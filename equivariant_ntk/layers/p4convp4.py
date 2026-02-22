from functools import partial
from jax import jit, vmap
from jax.lax import fori_loop
import jax.numpy as jnp
from jax.example_libraries.stax import GeneralConv
import numpy as onp
from jax import lax, random
from neural_tangents import Kernel
from neural_tangents._src.stax.linear import \
    _conv_kernel_full_spatial_shared as a_op
from neural_tangents._src.stax.requirements import layer
from neural_tangents.stax import Padding
import jax


@layer
def P4ConvP4(
    out_chan,
    filter_shape,
    padding=Padding.VALID.name,
    parameterization="ntk",
    s=(1, 1),
    W_std=1.0,
    b_std=None,
    kn_supp=4,
):
    """P4 equivariant Group Convolution
    Expects both input and output feature maps on P4.
    Args:
        out_chan (int): number of output channels
        filter_shape (Tuple[int, int]): width and height of filter
        padding (str): padding type, only `VALID` and `SAME` are supported.
        W_std (float): Standard deviation of the weight initialization
        b_std (float): Standard deviation of the bias initialization
        parameterization (str): Either `"ntk"` or `"standard"`.
        s (Tuple[int, int]):
            A tuple of integers, a direct convolutional analogue of the respective
            parameters for the `Dense` layer, see https://arxiv.org/abs/2001.07301.
    Returns:
        Tuple[Callable, Callable, Callable]: Returns initialization, apply, and kernel function
            (init_fn, apply_fn, kernel_fn).
    """

    padding = Padding(padding)
    if padding not in [Padding.VALID, Padding.SAME, Padding.CIRCULAR]:
        raise ValueError(f"Padding currently not supported: {padding}.")

    if padding == Padding.CIRCULAR:
        apply_padding = Padding.VALID
        init_padding = Padding.SAME
    else:
        apply_padding = padding
        init_padding = padding

    kn_supp_angles = [0, 1, 2, 3][:kn_supp]

    def init_fn(rng, input_shape):
        """Initialize parameters of group convolution
        Args:
            rng (PRNGKeyArray): a PRNG key
            input_shape (tuple): shape of input features of form `BGHWI`
        Returns:
            output (Tuple[Tuple[int], Tuple[jnp.array, jnp.array]]): output shape and initialized parameters W, b
        """
        b, g, h, w, c = input_shape
        assert g == 4, "Input has to have `G` dimension of 4."

        kernel_shape = (*filter_shape, kn_supp, c, out_chan)

        conv_init_fn, _ = GeneralConv(
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
            out_chan=out_chan,
            filter_shape=filter_shape,
            strides=(1, 1),
            padding=init_padding.name,
            W_init=random.normal,
            b_init=random.normal,
        )
        input_shape_wo_group = (b, h, w, c)
        output_shape_wo_group, _ = conv_init_fn(rng, input_shape_wo_group)
        output_shape = (output_shape_wo_group[0], 4, *output_shape_wo_group[1:])

        bias_shape = (1, 1, 1, 1, out_chan)

        k1, k2 = random.split(rng)
        W = random.normal(k1, kernel_shape)
        W = jnp.pad(W, ((0, 0),) * 2 + ((0, g - kn_supp),) + ((0, 0),) * 2)
        b = None if b_std is None else random.normal(k2, bias_shape)

        if parameterization == "standard":
            fan_in = c * g * onp.prod(filter_shape)
            W *= W_std / (fan_in / s[0]) ** 0.5
            b = None if b_std is None else b * b_std

        return output_shape, (W, b)

    def apply_fn(params, x, **kwargs):
        """Apply group convolution
        Args:
            params (tuple): Weights and bias parameters (W, b). Weight is `HWGIO` where `O` is number of out-channels
            x (jnp.array): Input features of form `BGHWI` where `I` is number of in-channels
        Returns:
            output (jnp.array): output of convolution of form BGHWO.
        """
        weight, bias = params
        assert x.ndim == 5
        assert x.shape[1] == 4, "Input has to have `G` dimension of 4."
        assert x.shape[2] == x.shape[3], "Feature map is not a square."

        if parameterization == "ntk":
            fan_in = x.shape[-1] * x.shape[1] * onp.prod(filter_shape)
            w_norm = W_std / fan_in**0.5
            b_norm = b_std
        elif parameterization == "standard":
            w_norm = 1.0 / s[0] ** 0.5
            b_norm = 1.0

        if padding == Padding.CIRCULAR:
            spatial_pads = lax.padtype_to_pads(x.shape[-3:-1], filter_shape, (1, 1), Padding.SAME.name)
            pads = [(0, 0)] * x.ndim
            pads[-3:-1] = spatial_pads
            x = jnp.pad(x, pads, mode="wrap")

        # lax conv use convention CHW, jax way of handling images is HWC
        # so we convert from BGHWC->BCGHW
        x = jnp.moveaxis(x, 4, 1)
        x = x.reshape(x.shape[0], -1, x.shape[3], x.shape[4])

        # HWGIO -> OIGHW
        weight = weight.transpose((4, 3, 2, 0, 1))

        # generate rotated and cyclic shifted kernel
        kco, _, _, kh, kw = weight.shape
        group_rotated_kernel = [
            jnp.rot90(jnp.roll(weight, k, 2), -k, (3, 4)).reshape(kco, -1, kh, kw) for k in range(4)
        ]
        group_rotated_kernel = jnp.concatenate(group_rotated_kernel, 0)

        y = w_norm * lax.conv(x, group_rotated_kernel, (1, 1), apply_padding.name)
        y = jnp.stack(jnp.split(y, 4, axis=1), axis=1)
        y = jnp.transpose(y, (0, 1, 3, 4, 2))
        if bias is not None:
            y = y + b_norm * bias

        return y

    def kernel_fn(k: Kernel, **kwargs):
        cov1, nngp, cov2, ntk, is_reversed = (k.cov1, k.nngp, k.cov2, k.ntk, k.is_reversed)

        def group_aop(kernel):
            # not traceable by jit:
            # if (kernel == onp.array([0.0])).all():
            #     return kernel
            *b, g1, g2, h1, h2, w1, w2 = kernel.shape
            if len(b) > 2:
                raise ValueError("Too many batch dimensions!")

            def rot_second_spatial_slot(kernel, inv=False):
                result = jnp.zeros_like(kernel)
                for r in range(4):
                    for r_prime in range(4):
                        if inv:
                            elem = result.at[..., r_prime, r, :, :, :, :]
                            kernel_slice = kernel[..., r_prime, r, :, :, :, :]
                        else:
                            elem = result.at[..., r, r_prime, :, :, :, :]
                            kernel_slice = kernel[..., r, r_prime, :, :, :, :]
                        result = elem.set(jnp.rot90(kernel_slice, r_prime - r, axes=(-3, -1)))
                return result


            @partial(jit, static_argnames=("axes",))
            def traceable_rot90(m, r, axes=(0, 1)):
                r %= 4
                return jax.lax.switch(r, [partial(jnp.rot90, m, k=i, axes=axes) for i in range(4)])

            @jit
            def rot_second_spatial_slot_for_r_diffs(kernel):
                rs = jnp.arange(4)
                r_diffs = rs[None, :] - rs[:, None]

                rot_fn = partial(traceable_rot90, axes=(-3, -1))

                func_over_r_prime = vmap(
                    rot_fn,
                    in_axes=(-5, 0),
                    out_axes=-5,
                )
                func_over_both_rs = vmap(func_over_r_prime, in_axes=(-6, 0), out_axes=-6)
                return func_over_both_rs(kernel, r_diffs)

            @jit
            def rot_second_spatial_slot_for_r_diffs_inv(kernel_pair_entry):
                rs = jnp.arange(4)
                r_diffs = rs[:, None] - rs[None, :]

                rot_fn = partial(traceable_rot90, axes=(-3, -1))

                func_over_r = vmap(
                    rot_fn,
                    in_axes=(-5, 0),
                    out_axes=-5,
                )
                func_over_both_rs = vmap(func_over_r, in_axes=(-6, 0), out_axes=-6)
                return func_over_both_rs(kernel_pair_entry, r_diffs)

            # transform nngp before A operator is applied:
            # K(t,t') -> K(t, R(r' r^{-1}) t')
            kernel = rot_second_spatial_slot_for_r_diffs(kernel)

            # absorb group dimensions in batch dimensions
            if len(b) == 2:
                kernel = jnp.moveaxis(kernel, 2, 1)
                kernel = kernel.reshape((b[0] * 4, b[1] * 4, h1, h2, w1, w2))
            else:
                kernel = kernel.reshape((b[0] * 16, h1, h2, w1, w2))

            # For a general gconv, the A operator is defined by
            # A(K)(t, t') = \sum_y c_in/fan_in K(t + y, t' + y) (fan_in = q^2 * c_in)
            # Since aop is implemented for standard convolutions, it assumes that c_in/fan-in is 1/q^2,
            # where q^2 is the number of spatial positions and c_in is the number of input channels of
            # the convolutional filter. For P4P4 convolutions, the c_in / fan-in is however
            # 1 / (|C_4| * q^2) resulting in the correction factor of 1/4.
            kernel = (
                a_op(kernel, filter_shape, strides=(1, 1), padding=padding, batch_ndim=len(b)) / kn_supp
            )
            # Actually, aop returns transposed of A(K)
            # in the sense that t_x <-> t_y and similarly for t'
            if len(b) == 2:
                kernel = kernel.transpose(0, 1, 4, 5, 2, 3)
                spatial_shape = kernel.shape[-4:]
                kernel = kernel.reshape((b[0], 4, b[1], 4, *spatial_shape))
                kernel = jnp.moveaxis(kernel, 1, 2)
            else:
                kernel = kernel.transpose(0, 3, 4, 1, 2)
                spatial_shape = kernel.shape[-4:]
                kernel = kernel.reshape((b[0], 4, 4, *spatial_shape))

            # transform kernel back
            kernel = rot_second_spatial_slot_for_r_diffs_inv(kernel)

            # sum over r_tilde
            kernel = jnp.sum(
                jnp.stack([jnp.roll(kernel, r_tilde, axis=(-6, -5)) for r_tilde in kn_supp_angles]),
                axis=0,
            )
            return kernel

        def affine_aop(kernel):
            kernel_unscaled = group_aop(kernel)
            kernel = W_std**2 * kernel_unscaled
            if b_std is not None:
                kernel = kernel + b_std**2
            return kernel, kernel_unscaled

        nngp, nngp_unscaled = affine_aop(nngp)
        cov1, _ = affine_aop(cov1)
        if cov2 is not None:
            cov2, _ = affine_aop(cov2)

        if ntk is not None:
            if ntk.ndim == 0:
                ntk = jnp.copy(k.nngp)
            if parameterization == "ntk":
                ntk = W_std**2 * group_aop(ntk) + nngp
            elif parameterization == "standard":
                fan_in = k.shape1[-1] * k.shape1[1] * onp.prod(filter_shape)
                factor = fan_in / s[0]
                ntk = factor * nngp_unscaled + W_std**2 * group_aop(ntk) + 1.0
            else:
                raise TypeError(f"Parameterization {parameterization} is invalid.")

        result = k.replace(
            cov1=cov1,
            nngp=nngp,
            cov2=cov2,
            ntk=ntk,
            is_gaussian=True,
            is_reversed=is_reversed,
            batch_axis=0,
            channel_axis=-1,
            is_input=False,
        )

        return result

    return init_fn, apply_fn, kernel_fn
