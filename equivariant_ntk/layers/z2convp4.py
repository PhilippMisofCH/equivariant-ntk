import jax.numpy as jnp
import numpy as onp
from jax import lax, random, vmap, jit
from jax.lax import fori_loop
from jax.example_libraries.stax import GeneralConv
from neural_tangents import Kernel
from neural_tangents._src.stax.linear import _conv_kernel_full_spatial_shared as a_op
from neural_tangents._src.stax.requirements import layer
from neural_tangents.stax import Padding
from functools import partial


@layer
def Z2ConvP4(
    out_chan,
    filter_shape,
    padding=Padding.VALID.name,
    parameterization="ntk",
    s=(1, 1),
    W_std=1.0,
    b_std=None,
):
    """Z2 equivariant Group Convolution
    Expects input and output feature maps on Z2 and P4 respectively.
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
        Tuple[Callable, Callable, Callable]: Returns initialization, apply, kernel function.
    """
    if parameterization not in ["ntk", "standard"]:
        raise ValueError(f"Parameterization not supported: {parameterization}.")

    padding = Padding(padding)
    if padding not in [Padding.VALID, Padding.SAME, Padding.CIRCULAR]:
        raise ValueError(f"Padding currently not supported: {padding}.")

    if padding == Padding.CIRCULAR:
        apply_padding = Padding.VALID
        init_padding = Padding.SAME
    else:
        apply_padding = padding
        init_padding = padding

    def init_fn(rng, input_shape):
        """Initialize parameters of group convolution
        Args:
            rng (PRNGKeyArray): a PRNG key
            input_shape (tuple): shape of input features of form `BHWI`
        Returns:
            output (Tuple[Tuple[int], Tuple[jnp.array, jnp.array]]): output shape and
                                                                     initialized parameters W, b
        """
        b, h, w, c = input_shape

        kernel_shape = (*filter_shape, c, out_chan)

        conv_init_fn, _ = GeneralConv(
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
            out_chan=out_chan,
            filter_shape=filter_shape,
            strides=(1, 1),
            padding=init_padding.name,
            W_init=random.normal,
            b_init=random.normal,
        )
        output_shape_wo_group, _ = conv_init_fn(rng, input_shape)
        output_shape = (output_shape_wo_group[0], 4, *output_shape_wo_group[1:])

        bias_shape = (1, 1, 1, 1, out_chan)

        k1, k2 = random.split(rng)
        W = random.normal(k1, kernel_shape)
        b = None if b_std is None else random.normal(k2, bias_shape)

        if parameterization == "standard":
            fan_in = c * onp.prod(filter_shape)
            W *= W_std / (fan_in / s[0]) ** 0.5
            b = None if b_std is None else b * b_std

        return output_shape, (W, b)

    def apply_fn(params, x, **kwargs):
        """Apply group convolution
        Args:
            params (tuple): Weights and bias parameters (W, b).
                            Weight is `HWIO` where `O` is number of out-channels
            x (jnp.array): Input features of form `BHWI` where `I` is number of in-channels
        Returns:
            output (jnp.array): output of convolution of form BGHWO.
        """
        weight, bias = params
        assert x.ndim == 4
        assert x.shape[1] == x.shape[2], "Feature map is not a square."

        if parameterization == "ntk":
            fan_in = x.shape[-1] * onp.prod(filter_shape)
            w_norm = W_std / fan_in**0.5
            b_norm = b_std
        elif parameterization == "standard":
            w_norm = 1.0 / s[0] ** 0.5
            b_norm = 1.0

        if padding == Padding.CIRCULAR:
            spatial_pads = lax.padtype_to_pads(
                x.shape[-3:-1], filter_shape, (1, 1), Padding.SAME.name
            )
            pads = [(0, 0)] * x.ndim
            pads[-3:-1] = spatial_pads
            x = jnp.pad(x, pads, mode="wrap")
        # lax conv use convention CHW, jax way of handling images is HWC
        # so we convert from BHWC->BCHW
        x = jnp.moveaxis(x, 3, 1)

        # HWIO -> OIHW
        weight = weight.transpose((3, 2, 0, 1))

        # generate rotated and cyclic shifted kernel
        group_rotated_kernel = [jnp.rot90(weight, -k, (-2, -1)) for k in range(4)]
        group_rotated_kernel = jnp.concatenate(group_rotated_kernel, 0)

        y = w_norm * lax.conv(x, group_rotated_kernel, (1, 1), apply_padding.name)
        y = jnp.stack(jnp.split(y, 4, axis=1), axis=1)
        y = jnp.transpose(y, (0, 1, 3, 4, 2))
        if bias is not None:
            y = y + b_norm * bias

        return y

    def kernel_fn(k: Kernel, **kwargs):
        cov1, nngp, cov2, ntk, is_reversed = (
            k.cov1,
            k.nngp,
            k.cov2,
            k.ntk,
            k.is_reversed,
        )

        def group_aop(kernel):
            # not traceable by jit:
            # if (kernel == onp.array([0.0])).all():
            #     return kernel

            n_batch_dim = len(kernel.shape) - 4
            *b, h1, h2, w1, w2 = kernel.shape
            if n_batch_dim > 2:
                raise ValueError("Too many batch dimensions!")

            def rot_second_spatial_slot(kernel, inv=False):
                result_shape = (*b, 4, 4, h1, h2, w1, w2)
                result = jnp.zeros(result_shape)
                for r in range(4):
                    for r_prime in range(4):
                        if inv:
                            elem = result.at[..., r_prime, r, :, :, :, :]
                            kernel_slice = kernel[..., r_prime, r, :, :, :, :]
                        else:
                            elem = result.at[..., r, r_prime, :, :, :, :]
                            kernel_slice = kernel
                        result = elem.set(
                            jnp.rot90(kernel_slice, r_prime - r, axes=(-3, -1))
                        )
                return result

            # May not be the most efficient way but still faster than python
            # loops. Problem is that jnp.rot90 is not vectorizeable over the
            # rotations
            @partial(jit, static_argnames=("axes",))
            def traceable_rot90(m, r, axes=(0, 1)):
                r %= 4
                return lax.switch(r, [partial(jnp.rot90, m, k=i, axes=axes) for i in range(4)])

            @jit
            def rot_second_spatial_slot_for_r_diffs(kernel):
                rs = jnp.arange(4)
                r_diffs = rs[None, :] - rs[:, None]

                rot_fn = partial(traceable_rot90, axes=(-3, -1))

                func_over_r_prime = vmap(
                    rot_fn,
                    in_axes=(None, 0),
                    out_axes=-5,
                )
                func_over_both_rs = vmap(
                    func_over_r_prime, in_axes=(None, 0), out_axes=-6
                )
                return func_over_both_rs(kernel, r_diffs)

            @jit
            def rot_second_spatial_slot_for_r_diffs_inv(kernel):
                rs = jnp.arange(4)
                r_diffs = rs[:, None] - rs[None, :]

                rot_fn = partial(traceable_rot90, axes=(-3, -1))

                func_over_r = vmap(
                    rot_fn,
                    in_axes=(-5, 0),
                    out_axes=-5,
                )
                func_over_both_rs = vmap(func_over_r, in_axes=(-6, 0), out_axes=-6)
                return func_over_both_rs(kernel, r_diffs)

            # transform nngp before A operator is applied:
            # K(t,t') -> K(t, R(r' r^{-1}) t')
            kernel = rot_second_spatial_slot_for_r_diffs(kernel)

            # absorb group dimensions in batch dimensions
            if n_batch_dim == 2:
                kernel = jnp.moveaxis(kernel, 2, 1)
                kernel = kernel.reshape((b[0] * 4, b[1] * 4, h1, h2, w1, w2))
            else:
                kernel = kernel.reshape((b[0] * 16, h1, h2, w1, w2))

            kernel = a_op(
                kernel,
                filter_shape,
                strides=(1, 1),
                padding=padding,
                batch_ndim=n_batch_dim,
            )
            # aop returns transposed of
            # A(K)(t, t') = \sum_y 1/q^2 K(t + y, t' + y)
            # in the sense that t_x <-> t_y and similarly for t'
            if n_batch_dim == 2:
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
                fan_in = k.shape1[-1] * onp.prod(filter_shape)
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
            channel_axis=-1 % (len(k.shape1)+1),
            is_input=False,
            # shape1=(k.shape1[-4], 4, *k.shape1[-3:]),
            # shape2=(k.shape1[-4], 4, *k.shape2[-3:]),
        )

        return result

    return init_fn, apply_fn, kernel_fn
