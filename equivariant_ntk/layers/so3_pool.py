import s2fft
from neural_tangents._src.stax.requirements import layer
from neural_tangents import Kernel
import jax.numpy as jnp
import jax
from math import pi
from functools import partial
from equivariant_ntk.utils.vmap_helpers import vmap_subarray_fn

jax.config.update("jax_enable_x64", True)


def _quad_weights_so3(L, sampling):
    weights = s2fft.utils.quadrature.quad_weights(L, sampling="dh")
    weights *= 2 * pi / (2 * L - 1)
    return weights


def _integrate_so3(so3_signal, sampling):
    grid_shape = so3_signal.shape
    L = grid_shape[1] // 2
    supported_grid_shape = s2fft.sampling.so3_samples.f_shape(L, L, sampling=sampling)

    if supported_grid_shape != grid_shape:
        raise ValueError(
            f"Input has to have spatial grid {supported_grid_shape} for SO(3) "
            f"pooling but has {grid_shape}"
        )
    weights = _quad_weights_so3(L, sampling)
    integral = jnp.sum(so3_signal * weights[None, :, None])
    return integral


def _integrate_so3_batched(so3_signal, sampling):
    integral_fn = partial(_integrate_so3, sampling=sampling)
    batched_integral_fn = vmap_subarray_fn(integral_fn, "aghoi->ai", "gho->")
    return batched_integral_fn(so3_signal)


def _integrate_so3_kernel_batched(so3_kernel, sampling):
    n_batch = so3_kernel.ndim - 6
    b_str = "ab" if n_batch == 2 else "a"

    integral_fn = partial(_integrate_so3, sampling=sampling)
    integral_second = vmap_subarray_fn(
        integral_fn, f"{b_str}gshtiu->{b_str}ghi", "stu->"
    )
    integral_first = vmap_subarray_fn(integral_fn, f"{b_str}ghi->{b_str}", "ghi->")
    return integral_first(integral_second(so3_kernel))


@layer
def SO3Pool():
    """Pooling over entire SO(3) group.
    Returns:
        Tuple[Callable, Callable, Callable]: Returns initialization, apply, and kernel function
            (init_fn, apply_fn, kernel_fn).
    """
    sampling = "dh"

    def init_fn(rng, input_shape):
        """No parameters to be initialized
        Args:
            input_shape (tuple): shape of input features of form `batch, gamma, beta, alpha,
            in_ch``
        Returns:
            output (Tuple[int]): output shape
        """
        b, *grid_shape, c = input_shape
        grid_shape = tuple(grid_shape)
        L = grid_shape[1] // 2

        supported_grid_shape = s2fft.sampling.so3_samples.f_shape(
            L, L, sampling=sampling
        )
        assert grid_shape == supported_grid_shape, (
            "Input has to have spatial grid "
            f"{supported_grid_shape} for SO(3) pooling but has {grid_shape}"
        )

        output_shape = (b, c)

        return output_shape, ()

    def apply_fn(_, x, **kwargs):
        """Apply SO(3) pooling
        Args:
            x (jnp.array): Input features of form `batch, gamma, beta, alpha, in_ch`
        Returns:
            output (jnp.array): output of pooling of form `batch, in_ch`.
        """
        y = _integrate_so3_batched(x, sampling)
        return y / (8 * pi**2)

    def _group_mean(kernel):
        if kernel is None:
            return kernel
        integral = _integrate_so3_kernel_batched(kernel, sampling)
        return integral / (8 * pi**2) ** 2

    def kernel_fn(k: Kernel, **kwargs):
        cov1, nngp, cov2, ntk, is_reversed = (
            k.cov1,
            k.nngp,
            k.cov2,
            k.ntk,
            k.is_reversed,
        )

        return k.replace(
            cov1=_group_mean(cov1),
            nngp=_group_mean(nngp),
            cov2=_group_mean(cov2),
            ntk=_group_mean(ntk),
            is_gaussian=True,
            channel_axis=1,
            is_input=False,
        )

    return init_fn, apply_fn, kernel_fn
