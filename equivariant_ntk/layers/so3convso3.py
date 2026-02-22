import jax
from neural_tangents.stax import layer
from neural_tangents import Kernel
import s2fft
import s2fft.precompute_transforms
from math import floor, pi, cos
from functools import partial
from jax import vmap, jit
import jax.numpy as jnp

from equivariant_ntk.utils.vmap_helpers import vmap_subarray_fn
from equivariant_ntk.utils.so3 import Precompute_Wig, get_bw_from_so3_shape


def _affine(mat, W_std, b_std):
    if mat is not None:
        mat *= W_std**2

        if b_std is not None:
            mat += b_std**2

    return mat


def get_filter_shape(n_gamma: int, n_beta: int, n_alpha: int, max_beta: float):
    """Given an angle from the north pole (when considering (beta, alpha) being on the sphere for a
    fixed gamma), returns the shape of the filter.

    Given an equiangular grid of n_gamma x n_beta x n_alpha, and a maximum angle from the north pole, we can
    compute the shape of the filter where for each gamma we obtain all grid points, from the north pole up to the maximum
    angle.

    Args:
        n_gamma: Number of gamma points of the full grid.
        n_beta: Number of beta points of the full grid.
        n_alpha: Number of alpha points of the full grid.
        max_beta: Opening angle of the filter measured from the north pole.

    Returns:
        Tuple of integers (n_gamma_ker, n_beta_ker, n_alpha_ker) representing the shape of the filter.
    """
    in_bw = n_beta // 2
    n_beta_ker = floor(2 * in_bw * max_beta / pi - 1 / 2) + 1
    assert n_beta_ker >= 1, "Max_beta is too small for the low bandwidth."

    return (n_gamma, n_beta_ker, n_alpha)


def pad_filter(filter: jax.Array, bw: int, sampling: str):
    """Pads the filter along the beta axis to match the grid of the input signal."""
    n_beta = s2fft.sampling.so3_samples.f_shape(bw, bw, sampling=sampling)[1]
    return jnp.pad(filter, ((0, 0), (0, n_beta - filter.shape[1]), (0, 0)))


def downsample_fourier(x: jax.Array, out_bw: int):
    # Fourier space is independent of sampling grid
    in_bw = x.shape[1]
    if out_bw > in_bw:
        raise NotImplementedError("Upsampling not supported.")
    bw_diff = in_bw - out_bw
    nm_slice = slice(bw_diff, bw_diff + 2 * out_bw - 1)
    return x[nm_slice, :out_bw, nm_slice]


def downsample_fourier_kernel(x: jax.Array, out_bw: int):
    # Fourier space is independent of sampling grid
    in_bw = x.shape[2]
    if out_bw > in_bw:
        raise NotImplementedError("Upsampling not supported.")
    bw_diff = in_bw - out_bw
    nm_slice = slice(bw_diff, bw_diff + 2 * out_bw - 1)
    return x[nm_slice, nm_slice, :out_bw, :out_bw, nm_slice, nm_slice]


@partial(vmap, in_axes=(None, 0, None, None), out_axes=(0))   # batch dimension
@partial(vmap, in_axes=(4, None, None, None), out_axes=(3))   # output channel
def convolute_in_fourier(filters: jax.Array, inputs: jax.Array, out_bw: int, precomps_wig: Precompute_Wig):
    """Applies SO3 convolution in Fourier space.

    Args:
        filters: convolutional filters of all input channels. Dimensions are (gamma, beta, alpha,
        input channel, output channel)
        inputs: input signals of all input channels. Dimensions are (batch, gamma, beta, alpha, input channel)
        out_bw: bandwidth of the output signal
        precomps_wig: Precomputed Wigner d kernel for Wigner transform

    Returns:
        The convolved signal in real space.
    """

    in_bw = get_bw_from_so3_shape(inputs.shape[0:3], precomps_wig.sampling)
    reality = True
    fourier = partial(s2fft.precompute_transforms.wigner.forward_transform_jax,
                      kernel=precomps_wig.get_kernel(in_bw, 'forward', reality=reality), L=in_bw,
                      N=in_bw,
                      sampling=precomps_wig.sampling, reality=reality, nside=None)

    # vmap over input channel dimension, map to last dimension in output
    inputs_F = vmap(fourier, -1, -1)(inputs)

    # pad filters that have input channel dimension in the last axis
    filters_pad = vmap(pad_filter, (-1, None, None), -1)(filters, in_bw, precomps_wig.sampling)
    filters_F = vmap(fourier, -1, -1)(filters_pad)
    filters_F_con = jnp.conjugate(filters_F)

    filters_F_con = downsample_fourier(filters_F_con, out_bw)
    inputs_F = downsample_fourier(inputs_F, out_bw)

    # contract along input channels and right lower index
    conv_F = jnp.einsum('slni,slmi->nlm', filters_F_con, inputs_F)

    prefactor = 8 * pi**2 / (2 * jnp.arange(out_bw) + 1)
    conv_F = jnp.einsum('nlm,l->nlm', conv_F, prefactor)

    conv = s2fft.precompute_transforms.wigner.inverse_transform_jax(
        conv_F,
        kernel=precomps_wig.get_kernel(out_bw, 'backward', reality=reality),
        L=out_bw, N=out_bw,
        sampling=precomps_wig.sampling,
        reality=reality, nside=None
    )
    return jnp.real(conv)


@layer
def SO3ConvSO3(
    out_ch: int,
    max_beta: float,
    bw: tuple[int, int],
    precomps_wig: Precompute_Wig,
    parametrization: str = "ntk",
    s: tuple[int, int] = (1, 1),
    kappa_std: float = 1.0,
    b_std: float = None,
):
    in_bw, out_bw = bw

    filter_volume = 4 * pi**2 * (1 - cos(max_beta))
    in_grid_shape = s2fft.sampling.so3_samples.f_shape(in_bw, in_bw,
                                                       precomps_wig.sampling)

    filter_shape = get_filter_shape(*in_grid_shape, max_beta)

    def affine(mat):
        return _affine(mat, kappa_std, b_std)

    def init_fn(rng: jax.Array, input_shape: tuple[int, int, int, int, int]):
        batch_size, n_gamma, n_beta, n_alpha, in_ch = input_shape
        assert input_shape[1:-1] == in_grid_shape, (f"Input shape {input_shape} does not match"
                                                    f"given input bandlimit L = {in_bw} "
                                                    f"and sampling = {precomps_wig.sampling}.")

        out_grid_shape = s2fft.sampling.so3_samples.f_shape(out_bw, out_bw, precomps_wig.sampling)

        kernel_shape = filter_shape + (in_ch, out_ch)

        output_shape = (batch_size, *out_grid_shape, out_ch)
        bias_shape = (len(kernel_shape) - 1) * (1,) + (out_ch,)

        k1, k2 = jax.random.split(rng)
        kappa = jax.random.normal(k1, kernel_shape)
        b = None if b_std is None else jax.random.normal(k2, bias_shape)

        if parametrization == "standard":
            fan_in = in_ch * filter_volume
            kappa *= kappa_std / (fan_in / s[0]) ** 0.5
            b = None if b_std is None else b * b_std

        return output_shape, (kappa, b)

    def apply_fn(params, x: jax.Array, **kwargs):
        kappa, b = params

        if parametrization == "ntk":
            fan_in = x.shape[-1] * filter_volume
            kappa_norm = kappa_std / fan_in ** 0.5
            b_norm = b_std

        elif parametrization == "standard":
            kappa_norm = 1.0 / s[0] ** 0.5
            b_norm = 1.0

        y = convolute_in_fourier(kappa, x, out_bw, precomps_wig)

        y *= kappa_norm
        if b is not None:
            y = y + b_norm * b
        return y

    @jit
    def _group_aop(kernel: jax.Array):
        fourier = partial(s2fft.precompute_transforms.wigner.forward_transform_jax,
                          kernel=precomps_wig.get_kernel(in_bw, 'forward', reality=False), L=in_bw,
                          N=in_bw,
                          sampling=precomps_wig.sampling, reality=False, nside=None)
        num_batch = kernel.ndim - 6
        if num_batch == 2:
            b_str = 'ab'
        elif num_batch == 1:
            b_str = 'a'
        else:
            raise NotImplementedError("Only 1 or 2 batch dimensions are supported.")

        fourier_first = vmap_subarray_fn(fourier, f'{b_str}gshtiu->{b_str}nsltmu', 'ghi->nlm')
        fourier_second = vmap_subarray_fn(fourier, f'{b_str}nsltmu->{b_str}nqlomp', 'stu->qop')

        kernel_F = fourier_second(fourier_first(kernel))

        downsample_kernel_fn = partial(downsample_fourier_kernel,
                                       out_bw=out_bw)

        kernel_F = vmap_subarray_fn(downsample_kernel_fn, f'{b_str}nqlomp->{b_str}nqlomp',
                                    'nqlomp->nqlomp')(kernel_F)

        signs = jnp.arange(0, (2 * out_bw - 1)**2)
        signs = -2 * (signs % 2) + 1
        signs = signs.reshape((2 * out_bw - 1,) * 2)

        # r = r, l=p, p=p', m=m, k=m', n = n
        kernel_sum = jnp.einsum('...rrlpmk,nr->...nlpmk', jnp.flip(kernel_F, axis=-5), signs)

        l_diag = jnp.eye(out_bw)
        n_flip_diag = jnp.flip(jnp.eye(2 * out_bw - 1), axis=1)

        # i = n'
        kernel_F = jnp.einsum('...nlpmk,lp,ni->...inlpmk', kernel_sum, l_diag, n_flip_diag)

        prefactors = 8 * pi**2 / (2 * jnp.arange(out_bw) + 1)
        kernel_F = jnp.einsum('...inlpmk,l->...inlpmk', kernel_F, prefactors)


        fourier_inv = partial(s2fft.precompute_transforms.wigner.inverse_transform_jax,
                              kernel=precomps_wig.get_kernel(out_bw, 'backward', reality=False),
                              L=out_bw, N=out_bw,
                              sampling=precomps_wig.sampling,
                              reality=False, nside=None)

        fourier_inv_second = vmap_subarray_fn(fourier_inv, f'{b_str}nqlomp->{b_str}nsltmu',
                                              'qop->stu')
        fourier_inv_first = vmap_subarray_fn(fourier_inv, f'{b_str}nsltmu->{b_str}gshtiu',
                                             'nlm->ghi')

        kernel = jnp.real(fourier_inv_first(fourier_inv_second(kernel_F)))
        kernel /= 8 * pi**2
        return kernel

    def kernel_fn(k: Kernel, **kwargs):
        if abs(max_beta - pi) > 1e-5:
            raise NotImplementedError("Only max_beta=pi is supported at the moment.")

        cov1, nngp, cov2, ntk = (k.cov1, k.nngp, k.cov2, k.ntk)

        if parametrization != "ntk":
            raise NotImplementedError("Only NTK parameterization is supported.")

        nngp = _group_aop(nngp)
        cov1 = _group_aop(cov1)

        cov1 = affine(cov1)
        if cov2 is not None:
            cov2 = _group_aop(cov2)
            cov2 = affine(cov2)
        if ntk is not None:
            if ntk.ndim == 0:
                ntk = jnp.copy(nngp)
            else:
                ntk = nngp + kappa_std**2 * (_group_aop(ntk))


        result = k.replace(cov1=cov1,
                           nngp=nngp,
                           cov2=cov2,
                           ntk=ntk,
                           is_gaussian=True,
                           # batch_axis=0,
                           # channel_axis=-1,
                           is_input=False)

        return result

    return init_fn, apply_fn, kernel_fn
