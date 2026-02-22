from functools import partial
from math import pi, floor, cos

from neural_tangents.stax import layer
from neural_tangents import Kernel
import s2fft
import s2fft.precompute_transforms
import jax.numpy as jnp
import jax
from jax import random, vmap, jit
from jax import config

from equivariant_ntk.utils.vmap_helpers import vmap_subarray_fn
from equivariant_ntk.utils.so3 import (
    Precompute_Wig,
    Precompute_Sph,
    get_bw_from_s2_shape,
)

config.update("jax_enable_x64", True)


def _affine(mat: jax.Array, W_std: float, b_std: float) -> jax.Array:
    """Apply an affine transformation :math:`W A + b` to a matrix :math:`A`.

    Args:
        mat: Matrix to be transformed.
        W_std: Standard deviation of the weights.
        b_std: Standard deviation of the biases.

    Returns:
        The transformed matrix.
    """
    if mat is not None:
        mat *= W_std**2

        if b_std is not None:
            mat += b_std**2

    return mat


def _get_filter_shape(n_theta: int, n_phi: int, max_beta: float) -> tuple[int, int]:
    r"""Given an angle from the north pole, returns the shape of the filter.

    Given an equiangular grid of :math:`n_\theta \times n_\phi`, and a maximum angle from the north
    pole, we can compute the shape of the filter containing all grid points from the north pole up
    to the maximum angle.

    Args:
        n_theta: Number of theta points of the full grid.
        n_phi: Number of phi points of the full grid.
        max_beta: Opening angle of the filter measured from the north pole.

    Returns:
        Tuple of integers (n_theta_ker, n_phi_ker) representing the shape of the filter.
    """
    in_bw = n_theta // 2
    n_theta_ker = floor(2 * in_bw * max_beta / pi - 1 / 2)
    assert n_theta_ker > 0, "Max_beta is too small for the low bandwidth."

    return (n_theta_ker, n_phi)


def _fourier_transform(input: jax.Array, precomps: Precompute_Sph) -> jax.Array:
    r"""Applies the :math:`S^2`-Fourier transform to the input signal.

    Args:
        input: Input signal on the sphere with grid :math:`(\theta, \phi)`.
        precomps: Precomputed Wigner d kernels for the spherical Fourier transform.

    Returns:
        The Fourier transformed signal.
    """
    reality = False
    L = get_bw_from_s2_shape(input.shape, precomps.sampling)
    input_F = s2fft.precompute_transforms.spherical.forward_transform_jax(
        input,
        kernel=precomps.get_kernel(L, "forward", reality),
        L=L,
        sampling=precomps.sampling,
        reality=reality,
        spin=0,
        nside=None,
    )

    return input_F


def _pad_filter(filter: jax.Array, bw: int, sampling: str) -> jax.Array:
    r"""Pads the filter along the :math:`\theta`-axis to match the grid of the input signal.

    Args:
        filter: Filter to be padded of shape :math:`(\theta, \phi)`.
        bw: Bandwidth of the input signal.
        sampling: Sampling scheme of the input signal.

    Returns:
        The padded filter.
    """

    n_theta = s2fft.sampling.s2_samples.f_shape(bw, sampling=sampling)[0]
    return jnp.pad(filter, ((0, n_theta - filter.shape[0]), (0, 0)))


def _fourier_transform_filter(
    filter: jax.Array, L: int, precomps: Precompute_Sph
) -> jax.Array:
    return _fourier_transform(_pad_filter(filter, L, precomps.sampling), precomps)


def _downsample_fourier(input: jax.Array, out_bw: int) -> jax.Array:
    r"""Downsamples a signal in the Fourier domain of :math:`S^2` to the desired bandwidth.

    Upsampling is currently not supported.

    Args:
        input: signal in Fourier space.
        out_bw: Bandwidth of the output signal.
    Returns:
        The downsampled signal.
    """

    # Fourier space is independent of sampling grid
    in_bw = input.shape[0]
    if out_bw > in_bw:
        raise NotImplementedError("Upsampling not supported.")
    bw_diff = in_bw - out_bw
    return input[:out_bw, bw_diff : bw_diff + 2 * out_bw - 1]


def _downsample_fourier_kernel(input: jax.Array, out_bw: int) -> jax.Array:
    # Fourier space is independent of sampling grid
    in_bw = input.shape[0]
    if out_bw > in_bw:
        raise NotImplementedError("Upsampling not supported.")
    bw_diff = in_bw - out_bw
    n_slice = slice(bw_diff, bw_diff + 2 * out_bw - 1)
    return input[:out_bw, :out_bw, n_slice, n_slice]


@partial(vmap, in_axes=(None, 0, None, None, None), out_axes=(0))  # batch dimension
@partial(vmap, in_axes=(3, None, None, None, None), out_axes=(3))  # output channel
def convolute_in_fourier(
    filters: jax.Array,
    inputs: jax.Array,
    out_bw: int,
    precomps_sph: Precompute_Sph,
    precomps_wig: Precompute_Wig,
):
    """Applies :math:`SO(3)` convolution via Fourier space.

    Args:
        filters: Convolutional filters of all input channels. Dimensions are `(theta, phi, input
            channel, output channel)`.
        inputs: input signals of all input channels. Dimensions are `(batch, theta, phi, input
            channel)`.
        out_bw: bandwidth of the output signal
        precomps_sph: Precomputed Wigner d kernel for spherical transform
        precomps_wig: Precomputed Wigner d kernel for Wigner transform

    Returns:
        The convolved signal in real space.
    """
    in_bw = get_bw_from_s2_shape(inputs.shape[0:2], precomps_sph.sampling)
    if out_bw > in_bw:
        raise NotImplementedError(
            "Upsampling of spherical convolution not yet supported."
        )
    # vmap over input channel dimension, map to last dimension in output
    inputs_F = vmap(_fourier_transform, (2, None), 2)(inputs, precomps_sph)

    filters_F = vmap(_fourier_transform_filter, (2, None, None), 2)(
        filters, in_bw, precomps_sph
    )
    filters_F_con = jnp.conjugate(filters_F)

    filters_F_con = _downsample_fourier(filters_F_con, out_bw)
    inputs_F = _downsample_fourier(inputs_F, out_bw)

    # contract along input channels
    conv_F = jnp.einsum("lni,lmi->nlm", filters_F_con, inputs_F)

    ls = jnp.arange(0, out_bw)
    prefactors = 8 * pi**2 / (2 * ls + 1)
    conv_F = jnp.einsum("nlm,l->nlm", conv_F, prefactors)

    reality = False
    conv = s2fft.precompute_transforms.wigner.inverse_transform_jax(
        conv_F,
        kernel=precomps_wig.get_kernel(out_bw, "backward", reality),
        L=out_bw,
        N=out_bw,
        sampling=precomps_wig.sampling,
        reality=reality,
        nside=None,
    )
    return jnp.real(conv)


@layer
def S2ConvSO3(
    out_ch: int,  # num of output channels
    max_beta: float,  # maximum angle from north pole where kernel is supported, from 0 to pi
    bw: tuple[int, int],  # in and out bandwidth of Fourier transform
    precomps_sph: Precompute_Sph,
    precomps_wig: Precompute_Wig,
    parametrization: str = "ntk",
    s: tuple[int, int] = (1, 1),
    kappa_std: float = 1.0,
    b_std: float = None,
):
    filter_volume = 2 * pi * (1 - cos(max_beta))
    in_bw, out_bw = bw

    in_grid_shape = s2fft.sampling.s2_samples.f_shape(
        in_bw, sampling=precomps_sph.sampling
    )

    def affine(mat):
        return _affine(mat, kappa_std, b_std)

    def init_fn(
        rng: jax.Array, input_shape: tuple[int, int, int, int]
    ) -> tuple[tuple[int, int, int, int], tuple[jax.Array, jax.Array]]:
        """Initializes the parameters of the layer.

        Args:
            rng: Random number key array for jax.
            input_shape: Shape of the input array for the layer (batch_size, n_theta, n_phi, in_ch).
        Returns:
            Tuple (output shape, (filter, biases)).
        """
        batch_size, n_theta, n_phi, in_ch = input_shape

        assert (n_theta, n_phi) == in_grid_shape, (
            f"Input grid {(n_theta, n_phi)} does not match the "
            f"given input bandlimit L = {in_bw} and sampling = {precomps_sph.sampling}."
        )

        filter_shape = _get_filter_shape(n_theta, n_phi, max_beta)
        kernel_shape = (*filter_shape, in_ch, out_ch)

        out_grid_shape = s2fft.sampling.so3_samples.f_shape(
            out_bw, out_bw, sampling="dh"
        )
        output_shape = (batch_size, *out_grid_shape, out_ch)
        bias_shape = (len(kernel_shape) - 1) * (1,) + (out_ch,)

        k1, k2 = random.split(rng)
        kappa = random.normal(k1, kernel_shape)
        b = None if b_std is None else random.normal(k2, bias_shape)

        if parametrization == "standard":
            fan_in = in_ch * filter_volume
            kappa *= kappa_std / (fan_in / s[0]) ** 0.5
            b = None if b_std is None else b * b_std

        return output_shape, (kappa, b)

    def apply_fn(params, x: jax.Array, **kwargs):
        # dims of x: (batches, thetas, phis, in_channels)
        # dims of kappa: (thetas, phis, in_channels, out_channels)
        # dims of y: (batches, gammas, betas, alphas, out_channels)
        kappa, b = params
        assert x.ndim == 4

        if parametrization == "ntk":
            fan_in = x.shape[-2] * filter_volume
            kappa_norm = kappa_std / fan_in**0.5
            b_norm = b_std

        elif parametrization == "standard":
            kappa_norm = 1.0 / s[0] ** 0.5
            b_norm = 1.0

        y = convolute_in_fourier(kappa, x, out_bw, precomps_sph, precomps_wig)
        y *= kappa_norm
        if b is not None:
            y = y + b_norm * b

        return y

    @jit
    def _group_aop(kernel: jax.Array):
        reality = False
        fourier_sphere = partial(
            s2fft.precompute_transforms.spherical.forward_transform_jax,
            kernel=precomps_sph.get_kernel(in_bw, "forward", reality),
            L=in_bw,
            sampling=precomps_sph.sampling,
            reality=reality,
            spin=0,
            nside=None,
        )

        num_batch = kernel.ndim - 4
        if num_batch == 2:
            b_str = "ab"
        elif num_batch == 1:
            b_str = "a"
        else:
            raise NotImplementedError("Only 1 or 2 batch dimensions are supported.")
        # nngp has shape (batch1, [batch2], theta1, [theta2], phi1, [phi2])
        fourier_first_sph = vmap_subarray_fn(
            fourier_sphere, f"{b_str}hist->{b_str}limt", "hs->lm"
        )
        fourier_second_sph = vmap_subarray_fn(
            fourier_sphere, f"{b_str}limt->{b_str}ljmn", "it->jn"
        )

        def fourier_sph_double(double_sph):
            return fourier_second_sph(fourier_first_sph(double_sph))

        # nngp_F has shape (batch1, [batch2], l, l', m, m')
        kernel_F = fourier_sph_double(kernel)
        downsample_kernel_fn = partial(_downsample_fourier_kernel, out_bw=out_bw)
        kernel_F = vmap_subarray_fn(
            downsample_kernel_fn, f"{b_str}ljmn->{b_str}ljmn", "ljmn->ljmn"
        )(kernel_F)

        signs = jnp.arange(0, (2 * out_bw - 1)) + (out_bw + 1) % 2
        signs = -2 * (signs % 2) + 1

        ls = jnp.arange(0, out_bw)
        ns = jnp.arange(-out_bw + 1, out_bw)
        nls = jnp.tile(ns, (out_bw, 1))
        # nmask has shape (l, n)
        n_mask = jnp.abs(nls) <= ls[:, None]

        prefactors = (8 * pi**2 / (2 * ls + 1)) ** 2

        l_diag = jnp.eye(out_bw)
        n_flip_diag = jnp.flip(jnp.eye(2 * out_bw - 1), axis=1)
        # # dimensions (batch1, batch2, l, l', m, m', n, n'); j=l', k=m', i=n'
        kernel_F = jnp.einsum(
            "l,...ljmk,n,ni,lj,ln->...ljmkni",
            prefactors,
            kernel_F,
            signs,
            n_flip_diag,
            l_diag,
            n_mask,
        )

        reality = False
        # remember inverse wigner Fourier expects (n,l,m) as dimensions
        fourier_inv_wig = partial(
            s2fft.precompute_transforms.wigner.inverse_transform_jax,
            kernel=precomps_wig.get_kernel(out_bw, "backward", reality=reality),
            L=out_bw,
            N=out_bw,
            sampling=precomps_wig.sampling,
            reality=reality,
            nside=None,
        )

        fourier_inv_wig_second = vmap_subarray_fn(
            fourier_inv_wig, f"{b_str}lompnq->{b_str}lsmtnu", "qop->stu"
        )
        fourier_inv_wig_first = vmap_subarray_fn(
            fourier_inv_wig, f"{b_str}lsmtnu->{b_str}gshtiu", "nlm->ghi"
        )

        def fourier_inv_wig_double(double_wig):
            return fourier_inv_wig_first(fourier_inv_wig_second(double_wig))

        new_kernel = jnp.real(fourier_inv_wig_double(kernel_F))
        # Assumes global support of the filter
        new_kernel /= 4 * pi
        return new_kernel

    def kernel_fn(k: Kernel, **kwargs):
        if abs(max_beta - pi) > 1e-5:
            raise NotImplementedError("Only max_beta=pi is supported at the moment.")

        cov1, nngp, cov2, ntk, is_reversed = (
            k.cov1,
            k.nngp,
            k.cov2,
            k.ntk,
            k.is_reversed,
        )

        if parametrization != "ntk":
            raise NotImplementedError("Only NTK parameterization is supported.")

        nngp = _group_aop(nngp)
        cov1 = _group_aop(cov1)

        nngp = affine(nngp)
        cov1 = affine(cov1)
        if cov2 is not None:
            cov2 = _group_aop(cov2)
            cov2 = affine(cov2)
        if ntk is not None:
            if ntk.ndim == 0:
                ntk = jnp.copy(nngp)
            else:
                ntk = nngp + kappa_std**2 * (_group_aop(ntk))

        result = k.replace(
            nngp=nngp,
            cov1=cov1,
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
