from scipy.spatial.transform import Rotation
import jax
import jax.numpy as jnp
from jax import vmap
from s2fft.utils.rotation import rotate_flms
import s2fft.precompute_transforms
import s2fft
import numpy as np
from math import pi
from itertools import product
from functools import partial, wraps
from equivariant_ntk.utils.vmap_helpers import vmap_subarray_fn
from equivariant_ntk.utils.so3 import Precompute_Wig, Precompute_Sph, get_bw_from_s2_shape, get_bw_from_so3_shape

jax.config.update("jax_enable_x64", True)


euler_conv = "zyz"


def create_rnd_euler_angles():
    return tuple(np.random.rand(3) * np.array([2 * pi, pi, 2 * pi]))


def invert_euler_angles(euler_angles):
    rot = Rotation.from_euler(euler_conv, euler_angles)
    inv_euler_angles = rot.inv().as_euler(euler_conv)
    return tuple(inv_euler_angles)


def create_wigner(L, euler_angs):
    """create full Wigner D matrix corresponding to rotation r

    Args:
        L (int): bandlimit
        euler_angs (tuple): Euler angles (alpha, beta, gamma) in 'zyz' convention

    Returns:
        jnp.ndarray: Wigner D matrix with dimensions (l, m, n) and size (L, 2L-1, 2L-1)
    """

    alph, bet, gam = euler_angs

    dlm_s = []
    dlm_s.append(np.zeros((2 * L - 1, 2 * L - 1), dtype=np.float64))
    dlm_s[0] = s2fft.recursions.risbo_jax.compute_full(dl=np.copy(dlm_s[0]), beta=bet, L=L, el=0)
    for el in range(1, L):
        dlm_s.append(s2fft.recursions.risbo_jax.compute_full(dl=np.copy(dlm_s[el - 1]), beta=bet, L=L, el=el))
    dlmn = jnp.stack(dlm_s, axis=0)

    ms = jnp.arange(-L + 1, L)
    ns = jnp.arange(-L + 1, L)

    alph_phase = jnp.exp(-1j * ms * alph)
    gam_phase = jnp.exp(-1j * ns * gam)
    Dlmn = jnp.einsum('lmn,n->lmn', dlmn, gam_phase)
    Dlmn = jnp.einsum('m,lmn->lmn', alph_phase, Dlmn)
    return Dlmn


def get_bandlim_from_real_s2shape(shape, sampling='dh'):
    L = shape[0] // 2
    if shape != s2fft.sampling.s2_samples.f_shape(L, sampling):
        raise ValueError(f"shape {shape} of spherical signal is not compatible with sampling {sampling}")
    return L


def get_bandlim_from_real_so3shape(shape, sampling='dh'):
    # shape = (beta, alpha, gamma), where (beta, alpha) corresponds to (theta, phi)
    L = shape[1] // 2
    if shape != s2fft.sampling.so3_samples.f_shape(L, L, sampling):
        raise ValueError(f"shape {shape} of spherical signal is not compatible with sampling {sampling}")
    return L


def upsample_so3(f, L_res, sampling='dh'):
    """Upsample a signal on SO(3) to a higher bandlimit.

    Args:
        f (jnp.ndarray): input signal with dimensions (gamma, beta, alpha)
        L_res (int): signal resolution bandlimit
        sampling (str, optional): Defaults to 'dh'. Sampling type.

    Returns:
        upsampled signal with dimensions (gamma, beta, alpha, input_channels)
    """
    L = get_bandlim_from_real_so3shape(f.shape, sampling='dh')
    flmn = s2fft.transforms.wigner.forward_jax(f, L, L, sampling='dh', reality=False)
    flmn = jnp.pad(flmn, ((L_res - L,) * 2, (0, L_res - L), (L_res - L,) * 2))
    return s2fft.transforms.wigner.inverse_jax(flmn, L_res, L_res, sampling='dh', reality=False)


@partial(vmap, in_axes=(0, None, None), out_axes=(0))   # batch dimension
@partial(vmap, in_axes=(2, None, None), out_axes=(2))   # input channel dimension
def rotate_batch_s2_signal(f: jax.Array, euler_angs: tuple[float, float, float], precomps: Precompute_Sph):
    """Rotate a signal on the sphere via Fourier space.

    This also handles batch and input channel dimensions.

    Args:
        f (jnp.ndarray): input signal with dimensions (batch, theta, phi, input_channels)
        euler_angs (tuple): Euler angles (alpha, beta, gamma) in 'zyz' convention
        precomps (Precompute_Sph): precomputed spherical harmonics kernels

    Returns:
        rotated signal with same dimensions as input f
    """
    L = get_bw_from_s2_shape(f.shape, precomps.sampling)
    reality = True
    flm = s2fft.precompute_transforms.spherical.forward_transform_jax(f, precomps.get_kernel(L,
                                                                                             'forward',
                                                                                             reality=reality),
                                                                      L,
                                                                      sampling=precomps.sampling,
                                                                      reality=reality,
                                                                      spin=0,
                                                                      nside=None)
    flm_rot = rotate_flms(flm, L, euler_angs)
    f_rot = s2fft.precompute_transforms.spherical.inverse_transform_jax(flm_rot,
                                                                        precomps.get_kernel(L,
                                                                                            'backward', reality=reality),
                                                                        L,
                                                                        sampling=precomps.sampling,
                                                                        reality=reality,
                                                                        spin=0,
                                                                        nside=None)
    return jnp.real(f_rot)


def generate_smooth_flm(L_gen: int, L_res: int, rng: np.random.Generator, reality: bool) -> np.ndarray:
    """Generate a smooth random signal on the sphere by sampling only random Fourier coefficients lower
    than the bandlimit.

    Args:
        L_gen(int): signal generation bandlimit
        L_res(int): signal resolution bandlimit
        rng(np.random.Generator): random number generator
        reality(bool): whether the signal should be real
    Returns:
        np.ndarray: random Fourier coefficients with shape (L, 2L+1)
    """
    flm = s2fft.utils.signal_generator.generate_flm(rng, L_gen, reality=reality)
    flm = np.pad(flm, ((0, L_res - L_gen), (L_res - L_gen,) * 2))
    return flm


# def generate_smooth_flm_batch(L_gen: int, L_res: int, rng: np.random.Generator, reality: bool, axes_sizes:
#                               tuple[int, ...]) -> np.ndarray:
#     """Generate a batch of smooth random signals on the sphere by sampling only random Fourier
#     coefficients lower than the bandlimit.
#
#     Args:
#         L_gen(int): signal generation bandlimit
#         L_res(int): signal resolution bandlimit
#         rng(np.random.Generator): random number generator
#         reality(bool): whether the signal should be real
#         axes_sizes(tuple): batch dimensions
#     Returns:
#         np.ndarray: random Fourier coefficients with shape (*batch dimensions, L, 2L+1)
#     """
#     axes = [range(size) for size in axes_sizes]
#     flm_batch = np.empty(axes_sizes + (L_res, 2 * L_res - 1), dtype=np.complex128)
#     for indices in product(*axes):
#         flm_batch[indices] = generate_smooth_flm(L_gen, L_res, rng, reality=reality)
#     return flm_batch


def batchify_func(batch_sizes, fn_output_shape, dtype=np.float64):
    def batchify_decorator(fn):
        @wraps(fn)
        def batchified_fn(*args, **kwargs):
            axes = [range(size) for size in batch_sizes]
            result = np.empty(batch_sizes + fn_output_shape, dtype=dtype)
            for indices in product(*axes):
                result[indices] = fn(*args, **kwargs)
            return result
        return batchified_fn
    return batchify_decorator


def generate_random_s2_signal_batch(L_gen: int, L_res: int, precomps: Precompute_Sph, rng: np.random.Generator,
                                    batch_size: int, channels: int) -> jax.Array:
    """Generate a random signal on the sphere with batch and channel dimensions.

    Args:
        L_gen(int): signal generation bandlimit
        L_res(int): signal resolution
        precomps(Precompute): precomputed spherical harmonics kernels
        rng(np.random.Generator): random number generator
        batch_size(int): batch size
        channels(int): number of input channels

    Returns:
        jax.Array: random signal on the sphere with dimensions (batch, theta, phi,
        input_channels)
    """
    sampling = precomps.sampling
    reality = True

    if L_res < L_gen:
        raise ValueError("L_res must be larger than L_gen")

    generate_flm_batch = batchify_func((batch_size, channels),
                                       s2fft.sampling.s2_samples.flm_shape(L_res),
                                       dtype=np.complex128)(generate_smooth_flm)
    flm_batch = generate_flm_batch(L_gen, L_res, rng, reality)
    fourier_inv = partial(s2fft.precompute_transforms.spherical.inverse_transform_jax,
                          kernel=precomps.get_kernel(L_res, 'backward', reality=reality), L=L_res, sampling=sampling,
                          reality=reality, spin=0, nside=None)
    fourier_inv = vmap_subarray_fn(fourier_inv, 'bilm->bghi', 'lm->gh')
    signal = fourier_inv(flm_batch)
    if reality:
        signal = jnp.real(signal)
    return signal


def generate_smooth_flmn(L_gen: int, L_res: int, rng: np.random.Generator, reality: bool) -> np.ndarray:
    """Generate a smooth random signal on SO(3) by sampling only random Fourier coefficients lower
    than the bandlimit.

    Args:
        L_gen(int): signal generation bandlimit
        L_res(int): signal resolution bandlimit
        rng(np.random.Generator): random number generator
        reality(bool): whether the signal should be real
    Returns:
        np.ndarray: random Fourier coefficients with shape (l, m, n)
    """
    flmn = s2fft.utils.signal_generator.generate_flmn(rng, L_gen, L_gen, reality=True)
    flmn = np.pad(flmn, ((L_res - L_gen,) * 2, (0, L_res - L_gen), (L_res - L_gen,) * 2))
    return flmn


def generate_random_so3_signal_batch(L_gen: int, L_res: int, precomps: Precompute_Wig, rng:
                                     np.random.Generator, batch_size: int,
                                     channels: int) -> jax.Array:
    """Generate a random signal on SO(3) with batch and channel dimensions.

    Args:
        L_gen(int): signal generation bandlimit
        precomps(Precompute): precomputed Wigner kernels
        rng(np.random.Generator): random number generator
        batch_size(int): batch size
        channels(int): number of input channels
    Returns:
        jax.Array: random signal on SO(3) with dimensions (batch, gamma, beta, alpha,
            input_channels)
    """
    sampling = precomps.sampling
    reality = True

    if L_res < L_gen:
        raise ValueError("L_res must be larger than or equal L_gen")

    generate_flmn_batch = batchify_func((batch_size, channels),
                                        s2fft.sampling.so3_samples.flmn_shape(L_res, L_res),
                                        dtype=np.complex128)(generate_smooth_flmn)
    flmn_batch = generate_flmn_batch(L_gen, L_res, rng, reality=reality)
    fourier_inv = partial(s2fft.precompute_transforms.wigner.inverse_transform_jax,
                          kernel=precomps.get_kernel(L_res, 'backward', reality=reality), L=L_res, N=L_res,
                          sampling=sampling, reality=reality, nside=None)
    fourier_inv = vmap_subarray_fn(fourier_inv, 'binlm->bgshi', 'nlm->gsh')
    signal = fourier_inv(flmn_batch)
    return jnp.real(signal)


def generate_random_so3_kernel(L_gen: int, L_res: int, batch_size: int, precomps: Precompute_Wig, rng: np.random.Generator) -> jax.Array:
    f1 = generate_random_so3_signal_batch(L_gen, L_res, precomps, rng, batch_size, 1)
    f1 = jnp.squeeze(f1, axis=-1)
    f2 = generate_random_so3_signal_batch(L_gen, L_res, precomps, rng, batch_size, 1)
    f2 = jnp.squeeze(f2, axis=-1)
    kernel = jnp.einsum('aghi,bstu->abgshtiu', f1, f2)

    return kernel


def rotate_flmns(flmn: jax.Array, euler_angs: tuple[float, float, float]) -> jax.Array:
    r"""Rotate fourier coefficients of a signal on SO(3)

    Computes
    .. math:: \hat{f}^l_{mn} \rightarrow \sum_{m}\hat{f}^l_{mn} \mathcal{D}^l_{pm}(R)}

    Args:
        flmn(jnp.ndarray): Fourier coefficients with dimensions(n, l, m)
        euler_angs(tuple): Euler angles(alpha, beta, gamma) in 'zyz' convention

    Returns:
        rotated Fourier coefficients with same dimensions as input flmn
    """
    L = flmn.shape[1]
    euler_angs = (euler_angs[0] % (2 * pi), euler_angs[1] % pi, euler_angs[2] % (2 * pi))
    Dlpm = create_wigner(L, euler_angs)
    return jnp.einsum('nlm,lpm->nlp', flmn, Dlpm)


def rotate_so3_signal(f: jax.Array, euler_angs: tuple[float, float, float], precomps:
                      Precompute_Wig) -> jax.Array:
    """Rotate a signal on SO(3) via Fourier space, i.e. compute :math:`f(R^{-1}Q)`.

    Args:
        f(jnp.ndarray): input signal with dimensions(gamma, beta, alpha)
        euler_angs(tuple): Euler angles(alpha, beta, gamma) in 'zyz' convention parametrizing
            :math:`R`
        precomps(Precompute): precomputed Wigner kernels

    Returns:
        rotated signal with same dimensions as input f
    """

    L = get_bw_from_so3_shape(f.shape, precomps.sampling)
    sampling = precomps.sampling

    reality = True
    flmn = s2fft.precompute_transforms.wigner.forward_transform_jax(f, precomps.get_kernel(L,
                                                                                           'forward',
                                                                                           reality=reality), L, L,
                                                                    sampling=sampling,
                                                                    reality=reality, nside=None)
    flmn_rot = rotate_flmns(flmn, euler_angs)
    f_rot = s2fft.precompute_transforms.wigner.inverse_transform_jax(flmn_rot,
                                                                     precomps.get_kernel(L,
                                                                                         'backward',
                                                                                         reality=reality), L, L,
                                                                     sampling=sampling,
                                                                     reality=reality,
                                                                     nside=None)
    f_rot = jnp.real(f_rot)
    return f_rot


def rotate_batch_so3_signal(f: jax.Array, euler_angs: tuple[float, float, float], precomps:
                            Precompute_Wig) -> jax.Array:
    """Rotate a signal on SO(3) via Fourier space, i.e. compute :math:`f(R^{-1}Q)`.

    This also handles batch and input channel dimensions.

    Args:
        f(jnp.ndarray): input signal with dimensions(batch, gamma, beta, alpha, input_channels)
        euler_angs(tuple): Euler angles(alpha, beta, gamma) in 'zyz' convention parametrizing
            :math:`R`
        precomps(Precompute): precomputed Wigner kernels

    Returns:
        rotated signal with same dimensions as input f
    """
    fn = vmap(rotate_so3_signal, in_axes=(-1, None, None), out_axes=-1)  # Channel dimension
    fn = vmap(fn, in_axes=(0, None, None), out_axes=0)  # Batch dimension
    return fn(f, euler_angs, precomps)


def rotate_kernel_args(kernel: jax.Array, euler_angs_1: tuple[float, float, float], euler_angs_2:
                       tuple[float, float, float], right: bool, precomps: Precompute_Wig) -> jax.Array:
    """Apply rotation to kernel arguments on SO(3) from the right via Fourier space, i.e. given rotations Q1, Q2
    (in euler angles), compute
    kernel(R Q1, R' Q2)

    Args:
        kernel(Array): input kernel(batch1, batch2, gam, gam', bet, bet', alph, alph')
        euler_angs_1 (tuple): Euler angles (alpha, beta, gamma) of the first argument in 'zyz' convention
        euler_angs_2 (tuple): Euler angles (alpha, beta, gamma) of the first argument in 'zyz' convention
        right (bool): whether to rotate from the right or left
        precomps (Precompute): precomputed Wigner kernels
    Returns:
        rotated kernel
    """
    b_str = 'ab'
    L = get_bw_from_so3_shape(kernel.shape[2:7:2], precomps.sampling)
    # L = kernel.shape[4] // 2
    fourier_kn = partial(s2fft.precompute_transforms.wigner.forward_transform_jax,
                         kernel=precomps.get_kernel(L, 'forward', reality=False), L=L, N=L, sampling=precomps.sampling,
                         reality=False, nside=None)
    fourier_first = vmap_subarray_fn(fourier_kn, f'{b_str}gshtiu->{b_str}nsltmu', 'ghi->nlm')
    fourier_second = vmap_subarray_fn(fourier_kn, f'{b_str}nsltmu->{b_str}nqlomp', 'stu->qop')

    kernel_F = fourier_first(fourier_second(kernel))  # (a,b,n,n',l,l',m,m')

    D1 = create_wigner(L, euler_angs_1)  # (lmn)
    D2 = create_wigner(L, euler_angs_2)  # (lmn)

    if right:
        kernel_F = jnp.einsum('...nqlomp,lrn->...rqlomp', kernel_F, jnp.conjugate(D1))  # contract n
        kernel_F = jnp.einsum('...rqlomp,ovq->...rvlomp', kernel_F, jnp.conjugate(D2))  # contract q
    else:
        kernel_F = jnp.einsum('...nqlomp,lmr->...nqlorp', kernel_F, jnp.conjugate(D1))  # contract m
        kernel_F = jnp.einsum('...nqlorp,opv->...nqlorv', kernel_F, jnp.conjugate(D2))  # contract p

    fourier_inv_kn = partial(s2fft.precompute_transforms.wigner.inverse_transform_jax,
                             kernel=precomps.get_kernel(L, 'backward', reality=False), L=L, N=L, sampling=precomps.sampling,
                             reality=False, nside=None)
    fourier_inv_second = vmap_subarray_fn(fourier_inv_kn, f'{b_str}nqlomp->{b_str}nsltmu', 'qop->stu')
    fourier_inv_first = vmap_subarray_fn(fourier_inv_kn, f'{b_str}nsltmu->{b_str}gshtiu', 'nlm->ghi')

    kernel_rot = fourier_inv_second(fourier_inv_first(kernel_F))
    kernel_rot = jnp.real(kernel_rot)
    return kernel_rot


def calc_downscaled_bws(L_max):
    bws = [L_max]
    while bws[-1] % 2 == 0 and bws[-1] > 2:
        bws.append(bws[-1] // 2)
    return bws
