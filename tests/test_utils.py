from s2_so3_utils import euler_conv
import s2_so3_utils
import utils
from equivariant_ntk.utils.so3 import Precompute_Wig, Precompute_Sph
from math import pi
from scipy.spatial.transform import Rotation
import jax.numpy as jnp
import pytest
import numpy as np
import s2fft
import jax
import random

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize(
    "euler_angles,L_res,L_gen",
    [
        ((0.0, 0.0, 0.0), 8, 4),
        (s2_so3_utils.create_rnd_euler_angles(), 8, 4),
    ],
)
def test_s2_rotation(euler_angles, L_res, L_gen, rng):
    precomps = Precompute_Sph(sampling="dh")
    precomps.compute_kernels([L_res])
    f = s2_so3_utils.generate_random_s2_signal_batch(
        L_gen, L_res, precomps, rng, batch_size=2, channels=3
    )
    f_rot = s2_so3_utils.rotate_batch_s2_signal(f, euler_angles, precomps)
    euler_angles_inv = s2_so3_utils.invert_euler_angles(euler_angles)
    f_back = s2_so3_utils.rotate_batch_s2_signal(f_rot, euler_angles_inv, precomps)
    assert jnp.allclose(f, f_back, atol=1e-8), (
        "Rotating S2 signal back and forth does not give the original signal"
    )


@pytest.mark.parametrize(
    "euler_angles, L", [(s2_so3_utils.create_rnd_euler_angles(), 4)]
)
def test_wigner_D(euler_angles, L):
    d = s2_so3_utils.create_wigner(L, euler_angles)
    d_inv = s2_so3_utils.create_wigner(
        L, s2_so3_utils.invert_euler_angles(euler_angles)
    )
    d_herm = jnp.transpose(jnp.conjugate(d), (0, 2, 1))
    assert jnp.allclose(d_herm, d_inv, atol=1e-8), (
        "Wigner D matrix for inverse rotation is not the hermitian transpose"
    )


def test_wigner_D1mn():
    Rot = Rotation.from_euler(euler_conv, (0, pi / 4, 0))
    d = s2_so3_utils.create_wigner(2, Rot.as_euler(euler_conv))
    d1_expected = jnp.array(
        [
            [0.85355339 + 0.0j, 0.5 + 0.0j, 0.14644661 + 0.0j],
            [-0.5 + 0.0j, 0.70710678 + 0.0j, 0.5 + 0.0j],
            [0.14644661 + 0.0j, -0.5 + 0.0j, 0.85355339 + 0.0j],
        ]
    )
    assert jnp.allclose(d[1, :, :], d1_expected, atol=1e-8), (
        "Wigner D matrix is incorrect"
    )


@pytest.mark.parametrize(
    "euler_angles,L_res,L_gen",
    [
        ((0.0, 0.0, 0.0), 8, 4),
        (s2_so3_utils.create_rnd_euler_angles(), 8, 4),
    ],
)
def test_so3_rotation_Fourier(euler_angles, L_res, L_gen):
    rng = np.random.default_rng()
    flmn = s2fft.utils.signal_generator.generate_flmn(rng, L_gen, L_gen, reality=True)
    flmn = jnp.pad(
        flmn, ((L_res - L_gen,) * 2, (0, L_res - L_gen), (L_res - L_gen,) * 2)
    )
    flmn_rot = s2_so3_utils.rotate_flmns(flmn, euler_angles)
    flmn_back = s2_so3_utils.rotate_flmns(
        flmn_rot, s2_so3_utils.invert_euler_angles(euler_angles)
    )
    assert jnp.allclose(flmn, flmn_back, atol=1e-8), (
        "Rotating SO3 signal in Fourier space back "
        "and forth does not give the original signal"
    )


@pytest.mark.parametrize(
    "L, euler_angles",
    [
        pytest.param(32, jnp.array([0, 0, 0]), id="identity"),
        pytest.param(32, s2_so3_utils.create_rnd_euler_angles(), id="random angles"),
    ],
)
def test_so3_rotation_identity(
    L: int, euler_angles: tuple[float, float, float], rng: np.random.Generator
):
    sampling = "dh"
    precomps = Precompute_Wig(sampling=sampling)
    precomps.compute_kernels([L])

    inv_euler_angles = s2_so3_utils.invert_euler_angles(euler_angles)

    f_in = s2_so3_utils.generate_random_so3_signal_batch(L // 2, L, precomps, rng, 2, 2)
    f_rot = s2_so3_utils.rotate_batch_so3_signal(f_in, euler_angles, precomps)
    f_rot_back = s2_so3_utils.rotate_batch_so3_signal(f_rot, inv_euler_angles, precomps)
    assert jnp.allclose(f_in, f_rot_back, atol=1e-8), (
        "Rotating SO3 signal back and forth by"
        f"{euler_angles} does not"
        " give the original signal"
    )


@pytest.mark.parametrize("right_mul", [True, False])
@pytest.mark.parametrize(
    "L, euler_angles1, euler_angles2",
    [
        pytest.param(6, jnp.array([0, 0, 0]), jnp.array([0, 0, 0]), id="identity"),
        pytest.param(
            4,
            s2_so3_utils.create_rnd_euler_angles(),
            s2_so3_utils.create_rnd_euler_angles(),
            id="random angles",
        ),
    ],
)
def test_so3_kernel_rotation_identity(
    L: int,
    euler_angles1: tuple[float, float, float],
    euler_angles2: tuple[float, float, float],
    right_mul: bool,
    rng: np.random.Generator,
):
    precomps_wig = Precompute_Wig(sampling="dh")
    precomps_wig.compute_kernels([L])
    inv_euler_angles1 = s2_so3_utils.invert_euler_angles(euler_angles1)
    inv_euler_angles2 = s2_so3_utils.invert_euler_angles(euler_angles2)

    kernel = s2_so3_utils.generate_random_so3_kernel(L // 2, L, 2, precomps_wig, rng)

    kernel_rot = s2_so3_utils.rotate_kernel_args(
        kernel, euler_angles1, euler_angles2, right_mul, precomps_wig
    )
    kernel_rot_back = s2_so3_utils.rotate_kernel_args(
        kernel_rot, inv_euler_angles1, inv_euler_angles2, right_mul, precomps_wig
    )

    assert jnp.allclose(kernel, kernel_rot_back, atol=1e-8, rtol=1e-8), (
        "Rotating SO3 kernel back "
        "and forth by "
        f"{euler_angles1}, "
        f"{euler_angles2} does not"
        " give the original kernel"
    )


@pytest.mark.parametrize("L", [4])
def test_so3_kernel_rotation_alpha(L: int, rng: np.random.Generator):
    precomps_wig = Precompute_Wig(sampling="dh")
    precomps_wig.compute_kernels([L])

    sampling = precomps_wig.sampling
    n_alphas = s2fft.sampling.s2_samples.nphi_equiang(L, sampling)
    a_ind_1 = random.choice(range(n_alphas))
    a_ind_2 = random.choice(range(n_alphas))
    alpha1 = s2fft.sampling.s2_samples.p2phi_equiang(L, a_ind_1, sampling)
    alpha2 = s2fft.sampling.s2_samples.p2phi_equiang(L, a_ind_2, sampling)
    # print(f"alphas: {alpha1}, {alpha2}")

    euler_angles1 = (alpha1, 0.0, 0.0)
    euler_angles2 = (alpha2, 0.0, 0.0)
    inv_euler_angles1 = s2_so3_utils.invert_euler_angles(euler_angles1)
    inv_euler_angles2 = s2_so3_utils.invert_euler_angles(euler_angles2)

    kernel = s2_so3_utils.generate_random_so3_kernel(L // 2, L, 2, precomps_wig, rng)

    right_mul = False
    kernel_rot = s2_so3_utils.rotate_kernel_args(
        kernel, inv_euler_angles1, inv_euler_angles2, right_mul, precomps_wig
    )

    rotated_a_ind_1 = jnp.mod(jnp.arange(0, n_alphas) - a_ind_1, n_alphas)
    rotated_a_ind_2 = jnp.mod(jnp.arange(0, n_alphas) - a_ind_2, n_alphas)
    expected_kernel = kernel[:, :, :, :, :, :, rotated_a_ind_1, :]
    expected_kernel = expected_kernel[:, :, :, :, :, :, :, rotated_a_ind_2]
    assert jnp.allclose(expected_kernel, kernel_rot)
