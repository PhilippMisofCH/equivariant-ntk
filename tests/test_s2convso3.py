from math import pi
import pytest
import s2_so3_utils
from equivariant_ntk.layers import S2ConvSO3
import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)


@pytest.fixture()
def c_in():
    return 3

@pytest.fixture()
def c_out():
    return 3


@pytest.fixture()
def finite_s2convso3(c_out, max_beta, out_bw, precomps_sph, precomps_wig):
    return S2ConvSO3(c_out, max_beta,
                     (out_bw, out_bw // 2),
                     precomps_sph,
                     precomps_wig)[:2]


@pytest.fixture()
def infinite_s2convso3(max_beta, out_bw, precomps_sph, precomps_wig):
    return S2ConvSO3(1, max_beta,
                     (out_bw, out_bw // 2),
                     precomps_sph,
                     precomps_wig)[2]


@pytest.mark.parametrize("max_beta, out_bw", [
    (pi / 4, 4),
    (pi, 4),
])
class TestS2ConvSO3Finite:

    def test_shape(self, finite_s2convso3, s2_signal, precomps_sph, rand_key):
        init_fn, apply_fn = finite_s2convso3

        output_shape, params = init_fn(rand_key, s2_signal.shape)
        output = apply_fn(params, s2_signal)
        assert output_shape == output.shape

    @pytest.mark.parametrize("euler_angles", [
        pytest.param((0.0, 0.0, 0.0), id="identity"),
        pytest.param(s2_so3_utils.create_rnd_euler_angles(), id="random angles"),
    ])
    def test_equivariance(self, finite_s2convso3, euler_angles, s2_signal, precomps_sph,
                          precomps_wig, rand_key):
        init_fn, apply_fn = finite_s2convso3
        _, params = init_fn(rand_key, s2_signal.shape)

        f_in_pre_rotated = s2_so3_utils.rotate_batch_s2_signal(s2_signal, euler_angles, precomps_sph)
        f_out_pre_rotated = apply_fn(params, f_in_pre_rotated)
        f_out = apply_fn(params, s2_signal)
        f_out_post_rotated = s2_so3_utils.rotate_batch_so3_signal(f_out, euler_angles, precomps_wig)

        assert jnp.allclose(f_out_post_rotated, f_out_pre_rotated, atol=1e-8)


@pytest.mark.parametrize("max_beta, out_bw", [
    (pi, 4),
    # pytest.param(pi / 4, 4, marks=pytest.mark.xfail),
])
class TestS2ConvSO3Infinite:

    @pytest.mark.parametrize("euler_angles", [
        pytest.param((0.0, 0.0, 0.0), id="identity"),
        pytest.param(s2_so3_utils.create_rnd_euler_angles(), id="random angles"),
    ])
    def test_kernel_simultan_shift_invariance(self, infinite_s2convso3, euler_angles, precomps_wig, s2_signal):
        kernel_fn = infinite_s2convso3

        ntk = kernel_fn(s2_signal, None, "ntk")
        right_mul = True
        ntk_rot = s2_so3_utils.rotate_kernel_args(ntk, euler_angles, euler_angles, right_mul, precomps_wig)
        assert jnp.allclose(ntk, ntk_rot, atol=1e-8)

    @pytest.mark.parametrize("euler_angles1, euler_angles2", [
        pytest.param(s2_so3_utils.create_rnd_euler_angles(), s2_so3_utils.create_rnd_euler_angles(), id="random angles"),
    ])
    def test_kernel_input_equivariance(self, infinite_s2convso3, s2_signal, euler_angles1,
                                       euler_angles2, precomps_sph,
                                       precomps_wig, rng):
        rot1_signal = s2_so3_utils.rotate_batch_s2_signal(s2_signal, euler_angles1, precomps_sph)
        rot2_signal = s2_so3_utils.rotate_batch_s2_signal(s2_signal, euler_angles2, precomps_sph)

        kernel_fn = infinite_s2convso3
        ntk_signal_rot = kernel_fn(rot1_signal, rot2_signal, "ntk")

        ntk = kernel_fn(s2_signal, None, "ntk")
        inv_euler_angles1 = s2_so3_utils.invert_euler_angles(euler_angles1)
        inv_euler_angles2 = s2_so3_utils.invert_euler_angles(euler_angles2)
        right_mul = False
        ntk_kernel_rot = s2_so3_utils.rotate_kernel_args(ntk, inv_euler_angles1, inv_euler_angles2,
                                                         right_mul, precomps_wig)

        assert jnp.allclose(ntk_signal_rot, ntk_kernel_rot, atol=1e-8)
