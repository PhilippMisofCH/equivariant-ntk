import pytest
from equivariant_ntk.layers import SO3ConvSO3
import s2_so3_utils
from math import pi
import jax.numpy as jnp


@pytest.fixture()
def c_in():
    return 3


@pytest.fixture()
def c_out():
    return 3


@pytest.fixture()
def finite_so3convso3(c_out, max_beta, out_bw, precomps_wig):
    return SO3ConvSO3(c_out, max_beta, (out_bw, out_bw // 2), precomps_wig)[:2]


@pytest.fixture()
def infinite_so3convso3(max_beta, out_bw, precomps_wig):
    return SO3ConvSO3(1, max_beta, (out_bw, out_bw // 2), precomps_wig)[2]


@pytest.mark.parametrize(
    "max_beta, out_bw",
    [
        (pi / 4, 4),
        (pi, 8),
    ],
)
class TestSO3ConvSO3Finite:
    def test_shape(self, finite_so3convso3, so3_signal, precomps_wig, rand_key):
        init_fn, apply_fn = finite_so3convso3

        output_shape, params = init_fn(rand_key, so3_signal.shape)
        output = apply_fn(params, so3_signal)
        assert output_shape == output.shape

    @pytest.mark.parametrize(
        "euler_angles",
        [
            pytest.param((0.0, 0.0, 0.0), id="identity"),
            pytest.param(s2_so3_utils.create_rnd_euler_angles(), id="random angles"),
        ],
    )
    def test_equivariance(
        self, finite_so3convso3, euler_angles, so3_signal, precomps_wig, rand_key
    ):
        init_fn, apply_fn = finite_so3convso3
        _, params = init_fn(rand_key, so3_signal.shape)

        f_in_pre_rotated = s2_so3_utils.rotate_batch_so3_signal(
            so3_signal, euler_angles, precomps_wig
        )
        f_out_pre_rotated = apply_fn(params, f_in_pre_rotated)
        f_out = apply_fn(params, so3_signal)
        f_out_post_rotated = s2_so3_utils.rotate_batch_so3_signal(
            f_out, euler_angles, precomps_wig
        )
        assert jnp.allclose(f_out_pre_rotated, f_out_post_rotated, atol=1e-8)


@pytest.mark.parametrize(
    "max_beta, out_bw",
    [
        (pi, 4),
    ],
)
class TestSO3ConvSO3Infinite:
    @pytest.mark.parametrize(
        "euler_angles",
        [
            pytest.param((0.0, 0.0, 0.0), id="identity"),
            pytest.param(s2_so3_utils.create_rnd_euler_angles(), id="random angles"),
        ],
    )
    def test_kernel_simultan_shift_invariance(
        self, infinite_so3convso3, euler_angles, precomps_wig, so3_signal
    ):
        kernel_fn = infinite_so3convso3
        ntk = kernel_fn(so3_signal, None, "ntk")
        right_mul = True
        ntk_rot = s2_so3_utils.rotate_kernel_args(
            ntk, euler_angles, euler_angles, right_mul, precomps_wig
        )
        assert jnp.allclose(ntk, ntk_rot, atol=1e-8)

    @pytest.mark.parametrize(
        "euler_angles1, euler_angles2",
        [
            pytest.param(
                s2_so3_utils.create_rnd_euler_angles(),
                s2_so3_utils.create_rnd_euler_angles(),
                id="random angles",
            ),
        ],
    )
    def test_kernel_input_equivariance(
        self,
        infinite_so3convso3,
        so3_signal,
        euler_angles1,
        euler_angles2,
        precomps_wig,
        rng,
    ):
        rot1_signal = s2_so3_utils.rotate_batch_so3_signal(
            so3_signal, euler_angles1, precomps_wig
        )
        rot2_signal = s2_so3_utils.rotate_batch_so3_signal(
            so3_signal, euler_angles2, precomps_wig
        )

        kernel_fn = infinite_so3convso3
        ntk_signal_rot = kernel_fn(rot1_signal, rot2_signal, "ntk")

        ntk = kernel_fn(so3_signal, None, "ntk")
        inv_euler_angles1 = s2_so3_utils.invert_euler_angles(euler_angles1)
        inv_euler_angles2 = s2_so3_utils.invert_euler_angles(euler_angles2)
        right_mul = False
        ntk_kernel_rot = s2_so3_utils.rotate_kernel_args(
            ntk, inv_euler_angles1, inv_euler_angles2, right_mul, precomps_wig
        )
        assert jnp.allclose(ntk_signal_rot, ntk_kernel_rot, atol=1e-8)
