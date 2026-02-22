import pytest
from math import pi
from neural_tangents import stax
from equivariant_ntk.layers import S2ConvSO3, SO3ConvSO3, SO3Pool
from equivariant_ntk.utils.so3 import Precompute_Sph, Precompute_Wig
import s2_so3_utils
import jax.numpy as jnp


def model(linear: bool, pooling: bool, out_bw: int, max_beta: float, precomps_sph: Precompute_Sph,
          precomps_wig: Precompute_Wig):
    layers = [S2ConvSO3(10, max_beta, (out_bw, out_bw), precomps_sph, precomps_wig),
              stax.Erf() if not linear else None,
              SO3ConvSO3(10, max_beta, (out_bw, out_bw), precomps_wig),
              stax.Erf() if not linear else None,
              SO3Pool() if pooling else None
              ]
    layers = list(filter(None, layers))
    return stax.serial(*layers)


@pytest.mark.parametrize("c_in, max_beta, out_bw", [(3, pi, 15)])
@pytest.mark.parametrize("euler_angles", [
    pytest.param(s2_so3_utils.create_rnd_euler_angles(), id="random angles"),
])
class TestS2NNFinite:

    @pytest.mark.parametrize("linear, tol", [(True, 1e-8), (False, 1e-2)])
    def test_finite_s2_model_invariance(self, linear, euler_angles, s2_signal, out_bw,
                                        max_beta, precomps_sph, precomps_wig, rand_key,
                                        tol):
        pooling = True
        init_fn, apply_fn, _ = model(linear, pooling, out_bw, max_beta, precomps_sph, precomps_wig)
        _, params = init_fn(rand_key, s2_signal.shape)

        f_in_rot = s2_so3_utils.rotate_batch_s2_signal(s2_signal, euler_angles, precomps_sph)
        f_out = apply_fn(params, s2_signal)
        f_out_rot = apply_fn(params, f_in_rot)

        assert jnp.allclose(f_out, f_out_rot, rtol=tol)

    @pytest.mark.parametrize("linear, tol", [(True, 1e-8), (False, 1e-2)])
    def test_finite_s2_model_equivariance(self, linear, euler_angles, s2_signal, out_bw,
                                          max_beta, precomps_sph, precomps_wig, rand_key,
                                          tol):
        pooling = False
        init_fn, apply_fn, _ = model(linear, pooling, out_bw, max_beta, precomps_sph, precomps_wig)
        _, params = init_fn(rand_key, s2_signal.shape)

        f_in_rot = s2_so3_utils.rotate_batch_s2_signal(s2_signal, euler_angles, precomps_sph)
        f_out = apply_fn(params, s2_signal)
        f_out_pre_rot = apply_fn(params, f_in_rot)
        f_out_post_rot = s2_so3_utils.rotate_batch_so3_signal(f_out, euler_angles, precomps_wig)

        diff = jnp.max(jnp.abs(f_out_pre_rot - f_out_post_rot))
        assert jnp.allclose(f_out_pre_rot, f_out_post_rot, atol=tol), f"Max diff: {diff}"


@pytest.mark.parametrize("c_in, max_beta, out_bw", [(3, pi, 6)])
class TestS2NNInfinite:

    @pytest.mark.parametrize("linear, tol", [(True, 1e-8), (False, 5e-2)])
    @pytest.mark.parametrize("euler_angles", [
        pytest.param(s2_so3_utils.create_rnd_euler_angles(), id="random angles"),
    ])
    def test_kernel_s2_model_simultan_shift_invariance(self, linear, euler_angles, s2_signal,
                                                       out_bw, max_beta, precomps_sph,
                                                       precomps_wig, tol):
        pooling = False
        _, _, kernel_fn = model(linear, pooling, out_bw, max_beta, precomps_sph, precomps_wig)
        ntk = kernel_fn(s2_signal, None, 'ntk')

        right_mul = True
        ntk_rot = s2_so3_utils.rotate_kernel_args(ntk, euler_angles, euler_angles, right_mul,
                                                  precomps_wig)
        max_diff = jnp.max(jnp.abs((ntk - ntk_rot)))
        assert jnp.allclose(ntk, ntk_rot, atol=tol), f"Max diff: {max_diff}"

    @pytest.mark.parametrize("linear, tol", [(True, 1e-8), (False, 2e-2)])
    @pytest.mark.parametrize("euler_angles1, euler_angles2", [
        pytest.param(s2_so3_utils.create_rnd_euler_angles(), s2_so3_utils.create_rnd_euler_angles(), id="random angles"),
    ])
    def test_kernel_s2_model_invariance(self, linear, euler_angles1, euler_angles2, s2_signal, out_bw,
                                        max_beta, precomps_sph, precomps_wig,
                                        tol):
        pooling = True
        _, _, kernel_fn = model(linear, pooling, out_bw, max_beta, precomps_sph, precomps_wig)

        f_rot1 = s2_so3_utils.rotate_batch_s2_signal(s2_signal, euler_angles1, precomps_sph)
        f_rot2 = s2_so3_utils.rotate_batch_s2_signal(s2_signal, euler_angles2, precomps_sph)
        ntk_signal_rot = kernel_fn(f_rot1, f_rot2, 'ntk')

        ntk = kernel_fn(s2_signal, None, 'ntk')

        jnp.allclose(ntk_signal_rot, ntk, atol=tol)

    @pytest.mark.parametrize("linear, tol", [(True, 1e-8), (False, 2e-2)])
    @pytest.mark.parametrize("euler_angles1, euler_angles2", [
        pytest.param(s2_so3_utils.create_rnd_euler_angles(), s2_so3_utils.create_rnd_euler_angles(), id="random angles"),
    ])
    def test_kernel_s2_model_equivariance(self, linear, euler_angles1, euler_angles2, s2_signal, out_bw,
                                          max_beta, precomps_sph, precomps_wig,
                                          tol):
        pooling = False
        _, _, kernel_fn = model(linear, pooling, out_bw, max_beta, precomps_sph, precomps_wig)

        f_rot1 = s2_so3_utils.rotate_batch_s2_signal(s2_signal, euler_angles1, precomps_sph)
        f_rot2 = s2_so3_utils.rotate_batch_s2_signal(s2_signal, euler_angles2, precomps_sph)
        ntk_signal_rot = kernel_fn(f_rot1, f_rot2, 'ntk')

        ntk = kernel_fn(s2_signal, None, 'ntk')
        inv_euler_angles1 = s2_so3_utils.invert_euler_angles(euler_angles1)
        inv_euler_angles2 = s2_so3_utils.invert_euler_angles(euler_angles2)
        right_mul = False
        ntk_kernel_rot = s2_so3_utils.rotate_kernel_args(ntk, inv_euler_angles1,
                                                         inv_euler_angles2, right_mul,
                                                         precomps_wig)
        max_diff = jnp.max(jnp.abs((ntk_signal_rot - ntk_kernel_rot)))
        assert jnp.allclose(ntk_signal_rot, ntk_kernel_rot, atol=tol), f"Max diff: {max_diff}"
