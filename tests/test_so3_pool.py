from equivariant_ntk.layers import SO3Pool
import s2_so3_utils
import pytest
import s2fft
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


@pytest.fixture
def c_in():
    return 3


@pytest.fixture
def finite_so3pool():
    return SO3Pool()[:2]


@pytest.fixture
def infinite_so3pool():
    return SO3Pool()[2]


@pytest.fixture
def so3_const_one_signal(out_bw):
    batch_size = 2
    channels = 3
    grid_shape = s2fft.sampling.so3_samples.f_shape(out_bw, out_bw, sampling="dh")
    return jnp.ones((batch_size, *grid_shape, channels))


@pytest.fixture
def cos_gamma_signal(out_bw):
    alphas = s2fft.sampling.s2_samples.phis_equiang(out_bw, sampling="dh")
    betas = s2fft.sampling.s2_samples.thetas(out_bw, sampling="dh")
    gammas = jnp.copy(alphas)
    a, b, g = jnp.meshgrid(alphas, betas, gammas, indexing="ij")
    signal = jnp.cos(g)[None, :, :, :, None]
    mean = 0.0
    return signal, mean


@pytest.mark.parametrize("out_bw", [4])
class TestSO3PoolFinite:
    def test_shape(self, finite_so3pool, so3_signal, rand_key):
        init_fn, apply_fn = finite_so3pool

        output_shape, _ = init_fn(rand_key, so3_signal.shape)
        output = apply_fn({}, so3_signal)
        assert output.shape == output_shape

    @pytest.mark.parametrize(
        "euler_angles",
        [
            pytest.param((0, 0, 0), id="identity"),
            pytest.param(s2_so3_utils.create_rnd_euler_angles(), id="random"),
        ],
    )
    def test_invariance(
        self, finite_so3pool, euler_angles, so3_signal, precomps_wig, rand_key
    ):
        init_fn, apply_fn = finite_so3pool

        f_out = apply_fn({}, so3_signal)
        rotated_so3_signal = s2_so3_utils.rotate_batch_so3_signal(
            so3_signal, euler_angles, precomps_wig
        )
        rotated_f_out = apply_fn({}, rotated_so3_signal)
        assert jnp.allclose(f_out, rotated_f_out, atol=1e-8)

    def test_integration_volume(self, finite_so3pool, so3_const_one_signal):
        init_fn, apply_fn = finite_so3pool

        output = apply_fn({}, so3_const_one_signal)
        batch_size = so3_const_one_signal.shape[0]
        channel_size = so3_const_one_signal.shape[-1]
        expected_output = jnp.ones((batch_size, channel_size))

        assert jnp.allclose(output, expected_output)

    def test_cos_gamma_mean(self, finite_so3pool, out_bw, cos_gamma_signal):
        init_fn, apply_fn = finite_so3pool

        signal, mean = cos_gamma_signal
        output = apply_fn({}, signal)
        assert jnp.allclose(output, mean, atol=1e-8)

    # Other possible test
    # signal_3 = jnp.cos(a / 4)
    # integral = jnp.sum(signal_3 * weights[None, :, None])
    # print(f"cos(alpha/4) = {integral}")  # rather coarse for low bandwidth


@pytest.mark.parametrize("out_bw", [4])
class TestSO3PoolInfinite:
    @pytest.mark.parametrize(
        "euler_angles",
        [
            pytest.param((0.0, 0.0, 0.0), id="identity"),
            pytest.param(s2_so3_utils.create_rnd_euler_angles(), id="random angles"),
        ],
    )
    def test_kernel_invariance(
        self, infinite_so3pool, euler_angles, so3_signal, precomps_wig
    ):
        kernel_fn = infinite_so3pool
        rot_so3_signal = s2_so3_utils.rotate_batch_so3_signal(
            so3_signal, euler_angles, precomps_wig
        )
        nngp = kernel_fn(so3_signal, None, "nngp")
        nngp_rotated = kernel_fn(rot_so3_signal, None, "nngp")
        assert jnp.allclose(nngp, nngp_rotated, atol=1e-8)

    def test_kernel_integration_volume(self, infinite_so3pool, so3_const_one_signal):
        kernel_fn = infinite_so3pool
        nngp = kernel_fn(so3_const_one_signal, None, "nngp")
        batch_size = so3_const_one_signal.shape[0]
        expected_output = jnp.ones((batch_size, batch_size))

        assert jnp.allclose(nngp, expected_output)
