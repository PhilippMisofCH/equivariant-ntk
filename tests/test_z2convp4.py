import pytest
import jax
import jax.numpy as jnp
import numpy as onp
from equivariant_ntk.layers import Z2ConvP4
from neural_tangents.stax import Padding

from utils import (
    random_kernel,
    transform_p4,
    transform_p4kernelp4,
    transform_z2,
    transform_z2kernelp4,
)
from common_args import roto_trans_params

z4_inits = pytest.mark.parametrize(
    "parameterization, W_std, b_std, c_out, c_in, s",
    [
        ("ntk", 2.0, 3.0, 3, 3, (5, 6)),
        ("standard", 2.0, 3.0, 3, 3, (5, 6)),
    ],
)


@z4_inits
@roto_trans_params
@pytest.mark.parametrize(
    "filter_size, img_size",
    [
        (3, 3),
        (5, 10),
    ],
)
def test_z2convp4(
    parameterization, W_std, b_std, c_out, c_in, s, filter_size, img_size, roto_trans
):
    filter_size = 2 * (filter_size,)
    img_size = 2 * (img_size,)
    init_fn, apply_fn, _ = Z2ConvP4(
        c_out,
        filter_size,
        padding=Padding.SAME,
        s=s,
        parameterization=parameterization,
        W_std=W_std,
        b_std=b_std,
    )
    input_shape = (2, *img_size, c_in)

    t = roto_trans.t
    t0, t1 = roto_trans.t
    r = roto_trans.r
    t_pad = max(abs(t0), abs(t1)) + (filter_size[0] - 1) // 2

    key1, key2 = jax.random.split(jax.random.key(0), num=2)
    batch = jax.random.normal(key1, input_shape)
    pad_width = ((0, 0),) + 2 * ((t_pad, t_pad),) + ((0, 0),)
    batch = jnp.pad(batch, pad_width)
    input_shape = batch.shape

    output_shape, params = init_fn(key2, input_shape)

    output = apply_fn(params, batch)
    assert output_shape == output.shape, f"expected {output_shape}, got {output.shape}"
    output = transform_p4(output, r=r, t=(t0, t1))

    rot_batch = transform_z2(batch, r=r, t=(t0, t1))
    rot_output = apply_fn(params, rot_batch)
    assert jnp.allclose(rot_output, output, atol=1e-5), (
        f"equivariance failed for r={r}, t={t}"
    )


@z4_inits
def test_z2convp4_parameterization(parameterization, W_std, b_std, c_out, c_in, s):
    filter_size = (100, 100)
    init_fn, apply_fn, _ = Z2ConvP4(
        c_out,
        filter_size,
        padding=Padding.SAME,
        s=s,
        parameterization=parameterization,
        W_std=W_std,
        b_std=b_std,
    )
    input_shape = (2, 500, 500, c_in)

    key, key2 = jax.random.split(jax.random.key(0), num=2)
    output_shape, params = init_fn(key, input_shape)

    if parameterization == "ntk":
        var_goal = 1.0
    elif parameterization == "standard":
        var_goal = W_std**2 / (c_in * onp.prod(filter_size) / s[0])
    assert jnp.allclose(jnp.var(params[0]), var_goal, atol=1e-2)


@roto_trans_params
def test_kernel_z2convp4_left(roto_trans):
    init_fn, apply_fn, kernel_fn = Z2ConvP4(3, (3, 3), padding=Padding.SAME)
    kernel_shape = (2, 2, 5, 5, 5, 5)

    key1, key2, key3 = jax.random.split(jax.random.key(0), num=3)
    rnd_kernel = random_kernel(key1, kernel_shape, 3)
    t0, t1 = roto_trans.t
    t_pad = max(abs(t0), abs(t1)) + (3 - 1) // 2

    padding = 2 * ((0, 0),) + 2 * ((t_pad, t_pad),) + 2 * ((t_pad, t_pad),)
    rnd_pad_kernel = rnd_kernel.replace(
        nngp=jnp.pad(rnd_kernel.nngp, pad_width=padding)
    )

    # apply kernel and then rotate
    kernel = kernel_fn(rnd_pad_kernel)
    rot_kernel = transform_p4kernelp4(kernel, roto_trans.r, (t0, t1))

    # rotate and then apply kernel
    rot_rnd_kernel = transform_z2kernelp4(rnd_pad_kernel, roto_trans.r, (t0, t1))
    kernel_rot = kernel_fn(rot_rnd_kernel)

    assert jnp.allclose(kernel_rot.nngp, rot_kernel.nngp, atol=1e-5)


@pytest.mark.parametrize(
    "padding, exp_output_shape",
    [
        (Padding.SAME, (2, 4, 16, 16, 10)),
        (Padding.CIRCULAR, (2, 4, 16, 16, 10)),
        (Padding.VALID, (2, 4, 14, 14, 10)),
    ],
)
def test_z2convp4_paddings(padding, exp_output_shape):
    init_fn, apply_fn, kernel_fn = Z2ConvP4(10, (3, 3), padding=padding)
    input_shape = (2, 16, 16, 3)

    key1, key2 = jax.random.split(jax.random.key(0), num=2)
    output_shape, params = init_fn(key1, input_shape)

    batch = jax.random.normal(key2, input_shape)
    output = apply_fn(params, batch)

    assert output.shape == output_shape
    assert output.shape == exp_output_shape
