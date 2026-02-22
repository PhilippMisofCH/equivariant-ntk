from itertools import product
import pytest

import jax
import jax.numpy as jnp
import numpy as onp
from equivariant_ntk.layers import P4ConvP4
from neural_tangents._src.stax.linear import _conv_kernel_full_spatial_shared as a_op
from neural_tangents.stax import Padding
from utils import transform_p4, random_kernel, create_kernel, transform_p4kernelp4
from common_args import roto_trans_params


@roto_trans_params
@pytest.mark.parametrize(
    "filter_size, img_size, c_in, c_out",
    [
        (3, 3, 3, 3),
        (5, 10, 3, 3),
    ],
)
def test_p4convp4(roto_trans, filter_size, img_size, c_in, c_out):
    filter_size = 2 * (filter_size,)
    img_size = 2 * (img_size,)
    init_fn, apply_fn, _ = P4ConvP4(c_out, filter_size, padding=Padding.SAME)
    input_shape = (2, 4, *img_size, c_in)

    t0, t1 = roto_trans.t
    t_pad = max(abs(t0), abs(t1)) + (filter_size[0] - 1) // 2

    key1, key2 = jax.random.split(jax.random.key(0), num=2)
    batch = jax.random.normal(key1, input_shape)
    pad_width = 2 * ((0, 0),) + 2 * ((t_pad, t_pad),) + ((0, 0),)
    batch = jnp.pad(batch, pad_width)
    input_shape = batch.shape

    output_shape, params = init_fn(key2, input_shape)

    output = apply_fn(params, batch)
    assert output_shape == output.shape
    output = transform_p4(output, r=roto_trans.r, t=(t0, t1))

    rot_batch = transform_p4(batch, r=roto_trans.r, t=(t0, t1))
    rot_output = apply_fn(params, rot_batch)
    assert jnp.allclose(rot_output, output, atol=1e-5)


def test_kernel_p4convp4_right():
    """This test acts with rotation from the right, i.e.
            K^{r, r'} -> K^{r \tilde{r}, r' \tilde{r}}
    which leaves the NTK invariant since it sums over the entire C4 group.
    """
    init_fn, apply_fn, kernel_fn = P4ConvP4(3, (3, 3), padding=Padding.SAME)
    ouput_shape = (2, 4, 5, 5, 2)

    key1, key2, key3 = jax.random.split(jax.random.key(0), num=3)
    batch1 = jax.random.normal(key1, ouput_shape)
    batch2 = jax.random.normal(key2, ouput_shape)

    kernel = kernel_fn(batch1, batch2, "nngp")

    r = jax.random.randint(key3, (1,), 0, 3).item()
    rot_kernel = jnp.roll(jnp.roll(kernel, r, 2), r, 3)

    assert jnp.allclose(kernel, rot_kernel, atol=1e-5)


def manual_aop(kernel, filter_size):
    result = onp.zeros(kernel.shape)
    kernel = onp.array(kernel)

    npad = (filter_size - 1) // 2
    ranges = 4 * (range(kernel.shape[-1]),)
    for th, th_prime, tw, tw_prime in product(*ranges):
        result[..., th, th_prime, tw, tw_prime] = sum(
            [
                sum(
                    [
                        kernel[
                            ...,
                            min(kernel.shape[-1] - 1, th + th_tilde),
                            min(kernel.shape[-1] - 1, th_prime + th_tilde),
                            min(kernel.shape[-1] - 1, tw + tw_tilde),
                            min(kernel.shape[-1] - 1, tw_prime + tw_tilde),
                        ]
                        for th_tilde in range(-npad, npad + 1)
                    ]
                )
                for tw_tilde in range(-npad, npad + 1)
            ]
        )
    return result / (filter_size**2)


def test_aop():
    kernel_shape = (1, 1, 5, 5, 5, 5)
    filter_size = 3
    key1, key2, key3 = jax.random.split(jax.random.key(0), num=3)
    nngp = jax.random.normal(key1, kernel_shape)

    npad = (filter_size - 1) // 2
    nngp = jnp.pad(nngp, 2 * ((0, 0),) + 4 * ((npad, npad),))

    man_aop = manual_aop(nngp, filter_size)

    filter_shape = (filter_size, filter_size)
    romans_aop = a_op(
        nngp, filter_shape, strides=(1, 1), padding=Padding.SAME, batch_ndim=1
    )
    romans_aop = romans_aop.transpose(0, 1, 4, 5, 2, 3)

    assert jnp.allclose(man_aop, romans_aop, atol=1e-5)


def test_aop_with_group_idx():
    kernel_shape = (1, 1, 4, 4, 5, 5, 5, 5)
    filter_size = 3
    key1, key2, key3 = jax.random.split(jax.random.key(0), num=3)
    nngp = jax.random.normal(key1, kernel_shape)

    npad = (filter_size - 1) // 2
    nngp = jnp.pad(nngp, 4 * ((0, 0),) + 4 * ((npad, npad),))

    man_aop = manual_aop(nngp, filter_size)

    filter_shape = (filter_size, filter_size)

    # absorb group dimensions in batch dimensions
    b1, b2, g1, g2, h1, h2, w1, w2 = nngp.shape
    nngp = jnp.moveaxis(nngp, 2, 1)
    nngp = nngp.reshape((b1 * g1, b2 * g2, h1, h2, w1, w2))
    romans_aop = a_op(
        nngp, filter_shape, strides=(1, 1), padding=Padding.SAME, batch_ndim=1
    )
    romans_aop = romans_aop.transpose(0, 1, 4, 5, 2, 3)
    romans_aop = romans_aop.reshape((b1, g1, b2, g2, h1, h2, w1, w2))
    romans_aop = jnp.moveaxis(romans_aop, 1, 2)

    assert jnp.allclose(man_aop, romans_aop, atol=1e-5)


@pytest.mark.parametrize("r", range(4))
def test_kernel_p4convp4_only_left_rot(r):
    init_fn, apply_fn, kernel_fn = P4ConvP4(3, (3, 3), padding=Padding.SAME)
    kernel_shape = (2, 2, 4, 4, 5, 5, 5, 5)

    key1, key2, key3 = jax.random.split(jax.random.key(0), num=3)
    rnd_kernel = random_kernel(key1, kernel_shape, 3)
    nngp = jnp.pad(rnd_kernel.nngp, 4 * ((0, 0),) + 4 * ((3, 3),))
    rnd_kernel = rnd_kernel.replace(nngp=nngp)

    # apply kernel and then rotate
    kernel = kernel_fn(rnd_kernel)
    rot_kernel = transform_p4kernelp4(kernel, r)

    # rotate and then apply kernel
    rot_rnd_kernel = transform_p4kernelp4(rnd_kernel, r)
    kernel_rot = kernel_fn(rot_rnd_kernel)

    assert jnp.allclose(kernel_rot.nngp, rot_kernel.nngp, atol=1e-5)


@roto_trans_params
def test_kernel_p4convp4_left(roto_trans):
    init_fn, apply_fn, kernel_fn = P4ConvP4(3, (3, 3), padding=Padding.SAME)
    kernel_shape = (2, 2, 4, 4, 5, 5, 5, 5)

    key1, key2, key3 = jax.random.split(jax.random.key(0), num=3)
    rnd_kernel = random_kernel(key1, kernel_shape, 3)
    t0, t1 = roto_trans.t
    t_pad = max(abs(t0), abs(t1)) + (3 - 1) // 2

    padding = 4 * ((0, 0),) + 2 * ((t_pad, t_pad),) + 2 * ((t_pad, t_pad),)
    rnd_pad_kernel = rnd_kernel.replace(
        nngp=jnp.pad(rnd_kernel.nngp, pad_width=padding)
    )

    # apply kernel and then rotate
    kernel = kernel_fn(rnd_pad_kernel)
    rot_kernel = transform_p4kernelp4(kernel, roto_trans.r, (t0, t1))

    # rotate and then apply kernel
    rot_rnd_kernel = transform_p4kernelp4(rnd_pad_kernel, roto_trans.r, (t0, t1))
    kernel_rot = kernel_fn(rot_rnd_kernel)

    assert jnp.allclose(kernel_rot.nngp, rot_kernel.nngp, atol=1e-5)


@pytest.mark.parametrize("r", range(4))
def test_kernel_group_action_on_c4(r):
    kernel_shape = (1, 1, 4, 4, 5, 5, 5, 5)
    kernel = onp.zeros(kernel_shape)
    for i in range(4):
        for j in range(4):
            kernel[:, :, i, j] = 4 * i + j
    inv_r = -r % 4
    rot_kernel = onp.array(
        transform_p4kernelp4(create_kernel(jnp.array(kernel)), inv_r, (0, 0)).nngp
    )

    for i in range(4):
        for j in range(4):
            rot_i = (i + inv_r) % 4
            rot_j = (j + inv_r) % 4

            assert onp.allclose(rot_kernel[0, 0, i, j], kernel[0, 0, rot_i, rot_j]), (
                f"Failed for ({i}, {j}) component"
            )


@pytest.mark.parametrize("r", range(4))
def test_kernel_group_action(r):
    kernel_shape = (1, 1, 4, 4, 3, 3, 3, 3)
    key = jax.random.key(0)
    rnd_kernel = random_kernel(key, kernel_shape, 3)
    inv_r = -r % 4
    rot_kernel = onp.array(transform_p4kernelp4(rnd_kernel, inv_r, (0, 0)).nngp)

    kernel = onp.array(rnd_kernel.nngp)
    for i in range(4):
        for j in range(4):
            rot_i = (i + inv_r) % 4
            rot_j = (j + inv_r) % 4
            ker = kernel[0, 0, rot_i, rot_j]
            ker = onp.rot90(ker, inv_r, axes=(0, 2))
            ker = onp.rot90(ker, inv_r, axes=(1, 3))

            assert onp.allclose(rot_kernel[0, 0, i, j], ker), (
                f"Failed for ({i}, {j}) component"
            )


@pytest.mark.parametrize(
    "padding, exp_output_shape",
    [
        (Padding.SAME, (2, 4, 16, 16, 10)),
        (Padding.CIRCULAR, (2, 4, 16, 16, 10)),
        (Padding.VALID, (2, 4, 14, 14, 10)),
    ],
)
def test_z2convp4_paddings(padding, exp_output_shape):
    init_fn, apply_fn, kernel_fn = P4ConvP4(10, (3, 3), padding=padding)
    input_shape = (2, 4, 16, 16, 3)

    key1, key2 = jax.random.split(jax.random.key(0), num=2)
    output_shape, params = init_fn(key1, input_shape)

    batch = jax.random.normal(key2, input_shape)
    output = apply_fn(params, batch)

    assert output.shape == output_shape
    assert output.shape == exp_output_shape
