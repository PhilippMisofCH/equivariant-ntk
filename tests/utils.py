from neural_tangents import Kernel
import jax.numpy as jnp
import jax


def transform_p4(batch, r=0, t=(0, 0), r_dim=1, t_dims=(2, 3)):
    """acts with (r, t) from the left"""
    batch = jnp.roll(batch, -r, r_dim)
    # perform shift assuming sufficent padding
    batch = jnp.roll(batch, t, t_dims)
    return jnp.rot90(batch, r, t_dims)


def transform_z2(batch, r=1, t=(0, 0), t_dims=(1, 2)):
    # perform shift assuming sufficent padding
    batch = jnp.roll(batch, t, t_dims)
    return jnp.rot90(batch, r, t_dims)


def random_kernel(key, kernel_shape, n_in_chan):
    cov_shape1 = (kernel_shape[0], *kernel_shape[2:])
    cov_shape2 = (kernel_shape[1], *kernel_shape[2:])
    key1, key2, key3 = jax.random.split(key, 3)
    nngp = jax.random.normal(key1, kernel_shape)
    cov1 = jax.random.normal(key2, cov_shape1)
    cov2 = jax.random.normal(key3, cov_shape2)
    return create_kernel(nngp, n_in_chan=n_in_chan, cov1=cov1, cov2=cov2)


def create_kernel(nngp, n_in_chan=1, ntk=None, cov1=None, cov2=None):
    input_shape = nngp.shape[::2] + (n_in_chan,)
    return Kernel(
        nngp=nngp,
        ntk=ntk,
        cov1=cov1,
        cov2=cov2,
        x1_is_x2=False,
        is_gaussian=True,
        is_reversed=False,
        is_input=False,
        diagonal_batch=True,
        diagonal_spatial=False,
        shape1=input_shape,
        shape2=input_shape,
        batch_axis=0,
        channel_axis=-1,
        mask1=None,
        mask2=None,
    )


def transform_p4kernelp4(kernel, r, t=(0, 0)):
    """Acts from the left with t and r"""
    rot_kernel = transform_p4(kernel.nngp, r=r, t=t, r_dim=2, t_dims=(4, 6))
    rot_kernel = transform_p4(rot_kernel, r=r, t=t, r_dim=3, t_dims=(5, 7))
    return kernel.replace(nngp=rot_kernel)


def transform_z2kernelp4(kernel, r, t=(0, 0)):
    """Acts from the left with t and r"""
    rot_kernel = transform_z2(kernel.nngp, r=r, t=t, t_dims=(2, 4))
    rot_kernel = transform_z2(rot_kernel, r=r, t=t, t_dims=(3, 5))
    return kernel.replace(nngp=rot_kernel)
