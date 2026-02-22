from equivariant_ntk.utils.vmap_helpers import vmap_subarray_fn
import jax.numpy as jnp
import numpy as np

# def fn(x):
#     return jnp.sum(x, axis=2)
#
# new_fn = vmap_subarray_fn(fn, 'abcdef->adcbf', 'bde->bd')


def fn(x):
    doub_size = np.array(x.shape, dtype=int) * 2
    return jnp.ones(doub_size)


new_fn = vmap_subarray_fn(fn, 'abcdef->amclef', 'bd->lm')

x = jnp.ones((2, 3, 5, 7, 11, 13))
y = new_fn(x)
print(y.shape)
