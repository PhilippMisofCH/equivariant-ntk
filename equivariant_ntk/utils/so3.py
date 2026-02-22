import itertools
import s2fft
import s2fft.precompute_transforms
from math import pi
import jax
import jax.numpy as jnp


class Precompute_Wig:
    def __init__(self, sampling: str = "dh"):
        self.cache = {}
        self.sampling = sampling

    def get_kernel(self, L, direction, reality=False):
        if direction not in ["forward", "backward"]:
            raise ValueError(
                f"Invalid value {direction} for kernel. Must be either 'forward' or "
                "'backward'."
            )
        if (L, direction, reality) in self.cache:
            return self.cache[(L, direction, reality)]
        else:
            raise ValueError(
                f"Kernel for L={L}, direction={direction}, reality={reality} was not "
                "precomputed"
            )

    def compute_kernels(self, Ls):
        realities = [True, False]
        directions = ["forward", "backward"]
        combinations = itertools.product(Ls, directions, realities)
        for L, direction, reality in combinations:
            is_forward = direction == "forward"
            kernel = s2fft.precompute_transforms.construct.wigner_kernel_jax(
                L, L, reality=reality, sampling=self.sampling, forward=is_forward
            )
            self.cache[(L, direction, reality)] = kernel


class Precompute_Sph:
    def __init__(self, sampling: str = "dh"):
        self.cache = {}
        self.sampling = sampling

    def get_kernel(self, L, direction, reality=False):
        if direction not in ["forward", "backward"]:
            raise ValueError(
                f"Invalid value {direction} for kernel. Must be either 'forward' or "
                "'backward'."
            )
        if (L, direction, reality) in self.cache:
            return self.cache[(L, direction, reality)]
        else:
            raise ValueError(
                f"Kernel for L={L}, direction={direction}, reality={reality} was not "
                "precomputed"
            )

    def compute_kernels(self, Ls):
        realities = [True, False]
        directions = ["forward", "backward"]
        combinations = itertools.product(Ls, directions, realities)
        for L, direction, reality in combinations:
            is_forward = direction == "forward"
            kernel = s2fft.precompute_transforms.construct.spin_spherical_kernel_jax(
                L, spin=0, reality=reality, sampling=self.sampling, forward=is_forward
            )
            self.cache[(L, direction, reality)] = kernel


def get_bw_from_so3_shape(shape, sampling):
    if sampling != "dh":
        raise NotImplementedError(
            f"Only 'dh' sampling is supported right now. Got {sampling}."
        )

    n_betas = shape[1]
    if n_betas % 2 != 0:
        raise ValueError(f"Number of betas must be even. Got {n_betas}.")

    L = n_betas // 2
    return L


def get_bw_from_s2_shape(shape, sampling):
    if sampling != "dh":
        raise NotImplementedError(
            f"Only 'dh' sampling is supported right now. Got {sampling}."
        )

    n_thetas = shape[0]
    if n_thetas % 2 != 0:
        raise ValueError(f"Number of thetas must be even. Got {n_thetas}.")

    L = n_thetas // 2
    return L


def sample_rotation(key, n):
    """Samples random SO(3) matrices.

    Follows the approach in Arvo, J. III.4 - FAST RANDOM ROTATION MATRICES. in Graphics Gems III
    (IBM Version) (ed. Kirk, D.) 117–120 (Morgan Kaufmann, San Francisco, 1992).
    doi:10.1016/B978-0-08-050755-2.50034-8.
    """

    xs = jax.random.uniform(key, shape=(n, 3))
    thetas = 2 * pi * xs[:, 0]
    phis = 2 * pi * xs[:, 1]
    z = xs[:, 2]

    rot_2d = jnp.array(
        [
            [jnp.cos(thetas), jnp.sin(thetas), jnp.zeros_like(thetas)],
            [-jnp.sin(thetas), jnp.cos(thetas), jnp.zeros_like(thetas)],
            [jnp.zeros_like(thetas), jnp.zeros_like(thetas), jnp.ones_like(thetas)],
        ]
    )
    v = jnp.array(
        [jnp.cos(phis) * jnp.sqrt(z), jnp.sin(phis) * jnp.sqrt(z), jnp.sqrt(1 - z)]
    )

    z_proj = 2 * jnp.einsum("ib,jb->bij", v, v) - jnp.expand_dims(
        jnp.identity(3), axis=0
    )
    rot_3d = jnp.einsum("bij,jkb->bik", z_proj, rot_2d)

    return rot_3d
