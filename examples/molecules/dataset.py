import warnings
from functools import partial
import tensorflow_datasets as tfds
import random
from typing import Any
from collections.abc import Iterable
import jax.numpy as jnp
import jax
from jax import vmap
import numpy as np
from math import pi
import math
from utils import create_sphere_vecs
from neural_tangents import Kernel
from neural_tangents.stax import layer, requires, Diagonal, Bool

jax.config.update("jax_enable_x64", True)

Ha_to_meV = 27.211386245981 * 1000
name = "qm9/dimenet"

data_dir = "./data"
qm9_meta = {
    'atom_types': [1.0, 6.0, 7.0, 8.0, 9.0],
    'max_num_atoms': 29,
    'unit_convs': {
        'U': Ha_to_meV,
        'U0': Ha_to_meV,
        'U0_atomization': Ha_to_meV,
    },
    # mean and std of train data
    'stats': {
        'U0': (-410.80448871, 39.74504515),
        'U0_atomization': (-2.79825228, 0.37907648),
    }
}


def create_data_source(name: str, data_dir: str, split: str) -> Iterable[Any]:
    """Create tensorflow dataset source dataset in `array_record` format.

    Args:
        name: Name of the dataset.
        data_dir: Directory where the dataset is stored.
        split: Split of the dataset to load.

    Returns:
        Iterable dataset source.
    """
    builder = tfds.builder(name, data_dir=data_dir)
    builder.download_and_prepare(file_format='array_record')
    ds = builder.as_data_source(split=split)
    return ds


def load_dataset(targets: list, data_source: Iterable[Any], shuffle: bool, seed: int = None,
                 max_samples: int = None, offset: int = 0) -> tuple[jax.Array, jax.Array, dict]:
    """Load the QM9 dataset from tensorflow into a jax array.

    Args:
        targets: List of targets to load from QM9.
        data_source: Iterable dataset source representing QM9
        shuffle: Whether to shuffle the dataset.
        seed: Seed used for shuffling/sampling.
        max_samples: Maximum number of samples to load. If None, all samples are loaded. Limited to
            the number of samples in the dataset.
        offset: Offset/Index to start loading samples from. Useful for batching data loading.

    Returns:
        Tuple of positions, charges, and target values (as dictionary).
    """
    if offset != 0 and shuffle and seed is None:
        warnings.warn("Offset and shuffle are set but the PRNG is not seeded. This may lead to "
                      "non-reproducible test sets.")
    n_samples = len(data_source)
    if max_samples:
        max_samples = min(max_samples, n_samples)
    else:
        max_samples = n_samples

    if shuffle:
        if seed:
            random.seed(seed)
        entries = random.sample(range(n_samples), max_samples + offset)
        entries = entries[offset:]
    else:
        entries = range(offset, offset + max_samples)

    positions = []
    charges = []
    target_vals = dict([(t, []) for t in targets])

    for index in entries:
        entry = data_source[index]
        positions.append(entry['positions'])
        charges.append(entry['charges'])
        for t in targets:
            value = jnp.array(entry[t])
            target_vals[t].append(value)

    positions = jnp.stack(positions)
    charges = jnp.stack(charges)
    for key, value in target_vals.items():
        target_vals[key] = jnp.stack(value, axis=0)
        if target_vals[key].ndim == 1:
            target_vals[key] = jnp.expand_dims(target_vals[key], axis=-1)

    return positions, charges, target_vals


def calc_beta(cutoff_angle: float) -> float:
    """Calculate the beta parameter for the Gaussian weight function.

    See https://doi.org/10.48550/arXiv.2306.05420 for details.

    Args:
        cutoff_angle: Cutoff angle in radians that defines 95% falloff.

    Returns:
        Beta parameter for Gaussian weight function.
    """

    return - (math.cos(cutoff_angle) - 1)**2 / math.log(0.05)


@partial(vmap, in_axes=(0, None, None), out_axes=0)
def sum_same_element_contributions(addends: jax.Array, element_segment_ids: jax.Array,
                                   num_elements: int) -> jax.Array:
    """Sum contributions in an array corresponding to the same element type.

    Args:
        addends: Array of contributions to sum.
        element_segment_ids: 1d array of element ids representing the associated element with each
            row in `addends`.
        num_elements: Number of different element types, important for correct output size.

    Returns:
        Summed contributions for each element type.
    """
    summed_contributions = jax.ops.segment_sum(addends, element_segment_ids,
                                               num_segments=num_elements)
    return summed_contributions


def reduce_size_to_actual_atoms(positions: jax.Array, charges: jax.Array) -> tuple[jax.Array,
                                                                                   jax.Array, int]:
    """Reduce the size of the input to the actual number of atoms in the molecule.

    This essentially removes the padding.

    Args:
        positions: padded array of positions of shape.
        charges: padded array of charges of shape.

    Returns:
        Tuple of non-padded positions, charges, and the number of atoms in the molecule.
    """
    num_atoms = np.count_nonzero(charges)
    positions = positions[:num_atoms]
    charges = charges[:num_atoms]
    return positions, charges, num_atoms


def create_spherical_potentials(positions: jax.Array, charges: jax.Array, bandlimit: int,
                                atom_types: list[float] | float, powers: list[int] | int) -> jax.Array:
    """Create spherical potentials for a molecule.

    Args:
        positions: Array of positions of shape (num_atoms, 3).
        charges: Array of charges of shape (num_atoms,).
        bandlimit: Bandlimit defining the grid size (Driscoll & Healy).
        atom_types: List of atom types in the molecule. These are usually just the atomic numbers.
        powers: List of powers to use in the power law decay with the distance. Determines the
            number of channels.

    Returns:
        Spherical potentials for the molecule with shape
        (num_atoms, thetas, phis, num_elements x num_powers).
    """
    if not isinstance(atom_types, list):
        atom_types = [atom_types]

    if not isinstance(powers, list):
        powers = [powers]

    cutoff_angle = pi / 4
    max_num_atoms = charges.size
    atom_types = jnp.array([0] + atom_types)
    powers = jnp.array(powers)

    # r_ij = r_j - r_i with (i, j)
    pos_vecs = positions[None, :, :] - positions[:, None, :]
    dists = jnp.linalg.norm(pos_vecs, axis=-1)
    dists_no_self = dists.at[jnp.diag_indices(max_num_atoms)].set(jnp.inf)

    sph_vecs = create_sphere_vecs(bandlimit)
    grid_size = sph_vecs.shape[:2]
    beta = calc_beta(cutoff_angle)

    # i = atom i, j = atom j, d = 3d vec, t = theta, p = phi
    inner_prods = jnp.einsum('ijd,tpd->ijtp', pos_vecs / dists_no_self[:, :, None], sph_vecs)

    gauss_weight = jnp.exp(-1 / beta * (inner_prods - 1)**2)
    charge_prods = charges[None, :] * charges[:, None]

    dist_powers = dists_no_self[:, :, None] ** powers[None, None, :]

    # Make all tensors broadcastable: must match shape (num_atoms, num_atoms, thetas, phis,
    # powers)
    gauss_weight = jnp.expand_dims(gauss_weight, axis=-1)
    charge_prods = jnp.expand_dims(charge_prods, axis=(2, 3, 4))
    dist_powers = jnp.expand_dims(dist_powers, axis=(2, 3))

    addends = charge_prods / dist_powers * gauss_weight

    element_segment_ids = jnp.where(charges[:, None] - atom_types[None, :] == 0,
                                    size=max_num_atoms)[1]

    # has shape (num_atoms, num_elements, thetas, phis, powers)
    # non-existing elements are summed to index 0 and removed afterwards
    summed_contributions = sum_same_element_contributions(addends, element_segment_ids,
                                                          atom_types.size)[..., 1:, :, :, :]
    summed_contributions = jnp.transpose(summed_contributions, (0, 2, 3, 1, 4))

    # signal.shape = (num_atoms, thetas, phis, num_types * powers)
    signal = jnp.reshape(summed_contributions, (max_num_atoms, *grid_size, -1))
    return signal


def rotate_positions(positions: jax.Array, rotation: jax.Array) -> jax.Array:
    """Rotate positions using a rotation matrices.

    Args:
        positions: Array of positions of shape (batch, num_atoms, 3).
        rotation: Rotation matrix of shape (rotation_batch, 3, 3).

    Returns:
        Rotated positions of shape (rotation_batch, batch, num_atoms, 3).
    """
    return jnp.einsum('bij,maj->mbai', rotation, positions)


def load_sphere_data(targets: list[str], data_source: Iterable[Any], shuffle: bool, bandlimit: int,
                     atom_types: list[float], powers: list[int], seed: int = None,
                     max_samples: int = None, offset: int = 0,
                     rotations: jax.Array = None) -> tuple[jax.Array, dict]:
    """Load the QM9 dataset and create spherical potentials.

    Args:
        targets: List of targets to load from QM9.
        data_source: Iterable dataset source representing QM9
        shuffle: Whether to shuffle the dataset.
        bandlimit: Bandlimit defining the grid size (Driscoll & Healy).
        atom_types: List of atom types in the molecule. These are usually just the atomic numbers.
        powers: List of powers to use in the power law decay with the distance. Determines the
            number of channels.
        seed: Seed used for shuffling/sampling.
        max_samples: Maximum number of samples to load. If None, all samples are loaded. Limited to
            the number of samples in the dataset.
        offset: Offset/Index to start loading samples from. Useful for batching data loading.
        rotations: Array of rotation matrices to augment the dataset.

    Returns:
        Tuple of spherical potentials and target values (in form of dictionary).
    """
    positions, charges, target_vals = load_dataset(targets, data_source, shuffle, seed,
                                                   max_samples, offset)
    if rotations is not None:
        num_rotations = rotations.shape[0]
        positions = rotate_positions(positions, rotations)
        positions = jnp.reshape(positions, (-1, positions.shape[-2], positions.shape[-1]))
        charges = jnp.repeat(charges, num_rotations, axis=0)
        for key, value in target_vals.items():
            if value.ndim > 2 or value.shape[1] > 1:
                raise NotImplementedError("Rotation augmentation for non-scalar targets not "
                                          " implemented.")
            target_vals[key] = jnp.repeat(value, num_rotations, axis=0)

    sph_pot_fn = partial(create_spherical_potentials, bandlimit=bandlimit, atom_types=atom_types,
                         powers=powers)
    batch_create_spherical_potentials = vmap(sph_pot_fn, in_axes=(0, 0), out_axes=0)
    sph_signals = batch_create_spherical_potentials(positions, charges)
    return sph_signals, target_vals


def calc_target_stats(targets: list[str], data_source: Iterable[Any]) -> tuple[dict, dict]:
    """Calculate mean and standard deviation of target values.

    Args:
        targets: List of targets to load from QM9.
        data_source: Iterable dataset source representing QM9
    Returns:
        Tuple of means and standard deviations for each target in form of a dictionary.
    """
    means = {}
    stds = {}
    target_vals = load_dataset(targets, data_source, False)[-1]
    for target in targets:
        vals = target_vals[target]
        means[target] = jnp.mean(vals)
        stds[target] = jnp.std(vals)
    return means, stds


@layer
def split_atom_features(max_num_atoms: int):
    """Fan a molecule tensor (batch, atoms, ...) into per-atom streams for stax.parallel."""

    def init_fn(rng, input_shape: tuple) -> tuple:
        per_atom_shape = (input_shape[0], *input_shape[2:])
        return [per_atom_shape] * max_num_atoms, ()

    def apply_fn(params, inputs: jax.Array, **kwargs) -> list:
        return [inputs[:, i] for i in range(max_num_atoms)]

    def _extract_subkernel(kernel: jax.Array, idx: int) -> jax.Array:
        if kernel is None or kernel.ndim == 0:
            return kernel
        if kernel.ndim == 8:
            return kernel[:, :, idx, idx, :, :, :, :]
        elif kernel.ndim == 7:
            return kernel[:, idx, idx, :, :, :, :]
        elif kernel.ndim == 5:
            return kernel[:, :, idx, :, :]
        elif kernel.ndim == 4:
            return kernel[:, idx, :, :]
        else:
            raise ValueError(f"Invalid kernel shape: {kernel.shape}")

    def _extract_all_subkernels(k: Kernel, idx: int) -> Kernel:
        return k.replace(
            nngp=_extract_subkernel(k.nngp, idx),
            ntk=_extract_subkernel(k.ntk, idx),
            cov1=_extract_subkernel(k.cov1, idx),
            cov2=_extract_subkernel(k.cov2, idx),
        )

    @requires(batch_axis=0, channel_axis=4,
              diagonal_spatial=Diagonal(output=Bool.MAYBE))
    def kernel_fn(k: Kernel, **kwargs) -> list:
        return [_extract_all_subkernels(k, i) for i in range(max_num_atoms)]

    return init_fn, apply_fn, kernel_fn
