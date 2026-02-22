# equivariant-ntk
[**ICML 2025 paper**](https://proceedings.mlr.press/v267/misof25a.html)
| [**arXiv version**](https://arxiv.org/abs/2406.06504)

This repository implements the NTK and NNGP recursions of several group convolutional layers from the paper [*Equivariant Neural Tangent Kernels*](https://proceedings.mlr.press/v267/misof25a.html) by [Philipp Misof](https://gapindnns.github.io/members/Philipp_Misof.html), [Pan Kessel](https://www.gene.com/scientists/our-scientists/pan-kessel) and [Jan E Gerken](https://gapindnns.github.io/members/Jan_Gerken.html) presented at the **ICML 2025**. In this work we extend the NTK framework to group convolutional neural networks (GCNNs) and compare the infinite width dynamics of those equivariant architectures to non-equivariant counterparts trained with data augmentation.

The package is written as an extension to the [`neural-tangents`](https://github.com/google/neural-tangents) library and is implemented in [JAX](https://github.com/jax-ml/jax). Supported symmetry groups are the planar roto-translation group $C_4 \ltimes \mathbb{R}^2$ and the group of 3D rotations $\mathrm{SO}(3)$. Simplified versions of the experiments in the paper are provided as well.

## Layers

### Overview

| Layer | Group | Description |
|-------|-------|-------------|
| `Z2ConvP4` | $C_4 \ltimes \mathbb{R}^2$ | Lifting convolution from images to the group domain |
| `P4ConvP4` | $C_4 \ltimes \mathbb{R}^2$ | Group convolution |
| `GroupPool` | $C_4 \ltimes \mathbb{R}^2$ | Invariant pooling over the group and spatial dimensions |
| `S2ConvSO3` | $\mathrm{SO}(3)$ | Lifting convolution from $S^2$ signals to $\mathrm{SO}(3)$ |
| `SO3ConvSO3` | $\mathrm{SO}(3)$ | Group convolution on $\mathrm{SO}(3)$ |
| `SO3Pool` | $\mathrm{SO}(3)$ | Invariant pooling over the group and spatial dimensions |


### API

The layers follow the `neural-tangents` API, which extends JAX's [`stax`](https://docs.jax.dev/en/latest/jax.example_libraries.stax.html) API to also provide infinite-width operations: Each layer function returns a triple `(init_fn, apply_fn, kernel_fn)`, where

- `init_fn(rng, input_shape)` — initialises and returns the layer's parameters.
- `apply_fn(params, x)` — evaluates the finite-width network for a given input `x`.
- `kernel_fn(x1, x2)` — propagates the NNGP and NTK kernel matrices analytically through the layer, as introduced by `neural-tangents`.

 Layers can be composed using `neural_tangents.stax.serial` and `neural_tangents.stax.parallel`

```python
from equivariant_ntk.layers import S2ConvSO3, SO3ConvSO3, SO3Pool
from neural_tangents import stax

init_fn, apply_fn, kernel_fn = stax.serial(
    S2ConvSO3(...),
    stax.Erf(),
    SO3ConvSO3(...),
    SO3Pool(),
    stax.Dense(1),
)

# finite-width forward pass
_, params = init_fn(rng, input_shape)
y = apply_fn(params, x)

# exact infinite-width NTK
kernel = kernel_fn(x, None, 'ntk')
```

For more details, see the [`neural-tangents` documentation](https://neural-tangents.readthedocs.io/en/latest/)

### Implementation details

#### $C_4 \ltimes \mathbb{R}^2$ layers
2D feature maps are extended to carry an explicit group dimension of size 4, one entry per element of $C_4$. The lifting convolution `Z2ConvP4` maps a standard 2D signal (no group dimension) to 2D signals over the group. Subsequent `P4ConvP4` layers keep this structure.

We exploit the fact that both the finite and infinite-width operations of the $C_4 \ltimes \mathbb{R}^2$ group convolution decompose into two parts: a standard spatial convolution, for which the optimized `lax` and `neural-tangents` operations exist as well as a $C_4$-specific operation that mixes the group dimension across the four rotations.

#### $\mathrm{SO}(3)$ layers
Input signals defined on the sphere $S^2$ are lifted to feature maps on $\mathrm{SO}(3)$ after applying the `S2ConvSO3` layer. Feature maps are discretized on the equiangular Driscoll-Healy grid and parametrized by the spherical coordinates $(\theta, \varphi)$ ($S^2$) or the Euler angles $(\alpha, \beta, \gamma)$ (${\mathrm{SO}(3)}$). The bandwidth parameter $L$ controls the grid size. The convolutions are computed by first Fourier transforming the feature maps and then applying the corresponding algebraic operations in Fourier space before transforming back. The necessary spherical and Wigner transformations are provided by  the [`s2fft`](https://github.com/astro-informatics/s2fft) library.

## Examples

The experiments in the corresponding paper required several GPU hours on a cluster. For demonstrative purposes, we have reduced the complexity of the setup significantly in the notebooks provided here. They should run on most laptops in 5-20 minutes.

### Histological image classification

Located in `examples/histological_image_classification/`. There we use a $C_4 \ltimes \mathbb{R}^2$-equivariant CNN for classifying histological tissue images found in the *NCT-CRC-HE-100K*[^1] dataset.

- `kernel_convergence.ipynb` — verifies that the empirical NTK/NNGP of finite-width networks converges to the analytic infinite-width kernel as width increases.
- `kernel_prediction.ipynb` — infinite-width NTK kernel regression on the classification task. Corresponds to the mean solution of an ensemble of infinitely wide neural networks trained under gradient flow until fully converged.

### Molecular property prediction (QM9)

Located in `examples/molecules/`. Predicts the internal energies of molecules from the *QM9*[^2] dataset using infinite-width NTK kernel regression.

Each atom's chemical environment is encoded as a multi-channel scalar field on the unit sphere (a spherical potential), discretised on a Driscoll-Healy grid. The notebook compares a per-atom MLP baseline against an $\mathrm{SO}(3)$-invariant CNN, demonstrating that the equivariant architecture achieves lower prediction error at the same training set size.

- `kernel_prediction.ipynb` — self-contained demonstration including data loading, input feature visualisation, model definitions, and kernel regression.


## Installation

First, clone the repository:
```bash
git clone https://github.com/PhilippMisofCH/equivariant-ntk.git
```

To install the package, we recommend using `uv`:
```bash
uv sync --all-extras
```
Alternatively, using pip:
```bash
pip install -e .[dev]
```


## Running tests

```bash
pytest tests/
```

## Citation
If you use this code or parts of it in a publication, please cite our paper:
```bibtex
@InProceedings{misof2025,
  title = 	 {Equivariant Neural Tangent Kernels},
  author =       {Misof, Philipp and Kessel, Pan and Gerken, Jan E},
  booktitle = 	 {Proceedings of the 42nd International Conference on Machine Learning},
  pages = 	 {44470--44503},
  year = 	 {2025},
  volume = 	 {267},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {13--19 Jul},
  publisher =    {PMLR},
  url = 	 {https://proceedings.mlr.press/v267/misof25a.html},
}
```

## References
[^1]: Kather, J. N., Halama, N., & Marx, A. (2018). 100,000 histological images of human colorectal cancer and healthy tissue (v0.1) \[Data set\]. Zenodo. [https://doi.org/10.5281/zenodo.1214456](https://doi.org/10.5281/zenodo.1214456)
[^2]: Ramakrishnan, Raghunathan; Dral, Pavlo; Rupp, Matthias; Anatole von Lilienfeld, O. (2014). Quantum chemistry structures and properties of 134 kilo molecules. figshare. Collection. [https://doi.org/10.6084/m9.figshare.c.978904.v5](https://doi.org/10.6084/m9.figshare.c.978904.v5)
