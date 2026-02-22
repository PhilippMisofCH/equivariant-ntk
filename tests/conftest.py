from itertools import product
from dataclasses import dataclass

from equivariant_ntk.utils.so3 import Precompute_Wig, Precompute_Sph
import s2_so3_utils

import numpy as np
import pytest
import jax


# @pytest.fixture(params=["ntk", "standard"])
# def parameterization(request):
#     return request.param


# @pytest.fixture(params=[0, 1, 2, 3])
# def r(request):
#     return request.param
#
#
# @pytest.fixture(params=product(range(-2, 3), range(-2, 3)))
# def t(request):
#     return request.param


# @pytest.fixture(params=[2.0])
# def W_std(request):
#     return request.param
#
#
# @pytest.fixture(params=[3.0])
# def b_std(request):
#     return request.param
#
#
# @pytest.fixture(params=[3])
# def c_out(request):
#     return request.param


@pytest.fixture(params=[3, 5])
def filter_size(request):
    return request.param


#
#
# @pytest.fixture(params=[(5, 6)])
# def s(request):
#     return request.param


# @pytest.fixture(params=[3])
# def c_in(request):
#     return request.param


# @pytest.fixture(params=[3, 10])
# def img_size(request):
#     return request.param


@pytest.fixture
def rng():
    return np.random.default_rng()


@pytest.fixture
def rand_key():
    seed = np.random.randint(0, 2**32 - 1)
    return jax.random.key(seed)


@pytest.fixture
def precomps_sph(out_bw):
    val = Precompute_Sph(sampling="dh")
    bws = s2_so3_utils.calc_downscaled_bws(out_bw)
    val.compute_kernels(bws)
    return val


@pytest.fixture
def precomps_wig(out_bw):
    val = Precompute_Wig(sampling="dh")
    bws = s2_so3_utils.calc_downscaled_bws(out_bw)
    val.compute_kernels(bws)
    return val


@pytest.fixture()
def s2_signal(out_bw, precomps_sph, c_in, rng):
    return s2_so3_utils.generate_random_s2_signal_batch(
        out_bw // 2, out_bw, precomps_sph, rng, batch_size=2, channels=c_in
    )


@pytest.fixture
def so3_signal(out_bw, precomps_wig, c_in, rng):
    return s2_so3_utils.generate_random_so3_signal_batch(
        out_bw // 2, out_bw, precomps_wig, rng, batch_size=2, channels=c_in
    )
