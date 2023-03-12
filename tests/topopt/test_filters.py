import numpy as np
import pytest
from autograd.test_util import check_grads
from numpy.testing import assert_almost_equal

from phodex.autograd.functions import _pad_modes
from phodex.topopt.filters import cone_filter, disk_filter, gaussian_filter


@pytest.fixture
def rng():
    seed = 36523523
    return np.random.default_rng(seed)


@pytest.mark.parametrize("ary_size", [10, 11])
@pytest.mark.parametrize("sigma", [1, 2])
@pytest.mark.parametrize(
    "mode",
    [
        "reflect",
        "constant",
        "wrap",
        pytest.param(
            "nearest",
            marks=pytest.mark.xfail(
                reason="Grads for mode `nearest` not implemented for Gaussian filter."
            ),
        ),
        pytest.param(
            "mirror",
            marks=pytest.mark.xfail(
                reason="Grads for mode `mirror` not implemented for Gaussian filter."
            ),
        ),
    ],
)
def test_gaussian_filter_grad(rng, ary_size, sigma, mode):
    x = rng.random((ary_size, ary_size))
    check_grads(gaussian_filter, modes=["rev"], order=2)(x, sigma, mode=mode)


@pytest.mark.parametrize("func", [cone_filter, disk_filter])
@pytest.mark.parametrize("ary_size", [10, 11])
@pytest.mark.parametrize("filter_size", [3, 5])
@pytest.mark.parametrize("mode", _pad_modes.__args__)
def test_filter_grad(rng, func, ary_size, filter_size, mode):
    x = rng.random((ary_size, ary_size))
    f = func(filter_size, mode)
    check_grads(f, modes=["rev"], order=2)(x)


@pytest.mark.parametrize("func", [cone_filter, disk_filter])
@pytest.mark.parametrize("ary_size", [10, 11])
@pytest.mark.parametrize("filter_size", [3, 5])
@pytest.mark.parametrize("mode", ["symmetric", "wrap"])
def test_filter_energy_conservation(rng, func, ary_size, filter_size, mode):
    x = rng.random((ary_size, ary_size))
    f = func(filter_size, mode)
    assert_almost_equal(np.mean(x), np.mean(f(x)))
