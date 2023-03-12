import numpy as np
import pytest
from autograd.test_util import check_grads
from numpy.testing import assert_allclose
from scipy.signal import convolve as convolve_sp

from phodex.autograd.functions import (
    _pad_modes,
    circular_pad,
    convolve,
    edge_pad,
    pad,
    reflection_pad,
    symmetric_pad,
)


@pytest.fixture
def rng():
    seed = 36523523
    return np.random.default_rng(seed)


@pytest.mark.parametrize(
    "func,mode",
    [
        (circular_pad, "wrap"),
        (edge_pad, "edge"),
        (reflection_pad, "reflect"),
        (symmetric_pad, "symmetric"),
    ],
)
@pytest.mark.parametrize("size", [10, 11])
@pytest.mark.parametrize("padding", [1, 2])
def test_pad_func_val(rng, func, mode, size, padding):
    x = rng.random((size, size))
    assert_allclose(func(x, padding), np.pad(x, padding, mode=mode))


@pytest.mark.parametrize(
    "func", [circular_pad, edge_pad, reflection_pad, symmetric_pad]
)
@pytest.mark.parametrize("size", [10, 11])
@pytest.mark.parametrize("padding", [1, 2])
def test_pad_func_grad(rng, func, size, padding):
    x = rng.random((size, size))
    check_grads(func, modes=["fwd", "rev"], order=2)(x, padding)


@pytest.mark.parametrize("mode", _pad_modes.__args__)
@pytest.mark.parametrize("size", [10, 11])
@pytest.mark.parametrize("padding", [0, 1, 2])
def test_pad_val(rng, mode, size, padding):
    x = rng.random((size, size))
    assert_allclose(pad(x, padding, mode=mode), np.pad(x, padding, mode=mode))


@pytest.mark.parametrize("mode", _pad_modes.__args__)
@pytest.mark.parametrize("size", [10, 11])
@pytest.mark.parametrize("padding", [0, 1, 2])
def test_pad_grad(rng, mode, size, padding):
    x = rng.random((size, size))
    check_grads(pad, modes=["fwd", "rev"], order=2)(x, padding, mode=mode)


@pytest.mark.parametrize("mode", _pad_modes.__args__)
@pytest.mark.parametrize("size", [10, 11])
@pytest.mark.parametrize("ks", [1, 3])
def test_convolve_shape(rng, mode, size, ks):
    x = rng.random((size, size))
    k = rng.random((ks, ks))
    c = convolve(x, k, mode=mode)
    assert_allclose(x.shape, c.shape)


@pytest.mark.parametrize("mode", _pad_modes.__args__)
@pytest.mark.parametrize("size", [10, 11])
@pytest.mark.parametrize("ks", [1, 3])
def test_convolve_val(rng, mode, size, ks):
    def _conv(x, k, mode):
        p = k.shape[-1] // 2
        x = np.pad(x, p, mode=mode)
        return convolve_sp(x, k, mode="valid")

    x = rng.random((size, size))
    k = rng.random((ks, ks))
    assert_allclose(convolve(x, k, mode=mode), _conv(x, k, mode=mode))


@pytest.mark.parametrize("mode", _pad_modes.__args__)
@pytest.mark.parametrize("size", [10, 11])
@pytest.mark.parametrize("ks", [1, 3])
def test_convolve_grad(rng, mode, size, ks):
    x = rng.random((size, size))
    k = rng.random((ks, ks))
    check_grads(convolve, modes=["rev"], order=2)(x, k, mode=mode)
