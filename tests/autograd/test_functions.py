import numpy as np
import pytest
import scipy.ndimage
from autograd.test_util import check_grads
from numpy.testing import assert_allclose
from scipy.signal import convolve as convolve_sp

from phodex.autograd.functions import (
    _mode_to_scipy,
    _pad_modes,
    circular_pad,
    convolve,
    edge_pad,
    grey_closing,
    grey_dilation,
    grey_erosion,
    grey_opening,
    morphological_gradient,
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
@pytest.mark.parametrize("ary_size", [10, 11])
@pytest.mark.parametrize("padding", [1, 2])
def test_pad_func_val(rng, func, mode, ary_size, padding):
    x = rng.random((ary_size, ary_size))
    assert_allclose(func(x, padding), np.pad(x, padding, mode=mode))


@pytest.mark.parametrize(
    "func", [circular_pad, edge_pad, reflection_pad, symmetric_pad]
)
@pytest.mark.parametrize("ary_size", [10, 11])
@pytest.mark.parametrize("padding", [1, 2])
def test_pad_func_grad(rng, func, ary_size, padding):
    x = rng.random((ary_size, ary_size))
    check_grads(func, modes=["fwd", "rev"], order=2)(x, padding)


@pytest.mark.parametrize("mode", _pad_modes.__args__)
@pytest.mark.parametrize("ary_size", [10, 11])
@pytest.mark.parametrize("padding", [0, 1, 2])
def test_pad_val(rng, mode, ary_size, padding):
    x = rng.random((ary_size, ary_size))
    assert_allclose(pad(x, padding, mode=mode), np.pad(x, padding, mode=mode))


@pytest.mark.parametrize("mode", _pad_modes.__args__)
@pytest.mark.parametrize("ary_size", [10, 11])
@pytest.mark.parametrize("padding", [0, 1, 2])
def test_pad_grad(rng, mode, ary_size, padding):
    x = rng.random((ary_size, ary_size))
    check_grads(pad, modes=["fwd", "rev"], order=2)(x, padding, mode=mode)


@pytest.mark.parametrize("mode", _pad_modes.__args__)
@pytest.mark.parametrize("ary_size", [10, 11])
@pytest.mark.parametrize("ks", [1, 2, 3])
def test_convolve_shape(rng, mode, ary_size, ks):
    x = rng.random((ary_size, ary_size))
    k = rng.random((ks, ks))
    c = convolve(x, k, mode=mode)
    assert_allclose(x.shape, c.shape)


@pytest.mark.parametrize("mode", _pad_modes.__args__)
@pytest.mark.parametrize("ary_size", [10, 11])
@pytest.mark.parametrize(
    "ks",
    [
        1,
        pytest.param(
            2,
            marks=pytest.mark.xfail(
                reason="Even kernels use a different convention than SciPy."
            ),
        ),
        3,
    ],
)
def test_convolve_val(rng, mode, ary_size, ks):
    def _conv(x, k, mode):
        p = k.shape[-1] // 2
        x = np.pad(x, p, mode=mode)
        return convolve_sp(x, k, mode="valid")

    x = rng.random((ary_size, ary_size))
    k = rng.random((ks, ks))
    assert_allclose(convolve(x, k, mode=mode), _conv(x, k, mode=mode))


@pytest.mark.parametrize("mode", _pad_modes.__args__)
@pytest.mark.parametrize("ary_size", [10, 11])
@pytest.mark.parametrize("ks", [1, 2, 3])
def test_convolve_grad(rng, mode, ary_size, ks):
    x = rng.random((ary_size, ary_size))
    k = rng.random((ks, ks))
    check_grads(convolve, modes=["rev"], order=2)(x, k, mode=mode)


@pytest.mark.parametrize(
    "op,sp_op",
    [
        (grey_dilation, scipy.ndimage.grey_dilation),
        (grey_erosion, scipy.ndimage.grey_erosion),
        (grey_opening, scipy.ndimage.grey_opening),
        (grey_closing, scipy.ndimage.grey_closing),
        (morphological_gradient, scipy.ndimage.morphological_gradient),
    ],
)
@pytest.mark.parametrize("mode", _pad_modes.__args__)
@pytest.mark.parametrize("ary_size", [10, 11])
@pytest.mark.parametrize("ks", [1, 3])
@pytest.mark.parametrize(
    "kind",
    [
        "flat",
        pytest.param(
            "full",
            marks=pytest.mark.xfail(
                reason="Full structuring elements are not supported yet."
            ),
        ),
    ],
)
def test_morphology_val(rng, op, sp_op, mode, ary_size, ks, kind):
    x = rng.random((ary_size, ary_size))

    match kind:
        case "flat":
            s = np.ones((ks, ks))
        case "full":
            s = rng.randint(0, 2, (ks, ks))

    ndimg_mode = _mode_to_scipy[mode]
    assert_allclose(op(x, s, mode=mode), sp_op(x, structure=s, mode=ndimg_mode))


@pytest.mark.parametrize(
    "op",
    [grey_dilation, grey_erosion, grey_opening, grey_closing, morphological_gradient],
)
@pytest.mark.parametrize("mode", ["reflect", "constant", "symmetric", "wrap"])
@pytest.mark.parametrize("ary_size", [10, 11])
@pytest.mark.parametrize("ks", [1, 3])
@pytest.mark.parametrize(
    "kind",
    [
        "flat",
        pytest.param(
            "full",
            marks=pytest.mark.xfail(
                reason="Full structuring elements are not supported yet."
            ),
        ),
    ],
)
def test_morphology_grad(rng, op, mode, ary_size, ks, kind):
    x = rng.random((ary_size, ary_size))

    match kind:
        case "flat":
            s = np.ones((ks, ks))
        case "full":
            s = rng.randint(0, 2, (ks, ks))

    check_grads(op, modes=["rev"], order=2)(x, s, mode=mode)
