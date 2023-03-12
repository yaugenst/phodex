from typing import Callable

import autograd.numpy as np
import numpy as onp

import phodex.autograd.primitives
from phodex.autograd.functions import _pad_modes, convolve

gaussian_filter = phodex.autograd.primitives.gaussian_filter


def cone_filter(size: int, mode: _pad_modes = "symmetric") -> Callable:
    xx, yy = onp.ogrid[-1 : 1 : 1j * size, -1 : 1 : 1j * size]
    kernel = 1 - np.sqrt(xx**2 + yy**2)
    kernel[kernel < 0] = 0
    kernel /= np.sum(kernel)

    def _filter(x: np.ndarray) -> np.ndarray:
        return convolve(x, kernel, mode=mode)

    return _filter


def disk_filter(size: int, mode: _pad_modes = "symmetric") -> Callable:
    xx, yy = onp.ogrid[-1 : 1 : 1j * size, -1 : 1 : 1j * size]
    kernel = np.sqrt(xx**2 + yy**2) <= 1
    kernel = kernel / np.sum(kernel)

    def _filter(x: np.ndarray) -> np.ndarray:
        return convolve(x, kernel, mode=mode)

    return _filter
