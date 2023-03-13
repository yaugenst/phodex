from functools import partial
from typing import Callable

import autograd.numpy as np
import numpy as onp

import phodex.autograd.primitives
from phodex.autograd.functions import _mode_to_scipy, _pad_modes, convolve


def gaussian_filter(size: float, mode: _pad_modes = "symmetric") -> Callable:
    f = phodex.autograd.primitives.gaussian_filter
    m = _mode_to_scipy[mode]
    sigma = size / (3 * np.sqrt(3))  # heuristic! but is roughly equivalent to cone
    return partial(f, sigma=sigma, mode=m)


def cone_filter(size: int, mode: _pad_modes = "symmetric") -> Callable:
    xx, yy = onp.ogrid[-1 : 1 : 1j * size, -1 : 1 : 1j * size]
    kernel = 1 - np.sqrt(xx**2 + yy**2)
    kernel[kernel < 0] = 0
    kernel /= np.sum(kernel)
    return partial(convolve, k=kernel, mode=mode)


def disk_filter(size: int, mode: _pad_modes = "symmetric") -> Callable:
    xx, yy = onp.ogrid[-1 : 1 : 1j * size, -1 : 1 : 1j * size]
    kernel = np.sqrt(xx**2 + yy**2) <= 1
    kernel = kernel / np.sum(kernel)
    return partial(convolve, k=kernel, mode=mode)
