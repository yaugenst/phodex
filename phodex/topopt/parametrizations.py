from typing import Callable, Iterable, Literal

import autograd.numpy as np

from phodex.autograd.functions import _pad_modes
from phodex.topopt.filters import cone_filter, gaussian_filter
from phodex.topopt.projections import sigmoid, simp


def sigmoid_parametrization(
    shape: Iterable[int],
    filter_size: int,
    projection_strength: float,
    projection_center: float = 0.5,
    filter_type: Literal["gaussian", "cone"] = "gaussian",
    filter_padding: _pad_modes = "symmetric",
    flat: bool = True,
) -> Callable:
    match filter_type:
        case "gaussian":
            filt = gaussian_filter(filter_size, mode=filter_padding)
        case "cone":
            filt = cone_filter(filter_size, mode=filter_padding)
        case _:
            raise ValueError(f"Unkown filter type: {filter_type}.")

    def _parametrization(x: np.ndarray) -> np.ndarray:
        x = np.reshape(x, shape)
        x = filt(x)
        x = sigmoid(x, projection_strength, projection_center)
        return x.ravel() if flat else x

    return _parametrization


def simp_parametrization(
    shape: Iterable[int],
    filter_size: int,
    vmin: float = 0.0,
    vmax: float = 1.0,
    penalty: float = 3.0,
    filter_type: Literal["gaussian", "cone"] = "gaussian",
    filter_padding: _pad_modes = "symmetric",
    flat: bool = True,
) -> Callable:
    match filter_type:
        case "gaussian":
            filt = gaussian_filter(filter_size, mode=filter_padding)
        case "cone":
            filt = cone_filter(filter_size, mode=filter_padding)
        case _:
            raise ValueError(f"Unkown filter type: {filter_type}.")

    def _parametrization(x: np.ndarray) -> np.ndarray:
        x = np.reshape(x, shape)
        x = filt(x)
        x = simp(x, vmin, vmax, penalty)
        return x.ravel() if flat else x

    return _parametrization
