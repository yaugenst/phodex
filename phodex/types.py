from typing import Iterable, TypedDict

from meep.simulation import FluxData
from numpy import ndarray


class StateDict(TypedDict):
    obj_hist: list[Iterable[float]]
    epivar_hist: list[float]
    cur_iter: int


class PICNormalizationData(TypedDict):
    flux_near: ndarray
    flux_near_data: FluxData
    flux_far: ndarray
    flux_far_data: FluxData
