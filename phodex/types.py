from typing import Iterable, NotRequired, TypedDict

from meep.simulation import FluxData
from numpy import ndarray


class StateDict(TypedDict):
    obj_hist: list[Iterable[float]]
    cur_iter: int
    epivar_hist: NotRequired[list[float]]


class PICNormalizationData(TypedDict):
    flux_near: ndarray
    flux_near_data: FluxData
    flux_far: ndarray
    flux_far_data: FluxData
