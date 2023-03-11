from typing import Iterable

import autograd.numpy as np


def reflection_pad(
    x: np.ndarray, pad: int, axis: int | Iterable | None = None
) -> np.ndarray:
    if axis is None:
        axis = list(range(x.ndim))
    if isinstance(axis, int):
        axis = [axis]

    for ax in axis:
        if ax >= len(x.shape):
            raise IndexError(f"axis {ax} out of range")

        idx = tuple(
            slice(None) if s != ax else slice(-1, -pad - 1, -1) for s in range(x.ndim)
        )
        x = np.concatenate([x, x[idx]], axis=ax)

        idx = tuple(
            slice(None) if s != ax else slice(pad - 1, None, -1) for s in range(x.ndim)
        )
        x = np.concatenate([x[idx], x], axis=ax)

    return x


def circular_pad(
    x: np.ndarray, pad: int, axis: int | Iterable | None = None
) -> np.ndarray:
    if axis is None:
        axis = list(range(x.ndim))
    if isinstance(axis, int):
        axis = [axis]

    for ax in axis:
        if ax >= len(x.shape):
            raise IndexError(f"axis {ax} out of range")

        idx = tuple(slice(None) if s != ax else slice(pad) for s in range(len(x.shape)))
        x = np.concatenate([x, x[idx]], axis=ax)

        idx = tuple(
            slice(None) if s != ax else slice(-2 * pad, -pad, None)
            for s in range(len(x.shape))
        )
        x = np.concatenate([x[idx], x], axis=ax)

    return x
