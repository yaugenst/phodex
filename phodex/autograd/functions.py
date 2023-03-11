from typing import Iterable, Literal

import autograd.numpy as np
from autograd.scipy.signal import convolve as convolve_ag


def edge_pad(x: np.ndarray, pad: int, axis: int | Iterable | None = None) -> np.ndarray:
    if axis is None:
        axis = list(range(x.ndim))
    if isinstance(axis, int):
        axis = [axis]

    for ax in axis:
        if ax >= len(x.shape):
            raise IndexError(f"axis {ax} out of range")

        idx = tuple(slice(None) if s != ax else -1 for s in range(len(x.shape)))
        x = np.concatenate([x, np.stack(pad // 2 * [x[idx]], axis=ax)], axis=ax)

        idx = tuple(slice(None) if s != ax else 0 for s in range(len(x.shape)))
        x = np.concatenate([np.stack(pad // 2 * [x[idx]], axis=ax), x], axis=ax)

    return x


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


def convolve(
    x: np.ndarray,
    k: np.ndarray,
    mode: Literal["constant", "edge", "reflect", "wrap"] = "constant",
) -> np.ndarray:
    if len(set(k.shape)) > 1:
        raise NotImplementedError(
            f"Non-square kernels not implemented yet! Got {k.shape}"
        )
    if k.shape[0] % 2 == 0:
        raise NotImplementedError(
            f"Even-sized kernels not supported yet! Got {k.shape}"
        )

    p = k.shape[0] // 2

    match mode:
        case "constant":
            x = np.pad(x, p, mode="constant")
        case "edge":
            x = edge_pad(x, p)
        case "reflect":
            x = reflection_pad(x, p)
        case "wrap":
            x = circular_pad(x, p)
        case _:
            raise ValueError(f"Unsupported padding mode: {mode}")

    x = convolve_ag(x, k, mode="valid")
    return x
