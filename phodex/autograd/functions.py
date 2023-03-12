from typing import Iterable, Literal

import autograd.numpy as np
from autograd.scipy.signal import convolve as convolve_ag

_pad_modes = Literal["constant", "edge", "reflect", "symmetric", "wrap"]


def edge_pad(x: np.ndarray, pad: int, axis: int | Iterable | None = None) -> np.ndarray:
    if axis is None:
        axis = list(range(x.ndim))
    if isinstance(axis, int):
        axis = [axis]

    for ax in axis:
        if ax < 0:
            ax = x.ndim + ax
        if ax >= len(x.shape):
            raise IndexError(f"axis {ax} out of range")

        idx = tuple(slice(None) if s != ax else -1 for s in range(len(x.shape)))
        x = np.concatenate([x, np.stack(pad * [x[idx]], axis=ax)], axis=ax)

        idx = tuple(slice(None) if s != ax else 0 for s in range(len(x.shape)))
        x = np.concatenate([np.stack(pad * [x[idx]], axis=ax), x], axis=ax)

    return x


def reflection_pad(x: np.ndarray, pad: int, axis=None) -> np.ndarray:
    if axis is None:
        axis = list(range(x.ndim))
    if isinstance(axis, int):
        axis = [axis]

    for ax in axis:
        if ax < 0:
            ax = x.ndim + ax
        if ax >= len(x.shape):
            raise IndexError(f"axis {ax} out of range")

        idx = tuple(
            slice(None) if s != ax else slice(-2, -pad - 2, -1) for s in range(x.ndim)
        )
        x = np.concatenate([x, x[idx]], axis=ax)

        idx = tuple(
            slice(None) if s != ax else slice(pad, 0, -1) for s in range(x.ndim)
        )
        x = np.concatenate([x[idx], x], axis=ax)

    return x


def symmetric_pad(
    x: np.ndarray, pad: int, axis: int | Iterable | None = None
) -> np.ndarray:
    if axis is None:
        axis = list(range(x.ndim))
    if isinstance(axis, int):
        axis = [axis]

    for ax in axis:
        if ax < 0:
            ax = x.ndim + ax
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
        if ax < 0:
            ax = x.ndim + ax
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


def pad(
    x: np.ndarray,
    pad: int,
    axis: int | Iterable | None = None,
    mode: _pad_modes = "constant",
) -> np.ndarray:
    if pad >= x.shape[-1]:
        raise NotImplementedError(
            "Padding larger than the input size is not supported!"
        )
    if pad < 0:
        raise ValueError("Padding must be >= 0!")
    if pad == 0:
        return x

    if axis is None:
        pad_axes = (-2, -1)

    match mode:
        case "constant":
            p = (x.ndim - 2) * [(0, 0)] + 2 * [(pad, pad)]
            return np.pad(x, p, mode="constant")
        case "edge":
            return edge_pad(x, pad, pad_axes)
        case "reflect":
            return reflection_pad(x, pad, pad_axes)
        case "symmetric":
            return symmetric_pad(x, pad, pad_axes)
        case "wrap":
            return circular_pad(x, pad, pad_axes)
        case _:
            raise ValueError(f"Unsupported padding mode: {mode}")


def convolve(
    x: np.ndarray,
    k: np.ndarray,
    axes: Iterable[Iterable[int]] | None = None,
    dot_axes: Iterable[Iterable[int]] = [(), ()],
    mode: _pad_modes = "constant",
) -> np.ndarray:
    if len(set(k.shape[-2:])) > 1:
        raise NotImplementedError(
            f"Non-square kernels not implemented yet! Got {k.shape}"
        )
    if k.shape[-1] % 2 == 0:
        raise NotImplementedError(
            f"Even-sized kernels not supported yet! Got {k.shape}"
        )

    pad_axes = axes[0] if axes is not None else None
    p = k.shape[-1] // 2

    x = pad(x, p, pad_axes, mode)
    x = convolve_ag(x, k, axes, dot_axes, mode="valid")
    return x
