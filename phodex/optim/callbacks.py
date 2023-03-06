from pathlib import Path
from typing import Callable, Iterable, TypedDict

import matplotlib.pyplot as plt
import meep as mp
import meep.adjoint as mpa
import numpy as np
from cycler import cycler
from loguru import logger
from matplotlib import figure, gridspec

from phodex.layout.meep import MultiportDevice2D
from phodex.plotting import add_legend_grid


class StateDict(TypedDict):
    obj_hist: list[Iterable[float]]
    epivar_hist: list[float]
    cur_iter: int


def combine(*functions: Callable) -> Callable:
    def _combined(*args, **kwargs):
        [f(*args, **kwargs) for f in functions]

    return _combined


def logging_callback(
    state_dict: StateDict | None = None, logscale: bool = False
) -> Callable:
    if state_dict is None:
        state_dict = {"obj_hist": [], "epivar_hist": [], "cur_iter": 0}

    def post(x):
        if logscale:
            return 10 * np.log10(x)
        return x

    def _callback(t, v, f0, grad) -> None:
        state_dict["obj_hist"].append(f0)
        state_dict["epivar_hist"].append(t)
        state_dict["cur_iter"] += 1

        if not mp.am_master():
            return

        logger.info(
            f'iteration: {state_dict["cur_iter"]-1:3d}, t: {t:11.4e}, objective (dB): '
            "[" + ", ".join(f"{post(ff):6.2f}" for ff in f0) + "]",
            flush=True,
        )

    return _callback


def plotting_callback(
    mpa_opt: mpa.OptimizationProblem,
    device: MultiportDevice2D,
    state_dict: StateDict,
    figure: figure.Figure | None = None,
    output_dir: Path | str | None = None,
) -> Callable:
    obj_funs = mpa_opt.objective_functions
    nrows = len(device.wavelengths)
    ncols = len(obj_funs)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving plots to {output_dir.resolve()}.")

    if figure is None:
        figure = plt.figure(figsize=(9, 6), tight_layout=True)

    def _callback(t, v, f0, grad):
        design = np.real(mpa_opt.sim.get_epsilon()) - device.n_clad**2
        design /= device.n_core**2 - device.n_clad**2
        xx, yy, _, _ = mpa_opt.sim.get_array_metadata()

        if not mp.am_master:
            return

        figure.clf()

        gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1])

        ax00 = figure.add_subplot(gs[0, 0])
        prop_cycler = cycler(
            color=plt.cm.inferno(np.linspace(0.1, 0.9, nrows))
        ) * cycler(linestyle=["-", "--", ":", "-."][:ncols])
        ax00.set_prop_cycle(prop_cycler)
        for wvl_id in range(nrows):
            for obj_id in range(ncols):
                idx = wvl_id * ncols + obj_id
                ax00.plot(np.asarray(state_dict["obj_hist"])[:, idx], label=" ")
        ax00.set_xlabel("iteration")
        ax00.set_ylabel("objectives")
        ax00.set_yscale("log")

        # legend
        row_names = [f"{int(1000 * w)} nm" for w in device.wavelengths]
        col_names = [f.__name__ for f in obj_funs]
        legend_ax = figure.add_subplot(gs[1, :])
        handles, labels = ax00.get_legend_handles_labels()
        add_legend_grid(handles, labels, row_names, col_names, legend_ax)

        # epigraph variable
        ax01 = ax00.twinx()
        color = "tab:blue"
        ax01.plot(state_dict["epivar_hist"], color=color)
        ax01.tick_params(axis="y", labelcolor=color)
        ax01.set_ylabel("epigraph dummy", color=color)

        # device
        ax1 = figure.add_subplot(gs[0, 1])
        ax1.pcolormesh(xx, yy, design.T, cmap="gray_r", vmin=0, vmax=1)
        ax1.set_aspect("equal")
        ax1.set_xlabel("x (μm)")
        ax1.set_ylabel("y (μm)")

        if output_dir is not None:
            figure.savefig(
                output_dir / f'out{state_dict["cur_iter"]-1:03d}.png',
                bbox_inches="tight",
            )

    return _callback
