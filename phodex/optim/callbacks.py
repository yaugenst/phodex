from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import meep as mp
import meep.adjoint as mpa
import numpy as np
from cycler import cycler
from loguru import logger
from matplotlib import figure, gridspec

from phodex.plotting import add_legend_grid
from phodex.types import StateDict


def combine(*functions: Callable) -> Callable:
    def _combined(*args, **kwargs):
        [f(*args, **kwargs) for f in functions]

    return _combined


def log_simple(state_dict: StateDict | None = None, logscale: bool = False) -> Callable:
    if state_dict is None:
        state_dict = {"obj_hist": [], "cur_iter": 0}

    def post(x: np.ndarray):
        if logscale:
            return 10 * np.log10(x)
        return x

    def _callback(x, f0, grad) -> None:
        state_dict["obj_hist"].append(f0)
        state_dict["cur_iter"] += 1

        if not mp.am_master():
            return

        logger.info(
            f'iteration: {state_dict["cur_iter"]-1:3d}, t: {t:11.4e}, '
            f"objective: {post(f0)}",
            flush=True,
        )

    return _callback, state_dict


def plot_simple(
    mpa_opt: mpa.OptimizationProblem,
    state_dict: StateDict,
    figure: figure.Figure | None = None,
    output_dir: Path | str | None = None,
) -> Callable:
    obj_funs = mpa_opt.objective_functions

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving plots to {output_dir.resolve()}.")

    if figure is None:
        figure, ax = plt.subplots(1, 2, figsize=(9, 4), tight_layout=True)

    def _callback(x: np.ndarray, f0: float, grad: np.ndarray) -> None:
        ax[0].cla()
        ax[0].plot(np.asarray(state_dict["obj_hist"]))
        ax[0].set_xlabel("iteration")
        ax[0].set_ylabel(obj_funs[0].__name__)
        ax[0].set_yscale("log")

        # device
        ax[1].cla()
        mpa_opt.plot2D(ax=ax[1], plot_monitors_flag=False, plot_sources_flag=False)

        if mp.am_master() and output_dir is not None:
            figure.savefig(
                output_dir / f'out{state_dict["cur_iter"]-1:03d}.png',
                bbox_inches="tight",
            )

    return _callback


def log_epigraph(
    state_dict: StateDict | None = None, logscale: bool = False
) -> Callable:
    if state_dict is None:
        state_dict = {"obj_hist": [], "epivar_hist": [], "cur_iter": 0}

    def post(x: np.ndarray) -> np.ndarray:
        if logscale:
            return 10 * np.log10(x)
        return x

    def _callback(x: np.ndarray, f0: float, grad: np.ndarray) -> None:
        t = x[0]

        state_dict["obj_hist"].append(f0)
        state_dict["epivar_hist"].append(t)
        state_dict["cur_iter"] += 1

        if not mp.am_master():
            return

        logger.info(
            f'iteration: {state_dict["cur_iter"]-1:3d}, t: {t:11.4e}, objective: '
            "[" + ", ".join(f"{post(ff):6.2f}" for ff in f0) + "]",
            flush=True,
        )

    return _callback, state_dict


def plot_epigraph(
    mpa_opt: mpa.OptimizationProblem,
    state_dict: StateDict,
    figure: figure.Figure | None = None,
    output_dir: Path | str | None = None,
) -> Callable:
    obj_funs = mpa_opt.objective_functions
    nrows = len(mpa_opt.frequencies)
    ncols = len(obj_funs)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving plots to {output_dir.resolve()}.")

    if figure is None:
        figure = plt.figure(figsize=(9, 6), tight_layout=True)

    def _callback(x: np.ndarray, f0: float, grad: np.ndarray) -> None:
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
        row_names = [f"{int(1000 * w)} nm" for w in 1 / mpa_opt.wavelengths]
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
        mpa_opt.plot2D(ax=ax1, plot_monitors_flag=False, plot_sources_flag=False)

        if mp.am_master() and output_dir is not None:
            figure.savefig(
                output_dir / f'out{state_dict["cur_iter"]-1:03d}.png',
                bbox_inches="tight",
            )

    return _callback
