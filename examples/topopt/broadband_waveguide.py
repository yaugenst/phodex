#!/usr/bin/env python3

import autograd.numpy as np
import matplotlib.pyplot as plt
import meep as mp
import nlopt
from cycler import cycler
from loguru import logger
from matplotlib import gridspec

from phodex.io import filter_stdout
from phodex.layout.meep import MultiportDevice2D, Port
from phodex.optim.nlopt import get_epigraph_formulation
from phodex.plotting import add_legend_grid
from phodex.topopt.parametrizations import sigmoid_parametrization


def main():
    sigma = 1
    beta = 64
    wvg_width = 0.5
    resolution = 20
    wavelengths = np.linspace(1.5, 1.6, 5)
    design_region = [4, 2]

    ports = [Port(wvg_width, "-x", source=True), Port(wvg_width, "+x")]

    p = MultiportDevice2D(
        ports=ports,
        n_core=3.4865,
        n_clad=1.4440,
        wvg_height=0.22,
        wavelengths=wavelengths,
        resolution=resolution,
        design_region_extent=design_region,
        monitor_size_fac=6,
    )

    input_flux, _ = p.normalizations[0]

    def tran(s11, s12):
        p = np.abs(s12) ** 2 / input_flux
        return 1 - p

    def refl(s11, s12):
        p = np.abs(s11) ** 2 / input_flux
        return p

    obj_funs = [tran, refl]
    mpa_opt = p.get_optimization_problem(obj_funs)

    parametrization = sigmoid_parametrization((p.nx, p.ny), sigma, beta)
    state_dict = {"obj_hist": [], "epivar_hist": [], "cur_iter": 0}

    def callback(t, v, f0, grad):
        design = np.real(mpa_opt.sim.get_epsilon()) - p.n_clad**2
        design /= p.n_core**2 - p.n_clad**2
        xx, yy, _, _ = mpa_opt.sim.get_array_metadata()

        if not mp.am_master():
            return  # only do callback stuff on master

        state_dict["obj_hist"].append(f0)
        state_dict["epivar_hist"].append(t)
        state_dict["cur_iter"] += 1

        fig = plt.figure(figsize=(9, 6), tight_layout=True)
        gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1])

        ax00 = fig.add_subplot(gs[0, 0])
        prop_cycler = cycler(
            color=plt.cm.inferno(np.linspace(0.1, 0.9, len(p.wavelengths)))
        ) * cycler(linestyle=["-", "--", ":", "-."][: len(obj_funs)])
        ax00.set_prop_cycle(prop_cycler)
        for wvl_id in range(len(p.wavelengths)):
            for obj_id in range(len(obj_funs)):
                idx = wvl_id * len(obj_funs) + obj_id
                ax00.plot(np.asarray(state_dict["obj_hist"])[:, idx], label=" ")
        ax00.set_xlabel("iteration")
        ax00.set_ylabel("objectives")
        ax00.set_yscale("log")

        # legend
        row_names = [f"{int(1000 * w)} nm" for w in p.wavelengths]
        col_names = [f.__name__ for f in obj_funs]
        legend_ax = fig.add_subplot(gs[1, :])
        handles, labels = ax00.get_legend_handles_labels()
        add_legend_grid(handles, labels, row_names, col_names, legend_ax)

        # epigraph variable
        ax01 = ax00.twinx()
        color = "tab:blue"
        ax01.plot(state_dict["epivar_hist"], color=color)
        ax01.tick_params(axis="y", labelcolor=color)
        ax01.set_ylabel("epigraph dummy", color=color)

        # device
        ax1 = fig.add_subplot(gs[0, 1])
        ax1.pcolormesh(xx, yy, design.T, cmap="gray_r", vmin=0, vmax=1)
        ax1.set_aspect("equal")
        ax1.set_xlabel("x (μm)")
        ax1.set_ylabel("y (μm)")

        fig.savefig(
            f'output/out{state_dict["cur_iter"]-1:03d}.png', bbox_inches="tight"
        )
        plt.close()

        logger.info(
            f'iteration: {state_dict["cur_iter"]-1:3d}, t: {t:11.4e}, objective (dB): '
            "[" + ", ".join(f"{10*np.log10(ff):6.2f}" for ff in f0) + "]",
            flush=True,
        )

    nlopt_obj, epi_cst = get_epigraph_formulation(mpa_opt, parametrization, callback)
    epi_tol = np.full(len(obj_funs) * p.nfreq, 1e-4)

    n = p.nx * p.ny + 1
    x0 = np.full(n, 0.5)
    lb = np.zeros_like(x0)
    ub = np.ones_like(x0)

    t0, _ = mpa_opt([parametrization(x0[1:])], need_gradient=False)
    x0[0] = 1.05 * np.max(t0)
    lb[0] = -np.inf
    ub[0] = np.inf

    opt = nlopt.opt(nlopt.LD_MMA, n)
    opt.set_lower_bounds(lb)
    opt.set_upper_bounds(ub)
    opt.set_min_objective(nlopt_obj)
    opt.add_inequality_mconstraint(epi_cst, epi_tol)
    opt.set_param("dual_ftol_rel", 1e-7)
    opt.set_maxeval(100)

    with filter_stdout("phodex"):
        x0[:] = opt.optimize(x0)


if __name__ == "__main__":
    main()
