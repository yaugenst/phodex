#!/usr/bin/env python3

import autograd.numpy as np
import matplotlib.pyplot as plt
import meep as mp
import nlopt

from phodex.io import filter_stdout
from phodex.optim.nlopt import get_epigraph_formulation
from phodex.topopt.parametrizations import sigmoid_parametrization
from phodex.topopt.problems import FilterProblem


def main():
    sigma = 1
    beta = 64
    wavelengths = np.linspace(1.5, 1.6, 3)
    p = FilterProblem(
        resolution=20,
        wavelengths=wavelengths,
        design_region_xy=(4, 2),
        use_effective_index=True,
    )
    input_flux, _ = p.normalization

    def extinction(_, s12):
        t = np.abs(s12) ** 2 / input_flux
        return np.array([t[0], 1 - t[1], t[2]])

    def reflection(s11, _):
        r = np.abs(s11) ** 2 / input_flux
        return r

    obj_funs = [extinction, reflection]
    # obj_funs = [extinction]
    mpa_opt = p.get_optimization_problem(obj_funs)
    parametrization = sigmoid_parametrization((p.nx, p.ny), sigma, beta)
    state_dict = {"obj_hist": [], "epivar_hist": [], "cur_iter": 0}

    def callback(t, v, f0, grad):
        design = np.real(mpa_opt.sim.get_epsilon())
        xx, yy, _, _ = mpa_opt.sim.get_array_metadata()
        if not mp.am_master():
            return
        fig, ax = plt.subplots(1, 1)
        ax.pcolormesh(xx, yy, design.T, cmap="gray_r")
        ax.set_aspect("equal")
        fig.tight_layout()
        fig.savefig(f'output/out{state_dict["cur_iter"]:03d}.png')
        plt.close()

        print(
            f'iteration: {state_dict["cur_iter"]:3d}, t: {t:9.5f}, objective (dB): '
            "[" + ", ".join(f"{10*np.log10(ff):6.2f}" for ff in f0) + "]",
            flush=True,
        )

        state_dict["obj_hist"].append(f0)
        state_dict["epivar_hist"].append(t)
        state_dict["cur_iter"] += 1

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

    opt = nlopt.opt(nlopt.LD_CCSAQ, n)
    opt.set_lower_bounds(lb)
    opt.set_upper_bounds(ub)
    opt.set_min_objective(nlopt_obj)
    opt.add_inequality_mconstraint(epi_cst, epi_tol)
    opt.set_param("dual_ftol_rel", 1e-7)
    opt.set_maxeval(1)

    with filter_stdout("iteration"):
        x0[:] = opt.optimize(x0)


if __name__ == "__main__":
    main()
