#!/usr/bin/env python3

import autograd.numpy as np
import nlopt

from phodex.io import filter_stdout
from phodex.layout.meep import MultiportDevice2D, Port
from phodex.optim.callbacks import combine, logging_callback, plotting_callback
from phodex.optim.nlopt import get_epigraph_formulation
from phodex.topopt.parametrizations import sigmoid_parametrization


def main():
    sigma = 4
    beta = 20
    wvg_width = 0.5
    resolution = 20
    wavelengths = np.linspace(1.5, 1.6, 5)
    design_region = [3, 1.5]

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
        damping=1e-2,
    )

    input_flux_far = p.normalizations[0]["flux_far"]
    input_flux_near = p.normalizations[0]["flux_near"]

    def loss(s11, s12):
        return 1 - np.abs(s12) ** 2 / input_flux_far

    def refl(s11, s12):
        refl = np.abs(s11) ** 2 / input_flux_near
        return refl

    obj_funs = [loss, refl]
    mpa_opt = p.get_optimization_problem(obj_funs)

    parametrization = sigmoid_parametrization((p.nx, p.ny), sigma, beta)

    state_dict = {"obj_hist": [], "epivar_hist": [], "cur_iter": 0}
    log_cb = logging_callback(state_dict, logscale=True)
    plot_cb = plotting_callback(mpa_opt, p, state_dict, output_dir="output")

    nlopt_obj, epi_cst = get_epigraph_formulation(
        mpa_opt, parametrization, combine(log_cb, plot_cb)
    )
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
