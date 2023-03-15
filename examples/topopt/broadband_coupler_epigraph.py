#!/usr/bin/env python3

import autograd.numpy as np
import nlopt

from phodex.autograd.functions import grey_closing, grey_opening
from phodex.io import filter_stdout
from phodex.layout.meep import MultiportDevice2D, Port
from phodex.optim.callbacks import combine, log_epigraph, plot_epigraph
from phodex.optim.nlopt import get_epigraph_formulation
from phodex.topopt.parametrizations import sigmoid_parametrization


def main():
    min_feat = 0.1
    beta = 64
    wvg_width = 0.5
    resolution = 30
    polarization = "te"
    wavelengths = np.linspace(1.53, 1.57, 5)
    design_region = [6, 3]

    ports = [
        Port(wvg_width, "-x", offset=1, source=True),
        Port(wvg_width, "-x", offset=-1),
        Port(wvg_width, "+x", offset=1),
        Port(wvg_width, "+x", offset=-1),
    ]

    p = MultiportDevice2D(
        ports=ports,
        n_core=3.4865,
        n_clad=1.4440,
        wvg_height=0.22,
        wavelengths=wavelengths,
        polarization=polarization,
        resolution=resolution,
        design_resolution=2 * resolution,
        design_region_extent=design_region,
        monitor_size_fac=4,
        damping=1e-3,
        mode_solver="mpb",
    )

    input_flux_far = p.normalizations[0]["flux_far"]

    def p13(s11, s12, s13, s14):
        p13 = np.abs(s13) ** 2 / input_flux_far
        return 0.5 - p13

    def p14(s11, s12, s13, s14):
        p14 = np.abs(s14) ** 2 / input_flux_far
        return 0.5 - p14

    mpa_opt = p.get_optimization_problem([p13, p14])

    ks = int(min_feat * p.design_resolution)
    if ks % 2 == 0:
        ks += 1

    filter_and_project = sigmoid_parametrization(
        (p.nx, p.ny), ks, beta, filter_type="gaussian", flat=False
    )

    def parametrization(x):
        x = np.reshape(x, (p.nx // 2, p.ny // 2))
        x = np.concatenate([x, np.fliplr(x)], axis=1)
        x = np.concatenate([x, np.flipud(x)], axis=0)
        x = filter_and_project(x)
        x = grey_opening(x, np.ones((ks, ks)), "symmetric")
        x = grey_closing(x, np.ones((ks, ks)), "symmetric")
        return x.ravel()

    log_cb, state_dict = log_epigraph(logscale=True)
    plot_cb = plot_epigraph(mpa_opt, state_dict, output_dir="output")
    nlopt_obj, epi_cst = get_epigraph_formulation(
        mpa_opt, parametrization, combine(log_cb, plot_cb)
    )
    epi_tol = np.full(len(mpa_opt.objective_functions) * p.nfreq, 1e-3)

    n = p.nx // 2 * p.ny // 2 + 1
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

    with filter_stdout("iteration"):
        x0[:] = opt.optimize(x0)


if __name__ == "__main__":
    main()
