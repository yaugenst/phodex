#!/usr/bin/env python3

import autograd.numpy as np
import nlopt

from phodex.io import filter_stdout
from phodex.layout.meep import MultiportDevice2D, Port
from phodex.optim.callbacks import combine, log_epigraph, plot_epigraph
from phodex.optim.nlopt import get_epigraph_formulation
from phodex.topopt.parametrizations import sigmoid_parametrization


def main():
    min_feat = 0.1
    beta = 64
    wvg_width = 0.45
    resolution = 30
    wavelengths = [1.45, 1.55]
    design_region = [4, 2]
    polarization = "te"

    ports = [
        Port(wvg_width, "-x", source=True),
        Port(wvg_width, "+x", offset=0.5),
        Port(wvg_width, "+x", offset=-0.5),
    ]

    p = [
        MultiportDevice2D(
            ports=ports,
            n_core=3.4865,
            n_clad=1.4440,
            wvg_height=0.22,
            wavelengths=[w],
            polarization=polarization,
            resolution=resolution,
            design_resolution=100,
            design_region_extent=design_region,
            monitor_size_fac=4,
            damping=1e-2,
            mode_solver="mpb",
        )
        for w in wavelengths
    ]

    input_flux_far = [pi.normalizations[0]["flux_far"] for pi in p]

    def loss_1450(s11, s12, s13):
        t12 = np.abs(s12) ** 2 / input_flux_far[0]
        return 1 - t12

    def loss_1550(s11, s12, s13):
        t13 = np.abs(s13) ** 2 / input_flux_far[1]
        return 1 - t13

    obj_funs = [loss_1450, loss_1550]
    mpa_opt = [
        pi.get_optimization_problem([obj_fun]) for pi, obj_fun in zip(p, obj_funs)
    ]

    ks = int(min_feat * p[0].design_resolution)
    if ks % 2 == 0:
        ks += 1

    parametrization = sigmoid_parametrization(
        (p[0].nx, p[0].ny), ks, beta, filter_type="cone"
    )

    log_cb, state_dict = log_epigraph(logscale=True)
    plot_cb = plot_epigraph(mpa_opt[0], state_dict, output_dir="output")
    nlopt_obj, epi_cst0 = get_epigraph_formulation(
        mpa_opt[0], parametrization, combine(log_cb, plot_cb)
    )
    _, epi_cst1 = get_epigraph_formulation(mpa_opt[1], parametrization)
    epi_tol = np.full(len(mpa_opt[0].objective_functions) * p[0].nfreq, 1e-3)

    n = p[0].nx * p[0].ny + 1
    x0 = np.full(n, 0.5)
    lb = np.zeros_like(x0)
    ub = np.ones_like(x0)

    t0, _ = mpa_opt[0]([parametrization(x0[1:])], need_gradient=False)
    x0[0] = 1.05 * np.max(t0)
    lb[0] = -np.inf
    ub[0] = np.inf

    opt = nlopt.opt(nlopt.LD_MMA, n)
    opt.set_lower_bounds(lb)
    opt.set_upper_bounds(ub)
    opt.set_min_objective(nlopt_obj)
    opt.add_inequality_mconstraint(epi_cst0, epi_tol)
    opt.add_inequality_mconstraint(epi_cst1, epi_tol)
    opt.set_param("dual_ftol_rel", 1e-7)
    opt.set_maxeval(100)

    with filter_stdout("phodex"):
        x0[:] = opt.optimize(x0)


if __name__ == "__main__":
    main()
