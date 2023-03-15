#!/usr/bin/env python3

import autograd.numpy as np
import nlopt

from phodex.io import filter_stdout
from phodex.layout.meep import MultiportDevice2D, Port
from phodex.optim.callbacks import combine, log_simple, plot_simple
from phodex.optim.nlopt import get_objective
from phodex.topopt.parametrizations import sigmoid_parametrization


def main():
    min_feat = 0.1
    beta = 32
    wvg_width = 0.45
    resolution = 20
    wavelengths = np.linspace(1.53, 1.57, 5)
    design_region = [3, 1.5]
    polarization = "te"

    ports = [Port(wvg_width, "-x", source=True), Port(wvg_width, "+x")]

    p = MultiportDevice2D(
        ports=ports,
        n_core=3.4865,
        n_clad=1.4440,
        wvg_height=0.22,
        wavelengths=wavelengths,
        polarization=polarization,
        resolution=resolution,
        design_region_extent=design_region,
        monitor_size_fac=4,
        damping=1e-2,
        mode_solver="mpb",
    )

    input_flux_far = p.normalizations[0]["flux_far"]

    def loss(s11, s12):
        tran = np.mean(np.abs(s12) ** 2 / input_flux_far)
        return 1 - tran

    obj_funs = [loss]
    mpa_opt = p.get_optimization_problem(obj_funs)

    ks = int(min_feat * p.design_resolution)
    parametrization = sigmoid_parametrization(
        (p.nx, p.ny), ks, beta, filter_type="cone"
    )

    log_cb, state_dict = log_simple(logscale=True)
    plot_cb = plot_simple(mpa_opt, state_dict, output_dir="output")
    nlopt_obj = get_objective(mpa_opt, parametrization, combine(log_cb, plot_cb))

    n = p.nx * p.ny
    x0 = np.full(n, 0.5)
    lb = np.zeros_like(x0)
    ub = np.ones_like(x0)

    opt = nlopt.opt(nlopt.LD_MMA, n)
    opt.set_lower_bounds(lb)
    opt.set_upper_bounds(ub)
    opt.set_min_objective(nlopt_obj)
    opt.set_maxeval(100)

    with filter_stdout("phodex"):
        x0[:] = opt.optimize(x0)


if __name__ == "__main__":
    main()
