#!/usr/bin/env python3

from itertools import count

import autograd.numpy as np
import h5py
import matplotlib.pyplot as plt
import meep as mp
import meep.adjoint as mpa
import nlopt
from autograd import tensor_jacobian_product, value_and_grad
from gdsfactory.simulation.modes import find_modes_waveguide
from matplotlib.colors import CenteredNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tofea.fea2d import FEA2D_T
from tofea.topopt_helpers import gaussian_filter, gray_indicator, sigmoid_projection

mp.Verbosity(0)

sim_res = 20
design_res = 100
n_si = 3.4777
n_sio2 = 1.444
d_wvg = 0.5  # waveguide width
z_wvg = 0.22  # waveguide height
lcen = 1.55
dpml = 1.0
design_region_x = 5
design_region_y = 3
buffer_air = 0.5  # air gap in y direction (outside of design region)
buffer_wvg = 1.0  # how much waveguide beyond the design region
bfac = 40  # initial binarization strength
sigma = 25  # feature size in resolution of design_res
con_tol = 1e-4  # tolerance for the connectivity constraint - just needs to be small

fcen = 1 / lcen
df = 0.2

nx = int(design_res * design_region_x)
ny = int(design_res * design_region_y)
dofs = (nx // 2, ny // 2)  # number of design variables in (x, y)


def create_parametrizations(shape, sigma, beta, cmin=1.0, cmax=1e6):
    """Create two paramettrizations, one for Meep, and one for the FEM solver.
    The FEM parametrization is the same as FDTD, but needs to rescale the materials
    to the appropriate conductivities cmin and cmax.
    """

    def _fdtd(x):
        x = np.reshape(x, shape)
        x = np.concatenate([x, np.fliplr(x)], axis=1)
        x = np.concatenate([x, np.flipud(x)], axis=0)
        x = gaussian_filter(x, sigma)
        x = sigmoid_projection(x, beta)
        return x

    def _fem(x):
        x = _fdtd(x)
        x = cmin + (cmax - cmin) * x
        return x

    return _fdtd, _fem


# first we calculate the waveguide mode so we can do effective index simulations
neff = find_modes_waveguide(
    parity=mp.NO_PARITY,
    wg_width=0.5,
    ncore=n_si,
    nclad=n_sio2,
    wg_thickness=z_wvg,
    resolution=2 * sim_res,
    sy=3,
    sz=3,
    nmodes=1,
)[1].neff

# simulation cell size
cell = mp.Vector3(
    2 * dpml + 2 * buffer_wvg + design_region_x,
    2 * dpml + 2 * buffer_air + design_region_y,
)

core = mp.Medium(index=n_si)
clad = mp.Medium(index=n_sio2)

design_variables = mp.MaterialGrid(mp.Vector3(nx, ny), clad, core, grid_type="U_MEAN")
design_region = mpa.DesignRegion(
    design_variables,
    volume=mp.Volume(
        center=mp.Vector3(),
        size=mp.Vector3(design_region_x, design_region_y),
    ),
)

# define our geometry - first, the design region
geometry = [
    mp.Block(
        center=design_region.center, size=design_region.size, material=design_variables
    ),
]

# and now the four waveguides
for y_offset in [0.75, -0.75]:
    for x_dir in [1, -1]:
        geometry.append(
            mp.Block(
                center=mp.Vector3(x_dir * (cell.x - dpml - buffer_wvg) / 2, y_offset),
                size=mp.Vector3(dpml + buffer_wvg, d_wvg),
                material=core,
            )
        )

# because we have mirror symmetry in both x and y, we only need one source
# due to reciprocity, even though we have two input ports
source = mp.EigenModeSource(
    mp.GaussianSource(frequency=fcen, fwidth=df),
    eig_band=1,
    direction=mp.NO_DIRECTION,
    eig_kpoint=mp.Vector3(1, 0),
    size=mp.Vector3(0, 1.2),
    center=mp.Vector3(-cell.x / 2 + dpml, -0.75),
)

sim = mp.Simulation(
    resolution=sim_res,
    cell_size=cell,
    boundary_layers=[mp.PML(dpml)],
    geometry=geometry,
    sources=[source],
    default_material=clad,
)

s11_monitor = mpa.EigenmodeCoefficient(
    sim,
    mp.Volume(
        size=mp.Vector3(0, 1.2),
        center=mp.Vector3(source.center.x + 0.1, source.center.y),
    ),
    mode=1,
)
s13_monitor = mpa.EigenmodeCoefficient(
    sim,
    mp.Volume(size=mp.Vector3(0, 1.2), center=mp.Vector3(cell.x / 2 - dpml, 0.75)),
    mode=1,
)
s14_monitor = mpa.EigenmodeCoefficient(
    sim,
    mp.Volume(size=mp.Vector3(0, 1.2), center=mp.Vector3(cell.x / 2 - dpml, -0.75)),
    mode=1,
)


def mpa_objective(s11, s13, s14):
    """Our actual objective - power in both output ports (normalized by input).
    We subtract their difference, so that when we maximize, we minimize the difference
    between the outputs. A more robust approach would be to do an epigraph formulation,
    but this will work fine in a pinch...
    """
    p13 = np.abs(s13 / s11) ** 2
    p14 = np.abs(s14 / s11) ** 2
    diff = np.abs(p13 - p14)
    return p13 + p14 - diff


mpa_opt = mpa.OptimizationProblem(
    simulation=sim,
    objective_functions=mpa_objective,
    objective_arguments=[s11_monitor, s13_monitor, s14_monitor],
    design_regions=[design_region],
    fcen=fcen,
    df=0,
    nf=1,
)

x0 = np.full(dofs, 0.5).ravel()
fdtd_parametrization, fem_parametrization = create_parametrizations(dofs, sigma, bfac)
bin_tol = 1.1
counter = count()
obj_vals = []

# now we set up the FEM calculation for the connectivity constraint
fem_dofs = np.arange((nx + 1) * (ny + 1)).reshape(nx + 1, ny + 1)
fixed = np.zeros_like(fem_dofs, dtype=bool)
load = np.zeros_like(fem_dofs)
load[0, (75, 225)] = 1  # heat sources at the inputs
fixed[-1, (75, 225)] = 1  # heat sinks at the output
fem = FEA2D_T((nx, ny), fem_dofs, fixed, load)

# we let the optimization converge and then restart with higher binarization until
# good binarization is achieved
while np.log10(gray_indicator(fdtd_parametrization(x0))) > -2:
    if mp.am_master():
        print(f"\n{bfac=}, {bin_tol=}\n", flush=True)

    def nlopt_obj(x, gd):
        xp = fdtd_parametrization(x)
        v, g = mpa_opt([xp.ravel()])
        g = np.reshape(g, xp.shape)
        gg = tensor_jacobian_product(fdtd_parametrization, 0)(x, g)

        if gd.size > 0:
            gd[:] = gg

        neval = next(counter)
        if mp.am_master():
            print(f"{neval:>4}: {v=}", flush=True)

            if neval % 10 == 0:
                fig, ax = plt.subplots(2, 1)
                im1 = ax[0].imshow(
                    np.reshape(gg, dofs).T, cmap="RdBu", norm=CenteredNorm()
                )
                divider1 = make_axes_locatable(ax[0])
                cax1 = divider1.append_axes("right", size="5%", pad=0.05)
                im2 = ax[1].imshow(xp.T, cmap="gray_r")
                divider2 = make_axes_locatable(ax[1])
                cax2 = divider2.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im1, cax=cax1)
                plt.colorbar(im2, cax=cax2)
                fig.savefig(f"output/out{neval}.png")

        obj_vals.append(v)

        return v[0]

    def cst_binarization(x, gd):
        xp = fdtd_parametrization(x)
        v, g = value_and_grad(gray_indicator)(xp)
        if gd.size > 0:
            gd[:] = tensor_jacobian_product(fdtd_parametrization, 0)(x, g)
        return v - bin_tol

    def cst_connectivity(x, gd):
        xp = fem_parametrization(x)
        v, g = value_and_grad(fem)(xp)
        if gd.size > 0:
            gd[:] = tensor_jacobian_product(fem_parametrization, 0)(x, g)
        return v - con_tol

    opt = nlopt.opt(nlopt.LD_MMA, x0.size)
    opt.set_max_objective(nlopt_obj)
    opt.set_lower_bounds(0)
    opt.set_upper_bounds(1)
    opt.add_inequality_constraint(cst_binarization)
    opt.add_inequality_constraint(cst_connectivity)
    opt.set_ftol_rel(1e-4)

    x0 = opt.optimize(x0)

    bfac *= 2
    cst_bnd = gray_indicator(fdtd_parametrization(x0)) + 1e-3
    fdtd_parametrization, fem_parametrization = create_parametrizations(
        dofs, sigma, bfac
    )

if mp.am_master():
    hist = np.array(obj_vals)
    design = fdtd_parametrization(x0)
    with h5py.File("optimized_coupler.h5", "w") as f:
        f.create_dataset("hist", data=hist)
        f.create_dataset("xopt", data=x0)
        f.create_dataset("design", data=design)
