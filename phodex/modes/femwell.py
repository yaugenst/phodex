from collections import OrderedDict
from typing import Literal

import numpy as np
import shapely
import shapely.affinity
from femwell.mesh import mesh_from_OrderedDict
from femwell.mode_solver import compute_modes
from scipy.constants import speed_of_light as c0
from scipy.optimize import approx_fprime
from shapely import box
from skfem import Basis, ElementDG, ElementTriP0, ElementTriP1
from skfem.io.meshio import from_meshio


def get_neff_femwell(
    lambda0: float,
    wvg_width: float,
    wvg_height: float,
    n_core: float,
    n_clad: float,
    resolution: int,
    polarization: Literal["te", "tm"] = "te",
    tol: float = 1e-9,
) -> float:
    core = shapely.geometry.box(-wvg_width / 2, 0, wvg_width / 2, wvg_height)
    env = shapely.affinity.scale(core.buffer(5, resolution=10), xfact=0.5)

    polygons = OrderedDict(core=core, clad=env)

    resolutions = dict(core={"resolution": 1 / (2 * resolution), "distance": 0.5})

    mesh = from_meshio(
        mesh_from_OrderedDict(polygons, resolutions, default_resolution_max=30)
    )

    basis0 = Basis(mesh, ElementTriP0(), intorder=4)
    epsilon = basis0.zeros()
    for subdomain, n in {"core": n_core, "clad": n_clad}.items():
        epsilon[basis0.get_dofs(elements=subdomain)] = n**2

    mode_idx = 0 if polarization == "te" else 1

    lams, _, _ = compute_modes(
        basis0, epsilon, wavelength=lambda0, num_modes=mode_idx + 1, order=2
    )

    return np.real(lams[mode_idx])


def bend_modes_box(
    lbda: float,
    n_core: float,
    n_clad: float,
    wvg_width: float,
    wvg_height: float,
    radius: float = np.inf,
    pml_thickness: float = 2.0,
    n_guess: float | complex | None = None,
) -> dict:
    pml_distance = wvg_width / 2 + 2  # distance from center
    pml_thickness = 2
    core = box(-wvg_width / 2, 0, wvg_width / 2, wvg_height)
    env = box(-1 - wvg_width / 2, -1, pml_distance + pml_thickness, wvg_height + 1)

    polygons = dict(core=core, clad=env)
    resolutions = dict(core={"resolution": 0.03, "distance": 1})
    mesh = from_meshio(
        mesh_from_OrderedDict(
            polygons, resolutions, default_resolution_max=0.2, filename="mesh.msh"
        )
    )

    basis0 = Basis(mesh, ElementDG(ElementTriP1()))
    epsilon = basis0.zeros(dtype=complex)
    for subdomain, n in {"core": n_core(lbda), "clad": n_clad(lbda)}.items():
        epsilon[basis0.get_dofs(elements=subdomain)] = n**2
    epsilon += basis0.project(
        lambda x: -10j * np.maximum(0, x[0] - pml_distance) ** 2,
        dtype=complex,
    )

    lams, basis, xs = compute_modes(
        basis0,
        epsilon,
        wavelength=lbda,
        num_modes=1,
        order=2,
        radius=radius,
        n_guess=n_guess,
    )
    return {"lams": lams, "basis": basis, "xs": xs}


def effective_index(
    lbda: float,
    n_core: float,
    n_clad: float,
    wvg_width: float,
    wvg_height: float,
    radius: float = np.inf,
    pml_thickness: float = 2.0,
) -> float:
    mode = bend_modes_box(
        lbda, n_core, n_clad, wvg_width, wvg_height, radius, pml_thickness
    )
    return np.real(mode["lams"])


def dneff_dlambda(
    lbda: float,
    n_core: float,
    n_clad: float,
    wvg_width: float,
    wvg_height: float,
    radius: float = np.inf,
    pml_thickness: float = 2.0,
) -> float:
    return approx_fprime(
        lbda,
        lambda x, *args: np.real(bend_modes_box(x, *args)["lams"]),
        1.4901161193847656e-08,
        n_core,
        n_clad,
        wvg_width,
        wvg_height,
        radius,
        pml_thickness,
    )


def group_velocity(
    lbda: float,
    n_core: float,
    n_clad: float,
    wvg_width: float,
    wvg_height: float,
    radius: float = np.inf,
    pml_thickness: float = 2.0,
) -> float:
    neff = np.real(
        bend_modes_box(
            lbda, n_core, n_clad, wvg_width, wvg_height, radius, pml_thickness
        )["lams"]
    )
    dndl = dneff_dlambda(
        lbda,
        n_core,
        n_clad,
        wvg_width,
        wvg_height,
        radius,
        pml_thickness,
    )
    return c0 / (neff - lbda * dndl)


def group_index(
    lbda: float,
    n_core: float,
    n_clad: float,
    wvg_width: float,
    wvg_height: float,
    radius: float = np.inf,
    pml_thickness: float = 2.0,
) -> float:
    vg = group_velocity(
        lbda, n_core, n_clad, wvg_width, wvg_height, radius, pml_thickness
    )
    return c0 / vg


def free_spectral_range(
    lbda: float,
    n_core: float,
    n_clad: float,
    wvg_width: float,
    wvg_height: float,
    radius: float = np.inf,
    pml_thickness: float = 2.0,
) -> float:
    prefac = lbda**2 / (2 * np.pi * radius)
    neff = effective_index(
        lbda, n_core, n_clad, wvg_width, wvg_height, radius, pml_thickness
    )
    dndl = approx_fprime(
        lbda,
        effective_index,
        1.4901161193847656e-08,
        n_core,
        n_clad,
        wvg_width,
        wvg_height,
        radius,
        pml_thickness,
    )
    return prefac / (neff - lbda * dndl)
