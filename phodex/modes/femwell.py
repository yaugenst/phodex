from collections import OrderedDict
from typing import Literal

import numpy as np
import shapely
import shapely.affinity
from femwell.mesh import mesh_from_OrderedDict
from femwell.mode_solver import compute_modes
from skfem import Basis, ElementTriP0
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
