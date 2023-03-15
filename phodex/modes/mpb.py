from typing import Literal

import meep as mp
from meep import mpb

mp.verbosity.mpb = 0


def get_neff_mpb(
    lambda0: float,
    wvg_width: float,
    wvg_height: float,
    n_core: float,
    n_clad: float,
    resolution: int,
    polarization: Literal["te", "tm"] = "te",
    tol: float = 1e-9,
) -> float:
    sy = 10 * wvg_width
    sz = 10 * wvg_height

    parity = mp.ODD_Y + mp.EVEN_Z if polarization == "te" else mp.ODD_Z + mp.EVEN_Y

    geometry = [
        mp.Block(
            size=mp.Vector3(mp.inf, wvg_width, wvg_height),
            material=mp.Medium(index=n_core),
            center=mp.Vector3(),
        ),
    ]

    geometry_lattice = mp.Lattice(size=mp.Vector3(0, sy, sz))

    mode_solver = mpb.ModeSolver(
        geometry_lattice=geometry_lattice,
        geometry=geometry,
        resolution=resolution,
        default_material=mp.Medium(index=n_clad),
    )

    k = mode_solver.find_k(
        parity,
        1 / lambda0,
        1,
        1,
        mp.Vector3(1, 0, 0),
        tol,
        2 / lambda0,
        0.1 / lambda0,
        10 / lambda0,
    )

    return k[0] * lambda0
