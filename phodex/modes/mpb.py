from typing import Iterable, Literal

import matplotlib.pyplot as plt
import meep as mp
import numpy as np
from matplotlib.colors import CenteredNorm
from matplotlib.figure import Figure
from meep import mpb
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import UnivariateSpline
from scipy.optimize import root_scalar

mp.verbosity.mpb = 0


def find_k(lcen: float, k_vals: Iterable[float], freqs: Iterable[float]) -> float:
    """Finds intersection of modal dispersion with some target wavelength."""
    dispersion = UnivariateSpline(k_vals, freqs, s=0)
    crossing = root_scalar(
        lambda x: dispersion(x) - 1 / lcen,
        method="brentq",
        bracket=[k_vals[0], k_vals[-1]],
    )
    return crossing.root


def neff_from_k(lcen: float, k_vals: Iterable[float], freqs: Iterable[float]) -> float:
    k = find_k(lcen, k_vals, freqs)
    return k * lcen


def get_bands(
    k_points: Iterable[mp.Vector3],
    resolution: int,
    num_bands: int,
    geometry: Iterable[mp.GeometricObject],
    lattice: mp.Lattice,
    default_material: mp.Medium | None = None,
    polarization: Literal["te", "tm"] | None = None,
) -> dict:
    ms = mpb.ModeSolver(
        geometry_lattice=lattice,
        geometry=geometry,
        resolution=resolution,
        k_points=k_points,
        num_bands=num_bands,
        default_material=default_material,
    )

    match polarization:
        case "te":
            ms.run_te()
        case "tm":
            ms.run_tm()
        case _:
            ms.run()

    return gather_ms_results(ms, 1)


def gather_ms_results(
    ms: mpb.ModeSolver, band: int = 1, geometry: list[mp.GeometricObject] | None = None
) -> dict:
    results = {"freqs": ms.all_freqs}

    ms.get_efield(band)
    ms.fix_field_phase()
    results["efield"] = ms.get_efield(band)
    ms.get_hfield(band)
    ms.fix_field_phase()
    results["hfield"] = ms.get_hfield(band)

    results["poynting"] = ms.get_poynting(band)

    if geometry is not None:
        results["confinement"] = ms.compute_energy_in_objects(geometry)

    return results


def plot_bands(
    bands: dict[str, Iterable],
    lcen: float,
    k_vecs: Iterable[float],
    n_clad: float,
    ax: plt.Axes | None = None,
) -> None:
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4), tight_layout=True)

    fcen = 1 / lcen
    max_freq = np.max([b for b in bands.values()])

    colors = ["tab:blue", "tab:red"]
    for (k, v), c in zip(bands.items(), colors):
        ax.plot(k_vecs, v[:, 0], c, label=k)
        ax.plot(k_vecs, v[:, 1:], c)
    ax.plot(k_vecs, k_vecs / n_clad, "k-", lw=1)
    ax.plot([k_vecs[0], k_vecs[-1]], [fcen, fcen], c="tab:green", ls="--", zorder=3)
    ax.fill_between(
        k_vecs, np.max(k_vecs / n_clad), k_vecs / n_clad, fc="0.8", zorder=2
    )
    ax.axis([k_vecs[0], k_vecs[-1], 0, max_freq])
    ax.text(0.4, fcen + max_freq / 50, rf"$\lambda = {lcen} \mu m$", size=12)
    ax.set_xlabel(r"$k_x$ ($\frac{2\pi}{\mu m}$)", size=13)
    ax.set_ylabel(r"frequency (300 THz)", size=13)
    ax.legend()


def plot_mode(
    eps: np.ndarray,
    fields: Iterable[np.ndarray],
    labels: Iterable[Iterable[str]],
    sy: float,
    sz: float,
    fig: Figure | None,
    ax: plt.Axes | None,
) -> None:
    if ax is None:
        fig, ax = plt.subplots(
            2, 3, figsize=(8, 4), sharex=True, sharey=True, tight_layout=True
        )

    for idx in range(2):
        for n, (axi, label) in enumerate(zip(ax[idx], labels[idx])):
            im = axi.imshow(
                np.real(fields[idx][..., n]).T,
                cmap="RdBu_r",
                norm=CenteredNorm(),
            )
            axi.contour(eps.T, cmap="binary", alpha=0.5, levels=0)
            divider = make_axes_locatable(axi)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im, cax=cax, orientation="vertical")
            axi.set_title(label)

    plt.setp(
        ax[:, 0],
        yticks=np.linspace(0, eps.shape[1], 3),
        yticklabels=np.linspace(-sz / 2, sz / 2, 3),
        ylabel=r"z ($\mu m$)",
    )
    plt.setp(
        ax[-1, :],
        xticks=np.linspace(0, eps.shape[0], 3),
        xticklabels=np.linspace(-sy / 2, sy / 2, 3),
        xlabel=r"y ($\mu m$)",
    )


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

    parity = mp.TE if polarization == "te" else mp.TM

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
