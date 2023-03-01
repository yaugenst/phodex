import warnings
from dataclasses import dataclass
from functools import cached_property, partial
from inspect import signature
from typing import Callable, Iterable

import meep as mp
import meep.adjoint as mpa
import numpy as np


@dataclass(frozen=True)
class FilterProblem:
    resolution: int = 20
    design_resolution: int = 2 * resolution
    wavelengths: Iterable[float] = (1.54, 1.55)
    use_effective_index: bool = True
    dpml: float = 1.0
    df: float = 0.2
    stop_tol: float = 1e-6
    parity: int = mp.EVEN_Y + mp.ODD_Z
    symmetries: Iterable[mp.Symmetry] = (mp.Mirror(direction=mp.Y),)

    n_core: float = 1.9963
    n_clad: float = 1.444
    wvg_width: float = 1.0
    wvg_height: float = 0.3

    design_region_xy: tuple[float, float] = (4.0, 2.0)
    buffer_xy: tuple[float, float] = (1.0, 1.0)

    def __post_init__(self) -> None:
        object.__setattr__(self, "wavelengths", np.array(self.wavelengths, dtype="f8"))

        if self.use_effective_index:
            object.__setattr__(self, "n_core_ref", self.n_core)
            object.__setattr__(self, "n_core", self.neff)

        min_wl = min(self.wavelengths) / max([self.n_core, self.n_clad])
        if (min_resolution := int(np.ceil(15 / min_wl))) > self.resolution:
            warnings.warn(
                f"Resolution {self.resolution} is below the minimum recommended "
                f"resolution of {min_resolution} - results will likely be inaccurate!",
                RuntimeWarning,
            )

    @property
    def core(self) -> mp.Medium:
        return mp.Medium(index=self.n_core)

    @property
    def cladding(self) -> mp.Medium:
        return mp.Medium(index=self.n_clad)

    @cached_property
    def neff(self) -> float:
        from gdsfactory.simulation.modes import find_modes_waveguide

        modes = find_modes_waveguide(
            wg_width=self.wvg_width,
            ncore=self.n_core,
            nclad=self.n_clad,
            wg_thickness=self.wvg_height,
            sz=10 * self.wvg_height,
            sy=10 * self.wvg_width,
            nmodes=1,
            resolution=self.resolution,
        )
        return float(modes[1].neff)

    @property
    def cell(self) -> mp.Vector3:
        return mp.Vector3(
            2 * self.dpml + 2 * self.buffer_xy[0] + self.design_region_xy[0],
            2 * self.dpml + 2 * self.buffer_xy[1] + self.design_region_xy[1],
        )

    @property
    def pml(self) -> mp.PML:
        return mp.PML(self.dpml)

    @property
    def stop_cond(self) -> Callable:
        return mp.stop_when_dft_decayed(tol=self.stop_tol)

    @property
    def frequencies(self) -> np.ndarray:
        return 1 / self.wavelengths

    @property
    def nfreq(self) -> int:
        return self.frequencies.size

    @property
    def fcen(self) -> float:
        return np.mean(self.frequencies)

    @property
    def lcen(self) -> float:
        return np.mean(self.wavelengths)

    @property
    def nx(self) -> int:
        return int(self.design_resolution * self.design_region_xy[0])

    @property
    def ny(self) -> int:
        return int(self.design_resolution * self.design_region_xy[1])

    @cached_property
    def design_region(self) -> mpa.DesignRegion:
        return mpa.DesignRegion(
            mp.MaterialGrid(mp.Vector3(self.nx, self.ny), self.cladding, self.core),
            volume=mp.Volume(
                center=mp.Vector3(),
                size=mp.Vector3(*self.design_region_xy),
            ),
        )

    @property
    def geometry(self) -> list[mp.Block]:
        geom = [
            mp.Block(
                center=self.design_region.center,
                size=self.design_region.size,
                material=self.design_region.design_parameters,
            ),
        ]
        geom.extend(
            [
                mp.Block(
                    center=mp.Vector3(
                        x_dir * (self.cell.x - self.dpml - self.buffer_xy[0]) / 2, 0
                    ),
                    size=mp.Vector3(self.dpml + self.buffer_xy[1], self.wvg_width),
                    material=self.core,
                )
                for x_dir in [1, -1]
            ]
        )
        return geom

    @property
    def source(self) -> mp.EigenModeSource:
        return mp.EigenModeSource(
            mp.GaussianSource(frequency=self.fcen, fwidth=self.df),
            eig_band=1,
            direction=mp.NO_DIRECTION,
            eig_kpoint=mp.Vector3(1, 0),
            size=mp.Vector3(0, self.cell.y),
            center=mp.Vector3(-self.cell.x / 2 + self.dpml, 0),
            eig_parity=self.parity,
        )

    @property
    def simulation(self) -> mp.Simulation:
        return mp.Simulation(
            resolution=self.resolution,
            default_material=self.cladding,
            cell_size=self.cell,
            sources=[self.source],
            geometry=self.geometry,
            boundary_layers=[self.pml],
            symmetries=self.symmetries,
            k_point=mp.Vector3(),
        )

    @cached_property
    def normalization(self) -> tuple[np.ndarray, mp.simulation.FluxData]:
        sim = self.simulation
        sim.geometry = [
            mp.Block(
                center=mp.Vector3(0, 0),
                size=mp.Vector3(self.cell.x, self.wvg_width),
                material=self.core,
            )
        ]

        mon = sim.add_mode_monitor(
            self.frequencies,
            mp.ModeRegion(
                center=mp.Vector3(self.cell.x / 2 - self.dpml),
                size=mp.Vector3(0, self.cell.y),
            ),
            yee_grid=True,
        )

        sim.run(until_after_sources=self.stop_cond)

        mode_data = sim.get_eigenmode_coefficients(mon, [1], eig_parity=self.parity)

        coeffs = mode_data.alpha
        input_flux = np.abs(coeffs[0, :, 0]) ** 2
        input_flux_data = sim.get_flux_data(mon)

        sim.reset_meep()

        return input_flux, input_flux_data

    @property
    def objective_monitors(self) -> list[Callable]:
        s11_monitor = partial(
            mpa.EigenmodeCoefficient,
            volume=mp.Volume(
                center=mp.Vector3(-self.cell.x / 2 + self.dpml + 0.5, 0),
                size=mp.Vector3(0, self.cell.y),
            ),
            mode=1,
            eig_parity=self.parity,
            forward=False,
            subtracted_dft_fields=self.normalization[1],
        )
        s12_monitor = partial(
            mpa.EigenmodeCoefficient,
            volume=mp.Volume(
                size=mp.Vector3(0, self.cell.y),
                center=mp.Vector3(self.cell.x / 2 - self.dpml, 0),
            ),
            mode=1,
            eig_parity=self.parity,
            forward=True,
        )
        return [s11_monitor, s12_monitor]

    def _check_objective_args(self, fn: Callable) -> None:
        if len(p := signature(fn).parameters) == len(self.objective_monitors):
            return
        raise ValueError(
            f"Invalid number of arguments for objective function '{fn.__name__}' - "
            f"need {len(self.objective_monitors)}, got {len(p)}: {list(p.keys())}."
        )

    def get_optimization_problem(
        self, objective_functions: Iterable[Callable]
    ) -> mpa.OptimizationProblem:
        if not objective_functions:
            raise RuntimeError("Need at least one objective function!")

        for f in objective_functions:
            self._check_objective_args(f)

        sim = self.simulation
        return mpa.OptimizationProblem(
            simulation=sim,
            objective_functions=objective_functions,
            objective_arguments=[m(sim) for m in self.objective_monitors],
            design_regions=[self.design_region],
            frequencies=self.frequencies,
        )
