from dataclasses import dataclass, field
from functools import cached_property, partial
from inspect import signature
from typing import Callable, Iterable, Literal

import meep as mp
import meep.adjoint as mpa
import numpy as np
from loguru import logger


@dataclass(frozen=True)
class Port:
    width: float
    axis: Literal["-x", "+x", "-y", "+y"]
    offset: float = 0.0
    source: bool = False

    @property
    def kpoint(self) -> np.ndarray:
        if not self.source:
            raise ValueError("Not a source port!")
        match self.axis:
            case "-x":
                return np.array([1.0, 0.0])
            case "+x":
                return np.array([-1.0, 0.0])
            case "-y":
                return np.array([0.0, 1.0])
            case "+y":
                return np.array([0.0, -1.0])

    @property
    def kpoint_idx(self):
        return np.argmax(np.abs(self.kpoint))

    @property
    def kpoint_val(self):
        return self.kpoint[self.kpoint_idx]

    def get_center_size(
        self,
        cell: mp.Vector3,
        dpml: float,
        size_fac: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        match self.axis:
            case "-x":
                size = (0, size_fac * self.width)
                center = (-cell.x / 2 + dpml, self.offset)
            case "+x":
                size = (0, size_fac * self.width)
                center = (cell.x / 2 - dpml, self.offset)
            case "-y":
                size = (size_fac * self.width, 0)
                center = (self.offset, -cell.y / 2 + dpml)
            case "+y":
                size = (size_fac * self.width, 0)
                center = (self.offset, cell.y / 2 - dpml)

        return np.array(center), np.array(size)


@dataclass(frozen=True)
class MultiportDevice2D:
    ports: Iterable[Port]

    n_core: float = 3.4865
    n_clad: float = 1.4440
    wvg_height: float | None = None
    monitor_size_fac: float = 2.0
    monitor_offset: float = 0.2

    resolution: int = 30
    design_resolution: int = 2 * resolution
    wavelengths: Iterable[float] = (1.55,)
    mode: int = 1
    dpml: float = 1.0
    df: float = 0.2
    stop_tol: float = 1e-6
    polarization: Literal["te", "tm"] = "tm"
    mirror_axis: Literal["x", "y"] | None = None
    sim_kwargs: dict = field(default_factory=dict)

    design_region_extent: tuple[float, float] = (3.0, 3.0)
    buffer_xy: tuple[float, float] = (1.0, 1.0)

    def __post_init__(self) -> None:
        object.__setattr__(self, "wavelengths", np.array(self.wavelengths, dtype="f8"))

        if self.wvg_height is not None:
            self._check_neff()
            object.__setattr__(self, "n_core_ref", self.n_core)
            object.__setattr__(self, "n_core", self.neff)
            logger.info(
                f"Using effective index, set {self.n_core=} and {self.n_core_ref=}."
            )

        self._check_resolution()
        self._check_dpml()
        self._check_ports()
        self._check_sources()

    def _check_neff(self) -> None:
        if len(set(p.width for p in self.source_ports)) > 1:
            raise ValueError(
                "Source regions have different widths and thus excite different modes. "
                "Cannot do effective index calculations with different modes!"
            )

    def _check_resolution(self) -> None:
        if (min_resolution := int(np.ceil(15 / self.min_wl))) > self.resolution:
            logger.warning(
                f"Resolution {self.resolution} is below the minimum recommended "
                f"resolution of {min_resolution} - results will likely be inaccurate!",
            )

    def _check_dpml(self) -> None:
        if self.dpml < self.max_wl / 2:
            logger.warning(
                f"PML thickness {self.dpml} is below the minimum recommended "
                f"thickness of {self.max_wl / 2} - it is generally recommended that "
                "PMLs have a thickness of at least half of the largest wavelength in "
                " the simulation.",
            )

    def _check_ports(self) -> None:
        if len(self.ports) == 0:
            raise ValueError("Device contains no ports!")
        if len(self.ports) != len(set(self.ports)):
            logger.warning("Portlist contains duplicates!")

    def _check_sources(self) -> None:
        n = len(self.source_ports)
        if n == 0:
            raise ValueError("No source port specified!")
        elif n > 1:
            logger.warning(
                f"Specified {n} source ports that will be simulated simultaneously. "
                "Please make sure that's intended!"
            )

    @property
    def parity(self) -> int:
        pol_to_par = {"te": mp.ODD_Y + mp.EVEN_Z, "tm": mp.EVEN_Y + mp.ODD_Z}
        return pol_to_par[self.polarization]

    @property
    def symmetries(self) -> list[mp.Symmetry]:
        d = mp.X if self.mirror_axis == "x" else mp.Y
        p = 1 if self.polarization == "tm" else -1
        return [mp.Mirror(direction=d, phase=p)]

    @property
    def min_wl(self) -> float:
        return min(self.wavelengths) / max([self.n_core, self.n_clad])

    @property
    def max_wl(self) -> float:
        return max(self.wavelengths) / min([self.n_core, self.n_clad])

    @property
    def core(self) -> mp.Medium:
        return mp.Medium(index=self.n_core)

    @property
    def cladding(self) -> mp.Medium:
        return mp.Medium(index=self.n_clad)

    @property
    def source_ports(self) -> list[Port]:
        return [p for p in self.ports if p.source]

    @cached_property
    def neff(self) -> float:
        from gdsfactory.simulation.modes import find_modes_waveguide

        modes = find_modes_waveguide(
            wg_width=self.source_ports[0].width,
            ncore=self.n_core,
            nclad=self.n_clad,
            wg_thickness=self.wvg_height,
            sz=10 * self.wvg_height,
            sy=self.monitor_size_fac * self.source_ports[0].width,
            nmodes=self.mode,
            resolution=self.resolution,
            parity=self.parity,
        )
        return float(modes[self.mode].neff)

    @property
    def cell(self) -> mp.Vector3:
        return mp.Vector3(
            2 * self.dpml + 2 * self.buffer_xy[0] + self.design_region_extent[0],
            2 * self.dpml + 2 * self.buffer_xy[1] + self.design_region_extent[1],
        )

    @property
    def pml(self) -> list[mp.PML]:
        return [mp.PML(self.dpml)]

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
        return int(self.design_resolution * self.design_region_extent[0])

    @property
    def ny(self) -> int:
        return int(self.design_resolution * self.design_region_extent[1])

    @cached_property
    def design_region(self) -> mpa.DesignRegion:
        return mpa.DesignRegion(
            mp.MaterialGrid(mp.Vector3(self.nx, self.ny), self.cladding, self.core),
            volume=mp.Volume(
                center=mp.Vector3(),
                size=mp.Vector3(*self.design_region_extent),
            ),
        )

    @property
    def geometry(self) -> list[mp.Block]:
        self.design_region.update_design_parameters(np.full(self.nx * self.ny, 0.5))
        geom = [
            mp.Block(
                center=self.design_region.center,
                size=self.design_region.size,
                material=self.design_region.design_parameters,
            ),
        ]
        for port in self.ports:
            c, s = port.get_center_size(self.cell, self.dpml)
            idx = 0 if "x" in port.axis else 1
            fac = -1 if "+" in port.axis else 1
            s[idx] = self.dpml + self.buffer_xy[idx]
            c[idx] += fac * (self.buffer_xy[idx] - self.dpml) / 2
            geom.append(mp.Block(center=c, size=s, material=self.core))
        return geom

    @property
    def sources(self) -> list[mp.EigenModeSource]:
        sources = []
        for port in self.source_ports:
            c, s = port.get_center_size(self.cell, self.dpml, self.monitor_size_fac)
            sources.append(
                mp.EigenModeSource(
                    mp.GaussianSource(frequency=self.fcen, fwidth=self.df),
                    eig_band=self.mode,
                    direction=mp.NO_DIRECTION,
                    eig_kpoint=port.kpoint,
                    size=s,
                    center=c,
                    eig_parity=self.parity,
                )
            )
        return sources

    @property
    def simulation(self) -> mp.Simulation:
        return mp.Simulation(
            resolution=self.resolution,
            default_material=self.cladding,
            cell_size=self.cell,
            sources=self.sources,
            geometry=self.geometry,
            boundary_layers=self.pml,
            symmetries=self.symmetries,
            k_point=mp.Vector3(),
            **self.sim_kwargs,
        )

    @cached_property
    def normalizations(self) -> list[tuple[np.ndarray, mp.simulation.FluxData]]:
        norms = []
        for source, port in zip(self.sources, self.source_ports):
            wvg_center, wvg_size = port.get_center_size(self.cell, self.dpml)
            wvg_size[port.kpoint_idx] = mp.inf
            sim = self.simulation
            sim.geometry = [
                mp.Block(center=wvg_center, size=wvg_size, material=self.core)
            ]
            sim.change_sources([source])

            mon_center, mon_size = port.get_center_size(
                self.cell, self.dpml, size_fac=self.monitor_size_fac
            )

            mon_center_near = np.copy(mon_center)
            mon_center_near[port.kpoint_idx] += port.kpoint_val * self.monitor_offset
            mon_near = sim.add_mode_monitor(
                self.frequencies,
                mp.ModeRegion(center=mon_center_near, size=mon_size),
                yee_grid=True,
            )

            mon_center_far = np.copy(mon_center)
            mon_center_far[port.kpoint_idx] *= -1
            mon_far = sim.add_mode_monitor(
                self.frequencies,
                mp.ModeRegion(center=mon_center_far, size=mon_size),
                yee_grid=True,
            )

            sim.run(until_after_sources=self.stop_cond)

            mode_data_near = sim.get_eigenmode_coefficients(
                mon_near, [self.mode], eig_parity=self.parity
            )

            mode_data_far = sim.get_eigenmode_coefficients(
                mon_far, [self.mode], eig_parity=self.parity
            )

            c_near = mode_data_near
            c_far = mode_data_far
            idx = int(port.kpoint_val < 0)

            flux_near = np.abs(c_near.alpha[self.mode - 1, :, idx]) ** 2
            flux_far = np.abs(c_far.alpha[self.mode - 1, :, idx]) ** 2

            if not np.allclose(flux_near, flux_far, rtol=1e-3):
                logger.warning(
                    f"Excited mode is lossy! Try increasing {self.monitor_size_fac=}."
                )

            flux_near_data = sim.get_flux_data(mon_near)
            norms.append((flux_near, flux_near_data))
        return norms

    @property
    def objective_monitors(self) -> list[Callable]:
        monitors = []
        for (_, flux_data), port in zip(self.normalizations, self.source_ports):
            c, s = port.get_center_size(self.cell, self.dpml, self.monitor_size_fac)
            c[port.kpoint_idx] += port.kpoint_val * self.monitor_offset
            monitors.append(
                partial(
                    mpa.EigenmodeCoefficient,
                    volume=mp.Volume(center=c, size=s),
                    mode=self.mode,
                    eig_parity=self.parity,
                    forward="+" in port.axis,
                    subtracted_dft_fields=flux_data,
                )
            )
        for port in [p for p in self.ports if p not in self.source_ports]:
            c, s = port.get_center_size(self.cell, self.dpml, self.monitor_size_fac)
            monitors.append(
                partial(
                    mpa.EigenmodeCoefficient,
                    volume=mp.Volume(center=c, size=s),
                    mode=self.mode,
                    eig_parity=self.parity,
                    forward="+" in port.axis,
                )
            )
        return monitors

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
