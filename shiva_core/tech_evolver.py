"""
SHIVA-1 :: Technology Evolution Engine
======================================

Evolves engineering designs *under the physics of a mutated universe* and
scores them for transfer back to ours.

Physics bridge
--------------
:class:`PhysicsContext` holds the engineering-scale constants a designer
needs (surface gravity, gas/liquid properties, adiabatic indices, material
property scalings, speed of light).  ``PhysicsContext.from_genome`` derives
them from a :class:`~shiva_core.physics_mutator.PhysicsGenome` by explicit,
documented dimensional arguments:

* gravity ratio = |F_genome(r=1)| / |F_Newton(r=1)| from the SymPy force law;
* materials (Press & Lightman 1983): with electron and nucleon masses fixed,
  the Bohr radius a0 ~ 1/alpha and bond energies ~ alpha^2, so moduli and
  strengths (energy / volume) scale as alpha^5 and densities as alpha^3;
* gas viscosity ~ sqrt(m k T) / sigma_coll with sigma_coll ~ a0^2, i.e. ~ alpha^2;
  liquid viscosity follows the genome's fluid viscosity ratio;
* drag-scaling exponent and adiabatic index come straight from the genome.

These are order-of-magnitude scaling laws, not predictions of a full
alternative chemistry - every downstream number inherits that caveat.

Design domains (all SI, first-principles reduced-order models)
--------------------------------------------------------------
========================  ====================================================
``propulsion_nozzle``     de Laval nozzle: isentropic area-Mach relation,
                          thrust coefficient, c*, divergence loss, Summerfield
                          separation criterion, hoop-stress wall sizing
``structure_truss``       2-D cantilever truss: direct-stiffness FEM, yield,
                          Euler buckling, deflection limit, self-weight
``energy_flywheel``       rotating annular disk (Timoshenko) stresses, stored
                          energy, von Karman windage, bearing losses
``drone_frame``           multirotor: momentum-theory hover power, arm bending,
                          Euler-Bernoulli natural frequency vs rotor speed,
                          tip-Mach and propeller-clearance limits
``pump_impeller``         centrifugal pump: Euler head with Wiesner slip,
                          incidence/friction/volute losses, disk friction,
                          NPSH (cavitation) and de Haller diffusion limits
``gear_pair``             spur gears: involute contact ratio, undercut, tip
                          thickness, Lewis bending + Barth velocity factor,
                          Hertz contact, Ohlendorf mesh-loss efficiency
``cosmic_habitat``        rotating space habitat: artificial gravity, comfort
                          (rpm, Coriolis, gravity gradient), hoop stress with
                          pressure + contents, major-axis spin stability
``cosmic_lightsail``      solar sail: radiation pressure ~ 1/c, characteristic
                          acceleration, lightness number, thermal equilibrium
========================  ====================================================

Every candidate is scored on five objectives in [0, 1] -
efficiency, stability, adaptability, real-world mappability and cosmic-scale
performance - and evolved with NSGA-II (``ai_brain``).  The Pareto set feeds
a PPCA generative model, and a REINFORCE agent learns to *adapt* designs to
new physics contexts.  :class:`GenerativeBracketDesigner` evolves free-form
part shapes with CPPN-NEAT against a lattice-spring finite-element model.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type

import numpy as np
from scipy.optimize import brentq
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import spsolve

from .ai_brain import (NSGA2, CPPNShapeGenerator, DesignEnvironment, DesignRLAgent, LatentDesignModel,
                       NEATPopulation, fast_non_dominated_sort)
from .real_world_mapper import MATERIALS, Material

__all__ = [
    "PhysicsContext",
    "PLANETS",
    "DesignEval",
    "DesignDomain",
    "DOMAINS",
    "EvolvedDesign",
    "EvolutionResult",
    "TechEvolver",
    "evolve_technologies",
    "GenerativeBracketDesigner",
]

G0 = 9.80665
SIGMA_SB = 5.670374419e-8
R_UNIVERSAL = 8.314462618


# ---------------------------------------------------------------------------
# Physics context
# ---------------------------------------------------------------------------

@dataclass
class PhysicsContext:
    name: str = "earth"
    g: float = G0
    gravity_exponent: float = 2.0
    air_density: float = 1.225
    air_viscosity: float = 1.81e-5
    ambient_pressure: float = 101325.0
    speed_of_sound: float = 343.0
    temperature: float = 288.0
    liquid_density: float = 998.0
    liquid_viscosity: float = 1.0e-3
    vapor_pressure: float = 2339.0
    drag_exponent: float = 2.0
    adiabatic_index: float = 1.4
    propellant_gamma: float = 1.22
    modulus_scale: float = 1.0
    strength_scale: float = 1.0
    density_scale: float = 1.0
    speed_of_light: float = 299792458.0
    alpha_ratio: float = 1.0
    habitable_chemistry: bool = True
    description: str = "Earth, sea level, 15 C"

    @classmethod
    def earth(cls) -> "PhysicsContext":
        return cls()

    @classmethod
    def from_genome(cls, genome: Any) -> "PhysicsContext":
        from .physics_mutator import ALPHA_CHEMISTRY_WINDOW, SymbolicPhysics

        sym = SymbolicPhysics(genome)
        F1 = float(abs(sym.force_profile(np.array([1.0]))[0]))
        g_ratio = float(np.clip(F1 / 1.0, 0.05, 20.0))
        a = float(genome.em.alpha_ratio)
        gam = float(genome.thermo.adiabatic_index)
        c = 299792458.0
        if genome.inertia.model == "relativistic":
            c = 299792458.0 * float(genome.inertia.c) / 10.0
        lo, hi = ALPHA_CHEMISTRY_WINDOW
        return cls(
            name=f"alien-{genome.genome_id}",
            g=G0 * g_ratio,
            gravity_exponent=float(genome.gravity.k_eff),
            air_viscosity=1.81e-5 * a ** 2,
            speed_of_sound=343.0 * math.sqrt(gam / 1.4),
            liquid_density=998.0 * a ** 3,
            liquid_viscosity=1.0e-3 * float(genome.fluid.viscosity_ratio),
            drag_exponent=float(genome.fluid.drag_exponent),
            adiabatic_index=gam,
            propellant_gamma=float(np.clip(1.0 + (gam - 1.0) * 0.55, 1.05, 1.6)),
            modulus_scale=a ** 5,
            strength_scale=a ** 5,
            density_scale=a ** 3,
            speed_of_light=c,
            alpha_ratio=a,
            habitable_chemistry=bool(lo <= a <= hi),
            description=(f"genome {genome.genome_id}: g={g_ratio:.3g} g0, alpha/alpha0={a:.3g},"
                         f" drag n={genome.fluid.drag_exponent:.3g}, gamma={gam:.3g}"),
        )

    def features(self) -> np.ndarray:
        """Normalised context features for learning agents."""
        return np.array([
            math.log(self.g / G0),
            math.log(max(self.air_density, 1e-6) / 1.225) / 5.0,
            math.log(self.liquid_viscosity / 1e-3),
            self.drag_exponent - 2.0,
            (self.adiabatic_index - 1.4) * 5.0,
            math.log(self.strength_scale),
        ])

    def perturbed(self, rng: np.random.Generator, scale: float = 0.1) -> "PhysicsContext":
        f = lambda: float(math.exp(rng.normal(0.0, scale)))
        return replace(self, name=self.name + "~", g=self.g * f(), air_density=self.air_density * f(),
                       liquid_viscosity=self.liquid_viscosity * f(), ambient_pressure=self.ambient_pressure * f(),
                       modulus_scale=self.modulus_scale * f(), strength_scale=self.strength_scale * f(),
                       density_scale=self.density_scale * float(math.exp(rng.normal(0.0, scale / 3))))

    def with_materials_of(self, other: "PhysicsContext") -> "PhysicsContext":
        return replace(self, modulus_scale=other.modulus_scale, strength_scale=other.strength_scale,
                       density_scale=other.density_scale, speed_of_light=other.speed_of_light,
                       alpha_ratio=other.alpha_ratio, name=f"{self.name}[{other.name}]")


#: Real planetary environments (approximate surface/cloud-layer values).
PLANETS: Dict[str, PhysicsContext] = {
    "earth": PhysicsContext(),
    "mars": PhysicsContext(name="mars", g=3.721, air_density=0.020, air_viscosity=1.08e-5, ambient_pressure=610.0,
                           speed_of_sound=226.0, temperature=210.0, adiabatic_index=1.29,
                           description="Mars surface (CO2, 6 mbar)"),
    "titan": PhysicsContext(name="titan", g=1.352, air_density=5.3, air_viscosity=6.3e-6, ambient_pressure=146700.0,
                            speed_of_sound=198.0, temperature=94.0, liquid_density=450.0, liquid_viscosity=1.8e-4,
                            vapor_pressure=14000.0, description="Titan surface (N2, liquid methane)"),
    "venus_clouds": PhysicsContext(name="venus_clouds", g=8.87, air_density=0.88, air_viscosity=1.5e-5,
                                   ambient_pressure=50000.0, speed_of_sound=270.0, temperature=300.0,
                                   adiabatic_index=1.29, description="Venus ~55 km cloud layer"),
    "moon": PhysicsContext(name="moon", g=1.62, air_density=0.0, air_viscosity=1.81e-5, ambient_pressure=0.0,
                           speed_of_sound=0.0, temperature=250.0, description="Lunar surface (vacuum)"),
}


# ---------------------------------------------------------------------------
# Domain framework
# ---------------------------------------------------------------------------

@dataclass
class ParamSpec:
    name: str
    lo: float
    hi: float
    kind: str = "lin"  # lin | log | int
    unit: str = ""


@dataclass
class DesignEval:
    metrics: Dict[str, float]
    efficiency: float
    stability: float
    constraints: Dict[str, float]

    @property
    def feasible(self) -> bool:
        return all(np.isfinite(v) and v <= 1e-9 for v in self.constraints.values())

    @property
    def violation(self) -> float:
        return float(sum(max(0.0, v) if np.isfinite(v) else 10.0 for v in self.constraints.values()))

    @property
    def soft_feasibility(self) -> float:
        return float(math.exp(-5.0 * self.violation))


def margin(x: float, req: float, full: float = 2.0) -> float:
    """0 at x = req, 1 at x = full * req (linear in between)."""
    if not np.isfinite(x) or req <= 0:
        return 0.0
    return float(np.clip((x / req - 1.0) / (full - 1.0), 0.0, 1.0))


def _combine_margins(ms: Sequence[float]) -> float:
    ms = [float(m) for m in ms]
    return 0.5 * min(ms) + 0.5 * float(np.mean(ms))


def _infeasible(metrics: Optional[Dict[str, float]] = None, reason: str = "invalid") -> DesignEval:
    return DesignEval(metrics or {}, 0.0, 0.0, {reason: 10.0})


class DesignDomain:
    name = "abstract"
    title = ""
    params: List[ParamSpec] = []
    reference_material = "al6061_t6"
    proof_factor = 1.5
    #: parameters snapped to discrete catalogues (held fixed when re-projecting)
    catalog_params: Tuple[str, ...] = ()

    # -- encoding ---------------------------------------------------------------
    @property
    def n(self) -> int:
        return len(self.params)

    def integer_mask(self) -> np.ndarray:
        return np.array([p.kind == "int" for p in self.params])

    def is_integer(self, name: str) -> bool:
        return any(p.name == name and p.kind == "int" for p in self.params)

    def int_step(self, i: int) -> float:
        p = self.params[i]
        return 1.0 / max(p.hi - p.lo, 1.0)

    def decode(self, u: Sequence[float]) -> Dict[str, float]:
        out = {}
        for p, x in zip(self.params, np.clip(np.asarray(u, float), 0, 1)):
            if p.kind == "log":
                v = math.exp(math.log(p.lo) + x * (math.log(p.hi) - math.log(p.lo)))
            elif p.kind == "int":
                v = float(int(round(p.lo + x * (p.hi - p.lo))))
            else:
                v = p.lo + x * (p.hi - p.lo)
            out[p.name] = float(v)
        return out

    def encode(self, params: Dict[str, float]) -> np.ndarray:
        u = []
        for p in self.params:
            v = float(params[p.name])
            if p.kind == "log":
                x = (math.log(max(v, 1e-300)) - math.log(p.lo)) / (math.log(p.hi) - math.log(p.lo))
            else:
                x = (v - p.lo) / (p.hi - p.lo)
            u.append(min(max(x, 0.0), 1.0))
        return np.array(u)

    # -- physics ------------------------------------------------------------------
    def material(self, material: Optional[Material], ctx: PhysicsContext) -> Material:
        return (material or MATERIALS[self.reference_material]).scaled(ctx)

    def evaluate(self, params: Dict[str, float], ctx: PhysicsContext, material: Optional[Material] = None
                 ) -> DesignEval:
        try:
            with np.errstate(all="ignore"):
                ev = self._evaluate(params, ctx, self.material(material, ctx))
        except (ValueError, ZeroDivisionError, OverflowError, np.linalg.LinAlgError, FloatingPointError):
            return _infeasible(reason="model_failure")
        ev.metrics = {k: float(v) for k, v in ev.metrics.items()}
        ev.constraints = {k: (float(v) if np.isfinite(v) else 10.0) for k, v in ev.constraints.items()}
        if not np.isfinite(ev.efficiency):
            ev.efficiency = 0.0
        if not np.isfinite(ev.stability):
            ev.stability = 0.0
        ev.efficiency = float(np.clip(ev.efficiency, 0, 1))
        ev.stability = float(np.clip(ev.stability, 0, 1))
        return ev

    def _evaluate(self, p: Dict[str, float], ctx: PhysicsContext, m: Material) -> DesignEval:
        raise NotImplementedError

    def cosmic_contexts(self, alien: PhysicsContext) -> List[PhysicsContext]:
        return [c.with_materials_of(alien) for c in PLANETS.values()]

    def cosmic_score(self, params: Dict[str, float], contexts: List[PhysicsContext]) -> float:
        vals = []
        for c in contexts:
            e = self.evaluate(params, c)
            vals.append(e.efficiency * e.soft_feasibility)
        return float(np.mean(vals))

    # -- real-world hooks -----------------------------------------------------------
    def material_ok(self, m: Material) -> bool:
        return True

    def merit_index(self, m: Material) -> float:
        return m.yield_strength / m.density

    def snap(self, params: Dict[str, float]) -> Dict[str, float]:
        return dict(params)

    def bom(self, params: Dict[str, float], m: Material) -> List[Dict[str, Any]]:
        return []

    def build_steps(self, params: Dict[str, float], m: Material) -> List[str]:
        return []


def _nearest(v: float, options: Sequence[float]) -> float:
    return float(min(options, key=lambda o: abs(o - v)))


# ---------------------------------------------------------------------------
# 1. Propulsion nozzle
# ---------------------------------------------------------------------------

def _area_ratio(M: float, g: float) -> float:
    return (1.0 / M) * ((2.0 / (g + 1.0)) * (1.0 + 0.5 * (g - 1.0) * M * M)) ** ((g + 1.0) / (2.0 * (g - 1.0)))


class NozzleDomain(DesignDomain):
    name = "propulsion_nozzle"
    title = "de Laval rocket nozzle (conical, regeneratively cooled wall)"
    params = [ParamSpec("chamber_pressure", 0.5e6, 20e6, "log", "Pa"),
              ParamSpec("expansion_ratio", 2.0, 100.0, "log"),
              ParamSpec("throat_radius", 0.005, 0.08, "log", "m"),
              ParamSpec("chamber_temperature", 2200.0, 3600.0, "lin", "K"),
              ParamSpec("molar_mass", 0.012, 0.030, "lin", "kg/mol"),
              ParamSpec("half_angle", 8.0, 25.0, "lin", "deg"),
              ParamSpec("wall_thickness", 0.0008, 0.010, "log", "m")]
    reference_material = "inconel718"
    proof_factor = 1.5

    def _evaluate(self, p, ctx, m):
        gam = ctx.propellant_gamma
        pc, eps, rt = p["chamber_pressure"], p["expansion_ratio"], p["throat_radius"]
        Tc, Mm, alpha, t = p["chamber_temperature"], p["molar_mass"], math.radians(p["half_angle"]), p["wall_thickness"]
        R = R_UNIVERSAL / Mm
        Me = brentq(lambda M: _area_ratio(M, gam) - eps, 1.0 + 1e-9, 200.0)
        pe = pc * (1.0 + 0.5 * (gam - 1.0) * Me * Me) ** (-gam / (gam - 1.0))
        Gam = math.sqrt(gam) * (2.0 / (gam + 1.0)) ** ((gam + 1.0) / (2.0 * (gam - 1.0)))
        cstar = math.sqrt(R * Tc) / Gam
        k = 2.0 * gam * gam / (gam - 1.0) * (2.0 / (gam + 1.0)) ** ((gam + 1.0) / (gam - 1.0))
        cf_mom = math.sqrt(k * (1.0 - (pe / pc) ** ((gam - 1.0) / gam)))
        lam = 0.5 * (1.0 + math.cos(alpha))
        pa = ctx.ambient_pressure
        CF = lam * cf_mom + (pe - pa) / pc * eps
        At = math.pi * rt * rt
        F = CF * pc * At
        mdot = pc * At / cstar
        Isp = F / (mdot * G0)
        Isp_ideal = math.sqrt(k) * cstar / G0
        rc = 2.0 * rt
        allow = 0.6 * m.yield_strength  # hot-wall derating
        SF = allow / (pc * rc / t)
        re = rt * math.sqrt(eps)
        L_div = (re - rt) / math.tan(alpha)
        L_ch = 0.25  # characteristic length L* = 1 m with contraction ratio 4
        area = (2 * math.pi * rc * L_ch + math.pi * (rc + rt) * math.hypot(rc - rt, rc - rt)
                + math.pi * (re + rt) * math.hypot(L_div, re - rt))
        mass = 2.5 * area * t * m.density + 0.2
        TW = F / (mass * ctx.g) if ctx.g > 0 else 1e3
        sep_ratio = pe / (0.4 * pa) if pa > 0 else 10.0
        cons = {"wall_safety": (1.5 - SF) / 1.5, "flow_separation": 1.0 - sep_ratio,
                "thrust_to_weight": (20.0 - TW) / 20.0}
        stab = _combine_margins([margin(SF, 1.5, 3.0), margin(sep_ratio, 1.0, 3.0), margin(TW, 20.0, 80.0)])
        metrics = {"Isp_s": Isp, "Isp_ideal_s": Isp_ideal, "thrust_N": F, "mass_flow_kg_s": mdot, "exit_mach": Me,
                   "exit_pressure_Pa": pe, "thrust_coefficient": CF, "c_star_m_s": cstar, "wall_SF": SF,
                   "thrust_to_weight": TW, "mass_kg": mass, "length_m": L_ch + (rc - rt) + L_div}
        return DesignEval(metrics, Isp / Isp_ideal, stab, cons)

    def material_ok(self, m):
        return m.max_service_temp >= 900 and ("cnc" in m.processes or "slm" in m.processes)

    def merit_index(self, m):
        return 0.6 * m.yield_strength / m.density

    def snap(self, p):
        q = dict(p)
        q["wall_thickness"] = max(0.0008, round(p["wall_thickness"] * 1e4) / 1e4)
        q["throat_radius"] = round(p["throat_radius"] * 1e4) / 1e4
        q["half_angle"] = round(p["half_angle"] * 2) / 2
        return q

    def bom(self, p, m):
        ev = self.evaluate(p, PhysicsContext.earth(), m)
        mass = ev.metrics.get("mass_kg", 1.0)
        return [{"item": f"nozzle/chamber body, {m.name} (SLM or 5-axis CNC)", "qty": 1, "mass_kg": mass * 0.6,
                 "cost_usd": mass * 0.6 * m.cost_per_kg * 8.0},
                {"item": "injector plate + manifold", "qty": 1, "mass_kg": mass * 0.3, "cost_usd": 1500.0},
                {"item": "flange bolts, seals, instrumentation ports", "qty": 1, "mass_kg": mass * 0.1,
                 "cost_usd": 300.0}]

    def build_steps(self, p, m):
        return [f"Generate the contour from the SCAD/STL (throat r = {p['throat_radius'] * 1e3:.1f} mm, expansion ratio"
                f" {p['expansion_ratio']:.1f}, half-angle {p['half_angle']:.1f} deg).",
                f"Fabricate in {m.name} by laser powder-bed fusion with integral regenerative cooling channels,"
                f" or machine from bar; minimum wall {p['wall_thickness'] * 1e3:.2f} mm.",
                "Stress-relieve and HIP printed parts; inspect channels by CT scan; flow-test the coolant circuit.",
                "Hydrostatically proof-test the chamber at 1.5x design chamber pressure.",
                "Hot-fire testing only at a licensed propulsion test facility under its safety procedures."]


# ---------------------------------------------------------------------------
# 2. Cantilever truss
# ---------------------------------------------------------------------------

class TrussDomain(DesignDomain):
    name = "structure_truss"
    title = "2-D cantilever truss bracket (7 joints, 10 solid round members)"
    params = [ParamSpec("h0", 0.2, 1.0, "lin", "m"), ParamSpec("h1", 0.1, 0.9, "lin", "m"),
              ParamSpec("h2", 0.05, 0.7, "lin", "m"), ParamSpec("area_top", 1e-5, 3e-3, "log", "m^2"),
              ParamSpec("area_bottom", 1e-5, 3e-3, "log", "m^2"), ParamSpec("area_web", 5e-6, 2e-3, "log", "m^2")]
    reference_material = "al6061_t6"
    catalog_params = ("area_top", "area_bottom", "area_web")
    SPAN = 2.0
    PAYLOAD = 300.0
    MEMBERS = [(0, 1, "bottom"), (1, 2, "bottom"), (2, 3, "bottom"), (4, 5, "top"), (5, 6, "top"), (6, 3, "top"),
               (1, 5, "web"), (2, 6, "web"), (4, 1, "web"), (5, 2, "web")]
    FIXED = (0, 4)

    def nodes(self, p):
        L = self.SPAN
        return np.array([[0, 0], [L / 3, 0], [2 * L / 3, 0], [L, 0], [0, p["h0"]], [L / 3, p["h1"]],
                         [2 * L / 3, p["h2"]]], float)

    def solve(self, p, m, g):
        xy = self.nodes(p)
        areas = {"top": p["area_top"], "bottom": p["area_bottom"], "web": p["area_web"]}
        nn = len(xy)
        K = np.zeros((2 * nn, 2 * nn))
        f = np.zeros(2 * nn)
        info = []
        for i, j, grp in self.MEMBERS:
            d = xy[j] - xy[i]
            Lm = float(np.hypot(*d))
            c, s = d / Lm
            A = areas[grp]
            k = m.E * A / Lm
            T = np.array([c, s, -c, -s])
            idx = [2 * i, 2 * i + 1, 2 * j, 2 * j + 1]
            K[np.ix_(idx, idx)] += k * np.outer(T, T)
            w = m.density * A * Lm * g
            f[2 * i + 1] -= 0.5 * w
            f[2 * j + 1] -= 0.5 * w
            info.append((i, j, grp, Lm, c, s, A))
        f[2 * 3 + 1] -= self.PAYLOAD * g
        free = [d for n in range(nn) if n not in self.FIXED for d in (2 * n, 2 * n + 1)]
        Kff = K[np.ix_(free, free)]
        if np.linalg.cond(Kff) > 1e14:
            raise np.linalg.LinAlgError("mechanism")
        u = np.zeros(2 * nn)
        u[free] = np.linalg.solve(Kff, f[free])
        forces = []
        for (i, j, grp, Lm, c, s, A) in info:
            du = (u[2 * j] - u[2 * i]) * c + (u[2 * j + 1] - u[2 * i + 1]) * s
            forces.append(m.E * A / Lm * du)
        return xy, u, np.array(forces), info

    def _evaluate(self, p, ctx, m):
        xy, u, N, info = self.solve(p, m, ctx.g)
        A = np.array([x[6] for x in info])
        Lm = np.array([x[3] for x in info])
        sigma = N / A
        allow = m.yield_strength / 1.5
        I = A * A / (4 * math.pi)
        Pcr = math.pi ** 2 * m.E * I / Lm ** 2
        comp = N < 0
        buck = float(np.max(-N[comp] / (Pcr[comp] / 1.5))) if comp.any() else 0.0
        defl = float(np.hypot(u[6], u[7]))
        dlim = self.SPAN / 250.0
        mass = float(np.sum(m.density * A * Lm))
        cons = {"yield": float(np.max(np.abs(sigma))) / allow - 1.0, "buckling": buck - 1.0,
                "deflection": defl / dlim - 1.0}
        eff = float(np.clip(math.log10(self.PAYLOAD / mass) / 3.0, 0, 1))
        stab = _combine_margins([margin(allow / max(np.max(np.abs(sigma)), 1e-9), 1.0, 2.0),
                                 margin(1.0 / max(buck, 1e-9), 1.0, 2.0), margin(dlim / max(defl, 1e-12), 1.0, 2.0)])
        metrics = {"mass_kg": mass, "tip_deflection_mm": defl * 1e3, "max_stress_MPa": float(np.max(np.abs(sigma))) / 1e6,
                   "buckling_utilisation": buck, "payload_to_mass": self.PAYLOAD / mass}
        return DesignEval(metrics, eff, stab, cons)

    def material_ok(self, m):
        return bool({"cnc", "tube", "extrusion", "sls"} & set(m.processes))

    def merit_index(self, m):
        return math.sqrt(m.E) / m.density * math.sqrt(m.yield_strength / 1e6)

    def snap(self, p):
        q = dict(p)
        for k in ("area_top", "area_bottom", "area_web"):
            d = math.ceil(math.sqrt(4 * p[k] / math.pi) * 1e3) / 1e3  # round diameter up to whole mm
            q[k] = math.pi * d * d / 4
        for k in ("h0", "h1", "h2"):
            q[k] = round(p[k] / 0.005) * 0.005
        return q

    def bom(self, p, m):
        xy = self.nodes(p)
        rows = []
        for grp, key in (("top", "area_top"), ("bottom", "area_bottom"), ("web", "area_web")):
            L = sum(float(np.hypot(*(xy[j] - xy[i]))) for i, j, g_ in self.MEMBERS if g_ == grp)
            d = math.sqrt(4 * p[key] / math.pi)
            mass = m.density * p[key] * L
            rows.append({"item": f"{grp} members: {d * 1e3:.0f} mm round bar, {m.name}", "qty": 1, "length_m": L,
                         "mass_kg": mass, "cost_usd": mass * m.cost_per_kg * 1.5})
        rows.append({"item": "gusset plates (6 mm) + M8 bolts, 7 joints", "qty": 7, "mass_kg": 0.6, "cost_usd": 70.0})
        return rows

    def build_steps(self, p, m):
        return ["Cut members to the joint-to-joint lengths in the BOM (add gusset overlap).",
                f"Machine gusset plates in {m.name}; drill M8 clearance holes per joint.",
                "Assemble on a flat jig; bolt joints (pin-jointed assumption) and torque to spec.",
                "Anchor joints 0 and 4 to a rigid wall; apply the payload gradually while monitoring tip deflection.",
                f"Acceptance: tip deflection under {self.PAYLOAD:.0f} kg below {self.SPAN / 250 * 1e3:.1f} mm."]


# ---------------------------------------------------------------------------
# 3. Flywheel
# ---------------------------------------------------------------------------

class FlywheelDomain(DesignDomain):
    name = "energy_flywheel"
    title = "annular flywheel rotor in a (partially) evacuated housing"
    params = [ParamSpec("outer_radius", 0.1, 1.0, "lin", "m"), ParamSpec("inner_ratio", 0.05, 0.9, "lin"),
              ParamSpec("thickness", 0.02, 0.4, "log", "m"), ParamSpec("rpm", 1000.0, 60000.0, "log", "1/min"),
              ParamSpec("housing_pressure_ratio", 1e-5, 1.0, "log")]
    reference_material = "cfrp_ud"
    proof_factor = 1.15

    def _evaluate(self, p, ctx, m):
        R, a, h = p["outer_radius"], p["inner_ratio"] * p["outer_radius"], p["thickness"]
        w = p["rpm"] * 2 * math.pi / 60.0
        nu = m.poisson
        smax = (3 + nu) / 4 * m.density * w * w * (R * R + (1 - nu) / (3 + nu) * a * a)
        SF = m.yield_strength / smax
        mass = m.density * math.pi * (R * R - a * a) * h
        I = 0.5 * mass * (R * R + a * a)
        E = 0.5 * I * w * w
        e = E / mass
        e_lim = 0.5 * m.yield_strength / m.density
        rho_h = ctx.air_density * p["housing_pressure_ratio"]
        vtip = w * R
        if rho_h > 0:
            Re = rho_h * w * R * R / ctx.air_viscosity
            Cm = 3.87 / math.sqrt(Re) if Re < 3e5 else 0.146 * Re ** -0.2
            Pw = 0.5 * Cm * rho_h * w ** 3 * R ** 5 * (vtip / 100.0) ** (ctx.drag_exponent - 2.0)
        else:
            Pw = 0.0
        Pb = 0.0015 * mass * ctx.g * w * 0.01
        tau_h = E / max(Pw + Pb, 1e-9) / 3600.0
        E_kwh = E / 3.6e6
        cons = {"burst_safety": (2.0 - SF) / 2.0, "aspect_ratio": (0.05 - h / R) / 0.05,
                "min_energy": (0.5 - E_kwh) / 0.5}
        eff = 0.6 * min(1.0, e / e_lim) + 0.4 * (1.0 - math.exp(-tau_h / 24.0))
        stab = _combine_margins([margin(SF, 2.0, 2.0), margin(h / R, 0.05, 4.0), margin(tau_h, 6.0, 8.0)])
        metrics = {"energy_kWh": E_kwh, "specific_energy_Wh_kg": e / 3600.0, "burst_SF": SF, "tip_speed_m_s": vtip,
                   "self_discharge_h": tau_h, "windage_W": Pw, "bearing_W": Pb, "mass_kg": mass,
                   "max_stress_MPa": smax / 1e6}
        return DesignEval(metrics, eff, stab, cons)

    def material_ok(self, m):
        return bool({"filament_winding", "cnc"} & set(m.processes))

    def merit_index(self, m):
        return m.yield_strength / m.density

    def snap(self, p):
        q = dict(p)
        q["rpm"] = round(p["rpm"] / 100) * 100
        q["thickness"] = round(p["thickness"] * 1e3) / 1e3
        q["outer_radius"] = round(p["outer_radius"] * 1e3) / 1e3
        return q

    def bom(self, p, m):
        ev = self.evaluate(p, PhysicsContext.earth(), m)
        rows = [{"item": f"rotor, {m.name}", "qty": 1, "mass_kg": ev.metrics.get("mass_kg", 0.0),
                 "cost_usd": ev.metrics.get("mass_kg", 0.0) * m.cost_per_kg * 3.0},
                {"item": "steel hub + shaft", "qty": 1, "mass_kg": 2.0, "cost_usd": 250.0},
                {"item": "precision bearings (or magnetic bearing set)", "qty": 2, "mass_kg": 0.5, "cost_usd": 300.0},
                {"item": "burst-containment steel housing", "qty": 1, "mass_kg": 3.0 * ev.metrics.get("mass_kg", 1.0),
                 "cost_usd": 400.0 + 20.0 * ev.metrics.get("mass_kg", 1.0)}]
        if p["housing_pressure_ratio"] < 0.01:
            rows.append({"item": "vacuum pump + gauge", "qty": 1, "mass_kg": 8.0, "cost_usd": 2000.0})
        return rows

    def build_steps(self, p, m):
        return [f"Filament-wind (or machine) the rotor in {m.name}: R = {p['outer_radius'] * 1e3:.0f} mm,"
                f" bore = {p['inner_ratio'] * p['outer_radius'] * 1e3:.0f} mm, thickness {p['thickness'] * 1e3:.0f} mm.",
                "Cure per resin schedule; machine faces; press onto the hub with controlled interference.",
                "Balance to ISO 21940 grade G2.5 or better.",
                "Install in the burst-containment housing; evacuate to the design pressure ratio"
                f" {p['housing_pressure_ratio']:.1e}.",
                f"Spin-test remotely in a spin pit up to 1.15x the design speed ({p['rpm'] * 1.15:.0f} rpm)"
                " before any operation near people."]


# ---------------------------------------------------------------------------
# 4. Drone frame
# ---------------------------------------------------------------------------

TUBE_OD_MM = (6, 8, 10, 12, 14, 16, 18, 20, 22, 25, 28, 30)
PROP_IN = tuple(range(3, 16))
PACK_MAH = (1300, 1500, 1800, 2200, 3000, 4000, 5000, 6000, 8000, 10000, 16000, 22000)


class DroneFrameDomain(DesignDomain):
    name = "drone_frame"
    title = "multirotor frame (tubular arms)"
    params = [ParamSpec("arm_length", 0.08, 0.5, "lin", "m"), ParamSpec("arm_od", 0.006, 0.03, "lin", "m"),
              ParamSpec("wall_ratio", 0.05, 0.25, "lin"), ParamSpec("prop_diameter", 0.076, 0.381, "lin", "m"),
              ParamSpec("n_arms", 3, 8, "int"), ParamSpec("battery_wh", 20.0, 400.0, "log", "Wh")]
    reference_material = "cfrp_qi"
    catalog_params = ("arm_od", "wall_ratio", "prop_diameter", "battery_wh")
    PAYLOAD = 0.3

    def _evaluate(self, p, ctx, m):
        L, D, wr, Dp, n, Wh = (p["arm_length"], p["arm_od"], p["wall_ratio"], p["prop_diameter"], int(p["n_arms"]),
                               p["battery_wh"])
        if ctx.air_density < 1e-3:
            return DesignEval({"hover_time_min": 0.0}, 0.0, 0.0, {"no_atmosphere": 1.0})
        d = D * (1 - 2 * wr)
        A = math.pi / 4 * (D * D - d * d)
        I = math.pi / 64 * (D ** 4 - d ** 4)
        m_arm = m.density * A * L
        m_motor = 0.012 + 1.0 * Dp * Dp
        m_esc = 0.012
        mass = n * (m_arm + m_motor + m_esc) + (0.04 + 0.25 * L) + Wh / 180.0 + self.PAYLOAD
        T = mass * ctx.g / n
        A_disk = math.pi * Dp * Dp / 4
        P_ind = T ** 1.5 / math.sqrt(2 * ctx.air_density * A_disk)
        P_elec = n * P_ind / (0.65 * 0.8) + 5.0
        t_min = 0.85 * Wh * 3600.0 / P_elec / 60.0
        n_rps = math.sqrt(T / (0.11 * ctx.air_density * Dp ** 4))
        tip_mach = math.pi * Dp * n_rps * math.sqrt(2.0) / max(ctx.speed_of_sound, 1.0)
        M = 2 * T * L
        sigma = M * (D / 2) / I
        SF = m.yield_strength / sigma
        defl = 2 * T * L ** 3 / (3 * m.E * I)
        k = 3 * m.E * I / L ** 3
        fn = math.sqrt(k / (m_motor + m_esc + 0.24 * m_arm)) / (2 * math.pi)
        sep = abs(fn / n_rps - 1.0)
        spacing = 2 * L * math.sin(math.pi / n)
        cons = {"arm_strength": (2.0 - SF) / 2.0, "arm_deflection": defl / (0.02 * L) - 1.0,
                "resonance": (0.2 - sep) / 0.2, "prop_clearance": 1.1 * Dp / spacing - 1.0,
                "tip_mach": tip_mach / 0.6 - 1.0, "wall_min": (0.0005 - D * wr) / 0.0005}
        stab = _combine_margins([margin(SF, 2.0, 2.0), margin(0.02 * L / max(defl, 1e-12), 1.0, 3.0),
                                 margin(sep, 0.2, 3.0), margin(spacing / (1.1 * Dp), 1.0, 1.3)])
        metrics = {"hover_time_min": t_min, "total_mass_kg": mass, "hover_power_W": P_elec, "rotor_hz": n_rps,
                   "arm_natural_hz": fn, "tip_mach": tip_mach, "arm_SF": SF, "thrust_per_motor_N": T,
                   "mass_kg": mass}
        return DesignEval(metrics, min(1.0, t_min / 45.0), stab, cons)

    def material_ok(self, m):
        return bool({"tube", "fdm", "sls"} & set(m.processes)) and m.max_service_temp >= 330

    def merit_index(self, m):
        return math.sqrt(m.E) / m.density

    def snap(self, p):
        q = dict(p)
        od = _nearest(p["arm_od"] * 1e3, TUBE_OD_MM)
        q["arm_od"] = od / 1e3
        wall = max(0.5, round(p["wall_ratio"] * od * 2) / 2)  # 0.5 mm steps
        q["wall_ratio"] = min(wall / od, 0.25)
        q["prop_diameter"] = _nearest(p["prop_diameter"] / 0.0254, PROP_IN) * 0.0254
        mah = _nearest(p["battery_wh"] / 14.8 * 1000, PACK_MAH)
        q["battery_wh"] = mah / 1000 * 14.8
        q["arm_length"] = round(p["arm_length"] * 1e3) / 1e3
        return q

    def bom(self, p, m):
        n = int(p["n_arms"])
        D = p["arm_od"]
        d = D * (1 - 2 * p["wall_ratio"])
        arm_mass = m.density * math.pi / 4 * (D * D - d * d) * p["arm_length"]
        inch = p["prop_diameter"] / 0.0254
        return [{"item": f"arm tube {D * 1e3:.0f}x{(D - d) / 2 * 1e3:.1f} mm, {m.name}", "qty": n, "mass_kg": n * arm_mass,
                 "cost_usd": n * max(8.0, arm_mass * m.cost_per_kg * 4)},
                {"item": "centre plates (2x 2 mm CFRP) + standoffs", "qty": 1, "mass_kg": 0.04 + 0.25 * p["arm_length"],
                 "cost_usd": 40.0},
                {"item": f"brushless motor for {inch:.0f}\" props", "qty": n, "mass_kg": n * (0.012 + p["prop_diameter"] ** 2),
                 "cost_usd": 25.0 * n},
                {"item": "ESC", "qty": n, "mass_kg": 0.012 * n, "cost_usd": 15.0 * n},
                {"item": f"{inch:.0f}\" propellers (CW/CCW pairs)", "qty": n, "mass_kg": 0.01 * n, "cost_usd": 3.0 * n},
                {"item": f"4S LiPo {p['battery_wh'] / 14.8 * 1000:.0f} mAh", "qty": 1, "mass_kg": p["battery_wh"] / 180.0,
                 "cost_usd": 1.2 * p["battery_wh"]},
                {"item": "flight controller + GNSS + receiver", "qty": 1, "mass_kg": 0.05, "cost_usd": 120.0}]

    def build_steps(self, p, m):
        n = int(p["n_arms"])
        return [f"Cut {n} arm tubes ({m.name}) to {p['arm_length'] * 1e3:.0f} mm; deburr and scuff bond areas.",
                "CNC-cut centre plates from the SCAD outline; clamp arms between plates with bonded inserts.",
                "Print/machine motor mounts; mount motors at arm tips; route ESC wiring along arms.",
                "Flash and configure the flight controller; set motor order and propeller direction.",
                "Bench-test without propellers, then tethered hover test outdoors away from people;"
                " verify vibration spectrum shows no peak near the arm natural frequency.",
                "Fly only in compliance with local aviation regulations."]


# ---------------------------------------------------------------------------
# 5. Centrifugal pump impeller
# ---------------------------------------------------------------------------

class ImpellerDomain(DesignDomain):
    name = "pump_impeller"
    title = "closed centrifugal pump impeller (backward-curved blades)"
    params = [ParamSpec("r1", 0.015, 0.07, "lin", "m"), ParamSpec("r2", 0.05, 0.22, "lin", "m"),
              ParamSpec("b2", 0.003, 0.04, "log", "m"), ParamSpec("beta1", 10.0, 40.0, "lin", "deg"),
              ParamSpec("beta2", 15.0, 45.0, "lin", "deg"), ParamSpec("blades", 4, 9, "int"),
              ParamSpec("rpm", 900.0, 3600.0, "lin", "1/min")]
    reference_material = "ss304"
    Q = 0.012
    DP_REQ = 250e3
    BLADE_T = 0.003
    proof_factor = 1.5

    def _evaluate(self, p, ctx, m):
        r1, r2, b2, Z = p["r1"], p["r2"], p["b2"], int(p["blades"])
        b1r, b2r = math.radians(p["beta1"]), math.radians(p["beta2"])
        rho, mu, g, Q = ctx.liquid_density, ctx.liquid_viscosity, ctx.g, self.Q
        w = p["rpm"] * 2 * math.pi / 60
        u1, u2 = w * r1, w * r2
        rh = 0.35 * r1
        A1 = math.pi * (r1 * r1 - rh * rh)
        cm1 = Q / A1
        tau2 = max(0.5, 1 - Z * self.BLADE_T / (2 * math.pi * r2 * math.sin(b2r)))
        cm2 = Q / (2 * math.pi * r2 * b2 * tau2)
        slip = 1 - math.sqrt(math.sin(b2r)) / Z ** 0.7
        cu2 = slip * u2 - cm2 / math.tan(b2r)
        if cu2 <= 0:
            return _infeasible({"head_m": 0.0}, "negative_work")
        Hth = u2 * cu2 / g
        dwu = u1 - cm1 / math.tan(b1r)
        h_inc = 0.6 * dwu * dwu / (2 * g)
        w1 = math.hypot(cm1, u1)
        w2 = cm2 / math.sin(b2r)
        wavg = 0.5 * (w1 + w2)
        bavg = 0.5 * (b2 + A1 / (2 * math.pi * r1))
        beta_avg = 0.5 * (b1r + b2r)
        Lch = (r2 - r1) / math.sin(beta_avg)
        wch = max(2 * math.pi * 0.5 * (r1 + r2) * math.sin(beta_avg) / Z - self.BLADE_T, 1e-4)
        Dh = 2 * bavg * wch / (bavg + wch)
        Re = rho * wavg * Dh / mu
        f = 64 / Re if Re < 2300 else 0.316 * Re ** -0.25
        h_f = f * Lch / Dh * wavg * wavg / (2 * g) * (wavg / 5.0) ** (ctx.drag_exponent - 2.0)
        h_v = 0.15 * (cm2 * cm2 + cu2 * cu2) / (2 * g)
        H = Hth - h_inc - h_f - h_v
        dp = rho * g * H
        Re_d = rho * w * r2 * r2 / mu
        CM = 0.0255 * Re_d ** -0.2
        P_df = CM * rho * w ** 3 * r2 ** 5
        P_sh = rho * g * Q * Hth / 0.96 + P_df + 0.02 * rho * g * Q * Hth
        eta = Q * dp / P_sh if H > 0 else 0.0
        npshr = (1.1 * cm1 * cm1 + 0.25 * w1 * w1) / (2 * g)
        npsha = (ctx.ambient_pressure + rho * g * 1.0 - ctx.vapor_pressure) / (rho * g)
        sc = m.density * u2 * u2
        SF = m.yield_strength / sc
        nq = p["rpm"] * math.sqrt(Q) / max(H, 1e-3) ** 0.75
        cons = {"head_min": 1.0 - dp / self.DP_REQ, "head_max": dp / (1.4 * self.DP_REQ) - 1.0,
                "cavitation": (1.3 * npshr / npsha - 1.0) if npsha > 0 else 5.0,
                "de_haller": (0.7 - w2 / w1) / 0.7, "geometry": (1.4 * r1 - r2) / r2, "rim_stress": (3.0 - SF) / 3.0}
        stab = _combine_margins([margin(npsha / max(1.3 * npshr, 1e-9), 1.0, 2.0), margin(w2 / w1, 0.7, 1.3),
                                 margin(dp / self.DP_REQ, 1.0, 1.15), margin(SF, 3.0, 3.0)])
        mass = (math.pi * r2 * r2 * 0.004 * 2 + Z * (r2 - r1) / math.sin(beta_avg) * b2 * self.BLADE_T) * m.density
        metrics = {"pressure_rise_kPa": dp / 1e3, "head_m": H, "efficiency": eta, "shaft_power_W": P_sh,
                   "disk_friction_W": P_df, "npsh_required_m": npshr, "npsh_available_m": npsha,
                   "specific_speed_nq": nq, "de_haller": w2 / w1, "tip_speed_m_s": u2, "mass_kg": mass}
        return DesignEval(metrics, eta, stab, cons)

    def material_ok(self, m):
        return m.corrosion_resistant and bool({"cnc", "casting", "slm", "sls"} & set(m.processes))

    def merit_index(self, m):
        return m.yield_strength / m.density * (1.5 if "casting" in m.processes else 1.0)

    def snap(self, p):
        q = dict(p)
        for k in ("beta1", "beta2"):
            q[k] = round(p[k] * 2) / 2
        for std in (960.0, 1450.0, 1750.0, 2900.0, 3500.0):
            if abs(p["rpm"] / std - 1) < 0.03:
                q["rpm"] = std
        for k in ("r1", "r2", "b2"):
            q[k] = round(p[k] * 1e4) / 1e4
        return q

    def bom(self, p, m):
        ev = self.evaluate(p, PhysicsContext.earth(), m)
        Pk = ev.metrics.get("shaft_power_W", 3000.0) / 1e3
        return [{"item": f"impeller, {m.name} (investment cast or 5-axis CNC)", "qty": 1,
                 "mass_kg": ev.metrics.get("mass_kg", 1.0), "cost_usd": 400.0 + 20 * ev.metrics.get("mass_kg", 1.0) * m.cost_per_kg},
                {"item": "volute casing (cast iron/SS)", "qty": 1, "mass_kg": 8.0, "cost_usd": 350.0},
                {"item": "shaft, key, mechanical seal, 2 bearings", "qty": 1, "mass_kg": 3.0, "cost_usd": 220.0},
                {"item": f"induction motor >= {1.15 * Pk:.1f} kW @ {p['rpm']:.0f} rpm (VFD if non-standard)", "qty": 1,
                 "mass_kg": 10.0 * Pk, "cost_usd": 150.0 * Pk + 200.0}]

    def build_steps(self, p, m):
        return [f"Export the impeller SCAD/STL: r1 = {p['r1'] * 1e3:.1f} mm, r2 = {p['r2'] * 1e3:.1f} mm,"
                f" b2 = {p['b2'] * 1e3:.1f} mm, {int(p['blades'])} blades, beta1/beta2 = {p['beta1']:.1f}/{p['beta2']:.1f} deg.",
                f"Produce in {m.name} by investment casting (production) or SLS/SLM (prototype); machine bore and"
                " wear rings.",
                "Dynamically balance to ISO 21940 G6.3.",
                "Assemble with volute sized for the design flow; prime the pump before start-up.",
                f"Measure the H-Q curve per ISO 9906 at {p['rpm']:.0f} rpm; verify {self.DP_REQ / 1e3:.0f} kPa at"
                f" {self.Q * 1e3:.0f} L/s and absence of cavitation noise."]


# ---------------------------------------------------------------------------
# 6. Spur gear pair
# ---------------------------------------------------------------------------

_LEWIS_Z = np.array([12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 24, 26, 28, 30, 34, 38, 43, 50, 60, 75, 100, 150, 300, 400])
_LEWIS_Y = np.array([0.245, 0.261, 0.277, 0.290, 0.296, 0.303, 0.309, 0.314, 0.322, 0.328, 0.331, 0.337, 0.346,
                     0.353, 0.359, 0.371, 0.384, 0.397, 0.409, 0.422, 0.435, 0.447, 0.460, 0.472, 0.480])
ISO_MODULES = (1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0)


def lewis_form_factor(z: float) -> float:
    """Lewis form factor Y for 20 deg full-depth teeth (Shigley, Table 14-2)."""
    return float(np.interp(z, _LEWIS_Z, _LEWIS_Y))


def _inv(a: float) -> float:
    return math.tan(a) - a


class GearDomain(DesignDomain):
    name = "gear_pair"
    title = "external spur gear pair, 5 kW @ 1500 rpm, ratio 3"
    params = [ParamSpec("module", 0.001, 0.006, "lin", "m"), ParamSpec("z1", 12, 40, "int"),
              ParamSpec("face_factor", 6.0, 16.0, "lin"), ParamSpec("pressure_angle", 14.5, 25.0, "lin", "deg"),
              ParamSpec("shift", -0.3, 0.6, "lin")]
    reference_material = "steel4140_qt"
    catalog_params = ("module", "pressure_angle", "shift")
    POWER = 5000.0
    RPM = 1500.0
    RATIO = 3.0
    proof_factor = 1.25

    def geometry(self, p):
        mod, z1, phi, x1 = p["module"], int(p["z1"]), math.radians(p["pressure_angle"]), p["shift"]
        z2 = int(round(self.RATIO * z1))
        x2 = -x1
        r1, r2 = mod * z1 / 2, mod * z2 / 2
        rb1, rb2 = r1 * math.cos(phi), r2 * math.cos(phi)
        ra1, ra2 = r1 + mod * (1 + x1), r2 + mod * (1 + x2)
        return mod, z1, z2, phi, x1, x2, r1, r2, rb1, rb2, ra1, ra2

    def _evaluate(self, p, ctx, m):
        mod, z1, z2, phi, x1, x2, r1, r2, rb1, rb2, ra1, ra2 = self.geometry(p)
        i = z2 / z1
        pb = math.pi * mod * math.cos(phi)
        L1 = math.sqrt(ra1 ** 2 - rb1 ** 2) - r1 * math.sin(phi)
        L2 = math.sqrt(ra2 ** 2 - rb2 ** 2) - r2 * math.sin(phi)
        e1, e2 = L1 / pb, L2 / pb
        eps = e1 + e2

        def tip_thickness(ra, rb, z, x):
            aa = math.acos(rb / ra)
            return 2 * ra * ((math.pi / 2 + 2 * x * math.tan(phi)) / z + _inv(phi) - _inv(aa))

        sa1, sa2 = tip_thickness(ra1, rb1, z1, x1), tip_thickness(ra2, rb2, z2, x2)
        zmin1 = 2 * (1 - x1) / math.sin(phi) ** 2
        zmin2 = 2 * (1 - x2) / math.sin(phi) ** 2
        w1 = self.RPM * 2 * math.pi / 60
        V = w1 * r1
        Wt = self.POWER / V
        Kv = (6.1 + V) / 6.1
        b = p["face_factor"] * mod
        fac = math.sqrt(math.tan(phi) / math.tan(math.radians(20.0)))
        Y1 = lewis_form_factor(z1) * fac * (1 + 0.25 * x1)
        Y2 = lewis_form_factor(z2) * fac * (1 + 0.25 * x2)
        sb = Kv * Wt / (b * mod * min(Y1, Y2))
        allow_b = m.yield_strength / 2.0
        Cp = math.sqrt(1.0 / (math.pi * 2 * (1 - m.poisson ** 2) / m.E))
        Ig = math.cos(phi) * math.sin(phi) / 2 * i / (i + 1)
        sH = Cp * math.sqrt(Kv * Wt / (b * 2 * r1 * Ig))
        Sc = ((2.22 * (m.ultimate_strength / 3.45e6) + 200.0) * 1e6) if m.E > 50e9 else 1.0 * m.ultimate_strength
        Hv = math.pi * (i + 1) / (z1 * i) * (1 - eps + e1 ** 2 + e2 ** 2)
        mu = 0.05 * (ctx.liquid_viscosity / 1e-3) ** 0.1
        eta = 1 - mu * Hv
        mass = m.density * math.pi * b * (r1 ** 2 + r2 ** 2) * 0.85
        pd = self.POWER / mass
        cons = {"ratio": abs(i - self.RATIO) / (0.02 * self.RATIO) - 1.0, "contact_ratio": 1.25 / eps - 1.0,
                "undercut_pinion": zmin1 / z1 - 1.0, "undercut_gear": zmin2 / z2 - 1.0,
                "tip_thickness": 0.25 * mod / max(min(sa1, sa2), 1e-12) - 1.0 if min(sa1, sa2) > 0 else 5.0,
                "bending": sb / allow_b - 1.0, "contact": sH / (Sc / 1.2) - 1.0}
        eff = 0.5 * float(np.clip((eta - 0.97) / 0.03, 0, 1)) + 0.5 * float(np.clip(math.log10(pd) / 4.3, 0, 1))
        stab = _combine_margins([margin(allow_b / sb, 1.0, 2.0), margin(Sc / 1.2 / sH, 1.0, 1.5),
                                 margin(eps, 1.25, 1.3), margin(z1 / zmin1, 1.0, 1.3)])
        metrics = {"mesh_efficiency": eta, "contact_ratio": eps, "bending_stress_MPa": sb / 1e6,
                   "contact_stress_MPa": sH / 1e6, "power_density_W_kg": pd, "center_distance_mm": (r1 + r2) * 1e3,
                   "z2": z2, "ratio": i, "pitch_velocity_m_s": V, "face_width_mm": b * 1e3, "mass_kg": mass}
        return DesignEval(metrics, eff, stab, cons)

    def material_ok(self, m):
        # Only gear-grade materials: the Brinell-based surface-endurance model is
        # valid for through-hardened steels; acetal is the standard polymer gear.
        return bool({"hobbing", "injection"} & set(m.processes)) and m.max_service_temp >= 330

    def merit_index(self, m):
        return m.ultimate_strength / 1e6 * (1.0 if "hobbing" in m.processes else 0.6)

    def snap(self, p):
        q = dict(p)
        q["module"] = _nearest(p["module"] * 1e3, ISO_MODULES) / 1e3
        q["pressure_angle"] = 20.0 if 17.0 <= p["pressure_angle"] <= 23.0 else _nearest(p["pressure_angle"], (14.5, 20.0, 25.0))
        q["shift"] = round(p["shift"] / 0.05) * 0.05
        b_mm = round(p["face_factor"] * p["module"] * 1e3)
        q["face_factor"] = b_mm / (q["module"] * 1e3)
        return q

    def bom(self, p, m):
        mod, z1, z2, *_ = self.geometry(p)
        b = p["face_factor"] * mod
        out = []
        for name, z in (("pinion", z1), ("gear", z2)):
            r = mod * z / 2
            mass = m.density * math.pi * (r + mod) ** 2 * b
            out.append({"item": f"{name} blank, {m.name}, z={z}, m={mod * 1e3:.2f} mm, b={b * 1e3:.0f} mm", "qty": 1,
                        "mass_kg": mass, "cost_usd": mass * m.cost_per_kg + 120.0})
        out.append({"item": "hobbing + heat treatment + grinding", "qty": 2, "mass_kg": 0.0, "cost_usd": 400.0})
        out.append({"item": "shafts, bearings, housing", "qty": 1, "mass_kg": 6.0, "cost_usd": 350.0})
        return out

    def build_steps(self, p, m):
        mod, z1, z2, phi, x1, *_ = self.geometry(p)
        return [f"Turn blanks in {m.name}; pinion z = {z1}, gear z = {z2}, module {mod * 1e3:.2f} mm,"
                f" pressure angle {p['pressure_angle']:.1f} deg, profile shift x1 = {x1:+.2f} / x2 = {-x1:+.2f}.",
                "Hob teeth (or wire-EDM/print for prototypes); through-harden or nitride as specified for steels.",
                "Grind/finish flanks to ISO 1328 grade 6-7; deburr tips.",
                f"Mount on parallel shafts at centre distance {(mod * (z1 + z2) / 2) * 1e3:.2f} mm (no shift sum),"
                " check backlash and contact pattern with marking compound.",
                "Run-in under 25 % load with ISO VG 220 oil, then load-test to 1.25x rated torque."]


# ---------------------------------------------------------------------------
# 7. Rotating space habitat
# ---------------------------------------------------------------------------

class HabitatDomain(DesignDomain):
    name = "cosmic_habitat"
    title = "rotating space habitat (pressurised drum section)"
    params = [ParamSpec("radius", 50.0, 3000.0, "log", "m"), ParamSpec("rpm", 0.3, 6.0, "log", "1/min"),
              ParamSpec("thickness", 0.002, 0.3, "log", "m"), ParamSpec("width", 10.0, 1000.0, "log", "m"),
              ParamSpec("pressure", 35e3, 101.3e3, "lin", "Pa")]
    reference_material = "cfrp_ud"
    AREAL_MASS = 2000.0  # rotating shielding + contents, kg/m^2
    proof_factor = 1.25

    def _evaluate(self, p, ctx, m):
        R, rpm, t, W, P = p["radius"], p["rpm"], p["thickness"], p["width"], p["pressure"]
        w = rpm * 2 * math.pi / 60
        a = w * w * R
        target = max(ctx.g, 0.1)
        sigma = (P + self.AREAL_MASS * a) * R / t + m.density * w * w * R * R
        SF = m.yield_strength / sigma
        cor = 2 * w * 1.0 / a
        h_side = min(20.0, 0.5 * W)
        mass = 2 * math.pi * R * (W + 2 * h_side) * t * m.density
        area = 2 * math.pi * R * W
        apk = area / mass
        Iratio = R * R / (0.5 * R * R + W * W / 12)
        cons = {"gravity_match": abs(a / target - 1.0) / 0.1 - 1.0, "rpm_comfort": rpm / 4.0 - 1.0,
                "coriolis": cor / 0.25 - 1.0, "gravity_gradient": (2.0 / R) / 0.15 - 1.0,
                "hull_strength": (2.0 - SF) / 2.0, "spin_stability": 1.2 / Iratio - 1.0}
        eff = float(np.clip((math.log10(apk) + 3.5) / 2.5, 0, 1)) * (P / 101.3e3) ** 0.2  # Stanford torus ~ 5e-3 m^2/kg
        stab = _combine_margins([margin(SF, 2.0, 1.5), margin(4.0 / rpm, 1.0, 2.0), margin(Iratio, 1.2, 1.4)])
        metrics = {"gravity_g": a / G0, "rim_speed_m_s": w * R, "hoop_stress_MPa": sigma / 1e6, "hull_SF": SF,
                   "habitable_area_km2": area / 1e6, "structure_mass_t": mass / 1e3, "population_capacity": area / 100.0,
                   "coriolis_ratio": cor, "spin_inertia_ratio": Iratio, "mass_kg": mass}
        return DesignEval(metrics, eff, stab, cons)

    def material_ok(self, m):
        return bool({"filament_winding", "sheet", "tube"} & set(m.processes))

    def merit_index(self, m):
        return m.yield_strength / m.density

    def snap(self, p):
        q = dict(p)
        q["rpm"] = round(p["rpm"] * 100) / 100
        q["thickness"] = max(0.002, round(p["thickness"] * 1e3) / 1e3)
        q["radius"] = round(p["radius"])
        q["width"] = round(p["width"])
        return q

    def bom(self, p, m):
        ev = self.evaluate(p, PhysicsContext.earth(), m)
        hull = ev.metrics.get("structure_mass_t", 0.0) * 1e3
        area = 2 * math.pi * p["radius"] * p["width"]
        return [{"item": f"pressure hull, {m.name}", "qty": 1, "mass_kg": hull, "cost_usd": hull * m.cost_per_kg},
                {"item": "radiation shielding (regolith, 2 t/m^2), in-space sourced", "qty": 1,
                 "mass_kg": area * self.AREAL_MASS, "cost_usd": 0.0},
                {"item": "1:1000 desktop model (PLA, from STL)", "qty": 1, "mass_kg": 0.2, "cost_usd": 10.0}]

    def build_steps(self, p, m):
        return ["Megastructure: the full-scale design is a conceptual study (launch/in-space manufacturing not costed).",
                f"Print the 1:1000 scale model from the STL (R = {p['radius']:.0f} m -> {p['radius']:.0f} mm).",
                f"Spin-up schedule: ramp to {p['rpm']:.2f} rpm over weeks for vestibular adaptation (Hall 1999).",
                "Hull segments must be proof-pressurised before habitation; spin about the maximum-inertia axis."]


# ---------------------------------------------------------------------------
# 8. Solar light sail
# ---------------------------------------------------------------------------

FILM_GAUGES_UM = (0.9, 1.5, 2.0, 2.5, 3.0, 5.0, 7.5, 12.5, 25.0)


class LightSailDomain(DesignDomain):
    name = "cosmic_lightsail"
    title = "square solar sail with diagonal booms"
    params = [ParamSpec("film_thickness", 0.5e-6, 25e-6, "log", "m"), ParamSpec("side_length", 5.0, 500.0, "log", "m"),
              ParamSpec("reflectivity", 0.6, 0.95, "lin"), ParamSpec("boom_linear_density", 0.002, 0.1, "log", "kg/m"),
              ParamSpec("distance_au", 0.25, 1.5, "lin", "AU")]
    reference_material = "kapton"
    catalog_params = ("film_thickness",)
    PAYLOAD = 5.0
    proof_factor = 1.0

    def _evaluate(self, p, ctx, m):
        t, L, eta_r, lam, r = (p["film_thickness"], p["side_length"], p["reflectivity"], p["boom_linear_density"],
                               p["distance_au"])
        S = 1361.0 / r ** 2
        A = L * L
        F = (1 + eta_r) * S / ctx.speed_of_light * A
        m_film = m.density * t * A
        m_boom = 2 * math.sqrt(2) * L * lam
        mass = m_film + m_boom + self.PAYLOAD
        ac = F / mass
        g_sun = 5.93e-3 / r ** 2 * ctx.g / G0
        T = ((1 - eta_r) * S / (0.65 * SIGMA_SB)) ** 0.25
        lam_req = 0.01 * (L / 50.0) ** 0.5
        cons = {"temperature": T / (0.8 * m.max_service_temp) - 1.0, "boom_stiffness": lam_req / lam - 1.0,
                "film_handling": 0.9e-6 / t - 1.0}
        eff = float(np.clip((math.log10(ac * 1e3) + 2.0) / 3.0, 0, 1))
        stab = _combine_margins([margin(0.8 * m.max_service_temp / T, 1.0, 1.5), margin(lam / lam_req, 1.0, 2.0)])
        metrics = {"char_accel_mm_s2": ac * 1e3, "lightness_number": ac / g_sun, "sail_temperature_K": T,
                   "total_mass_kg": mass, "area_m2": A, "force_mN": F * 1e3, "mass_kg": mass}
        return DesignEval(metrics, eff, stab, cons)

    def cosmic_contexts(self, alien):
        return [alien]

    def cosmic_score(self, params, contexts):
        vals = []
        for c in contexts:
            for r in (0.5, 1.0, 1.5):
                e = self.evaluate(dict(params, distance_au=r), c)
                vals.append(e.efficiency * e.soft_feasibility)
        return float(np.mean(vals))

    def material_ok(self, m):
        return "film" in m.processes

    def merit_index(self, m):
        return m.max_service_temp / m.density

    def snap(self, p):
        q = dict(p)
        q["film_thickness"] = _nearest(p["film_thickness"] * 1e6, FILM_GAUGES_UM) / 1e6
        q["reflectivity"] = min(p["reflectivity"], 0.91)  # vapour-deposited aluminium, practical
        return q

    def bom(self, p, m):
        A = p["side_length"] ** 2
        mf = m.density * p["film_thickness"] * A
        return [{"item": f"aluminised {m.name} film {p['film_thickness'] * 1e6:.1f} um", "qty": 1, "area_m2": A,
                 "mass_kg": mf, "cost_usd": 3.0 * A + mf * m.cost_per_kg},
                {"item": "deployable CFRP booms", "qty": 4, "mass_kg": 2 * math.sqrt(2) * p["side_length"] * p["boom_linear_density"],
                 "cost_usd": 4 * (2000 + 50 * p["side_length"])},
                {"item": "deployer, tip vanes, avionics (payload)", "qty": 1, "mass_kg": self.PAYLOAD, "cost_usd": 50000.0}]

    def build_steps(self, p, m):
        return [f"Procure {p['film_thickness'] * 1e6:.1f} um {m.name} with vapour-deposited aluminium front coat and"
                " high-emissivity back coat.",
                f"Cut four triangular quadrants for a {p['side_length']:.1f} m square sail; add rip-stop tape grid.",
                "Fold (frog-leg or Miura) onto the deployer spools; integrate booms and tip vanes.",
                "Ground deployment test in a gravity-offload rig; thermal-vacuum test the coated film."]


DOMAINS: Dict[str, Type[DesignDomain]] = {c.name: c for c in (NozzleDomain, TrussDomain, FlywheelDomain,
                                                                DroneFrameDomain, ImpellerDomain, GearDomain,
                                                                HabitatDomain, LightSailDomain)}


# ---------------------------------------------------------------------------
# Evolution
# ---------------------------------------------------------------------------

@dataclass
class EvolvedDesign:
    domain: str
    design_id: str
    u: List[float]
    params: Dict[str, float]
    objectives: Dict[str, float]
    metrics: Dict[str, float]
    feasible: bool
    context: str
    score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvolutionResult:
    domain: str
    context: str
    pareto: List[EvolvedDesign]
    champion: EvolvedDesign
    history: List[Dict[str, Any]]
    evaluations: int
    generative: Dict[str, Any] = field(default_factory=dict)
    rl: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> Dict[str, Any]:
        return {"domain": self.domain, "context": self.context, "pareto_size": len(self.pareto),
                "evaluations": self.evaluations, "champion": {"objectives": self.champion.objectives,
                                                               "params": self.champion.params,
                                                               "feasible": self.champion.feasible},
                "generative": self.generative, "rl": self.rl}


class TechEvolver:
    """NSGA-II over one design domain under one physics context."""

    OBJECTIVES = ("efficiency", "stability", "adaptability", "mappability", "cosmic")
    WEIGHTS = np.array([0.35, 0.2, 0.15, 0.2, 0.1])

    def __init__(self, domain: Any, context: Optional[PhysicsContext] = None, seed: int = 0, n_adapt: int = 5):
        self.domain = DOMAINS[domain]() if isinstance(domain, str) else domain
        self.ctx = context or PhysicsContext.earth()
        self.earth = PhysicsContext.earth()
        self.seed = seed
        rng = np.random.default_rng(seed)
        self.adapt_ctxs = [self.ctx.perturbed(rng, 0.15) for _ in range(n_adapt)]
        self.cosmic_ctxs = self.domain.cosmic_contexts(self.ctx)
        self.evaluations = 0

    def evaluate_u(self, u: np.ndarray) -> Tuple[np.ndarray, float, DesignEval]:
        d = self.domain
        p = d.decode(u)
        ev = d.evaluate(p, self.ctx)
        self.evaluations += 1
        eff = ev.efficiency * ev.soft_feasibility
        ad = []
        for c in self.adapt_ctxs:
            e = d.evaluate(p, c)
            ad.append(e.efficiency * e.soft_feasibility)
        ad = np.array(ad)
        adapt = float(ad.mean() * (1.0 - min(1.0, ad.std() / (ad.mean() + 1e-9)))) if ad.mean() > 0 else 0.0
        ee = d.evaluate(p, self.earth)
        mapp = ee.soft_feasibility * min(1.0, (ee.efficiency + 1e-9) / (ev.efficiency + 1e-9)) * (0.5 + 0.5 * ee.efficiency)
        cosmic = d.cosmic_score(p, self.cosmic_ctxs)
        F = np.array([eff, ev.stability * ev.soft_feasibility, adapt, mapp, cosmic])
        return F, ev.violation, ev

    def _batch(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        out = [self.evaluate_u(x) for x in X]
        return np.array([o[0] for o in out]), np.array([o[1] for o in out])

    def _design(self, u: np.ndarray, F: np.ndarray, ev: DesignEval) -> EvolvedDesign:
        return EvolvedDesign(domain=self.domain.name, design_id=uuid.uuid4().hex[:8], u=[float(x) for x in u],
                             params=self.domain.decode(u), objectives=dict(zip(self.OBJECTIVES, map(float, F))),
                             metrics=ev.metrics, feasible=ev.feasible, context=self.ctx.name,
                             score=float(self.WEIGHTS @ F))

    def evolve(self, generations: int = 25, pop_size: int = 32, generative_samples: int = 32) -> EvolutionResult:
        nsga = NSGA2(self.domain.n, len(self.OBJECTIVES), pop_size=pop_size, seed=self.seed)
        res = nsga.run(self._batch, generations)
        idx = res.front
        designs = []
        for i in idx:
            _, _, ev = self.evaluate_u(res.X[i])
            designs.append(self._design(res.X[i], res.F[i], ev))
        feas = [d for d in designs if d.feasible]
        pool = feas or designs
        champion = max(pool, key=lambda d: d.score)
        gen_info: Dict[str, Any] = {}
        if len(pool) >= 4 and generative_samples > 0:
            U = np.array([d.u for d in pool])
            model = LatentDesignModel(latent_dim=min(3, U.shape[1] - 1)).fit(U)
            S = np.clip(model.sample(generative_samples, np.random.default_rng(self.seed + 1)), 0, 1)
            FS, CVS = self._batch(S)
            allF = np.concatenate([res.F[idx], FS])
            allCV = np.concatenate([res.CV[idx], CVS])
            front = set(fast_non_dominated_sort(allF, allCV)[0].tolist())
            novel = [k for k in range(len(S)) if (len(idx) + k) in front]
            for k in novel:
                _, _, ev = self.evaluate_u(S[k])
                d = self._design(S[k], FS[k], ev)
                designs.append(d)
                if d.feasible and d.score > champion.score:
                    champion = d
            gen_info = {"latent_dim": model.q, "samples": int(len(S)), "feasible": int(np.sum(CVS <= 0)),
                        "joined_pareto_front": len(novel)}
        return EvolutionResult(domain=self.domain.name, context=self.ctx.name, pareto=designs, champion=champion,
                               history=res.history, evaluations=self.evaluations, generative=gen_info)

    # -- reinforcement learning -------------------------------------------------------
    def train_adaptive_designer(self, iterations: int = 40, episodes: int = 8, horizon: int = 8
                                ) -> Dict[str, Any]:
        """Learn a context-conditioned editing policy (REINFORCE) over
        planetary + alien contexts; report gain over random edits."""
        contexts = [(c.features(), c) for c in [self.ctx, self.earth, PLANETS["mars"], PLANETS["titan"],
                                                 PLANETS["venus_clouds"]]]
        d = self.domain

        def fit(u, c):
            e = d.evaluate(d.decode(u), c)
            return e.efficiency * e.soft_feasibility

        env = DesignEnvironment(fit, d.n, contexts, horizon=horizon, seed=self.seed)
        agent = DesignRLAgent(env.state_dim, d.n, seed=self.seed)
        before = agent.evaluate(env)
        agent.train(env, iterations=iterations, episodes_per_iter=episodes)
        after = agent.evaluate(env)
        rnd = agent.evaluate(env, random_policy=True)
        return {"untrained_gain": before, "trained_gain": after, "random_gain": rnd,
                "final_sigma": agent.history[-1]["sigma"] if agent.history else None,
                "learning_curve": [h["mean_return"] for h in agent.history]}


def evolve_technologies(context: PhysicsContext, domains: Optional[Sequence[str]] = None, generations: int = 25,
                        pop_size: int = 32, seed: int = 0, rl_iterations: int = 0, verbose: bool = True
                        ) -> Dict[str, EvolutionResult]:
    out: Dict[str, EvolutionResult] = {}
    for k, name in enumerate(domains or list(DOMAINS)):
        ev = TechEvolver(name, context, seed=seed + k)
        r = ev.evolve(generations=generations, pop_size=pop_size)
        if rl_iterations:
            r.rl = ev.train_adaptive_designer(iterations=rl_iterations)
        out[name] = r
        if verbose:
            c = r.champion
            print(f"  [{name:18s}] pareto={len(r.pareto):3d} feasible={c.feasible} score={c.score:.3f} "
                  + " ".join(f"{k[:4]}={v:.2f}" for k, v in c.objectives.items()), flush=True)
    return out


# ---------------------------------------------------------------------------
# Generative bracket (CPPN-NEAT + lattice-spring FEM)
# ---------------------------------------------------------------------------

class GenerativeBracketDesigner:
    """Free-form cantilever bracket evolved as a CPPN density field.

    The plate (nx x ny pixels of ``pixel`` metres, thickness ``t``) is
    clamped on its left edge and loaded by a payload at the middle of its
    right edge plus self-weight.  Material pixels are joined by axial springs
    to their 8 neighbours (k_orth = E t, k_diag = E t / 2, the square-lattice
    spring model that is isotropic with Poisson ratio 1/3).  Fitness is the
    stiffness relative to the solid plate under a volume budget, with
    stress and deflection penalties: the classical minimum-compliance problem
    of topology optimisation (Bendsoe & Sigmund 2003).
    """

    def __init__(self, ctx: Optional[PhysicsContext] = None, nx: int = 32, ny: int = 16, pixel: float = 0.005,
                 thickness: float = 0.006, payload: float = 5.0, volume_budget: float = 0.45,
                 material: str = "pa12", seed: int = 0):
        self.ctx = ctx or PhysicsContext.earth()
        self.nx, self.ny, self.pixel, self.t = nx, ny, pixel, thickness
        self.payload, self.vbudget = payload, volume_budget
        self.mat = MATERIALS[material].scaled(self.ctx)
        self.seed = seed
        self.gen = CPPNShapeGenerator(nx, ny, mirror_y=True)
        full = np.ones((ny, nx), bool)
        self.C_full = self.compliance(full)["compliance"]

    def compliance(self, shape: np.ndarray) -> Dict[str, Any]:
        ny, nx = shape.shape
        idx = -np.ones(shape.shape, int)
        filled = np.argwhere(shape)
        if len(filled) < 4:
            return {"ok": False}
        idx[shape] = np.arange(len(filled))
        E_t = self.mat.E * self.t
        rows, cols, ks, dirs = [], [], [], []
        for dy, dx, kf in ((0, 1, 1.0), (1, 0, 1.0), (1, 1, 0.5), (1, -1, 0.5)):
            x0, x1 = max(0, -dx), nx - max(0, dx)
            a = idx[0:ny - dy, x0:x1]
            b = idx[dy:ny, x0 + dx:x1 + dx]
            m = (a >= 0) & (b >= 0)
            rows.append(a[m])
            cols.append(b[m])
            e = np.array([dx, dy], float) / math.hypot(dx, dy)
            ks.append(np.full(int(m.sum()), kf * E_t))
            dirs.append(np.repeat(e[None], int(m.sum()), axis=0))
        I, J, K, Dv = (np.concatenate(rows), np.concatenate(cols), np.concatenate(ks), np.concatenate(dirs))
        n = len(filled)
        adj = coo_matrix((np.ones(len(I)), (I, J)), shape=(n, n))
        _, labels = connected_components(adj, directed=False)
        fixed_nodes = idx[:, 0][idx[:, 0] >= 0]
        if fixed_nodes.size == 0:
            return {"ok": False}
        main = labels == labels[fixed_nodes[0]]
        load_rows = range(ny // 2 - 2, ny // 2 + 2)
        load_nodes = [idx[r, nx - 1] for r in load_rows if idx[r, nx - 1] >= 0 and main[idx[r, nx - 1]]]
        if not load_nodes:
            return {"ok": False}
        keep = main[I] & main[J]
        I, J, K, Dv = I[keep], J[keep], K[keep], Dv[keep]
        blocks_r, blocks_c, vals = [], [], []
        for a_ in range(2):
            for b_ in range(2):
                kab = K * Dv[:, a_] * Dv[:, b_]
                blocks_r += [2 * I + a_, 2 * J + a_, 2 * I + a_, 2 * J + a_]
                blocks_c += [2 * I + b_, 2 * J + b_, 2 * J + b_, 2 * I + b_]
                vals += [kab, kab, -kab, -kab]
        Kg = coo_matrix((np.concatenate(vals), (np.concatenate(blocks_r), np.concatenate(blocks_c))),
                        shape=(2 * n, 2 * n)).tocsr()
        f = np.zeros(2 * n)
        g = self.ctx.g
        mpix = self.mat.density * self.pixel ** 2 * self.t
        f[1::2] -= np.where(main, mpix * g, 0.0)  # +y is "down" index-wise; sign irrelevant for compliance
        for ln in load_nodes:
            f[2 * ln + 1] -= self.payload * g / len(load_nodes)
        fixed = set()
        for fn_ in fixed_nodes:
            fixed.update((2 * fn_, 2 * fn_ + 1))
        active = np.repeat(main, 2)
        free = np.array([d for d in range(2 * n) if d not in fixed and active[d]])
        Kff = Kg[free][:, free] + 1e-9 * E_t * coo_matrix((np.ones(len(free)), (range(len(free)), range(len(free)))),
                                                          shape=(len(free), len(free))).tocsr()
        u = np.zeros(2 * n)
        u[free] = spsolve(Kff.tocsc(), f[free])
        if not np.all(np.isfinite(u)):
            return {"ok": False}
        du = (u[2 * J] - u[2 * I]) * Dv[:, 0] + (u[2 * J + 1] - u[2 * I + 1]) * Dv[:, 1]
        spring_force = K * du  # k already includes E t; force per unit strain * length ~ N
        stress = np.abs(spring_force) / (self.pixel * self.t) if spring_force.size else np.zeros(1)
        return {"ok": True, "compliance": float(f @ u), "max_disp": float(np.max(np.abs(u))),
                "max_stress": float(np.max(stress)), "volume_fraction": float(main.sum() / shape.size),
                "islands": int(len(filled) - main.sum())}

    def fitness_of_shape(self, shape: np.ndarray) -> Tuple[float, Dict[str, Any]]:
        r = self.compliance(shape)
        if not r.get("ok"):
            return 0.0, r
        vf = r["volume_fraction"]
        f = self.C_full / max(r["compliance"], 1e-30)
        if vf > self.vbudget:
            f *= math.exp(-15 * (vf - self.vbudget))
        if r["max_stress"] > self.mat.yield_strength / 2:
            f *= 0.5
        if r["max_disp"] > 1e-3:
            f *= math.exp(-(r["max_disp"] - 1e-3) / 1e-3)
        f *= 1.0 / (1.0 + 0.01 * r["islands"])
        return float(f), r

    def shape_of(self, genome: Any) -> np.ndarray:
        """Volume-constrained projection: keep the top ``volume_budget``
        fraction of the CPPN density field (ties -> solid plate, penalised)."""
        dens = self.gen.density(genome)
        if float(np.ptp(dens)) < 1e-9:
            return np.ones_like(dens, bool)
        thr = float(np.quantile(dens, 1.0 - self.vbudget))
        return dens > thr

    def run(self, generations: int = 25, pop_size: int = 40, verbose: bool = False) -> Dict[str, Any]:
        popn = NEATPopulation(CPPNShapeGenerator.N_INPUTS, 1, pop_size=pop_size, seed=self.seed)

        def fit(genome):
            return self.fitness_of_shape(self.shape_of(genome))[0]

        for gen in range(generations):
            popn.step(fit)
            if verbose:
                h = popn.history[-1]
                print(f"  [cppn {gen + 1}] best={h['best']:.3f} species={h['species']} complexity={h['complexity']}")
        best = popn.best
        shape = self.shape_of(best)
        f, info = self.fitness_of_shape(shape)
        return {"genome": best, "shape": shape, "density": self.gen.density(best), "fitness": f, "metrics": info,
                "history": popn.history, "pixel_m": self.pixel, "thickness_m": self.t, "material": self.mat.key,
                "context": self.ctx.name}
