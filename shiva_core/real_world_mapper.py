"""
SHIVA-1 :: Real-World Mapper
============================

Translates designs evolved under *alien* physics into buildable artefacts in
*our* universe.

Pipeline for one design::

    alien design u*  (evolved under PhysicsContext.from_genome(genome))
      |  1. candidate materials: buildable, process-compatible, ranked by an
      |     Ashby merit index for the domain (Ashby 2011)
      |  2. constrained projection  u_real = argmin ||u - u*||^2
      |                               s.t. g_i(u; Earth, material) <= 0   (SLSQP)
      |  3. snap to manufacturable standards (ISO modules, stock tubes, film
      |     gauges), re-verify
      |  4. Monte-Carlo performance prediction (material scatter, tolerances)
      |  5. bill of materials, cost, build & test instructions
      v  6. parametric CAD (OpenSCAD / STL / FreeCAD macro) via cad_generator

Material values are typical room-temperature handbook values (MatWeb / ASM
Handbook class sources).  **They are planning values: verify against the
supplier's certified datasheet, and have any load-bearing or pressurised part
reviewed by a qualified engineer before fabrication or test.**
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize

__all__ = ["Material", "MATERIALS", "MappedPrototype", "RealWorldMapper"]


@dataclass(frozen=True)
class Material:
    key: str
    name: str
    density: float  # kg/m^3
    E: float  # Pa
    yield_strength: float  # Pa (tensile strength for brittle composites/fibres)
    ultimate_strength: float  # Pa
    poisson: float
    cost_per_kg: float  # USD/kg, raw stock, order-of-magnitude
    max_service_temp: float  # K
    processes: Tuple[str, ...]
    buildable: bool = True
    corrosion_resistant: bool = False
    notes: str = ""

    def scaled(self, ctx: Any) -> "Material":
        """Material as it would exist in the context's universe (Press-Lightman
        scaling of moduli/strengths ~ alpha^5 and density ~ alpha^3)."""
        ms = getattr(ctx, "modulus_scale", 1.0)
        ss = getattr(ctx, "strength_scale", 1.0)
        ds = getattr(ctx, "density_scale", 1.0)
        if ms == ss == ds == 1.0:
            return self
        return replace(self, density=self.density * ds, E=self.E * ms, yield_strength=self.yield_strength * ss,
                       ultimate_strength=self.ultimate_strength * ss)

    def perturbed(self, rng: np.random.Generator, cov: float = 0.05) -> "Material":
        f = lambda: float(np.clip(rng.normal(1.0, cov), 0.7, 1.3))
        return replace(self, density=self.density * float(np.clip(rng.normal(1.0, cov / 3), 0.9, 1.1)),
                       E=self.E * f(), yield_strength=self.yield_strength * f(),
                       ultimate_strength=self.ultimate_strength * f())


MATERIALS: Dict[str, Material] = {m.key: m for m in [
    Material("al6061_t6", "Aluminium 6061-T6", 2700, 68.9e9, 276e6, 310e6, 0.33, 4.0, 423,
             ("cnc", "sheet", "extrusion", "tube"), True, True, "general-purpose structural aluminium"),
    Material("al7075_t6", "Aluminium 7075-T6", 2810, 71.7e9, 503e6, 572e6, 0.33, 6.0, 393,
             ("cnc", "sheet", "tube"), True, False, "high-strength aerospace aluminium"),
    Material("ti6al4v", "Titanium Ti-6Al-4V (annealed)", 4430, 113.8e9, 880e6, 950e6, 0.342, 30.0, 673,
             ("cnc", "slm", "tube"), True, True, "high specific strength, corrosion resistant"),
    Material("ss304", "Stainless steel 304", 8000, 193e9, 215e6, 505e6, 0.29, 3.5, 1073,
             ("cnc", "sheet", "casting", "tube"), True, True, "corrosion resistant, weldable"),
    Material("steel4140_qt", "Alloy steel 4140 (Q&T)", 7850, 205e9, 655e6, 1020e6, 0.29, 2.5, 700,
             ("cnc", "forging", "hobbing"), True, False, "through-hardening gear/shaft steel"),
    Material("inconel718", "Inconel 718 (aged)", 8190, 200e9, 1034e6, 1241e6, 0.29, 45.0, 973,
             ("cnc", "slm"), True, True, "hot-section superalloy"),
    Material("cfrp_qi", "CFRP quasi-isotropic laminate", 1600, 60e9, 570e6, 600e6, 0.30, 40.0, 393,
             ("layup", "tube", "sheet"), True, True, "carbon/epoxy, 0/45/90/-45 layup"),
    Material("cfrp_ud", "CFRP unidirectional (fibre direction)", 1600, 135e9, 1500e6, 1500e6, 0.30, 50.0, 393,
             ("filament_winding", "tube", "layup"), True, True, "hoop-wound or pultruded carbon/epoxy"),
    Material("pla", "PLA (FDM printed)", 1240, 3.5e9, 50e6, 55e6, 0.36, 25.0, 330,
             ("fdm",), True, False, "prototype-grade; creeps above ~50 C"),
    Material("petg", "PETG (FDM printed)", 1270, 2.1e9, 50e6, 53e6, 0.38, 25.0, 343,
             ("fdm",), True, True, "tougher than PLA"),
    Material("pa12", "Nylon PA12 (SLS printed)", 1010, 1.7e9, 48e6, 50e6, 0.40, 80.0, 368,
             ("sls",), True, True, "isotropic SLS polymer"),
    Material("pom", "Acetal POM (Delrin)", 1410, 3.1e9, 70e6, 72e6, 0.35, 5.0, 363,
             ("cnc", "injection", "hobbing"), True, True, "self-lubricating gear polymer"),
    Material("kapton", "Polyimide film (Kapton HN)", 1420, 2.5e9, 69e6, 231e6, 0.34, 100.0, 673,
             ("film",), True, True, "space-qualified thin film"),
    Material("zylon", "Zylon PBO fibre composite", 1560, 270e9, 5.8e9, 5.8e9, 0.30, 150.0, 773,
             ("filament_winding", "tube"), True, False, "fibre-direction values; UV/moisture sensitive"),
    Material("cnt_yarn", "Carbon-nanotube yarn (laboratory)", 1300, 100e9, 3.0e9, 3.0e9, 0.30, 5000.0, 773,
             ("filament_winding",), False, False, "not available at engineering scale - reference only"),
]}


@dataclass
class MappedPrototype:
    domain: str
    design_id: str
    material: str
    material_name: str
    alien_params: Dict[str, float]
    real_params: Dict[str, float]
    fidelity: float
    alien_efficiency: float
    real_efficiency: float
    feasible: bool
    snapped: bool
    constraint_values: Dict[str, float]
    prediction: Dict[str, Dict[str, float]]
    reliability: float
    bom: List[Dict[str, Any]]
    cost_usd: float
    mass_kg: float
    build_instructions: List[str]
    material_ranking: List[Dict[str, Any]]
    cad: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return json.loads(json.dumps(asdict(self), default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))

    def report(self) -> str:
        lines = [f"### {self.domain} prototype ({self.material_name})",
                 f"- feasible on Earth: **{self.feasible}**  (Monte-Carlo reliability {self.reliability:.0%})",
                 f"- fidelity to alien design: {self.fidelity:.3f}; efficiency alien {self.alien_efficiency:.3f}"
                 f" -> real {self.real_efficiency:.3f}",
                 f"- mass {self.mass_kg:.3g} kg, est. material+parts cost ${self.cost_usd:,.0f}",
                 "", "| parameter | alien | real |", "|---|---|---|"]
        for k in self.real_params:
            lines.append(f"| {k} | {self.alien_params.get(k, float('nan')):.4g} | {self.real_params[k]:.4g} |")
        lines.append("")
        lines.append("| predicted metric | mean | std |")
        lines.append("|---|---|---|")
        for k, v in self.prediction.items():
            lines.append(f"| {k} | {v['mean']:.4g} | {v['std']:.2g} |")
        lines.append("")
        lines.append("Build instructions:")
        lines.extend(f"{i + 1}. {s}" for i, s in enumerate(self.build_instructions))
        if self.notes:
            lines.append("")
            lines.extend(f"> {n}" for n in self.notes)
        return "\n".join(lines)


class RealWorldMapper:
    """Map evolved designs onto real materials, real constraints and real tooling."""

    def __init__(self, n_monte_carlo: int = 200, seed: int = 0, max_materials: int = 4, cad_generator: Any = None,
                 design_margin: float = 0.03):
        self.n_mc = n_monte_carlo
        self.design_margin = design_margin
        self.rng = np.random.default_rng(seed)
        self.max_materials = max_materials
        self.cad = cad_generator

    # -- materials ---------------------------------------------------------------
    def candidate_materials(self, domain: Any) -> List[Tuple[Material, float]]:
        out = []
        for m in MATERIALS.values():
            if not m.buildable or not domain.material_ok(m):
                continue
            out.append((m, float(domain.merit_index(m))))
        out.sort(key=lambda t: -t[1])
        return out

    # -- projection ----------------------------------------------------------------
    def project(self, domain: Any, u_alien: np.ndarray, ctx: Any, material: Material,
                frozen: Optional[np.ndarray] = None, start: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Any]:
        """Closest feasible design (normalised space) for ``material`` in ``ctx``.

        Constraints are first tightened to g_i <= -margin so the mapped design
        keeps a manufacturing/tolerance reserve (an optimiser otherwise parks
        designs exactly on constraint boundaries, where half of all as-built
        parts would fail); if no design meets the margin, plain feasibility is
        accepted."""
        for delta in (self.design_margin, 0.0):
            u, ev = self._project(domain, u_alien, ctx, material, delta, frozen, start)
            if ev.feasible and max(ev.constraints.values(), default=-1.0) <= -0.5 * delta:
                return u, ev
        return u, ev

    def _project(self, domain: Any, u_alien: np.ndarray, ctx: Any, material: Material, delta: float,
                 frozen: Optional[np.ndarray] = None, start: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Any]:
        int_mask = domain.integer_mask()
        u0 = np.clip(np.asarray(u_alien, float), 0, 1)
        best_u, best_ev, best_d = None, None, np.inf
        if frozen is not None:
            # repair mode: integers and catalogue values held at ``start``
            int_mask = int_mask | frozen
            options = [np.clip(np.asarray(start, float), 0, 1)]
        else:
            # integer neighbourhoods: alien value and +/- one step
            options = [u0.copy()]
            for i in np.where(int_mask)[0]:
                step = domain.int_step(i)
                for s in (-1, 1):
                    v = u0.copy()
                    v[i] = np.clip(v[i] + s * step, 0, 1)
                    options.append(v)
        cache: Dict[bytes, Any] = {}

        for start in options:
            fixed = start.copy()
            free = ~int_mask

            def full(z: np.ndarray) -> np.ndarray:
                u = fixed.copy()
                u[free] = z
                return u

            def ev(z: np.ndarray) -> Any:
                key = np.round(z, 10).tobytes()
                if key not in cache:
                    cache[key] = domain.evaluate(domain.decode(full(z)), ctx, material)
                return cache[key]

            def cons(z: np.ndarray) -> np.ndarray:
                g = np.array(list(ev(z).constraints.values()), float)
                return -np.nan_to_num(g, nan=10.0, posinf=10.0, neginf=-10.0) - delta

            z0 = start[free]
            res = minimize(lambda z: float(np.sum((z - u0[free]) ** 2)), z0, method="SLSQP",
                           bounds=[(0.0, 1.0)] * int(free.sum()),
                           constraints=[{"type": "ineq", "fun": cons}], options={"maxiter": 200, "ftol": 1e-9})
            u = full(np.clip(res.x, 0, 1))
            e = domain.evaluate(domain.decode(u), ctx, material)
            d = float(np.sum((u - u0) ** 2))
            if e.feasible and d < best_d:
                best_u, best_ev, best_d = u, e, d
        if best_u is None:
            # fall back: penalty search (Nelder-Mead on violation + distance)
            base = options[0]

            def pen(z: np.ndarray) -> float:
                u = base.copy()
                u[~int_mask] = np.clip(z, 0, 1)
                e = domain.evaluate(domain.decode(u), ctx, material)
                viol = sum(max(0.0, g) for g in e.constraints.values())
                return float(1e3 * viol + np.sum((u - u0) ** 2))
            res = minimize(pen, base[~int_mask], method="Nelder-Mead", options={"maxiter": 2000, "xatol": 1e-6})
            u = base.copy()
            u[~int_mask] = np.clip(res.x, 0, 1)
            best_u, best_ev = u, domain.evaluate(domain.decode(u), ctx, material)
        return best_u, best_ev

    # -- prediction -----------------------------------------------------------------
    def monte_carlo(self, domain: Any, params: Dict[str, float], ctx: Any, material: Material
                    ) -> Tuple[Dict[str, Dict[str, float]], float]:
        rows: Dict[str, List[float]] = {}
        feas = 0
        for _ in range(self.n_mc):
            mat = material.perturbed(self.rng)
            p = {k: (v if domain.is_integer(k) else v * float(self.rng.normal(1.0, 0.01))) for k, v in params.items()}
            e = domain.evaluate(p, ctx.perturbed(self.rng, 0.02), mat)
            feas += int(e.feasible)
            for k, v in e.metrics.items():
                if isinstance(v, (int, float)) and np.isfinite(v):
                    rows.setdefault(k, []).append(float(v))
        pred = {k: {"mean": float(np.mean(v)), "std": float(np.std(v)), "p05": float(np.percentile(v, 5)),
                    "p95": float(np.percentile(v, 95))} for k, v in rows.items()}
        return pred, feas / self.n_mc

    # -- main entry -------------------------------------------------------------------
    def map(self, design: Any, ctx: Any = None, out_dir: Optional[str] = None, name: Optional[str] = None
            ) -> MappedPrototype:
        """Map an :class:`~shiva_core.tech_evolver.EvolvedDesign` to the real world."""
        from .tech_evolver import DOMAINS, PhysicsContext

        domain = DOMAINS[design.domain]()
        ctx = ctx or PhysicsContext.earth()
        u_alien = np.asarray(design.u, float)
        alien_params = domain.decode(u_alien)
        ranking = []
        best = None
        for mat, merit in self.candidate_materials(domain)[: self.max_materials]:
            u_real, ev = self.project(domain, u_alien, ctx, mat)
            fidelity = float(1.0 - np.linalg.norm(u_real - u_alien) / math.sqrt(len(u_alien)))
            cost_term = 0.03 * math.log10(max(mat.cost_per_kg, 1.0) / 4.0)  # tie-break towards common stock
            score = (0.6 * ev.efficiency + 0.4 * fidelity - cost_term) if ev.feasible else -1.0 + 0.1 * fidelity
            ranking.append({"material": mat.key, "merit_index": merit, "feasible": ev.feasible,
                            "efficiency": ev.efficiency, "fidelity": fidelity, "score": score})
            if best is None or score > best[0]:
                best = (score, mat, u_real, ev, fidelity)
        _, mat, u_real, ev, fidelity = best
        params = domain.decode(u_real)
        notes: List[str] = []
        snapped_params = domain.snap(params)
        ev_snap = domain.evaluate(snapped_params, ctx, mat)
        catalog = np.array([p.name in domain.catalog_params for p in domain.params])
        if not (ev_snap.feasible and max(ev_snap.constraints.values(), default=-1.0) <= -0.5 * self.design_margin):
            # repair: hold the standard (catalogue) values and re-project the free geometry
            u_s, ev_s = self.project(domain, u_alien, ctx, mat, frozen=catalog, start=domain.encode(snapped_params))
            rep = domain.decode(u_s)
            for k in domain.catalog_params:
                rep[k] = snapped_params[k]  # exact catalogue value (decode(encode(x)) == x up to rounding)
            snapped_params, ev_snap = rep, domain.evaluate(rep, ctx, mat)
            if ev_snap.feasible:
                notes.append("Free dimensions were re-optimised after snapping to standard stock sizes.")
        snapped = bool(ev_snap.feasible) or not ev.feasible
        if snapped:
            params, ev = snapped_params, ev_snap
            fidelity = float(1.0 - np.linalg.norm(domain.encode(params) - u_alien) / math.sqrt(len(u_alien)))
        else:
            notes.append("No design using standard stock sizes satisfies every constraint; the unsnapped geometry"
                         " needs custom tooling or stock.")
        pred, reliability = self.monte_carlo(domain, params, ctx, mat)
        bom = domain.bom(params, mat)
        cost = float(sum(item.get("cost_usd", 0.0) for item in bom))
        mass = float(ev.metrics.get("mass_kg", sum(item.get("mass_kg", 0.0) for item in bom)))
        steps = domain.build_steps(params, mat) + [
            "Inspect all critical dimensions against the CAD model (tolerances as noted) before assembly.",
            f"Proof-test to {domain.proof_factor:.2g}x the design load/pressure/speed behind a barrier before use;"
            " record results.",
            "Planning values only: confirm material certificates and have a qualified engineer review"
            " load-bearing, rotating or pressurised parts.",
        ]
        if not ev.feasible:
            notes.append("No candidate material satisfied every Earth constraint; the closest design is reported"
                         " and must not be built as-is.")
        alien_eff = float(design.objectives.get("efficiency", float("nan")))
        proto = MappedPrototype(
            domain=design.domain, design_id=design.design_id, material=mat.key, material_name=mat.name,
            alien_params=alien_params, real_params=params, fidelity=fidelity, alien_efficiency=alien_eff,
            real_efficiency=float(ev.efficiency), feasible=bool(ev.feasible), snapped=bool(snapped),
            constraint_values={k: float(v) for k, v in ev.constraints.items()}, prediction=pred,
            reliability=float(reliability), bom=bom, cost_usd=cost, mass_kg=mass, build_instructions=steps,
            material_ranking=ranking, notes=notes)
        if out_dir is not None and self.cad is not None:
            stem = name or f"{design.domain}_{design.design_id}"
            proto.cad = self.cad.generate(design.domain, params, material=mat, name=stem, out_dir=out_dir)
        return proto
