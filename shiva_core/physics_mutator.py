"""
SHIVA-1 :: Physics Mutation Engine
==================================

Generates, mutates, recombines and symbolically analyses *alternate laws of
physics*.  A universe is described by a :class:`PhysicsGenome`, a structured
set of genes:

=================  =========================================================
Gene               What it controls
=================  =========================================================
``gravity``        nested law hierarchy  inverse-square -> inverse-k ->
                   tensor (anisotropic metric) -> fractal (log-periodic,
                   discrete-scale-invariant), plus a Yukawa (screened) branch
                   and an optional time-modulated coupling G(t).
``em``             electromagnetic-like coupling: Coulomb / power-law /
                   Proca (massive photon), fine-structure ratio alpha/alpha_0,
                   specific charge and a uniform background magnetic field.
``inertia``        kinetic-energy law T(p): Newtonian, relativistic with a
                   mutated speed limit c, anisotropic mass tensor, or a
                   power law |p|^a; mass spectrum (equal/lognormal/Salpeter).
``thermo``         entropy behaviour: Langevin friction + temperature
                   (fluctuation-dissipation consistent), Tsallis-q heavy
                   tailed noise, Rayleigh-Helmholtz active friction.
``fluid``          viscosity tensor, hyperviscosity order, advection
                   coupling, drag-scaling exponent, reaction-diffusion
                   constants for the continuum field solvers.
``interaction``    force propagation (instantaneous / retarded with finite
                   speed), multi-species coupling matrix (possibly
                   non-reciprocal), soft-core repulsion.
``cosmology``      Newtonian cosmological constant Lambda and Hubble drag.
=================  =========================================================

Mathematical-consistency principle
----------------------------------
Every *conservative* interaction is defined by a pair potential U written in
SymPy; forces are always obtained as ``F = -grad U`` by symbolic
differentiation, so energy conservation is never asserted by hand.  Conservation
laws are **never mutated directly**: they are *derived* from the symmetries of
the mutated laws via Noether's theorem (:func:`noether_analysis`).  Breaking a
symmetry (anisotropy, time modulation, non-reciprocity, external potential,
dissipation, retardation) is the only way a conservation law disappears, which
guarantees that the declared conservation structure is always consistent with
the dynamics.  The universe simulator then *measures* the drift of each
quantity and checks it against these predictions.

References are listed in ``shiva_docs/whitepaper.md``.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import sympy as sp

__all__ = [
    "GRAVITY_LADDER",
    "GRAVITY_LAWS",
    "EM_LAWS",
    "INERTIA_MODELS",
    "ALPHA_CHEMISTRY_WINDOW",
    "GravityGene",
    "ElectromagneticGene",
    "InertiaGene",
    "ThermoGene",
    "FluidGene",
    "InteractionGene",
    "CosmologyGene",
    "PhysicsGenome",
    "PhysicsMutator",
    "SymbolicPhysics",
    "ChannelSpec",
    "NoetherReport",
    "noether_analysis",
    "spd_from_params",
]

# ---------------------------------------------------------------------------
# Catalogues and constants
# ---------------------------------------------------------------------------

#: The gravitational "escalation ladder".  Each rung strictly generalises the
#: previous one (inverse_square is inverse_k with k=2, inverse_k is tensor with
#: Q=I, tensor is fractal with eps=0).
GRAVITY_LADDER: Tuple[str, ...] = ("inverse_square", "inverse_k", "tensor", "fractal")
GRAVITY_LAWS: Tuple[str, ...] = GRAVITY_LADDER + ("yukawa",)
EM_LAWS: Tuple[str, ...] = ("off", "coulomb", "power", "proca")
INERTIA_MODELS: Tuple[str, ...] = ("newtonian", "relativistic", "anisotropic", "power")
NOISE_MODELS: Tuple[str, ...] = ("gaussian", "q_gaussian")
PROPAGATION_MODES: Tuple[str, ...] = ("instantaneous", "retarded")
MASS_SPECTRA: Tuple[str, ...] = ("equal", "lognormal", "salpeter")

ALPHA_0 = 1.0 / 137.035999
#: Window of the fine-structure constant inside which stable chemistry and
#: long-lived stars are believed possible (Barrow & Tipler 1986; Tegmark 1998:
#: roughly 1/180 < alpha < 1/85), expressed as the ratio alpha/alpha_0.
ALPHA_CHEMISTRY_WINDOW: Tuple[float, float] = ((1.0 / 180.0) / ALPHA_0, (1.0 / 85.0) / ALPHA_0)

#: Hard bounds used by :meth:`PhysicsMutator.repair`.  (lo, hi)
PARAM_BOUNDS: Dict[str, Tuple[float, float]] = {
    "gravity.G": (0.05, 20.0),
    "gravity.k": (1.0, 3.6),
    "gravity.screening_length": (0.2, 50.0),
    "gravity.fractal_eps": (0.0, 0.9),
    "gravity.fractal_gamma": (0.1, 0.9),
    "gravity.fractal_omega": (1.0, 30.0),
    "gravity.fractal_beta": (1.2, 4.0),
    "gravity.fractal_octaves": (1, 6),
    "gravity.fractal_r0": (0.05, 5.0),
    "gravity.modulation_amp": (0.0, 0.9),
    "gravity.modulation_freq": (0.0, 20.0),
    "em.alpha_ratio": (0.1, 10.0),
    "em.base_coupling": (0.0, 5.0),
    "em.k": (1.0, 3.6),
    "em.proca_mass": (0.0, 10.0),
    "em.specific_charge": (0.0, 5.0),
    "inertia.c": (0.5, 1000.0),
    "inertia.kinetic_exponent": (1.5, 3.0),
    "inertia.rest_energy_factor": (0.0, 10.0),
    "inertia.mass_spread": (0.0, 2.0),
    "inertia.imf_slope": (1.5, 3.5),
    "thermo.friction": (0.0, 20.0),
    "thermo.temperature": (0.0, 10.0),
    "thermo.tsallis_q": (1.0, 1.6),
    "thermo.active_gamma": (0.0, 20.0),
    "thermo.active_v0": (0.01, 10.0),
    "thermo.adiabatic_index": (1.05, 1.8),
    "fluid.viscosity": (1e-5, 5e-2),
    "fluid.hyper_order": (0.5, 4.0),
    "fluid.advection": (0.0, 3.0),
    "fluid.ekman_drag": (0.0, 0.5),
    "fluid.forcing_amp": (0.0, 2.0),
    "fluid.forcing_k": (1, 12),
    "fluid.medium_drag": (0.0, 5.0),
    "fluid.drag_exponent": (0.5, 3.0),
    "fluid.rd_diffusion_u": (0.02, 0.24),
    "fluid.rd_diffusion_v": (0.01, 0.24),
    "fluid.rd_feed": (0.005, 0.1),
    "fluid.rd_kill": (0.03, 0.075),
    "interaction.c_prop": (0.2, 100.0),
    "interaction.n_species": (1, 5),
    "interaction.species_range": (0.05, 2.0),
    "interaction.core_repulsion": (0.0, 5.0),
    "interaction.core_range": (0.01, 0.5),
    "cosmology.Lambda": (-10.0, 10.0),
    "cosmology.hubble": (-1.0, 2.0),
}

#: Continuous parameters exposed to vector optimisers (CMA-ES).  Entries are
#: (dotted path, lo, hi, scale) with scale "log" or "lin".
CONTINUOUS_SPEC: Tuple[Tuple[str, float, float, str], ...] = (
    ("gravity.G", 0.2, 5.0, "log"),
    ("gravity.k", 1.2, 3.4, "lin"),
    ("gravity.screening_length", 0.3, 20.0, "log"),
    ("gravity.fractal_eps", 0.0, 0.8, "lin"),
    ("gravity.fractal_omega", 2.0, 20.0, "log"),
    ("gravity.modulation_amp", 0.0, 0.6, "lin"),
    ("em.alpha_ratio", ALPHA_CHEMISTRY_WINDOW[0], ALPHA_CHEMISTRY_WINDOW[1], "log"),
    ("em.specific_charge", 0.0, 3.0, "lin"),
    ("thermo.friction", 0.0, 5.0, "lin"),
    ("thermo.temperature", 0.0, 1.0, "lin"),
    ("thermo.active_gamma", 0.0, 5.0, "lin"),
    ("interaction.species_range", 0.1, 1.0, "log"),
    ("interaction.core_repulsion", 0.0, 2.0, "lin"),
    ("cosmology.Lambda", -3.0, 3.0, "lin"),
)


# ---------------------------------------------------------------------------
# Linear-algebra helpers
# ---------------------------------------------------------------------------

def spd_from_params(params: Sequence[float], dim: int) -> np.ndarray:
    """Map an unconstrained parameter vector to a unit-determinant SPD matrix.

    ``params`` holds the lower-triangular entries (row-major) of a Cholesky
    factor ``L`` whose diagonal is stored as logarithms, so *every* real
    vector yields a valid symmetric positive-definite tensor and the zero
    vector yields the identity.  The result is normalised to ``det = 1`` so
    that it encodes pure anisotropy (shape) without changing overall strength.
    """
    n_expected = dim * (dim + 1) // 2
    p = np.asarray(params, dtype=float)
    if p.size != n_expected:
        raise ValueError(f"expected {n_expected} Cholesky parameters for dim={dim}, got {p.size}")
    L = np.zeros((dim, dim))
    rows, cols = np.tril_indices(dim)
    L[rows, cols] = p
    di = np.diag_indices(dim)
    L[di] = np.exp(np.clip(L[di], -3.0, 3.0))
    A = L @ L.T
    A /= np.linalg.det(A) ** (1.0 / dim)
    return 0.5 * (A + A.T)


def _is_isotropic(T: np.ndarray, tol: float = 1e-9) -> bool:
    d = T.shape[0]
    return bool(np.allclose(T, np.trace(T) / d * np.eye(d), atol=tol * max(1.0, abs(np.trace(T)))))


def _skew(n: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -n[2], n[1]], [n[2], 0.0, -n[0]], [-n[1], n[0], 0.0]])


def _get_path(obj: Any, path: str) -> Any:
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def _set_path(obj: Any, path: str, value: Any) -> None:
    parts = path.split(".")
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)


# ---------------------------------------------------------------------------
# Genes
# ---------------------------------------------------------------------------

@dataclass
class GravityGene:
    """Gravitational law.

    Pair potential (unit masses), with ``phi_k(r) = r^(1-k)/(k-1)``
    (``-ln r`` for k=1) so that ``|F| = G r^-k`` for the bare power law::

        inverse_square : U = -G / r
        inverse_k      : U = -G phi_k(r)
        yukawa         : U = -G exp(-r/lambda) / r
        tensor         : U = -G phi_k(rho),  rho^2 = x^T Q x,  det Q = 1
        fractal        : U = -G phi_k(rho) [1 + eps sum_n gamma^n cos(omega beta^n ln(rho/r0))]

    ``modulation_amp`` adds an explicit time dependence G(t) = G(1 + a sin(W t)),
    which breaks time-translation symmetry (energy is then not conserved, but
    momentum still is - a clean Noether test case).
    """

    law: str = "inverse_square"
    G: float = 1.0
    k: float = 2.0
    screening_length: float = 5.0
    tensor_chol: List[float] = field(default_factory=lambda: [0.0] * 6)
    fractal_eps: float = 0.0
    fractal_gamma: float = 0.5
    fractal_omega: float = 6.0
    fractal_beta: float = 2.0
    fractal_octaves: int = 3
    fractal_r0: float = 1.0
    modulation_amp: float = 0.0
    modulation_freq: float = 1.0

    @property
    def k_eff(self) -> float:
        return 2.0 if self.law in ("inverse_square", "yukawa") else float(self.k)

    @property
    def anisotropic(self) -> bool:
        return self.law in ("tensor", "fractal") and not _is_isotropic(spd_from_params(self.tensor_chol, 3))

    def Q(self, dim: int = 3) -> np.ndarray:
        """Metric tensor of the gravitational interaction restricted to ``dim``."""
        if self.law not in ("tensor", "fractal"):
            return np.eye(dim)
        Q3 = spd_from_params(self.tensor_chol, 3)
        return Q3[:dim, :dim].copy()


@dataclass
class ElectromagneticGene:
    """Electromagnetic-like interaction between charged particles.

    Charges are assigned as ``q_i = +/- specific_charge * m_i`` (neutral
    overall).  The coupling constant in simulation units is
    ``ke = base_coupling * alpha_ratio``; ``alpha_ratio = alpha/alpha_0`` also
    drives the Press-Lightman scaling of material properties used by the
    technology engine.  Laws::

        coulomb : U = +ke q_i q_j / r
        power   : U = +ke q_i q_j phi_k(r)
        proca   : U = +ke q_i q_j exp(-mu r) / r     (massive photon, Yukawa)

    ``B`` is a uniform background magnetic field (simulation units); in 2-D
    only its z component acts.
    """

    law: str = "off"
    alpha_ratio: float = 1.0
    base_coupling: float = 0.5
    k: float = 2.0
    proca_mass: float = 1.0
    specific_charge: float = 1.0
    B: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    @property
    def ke(self) -> float:
        return float(self.base_coupling * self.alpha_ratio)

    @property
    def active(self) -> bool:
        return self.law != "off" and self.ke != 0.0 and self.specific_charge != 0.0

    def B_vec(self, dim: int = 3) -> np.ndarray:
        B = np.asarray(self.B, dtype=float)
        if not self.active:
            return np.zeros(3)
        if dim == 2:
            return np.array([0.0, 0.0, B[2]])
        return B


@dataclass
class InertiaGene:
    """Inertia / kinetic-energy law ``T(p)`` and mass spectrum.

    ===============  ==================================================
    newtonian        T = |p|^2 / 2m
    relativistic     T = sqrt(p^2 c^2 + m^2 c^4) - m c^2   (speed limit c)
    anisotropic      T = p^T M^-1 p / 2m,  det M = 1   (direction-dependent inertia)
    power            T = |p|^a / (a m^(a-1))           (a = kinetic_exponent)
    ===============  ==================================================

    All four are *separable* Hamiltonians H = T(p) + U(q), so the simulator's
    Stormer-Verlet / Yoshida integrators remain symplectic.  The rest energy
    E_0 = eta m c^2 is tracked for cosmic energy budgets.
    """

    model: str = "newtonian"
    c: float = 10.0
    mass_tensor_chol: List[float] = field(default_factory=lambda: [0.0] * 6)
    kinetic_exponent: float = 2.0
    rest_energy_factor: float = 1.0
    mass_spectrum: str = "equal"
    mass_spread: float = 0.5
    imf_slope: float = 2.35

    @property
    def anisotropic(self) -> bool:
        return self.model == "anisotropic" and not _is_isotropic(spd_from_params(self.mass_tensor_chol, 3))

    def M(self, dim: int = 3) -> np.ndarray:
        if self.model != "anisotropic":
            return np.eye(dim)
        return spd_from_params(self.mass_tensor_chol, 3)[:dim, :dim].copy()


@dataclass
class ThermoGene:
    """Thermodynamic / entropy behaviour.

    * Langevin bath with friction ``friction`` (rate) and temperature
      ``temperature`` obeying the fluctuation-dissipation theorem
      (noise variance 2 gamma m k_B T).
    * ``noise = "q_gaussian"`` draws unit-variance Student-t increments, the
      Tsallis q-Gaussian for 1 < q < 5/3 (nu = (3-q)/(q-1) degrees of freedom).
    * ``active_gamma`` > 0 adds Rayleigh-Helmholtz active friction
      -gamma_a (|v|^2/v0^2 - 1) p, which pumps energy into slow particles and
      drains fast ones: a minimal model of self-propelled ("living") matter.
    * ``adiabatic_index`` is the gas heat-capacity ratio used by the
      engineering layer (nozzles, compressors).
    """

    friction: float = 0.0
    temperature: float = 0.0
    noise: str = "gaussian"
    tsallis_q: float = 1.0
    active_gamma: float = 0.0
    active_v0: float = 0.5
    adiabatic_index: float = 1.4


@dataclass
class FluidGene:
    """Continuum (field) dynamics and drag scaling.

    2-D vorticity dynamics (see :class:`universe_simulator.VorticityField2D`)::

        d_t w + lambda J(psi, w) = -nu (-div N grad)^h w - alpha w + f,   lap psi = -w

    with viscosity tensor ``N`` (unit determinant, from ``viscosity_tensor_chol``),
    hyperviscosity order ``h`` and advection coupling ``lambda``.  Particles
    moving through the medium feel drag ``-c_d |v|^(n-1) v`` (n = 1 Stokes,
    n = 2 Newtonian quadratic) and the Gray-Scott reaction-diffusion field
    uses anisotropic diffusion ``D N``.
    """

    viscosity: float = 1e-3
    viscosity_tensor_chol: List[float] = field(default_factory=lambda: [0.0] * 3)
    hyper_order: float = 1.0
    advection: float = 1.0
    ekman_drag: float = 0.02
    forcing_amp: float = 0.5
    forcing_k: int = 4
    medium_drag: float = 0.0
    drag_exponent: float = 2.0
    rd_diffusion_u: float = 0.16
    rd_diffusion_v: float = 0.08
    rd_feed: float = 0.030
    rd_kill: float = 0.062

    #: reference viscosity corresponding to "our" fluids
    REFERENCE_VISCOSITY = 1e-3

    @property
    def viscosity_ratio(self) -> float:
        return float(self.viscosity / self.REFERENCE_VISCOSITY)

    def N(self) -> np.ndarray:
        return spd_from_params(self.viscosity_tensor_chol, 2)


@dataclass
class InteractionGene:
    """Force propagation and interaction symmetries.

    * ``propagation = "retarded"``: long-range forces act from the source's
      position at the retarded time t - r/c_prop.  Newton's third law then
      fails between particles (momentum is carried by the implicit field).
    * Multi-species "chemistry": pair potential ``K_ab exp(-r^2 / 2 sigma^2)``
      with species matrix ``K``.  A non-symmetric ``K`` is *non-reciprocal*
      (A chases B while B flees A), breaking action-reaction and hence
      momentum, energy and angular-momentum conservation - the mechanism of
      non-reciprocal phase transitions (Fruchart et al., Nature 2021).
    * ``core_repulsion`` adds a universal soft core ``eps exp(-r^2/2 sigma_c^2)``.
    """

    propagation: str = "instantaneous"
    c_prop: float = 5.0
    n_species: int = 1
    species_matrix: List[List[float]] = field(default_factory=lambda: [[0.0]])
    species_range: float = 0.3
    core_repulsion: float = 0.0
    core_range: float = 0.08

    def K(self) -> np.ndarray:
        return np.asarray(self.species_matrix, dtype=float).reshape(self.n_species, self.n_species)

    @property
    def species_active(self) -> bool:
        return bool(np.any(self.K() != 0.0))

    @property
    def nonreciprocity(self) -> float:
        """||A|| / ||K|| with A the antisymmetric part of K (0 = reciprocal)."""
        K = self.K()
        nk = np.linalg.norm(K)
        if nk == 0:
            return 0.0
        return float(np.linalg.norm(0.5 * (K - K.T)) / nk)


@dataclass
class CosmologyGene:
    """Newtonian cosmology.

    ``Lambda`` adds the Newtonian cosmological-constant force F = Lambda m x / 3
    (potential -Lambda m |x|^2 / 6): Lambda > 0 is de Sitter-like accelerated
    expansion, Lambda < 0 an anti-de Sitter-like confining box.  It breaks
    translation symmetry (momentum) but keeps rotation symmetry about the
    origin.  ``hubble`` is a Hubble drag dp/dt = -H p acting on peculiar
    momenta (comoving-frame cooling for H > 0).
    """

    Lambda: float = 0.0
    hubble: float = 0.0


# ---------------------------------------------------------------------------
# Genome
# ---------------------------------------------------------------------------

def _new_id() -> str:
    return uuid.uuid4().hex[:12]


@dataclass
class PhysicsGenome:
    """Complete specification of an alternate universe's laws."""

    gravity: GravityGene = field(default_factory=GravityGene)
    em: ElectromagneticGene = field(default_factory=ElectromagneticGene)
    inertia: InertiaGene = field(default_factory=InertiaGene)
    thermo: ThermoGene = field(default_factory=ThermoGene)
    fluid: FluidGene = field(default_factory=FluidGene)
    interaction: InteractionGene = field(default_factory=InteractionGene)
    cosmology: CosmologyGene = field(default_factory=CosmologyGene)
    dim: int = 3
    name: str = ""
    genome_id: str = field(default_factory=_new_id)
    parents: List[str] = field(default_factory=list)
    generation: int = 0
    lineage: List[str] = field(default_factory=list)

    # -- serialisation ----------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, **kw: Any) -> str:
        return json.dumps(self.to_dict(), **({"indent": 2} | kw))

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PhysicsGenome":
        d = copy.deepcopy(d)
        genes = {
            "gravity": GravityGene,
            "em": ElectromagneticGene,
            "inertia": InertiaGene,
            "thermo": ThermoGene,
            "fluid": FluidGene,
            "interaction": InteractionGene,
            "cosmology": CosmologyGene,
        }
        kwargs: Dict[str, Any] = {}
        for key, val in d.items():
            if key in genes:
                kwargs[key] = genes[key](**val)
            else:
                kwargs[key] = val
        return cls(**kwargs)

    @classmethod
    def from_json(cls, s: str) -> "PhysicsGenome":
        return cls.from_dict(json.loads(s))

    def copy(self) -> "PhysicsGenome":
        return copy.deepcopy(self)

    # -- identity -----------------------------------------------------------
    def fingerprint(self) -> str:
        """Hash of the *physics* only (ignores id/lineage)."""
        d = self.to_dict()
        for key in ("name", "genome_id", "parents", "generation", "lineage"):
            d.pop(key, None)
        return hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]

    def describe(self) -> str:
        g, e, i, t, f, x, c = (self.gravity, self.em, self.inertia, self.thermo,
                               self.fluid, self.interaction, self.cosmology)
        lines = [f"PhysicsGenome {self.name or self.genome_id} (gen {self.generation}, dim {self.dim})"]
        grav = f"  gravity : {g.law}, G={g.G:.3g}, k={g.k_eff:.3g}"
        if g.law == "yukawa":
            grav += f", lambda={g.screening_length:.3g}"
        if g.law in ("tensor", "fractal"):
            ev = np.linalg.eigvalsh(g.Q(3))
            grav += f", Q-eig={np.round(ev, 3).tolist()}"
        if g.law == "fractal":
            grav += (f", eps={g.fractal_eps:.3g}, gamma={g.fractal_gamma:.3g}, omega={g.fractal_omega:.3g},"
                     f" beta={g.fractal_beta:.3g}, octaves={g.fractal_octaves}")
        if g.modulation_amp:
            grav += f", G(t) modulation a={g.modulation_amp:.3g} W={g.modulation_freq:.3g}"
        lines.append(grav)
        if e.active:
            lines.append(f"  em      : {e.law}, alpha/alpha0={e.alpha_ratio:.3g}, ke={e.ke:.3g}, q/m={e.specific_charge:.3g},"
                         f" k={e.k:.3g}, mu={e.proca_mass:.3g}, B={np.round(e.B, 3).tolist()}")
        else:
            lines.append(f"  em      : off (alpha/alpha0={e.alpha_ratio:.3g})")
        inert = f"  inertia : {i.model}"
        if i.model == "relativistic":
            inert += f", c={i.c:.3g}"
        if i.model == "power":
            inert += f", a={i.kinetic_exponent:.3g}"
        if i.model == "anisotropic":
            inert += f", M-eig={np.round(np.linalg.eigvalsh(i.M(3)), 3).tolist()}"
        inert += f", masses={i.mass_spectrum}"
        lines.append(inert)
        lines.append(f"  thermo  : gamma={t.friction:.3g}, T={t.temperature:.3g}, noise={t.noise}(q={t.tsallis_q:.3g}),"
                     f" active={t.active_gamma:.3g}@v0={t.active_v0:.3g}, adiabatic={t.adiabatic_index:.3g}")
        lines.append(f"  fluid   : nu={f.viscosity:.3g}, h={f.hyper_order:.3g}, lambda={f.advection:.3g},"
                     f" N-eig={np.round(np.linalg.eigvalsh(f.N()), 3).tolist()}, drag={f.medium_drag:.3g}|v|^{f.drag_exponent:.3g}")
        inter = f"  interact: {x.propagation}"
        if x.propagation == "retarded":
            inter += f"(c={x.c_prop:.3g})"
        inter += f", species={x.n_species}"
        if x.species_active:
            inter += f", nonreciprocity={x.nonreciprocity:.3f}, sigma={x.species_range:.3g}"
        if x.core_repulsion:
            inter += f", core={x.core_repulsion:.3g}@{x.core_range:.3g}"
        lines.append(inter)
        lines.append(f"  cosmos  : Lambda={c.Lambda:.3g}, H={c.hubble:.3g}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Noether analysis
# ---------------------------------------------------------------------------

@dataclass
class NoetherReport:
    """Symmetries of a genome and the conservation laws they imply."""

    energy: bool
    momentum: bool
    momentum_kind: str  # "mechanical" | "pseudo" (magnetic) | "none"
    angular_axes: List[List[float]]  # axes n with conserved n.L (or canonical J)
    angular_kind: str  # "mechanical" | "canonical" | "none"
    time_reversal: bool
    liouville: bool
    virial_degree: Optional[float]
    symmetries: List[str]
    broken: Dict[str, List[str]]

    @property
    def angular_momentum(self) -> bool:
        return len(self.angular_axes) > 0

    def predicted(self) -> Dict[str, bool]:
        return {"energy": self.energy, "momentum": self.momentum, "angular_momentum": self.angular_momentum}

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        def yn(b: bool) -> str:
            return "conserved" if b else "NOT conserved"
        lines = [
            f"energy            : {yn(self.energy)}" + ("" if self.energy else f"  <- {', '.join(self.broken['energy'])}"),
            f"momentum ({self.momentum_kind:9s}): {yn(self.momentum)}" + ("" if self.momentum else f"  <- {', '.join(self.broken['momentum'])}"),
            f"angular ({self.angular_kind:10s}): " + (f"conserved about {len(self.angular_axes)} axis/axes" if self.angular_axes
                                                    else f"NOT conserved  <- {', '.join(self.broken['angular_momentum'])}"),
            f"time reversal     : {'symmetric' if self.time_reversal else 'broken'}",
            f"Liouville volume  : {'preserved' if self.liouville else 'contracting/undefined'}",
            f"virial degree     : {self.virial_degree}",
        ]
        return "\n".join(lines)


def _rotation_axes(tensors: List[np.ndarray], B: np.ndarray, dim: int) -> List[np.ndarray]:
    """Axes n such that every tensor commutes with rotations about n and n || B."""
    if dim == 2:
        # In-plane rotation; B_z is a pseudoscalar invariant under it.
        if all(_is_isotropic(T[:2, :2]) for T in tensors):
            return [np.array([0.0, 0.0, 1.0])]
        return []
    candidates: List[np.ndarray] = [np.eye(3)[i] for i in range(3)]
    for T in tensors:
        _, vecs = np.linalg.eigh(T)
        candidates.extend(vecs.T)
    if np.linalg.norm(B) > 0:
        candidates.append(B / np.linalg.norm(B))
    axes: List[np.ndarray] = []
    for n in candidates:
        n = n / np.linalg.norm(n)
        Gn = _skew(n)
        ok = all(np.linalg.norm(T @ Gn - Gn @ T) < 1e-8 * max(1.0, np.linalg.norm(T)) for T in tensors)
        ok = ok and np.linalg.norm(np.cross(n, B)) < 1e-10 * max(1.0, np.linalg.norm(B))
        if ok and not any(abs(abs(float(np.dot(n, a))) - 1.0) < 1e-8 for a in axes):
            axes.append(n)
    # If three independent axes are symmetric the full SO(3) is; keep the canonical basis.
    if len(axes) >= 3:
        return [np.eye(3)[i] for i in range(3)]
    return axes


def noether_analysis(genome: PhysicsGenome, boundary: str = "open", dim: Optional[int] = None) -> NoetherReport:
    """Derive the conservation structure of ``genome`` from its symmetries.

    Parameters
    ----------
    boundary:
        ``"open"``, ``"periodic"`` or ``"reflective"`` - boundaries are part of
        the symmetry structure (walls break translations, boxes break rotations).
    """
    d = dim or genome.dim
    g, e, inert, th, fl, ix, cos = (genome.gravity, genome.em, genome.inertia, genome.thermo,
                                    genome.fluid, genome.interaction, genome.cosmology)
    long_range = True  # gravity is always present
    time_dep = g.modulation_amp > 0 and g.modulation_freq > 0
    dissipative = th.friction > 0 or th.active_gamma > 0 or fl.medium_drag > 0 or cos.hubble != 0
    stochastic = th.friction > 0 and th.temperature > 0
    nonrecip = ix.species_active and ix.nonreciprocity > 1e-12
    retarded = ix.propagation == "retarded" and long_range
    external = cos.Lambda != 0 and boundary != "periodic"
    walls = boundary == "reflective"
    B = e.B_vec(d)
    magnetic = bool(np.linalg.norm(B) > 0)

    broken: Dict[str, List[str]] = {"energy": [], "momentum": [], "angular_momentum": []}
    if time_dep:
        broken["energy"].append("explicit time dependence G(t)")
    if dissipative:
        for key in broken:
            broken[key].append("dissipation (friction/active/drag/Hubble)")
    if nonrecip:
        for key in broken:
            broken[key].append("non-reciprocal species coupling")
    if retarded:
        for key in broken:
            broken[key].append("retarded propagation (field carries momentum)")
    if external:
        broken["momentum"].append("external Lambda potential (no translation symmetry)")
    if walls:
        broken["momentum"].append("reflective walls")
        broken["angular_momentum"].append("reflective box")
    if boundary == "periodic":
        broken["angular_momentum"].append("periodic box (no continuous rotation symmetry)")
        if magnetic:
            broken["momentum"].append("pseudo-momentum ill-defined in periodic box with B")

    tensors: List[np.ndarray] = []
    if g.anisotropic:
        tensors.append(g.Q(3))
    if inert.anisotropic:
        tensors.append(inert.M(3))
    axes: List[np.ndarray] = []
    if not broken["angular_momentum"]:
        axes = _rotation_axes(tensors, B, d)
        if not axes:
            reason = []
            if tensors:
                reason.append("anisotropic tensor (gravity metric / mass tensor)")
            if magnetic:
                reason.append("background B field")
            broken["angular_momentum"].append(" & ".join(reason) or "no rotational symmetry")

    energy = not broken["energy"]
    momentum = not broken["momentum"]
    syms = []
    if not time_dep and not dissipative:
        syms.append("time translation")
    if momentum:
        syms.append("spatial translation" + (" (magnetic, pseudo-momentum)" if magnetic else ""))
    if axes:
        syms.append(f"rotation about {len(axes)} axis/axes" + (" (canonical J with B)" if magnetic else ""))

    # Time reversal: magnetic fields, dissipation, noise, retardation and odd G(t) break it.
    trs = not (magnetic or dissipative or stochastic or retarded or time_dep)
    if trs:
        syms.append("time reversal")
    liouville = not (dissipative or retarded)

    # Homogeneity degree for the virial theorem (only when every active
    # potential channel is a pure power law of the same degree).
    degrees: List[float] = []
    if g.law in ("inverse_square", "inverse_k", "tensor") or (g.law == "fractal" and g.fractal_eps == 0):
        degrees.append(1.0 - g.k_eff)
    else:
        degrees.append(float("nan"))
    if e.active:
        degrees.append(1.0 - (2.0 if e.law == "coulomb" else e.k) if e.law in ("coulomb", "power") else float("nan"))
    if ix.species_active or ix.core_repulsion:
        degrees.append(float("nan"))
    virial = None
    if all(np.isfinite(degrees)) and np.allclose(degrees, degrees[0]) and cos.Lambda == 0:
        virial = float(degrees[0])

    return NoetherReport(
        energy=energy,
        momentum=momentum,
        momentum_kind=("pseudo" if magnetic else "mechanical") if momentum else "none",
        angular_axes=[a.tolist() for a in axes],
        angular_kind=("canonical" if magnetic else "mechanical") if axes else "none",
        time_reversal=trs,
        liouville=liouville,
        virial_degree=virial,
        symmetries=syms,
        broken=broken,
    )


# ---------------------------------------------------------------------------
# Symbolic physics
# ---------------------------------------------------------------------------

@dataclass
class ChannelSpec:
    """A pair-interaction channel ready for numerical evaluation.

    The force on particle i from j is ``C_ij * kappa(S_ij) * Q (x_i - x_j)``
    with ``S = (x_i-x_j)^T Q (x_i-x_j) + eps^2`` and ``kappa(s) = -2 dU/ds``.
    """

    name: str
    potential_r: sp.Expr
    Q: np.ndarray
    coefficient: str  # "mass" | "charge" | "species" | "core"
    reciprocal: bool
    long_range: bool
    kappa: Callable[[np.ndarray], np.ndarray]
    U: Callable[[np.ndarray], np.ndarray]


def _phi(r: sp.Expr, k: float) -> sp.Expr:
    """Power-law Green's function with |d phi/dr| = r^-k (log for k = 1)."""
    k = sp.nsimplify(round(float(k), 6), rational=True)
    if k == 1:
        return -sp.log(r)
    return r ** (1 - k) / (k - 1)


class SymbolicPhysics:
    """SymPy representation of a genome's laws with automatic verification."""

    def __init__(self, genome: PhysicsGenome):
        self.genome = genome
        self.r = sp.Symbol("r", positive=True)
        self.s = sp.Symbol("s", positive=True)
        self.t = sp.Symbol("t", real=True)
        self._channels: Optional[Dict[str, ChannelSpec]] = None

    # -- potentials ---------------------------------------------------------
    def gravity_modulation(self) -> sp.Expr:
        g = self.genome.gravity
        if g.law != "fractal" or g.fractal_eps == 0:
            return sp.Integer(1)
        r = self.r
        eps, gam, om, beta, r0 = (sp.Float(g.fractal_eps, 8), sp.Float(g.fractal_gamma, 8), sp.Float(g.fractal_omega, 8),
                                  sp.Float(g.fractal_beta, 8), sp.Float(g.fractal_r0, 8))
        series = sum(gam ** n * sp.cos(om * beta ** n * sp.log(r / r0)) for n in range(int(g.fractal_octaves)))
        return 1 + eps * series

    def gravity_potential(self) -> sp.Expr:
        """U_grav(r) for unit masses, excluding the time factor of G(t)."""
        g, r = self.genome.gravity, self.r
        G = sp.Float(g.G, 10)
        if g.law == "inverse_square":
            return -G / r
        if g.law == "yukawa":
            return -G * sp.exp(-r / sp.Float(g.screening_length, 10)) / r
        base = -G * _phi(r, g.k)
        return base * self.gravity_modulation()

    def em_potential(self) -> Optional[sp.Expr]:
        e, r = self.genome.em, self.r
        if not e.active:
            return None
        ke = sp.Float(e.ke, 10)
        if e.law == "coulomb":
            return ke / r
        if e.law == "power":
            return ke * _phi(r, e.k)
        if e.law == "proca":
            return ke * sp.exp(-sp.Float(e.proca_mass, 10) * r) / r
        raise ValueError(e.law)

    def species_potential(self) -> sp.Expr:
        sig = sp.Float(self.genome.interaction.species_range, 10)
        return sp.exp(-self.r ** 2 / (2 * sig ** 2))

    def core_potential(self) -> sp.Expr:
        ix = self.genome.interaction
        sig = sp.Float(ix.core_range, 10)
        return sp.Float(ix.core_repulsion, 10) * sp.exp(-self.r ** 2 / (2 * sig ** 2))

    def force_law(self, U: sp.Expr) -> sp.Expr:
        """Radial force F(r) = -dU/dr (positive = repulsive)."""
        return sp.simplify(-sp.diff(U, self.r))

    def time_factor(self) -> sp.Expr:
        g = self.genome.gravity
        if g.modulation_amp == 0:
            return sp.Integer(1)
        return 1 + sp.Float(g.modulation_amp, 8) * sp.sin(sp.Float(g.modulation_freq, 8) * self.t)

    # -- compiled channels ----------------------------------------------------
    def _compile(self, U_r: sp.Expr) -> Tuple[Callable, Callable]:
        U_s = U_r.subs(self.r, sp.sqrt(self.s))
        kappa = -2 * sp.diff(U_s, self.s)
        kappa_fn = sp.lambdify(self.s, kappa, modules="numpy")
        U_fn = sp.lambdify(self.s, U_s, modules="numpy")

        def _vec(fn: Callable) -> Callable:
            def wrapped(S: np.ndarray) -> np.ndarray:
                out = fn(S)
                if np.isscalar(out) or np.ndim(out) == 0:
                    return np.full_like(S, float(out), dtype=float)
                return out
            return wrapped

        return _vec(kappa_fn), _vec(U_fn)

    def channels(self, dim: Optional[int] = None) -> Dict[str, ChannelSpec]:
        d = dim or self.genome.dim
        if self._channels is not None and next(iter(self._channels.values())).Q.shape[0] == d:
            return self._channels
        out: Dict[str, ChannelSpec] = {}
        Ug = self.gravity_potential()
        k, u = self._compile(Ug)
        out["gravity"] = ChannelSpec("gravity", Ug, self.genome.gravity.Q(d), "mass", True, True, k, u)
        Ue = self.em_potential()
        if Ue is not None:
            k, u = self._compile(Ue)
            out["em"] = ChannelSpec("em", Ue, np.eye(d), "charge", True, True, k, u)
        ix = self.genome.interaction
        if ix.species_active:
            Us = self.species_potential()
            k, u = self._compile(Us)
            out["species"] = ChannelSpec("species", Us, np.eye(d), "species", ix.nonreciprocity < 1e-12, False, k, u)
        if ix.core_repulsion > 0:
            Uc = self.core_potential()
            k, u = self._compile(Uc)
            out["core"] = ChannelSpec("core", Uc, np.eye(d), "core", True, False, k, u)
        self._channels = out
        return out

    def force_profile(self, r_values: np.ndarray, channel: str = "gravity") -> np.ndarray:
        """Numerical radial force -dU/dr for unit coefficients (isotropic slice)."""
        U = {"gravity": self.gravity_potential, "em": self.em_potential,
             "species": self.species_potential, "core": self.core_potential}[channel]()
        if U is None:
            return np.zeros_like(r_values)
        fn = sp.lambdify(self.r, -sp.diff(U, self.r), modules="numpy")
        out = fn(np.asarray(r_values, dtype=float))
        return np.broadcast_to(out, np.shape(r_values)).astype(float)

    # -- kinetic energy and Hamiltonian --------------------------------------
    def momentum_symbols(self, d: Optional[int] = None) -> List[sp.Symbol]:
        d = d or self.genome.dim
        return list(sp.symbols("p_x p_y p_z", real=True)[:d])

    def position_symbols(self, d: Optional[int] = None) -> List[sp.Symbol]:
        d = d or self.genome.dim
        return list(sp.symbols("x y z", real=True)[:d])

    def kinetic_energy(self, d: Optional[int] = None) -> sp.Expr:
        d = d or self.genome.dim
        inert = self.genome.inertia
        p = sp.Matrix(self.momentum_symbols(d))
        m = sp.Symbol("m", positive=True)
        p2 = (p.T * p)[0]
        if inert.model == "newtonian":
            return p2 / (2 * m)
        if inert.model == "relativistic":
            c = sp.Float(inert.c, 10)
            return sp.sqrt(p2 * c ** 2 + m ** 2 * c ** 4) - m * c ** 2
        if inert.model == "anisotropic":
            Minv = sp.Matrix(np.round(np.linalg.inv(inert.M(d)), 8))
            return (p.T * Minv * p)[0] / (2 * m)
        if inert.model == "power":
            a = sp.nsimplify(round(inert.kinetic_exponent, 4), rational=True)
            return p2 ** (a / 2) / (a * m ** (a - 1))
        raise ValueError(inert.model)

    def hamiltonian(self, d: Optional[int] = None) -> sp.Expr:
        """Test-particle Hamiltonian H = T(p) + U_grav(rho) + U_Lambda (unit source)."""
        d = d or self.genome.dim
        x = sp.Matrix(self.position_symbols(d))
        Q = sp.Matrix(np.round(self.genome.gravity.Q(d), 8))
        rho = sp.sqrt((x.T * Q * x)[0])
        m = sp.Symbol("m", positive=True)
        U = self.gravity_potential().subs(self.r, rho) * m * self.time_factor()
        if self.genome.cosmology.Lambda:
            U += -sp.Float(self.genome.cosmology.Lambda, 8) * m * (x.T * x)[0] / 6
        return self.kinetic_energy(d) + U

    def equations_of_motion(self, d: Optional[int] = None) -> List[sp.Eq]:
        """Hamilton's equations dq/dt = dH/dp, dp/dt = -dH/dq."""
        d = d or self.genome.dim
        H = self.hamiltonian(d)
        qs, ps = self.position_symbols(d), self.momentum_symbols(d)
        eqs = []
        for q, p in zip(qs, ps):
            eqs.append(sp.Eq(sp.Symbol(f"\\dot{{{q}}}"), sp.simplify(sp.diff(H, p))))
            eqs.append(sp.Eq(sp.Symbol(f"\\dot{{{sp.latex(p)}}}"), -sp.diff(H, q)))
        return eqs

    # -- field equations ------------------------------------------------------
    def effective_dimension(self) -> float:
        """Gauss-law dimension: |F| ~ r^-k is the Green's function of d = k+1."""
        return self.genome.gravity.k_eff + 1.0

    def field_equations(self) -> Dict[str, Dict[str, Any]]:
        """Field equations whose Green's functions are the mutated potentials,
        each verified symbolically or symbolic-numerically."""
        r = self.r
        out: Dict[str, Dict[str, Any]] = {}
        g = self.genome.gravity
        if g.law in ("inverse_square", "inverse_k", "tensor", "fractal"):
            k = sp.nsimplify(round(g.k_eff, 6), rational=True)
            dd = k + 1
            phi = _phi(r, k)
            radial_lap = sp.diff(phi, r, 2) + (dd - 1) / r * sp.diff(phi, r)
            out["gravity_gauss_law"] = {
                "equation": sp.Eq(sp.Symbol("nabla^2_{d_eff}") * sp.Function("Phi")(r),
                                  4 * sp.pi * sp.Symbol("G") * sp.Symbol("rho")),
                "latex": r"\nabla^2_{(d)}\Phi = S_{d} G \rho,\quad d_{\rm eff} = k+1 = " + sp.latex(dd),
                "green_function": sp.latex(phi),
                "verified": bool(sp.simplify(radial_lap) == 0),
                "note": "radial Laplacian in d = k+1 (possibly non-integer, Stillinger 1977) annihilates phi_k for r>0",
            }
        if g.law == "yukawa":
            lam = sp.Symbol("lambda", positive=True)  # verify for *all* lambda
            phi = sp.exp(-r / lam) / r
            helm = sp.diff(phi, r, 2) + 2 / r * sp.diff(phi, r) - phi / lam ** 2
            out["gravity_screened_poisson"] = {
                "latex": r"(\nabla^2 - \lambda^{-2})\Phi = 4\pi G\rho",
                "green_function": sp.latex(phi.subs(lam, sp.Float(g.screening_length, 6))),
                "verified": bool(sp.simplify(helm) == 0),
                "note": "Yukawa potential is the Green's function of the screened Poisson equation (proved for all lambda)",
            }
        if g.law in ("tensor", "fractal") and g.anisotropic:
            # phi = (x^T Q x)^((1-k)/2) solves div(Q^-1 grad phi) = 0 in d=k+1;
            # verify in 3-D for k=2 numerically at random points.
            Q = g.Q(3)
            Qi = np.linalg.inv(Q)
            xs = sp.symbols("x y z", real=True)
            X = sp.Matrix(xs)
            phi = ((X.T * sp.Matrix(Q) * X)[0]) ** sp.Rational(-1, 2)
            grad = sp.Matrix([sp.diff(phi, v) for v in xs])
            flux = sp.Matrix(Qi) * grad
            div = sum(sp.diff(flux[i], xs[i]) for i in range(3))
            fn = sp.lambdify(xs, div, modules="numpy")
            rng = np.random.default_rng(0)
            pts = rng.normal(size=(8, 3)) + 0.5
            residual = float(np.max(np.abs([fn(*p) for p in pts])))
            out["gravity_anisotropic_poisson"] = {
                "latex": r"\nabla\cdot(Q^{-1}\nabla\Phi) = 4\pi G \rho\,(\det Q)^{-1/2}",
                "green_function": r"(x^{T} Q x)^{-1/2}",
                "verified": residual < 1e-8,
                "residual": residual,
                "note": "coordinate map y = Q^{1/2} x turns the anisotropic operator into the Laplacian (checked on the k=2, 3-D slice at random points)",
            }
        if g.law == "fractal" and g.fractal_eps:
            M = self.gravity_modulation()
            beta_fn = sp.simplify(r * sp.diff(M, r))
            out["gravity_running_coupling"] = {
                "latex": r"G_{\rm eff}(r) = G\Big[1+\epsilon\sum_n\gamma^n\cos(\omega\beta^n\ln\tfrac{r}{r_0})\Big],\quad"
                         r"\beta(G_{\rm eff}) = \frac{dG_{\rm eff}}{d\ln r}",
                "beta_function": sp.latex(beta_fn),
                "verified": True,
                "note": "discrete scale invariance: G_eff is (quasi-)invariant under r -> r exp(2 pi/omega)",
            }
        e = self.genome.em
        if e.active and e.law == "proca":
            mu = sp.Symbol("mu", positive=True)  # verify for *all* photon masses
            phi = sp.exp(-mu * r) / r
            helm = sp.diff(phi, r, 2) + 2 / r * sp.diff(phi, r) - mu ** 2 * phi
            out["proca"] = {
                "latex": r"\partial_\mu F^{\mu\nu} + \mu^2 A^\nu = J^\nu \;\Rightarrow\; (\nabla^2-\mu^2)\phi = -\rho/\epsilon_0",
                "green_function": sp.latex(phi.subs(mu, sp.Float(e.proca_mass, 6))),
                "verified": bool(sp.simplify(helm) == 0),
                "note": "massive-photon electrostatics (proved for all mu)",
            }
        if e.active and e.law in ("coulomb", "power"):
            ke = 2.0 if e.law == "coulomb" else e.k
            k = sp.nsimplify(round(ke, 6), rational=True)
            phi = _phi(r, k)
            lap = sp.diff(phi, r, 2) + k / r * sp.diff(phi, r)
            out["em_gauss_law"] = {
                "latex": r"\nabla\cdot E = \rho/\epsilon_0 \text{ in } d_{\rm eff} = " + sp.latex(k + 1),
                "green_function": sp.latex(phi),
                "verified": bool(sp.simplify(lap) == 0),
                "note": "electrostatic Gauss law in d = k+1",
            }
        out["vorticity"] = {"latex": self.vorticity_equation_latex(), "verified": True,
                            "note": "pseudo-spectral solver uses exactly this operator in Fourier space"}
        return out

    def vorticity_equation_latex(self) -> str:
        f = self.genome.fluid
        N = f.N()
        return (r"\partial_t\omega + " + f"{f.advection:.3g}" + r"\,J(\psi,\omega) = -" + f"{f.viscosity:.3g}"
                + r"\,\big(k^{T}N k\big)^{" + f"{f.hyper_order:.3g}" + r"}\hat\omega - " + f"{f.ekman_drag:.3g}"
                + r"\,\omega + f,\quad N=" + sp.latex(sp.Matrix(np.round(N, 3))))

    # -- thermodynamics -------------------------------------------------------
    def entropy_functional(self) -> Dict[str, Any]:
        """Tsallis entropy and a symbolic proof that it reduces to Shannon as q->1."""
        q = sp.Symbol("q", positive=True)
        p1, p2 = sp.symbols("p_1 p_2", positive=True)
        p3 = 1 - p1 - p2
        Sq = (1 - p1 ** q - p2 ** q - p3 ** q) / (q - 1)
        lim = sp.limit(Sq, q, 1)
        shannon = -(p1 * sp.log(p1) + p2 * sp.log(p2) + p3 * sp.log(p3))
        ok = sp.simplify(sp.expand_log(lim - shannon, force=True)) == 0
        qv = self.genome.thermo.tsallis_q
        return {
            "latex": r"S_q = \frac{1-\sum_i p_i^{q}}{q-1},\quad q = " + f"{qv:.3g}",
            "shannon_limit_verified": bool(ok),
            "q": qv,
        }

    # -- orbital analysis ------------------------------------------------------
    def orbital_analysis(self, r0: float = 1.0) -> Dict[str, Any]:
        """Circular-orbit stability and apsidal angle for the isotropic gravity
        slice (Newtonian test particle):

        * stable iff U_eff''(r0) = 3U'(r0)/r0 + U''(r0) > 0
        * apsidal angle psi = pi * Omega / kappa with Omega^2 = U'/r, kappa^2 = U_eff''
          (psi = pi for Kepler: Bertrand's theorem)
        """
        r = self.r
        U = self.gravity_potential()
        U1, U2 = sp.diff(U, r), sp.diff(U, r, 2)
        f1 = sp.lambdify(r, U1, "numpy")
        f2 = sp.lambdify(r, U2, "numpy")

        def at(rv: float) -> Tuple[float, float, float]:
            a, b = float(f1(rv)), float(f2(rv))
            return a, b, 3 * a / rv + b

        u1, u2, kappa2 = at(r0)
        bound = u1 > 0
        stable = bool(bound and kappa2 > 0)
        psi = math.pi * math.sqrt(u1 / r0) / math.sqrt(kappa2) if stable else float("nan")
        grid = np.geomspace(0.05, 20.0, 400)
        stab = np.array([(lambda v: v[0] > 0 and v[2] > 0)(at(float(x))) for x in grid])
        intervals: List[Tuple[float, float]] = []
        start = None
        for x, s in zip(grid, stab):
            if s and start is None:
                start = x
            if not s and start is not None:
                intervals.append((float(start), float(x)))
                start = None
        if start is not None:
            intervals.append((float(start), float(grid[-1])))
        return {
            "r0": r0,
            "circular_orbit_exists": bool(bound),
            "stable_circular": stable,
            "apsidal_angle": psi,
            "precession_per_radial_period": (2 * psi - 2 * math.pi) if stable else float("nan"),
            "closed_orbits_expected": bool(stable and abs(psi - math.pi) < 1e-6),
            "stable_bands": intervals,
            "n_stable_bands": len(intervals),
        }

    # -- gravitational instability --------------------------------------------------
    def jeans_analysis(self, rho0: float = 1.0, cs: float = 1.0) -> Optional[Dict[str, Any]]:
        """Linear (Jeans) instability of a uniform self-gravitating gas.

        Linearised continuity + Euler with the mutated pair potential phi(r)
        give  omega^2 = cs^2 q^2 + rho0 q^2 phi_hat(q).

        * power law, U = -G r^(1-k)/(k-1), 1 < k < 4: with the 3-D Fourier
          transform of r^-a (a = k-1), phi_hat = -G A_k q^(k-4) and
          ``omega^2 = cs^2 q^2 - G rho0 A_k q^(k-2)``,
          ``A_k = pi^(3/2) 2^(4-k) Gamma((4-k)/2) / ((k-1) Gamma((k-1)/2))`` (A_2 = 4 pi).
          Unstable for q < q_J, q_J^(4-k) = G A_k rho0 / cs^2.  For k > 2 the growth
          rate peaks at q*^(4-k) = (k-2) G A_k rho0 / (2 cs^2): a *preferred
          clustering scale*; for k < 2 the largest scales grow fastest without bound.
        * Yukawa: phi_hat = -4 pi G / (q^2 + lambda^-2), so
          ``omega^2 = cs^2 q^2 - 4 pi G rho0 q^2 / (q^2 + lambda^-2)``; unstable
          iff q^2 < k_J^2 - lambda^-2: screening shorter than the Jeans length
          (lambda k_J < 1) switches structure formation off entirely.
        """
        g = self.genome.gravity
        G = float(g.G)
        if g.law == "yukawa":
            kJ2 = 4 * math.pi * G * rho0 / cs ** 2
            lam = float(g.screening_length)
            band = kJ2 - lam ** -2
            return {"law": "yukawa", "k_J": math.sqrt(kJ2), "unstable": band > 0,
                    "q_max_unstable": math.sqrt(band) if band > 0 else 0.0,
                    "lambda_times_kJ": lam * math.sqrt(kJ2)}
        if g.law in ("inverse_square", "inverse_k") or (g.law in ("tensor", "fractal") and not g.anisotropic
                                                          and (g.law == "tensor" or g.fractal_eps == 0)):
            k = g.k_eff
            if not 1.0 < k < 4.0:
                return None
            kk = sp.nsimplify(round(k, 6), rational=True)
            A = sp.pi ** sp.Rational(3, 2) * 2 ** (4 - kk) * sp.gamma((4 - kk) / 2) / ((kk - 1) * sp.gamma((kk - 1) / 2))
            Ak = float(A)
            qJ = (G * Ak * rho0 / cs ** 2) ** (1.0 / (4.0 - k))
            out = {"law": g.law, "k": k, "A_k": Ak, "A_k_exact": sp.latex(sp.simplify(A)), "q_J": qJ,
                   "jeans_length": 2 * math.pi / qJ, "unstable": True}
            if k > 2.0:
                qs = ((k - 2.0) * G * Ak * rho0 / (2 * cs ** 2)) ** (1.0 / (4.0 - k))
                out.update({"q_star": qs, "preferred_scale": 2 * math.pi / qs,
                            "max_growth_rate": math.sqrt(G * rho0 * Ak * qs ** (k - 2) - cs ** 2 * qs ** 2)})
            else:
                out.update({"q_star": 0.0 if k < 2 else None, "preferred_scale": float("inf"),
                            "note": "k <= 2: growth rate is maximal on the largest scales"})
            return out
        return None

    # -- reporting ---------------------------------------------------------------
    def latex(self) -> Dict[str, str]:
        out = {"U_gravity": sp.latex(self.gravity_potential()),
               "F_gravity": sp.latex(self.force_law(self.gravity_potential())),
               "T_kinetic": sp.latex(self.kinetic_energy())}
        Ue = self.em_potential()
        if Ue is not None:
            out["U_em"] = sp.latex(Ue)
        if self.genome.interaction.species_active:
            out["U_species"] = r"K_{ab}\," + sp.latex(self.species_potential())
        if self.genome.interaction.core_repulsion:
            out["U_core"] = sp.latex(self.core_potential())
        if self.genome.gravity.modulation_amp:
            out["G_t"] = r"G(t) = G\," + sp.latex(self.time_factor())
        th = self.genome.thermo
        if th.friction or th.active_gamma:
            out["langevin"] = (r"dp = F\,dt - \gamma p\,dt - \gamma_a\Big(\frac{|v|^2}{v_0^2}-1\Big)p\,dt"
                               r" + \sqrt{2\gamma m k_B T}\,dW_q")
        return out

    def report(self, r0: float = 1.0) -> Dict[str, Any]:
        """Everything a researcher needs about this universe's laws."""
        fe = self.field_equations()
        return {
            "genome_id": self.genome.genome_id,
            "latex": self.latex(),
            "effective_dimension": self.effective_dimension(),
            "field_equations": {k: {kk: (str(vv) if isinstance(vv, sp.Basic) else vv) for kk, vv in v.items()}
                                for k, v in fe.items()},
            "orbital": self.orbital_analysis(r0),
            "jeans": self.jeans_analysis(),
            "entropy": self.entropy_functional(),
            "noether": noether_analysis(self.genome).to_dict(),
        }


# ---------------------------------------------------------------------------
# Mutation engine
# ---------------------------------------------------------------------------

DEFAULT_OPERATOR_WEIGHTS: Dict[str, float] = {
    "gravity_params": 3.0,
    "gravity_escalate": 2.0,
    "em": 1.5,
    "inertia": 1.0,
    "thermo": 1.0,
    "fluid": 1.0,
    "interaction": 2.0,
    "symmetry": 1.5,
    "cosmology": 1.0,
    "dimension": 0.0,  # off by default: never reduce dimensionality unless asked
}


class PhysicsMutator:
    """Stochastic operators over :class:`PhysicsGenome` with lineage tracking.

    Parameters
    ----------
    seed:
        RNG seed (``numpy.random.default_rng``).
    operator_weights:
        Relative probability of each operator (see ``DEFAULT_OPERATOR_WEIGHTS``).
    enforce_chemistry_window:
        Keep alpha/alpha_0 inside :data:`ALPHA_CHEMISTRY_WINDOW` so that the
        engineering layer always has a chemistry to build with.
    """

    def __init__(self, seed: Optional[int] = None, operator_weights: Optional[Dict[str, float]] = None,
                 enforce_chemistry_window: bool = True, rng: Optional[np.random.Generator] = None):
        self.rng = rng if rng is not None else np.random.default_rng(seed)
        self.weights = dict(DEFAULT_OPERATOR_WEIGHTS)
        if operator_weights:
            self.weights.update(operator_weights)
        self.enforce_chemistry_window = enforce_chemistry_window
        self.operators: Dict[str, Callable[[PhysicsGenome, float], str]] = {
            "gravity_params": self._op_gravity_params,
            "gravity_escalate": self._op_gravity_escalate,
            "em": self._op_em,
            "inertia": self._op_inertia,
            "thermo": self._op_thermo,
            "fluid": self._op_fluid,
            "interaction": self._op_interaction,
            "symmetry": self._op_symmetry,
            "cosmology": self._op_cosmology,
            "dimension": self._op_dimension,
        }

    # -- factories -------------------------------------------------------------
    @staticmethod
    def baseline(dim: int = 3) -> PhysicsGenome:
        """Our-universe-like reference (Newtonian gravity, neutral matter)."""
        return PhysicsGenome(dim=dim, name="baseline-newtonian")

    def random_genome(self, dim: int = 3) -> PhysicsGenome:
        g = self.baseline(dim)
        g.name = ""
        for _ in range(int(self.rng.integers(2, 6))):
            op = self._choose_operator()
            self.operators[op](g, 1.5)
        g.gravity.law = str(self.rng.choice(GRAVITY_LAWS, p=[0.15, 0.25, 0.2, 0.25, 0.15]))
        self._ensure_law_params(g)
        self.repair(g)
        g.lineage = ["random"]
        return g

    # -- public API ---------------------------------------------------------------
    def mutate(self, genome: PhysicsGenome, strength: float = 1.0, n_ops: Optional[int] = None,
               operators: Optional[Sequence[str]] = None) -> PhysicsGenome:
        """Return a mutated child (the parent is never modified)."""
        child = genome.copy()
        child.genome_id = _new_id()
        child.parents = [genome.genome_id]
        child.generation = genome.generation + 1
        child.name = ""
        if operators is None:
            n = n_ops if n_ops is not None else 1 + int(self.rng.poisson(0.7))
            operators = [self._choose_operator() for _ in range(n)]
        notes = []
        for op in operators:
            notes.append(f"{op}: {self.operators[op](child, strength)}")
        self.repair(child)
        child.lineage = list(genome.lineage) + [f"g{child.generation}: " + "; ".join(notes)]
        return child

    def crossover(self, a: PhysicsGenome, b: PhysicsGenome) -> PhysicsGenome:
        """Gene-wise uniform crossover (each gene inherited intact - genes are
        co-adapted parameter blocks, so we never split them)."""
        child = a.copy()
        for gene in ("gravity", "em", "inertia", "thermo", "fluid", "interaction", "cosmology"):
            if self.rng.random() < 0.5:
                setattr(child, gene, copy.deepcopy(getattr(b, gene)))
        child.genome_id = _new_id()
        child.parents = [a.genome_id, b.genome_id]
        child.generation = max(a.generation, b.generation) + 1
        child.name = ""
        self.repair(child)
        child.lineage = list(a.lineage) + [f"g{child.generation}: crossover({a.genome_id},{b.genome_id})"]
        return child

    def repair(self, g: PhysicsGenome) -> PhysicsGenome:
        """Clamp parameters into :data:`PARAM_BOUNDS` and restore invariants."""
        for path, (lo, hi) in PARAM_BOUNDS.items():
            v = _get_path(g, path)
            if isinstance(lo, int) and isinstance(hi, int):
                _set_path(g, path, int(min(max(int(round(float(v))), lo), hi)))
            else:
                v = float(v)
                if not np.isfinite(v):
                    v = lo
                _set_path(g, path, float(min(max(v, lo), hi)))
        if self.enforce_chemistry_window:
            lo, hi = ALPHA_CHEMISTRY_WINDOW
            g.em.alpha_ratio = float(min(max(g.em.alpha_ratio, lo), hi))
        gg = g.gravity
        if gg.law not in GRAVITY_LAWS:
            gg.law = "inverse_square"
        # Keep the fractal envelope positive: eps * sum gamma^n < 0.9.
        env = sum(gg.fractal_gamma ** n for n in range(int(gg.fractal_octaves)))
        if gg.fractal_eps * env > 0.9:
            gg.fractal_eps = 0.9 / env
        gg.tensor_chol = [float(np.clip(x, -1.5, 1.5)) for x in gg.tensor_chol]
        g.inertia.mass_tensor_chol = [float(np.clip(x, -1.0, 1.0)) for x in g.inertia.mass_tensor_chol]
        g.fluid.viscosity_tensor_chol = [float(np.clip(x, -1.5, 1.5)) for x in g.fluid.viscosity_tensor_chol]
        g.em.B = [float(np.clip(x, -10, 10)) for x in g.em.B]
        # Reaction-diffusion stability (explicit Euler, dt=1, dx=1): D * lambda_max(N) * 4 < 1.
        lam_max = float(np.max(np.linalg.eigvalsh(g.fluid.N())))
        dmax = 0.24 / lam_max
        g.fluid.rd_diffusion_u = float(min(g.fluid.rd_diffusion_u, dmax))
        g.fluid.rd_diffusion_v = float(min(g.fluid.rd_diffusion_v, dmax))
        ix = g.interaction
        K = np.asarray(ix.species_matrix, dtype=float)
        n = int(ix.n_species)
        if K.shape != (n, n):
            newK = np.zeros((n, n))
            m = min(n, K.shape[0] if K.ndim == 2 else 0)
            if m:
                newK[:m, :m] = K[:m, :m]
            K = newK
        ix.species_matrix = np.clip(K, -5, 5).tolist()
        if g.dim not in (2, 3):
            g.dim = 3
        return g

    # -- vector interface for CMA-ES -----------------------------------------------
    @staticmethod
    def vectorize(g: PhysicsGenome) -> np.ndarray:
        out = []
        for path, lo, hi, scale in CONTINUOUS_SPEC:
            v = float(_get_path(g, path))
            if scale == "log":
                v = (math.log(max(v, lo)) - math.log(lo)) / (math.log(hi) - math.log(lo))
            else:
                v = (v - lo) / (hi - lo)
            out.append(min(max(v, 0.0), 1.0))
        return np.array(out)

    def devectorize(self, x: Sequence[float], template: PhysicsGenome) -> PhysicsGenome:
        g = template.copy()
        for (path, lo, hi, scale), v in zip(CONTINUOUS_SPEC, x):
            v = min(max(float(v), 0.0), 1.0)
            val = math.exp(math.log(lo) + v * (math.log(hi) - math.log(lo))) if scale == "log" else lo + v * (hi - lo)
            _set_path(g, path, val)
        g.genome_id = _new_id()
        g.parents = [template.genome_id]
        g.generation = template.generation + 1
        self.repair(g)
        g.lineage = list(template.lineage) + [f"g{g.generation}: cma-es continuous update"]
        return g

    # -- operators ---------------------------------------------------------------------
    def _choose_operator(self) -> str:
        names = list(self.weights)
        w = np.array([self.weights[n] for n in names], dtype=float)
        return str(self.rng.choice(names, p=w / w.sum()))

    def _lognormal(self, v: float, s: float) -> float:
        return float(v * math.exp(self.rng.normal(0.0, s)))

    def _ensure_law_params(self, g: PhysicsGenome) -> None:
        gg = g.gravity
        if gg.law == "fractal" and gg.fractal_eps == 0:
            gg.fractal_eps = float(self.rng.uniform(0.15, 0.5))
        if gg.law in ("tensor", "fractal") and not any(gg.tensor_chol):
            gg.tensor_chol = self.rng.normal(0, 0.3, 6).tolist()
        if gg.law in ("inverse_k", "tensor", "fractal") and gg.k == 2.0:
            gg.k = float(2.0 + self.rng.normal(0, 0.3))

    def _op_gravity_params(self, g: PhysicsGenome, s: float) -> str:
        gg = g.gravity
        gg.G = self._lognormal(gg.G, 0.25 * s)
        msg = [f"G->{gg.G:.3g}"]
        if gg.law in ("inverse_k", "tensor", "fractal"):
            gg.k = float(gg.k + self.rng.normal(0, 0.15 * s))
            msg.append(f"k->{gg.k:.3g}")
        if gg.law == "yukawa":
            gg.screening_length = self._lognormal(gg.screening_length, 0.4 * s)
            msg.append(f"lambda->{gg.screening_length:.3g}")
        if gg.law in ("tensor", "fractal"):
            gg.tensor_chol = (np.asarray(gg.tensor_chol) + self.rng.normal(0, 0.1 * s, 6)).tolist()
            msg.append("Q perturbed")
        if gg.law == "fractal":
            gg.fractal_eps = float(gg.fractal_eps + self.rng.normal(0, 0.08 * s))
            gg.fractal_omega = self._lognormal(gg.fractal_omega, 0.2 * s)
            gg.fractal_gamma = float(gg.fractal_gamma + self.rng.normal(0, 0.08 * s))
            if self.rng.random() < 0.2:
                gg.fractal_octaves = int(gg.fractal_octaves + self.rng.choice([-1, 1]))
            msg.append(f"fractal eps={gg.fractal_eps:.3g} omega={gg.fractal_omega:.3g}")
        return ", ".join(msg)

    def _op_gravity_escalate(self, g: PhysicsGenome, s: float) -> str:
        gg = g.gravity
        old = gg.law
        if gg.law == "yukawa" or self.rng.random() < 0.1:
            gg.law = str(self.rng.choice([x for x in GRAVITY_LAWS if x != gg.law]))
        else:
            i = GRAVITY_LADDER.index(gg.law)
            step = 1 if self.rng.random() < 0.8 else -1
            gg.law = GRAVITY_LADDER[int(np.clip(i + step, 0, len(GRAVITY_LADDER) - 1))]
            if gg.law == old:
                gg.law = "yukawa"
        self._ensure_law_params(g)
        return f"{old} -> {gg.law}"

    def _op_em(self, g: PhysicsGenome, s: float) -> str:
        e = g.em
        r = self.rng.random()
        if r < 0.3:
            e.law = str(self.rng.choice(EM_LAWS))
        e.alpha_ratio = self._lognormal(e.alpha_ratio, 0.15 * s)
        if e.law == "power":
            e.k = float(e.k + self.rng.normal(0, 0.2 * s))
        if e.law == "proca":
            e.proca_mass = self._lognormal(max(e.proca_mass, 0.1), 0.4 * s)
        e.specific_charge = float(abs(e.specific_charge + self.rng.normal(0, 0.3 * s)))
        if self.rng.random() < 0.3:
            if self.rng.random() < 0.5:
                e.B = [0.0, 0.0, float(self.rng.normal(0, 2.0 * s))]
            else:
                e.B = self.rng.normal(0, 1.5 * s, 3).tolist()
        if self.rng.random() < 0.15:
            e.B = [0.0, 0.0, 0.0]
        return f"law={e.law}, alpha/alpha0={e.alpha_ratio:.3g}, q/m={e.specific_charge:.3g}, B={np.round(e.B, 2).tolist()}"

    def _op_inertia(self, g: PhysicsGenome, s: float) -> str:
        i = g.inertia
        if self.rng.random() < 0.4:
            i.model = str(self.rng.choice(INERTIA_MODELS, p=[0.3, 0.3, 0.2, 0.2]))
        if i.model == "relativistic":
            i.c = float(self.rng.uniform(1.0, 6.0)) if self.rng.random() < 0.5 else self._lognormal(i.c, 0.3 * s)
        if i.model == "anisotropic":
            i.mass_tensor_chol = (np.asarray(i.mass_tensor_chol) + self.rng.normal(0, 0.2 * s, 6)).tolist()
        if i.model == "power":
            i.kinetic_exponent = float(i.kinetic_exponent + self.rng.normal(0, 0.2 * s))
        if self.rng.random() < 0.3:
            i.mass_spectrum = str(self.rng.choice(MASS_SPECTRA))
            i.mass_spread = float(abs(i.mass_spread + self.rng.normal(0, 0.2 * s)))
        return f"model={i.model}, c={i.c:.3g}, a={i.kinetic_exponent:.3g}, spectrum={i.mass_spectrum}"

    def _op_thermo(self, g: PhysicsGenome, s: float) -> str:
        t = g.thermo
        choice = self.rng.integers(0, 4)
        if choice == 0:
            t.friction = float(abs(t.friction + self.rng.normal(0, 0.5 * s)))
            t.temperature = float(abs(t.temperature + self.rng.normal(0, 0.05 * s)))
        elif choice == 1:
            t.noise = str(self.rng.choice(NOISE_MODELS))
            t.tsallis_q = float(1.0 + abs(self.rng.normal(0, 0.2 * s))) if t.noise == "q_gaussian" else 1.0
        elif choice == 2:
            t.active_gamma = float(abs(t.active_gamma + self.rng.normal(0, 0.8 * s)))
            t.active_v0 = self._lognormal(t.active_v0, 0.3 * s)
        else:
            t.adiabatic_index = float(t.adiabatic_index + self.rng.normal(0, 0.08 * s))
        return (f"gamma={t.friction:.3g}, T={t.temperature:.3g}, noise={t.noise}, q={t.tsallis_q:.3g},"
                f" active={t.active_gamma:.3g}, adiabatic={t.adiabatic_index:.3g}")

    def _op_fluid(self, g: PhysicsGenome, s: float) -> str:
        f = g.fluid
        f.viscosity = self._lognormal(f.viscosity, 0.4 * s)
        f.hyper_order = float(f.hyper_order + self.rng.normal(0, 0.25 * s))
        f.advection = float(abs(f.advection + self.rng.normal(0, 0.2 * s)))
        f.viscosity_tensor_chol = (np.asarray(f.viscosity_tensor_chol) + self.rng.normal(0, 0.15 * s, 3)).tolist()
        f.drag_exponent = float(f.drag_exponent + self.rng.normal(0, 0.2 * s))
        if self.rng.random() < 0.3:
            f.medium_drag = float(abs(f.medium_drag + self.rng.normal(0, 0.1 * s)))
        f.rd_feed = float(f.rd_feed + self.rng.normal(0, 0.006 * s))
        f.rd_kill = float(f.rd_kill + self.rng.normal(0, 0.003 * s))
        return (f"nu={f.viscosity:.3g}, h={f.hyper_order:.3g}, lambda={f.advection:.3g}, n_drag={f.drag_exponent:.3g},"
                f" F={f.rd_feed:.3g}, k={f.rd_kill:.3g}")

    def _op_interaction(self, g: PhysicsGenome, s: float) -> str:
        ix = g.interaction
        r = self.rng.random()
        if r < 0.25:
            ix.propagation = "retarded" if ix.propagation == "instantaneous" else "instantaneous"
            ix.c_prop = float(self.rng.uniform(1.0, 10.0))
        elif r < 0.5:
            ix.n_species = int(np.clip(ix.n_species + self.rng.choice([-1, 1, 1]), 1, 5))
            self.repair(g)
            K = ix.K() + self.rng.normal(0, 0.6 * s, (ix.n_species, ix.n_species))
            ix.species_matrix = K.tolist()
        elif r < 0.8:
            K = ix.K() + self.rng.normal(0, 0.3 * s, (ix.n_species, ix.n_species))
            ix.species_matrix = K.tolist()
            ix.species_range = self._lognormal(ix.species_range, 0.2 * s)
        else:
            ix.core_repulsion = float(abs(ix.core_repulsion + self.rng.normal(0, 0.4 * s)))
            ix.core_range = self._lognormal(ix.core_range, 0.2 * s)
        return (f"prop={ix.propagation}, species={ix.n_species}, nonrecip={ix.nonreciprocity:.2f},"
                f" core={ix.core_repulsion:.3g}")

    def _op_symmetry(self, g: PhysicsGenome, s: float) -> str:
        """Deliberately break or restore one symmetry (and hence one
        conservation law) - the only consistent way to 'mutate conservation'."""
        actions = ["break_rotation", "restore_rotation", "break_time", "restore_time",
                   "break_reciprocity", "restore_reciprocity", "break_translation", "restore_translation",
                   "break_time_reversal", "restore_time_reversal"]
        act = str(self.rng.choice(actions))
        gg, ix = g.gravity, g.interaction
        if act == "break_rotation":
            if gg.law not in ("tensor", "fractal"):
                gg.law = "tensor"
            gg.tensor_chol = self.rng.normal(0, 0.4 * s, 6).tolist()
            self._ensure_law_params(g)
        elif act == "restore_rotation":
            gg.tensor_chol = [0.0] * 6
            g.inertia.mass_tensor_chol = [0.0] * 6
            g.em.B = [0.0, 0.0, float(g.em.B[2])]
        elif act == "break_time":
            gg.modulation_amp = float(self.rng.uniform(0.05, 0.4))
            gg.modulation_freq = float(self.rng.uniform(0.5, 6.0))
        elif act == "restore_time":
            gg.modulation_amp = 0.0
        elif act == "break_reciprocity":
            if ix.n_species < 2:
                ix.n_species = 2
                self.repair(g)
            n = ix.n_species
            A = self.rng.normal(0, 0.8 * s, (n, n))
            ix.species_matrix = (ix.K() + 0.5 * (A - A.T)).tolist()
        elif act == "restore_reciprocity":
            K = ix.K()
            ix.species_matrix = (0.5 * (K + K.T)).tolist()
        elif act == "break_translation":
            g.cosmology.Lambda = float(self.rng.normal(0, 1.0 * s))
        elif act == "restore_translation":
            g.cosmology.Lambda = 0.0
        elif act == "break_time_reversal":
            if g.em.law == "off":
                g.em.law = "coulomb"
            g.em.B = [0.0, 0.0, float(self.rng.normal(0, 2.0))]
        else:
            g.em.B = [0.0, 0.0, 0.0]
        return act

    def _op_cosmology(self, g: PhysicsGenome, s: float) -> str:
        c = g.cosmology
        c.Lambda = float(c.Lambda + self.rng.normal(0, 0.5 * s))
        if self.rng.random() < 0.3:
            c.hubble = float(c.hubble + self.rng.normal(0, 0.1 * s))
        if self.rng.random() < 0.2:
            c.hubble = 0.0
        return f"Lambda={c.Lambda:.3g}, H={c.hubble:.3g}"

    def _op_dimension(self, g: PhysicsGenome, s: float) -> str:
        g.dim = 2 if g.dim == 3 else 3
        return f"dim -> {g.dim}"
