"""
SHIVA-1 :: Universe Simulator
=============================

Evolves alternate universes defined by a :class:`~shiva_core.physics_mutator.PhysicsGenome`
on three coupled levels of description:

1. **Particles** (``UniverseSimulator``): N-body dynamics in d = 2 or 3
   dimensions with every pair channel compiled from the genome's SymPy
   potentials.  The integrator is a symmetric Strang splitting of the flow::

       K(h/2) R(h/2) D(h/2) O(h) D(h/2) R(h/2) K(h/2)

   * ``K`` - kick with all position-dependent forces (exact sub-flow),
   * ``R`` - exact rotation of momenta about the background magnetic field
     (Boris-type; preserves |p| and therefore T(|p|) exactly),
   * ``D`` - drift with the *mutated* velocity v = dT/dp (Newtonian,
     relativistic, anisotropic or power-law inertia),
   * ``O`` - dissipative / stochastic sub-flow: Ornstein-Uhlenbeck Langevin
     bath (BAOAB, Leimkuhler & Matthews 2013), Hubble drag, medium drag and
     Rayleigh-Helmholtz active friction (exact logistic speed update).

   Without the ``O`` and ``R`` pieces this is the Stormer-Verlet leapfrog,
   which is symplectic for every separable H = T(p) + U(q) offered by the
   mutator.  ``yoshida4`` composes it into a 4th-order symplectic scheme
   (Yoshida 1990); ``rk4`` is provided as a non-symplectic control.

   All state arrays carry a leading *batch* axis.  The batch holds a shadow
   ("twin") trajectory displaced by 1e-7 in phase space and driven by the
   same noise, giving the maximal Lyapunov exponent by Benettin
   renormalisation at almost no extra cost.

2. **Rigid bodies** (``RigidBodyEnsemble``): free rigid-body rotation by the
   exact axis-splitting method (McLachlan 1993; Touma & Wisdom 1994), with
   optional internal dissipation at constant |L| (the "major-axis rule") and
   centre-of-mass motion under the mutated gravity.  An optional PyBullet
   backend is used when the package is installed.

3. **Continuum fields**: a pseudo-spectral 2-D vorticity solver with mutated
   viscosity tensor / hyperviscosity / advection coupling (integrating-factor
   RK4, 2/3 de-aliasing) and an anisotropic Gray-Scott reaction-diffusion
   field.

``evaluate_simulation`` scores stability, Noether consistency (predicted vs.
measured conservation) and cosmic-scale feasibility.
"""

from __future__ import annotations

import importlib.util
import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .physics_mutator import NoetherReport, PhysicsGenome, SymbolicPhysics, noether_analysis

__all__ = [
    "SimulationConfig",
    "SimulationResult",
    "UniverseSimulator",
    "Kinetics",
    "ForceModel",
    "RigidBodyConfig",
    "RigidBodyEnsemble",
    "RigidBodyResult",
    "PyBulletRigidBackend",
    "FieldResult",
    "VorticityField2D",
    "ReactionDiffusion2D",
    "UniverseEvaluation",
    "UniverseRun",
    "evaluate_simulation",
    "run_universe",
]


# ---------------------------------------------------------------------------
# Configuration and results
# ---------------------------------------------------------------------------

@dataclass
class SimulationConfig:
    """Numerical set-up of a particle universe (simulation units: G = M = R = 1)."""

    n_particles: int = 96
    dim: Optional[int] = None  # defaults to genome.dim
    dt: float = 2e-3
    steps: int = 2000
    record_every: int = 20
    integrator: str = "leapfrog"  # leapfrog | yoshida4 | rk4
    softening: float = 0.05
    boundary: str = "open"  # open | periodic | reflective
    box_size: float = 4.0
    initial_condition: str = "cold_collapse"  # cold_collapse | virialized | uniform | rotating_disk | lattice | binary
    ic_radius: float = 1.0
    ic_spin: float = 0.0
    ic_velocity_dispersion: float = 0.05
    seed: int = 0
    lyapunov: bool = True
    lyapunov_delta0: float = 1e-7
    lyapunov_renorm_every: int = 10
    max_speed: float = 1e3
    max_history: int = 1500  # retarded-propagation horizon (steps)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SimulationResult:
    genome_id: str
    config: Dict[str, Any]
    dim: int
    times: np.ndarray
    positions: np.ndarray  # (R, N, d)
    velocities: np.ndarray  # (R, N, d)
    masses: np.ndarray
    charges: np.ndarray
    species: np.ndarray
    kinetic: np.ndarray
    potential: np.ndarray
    momentum: np.ndarray  # (R, 3) mechanical or pseudo
    angular_momentum: np.ndarray  # (R, 3) mechanical or canonical
    virial_ratio: np.ndarray  # -<x.F>/<p.v>
    bound_fraction: np.ndarray
    noether: NoetherReport
    lyapunov: Optional[float] = None
    lyapunov_series: Optional[np.ndarray] = None
    blew_up: bool = False
    blowup_time: Optional[float] = None
    wall_time: float = 0.0
    reference_length: float = 1.0
    energy_scale: float = 1.0
    momentum_scale: float = 1.0
    angular_scale: float = 1.0

    @property
    def energy(self) -> np.ndarray:
        return self.kinetic + self.potential

    @property
    def t_end(self) -> float:
        return float(self.times[-1]) if len(self.times) else 0.0

    def summary(self) -> Dict[str, Any]:
        return {
            "genome_id": self.genome_id,
            "dim": self.dim,
            "n_particles": int(self.masses.size),
            "t_end": self.t_end,
            "records": int(len(self.times)),
            "blew_up": self.blew_up,
            "blowup_time": self.blowup_time,
            "lyapunov": self.lyapunov,
            "final_virial_ratio": float(self.virial_ratio[-1]) if len(self.virial_ratio) else None,
            "final_bound_fraction": float(self.bound_fraction[-1]) if len(self.bound_fraction) else None,
            "wall_time_s": self.wall_time,
        }


# ---------------------------------------------------------------------------
# Kinetics: mutated inertia
# ---------------------------------------------------------------------------

class Kinetics:
    """Kinetic energy T(p), velocity dT/dp and their inverses for each inertia model."""

    def __init__(self, genome: PhysicsGenome, masses: np.ndarray, dim: int):
        g = genome.inertia
        self.model = g.model
        self.m = masses[None, :]  # (1, N)
        self.mc = masses[None, :, None]  # (1, N, 1)
        self.c = float(g.c)
        self.a = float(g.kinetic_exponent)
        self.M = g.M(dim)
        self.Minv = np.linalg.inv(self.M)

    def T(self, P: np.ndarray) -> np.ndarray:
        """Per-particle kinetic energy, shape (B, N)."""
        if self.model == "newtonian":
            return 0.5 * np.sum(P * P, axis=-1) / self.m
        if self.model == "relativistic":
            p2 = np.sum(P * P, axis=-1)
            c2 = self.c ** 2
            E = np.sqrt(p2 * c2 + (self.m * c2) ** 2)
            return p2 * c2 / (E + self.m * c2)  # = E - m c^2 without cancellation
        if self.model == "anisotropic":
            return 0.5 * np.einsum("bni,ij,bnj->bn", P, self.Minv, P) / self.m
        if self.model == "power":
            pn = np.linalg.norm(P, axis=-1)
            return pn ** self.a / (self.a * self.m ** (self.a - 1.0))
        raise ValueError(self.model)

    def velocity(self, P: np.ndarray) -> np.ndarray:
        if self.model == "newtonian":
            return P / self.mc
        if self.model == "relativistic":
            p2 = np.sum(P * P, axis=-1, keepdims=True)
            c2 = self.c ** 2
            E = np.sqrt(p2 * c2 + (self.mc * c2) ** 2)
            return P * c2 / E
        if self.model == "anisotropic":
            return (P @ self.Minv) / self.mc
        if self.model == "power":
            pn = np.linalg.norm(P, axis=-1, keepdims=True)
            with np.errstate(divide="ignore", invalid="ignore"):
                fac = np.where(pn > 1e-300, pn ** (self.a - 2.0), 0.0) / self.mc ** (self.a - 1.0)
            return fac * P
        raise ValueError(self.model)

    def speed_factor(self, P: np.ndarray) -> np.ndarray:
        """Scalar f with |v| = f |p| (exact for isotropic laws), shape (B, N)."""
        pn = np.linalg.norm(P, axis=-1)
        vn = np.linalg.norm(self.velocity(P), axis=-1)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(pn > 1e-300, vn / pn, 1.0 / self.m)

    def momentum_from_velocity(self, V: np.ndarray) -> np.ndarray:
        if self.model == "newtonian":
            return V * self.mc
        if self.model == "relativistic":
            vn = np.linalg.norm(V, axis=-1, keepdims=True)
            vmax = 0.95 * self.c
            V = np.where(vn > vmax, V * vmax / np.maximum(vn, 1e-300), V)
            v2 = np.sum(V * V, axis=-1, keepdims=True)
            gamma = 1.0 / np.sqrt(1.0 - v2 / self.c ** 2)
            return gamma * self.mc * V
        if self.model == "anisotropic":
            return (V @ self.M) * self.mc
        if self.model == "power":
            vn = np.linalg.norm(V, axis=-1, keepdims=True)
            with np.errstate(divide="ignore", invalid="ignore"):
                pn = self.mc * np.where(vn > 0, vn ** (1.0 / (self.a - 1.0)), 0.0)
                return np.where(vn > 0, pn * V / vn, 0.0)
        raise ValueError(self.model)


# ---------------------------------------------------------------------------
# Forces
# ---------------------------------------------------------------------------

class _History:
    """Ring buffer of past positions for retarded interactions."""

    def __init__(self, X0: np.ndarray, capacity: int):
        self.buf = np.repeat(X0[None], capacity, axis=0)
        self.capacity = capacity
        self.idx = 0
        self.filled = 1

    def push(self, X: np.ndarray) -> None:
        self.idx = (self.idx + 1) % self.capacity
        self.buf[self.idx] = X
        self.filled = min(self.filled + 1, self.capacity)

    def ordered(self) -> np.ndarray:
        """States ordered by lag: element k is k+1 steps in the past."""
        order = (self.idx - np.arange(self.filled)) % self.capacity
        return self.buf[order]


class ForceModel:
    """All pair channels of a genome, compiled for vectorised evaluation."""

    def __init__(self, genome: PhysicsGenome, cfg: SimulationConfig, masses: np.ndarray,
                 charges: np.ndarray, species: np.ndarray, dim: int):
        self.genome = genome
        self.dim = dim
        self.cfg = cfg
        sym = SymbolicPhysics(genome)
        self.channels = sym.channels(dim)
        self.masses = masses
        N = masses.size
        mm = np.outer(masses, masses)
        K = genome.interaction.K()
        coeffs = {
            "mass": mm,
            "charge": np.outer(charges, charges),
            "species": K[species][:, species] * mm,
            "core": mm,
        }
        self.C: Dict[str, np.ndarray] = {}
        self.Csym: Dict[str, np.ndarray] = {}
        self.iso: Dict[str, bool] = {}
        for name, ch in self.channels.items():
            C = coeffs[ch.coefficient].copy()
            np.fill_diagonal(C, 0.0)
            self.C[name] = C
            self.Csym[name] = 0.5 * (C + C.T)
            self.iso[name] = bool(np.allclose(ch.Q, np.eye(dim)))
        g = genome.gravity
        self.mod_amp, self.mod_freq = float(g.modulation_amp), float(g.modulation_freq)
        self.periodic = cfg.boundary == "periodic"
        self.L = float(cfg.box_size)
        self.Lambda = float(genome.cosmology.Lambda) if not self.periodic else 0.0
        self.retarded = genome.interaction.propagation == "retarded"
        self.c_prop = float(genome.interaction.c_prop)
        self.eps2 = float(cfg.softening) ** 2
        self.dt = float(cfg.dt)
        self.N = N
        self.confining = genome.gravity.k_eff <= 1.0 and g.law != "yukawa"

    # -- geometry ---------------------------------------------------------------
    def displacements(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        """D[b, i, j] = x_i - y_j (minimum image if periodic)."""
        Y = X if Y is None else Y
        D = X[:, :, None, :] - Y[:, None, :, :] if Y.ndim == 3 else X[:, :, None, :] - Y
        if self.periodic:
            D = D - self.L * np.round(D / self.L)
        return D

    def _retarded_displacements(self, X: np.ndarray, D: np.ndarray, hist: Optional[_History]) -> np.ndarray:
        if hist is None or hist.filled <= 1:
            return D
        r = np.sqrt(np.sum(D * D, axis=-1))
        lag = np.clip(np.rint(r / (self.c_prop * self.dt)).astype(int), 0, hist.filled)
        past = hist.ordered()  # (H, B, N, d), lag k+1 at index k
        stack = np.concatenate([X[None], past], axis=0)  # lag 0 = current
        B, N = X.shape[0], X.shape[1]
        b_idx = np.arange(B)[:, None, None]
        j_idx = np.arange(N)[None, None, :]
        Xsrc = stack[lag, b_idx, j_idx]  # (B, N, N, d)
        Dr = X[:, :, None, :] - Xsrc
        if self.periodic:
            Dr = Dr - self.L * np.round(Dr / self.L)
        return Dr

    def time_factor(self, t: float) -> float:
        return 1.0 + self.mod_amp * math.sin(self.mod_freq * t) if self.mod_amp else 1.0

    # -- forces and energies -----------------------------------------------------------
    def forces(self, X: np.ndarray, t: float, hist: Optional[_History] = None) -> np.ndarray:
        D = self.displacements(X)
        Dret = self._retarded_displacements(X, D, hist) if self.retarded else D
        F = np.zeros_like(X)
        for name, ch in self.channels.items():
            Dc = Dret if (ch.long_range and self.retarded) else D
            if self.iso[name]:
                QD = Dc
                S = np.sum(Dc * Dc, axis=-1)
            else:
                QD = Dc @ ch.Q
                S = np.sum(QD * Dc, axis=-1)
            Kmat = self.C[name] * ch.kappa(S + self.eps2)
            if name == "gravity" and self.mod_amp:
                Kmat = Kmat * self.time_factor(t)
            F += np.einsum("bij,bijd->bid", Kmat, QD)
        if self.Lambda:
            F += (self.Lambda / 3.0) * self.masses[None, :, None] * X
        return F

    def pair_energy_matrix(self, X: np.ndarray, t: float) -> np.ndarray:
        """E[b, i, j] = symmetric-part pair energy between i and j."""
        D = self.displacements(X)
        E = np.zeros(D.shape[:3])
        for name, ch in self.channels.items():
            if self.iso[name]:
                S = np.sum(D * D, axis=-1)
            else:
                S = np.sum((D @ ch.Q) * D, axis=-1)
            e = self.Csym[name] * ch.U(S + self.eps2)
            if name == "gravity" and self.mod_amp:
                e = e * self.time_factor(t)
            E += e
        return E

    def potential(self, X: np.ndarray, t: float) -> np.ndarray:
        """Total potential energy per batch member (B,).  For non-reciprocal
        couplings only the symmetric part admits a potential; it is reported
        as a pseudo-potential."""
        U = 0.5 * np.sum(self.pair_energy_matrix(X, t), axis=(1, 2))
        if self.Lambda:
            U = U - (self.Lambda / 6.0) * np.sum(self.masses[None, :] * np.sum(X * X, axis=-1), axis=1)
        return U

    def particle_energies(self, X: np.ndarray, t: float) -> np.ndarray:
        e = np.sum(self.pair_energy_matrix(X, t), axis=2)
        if self.Lambda:
            e = e - (self.Lambda / 6.0) * self.masses[None, :] * np.sum(X * X, axis=-1)
        return e


# ---------------------------------------------------------------------------
# Particle universe
# ---------------------------------------------------------------------------

class UniverseSimulator:
    """N-body integrator for a mutated universe."""

    YOSHIDA_W1 = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
    YOSHIDA_W0 = -(2.0 ** (1.0 / 3.0)) / (2.0 - 2.0 ** (1.0 / 3.0))

    def __init__(self, genome: PhysicsGenome, config: Optional[SimulationConfig] = None):
        self.genome = genome
        self.cfg = config or SimulationConfig()
        self.dim = int(self.cfg.dim or genome.dim)
        self.rng = np.random.default_rng(self.cfg.seed)
        self.noise_rng = np.random.default_rng(self.cfg.seed + 7919)
        N = int(self.cfg.n_particles)
        self.masses = self._sample_masses(N)
        n_sp = int(genome.interaction.n_species)
        self.species = self.rng.integers(0, n_sp, N) if n_sp > 1 else np.zeros(N, dtype=int)
        if genome.em.active:
            signs = np.where(np.arange(N) % 2 == 0, 1.0, -1.0)
            self.rng.shuffle(signs)
            self.charges = signs * genome.em.specific_charge * self.masses
        else:
            self.charges = np.zeros(N)
        self.kin = Kinetics(genome, self.masses, self.dim)
        self.force = ForceModel(genome, self.cfg, self.masses, self.charges, self.species, self.dim)
        self.noether = noether_analysis(genome, boundary=self.cfg.boundary, dim=self.dim)
        th, fl, cos = genome.thermo, genome.fluid, genome.cosmology
        self.B = genome.em.B_vec(self.dim)
        self.has_R = bool(np.linalg.norm(self.B) > 0 and np.any(self.charges))
        self.has_O = th.friction > 0 or th.active_gamma > 0 or fl.medium_drag > 0 or cos.hubble != 0
        integ = self.cfg.integrator
        if integ not in ("leapfrog", "yoshida4", "rk4"):
            raise ValueError(f"unknown integrator {integ}")
        if integ == "yoshida4" and (self.has_O or self.genome.interaction.propagation == "retarded"):
            integ = "leapfrog"  # high-order composition is only meaningful for deterministic, Markovian flows
        self.integrator = integ

    # -- initial conditions ---------------------------------------------------------
    def _sample_masses(self, N: int) -> np.ndarray:
        g = self.genome.inertia
        if g.mass_spectrum == "lognormal":
            m = np.exp(self.rng.normal(0.0, g.mass_spread, N))
        elif g.mass_spectrum == "salpeter":
            a = g.imf_slope
            lo, hi = 1.0, 30.0
            u = self.rng.random(N)
            m = (lo ** (1 - a) + u * (hi ** (1 - a) - lo ** (1 - a))) ** (1.0 / (1 - a))
        else:
            m = np.ones(N)
        return m / m.sum()

    def _ball(self, N: int, R: float) -> np.ndarray:
        d = self.dim
        v = self.rng.normal(size=(N, d))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        return v * R * self.rng.random((N, 1)) ** (1.0 / d)

    def _virial_scale(self, X: np.ndarray, V: np.ndarray) -> np.ndarray:
        """Rescale velocities so that sum m v^2 = -sum x.F (generalised virial
        theorem for *any* mutated force law)."""
        F = self.force.forces(X[None], 0.0)[0]
        xc = X - np.average(X, axis=0, weights=self.masses)
        W = -float(np.sum(xc * F))
        K2 = float(np.sum(self.masses[:, None] * V * V))
        if W > 0 and K2 > 0:
            V = V * math.sqrt(W / K2)
        return V

    def initial_conditions(self) -> Tuple[np.ndarray, np.ndarray]:
        cfg, d, N, R = self.cfg, self.dim, self.cfg.n_particles, self.cfg.ic_radius
        ic = cfg.initial_condition
        sig = cfg.ic_velocity_dispersion
        if ic == "cold_collapse":
            X = self._ball(N, R)
            V = self.rng.normal(0.0, sig, (N, d))
        elif ic == "virialized":
            # Plummer radial profile, isotropic directions, virial-scaled speeds.
            u = self.rng.uniform(0.02, 0.98, N)
            r = R * 0.6 / np.sqrt(u ** (-2.0 / 3.0) - 1.0)
            r = np.minimum(r, 6 * R)
            dirs = self.rng.normal(size=(N, d))
            dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
            X = dirs * r[:, None]
            V = self.rng.normal(0.0, 1.0, (N, d))
            V = self._virial_scale(X, V)
        elif ic == "uniform":
            X = self.rng.uniform(-0.5, 0.5, (N, d)) * cfg.box_size
            V = self.rng.normal(0.0, sig, (N, d))
        elif ic == "lattice":
            n_side = int(math.ceil(N ** (1.0 / d)))
            grid = np.stack(np.meshgrid(*[np.linspace(-R, R, n_side)] * d, indexing="ij"), -1).reshape(-1, d)
            X = grid[self.rng.choice(len(grid), N, replace=False)] + self.rng.normal(0, 0.02 * R, (N, d))
            V = np.zeros((N, d))
        elif ic == "rotating_disk":
            rr = R * np.sqrt(self.rng.uniform(0.01, 1.0, N))
            th = self.rng.uniform(0, 2 * np.pi, N)
            X = np.zeros((N, d))
            X[:, 0], X[:, 1] = rr * np.cos(th), rr * np.sin(th)
            if d == 3:
                X[:, 2] = self.rng.normal(0, 0.05 * R, N)
            F = self.force.forces(X[None], 0.0)[0]
            xy = X[:, :2]
            radial = -np.sum(F[:, :2] * xy, axis=1) / np.maximum(np.linalg.norm(xy, axis=1), 1e-12)
            vc = np.sqrt(np.maximum(radial * np.linalg.norm(xy, axis=1) / self.masses, 0.0))
            V = np.zeros((N, d))
            V[:, 0], V[:, 1] = -vc * np.sin(th), vc * np.cos(th)
            V += self.rng.normal(0, sig, (N, d))
        elif ic == "binary":
            h = N // 2
            X1, X2 = self._ball(h, 0.5 * R), self._ball(N - h, 0.5 * R)
            off = np.zeros(d)
            off[0] = 1.5 * R
            X = np.concatenate([X1 - off, X2 + off])
            V = self.rng.normal(0.0, sig, (N, d))
            V[:h, 0] += 0.3
            V[h:, 0] -= 0.3
            if d >= 2:
                V[:h, 1] += 0.15
                V[h:, 1] -= 0.15
        else:
            raise ValueError(f"unknown initial condition {ic}")
        if cfg.ic_spin and d >= 2:
            V[:, 0] += -cfg.ic_spin * X[:, 1]
            V[:, 1] += cfg.ic_spin * X[:, 0]
        if cfg.boundary == "open":  # boxed universes must start inside their walls
            X = X - np.average(X, axis=0, weights=self.masses)
        P = self.kin.momentum_from_velocity(V[None])[0]
        P = P - P.mean(axis=0)  # zero total momentum
        return X, P

    # -- sub-flows -------------------------------------------------------------------
    def _drift(self, X: np.ndarray, P: np.ndarray, h: float) -> Tuple[np.ndarray, np.ndarray]:
        X = X + h * self.kin.velocity(P)
        bnd = self.cfg.boundary
        if bnd == "periodic":
            L = self.cfg.box_size
            X = (X + 0.5 * L) % L - 0.5 * L
        elif bnd == "reflective":
            L2 = 0.5 * self.cfg.box_size
            over = np.abs(X) > L2
            if np.any(over):
                X = np.where(X > L2, 2 * L2 - X, X)
                X = np.where(X < -L2, -2 * L2 - X, X)
                P = np.where(over, -P, P)
        return X, P

    def _rotate(self, P: np.ndarray, h: float) -> np.ndarray:
        if not self.has_R:
            return P
        q = self.charges[None, :]
        f = self.kin.speed_factor(P)
        if self.dim == 2:
            theta = -q * self.B[2] * f * h
            c, s = np.cos(theta), np.sin(theta)
            px, py = P[..., 0], P[..., 1]
            return np.stack([c * px - s * py, s * px + c * py], axis=-1)
        Bn = float(np.linalg.norm(self.B))
        b = self.B / Bn
        theta = (-q * Bn * f * h)[..., None]
        bxp = np.cross(np.broadcast_to(b, P.shape), P)
        bdp = np.sum(P * b, axis=-1, keepdims=True)
        return P * np.cos(theta) + bxp * np.sin(theta) + b * bdp * (1 - np.cos(theta))

    def _noise(self, shape: Tuple[int, ...]) -> np.ndarray:
        th = self.genome.thermo
        if th.noise == "q_gaussian" and th.tsallis_q > 1.0 + 1e-9:
            q = min(th.tsallis_q, 1.6)
            nu = (3.0 - q) / (q - 1.0)
            xi = self.noise_rng.standard_t(nu, size=shape)
            return xi / math.sqrt(nu / (nu - 2.0))
        return self.noise_rng.standard_normal(shape)

    def _O(self, P: np.ndarray, h: float) -> np.ndarray:
        th, fl, cos = self.genome.thermo, self.genome.fluid, self.genome.cosmology
        if cos.hubble:
            P = P * math.exp(-cos.hubble * h)
        if th.friction > 0:
            c1 = math.exp(-th.friction * h)
            P = c1 * P
            if th.temperature > 0:
                xi = self._noise((1,) + P.shape[1:])  # shared by the Lyapunov twin
                P = P + math.sqrt((1.0 - c1 * c1) * th.temperature) * np.sqrt(self.kin.mc) * xi
        if fl.medium_drag > 0:
            f = self.kin.speed_factor(P)
            vn = f * np.linalg.norm(P, axis=-1)
            P = P / (1.0 + h * fl.medium_drag * vn ** (fl.drag_exponent - 1.0) * f)[..., None]
        if th.active_gamma > 0:
            v = np.linalg.norm(self.kin.velocity(P), axis=-1)
            v2, v02 = v * v, th.active_v0 ** 2
            e = math.exp(2.0 * th.active_gamma * h)
            v2n = v02 * v2 * e / (v02 + v2 * (e - 1.0))
            with np.errstate(divide="ignore", invalid="ignore"):
                scale = np.where(v2 > 1e-300, np.sqrt(v2n / v2), 1.0)
            P = P * scale[..., None]
        return P

    def _step_splitting(self, X: np.ndarray, P: np.ndarray, F: np.ndarray, t: float, h: float,
                        hist: Optional[_History]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        P = P + 0.5 * h * F
        P = self._rotate(P, 0.5 * h)
        X_old = X
        if self.has_O:
            X, P = self._drift(X, P, 0.5 * h)
            P = self._O(P, h)
            X, P = self._drift(X, P, 0.5 * h)
        else:
            X, P = self._drift(X, P, h)
        P = self._rotate(P, 0.5 * h)
        if hist is not None:
            hist.push(X_old)
        F = self.force.forces(X, t + h, hist)
        P = P + 0.5 * h * F
        return X, P, F

    def _magnetic_force(self, P: np.ndarray) -> np.ndarray:
        if not self.has_R:
            return np.zeros_like(P)
        V = self.kin.velocity(P)
        q = self.charges[None, :, None]
        if self.dim == 2:
            Bz = self.B[2]
            return q * Bz * np.stack([V[..., 1], -V[..., 0]], axis=-1)
        return q * np.cross(V, np.broadcast_to(self.B, V.shape))

    def _step_rk4(self, X: np.ndarray, P: np.ndarray, t: float, h: float, hist: Optional[_History]):
        def rhs(Xs, Ps, ts):
            return self.kin.velocity(Ps), self.force.forces(Xs, ts, hist) + self._magnetic_force(Ps)

        k1x, k1p = rhs(X, P, t)
        k2x, k2p = rhs(X + 0.5 * h * k1x, P + 0.5 * h * k1p, t + 0.5 * h)
        k3x, k3p = rhs(X + 0.5 * h * k2x, P + 0.5 * h * k2p, t + 0.5 * h)
        k4x, k4p = rhs(X + h * k3x, P + h * k3p, t + h)
        Xn = X + h / 6.0 * (k1x + 2 * k2x + 2 * k3x + k4x)
        Pn = P + h / 6.0 * (k1p + 2 * k2p + 2 * k3p + k4p)
        Xn, Pn = self._drift(Xn, Pn, 0.0)  # apply boundary conditions
        if self.has_O:
            Pn = self._O(Pn, h)
        if hist is not None:
            hist.push(X)
        return Xn, Pn

    # -- diagnostics ------------------------------------------------------------------
    def _to3(self, A: np.ndarray) -> np.ndarray:
        if A.shape[-1] == 3:
            return A
        out = np.zeros(A.shape[:-1] + (3,))
        out[..., : A.shape[-1]] = A
        return out

    def _diagnostics(self, X: np.ndarray, P: np.ndarray, t: float, F: np.ndarray) -> Dict[str, Any]:
        X0, P0 = X[0], P[0]
        T = float(np.sum(self.kin.T(P[:1])))
        U = float(self.force.potential(X[:1], t)[0])
        X3, P3 = self._to3(X0), self._to3(P0)
        q = self.charges[:, None]
        if self.has_R:
            mom = np.sum(P3 - q * np.cross(X3, self.B), axis=0)
            r2 = np.sum(X3 * X3, axis=1, keepdims=True)
            rB = np.sum(X3 * self.B, axis=1, keepdims=True)
            ang = np.sum(np.cross(X3, P3), axis=0) + np.sum(0.5 * q * (self.B[None, :] * r2 - X3 * rB), axis=0)
        else:
            mom = np.sum(P3, axis=0)
            ang = np.sum(np.cross(X3, P3), axis=0)
        V = self.kin.velocity(P[:1])[0]
        pv = float(np.sum(P0 * V))
        if self.cfg.boundary == "periodic":
            vir = float("nan")
        else:
            xc = X0 - np.average(X0, axis=0, weights=self.masses)
            vir = -float(np.sum(xc * F[0])) / pv if pv > 1e-300 else float("nan")
        if self.force.confining:
            bound = 1.0
        else:
            e = self.kin.T(P[:1])[0] + self.force.particle_energies(X[:1], t)[0]
            bound = float(np.mean(e < 0))
        return {"T": T, "U": U, "mom": mom, "ang": ang, "vir": vir, "bound": bound, "V": V,
                "pabs": float(np.sum(np.linalg.norm(P0, axis=1))),
                "labs": float(np.sum(np.linalg.norm(X3, axis=1) * np.linalg.norm(P3, axis=1)))}

    # -- main loop --------------------------------------------------------------------
    def run(self, progress: bool = False, initial: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> SimulationResult:
        """Integrate the universe.  ``initial = (X, P)`` overrides the sampled
        initial conditions (positions and canonical momenta, shape (N, d))."""
        cfg = self.cfg
        t0 = time.time()
        X0, P0 = self.initial_conditions() if initial is None else (np.asarray(initial[0], float).copy(),
                                                                      np.asarray(initial[1], float).copy())
        B = 2 if cfg.lyapunov else 1
        X = np.repeat(X0[None], B, axis=0)
        P = np.repeat(P0[None], B, axis=0)
        mbar = float(np.mean(self.masses))
        if cfg.lyapunov:
            dq = self.rng.normal(size=X0.shape)
            dp = self.rng.normal(size=P0.shape) * mbar
            norm = math.sqrt(np.sum(dq * dq) + np.sum((dp / mbar) ** 2))
            X[1] += cfg.lyapunov_delta0 * dq / norm
            P[1] += cfg.lyapunov_delta0 * dp / norm
        hist = _History(X, min(cfg.max_history, cfg.steps + 2)) if self.genome.interaction.propagation == "retarded" else None
        t = 0.0
        F = self.force.forces(X, t, hist)
        rec: Dict[str, List[Any]] = {k: [] for k in ("t", "X", "V", "T", "U", "mom", "ang", "vir", "bound")}
        pabs, labs = [], []
        lyap_sum, lyap_series = 0.0, []
        blew_up, t_blow = False, None
        h = cfg.dt

        def record(F_now: np.ndarray) -> None:
            dgn = self._diagnostics(X, P, t, F_now)
            rec["t"].append(t)
            rec["X"].append(X[0].copy())
            rec["V"].append(dgn["V"].copy())
            for key in ("T", "U", "mom", "ang", "vir", "bound"):
                rec[key].append(dgn[key])
            pabs.append(dgn["pabs"])
            labs.append(dgn["labs"])

        record(F)
        for step in range(1, cfg.steps + 1):
            if self.integrator == "leapfrog":
                X, P, F = self._step_splitting(X, P, F, t, h, hist)
            elif self.integrator == "yoshida4":
                tt = t
                for w in (self.YOSHIDA_W1, self.YOSHIDA_W0, self.YOSHIDA_W1):
                    X, P, F = self._step_splitting(X, P, F, tt, w * h, hist)
                    tt += w * h
            else:
                X, P = self._step_rk4(X, P, t, h, hist)
                F = self.force.forces(X, t + h, hist)
            t += h
            if cfg.lyapunov and step % cfg.lyapunov_renorm_every == 0:
                dq = X[1] - X[0]
                if cfg.boundary == "periodic":
                    dq = dq - cfg.box_size * np.round(dq / cfg.box_size)
                dp = P[1] - P[0]
                dist = math.sqrt(float(np.sum(dq * dq) + np.sum((dp / mbar) ** 2)))
                if dist > 0 and np.isfinite(dist):
                    lyap_sum += math.log(dist / cfg.lyapunov_delta0)
                    lyap_series.append(lyap_sum / t)
                    X[1] = X[0] + dq * (cfg.lyapunov_delta0 / dist)
                    P[1] = P[0] + dp * (cfg.lyapunov_delta0 / dist)
            if step % cfg.record_every == 0 or step == cfg.steps:
                finite = np.all(np.isfinite(X)) and np.all(np.isfinite(P))
                vmax = float(np.max(np.linalg.norm(self.kin.velocity(P[:1]), axis=-1))) if finite else np.inf
                if not finite or vmax > cfg.max_speed:
                    blew_up, t_blow = True, t
                    break
                record(F)
            if progress and step % max(1, cfg.steps // 10) == 0:
                print(f"  step {step}/{cfg.steps}  t={t:.3f}")
        ref_len = float(np.sqrt(np.mean(np.sum((X0 - X0.mean(0)) ** 2, axis=1))))
        Ts, Us = np.array(rec["T"]), np.array(rec["U"])
        e_scale = float(max(np.median(Ts + np.abs(Us)), 1e-12))
        res = SimulationResult(
            genome_id=self.genome.genome_id,
            config=cfg.to_dict() | {"integrator_used": self.integrator},
            dim=self.dim,
            times=np.array(rec["t"]),
            positions=np.array(rec["X"]),
            velocities=np.array(rec["V"]),
            masses=self.masses,
            charges=self.charges,
            species=self.species,
            kinetic=Ts,
            potential=Us,
            momentum=np.array(rec["mom"]),
            angular_momentum=np.array(rec["ang"]),
            virial_ratio=np.array(rec["vir"]),
            bound_fraction=np.array(rec["bound"]),
            noether=self.noether,
            lyapunov=(lyap_sum / t) if (cfg.lyapunov and t > 0 and lyap_series) else None,
            lyapunov_series=np.array(lyap_series) if lyap_series else None,
            blew_up=blew_up,
            blowup_time=t_blow,
            wall_time=time.time() - t0,
            reference_length=ref_len,
            energy_scale=e_scale,
            momentum_scale=float(max(np.mean(pabs), 1e-12)),
            angular_scale=float(max(np.mean(labs), 1e-12)),
        )
        return res


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@dataclass
class UniverseEvaluation:
    stability: float
    conservation_consistency: float
    feasibility: float
    conservation: Dict[str, Dict[str, Any]]
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


#: Relative-drift tolerances used to decide whether a quantity is "measured
#: conserved".  Mechanical momentum and angular momentum are exact invariants
#: of the splitting for central reciprocal forces, so the tolerance is tight;
#: energy and canonical (magnetic) quantities carry O(dt^2) bounded error.
CONSERVATION_TOL = {"energy": 2e-2, "momentum": 1e-6, "angular_momentum": 1e-6, "canonical": 2e-2}


def evaluate_simulation(res: SimulationResult) -> UniverseEvaluation:
    """Score a particle run: stability, Noether consistency, cosmic feasibility."""
    n = res.noether
    out: Dict[str, Dict[str, Any]] = {}
    if len(res.times) >= 2:
        E = res.energy
        dE = float(np.max(np.abs(E - E[0])) / res.energy_scale)
        dP = float(np.max(np.linalg.norm(res.momentum - res.momentum[0], axis=1)) / res.momentum_scale)
        axes = np.array(n.angular_axes) if n.angular_axes else np.eye(3)
        Lproj = res.angular_momentum @ axes.T
        dL = float(np.max(np.abs(Lproj - Lproj[0])) / res.angular_scale)
    else:
        dE = dP = dL = float("nan")
    tolP = CONSERVATION_TOL["canonical"] if n.momentum_kind == "pseudo" else CONSERVATION_TOL["momentum"]
    tolL = CONSERVATION_TOL["canonical"] if n.angular_kind == "canonical" else CONSERVATION_TOL["angular_momentum"]
    for key, pred, drift, tol in (("energy", n.energy, dE, CONSERVATION_TOL["energy"]),
                                  ("momentum", n.momentum, dP, tolP),
                                  ("angular_momentum", n.angular_momentum, dL, tolL)):
        measured = bool(np.isfinite(drift) and drift < tol)
        out[key] = {"predicted": bool(pred), "relative_drift": drift, "tolerance": tol,
                    "measured_conserved": measured,
                    "consistent": (measured if pred else True)}
    preds = [v for v in out.values() if v["predicted"]]
    consistency = float(np.mean([v["consistent"] for v in preds])) if preds else 1.0
    survival = (res.blowup_time / (res.config["steps"] * res.config["dt"])) if res.blew_up else 1.0
    collapse = 0.0
    if len(res.positions):
        Xf = res.positions[-1]
        rg = float(np.sqrt(np.mean(np.sum((Xf - Xf.mean(0)) ** 2, axis=1))))
        if rg < 2.0 * res.config["softening"]:
            collapse = 0.5
    else:
        rg = float("nan")
    stability = float(survival * (1.0 - collapse))
    if n.energy and out["energy"]["relative_drift"] > 0.2:
        stability *= 0.5  # numerically unresolved dynamics
    bound = float(res.bound_fraction[-1]) if len(res.bound_fraction) else 0.0
    vir = res.virial_ratio[-max(1, len(res.virial_ratio) // 4):]
    vir = vir[np.isfinite(vir)]
    vir_dev = float(abs(np.mean(vir) - 1.0)) if vir.size else 1.0
    feasibility = float(survival * math.sqrt(max(bound, 0.0)) * math.exp(-min(vir_dev, 5.0)) * (1.0 - collapse))
    return UniverseEvaluation(
        stability=stability,
        conservation_consistency=consistency,
        feasibility=feasibility,
        conservation=out,
        details={"survival": survival, "final_rg": rg, "bound_fraction": bound, "virial_deviation": vir_dev,
                 "lyapunov": res.lyapunov},
    )


# ---------------------------------------------------------------------------
# Rigid bodies
# ---------------------------------------------------------------------------

def _qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = np.moveaxis(a, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(b, -1, 0)
    return np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], axis=-1)


def _qrot(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate body-frame vectors v by unit quaternions q (body -> space)."""
    qv = np.concatenate([np.zeros(v.shape[:-1] + (1,)), v], axis=-1)
    qc = q * np.array([1.0, -1.0, -1.0, -1.0])
    return _qmul(_qmul(q, qv), qc)[..., 1:]


def _axis_angle_quat(axis: np.ndarray, angle: np.ndarray) -> np.ndarray:
    half = 0.5 * angle[..., None]
    return np.concatenate([np.cos(half), np.sin(half) * axis], axis=-1)


@dataclass
class RigidBodyConfig:
    n_bodies: int = 4
    steps: int = 6000
    dt: float = 2e-3
    record_every: int = 10
    spin: float = 6.0
    perturbation: float = 1e-3
    internal_dissipation: Optional[float] = None  # None -> derived from genome.thermo.friction
    seed: int = 0


@dataclass
class RigidBodyResult:
    times: np.ndarray
    omega_body: np.ndarray  # (R, nb, 3)
    L_space: np.ndarray  # (R, nb, 3)
    energy_rot: np.ndarray  # (R, nb)
    com: np.ndarray  # (R, nb, 3)
    inertia: np.ndarray  # (nb, 3)
    flips: np.ndarray  # (nb,) number of intermediate-axis sign reversals
    final_axis_alignment: np.ndarray  # (nb,) |L_body . e_major| / |L|
    L_norm_drift: float
    dissipation: float

    def summary(self) -> Dict[str, Any]:
        return {"flips": self.flips.tolist(), "final_major_axis_alignment": self.final_axis_alignment.round(4).tolist(),
                "L_norm_drift": self.L_norm_drift, "dissipation": self.dissipation}


class RigidBodyEnsemble:
    """Rigid bodies spinning near their unstable intermediate axis.

    Free rotation uses the exact split flows of H_i = L_i^2 / 2 I_i, composed
    as R1(h/2) R2(h/2) R3(h) R2(h/2) R1(h/2), which preserves |L| exactly and
    keeps energy error bounded.  Internal dissipation dL/dt = eta L x (L x Omega)
    (orthogonal to L, so |L| is preserved while energy decays) is applied with
    the compensating counter-rotation of the attitude so that the *spatial*
    angular momentum stays fixed - producing the major-axis rule.
    """

    def __init__(self, genome: PhysicsGenome, config: Optional[RigidBodyConfig] = None):
        self.genome = genome
        self.cfg = config or RigidBodyConfig()
        self.rng = np.random.default_rng(self.cfg.seed)

    def run(self) -> RigidBodyResult:
        cfg, nb = self.cfg, self.cfg.n_bodies
        base = np.array([1.0, 2.0, 3.0])
        I = base[None, :] * self.rng.uniform(0.9, 1.1, (nb, 3))  # keep moments well separated
        I = np.sort(I, axis=1)
        L = np.zeros((nb, 3))
        L[:, 1] = cfg.spin * I[:, 1]
        L += self.rng.normal(0, cfg.perturbation * cfg.spin, (nb, 3))
        q = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (nb, 1))
        eta = cfg.internal_dissipation
        if eta is None:
            eta = 0.02 * float(self.genome.thermo.friction)
        # Centre-of-mass motion under the mutated gravity (bodies on a ring).
        sim_cfg = SimulationConfig(n_particles=nb, dim=3, dt=cfg.dt, softening=0.1, lyapunov=False)
        masses = np.full(nb, 1.0 / nb)
        force = ForceModel(self.genome, sim_cfg, masses, np.zeros(nb), np.zeros(nb, dtype=int), 3)
        ang = 2 * np.pi * np.arange(nb) / nb
        X = np.stack([np.cos(ang), np.sin(ang), np.zeros(nb)], axis=1)[None]
        Fr = force.forces(X, 0.0)[0]
        vc = np.sqrt(np.maximum(-np.sum(Fr * X[0], axis=1) / masses, 0.0))
        P = (np.stack([-np.sin(ang), np.cos(ang), np.zeros(nb)], axis=1) * (vc * masses)[:, None])[None]
        F = force.forces(X, 0.0)
        h = cfg.dt
        e = np.eye(3)
        recs: Dict[str, List[Any]] = {k: [] for k in ("t", "w", "Ls", "E", "x")}
        L0n = np.linalg.norm(L, axis=1)

        def axis_flow(L: np.ndarray, q: np.ndarray, i: int, tau: float) -> Tuple[np.ndarray, np.ndarray]:
            ang_i = L[:, i] / I[:, i] * tau
            c, s = np.cos(-ang_i), np.sin(-ang_i)
            j, k = (i + 1) % 3, (i + 2) % 3
            Lj, Lk = L[:, j].copy(), L[:, k].copy()
            L = L.copy()
            L[:, j] = c * Lj - s * Lk
            L[:, k] = s * Lj + c * Lk
            q = _qmul(q, _axis_angle_quat(np.broadcast_to(e[i], (nb, 3)), ang_i))
            return L, q

        def record(t: float) -> None:
            recs["t"].append(t)
            recs["w"].append(L / I)
            recs["Ls"].append(_qrot(q, L))
            recs["E"].append(0.5 * np.sum(L * L / I, axis=1))
            recs["x"].append(X[0].copy())

        record(0.0)
        t = 0.0
        for step in range(1, cfg.steps + 1):
            for i, tau in ((0, 0.5 * h), (1, 0.5 * h), (2, h), (1, 0.5 * h), (0, 0.5 * h)):
                L, q = axis_flow(L, q, i, tau)
            if eta > 0:
                W = L / I
                dL = eta * np.cross(L, np.cross(L, W)) * h
                Ln = L + dL
                Ln *= (np.linalg.norm(L, axis=1) / np.linalg.norm(Ln, axis=1))[:, None]
                ax = np.cross(L, Ln)
                s = np.linalg.norm(ax, axis=1)
                ok = s > 1e-300
                if np.any(ok):
                    angle = np.arctan2(s, np.sum(L * Ln, axis=1))
                    axis = np.where(ok[:, None], ax / np.maximum(s, 1e-300)[:, None], e[0])
                    # body-frame rotation S maps L -> Ln; compensate attitude with S^-1
                    q = _qmul(q, _axis_angle_quat(axis, -angle * ok))
                L = Ln
            q /= np.linalg.norm(q, axis=1, keepdims=True)
            P = P + 0.5 * h * F
            X = X + h * P / masses[None, :, None]
            F = force.forces(X, t + h)
            P = P + 0.5 * h * F
            t += h
            if step % cfg.record_every == 0:
                record(t)
        w = np.array(recs["w"])
        signs = np.sign(w[:, :, 1])
        flips = np.sum(np.abs(np.diff(signs, axis=0)) > 0, axis=0)
        Lb = np.array(L)
        align = np.abs(Lb[:, 2]) / np.linalg.norm(Lb, axis=1)
        drift = float(np.max(np.abs(np.linalg.norm(L, axis=1) - L0n) / L0n))
        return RigidBodyResult(times=np.array(recs["t"]), omega_body=w, L_space=np.array(recs["Ls"]),
                               energy_rot=np.array(recs["E"]), com=np.array(recs["x"]), inertia=I,
                               flips=flips, final_axis_alignment=align, L_norm_drift=drift, dissipation=eta)


class PyBulletRigidBackend:
    """Optional PyBullet rigid-body sandbox under the genome's surface gravity.

    Boxes of random size are dropped onto a plane with gravity scaled by the
    genome's force law evaluated at one length unit.  Only used when
    ``pybullet`` is importable; returns ``None`` otherwise.
    """

    available = importlib.util.find_spec("pybullet") is not None

    def __init__(self, genome: PhysicsGenome, n_boxes: int = 8, seed: int = 0):
        self.genome, self.n_boxes, self.seed = genome, n_boxes, seed

    def gravity_ratio(self) -> float:
        sym = SymbolicPhysics(self.genome)
        return float(abs(sym.force_profile(np.array([1.0]))[0]))

    def run(self, steps: int = 1200) -> Optional[Dict[str, Any]]:
        if not self.available:
            return None
        import pybullet as pb  # type: ignore

        rng = np.random.default_rng(self.seed)
        cid = pb.connect(pb.DIRECT)
        try:
            pb.setGravity(0, 0, -9.81 * self.gravity_ratio(), physicsClientId=cid)
            plane = pb.createCollisionShape(pb.GEOM_PLANE, physicsClientId=cid)
            pb.createMultiBody(0, plane, physicsClientId=cid)
            bodies = []
            for i in range(self.n_boxes):
                he = rng.uniform(0.05, 0.2, 3)
                col = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=he.tolist(), physicsClientId=cid)
                pos = [rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5), 0.5 + 0.5 * i]
                orn = pb.getQuaternionFromEuler(rng.uniform(0, np.pi, 3).tolist())
                bodies.append(pb.createMultiBody(float(np.prod(he) * 8000), col, -1, pos, orn, physicsClientId=cid))
            traj = []
            for _ in range(steps):
                pb.stepSimulation(physicsClientId=cid)
                traj.append([pb.getBasePositionAndOrientation(b, physicsClientId=cid)[0] for b in bodies])
            traj = np.array(traj)
            settle = float(np.mean(np.linalg.norm(traj[-1] - traj[-50], axis=-1)))
            return {"trajectory": traj, "final_heights": traj[-1, :, 2], "residual_motion": settle,
                    "gravity_ratio": self.gravity_ratio()}
        finally:
            pb.disconnect(physicsClientId=cid)


# ---------------------------------------------------------------------------
# Continuum fields
# ---------------------------------------------------------------------------

@dataclass
class FieldResult:
    kind: str
    n: int
    times: np.ndarray
    snapshots: np.ndarray  # (S, n, n)
    final: np.ndarray
    energy: np.ndarray
    enstrophy: np.ndarray
    spectrum_k: np.ndarray
    spectrum_E: np.ndarray
    params: Dict[str, Any]
    blew_up: bool = False
    wall_time: float = 0.0


def _shell_spectrum(field2d: np.ndarray, weight_by_k2_inv: bool) -> Tuple[np.ndarray, np.ndarray]:
    n = field2d.shape[0]
    fh = np.fft.fft2(field2d) / n ** 2
    k = np.fft.fftfreq(n) * n
    KX, KY = np.meshgrid(k, k, indexing="ij")
    K = np.sqrt(KX ** 2 + KY ** 2)
    power = np.abs(fh) ** 2
    if weight_by_k2_inv:  # energy spectrum from vorticity: |u_k|^2 = |w_k|^2 / k^2
        with np.errstate(divide="ignore", invalid="ignore"):
            power = np.where(K > 0, power / K ** 2, 0.0)
    kb = np.rint(K).astype(int)
    kmax = n // 2
    E = np.bincount(kb.ravel(), weights=0.5 * power.ravel(), minlength=kmax + 1)[: kmax + 1]
    return np.arange(kmax + 1, dtype=float), E


class VorticityField2D:
    """Pseudo-spectral 2-D incompressible flow with mutated dissipation.

    ``d_t w_k = -D(k) w_k + NL_k``,  ``D(k) = nu (k^T N k)^h + alpha``; the
    stiff linear part is integrated exactly (integrating factor) and the
    nonlinear advection with classical RK4, adaptive CFL time step.
    """

    def __init__(self, genome: PhysicsGenome, n: int = 64, seed: int = 0, cfl: float = 0.4):
        f = genome.fluid
        self.f, self.n, self.cfl = f, n, cfl
        self.rng = np.random.default_rng(seed)
        k = np.fft.fftfreq(n) * n
        kr = np.fft.rfftfreq(n) * n
        self.KX, self.KY = np.meshgrid(k, kr, indexing="ij")
        self.K2 = self.KX ** 2 + self.KY ** 2
        self.K2inv = np.where(self.K2 > 0, 1.0 / np.where(self.K2 > 0, self.K2, 1.0), 0.0)
        N = f.N()
        kNk = N[0, 0] * self.KX ** 2 + 2 * N[0, 1] * self.KX * self.KY + N[1, 1] * self.KY ** 2
        self.D = f.viscosity * np.maximum(kNk, 0.0) ** f.hyper_order + f.ekman_drag
        self.dealias = (np.abs(self.KX) < n / 3.0) & (np.abs(self.KY) < n / 3.0)
        x = np.arange(n) * 2 * np.pi / n
        Xg, Yg = np.meshgrid(x, x, indexing="ij")
        kf = int(f.forcing_k)
        forcing = -f.forcing_amp * kf * np.cos(kf * Yg)  # curl of F0 sin(kf y) x-hat
        self.f_hat = np.fft.rfft2(forcing)
        self.dx = 2 * np.pi / n

    def _nonlinear(self, wh: np.ndarray) -> np.ndarray:
        psi = wh * self.K2inv
        u = np.fft.irfft2(1j * self.KY * psi, s=(self.n, self.n))
        v = np.fft.irfft2(-1j * self.KX * psi, s=(self.n, self.n))
        wx = np.fft.irfft2(1j * self.KX * wh, s=(self.n, self.n))
        wy = np.fft.irfft2(1j * self.KY * wh, s=(self.n, self.n))
        adv = np.fft.rfft2(u * wx + v * wy) * self.dealias
        return -self.f.advection * adv + self.f_hat

    def _velocity_max(self, wh: np.ndarray) -> float:
        psi = wh * self.K2inv
        u = np.fft.irfft2(1j * self.KY * psi, s=(self.n, self.n))
        v = np.fft.irfft2(-1j * self.KX * psi, s=(self.n, self.n))
        return float(np.max(np.sqrt(u * u + v * v)))

    def run(self, t_end: float = 30.0, n_snapshots: int = 8, max_steps: int = 20000) -> FieldResult:
        t0 = time.time()
        n = self.n
        # random initial vorticity with spectrum peaked at k ~ 6
        K = np.sqrt(self.K2)
        amp = K ** 2 * np.exp(-((K / 6.0) ** 2))
        phase = np.exp(2j * np.pi * self.rng.random(self.K2.shape))
        wh = amp * phase * self.dealias
        wh[0, 0] = 0.0
        w = np.fft.irfft2(wh, s=(n, n))
        w *= 2.0 / max(np.std(w), 1e-12)
        wh = np.fft.rfft2(w)
        t, steps = 0.0, 0
        snaps, stimes, Es, Zs = [], [], [], []
        snap_times = np.linspace(0, t_end, n_snapshots)
        si = 0
        blew = False
        while t < t_end and steps < max_steps:
            if si < len(snap_times) and t >= snap_times[si] - 1e-12:
                w = np.fft.irfft2(wh, s=(n, n))
                snaps.append(w)
                stimes.append(t)
                si += 1
            umax = self._velocity_max(wh)
            dt = min(self.cfl * self.dx / max(umax, 1e-6), 0.1, t_end - t + 1e-12)
            E = np.exp(-self.D * dt)
            E2 = np.exp(-self.D * dt / 2)
            k1 = self._nonlinear(wh)
            k2 = self._nonlinear(E2 * (wh + 0.5 * dt * k1))
            k3 = self._nonlinear(E2 * wh + 0.5 * dt * k2)
            k4 = self._nonlinear(E * wh + dt * E2 * k3)
            wh = E * wh + dt / 6.0 * (E * k1 + 2 * E2 * (k2 + k3) + k4)
            wh *= self.dealias
            t += dt
            steps += 1
            if steps % 5 == 0:
                w = np.fft.irfft2(wh, s=(n, n))
                psi = np.fft.irfft2(wh * self.K2inv, s=(n, n))
                Zs.append(0.5 * float(np.mean(w * w)))
                Es.append(0.5 * float(np.mean(psi * w)))
                if not np.isfinite(Zs[-1]) or Zs[-1] > 1e8:
                    blew = True
                    break
        w = np.fft.irfft2(wh, s=(n, n))
        if len(snaps) < n_snapshots and not blew:
            snaps.append(w)
            stimes.append(t)
        k, Ek = _shell_spectrum(w, weight_by_k2_inv=True)
        return FieldResult("vorticity", n, np.array(stimes), np.array(snaps), w, np.array(Es), np.array(Zs), k, Ek,
                           params={"nu": self.f.viscosity, "h": self.f.hyper_order, "lambda": self.f.advection,
                                   "N": self.f.N().tolist(), "steps": steps, "t_end": t},
                           blew_up=blew, wall_time=time.time() - t0)


class ReactionDiffusion2D:
    """Gray-Scott reaction-diffusion with an anisotropic diffusion tensor.

    ``u_t = Du div(N grad u) - u v^2 + F (1-u)``,
    ``v_t = Dv div(N grad v) + u v^2 - (F+k) v``  (Pearson 1993 parameterisation).
    """

    def __init__(self, genome: PhysicsGenome, n: int = 96, seed: int = 0):
        self.f, self.n = genome.fluid, n
        self.rng = np.random.default_rng(seed)
        self.N = self.f.N()

    def _lap(self, a: np.ndarray) -> np.ndarray:
        N = self.N
        axx = np.roll(a, 1, 0) + np.roll(a, -1, 0) - 2 * a
        ayy = np.roll(a, 1, 1) + np.roll(a, -1, 1) - 2 * a
        axy = 0.25 * (np.roll(np.roll(a, -1, 0), -1, 1) - np.roll(np.roll(a, -1, 0), 1, 1)
                      - np.roll(np.roll(a, 1, 0), -1, 1) + np.roll(np.roll(a, 1, 0), 1, 1))
        return N[0, 0] * axx + 2 * N[0, 1] * axy + N[1, 1] * ayy

    def run(self, steps: int = 5000, n_snapshots: int = 8) -> FieldResult:
        t0 = time.time()
        f, n = self.f, self.n
        u = np.ones((n, n))
        v = np.zeros((n, n))
        for _ in range(6):
            cx, cy = self.rng.integers(0, n, 2)
            s = max(3, n // 16)
            u[cx - s:cx + s, cy - s:cy + s] = 0.5
            v[cx - s:cx + s, cy - s:cy + s] = 0.25
        u += 0.02 * self.rng.random((n, n))
        v += 0.02 * self.rng.random((n, n))
        Dmax = max(f.rd_diffusion_u, f.rd_diffusion_v)
        N = self.N
        dt = min(1.0, 0.9 / (Dmax * (2 * N[0, 0] + 2 * N[1, 1] + 2 * abs(N[0, 1]))))
        snaps, times, mass = [], [], []
        snap_steps = set(np.linspace(0, steps, n_snapshots).astype(int).tolist())
        blew = False
        for step in range(steps + 1):
            if step in snap_steps:
                snaps.append(v.copy())
                times.append(step * dt)
            if step == steps:
                break
            uvv = u * v * v
            u = u + dt * (f.rd_diffusion_u * self._lap(u) - uvv + f.rd_feed * (1 - u))
            v = v + dt * (f.rd_diffusion_v * self._lap(v) + uvv - (f.rd_feed + f.rd_kill) * v)
            if step % 50 == 0:
                mass.append(float(np.mean(v)))
                if not np.isfinite(mass[-1]):
                    blew = True
                    break
        k, Ek = _shell_spectrum(v - v.mean(), weight_by_k2_inv=False)
        return FieldResult("reaction_diffusion", n, np.array(times), np.array(snaps), v, np.array(mass),
                           np.array([float(np.std(v))]), k, Ek,
                           params={"F": f.rd_feed, "k": f.rd_kill, "Du": f.rd_diffusion_u, "Dv": f.rd_diffusion_v,
                                   "N": N.tolist(), "dt": dt, "steps": steps},
                           blew_up=blew, wall_time=time.time() - t0)


# ---------------------------------------------------------------------------
# Whole-universe convenience
# ---------------------------------------------------------------------------

@dataclass
class UniverseRun:
    genome: PhysicsGenome
    particles: SimulationResult
    evaluation: UniverseEvaluation
    rigid: Optional[RigidBodyResult] = None
    fields: Dict[str, FieldResult] = field(default_factory=dict)
    pybullet: Optional[Dict[str, Any]] = None


def run_universe(genome: PhysicsGenome, sim_config: Optional[SimulationConfig] = None,
                 fields: bool = True, rigid: bool = True, pybullet: bool = False,
                 field_resolution: int = 64, rd_steps: int = 4000, vort_t_end: float = 25.0) -> UniverseRun:
    """Run every level of a universe and evaluate it."""
    res = UniverseSimulator(genome, sim_config).run()
    ev = evaluate_simulation(res)
    run = UniverseRun(genome=genome, particles=res, evaluation=ev)
    seed = (sim_config.seed if sim_config else 0)
    if rigid:
        run.rigid = RigidBodyEnsemble(genome, RigidBodyConfig(seed=seed)).run()
    if fields:
        run.fields["vorticity"] = VorticityField2D(genome, n=field_resolution, seed=seed).run(t_end=vort_t_end)
        run.fields["reaction_diffusion"] = ReactionDiffusion2D(genome, n=max(64, field_resolution), seed=seed).run(steps=rd_steps)
    if pybullet and PyBulletRigidBackend.available:
        run.pybullet = PyBulletRigidBackend(genome, seed=seed).run()
    return run
