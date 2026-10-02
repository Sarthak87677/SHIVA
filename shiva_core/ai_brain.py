"""
SHIVA-1 :: AI Brain
===================

The learning and search machinery of SHIVA-1, implemented from first
principles (NumPy only) so every algorithm is inspectable:

==========================  ====================================================
Component                   Reference
==========================  ====================================================
``GeneticAlgorithm``        real-coded GA, SBX crossover + polynomial mutation
                            (Deb & Agrawal 1995), tournament selection, elitism
``CMAES``                   (mu/mu_w, lambda)-CMA-ES with rank-one and rank-mu
                            updates and cumulative step-size adaptation
                            (Hansen & Ostermeier 2001; Hansen 2016 tutorial)
``NSGA2``                   fast non-dominated sorting, crowding distance,
                            constrained domination (Deb et al. 2002)
``NEATPopulation``          NEAT: historical markings, speciation, fitness
                            sharing, complexification (Stanley & Miikkulainen 2002)
``CPPNShapeGenerator``      compositional pattern-producing networks evolved by
                            NEAT as a generative shape model (Stanley 2007)
``LatentDesignModel``       probabilistic PCA latent generative model fitted to
                            elite designs (Tipping & Bishop 1999)
``DesignRLAgent``           REINFORCE with baseline, linear-Gaussian policy,
                            Adam optimiser (Williams 1992; Kingma & Ba 2015)
``OperatorBandit``          UCB1 adaptive operator selection (Auer et al. 2002)
``ShivaBrain``              meta-level universe search: evolutionary search over
                            physics genomes with novelty archive (Lehman &
                            Stanley 2011), bandit-chosen mutation operators,
                            1/5th-success-rule step adaptation (Rechenberg 1973)
                            and CMA-ES refinement of continuous constants.
==========================  ====================================================

All optimisers *maximise* unless stated otherwise (CMA-ES minimises
internally and exposes a maximising ``optimize`` wrapper).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "GeneticAlgorithm",
    "CMAES",
    "NSGA2",
    "NSGA2Result",
    "fast_non_dominated_sort",
    "crowding_distance",
    "hypervolume_2d",
    "NodeGene",
    "ConnGene",
    "InnovationTracker",
    "NEATGenome",
    "FeedForwardNetwork",
    "NEATPopulation",
    "CPPNShapeGenerator",
    "LatentDesignModel",
    "DesignEnvironment",
    "DesignRLAgent",
    "OperatorBandit",
    "UniverseCandidate",
    "ShivaBrain",
]


# ---------------------------------------------------------------------------
# Variation operators
# ---------------------------------------------------------------------------

def sbx_crossover(p1: np.ndarray, p2: np.ndarray, lo: np.ndarray, hi: np.ndarray, eta: float,
                  rng: np.random.Generator, prob: float = 0.9) -> Tuple[np.ndarray, np.ndarray]:
    """Simulated binary crossover (per-variable, with swap)."""
    c1, c2 = p1.copy(), p2.copy()
    if rng.random() > prob:
        return c1, c2
    u = rng.random(p1.shape)
    beta = np.where(u <= 0.5, (2 * u) ** (1 / (eta + 1)), (1 / (2 * (1 - u))) ** (1 / (eta + 1)))
    active = rng.random(p1.shape) < 0.5
    a = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
    b = 0.5 * ((1 - beta) * p1 + (1 + beta) * p2)
    c1 = np.where(active, a, p1)
    c2 = np.where(active, b, p2)
    return np.clip(c1, lo, hi), np.clip(c2, lo, hi)


def polynomial_mutation(x: np.ndarray, lo: np.ndarray, hi: np.ndarray, eta: float, p: float,
                        rng: np.random.Generator) -> np.ndarray:
    u = rng.random(x.shape)
    delta = np.where(u < 0.5, (2 * u) ** (1 / (eta + 1)) - 1, 1 - (2 * (1 - u)) ** (1 / (eta + 1)))
    mask = rng.random(x.shape) < p
    return np.clip(np.where(mask, x + delta * (hi - lo), x), lo, hi)


# ---------------------------------------------------------------------------
# Genetic algorithm
# ---------------------------------------------------------------------------

class GeneticAlgorithm:
    """Elitist real-coded GA with an ask/tell interface (maximisation)."""

    def __init__(self, n_dim: int, bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None, pop_size: int = 32,
                 elite: int = 2, eta_c: float = 15.0, eta_m: float = 20.0, p_mut: Optional[float] = None,
                 tournament: int = 2, seed: Optional[int] = None):
        self.n = n_dim
        self.lo, self.hi = (np.zeros(n_dim), np.ones(n_dim)) if bounds is None else (np.asarray(bounds[0], float),
                                                                                       np.asarray(bounds[1], float))
        self.pop_size, self.elite, self.eta_c, self.eta_m = pop_size, elite, eta_c, eta_m
        self.p_mut = p_mut if p_mut is not None else 1.0 / n_dim
        self.k = tournament
        self.rng = np.random.default_rng(seed)
        self.pop: Optional[np.ndarray] = None
        self.fit: Optional[np.ndarray] = None
        self.best_x: Optional[np.ndarray] = None
        self.best_f = -np.inf
        self.history: List[Dict[str, float]] = []

    def _tournament(self) -> np.ndarray:
        idx = self.rng.integers(0, len(self.pop), self.k)
        return self.pop[idx[np.argmax(self.fit[idx])]]

    def ask(self) -> np.ndarray:
        if self.pop is None:
            return self.lo + self.rng.random((self.pop_size, self.n)) * (self.hi - self.lo)
        kids = []
        while len(kids) < self.pop_size:
            c1, c2 = sbx_crossover(self._tournament(), self._tournament(), self.lo, self.hi, self.eta_c, self.rng)
            kids.append(polynomial_mutation(c1, self.lo, self.hi, self.eta_m, self.p_mut, self.rng))
            kids.append(polynomial_mutation(c2, self.lo, self.hi, self.eta_m, self.p_mut, self.rng))
        return np.array(kids[: self.pop_size])

    def tell(self, X: np.ndarray, f: np.ndarray) -> None:
        f = np.asarray(f, float)
        if self.pop is None:
            self.pop, self.fit = X.copy(), f.copy()
        else:
            allX = np.concatenate([self.pop, X])
            allf = np.concatenate([self.fit, f])
            order = np.argsort(-allf)
            keep = list(order[: self.elite])
            rest = order[self.elite:]
            while len(keep) < self.pop_size:
                idx = self.rng.choice(rest, size=min(self.k, len(rest)), replace=False)
                keep.append(int(idx[np.argmax(allf[idx])]))
            self.pop, self.fit = allX[keep], allf[keep]
        i = int(np.argmax(f))
        if f[i] > self.best_f:
            self.best_f, self.best_x = float(f[i]), X[i].copy()
        self.history.append({"best": self.best_f, "mean": float(np.mean(f))})

    def optimize(self, fn: Callable[[np.ndarray], float], generations: int) -> Tuple[np.ndarray, float]:
        for _ in range(generations):
            X = self.ask()
            self.tell(X, np.array([fn(x) for x in X]))
        return self.best_x, self.best_f


# ---------------------------------------------------------------------------
# CMA-ES
# ---------------------------------------------------------------------------

class CMAES:
    """Covariance Matrix Adaptation Evolution Strategy (minimisation core).

    Bounds are handled by mirroring samples back into the box (repair).
    """

    def __init__(self, x0: Sequence[float], sigma0: float, bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                 popsize: Optional[int] = None, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)
        self.mean = np.asarray(x0, dtype=float).copy()
        n = self.n = self.mean.size
        self.sigma = float(sigma0)
        self.bounds = None if bounds is None else (np.asarray(bounds[0], float), np.asarray(bounds[1], float))
        self.lam = popsize or 4 + int(3 * math.log(n))
        self.mu = self.lam // 2
        w = math.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = w / w.sum()
        self.mueff = 1.0 / np.sum(self.weights ** 2)
        self.cc = (4 + self.mueff / n) / (n + 4 + 2 * self.mueff / n)
        self.cs = (self.mueff + 2) / (n + self.mueff + 5)
        self.c1 = 2 / ((n + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((n + 2) ** 2 + self.mueff))
        self.damps = 1 + 2 * max(0.0, math.sqrt((self.mueff - 1) / (n + 1)) - 1) + self.cs
        self.chiN = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
        self.pc = np.zeros(n)
        self.ps = np.zeros(n)
        self.B = np.eye(n)
        self.D = np.ones(n)
        self.C = np.eye(n)
        self.invsqrtC = np.eye(n)
        self.eigeneval = 0
        self.counteval = 0
        self.best_x: Optional[np.ndarray] = None
        self.best_f = np.inf
        self.history: List[Dict[str, float]] = []

    def _repair(self, x: np.ndarray) -> np.ndarray:
        if self.bounds is None:
            return x
        lo, hi = self.bounds
        span = hi - lo
        y = (x - lo) % (2 * span)
        y = np.where(y > span, 2 * span - y, y)
        return lo + y

    def ask(self) -> np.ndarray:
        z = self.rng.standard_normal((self.lam, self.n))
        y = (z * self.D) @ self.B.T
        return np.array([self._repair(self.mean + self.sigma * yi) for yi in y])

    def tell(self, X: np.ndarray, f: np.ndarray) -> None:
        f = np.asarray(f, float)
        n = self.n
        self.counteval += len(f)
        order = np.argsort(f)
        if f[order[0]] < self.best_f:
            self.best_f, self.best_x = float(f[order[0]]), X[order[0]].copy()
        xsel = X[order[: self.mu]]
        old = self.mean.copy()
        self.mean = self.weights @ xsel
        dm = (self.mean - old) / self.sigma
        self.ps = (1 - self.cs) * self.ps + math.sqrt(self.cs * (2 - self.cs) * self.mueff) * (self.invsqrtC @ dm)
        hsig = (np.linalg.norm(self.ps) / math.sqrt(1 - (1 - self.cs) ** (2 * self.counteval / self.lam)) / self.chiN
                < 1.4 + 2 / (n + 1))
        self.pc = (1 - self.cc) * self.pc + hsig * math.sqrt(self.cc * (2 - self.cc) * self.mueff) * dm
        art = (xsel - old) / self.sigma
        self.C = ((1 - self.c1 - self.cmu) * self.C
                  + self.c1 * (np.outer(self.pc, self.pc) + (1 - hsig) * self.cc * (2 - self.cc) * self.C)
                  + self.cmu * (art.T * self.weights) @ art)
        self.sigma *= math.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chiN - 1))
        self.sigma = float(min(self.sigma, 1e6))
        if self.counteval - self.eigeneval > self.lam / (self.c1 + self.cmu) / n / 10:
            self.eigeneval = self.counteval
            self.C = np.triu(self.C) + np.triu(self.C, 1).T
            evals, self.B = np.linalg.eigh(self.C)
            self.D = np.sqrt(np.maximum(evals, 1e-20))
            self.invsqrtC = self.B @ np.diag(1 / self.D) @ self.B.T
        self.history.append({"best": self.best_f, "mean": float(np.mean(f)), "sigma": self.sigma})

    def optimize(self, fn: Callable[[np.ndarray], float], iterations: int, maximize: bool = True,
                 tol: float = 1e-12) -> Tuple[np.ndarray, float]:
        sgn = -1.0 if maximize else 1.0
        for _ in range(iterations):
            X = self.ask()
            self.tell(X, np.array([sgn * fn(x) for x in X]))
            if self.sigma * np.max(self.D) < tol:
                break
        return self.best_x, sgn * self.best_f


# ---------------------------------------------------------------------------
# NSGA-II
# ---------------------------------------------------------------------------

def _dominance(F: np.ndarray, CV: np.ndarray) -> np.ndarray:
    """dom[i, j] = True if i constrained-dominates j (maximisation)."""
    ge = np.all(F[:, None, :] >= F[None, :, :], axis=2)
    gt = np.any(F[:, None, :] > F[None, :, :], axis=2)
    fdom = ge & gt
    fi, fj = CV[:, None] <= 0, CV[None, :] <= 0
    dom = np.where(fi & fj, fdom, False)
    dom |= fi & ~fj
    dom |= (~fi & ~fj) & (CV[:, None] < CV[None, :])
    np.fill_diagonal(dom, False)
    return dom


def fast_non_dominated_sort(F: np.ndarray, CV: Optional[np.ndarray] = None) -> List[np.ndarray]:
    CV = np.zeros(len(F)) if CV is None else np.asarray(CV, float)
    dom = _dominance(F, CV)
    n_dominated_by = dom.sum(axis=0)
    fronts: List[np.ndarray] = []
    current = np.where(n_dominated_by == 0)[0]
    remaining = n_dominated_by.copy()
    assigned = np.zeros(len(F), bool)
    while current.size:
        fronts.append(current)
        assigned[current] = True
        remaining = remaining - dom[current].sum(axis=0)
        current = np.where((remaining == 0) & ~assigned)[0]
    return fronts


def crowding_distance(F: np.ndarray) -> np.ndarray:
    n, m = F.shape
    d = np.zeros(n)
    if n <= 2:
        return np.full(n, np.inf)
    for k in range(m):
        o = np.argsort(F[:, k])
        span = F[o[-1], k] - F[o[0], k]
        d[o[0]] = d[o[-1]] = np.inf
        if span > 0:
            d[o[1:-1]] += (F[o[2:], k] - F[o[:-2], k]) / span
    return d


def hypervolume_2d(F: np.ndarray, ref: Sequence[float]) -> float:
    """Exact dominated hypervolume of a 2-objective maximisation front."""
    P = F[np.all(F > np.asarray(ref), axis=1)]
    if not len(P):
        return 0.0
    P = P[np.argsort(-P[:, 0])]
    hv, best_y = 0.0, ref[1]
    for x, y in P:
        if y > best_y:
            hv += (x - ref[0]) * (y - best_y)
            best_y = y
    return float(hv)


@dataclass
class NSGA2Result:
    X: np.ndarray
    F: np.ndarray
    CV: np.ndarray
    front: np.ndarray  # indices of the final first front
    history: List[Dict[str, Any]]
    evaluations: int

    @property
    def pareto_X(self) -> np.ndarray:
        return self.X[self.front]

    @property
    def pareto_F(self) -> np.ndarray:
        return self.F[self.front]


class NSGA2:
    """Multi-objective evolutionary optimiser (maximisation, constraints via CV >= 0)."""

    def __init__(self, n_dim: int, n_obj: int, bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                 pop_size: int = 40, eta_c: float = 15.0, eta_m: float = 20.0, p_mut: Optional[float] = None,
                 seed: Optional[int] = None):
        self.n, self.m = n_dim, n_obj
        self.lo, self.hi = (np.zeros(n_dim), np.ones(n_dim)) if bounds is None else (np.asarray(bounds[0], float),
                                                                                       np.asarray(bounds[1], float))
        self.pop_size = pop_size + (pop_size % 2)
        self.eta_c, self.eta_m = eta_c, eta_m
        self.p_mut = p_mut if p_mut is not None else 1.0 / n_dim
        self.rng = np.random.default_rng(seed)

    def _rank_crowd(self, F: np.ndarray, CV: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        rank = np.zeros(len(F), int)
        crowd = np.zeros(len(F))
        for r, fr in enumerate(fast_non_dominated_sort(F, CV)):
            rank[fr] = r
            crowd[fr] = crowding_distance(F[fr])
        return rank, crowd

    def _select(self, rank: np.ndarray, crowd: np.ndarray) -> int:
        i, j = self.rng.integers(0, len(rank), 2)
        if rank[i] != rank[j]:
            return int(i if rank[i] < rank[j] else j)
        return int(i if crowd[i] >= crowd[j] else j)

    def run(self, evaluate: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]], generations: int,
            initial: Optional[np.ndarray] = None, callback: Optional[Callable[[int, np.ndarray, np.ndarray], None]] = None
            ) -> NSGA2Result:
        X = self.lo + self.rng.random((self.pop_size, self.n)) * (self.hi - self.lo)
        if initial is not None:
            k = min(len(initial), self.pop_size)
            X[:k] = np.clip(initial[:k], self.lo, self.hi)
        F, CV = evaluate(X)
        evals = len(X)
        history: List[Dict[str, Any]] = []
        for gen in range(generations):
            rank, crowd = self._rank_crowd(F, CV)
            kids = []
            while len(kids) < self.pop_size:
                a, b = X[self._select(rank, crowd)], X[self._select(rank, crowd)]
                c1, c2 = sbx_crossover(a, b, self.lo, self.hi, self.eta_c, self.rng)
                kids.append(polynomial_mutation(c1, self.lo, self.hi, self.eta_m, self.p_mut, self.rng))
                kids.append(polynomial_mutation(c2, self.lo, self.hi, self.eta_m, self.p_mut, self.rng))
            Q = np.array(kids[: self.pop_size])
            FQ, CVQ = evaluate(Q)
            evals += len(Q)
            RX, RF, RCV = np.concatenate([X, Q]), np.concatenate([F, FQ]), np.concatenate([CV, CVQ])
            keep: List[int] = []
            for fr in fast_non_dominated_sort(RF, RCV):
                if len(keep) + len(fr) <= self.pop_size:
                    keep.extend(fr.tolist())
                else:
                    cd = crowding_distance(RF[fr])
                    keep.extend(fr[np.argsort(-cd)][: self.pop_size - len(keep)].tolist())
                    break
            X, F, CV = RX[keep], RF[keep], RCV[keep]
            feas = CV <= 0
            history.append({"generation": gen, "feasible_fraction": float(feas.mean()),
                            "best": (F[feas].max(axis=0).tolist() if feas.any() else None),
                            "mean": (F[feas].mean(axis=0).tolist() if feas.any() else None)})
            if callback:
                callback(gen, X, F)
        front = fast_non_dominated_sort(F, CV)[0]
        return NSGA2Result(X=X, F=F, CV=CV, front=front, history=history, evaluations=evals)


# ---------------------------------------------------------------------------
# NEAT
# ---------------------------------------------------------------------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(4.9 * x, -60, 60)))


ACTIVATIONS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "sigmoid": _sigmoid,
    "tanh": np.tanh,
    "sin": lambda x: np.sin(np.pi * x),
    "gauss": lambda x: np.exp(-2.5 * x * x),
    "abs": np.abs,
    "relu": lambda x: np.maximum(x, 0.0),
    "identity": lambda x: x,
}
CPPN_ACTIVATIONS: Tuple[str, ...] = ("sigmoid", "tanh", "sin", "gauss", "abs", "identity")


@dataclass
class NodeGene:
    id: int
    kind: str  # input | bias | output | hidden
    activation: str = "sigmoid"
    bias: float = 0.0


@dataclass
class ConnGene:
    innov: int
    src: int
    dst: int
    weight: float
    enabled: bool = True


class InnovationTracker:
    """Global historical markings shared by a population."""

    def __init__(self, next_node_id: int):
        self.next_node = next_node_id
        self.next_innov = 0
        self.conn: Dict[Tuple[int, int], int] = {}
        self.split: Dict[int, int] = {}

    def conn_innov(self, src: int, dst: int) -> int:
        key = (src, dst)
        if key not in self.conn:
            self.conn[key] = self.next_innov
            self.next_innov += 1
        return self.conn[key]

    def split_node(self, innov: int) -> int:
        if innov not in self.split:
            self.split[innov] = self.next_node
            self.next_node += 1
        return self.split[innov]


class NEATGenome:
    def __init__(self, nodes: Dict[int, NodeGene], conns: Dict[int, ConnGene], n_in: int, n_out: int):
        self.nodes, self.conns, self.n_in, self.n_out = nodes, conns, n_in, n_out
        self.fitness: float = 0.0
        self.adjusted: float = 0.0

    @classmethod
    def minimal(cls, n_in: int, n_out: int, tracker: InnovationTracker, rng: np.random.Generator,
                output_activation: str = "sigmoid") -> "NEATGenome":
        nodes: Dict[int, NodeGene] = {}
        for i in range(n_in):
            nodes[i] = NodeGene(i, "input", "identity")
        nodes[n_in] = NodeGene(n_in, "bias", "identity")
        for o in range(n_out):
            nid = n_in + 1 + o
            nodes[nid] = NodeGene(nid, "output", output_activation, 0.0)
        conns: Dict[int, ConnGene] = {}
        for s in range(n_in + 1):
            for o in range(n_out):
                d = n_in + 1 + o
                inn = tracker.conn_innov(s, d)
                conns[inn] = ConnGene(inn, s, d, float(rng.normal(0, 1.0)))
        return cls(nodes, conns, n_in, n_out)

    def copy(self) -> "NEATGenome":
        g = NEATGenome({k: NodeGene(**vars(v)) for k, v in self.nodes.items()},
                       {k: ConnGene(**vars(v)) for k, v in self.conns.items()}, self.n_in, self.n_out)
        g.fitness = self.fitness
        return g

    # -- structure -------------------------------------------------------------
    def _creates_cycle(self, src: int, dst: int) -> bool:
        if src == dst:
            return True
        adj: Dict[int, List[int]] = {}
        for c in self.conns.values():
            adj.setdefault(c.src, []).append(c.dst)
        stack, seen = [dst], set()
        while stack:
            n = stack.pop()
            if n == src:
                return True
            if n in seen:
                continue
            seen.add(n)
            stack.extend(adj.get(n, []))
        return False

    def mutate(self, rng: np.random.Generator, tracker: InnovationTracker, activations: Sequence[str] = CPPN_ACTIVATIONS,
               p_add_conn: float = 0.15, p_add_node: float = 0.08, p_weight: float = 0.8,
               p_activation: float = 0.1) -> None:
        if rng.random() < p_weight:
            for c in self.conns.values():
                if rng.random() < 0.9:
                    c.weight += float(rng.normal(0, 0.4))
                else:
                    c.weight = float(rng.normal(0, 1.0))
                c.weight = float(np.clip(c.weight, -8, 8))
            for n in self.nodes.values():
                if n.kind in ("hidden", "output") and rng.random() < 0.2:
                    n.bias = float(np.clip(n.bias + rng.normal(0, 0.3), -5, 5))
        if rng.random() < p_add_conn:
            srcs = [n.id for n in self.nodes.values() if n.kind != "output"]
            dsts = [n.id for n in self.nodes.values() if n.kind in ("hidden", "output")]
            for _ in range(20):
                s, d = int(rng.choice(srcs)), int(rng.choice(dsts))
                exists = any(c.src == s and c.dst == d for c in self.conns.values())
                if not exists and not self._creates_cycle(s, d):
                    inn = tracker.conn_innov(s, d)
                    self.conns[inn] = ConnGene(inn, s, d, float(rng.normal(0, 1.0)))
                    break
        if rng.random() < p_add_node and self.conns:
            enabled = [c for c in self.conns.values() if c.enabled]
            if enabled:
                c = enabled[int(rng.integers(0, len(enabled)))]
                nid = tracker.split_node(c.innov)
                if nid not in self.nodes:
                    c.enabled = False
                    self.nodes[nid] = NodeGene(nid, "hidden", str(rng.choice(activations)))
                    i1, i2 = tracker.conn_innov(c.src, nid), tracker.conn_innov(nid, c.dst)
                    self.conns[i1] = ConnGene(i1, c.src, nid, 1.0)
                    self.conns[i2] = ConnGene(i2, nid, c.dst, c.weight)
        if rng.random() < p_activation:
            hidden = [n for n in self.nodes.values() if n.kind == "hidden"]
            if hidden:
                hidden[int(rng.integers(0, len(hidden)))].activation = str(rng.choice(activations))
        if rng.random() < 0.02 and self.conns:
            c = list(self.conns.values())[int(rng.integers(0, len(self.conns)))]
            if c.enabled or not self._creates_cycle(c.src, c.dst):
                c.enabled = not c.enabled

    @staticmethod
    def crossover(a: "NEATGenome", b: "NEATGenome", rng: np.random.Generator) -> "NEATGenome":
        """``a`` must be the fitter parent; disjoint/excess genes come from it."""
        conns: Dict[int, ConnGene] = {}
        for inn, ca in a.conns.items():
            src = ca
            if inn in b.conns and rng.random() < 0.5:
                src = b.conns[inn]
            g = ConnGene(**vars(src))
            if inn in b.conns and (not ca.enabled or not b.conns[inn].enabled):
                g.enabled = rng.random() > 0.75
            conns[inn] = g
        nodes = {k: NodeGene(**vars(v)) for k, v in a.nodes.items()}
        for k, v in b.nodes.items():
            if k in nodes and rng.random() < 0.5 and v.kind == nodes[k].kind:
                nodes[k] = NodeGene(**vars(v))
        child = NEATGenome(nodes, conns, a.n_in, a.n_out)
        # Guard against cycles introduced by re-enabled genes.
        for c in list(child.conns.values()):
            if c.enabled:
                c.enabled = False
                if child._creates_cycle(c.src, c.dst):
                    continue
                c.enabled = True
        return child

    def distance(self, other: "NEATGenome", c1: float = 1.0, c2: float = 1.0, c3: float = 0.4) -> float:
        ia, ib = set(self.conns), set(other.conns)
        if not ia and not ib:
            return 0.0
        max_a, max_b = max(ia, default=-1), max(ib, default=-1)
        cutoff = min(max_a, max_b)
        non_match = ia ^ ib
        excess = sum(1 for i in non_match if i > cutoff)
        disjoint = len(non_match) - excess
        match = ia & ib
        W = float(np.mean([abs(self.conns[i].weight - other.conns[i].weight) for i in match])) if match else 0.0
        N = max(len(ia), len(ib))
        N = 1.0 if N < 20 else float(N)
        return c1 * excess / N + c2 * disjoint / N + c3 * W

    @property
    def complexity(self) -> Tuple[int, int]:
        return (sum(1 for n in self.nodes.values() if n.kind == "hidden"), sum(1 for c in self.conns.values() if c.enabled))


class FeedForwardNetwork:
    """Vectorised evaluation of a feed-forward NEAT genome."""

    def __init__(self, genome: NEATGenome):
        self.g = genome
        self.inputs = [n.id for n in genome.nodes.values() if n.kind == "input"]
        self.bias = [n.id for n in genome.nodes.values() if n.kind == "bias"]
        self.outputs = [n.id for n in genome.nodes.values() if n.kind == "output"]
        enabled = [c for c in genome.conns.values() if c.enabled]
        self.incoming: Dict[int, List[ConnGene]] = {}
        for c in enabled:
            self.incoming.setdefault(c.dst, []).append(c)
        # topological order (Kahn) over nodes reachable from inputs
        indeg = {n: 0 for n in genome.nodes}
        for c in enabled:
            indeg[c.dst] += 1
        order, queue = [], [n for n, d in indeg.items() if d == 0]
        adj: Dict[int, List[int]] = {}
        for c in enabled:
            adj.setdefault(c.src, []).append(c.dst)
        while queue:
            n = queue.pop()
            order.append(n)
            for m in adj.get(n, []):
                indeg[m] -= 1
                if indeg[m] == 0:
                    queue.append(m)
        self.order = order

    def activate(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, float))
        vals: Dict[int, np.ndarray] = {}
        for i, nid in enumerate(self.inputs):
            vals[nid] = X[:, i]
        for nid in self.bias:
            vals[nid] = np.ones(len(X))
        for nid in self.order:
            node = self.g.nodes[nid]
            if node.kind in ("input", "bias"):
                continue
            s = np.full(len(X), node.bias)
            for c in self.incoming.get(nid, []):
                s = s + c.weight * vals.get(c.src, 0.0)
            vals[nid] = ACTIVATIONS[node.activation](s)
        return np.stack([vals.get(o, np.zeros(len(X))) for o in self.outputs], axis=1)


class NEATPopulation:
    """Speciated NEAT population (maximises a non-negative fitness)."""

    def __init__(self, n_in: int, n_out: int, pop_size: int = 50, seed: Optional[int] = None,
                 activations: Sequence[str] = CPPN_ACTIVATIONS, output_activation: str = "sigmoid",
                 target_species: int = 5, compat_threshold: float = 2.0, stagnation: int = 15, elitism: int = 1):
        self.rng = np.random.default_rng(seed)
        self.n_in, self.n_out = n_in, n_out
        self.tracker = InnovationTracker(n_in + 1 + n_out)
        self.activations = activations
        self.pop = [NEATGenome.minimal(n_in, n_out, self.tracker, self.rng, output_activation) for _ in range(pop_size)]
        self.pop_size = pop_size
        self.target_species = target_species
        self.threshold = compat_threshold
        self.stagnation = stagnation
        self.elitism = elitism
        self.species: List[Dict[str, Any]] = []
        self.best: Optional[NEATGenome] = None
        self.history: List[Dict[str, Any]] = []

    def _speciate(self) -> None:
        for s in self.species:
            s["members"] = []
        for g in self.pop:
            for s in self.species:
                if g.distance(s["rep"]) < self.threshold:
                    s["members"].append(g)
                    break
            else:
                self.species.append({"rep": g, "members": [g], "best": -np.inf, "stale": 0})
        self.species = [s for s in self.species if s["members"]]
        for s in self.species:
            s["rep"] = s["members"][int(self.rng.integers(0, len(s["members"])))]
        if len(self.species) < self.target_species:
            self.threshold *= 0.9
        elif len(self.species) > self.target_species:
            self.threshold *= 1.1

    def step(self, fitness_fn: Callable[[NEATGenome], float]) -> None:
        for g in self.pop:
            g.fitness = max(0.0, float(fitness_fn(g)))
        gen_best = max(self.pop, key=lambda g: g.fitness)
        if self.best is None or gen_best.fitness > self.best.fitness:
            self.best = gen_best.copy()
        self._speciate()
        for s in self.species:
            top = max(m.fitness for m in s["members"])
            if top > s["best"]:
                s["best"], s["stale"] = top, 0
            else:
                s["stale"] += 1
            for m in s["members"]:
                m.adjusted = m.fitness / len(s["members"])
        alive = [s for s in self.species if s["stale"] < self.stagnation or any(m is gen_best for m in s["members"])]
        if not alive:
            alive = self.species
        totals = np.array([sum(m.adjusted for m in s["members"]) for s in alive])
        if totals.sum() <= 0:
            totals = np.ones(len(alive))
        quota = np.maximum(1, np.floor(totals / totals.sum() * self.pop_size)).astype(int)
        while quota.sum() > self.pop_size:
            quota[int(np.argmax(quota))] -= 1
        while quota.sum() < self.pop_size:
            quota[int(np.argmax(totals))] += 1
        new_pop: List[NEATGenome] = []
        cumulative = np.cumsum(quota)
        for si, (s, q) in enumerate(zip(alive, quota)):
            members = sorted(s["members"], key=lambda g: -g.fitness)
            parents = members[: max(1, int(math.ceil(0.5 * len(members))))]
            for e in range(min(self.elitism, q, len(members)) if len(members) >= 3 else min(1, q)):
                new_pop.append(members[e].copy())
            while len(new_pop) < cumulative[si]:
                if len(parents) > 1 and self.rng.random() < 0.75:
                    a, b = (parents[int(i)] for i in self.rng.choice(len(parents), 2, replace=False))
                    if b.fitness > a.fitness:
                        a, b = b, a
                    child = NEATGenome.crossover(a, b, self.rng)
                else:
                    child = parents[int(self.rng.integers(0, len(parents)))].copy()
                child.mutate(self.rng, self.tracker, self.activations)
                new_pop.append(child)
        self.pop = new_pop[: self.pop_size]
        self.history.append({"best": float(self.best.fitness), "gen_best": float(gen_best.fitness),
                             "species": len(self.species), "threshold": self.threshold,
                             "complexity": self.best.complexity})

    def run(self, fitness_fn: Callable[[NEATGenome], float], generations: int,
            target: Optional[float] = None) -> NEATGenome:
        for _ in range(generations):
            self.step(fitness_fn)
            if target is not None and self.best.fitness >= target:
                break
        return self.best


class CPPNShapeGenerator:
    """Generative shape model: a NEAT-evolved CPPN maps coordinates (x, y, r)
    to material density; thresholding yields a 2-D part outline."""

    N_INPUTS = 3

    def __init__(self, nx: int = 32, ny: int = 16, mirror_y: bool = True):
        self.nx, self.ny, self.mirror_y = nx, ny, mirror_y
        xs = np.linspace(-1, 1, nx)
        ys = np.linspace(-1, 1, ny)
        X, Y = np.meshgrid(xs, ys)
        if mirror_y:
            Y = np.abs(Y)
        self.inputs = np.stack([X.ravel(), Y.ravel(), np.sqrt(X ** 2 + Y ** 2).ravel() / math.sqrt(2)], axis=1)

    def density(self, genome: NEATGenome) -> np.ndarray:
        out = FeedForwardNetwork(genome).activate(self.inputs)[:, 0]
        return out.reshape(self.ny, self.nx)

    def shape(self, genome: NEATGenome, threshold: float = 0.5) -> np.ndarray:
        return self.density(genome) > threshold


# ---------------------------------------------------------------------------
# Probabilistic PCA generative model
# ---------------------------------------------------------------------------

class LatentDesignModel:
    """x = mu + W z + eps, z ~ N(0, I_q), eps ~ N(0, s^2 I): closed-form ML fit."""

    def __init__(self, latent_dim: int = 2):
        self.q = latent_dim
        self.mu: Optional[np.ndarray] = None
        self.W: Optional[np.ndarray] = None
        self.sigma2 = 0.0

    def fit(self, X: np.ndarray) -> "LatentDesignModel":
        X = np.asarray(X, float)
        n, d = X.shape
        q = min(self.q, d - 1, max(n - 1, 1))
        self.q = q
        self.mu = X.mean(axis=0)
        S = np.cov(X.T, bias=True) if n > 1 else np.zeros((d, d))
        S = np.atleast_2d(S)
        evals, evecs = np.linalg.eigh(S)
        evals, evecs = evals[::-1], evecs[:, ::-1]
        self.sigma2 = float(max(np.mean(evals[q:]) if d > q else 0.0, 1e-10))
        self.W = evecs[:, :q] * np.sqrt(np.maximum(evals[:q] - self.sigma2, 0.0))
        return self

    def sample(self, n: int, rng: Optional[np.random.Generator] = None, temperature: float = 1.0) -> np.ndarray:
        rng = rng or np.random.default_rng()
        z = rng.standard_normal((n, self.q)) * temperature
        eps = rng.standard_normal((n, self.mu.size)) * math.sqrt(self.sigma2) * temperature
        return self.mu + z @ self.W.T + eps

    def encode(self, X: np.ndarray) -> np.ndarray:
        M = self.W.T @ self.W + self.sigma2 * np.eye(self.q)
        return np.linalg.solve(M, self.W.T @ (np.atleast_2d(X) - self.mu).T).T

    def decode(self, Z: np.ndarray) -> np.ndarray:
        return self.mu + np.atleast_2d(Z) @ self.W.T

    def log_likelihood(self, X: np.ndarray) -> float:
        d = self.mu.size
        C = self.W @ self.W.T + self.sigma2 * np.eye(d)
        sign, logdet = np.linalg.slogdet(C)
        Xc = np.atleast_2d(X) - self.mu
        quad = np.sum(Xc @ np.linalg.inv(C) * Xc, axis=1)
        return float(np.mean(-0.5 * (d * math.log(2 * math.pi) + logdet + quad)))


# ---------------------------------------------------------------------------
# Reinforcement learning for adaptive design
# ---------------------------------------------------------------------------

class DesignEnvironment:
    """Sequential design editing as an MDP.

    State  = (normalised design vector x in [0,1]^n, physics-context features c).
    Action = bounded edit dx in [-step, step]^n.
    Reward = f(x_{t+1}, ctx) - f(x_t, ctx)  (telescopes to total improvement).
    """

    def __init__(self, fitness_fn: Callable[[np.ndarray, Any], float], n_params: int,
                 contexts: Sequence[Tuple[np.ndarray, Any]], horizon: int = 8, step: float = 0.08,
                 seed: Optional[int] = None):
        self.f, self.n, self.contexts, self.horizon, self.step_size = fitness_fn, n_params, list(contexts), horizon, step
        self.rng = np.random.default_rng(seed)
        self.state_dim = n_params + len(self.contexts[0][0])

    def reset(self, context_index: Optional[int] = None, x0: Optional[np.ndarray] = None) -> np.ndarray:
        i = int(self.rng.integers(0, len(self.contexts))) if context_index is None else context_index
        self.c_feat, self.ctx = self.contexts[i]
        self.x = self.rng.random(self.n) if x0 is None else np.asarray(x0, float).copy()
        self.fx = float(self.f(self.x, self.ctx))
        self.t = 0
        return np.concatenate([self.x, self.c_feat])

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        a = np.clip(action, -1, 1) * self.step_size
        self.x = np.clip(self.x + a, 0, 1)
        fn = float(self.f(self.x, self.ctx))
        r = fn - self.fx
        self.fx = fn
        self.t += 1
        return np.concatenate([self.x, self.c_feat]), r, self.t >= self.horizon


class DesignRLAgent:
    """REINFORCE with a learned baseline and a linear-Gaussian policy."""

    def __init__(self, state_dim: int, action_dim: int, lr: float = 0.03, sigma0: float = 0.5,
                 seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)
        self.sd, self.ad = state_dim, action_dim
        self.W = np.zeros((action_dim, state_dim + 1))
        self.log_sigma = np.full(action_dim, math.log(sigma0))
        self.v = np.zeros(state_dim + 1)  # linear value baseline
        self.lr = lr
        self._m = [np.zeros_like(self.W), np.zeros_like(self.log_sigma)]
        self._s = [np.zeros_like(self.W), np.zeros_like(self.log_sigma)]
        self._t = 0
        self.history: List[Dict[str, float]] = []

    @staticmethod
    def phi(s: np.ndarray) -> np.ndarray:
        return np.concatenate([s, [1.0]])

    def act(self, s: np.ndarray, deterministic: bool = False) -> np.ndarray:
        mu = np.tanh(self.W @ self.phi(s))
        if deterministic:
            return mu
        return mu + np.exp(self.log_sigma) * self.rng.standard_normal(self.ad)

    def _adam(self, grads: List[np.ndarray]) -> None:
        self._t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        params = [self.W, self.log_sigma]
        for i, g in enumerate(grads):
            self._m[i] = b1 * self._m[i] + (1 - b1) * g
            self._s[i] = b2 * self._s[i] + (1 - b2) * g * g
            mh = self._m[i] / (1 - b1 ** self._t)
            sh = self._s[i] / (1 - b2 ** self._t)
            params[i] += self.lr * mh / (np.sqrt(sh) + eps)  # gradient *ascent*
        self.log_sigma[:] = np.clip(self.log_sigma, math.log(0.05), math.log(1.0))

    def train(self, env: DesignEnvironment, iterations: int = 60, episodes_per_iter: int = 8,
              gamma: float = 1.0) -> List[Dict[str, float]]:
        for it in range(iterations):
            batch = []
            for _ in range(episodes_per_iter):
                s = env.reset()
                traj = []
                done = False
                while not done:
                    a = self.act(s)
                    s2, r, done = env.step(a)
                    traj.append((s, a, r))
                    s = s2
                G, rets = 0.0, []
                for (_, _, r) in reversed(traj):
                    G = r + gamma * G
                    rets.append(G)
                batch.append((traj, rets[::-1]))
            # advantages with linear baseline, normalised across the batch
            phis = np.array([self.phi(st) for traj, _ in batch for (st, _, _) in traj])
            rets = np.array([g for _, rs in batch for g in rs])
            base = phis @ self.v
            adv = rets - base
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            # fit baseline by least squares (ridge)
            A = phis.T @ phis + 1e-3 * np.eye(phis.shape[1])
            self.v = np.linalg.solve(A, phis.T @ rets)
            gW = np.zeros_like(self.W)
            gS = np.zeros_like(self.log_sigma)
            k = 0
            sig = np.exp(self.log_sigma)
            for traj, _ in batch:
                for (st, a, _) in traj:
                    ph = self.phi(st)
                    pre = self.W @ ph
                    mu = np.tanh(pre)
                    z = (a - mu) / sig
                    gW += adv[k] * np.outer((z / sig) * (1 - mu * mu), ph)
                    gS += adv[k] * (z * z - 1.0)
                    k += 1
            gW /= k
            gS /= k
            self._adam([gW, gS])
            mean_ret = float(np.mean([rs[0] for _, rs in batch]))
            self.history.append({"iteration": it, "mean_return": mean_ret, "sigma": float(np.mean(np.exp(self.log_sigma)))})
        return self.history

    def evaluate(self, env: DesignEnvironment, episodes: int = 20, deterministic: bool = True,
                 random_policy: bool = False, seed: int = 123) -> float:
        """Mean total improvement per episode (vs. a random-edit baseline when
        ``random_policy``)."""
        rng = np.random.default_rng(seed)
        env.rng = np.random.default_rng(seed)
        tot = []
        for _ in range(episodes):
            s = env.reset()
            done, G = False, 0.0
            while not done:
                a = rng.uniform(-1, 1, self.ad) if random_policy else self.act(s, deterministic)
                s, r, done = env.step(a)
                G += r
            tot.append(G)
        return float(np.mean(tot))


class OperatorBandit:
    """UCB1 bandit over named mutation operators."""

    def __init__(self, arms: Sequence[str], c: float = 1.0, seed: Optional[int] = None):
        self.arms = list(arms)
        self.c = c
        self.n = {a: 0 for a in self.arms}
        self.value = {a: 0.0 for a in self.arms}
        self.t = 0
        self.rng = np.random.default_rng(seed)

    def select(self) -> str:
        self.t += 1
        unplayed = [a for a in self.arms if self.n[a] == 0]
        if unplayed:
            return str(self.rng.choice(unplayed))
        ucb = {a: self.value[a] + self.c * math.sqrt(math.log(self.t) / self.n[a]) for a in self.arms}
        return max(ucb, key=ucb.get)

    def update(self, arm: str, reward: float) -> None:
        self.n[arm] += 1
        self.value[arm] += (reward - self.value[arm]) / self.n[arm]

    def stats(self) -> Dict[str, Dict[str, float]]:
        return {a: {"pulls": self.n[a], "mean_reward": self.value[a]} for a in self.arms}


# ---------------------------------------------------------------------------
# Universe search
# ---------------------------------------------------------------------------

@dataclass
class UniverseCandidate:
    genome: Any
    fitness: float
    richness: float
    novelty: float
    stability: float
    consistency: float
    feasibility: float
    descriptor: np.ndarray
    report: Any = None
    run: Any = None
    wall_time: float = 0.0

    def brief(self) -> Dict[str, Any]:
        return {"genome_id": self.genome.genome_id, "law": self.genome.gravity.law, "fitness": round(self.fitness, 4),
                "richness": round(self.richness, 4), "novelty": round(self.novelty, 4),
                "stability": round(self.stability, 3), "consistency": round(self.consistency, 3),
                "feasibility": round(self.feasibility, 3)}


class ShivaBrain:
    """Meta-level search over physics genomes for emergent richness.

    fitness(u) = R(u) * (0.75 + 0.25 * feasibility(u)) * consistency(u)

    where R is the emergent richness and consistency is the Noether
    consistency (1.0 unless a predicted conservation law is violated, which
    flags an unresolved simulation).  Selection uses
    ``(1 - w) * fitness + w * novelty`` with novelty the mean distance to the
    k nearest behaviour descriptors in an archive.
    """

    def __init__(self, seed: int = 0, sim_config: Any = None, include_fields: bool = False,
                 include_rigid: bool = False, novelty_weight: float = 0.25, archive_k: int = 5,
                 verbose: bool = True, dim: int = 3, n_seeds: int = 1):
        from .physics_mutator import DEFAULT_OPERATOR_WEIGHTS, PhysicsMutator
        from .universe_simulator import SimulationConfig

        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.mutator = PhysicsMutator(rng=self.rng)
        self.sim_config = sim_config or SimulationConfig(n_particles=64, steps=1200, record_every=15,
                                                         initial_condition="cold_collapse", seed=seed)
        self.include_fields, self.include_rigid = include_fields, include_rigid
        self.w_nov, self.k = novelty_weight, archive_k
        self.verbose = verbose
        self.dim = dim
        # Operators whose genes are invisible to the evaluation are excluded
        # (fluid genes only act on the continuum fields).
        arms = [k for k, v in DEFAULT_OPERATOR_WEIGHTS.items() if v > 0 and (include_fields or k != "fluid")]
        self.bandit = OperatorBandit(arms + ["crossover"], seed=seed)
        self.n_seeds = max(1, int(n_seeds))
        self.archive: List[np.ndarray] = []
        self.cache: Dict[str, UniverseCandidate] = {}
        self.history: List[Dict[str, Any]] = []
        self.strength = 1.0
        self.evaluations = 0

    def log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    # -- evaluation ------------------------------------------------------------
    def evaluate_genome(self, genome: Any, keep_run: bool = False) -> UniverseCandidate:
        from .emergence_detector import EmergenceDetector
        from .universe_simulator import run_universe

        fp = genome.fingerprint()
        if fp in self.cache:
            c = self.cache[fp]
            return UniverseCandidate(genome, c.fitness, c.richness, c.novelty, c.stability, c.consistency,
                                     c.feasibility, c.descriptor, c.report, c.run if keep_run else None, 0.0)
        t0 = time.time()
        fits, rich, stab, cons, feas, descs = [], [], [], [], [], []
        for k in range(self.n_seeds):  # average over initial-condition realisations
            cfg = replace(self.sim_config, seed=self.sim_config.seed + 1000 * k)
            run = run_universe(genome, cfg, fields=self.include_fields, rigid=self.include_rigid,
                               field_resolution=48, rd_steps=2500, vort_t_end=15.0)
            rep = EmergenceDetector(seed=self.seed).analyze(run)
            ev = run.evaluation
            fits.append(rep.richness * (0.75 + 0.25 * ev.feasibility) * ev.conservation_consistency)
            rich.append(rep.richness)
            stab.append(ev.stability)
            cons.append(ev.conservation_consistency)
            feas.append(ev.feasibility)
            descs.append(rep.descriptor)
        cand = UniverseCandidate(genome=genome, fitness=float(np.mean(fits)), richness=float(np.mean(rich)),
                                 novelty=0.0, stability=float(np.mean(stab)), consistency=float(np.min(cons)),
                                 feasibility=float(np.mean(feas)), descriptor=np.mean(descs, axis=0), report=rep,
                                 run=run if keep_run else None, wall_time=time.time() - t0)
        self.evaluations += 1
        self.cache[fp] = cand
        return cand

    def _novelty(self, d: np.ndarray, others: List[np.ndarray]) -> float:
        if not others:
            return 1.0
        dist = np.sort(np.linalg.norm(np.array(others) - d, axis=1))
        dist = dist[dist > 1e-12] if np.any(dist > 1e-12) else dist
        return float(np.mean(dist[: self.k])) if dist.size else 0.0

    def _score(self, pop: List[UniverseCandidate]) -> np.ndarray:
        descs = [c.descriptor for c in pop] + self.archive
        for c in pop:
            c.novelty = self._novelty(c.descriptor, descs)
        nov = np.array([c.novelty for c in pop])
        nov = nov / nov.max() if nov.max() > 0 else nov
        fit = np.array([c.fitness for c in pop])
        return (1 - self.w_nov) * fit + self.w_nov * nov * (fit.max() if fit.max() > 0 else 1.0)

    # -- search -----------------------------------------------------------------
    def search(self, generations: int = 5, population: int = 8, elite: int = 2,
               seed_genomes: Optional[List[Any]] = None) -> Dict[str, Any]:
        t0 = time.time()
        genomes = list(seed_genomes or [])
        if not genomes:
            genomes.append(self.mutator.baseline(self.dim))
        while len(genomes) < population:
            genomes.append(self.mutator.random_genome(self.dim))
        pop = []
        for g in genomes:
            c = self.evaluate_genome(g)
            pop.append(c)
            self.log(f"  [init] {g.gravity.law:15s} fitness={c.fitness:.3f} R={c.richness:.3f} ({c.wall_time:.1f}s)")
        best = max(pop, key=lambda c: c.fitness)
        for gen in range(generations):
            scores = self._score(pop)
            order = np.argsort(-scores)
            elites = [pop[i] for i in order[:elite]]
            children: List[UniverseCandidate] = []
            successes = 0
            while len(children) < population - elite:
                a, b = self.rng.integers(0, len(pop), 2)
                parent = pop[int(a) if scores[a] >= scores[b] else int(b)]  # binary tournament
                op = self.bandit.select()
                if op == "crossover":
                    j = int(order[int(self.rng.integers(0, max(1, len(pop) // 2)))])
                    child_g = self.mutator.crossover(parent.genome, pop[j].genome)
                    child_g = self.mutator.mutate(child_g, strength=0.5 * self.strength, n_ops=1)
                else:
                    child_g = self.mutator.mutate(parent.genome, strength=self.strength, operators=[op])
                child = self.evaluate_genome(child_g)
                improvement = child.fitness - parent.fitness
                self.bandit.update(op, float(np.tanh(10 * improvement)) * 0.5 + 0.5)
                successes += improvement > 0
                children.append(child)
                self.log(f"  [gen {gen + 1}] {op:17s} -> {child_g.gravity.law:15s} fitness={child.fitness:.3f}"
                         f" (parent {parent.fitness:.3f})")
            rate = successes / max(1, len(children))
            self.strength = float(np.clip(self.strength * (1.2 if rate > 0.2 else 0.85), 0.25, 3.0))
            pop = elites + children
            for c in children:
                if c.novelty > np.median([p.novelty for p in pop]) or self.rng.random() < 0.1:
                    self.archive.append(c.descriptor)
            gb = max(pop, key=lambda c: c.fitness)
            if gb.fitness > best.fitness:
                best = gb
            self.history.append({"generation": gen + 1, "best_fitness": best.fitness,
                                 "mean_fitness": float(np.mean([c.fitness for c in pop])),
                                 "best_richness": best.richness, "success_rate": rate, "strength": self.strength,
                                 "laws": [c.genome.gravity.law for c in pop]})
            self.log(f"== generation {gen + 1}: best fitness {best.fitness:.3f} ({best.genome.gravity.law}),"
                     f" success rate {rate:.2f}, strength {self.strength:.2f}")
        ranked = sorted(self.cache.values(), key=lambda c: -c.fitness)
        return {"best": best, "population": pop, "ranked": ranked, "history": self.history,
                "bandit": self.bandit.stats(), "evaluations": self.evaluations, "wall_time": time.time() - t0}

    def refine_cmaes(self, genome: Any, iterations: int = 4, popsize: int = 6, sigma0: float = 0.15
                     ) -> Tuple[Any, float, List[Dict[str, float]]]:
        """CMA-ES over the continuous constants of ``genome`` (structure fixed)."""
        x0 = self.mutator.vectorize(genome)
        es = CMAES(x0, sigma0, bounds=(np.zeros_like(x0), np.ones_like(x0)), popsize=popsize, seed=self.seed)
        best_g, best_f = genome, self.evaluate_genome(genome).fitness
        for it in range(iterations):
            X = es.ask()
            fs = []
            for x in X:
                g = self.mutator.devectorize(x, genome)
                c = self.evaluate_genome(g)
                fs.append(c.fitness)
                if c.fitness > best_f:
                    best_g, best_f = g, c.fitness
            es.tell(X, -np.array(fs))
            self.log(f"  [cma-es {it + 1}] best {best_f:.3f} sigma {es.sigma:.3f}")
        return best_g, best_f, es.history
