"""
SHIVA-1 :: Emergence Detector
=============================

Quantifies emergent phenomena in simulated universes and condenses them into
an *emergent-richness* score.  Every metric is a standard, citable estimator:

====================  ===========================================================
Phenomenon            Estimator
====================  ===========================================================
self-organisation     drop of coarse-grained spatial Shannon entropy, LMC
                      statistical complexity C = H * D (Lopez-Ruiz, Mancini &
                      Calbet 1995), collective order parameters (polarisation
                      and milling, Vicsek 1995 / Couzin 2002)
oscillation           peak-to-background ratio of the detrended periodogram of
                      macro-observables (kinetic energy, radius of gyration)
stable attractors     recurrence quantification: determinism DET at fixed
                      recurrence rate (Marwan et al. 2007) x stationarity
chaos                 maximal Lyapunov exponent from the simulator's shadow
                      trajectory (Benettin et al. 1980) or Rosenstein et al. 1993
fractal structure     correlation dimension D2 of the final point set
                      (Grassberger & Procaccia 1983) + box-counting dimension
symmetry breaking     change-point analysis (CUSUM) of order parameters
                      (asphericity, polarisation, milling, net spin); flagged
                      *spontaneous* when the laws retain the symmetry
cosmic clustering     friends-of-friends haloes, b = 0.2 x mean separation
                      (Davis et al. 1985) and the Landy-Szalay (1993) two-point
                      correlation function with power-law fit xi = (r/r0)^-gamma
field patterns        Turing-pattern counts / wavelength (reaction-diffusion),
                      coherent vortices and vorticity kurtosis (2-D flow)
====================  ===========================================================

Richness::

    R = stability * (0.7 * sum_i w_i s_i / sum_i w_i + 0.3 * coverage)

where ``s_i`` are the phenomenon scores in [0, 1] and ``coverage`` is the
fraction of phenomena with ``s_i >= 0.25``: a universe is *rich* when it is
stable and exhibits many distinct phenomena at once.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist, pdist

__all__ = [
    "EmergenceReport",
    "EmergenceDetector",
    "spatial_entropy",
    "lmc_complexity",
    "correlation_dimension",
    "box_counting_dimension",
    "friends_of_friends",
    "landy_szalay",
    "rqa",
    "rosenstein_lyapunov",
    "cusum_change_point",
    "order_parameters",
    "PHENOMENA",
    "DEFAULT_WEIGHTS",
]

PHENOMENA: Tuple[str, ...] = ("self_organization", "oscillation", "attractor", "chaos",
                              "fractal", "symmetry_breaking", "clustering")
FIELD_PHENOMENA: Tuple[str, ...] = ("pattern_formation", "coherent_vortices", "rotational_symmetry_breaking")
DEFAULT_WEIGHTS: Dict[str, float] = {
    "self_organization": 1.2, "oscillation": 0.8, "attractor": 0.8, "chaos": 0.6, "fractal": 1.0,
    "symmetry_breaking": 1.0, "clustering": 1.2, "pattern_formation": 0.6, "coherent_vortices": 0.4,
    "rotational_symmetry_breaking": 0.4,
}


# ---------------------------------------------------------------------------
# Elementary estimators
# ---------------------------------------------------------------------------

def _central_box(X: np.ndarray, q: float = 2.5) -> Tuple[np.ndarray, np.ndarray]:
    lo = np.percentile(X, q, axis=0)
    hi = np.percentile(X, 100 - q, axis=0)
    span = np.maximum(hi - lo, 1e-9)
    return lo - 0.05 * span, hi + 0.05 * span


def _occupation(X: np.ndarray, bins: int) -> np.ndarray:
    lo, hi = _central_box(X)
    Xc = np.clip(X, lo, hi - 1e-12 * (hi - lo))
    hist, _ = np.histogramdd(Xc, bins=[np.linspace(l, h, bins + 1) for l, h in zip(lo, hi)])
    p = hist.ravel()
    return p / max(p.sum(), 1e-300)


def spatial_entropy(X: np.ndarray, bins: Optional[int] = None) -> float:
    """Normalised Shannon entropy H/H_max of the coarse-grained density in a
    scale-free (percentile) box, so it measures *shape*, not overall size."""
    n, d = X.shape
    bins = bins or max(3, int(round((n / 2.0) ** (1.0 / d))))
    p = _occupation(X, bins)
    nz = p[p > 0]
    return float(-np.sum(nz * np.log(nz)) / math.log(p.size))


def lmc_complexity(X: np.ndarray, bins: Optional[int] = None) -> Tuple[float, float, float]:
    """LMC statistical complexity C = H * D with normalised disequilibrium."""
    n, d = X.shape
    bins = bins or max(3, int(round((n / 2.0) ** (1.0 / d))))
    p = _occupation(X, bins)
    K = p.size
    nz = p[p > 0]
    H = float(-np.sum(nz * np.log(nz)) / math.log(K))
    D = float(np.sum((p - 1.0 / K) ** 2) / (1.0 - 1.0 / K))
    return H * D, H, D


def order_parameters(X: np.ndarray, V: np.ndarray, masses: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Polarisation, milling, asphericity and net spin of a snapshot."""
    n, d = X.shape
    w = masses if masses is not None else np.full(n, 1.0 / n)
    xc = X - np.average(X, axis=0, weights=w)
    vc = V - np.average(V, axis=0, weights=w)
    vn = np.linalg.norm(V, axis=1, keepdims=True)
    vhat = np.where(vn > 1e-300, V / np.maximum(vn, 1e-300), 0.0)
    polar = float(np.linalg.norm(vhat.mean(axis=0)))
    rn = np.linalg.norm(xc, axis=1, keepdims=True)
    rhat = np.where(rn > 1e-300, xc / np.maximum(rn, 1e-300), 0.0)
    vcn = np.linalg.norm(vc, axis=1, keepdims=True)
    vchat = np.where(vcn > 1e-300, vc / np.maximum(vcn, 1e-300), 0.0)
    if d == 2:
        mill = float(abs(np.mean(rhat[:, 0] * vchat[:, 1] - rhat[:, 1] * vchat[:, 0])))
        spin_vec = np.array([np.sum(w * (xc[:, 0] * vc[:, 1] - xc[:, 1] * vc[:, 0]))])
    else:
        mill = float(np.linalg.norm(np.mean(np.cross(rhat, vchat), axis=0)))
        spin_vec = np.sum(w[:, None] * np.cross(xc, vc), axis=0)
    norm = float(np.sum(w * np.linalg.norm(xc, axis=1) * np.linalg.norm(vc, axis=1)))
    spin = float(np.linalg.norm(spin_vec) / norm) if norm > 1e-300 else 0.0
    S = (w[:, None] * xc).T @ xc / w.sum()
    ev = np.sort(np.linalg.eigvalsh(S))[::-1]
    tr = max(float(ev.sum()), 1e-300)
    if d == 3:
        asph = float((ev[0] - 0.5 * (ev[1] + ev[2])) / tr)
    else:
        asph = float((ev[0] - ev[1]) / tr)
    rg = float(math.sqrt(tr))
    return {"polarization": polar, "milling": mill, "asphericity": asph, "spin": spin, "radius_of_gyration": rg}


def correlation_dimension(X: np.ndarray, r_min: Optional[float] = None, min_decades: float = 0.6,
                          fixed_range: Optional[Tuple[float, float]] = None) -> Dict[str, Any]:
    """Grassberger-Procaccia D2: slope of log C(r) vs log r over the most
    linear window spanning at least ``min_decades`` (or over ``fixed_range``)."""
    dist = pdist(X)
    dist = dist[dist > 0]
    if fixed_range is not None and dist.size >= 20:
        rs = np.geomspace(fixed_range[0], fixed_range[1], 12)
        C = np.searchsorted(np.sort(dist), rs) / dist.size
        ok = C > 0
        if ok.sum() >= 3:
            coef = np.polyfit(np.log10(rs[ok]), np.log10(C[ok]), 1)
            return {"D2": float(coef[0]), "r2": float("nan"), "range": fixed_range}
        return {"D2": float("nan"), "r2": 0.0, "range": fixed_range}
    if dist.size < 20:
        return {"D2": float("nan"), "r2": 0.0, "range": (float("nan"), float("nan"))}
    lo = max(np.percentile(dist, 1), r_min or 0.0)
    hi = np.percentile(dist, 60)
    if not hi > lo * 1.5:
        return {"D2": float("nan"), "r2": 0.0, "range": (float(lo), float(hi))}
    rs = np.geomspace(lo, hi, 24)
    ds = np.sort(dist)
    C = np.searchsorted(ds, rs) / ds.size
    ok = C > 0
    lr, lc = np.log10(rs[ok]), np.log10(C[ok])
    best = None
    n = lr.size
    for i in range(n):
        for j in range(i + 4, n + 1):
            span = lr[j - 1] - lr[i]
            if span < min_decades:
                continue
            x, y = lr[i:j], lc[i:j]
            coef = np.polyfit(x, y, 1)
            ss = float(np.sum((y - y.mean()) ** 2))
            r2 = 1.0 - float(np.sum((y - np.polyval(coef, x)) ** 2)) / ss if ss > 0 else 0.0
            if r2 < 0.95:
                continue
            key = (round(span, 6), r2)  # widest scaling range first, then best fit
            if best is None or key > best[0]:
                best = (key, float(coef[0]), r2, (float(10 ** lr[i]), float(10 ** lr[j - 1])))
    if best is None:
        coef = np.polyfit(lr, lc, 1)
        resid = lc - np.polyval(coef, lr)
        r2 = 1 - np.sum(resid ** 2) / max(np.sum((lc - lc.mean()) ** 2), 1e-300)
        return {"D2": float(coef[0]), "r2": float(r2), "range": (float(lo), float(hi))}
    return {"D2": best[1], "r2": float(best[2]), "range": best[3]}


def box_counting_dimension(X: np.ndarray, n_scales: int = 8) -> float:
    lo, hi = _central_box(X, q=0.5)
    span = float(np.max(hi - lo))
    Xn = (X - lo) / span
    eps = np.geomspace(0.5, 2.0 / max(len(X) ** (1.0 / X.shape[1]), 2.0) / 4, n_scales)
    counts = [len(np.unique(np.floor(np.clip(Xn, 0, 1 - 1e-12) / e).astype(int), axis=0)) for e in eps]
    coef = np.polyfit(np.log(1 / eps), np.log(counts), 1)
    return float(coef[0])


def friends_of_friends(X: np.ndarray, linking_length: float, min_members: int = 5,
                       boxsize: Optional[float] = None) -> Dict[str, Any]:
    """Friends-of-friends halo finder (union of all pairs closer than b)."""
    n = len(X)
    if boxsize:
        tree = cKDTree(np.mod(X + 0.5 * boxsize, boxsize), boxsize=boxsize)
    else:
        tree = cKDTree(X)
    pairs = tree.query_pairs(linking_length, output_type="ndarray")
    if pairs.size:
        A = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
        ncomp, labels = connected_components(A, directed=False)
    else:
        labels = np.arange(n)
    sizes = np.bincount(labels)
    groups = np.where(sizes >= min_members)[0]
    members = np.isin(labels, groups)
    return {"labels": labels, "group_sizes": np.sort(sizes[groups])[::-1].tolist(), "n_groups": int(groups.size),
            "clustered_fraction": float(members.mean()), "largest_fraction": float(sizes.max() / n)}


def landy_szalay(X: np.ndarray, r_edges: np.ndarray, n_random_factor: int = 4,
                 rng: Optional[np.random.Generator] = None, boxsize: Optional[float] = None) -> np.ndarray:
    """xi(r) = (DD - 2 DR + RR) / RR with normalised pair counts."""
    rng = rng or np.random.default_rng(0)
    n, d = X.shape
    nr = n_random_factor * n
    if boxsize:
        Xd = np.mod(X + 0.5 * boxsize, boxsize)
        R = rng.random((nr, d)) * boxsize
        td, tr = cKDTree(Xd, boxsize=boxsize), cKDTree(R, boxsize=boxsize)
    else:
        lo, hi = _central_box(X, q=1.0)
        R = lo + rng.random((nr, d)) * (hi - lo)
        td, tr = cKDTree(X), cKDTree(R)
    DD = np.diff(td.count_neighbors(td, r_edges)).astype(float)
    RR = np.diff(tr.count_neighbors(tr, r_edges)).astype(float)
    DR = np.diff(td.count_neighbors(tr, r_edges)).astype(float)
    DD /= n * (n - 1)
    RR /= nr * (nr - 1)
    DR /= n * nr
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(RR > 0, (DD - 2 * DR + RR) / RR, np.nan)


def rqa(Z: np.ndarray, recurrence_rate: float = 0.1, lmin: int = 2) -> Dict[str, float]:
    """Recurrence quantification at a fixed recurrence rate."""
    n = len(Z)
    if n < 10:
        return {"RR": float("nan"), "DET": 0.0, "eps": float("nan")}
    D = cdist(Z, Z)
    off = D[~np.eye(n, dtype=bool)]
    eps = float(np.quantile(off, recurrence_rate))
    R = D <= eps
    total = int(R.sum() - n)
    if total <= 0:
        return {"RR": 0.0, "DET": 0.0, "eps": eps}
    in_lines = 0
    for k in range(1, n):
        diag = np.diagonal(R, k).astype(np.int8)
        if not diag.any():
            continue
        padded = np.concatenate([[0], diag, [0]])
        dd = np.diff(padded)
        starts, ends = np.where(dd == 1)[0], np.where(dd == -1)[0]
        lens = ends - starts
        in_lines += int(lens[lens >= lmin].sum())
    return {"RR": total / (n * n - n), "DET": 2.0 * in_lines / total, "eps": eps}


def rosenstein_lyapunov(x: np.ndarray, dt: float, m: int = 4, tau: int = 1, horizon: int = 12) -> float:
    """Largest Lyapunov exponent of a scalar series (Rosenstein et al. 1993)."""
    x = np.asarray(x, dtype=float)
    n = len(x) - (m - 1) * tau
    if n < 3 * horizon:
        return float("nan")
    Y = np.stack([x[i * tau:i * tau + n] for i in range(m)], axis=1)
    D = cdist(Y, Y)
    sep = max(horizon, 3)
    for i in range(n):
        D[i, max(0, i - sep):i + sep + 1] = np.inf
    nn = np.argmin(D, axis=1)
    div = []
    for k in range(horizon):
        idx = np.arange(n - k)
        valid = nn[idx] + k < n
        a, b = idx[valid], nn[idx][valid] + k
        a = a[a + k < n]
        b = b[: len(a)]
        dist = np.linalg.norm(Y[a + k] - Y[b], axis=1)
        dist = dist[dist > 0]
        div.append(np.mean(np.log(dist)) if dist.size else np.nan)
    div = np.array(div)
    ok = np.isfinite(div)
    if ok.sum() < 3:
        return float("nan")
    return float(np.polyfit(np.arange(horizon)[ok] * dt, div[ok], 1)[0])


def cusum_change_point(x: np.ndarray) -> Dict[str, float]:
    """Single change point maximising |CUSUM|; effect size in pooled std units."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 6:
        return {"index": -1, "effect": 0.0, "delta": 0.0}
    S = np.cumsum(x - x.mean())
    k = int(np.argmax(np.abs(S[1:-2]))) + 2
    a, b = x[:k], x[k:]
    pooled = math.sqrt(0.5 * (a.var() + b.var())) + 1e-9
    return {"index": k, "effect": float(abs(b.mean() - a.mean()) / pooled), "delta": float(b.mean() - a.mean())}


def _peak_ratio(x: np.ndarray, t: np.ndarray) -> Tuple[float, float]:
    """Max periodogram peak / running-median background (>= 2 cycles)."""
    x = np.asarray(x, dtype=float)
    if len(x) < 16 or not np.all(np.isfinite(x)) or np.std(x) < 1e-12:
        return 1.0, float("nan")
    tt = t - t[0]
    x = x - np.polyval(np.polyfit(tt, x, 1), tt)
    if np.std(x) < 1e-12:
        return 1.0, float("nan")
    x = x / np.std(x)
    P = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    f = np.fft.rfftfreq(len(x), d=float(np.mean(np.diff(t))))
    P, f = P[1:], f[1:]
    bg = ndimage.median_filter(np.log(P + 1e-300), size=9, mode="nearest")
    ratio = P / np.exp(bg)
    T = tt[-1]
    ok = f * T >= 2.0
    if not ok.any():
        return 1.0, float("nan")
    i = int(np.argmax(np.where(ok, ratio, 0)))
    return float(ratio[i]), float(1.0 / f[i])


def _clip01(x: float) -> float:
    return float(min(max(x, 0.0), 1.0)) if np.isfinite(x) else 0.0


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

@dataclass
class EmergenceReport:
    scores: Dict[str, float]
    detected: Dict[str, bool]
    metrics: Dict[str, Dict[str, Any]]
    richness: float
    stability: float
    descriptor: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return _jsonable(d)

    def summary(self) -> str:
        lines = [f"emergent richness R = {self.richness:.3f}  (stability {self.stability:.2f})"]
        for k, v in self.scores.items():
            bar = "#" * int(round(20 * v))
            lines.append(f"  {k:30s} {v:5.2f} {bar:20s} {'DETECTED' if self.detected.get(k) else ''}")
        return "\n".join(lines)


def _jsonable(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class EmergenceDetector:
    """Analyse a :class:`~shiva_core.universe_simulator.UniverseRun` (or a bare
    ``SimulationResult``) and score its emergent richness."""

    def __init__(self, weights: Optional[Dict[str, float]] = None, detection_threshold: float = 0.25, seed: int = 0):
        self.weights = dict(DEFAULT_WEIGHTS)
        if weights:
            self.weights.update(weights)
        self.threshold = detection_threshold
        self.rng = np.random.default_rng(seed)

    # -- particle phenomena ---------------------------------------------------
    def order_parameter_series(self, res: Any) -> Dict[str, np.ndarray]:
        ops = [order_parameters(X, V, res.masses) for X, V in zip(res.positions, res.velocities)]
        return {k: np.array([o[k] for o in ops]) for k in ops[0]}

    def self_organization(self, res: Any, ops: Dict[str, np.ndarray]) -> Dict[str, Any]:
        X0, Xf = res.positions[0], res.positions[-1]
        H0, Hf = spatial_entropy(X0), spatial_entropy(Xf)
        C0, Cf = lmc_complexity(X0)[0], lmc_complexity(Xf)[0]
        tail = max(1, len(ops["polarization"]) // 5)
        order_gain = max(float(np.mean(ops["polarization"][-tail:]) - ops["polarization"][0]),
                         float(np.mean(ops["milling"][-tail:]) - ops["milling"][0]))
        score = _clip01(max((H0 - Hf) / 0.3, (Cf - C0) / 0.1, order_gain / 0.5))
        return {"score": score, "entropy_initial": H0, "entropy_final": Hf, "lmc_initial": C0, "lmc_final": Cf,
                "order_gain": order_gain, "polarization_final": float(ops["polarization"][-1]),
                "milling_final": float(ops["milling"][-1])}

    def oscillation(self, res: Any, ops: Dict[str, np.ndarray]) -> Dict[str, Any]:
        t = res.times
        series = {"kinetic": res.kinetic, "radius_of_gyration": ops["radius_of_gyration"],
                  "asphericity": ops["asphericity"]}
        best = (1.0, float("nan"), "")
        for name, x in series.items():
            r, per = _peak_ratio(x, t)
            if r > best[0]:
                best = (r, per, name)
        score = _clip01((math.log10(best[0]) - 0.5) / 1.5)
        return {"score": score, "peak_ratio": best[0], "period": best[1], "observable": best[2]}

    def attractor(self, res: Any, ops: Dict[str, np.ndarray]) -> Dict[str, Any]:
        n = len(res.times)
        half = slice(n // 2, None)
        Z = np.stack([res.kinetic[half], ops["radius_of_gyration"][half], ops["asphericity"][half]], axis=1)
        Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-12)
        q = rqa(Z)
        rg = ops["radius_of_gyration"][half]
        if rg.size > 3:
            tt = res.times[half] - res.times[half][0]
            slope = np.polyfit(tt, rg, 1)[0] * max(tt[-1], 1e-12)
            stationarity = math.exp(-abs(slope) / (np.mean(np.abs(rg)) + 1e-12) * 3.0)
        else:
            stationarity = 0.0
        score = _clip01(q["DET"] * stationarity)
        return {"score": score, "DET": q["DET"], "RR": q["RR"], "stationarity": stationarity}

    def chaos(self, res: Any, ops: Dict[str, np.ndarray]) -> Dict[str, Any]:
        lam = res.lyapunov
        method = "benettin-twin"
        if lam is None or not np.isfinite(lam):
            dt = float(np.mean(np.diff(res.times))) if len(res.times) > 1 else 1.0
            lam = rosenstein_lyapunov(ops["radius_of_gyration"], dt)
            method = "rosenstein"
        lam = float(lam) if lam is not None and np.isfinite(lam) else 0.0
        return {"score": _clip01(1.0 - math.exp(-max(lam, 0.0) / 1.0)), "lyapunov": lam, "method": method,
                "lyapunov_time": (1.0 / lam) if lam > 0 else float("inf")}

    def fractal(self, res: Any) -> Dict[str, Any]:
        Xf = res.positions[-1]
        d = Xf.shape[1]
        soft = float(res.config.get("softening", 0.0))
        cd = correlation_dimension(Xf, r_min=2.0 * soft)
        D2 = cd["D2"]
        # Self-calibration: GP estimates on ~1e2 points are biased low by edge
        # and finite-size effects, so compare with Poisson samples of the same
        # size in the same box, measured over the *same* scaling range.
        lo, hi = _central_box(Xf, q=1.0)
        refs = []
        if np.isfinite(D2) and cd["range"][1] > cd["range"][0]:
            for _ in range(3):
                U = lo + self.rng.random(Xf.shape) * (hi - lo)
                refs.append(correlation_dimension(U, fixed_range=cd["range"])["D2"])
        finite_refs = [x for x in refs if np.isfinite(x)]
        D2_ref = float(np.mean(finite_refs)) if finite_refs else float("nan")
        deficit = (D2_ref - D2) / D2_ref if np.isfinite(D2_ref) and D2_ref > 0 and np.isfinite(D2) else 0.0
        valid = np.isfinite(D2) and D2 > 0.2
        score = _clip01(cd["r2"] * _clip01(deficit / 0.5)) if valid else 0.0
        bc = box_counting_dimension(Xf)
        return {"score": score, "D2": D2, "D2_poisson_reference": D2_ref, "dimension_deficit": deficit,
                "fit_r2": cd["r2"], "fit_range": cd["range"], "box_dimension": bc, "embedding_dimension": d}

    def symmetry_breaking(self, res: Any, ops: Dict[str, np.ndarray]) -> Dict[str, Any]:
        noe = res.noether
        laws_isotropic = len(noe.angular_axes) >= (3 if res.dim == 3 else 1)
        out: Dict[str, Any] = {"events": {}}
        best = 0.0
        for name in ("asphericity", "polarization", "milling", "spin"):
            x = ops[name]
            cp = cusum_change_point(x)
            magnitude = abs(cp["delta"])
            s = _clip01(magnitude / 0.3) * _clip01(cp["effect"] / 3.0)
            spontaneous = bool(cp["delta"] > 0 and laws_isotropic and name in ("asphericity", "spin", "milling"))
            if not spontaneous:
                s *= 0.5  # explicitly broken by the laws, or symmetry-restoring
            out["events"][name] = {"delta": cp["delta"], "effect": cp["effect"], "index": cp["index"],
                                   "spontaneous": spontaneous, "score": s}
            best = max(best, s)
        out["score"] = best
        out["laws_rotationally_symmetric"] = bool(laws_isotropic)
        return out

    def clustering(self, res: Any) -> Dict[str, Any]:
        Xf = res.positions[-1]
        n, d = Xf.shape
        periodic = res.config.get("boundary") == "periodic"
        box = float(res.config.get("box_size")) if periodic else None
        if periodic:
            V0 = box ** d
        else:
            X0 = res.positions[0]
            r90 = float(np.percentile(np.linalg.norm(X0 - X0.mean(0), axis=1), 90))
            V0 = (math.pi ** (d / 2) / math.gamma(d / 2 + 1)) * r90 ** d / 0.9
        ell = (V0 / n) ** (1.0 / d)
        b = 0.2 * ell
        fof = friends_of_friends(Xf, b, min_members=5, boxsize=box)
        soft = float(res.config.get("softening", 0.01))
        lo, hi = _central_box(Xf, q=1.0)
        rmax = 0.5 * float(np.min(hi - lo)) if not periodic else 0.5 * box
        rmin = max(2 * soft, 1e-3)
        xi_fit: Dict[str, Any] = {"r0": float("nan"), "gamma": float("nan")}
        if rmax > rmin * 2:
            edges = np.geomspace(rmin, rmax, 11)
            xi = landy_szalay(Xf, edges, rng=self.rng, boxsize=box)
            rc = np.sqrt(edges[1:] * edges[:-1])
            ok = np.isfinite(xi) & (xi > 0)
            if ok.sum() >= 3:
                coef = np.polyfit(np.log(rc[ok]), np.log(xi[ok]), 1)
                gamma = -coef[0]
                r0 = math.exp(coef[1] / gamma) if gamma > 0 else float("nan")
                xi_fit = {"r0": float(r0), "gamma": float(gamma)}
            xi_fit.update({"r": rc.tolist(), "xi": xi.tolist()})
        multi = 0.5 + 0.5 * min(1.0, max(fof["n_groups"] - 1, 0) / 4.0)
        score = _clip01(fof["clustered_fraction"] * multi)
        return {"score": score, "linking_length": b, "n_groups": fof["n_groups"], "group_sizes": fof["group_sizes"][:20],
                "clustered_fraction": fof["clustered_fraction"], "largest_fraction": fof["largest_fraction"],
                "correlation": xi_fit}

    # -- field phenomena ------------------------------------------------------------------
    def pattern_formation(self, fr: Any) -> Dict[str, Any]:
        v = fr.final
        sd = float(np.std(v))
        thr = v.mean() + 0.5 * sd
        lab, ncomp = ndimage.label(v > thr)
        k, E = fr.spectrum_k, fr.spectrum_E
        kp = int(np.argmax(E[1:]) + 1) if E.size > 1 else 0
        wavelength = fr.n / kp if kp else float("nan")
        coarse = v.reshape(16, v.shape[0] // 16, 16, -1).mean(axis=(1, 3)) if v.shape[0] % 16 == 0 else v
        p = coarse.ravel() - coarse.min()
        p = p / max(p.sum(), 1e-300)
        nz = p[p > 0]
        H = float(-np.sum(nz * np.log(nz)) / math.log(p.size)) if p.size > 1 else 0.0
        score = _clip01(min(1.0, sd / 0.08) * (1.0 - math.exp(-ncomp / 6.0))) if not fr.blew_up else 0.0
        return {"score": score, "std": sd, "components": int(ncomp), "wavelength": wavelength, "entropy": H}

    def coherent_vortices(self, fr: Any) -> Dict[str, Any]:
        w = fr.final
        sd = float(np.std(w))
        if sd < 1e-12 or fr.blew_up:
            return {"score": 0.0, "n_vortices": 0, "kurtosis": float("nan"), "spectral_slope": float("nan")}
        z = (w - w.mean()) / sd
        kurt = float(np.mean(z ** 4))
        n_v = ndimage.label(z > 2.0)[1] + ndimage.label(z < -2.0)[1]
        k, E = fr.spectrum_k, fr.spectrum_E
        sel = (k >= 3) & (k <= fr.n // 4) & (E > 0)
        slope = float(np.polyfit(np.log(k[sel]), np.log(E[sel]), 1)[0]) if sel.sum() >= 3 else float("nan")
        score = _clip01(min(1.0, n_v / 10.0) * _clip01((kurt - 3.0) / 3.0 + 0.3))
        return {"score": score, "n_vortices": int(n_v), "kurtosis": kurt, "spectral_slope": slope}

    def rigid_symmetry_breaking(self, rb: Any) -> Dict[str, Any]:
        flips = np.asarray(rb.flips)
        align = np.asarray(rb.final_axis_alignment)
        score = _clip01(0.5 * min(1.0, float(np.mean(flips)) / 2.0) + 0.5 * float(np.mean(align > 0.9)))
        return {"score": score, "mean_flips": float(np.mean(flips)), "major_axis_fraction": float(np.mean(align > 0.9))}

    # -- top level ---------------------------------------------------------------------------
    def analyze(self, run: Any, stability: Optional[float] = None) -> EmergenceReport:
        res = getattr(run, "particles", run)
        fields = getattr(run, "fields", {}) or {}
        rigid = getattr(run, "rigid", None)
        if stability is None:
            ev = getattr(run, "evaluation", None)
            if ev is not None:
                stability = ev.stability
            else:
                total = res.config["steps"] * res.config["dt"]
                stability = (res.blowup_time / total) if res.blew_up else 1.0
        metrics: Dict[str, Dict[str, Any]] = {}
        scores: Dict[str, float] = {}
        if len(res.times) < 8:
            for p in PHENOMENA:
                scores[p] = 0.0
                metrics[p] = {"score": 0.0, "note": "too few records"}
        else:
            ops = self.order_parameter_series(res)
            metrics["self_organization"] = self.self_organization(res, ops)
            metrics["oscillation"] = self.oscillation(res, ops)
            metrics["attractor"] = self.attractor(res, ops)
            metrics["chaos"] = self.chaos(res, ops)
            metrics["fractal"] = self.fractal(res)
            metrics["symmetry_breaking"] = self.symmetry_breaking(res, ops)
            metrics["clustering"] = self.clustering(res)
            metrics["order_parameters"] = {k: v.tolist() for k, v in ops.items()}
            for p in PHENOMENA:
                scores[p] = float(metrics[p]["score"])
        if "reaction_diffusion" in fields:
            metrics["pattern_formation"] = self.pattern_formation(fields["reaction_diffusion"])
            scores["pattern_formation"] = metrics["pattern_formation"]["score"]
        if "vorticity" in fields:
            metrics["coherent_vortices"] = self.coherent_vortices(fields["vorticity"])
            scores["coherent_vortices"] = metrics["coherent_vortices"]["score"]
        if rigid is not None:
            metrics["rotational_symmetry_breaking"] = self.rigid_symmetry_breaking(rigid)
            scores["rotational_symmetry_breaking"] = metrics["rotational_symmetry_breaking"]["score"]
        detected = {k: v >= self.threshold for k, v in scores.items()}
        w = np.array([self.weights.get(k, 1.0) for k in scores])
        s = np.array(list(scores.values()))
        mean_w = float(np.sum(w * s) / np.sum(w)) if s.size else 0.0
        coverage = float(np.mean(s >= self.threshold)) if s.size else 0.0
        richness = float(stability * (0.7 * mean_w + 0.3 * coverage))
        descriptor = [scores.get(p, 0.0) for p in PHENOMENA + FIELD_PHENOMENA]
        return EmergenceReport(scores=scores, detected=detected, metrics=metrics, richness=richness,
                               stability=float(stability), descriptor=descriptor)
