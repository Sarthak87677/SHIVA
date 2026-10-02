"""
SHIVA-1 :: Visualizer
=====================

Publication-style static figures (matplotlib, Agg) for every layer of the
system, plus a multi-domain dashboard.

Design rules (applied uniformly):

* one validated categorical palette in a **fixed slot order** (identity never
  depends on rank); scatter plots use at most three hues and add marker shape
  as a secondary channel;
* magnitude uses a single-hue light->dark ramp; signed fields use a blue<->red
  diverging map with a neutral grey midpoint;
* thin marks, hairline solid gridlines, text in ink colours (never in series
  colours), a legend whenever two or more series share an axis, and never a
  second y-axis.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import matplotlib

if "ipykernel" not in sys.modules and not os.environ.get("MPLBACKEND"):
    matplotlib.use("Agg")  # headless by default; Jupyter keeps its inline backend
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
from cycler import cycler  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

__all__ = [
    "CATEGORICAL", "STYLE", "SEQUENTIAL", "DIVERGING",
    "plot_force_laws", "plot_conservation", "plot_particles", "plot_emergence", "plot_correlation",
    "plot_fields", "plot_rigid", "plot_search", "plot_pareto", "plot_design_gallery", "plot_bracket",
    "plot_dashboard", "plot_rl",
]

# ---------------------------------------------------------------------------
# Palette (validated reference instance; light mode)
# ---------------------------------------------------------------------------

CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf",
             "#1c5cab", "#184f95", "#104281", "#0d366b"]
SEQUENTIAL = LinearSegmentedColormap.from_list("shiva_seq", BLUE_RAMP)
DIVERGING = LinearSegmentedColormap.from_list(
    "shiva_div", ["#0d366b", "#2a78d6", "#9ec5f4", "#f0efec", "#f2aaa9", "#e34948", "#8f2222"])
EMPHASIS_OTHER = "#c3c2b7"
for _cm in (SEQUENTIAL, DIVERGING):
    if _cm.name not in matplotlib.colormaps:
        matplotlib.colormaps.register(_cm)

STYLE = {
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": BASELINE, "axes.labelcolor": INK2, "axes.titlecolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
    "lines.linewidth": 1.8, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
    "font.family": "sans-serif", "font.size": 8.5, "axes.titlesize": 9.5, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.labelsize": 8.5, "legend.frameon": False, "legend.fontsize": 7.5,
    "axes.prop_cycle": cycler(color=CATEGORICAL), "figure.dpi": 110, "savefig.dpi": 130,
    "savefig.bbox": "tight", "image.cmap": "shiva_seq",
}


def _save(fig: plt.Figure, path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def _ctx():
    return plt.rc_context(STYLE)


# ---------------------------------------------------------------------------
# Physics
# ---------------------------------------------------------------------------

def plot_force_laws(genomes: Sequence[Any], path: Optional[str] = None, labels: Optional[Sequence[str]] = None,
                    r: Optional[np.ndarray] = None) -> Optional[str]:
    """|F(r)| on log-log axes and F(r)/F_Newton(r) for up to 8 genomes."""
    from .physics_mutator import SymbolicPhysics

    r = np.geomspace(0.05, 10.0, 500) if r is None else r
    genomes = list(genomes)[:8]
    labels = list(labels or [g.name or f"{g.gravity.law} ({g.genome_id[:6]})" for g in genomes])
    with _ctx():
        fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
        for k, (g, lab) in enumerate(zip(genomes, labels)):
            sym = SymbolicPhysics(g)
            if g.gravity.law in ("tensor", "fractal") and g.gravity.anisotropic:
                # along the eigen-direction with the smallest eigenvalue: rho = sqrt(lam) r,
                # |F| = sqrt(lam) U'(sqrt(lam) r)
                lam = float(np.min(np.linalg.eigvalsh(g.gravity.Q(3))))
                F = math.sqrt(lam) * sym.force_profile(math.sqrt(lam) * r)
                lab = f"{lab}, softest axis"
            else:
                F = sym.force_profile(r)
            Fn = g.gravity.G / r ** 2
            axes[0].plot(r, np.abs(F), color=CATEGORICAL[k], label=lab)
            axes[1].plot(r, -F / Fn * (1 if g.gravity.G else 0), color=CATEGORICAL[k], label=lab)
        axes[0].set_xscale("log")
        axes[0].set_yscale("log")
        axes[0].set_xlabel("separation r")
        axes[0].set_ylabel("|F(r)|  (unit masses)")
        axes[0].set_title("Mutated gravitational force laws")
        axes[1].set_xscale("log")
        axes[1].axhline(1.0, color=BASELINE, lw=0.8)
        axes[1].set_xlabel("separation r")
        axes[1].set_ylabel("attraction relative to Newton")
        axes[1].set_title("Running coupling  G_eff(r) / G  (negative = repulsive shell)")
        handles, labs = axes[0].get_legend_handles_labels()
        fig.legend(handles, labs, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
        fig.tight_layout(rect=(0, 0.12, 1, 1))
        return _save(fig, path)


def plot_conservation(res: Any, path: Optional[str] = None, evaluation: Any = None) -> Optional[str]:
    """Relative drift of energy, (pseudo-)momentum and (canonical) angular
    momentum against the Noether prediction."""
    n = res.noether
    t = res.times
    E = res.energy
    axes_vec = np.array(n.angular_axes) if n.angular_axes else np.eye(3)
    L = res.angular_momentum @ axes_vec.T
    series = [
        ("energy", np.abs(E - E[0]) / res.energy_scale, n.energy),
        (f"momentum ({n.momentum_kind if n.momentum else 'mechanical'})",
         np.linalg.norm(res.momentum - res.momentum[0], axis=1) / res.momentum_scale, n.momentum),
        (f"angular momentum ({n.angular_kind if n.angular_axes else 'mechanical'})",
         np.max(np.abs(L - L[0]), axis=1) / res.angular_scale, n.angular_momentum),
    ]
    with _ctx():
        fig, ax = plt.subplots(figsize=(6.4, 3.4))
        for k, (name, y, pred) in enumerate(series):
            ax.plot(t, np.maximum(y, 1e-17), color=CATEGORICAL[k],
                    label=f"{name}: {'conserved' if pred else 'broken'} (Noether)")
        ax.set_yscale("log")
        ax.set_ylim(1e-18, max(1e4, 1e4 * ax.get_ylim()[1]))
        ax.set_xlabel("time")
        ax.set_ylabel("relative drift |Q(t) - Q(0)| / scale")
        ax.set_title("Conservation laws: predicted (symmetry) vs measured (simulation)")
        ax.legend(loc="upper right")
        fig.tight_layout()
        return _save(fig, path)


def plot_particles(res: Any, path: Optional[str] = None, frames: Optional[Sequence[int]] = None,
                   title: str = "") -> Optional[str]:
    R = len(res.times)
    frames = list(frames or [0, R // 3, 2 * R // 3, R - 1])
    sp = np.asarray(res.species)
    n_sp = int(sp.max()) + 1 if sp.size else 1
    with _ctx():
        fig, axes = plt.subplots(1, len(frames), figsize=(2.6 * len(frames), 2.75), squeeze=False)
        lim = np.percentile(np.abs(res.positions[frames[-1]][:, :2]), 97) * 1.25 + 1e-9
        lim = max(lim, np.percentile(np.abs(res.positions[0][:, :2]), 99) * 1.1)
        for ax, f in zip(axes[0], frames):
            X = res.positions[f]
            for s in range(n_sp):
                m = sp == s
                ax.scatter(X[m, 0], X[m, 1], s=7, color=CATEGORICAL[s % 3], marker=MARKERS[s % len(MARKERS)],
                           linewidths=0, alpha=0.9, label=f"species {s}" if n_sp > 1 else None)
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_aspect("equal")
            ax.set_title(f"t = {res.times[f]:.2f}")
            ax.set_xlabel("x")
        axes[0][0].set_ylabel("y")
        if n_sp > 1:
            axes[0][-1].legend(loc="upper right", markerscale=1.5)
        if title:
            fig.suptitle(title, x=0.01, ha="left", fontsize=10, fontweight="bold")
        fig.tight_layout()
        return _save(fig, path)


def plot_emergence(report: Any, path: Optional[str] = None, title: str = "Emergent phenomena") -> Optional[str]:
    items = list(report.scores.items())
    names = [k.replace("_", " ") for k, _ in items]
    vals = np.array([v for _, v in items])
    with _ctx():
        fig, ax = plt.subplots(figsize=(6.0, 0.32 * len(items) + 1.0))
        y = np.arange(len(items))[::-1]
        ax.barh(y, vals, height=0.55, color=CATEGORICAL[0])
        ax.axvline(0.25, color=MUTED, lw=0.8)
        ax.text(0.255, y.max() + 0.55, "detection threshold", color=INK2, fontsize=7, va="bottom")
        for yi, v in zip(y, vals):
            ax.text(min(v + 0.015, 0.93), yi, f"{v:.2f}", va="center", fontsize=7.5, color=INK2)
        ax.set_yticks(y, names)
        ax.set_xlim(0, 1.0)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("score")
        ax.set_title(f"{title}   (richness R = {report.richness:.3f})")
        fig.tight_layout()
        return _save(fig, path)


def plot_correlation(report: Any, path: Optional[str] = None) -> Optional[str]:
    c = report.metrics.get("clustering", {}).get("correlation", {})
    r, xi = np.array(c.get("r", []), float), np.array(c.get("xi", []), float)
    with _ctx():
        fig, ax = plt.subplots(figsize=(4.6, 3.3))
        ok = np.isfinite(xi) & (xi > 0)
        if ok.any():
            ax.plot(r[ok], xi[ok], marker="o", ms=4, color=CATEGORICAL[0], label="Landy-Szalay estimate")
            g, r0 = c.get("gamma"), c.get("r0")
            if g and r0 and np.isfinite(g) and np.isfinite(r0):
                rr = np.geomspace(r[ok].min(), r[ok].max(), 50)
                ax.plot(rr, (rr / r0) ** (-g), color=CATEGORICAL[1], lw=1.2,
                        label=f"fit (r/{r0:.2f})^-{g:.2f}")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.legend(loc="upper right")
        ax.set_xlabel("r")
        ax.set_ylabel("xi(r)")
        ax.set_title("Two-point correlation function")
        fig.tight_layout()
        return _save(fig, path)


def plot_fields(fields: Dict[str, Any], path: Optional[str] = None) -> Optional[str]:
    with _ctx():
        fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.3))
        vo = fields.get("vorticity")
        rd = fields.get("reaction_diffusion")
        if vo is not None:
            w = vo.final
            lim = np.percentile(np.abs(w), 99)
            im = axes[0].imshow(w.T, origin="lower", cmap=DIVERGING, vmin=-lim, vmax=lim)
            axes[0].set_title("Vorticity (mutated viscosity)")
            fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.03)
            k, E = vo.spectrum_k, vo.spectrum_E
            ok = (k > 0) & (E > 0)
            axes[2].loglog(k[ok], E[ok], color=CATEGORICAL[0], label="E(k)")
            kk = np.array([3.0, k[ok].max()])
            ref = E[ok][np.argmin(np.abs(k[ok] - 3))]
            axes[2].loglog(kk, ref * (kk / 3) ** -3, color=CATEGORICAL[1], lw=1.0, label="k^-3 (enstrophy cascade)")
            axes[2].loglog(kk, ref * (kk / 3) ** (-5 / 3), color=CATEGORICAL[2], lw=1.0, label="k^-5/3 (energy cascade)")
            axes[2].set_xlabel("wavenumber k")
            axes[2].set_ylabel("E(k)")
            axes[2].set_title("Energy spectrum")
            axes[2].legend(loc="lower left")
        if rd is not None:
            im = axes[1].imshow(rd.final.T, origin="lower", cmap=SEQUENTIAL)
            axes[1].set_title("Reaction-diffusion (Gray-Scott, v)")
            fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.03)
        for ax in axes[:2]:
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.tight_layout()
        return _save(fig, path)


def plot_rigid(rb: Any, path: Optional[str] = None, body: int = 0) -> Optional[str]:
    with _ctx():
        fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.2))
        w = rb.omega_body[:, body]
        for i, lab in enumerate(("omega_1 (minor axis)", "omega_2 (intermediate)", "omega_3 (major axis)")):
            axes[0].plot(rb.times, w[:, i], color=CATEGORICAL[i], lw=1.3, label=lab)
        axes[0].set_xlabel("time")
        axes[0].set_ylabel("body angular velocity")
        axes[0].set_title(f"Intermediate-axis instability (body {body}, {int(rb.flips[body])} flips)")
        axes[0].legend(loc="lower right")
        E = rb.energy_rot
        for b in range(E.shape[1]):
            axes[1].plot(rb.times, E[:, b] / E[0, b], color=CATEGORICAL[0] if b == body else EMPHASIS_OTHER, lw=1.3)
        axes[1].set_xlabel("time")
        axes[1].set_ylabel("rotational energy / initial")
        axes[1].set_title(f"Energy at constant |L| (dissipation {rb.dissipation:.3g})")
        fig.tight_layout()
        return _save(fig, path)


# ---------------------------------------------------------------------------
# Search and evolution
# ---------------------------------------------------------------------------

def plot_search(history: Sequence[Dict[str, Any]], path: Optional[str] = None,
                title: str = "Universe search") -> Optional[str]:
    g = [h["generation"] for h in history]
    with _ctx():
        fig, ax = plt.subplots(figsize=(5.2, 3.2))
        ax.plot(g, [h["best_fitness"] for h in history], marker="o", ms=4, color=CATEGORICAL[0], label="best")
        ax.plot(g, [h["mean_fitness"] for h in history], marker="o", ms=4, color=CATEGORICAL[1], label="population mean")
        ax.set_xlabel("generation")
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax.set_ylabel("fitness = R x (0.75 + 0.25 F) x C")
        ax.set_title(title)
        ax.legend(loc="lower right")
        fig.tight_layout()
        return _save(fig, path)


def plot_pareto(results: Dict[str, Any], path: Optional[str] = None, x: str = "efficiency",
                y: str = "mappability") -> Optional[str]:
    names = list(results)
    cols = 4
    rows = int(math.ceil(len(names) / cols))
    with _ctx():
        fig, axes = plt.subplots(rows, cols, figsize=(2.6 * cols, 2.4 * rows), squeeze=False)
        for ax, name in zip(axes.ravel(), names):
            r = results[name]
            P = np.array([[d.objectives[x], d.objectives[y]] for d in r.pareto])
            ax.scatter(P[:, 0], P[:, 1], s=12, color=EMPHASIS_OTHER, linewidths=0, label="Pareto set")
            c = r.champion
            ax.scatter([c.objectives[x]], [c.objectives[y]], s=40, color=CATEGORICAL[0], edgecolors=SURFACE,
                       linewidths=1.5, label="champion", zorder=3)
            ax.set_xlim(-0.03, 1.03)
            ax.set_ylim(-0.03, 1.03)
            ax.set_title(name.replace("_", " "), fontsize=8.5)
            ax.set_xlabel(x, fontsize=7.5)
            ax.set_ylabel(y, fontsize=7.5)
        for ax in axes.ravel()[len(names):]:
            ax.axis("off")
        axes[0][0].legend(loc="lower left", fontsize=6.5)
        fig.suptitle(f"Evolved technologies: {x} vs {y}", x=0.01, ha="left", fontsize=10, fontweight="bold")
        fig.tight_layout()
        return _save(fig, path)


def plot_rl(rl: Dict[str, Dict[str, Any]], path: Optional[str] = None) -> Optional[str]:
    with _ctx():
        fig, ax = plt.subplots(figsize=(5.4, 3.2))
        for k, (name, d) in enumerate(list(rl.items())[:8]):
            curve = d.get("learning_curve") or []
            if curve:
                ax.plot(np.arange(1, len(curve) + 1), curve, color=CATEGORICAL[k], lw=1.3, label=name.replace("_", " "))
        ax.set_xlabel("policy-gradient iteration")
        ax.set_ylabel("mean episode improvement")
        ax.set_title("Adaptive design agent (REINFORCE)")
        ax.legend(loc="lower right", fontsize=6.5)
        fig.tight_layout()
        return _save(fig, path)


# ---------------------------------------------------------------------------
# Designs
# ---------------------------------------------------------------------------

def _draw_design(ax: plt.Axes, domain: str, p: Dict[str, float]) -> None:
    from .cad_generator import impeller_blade, involute_gear_polygon, nozzle_profile

    ax.grid(False)
    ax.set_aspect("equal")
    col, light = CATEGORICAL[0], "#cde2fb"
    if domain == "gear_pair":
        from .tech_evolver import GearDomain
        mod = p["module"] * 1e3
        z1 = int(p["z1"])
        z2 = int(round(GearDomain.RATIO * z1))
        g1 = involute_gear_polygon(mod, z1, p["pressure_angle"], p["shift"])
        g2 = involute_gear_polygon(mod, z2, p["pressure_angle"], -p["shift"])
        rot = math.pi / z2 if z2 % 2 == 0 else 0.0
        c, s = math.cos(rot), math.sin(rot)
        g2 = g2 @ np.array([[c, s], [-s, c]]) + [mod * (z1 + z2) / 2, 0]
        for g in (g1, g2):
            ax.fill(g[:, 0], g[:, 1], color=light, lw=0)
            ax.plot(np.r_[g[:, 0], g[0, 0]], np.r_[g[:, 1], g[0, 1]], color=col, lw=0.6)
    elif domain == "propulsion_nozzle":
        inner, _ = nozzle_profile(p)
        t = p["wall_thickness"] * 1e3
        for sgn in (1, -1):
            ax.fill_betweenx(inner[:, 1], sgn * inner[:, 0], sgn * (inner[:, 0] + t), color=col, lw=0)
            ax.plot(sgn * inner[:, 0], inner[:, 1], color=col, lw=0.8)
    elif domain == "structure_truss":
        from .tech_evolver import TrussDomain
        d = TrussDomain()
        xy = d.nodes(p)
        for i, j, g in d.MEMBERS:
            w = 1.0 + 6.0 * math.sqrt(p[f"area_{g}"] / 3e-3)
            ax.plot(xy[[i, j], 0], xy[[i, j], 1], color=col, lw=w, solid_capstyle="round")
        ax.plot([0, 0], [-0.1, max(p["h0"], 0.3) + 0.1], color=INK2, lw=2.5)
        ax.annotate("", xy=(xy[3, 0], -0.25), xytext=(xy[3, 0], 0.0), arrowprops=dict(arrowstyle="-|>", color=INK2))
    elif domain == "energy_flywheel":
        R, a, h = p["outer_radius"], p["inner_ratio"] * p["outer_radius"], p["thickness"]
        hub = max(0.6 * a, 0.015)
        for sgn in (1, -1):
            ax.add_patch(plt.Rectangle((a if sgn > 0 else -R, -h / 2), R - a, h, color=col, lw=0))
            ax.add_patch(plt.Rectangle((hub if sgn > 0 else -a, -0.15 * h), max(a - hub, 0), 0.3 * h, color=light, lw=0))
        ax.add_patch(plt.Rectangle((-hub, -0.6 * h), 2 * hub, 1.2 * h, color=light, lw=0))
        ax.plot([0, 0], [-0.8 * R, 0.8 * R], color=MUTED, lw=0.8)
        ax.text(0, -0.55 * R, "cross-section", ha="center", fontsize=6.5, color=INK2)
        ax.set_xlim(-1.1 * R, 1.1 * R)
        ax.set_ylim(-0.6 * R, 0.6 * R)
    elif domain == "drone_frame":
        n, L, Dp = int(p["n_arms"]), p["arm_length"], p["prop_diameter"]
        for k in range(n):
            ang = 2 * math.pi * k / n
            ax.plot([0, L * math.cos(ang)], [0, L * math.sin(ang)], color=col, lw=2.5)
            ax.add_patch(plt.Circle((L * math.cos(ang), L * math.sin(ang)), Dp / 2, fill=False, color=MUTED, lw=0.8))
        ax.add_patch(plt.Circle((0, 0), max(0.035, 0.3 * L), color=light, lw=0))
        ax.set_xlim(-(L + Dp / 2) * 1.05, (L + Dp / 2) * 1.05)
        ax.set_ylim(-(L + Dp / 2) * 1.05, (L + Dp / 2) * 1.05)
    elif domain == "pump_impeller":
        blade = impeller_blade(p)
        r2 = p["r2"] * 1e3
        ax.add_patch(plt.Circle((0, 0), r2, color=light, lw=0))
        ax.add_patch(plt.Circle((0, 0), p["r1"] * 1e3, fill=False, color=MUTED, lw=0.8))
        Z = int(p["blades"])
        for k in range(Z):
            c, s = math.cos(2 * math.pi * k / Z), math.sin(2 * math.pi * k / Z)
            b = blade @ np.array([[c, s], [-s, c]])
            ax.fill(b[:, 0], b[:, 1], color=col, lw=0)
        ax.set_xlim(-1.05 * r2, 1.05 * r2)
        ax.set_ylim(-1.05 * r2, 1.05 * r2)
    elif domain == "cosmic_habitat":
        R, W = p["radius"], p["width"]
        th = np.linspace(0, 2 * np.pi, 200)
        ax.plot(R * np.cos(th), R * np.sin(th), color=col, lw=3)
        for k in range(6):
            a = 2 * np.pi * k / 6
            ax.plot([0, R * math.cos(a)], [0, R * math.sin(a)], color=MUTED, lw=0.8)
        ax.text(0, -1.25 * R, f"R = {R:.0f} m, W = {W:.0f} m, {p['rpm']:.2f} rpm", ha="center", fontsize=6.5, color=INK2)
        ax.set_xlim(-1.3 * R, 1.3 * R)
        ax.set_ylim(-1.4 * R, 1.2 * R)
    elif domain == "cosmic_lightsail":
        L = p["side_length"]
        ax.add_patch(plt.Rectangle((-L / 2, -L / 2), L, L, color=light, lw=0))
        ax.plot([-L / 2, L / 2], [-L / 2, L / 2], color=col, lw=1.5)
        ax.plot([-L / 2, L / 2], [L / 2, -L / 2], color=col, lw=1.5)
        ax.set_xlim(-0.6 * L, 0.6 * L)
        ax.set_ylim(-0.6 * L, 0.6 * L)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def plot_design_gallery(designs: Dict[str, Dict[str, float]], path: Optional[str] = None,
                        captions: Optional[Dict[str, str]] = None, title: str = "Evolved champions") -> Optional[str]:
    names = list(designs)
    cols = 4
    rows = int(math.ceil(len(names) / cols))
    with _ctx():
        fig, axes = plt.subplots(rows, cols, figsize=(2.7 * cols, 2.7 * rows), squeeze=False)
        for ax, name in zip(axes.ravel(), names):
            _draw_design(ax, name, designs[name])
            ax.set_title(name.replace("_", " "), fontsize=8.5)
            if captions and name in captions:
                ax.set_xlabel(captions[name], fontsize=6.5, color=INK2)
        for ax in axes.ravel()[len(names):]:
            ax.axis("off")
        fig.suptitle(title, x=0.01, ha="left", fontsize=10, fontweight="bold")
        fig.tight_layout()
        return _save(fig, path)


def plot_bracket(br: Dict[str, Any], path: Optional[str] = None) -> Optional[str]:
    with _ctx():
        fig, axes = plt.subplots(1, 3, figsize=(10.0, 2.8))
        axes[0].imshow(br["density"], cmap=SEQUENTIAL, origin="upper")
        axes[0].set_title("CPPN density field")
        axes[1].imshow(br["shape"], cmap=LinearSegmentedColormap.from_list("bin", [SURFACE, CATEGORICAL[0]]),
                       origin="upper")
        axes[1].set_title(f"Projected shape (volume {br['metrics'].get('volume_fraction', 0):.2f})")
        axes[1].annotate("clamped", xy=(0, br["shape"].shape[0] / 2), xytext=(-6, br["shape"].shape[0] / 2),
                         fontsize=7, color=INK2, ha="right", va="center")
        for ax in axes[:2]:
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
        h = br["history"]
        axes[2].plot(np.arange(1, len(h) + 1), [x["best"] for x in h], color=CATEGORICAL[0], marker="o", ms=3)
        axes[2].set_xlabel("NEAT generation")
        axes[2].set_ylabel("stiffness / solid-plate stiffness")
        axes[2].set_title("Generative bracket evolution")
        fig.tight_layout()
        return _save(fig, path)


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

def plot_dashboard(run: Any, report: Any, search: Optional[Dict[str, Any]] = None,
                   tech: Optional[Dict[str, Any]] = None, prototypes: Optional[Dict[str, Any]] = None,
                   path: Optional[str] = None) -> Optional[str]:
    """One-page multi-domain overview."""
    res = run.particles
    with _ctx():
        fig = plt.figure(figsize=(13.0, 8.0))
        gs = fig.add_gridspec(3, 4, height_ratios=[0.32, 1, 1], width_ratios=[1, 1, 0.42, 1.55], hspace=0.5,
                              wspace=0.3)
        top = gs[0, :].subgridspec(1, 4)
        # Row 0: stat tiles
        tiles = [("emergent richness", f"{report.richness:.3f}"),
                 ("Noether consistency", f"{run.evaluation.conservation_consistency:.0%}"),
                 ("Lyapunov exponent", f"{res.lyapunov:.2f}" if res.lyapunov is not None else "n/a"),
                 ("buildable prototypes",
                  f"{sum(1 for p in (prototypes or {}).values() if p.feasible)}/{len(prototypes or {})}")]
        for k, (lab, val) in enumerate(tiles):
            ax = fig.add_subplot(top[0, k])
            ax.axis("off")
            ax.text(0.0, 0.45, val, fontsize=24, fontweight="bold", color=INK, transform=ax.transAxes)
            ax.text(0.0, 0.0, lab, fontsize=9, color=INK2, transform=ax.transAxes)
        # Row 1
        ax = fig.add_subplot(gs[1, 0])
        X = res.positions[-1]
        sp = np.asarray(res.species)
        for s in range(int(sp.max()) + 1):
            m = sp == s
            ax.scatter(X[m, 0], X[m, 1], s=5, color=CATEGORICAL[s % 3], marker=MARKERS[s % len(MARKERS)], linewidths=0)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(f"Universe at t = {res.times[-1]:.1f}")
        ax = fig.add_subplot(gs[1, 1])
        E = res.energy
        ax.plot(res.times, np.maximum(np.abs(E - E[0]) / res.energy_scale, 1e-17), color=CATEGORICAL[0], label="energy")
        ax.plot(res.times, np.maximum(np.linalg.norm(res.momentum - res.momentum[0], axis=1) / res.momentum_scale, 1e-17),
                color=CATEGORICAL[1], label="momentum")
        ax.set_yscale("log")
        ax.set_ylim(1e-18, 1e4)
        ax.set_title("Conservation drift")
        ax.set_xlabel("time")
        ax.legend(loc="upper right")
        ax = fig.add_subplot(gs[1, 3])
        items = list(report.scores.items())
        y = np.arange(len(items))[::-1]
        ax.barh(y, [v for _, v in items], height=0.55, color=CATEGORICAL[0])
        ax.axvline(0.25, color=MUTED, lw=0.8)
        ax.set_yticks(y, [k.replace("_", " ") for k, _ in items], fontsize=7)
        ax.set_xlim(0, 1)
        ax.grid(axis="y", visible=False)
        ax.set_title("Emergent phenomena")
        # Row 2
        ax = fig.add_subplot(gs[2, 0])
        if search and search.get("history"):
            h = search["history"]
            ax.plot([x["generation"] for x in h], [x["best_fitness"] for x in h], color=CATEGORICAL[0], marker="o",
                    ms=3, label="best")
            ax.plot([x["generation"] for x in h], [x["mean_fitness"] for x in h], color=CATEGORICAL[1], marker="o",
                    ms=3, label="mean")
            ax.legend(loc="lower right")
            ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax.set_title("Universe search")
        ax.set_xlabel("generation")
        ax = fig.add_subplot(gs[2, 1])
        vo = run.fields.get("vorticity") if run.fields else None
        if vo is not None:
            lim = np.percentile(np.abs(vo.final), 99)
            ax.imshow(vo.final.T, origin="lower", cmap=DIVERGING, vmin=-lim, vmax=lim)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("Vorticity field")
        ax = fig.add_subplot(gs[2, 3])
        if tech:
            names = list(tech)
            objs = ("efficiency", "stability", "adaptability", "mappability", "cosmic")
            M = np.array([[tech[n].champion.objectives[o] for o in objs] for n in names])
            im = ax.imshow(M, cmap=SEQUENTIAL, vmin=0, vmax=1, aspect="auto")
            ax.set_xticks(range(len(objs)), objs, fontsize=7)
            ax.set_yticks(range(len(names)), [n.replace("_", " ") for n in names], fontsize=7)
            for i in range(M.shape[0]):
                for j in range(M.shape[1]):
                    ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                            color="#ffffff" if M[i, j] > 0.55 else INK)
            ax.grid(False)
        ax.set_title("Technology champions (objective scores)")
        fig.suptitle("SHIVA-1 multi-domain dashboard", x=0.01, ha="left", fontsize=12, fontweight="bold")
        return _save(fig, path)
