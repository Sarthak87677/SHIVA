"""
SHIVA-1 :: Interactive web dashboard (data export + local server)
=================================================================

``python -m shiva_core dashboard`` turns a pipeline run directory into a live
browser dashboard:

* ``build_dashboard_data`` condenses ``summary.json`` and the particle
  trajectories into ``dashboard_data.json`` (re-simulating the two universes
  quickly if the run predates trajectory export);
* ``serve`` starts a local HTTP server (``http://localhost:8765``) that serves
  the web app from ``shiva_dashboard/web``, the run files under ``/run/``, CAD
  models under ``/cad/`` and figures under ``/fig/``.

``localhost`` is a secure context, so the browser may use the webcam for the
optional hand-gesture control.  The 3-D engine (three.js) is vendored and works
offline; hand tracking (MediaPipe) is fetched from a CDN when gestures are
switched on.
"""

from __future__ import annotations

import functools
import json
import math
import threading
import time
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import unquote, urlparse

import numpy as np

__all__ = ["WEB_DIR", "find_latest_run", "trajectory_payload", "build_dashboard_data", "write_dashboard_data",
           "make_server", "serve"]

WEB_DIR = Path(__file__).resolve().parent.parent / "shiva_dashboard" / "web"
DATA_NAME = "dashboard_data.json"

PHENOMENON_INFO = {
    "self_organization": "drop of spatial entropy and growth of statistical complexity as structure forms",
    "oscillation": "resolved periodic modes in global observables (kinetic energy, radius, shape)",
    "attractor": "how deterministic and settled the late-time macro-dynamics are (recurrence analysis)",
    "chaos": "maximal Lyapunov exponent from a shadow trajectory: sensitivity to initial conditions",
    "fractal": "deficit of the correlation dimension relative to random points: scale-dependent clumping",
    "symmetry_breaking": "abrupt growth of shape/flow order parameters; halved if the laws themselves are anisotropic",
    "clustering": "friends-of-friends haloes and the two-point correlation function",
    "pattern_formation": "Turing spots/stripes in the reaction-diffusion field",
    "coherent_vortices": "long-lived vortices in the 2-D flow field",
    "rotational_symmetry_breaking": "intermediate-axis flips and major-axis capture of spinning rigid bodies",
}


def find_latest_run(root: Path = Path("shiva_output")) -> Optional[Path]:
    """Most recently written run directory (one containing summary.json)."""
    cands = sorted(root.glob("*/summary.json"), key=lambda p: p.stat().st_mtime, reverse=True) if root.exists() else []
    if cands:
        return cands[0].parent
    docs = Path(__file__).resolve().parent.parent / "shiva_docs" / "results"
    return docs if (docs / "summary.json").exists() else None


def _cad_dir(run_dir: Path) -> Optional[Path]:
    for cand in (run_dir / "cad", run_dir.parent / "prototypes"):
        if cand.is_dir() and any(cand.glob("*.stl")):
            return cand
    return None


def _fig_dir(run_dir: Path) -> Optional[Path]:
    """``<run>/figures`` for pipeline runs, ``shiva_docs/figures`` for the bundled reference results."""
    for cand in (run_dir / "figures", run_dir.parent / "figures"):
        if cand.is_dir() and any(cand.glob("*.png")):
            return cand
    return None


def trajectory_payload(particles: Any, label: str, description: str, max_frames: int = 150) -> Dict[str, Any]:
    """Downsample a SimulationResult to a compact, JSON-friendly trajectory."""
    X = np.asarray(particles.positions, float)  # (R, N, d)
    R, N, d = X.shape
    idx = np.unique(np.linspace(0, R - 1, min(R, max_frames)).round().astype(int))
    if d == 2:
        X = np.concatenate([X, np.zeros((R, N, 1))], axis=2)
    E = np.asarray(particles.kinetic) + np.asarray(particles.potential)
    drift = np.abs(E - E[0]) / max(float(getattr(particles, "energy_scale", 1.0)), 1e-300)
    rg = [float(np.sqrt(np.mean(np.sum((x - x.mean(0)) ** 2, axis=1)))) for x in X]
    species = np.asarray(getattr(particles, "species", np.zeros(N)), int)
    return {
        "label": label,
        "description": description,
        "n": int(N),
        "times": [round(float(t), 4) for t in np.asarray(particles.times)[idx]],
        "frames": [np.round(X[i], 3).ravel().tolist() for i in idx],
        "energy_drift": [float(f"{v:.3e}") for v in drift[idx]],
        "radius_of_gyration": [round(rg[i], 4) for i in idx],
        "species": species.tolist(),
        "lyapunov": None if particles.lyapunov is None else float(particles.lyapunov),
    }


def _quick_trajectories(run_dir: Path, summary: Dict[str, Any], n: int, steps: int) -> Dict[str, Any]:
    """Re-simulate best and baseline universes (for runs that predate trajectory export)."""
    from .physics_mutator import PhysicsGenome
    from .pipeline import showcase_genomes
    from .universe_simulator import SimulationConfig, UniverseSimulator

    seed = int(summary.get("seed", 42))
    best_path = run_dir / "best_genome.json"
    best = PhysicsGenome.from_json(best_path.read_text(encoding="utf-8")) if best_path.exists() else \
        PhysicsGenome.from_dict(summary["stages"]["search"]["best_genome"])
    base = showcase_genomes(seed)[0]
    out = {}
    for label, g in (("best", best), ("baseline", base)):
        cfg = SimulationConfig(n_particles=n, steps=steps, record_every=max(1, steps // 150),
                               initial_condition="cold_collapse", seed=seed)
        res = UniverseSimulator(g, cfg).run()
        out[label] = trajectory_payload(res, label, g.describe())
    return out


def _cad_info(cad: Optional[Path], name: str) -> Dict[str, Any]:
    """Bounding box, mesh size and model scale of a prototype's CAD export."""
    info: Dict[str, Any] = {}
    if cad is None:
        return info
    meta = cad / f"{name}.json"
    if meta.exists():
        try:
            c = json.loads(meta.read_text(encoding="utf-8")).get("cad", {})
            info = {"bbox_mm": c.get("bbox_mm"), "triangles": c.get("triangles"), "components": c.get("components"),
                    "watertight": c.get("components_watertight")}
        except (ValueError, OSError):
            pass
    scad = cad / f"{name}.scad"
    if scad.exists():
        for line in scad.read_text(encoding="utf-8", errors="replace").splitlines()[:12]:
            if line.startswith("scale_note"):
                info["scale"] = line.split("=", 1)[1].strip().strip(";").strip('"')
    return info


def _force_laws(summary: Dict[str, Any]) -> Dict[str, Any]:
    from .physics_mutator import PhysicsGenome, SymbolicPhysics

    r = np.geomspace(0.05, 10.0, 160)
    laws = []
    entries = [(p["name"], p["genome"]) for p in summary["stages"].get("physics", [])]
    bg = summary["stages"].get("search", {}).get("best_genome")
    if bg:
        entries.append(("evolved best universe", bg))
    for name, gd in entries[:8]:
        g = PhysicsGenome.from_dict(gd)
        sym = SymbolicPhysics(g)
        if g.gravity.law in ("tensor", "fractal") and g.gravity.anisotropic:
            lam = float(np.min(np.linalg.eigvalsh(g.gravity.Q(3))))
            F = math.sqrt(lam) * sym.force_profile(math.sqrt(lam) * r)
            name = f"{name} (softest axis)"
        else:
            F = sym.force_profile(r)
        laws.append({"name": name, "absF": [float(f"{abs(v):.4e}") for v in F],
                     "ratio": [round(float(-v / (g.gravity.G / rr ** 2)), 4) for v, rr in zip(F, r)]})
    return {"r": [round(float(x), 5) for x in r], "laws": laws}


def build_dashboard_data(run_dir: Path, summary: Optional[Dict[str, Any]] = None,
                         trajectories: Optional[Dict[str, Any]] = None, quick_n: int = 128,
                         quick_steps: int = 2000, verbose: bool = True) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    if summary is None:
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    st = summary["stages"]
    if trajectories is None:
        if verbose:
            print("  simulating best + baseline universes for the 3-D viewer (one-off, ~10-40 s)...", flush=True)
        trajectories = _quick_trajectories(run_dir, summary, quick_n, quick_steps)
    u = st.get("universe", {})
    comp = u.get("comparison") or {}
    phen = list(u.get("best", {}).get("emergence", {}).get("scores", {}).keys())
    emergence = {
        "phenomena": [{"key": k, "label": k.replace("_", " "), "info": PHENOMENON_INFO.get(k, ""),
                       "best": u["best"]["emergence"]["scores"].get(k),
                       "baseline": u["baseline"]["emergence"]["scores"].get(k)} for k in phen],
        "best_stats": u.get("best", {}).get("richness_stats", {}),
        "baseline_stats": u.get("baseline", {}).get("richness_stats", {}),
        "welch": comp,
    }
    cad = _cad_dir(run_dir)
    from .tech_evolver import DOMAINS

    tech = []
    for name, t in st.get("technology", {}).items():
        if name == "context":
            continue
        m = st.get("mapping", {}).get(name, {})
        pred = m.get("prediction", {})
        keys = [k for k in pred if k != "mass_kg"][:6]
        stl = f"cad/{name}.stl" if cad is not None and (cad / f"{name}.stl").exists() else None
        tech.append({
            "domain": name, "label": name.replace("_", " "),
            "objectives": t["champion"]["objectives"], "feasible_alien": t["champion"]["feasible"],
            "pareto_size": t.get("pareto_size"),
            "material": m.get("material_name"), "feasible": m.get("feasible"), "fidelity": m.get("fidelity"),
            "reliability": m.get("reliability"), "mass_kg": m.get("mass_kg"), "cost_usd": m.get("cost_usd"),
            "efficiency": [m.get("alien_efficiency"), m.get("real_efficiency")],
            "params": m.get("real_params", {}),
            "units": {ps.name: ps.unit for ps in getattr(DOMAINS.get(name), "params", [])},
            "prediction": [{"metric": k, "mean": pred[k]["mean"], "std": pred[k]["std"]} for k in keys],
            "build": m.get("build_instructions", []), "bom": m.get("bom", []), "notes": m.get("notes", []),
            "stl": stl, "cad": _cad_info(cad, name) if stl else {},
        })
    se = st.get("search", {})
    fig_dir = _fig_dir(run_dir)
    figures = sorted(p.name for p in fig_dir.glob("*.png")) if fig_dir else []
    data = {
        "meta": {"version": summary.get("version"), "seed": summary.get("seed"), "profile": summary.get("profile"),
                 "wall_time_s": summary.get("wall_time_s"), "run_dir": run_dir.as_posix(),
                 "generated": time.strftime("%Y-%m-%d %H:%M")},
        "physics": [{"name": p["name"], "d_eff": p["report"]["effective_dimension"],
                     "apsidal": p["report"]["orbital"]["apsidal_angle"],
                     "bands": p["report"]["orbital"]["n_stable_bands"],
                     "verified": all(v.get("verified") for v in p["report"]["field_equations"].values())}
                    for p in st.get("physics", [])],
        "noether": st.get("noether", []),
        "search": {"history": se.get("history", []), "validation": se.get("validation", []),
                   "evaluations": se.get("evaluations"), "best_description": se.get("best_description", ""),
                   "bandit": se.get("bandit", {})},
        "context": st.get("technology", {}).get("context", {}).get("description", ""),
        "emergence": emergence,
        "universes": trajectories,
        "force_laws": _force_laws(summary),
        "technology": tech,
        "bracket": {k: st.get("generative", {}).get(k) for k in ("shape", "fitness_alien", "fitness_earth")},
        "figures": figures,
    }
    return data


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _clean(o: Any) -> Any:
    """Replace NaN/inf (invalid JSON) by null."""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_clean(v) for v in o]
    return o


def write_dashboard_data(run_dir: Path, **kw: Any) -> Path:
    run_dir = Path(run_dir)
    data = build_dashboard_data(run_dir, **kw)
    path = run_dir / DATA_NAME
    path.write_text(json.dumps(_clean(json.loads(json.dumps(data, default=_json_default))), separators=(",", ":")),
                    encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

#: Explicit types: on Windows ``mimetypes`` may map .js to text/plain from the
#: registry, which browsers refuse for ES modules.
CONTENT_TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
                 ".mjs": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8",
                 ".json": "application/json", ".png": "image/png", ".svg": "image/svg+xml",
                 ".stl": "application/octet-stream", ".wasm": "application/wasm", ".txt": "text/plain; charset=utf-8",
                 ".md": "text/plain; charset=utf-8", ".scad": "text/plain; charset=utf-8"}


class _Handler(SimpleHTTPRequestHandler):
    web_dir: Path = WEB_DIR
    run_dir: Path = Path(".")
    cad_dir: Optional[Path] = None
    fig_dir: Optional[Path] = None

    def log_message(self, fmt: str, *args: Any) -> None:  # keep the console quiet
        pass

    def _resolve(self) -> Optional[Path]:
        path = unquote(urlparse(self.path).path)
        if path in ("", "/"):
            path = "/index.html"
        for prefix, base in (("/run/", self.run_dir), ("/cad/", self.cad_dir), ("/fig/", self.fig_dir)):
            if path.startswith(prefix):
                if base is None:
                    return None
                target = (base / path[len(prefix):]).resolve()
                return target if target.is_relative_to(base.resolve()) else None
        target = (self.web_dir / path.lstrip("/")).resolve()
        return target if target.is_relative_to(self.web_dir.resolve()) else None

    def do_GET(self) -> None:  # noqa: N802
        target = self._resolve()
        if target is None or not target.is_file():
            self.send_error(404, "not found")
            return
        body = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", CONTENT_TYPES.get(target.suffix.lower(), "application/octet-stream"))
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    do_HEAD = do_GET


def make_server(run_dir: Path, port: int = 8765, host: str = "127.0.0.1") -> ThreadingHTTPServer:
    run_dir = Path(run_dir).resolve()
    handler = type("ShivaHandler", (_Handler,), {"run_dir": run_dir, "cad_dir": _cad_dir(run_dir),
                                                  "fig_dir": _fig_dir(run_dir)})
    return ThreadingHTTPServer((host, port), handler)


def serve(run_dir: Optional[Path] = None, port: int = 8765, open_browser: bool = True, rebuild: bool = False) -> None:
    run_dir = Path(run_dir) if run_dir else find_latest_run()
    if run_dir is None or not (run_dir / "summary.json").exists():
        raise SystemExit("No SHIVA-1 run found. Run `python -m shiva_core demo` first (or pass --run <dir>).")
    if rebuild or not (run_dir / DATA_NAME).exists():
        print(f"Preparing dashboard data for {run_dir} ...", flush=True)
        write_dashboard_data(run_dir)
    for p in range(port, port + 20):
        try:
            server = make_server(run_dir, p)
            break
        except OSError:
            continue
    else:
        raise SystemExit(f"No free port between {port} and {port + 19}.")
    url = f"http://localhost:{server.server_address[1]}/"
    print(f"\nSHIVA-1 dashboard for {run_dir}\n  open  {url}\n  (Ctrl+C to stop)\n", flush=True)
    if open_browser:
        threading.Timer(0.8, functools.partial(webbrowser.open, url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("dashboard stopped")
    finally:
        server.server_close()
