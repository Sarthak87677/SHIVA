"""
SHIVA-1 :: CAD Generator
========================

Produces three artefacts for every design:

* **OpenSCAD source** (``.scad``) - fully parametric; every design variable is
  a named top-level variable so the part can be edited in the OpenSCAD
  Customizer.  Involute gear teeth are generated *inside* OpenSCAD from the
  module / tooth count / pressure angle / profile shift.
* **STL** - written by a dependency-free mesh kernel (revolve, extrude with
  ear-clipping, oriented cylinders, UV spheres, voxel surfaces).  Each
  component is a closed, consistently oriented 2-manifold (checked); an
  assembly is a multi-shell STL that slicers union.  When the ``openscad``
  binary is available the ``.scad`` is also rendered to ``*_openscad.stl``
  (true CSG union).
* **FreeCAD macro** (``.FCMacro``) - imports the STL, converts it to a solid,
  and stores all design parameters in a Spreadsheet; executed directly if the
  FreeCAD Python API is importable.

Units: millimetres in all CAD outputs (SI metres in the design dictionaries).
"""

from __future__ import annotations

import json
import math
import shutil
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["Mesh", "CADGenerator", "involute_gear_polygon", "revolve", "extrude", "cylinder_between", "sphere",
           "voxel_surface", "box", "triangulate_polygon"]


# ---------------------------------------------------------------------------
# Mesh kernel
# ---------------------------------------------------------------------------

@dataclass
class Mesh:
    V: np.ndarray  # (n, 3)
    F: np.ndarray  # (m, 3) int

    @staticmethod
    def merge(meshes: Sequence["Mesh"]) -> "Mesh":
        Vs, Fs, off = [], [], 0
        for m in meshes:
            Vs.append(m.V)
            Fs.append(m.F + off)
            off += len(m.V)
        return Mesh(np.concatenate(Vs), np.concatenate(Fs))

    def transformed(self, R: Optional[np.ndarray] = None, t: Sequence[float] = (0, 0, 0), s: float = 1.0) -> "Mesh":
        V = self.V * s
        if R is not None:
            V = V @ np.asarray(R).T
        return Mesh(V + np.asarray(t, float), self.F.copy())

    def signed_volume(self) -> float:
        a, b, c = self.V[self.F[:, 0]], self.V[self.F[:, 1]], self.V[self.F[:, 2]]
        return float(np.sum(np.einsum("ij,ij->i", a, np.cross(b, c))) / 6.0)

    def oriented(self) -> "Mesh":
        """Flip faces if the (closed) mesh is inside-out."""
        return Mesh(self.V, self.F[:, ::-1].copy()) if self.signed_volume() < 0 else self

    def area(self) -> float:
        a, b, c = self.V[self.F[:, 0]], self.V[self.F[:, 1]], self.V[self.F[:, 2]]
        return float(0.5 * np.sum(np.linalg.norm(np.cross(b - a, c - a), axis=1)))

    def is_watertight(self) -> bool:
        """Every directed edge has exactly one opposite twin (closed, oriented 2-manifold)."""
        e = np.concatenate([self.F[:, [0, 1]], self.F[:, [1, 2]], self.F[:, [2, 0]]])
        fwd = {tuple(x) for x in e.tolist()}
        if len(fwd) != len(e):
            return False
        return all((b, a) in fwd for a, b in fwd)

    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.V.min(axis=0), self.V.max(axis=0)

    def write_stl(self, path: str, name: str = "shiva") -> None:
        a, b, c = self.V[self.F[:, 0]], self.V[self.F[:, 1]], self.V[self.F[:, 2]]
        n = np.cross(b - a, c - a)
        nn = np.linalg.norm(n, axis=1, keepdims=True)
        n = np.where(nn > 0, n / np.maximum(nn, 1e-300), 0.0)
        data = np.zeros(len(self.F), dtype=np.dtype([("n", "<f4", 3), ("a", "<f4", 3), ("b", "<f4", 3),
                                                      ("c", "<f4", 3), ("attr", "<u2")]))
        data["n"], data["a"], data["b"], data["c"] = n, a, b, c
        with open(path, "wb") as fh:
            fh.write(struct.pack("<80sI", name.encode()[:80].ljust(80, b" "), len(self.F)))
            fh.write(data.tobytes())


def read_stl(path: str) -> Mesh:
    """Read a binary or ASCII STL (vertices de-duplicated)."""
    raw = Path(path).read_bytes()
    tris: np.ndarray
    if raw[:5].lower() == b"solid" and b"facet" in raw[:400]:
        nums = [list(map(float, line.split()[1:4])) for line in raw.decode(errors="ignore").splitlines()
                if line.strip().startswith("vertex")]
        tris = np.array(nums, float).reshape(-1, 3, 3)
    else:
        n = struct.unpack("<I", raw[80:84])[0]
        rec = np.frombuffer(raw[84:84 + 50 * n], dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)),
                                                                ("attr", "<u2")]))
        tris = rec["v"].astype(float)
    V, inv = np.unique(np.round(tris.reshape(-1, 3), 5), axis=0, return_inverse=True)
    return Mesh(V, inv.reshape(-1, 3))


def _cross2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _clean_polygon(P: np.ndarray) -> np.ndarray:
    P = np.asarray(P, float)
    keep = np.linalg.norm(P - np.roll(P, -1, axis=0), axis=1) > 1e-12
    P = P[keep]
    area = 0.5 * np.sum(_cross2(P, np.roll(P, -1, axis=0)))
    return P if area > 0 else P[::-1]


def triangulate_polygon(P: np.ndarray) -> List[Tuple[int, int, int]]:
    """Ear clipping for a simple CCW polygon (vectorised point-in-triangle)."""
    n = len(P)
    idx = list(range(n))
    tris: List[Tuple[int, int, int]] = []
    guard = 0
    k = 0
    while len(idx) > 3 and guard < 10 * n * n:
        m = len(idx)
        found = False
        for step in range(m):
            ii = (k + step) % m
            i0, i1, i2 = idx[ii - 1], idx[ii], idx[(ii + 1) % m]
            a, b, c = P[i0], P[i1], P[i2]
            if _cross2(b - a, c - b) <= 1e-14:
                continue
            others = [j for j in idx if j not in (i0, i1, i2)]
            if others:
                Q = P[others]
                d1 = _cross2(b - a, Q - a)
                d2 = _cross2(c - b, Q - b)
                d3 = _cross2(a - c, Q - c)
                if np.any((d1 >= -1e-14) & (d2 >= -1e-14) & (d3 >= -1e-14)):
                    continue
            tris.append((i0, i1, i2))
            idx.pop(ii)
            k = ii % max(len(idx), 1)
            found = True
            break
        guard += 1
        if not found:  # numerically degenerate remainder: fan it
            for j in range(1, len(idx) - 1):
                tris.append((idx[0], idx[j], idx[j + 1]))
            return tris
    if len(idx) == 3:
        tris.append((idx[0], idx[1], idx[2]))
    return tris


def extrude(poly: np.ndarray, height: float, z0: float = 0.0, star_center: Optional[Sequence[float]] = None) -> Mesh:
    """Linear extrusion of a simple polygon.  ``star_center`` uses a fan from
    that point (valid for star-shaped outlines such as gears; much faster)."""
    P = _clean_polygon(poly)
    n = len(P)
    bot = np.column_stack([P, np.full(n, z0)])
    top = np.column_stack([P, np.full(n, z0 + height)])
    V = [bot, top]
    F: List[Tuple[int, int, int]] = []
    if star_center is not None:
        c = np.asarray(star_center, float)
        V.append(np.array([[c[0], c[1], z0], [c[0], c[1], z0 + height]]))
        cb, ct = 2 * n, 2 * n + 1
        for i in range(n):
            j = (i + 1) % n
            F.append((cb, j, i))
            F.append((ct, n + i, n + j))
    else:
        for a, b, c in triangulate_polygon(P):
            F.append((c, b, a))
            F.append((n + a, n + b, n + c))
    for i in range(n):
        j = (i + 1) % n
        F.append((i, j, n + j))
        F.append((i, n + j, n + i))
    return Mesh(np.concatenate(V), np.array(F)).oriented()


def revolve(profile: np.ndarray, segments: int = 72) -> Mesh:
    """Revolve a closed (r, z) polygon (r > 0) about the z axis."""
    P = _clean_polygon(profile)
    n = len(P)
    th = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    V = np.zeros((segments * n, 3))
    for j, t in enumerate(th):
        V[j * n:(j + 1) * n] = np.column_stack([P[:, 0] * math.cos(t), P[:, 0] * math.sin(t), P[:, 1]])
    F = []
    for j in range(segments):
        jn = (j + 1) % segments
        for i in range(n):
            i2 = (i + 1) % n
            a, b, c, d = j * n + i, j * n + i2, jn * n + i2, jn * n + i
            F.append((a, b, c))
            F.append((a, c, d))
    return Mesh(V, np.array(F)).oriented()


def _frame(axis: np.ndarray) -> np.ndarray:
    z = axis / np.linalg.norm(axis)
    tmp = np.array([1.0, 0, 0]) if abs(z[0]) < 0.9 else np.array([0, 1.0, 0])
    x = np.cross(tmp, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def cylinder_between(p0: Sequence[float], p1: Sequence[float], r: float, segments: int = 20) -> Mesh:
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    L = float(np.linalg.norm(p1 - p0))
    th = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    circ = np.column_stack([r * np.cos(th), r * np.sin(th)])
    m = extrude(circ, L, 0.0, star_center=(0, 0))
    return m.transformed(R=_frame(p1 - p0), t=p0)


def sphere(center: Sequence[float], r: float, n_lat: int = 10, n_lon: int = 16) -> Mesh:
    V = [[0, 0, r]]
    for i in range(1, n_lat):
        ph = math.pi * i / n_lat
        for j in range(n_lon):
            t = 2 * math.pi * j / n_lon
            V.append([r * math.sin(ph) * math.cos(t), r * math.sin(ph) * math.sin(t), r * math.cos(ph)])
    V.append([0, 0, -r])
    V = np.array(V) + np.asarray(center, float)
    F = []
    for j in range(n_lon):
        F.append((0, 1 + j, 1 + (j + 1) % n_lon))
    for i in range(n_lat - 2):
        for j in range(n_lon):
            a = 1 + i * n_lon + j
            b = 1 + i * n_lon + (j + 1) % n_lon
            c = a + n_lon
            d = b + n_lon
            F.append((a, c, d))
            F.append((a, d, b))
    last = len(V) - 1
    base = 1 + (n_lat - 2) * n_lon
    for j in range(n_lon):
        F.append((last, base + (j + 1) % n_lon, base + j))
    return Mesh(V, np.array(F)).oriented()


def box(size: Sequence[float], center: Sequence[float] = (0, 0, 0)) -> Mesh:
    sx, sy, sz = (0.5 * np.asarray(size, float))
    poly = np.array([[-sx, -sy], [sx, -sy], [sx, sy], [-sx, sy]])
    m = extrude(poly, 2 * sz, -sz)
    return m.transformed(t=center)


def voxel_surface(grid: np.ndarray, pitch: float, height: float) -> Mesh:
    """Watertight surface of a 2-D pixel mask extruded to ``height`` (only exposed faces)."""
    ny, nx = grid.shape
    verts: Dict[Tuple[int, int, int], int] = {}
    F: List[Tuple[int, int, int]] = []

    def vid(i: int, j: int, k: int) -> int:
        key = (i, j, k)
        if key not in verts:
            verts[key] = len(verts)
        return verts[key]

    def quad(a, b, c, d):
        F.append((a, b, c))
        F.append((a, c, d))

    filled = lambda y, x: 0 <= y < ny and 0 <= x < nx and bool(grid[y, x])
    for y in range(ny):
        for x in range(nx):
            if not grid[y, x]:
                continue
            quad(vid(x, y, 0), vid(x, y + 1, 0), vid(x + 1, y + 1, 0), vid(x + 1, y, 0))  # bottom (-z)
            quad(vid(x, y, 1), vid(x + 1, y, 1), vid(x + 1, y + 1, 1), vid(x, y + 1, 1))  # top (+z)
            if not filled(y, x - 1):
                quad(vid(x, y, 0), vid(x, y, 1), vid(x, y + 1, 1), vid(x, y + 1, 0))
            if not filled(y, x + 1):
                quad(vid(x + 1, y, 0), vid(x + 1, y + 1, 0), vid(x + 1, y + 1, 1), vid(x + 1, y, 1))
            if not filled(y - 1, x):
                quad(vid(x, y, 0), vid(x + 1, y, 0), vid(x + 1, y, 1), vid(x, y, 1))
            if not filled(y + 1, x):
                quad(vid(x, y + 1, 0), vid(x, y + 1, 1), vid(x + 1, y + 1, 1), vid(x + 1, y + 1, 0))
    V = np.zeros((len(verts), 3))
    for (i, j, k), n in verts.items():
        V[n] = [i * pitch, j * pitch, k * height]
    return Mesh(V, np.array(F))


# ---------------------------------------------------------------------------
# Geometry generators
# ---------------------------------------------------------------------------

def _inv(a: float) -> float:
    return math.tan(a) - a


def involute_gear_polygon(module: float, teeth: int, pressure_angle_deg: float = 20.0, shift: float = 0.0,
                          n_flank: int = 10) -> np.ndarray:
    """Outline (CCW) of an external involute spur gear, centred at the origin."""
    phi = math.radians(pressure_angle_deg)
    rp = module * teeth / 2
    rb = rp * math.cos(phi)
    ra = rp + module * (1 + shift)
    rf = rp - module * (1.25 - shift)
    psi = (math.pi / 2 + 2 * shift * math.tan(phi)) / teeth
    r0 = max(rb, rf)
    rs = np.linspace(r0, ra, n_flank)
    theta = psi + _inv(phi) - np.array([_inv(math.acos(min(1.0, rb / r))) for r in rs])
    pts = []
    for k in range(teeth):
        c = 2 * math.pi * k / teeth
        tooth = [(rf, c - theta[0])] + [(r, c - t) for r, t in zip(rs, theta)] + \
                [(r, c + t) for r, t in zip(rs[::-1], theta[::-1])] + [(rf, c + theta[0])]
        pts.extend(tooth)
    P = np.array([[r * math.cos(a), r * math.sin(a)] for r, a in pts])
    return P


def nozzle_profile(p: Dict[str, float], segments: int = 24) -> Tuple[np.ndarray, np.ndarray]:
    """Inner contour (r, z) in mm: chamber -> 45 deg convergent -> throat arc -> conical divergent."""
    rt = p["throat_radius"] * 1e3
    rc = 2 * rt
    re = rt * math.sqrt(p["expansion_ratio"])
    a = math.radians(p["half_angle"])
    Lc = 250.0
    zc0, zc1 = -Lc - (rc - rt), -(rc - rt)
    Rarc = 1.5 * rt
    pts = [(rc, zc0), (rc, zc1)]
    pts.append((rt + Rarc * (1 - math.cos(math.radians(45))), -Rarc * math.sin(math.radians(45))))
    for t in np.linspace(-math.radians(45), a, segments):
        pts.append((rt + Rarc * (1 - math.cos(t)), Rarc * math.sin(t)))
    r_end, z_end = pts[-1]
    L_div = (re - r_end) / math.tan(a)
    pts.append((re, z_end + L_div))
    inner = np.array(pts)
    return inner, np.array([rc, re])


def impeller_blade(p: Dict[str, float], n: int = 24, thickness: float = 3.0) -> np.ndarray:
    """2-D blade outline (mm): centreline with blade angle varying linearly from
    beta1 to beta2 (from tangential), d(theta)/dr = 1/(r tan beta)."""
    r1, r2 = p["r1"] * 1e3, p["r2"] * 1e3
    b1, b2 = math.radians(p["beta1"]), math.radians(p["beta2"])
    rs = np.linspace(r1, r2, n)
    beta = b1 + (b2 - b1) * (rs - r1) / (r2 - r1)
    dth = 1.0 / (rs * np.tan(beta))
    th = np.concatenate([[0.0], np.cumsum(0.5 * (dth[1:] + dth[:-1]) * np.diff(rs))])
    th = -th  # backward-curved for counter-clockwise rotation
    c = np.column_stack([rs * np.cos(th), rs * np.sin(th)])
    t = np.gradient(c, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    nrm = np.column_stack([-t[:, 1], t[:, 0]])
    left = c + 0.5 * thickness * nrm
    right = c - 0.5 * thickness * nrm
    return np.concatenate([left, right[::-1]])


# ---------------------------------------------------------------------------
# CAD generator
# ---------------------------------------------------------------------------

_SCAD_HEADER = "// Generated by SHIVA-1 cad_generator - parametric OpenSCAD model\n// Units: millimetres.\n"


def _scad_vars(d: Dict[str, Any]) -> str:
    return "".join(f"{k} = {v:.6g};\n" if isinstance(v, (int, float)) else f"{k} = {v};\n" for k, v in d.items())


class CADGenerator:
    """Generate SCAD / STL / FreeCAD artefacts for SHIVA design domains."""

    def __init__(self, out_dir: str = "shiva_output/cad", render_openscad: Optional[bool] = None,
                 openscad_timeout: int = 240):
        self.out_dir = Path(out_dir)
        self.openscad = shutil.which("openscad")
        self.render = (self.openscad is not None) if render_openscad is None else (render_openscad and self.openscad is not None)
        self.timeout = openscad_timeout
        try:
            import FreeCAD  # type: ignore  # noqa: F401
            self.freecad = True
        except Exception:
            self.freecad = False

    # -- public -------------------------------------------------------------------
    def generate(self, domain: str, params: Dict[str, float], material: Any = None, name: Optional[str] = None,
                 out_dir: Optional[str] = None) -> Dict[str, Any]:
        out = Path(out_dir) if out_dir else self.out_dir
        out.mkdir(parents=True, exist_ok=True)
        stem = name or domain
        builder = getattr(self, f"_build_{domain}")
        scad, parts = builder(params)
        scad_path = out / f"{stem}.scad"
        scad_path.write_text(_SCAD_HEADER + f"// domain: {domain}\n" +
                             (f"// material: {getattr(material, 'name', material)}\n" if material else "") + scad)
        mesh = Mesh.merge(parts)
        stl_path = out / f"{stem}.stl"
        mesh.write_stl(str(stl_path), name=stem)
        lo, hi = mesh.bounds()
        info: Dict[str, Any] = {
            "scad": str(scad_path), "stl": str(stl_path), "components": len(parts),
            "triangles": int(len(mesh.F)), "components_watertight": all(p.is_watertight() for p in parts),
            "bbox_mm": (hi - lo).round(3).tolist(),
            "volume_mm3_sum_of_shells": float(sum(p.signed_volume() for p in parts)),
        }
        macro = self._freecad_macro(stem, stl_path, params, material)
        macro_path = out / f"{stem}.FCMacro"
        macro_path.write_text(macro)
        info["freecad_macro"] = str(macro_path)
        if self.render:
            info["openscad_stl"] = self._render_scad(scad_path)
        if self.freecad:
            info["freecad_fcstd"] = self._run_freecad(stem, stl_path, params, out)
        (out / f"{stem}.json").write_text(json.dumps({"domain": domain, "params": params, "cad": info,
                                                       "material": getattr(material, "key", material)}, indent=2))
        return info

    def render_scad(self, scad_path: str) -> Optional[str]:
        return self._render_scad(Path(scad_path))

    def _render_scad(self, scad_path: Path) -> Optional[str]:
        target = scad_path.with_name(scad_path.stem + "_openscad.stl")
        try:
            proc = subprocess.run([self.openscad, "-o", str(target), str(scad_path)], capture_output=True, text=True,
                                  timeout=self.timeout)
            if proc.returncode == 0 and target.exists():
                return str(target)
            return None
        except (subprocess.TimeoutExpired, OSError):
            return None

    def _freecad_macro(self, stem: str, stl: Path, params: Dict[str, float], material: Any) -> str:
        rows = "\n".join(f"sheet.set('A{i + 2}', {k!r}); sheet.set('B{i + 2}', {repr(str(v))})"
                         for i, (k, v) in enumerate(params.items()))
        return f'''# FreeCAD macro generated by SHIVA-1 (Macro > Macros... > Execute, or freecadcmd {stem}.FCMacro)
import os
import FreeCAD as App
import Mesh
import Part

here = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
doc = App.newDocument("SHIVA_{stem}")
mesh = Mesh.Mesh(os.path.join(here, "{stl.name}"))
shape = Part.Shape()
shape.makeShapeFromMesh(mesh.Topology, 0.05)
try:
    solid = Part.makeSolid(shape).removeSplitter()
except Exception:
    solid = shape
part = doc.addObject("Part::Feature", "{stem}")
part.Shape = solid
sheet = doc.addObject("Spreadsheet::Sheet", "Parameters")
sheet.set("A1", "parameter"); sheet.set("B1", "value (SI)")
{rows}
sheet.set("D1", "material"); sheet.set("E1", {repr(str(getattr(material, "name", material)))})
doc.recompute()
doc.saveAs(os.path.join(here, "{stem}.FCStd"))
'''

    def _run_freecad(self, stem: str, stl: Path, params: Dict[str, float], out: Path) -> Optional[str]:  # pragma: no cover
        try:
            import FreeCAD as App  # type: ignore
            import Mesh as FMesh  # type: ignore
            import Part  # type: ignore

            doc = App.newDocument(f"SHIVA_{stem}")
            m = FMesh.Mesh(str(stl))
            sh = Part.Shape()
            sh.makeShapeFromMesh(m.Topology, 0.05)
            obj = doc.addObject("Part::Feature", stem)
            obj.Shape = Part.makeSolid(sh)
            doc.recompute()
            path = out / f"{stem}.FCStd"
            doc.saveAs(str(path))
            return str(path)
        except Exception:
            return None

    # -- domain builders: return (scad_source, [component meshes]) ----------------------------
    def _build_propulsion_nozzle(self, p):
        inner, (rc, re) = nozzle_profile(p)
        t = p["wall_thickness"] * 1e3
        tang = np.gradient(inner, axis=0)
        tang /= np.linalg.norm(tang, axis=1, keepdims=True)
        normal = np.column_stack([tang[:, 1], -tang[:, 0]])
        normal[normal[:, 0] < 0] *= -1  # outward (+r)
        outer = inner + t * normal
        prof = np.concatenate([inner, outer[::-1]])
        mesh = revolve(prof, 72)
        pts = ",".join(f"[{r:.4f},{z:.4f}]" for r, z in prof)
        scad = _scad_vars({"throat_radius": p["throat_radius"] * 1e3, "expansion_ratio": p["expansion_ratio"],
                           "half_angle": p["half_angle"], "wall": t, "chamber_pressure_MPa": p["chamber_pressure"] / 1e6})
        scad += ("// wall contour (r, z) generated from the parameters above; chamber + 45 deg convergent +"
                 " throat arc (1.5 r_t) + conical divergent\n")
        scad += f"rotate_extrude($fn = 96) polygon(points = [{pts}]);\n"
        return scad, [mesh]

    def _build_structure_truss(self, p):
        from .tech_evolver import TrussDomain

        d = TrussDomain()
        xy = d.nodes(p) * 1e3
        rad = {g: math.sqrt(p[f"area_{g}"] / math.pi) * 1e3 for g in ("top", "bottom", "web")}
        parts = []
        for i, j, g in d.MEMBERS:
            a = [xy[i, 0], 0.0, xy[i, 1]]
            b = [xy[j, 0], 0.0, xy[j, 1]]
            parts.append(cylinder_between(a, b, rad[g], 20))
        rj = 1.6 * max(rad.values())
        for k in range(len(xy)):
            parts.append(sphere([xy[k, 0], 0.0, xy[k, 1]], rj, 8, 14))
        nodes = ",".join(f"[{x:.3f},{y:.3f}]" for x, y in xy)
        members = ",".join(f"[{i},{j},{['top', 'bottom', 'web'].index(g)}]" for i, j, g in d.MEMBERS)
        scad = _scad_vars({"r_top": rad["top"], "r_bottom": rad["bottom"], "r_web": rad["web"], "joint_r": rj})
        scad += f"nodes = [{nodes}];\nmembers = [{members}];\nradii = [r_top, r_bottom, r_web];\n"
        scad += """module bar(a, b, r) { hull() { translate([a[0], 0, a[1]]) sphere(r = r, $fn = 24);
                                  translate([b[0], 0, b[1]]) sphere(r = r, $fn = 24); } }
union() {
  for (m = members) bar(nodes[m[0]], nodes[m[1]], radii[m[2]]);
  for (n = nodes) translate([n[0], 0, n[1]]) sphere(r = joint_r, $fn = 32);
}
"""
        return scad, parts

    def _build_energy_flywheel(self, p):
        R = p["outer_radius"] * 1e3
        a = p["inner_ratio"] * R
        h = p["thickness"] * 1e3
        hub_r = max(0.6 * a, 15.0)
        rotor = revolve(np.array([[a, -h / 2], [R, -h / 2], [R, h / 2], [a, h / 2]]), 128)
        web = revolve(np.array([[hub_r * 0.99, -0.15 * h], [a * 1.001, -0.15 * h], [a * 1.001, 0.15 * h],
                                [hub_r * 0.99, 0.15 * h]]), 96) if a > hub_r * 1.05 else None
        hub = revolve(np.array([[10.0, -0.6 * h], [hub_r, -0.6 * h], [hub_r, 0.6 * h], [10.0, 0.6 * h]]), 64)
        parts = [rotor, hub] + ([web] if web is not None else [])
        scad = _scad_vars({"R": R, "bore": a, "thickness": h, "hub_r": hub_r, "shaft_r": 10.0})
        scad += """union() {
  difference() { cylinder(r = R, h = thickness, center = true, $fn = 180); cylinder(r = bore, h = thickness + 2, center = true, $fn = 120); }
  if (bore > hub_r) difference() { cylinder(r = bore + 0.5, h = 0.3 * thickness, center = true, $fn = 120); cylinder(r = hub_r - 0.5, h = thickness, center = true, $fn = 96); }
  difference() { cylinder(r = hub_r, h = 1.2 * thickness, center = true, $fn = 96); cylinder(r = shaft_r, h = 2 * thickness, center = true, $fn = 48); }
}
"""
        return scad, parts

    def _build_drone_frame(self, p):
        n = int(p["n_arms"])
        L = p["arm_length"] * 1e3
        D = p["arm_od"] * 1e3
        Dp = p["prop_diameter"] * 1e3
        plate_r = max(35.0, 0.3 * L)
        plate = extrude(np.array([[plate_r * math.cos(2 * math.pi * k / n + math.pi / n),
                                   plate_r * math.sin(2 * math.pi * k / n + math.pi / n)] for k in range(n)]),
                        3.0, -1.5, star_center=(0, 0))
        parts = [plate]
        for k in range(n):
            ang = 2 * math.pi * k / n
            tip = np.array([L * math.cos(ang), L * math.sin(ang), 0.0])
            parts.append(cylinder_between(0.25 * tip, tip, D / 2, 24))
            parts.append(cylinder_between(tip + [0, 0, D / 2], tip + [0, 0, D / 2 + 18.0], 0.07 * Dp + 6.0, 32))
        scad = _scad_vars({"n_arms": n, "arm_length": L, "arm_od": D, "arm_wall": p["wall_ratio"] * D,
                           "prop_diameter": Dp, "plate_r": plate_r, "plate_t": 3.0, "show_props": "false"})
        scad += """union() {
  rotate([0, 0, 180 / n_arms]) cylinder(r = plate_r, h = plate_t, center = true, $fn = n_arms);
  for (k = [0 : n_arms - 1]) rotate([0, 0, 360 * k / n_arms]) {
    translate([0.25 * arm_length, 0, 0]) rotate([0, 90, 0])
      difference() { cylinder(d = arm_od, h = 0.75 * arm_length, $fn = 32); translate([0, 0, -1]) cylinder(d = arm_od - 2 * arm_wall, h = arm_length, $fn = 32); }
    translate([arm_length, 0, arm_od / 2]) cylinder(r = 0.07 * prop_diameter + 6, h = 18, $fn = 48);
    if (show_props) %translate([arm_length, 0, arm_od / 2 + 20]) cylinder(d = prop_diameter, h = 1, $fn = 96);
  }
}
"""
        return scad, parts

    def _build_pump_impeller(self, p):
        r1, r2, b2 = p["r1"] * 1e3, p["r2"] * 1e3, p["b2"] * 1e3
        Z = int(p["blades"])
        back_t = 4.0
        hub_r = 0.35 * r1
        backplate = revolve(np.array([[hub_r * 0.5, -back_t], [r2, -back_t], [r2, 0.0], [hub_r * 0.5, 0.0]]), 128)
        hub = revolve(np.array([[hub_r * 0.5, 0.0], [hub_r, 0.0], [hub_r * 0.8, 1.5 * b2], [hub_r * 0.5, 1.5 * b2]]), 64)
        blade = impeller_blade(p)
        parts = [backplate, hub]
        for k in range(Z):
            c, s = math.cos(2 * math.pi * k / Z), math.sin(2 * math.pi * k / Z)
            Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            parts.append(extrude(blade, b2, 0.0).transformed(R=Rz))
        pts = ",".join(f"[{x:.4f},{y:.4f}]" for x, y in blade)
        scad = _scad_vars({"r1": r1, "r2": r2, "b2": b2, "blades": Z, "beta1": p["beta1"], "beta2": p["beta2"],
                           "backplate_t": back_t, "hub_r": hub_r, "show_shroud": "false"})
        scad += "// blade outline from d(theta)/dr = 1/(r tan(beta)), beta varying linearly beta1 -> beta2\n"
        scad += f"blade = [{pts}];\n"
        scad += """union() {
  translate([0, 0, -backplate_t]) difference() { cylinder(r = r2, h = backplate_t, $fn = 180); translate([0, 0, -1]) cylinder(r = 0.5 * hub_r, h = backplate_t + 2, $fn = 48); }
  difference() { cylinder(r1 = hub_r, r2 = 0.8 * hub_r, h = 1.5 * b2, $fn = 64); translate([0, 0, -1]) cylinder(r = 0.5 * hub_r, h = 3 * b2, $fn = 48); }
  for (k = [0 : blades - 1]) rotate([0, 0, 360 * k / blades]) linear_extrude(height = b2) polygon(blade);
  if (show_shroud) %translate([0, 0, b2]) difference() { cylinder(r = r2, h = 3, $fn = 180); translate([0, 0, -1]) cylinder(r = r1, h = 5, $fn = 96); }
}
"""
        return scad, parts

    def _build_gear_pair(self, p):
        from .tech_evolver import GearDomain

        mod = p["module"] * 1e3
        z1 = int(p["z1"])
        z2 = int(round(GearDomain.RATIO * z1))
        pa, x = p["pressure_angle"], p["shift"]
        b = p["face_factor"] * mod
        a = mod * (z1 + z2) / 2
        g1 = involute_gear_polygon(mod, z1, pa, x)
        g2 = involute_gear_polygon(mod, z2, pa, -x)
        rot2 = math.pi / z2 if z2 % 2 == 0 else 0.0  # mesh phasing
        c, s = math.cos(rot2), math.sin(rot2)
        g2 = g2 @ np.array([[c, s], [-s, c]])
        m1 = extrude(g1, b, 0.0, star_center=(0, 0))
        m2 = extrude(g2, b, 0.0, star_center=(0, 0)).transformed(t=(a, 0, 0))
        scad = _scad_vars({"module_mm": mod, "z1": z1, "z2": z2, "pressure_angle": pa, "shift": x, "face_width": b,
                           "bore1": max(0.3 * mod * z1 / 2, 5.0), "bore2": max(0.3 * mod * z2 / 2, 5.0)})
        scad += """// involute geometry evaluated in OpenSCAD (degrees)
function inv_deg(a) = (tan(a) - a * PI / 180) * 180 / PI;
function polar(r, t) = [r * cos(t), r * sin(t)];
module involute_gear(m, z, pa, x, width, bore, steps = 12) {
  rp = m * z / 2; rb = rp * cos(pa); ra = rp + m * (1 + x); rf = rp - m * (1.25 - x);
  half_p = (90 + 2 * x * tan(pa) * 180 / PI) / z;
  r0 = max(rb, rf);
  rs = [for (i = [0 : steps]) r0 + (ra - r0) * i / steps];
  function theta(r) = half_p + inv_deg(pa) - inv_deg(acos(min(1, rb / r)));
  pts = [for (k = [0 : z - 1]) let(c = k * 360 / z) each concat(
           [polar(rf, c - theta(r0))],
           [for (r = rs) polar(r, c - theta(r))],
           [for (i = [steps : -1 : 0]) polar(rs[i], c + theta(rs[i]))],
           [polar(rf, c + theta(r0))])];
  difference() {
    linear_extrude(height = width) polygon(pts);
    translate([0, 0, -1]) cylinder(r = bore, h = width + 2, $fn = 48);
  }
}
involute_gear(module_mm, z1, pressure_angle, shift, face_width, bore1);
translate([module_mm * (z1 + z2) / 2, 0, 0]) rotate([0, 0, (z2 % 2 == 0) ? 180 / z2 : 0])
  involute_gear(module_mm, z2, pressure_angle, -shift, face_width, bore2);
"""
        return scad, [m1, m2]

    def _build_cosmic_habitat(self, p):
        k = 1e-3 * 1e3  # 1:1000 model, metres -> mm
        R, W = p["radius"] * k, p["width"] * k
        h = max(min(20.0, 0.5 * p["width"]) * k, 1.0)
        t = max(p["thickness"] * 1e3 * 1e-3, 0.8)  # hull thickness at model scale, printable minimum 0.8 mm
        ring = revolve(np.array([[R - h, -W / 2], [R, -W / 2], [R, W / 2], [R - h, W / 2]]), 160)
        hub = revolve(np.array([[1.0, -W / 4], [0.08 * R + 2, -W / 4], [0.08 * R + 2, W / 4], [1.0, W / 4]]), 48)
        parts = [ring, hub]
        for j in range(6):
            ang = 2 * math.pi * j / 6
            parts.append(cylinder_between([0.08 * R * math.cos(ang), 0.08 * R * math.sin(ang), 0],
                                          [(R - h) * math.cos(ang), (R - h) * math.sin(ang), 0], max(0.01 * R, 0.8), 12))
        scad = _scad_vars({"scale_note": '"1:1000 model"', "R": R, "W": W, "h": h, "hull_t": t, "spokes": 6})
        scad += """union() {
  rotate_extrude($fn = 240) translate([R - h, -W / 2]) difference() { square([h, W]); translate([hull_t, hull_t]) square([h - 2 * hull_t, W - 2 * hull_t]); }
  cylinder(r = 0.08 * R + 2, h = W / 2, center = true, $fn = 64);
  for (j = [0 : spokes - 1]) rotate([0, 0, 360 * j / spokes]) translate([0.08 * R, 0, 0]) rotate([0, 90, 0]) cylinder(r = max(0.01 * R, 0.8), h = R - h - 0.08 * R, $fn = 16);
}
"""
        return scad, parts

    def _build_cosmic_lightsail(self, p):
        k = 10.0  # 1:100 model: metres -> mm * 1e3 / 100
        L = p["side_length"] * k
        film = box([L, L, 0.4], [0, 0, 0])
        boom_r = max(0.004 * L, 0.6)
        d = L / math.sqrt(2)
        parts = [film]
        for ang in (math.pi / 4, 3 * math.pi / 4):
            parts.append(cylinder_between([-d * math.cos(ang), -d * math.sin(ang), 1.0],
                                          [d * math.cos(ang), d * math.sin(ang), 1.0], boom_r, 12))
        parts.append(box([0.06 * L, 0.06 * L, 0.04 * L], [0, 0, 0.02 * L + 1.0]))
        scad = _scad_vars({"scale_note": '"1:100 model"', "side": L, "film_t": 0.4, "boom_r": boom_r})
        scad += """union() {
  cube([side, side, film_t], center = true);
  for (a = [45, 135]) rotate([0, 0, a]) translate([-side / sqrt(2), 0, 1]) rotate([0, 90, 0]) cylinder(r = boom_r, h = 2 * side / sqrt(2), $fn = 16);
  translate([0, 0, 0.02 * side + 1]) cube([0.06 * side, 0.06 * side, 0.04 * side], center = true);
}
"""
        return scad, parts

    def _build_generative_bracket(self, p):
        grid = np.asarray(p["shape"], bool)
        pitch = p["pixel_m"] * 1e3
        t = p["thickness_m"] * 1e3
        mesh = voxel_surface(grid[::-1], pitch, t)  # row 0 at the top in image space
        rects = []
        g = grid[::-1]
        for y in range(g.shape[0]):
            x = 0
            while x < g.shape[1]:
                if g[y, x]:
                    x0 = x
                    while x < g.shape[1] and g[y, x]:
                        x += 1
                    rects.append((x0, y, x - x0))
                else:
                    x += 1
        cubes = "\n".join(f"  translate([{x0 * pitch:.3f}, {y * pitch:.3f}, 0]) cube([{w * pitch:.3f}, {pitch:.3f}, thickness]);"
                          for x0, y, w in rects)
        scad = _scad_vars({"pixel": pitch, "thickness": t}) + "// CPPN-NEAT generated bracket (run-length merged pixels)\nunion() {\n" + cubes + "\n}\n"
        return scad, [mesh]
