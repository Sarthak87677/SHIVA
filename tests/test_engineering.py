import math
import shutil

import numpy as np
import pytest

from shiva_core.cad_generator import (CADGenerator, Mesh, box, cylinder_between, extrude, involute_gear_polygon,
                                      read_stl, revolve, sphere, voxel_surface)
from shiva_core.physics_mutator import PhysicsMutator
from shiva_core.real_world_mapper import MATERIALS, RealWorldMapper
from shiva_core.tech_evolver import (DOMAINS, FlywheelDomain, GearDomain, GenerativeBracketDesigner, NozzleDomain,
                                     PhysicsContext, TechEvolver, TrussDomain, _area_ratio, lewis_form_factor)

EARTH = PhysicsContext.earth()


def test_baseline_genome_maps_to_earth_context():
    ctx = PhysicsContext.from_genome(PhysicsMutator.baseline())
    assert math.isclose(ctx.g, 9.80665, rel_tol=1e-9)
    assert ctx.modulus_scale == ctx.strength_scale == ctx.density_scale == 1.0
    assert ctx.habitable_chemistry


def test_alpha_scaling_follows_press_lightman():
    g = PhysicsMutator.baseline()
    g.em.alpha_ratio = 1.2
    ctx = PhysicsContext.from_genome(g)
    assert math.isclose(ctx.modulus_scale, 1.2 ** 5) and math.isclose(ctx.density_scale, 1.2 ** 3)


@pytest.mark.parametrize("name", list(DOMAINS))
def test_domains_evaluate_finite_and_roundtrip(name):
    d = DOMAINS[name]()
    rng = np.random.default_rng(0)
    for _ in range(25):
        u = rng.random(d.n)
        p = d.decode(u)
        ev = d.evaluate(p, EARTH)
        assert 0 <= ev.efficiency <= 1 and 0 <= ev.stability <= 1
        assert all(np.isfinite(v) for v in ev.constraints.values())
        p2 = d.decode(d.encode(p))
        assert all(math.isclose(p2[k], p[k], rel_tol=1e-9, abs_tol=1e-12) for k in p)


def test_isentropic_area_ratio():
    assert math.isclose(_area_ratio(2.0, 1.4), 1.6875, rel_tol=1e-4)  # textbook value


def test_nozzle_vacuum_isp_exceeds_sea_level():
    d = NozzleDomain()
    p = {"chamber_pressure": 7e6, "expansion_ratio": 40, "throat_radius": 0.03, "chamber_temperature": 3400,
         "molar_mass": 0.022, "half_angle": 15, "wall_thickness": 0.004}
    sl = d.evaluate(p, EARTH)
    vac = d.evaluate(p, PhysicsContext(ambient_pressure=0.0))
    assert vac.metrics["Isp_s"] > sl.metrics["Isp_s"]
    assert 250 < vac.metrics["Isp_s"] < 450


def test_gear_contact_ratio_and_lewis_table():
    d = GearDomain()
    ev = d.evaluate({"module": 0.003, "z1": 20, "face_factor": 10, "pressure_angle": 20, "shift": 0.0}, EARTH)
    assert 1.5 < ev.metrics["contact_ratio"] < 1.8  # 20/60 teeth, 20 deg full depth
    assert math.isclose(lewis_form_factor(20), 0.322)
    assert 0.98 < ev.metrics["mesh_efficiency"] < 1.0


def test_flywheel_thin_rim_limit():
    d = FlywheelDomain()
    m = MATERIALS["al6061_t6"]
    p = {"outer_radius": 0.5, "inner_ratio": 0.999, "thickness": 0.05, "rpm": 6000, "housing_pressure_ratio": 1e-3}
    ev = d.evaluate(p, EARTH, m)
    w = 6000 * 2 * math.pi / 60
    assert math.isclose(ev.metrics["max_stress_MPa"] * 1e6, m.density * w ** 2 * 0.25, rel_tol=0.01)


def test_truss_deflection_scales_inversely_with_modulus():
    d = TrussDomain()
    p = {"h0": 0.6, "h1": 0.4, "h2": 0.2, "area_top": 4e-4, "area_bottom": 4e-4, "area_web": 2e-4}
    m = MATERIALS["al6061_t6"]
    stiff = type(m)(**{**m.__dict__, "E": 2 * m.E, "density": 1e-9})
    soft = type(m)(**{**m.__dict__, "density": 1e-9})
    d1 = d.evaluate(p, EARTH, soft).metrics["tip_deflection_mm"]
    d2 = d.evaluate(p, EARTH, stiff).metrics["tip_deflection_mm"]
    assert math.isclose(d1 / d2, 2.0, rel_tol=1e-6)


def test_tech_evolver_and_mapping_produce_buildable_gear():
    r = TechEvolver("gear_pair", EARTH, seed=0).evolve(generations=10, pop_size=20, generative_samples=8)
    assert r.champion.feasible
    pr = RealWorldMapper(n_monte_carlo=30).map(r.champion)
    assert pr.feasible and pr.material in ("steel4140_qt", "pom")
    assert round(pr.real_params["module"] * 1e3, 6) in (1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0)
    assert pr.build_instructions and pr.bom


def test_generative_bracket_improves():
    out = GenerativeBracketDesigner(seed=0).run(generations=6, pop_size=16)
    assert out["fitness"] > 0.05 and out["metrics"]["volume_fraction"] <= 0.46


# ---------------------------------------------------------------- CAD kernel
def test_primitives_are_watertight_with_correct_volume():
    c = cylinder_between([0, 0, 0], [0, 0, 10], 2.0, 256)
    assert c.is_watertight() and math.isclose(c.signed_volume(), math.pi * 4 * 10, rel_tol=1e-3)
    a = revolve(np.array([[1, 0], [2, 0], [2, 1], [1, 1]]), 512)
    assert a.is_watertight() and math.isclose(a.signed_volume(), math.pi * 3, rel_tol=1e-3)
    s = sphere([0, 0, 0], 1.0, 40, 80)
    assert s.is_watertight() and math.isclose(s.signed_volume(), 4 / 3 * math.pi, rel_tol=0.01)
    b = box([1, 2, 3])
    assert b.is_watertight() and math.isclose(b.signed_volume(), 6.0)
    L = extrude(np.array([[0, 0], [3, 0], [3, 1], [1, 1], [1, 3], [0, 3]]), 2.0)  # non-convex
    assert L.is_watertight() and math.isclose(L.signed_volume(), 10.0)
    v = voxel_surface(np.array([[1, 1, 0], [0, 1, 1]], bool), 1.0, 1.0)
    assert v.is_watertight() and math.isclose(v.signed_volume(), 4.0)


def test_involute_gear_polygon():
    P = involute_gear_polygon(2.0, 24, 20.0, 0.0)
    m = extrude(P, 5.0, star_center=(0, 0))
    assert m.is_watertight()
    r = np.linalg.norm(P, axis=1)
    assert math.isclose(r.max(), 2.0 * 24 / 2 + 2.0, rel_tol=1e-9)  # addendum radius
    area = m.signed_volume() / 5.0
    assert math.pi * (24 - 2.5) ** 2 < area < math.pi * 26 ** 2  # between root and tip circles


def test_cad_generator_all_domains(tmp_path):
    cad = CADGenerator(str(tmp_path), render_openscad=False)
    rng = np.random.default_rng(1)
    for name, cls in DOMAINS.items():
        d = cls()
        p = d.decode(rng.random(d.n) * 0.6 + 0.2)
        info = cad.generate(name, p, MATERIALS[d.reference_material], name=name)
        assert info["components_watertight"], name
        mesh = read_stl(info["stl"])
        assert len(mesh.F) == info["triangles"]
        assert (tmp_path / f"{name}.scad").read_text().startswith("// Generated by SHIVA-1")
        assert "FreeCAD" in (tmp_path / f"{name}.FCMacro").read_text()


@pytest.mark.skipif(shutil.which("openscad") is None, reason="openscad not installed")
def test_openscad_renders_gear(tmp_path):
    cad = CADGenerator(str(tmp_path), render_openscad=True)
    info = cad.generate("gear_pair", {"module": 0.002, "z1": 14, "face_factor": 8, "pressure_angle": 20, "shift": 0.2})
    assert info.get("openscad_stl")
    m = read_stl(info["openscad_stl"])
    assert m.signed_volume() > 0
