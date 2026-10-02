import math

import numpy as np
import pytest

from shiva_core.physics_mutator import PhysicsMutator
from shiva_core.pipeline import noether_battery
from shiva_core.universe_simulator import (Kinetics, ReactionDiffusion2D, RigidBodyConfig, RigidBodyEnsemble,
                                           SimulationConfig, UniverseSimulator, VorticityField2D, evaluate_simulation)


def small(**kw):
    base = dict(n_particles=32, steps=400, initial_condition="virialized", seed=0, lyapunov=False)
    base.update(kw)
    return SimulationConfig(**base)


def test_leapfrog_conserves_momentum_and_angular_momentum_exactly():
    res = UniverseSimulator(PhysicsMutator.baseline(), small()).run()
    ev = evaluate_simulation(res)
    assert ev.conservation["momentum"]["relative_drift"] < 1e-12
    assert ev.conservation["angular_momentum"]["relative_drift"] < 1e-12
    assert ev.conservation["energy"]["relative_drift"] < 1e-3


def test_yoshida_beats_leapfrog_on_energy():
    g = PhysicsMutator.baseline()
    e_lf = evaluate_simulation(UniverseSimulator(g, small(integrator="leapfrog")).run()).conservation["energy"]
    e_y4 = evaluate_simulation(UniverseSimulator(g, small(integrator="yoshida4")).run()).conservation["energy"]
    assert e_y4["relative_drift"] < 0.1 * e_lf["relative_drift"]


@pytest.mark.parametrize("g", noether_battery(0), ids=lambda g: g.name)
def test_noether_battery_is_consistent(g):
    res = UniverseSimulator(g, small(steps=300)).run()
    assert not res.blew_up
    assert evaluate_simulation(res).conservation_consistency == 1.0


def test_broken_symmetries_produce_measurable_drift():
    for name in ("time-varying G", "non-reciprocal species", "anisotropic gravity"):
        g = next(x for x in noether_battery(0) if x.name == name)
        ev = evaluate_simulation(UniverseSimulator(g, small(steps=400)).run())
        broken = [k for k, v in ev.conservation.items() if not v["predicted"]]
        assert broken and all(ev.conservation[k]["relative_drift"] > 1e-4 for k in broken), name


@pytest.mark.parametrize("k", [2.0, 2.5])
def test_two_body_precession_matches_symbolic_apsidal_angle(k):
    g = PhysicsMutator.baseline()
    g.gravity.law, g.gravity.k = ("inverse_square", 2.0) if k == 2.0 else ("inverse_k", k)
    cfg = SimulationConfig(n_particles=2, steps=40000, dt=5e-4, record_every=5, softening=1e-5, lyapunov=False)
    sim = UniverseSimulator(g, cfg)
    m = sim.masses  # 0.5 each, G M_total = 1
    r0 = 1.0
    v_circ = math.sqrt(r0 ** (1 - k))  # relative circular speed for |F| = G m1 m2 r^-k
    vrel = 0.97 * v_circ
    X = np.array([[0.5 * r0, 0, 0], [-0.5 * r0, 0, 0]])
    V = np.array([[0, 0.5 * vrel, 0], [0, -0.5 * vrel, 0]])
    res = sim.run(initial=(X, V * m[:, None]))
    rel = res.positions[:, 0] - res.positions[:, 1]
    rad = np.linalg.norm(rel, axis=1)
    ang = np.unwrap(np.arctan2(rel[:, 1], rel[:, 0]))
    peri = [i for i in range(1, len(rad) - 1) if rad[i] < rad[i - 1] and rad[i] <= rad[i + 1]]
    assert len(peri) >= 3
    swept = np.diff(ang[peri])  # angle between successive pericentres = 2 * psi
    psi_theory = math.pi / math.sqrt(3 - k)
    assert np.allclose(swept, 2 * psi_theory, rtol=0.02)


@pytest.mark.parametrize("model", ["newtonian", "relativistic", "anisotropic", "power"])
def test_kinetics_inverse_consistency(model):
    g = PhysicsMutator.baseline()
    g.inertia.model = model
    g.inertia.c = 3.0
    g.inertia.kinetic_exponent = 2.5
    g.inertia.mass_tensor_chol = [0.2, 0.1, 0, 0, 0.1, -0.2]
    m = np.array([0.3, 0.7])
    kin = Kinetics(g, m, 3)
    V = np.array([[[0.4, -0.2, 0.1], [0.05, 0.3, -0.6]]])
    assert np.allclose(kin.velocity(kin.momentum_from_velocity(V)), V, rtol=1e-9)
    P = kin.momentum_from_velocity(V)
    # v = dT/dp by finite differences
    h = 1e-7
    for i in range(3):
        dP = np.zeros_like(P)
        dP[..., i] = h
        dT = (kin.T(P + dP) - kin.T(P - dP)) / (2 * h)
        assert np.allclose(dT, kin.velocity(P)[..., i], rtol=1e-5, atol=1e-9)


def test_rigid_body_flips_and_major_axis_rule():
    g = PhysicsMutator.baseline()
    rb = RigidBodyEnsemble(g, RigidBodyConfig(n_bodies=3, steps=4000)).run()
    assert rb.L_norm_drift < 1e-12
    assert np.all(rb.flips >= 1)  # Dzhanibekov effect
    g.thermo.friction = 5.0
    rb = RigidBodyEnsemble(g, RigidBodyConfig(n_bodies=3, steps=6000)).run()
    assert np.all(rb.final_axis_alignment > 0.9)
    assert np.max(np.abs(rb.L_space - rb.L_space[0])) < 1e-10 * np.max(np.abs(rb.L_space))


def test_fields_run_and_form_patterns():
    from scipy import ndimage

    g = PhysicsMutator.baseline()
    v = VorticityField2D(g, n=32).run(t_end=5.0)
    assert not v.blew_up and np.all(np.isfinite(v.final))
    rd = ReactionDiffusion2D(g, n=64).run(steps=2500)
    assert not rd.blew_up
    assert ndimage.label(rd.final > rd.final.mean() + 0.5 * rd.final.std())[1] >= 3


def test_lyapunov_positive_for_nbody_chaos():
    res = UniverseSimulator(PhysicsMutator.baseline(), small(lyapunov=True, steps=600)).run()
    assert res.lyapunov is not None and res.lyapunov > 0.1


def test_periodic_and_reflective_boundaries_stay_in_box():
    g = PhysicsMutator.baseline()
    for b in ("periodic", "reflective"):
        res = UniverseSimulator(g, small(boundary=b, initial_condition="uniform", box_size=3.0)).run()
        assert np.all(np.abs(res.positions) <= 1.5 + 1e-9)


def test_anti_de_sitter_universe_is_fully_bound():
    g = PhysicsMutator.baseline()
    g.cosmology.Lambda = -0.5
    res = UniverseSimulator(g, small(steps=200)).run()
    assert np.all(res.bound_fraction == 1.0)
