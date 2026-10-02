import json
import math

import numpy as np
import pytest

from shiva_core.physics_mutator import (ALPHA_CHEMISTRY_WINDOW, GRAVITY_LADDER, PARAM_BOUNDS, PhysicsGenome,
                                        PhysicsMutator, SymbolicPhysics, _get_path, noether_analysis,
                                        spd_from_params)


def genome(**mods):
    g = PhysicsMutator.baseline()
    for path, v in mods.items():
        obj = g
        parts = path.split("__")
        for p in parts[:-1]:
            obj = getattr(obj, p)
        setattr(obj, parts[-1], v)
    PhysicsMutator().repair(g)
    return g


def test_spd_from_params_is_unit_det_spd():
    rng = np.random.default_rng(0)
    for dim in (2, 3):
        for _ in range(20):
            Q = spd_from_params(rng.normal(0, 1, dim * (dim + 1) // 2), dim)
            assert np.allclose(Q, Q.T)
            assert np.all(np.linalg.eigvalsh(Q) > 0)
            assert math.isclose(np.linalg.det(Q), 1.0, rel_tol=1e-9)
    assert np.allclose(spd_from_params([0] * 6, 3), np.eye(3))


def test_kepler_orbits_are_closed():
    orb = SymbolicPhysics(PhysicsMutator.baseline()).orbital_analysis(1.0)
    assert orb["stable_circular"]
    assert math.isclose(orb["apsidal_angle"], math.pi, rel_tol=1e-12)  # Bertrand
    assert orb["closed_orbits_expected"]


@pytest.mark.parametrize("k", [1.5, 2.5, 2.9])
def test_power_law_apsidal_angle(k):
    orb = SymbolicPhysics(genome(gravity__law="inverse_k", gravity__k=k)).orbital_analysis(1.3)
    assert math.isclose(orb["apsidal_angle"], math.pi / math.sqrt(3 - k), rel_tol=1e-9)


def test_power_law_k_above_3_has_no_stable_circular_orbit():
    orb = SymbolicPhysics(genome(gravity__law="inverse_k", gravity__k=3.3)).orbital_analysis(1.0)
    assert not orb["stable_circular"]


def test_yukawa_stability_edge_matches_classical_result():
    lam = 2.0
    orb = SymbolicPhysics(genome(gravity__law="yukawa", gravity__screening_length=lam)).orbital_analysis()
    edge = orb["stable_bands"][0][1]
    assert abs(edge - lam * (1 + math.sqrt(5)) / 2) / edge < 0.03


def test_fractal_gravity_has_multiple_stability_bands():
    orb = SymbolicPhysics(genome(gravity__law="fractal", gravity__fractal_eps=0.4)).orbital_analysis()
    assert orb["n_stable_bands"] > 3


@pytest.mark.parametrize("law", ["inverse_square", "inverse_k", "tensor", "fractal", "yukawa"])
def test_field_equations_verified(law):
    g = genome(gravity__law=law, gravity__k=2.3 if law in ("inverse_k",) else 2.0,
               gravity__tensor_chol=[0.3, 0, 0, 0.1, 0, -0.2], gravity__fractal_eps=0.3,
               em__law="proca", em__proca_mass=0.7)
    fe = SymbolicPhysics(g).field_equations()
    assert fe and all(v["verified"] for v in fe.values()), {k: v["verified"] for k, v in fe.items()}


def test_effective_dimension():
    assert SymbolicPhysics(genome(gravity__law="inverse_k", gravity__k=2.5)).effective_dimension() == 3.5


def test_tsallis_reduces_to_shannon():
    assert SymbolicPhysics(PhysicsMutator.baseline()).entropy_functional()["shannon_limit_verified"]


def test_compiled_kernel_matches_newton():
    ch = SymbolicPhysics(PhysicsMutator.baseline()).channels(3)["gravity"]
    S = np.array([0.25, 1.0, 4.0])
    assert np.allclose(ch.kappa(S), -S ** -1.5)
    assert np.allclose(ch.U(S), -S ** -0.5)


def test_force_is_minus_gradient_for_every_law():
    for law in ("inverse_k", "yukawa", "fractal"):
        sym = SymbolicPhysics(genome(gravity__law=law, gravity__k=2.4, gravity__fractal_eps=0.3))
        ch = sym.channels(3)["gravity"]
        r = np.linspace(0.3, 3.0, 50)
        h = 1e-6
        dU = (ch.U((r + h) ** 2) - ch.U((r - h) ** 2)) / (2 * h)
        # F_i = kappa * r_vec  -> radial component kappa * r = -dU/dr
        assert np.allclose(ch.kappa(r ** 2) * r, -dU, rtol=1e-5, atol=1e-8)


def test_noether_predictions():
    assert noether_analysis(PhysicsMutator.baseline()).predicted() == {"energy": True, "momentum": True,
                                                                       "angular_momentum": True}
    n = noether_analysis(genome(gravity__law="tensor", gravity__tensor_chol=[0.35, 0, 0, 0, 0, -0.35]))
    assert n.energy and n.momentum and not n.angular_momentum
    n = noether_analysis(genome(gravity__law="tensor", gravity__tensor_chol=[0, 0, 0, 0, 0, 0.5]))
    assert len(n.angular_axes) == 1 and np.allclose(np.abs(n.angular_axes[0]), [0, 0, 1])
    n = noether_analysis(genome(gravity__modulation_amp=0.2))
    assert not n.energy and n.momentum and n.angular_momentum
    n = noether_analysis(genome(cosmology__Lambda=-1.0))
    assert n.energy and not n.momentum and n.angular_momentum
    n = noether_analysis(genome(em__law="coulomb", em__B=[0, 0, 2.0]))
    assert n.momentum_kind == "pseudo" and n.angular_kind == "canonical" and not n.time_reversal
    n = noether_analysis(genome(interaction__n_species=2, interaction__species_matrix=[[0, 1], [-1, 0]]))
    assert not (n.energy or n.momentum or n.angular_momentum)
    assert n.liouville  # position-dependent non-reciprocal forces still preserve phase-space volume
    n = noether_analysis(PhysicsMutator.baseline(), boundary="periodic")
    assert n.momentum and not n.angular_momentum
    assert noether_analysis(PhysicsMutator.baseline()).virial_degree == -1.0


def test_mutation_creates_valid_lineage_and_respects_bounds():
    m = PhysicsMutator(seed=1)
    g = m.baseline()
    for _ in range(60):
        c = m.mutate(g, strength=2.0)
        assert c.parents == [g.genome_id] and c.generation == g.generation + 1
        assert len(c.lineage) == len(g.lineage) + 1
        for path, (lo, hi) in PARAM_BOUNDS.items():
            v = _get_path(c, path)
            assert lo - 1e-9 <= v <= hi + 1e-9, path
        lo, hi = ALPHA_CHEMISTRY_WINDOW
        assert lo <= c.em.alpha_ratio <= hi
        assert c.gravity.law in ("yukawa",) + GRAVITY_LADDER
        g = c
    assert g.dim == 3  # dimension operator is off by default


def test_crossover_and_serialisation_roundtrip():
    m = PhysicsMutator(seed=2)
    a, b = m.random_genome(), m.random_genome()
    c = m.crossover(a, b)
    assert set(c.parents) == {a.genome_id, b.genome_id}
    d = PhysicsGenome.from_json(c.to_json())
    assert d.fingerprint() == c.fingerprint()
    assert json.loads(c.to_json())["gravity"]["law"] == c.gravity.law


def test_vectorize_roundtrip():
    m = PhysicsMutator(seed=3)
    g = m.random_genome()
    x = m.vectorize(g)
    assert np.all((x >= 0) & (x <= 1))
    g2 = m.devectorize(x, g)
    assert np.allclose(m.vectorize(g2), x, atol=1e-9)


def test_jeans_analysis_limits():
    import math as _m

    j = SymbolicPhysics(PhysicsMutator.baseline()).jeans_analysis(rho0=2.0, cs=0.5)
    assert _m.isclose(j["A_k"], 4 * _m.pi)
    assert _m.isclose(j["q_J"] ** 2, 4 * _m.pi * 2.0 / 0.25)
    j = SymbolicPhysics(genome(gravity__law="inverse_k", gravity__k=2.6)).jeans_analysis()
    assert 0 < j["q_star"] < j["q_J"] and j["max_growth_rate"] > 0
    y = SymbolicPhysics(genome(gravity__law="yukawa", gravity__screening_length=0.05)).jeans_analysis()
    assert not y["unstable"]  # screening shorter than the Jeans length switches clustering off


def test_fractal_orbit_closed_form():
    """psi = pi sqrt((M - M')/(M - M'')) for U = -G M(ln r)/r (derived in the cosmic appendix)."""
    g = genome(gravity__law="fractal", gravity__fractal_eps=0.4, gravity__k=2.0, gravity__tensor_chol=[0] * 6)
    gg = g.gravity
    om = [gg.fractal_omega * gg.fractal_beta ** n for n in range(gg.fractal_octaves)]
    amp = [gg.fractal_eps * gg.fractal_gamma ** n for n in range(gg.fractal_octaves)]
    M = 1 + sum(amp)
    M2 = -sum(a * w * w for a, w in zip(amp, om))
    psi = math.pi * math.sqrt(M / (M - M2))  # at r = r0 = 1: M' = 0
    assert math.isclose(SymbolicPhysics(g).orbital_analysis(1.0)["apsidal_angle"], psi, rel_tol=1e-9)
