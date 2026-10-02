import numpy as np

from shiva_core.emergence_detector import (PHENOMENA, EmergenceDetector, correlation_dimension, cusum_change_point,
                                           friends_of_friends, landy_szalay, lmc_complexity, rqa, spatial_entropy)
from shiva_core.physics_mutator import PhysicsMutator
from shiva_core.universe_simulator import SimulationConfig, run_universe


def test_entropy_and_complexity_distinguish_clumps_from_uniform():
    rng = np.random.default_rng(0)
    U = rng.uniform(-1, 1, (300, 3))
    C = np.concatenate([rng.normal(c, 0.05, (100, 3)) for c in ([0.5, 0.5, 0.5], [-0.5, 0, 0.2], [0, -0.6, -0.4])])
    assert spatial_entropy(U) > spatial_entropy(C)
    assert lmc_complexity(C)[0] > lmc_complexity(U)[0]


def test_correlation_dimension_of_line_and_plane():
    rng = np.random.default_rng(1)
    line = np.column_stack([rng.random(400), np.zeros(400), np.zeros(400)])
    plane = np.column_stack([rng.random(800), rng.random(800), np.zeros(800)])
    assert abs(correlation_dimension(line)["D2"] - 1.0) < 0.15
    assert abs(correlation_dimension(plane)["D2"] - 2.0) < 0.3


def test_fof_finds_planted_haloes():
    rng = np.random.default_rng(2)
    X = np.concatenate([rng.normal(c, 0.02, (30, 3)) for c in ([0, 0, 0], [1, 0, 0], [0, 1, 0])]
                       + [rng.uniform(-1, 2, (20, 3))])
    fof = friends_of_friends(X, 0.05, min_members=5)
    assert fof["n_groups"] == 3


def test_landy_szalay_positive_for_clustered_and_flat_for_poisson():
    rng = np.random.default_rng(3)
    edges = np.geomspace(0.02, 0.5, 8)
    C = np.concatenate([rng.normal(rng.uniform(-1, 1, 3), 0.05, (20, 3)) for _ in range(10)])
    U = rng.uniform(-1, 1, (200, 3))
    assert np.nanmean(landy_szalay(C, edges, rng=rng)[:3]) > 3.0
    assert abs(np.nanmean(landy_szalay(U, edges, rng=rng)[2:])) < 0.5


def test_rqa_determinism_sine_vs_noise():
    t = np.linspace(0, 20 * np.pi, 300)
    sine = np.column_stack([np.sin(t), np.cos(t)])
    noise = np.random.default_rng(4).normal(size=(300, 2))
    assert rqa(sine)["DET"] > rqa(noise)["DET"] + 0.3


def test_cusum_detects_step():
    x = np.r_[np.zeros(50), np.ones(50)] + np.random.default_rng(5).normal(0, 0.1, 100)
    cp = cusum_change_point(x)
    assert abs(cp["index"] - 50) <= 3 and cp["effect"] > 5


def test_analyze_full_run_scores_in_unit_interval():
    cfg = SimulationConfig(n_particles=48, steps=600, seed=0)
    run = run_universe(PhysicsMutator.baseline(), cfg, fields=True, rigid=True, field_resolution=32, rd_steps=800,
                       vort_t_end=3.0)
    rep = EmergenceDetector().analyze(run)
    assert set(PHENOMENA) <= set(rep.scores)
    assert all(0.0 <= v <= 1.0 for v in rep.scores.values())
    assert 0.0 <= rep.richness <= 1.0
    assert rep.metrics["chaos"]["lyapunov"] > 0
    assert rep.to_dict()["richness"] == rep.richness
