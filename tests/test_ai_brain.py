import numpy as np

from shiva_core.ai_brain import (CMAES, NSGA2, DesignEnvironment, DesignRLAgent, FeedForwardNetwork,
                                 GeneticAlgorithm, LatentDesignModel, NEATPopulation, OperatorBandit, ShivaBrain,
                                 fast_non_dominated_sort, hypervolume_2d)


def test_cmaes_solves_rosenbrock():
    f = lambda x: float(np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))
    x, fx = CMAES(np.zeros(5), 0.5, seed=1).optimize(f, 600, maximize=False)
    assert fx < 1e-8 and np.allclose(x, 1, atol=1e-3)


def test_ga_sphere():
    ga = GeneticAlgorithm(6, (np.full(6, -5.0), np.full(6, 5.0)), pop_size=40, seed=1)
    x, fx = ga.optimize(lambda x: -float(np.sum((x - 1.5) ** 2)), 120)
    assert fx > -1e-2


def test_nsga2_zdt1_front():
    def zdt1(X):
        f1 = X[:, 0]
        g = 1 + 9 * X[:, 1:].mean(1)
        return np.stack([-f1, -(g * (1 - np.sqrt(f1 / g)))], 1), np.zeros(len(X))

    r = NSGA2(10, 2, pop_size=60, seed=2).run(zdt1, 120)
    hv = hypervolume_2d(r.pareto_F, [-1.1, -1.1])
    assert hv > 0.95 * (1.21 - 1 / 3)


def test_non_dominated_sort_with_constraints():
    F = np.array([[1, 1], [2, 2], [0, 3], [5, 5]], float)
    CV = np.array([0, 0, 0, 1.0])
    fronts = fast_non_dominated_sort(F, CV)
    assert set(fronts[0].tolist()) == {1, 2} and 3 in fronts[-1]


def test_neat_solves_xor():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float)
    y = np.array([0, 1, 1, 0], float)
    fit = lambda g: 4 - float(np.sum((FeedForwardNetwork(g).activate(X)[:, 0] - y) ** 2))
    pop = NEATPopulation(2, 1, pop_size=150, seed=3, activations=("sigmoid",))
    best = pop.run(fit, 200, target=3.9)
    out = FeedForwardNetwork(best).activate(X)[:, 0]
    assert np.all((out > 0.5) == (y > 0.5))
    assert best.complexity[0] >= 1  # needed hidden structure


def test_ppca_recovers_noise_level():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 2)) @ rng.normal(size=(2, 6)) + 0.05 * rng.normal(size=(600, 6))
    m = LatentDesignModel(2).fit(X)
    assert abs(m.sigma2 - 0.0025) < 0.0008
    Z = m.encode(X[:5])
    assert Z.shape == (5, 2)


def test_bandit_prefers_best_arm():
    b = OperatorBandit(["a", "b", "c"], seed=0)
    rng = np.random.default_rng(0)
    for _ in range(400):
        a = b.select()
        b.update(a, {"a": 0.2, "b": 0.8, "c": 0.5}[a] + rng.normal(0, 0.1))
    assert max(b.n, key=b.n.get) == "b"


def test_reinforce_agent_learns_context_dependent_edits():
    ctxs = [(np.array([a, b]), np.array([a, b])) for a in (0.2, 0.8) for b in (0.3, 0.7)]
    env = DesignEnvironment(lambda x, c: -float(np.sum((x - c) ** 2)), 2, ctxs, horizon=8, seed=0)
    ag = DesignRLAgent(env.state_dim, 2, seed=0)
    ag.train(env, iterations=120, episodes_per_iter=8)
    assert ag.evaluate(env) > 3 * max(ag.evaluate(env, random_policy=True), 1e-3)


def test_shiva_brain_search_smoke():
    from shiva_core.universe_simulator import SimulationConfig

    brain = ShivaBrain(seed=0, verbose=False, sim_config=SimulationConfig(n_particles=16, steps=150, record_every=10))
    out = brain.search(generations=1, population=3, elite=1)
    assert out["best"].fitness >= 0 and out["evaluations"] >= 3
    assert "fluid" not in brain.bandit.arms
