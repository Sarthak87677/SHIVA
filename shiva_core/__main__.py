"""Command-line interface: ``python -m shiva_core <command>``.

Commands
--------
run        full pipeline (physics -> universes -> emergence -> tech -> real world -> CAD)
demo       the full pipeline with the ``quick`` profile (~2-4 minutes)
mutate     print mutated genomes with their symbolic laws and Noether structure
simulate   simulate one genome (baseline / random / JSON file) and report emergence
search     universe search only
evolve     evolve technology domains under a genome's physics
cad        regenerate CAD for a champion JSON or a domain's reference design
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_genome(spec: str, seed: int):
    from .physics_mutator import PhysicsGenome, PhysicsMutator

    m = PhysicsMutator(seed=seed)
    if spec == "baseline":
        return m.baseline()
    if spec == "random":
        return m.random_genome()
    if spec in ("inverse_k", "tensor", "fractal", "yukawa"):
        from .pipeline import showcase_genomes

        return next(g for g in showcase_genomes(seed) if g.gravity.law == spec)
    return PhysicsGenome.from_json(Path(spec).read_text())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m shiva_core", description="SHIVA-1 meta-science system")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run", help="full pipeline")
    p.add_argument("--profile", choices=("quick", "standard", "deep"), default="standard")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="shiva_output/run")
    p.add_argument("--no-cad", action="store_true")
    p.add_argument("--no-figures", action="store_true")

    p = sub.add_parser("demo", help="quick full pipeline")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", default="shiva_output/demo")

    p = sub.add_parser("mutate", help="generate mutated physics")
    p.add_argument("-n", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--latex", action="store_true")

    p = sub.add_parser("simulate", help="simulate one universe")
    p.add_argument("--genome", default="baseline", help="baseline | random | inverse_k | tensor | fractal | yukawa | path.json")
    p.add_argument("--n", type=int, default=128)
    p.add_argument("--steps", type=int, default=2500)
    p.add_argument("--ic", default="cold_collapse")
    p.add_argument("--integrator", default="leapfrog")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fields", action="store_true")
    p.add_argument("--out", default="shiva_output/simulate")

    p = sub.add_parser("search", help="universe search")
    p.add_argument("--generations", type=int, default=4)
    p.add_argument("--population", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="shiva_output/search")

    p = sub.add_parser("evolve", help="technology evolution")
    p.add_argument("--genome", default="baseline")
    p.add_argument("--domains", default="all")
    p.add_argument("--generations", type=int, default=25)
    p.add_argument("--population", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--map", action="store_true", help="also map champions to the real world and write CAD")
    p.add_argument("--out", default="shiva_output/evolve")

    a = ap.parse_args(argv)

    if a.cmd in ("run", "demo"):
        from .pipeline import run_pipeline

        if a.cmd == "demo":
            run_pipeline(a.out, seed=a.seed, profile="quick")
        else:
            run_pipeline(a.out, seed=a.seed, profile=a.profile, figures=not a.no_figures, cad=not a.no_cad)
        return 0

    if a.cmd == "mutate":
        from .physics_mutator import PhysicsMutator, SymbolicPhysics, noether_analysis

        m = PhysicsMutator(seed=a.seed)
        g = m.baseline()
        for i in range(a.n):
            g = m.mutate(g, strength=1.5, n_ops=2)
            print(g.describe())
            print("  lineage:", g.lineage[-1])
            print("  " + noether_analysis(g).summary().replace("\n", "\n  "))
            if a.latex:
                for k, v in SymbolicPhysics(g).latex().items():
                    print(f"  {k}: {v}")
            print()
        return 0

    if a.cmd == "simulate":
        from . import visualizer as viz
        from .emergence_detector import EmergenceDetector
        from .universe_simulator import SimulationConfig, run_universe

        g = _load_genome(a.genome, a.seed)
        print(g.describe())
        cfg = SimulationConfig(n_particles=a.n, steps=a.steps, initial_condition=a.ic, integrator=a.integrator, seed=a.seed)
        run = run_universe(g, cfg, fields=a.fields, rigid=a.fields)
        rep = EmergenceDetector().analyze(run)
        print(run.particles.noether.summary())
        print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "tolerance"}
                          for k, v in run.evaluation.conservation.items()}, indent=1))
        print(rep.summary())
        out = Path(a.out)
        viz.plot_particles(run.particles, out / "particles.png")
        viz.plot_conservation(run.particles, out / "conservation.png")
        viz.plot_emergence(rep, out / "emergence.png")
        if a.fields:
            viz.plot_fields(run.fields, out / "fields.png")
        print(f"figures -> {out}")
        return 0

    if a.cmd == "search":
        from . import visualizer as viz
        from .ai_brain import ShivaBrain

        brain = ShivaBrain(seed=a.seed)
        res = brain.search(generations=a.generations, population=a.population)
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "best_genome.json").write_text(res["best"].genome.to_json())
        viz.plot_search(res["history"], out / "search.png")
        print(res["best"].genome.describe())
        print(json.dumps(res["best"].brief(), indent=1))
        return 0

    if a.cmd == "evolve":
        from .cad_generator import CADGenerator
        from .real_world_mapper import RealWorldMapper
        from .tech_evolver import DOMAINS, PhysicsContext, evolve_technologies

        g = _load_genome(a.genome, a.seed)
        ctx = PhysicsContext.from_genome(g)
        print(ctx.description)
        doms = list(DOMAINS) if a.domains == "all" else a.domains.split(",")
        res = evolve_technologies(ctx, doms, generations=a.generations, pop_size=a.population, seed=a.seed)
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "champions.json").write_text(json.dumps({k: r.champion.to_dict() for k, r in res.items()}, indent=1))
        if a.map:
            mapper = RealWorldMapper(cad_generator=CADGenerator(str(out / "cad")))
            for k, r in res.items():
                pr = mapper.map(r.champion, out_dir=str(out / "cad"), name=k)
                print(pr.report())
                print()
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
