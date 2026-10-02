"""Builds ``shiva_dashboard/dashboard.ipynb`` from source (keeps the notebook diff-able).

    python shiva_dashboard/build_dashboard.py            # write the notebook
    python shiva_dashboard/build_dashboard.py --execute  # write and execute it headless (needs nbclient, ipykernel)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent

CELLS = [
    ("md", """# SHIVA-1 — multi-domain research dashboard

Interactive tour of the full meta-science loop:

**alternate physics → universe simulation → emergence detection → AI technology evolution → real-world mapping → CAD.**

Every cell runs on a laptop CPU. `QUICK = True` keeps the whole notebook to a few minutes; set it to `False`
for research-grade budgets. Figures are written to `shiva_dashboard/output/` and displayed inline.
The end-to-end, reproducible version of everything below is `python -m shiva_core run --profile standard`."""),
    ("code", """import sys, json, time
from pathlib import Path
ROOT = Path.cwd().resolve().parent if Path.cwd().name == "shiva_dashboard" else Path.cwd().resolve()
sys.path.insert(0, str(ROOT))
OUT = ROOT / "shiva_dashboard" / "output"
OUT.mkdir(parents=True, exist_ok=True)

import numpy as np
from IPython.display import Image, Markdown, Math, display

from shiva_core import visualizer as viz
from shiva_core.physics_mutator import PhysicsMutator, SymbolicPhysics, noether_analysis
from shiva_core.universe_simulator import SimulationConfig, run_universe
from shiva_core.emergence_detector import EmergenceDetector
from shiva_core.ai_brain import ShivaBrain
from shiva_core.tech_evolver import DOMAINS, PhysicsContext, TechEvolver, GenerativeBracketDesigner
from shiva_core.real_world_mapper import RealWorldMapper
from shiva_core.cad_generator import CADGenerator
from shiva_core.pipeline import showcase_genomes

QUICK = True
SEED = 11
B = dict(n=64 if QUICK else 160, steps=1200 if QUICK else 4000, tech_gens=10 if QUICK else 30,
         tech_pop=20 if QUICK else 36, search_gens=1 if QUICK else 5, search_pop=4 if QUICK else 10)

def show(path, width=900):
    display(Image(filename=str(path), width=width))"""),
    ("md", """## 1 · Physics mutation engine
Five universes along the gravitational escalation ladder. Each law is a SymPy potential; forces are its exact
negative gradient, so energy conservation is never assumed by hand."""),
    ("code", """genomes = showcase_genomes(SEED)
for g in genomes:
    sym = SymbolicPhysics(g)
    display(Markdown(f"**{g.name}**  —  effective Gauss-law dimension $d_{{eff}} = {sym.effective_dimension():.2f}$"))
    display(Math(r"U(r) = " + sym.latex()["U_gravity"]))
viz.plot_force_laws(genomes, OUT / "force_laws.png", labels=[g.name for g in genomes])
show(OUT / "force_laws.png")"""),
    ("md", """### Symmetries → conservation laws (Noether)
Conservation laws are *derived*, never mutated directly: breaking a symmetry is the only way a conserved quantity
disappears."""),
    ("code", """rows = ["| universe | energy | momentum | angular momentum | time reversal |", "|---|---|---|---|---|"]
for g in genomes:
    n = noether_analysis(g)
    rows.append(f"| {g.name} | {n.energy} | {n.momentum} ({n.momentum_kind}) | {len(n.angular_axes)} axes | {n.time_reversal} |")
display(Markdown("\\n".join(rows)))
orb = SymbolicPhysics(genomes[3]).orbital_analysis()
display(Markdown(f"Fractal gravity: **{orb['n_stable_bands']} separate bands** of stable circular orbits between r = 0.05 and 20."))"""),
    ("md", "## 2 · Universe simulation (particles + rigid bodies + continuum fields)"),
    ("code", """g = genomes[3]                      # fractal gravity
cfg = SimulationConfig(n_particles=B["n"], steps=B["steps"], initial_condition="cold_collapse", seed=SEED)
t0 = time.time()
run = run_universe(g, cfg, fields=True, rigid=True, field_resolution=48 if QUICK else 64)
print(f"simulated in {time.time()-t0:.1f}s;  Lyapunov exponent = {run.particles.lyapunov:.2f}")
print(json.dumps({k: {kk: v[kk] for kk in ("predicted", "relative_drift", "consistent")}
                  for k, v in run.evaluation.conservation.items()}, indent=1))
viz.plot_particles(run.particles, OUT / "particles.png", title=g.name); show(OUT / "particles.png")
viz.plot_conservation(run.particles, OUT / "conservation.png"); show(OUT / "conservation.png", 650)"""),
    ("code", """viz.plot_fields(run.fields, OUT / "fields.png"); show(OUT / "fields.png")
viz.plot_rigid(run.rigid, OUT / "rigid.png"); show(OUT / "rigid.png")"""),
    ("md", "## 3 · Emergence detector"),
    ("code", """rep = EmergenceDetector(seed=SEED).analyze(run)
print(rep.summary())
viz.plot_emergence(rep, OUT / "emergence.png"); show(OUT / "emergence.png", 650)
viz.plot_correlation(rep, OUT / "correlation.png"); show(OUT / "correlation.png", 500)"""),
    ("md", """## 4 · AI brain: universe search
Evolution over physics genomes with a novelty archive, UCB1-chosen mutation operators and 1/5th-rule step-size
adaptation. (Use the pipeline for held-out validation of the winner.)"""),
    ("code", """brain = ShivaBrain(seed=SEED, verbose=False,
                   sim_config=SimulationConfig(n_particles=40, steps=600, record_every=15, seed=SEED))
search = brain.search(generations=B["search_gens"], population=B["search_pop"], seed_genomes=genomes[:B["search_pop"]])
best = search["best"]
print(best.genome.describe()); print(best.brief())
viz.plot_search(search["history"], OUT / "search.png"); show(OUT / "search.png", 500)"""),
    ("md", "## 5 · Technology evolution under the winner's physics"),
    ("code", """alien = PhysicsContext.from_genome(best.genome)
print(alien.description)
tech = {}
for k, name in enumerate(DOMAINS):
    tech[name] = TechEvolver(name, alien, seed=SEED + k).evolve(generations=B["tech_gens"], pop_size=B["tech_pop"])
    c = tech[name].champion
    print(f"{name:18s} feasible={c.feasible} " + " ".join(f"{o}={v:.2f}" for o, v in c.objectives.items()))
viz.plot_pareto(tech, OUT / "pareto.png"); show(OUT / "pareto.png")
viz.plot_design_gallery({n: r.champion.params for n, r in tech.items()}, OUT / "gallery.png"); show(OUT / "gallery.png")"""),
    ("md", "## 6 · Real-world mapping, build instructions and CAD"),
    ("code", """cad = CADGenerator(str(OUT / "cad"), render_openscad=False)
mapper = RealWorldMapper(n_monte_carlo=60, seed=SEED, cad_generator=cad)
protos = {n: mapper.map(r.champion, out_dir=str(OUT / "cad"), name=n) for n, r in tech.items()}
for n, p in protos.items():
    print(f"{n:18s} {p.material_name:40s} feasible={p.feasible} fidelity={p.fidelity:.3f} reliability={p.reliability:.0%} -> {Path(p.cad['scad']).name}")
display(Markdown(protos["gear_pair"].report()))"""),
    ("md", "## 7 · Generative design (CPPN-NEAT + lattice FEM)"),
    ("code", """br = GenerativeBracketDesigner(ctx=alien, seed=SEED).run(generations=10 if QUICK else 30, pop_size=24 if QUICK else 40)
print(f"stiffness relative to solid plate: {br['fitness']:.3f} at volume fraction {br['metrics']['volume_fraction']:.2f}")
viz.plot_bracket(br, OUT / "bracket.png"); show(OUT / "bracket.png")"""),
    ("md", "## 8 · Dashboard"),
    ("code", """viz.plot_dashboard(run, rep, search, tech, protos, OUT / "dashboard.png"); show(OUT / "dashboard.png", 1000)"""),
    ("md", """## 9 · Reproducible pipeline results
If you have run `python -m shiva_core run --profile standard --seed 42`, its auto-generated results are shown here."""),
    ("code", """res = ROOT / "shiva_output" / "standard_seed42" / "results.md"
alt = ROOT / "shiva_docs" / "results" / "results.md"
display(Markdown(res.read_text() if res.exists() else alt.read_text() if alt.exists() else "_no pipeline run found_"))"""),
]


def build() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    for kind, src in CELLS:
        nb.cells.append(nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src))
    return nb


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    nb = build()
    path = HERE / "dashboard.ipynb"
    if a.execute:
        from nbclient import NotebookClient

        NotebookClient(nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": str(HERE)}}).execute()
        # keep the committed notebook light: drop images, keep text outputs
        for cell in nb.cells:
            if cell.cell_type == "code":
                cell.outputs = [o for o in cell.outputs if not ("data" in o and "image/png" in o.get("data", {}))]
    nbf.validate(nb)
    nbf.write(nb, path)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
