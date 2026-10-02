"""
SHIVA-1 :: End-to-end pipeline
==============================

``run_pipeline`` executes the complete meta-science loop and writes every
artefact to an output directory::

    A  physics      showcase genomes along the gravity ladder, SymPy reports,
                    force-law figure
    B  noether      predicted-vs-measured conservation for a battery of
                    symmetry-breaking universes (validation of the simulator)
    C  search       ShivaBrain evolutionary/novelty search over genomes,
                    then CMA-ES refinement of the winner's constants
    D  universe     high-resolution particle + rigid-body + field run of the
                    winner and of the Newtonian baseline; emergence reports
    E  technology   NSGA-II evolution of 8 design domains under the winner's
                    physics, PPCA generative augmentation, REINFORCE agents
    F  generative   CPPN-NEAT free-form bracket under the winner's physics
    G  mapping      real-world projection, Monte-Carlo prediction, BOM,
                    build steps and CAD (SCAD / STL / FreeCAD macro)
    H  reporting    figures, dashboard, ``summary.json`` and ``results.md``

Profiles (``quick`` | ``standard`` | ``deep``) scale every budget.
"""

from __future__ import annotations

import json
import math
import platform
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from . import __version__
from . import visualizer as viz
from .ai_brain import ShivaBrain
from .cad_generator import CADGenerator
from .emergence_detector import EmergenceDetector, _jsonable
from .physics_mutator import PhysicsGenome, PhysicsMutator, SymbolicPhysics, noether_analysis
from .real_world_mapper import RealWorldMapper
from .tech_evolver import DOMAINS, GenerativeBracketDesigner, PhysicsContext, TechEvolver
from .universe_simulator import SimulationConfig, UniverseSimulator, evaluate_simulation, run_universe

__all__ = ["PROFILES", "showcase_genomes", "noether_battery", "run_pipeline"]

PROFILES: Dict[str, Dict[str, Any]] = {
    "smoke": dict(search_gens=1, search_pop=3, search_n=24, search_steps=200, final_n=32, final_steps=300,
                  tech_gens=3, tech_pop=8, rl_iters=2, rl_domains=1, bracket_gens=2, bracket_pop=8, mc=10,
                  cma_iters=1, field_n=32, noether_n=16, noether_steps=100, search_seeds=1, final_seeds=2,
                  validate_k=1, test_seeds=2),
    "quick": dict(search_gens=2, search_pop=6, search_n=48, search_steps=900, final_n=96, final_steps=2000,
                  tech_gens=12, tech_pop=24, rl_iters=12, rl_domains=2, bracket_gens=12, bracket_pop=24, mc=80,
                  cma_iters=1, field_n=48, noether_n=48, noether_steps=600, search_seeds=1, final_seeds=2, validate_k=2, test_seeds=2),
    "standard": dict(search_gens=5, search_pop=10, search_n=64, search_steps=1200, final_n=160, final_steps=4000,
                     tech_gens=30, tech_pop=36, rl_iters=40, rl_domains=3, bracket_gens=30, bracket_pop=40, mc=200,
                     cma_iters=2, field_n=64, noether_n=64, noether_steps=1000, search_seeds=2, final_seeds=3, validate_k=4, test_seeds=5),
    "deep": dict(search_gens=12, search_pop=16, search_n=96, search_steps=2000, final_n=256, final_steps=6000,
                 tech_gens=60, tech_pop=48, rl_iters=80, rl_domains=8, bracket_gens=60, bracket_pop=60, mc=400,
                 cma_iters=4, field_n=96, noether_n=96, noether_steps=1500, search_seeds=3, final_seeds=5, validate_k=6, test_seeds=8),
}


def showcase_genomes(seed: int = 0) -> List[PhysicsGenome]:
    """Baseline plus one representative per rung of the gravity ladder."""
    m = PhysicsMutator(seed=seed)
    out = []
    g = m.baseline()
    g.name = "newtonian (inverse-square)"
    out.append(g)
    g = m.baseline()
    g.gravity.law, g.gravity.k, g.name = "inverse_k", 2.5, "inverse-k (k = 2.5, d_eff = 3.5)"
    out.append(g)
    g = m.baseline()
    g.gravity.law, g.gravity.k = "tensor", 2.0
    g.gravity.tensor_chol = [0.35, 0.0, 0.0, 0.0, 0.0, -0.35]
    g.name = "tensor (anisotropic metric Q)"
    out.append(g)
    g = m.baseline()
    g.gravity.law, g.gravity.k, g.gravity.fractal_eps = "fractal", 2.0, 0.4
    g.name = "fractal (log-periodic, eps = 0.4)"
    out.append(g)
    g = m.baseline()
    g.gravity.law, g.gravity.screening_length, g.name = "yukawa", 1.0, "yukawa (lambda = 1)"
    out.append(g)
    for x in out:
        m.repair(x)
    return out


def noether_battery(seed: int = 0) -> List[PhysicsGenome]:
    """Universes that each break (or keep) a specific symmetry."""
    m = PhysicsMutator(seed=seed)
    cases = []

    def add(name: str, **mods: Any) -> None:
        g = m.baseline()
        for path, v in mods.items():
            obj = g
            parts = path.split("__")
            for p in parts[:-1]:
                obj = getattr(obj, p)
            setattr(obj, parts[-1], v)
        g.name = name
        m.repair(g)
        cases.append(g)

    add("newtonian")
    add("inverse-k 2.5", gravity__law="inverse_k", gravity__k=2.5)
    add("anisotropic gravity", gravity__law="tensor", gravity__tensor_chol=[0.35, 0, 0, 0, 0, -0.35])
    add("axisymmetric gravity", gravity__law="tensor", gravity__tensor_chol=[0, 0, 0, 0, 0, 0.5])
    add("relativistic inertia c=1.5", inertia__model="relativistic", inertia__c=1.5)
    add("anisotropic inertia", inertia__model="anisotropic", inertia__mass_tensor_chol=[0.3, 0, 0, 0, 0, -0.3])
    add("time-varying G", gravity__modulation_amp=0.3, gravity__modulation_freq=2.0)
    add("Lambda < 0 (AdS box)", cosmology__Lambda=-1.0)
    add("Coulomb + B field", em__law="coulomb", em__B=[0.0, 0.0, 2.0])
    add("non-reciprocal species", interaction__n_species=2, interaction__species_matrix=[[0.5, -1.5], [1.5, 0.5]],
        interaction__core_repulsion=1.0)
    add("retarded gravity c=3", interaction__propagation="retarded", interaction__c_prop=3.0)
    add("Langevin bath", thermo__friction=0.5, thermo__temperature=0.05)
    return cases


def _md_table(rows: List[List[Any]], header: List[str]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def run_pipeline(out_dir: str = "shiva_output/run", seed: int = 42, profile: str = "standard",
                 verbose: bool = True, figures: bool = True, cad: bool = True,
                 render_openscad: Optional[bool] = None) -> Dict[str, Any]:
    P = PROFILES[profile]
    out = Path(out_dir)
    figs = out / "figures"
    out.mkdir(parents=True, exist_ok=True)
    figs.mkdir(exist_ok=True)
    t_start = time.time()
    log = (lambda s: print(s, flush=True)) if verbose else (lambda s: None)
    summary: Dict[str, Any] = {"version": __version__, "seed": seed, "profile": profile,
                               "python": platform.python_version(), "stages": {}}

    # ---------------------------------------------------------------- A. physics
    log("[A] physics mutation: showcase genomes along the gravity ladder")
    show = showcase_genomes(seed)
    phys = []
    for g in show:
        rep = SymbolicPhysics(g).report()
        phys.append({"name": g.name, "genome": g.to_dict(), "report": rep})
        orb = rep["orbital"]
        log(f"    {g.name:38s} d_eff={rep['effective_dimension']:.2f} stable_circular={orb['stable_circular']}"
            f" apsidal={orb['apsidal_angle']:.3f} bands={orb['n_stable_bands']}"
            f" fields_verified={all(v['verified'] for v in rep['field_equations'].values())}")
    summary["stages"]["physics"] = phys
    if figures:
        viz.plot_force_laws(show, figs / "force_laws.png", labels=[g.name for g in show])

    # ---------------------------------------------------------------- B. noether
    log("[B] Noether validation battery")
    nrows = []
    for g in noether_battery(seed):
        cfg = SimulationConfig(n_particles=P["noether_n"], steps=P["noether_steps"], initial_condition="virialized",
                               seed=seed, lyapunov=False)
        res = UniverseSimulator(g, cfg).run()
        ev = evaluate_simulation(res)
        row = {"name": g.name}
        for q in ("energy", "momentum", "angular_momentum"):
            c = ev.conservation[q]
            row[q] = {"predicted": c["predicted"], "drift": c["relative_drift"], "consistent": c["consistent"]}
        row["consistency"] = ev.conservation_consistency
        nrows.append(row)
        log(f"    {g.name:28s} " + "  ".join(
            f"{q[:3]}:{'P' if row[q]['predicted'] else '-'} {row[q]['drift']:.1e}" for q in ("energy", "momentum",
                                                                                            "angular_momentum"))
            + f"  consistent={row['consistency']:.0%}")
    summary["stages"]["noether"] = nrows

    # ---------------------------------------------------------------- C. search
    log("[C] universe search (evolution + novelty + bandit operators)")
    brain = ShivaBrain(seed=seed, verbose=verbose, n_seeds=P["search_seeds"],
                       sim_config=SimulationConfig(n_particles=P["search_n"], steps=P["search_steps"], record_every=15,
                                                   initial_condition="cold_collapse", seed=seed))
    search = brain.search(generations=P["search_gens"], population=P["search_pop"], seed_genomes=show)
    best = search["best"]
    refined, refined_f, cma_hist = brain.refine_cmaes(best.genome, iterations=P["cma_iters"], popsize=6)
    if refined_f > best.fitness:
        best_genome = refined
        log(f"    CMA-ES improved fitness {best.fitness:.3f} -> {refined_f:.3f}")
    else:
        best_genome = best.genome
    best_genome.name = best_genome.name or f"shiva-best-{best_genome.genome_id}"
    summary["stages"]["search"] = {
        "best": best.brief(), "best_genome": best_genome.to_dict(), "best_description": best_genome.describe(),
        "history": search["history"], "bandit": search["bandit"], "evaluations": brain.evaluations,
        "top10": [c.brief() for c in search["ranked"][:10]], "cma_es": cma_hist, "refined_fitness": refined_f,
        "noether": noether_analysis(best_genome).to_dict(), "wall_time_s": search["wall_time"]}
    (out / "best_genome.json").write_text(best_genome.to_json())
    if figures:
        viz.plot_search(search["history"], figs / "universe_search.png")

    # ---------------------------------------------------------------- D. universe
    log("[D] validation: re-evaluate top candidates at final resolution on fresh seeds")
    det = EmergenceDetector(seed=seed)

    def final_runs(g: PhysicsGenome, keep_first: bool, n_seeds: Optional[int] = None,
                   seed_base: int = 7919) -> Tuple[List[float], Any]:
        Rs, first = [], None
        for k in range(n_seeds or P["final_seeds"]):
            cfg = SimulationConfig(n_particles=P["final_n"], steps=P["final_steps"], record_every=20,
                                   initial_condition="cold_collapse", seed=seed + seed_base + 101 * k)
            run = run_universe(g, cfg, fields=True, rigid=True, field_resolution=P["field_n"])
            rep = det.analyze(run)
            Rs.append(rep.richness)
            if k == 0 and keep_first:
                first = (run, rep)
        return Rs, first

    # distinct alien candidates: CMA-ES refinement + top of the search ranking
    cands: List[PhysicsGenome] = []
    seen = {show[0].fingerprint()}
    for g in [refined] + [c.genome for c in search["ranked"]]:
        if g.fingerprint() not in seen and len(cands) < P["validate_k"]:
            seen.add(g.fingerprint())
            cands.append(g)
    validation = []
    best_val = None
    for g in cands:
        Rs, first = final_runs(g, keep_first=True)
        entry = {"genome_id": g.genome_id, "law": g.gravity.law, "search_fitness": brain.cache[g.fingerprint()].fitness
                 if g.fingerprint() in brain.cache else None, "richness_per_seed": Rs,
                 "richness_mean": float(np.mean(Rs)), "richness_std": float(np.std(Rs, ddof=1)) if len(Rs) > 1 else 0.0}
        validation.append(entry)
        log(f"    candidate {g.genome_id} ({g.gravity.law:14s}) search={entry['search_fitness'] or float('nan'):.3f}"
            f" -> validated R = {entry['richness_mean']:.3f} +/- {entry['richness_std']:.3f}")
        if best_val is None or entry["richness_mean"] > best_val[0]["richness_mean"]:
            best_val = (entry, g, first)
    best_entry, best_genome, best_first = best_val
    best_genome.name = best_genome.name or f"shiva-best-{best_genome.genome_id}"
    (out / "best_genome.json").write_text(best_genome.to_json())
    summary["stages"]["search"]["validation"] = validation
    summary["stages"]["search"]["validated_best"] = best_entry
    summary["stages"]["search"]["best_genome"] = best_genome.to_dict()
    summary["stages"]["search"]["best_description"] = best_genome.describe()
    summary["stages"]["search"]["noether"] = noether_analysis(best_genome).to_dict()
    # Held-out TEST: the winner was selected on the validation seeds, so compare it with the
    # baseline on fresh seeds that played no role in any selection (unbiased estimate).
    log(f"    test: best vs baseline on {P['test_seeds']} fresh seeds")
    best_R, best_first = final_runs(best_genome, keep_first=True, n_seeds=P["test_seeds"], seed_base=50021)
    base_R, base_first = final_runs(show[0], keep_first=True, n_seeds=P["test_seeds"], seed_base=50021)
    runs = {"best": best_first, "baseline": base_first}

    def _st(R: List[float]) -> Dict[str, Any]:
        return {"richness_per_seed": R, "richness_mean": float(np.mean(R)),
                "richness_std": float(np.std(R, ddof=1)) if len(R) > 1 else 0.0}

    stats: Dict[str, Dict[str, Any]] = {"best": _st(best_R), "baseline": _st(base_R)}
    for label, g in (("best", best_genome), ("baseline", show[0])):
        run, rep = runs[label]
        log(f"    {label:8s} R={stats[label]['richness_mean']:.3f}+/-{stats[label]['richness_std']:.3f}"
            f" (n={len(stats[label]['richness_per_seed'])}) stability={run.evaluation.stability:.2f}"
            f" consistency={run.evaluation.conservation_consistency:.0%} feasibility={run.evaluation.feasibility:.2f}"
            f" lyapunov={run.particles.lyapunov:.3f}")
        if figures:
            viz.plot_particles(run.particles, figs / f"particles_{label}.png", title=f"{label}: {g.name}")
            viz.plot_conservation(run.particles, figs / f"conservation_{label}.png")
            viz.plot_emergence(rep, figs / f"emergence_{label}.png", title=f"Emergent phenomena ({label})")
            viz.plot_correlation(rep, figs / f"correlation_{label}.png")
            viz.plot_fields(run.fields, figs / f"fields_{label}.png")
            if run.rigid is not None:
                viz.plot_rigid(run.rigid, figs / f"rigid_{label}.png")
    if P["test_seeds"] > 1:
        from scipy.stats import ttest_ind

        t = ttest_ind(stats["best"]["richness_per_seed"], stats["baseline"]["richness_per_seed"], equal_var=False)
        stats["welch_t"] = {"t": float(t.statistic), "p": float(t.pvalue)}
        log(f"    Welch t-test (test seeds) best vs baseline: t = {t.statistic:.2f}, p = {t.pvalue:.3f}")
    summary["stages"]["universe"] = {
        label: {"evaluation": run.evaluation.to_dict(), "emergence": rep.to_dict(),
                "particles": run.particles.summary(), "rigid": run.rigid.summary() if run.rigid else None,
                "richness_stats": stats[label],
                "fields": {k: {"blew_up": f.blew_up, "params": f.params, "wall_time": f.wall_time}
                           for k, f in run.fields.items()}}
        for label, (run, rep) in runs.items()}
    summary["stages"]["universe"]["comparison"] = stats.get("welch_t")

    # ---------------------------------------------------------------- E. technology
    alien = PhysicsContext.from_genome(best_genome)
    log(f"[E] technology evolution under {alien.description}")
    tech = {}
    rl = {}
    for k, name in enumerate(DOMAINS):
        te = TechEvolver(name, alien, seed=seed + k)
        r = te.evolve(generations=P["tech_gens"], pop_size=P["tech_pop"])
        if k < P["rl_domains"]:
            r.rl = te.train_adaptive_designer(iterations=P["rl_iters"])
            rl[name] = r.rl
        tech[name] = r
        c = r.champion
        log(f"    {name:18s} pareto={len(r.pareto):3d} feasible={c.feasible} score={c.score:.3f} "
            + " ".join(f"{o[:4]}={v:.2f}" for o, v in c.objectives.items()))
    summary["stages"]["technology"] = {"context": alien.__dict__, **{n: r.summary() for n, r in tech.items()}}
    if figures:
        viz.plot_pareto(tech, figs / "pareto_efficiency_mappability.png")
        viz.plot_pareto(tech, figs / "pareto_efficiency_cosmic.png", y="cosmic")
        viz.plot_design_gallery({n: r.champion.params for n, r in tech.items()}, figs / "design_gallery.png",
                                title="Champions evolved under alien physics")
        if rl:
            viz.plot_rl(rl, figs / "rl_learning.png")

    # ---------------------------------------------------------------- F. generative
    log("[F] generative bracket (CPPN-NEAT + lattice FEM)")
    br = GenerativeBracketDesigner(ctx=alien, seed=seed).run(generations=P["bracket_gens"], pop_size=P["bracket_pop"])
    br_earth = GenerativeBracketDesigner(seed=seed)
    f_earth, m_earth = br_earth.fitness_of_shape(br["shape"])
    log(f"    alien fitness {br['fitness']:.3f}  (same shape on Earth: {f_earth:.3f}) volume {br['metrics'].get('volume_fraction', 0):.2f}")
    summary["stages"]["generative"] = {"fitness_alien": br["fitness"], "fitness_earth": f_earth,
                                       "metrics_alien": br["metrics"], "metrics_earth": m_earth,
                                       "complexity": br["history"][-1]["complexity"] if br["history"] else None,
                                       "shape": br["shape"].astype(int).tolist()}
    if figures:
        viz.plot_bracket(br, figs / "generative_bracket.png")

    # ---------------------------------------------------------------- G. mapping
    log("[G] real-world mapping + CAD")
    cadgen = CADGenerator(str(out / "cad"), render_openscad=render_openscad) if cad else None
    mapper = RealWorldMapper(n_monte_carlo=P["mc"], seed=seed, cad_generator=cadgen)
    protos = {}
    for name, r in tech.items():
        pr = mapper.map(r.champion, out_dir=str(out / "cad") if cad else None, name=name)
        protos[name] = pr
        log(f"    {name:18s} {pr.material:12s} feasible={pr.feasible} fidelity={pr.fidelity:.3f}"
            f" eff {pr.alien_efficiency:.2f}->{pr.real_efficiency:.2f} reliability={pr.reliability:.0%}"
            f" cad={'ok' if pr.cad.get('components_watertight') else '-'}"
            f" openscad={'ok' if pr.cad.get('openscad_stl') else '-'}")
    if cad:
        info = cadgen.generate("generative_bracket", {"shape": br["shape"].tolist(), "pixel_m": br["pixel_m"],
                                                      "thickness_m": br["thickness_m"]}, name="generative_bracket",
                               out_dir=str(out / "cad"))
        summary["stages"]["generative"]["cad"] = info
    summary["stages"]["mapping"] = {n: p.to_dict() for n, p in protos.items()}
    (out / "prototypes.md").write_text("# Real-world prototypes\n\n" + "\n\n".join(p.report() for p in protos.values()))
    if figures:
        viz.plot_design_gallery({n: p.real_params for n, p in protos.items()}, figs / "prototype_gallery.png",
                                captions={n: f"{p.material_name}" for n, p in protos.items()},
                                title="Real-world prototypes (Earth physics, real materials)")
        best_run, best_rep = runs["best"]
        viz.plot_dashboard(best_run, best_rep, search, tech, protos, figs / "dashboard.png")

    # ---------------------------------------------------------------- H. reporting
    summary["wall_time_s"] = time.time() - t_start
    (out / "summary.json").write_text(json.dumps(_jsonable(summary), indent=1, default=str))
    (out / "results.md").write_text(results_markdown(summary))
    log(f"[H] done in {summary['wall_time_s']:.0f}s -> {out}")
    summary["_objects"] = {"runs": runs, "tech": tech, "protos": protos, "bracket": br, "best_genome": best_genome,
                           "search": search}
    return summary


def results_markdown(s: Dict[str, Any]) -> str:
    """Auto-generated results section (tables) from a pipeline summary."""
    st = s["stages"]
    lines = [f"# SHIVA-1 results (seed {s['seed']}, profile {s['profile']}, v{s['version']})", ""]
    lines.append("## Noether validation")
    rows = []
    for r in st["noether"]:
        cells = [r["name"]]
        for q in ("energy", "momentum", "angular_momentum"):
            c = r[q]
            cells.append(f"{'conserved' if c['predicted'] else 'broken'} / {c['drift']:.1e}")
        cells.append(f"{r['consistency']:.0%}")
        rows.append(cells)
    lines.append(_md_table(rows, ["universe", "energy (pred / drift)", "momentum", "angular momentum", "consistency"]))
    lines.append("")
    lines.append("## Gravity ladder (symbolic analysis)")
    rows = []
    for p in st["physics"]:
        o = p["report"]["orbital"]
        rows.append([p["name"], f"{p['report']['effective_dimension']:.2f}", o["stable_circular"],
                     f"{o['apsidal_angle']:.4f}" if o["apsidal_angle"] == o["apsidal_angle"] else "-",
                     o["n_stable_bands"], all(v["verified"] for v in p["report"]["field_equations"].values())])
    lines.append(_md_table(rows, ["law", "d_eff", "stable circular orbit (r=1)", "apsidal angle", "stability bands",
                                  "field eqs verified"]))
    lines.append("")
    se = st["search"]
    lines.append("## Universe search")
    lines.append(f"Evaluations: {se['evaluations']}; best fitness {se['best']['fitness']} (after CMA-ES:"
                 f" {se['refined_fitness']:.4f}).")
    lines.append("")
    lines.append("```\n" + se["best_description"] + "\n```")
    lines.append("")
    rows = [[c["genome_id"], c["law"], c["fitness"], c["richness"], c["novelty"], c["feasibility"]] for c in se["top10"]]
    lines.append(_md_table(rows, ["genome", "gravity law", "fitness", "richness", "novelty", "feasibility"]))
    lines.append("")
    if se.get("validation"):
        lines.append("Held-out validation at final resolution (fresh seeds):")
        lines.append("")
        rows = [[v["genome_id"], v["law"], f"{v['search_fitness']:.3f}" if v["search_fitness"] is not None else "-",
                 f"{v['richness_mean']:.3f} +/- {v['richness_std']:.3f}"] for v in se["validation"]]
        lines.append(_md_table(rows, ["candidate", "gravity law", "search fitness", "validated richness"]))
        lines.append("")
    lines.append("## Emergence: best universe vs Newtonian baseline")
    u = st["universe"]
    keys = list(u["best"]["emergence"]["scores"])
    rows = [[k.replace("_", " "), f"{u['best']['emergence']['scores'][k]:.3f}",
             f"{u['baseline']['emergence']['scores'].get(k, float('nan')):.3f}"] for k in keys]
    rows.append(["**richness R (seed 0)**", f"**{u['best']['emergence']['richness']:.3f}**",
                 f"**{u['baseline']['emergence']['richness']:.3f}**"])
    rb, rB = u["best"]["richness_stats"], u["baseline"]["richness_stats"]
    rows.append([f"richness mean +/- sd, {len(rb['richness_per_seed'])} held-out test seeds",
                 f"{rb['richness_mean']:.3f} +/- {rb['richness_std']:.3f}",
                 f"{rB['richness_mean']:.3f} +/- {rB['richness_std']:.3f}"])
    lines.append(_md_table(rows, ["phenomenon", "best", "baseline"]))
    if u.get("comparison"):
        lines.append("")
        lines.append(f"Welch t-test on held-out test seeds (best vs baseline richness): t = {u['comparison']['t']:.2f},"
                     f" p = {u['comparison']['p']:.3f}.")
    lines.append("")
    for lab in ("best", "baseline"):
        m = u[lab]["emergence"]["metrics"]
        lines.append(f"- {lab}: Lyapunov {m['chaos']['lyapunov']:.3f}, D2 = {m['fractal']['D2']:.2f} (Poisson ref"
                     f" {m['fractal'].get('D2_poisson_reference', float('nan')):.2f}), FoF groups {m['clustering']['n_groups']},"
                     f" clustered fraction {m['clustering']['clustered_fraction']:.2f}, xi slope"
                     f" {m['clustering']['correlation'].get('gamma')}, dominant period {m['oscillation']['period']}")
    lines.append("")
    lines.append("## Technology champions (alien physics)")
    t = st["technology"]
    rows = []
    for name in DOMAINS:
        c = t[name]["champion"]
        o = c["objectives"]
        rows.append([name, c["feasible"]] + [f"{o[k]:.2f}" for k in ("efficiency", "stability", "adaptability",
                                                                      "mappability", "cosmic")]
                    + [t[name]["pareto_size"], t[name]["generative"].get("joined_pareto_front", "-")])
    lines.append(_md_table(rows, ["domain", "feasible", "eff", "stab", "adapt", "map", "cosmic", "pareto",
                                  "PPCA joined front"]))
    lines.append("")
    rl_rows = [[n, f"{t[n]['rl']['trained_gain']:.4f}", f"{t[n]['rl']['random_gain']:.4f}",
                f"{t[n]['rl']['untrained_gain']:.4f}"] for n in DOMAINS if t[n].get("rl")]
    if rl_rows:
        lines.append("Adaptive-design agent (mean improvement per 8-edit episode across 5 physics contexts):")
        lines.append("")
        lines.append(_md_table(rl_rows, ["domain", "trained policy", "random edits", "untrained"]))
        lines.append("")
    g = st["generative"]
    lines.append(f"Generative bracket: fitness {g['fitness_alien']:.3f} under alien physics,"
                 f" {g['fitness_earth']:.3f} for the same shape on Earth; volume fraction"
                 f" {g['metrics_alien'].get('volume_fraction', float('nan')):.2f}.")
    lines.append("")
    lines.append("## Real-world prototypes")
    rows = []
    for name, p in st["mapping"].items():
        rows.append([name, p["material_name"], p["feasible"], f"{p['fidelity']:.3f}",
                     f"{p['alien_efficiency']:.2f} -> {p['real_efficiency']:.2f}", f"{p['reliability']:.0%}",
                     f"{p['mass_kg']:.3g}", f"${p['cost_usd']:,.0f}"])
    lines.append(_md_table(rows, ["domain", "material", "feasible", "fidelity", "efficiency alien -> real",
                                  "MC reliability", "mass kg", "cost"]))
    lines.append("")
    lines.append(f"Total wall time: {s['wall_time_s']:.0f} s.")
    return "\n".join(lines) + "\n"
