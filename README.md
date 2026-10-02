# SHIVA-1

**A self-evolving meta-science system** that generates alternate laws of physics, simulates the universes
they produce, detects emergent phenomena, evolves technologies inside those universes, and translates the
best designs back into buildable, CAD-ready prototypes in our universe.

```
 physics_mutator ──► universe_simulator ──► emergence_detector ──► ai_brain (universe search)
   SymPy laws          particles · rigid       12 estimators,          GA · CMA-ES · NSGA-II ·
   Noether analysis    bodies · fields         richness score          NEAT/CPPN · PPCA · RL
                                                                             │
        cad_generator ◄── real_world_mapper ◄── tech_evolver ◄──────────────┘
        SCAD · STL ·      materials · SLSQP      8 engineering domains
        FreeCAD macro     projection · BOM       under alien physics
```

Everything is first-principles NumPy/SciPy/SymPy code — no black boxes — and is covered by 84 tests,
including end-to-end checks that simulated orbits precess by exactly the SymPy-predicted apsidal angle
and that every predicted conservation law holds to machine precision.

## Layout

| Path | Contents |
|---|---|
| `shiva_core/physics_mutator.py` | physics genomes, mutation/crossover, symbolic laws, field-equation verification, orbital analysis, Noether analysis |
| `shiva_core/universe_simulator.py` | symplectic N-body (leapfrog / Yoshida-4 / RK4, BAOAB thermostat, Boris rotation, retarded forces), Lyapunov twin trajectories, rigid bodies, 2-D vorticity and reaction-diffusion fields |
| `shiva_core/emergence_detector.py` | entropy/LMC complexity, order parameters, periodograms, RQA, Lyapunov, correlation dimension, CUSUM, friends-of-friends, Landy–Szalay ξ(r), pattern metrics; emergent-richness score |
| `shiva_core/ai_brain.py` | GA, CMA-ES, NSGA-II, NEAT + CPPN, probabilistic PCA, REINFORCE, UCB1 bandit, `ShivaBrain` universe search |
| `shiva_core/tech_evolver.py` | `PhysicsContext` (alien genome → engineering constants), real planetary environments, 8 design domains, NSGA-II evolution, adaptive-design RL, CPPN-NEAT generative bracket |
| `shiva_core/real_world_mapper.py` | materials database, Ashby ranking, constrained projection with design margin, catalogue snapping, Monte-Carlo reliability, BOM, build steps |
| `shiva_core/cad_generator.py` | mesh kernel (watertight revolve/extrude/cylinders/voxels), parametric OpenSCAD (involute gears computed in SCAD), binary STL, FreeCAD macro |
| `shiva_core/visualizer.py` | publication figures and dashboard |
| `shiva_core/pipeline.py`, `__main__.py` | end-to-end pipeline and CLI |
| `shiva_docs/whitepaper.md` | research whitepaper (results from a reproducible run) |
| `shiva_docs/cosmic_appendix.md` | cosmic-scale theory appendix |
| `shiva_docs/results/`, `shiva_docs/figures/`, `shiva_docs/prototypes/` | artefacts of the reference run (seed 42, standard profile) |
| `shiva_dashboard/dashboard.ipynb` | interactive notebook (regenerate with `build_dashboard.py`) |
| `tests/` | pytest suite |

## Quick start

```bash
pip install -r requirements.txt          # numpy scipy sympy matplotlib (+ pytest, nbformat...)
python -m pytest -q                      # 84 tests, ~1 minute
python -m shiva_core demo                # full loop, quick profile, ~2 minutes -> shiva_output/demo/
python -m shiva_core run --profile standard --seed 42   # reference run used in the whitepaper (~10 min)
```

Other commands:

```bash
python -m shiva_core mutate -n 5 --latex                       # mutated laws + Noether structure
python -m shiva_core simulate --genome fractal --fields         # one universe, figures in shiva_output/simulate
python -m shiva_core search --generations 4 --population 8      # universe search only
python -m shiva_core evolve --genome shiva_output/standard_seed42/best_genome.json --map   # tech + CAD
python shiva_dashboard/build_dashboard.py --execute             # rebuild and run the notebook
```

Each run writes `summary.json` (machine-readable), `results.md` (tables), `prototypes.md` (build plans),
`figures/*.png` and `cad/*.{scad,stl,FCMacro,json}`.

### Optional tools

* **OpenSCAD** (`apt install openscad`): every `.scad` is also rendered to `*_openscad.stl` (true CSG union).
* **FreeCAD**: run any generated `*.FCMacro` (Macro → Execute, or `freecadcmd file.FCMacro`) to obtain a
  solid with a parameter spreadsheet; executed automatically when the FreeCAD Python API is importable.
* **PyBullet**: `PyBulletRigidBackend` drops rigid boxes under the genome's surface gravity when installed.

## Running inside Claude Code

Open the repository in Claude Code (CLI, desktop, IDE or claude.ai/code); `CLAUDE.md` tells Claude how
the project is organised and how to verify changes. Useful prompts:

* "Run the test suite and the quick demo, then summarise `shiva_output/demo/results.md`."
* "Mutate gravity to an inverse-2.8 law with a Yukawa screen, simulate it and compare its emergence report
  with the Newtonian baseline."
* "Add a new design domain for a heat-exchanger fin array to `tech_evolver.py`, with tests, CAD and a
  mapping hook."
* "Rerun the standard pipeline with seed 7 and update the whitepaper's results section."

## Safety and scope

SHIVA-1 explores *toy universes* to study how laws shape emergence and design; its alternate-physics
results are not claims about nature. Engineering outputs are reduced-order models with typical handbook
material values — treat them as concept-stage designs. Anything rotating at speed, pressurised,
load-bearing or propulsive must be reviewed by a qualified engineer and tested under appropriate safety
procedures (the generated build instructions say so explicitly).
