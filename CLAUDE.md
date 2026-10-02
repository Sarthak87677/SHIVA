# CLAUDE.md

SHIVA-1: alternate physics → universe simulation → emergence → technology evolution → real-world mapping → CAD.
Pure NumPy/SciPy/SymPy/matplotlib; optional OpenSCAD, FreeCAD, PyBullet, Jupyter.

## Commands

- Tests: `python -m pytest -q` (~1 min). Single file: `python -m pytest tests/test_universe_simulator.py -q`.
- Quick end-to-end check: `python -m shiva_core demo --out shiva_output/demo` (~2 min).
- Reference run for the docs: `python -m shiva_core run --profile standard --seed 42 --out shiva_output/standard_seed42`.
- Notebook: `python shiva_dashboard/build_dashboard.py --execute` (edit `build_dashboard.py`, never the .ipynb by hand).

## Architecture (dependency order)

`physics_mutator` → `universe_simulator` → `emergence_detector` → `ai_brain` → `tech_evolver`
(+ `real_world_mapper` materials) → `cad_generator` → `visualizer` → `pipeline` / `__main__`.
`real_world_mapper` imports `tech_evolver` lazily (inside `map`) to avoid a cycle.

## Invariants to preserve

- Conservative forces come only from SymPy potentials (`SymbolicPhysics.channels`): never hand-code a force.
- Conservation laws are derived by `noether_analysis`; never "mutate" a conservation law directly.
  Any new physics feature must update `noether_analysis` and keep `tests/test_universe_simulator.py::test_noether_battery_is_consistent` green.
- State arrays in the simulator carry a leading batch axis (main + Lyapunov twin).
- Design domains: SI units, constraints as normalised `g <= 0`, efficiency/stability in [0, 1].
  New domains need `material_ok`, `merit_index`, `snap`, `catalog_params`, `bom`, `build_steps`, a CAD builder
  `CADGenerator._build_<name>` and tests.
- CAD components must stay watertight (`Mesh.is_watertight`).
- Figures follow the palette/rules at the top of `visualizer.py` (fixed categorical order, no dual axes).
- Whitepaper numbers come from `shiva_docs/results/results.md`; regenerate rather than hand-edit.
