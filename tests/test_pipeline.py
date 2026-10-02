import json

from shiva_core.pipeline import run_pipeline


def test_pipeline_smoke(tmp_path):
    s = run_pipeline(str(tmp_path / "run"), seed=1, profile="smoke", verbose=False, render_openscad=False)
    out = tmp_path / "run"
    for f in ("summary.json", "results.md", "best_genome.json", "prototypes.md", "figures/dashboard.png",
              "figures/force_laws.png", "cad/gear_pair.scad", "cad/generative_bracket.stl"):
        assert (out / f).exists(), f
    summary = json.loads((out / "summary.json").read_text())
    assert all(r["consistency"] == 1.0 for r in summary["stages"]["noether"])
    assert set(summary["stages"]["mapping"]) == set(summary["stages"]["technology"]) - {"context"}
    assert "Noether validation" in (out / "results.md").read_text()
    assert s["stages"]["search"]["validated_best"]["richness_mean"] >= 0
    # interactive dashboard export
    dash = json.loads((out / "dashboard_data.json").read_text())
    for key in ("meta", "physics", "noether", "search", "emergence", "universes", "force_laws", "technology", "figures"):
        assert key in dash, key
    for u in dash["universes"].values():
        assert len(u["frames"]) == len(u["times"]) == len(u["energy_drift"]) >= 2
        assert all(len(f) == 3 * u["n"] for f in u["frames"])
    assert {t["domain"] for t in dash["technology"]} == set(summary["stages"]["mapping"])
    assert all((out / t["stl"]).exists() for t in dash["technology"] if t["stl"])
