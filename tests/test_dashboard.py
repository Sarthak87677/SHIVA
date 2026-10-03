import json
import shutil
import subprocess
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from shiva_core.dashboard_server import WEB_DIR, find_latest_run, make_server, trajectory_payload
from shiva_core.physics_mutator import PhysicsMutator
from shiva_core.universe_simulator import SimulationConfig, run_universe

ROOT = Path(__file__).resolve().parent.parent


def test_web_app_files_present():
    for f in ("index.html", "style.css", "app.js", "charts.js", "gestures.js", "airdraw.js", "vendor/three.module.js",
              "vendor/OrbitControls.js", "vendor/STLLoader.js", "vendor/RoomEnvironment.js"):
        assert (WEB_DIR / f).is_file(), f
    html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
    assert './vendor/three.module.js' in html and 'src="app.js"' in html


def test_trajectory_payload_shapes():
    g = PhysicsMutator(seed=0).baseline()
    run = run_universe(g, SimulationConfig(n_particles=24, steps=200, seed=0))
    p = trajectory_payload(run.particles, "baseline", g.describe(), max_frames=40)
    assert p["n"] == 24 and 2 <= len(p["frames"]) <= 40
    assert len(p["frames"]) == len(p["times"]) == len(p["energy_drift"]) == len(p["radius_of_gyration"])
    assert all(len(f) == 3 * 24 for f in p["frames"])
    assert p["energy_drift"][0] == 0.0 and max(p["energy_drift"]) < 1e-2
    json.dumps(p)


@pytest.fixture()
def server(tmp_path):
    run = tmp_path / "run"
    (run / "cad").mkdir(parents=True)
    (run / "summary.json").write_text("{}")
    (run / "dashboard_data.json").write_text('{"ok": true}')
    (run / "cad" / "part.stl").write_bytes(b"\0" * 84)
    (run / "figures").mkdir()
    (run / "figures" / "a.png").write_bytes(b"\x89PNG")
    (tmp_path / "secret.txt").write_text("nope")
    srv = make_server(run, port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def _get(url, method="GET"):
    with urllib.request.urlopen(urllib.request.Request(url, method=method), timeout=10) as r:
        return r.status, r.headers.get("Content-Type", ""), r.read()


def test_server_routes_and_types(server):
    status, ctype, body = _get(server + "/")
    assert status == 200 and ctype.startswith("text/html") and b"SHIVA-1" in body
    assert _get(server + "/app.js")[1].startswith("text/javascript")  # ES modules need a JS mime type
    assert _get(server + "/vendor/three.module.js")[1].startswith("text/javascript")
    status, ctype, body = _get(server + "/run/dashboard_data.json")
    assert ctype == "application/json" and json.loads(body) == {"ok": True}
    assert len(_get(server + "/cad/part.stl")[2]) == 84
    assert _get(server + "/fig/a.png")[1] == "image/png"
    assert _get(server + "/app.js", method="HEAD")[2] == b""


@pytest.mark.parametrize("path", ["/run/../secret.txt", "/run/%2e%2e/secret.txt", "/../secret.txt", "/missing.js"])
def test_server_blocks_traversal_and_missing(server, path):
    with pytest.raises(urllib.error.HTTPError) as e:
        _get(server + path)
    assert e.value.code == 404


def test_find_latest_run_prefers_newest(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name in ("old", "new"):
        d = tmp_path / "shiva_output" / name
        d.mkdir(parents=True)
        (d / "summary.json").write_text("{}")
    import os
    os.utime(tmp_path / "shiva_output" / "old" / "summary.json", (1, 1))
    assert find_latest_run().name == "new"


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not installed")
def test_gesture_classifier_js():
    r = subprocess.run(["node", str(ROOT / "tests" / "js" / "gestures.test.mjs")], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
