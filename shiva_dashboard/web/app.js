// SHIVA-1 dashboard :: main application.
// Loads run/dashboard_data.json (written by shiva_core.dashboard_server) and drives
// the 3-D universe and prototype viewers, the charts, the console and the
// optional hand-gesture control (gestures.js, loaded on demand).

import * as THREE from 'three';
import { OrbitControls } from './vendor/OrbitControls.js';
import { STLLoader } from './vendor/STLLoader.js';
import { RoomEnvironment } from './vendor/RoomEnvironment.js';
import { cssVar, esc, fmt, hBarChart, hideTip, legend, lineChart, renderTable } from './charts.js';

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const C = {};
const state = {
  data: null,
  mode: 'best', // best | baseline | both
  playing: true,
  speed: 1,
  phase: 0, // 0..1 through the simulated time span
  scrubbing: false,
  focus: 'universe', // universe | cad
  proto: 0,
  gestures: null,
  tables: {},
  hiddenForces: new Set(),
};
const MODES = ['best', 'baseline', 'both', 'sandbox'];
const MODE_NAME = { best: 'evolved', baseline: 'Newtonian', both: 'side by side', sandbox: 'live sandbox' };
const PLAY_SECONDS = 22; // one pass through the recorded trajectory at 1x

// ===================================================================== 3-D helpers

function makeRenderer(canvas) {
  const r = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: 'high-performance' });
  r.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  return r;
}

/** Shared orbit / dolly used by the mouse-free (keyboard, gesture) controls. */
function orbit(controls, dTheta, dPhi) {
  const cam = controls.object;
  const off = cam.position.clone().sub(controls.target);
  const sph = new THREE.Spherical().setFromVector3(off);
  sph.theta -= dTheta;
  sph.phi = clamp(sph.phi - dPhi, 0.08, Math.PI - 0.08);
  off.setFromSpherical(sph);
  cam.position.copy(controls.target).add(off);
  controls.update();
}

function dolly(controls, factor) {
  const cam = controls.object;
  const off = cam.position.clone().sub(controls.target);
  const len = clamp(off.length() * factor, controls.minDistance, controls.maxDistance);
  cam.position.copy(controls.target).add(off.setLength(len));
  controls.update();
}

function glowMaterial(color, size) {
  return new THREE.ShaderMaterial({
    uniforms: { uColor: { value: new THREE.Color(color) }, uSize: { value: size }, uScale: { value: 600 } },
    vertexShader: /* glsl */ `
      attribute float aBright;
      uniform float uSize;
      uniform float uScale;
      varying float vB;
      void main() {
        vec4 mv = modelViewMatrix * vec4(position, 1.0);
        gl_Position = projectionMatrix * mv;
        gl_PointSize = clamp(uSize * uScale * (1.0 + 0.9 * aBright) / max(-mv.z, 0.05), 1.0, 64.0);
        vB = aBright;
      }`,
    fragmentShader: /* glsl */ `
      uniform vec3 uColor;
      varying float vB;
      void main() {
        float d = length(gl_PointCoord - 0.5);
        if (d > 0.5) discard;
        float core = smoothstep(0.5, 0.0, d);
        vec3 c = mix(uColor, vec3(1.0), clamp(0.3 * vB + 0.7 * pow(core, 10.0), 0.0, 1.0));
        gl_FragColor = vec4(c, core * core);
        #include <colorspace_fragment>
      }`,
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
  });
}

class Viewport {
  constructor(canvas, panel) {
    this.canvas = canvas;
    this.panel = panel;
    this.renderer = makeRenderer(canvas);
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(46, 1, 0.01, 2000);
    this.controls = new OrbitControls(this.camera, canvas);
    Object.assign(this.controls, { enableDamping: true, dampingFactor: 0.08, autoRotate: true, autoRotateSpeed: 0.6 });
    this.lastTouch = -1e9;
    this.visible = true;
    this.controls.addEventListener('start', () => this.touch());
    new ResizeObserver(() => this.resize()).observe(panel);
    new IntersectionObserver((e) => { this.visible = e[0].isIntersecting; }).observe(panel);
    this.resize();
  }

  touch() { this.lastTouch = performance.now(); this.tween = null; }

  resize() {
    const w = Math.max(1, this.panel.clientWidth);
    const h = Math.max(1, this.panel.clientHeight);
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.onResize?.(w, h);
  }

  rotate(dTheta, dPhi) { this.touch(); orbit(this.controls, dTheta, dPhi); }
  zoom(f) { this.touch(); dolly(this.controls, f); }

  flyTo(pos, target = new THREE.Vector3()) {
    this.tween = { pos: pos.clone(), target: target.clone() };
  }

  render(dt) {
    if (this.tween) {
      const k = 1 - Math.exp(-dt * 5);
      this.camera.position.lerp(this.tween.pos, k);
      this.controls.target.lerp(this.tween.target, k);
      if (this.camera.position.distanceTo(this.tween.pos) < 0.01) this.tween = null;
    }
    this.controls.autoRotate = performance.now() - this.lastTouch > 6000;
    this.controls.update();
    if (this.visible) this.renderer.render(this.scene, this.camera);
  }
}

// ===================================================================== live sandbox physics

/**
 * Real-time N-body "sandbox": the run's initial particle cloud evolves under a radial force law
 * whose magnitude |F(r)| was tabulated from the SymPy potential on the Python side
 * (dashboard_data.force_laws), plus a movable heavy "star" driven by the user's hand or mouse.
 * Kick-drift-kick leapfrog with Plummer softening; masses sum to 1 as in the simulator.
 */
class Sandbox {
  constructor(frame0, n, laws, softening = 0.05) {
    this.n = n;
    this.frame0 = Float64Array.from(frame0);
    this.laws = laws; // {key: {name, logr, logF}}
    this.eps2 = softening * softening;
    this.law = Object.keys(laws)[0];
    this.m = 1 / n;
    this.star = { x: 0, y: 0, z: 0, active: false, mass: 0.35, heavy: false };
    this.reset();
  }

  static table(r, F) {
    const logr = [];
    const logF = [];
    r.forEach((x, i) => { if (x > 0 && F[i] > 0) { logr.push(Math.log(x)); logF.push(Math.log(F[i])); } });
    return { logr: Float64Array.from(logr), logF: Float64Array.from(logF) };
  }

  /** |F(r)| between unit masses: log-log interpolation of the SymPy table, power-law extrapolation beyond it. */
  force(r) {
    const { logr, logF } = this.laws[this.law];
    const x = Math.log(r);
    const N = logr.length;
    let i;
    if (x <= logr[0]) i = 0;
    else if (x >= logr[N - 1]) i = N - 2;
    else { let lo = 0; let hi = N - 1; while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (logr[mid] <= x) lo = mid; else hi = mid; } i = lo; }
    const t = (x - logr[i]) / (logr[i + 1] - logr[i]);
    return Math.exp(logF[i] + t * (logF[i + 1] - logF[i]));
  }

  reset() {
    this.pos = Float64Array.from(this.frame0);
    this.vel = new Float64Array(this.n * 3);
    let s = 12345; // deterministic small velocity dispersion (sigma 0.05, as the simulator's cold collapse)
    const rnd = () => { s = (s * 1103515245 + 12345) % 2147483648; return (s + 0.5) / 2147483648; };
    for (let k = 0; k < this.vel.length; k++) this.vel[k] = 0.05 * Math.sqrt(-2 * Math.log(rnd())) * Math.cos(2 * Math.PI * rnd());
    this.acc = new Float64Array(this.n * 3);
    this.t = 0;
    this.accel();
  }

  accel() {
    const { pos, acc, n, m, eps2 } = this;
    acc.fill(0);
    for (let i = 0; i < n; i++) {
      const xi = pos[3 * i]; const yi = pos[3 * i + 1]; const zi = pos[3 * i + 2];
      for (let j = i + 1; j < n; j++) {
        const dx = pos[3 * j] - xi; const dy = pos[3 * j + 1] - yi; const dz = pos[3 * j + 2] - zi;
        const re = Math.sqrt(dx * dx + dy * dy + dz * dz + eps2);
        const f = (m * this.force(re)) / re;
        acc[3 * i] += f * dx; acc[3 * i + 1] += f * dy; acc[3 * i + 2] += f * dz;
        acc[3 * j] -= f * dx; acc[3 * j + 1] -= f * dy; acc[3 * j + 2] -= f * dz;
      }
    }
    const S = this.star;
    if (S.active) {
      const M = S.mass * (S.heavy ? 4 : 1);
      const se2 = 0.15 * 0.15; // the star is softer, so close passes slingshot instead of exploding
      for (let i = 0; i < n; i++) {
        const dx = S.x - pos[3 * i]; const dy = S.y - pos[3 * i + 1]; const dz = S.z - pos[3 * i + 2];
        const re = Math.sqrt(dx * dx + dy * dy + dz * dz + se2);
        const f = (M * this.force(re)) / re;
        acc[3 * i] += f * dx; acc[3 * i + 1] += f * dy; acc[3 * i + 2] += f * dz;
      }
    }
  }

  step(dt) {
    const sub = Math.max(1, Math.ceil(dt / 0.0015));
    const h = dt / sub;
    const { pos, vel, acc } = this;
    const vmax = 25;
    for (let s = 0; s < sub; s++) {
      for (let k = 0; k < pos.length; k++) { vel[k] += 0.5 * h * acc[k]; pos[k] += h * vel[k]; }
      this.accel();
      for (let k = 0; k < pos.length; k += 3) {
        vel[k] += 0.5 * h * acc[k]; vel[k + 1] += 0.5 * h * acc[k + 1]; vel[k + 2] += 0.5 * h * acc[k + 2];
        const v = Math.hypot(vel[k], vel[k + 1], vel[k + 2]);
        if (v > vmax) { const c = vmax / v; vel[k] *= c; vel[k + 1] *= c; vel[k + 2] *= c; }
      }
    }
    this.t += dt;
  }

  radiusOfGyration() {
    const { pos, n } = this;
    let cx = 0; let cy = 0; let cz = 0;
    for (let i = 0; i < n; i++) { cx += pos[3 * i]; cy += pos[3 * i + 1]; cz += pos[3 * i + 2]; }
    cx /= n; cy /= n; cz /= n;
    let s = 0;
    for (let i = 0; i < n; i++) s += (pos[3 * i] - cx) ** 2 + (pos[3 * i + 1] - cy) ** 2 + (pos[3 * i + 2] - cz) ** 2;
    return Math.sqrt(s / n);
  }
}

// ===================================================================== universe view

const TRAIL = 18; // samples per trail
const TRAIL_STEP = 0.4; // frames between trail samples

class UniverseView extends Viewport {
  constructor(canvas, panel, universes, forceLaws) {
    super(canvas, panel);
    this.renderer.setClearColor(0x0b0b0b, 1);
    // normalise by the typical (median) particle radius over the run, so a collapsing
    // cloud still fills the view; both universes share one scale for a fair comparison
    this.scale = 1 / Math.max(1e-6, (UniverseView.typicalRadius(universes.best) + UniverseView.typicalRadius(universes.baseline)) / 2);
    this.offset = 2.5;
    this.controls.minDistance = 0.8;
    this.controls.maxDistance = 80;
    this.u = {
      best: this.build(universes.best, C.s1),
      baseline: this.build(universes.baseline, C.s2),
    };
    this.initSandbox(universes, forceLaws);
    this.backdrop();
    this.onResize = (w, h) => {
      const s = this.renderer.getPixelRatio() * h / (2 * Math.tan((this.camera.fov * Math.PI) / 360));
      for (const m of this.materials) m.uniforms.uScale.value = s;
    };
    this.resize();
    this.setMode('best', true);
  }

  static typicalRadius(u) {
    const med = (a) => { const b = [...a].sort((x, y) => x - y); return b[Math.floor(b.length / 2)] ?? 1; };
    const per = [];
    const step = Math.max(1, Math.floor(u.frames.length / 40));
    for (let i = 0; i < u.frames.length; i += step) {
      const f = u.frames[i];
      const r = [];
      for (let k = 0; k + 2 < f.length; k += 3) r.push(Math.hypot(f[k], f[k + 1], f[k + 2]));
      per.push(med(r));
    }
    return med(per) || 1;
  }

  get materials() {
    const m = [this.u.best.points.material, this.u.baseline.points.material, this.stars.material];
    if (this.live) m.push(this.live.points.material, this.starMesh.material);
    return m;
  }

  /** Points + fading trails + radius-of-gyration ring + floor grid for n particles. */
  makeSet(n, color) {
    const group = new THREE.Group();
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(n * 3), 3));
    g.setAttribute('aBright', new THREE.BufferAttribute(new Float32Array(n), 1));
    const points = new THREE.Points(g, glowMaterial(color, 0.14));
    points.frustumCulled = false;
    const tg = new THREE.BufferGeometry();
    tg.setAttribute('position', new THREE.BufferAttribute(new Float32Array(n * TRAIL * 2 * 3), 3));
    tg.setAttribute('color', new THREE.BufferAttribute(new Float32Array(n * TRAIL * 2 * 3), 3));
    const trails = new THREE.LineSegments(tg, new THREE.LineBasicMaterial({
      vertexColors: true, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    }));
    trails.frustumCulled = false;
    const ringPts = [];
    for (let k = 0; k <= 128; k++) { const a = (k / 128) * Math.PI * 2; ringPts.push(new THREE.Vector3(Math.cos(a), 0, Math.sin(a))); }
    const ring = new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(ringPts),
      new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.35 }));
    const polar = new THREE.PolarGridHelper(2.4, 12, 4, 96, 0x383835, 0x2c2c2a);
    polar.position.y = -2.2;
    polar.material.transparent = true;
    polar.material.opacity = 0.5;
    group.add(points, trails, ring, polar);
    this.scene.add(group);
    return { n, group, points, trails, ring, col: new THREE.Color(color), tmp: new Float32Array((TRAIL + 1) * n * 3) };
  }

  build(u, color) {
    const n = u.n;
    const F = u.frames.length;
    const pos = new Float32Array(F * n * 3);
    u.frames.forEach((fr, i) => { for (let k = 0; k < n * 3; k++) pos[i * n * 3 + k] = fr[k] * this.scale; });
    // typical per-frame displacement, for the speed -> brightness mapping
    const steps = [];
    for (let i = 0; i + 1 < F; i += Math.max(1, Math.floor(F / 30))) {
      for (let p = 0; p < n; p += 2) {
        const a = (i * n + p) * 3;
        const b = a + n * 3;
        steps.push(Math.hypot(pos[b] - pos[a], pos[b + 1] - pos[a + 1], pos[b + 2] - pos[a + 2]));
      }
    }
    steps.sort((a, b) => a - b);
    const vref = Math.max(steps[Math.floor(steps.length / 2)] || 1e-3, 1e-4) / TRAIL_STEP;
    return { ...this.makeSet(n, color), u, F, pos, vref };
  }

  backdrop() {
    const N = 1400;
    const p = new Float32Array(N * 3);
    for (let i = 0; i < N; i++) {
      const v = new THREE.Vector3().randomDirection().multiplyScalar(60 + Math.random() * 60);
      p.set([v.x, v.y, v.z], i * 3);
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(p, 3));
    g.setAttribute('aBright', new THREE.BufferAttribute(new Float32Array(N), 1));
    this.stars = new THREE.Points(g, glowMaterial('#5b5a56', 0.32));
    this.scene.add(this.stars);
  }

  initSandbox(universes, fl) {
    const laws = {};
    const pick = (re) => (fl?.laws || []).find((l) => re.test(l.name));
    const ev = pick(/evolved/i);
    const nw = pick(/newton/i);
    if (ev) laws.evolved = { name: ev.name, ...Sandbox.table(fl.r, ev.absF) };
    if (nw) laws.newtonian = { name: nw.name, ...Sandbox.table(fl.r, nw.absF) };
    if (!Object.keys(laws).length) return;
    const u = universes.best;
    this.sim = new Sandbox(u.frames[0], u.n, laws, fl.softening ?? 0.05);
    this.live = this.makeSet(u.n, C.s1);
    this.live.vref = 0.4;
    this.live.hist = []; // past snapshots (display units), newest first
    this.live.group.visible = false;
    // the user's star: a big glow plus a pulsing halo ring
    const sg = new THREE.BufferGeometry();
    sg.setAttribute('position', new THREE.BufferAttribute(new Float32Array(3), 3));
    sg.setAttribute('aBright', new THREE.BufferAttribute(new Float32Array([1]), 1));
    this.starMesh = new THREE.Points(sg, glowMaterial(C.s4, 0.9));
    this.starMesh.frustumCulled = false;
    const hp = [];
    for (let k = 0; k <= 64; k++) { const a = (k / 64) * Math.PI * 2; hp.push(new THREE.Vector3(Math.cos(a), Math.sin(a), 0)); }
    this.halo = new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(hp), new THREE.LineBasicMaterial({ color: new THREE.Color(C.s4), transparent: true, opacity: 0.6 }));
    this.starGroup = new THREE.Group();
    this.starGroup.add(this.starMesh, this.halo);
    this.starGroup.visible = false;
    this.scene.add(this.starGroup);
    this.raycaster = new THREE.Raycaster();
    this.setLaw(laws.evolved ? 'evolved' : 'newtonian');
    // mouse drives the star when no hand is doing so
    this.canvas.addEventListener('pointermove', (e) => { if (this.mode === 'sandbox' && !this.handStar) this.setStarClient(e.clientX, e.clientY, true); });
    this.canvas.addEventListener('pointerleave', () => { if (!this.handStar) this.setStarActive(false); });
  }

  setLaw(key) {
    if (!this.sim || !this.sim.laws[key]) return;
    this.sim.law = key;
    this.live.col.set(key === 'evolved' ? C.s1 : C.s2);
    this.live.points.material.uniforms.uColor.value.set(key === 'evolved' ? C.s1 : C.s2);
    this.live.ring.material.color.set(key === 'evolved' ? C.s1 : C.s2);
    this.sim.accel();
  }

  resetSandbox() { if (this.sim) { this.sim.reset(); this.live.hist = []; } }

  /** Put the star where the screen point (clientX, clientY) meets the plane through the view centre. */
  setStarClient(cx, cy, active = true) {
    if (!this.sim) return;
    const r = this.canvas.getBoundingClientRect();
    const ndc = new THREE.Vector2(((cx - r.left) / r.width) * 2 - 1, -((cy - r.top) / r.height) * 2 + 1);
    if (Math.abs(ndc.x) > 1 || Math.abs(ndc.y) > 1) { this.setStarActive(false); return; }
    this.raycaster.setFromCamera(ndc, this.camera);
    const n = this.camera.getWorldDirection(new THREE.Vector3());
    const plane = new THREE.Plane().setFromNormalAndCoplanarPoint(n, this.controls.target);
    const p = this.raycaster.ray.intersectPlane(plane, new THREE.Vector3());
    if (!p) return;
    Object.assign(this.sim.star, { x: p.x / this.scale, y: p.y / this.scale, z: p.z / this.scale });
    this.setStarActive(active);
  }

  setStarActive(on) {
    if (!this.sim) return;
    this.sim.star.active = on && this.mode === 'sandbox';
  }

  setMode(mode, instant = false) {
    if (mode === 'sandbox' && !this.sim) mode = 'best';
    this.mode = mode;
    const both = mode === 'both';
    this.u.best.group.visible = mode === 'best' || both;
    this.u.baseline.group.visible = mode === 'baseline' || both;
    if (this.live) this.live.group.visible = mode === 'sandbox';
    if (mode !== 'sandbox') this.setStarActive(false);
    this.u.best.group.position.x = both ? -this.offset : 0;
    this.u.baseline.group.position.x = both ? this.offset : 0;
    const pos = both ? new THREE.Vector3(0, 2.6, 10.8) : mode === 'sandbox' ? new THREE.Vector3(0, 1.2, 6.2) : new THREE.Vector3(0, 1.5, 5.4);
    if (instant) { this.camera.position.copy(pos); this.controls.target.set(0, 0, 0); this.controls.update(); }
    else this.flyTo(pos);
  }

  reset() { this.touch(); this.setMode(this.mode); }

  /** Write points (with speed glow) and fading trails from U.tmp = [now, older, ...] snapshots. */
  draw(U, samples, stepLen) {
    const n = U.n;
    const m = n * 3;
    const T = U.tmp;
    const P = U.points.geometry.attributes.position.array;
    const Bt = U.points.geometry.attributes.aBright.array;
    P.set(T.subarray(0, m));
    for (let p = 0; p < n; p++) {
      const a = p * 3;
      const b = m + a;
      const v = samples > 1 ? Math.hypot(T[a] - T[b], T[a + 1] - T[b + 1], T[a + 2] - T[b + 2]) / stepLen : 0;
      Bt[p] = clamp(v / (3 * U.vref) - 0.15, 0, 1);
    }
    U.points.geometry.attributes.position.needsUpdate = true;
    U.points.geometry.attributes.aBright.needsUpdate = true;
    const TP = U.trails.geometry.attributes.position.array;
    const TC = U.trails.geometry.attributes.color.array;
    const jump = 1.2;
    for (let p = 0; p < n; p++) {
      for (let j = 0; j < TRAIL; j++) {
        const s0 = j * m + p * 3;
        const s1 = (j + 1 < samples ? j + 1 : j) * m + p * 3;
        const o = (p * TRAIL + j) * 6;
        let x1 = T[s1];
        let y1 = T[s1 + 1];
        let z1 = T[s1 + 2];
        if (Math.hypot(x1 - T[s0], y1 - T[s0 + 1], z1 - T[s0 + 2]) > jump) { x1 = T[s0]; y1 = T[s0 + 1]; z1 = T[s0 + 2]; }
        TP[o] = T[s0]; TP[o + 1] = T[s0 + 1]; TP[o + 2] = T[s0 + 2];
        TP[o + 3] = x1; TP[o + 4] = y1; TP[o + 5] = z1;
        const w0 = 0.75 * (1 - j / TRAIL) ** 1.6;
        const w1 = 0.75 * (1 - (j + 1) / TRAIL) ** 1.6;
        TC[o] = U.col.r * w0; TC[o + 1] = U.col.g * w0; TC[o + 2] = U.col.b * w0;
        TC[o + 3] = U.col.r * w1; TC[o + 4] = U.col.g * w1; TC[o + 5] = U.col.b * w1;
      }
    }
    U.trails.geometry.attributes.position.needsUpdate = true;
    U.trails.geometry.attributes.color.needsUpdate = true;
  }

  update(phase, dtSim = 0) {
    for (const key of ['best', 'baseline']) {
      const U = this.u[key];
      if (!U.group.visible) continue;
      const f = phase * (U.F - 1);
      for (let j = 0; j <= TRAIL; j++) this.sample(U, Math.max(0, f - j * TRAIL_STEP), U.tmp, j * U.n * 3);
      this.draw(U, TRAIL + 1, TRAIL_STEP);
      U.ring.scale.setScalar(Math.max(this.at(U.u.radius_of_gyration, phase) * this.scale, 1e-3));
    }
    if (this.live && this.mode === 'sandbox') this.updateSandbox(dtSim);
    this.placeLabels();
  }

  updateSandbox(dtSim) {
    const L = this.live;
    const S = this.sim;
    if (dtSim > 0) S.step(dtSim);
    const m = L.n * 3;
    const cur = new Float32Array(m);
    for (let k = 0; k < m; k++) cur[k] = S.pos[k] * this.scale;
    if (dtSim > 0 && (!L.hist.length || (L.tick = (L.tick || 0) + 1) % 2 === 0)) {
      L.hist.unshift(cur);
      if (L.hist.length > TRAIL) L.hist.pop();
    }
    L.tmp.set(cur, 0);
    const samples = Math.min(TRAIL + 1, L.hist.length + 1);
    for (let j = 1; j <= TRAIL; j++) L.tmp.set(L.hist[Math.min(j - 1, L.hist.length - 1)] || cur, j * m);
    // brightness reference: running median-ish speed of the cloud
    if (samples > 1) {
      let s = 0;
      for (let p = 0; p < L.n; p += 4) { const a = p * 3; s += Math.hypot(L.tmp[a] - L.tmp[m + a], L.tmp[a + 1] - L.tmp[m + a + 1], L.tmp[a + 2] - L.tmp[m + a + 2]); }
      L.vref = 0.95 * L.vref + 0.05 * Math.max(1e-4, s / Math.ceil(L.n / 4));
    }
    this.draw(L, samples, 1);
    L.ring.scale.setScalar(Math.max(S.radiusOfGyration() * this.scale, 1e-3));
    const st = S.star;
    this.starGroup.visible = st.active;
    if (st.active) {
      this.starGroup.position.set(st.x * this.scale, st.y * this.scale, st.z * this.scale);
      this.halo.quaternion.copy(this.camera.quaternion);
      const pulse = 1 + 0.12 * Math.sin(performance.now() / 160);
      this.halo.scale.setScalar((st.heavy ? 0.55 : 0.3) * pulse);
      this.starMesh.material.uniforms.uSize.value = st.heavy ? 1.6 : 0.9;
    }
  }

  sample(U, f, out, off = 0) {
    const i0 = clamp(Math.floor(f), 0, U.F - 1);
    const i1 = Math.min(i0 + 1, U.F - 1);
    const a = clamp(f - i0, 0, 1);
    const m = U.n * 3;
    const A = i0 * m;
    const B = i1 * m;
    for (let k = 0; k < m; k++) out[off + k] = U.pos[A + k] * (1 - a) + U.pos[B + k] * a;
  }

  /** Linear interpolation of a per-frame series at the given phase. */
  at(series, phase) {
    const f = phase * (series.length - 1);
    const i = Math.floor(f);
    const a = f - i;
    const v0 = series[i];
    const v1 = series[Math.min(i + 1, series.length - 1)];
    return v0 == null ? v1 : v1 == null ? v0 : v0 * (1 - a) + v1 * a;
  }

  placeLabels() {
    const both = this.mode === 'both';
    for (const [key, el] of [['best', $('#lbl-best')], ['baseline', $('#lbl-baseline')]]) {
      if (!both) { el.style.display = 'none'; continue; }
      const v = new THREE.Vector3(this.u[key].group.position.x, 2.3, 0).project(this.camera);
      const visible = v.z < 1 && Math.abs(v.x) < 1.1 && Math.abs(v.y) < 1.1;
      el.style.display = visible ? 'block' : 'none';
      el.style.left = `${((v.x + 1) / 2) * this.panel.clientWidth}px`;
      el.style.top = `${((1 - v.y) / 2) * this.panel.clientHeight}px`;
    }
  }
}

// ===================================================================== CAD view

class CadView extends Viewport {
  constructor(canvas, panel) {
    super(canvas, panel);
    this.renderer.setClearColor(0x121211, 1);
    this.renderer.localClippingEnabled = true;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    const pmrem = new THREE.PMREMGenerator(this.renderer);
    this.scene.environment = pmrem.fromScene(new RoomEnvironment(this.renderer), 0.04).texture;
    this.controls.autoRotateSpeed = 1.4;
    this.controls.minDistance = 1.2;
    this.controls.maxDistance = 30;
    this.scene.add(new THREE.HemisphereLight(0xdfe8ff, 0x101012, 0.6));
    const key = new THREE.DirectionalLight(0xffffff, 1.6);
    key.position.set(3, 5, 4);
    const rim = new THREE.DirectionalLight(new THREE.Color(C.s1), 2.4);
    rim.position.set(-4, 2, -5);
    this.scene.add(key, rim);
    const grid = new THREE.GridHelper(10, 40, 0x383835, 0x2c2c2a);
    grid.position.y = -1.5;
    this.scene.add(grid);
    this.clip = new THREE.Plane(new THREE.Vector3(0, -1, 0), 10);
    this.material = new THREE.MeshStandardMaterial({ color: 0xb9c2cf, metalness: 0.7, roughness: 0.32, clippingPlanes: [this.clip] });
    this.edgeMaterial = new THREE.LineBasicMaterial({ color: new THREE.Color(C.s1), transparent: true, opacity: 0.45, clippingPlanes: [this.clip] });
    this.scanMaterial = new THREE.MeshBasicMaterial({ color: new THREE.Color(C.s1), transparent: true, opacity: 0.25, side: THREE.DoubleSide, depthWrite: false, blending: THREE.AdditiveBlending });
    this.scan = new THREE.Mesh(new THREE.RingGeometry(1.7, 1.78, 96), this.scanMaterial);
    this.scan.rotation.x = -Math.PI / 2;
    this.scan.visible = false;
    this.scene.add(this.scan);
    this.group = new THREE.Group();
    this.scene.add(this.group);
    this.cache = new Map();
    this.loader = new STLLoader();
    this.token = 0;
    this.camera.position.set(2.6, 1.8, 3.4);
  }

  async load(url) {
    if (!this.cache.has(url)) {
      this.cache.set(url, (async () => {
        const res = await fetch(url);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const geo = this.loader.parse(await res.arrayBuffer());
        geo.rotateX(-Math.PI / 2); // CAD files are z-up
        geo.computeBoundingBox();
        const bb = geo.boundingBox;
        const size = bb.getSize(new THREE.Vector3());
        const c = bb.getCenter(new THREE.Vector3());
        geo.translate(-c.x, -c.y, -c.z);
        geo.scale(2.8 / Math.max(size.x, size.y, size.z, 1e-9), 2.8 / Math.max(size.x, size.y, size.z, 1e-9), 2.8 / Math.max(size.x, size.y, size.z, 1e-9));
        geo.computeBoundingBox();
        const edges = new THREE.EdgesGeometry(geo, 28);
        return { geo, edges };
      })());
    }
    return this.cache.get(url);
  }

  async show(tech) {
    const token = ++this.token;
    this.group.clear();
    $('#cad-empty').hidden = true;
    $('#cad-empty').textContent = 'No CAD model for this design.';
    if (!tech.stl) { $('#cad-empty').hidden = false; return; }
    let m;
    try {
      m = await this.load(tech.stl);
    } catch (err) {
      if (token === this.token) { $('#cad-empty').hidden = false; $('#cad-empty').textContent = `Could not load ${tech.stl} (${err.message}).`; }
      return;
    }
    if (token !== this.token) return;
    const mesh = new THREE.Mesh(m.geo, this.material);
    const lines = new THREE.LineSegments(m.edges, this.edgeMaterial);
    this.group.add(mesh, lines);
    const bb = m.geo.boundingBox;
    this.group.position.y = -1.5 - bb.min.y + 0.02;
    this.build = { t: 0, y0: this.group.position.y + bb.min.y, y1: this.group.position.y + bb.max.y };
    this.scan.visible = true;
    this.touch();
    this.flyTo(new THREE.Vector3(2.6, 1.6, 3.4), new THREE.Vector3(0, this.group.position.y + (bb.min.y + bb.max.y) / 2, 0));
  }

  reset() {
    this.touch();
    const bb = this.group.children[0]?.geometry.boundingBox;
    const cy = bb ? this.group.position.y + (bb.min.y + bb.max.y) / 2 : 0;
    this.flyTo(new THREE.Vector3(2.6, 1.6, 3.4), new THREE.Vector3(0, cy, 0));
  }

  render(dt) {
    if (this.build) {
      // "materialise" the model bottom-up with a scan ring
      this.build.t = Math.min(1, this.build.t + dt / 1.1);
      const e = 1 - (1 - this.build.t) ** 3;
      const y = this.build.y0 + (this.build.y1 - this.build.y0 + 0.05) * e;
      this.clip.constant = y;
      this.scan.position.y = y;
      this.scanMaterial.opacity = 0.35 * (1 - this.build.t);
      if (this.build.t >= 1) { this.build = null; this.clip.constant = 1e3; this.scan.visible = false; }
    }
    super.render(dt);
  }
}

// ===================================================================== formatting helpers

const pct = (v) => (v == null ? '–' : `${Math.round(v * 100)}%`);
const money = (v) => (v == null ? '–' : `$${v >= 1e4 ? fmt(v, 3) : Math.round(v).toLocaleString('en-US')}`);
const kg = (v) => (v == null ? '–' : v >= 1000 ? `${fmt(v / 1000, 3)} t` : `${fmt(v, 3)} kg`);
const pm = (m, s, d = 3) => `${fmt(m, d)} ± ${fmt(s, 2)}`;

const METRIC_UNITS = [['_mm_s2', 'mm/s²'], ['_Wh_kg', 'Wh/kg'], ['_W_kg', 'W/kg'], ['_kg_s', 'kg/s'], ['_m_s', 'm/s'],
  ['_km2', 'km²'], ['_m2', 'm²'], ['_kWh', 'kWh'], ['_MPa', 'MPa'], ['_kPa', 'kPa'], ['_Pa', 'Pa'], ['_mN', 'mN'],
  ['_mm', 'mm'], ['_min', 'min'], ['_kg', 'kg'], ['_hz', 'Hz'], ['_Hz', 'Hz'], ['_N', 'N'], ['_W', 'W'], ['_K', 'K'],
  ['_s', 's'], ['_m', 'm'], ['_h', 'h'], ['_t', 't'], ['_g', 'g']];

function metricLabel(key) {
  if (key.endsWith('_SF')) return `${key.slice(0, -3).replace(/_/g, ' ')} safety factor`;
  for (const [suf, unit] of METRIC_UNITS) {
    if (key.endsWith(suf)) return `${key.slice(0, -suf.length).replace(/_/g, ' ')} (${unit})`;
  }
  return key.replace(/_/g, ' ');
}

function paramValue(v, unit) {
  if (unit === 'm' && Math.abs(v) < 1) return `${fmt(v * 1000)} mm`;
  if (unit === 'm^2') return `${fmt(v)} m²`;
  if (unit === '1/min') return `${fmt(v)} rpm`;
  if (unit === 'deg') return `${fmt(v)}°`;
  if (unit === 'Pa' && Math.abs(v) >= 1e5) return `${fmt(v / 1e6)} MPa`;
  return unit ? `${fmt(v)} ${unit}` : fmt(v);
}

const gravityLine = (desc) => (desc || '').split('\n').map((s) => s.trim()).find((s) => s.startsWith('gravity')) || '';

// ===================================================================== panels

function renderHeader(d) {
  const m = d.meta;
  const evals = d.search.evaluations;
  $('#run-meta').textContent = `seed ${m.seed} · ${m.profile} profile${evals ? ` · ${evals} universes simulated` : ''} · ${m.generated}`;
  document.title = `SHIVA-1 · seed ${m.seed}`;
}

function countUp(el, to, format, ms = 1100) {
  const t0 = performance.now();
  const step = (now) => {
    const k = Math.min(1, (now - t0) / ms);
    el.textContent = format(to * (1 - (1 - k) ** 3));
    if (k < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

function renderTiles(d) {
  const e = d.emergence;
  const tech = d.technology;
  const feasible = tech.filter((t) => t.feasible).length;
  const lb = d.universes.best.lyapunov;
  const ln = d.universes.baseline.lyapunov;
  const p = e.welch && e.welch.p;
  const tiles = [
    { label: 'Emergence richness', key: C.s1, value: e.best_stats.richness_mean, f: (v) => v.toFixed(3),
      delta: `Newtonian ${pm(e.baseline_stats.richness_mean, e.baseline_stats.richness_std)}${p != null ? ` · Welch p = ${fmt(p, 2)}` : ''}` },
    { label: 'Chaos (Lyapunov λ)', key: C.s1, value: lb, f: (v) => v.toFixed(2),
      delta: ln != null ? `Newtonian λ = ${fmt(ln, 3)}` : 'largest Lyapunov exponent' },
    { label: 'Universes simulated', value: d.search.evaluations || 0, f: (v) => String(Math.round(v)),
      delta: `${d.search.history.length} generations · ${d.search.validation.length} re-tested on new seeds` },
    { label: 'Prototypes feasible', value: feasible, f: (v) => `${Math.round(v)} / ${tech.length}`,
      delta: 'after mapping to real materials' },
  ];
  const root = $('#tiles');
  root.innerHTML = '';
  for (const t of tiles) {
    const el = document.createElement('div');
    el.className = 'tile';
    el.innerHTML = `<div class="label">${t.key ? `<span class="key" style="background:${t.key}"></span>` : ''}${esc(t.label)}</div>` +
      `<div class="value">–</div><div class="delta">${esc(t.delta)}</div>`;
    root.appendChild(el);
    if (t.value != null && Number.isFinite(t.value)) countUp($('.value', el), t.value, t.f);
  }
}

// -------------------------------------------------------------- charts
function forceColor(name, i) {
  if (/evolved/i.test(name)) return C.s1;
  if (/newton/i.test(name)) return C.s2;
  return [C.s3, C.s4, C.s5, C.s7, C.s8, C.s6][i % 6];
}

function chartSpecs(d) {
  const ph = d.emergence.phenomena;
  const h = d.search.history;
  const fl = d.force_laws;
  let other = 0;
  const forceSeries = (fl.laws || []).map((l) => ({
    name: l.name, values: l.absF, color: /evolved|newton/i.test(l.name) ? forceColor(l.name) : forceColor(l.name, other++),
  }));
  return {
    emergence: {
      legend: [{ name: 'Evolved', color: C.s1, box: true }, { name: 'Newtonian', color: C.s2, box: true }],
      chart: (el) => hBarChart(el, {
        categories: ph.map((p) => ({ label: p.label, info: p.info })),
        series: [{ name: 'Evolved', color: C.s1, values: ph.map((p) => p.best) },
          { name: 'Newtonian', color: C.s2, values: ph.map((p) => p.baseline) }],
      }),
      table: (el) => renderTable(el, [{ label: 'phenomenon' }, { label: 'evolved', num: true }, { label: 'Newtonian', num: true }],
        ph.map((p) => [p.label, fmt(p.best, 3), fmt(p.baseline, 3)])),
    },
    search: {
      legend: [{ name: 'best fitness', color: C.s1 }, { name: 'mean fitness', color: C.s3 }, { name: 'best richness', color: C.s4 }],
      chart: (el) => lineChart(el, {
        x: h.map((r) => r.generation),
        series: [{ name: 'best fitness', color: C.s1, values: h.map((r) => r.best_fitness) },
          { name: 'mean fitness', color: C.s3, values: h.map((r) => r.mean_fitness) },
          { name: 'best richness', color: C.s4, values: h.map((r) => r.best_richness) }],
        xLabel: 'generation', yLabel: 'score (0–1)', integerX: true, xFormat: (v) => String(v), yFormat: (v) => fmt(v, 3), height: 250,
      }),
      table: (el) => renderTable(el, [{ label: 'gen', num: true }, { label: 'best fitness', num: true }, { label: 'mean fitness', num: true },
        { label: 'best richness', num: true }, { label: 'success rate', num: true }],
      h.map((r) => [r.generation, fmt(r.best_fitness), fmt(r.mean_fitness), fmt(r.best_richness), pct(r.success_rate)])),
    },
    forces: {
      legend: forceSeries.map((s) => ({ name: s.name, color: s.color })),
      toggle: true,
      chart: (el) => lineChart(el, {
        x: fl.r, series: forceSeries, logX: true, logY: true, hidden: state.hiddenForces,
        xLabel: 'r', yLabel: '|F(r)|  (log–log)', xFormat: (v) => fmt(v, 3), yFormat: (v) => fmt(v, 3), height: 250,
      }),
      table: (el) => {
        const pick = [0.1, 0.3, 1, 3, 10].map((t) => fl.r.reduce((b, v, i) => (Math.abs(Math.log(v / t)) < Math.abs(Math.log(fl.r[b] / t)) ? i : b), 0));
        renderTable(el, [{ label: 'law' }, ...pick.map((i) => ({ label: `r = ${fmt(fl.r[i], 2)}`, num: true }))],
          forceSeries.map((s) => [s.name, ...pick.map((i) => fmt(s.values[i], 3))]));
      },
    },
  };
}

function renderCharts(d) {
  const specs = chartSpecs(d);
  for (const [key, spec] of Object.entries(specs)) {
    const el = $(`#chart-${key}`);
    const lg = $(`#legend-${key}`);
    const btn = $(`[data-table="${key}"]`);
    const draw = () => {
      const table = !!state.tables[key];
      btn.textContent = table ? 'Chart' : 'Table';
      btn.setAttribute('aria-pressed', String(table));
      if (el._ro) { el._ro.disconnect(); el._ro = null; }
      el.innerHTML = '';
      lg.hidden = table;
      if (table) { spec.table(el); return; }
      legend(lg, spec.legend, spec.toggle ? {
        hidden: state.hiddenForces,
        onToggle: (i) => {
          if (state.hiddenForces.has(i)) state.hiddenForces.delete(i);
          else if (state.hiddenForces.size < spec.legend.length - 1) state.hiddenForces.add(i);
          draw();
        },
      } : {});
      spec.chart(el);
    };
    btn.onclick = () => { state.tables[key] = !state.tables[key]; hideTip(); draw(); };
    draw();
  }
  const v = d.search.validation || [];
  const bestId = (d.search.best_description.match(/([0-9a-f]{12})/) || [])[1];
  const win = v.find((r) => r.genome_id === bestId);
  $('#search-note').textContent = v.length
    ? `${v.length} finalists were re-simulated on held-out seeds${win ? `; the winner (${win.law} gravity) scored richness ${pm(win.richness_mean, win.richness_std)}` : ''}.`
    : '';
}

// -------------------------------------------------------------- noether
function renderNoether(d) {
  const root = $('#noether');
  const rows = d.noether || [];
  const Q = [['energy', 'Energy'], ['momentum', 'Momentum'], ['angular_momentum', 'Angular momentum']];
  let html = '<div class="h">universe</div>' + Q.map(([, l]) => `<div class="h">${l}</div>`).join('') + '<div class="h">match</div>';
  let ok = 0;
  for (const r of rows) {
    html += `<div class="name">${esc(r.name)}</div>`;
    for (const [k] of Q) {
      const c = r[k];
      if (!c) { html += '<div><span class="chip neutral"><span class="ic">–</span>n/a</span></div>'; continue; }
      if (!c.predicted) { // no symmetry, so Noether makes no claim
        html += `<div title="no symmetry: conservation is not guaranteed; measured relative drift ${fmt(c.drift, 2)}">` +
          `<span class="chip neutral"><span class="ic">–</span>no symmetry <span class="v">${fmt(c.drift, 2)}</span></span></div>`;
        continue;
      }
      const good = c.consistent;
      html += `<div title="symmetry predicts conservation; measured relative drift ${fmt(c.drift, 2)}">` +
        `<span class="chip ${good ? 'good' : 'bad'}"><span class="ic">${good ? '✓' : '✕'}</span>${good ? 'conserved' : 'violated'}` +
        ` <span class="v">${fmt(c.drift, 2)}</span></span></div>`;
    }
    const full = (r.consistency ?? 0) >= 0.999;
    if (full) ok += 1;
    html += `<div><span class="chip ${full ? 'good' : 'bad'}"><span class="ic">${full ? '✓' : '✕'}</span>${Math.round((r.consistency ?? 0) * 100)}%</span></div>`;
  }
  root.innerHTML = html;
  $('#noether-summary').textContent = rows.length ? `${ok} / ${rows.length} universes: symmetry prediction matches the simulation` : 'no Noether data in this run';
}

// -------------------------------------------------------------- lab
const OBJECTIVES = ['efficiency', 'stability', 'adaptability', 'mappability', 'cosmic'];

function renderLab(d) {
  $('#lab-context').textContent = d.context ? `designed under ${d.context}` : '';
  const list = $('#proto-list');
  list.innerHTML = '';
  d.technology.forEach((t, i) => {
    const el = document.createElement('div');
    el.className = 'proto';
    el.setAttribute('role', 'option');
    el.tabIndex = 0;
    const ob = t.objectives || {};
    el.innerHTML = `<div class="t"><b>${esc(t.label)}</b>` +
      `<span class="chip ${t.feasible ? 'good' : 'bad'}"><span class="ic">${t.feasible ? '✓' : '✕'}</span>${t.feasible ? 'feasible' : 'infeasible'}</span></div>` +
      `<div class="m">${esc(t.material || '–')} · ${kg(t.mass_kg)} · ${money(t.cost_usd)}</div>` +
      `<div class="objbars">${OBJECTIVES.filter((k) => ob[k] != null).map((k) =>
        `<span>${k}</span><span class="track"><span class="fill" style="width:${clamp(ob[k], 0, 1) * 100}%"></span></span><span class="val">${ob[k].toFixed(2)}</span>`).join('')}</div>`;
    el.addEventListener('click', () => selectProto(i));
    el.addEventListener('keydown', (e) => { if (e.key === 'Enter') selectProto(i); });
    list.appendChild(el);
  });
}

function renderSpecs(t) {
  const units = t.units || {};
  const cad = t.cad || {};
  const bbox = cad.bbox_mm ? cad.bbox_mm.map((v) => fmt(v, 3)).join(' × ') + ' mm' : null;
  const rows = [
    ['Material', t.material || '–'],
    ['Feasible after mapping', t.feasible ? '✓ yes' : '✕ no'],
    ['Efficiency (alien → real)', `${pct(t.efficiency[0])} → ${pct(t.efficiency[1])}`],
    ['Fidelity kept', pct(t.fidelity)],
    ['Reliability (Monte-Carlo)', pct(t.reliability)],
    ['Mass', kg(t.mass_kg)],
    ['Cost estimate', money(t.cost_usd)],
    ['Pareto front size', t.pareto_size ?? '–'],
  ];
  if (bbox) rows.push(['CAD bounding box', `${bbox}${cad.scale ? ` (${cad.scale})` : ''}`]);
  const kv = (rs) => `<table class="kv">${rs.map(([k, v]) => `<tr><td>${esc(k)}</td><td class="num">${esc(v)}</td></tr>`).join('')}</table>`;
  let html = `<div class="panel-head"><div><h2 style="text-transform:capitalize">${esc(t.label)}</h2><div class="sub">real-world specification</div></div></div>`;
  html += kv(rows);
  html += `<h3>Design parameters</h3>${kv(Object.entries(t.params || {}).map(([k, v]) => [k.replace(/_/g, ' '), paramValue(v, units[k])]))}`;
  if (t.prediction?.length) {
    html += `<h3>Predicted performance (mean ± std over tolerances)</h3>${kv(t.prediction.map((p) => [metricLabel(p.metric), pm(p.mean, p.std)]))}`;
  }
  if (t.bom?.length) {
    html += '<h3>Bill of materials</h3><table class="data"><thead><tr><th>item</th><th class="num">qty</th><th class="num">mass</th><th class="num">cost</th></tr></thead><tbody>' +
      t.bom.map((b) => `<tr><td>${esc(b.item)}</td><td class="num">${esc(b.qty)}</td><td class="num">${kg(b.mass_kg)}</td><td class="num">${money(b.cost_usd)}</td></tr>`).join('') +
      '</tbody></table>';
  }
  if (t.build?.length) html += `<h3>Build steps</h3><ol>${t.build.map((s) => `<li>${esc(s)}</li>`).join('')}</ol>`;
  if (t.notes?.length) html += `<h3>Notes</h3><ol>${t.notes.map((s) => `<li>${esc(s)}</li>`).join('')}</ol>`;
  $('#specs').innerHTML = html;
}

function selectProto(i, { announce = false } = {}) {
  const tech = state.data.technology;
  if (!tech.length) return;
  state.proto = (i + tech.length) % tech.length;
  const t = tech[state.proto];
  $$('.proto').forEach((el, k) => { el.classList.toggle('on', k === state.proto); el.setAttribute('aria-selected', String(k === state.proto)); });
  const list = $('#proto-list');
  const item = $$('.proto')[state.proto];
  if (item) { // scroll the list only, never the page
    const top = item.offsetTop - list.offsetTop;
    if (top < list.scrollTop) list.scrollTop = top;
    else if (top + item.offsetHeight > list.scrollTop + list.clientHeight) list.scrollTop = top + item.offsetHeight - list.clientHeight;
  }
  $('#cad-title').textContent = t.label;
  const cad = t.cad || {};
  $('#cad-sub').textContent = [t.material, cad.triangles ? `${Number(cad.triangles).toLocaleString('en-US')} triangles` : null,
    cad.watertight ? 'watertight' : null, cad.scale].filter(Boolean).join(' · ');
  renderSpecs(t);
  C.cad?.show(t);
  if (announce) log(`prototype ${state.proto + 1}/${tech.length}: ${t.label} (${t.material || 'no material'}, ${t.feasible ? 'feasible' : 'infeasible'})`, 'sys');
}

function renderFigures(d) {
  const root = $('#figures');
  root.innerHTML = '';
  for (const name of d.figures || []) {
    const f = document.createElement('figure');
    const cap = name.replace(/\.png$/, '').replace(/_/g, ' ');
    f.innerHTML = `<img loading="lazy" src="fig/${encodeURIComponent(name)}" alt="${esc(cap)}"><figcaption>${esc(cap)}</figcaption>`;
    f.addEventListener('click', () => { $('#lightbox-img').src = `fig/${encodeURIComponent(name)}`; $('#lightbox-img').alt = cap; $('#lightbox').hidden = false; });
    root.appendChild(f);
  }
  if (!root.children.length) root.innerHTML = '<div class="sub">No figures in this run.</div>';
}

// ===================================================================== console

const queue = [];
let typing = false;

function log(text, cls = '', { instant = false } = {}) {
  queue.push({ text, cls, instant });
  if (!typing) drain();
}

async function drain() {
  typing = true;
  const box = $('#console-log');
  while (queue.length) {
    const { text, cls, instant } = queue.shift();
    const ln = document.createElement('div');
    ln.className = `ln ${cls}`;
    box.appendChild(ln);
    if (instant || document.hidden) {
      ln.textContent = text;
    } else {
      ln.classList.add('cursor');
      const chunk = Math.max(2, Math.ceil(text.length / 40));
      for (let i = 0; i < text.length; i += chunk) {
        ln.textContent = text.slice(0, i + chunk);
        box.scrollTop = box.scrollHeight;
        await sleep(queue.length > 3 ? 0 : 12);
      }
      ln.classList.remove('cursor');
    }
    box.scrollTop = box.scrollHeight;
  }
  typing = false;
}

function narrative(d) {
  const e = d.emergence;
  const m = d.meta;
  const nPhys = d.physics.length;
  const verified = d.physics.filter((p) => p.verified).length;
  log(`SHIVA-1 online · run ${m.run_dir} · seed ${m.seed} · ${m.profile} profile`, 'sys');
  log(`physics: ${nPhys} law families derived symbolically; field equations verified for ${verified}/${nPhys}.`);
  if (d.search.evaluations) {
    log(`search: ${d.search.evaluations} universes simulated over ${d.search.history.length} generations; ` +
      `${d.search.validation.length} finalists re-tested on seeds the search never saw.`);
  }
  const gl = gravityLine(d.search.best_description);
  if (gl) log(`best universe → ${gl.replace(/\s+:\s+/, ': ')}`, 'em');
  const p = e.welch && e.welch.p;
  const verdict = p == null ? '' : p < 0.05 ? ' — significant at the 5% level.' : ' — not significant at 5%; treat as suggestive.';
  log(`emergence richness ${pm(e.best_stats.richness_mean, e.best_stats.richness_std)} vs Newtonian ` +
    `${pm(e.baseline_stats.richness_mean, e.baseline_stats.richness_std)}${p != null ? ` (Welch p = ${fmt(p, 2)})` : ''}${verdict}`);
  const gains = e.phenomena.filter((x) => x.best != null && x.baseline != null).map((x) => [x.label, x.best - x.baseline])
    .filter(([, g]) => g > 0.02).sort((a, b) => b[1] - a[1]).slice(0, 3);
  log(gains.length ? `largest gains over Newtonian: ${gains.map(([l, g]) => `${l} +${g.toFixed(2)}`).join(', ')}.`
    : 'no phenomenon scored clearly higher than in the Newtonian universe.');
  const nn = (d.noether || []).length;
  if (nn) log(`Noether check: ${(d.noether || []).filter((r) => (r.consistency ?? 0) >= 0.999).length}/${nn} universes conserve exactly what their symmetries predict.`);
  const t = d.technology;
  log(`technology: ${t.filter((x) => x.feasible).length}/${t.length} designs survive mapping to real materials; ${t.filter((x) => x.stl).length} have CAD models.`);
  log('ready. press G (or the ✋ button) for hand control, S for the live sandbox, or type "help".', 'sys');
}

const HELP = [
  ['help', 'this list'],
  ['evolved · newtonian · both', 'switch the 3-D universe'],
  ['sandbox', 'live simulation you can push around with your hand'],
  ['law evolved | newtonian', 'force law used by the sandbox'],
  ['play · pause · speed <0.5–4>', 'playback'],
  ['show <design>', 'open a prototype, e.g. "show gear"'],
  ['designs', 'list the prototypes'],
  ['explain <phenomenon>', 'e.g. "explain chaos"'],
  ['compare', 'side-by-side universes with statistics'],
  ['laws', 'the physics families and their checks'],
  ['best', 'full description of the evolved universe'],
  ['noether', 'symmetry → conservation summary'],
  ['search', 'search progress and validation'],
  ['hands', 'hand-gesture control on / off'],
  ['write', 'air writing: draw on your camera view with a finger'],
  ['reset · clear', 'reset cameras · clear the console'],
];

function command(raw) {
  const d = state.data;
  const text = raw.trim();
  if (!text) return;
  log(text, 'you', { instant: true });
  const [cmd, ...rest] = text.toLowerCase().split(/\s+/);
  const arg = rest.join(' ');
  const findTech = (q) => d.technology.findIndex((t) => t.domain.includes(q.replace(/\s+/g, '_')) || t.label.includes(q));
  switch (cmd) {
    case 'help': case '?':
      HELP.forEach(([c, h]) => log(`${c.padEnd(30)} ${h}`, 'sys', { instant: true }));
      break;
    case 'evolved': case 'best-universe': setMode('best'); break;
    case 'newtonian': case 'baseline': setMode('baseline'); break;
    case 'both': case 'side': setMode('both'); break;
    case 'sandbox': case 'play-god': case 'live': setMode('sandbox'); break;
    case 'law': {
      if (/newt|base/.test(arg)) setLaw('newtonian'); else if (/evol|best|alien/.test(arg)) setLaw('evolved');
      else log('usage: law evolved | law newtonian (sandbox force law)', 'sys');
      break;
    }
    case 'compare': {
      setMode('both');
      const e = d.emergence;
      log(`evolved ${pm(e.best_stats.richness_mean, e.best_stats.richness_std)} vs Newtonian ${pm(e.baseline_stats.richness_mean, e.baseline_stats.richness_std)} ` +
        `over ${e.best_stats.richness_per_seed?.length ?? '?'} seeds${e.welch?.p != null ? `; Welch t = ${fmt(e.welch.t, 3)}, p = ${fmt(e.welch.p, 2)}` : ''}.`);
      break;
    }
    case 'play': setPlaying(true); break;
    case 'pause': case 'stop': setPlaying(false); break;
    case 'speed': {
      const s = parseFloat(arg);
      if (s > 0 && s <= 8) { state.speed = s; $('#speed').value = String([0.5, 1, 2, 4].reduce((a, b) => (Math.abs(b - s) < Math.abs(a - s) ? b : a))); log(`speed ${s}×`, 'sys'); }
      else log('usage: speed 0.5 | 1 | 2 | 4', 'sys');
      break;
    }
    case 'show': case 'open': {
      const i = arg ? findTech(arg) : -1;
      if (i < 0) { log(`no design matches "${arg}". try: ${d.technology.map((t) => t.label).join(', ')}`, 'sys'); break; }
      selectProto(i, { announce: true });
      setFocus('cad', true);
      break;
    }
    case 'designs': case 'prototypes':
      d.technology.forEach((t, i) => log(`${String(i + 1).padStart(2)}. ${t.label.padEnd(20)} ${t.feasible ? '✓' : '✕'} ${t.material || ''}`, 'sys', { instant: true }));
      break;
    case 'explain': case 'what': {
      const q = arg.replace(/^is\s+/, '');
      const p = d.emergence.phenomena.find((x) => q && (x.key.includes(q.replace(/\s+/g, '_')) || x.label.includes(q)));
      if (!p) { log(`phenomena: ${d.emergence.phenomena.map((x) => x.label).join(', ')}`, 'sys'); break; }
      log(`${p.label}: ${p.info || 'no description'}.`);
      log(`score evolved ${fmt(p.best, 3)} vs Newtonian ${fmt(p.baseline, 3)} (0 = absent, 1 = strongly present).`);
      break;
    }
    case 'laws': case 'physics':
      d.physics.forEach((p) => log(`${p.verified ? '✓' : '✕'} ${p.name} · d_eff ${fmt(p.d_eff, 3)} · apsidal angle ${fmt(p.apsidal, 3)} rad · ${p.bands} stable orbit band${p.bands === 1 ? '' : 's'}`, 'sys', { instant: true }));
      break;
    case 'best': case 'describe':
      (d.search.best_description || d.universes.best.description || '').split('\n').forEach((s) => log(s, 'sys', { instant: true }));
      break;
    case 'noether': {
      const bad = (d.noether || []).filter((r) => (r.consistency ?? 0) < 0.999);
      log($('#noether-summary').textContent + (bad.length ? `; mismatches: ${bad.map((r) => r.name).join(', ')}` : '.'));
      $('#noether').scrollIntoView({ behavior: 'smooth', block: 'center' });
      break;
    }
    case 'search':
      d.search.history.forEach((h) => log(`gen ${h.generation}: best ${fmt(h.best_fitness)} · mean ${fmt(h.mean_fitness)} · success ${pct(h.success_rate)}`, 'sys', { instant: true }));
      log($('#search-note').textContent || 'no validation data.', 'sys');
      break;
    case 'hands': case 'gestures': case 'hand': toggleGestures(); break;
    case 'write': case 'draw': case 'air': openAirDraw(); break;
    case 'reset': C.uni?.reset(); C.cad?.reset(); if (state.mode === 'sandbox') C.uni?.resetSandbox(); log(state.mode === 'sandbox' ? 'sandbox and cameras reset' : 'cameras reset', 'sys'); break;
    case 'clear': $('#console-log').innerHTML = ''; break;
    default: {
      const t = findTech(text.toLowerCase());
      if (t >= 0) { selectProto(t, { announce: true }); setFocus('cad', true); break; }
      log(`unknown command "${cmd}". type "help".`, 'sys');
    }
  }
}

// ===================================================================== interaction

function setMode(mode, { announce = true } = {}) {
  if (mode === 'sandbox' && !C.uni?.sim) mode = 'best';
  state.mode = mode;
  $$('#universe-panel .seg button[data-mode]').forEach((b) => b.classList.toggle('on', b.dataset.mode === mode));
  C.uni?.setMode(mode);
  document.body.classList.toggle('sandbox', mode === 'sandbox');
  const gl = gravityLine(state.data.universes.best.description).replace(/^gravity\s*:\s*/, '');
  const sub = {
    best: `evolved laws · ${gl} gravity`,
    baseline: 'Newtonian inverse-square gravity (reference)',
    both: 'evolved (blue, left) vs Newtonian (orange, right)',
    sandbox: 'live simulation · you are the gold star (hand or mouse) · pinch = heavy',
  };
  $('#universe-sub').textContent = sub[mode];
  if (announce) log(`universe view: ${MODE_NAME[mode]}`, 'sys');
  if (mode === 'sandbox' && announce) {
    log(`sandbox: ${C.uni.sim.n} particles from the run's initial cloud, integrated live in your browser (leapfrog, softening ${C.uni.sim.eps2 ** 0.5}) under a radial force |F(r)| tabulated from the SymPy potential. ` +
      'Newtonian reproduces the recorded collapse; "Evolved" uses only the evolved gravity law (radial profile, softest axis), so it is a simplified version of the full evolved universe.');
  }
}

function setLaw(key) {
  if (!C.uni?.sim) return;
  C.uni.setLaw(key);
  $$('#sandbox-bar [data-law]').forEach((b) => b.classList.toggle('on', b.dataset.law === key));
  log(`sandbox law: ${C.uni.sim.laws[key].name}${key === 'evolved' ? ' — gravity channel only, radial approximation' : ''}`, 'sys');
}

function setPlaying(on) {
  state.playing = on;
  $('#btn-play').textContent = on ? '❚❚' : '▶';
  $('#btn-play').setAttribute('aria-label', on ? 'Pause' : 'Play');
}

function setFocus(name, scroll = false) {
  if (state.focus !== name || scroll) {
    state.focus = name;
    $$('.focusable').forEach((p) => p.classList.toggle('focused', p.dataset.focus === name));
  }
  if (scroll) $(`[data-focus="${name}"]`).scrollIntoView({ behavior: 'smooth', block: 'center' });
}

const focusedView = () => (state.focus === 'cad' ? C.cad : C.uni);

function cycleMode() { setMode(MODES[(MODES.indexOf(state.mode) + 1) % MODES.length], { announce: false }); }

function onAction(a) {
  if (a === 'toggle-play') { setPlaying(!state.playing); log(`✊ ${state.playing ? 'play' : 'pause'}`, 'sys'); }
  else if (a === 'cycle-mode') { cycleMode(); log(`✌️ universe view: ${MODE_NAME[state.mode]}`, 'sys'); }
  else if (a === 'toggle-draw') { if (C.air?.open) closeAirDraw(); else openAirDraw(); }
}

// -------------------------------------------------------------- air writing
async function openAirDraw() {
  if (C.air?.open) return;
  closeWelcome();
  if (!state.gestures?.running) await toggleGestures(); // if the camera fails, air writing still works with the mouse
  if (!C.air) {
    const mod = await import('./airdraw.js');
    C.air = new mod.AirDraw({ root: $('#airdraw'), log: (m) => log(`✍️ ${m}`, 'sys'), onExit: closeAirDraw });
  }
  const live = !!state.gestures?.running;
  C.air.show(live ? state.gestures.stream : null);
  document.body.classList.add('air-writing');
  if (live) {
    state.gestures.onDraw = (f) => C.air.frame(f);
    state.gestures.setMode('draw');
    log('✍️ air writing ON — ☝️ index finger = write · ✌️ = pen up · hover a button (or 🤏 pinch it) to pick colour / size / pen / eraser / text · ✊ = rub out · 🤙 or 👍 hold = exit', 'em');
  } else {
    log('✍️ air writing with the mouse (hand control is off): drag to draw, Esc to exit', 'em');
  }
}

function closeAirDraw() {
  if (!C.air?.open) return;
  C.air.hide();
  state.gestures?.setMode('ui');
  document.body.classList.remove('air-writing');
  log('✍️ air writing off', 'sys');
}

// -------------------------------------------------------------- hand cursor
// The hand drives an on-screen cursor: pinch = click, pinch-and-move = drag
// (rotate a 3-D view, move the time slider, scroll a list or the page).

const CLICKABLE = 'button, a, select, input, .proto, figure, .legend .item.toggle, [data-click]';
const hand = { x: 0, y: 0, visible: false, hover: null, drag: null };

function clientXY(x, y) { return [x * innerWidth, y * innerHeight]; }

function hitAt(cx, cy) {
  const el = document.elementFromPoint(cx, cy);
  return { el, clickable: el && el.closest(CLICKABLE), panel: el && el.closest('.focusable'), canvas: el && el.tagName === 'CANVAS' ? el : null };
}

function viewFor(panel) {
  if (!panel) return null;
  return panel.dataset.focus === 'cad' ? C.cad : C.uni;
}

function onCursor(c) {
  const el = $('#hand-cursor');
  hand.visible = c.visible;
  el.hidden = !c.visible;
  document.body.classList.toggle('hand-present', c.visible);
  if (!c.visible) {
    if (hand.hover) { hand.hover.classList.remove('hand-hover'); hand.hover = null; }
    if (C.uni) { C.uni.handStar = false; C.uni.setStarActive(false); }
    return;
  }
  const [cx, cy] = clientXY(c.x, c.y);
  hand.x = cx;
  hand.y = cy;
  el.style.transform = `translate(${cx}px, ${cy}px)`;
  el.dataset.g = c.gesture;
  el.classList.toggle('pinch', !!c.pinching);
  el.style.setProperty('--p', String(c.progress || 0));
  const hit = hitAt(cx, cy);
  const hov = hit.clickable && !hit.clickable.disabled ? hit.clickable : null;
  if (hov !== hand.hover) {
    hand.hover?.classList.remove('hand-hover');
    hov?.classList.add('hand-hover');
    hand.hover = hov;
  }
  if (hit.panel && !hand.drag) setFocus(hit.panel.dataset.focus);
  // sandbox: the hand is the star while it is over the universe
  if (C.uni?.sim) {
    const over = state.mode === 'sandbox' && hit.panel?.dataset.focus === 'universe' && c.gesture !== 'fist';
    C.uni.handStar = over;
    if (over) C.uni.setStarClient(cx, cy, true);
    else C.uni.setStarActive(false);
    C.uni.sim.star.heavy = over && !!c.pinching;
  }
}

function tapAt(x, y) {
  const [cx, cy] = clientXY(x, y);
  const cur = $('#hand-cursor');
  cur.classList.remove('click');
  void cur.offsetWidth; // restart the click ripple
  cur.classList.add('click');
  const hit = hitAt(cx, cy);
  const t = hit.clickable;
  if (!t) {
    const modal = hit.el && hit.el.closest('.modal');
    if (modal) modal.hidden = true;
    return;
  }
  if (t.tagName === 'SELECT') {
    t.selectedIndex = (t.selectedIndex + 1) % t.options.length;
    t.dispatchEvent(new Event('change', { bubbles: true }));
  } else if (t.tagName === 'INPUT' && t.type === 'range') {
    setRange(t, cx);
  } else if (t.tagName === 'INPUT') {
    t.focus();
  } else {
    t.click();
  }
}

function setRange(input, cx) {
  const r = input.getBoundingClientRect();
  const f = clamp((cx - r.left) / r.width, 0, 1);
  input.value = String(Math.round(Number(input.min) + f * (Number(input.max) - Number(input.min))));
  input.dispatchEvent(new Event('input', { bubbles: true }));
}

function scrollableAt(el) {
  for (let e = el; e && e !== document.body; e = e.parentElement) {
    const s = getComputedStyle(e);
    if (/(auto|scroll)/.test(s.overflowY) && e.scrollHeight > e.clientHeight + 2) return e;
  }
  return null;
}

function onDrag(phase, x, y, dx, dy) {
  const [cx, cy] = clientXY(x, y);
  if (phase === 'start') {
    const hit = hitAt(cx, cy);
    if (hit.el && hit.el.id === 'scrub') { hand.drag = { kind: 'range', el: hit.el }; state.scrubbing = true; }
    else if (hit.canvas && hit.panel) {
      hand.drag = state.mode === 'sandbox' && hit.panel.dataset.focus === 'universe' ? { kind: 'star' } : { kind: 'rotate', view: viewFor(hit.panel) };
    } else hand.drag = { kind: 'scroll', el: scrollableAt(hit.el) };
    $('#hand-cursor').classList.add('dragging');
    return;
  }
  const d = hand.drag;
  if (!d) return;
  if (phase === 'end') {
    if (d.kind === 'range') state.scrubbing = false;
    hand.drag = null;
    $('#hand-cursor').classList.remove('dragging');
    return;
  }
  if (d.kind === 'range') setRange(d.el, cx);
  else if (d.kind === 'rotate') d.view?.rotate(dx * 5.5, dy * 4);
  else if (d.kind === 'scroll') {
    const by = -dy * innerHeight * 1.6; // grab and pull, like a touch screen
    if (d.el) d.el.scrollTop += by; else window.scrollBy(0, by);
  }
}

function onPalm(dx, dy, x, y) {
  if (state.mode === 'sandbox' && C.uni?.handStar) return; // the palm is moving the star
  const hit = hitAt(...clientXY(x, y));
  (viewFor(hit.panel) || focusedView())?.rotate(dx * 5, dy * 3.6);
}

function onZoom(f) {
  const hit = hand.visible ? hitAt(hand.x, hand.y) : {};
  (viewFor(hit.panel) || focusedView())?.zoom(f);
}

function gestureHint(g, hint) {
  if (state.mode === 'sandbox') {
    if (g === 'open' || g === 'point') return 'you are the star — move through the cloud';
    if (g === 'pinch') return 'heavy star! (release to lighten)';
  }
  return hint;
}

async function toggleGestures() {
  const btn = $('#btn-gesture');
  const dock = $('#gesture-dock');
  closeWelcome();
  if (state.gestures?.running) {
    closeAirDraw();
    state.gestures.stop();
    dock.hidden = true;
    btn.classList.remove('on');
    document.body.classList.remove('gesture-on');
    log('hand control off', 'sys');
    return;
  }
  dock.hidden = false;
  btn.classList.add('on');
  document.body.classList.add('gesture-on');
  $('#gesture-name').textContent = 'starting…';
  try {
    await ensureController();
    log('hand control: starting camera — click "Allow" if the browser asks…', 'sys');
    await state.gestures.start();
    log('hand control ON — raise a hand: ☝️ move the cursor · 🤏 pinch = click · 🤏 hold + move = drag/rotate/scroll · ✋ rotate · 🙌 spread = zoom · ✊ hold = play/pause · ✌️ hold = next view · 🤙 or 👍 hold = air writing', 'em');
  } catch (err) {
    console.error(err);
    btn.classList.remove('on');
    document.body.classList.remove('gesture-on');
    $('#gesture-name').textContent = 'hand control unavailable';
    $('#gesture-status').textContent = err.message;
    log(err.message, 'sys');
    setTimeout(() => { if (!state.gestures?.running) dock.hidden = true; }, 12000);
  }
}

async function ensureController() {
  if (state.gestures) return state.gestures;
  const mod = await import('./gestures.js');
  C.GESTURES = mod.GESTURES;
  state.gestures = new mod.GestureController({
    video: $('#gesture-video'),
    overlay: $('#gesture-overlay'),
    colors: { line: C.s1, joint: '#ffffff', ring: C.good },
    onCursor,
    onTap: tapAt,
    onDrag,
    onPalm,
    onZoom,
    onAction,
    onState: (g, hint, progress = 0) => {
      const toast = $('#hold-toast');
      const toggling = (g === 'thumbsup' || g === 'pinky') && progress > 0 && progress < 1;
      toast.hidden = !toggling;
      if (toggling) {
        $('b', toast).textContent = g === 'thumbsup' ? '👍' : '🤙';
        $('span', toast).textContent = C.air?.open ? 'keep holding… closing air writing' : 'keep holding… opening air writing';
        $('i', toast).style.width = `${Math.round(progress * 100)}%`;
      }
      const info = C.GESTURES[g] || C.GESTURES.none;
      $('#gesture-name').textContent = g === 'none' ? '🖐️ waiting for a hand' : `${info.icon} ${info.name}`;
      $('#gesture-status').textContent = gestureHint(g, hint);
      $$('#gesture-help li').forEach((li) => li.classList.toggle('on', li.dataset.g === g || (li.dataset.g === 'pinky' && g === 'thumbsup')));
    },
  });
  return state.gestures;
}

function closeWelcome() {
  $('#welcome').hidden = true;
  try { localStorage.setItem('shiva.welcomed', '1'); } catch (e) { /* storage unavailable */ }
}

function bindUI() {
  $$('#universe-panel .seg button[data-mode]').forEach((b) => b.addEventListener('click', () => setMode(b.dataset.mode)));
  $$('#sandbox-bar [data-law]').forEach((b) => b.addEventListener('click', () => setLaw(b.dataset.law)));
  $('#sandbox-reset').addEventListener('click', () => { C.uni?.resetSandbox(); log('sandbox reset to the run\'s initial cloud', 'sys'); });
  $('#btn-play').addEventListener('click', () => setPlaying(!state.playing));
  $('#speed').addEventListener('change', (e) => { state.speed = parseFloat(e.target.value); });
  const scrub = $('#scrub');
  scrub.addEventListener('input', () => { state.phase = scrub.value / 1000; });
  scrub.addEventListener('pointerdown', () => { state.scrubbing = true; });
  window.addEventListener('pointerup', () => { if (!hand.drag) state.scrubbing = false; });
  $('#cad-prev').addEventListener('click', () => selectProto(state.proto - 1));
  $('#cad-next').addEventListener('click', () => selectProto(state.proto + 1));
  $$('.focusable').forEach((p) => p.addEventListener('pointerdown', () => setFocus(p.dataset.focus)));
  $('#btn-gesture').addEventListener('click', toggleGestures);
  $('#btn-write').addEventListener('click', openAirDraw);
  $$('[data-start-write]').forEach((b) => b.addEventListener('click', openAirDraw));
  $$('[data-start-hands]').forEach((b) => b.addEventListener('click', () => { if (!state.gestures?.running) toggleGestures(); else closeWelcome(); }));
  $$('[data-start-sandbox]').forEach((b) => b.addEventListener('click', () => { closeWelcome(); setMode('sandbox'); setFocus('universe', true); }));
  $('#btn-help').addEventListener('click', () => { $('#help-modal').hidden = false; });
  $$('[data-close]').forEach((b) => b.addEventListener('click', () => { b.closest('.modal').hidden = true; if (b.closest('#welcome')) closeWelcome(); }));
  $$('.modal').forEach((m) => m.addEventListener('click', (e) => { if (e.target === m || m.id === 'lightbox') { m.hidden = true; if (m.id === 'welcome') closeWelcome(); } }));
  $('#console-form').addEventListener('submit', (e) => {
    e.preventDefault();
    const inp = $('#console-input');
    command(inp.value);
    inp.value = '';
  });
  window.addEventListener('keydown', (e) => {
    if (C.air?.open && C.air.key(e)) return;
    if (e.key === 'Escape') { $$('.modal').forEach((m) => { m.hidden = true; }); return; }
    if (e.target.closest('input, select, textarea') || e.ctrlKey || e.metaKey || e.altKey) return;
    const k = e.key.toLowerCase();
    if (k === ' ') { e.preventDefault(); setPlaying(!state.playing); }
    else if (k === 'b') setMode(MODES[(MODES.indexOf(state.mode) + 1) % MODES.length]);
    else if (k === 's') setMode('sandbox');
    else if (k === 'f') setFocus(state.focus === 'universe' ? 'cad' : 'universe');
    else if (k === 'arrowright') selectProto(state.proto + 1);
    else if (k === 'arrowleft') selectProto(state.proto - 1);
    else if (k === 'r') { C.uni?.reset(); C.cad?.reset(); }
    else if (k === 'g') toggleGestures();
    else if (k === 'w') openAirDraw();
    else if (k === 'h' || k === '?') $('#help-modal').hidden = !$('#help-modal').hidden;
    else if (k === '/') { e.preventDefault(); $('#console-input').focus(); }
  });
}

// ===================================================================== readouts + loop

function readouts() {
  const U = state.data.universes;
  if (state.mode === 'sandbox' && C.uni?.sim) {
    const S = C.uni.sim;
    $('#ro-time').textContent = fmt(S.t, 3);
    $('#ro-energy').textContent = 'open system';
    $('#ro-radius').textContent = fmt(S.radiusOfGyration(), 3);
    return;
  }
  const keys = state.mode === 'both' ? ['best', 'baseline'] : [state.mode];
  const at = (s) => (C.uni ? C.uni.at(s, state.phase) : s[Math.round(state.phase * (s.length - 1))]);
  const dot = (k) => (keys.length > 1 ? `<span class="key" style="background:${k === 'best' ? C.s1 : C.s2}"></span>` : '');
  $('#ro-time').textContent = fmt(at(U[keys[0]].times), 3);
  $('#ro-energy').innerHTML = keys.map((k) => `${dot(k)}${fmt(at(U[k].energy_drift), 2)}`).join(' ');
  $('#ro-radius').innerHTML = keys.map((k) => `${dot(k)}${fmt(at(U[k].radius_of_gyration), 3)}`).join(' ');
  if (!state.scrubbing) $('#scrub').value = String(Math.round(state.phase * 1000));
}

function loop() {
  let last = performance.now();
  let lastRO = 0;
  const times = state.data.universes.best.times;
  const simPerSecond = (times[times.length - 1] - times[0] || 4) / PLAY_SECONDS;
  const frame = (now) => {
    // rAF timestamps can precede the loop's start: never let dt go negative
    const dt = clamp((now - last) / 1000, 0, 0.1);
    last = Math.max(last, now);
    const live = state.playing && !state.scrubbing;
    if (live && state.mode !== 'sandbox') {
      state.phase += (dt * state.speed) / PLAY_SECONDS;
      if (state.phase > 1) state.phase = 0;
    }
    state.phase = clamp(state.phase, 0, 1);
    if (C.uni) { C.uni.update(state.phase, live ? Math.min(dt, 0.05) * state.speed * simPerSecond : 0); C.uni.render(dt); }
    if (C.cad) C.cad.render(dt);
    if (now - lastRO > 90) { readouts(); lastRO = now; }
    requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);
}

function webglFailed(panel, err) {
  console.error(err);
  const msg = document.createElement('div');
  msg.className = 'cad-empty';
  msg.textContent = '3-D view needs WebGL — turn on hardware acceleration in your browser settings.';
  panel.appendChild(msg);
}

function fatal(msg) {
  const el = $('#fatal');
  el.hidden = false;
  el.innerHTML = `<div><h2>SHIVA-1 dashboard could not start</h2><p>${esc(msg)}</p>` +
    '<p class="sub">Start it with <kbd>python -m shiva_core dashboard</kbd> (add <kbd>--rebuild</kbd> to re-export the data).</p></div>';
}

async function main() {
  for (const k of ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 'good', 'ink', 'muted']) C[k] = cssVar(`--${k}`);
  let d;
  try {
    const res = await fetch('run/dashboard_data.json', { cache: 'no-store' });
    if (!res.ok) throw new Error(`run/dashboard_data.json → HTTP ${res.status}`);
    d = await res.json();
  } catch (err) {
    fatal(location.protocol === 'file:' ? 'This page must be served by the SHIVA-1 server, not opened as a file.' : err.message);
    return;
  }
  state.data = d;
  bindUI();
  renderHeader(d);
  renderTiles(d);
  renderCharts(d);
  renderNoether(d);
  renderLab(d);
  renderFigures(d);
  try { C.uni = new UniverseView($('#universe-canvas'), $('#universe-panel'), d.universes, d.force_laws); } catch (err) { webglFailed($('#universe-panel'), err); }
  try { C.cad = new CadView($('#cad-canvas'), $('#cad-panel')); } catch (err) { webglFailed($('#cad-panel'), err); }
  if (!C.uni?.sim) $$('[data-mode="sandbox"], [data-start-sandbox]').forEach((b) => { b.hidden = true; });
  setMode('best', { announce: false });
  setFocus('universe');
  selectProto(0);
  narrative(d);
  loop();
  let welcomed = false;
  try { welcomed = localStorage.getItem('shiva.welcomed') === '1'; } catch (e) { /* storage unavailable */ }
  if (!welcomed && !/[?&]nowelcome/.test(location.search)) $('#welcome').hidden = false;
  // handy from the dev console, and used by the browser tests to feed synthetic hands
  window.shiva = { state, command, onAction, setMode, setFocus, selectProto, setLaw, ensureController, openAirDraw, closeAirDraw, views: C };
}

main();
