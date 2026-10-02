// SHIVA-1 dashboard :: webcam hand-gesture control.
//
// Hand landmarks come from MediaPipe Tasks Vision (HandLandmarker), which runs
// entirely in the browser (WebAssembly + WebGL); no video leaves the machine.
// The library and the model are downloaded from public CDNs the first time
// hand control is switched on.
//
// classifyHand() is a pure function of the 21 landmarks, so it can be tested
// without a camera; GestureController turns the stream of classified hands
// into rotate / zoom deltas and discrete actions.

export const VISION_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14';
export const MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';

// MediaPipe hand topology (landmark indices)
const WRIST = 0;
const THUMB_TIP = 4;
const INDEX_MCP = 5;
const INDEX_PIP = 6;
const INDEX_TIP = 8;
const MIDDLE_MCP = 9;
const FINGERS = [[8, 6], [12, 10], [16, 14], [20, 18]]; // [tip, pip] for index, middle, ring, pinky
const PALM = [0, 5, 9, 13, 17];
const CONNECTIONS = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [17, 18], [18, 19], [19, 20], [0, 17]];

export const GESTURES = {
  open: { icon: '✋', name: 'open palm', hint: 'move to rotate' },
  pinch: { icon: '🤏', name: 'pinch', hint: 'raise / lower to zoom' },
  fist: { icon: '✊', name: 'fist', hint: 'hold: play / pause' },
  two: { icon: '✌️', name: 'two fingers', hint: 'hold: switch universe' },
  three: { icon: '3️⃣', name: 'three fingers', hint: 'hold: switch focus' },
  point: { icon: '☝️', name: 'point', hint: 'swipe left / right' },
  none: { icon: '·', name: 'hand', hint: 'open palm, pinch, fist, ✌️, 3 or ☝️' },
};

const HOLD_ACTIONS = { fist: 'toggle-play', two: 'cycle-mode', three: 'switch-focus' };
const HOLD_MS = 650;
const SWIPE_WINDOW_MS = 420;
const SWIPE_DIST = 0.16; // fraction of the frame width
const SWIPE_COOLDOWN_MS = 750;
const STABLE_FRAMES = 3;

/**
 * Classify one hand from its 21 normalised landmarks ({x, y, z} in image coordinates).
 * Rotation-invariant: a finger counts as extended when its tip is clearly farther from
 * the wrist than its middle joint. `aspect` = image width / height (x and y are normalised
 * separately).
 */
export function classifyHand(lm, aspect = 4 / 3) {
  const P = (i) => [lm[i].x * aspect, lm[i].y];
  const d = (i, j) => { const a = P(i); const b = P(j); return Math.hypot(a[0] - b[0], a[1] - b[1]); };
  const palm = Math.max(d(WRIST, MIDDLE_MCP), 1e-6);
  const extended = FINGERS.map(([tip, pip]) => d(WRIST, tip) > 1.12 * d(WRIST, pip));
  const pinch = d(THUMB_TIP, INDEX_TIP) / palm;
  const thumbOut = d(THUMB_TIP, INDEX_MCP) / palm > 0.6;
  const indexReach = d(WRIST, INDEX_TIP) / Math.max(d(WRIST, INDEX_PIP), 1e-6);
  const [i, m, r, p] = extended;
  let gesture = 'none';
  if (pinch < 0.3 && indexReach > 0.85) gesture = 'pinch'; // thumb and index tips touch, index not curled
  else if (!i && !m && !r && !p) gesture = 'fist';
  else if (i && !m && !r && !p) gesture = 'point';
  else if (i && m && !r && !p) gesture = 'two';
  else if (i && m && r && !p) gesture = 'three';
  else if (i && m && r && p) gesture = 'open';
  const cx = PALM.reduce((s, k) => s + lm[k].x, 0) / PALM.length;
  const cy = PALM.reduce((s, k) => s + lm[k].y, 0) / PALM.length;
  return { gesture, extended, thumbOut, pinch, palm: { x: cx, y: cy }, tip: { x: lm[INDEX_TIP].x, y: lm[INDEX_TIP].y }, size: palm };
}

async function createLandmarker() {
  const vision = await import(/* @vite-ignore */ `${VISION_URL}/vision_bundle.mjs`);
  const fileset = await vision.FilesetResolver.forVisionTasks(`${VISION_URL}/wasm`);
  const options = (delegate) => ({
    baseOptions: { modelAssetPath: MODEL_URL, delegate },
    runningMode: 'VIDEO',
    numHands: 1,
    minHandDetectionConfidence: 0.6,
    minHandPresenceConfidence: 0.6,
    minTrackingConfidence: 0.5,
  });
  try {
    return await vision.HandLandmarker.createFromOptions(fileset, options('GPU'));
  } catch (err) {
    console.warn('GPU hand tracking unavailable, falling back to CPU:', err);
    return vision.HandLandmarker.createFromOptions(fileset, options('CPU'));
  }
}

function explain(err) {
  const n = err && err.name;
  if (n === 'NotAllowedError' || n === 'SecurityError') return 'Camera permission was denied. Allow the camera in the address bar and try again.';
  if (n === 'NotFoundError' || n === 'OverconstrainedError') return 'No webcam found.';
  if (n === 'NotReadableError') return 'The webcam is busy in another app (close Zoom/Teams/Camera) and try again.';
  if (err && /import|fetch|Failed to fetch|network|dynamically imported/i.test(String(err.message || err))) {
    return 'Could not download the hand-tracking model. It needs internet the first time.';
  }
  return `Hand control failed: ${err && err.message ? err.message : err}`;
}

export class GestureController {
  /**
   * @param {object} o
   * @param {HTMLVideoElement} o.video
   * @param {HTMLCanvasElement} o.overlay
   * @param {(dx:number, dy:number)=>void} o.onRotate  palm displacement (fraction of the frame, mirrored)
   * @param {(factor:number)=>void} o.onZoom          distance multiplier (<1 = closer)
   * @param {(action:string)=>void} o.onAction        toggle-play | cycle-mode | switch-focus | swipe-left | swipe-right
   * @param {(gesture:string, text:string)=>void} o.onState
   */
  constructor(o) {
    Object.assign(this, o);
    this.running = false;
    this.landmarker = null;
    this.colors = o.colors || { line: '#3987e5', joint: '#ffffff', ring: '#0ca30c' };
    this._reset();
  }

  _reset() {
    this.candidate = 'none';
    this.count = 0;
    this.stable = 'none';
    this.since = 0;
    this.fired = false;
    this.prev = null;
    this.sx = null;
    this.sy = null;
    this.track = [];
    this.lastSwipe = 0;
    this.lost = 0;
    this.lastVideoTime = -1;
  }

  async start() {
    if (this.running) return;
    try {
      this.onState('none', 'starting camera…');
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw Object.assign(new Error('This browser cannot access the camera here. Open the dashboard at http://localhost in Chrome, Edge or Firefox.'), { name: 'Unsupported' });
      }
      this.stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' }, audio: false });
      this.video.srcObject = this.stream;
      await this.video.play();
      if (!this.landmarker) {
        this.onState('none', 'loading hand model… (first start downloads ~18 MB)');
        this.landmarker = await createLandmarker();
      }
      this._reset();
      this.running = true;
      this.onState('none', 'show one hand to the camera');
      this._loop();
    } catch (err) {
      this.stop();
      throw Object.assign(new Error(explain(err)), { cause: err });
    }
  }

  stop() {
    this.running = false;
    cancelAnimationFrame(this.raf);
    if (this.stream) this.stream.getTracks().forEach((t) => t.stop());
    this.stream = null;
    if (this.video) this.video.srcObject = null;
    const ctx = this.overlay && this.overlay.getContext('2d');
    if (ctx) ctx.clearRect(0, 0, this.overlay.width, this.overlay.height);
  }

  _loop() {
    if (!this.running) return;
    const v = this.video;
    if (v.readyState >= 2 && v.currentTime !== this.lastVideoTime) {
      this.lastVideoTime = v.currentTime;
      const now = performance.now();
      let lm = null;
      try {
        const res = this.landmarker.detectForVideo(v, now);
        lm = res && res.landmarks && res.landmarks.length ? res.landmarks[0] : null;
      } catch (err) {
        console.warn(err);
      }
      this.process(lm, now, v.videoWidth && v.videoHeight ? v.videoWidth / v.videoHeight : 4 / 3);
    }
    this.raf = requestAnimationFrame(() => this._loop());
  }

  /** Feed one detection (or null). Exposed for testing with synthetic landmarks. */
  process(lm, now, aspect = 4 / 3) {
    if (!lm) {
      this.lost += 1;
      if (this.lost > 5 && this.stable !== 'none') { this.stable = 'none'; this.prev = null; this.sx = null; }
      if (this.lost === 6) this.onState('none', 'show one hand to the camera');
      this._draw(null);
      return null;
    }
    this.lost = 0;
    const h = classifyHand(lm, aspect);
    if (h.gesture === this.candidate) this.count += 1;
    else { this.candidate = h.gesture; this.count = 1; }
    if (this.count >= STABLE_FRAMES && this.stable !== h.gesture) {
      this.stable = h.gesture;
      this.since = now;
      this.fired = false;
      this.prev = null;
      this.track = [];
    }
    // mirrored coordinates: moving the hand to the user's right moves things right
    const px = 1 - h.palm.x;
    const py = h.palm.y;
    const k = 0.5;
    this.sx = this.sx == null ? px : this.sx + k * (px - this.sx);
    this.sy = this.sy == null ? py : this.sy + k * (py - this.sy);
    const g = this.stable;
    if (g === 'open' || g === 'pinch') {
      if (this.prev) {
        const dx = this.sx - this.prev.x;
        const dy = this.sy - this.prev.y;
        if (g === 'open' && Math.hypot(dx, dy) > 0.0015) this.onRotate(dx, dy);
        if (g === 'pinch' && Math.abs(dy) > 0.0015) this.onZoom(Math.exp(dy * 3.2));
      }
      this.prev = { x: this.sx, y: this.sy };
    } else {
      this.prev = null;
    }
    let progress = 0;
    if (HOLD_ACTIONS[g]) {
      progress = this.fired ? 1 : Math.min(1, (now - this.since) / HOLD_MS);
      if (progress >= 1 && !this.fired) { this.fired = true; this.onAction(HOLD_ACTIONS[g]); }
    }
    if (g === 'point') {
      const tx = 1 - h.tip.x;
      this.track.push([now, tx]);
      while (this.track.length && now - this.track[0][0] > SWIPE_WINDOW_MS) this.track.shift();
      const dx = tx - this.track[0][1];
      if (now - this.lastSwipe > SWIPE_COOLDOWN_MS && Math.abs(dx) > SWIPE_DIST) {
        this.lastSwipe = now;
        this.track = [];
        this.onAction(dx > 0 ? 'swipe-right' : 'swipe-left');
      }
    }
    const info = GESTURES[g] || GESTURES.none;
    this.onState(g, info.hint, progress);
    this._draw(lm, g, progress);
    return g;
  }

  _draw(lm, g = 'none', progress = 0) {
    const c = this.overlay;
    if (!c) return;
    const W = (this.video && this.video.videoWidth) || 640;
    const H = (this.video && this.video.videoHeight) || 480;
    if (c.width !== W || c.height !== H) { c.width = W; c.height = H; }
    const ctx = c.getContext('2d');
    ctx.clearRect(0, 0, W, H);
    if (!lm) return;
    const X = (p) => (1 - p.x) * W;
    const Y = (p) => p.y * H;
    const active = g !== 'none';
    ctx.lineWidth = 3;
    ctx.strokeStyle = active ? this.colors.line : 'rgba(255,255,255,0.45)';
    ctx.shadowColor = this.colors.line;
    ctx.shadowBlur = active ? 12 : 0;
    ctx.beginPath();
    for (const [a, b] of CONNECTIONS) { ctx.moveTo(X(lm[a]), Y(lm[a])); ctx.lineTo(X(lm[b]), Y(lm[b])); }
    ctx.stroke();
    ctx.shadowBlur = 0;
    ctx.fillStyle = this.colors.joint;
    for (const p of lm) { ctx.beginPath(); ctx.arc(X(p), Y(p), 3.5, 0, Math.PI * 2); ctx.fill(); }
    if (progress > 0 && HOLD_ACTIONS[g]) {
      const cx = PALM.reduce((s, k) => s + X(lm[k]), 0) / PALM.length;
      const cy = PALM.reduce((s, k) => s + Y(lm[k]), 0) / PALM.length;
      ctx.lineWidth = 6;
      ctx.strokeStyle = 'rgba(255,255,255,0.18)';
      ctx.beginPath(); ctx.arc(cx, cy, 34, 0, Math.PI * 2); ctx.stroke();
      ctx.strokeStyle = this.colors.ring;
      ctx.beginPath(); ctx.arc(cx, cy, 34, -Math.PI / 2, -Math.PI / 2 + progress * Math.PI * 2); ctx.stroke();
    }
  }
}
