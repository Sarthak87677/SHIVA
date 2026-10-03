// SHIVA-1 dashboard :: webcam hand-gesture control.
//
// Hand landmarks come from MediaPipe Tasks Vision (HandLandmarker), which runs
// entirely in the browser (WebAssembly + WebGL); no video leaves the machine.
// The library and the model are downloaded from public CDNs the first time
// hand control is switched on.
//
// classifyHand() is a pure function of the 21 landmarks, so it can be tested
// without a camera. GestureController turns the stream of detected hands into
// a screen cursor plus events:
//   cursor  - hand position mapped to the screen (index knuckles, smoothed)
//   tap     - quick pinch (thumb + index) = click under the cursor
//   drag    - pinch held and moved (start / move / end)
//   palm    - open palm moved (rotate a view, or move the sandbox star)
//   zoom    - two open hands spread/closed, or three fingers moved up/down
//   action  - held poses: fist = toggle-play, two fingers = cycle-mode,
//             little finger up / thumbs up = toggle-draw (air writing)
// In 'draw' mode (air writing) the controller instead reports the index
// fingertip in camera-view coordinates every frame through onDraw().

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
  point: { icon: '☝️', name: 'point', hint: 'move your hand to move the cursor' },
  pinch: { icon: '🤏', name: 'pinch', hint: 'quick pinch = click · hold + move = drag' },
  open: { icon: '✋', name: 'open palm', hint: 'move to rotate the view under the cursor' },
  fist: { icon: '✊', name: 'fist', hint: 'hold: play / pause' },
  two: { icon: '✌️', name: 'two fingers', hint: 'hold: next universe view' },
  three: { icon: '3️⃣', name: 'three fingers', hint: 'move up / down to zoom' },
  twohands: { icon: '🙌', name: 'two hands', hint: 'spread / close to zoom' },
  pinky: { icon: '🤙', name: 'little finger', hint: 'hold: air writing on / off' },
  thumbsup: { icon: '👍', name: 'thumbs up', hint: 'hold: air writing on / off' },
  none: { icon: '🖐️', name: 'hand', hint: 'point to move the cursor, pinch to click' },
};

const HOLD_ACTIONS = { fist: 'toggle-play', two: 'cycle-mode', pinky: 'toggle-draw', thumbsup: 'toggle-draw' };
const DRAW_HINTS = {
  point: 'writing — move your index finger',
  pinch: 'pinch = click a button / place text',
  fist: 'erasing — rub with your fist',
  pinky: 'hold to leave air writing',
  thumbsup: 'hold to leave air writing',
};
const HOLD_MS = 700;
const STABLE_FRAMES = 3;
const PINCH_ON = 0.27; // thumb–index distance / palm size
const PINCH_OFF = 0.42; // hysteresis: release only when clearly apart
const TAP_MS = 450;
const DRAG_START_MS = 320;
const DRAG_START_DIST = 0.025; // fraction of the screen
// usable part of the camera frame -> full screen (hands rarely reach the frame edges)
const REACH = { x0: 0.12, x1: 0.88, y0: 0.1, y1: 0.75 };

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
  // thumbs up: thumb clearly away from the curled fingers and straightened (tip beyond its IP joint)
  const thumbUp = d(THUMB_TIP, MIDDLE_MCP) / palm > 0.85 && thumbOut && d(WRIST, THUMB_TIP) > d(WRIST, 3);
  const reach = d(WRIST, INDEX_TIP) / Math.max(d(WRIST, INDEX_PIP), 1e-6);
  const [i, m, r, p] = extended;
  let gesture = 'none';
  if (pinch < 0.3 && reach > 0.85) gesture = 'pinch'; // thumb and index tips touch, index not curled
  else if (!i && !m && !r && !p) gesture = thumbUp ? 'thumbsup' : 'fist';
  else if (!i && !m && !r && p) gesture = 'pinky';
  else if (i && !m && !r && !p) gesture = 'point';
  else if (i && m && !r && !p) gesture = 'two';
  else if (i && m && r && !p) gesture = 'three';
  else if (i && m && r && p) gesture = 'open';
  const cx = PALM.reduce((s, k) => s + lm[k].x, 0) / PALM.length;
  const cy = PALM.reduce((s, k) => s + lm[k].y, 0) / PALM.length;
  // cursor anchor: the index/middle knuckles barely move when the fingers pinch or curl
  const ax = (lm[INDEX_MCP].x + lm[MIDDLE_MCP].x) / 2;
  const ay = (lm[INDEX_MCP].y + lm[MIDDLE_MCP].y) / 2;
  return { gesture, extended, thumbOut, thumbUp, pinch, reach, palm: { x: cx, y: cy }, anchor: { x: ax, y: ay },
    tip: { x: lm[INDEX_TIP].x, y: lm[INDEX_TIP].y }, size: palm };
}

/** Camera coordinates (unmirrored, 0..1) -> screen fraction (mirrored like a selfie view). */
export function toScreen(p) {
  const clamp01 = (v) => Math.min(1, Math.max(0, v));
  return { x: clamp01((1 - p.x - REACH.x0) / (REACH.x1 - REACH.x0)), y: clamp01((p.y - REACH.y0) / (REACH.y1 - REACH.y0)) };
}

async function createLandmarker() {
  const vision = await import(/* @vite-ignore */ `${VISION_URL}/vision_bundle.mjs`);
  const fileset = await vision.FilesetResolver.forVisionTasks(`${VISION_URL}/wasm`);
  const options = (delegate) => ({
    baseOptions: { modelAssetPath: MODEL_URL, delegate },
    runningMode: 'VIDEO',
    numHands: 2,
    minHandDetectionConfidence: 0.55,
    minHandPresenceConfidence: 0.55,
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
  if (n === 'NotAllowedError' || n === 'SecurityError') return 'Camera permission was denied. Click the camera icon in the address bar, choose "Allow", then press Hand control again.';
  if (n === 'NotFoundError' || n === 'OverconstrainedError') return 'No webcam found.';
  if (n === 'NotReadableError') return 'The webcam is busy in another app (close Zoom / Teams / Camera) and try again.';
  if (err && /import|fetch|Failed to fetch|network|dynamically imported/i.test(String(err.message || err))) {
    return 'Could not download the hand-tracking model. It needs internet the first time.';
  }
  return `Hand control failed: ${err && err.message ? err.message : err}`;
}

const noop = () => {};

export class GestureController {
  /**
   * @param {object} o  video, overlay, colors and callbacks:
   *   onCursor({x, y, visible, gesture, pinching, dragging, progress})  x, y in 0..1 of the screen
   *   onTap(x, y) · onDrag(phase, x, y, dx, dy) · onPalm(dx, dy, x, y) · onZoom(factor < 1 = closer)
   *   onAction('toggle-play' | 'cycle-mode') · onState(gesture, hint, progress)
   */
  constructor(o) {
    Object.assign(this, { onCursor: noop, onTap: noop, onDrag: noop, onPalm: noop, onZoom: noop, onAction: noop, onState: noop,
      onDraw: noop }, o);
    this.mode = 'ui';
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
    this.cursor = null; // smoothed screen position
    this.prevCursor = null;
    this.pinchOn = false;
    this.pinchFrames = 0;
    this.pinchStart = 0;
    this.pinchAt = null;
    this.dragging = false;
    this.two = null; // previous two-hand distance
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
      this.onState('none', 'raise one hand in front of the camera');
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
    this._release(performance.now(), false);
    this.onCursor({ x: 0, y: 0, visible: false, gesture: 'none' });
    const ctx = this.overlay && this.overlay.getContext('2d');
    if (ctx) ctx.clearRect(0, 0, this.overlay.width, this.overlay.height);
  }

  _loop() {
    if (!this.running) return;
    const v = this.video;
    if (v.readyState >= 2 && v.currentTime !== this.lastVideoTime) {
      this.lastVideoTime = v.currentTime;
      const now = performance.now();
      let hands = [];
      try {
        const res = this.landmarker.detectForVideo(v, now);
        hands = (res && res.landmarks) || [];
      } catch (err) {
        console.warn(err);
      }
      this.process(hands, now, v.videoWidth && v.videoHeight ? v.videoWidth / v.videoHeight : 4 / 3);
    }
    this.raf = requestAnimationFrame(() => this._loop());
  }

  /** 'ui' (cursor, clicks, drags) or 'draw' (air writing: fingertip pen). */
  setMode(mode) {
    if (mode === this.mode) return;
    this._release(performance.now(), false);
    this.onCursor({ x: 0, y: 0, visible: false, gesture: 'none' });
    this.mode = mode;
    this.candidate = 'none';
    this.count = 0;
    this.stable = 'none';
    this.fired = true;
    this.toggleLock = true; // the pose that switched modes must be released before it can toggle again
    this.tip = null;
    this.prevTip = null;
    this.cursor = null;
    this.prevCursor = null;
  }

  /** Air-writing frame: report the (smoothed, mirrored) index fingertip, palm, pose and pinch taps. */
  _processDraw(hands, now, aspect) {
    if (!hands.length) {
      this.lost += 1;
      if (this.lost === 3) {
        this.toggleLock = false; // lowering the hand also releases the toggle pose
        this.pinchOn = false;
        this.pinchFrames = 0;
        this.stable = 'none';
        this.candidate = 'none';
        this.onDraw({ visible: false, gesture: 'none' });
        this.onState('none', 'raise one hand in front of the camera', 0);
      }
      this._draw([], 'none', 0);
      return null;
    }
    this.lost = 0;
    const cls = hands.map((lm) => classifyHand(lm, aspect));
    let k = 0;
    if (cls.length > 1 && this.tip) {
      const dd = cls.map((c) => Math.hypot(1 - c.tip.x - this.tip.x, c.tip.y - this.tip.y));
      k = dd[1] < dd[0] ? 1 : 0;
    }
    const h = cls[k];
    const raw = { x: 1 - h.tip.x, y: h.tip.y };
    if (!this.tip) this.tip = raw;
    else {
      // adaptive smoothing: kills jitter when slow, keeps up when writing fast
      const v = Math.hypot(raw.x - this.tip.x, raw.y - this.tip.y);
      const a = Math.min(0.9, Math.max(0.3, v * 18));
      this.tip = { x: this.tip.x + a * (raw.x - this.tip.x), y: this.tip.y + a * (raw.y - this.tip.y) };
    }
    let tap = null;
    if (!this.pinchOn) {
      if (h.pinch < PINCH_ON && h.reach > 0.85) this.pinchFrames += 1;
      else this.pinchFrames = 0;
      if (this.pinchFrames >= 2) { this.pinchOn = true; this.pinchStart = now; this.pinchAt = { ...(this.prevTip || this.tip) }; }
    } else if (h.pinch > PINCH_OFF || h.reach < 0.7) {
      this.pinchOn = false;
      this.pinchFrames = 0;
      if (now - this.pinchStart < TAP_MS) tap = { ...this.pinchAt };
    }
    const pose = this.pinchOn ? 'pinch' : h.gesture;
    if (pose === this.candidate) this.count += 1;
    else { this.candidate = pose; this.count = 1; }
    if (this.count >= 2 && this.stable !== pose) {
      this.stable = pose;
      this.since = now;
      this.fired = false;
    }
    const g = this.stable;
    const progress = this._hold(g, now, (a) => a === 'toggle-draw');
    this.prevTip = { ...this.tip };
    this.onDraw({ visible: true, gesture: g, tip: { ...this.tip }, palm: { x: 1 - h.palm.x, y: h.palm.y }, tap, progress, size: h.size });
    this.onState(g, DRAW_HINTS[g] || 'pen up — hover a button to pick it', progress);
    this._draw(hands, g, progress, k);
    return g;
  }

  /** Progress (0..1) of a held pose; fires its action once when the hold completes. */
  _hold(g, now, allowed = () => true) {
    const action = HOLD_ACTIONS[g];
    if (action === 'toggle-draw' && this.toggleLock) return 0;
    if (action !== 'toggle-draw' && g !== 'none') this.toggleLock = false; // a different, real pose releases it
    if (!action || !allowed(action)) return 0;
    const progress = this.fired ? 1 : Math.min(1, (now - this.since) / HOLD_MS);
    if (progress >= 1 && !this.fired) { this.fired = true; this.onAction(action); }
    return progress;
  }

  /** End a pinch: a short, still pinch is a tap (click); a held one ends its drag. */
  _release(now, allowTap = true) {
    if (!this.pinchOn) return;
    this.pinchOn = false;
    this.pinchFrames = 0;
    if (this.dragging) {
      this.dragging = false;
      const c = this.cursor || this.pinchAt;
      this.onDrag('end', c.x, c.y, 0, 0);
    } else if (allowTap && this.pinchAt && now - this.pinchStart < TAP_MS) {
      this.onTap(this.pinchAt.x, this.pinchAt.y);
    }
  }

  _smooth(target) {
    if (!this.cursor) return { ...target };
    // adaptive smoothing: steady when the hand is still, responsive when it moves fast
    const v = Math.hypot(target.x - this.cursor.x, target.y - this.cursor.y);
    const a = Math.min(0.85, Math.max(0.22, v * 14));
    return { x: this.cursor.x + a * (target.x - this.cursor.x), y: this.cursor.y + a * (target.y - this.cursor.y) };
  }

  /**
   * Feed one camera frame: `hands` is a list of 21-landmark hands (a single hand is also accepted;
   * null or [] = no hand). Exposed for testing with synthetic landmarks.
   */
  process(hands, now, aspect = 4 / 3) {
    if (hands && hands.length === 21 && hands[0] && hands[0].x !== undefined) hands = [hands];
    hands = (hands || []).filter(Boolean);
    if (this.mode === 'draw') return this._processDraw(hands, now, aspect);
    if (!hands.length) {
      this.lost += 1;
      if (this.lost > 4) {
        this.toggleLock = false;
        this._release(now, false);
        this.stable = 'none';
        this.candidate = 'none';
        this.two = null;
        this.prevCursor = null;
        if (this.lost === 5) {
          this.onCursor({ x: this.cursor ? this.cursor.x : 0.5, y: this.cursor ? this.cursor.y : 0.5, visible: false, gesture: 'none' });
          this.onState('none', 'raise one hand in front of the camera', 0);
        }
      }
      this._draw([], 'none', 0);
      return null;
    }
    this.lost = 0;
    const cls = hands.map((lm) => classifyHand(lm, aspect));

    // ---- two hands: spread / close to zoom
    if (cls.length >= 2 && cls[0].gesture === cls[1].gesture && (cls[0].gesture === 'open' || cls[0].gesture === 'pinch')) {
      this._release(now, false);
      const a = toScreen(cls[0].palm);
      const b = toScreen(cls[1].palm);
      const dist = Math.hypot((a.x - b.x) * aspect, a.y - b.y);
      if (this.two && dist > 0.02) {
        const f = this.two / dist;
        if (Math.abs(f - 1) > 0.004) this.onZoom(Math.min(1.15, Math.max(0.87, f)));
      }
      this.two = dist;
      this.cursor = this._smooth({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 });
      this.prevCursor = null;
      this.stable = 'twohands';
      this.onCursor({ ...this.cursor, visible: true, gesture: 'twohands' });
      this.onState('twohands', GESTURES.twohands.hint, 0);
      this._draw(hands, 'twohands', 0);
      return 'twohands';
    }
    this.two = null;

    // ---- one (primary) hand: the one nearest the current cursor
    let k = 0;
    if (cls.length > 1 && this.cursor) {
      const dd = cls.map((c) => { const s = toScreen(c.anchor); return Math.hypot(s.x - this.cursor.x, s.y - this.cursor.y); });
      k = dd[1] < dd[0] ? 1 : 0;
    }
    const h = cls[k];
    this.cursor = this._smooth(toScreen(h.anchor));

    // pinch with hysteresis (fast onset, firm release) so clicks don't flicker
    if (!this.pinchOn) {
      if (h.pinch < PINCH_ON && h.reach > 0.85) this.pinchFrames += 1;
      else this.pinchFrames = 0;
      if (this.pinchFrames >= 2) {
        this.pinchOn = true;
        this.pinchStart = now;
        this.pinchAt = { ...(this.prevCursor || this.cursor) };
        this.dragging = false;
      }
    } else if (h.pinch > PINCH_OFF || h.reach < 0.7) {
      this._release(now);
    }

    let g;
    if (this.pinchOn) {
      g = 'pinch';
      if (this.stable !== 'pinch') { this.stable = 'pinch'; this.since = now; this.fired = false; }
      const moved = Math.hypot(this.cursor.x - this.pinchAt.x, this.cursor.y - this.pinchAt.y);
      if (!this.dragging && (now - this.pinchStart > DRAG_START_MS || moved > DRAG_START_DIST)) {
        this.dragging = true;
        this.onDrag('start', this.pinchAt.x, this.pinchAt.y, 0, 0);
        this.prevCursor = { ...this.pinchAt };
      }
      if (this.dragging && this.prevCursor) {
        const dx = this.cursor.x - this.prevCursor.x;
        const dy = this.cursor.y - this.prevCursor.y;
        if (dx || dy) this.onDrag('move', this.cursor.x, this.cursor.y, dx, dy);
      }
    } else {
      const raw = h.gesture === 'pinch' ? this.candidate : h.gesture; // pinch onset is handled above
      if (raw === this.candidate) this.count += 1;
      else { this.candidate = raw; this.count = 1; }
      if (this.count >= STABLE_FRAMES && this.stable !== raw) {
        this.stable = raw;
        this.since = now;
        this.fired = false;
        this.prevCursor = null;
      }
      g = this.stable;
      if (this.prevCursor) {
        const dx = this.cursor.x - this.prevCursor.x;
        const dy = this.cursor.y - this.prevCursor.y;
        if (g === 'open' && Math.hypot(dx, dy) > 0.0008) this.onPalm(dx, dy, this.cursor.x, this.cursor.y);
        if (g === 'three' && Math.abs(dy) > 0.0008) this.onZoom(Math.exp(dy * 2.6));
      }
    }

    const progress = this._hold(g, now);
    this.prevCursor = { ...this.cursor };
    this.onCursor({ ...this.cursor, visible: true, gesture: g, pinching: this.pinchOn, dragging: this.dragging, progress });
    const info = GESTURES[g] || GESTURES.none;
    this.onState(g, info.hint, progress);
    this._draw(hands, g, progress, k);
    return g;
  }

  _draw(hands, g = 'none', progress = 0, primary = 0) {
    const c = this.overlay;
    if (!c) return;
    const W = (this.video && this.video.videoWidth) || 640;
    const H = (this.video && this.video.videoHeight) || 480;
    if (c.width !== W || c.height !== H) { c.width = W; c.height = H; }
    const ctx = c.getContext('2d');
    ctx.clearRect(0, 0, W, H);
    const X = (p) => (1 - p.x) * W;
    const Y = (p) => p.y * H;
    hands.forEach((lm, i) => {
      const active = g !== 'none' && (i === primary || g === 'twohands');
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
      if (i === primary && progress > 0 && HOLD_ACTIONS[g]) {
        const cx = PALM.reduce((s, q) => s + X(lm[q]), 0) / PALM.length;
        const cy = PALM.reduce((s, q) => s + Y(lm[q]), 0) / PALM.length;
        ctx.lineWidth = 6;
        ctx.strokeStyle = 'rgba(255,255,255,0.18)';
        ctx.beginPath(); ctx.arc(cx, cy, 34, 0, Math.PI * 2); ctx.stroke();
        ctx.strokeStyle = this.colors.ring;
        ctx.beginPath(); ctx.arc(cx, cy, 34, -Math.PI / 2, -Math.PI / 2 + progress * Math.PI * 2); ctx.stroke();
      }
    });
  }
}
