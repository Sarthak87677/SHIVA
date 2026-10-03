// SHIVA-1 dashboard :: air writing.
//
// Write in the air with your index finger: the camera view is shown large and
// mirrored (like a mirror), and the ink is drawn on top of it at the fingertip.
// Strokes are stored as vectors in camera-view coordinates (0..1), so they stay
// on your face/scene when the window is resized, can be undone, and are
// re-rendered exactly by renderStroke() (also used for the PNG export).
//
//   ☝️ index finger only  = pen down (write)       ✌️ / ✋ = pen up (move freely)
//   hover a toolbar button ≈ 0.6 s, or pinch it    = select
//   ✊ fist                 = erase where your fist is
//   🤙 little finger / 👍 thumbs up (hold)         = leave air writing

export const COLORS = [
  ['white', '#ffffff'], ['black', '#111111'], ['red', '#ff3b30'], ['orange', '#ff9500'], ['yellow', '#ffd60a'],
  ['green', '#34c759'], ['cyan', '#32d1ff'], ['blue', '#2f6bff'], ['purple', '#af52de'], ['pink', '#ff4fa3'],
];
export const STYLES = [
  ['pen', '✒️', 'Pen'], ['neon', '💡', 'Neon'], ['brush', '🖌️', 'Brush'],
  ['rainbow', '🌈', 'Rainbow'], ['dotted', '⋯', 'Dotted'], ['spray', '💨', 'Spray'],
];
export const SIZES = [['S', 0.006], ['M', 0.012], ['L', 0.022], ['XL', 0.04]]; // fraction of the view height
export const FONTS = [
  ['sans', 'Clean', 'system-ui, "Segoe UI", Arial, sans-serif'],
  ['serif', 'Classic', 'Georgia, "Times New Roman", serif'],
  ['script', 'Script', '"Segoe Script", "Lucida Handwriting", "Brush Script MT", cursive'],
  ['comic', 'Comic', '"Comic Sans MS", "Comic Neue", "Chalkboard SE", cursive'],
  ['display', 'Bold', 'Impact, "Arial Black", "Helvetica Neue", sans-serif'],
  ['mono', 'Code', 'Consolas, "Cascadia Mono", ui-monospace, monospace'],
];
const FONT_STACK = Object.fromEntries(FONTS.map(([k, , stack]) => [k, stack]));
const ERASER_SCALE = 3; // eraser is wider than the pen
const FIST_ERASER = 0.07; // fist eraser radius, fraction of the view height
const DWELL_MS = 650;
const MIN_STEP = 0.0025; // ignore sub-pixel jitter (fraction of the view height)

// ------------------------------------------------------------------ rendering (pure)

function rng(seed) {
  let s = seed >>> 0 || 1;
  return () => { s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0; return s / 4294967296; };
}

const px = (s, i, W, H) => [s.pts[i][0] * W, s.pts[i][1] * H];
const mid = (a, b) => [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];

/** Draw the piece of stroke `s` from a to b (with optional quadratic control c), ending at point index i. */
function piece(ctx, s, a, c, b, i, W, H) {
  const w = Math.max(1, s.size * H);
  ctx.save();
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  const path = () => { ctx.beginPath(); ctx.moveTo(a[0], a[1]); if (c) ctx.quadraticCurveTo(c[0], c[1], b[0], b[1]); else ctx.lineTo(b[0], b[1]); };
  if (s.tool === 'erase') {
    ctx.globalCompositeOperation = 'destination-out';
    ctx.lineWidth = w;
    path();
    ctx.stroke();
  } else if (s.style === 'neon') {
    ctx.globalCompositeOperation = 'lighter';
    ctx.shadowColor = s.color;
    ctx.shadowBlur = w * 2.2;
    ctx.strokeStyle = s.color;
    ctx.globalAlpha = 0.7;
    ctx.lineWidth = w * 1.3;
    path();
    ctx.stroke();
    ctx.globalCompositeOperation = 'source-over';
    ctx.shadowBlur = 0;
    ctx.globalAlpha = 1;
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = Math.max(1, w * 0.35);
    path();
    ctx.stroke();
  } else if (s.style === 'brush') {
    const p0 = s.pts[Math.max(0, i - 1)];
    const p1 = s.pts[Math.min(i, s.pts.length - 1)];
    ctx.strokeStyle = s.color;
    ctx.lineWidth = w * ((p0[2] ?? 1) + (p1[2] ?? 1)) / 2;
    path();
    ctx.stroke();
  } else if (s.style === 'rainbow') {
    const L = s.pts[Math.min(i, s.pts.length - 1)][3] ?? 0;
    ctx.strokeStyle = `hsl(${(s.hue + L * 700) % 360}, 100%, 60%)`;
    ctx.lineWidth = w;
    path();
    ctx.stroke();
  } else if (s.style === 'dotted') {
    ctx.fillStyle = s.color;
    const gap = w * 2.4;
    const d = Math.hypot(b[0] - a[0], b[1] - a[1]);
    let t = s._carry ?? 0;
    for (; t <= d; t += gap) {
      const f = d ? t / d : 0;
      ctx.beginPath();
      ctx.arc(a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, w / 2, 0, Math.PI * 2);
      ctx.fill();
    }
    s._carry = t - d;
  } else if (s.style === 'spray') {
    const r = rng((s.seed || 1) * 7919 + i);
    ctx.fillStyle = s.color;
    const d = Math.hypot(b[0] - a[0], b[1] - a[1]);
    const n = Math.round(8 + w * 0.8 + d * 0.6);
    for (let k = 0; k < n; k++) {
      const f = r();
      const ang = r() * Math.PI * 2;
      const rad = Math.sqrt(r()) * w * 1.8;
      ctx.globalAlpha = 0.35 + 0.65 * r();
      ctx.fillRect(a[0] + (b[0] - a[0]) * f + Math.cos(ang) * rad, a[1] + (b[1] - a[1]) * f + Math.sin(ang) * rad, Math.max(1, w * 0.12), Math.max(1, w * 0.12));
    }
  } else {
    ctx.strokeStyle = s.color;
    ctx.lineWidth = w;
    path();
    ctx.stroke();
  }
  ctx.restore();
}

/** Incrementally draw the newest part of stroke s after point i was appended (i >= 1). */
export function drawSegment(ctx, s, i, W, H) {
  const p1 = px(s, i - 1, W, H);
  const p2 = px(s, i, W, H);
  const start = i >= 2 ? mid(px(s, i - 2, W, H), p1) : p1;
  piece(ctx, s, start, i >= 2 ? p1 : null, mid(p1, p2), i, W, H);
}

/** Finish a stroke: the last half-segment, or a dot for a single tap. */
export function drawTail(ctx, s, W, H) {
  const n = s.pts.length;
  if (!n) return;
  if (n === 1) {
    const p = px(s, 0, W, H);
    piece(ctx, s, p, null, [p[0] + 0.01, p[1]], 0, W, H);
    return;
  }
  const last = px(s, n - 1, W, H);
  piece(ctx, s, mid(px(s, n - 2, W, H), last), null, last, n - 1, W, H);
}

function drawText(ctx, t, W, H) {
  const fpx = Math.max(10, t.size * H * 6);
  ctx.save();
  ctx.font = `${t.font === 'display' ? '' : '600 '}${fpx}px ${FONT_STACK[t.font] || FONT_STACK.sans}`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  const x = t.x * W;
  const y = t.y * H;
  if (t.style === 'rainbow') {
    const wdt = ctx.measureText(t.text).width;
    const g = ctx.createLinearGradient(x - wdt / 2, y, x + wdt / 2, y);
    for (let k = 0; k <= 6; k++) g.addColorStop(k / 6, `hsl(${(t.hue + k * 60) % 360}, 100%, 60%)`);
    ctx.fillStyle = g;
  } else ctx.fillStyle = t.color;
  if (t.style === 'neon') { ctx.shadowColor = t.color; ctx.shadowBlur = fpx * 0.35; }
  else { ctx.shadowColor = 'rgba(0,0,0,0.55)'; ctx.shadowBlur = fpx * 0.08; }
  ctx.fillText(t.text, x, y);
  if (t.style === 'neon') { ctx.shadowBlur = 0; ctx.fillStyle = 'rgba(255,255,255,0.85)'; ctx.font = ctx.font; ctx.globalAlpha = 0.6; ctx.fillText(t.text, x, y); }
  ctx.restore();
}

/** Render one finished stroke or text item from scratch. */
export function renderStroke(ctx, s, W, H) {
  if (s.tool === 'text') { drawText(ctx, s, W, H); return; }
  s._carry = 0;
  for (let i = 1; i < s.pts.length; i++) drawSegment(ctx, s, i, W, H);
  drawTail(ctx, s, W, H);
}

// ------------------------------------------------------------------ the air-writing studio

export class AirDraw {
  /**
   * @param {object} o
   * @param {HTMLElement} o.root   the #airdraw overlay (built by index.html)
   * @param {(msg:string)=>void} o.log
   * @param {()=>void} o.onExit    called when the user leaves air writing
   */
  constructor(o) {
    this.root = o.root;
    this.log = o.log || (() => {});
    this.onExit = o.onExit || (() => {});
    this.stage = this.root.querySelector('.ad-stage');
    this.video = this.root.querySelector('.ad-video');
    this.ink = this.root.querySelector('.ad-ink');
    this.ctx = this.ink.getContext('2d');
    this.cursor = this.root.querySelector('.ad-cursor');
    this.ghost = this.root.querySelector('.ad-ghost');
    this.status = this.root.querySelector('.ad-status');
    this.textInput = this.root.querySelector('.ad-text-input');
    Object.assign(this, { tool: 'pen', style: 'pen', color: '#32d1ff', size: 1, font: 'script', bg: 'camera' });
    this.strokes = [];
    this.undoStack = [];
    this.redoStack = [];
    this.cur = null;
    this.dwell = null;
    this.open = false;
    this.buildToolbars();
    this.bindMouse();
    new ResizeObserver(() => this.fit()).observe(this.root);
    this.video.addEventListener('loadedmetadata', () => this.fit());
  }

  // ---------------------------------------------------------------- UI
  buildToolbars() {
    const top = this.root.querySelector('.ad-top');
    const bottom = this.root.querySelector('.ad-bottom');
    const fonts = this.root.querySelector('.ad-fonts');
    top.innerHTML = COLORS.map(([n, c]) => `<button data-ad="color" data-v="${c}" title="${n}" style="--c:${c}"><span></span></button>`).join('') +
      '<i class="ad-sep"></i>' +
      SIZES.map(([k, f], i) => `<button data-ad="size" data-v="${i}" title="size ${k}"><span class="ad-dot" style="--d:${Math.round(6 + f * 600)}px"></span><em>${k}</em></button>`).join('');
    bottom.innerHTML = STYLES.map(([k, ic, n]) => `<button data-ad="style" data-v="${k}"><b>${ic}</b><em>${n}</em></button>`).join('') +
      '<i class="ad-sep"></i>' +
      '<button data-ad="tool" data-v="eraser"><b>🧽</b><em>Eraser</em></button>' +
      '<button data-ad="tool" data-v="text"><b>Aa</b><em>Text</em></button>' +
      '<i class="ad-sep"></i>' +
      '<button data-ad="undo"><b>↶</b><em>Undo</em></button>' +
      '<button data-ad="redo"><b>↷</b><em>Redo</em></button>' +
      '<button data-ad="clear"><b>🗑️</b><em>Clear</em></button>' +
      '<button data-ad="bg"><b>🎥</b><em>Camera</em></button>' +
      '<button data-ad="save"><b>💾</b><em>Save</em></button>' +
      '<button data-ad="exit" class="ad-exit"><b>✕</b><em>Exit</em></button>';
    fonts.innerHTML = FONTS.map(([k, n, stack]) => `<button data-ad="font" data-v="${k}" style="font-family:${stack.replace(/"/g, '&quot;')}">${n}</button>`).join('');
    this.root.addEventListener('click', (e) => {
      const b = e.target.closest('[data-ad]');
      if (b && this.root.contains(b)) this.press(b);
    });
    this.textInput.addEventListener('keydown', (e) => { e.stopPropagation(); if (e.key === 'Enter') this.textInput.blur(); });
    this.refresh();
  }

  /** Activate a toolbar button (mouse click, finger dwell or pinch). */
  press(b) {
    const kind = b.dataset.ad;
    const v = b.dataset.v;
    b.classList.remove('ad-flash');
    void b.offsetWidth;
    b.classList.add('ad-flash');
    if (kind === 'color') { this.color = v; if (this.tool === 'eraser') this.tool = 'pen'; }
    else if (kind === 'size') this.size = Number(v);
    else if (kind === 'style') { this.style = v; if (this.tool === 'eraser') this.tool = 'pen'; }
    else if (kind === 'tool') this.tool = this.tool === v ? 'pen' : v;
    else if (kind === 'font') this.font = v;
    else if (kind === 'undo') this.undo();
    else if (kind === 'redo') this.redo();
    else if (kind === 'clear') this.clear();
    else if (kind === 'bg') { this.bg = { camera: 'dark', dark: 'light', light: 'camera' }[this.bg]; }
    else if (kind === 'save') this.save();
    else if (kind === 'exit') { this.onExit(); return; }
    if (kind === 'tool' && this.tool === 'text') {
      if (!this.textInput.value.trim()) this.textInput.value = 'SHIVA';
      this.log('text tool: type your word in the box, then point where it goes and pinch (or click) to place it');
    }
    this.refresh();
  }

  refresh() {
    const on = (kind, v) => this.root.querySelectorAll(`[data-ad="${kind}"]`).forEach((b) => b.classList.toggle('on', b.dataset.v === v));
    on('color', this.color);
    on('size', String(this.size));
    on('style', this.tool === 'eraser' ? '' : this.style);
    on('font', this.font);
    this.root.querySelectorAll('[data-ad="tool"]').forEach((b) => b.classList.toggle('on', b.dataset.v === this.tool));
    this.root.classList.toggle('ad-text-mode', this.tool === 'text');
    this.root.dataset.bg = this.bg;
    const bgBtn = this.root.querySelector('[data-ad="bg"]');
    bgBtn.innerHTML = { camera: '<b>🎥</b><em>Camera</em>', dark: '<b>⬛</b><em>Board</em>', light: '<b>⬜</b><em>Paper</em>' }[this.bg];
    this.root.querySelector('[data-ad="undo"]').disabled = !this.undoStack.length;
    this.root.querySelector('[data-ad="redo"]').disabled = !this.redoStack.length;
    const styleName = STYLES.find((s) => s[0] === this.style)[2];
    const what = this.tool === 'eraser' ? '🧽 eraser' : this.tool === 'text' ? `Aa text · ${FONTS.find((f) => f[0] === this.font)[1]}` : `${styleName} pen`;
    this.status.innerHTML = `<span class="ad-swatch" style="background:${this.color}"></span>${what} · size ${SIZES[this.size][0]}` +
      '<span class="ad-hint">☝️ write · ✌️ pen up · hover or 🤏 pinch a button · ✊ erase · 🤙/👍 hold = exit</span>';
    this.ghost.style.fontFamily = FONT_STACK[this.font];
    this.ghost.style.color = this.color;
  }

  // ---------------------------------------------------------------- open / close / layout
  show(stream) {
    this.open = true;
    this.root.hidden = false;
    this.video.srcObject = stream || null;
    if (stream) this.video.play().catch(() => {});
    if (!stream && this.bg === 'camera') this.bg = 'dark';
    this.refresh();
    this.fit();
  }

  hide() {
    this.endStroke();
    this.open = false;
    this.root.hidden = true;
    this.video.srcObject = null;
  }

  fit() {
    if (!this.open) return;
    const ar = this.video.videoWidth && this.video.videoHeight ? this.video.videoWidth / this.video.videoHeight : 4 / 3;
    const maxW = Math.max(200, this.root.clientWidth - 24);
    const maxH = Math.max(150, this.root.clientHeight - 24);
    const w = Math.min(maxW, maxH * ar);
    const h = w / ar;
    this.stage.style.width = `${w}px`;
    this.stage.style.height = `${h}px`;
    // keep the status line and text panel clear of the toolbars, however they wrap
    this.stage.style.setProperty('--top-h', `${this.root.querySelector('.ad-top').offsetHeight}px`);
    this.stage.style.setProperty('--bottom-h', `${this.root.querySelector('.ad-bottom').offsetHeight}px`);
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const W = Math.round(w * dpr);
    const H = Math.round(h * dpr);
    if (this.ink.width !== W || this.ink.height !== H) {
      this.ink.width = W;
      this.ink.height = H;
      this.repaint();
    }
  }

  repaint() {
    const { ctx } = this;
    ctx.clearRect(0, 0, this.ink.width, this.ink.height);
    for (const s of this.strokes) renderStroke(ctx, s, this.ink.width, this.ink.height);
    if (this.cur) renderStroke(ctx, this.cur, this.ink.width, this.ink.height);
  }

  // ---------------------------------------------------------------- strokes
  snapshot() {
    this.undoStack.push(this.strokes.slice());
    if (this.undoStack.length > 200) this.undoStack.shift();
    this.redoStack = [];
  }

  beginStroke(x, y, erase = false, size = null) {
    this.endStroke();
    this.snapshot();
    const fist = size != null;
    this.cur = erase
      ? { tool: 'erase', size: size ?? SIZES[this.size][1] * ERASER_SCALE, pts: [[x, y, 1, 0]], fist }
      : { tool: 'pen', style: this.style, color: this.color, size: SIZES[this.size][1], pts: [[x, y, 1, 0]],
        hue: Math.floor(Math.random() * 360), seed: 1 + Math.floor(Math.random() * 1e6) };
    this.cur._carry = 0;
  }

  addPoint(x, y) {
    const s = this.cur;
    if (!s) return;
    const last = s.pts[s.pts.length - 1];
    const ar = this.ink.width / Math.max(1, this.ink.height);
    const d = Math.hypot((x - last[0]) * ar, y - last[1]); // in units of the view height
    if (d < MIN_STEP) return;
    // brush: thinner when the finger moves fast, like a calligraphy brush
    const w = s.style === 'brush' ? Math.min(1.7, Math.max(0.35, 1.7 - d * 30)) : 1;
    const smoothW = last[2] + 0.35 * (w - last[2]);
    s.pts.push([x, y, smoothW, last[3] + d]);
    drawSegment(this.ctx, s, s.pts.length - 1, this.ink.width, this.ink.height);
  }

  endStroke() {
    const s = this.cur;
    if (!s) return;
    this.cur = null;
    // the last couple of samples are the finger already changing pose; trim that tail
    if (s.tool === 'pen' && s.pts.length > 8) s.pts.splice(-2, 2);
    this.strokes.push(s);
    this.repaint();
    this.refresh();
  }

  stampText(x, y) {
    const text = this.textInput.value.trim();
    if (!text) { this.log('type a word in the text box first'); return; }
    this.endStroke();
    this.snapshot();
    this.strokes.push({ tool: 'text', text, font: this.font, color: this.color, style: this.style, size: SIZES[this.size][1], x, y,
      hue: Math.floor(Math.random() * 360) });
    this.ghostPause = performance.now() + 1200; // don't double the freshly placed word with its preview
    this.ghost.hidden = true;
    this.repaint();
    this.refresh();
  }

  undo() {
    this.endStroke();
    if (!this.undoStack.length) return;
    this.redoStack.push(this.strokes);
    this.strokes = this.undoStack.pop();
    this.repaint();
  }

  redo() {
    if (!this.redoStack.length) return;
    this.undoStack.push(this.strokes);
    this.strokes = this.redoStack.pop();
    this.repaint();
  }

  clear() {
    this.endStroke();
    if (!this.strokes.length) return;
    this.snapshot();
    this.strokes = [];
    this.repaint();
  }

  /** Download the drawing (over the mirrored camera frame, or the board/paper background) as PNG. */
  save() {
    const W = this.ink.width;
    const H = this.ink.height;
    const c = document.createElement('canvas');
    c.width = W;
    c.height = H;
    const g = c.getContext('2d');
    if (this.bg === 'camera' && this.video.videoWidth) {
      g.save();
      g.translate(W, 0);
      g.scale(-1, 1);
      g.drawImage(this.video, 0, 0, W, H);
      g.restore();
    } else {
      g.fillStyle = this.bg === 'light' ? '#f7f5ef' : '#101418';
      g.fillRect(0, 0, W, H);
    }
    g.drawImage(this.ink, 0, 0);
    const a = document.createElement('a');
    a.download = `shiva-air-writing-${new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19)}.png`;
    a.href = c.toDataURL('image/png');
    document.body.appendChild(a);
    a.click();
    a.remove();
    this.log(`saved ${a.download} (check your Downloads folder)`);
  }

  // ---------------------------------------------------------------- input
  toClient(p) {
    const r = this.stage.getBoundingClientRect();
    return [r.left + p.x * r.width, r.top + p.y * r.height];
  }

  buttonAt(p) {
    const [cx, cy] = this.toClient(p);
    const el = document.elementFromPoint(cx, cy);
    const b = el && el.closest('[data-ad]');
    return b && this.root.contains(b) && !b.disabled ? b : null;
  }

  /** One hand-tracking frame (from GestureController.onDraw). */
  frame(f) {
    if (!this.open) return;
    if (!f.visible) {
      this.endStroke();
      this.cursor.hidden = true;
      this.ghost.hidden = true;
      this.setHover(null);
      return;
    }
    const tip = f.tip;
    const r = this.stage.getBoundingClientRect();
    this.cursor.hidden = false;
    const at = f.gesture === 'fist' ? f.palm : tip;
    this.cursor.style.transform = `translate(${at.x * r.width}px, ${at.y * r.height}px)`;
    this.cursor.dataset.g = f.gesture;
    const btn = this.buttonAt(tip);
    // pinch: click the button under the fingertip, or place text
    if (f.tap) {
      const tb = this.buttonAt(f.tap);
      if (tb) this.press(tb);
      else if (this.tool === 'text') this.stampText(f.tap.x, f.tap.y);
    }
    const writing = f.gesture === 'point' && !btn;
    const erasing = f.gesture === 'fist';
    // ghost preview for the text tool
    this.ghost.hidden = !(this.tool === 'text' && !btn && f.gesture !== 'fist' && performance.now() > (this.ghostPause || 0));
    if (!this.ghost.hidden) {
      this.ghost.textContent = this.textInput.value.trim() || 'type a word';
      this.ghost.style.fontSize = `${SIZES[this.size][1] * r.height * 6}px`;
      this.ghost.style.transform = `translate(${tip.x * r.width}px, ${tip.y * r.height}px) translate(-50%, -50%)`;
    }
    if (erasing) {
      if (!this.cur || this.cur.tool !== 'erase' || !this.cur.fist) this.beginStroke(f.palm.x, f.palm.y, true, FIST_ERASER);
      else this.addPoint(f.palm.x, f.palm.y);
    } else if (writing && this.tool !== 'text') {
      const erase = this.tool === 'eraser';
      if (!this.cur || (this.cur.tool === 'erase') !== erase || this.cur.fist) this.beginStroke(tip.x, tip.y, erase);
      else this.addPoint(tip.x, tip.y);
    } else {
      this.endStroke();
    }
    this.cursor.classList.toggle('ad-down', writing || erasing);
    this.cursor.style.setProperty('--ink', this.tool === 'eraser' || erasing ? '#ffffff' : this.color);
    this.cursor.style.setProperty('--sz', `${Math.max(8, (erasing ? FIST_ERASER * 2 : SIZES[this.size][1] * (this.tool === 'eraser' ? ERASER_SCALE : 1)) * r.height)}px`);
    // dwell: hold the finger on a button to press it
    this.setHover(writing || erasing ? null : btn);
    let prog = 0;
    if (this.dwell && this.dwell.btn === btn && !this.dwell.done) {
      prog = Math.min(1, (performance.now() - this.dwell.t0) / DWELL_MS);
      if (prog >= 1) { this.dwell.done = true; this.press(btn); }
    }
    this.cursor.style.setProperty('--p', String(f.progress > 0 ? f.progress : prog));
  }

  setHover(btn) {
    if (this.dwell && this.dwell.btn === btn) return;
    if (this.dwell) this.dwell.btn.classList.remove('ad-hover');
    this.dwell = btn ? { btn, t0: performance.now(), done: false } : null;
    if (btn) btn.classList.add('ad-hover');
  }

  bindMouse() {
    const pos = (e) => {
      const r = this.stage.getBoundingClientRect();
      return { x: (e.clientX - r.left) / r.width, y: (e.clientY - r.top) / r.height };
    };
    this.ink.addEventListener('pointerdown', (e) => {
      const p = pos(e);
      if (this.tool === 'text') { this.stampText(p.x, p.y); return; }
      this.ink.setPointerCapture(e.pointerId);
      this.mouse = true;
      this.beginStroke(p.x, p.y, this.tool === 'eraser');
    });
    this.ink.addEventListener('pointermove', (e) => {
      const p = pos(e);
      if (this.tool === 'text') {
        const r = this.stage.getBoundingClientRect();
        this.ghost.hidden = false;
        this.ghost.textContent = this.textInput.value.trim() || 'type a word';
        this.ghost.style.fontSize = `${SIZES[this.size][1] * r.height * 6}px`;
        this.ghost.style.transform = `translate(${p.x * r.width}px, ${p.y * r.height}px) translate(-50%, -50%)`;
      }
      if (this.mouse) this.addPoint(p.x, p.y);
    });
    const up = () => { if (this.mouse) { this.mouse = false; this.endStroke(); } };
    this.ink.addEventListener('pointerup', up);
    this.ink.addEventListener('pointercancel', up);
  }

  key(e) {
    if (!this.open) return false;
    if (e.target === this.textInput) return false;
    const k = e.key.toLowerCase();
    if (e.key === 'Escape') { this.onExit(); return true; }
    if ((e.ctrlKey || e.metaKey) && k === 'z') { e.preventDefault(); if (e.shiftKey) this.redo(); else this.undo(); this.refresh(); return true; }
    if ((e.ctrlKey || e.metaKey) && k === 'y') { e.preventDefault(); this.redo(); this.refresh(); return true; }
    if ((e.ctrlKey || e.metaKey) && k === 's') { e.preventDefault(); this.save(); return true; }
    if (k === 'e') { this.tool = this.tool === 'eraser' ? 'pen' : 'eraser'; this.refresh(); return true; }
    if (k === 'w') { this.onExit(); return true; }
    return true; // swallow dashboard shortcuts while writing
  }
}
