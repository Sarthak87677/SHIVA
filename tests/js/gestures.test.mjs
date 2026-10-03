// Node test for the dashboard's hand-gesture classifier and controller
// (run by tests/test_dashboard.py when Node.js is available).
import assert from 'node:assert/strict';
import { classifyHand, GestureController } from '../../shiva_dashboard/web/gestures.js';

// Synthetic hand in palm-size units: wrist at the origin, fingers pointing up (image y grows down).
function hand({ fingers = ['ext', 'ext', 'ext', 'ext'], thumb = 'out', angle = 0, cx = 0.5, cy = 0.6, s = 0.12 }) {
  const pts = new Array(21);
  const put = (i, x, y) => { pts[i] = [x, y]; };
  put(0, 0, 0);
  const bases = [[-0.35, -0.95], [-0.1, -1.0], [0.13, -0.95], [0.33, -0.85]];
  fingers.forEach((st, f) => {
    const [bx, by] = bases[f];
    const mcp = 5 + 4 * f;
    put(mcp, bx, by);
    if (st === 'ext') { put(mcp + 1, bx, by - 0.45); put(mcp + 2, bx, by - 0.75); put(mcp + 3, bx, by - 1.0); }
    else if (st === 'curl') { put(mcp + 1, bx, by - 0.35); put(mcp + 2, bx + 0.02, by - 0.15); put(mcp + 3, bx + 0.02, by + 0.1); }
    else { put(mcp + 1, bx - 0.05, by - 0.4); put(mcp + 2, bx - 0.2, by - 0.6); put(mcp + 3, bx - 0.35, by - 0.62); } // index bent to the thumb
  });
  put(1, -0.3, -0.2);
  if (thumb === 'out') { put(2, -0.55, -0.35); put(3, -0.75, -0.5); put(4, -0.9, -0.62); }
  else if (thumb === 'in') { put(2, -0.4, -0.45); put(3, -0.25, -0.65); put(4, -0.05, -0.75); }
  else { put(2, -0.5, -0.45); put(3, -0.65, -0.6); put(4, -0.72, -1.55); } // pinch
  const ca = Math.cos(angle);
  const sa = Math.sin(angle);
  return pts.map(([x, y]) => ({ x: cx + (s * (ca * x - sa * y)) / (4 / 3), y: cy + s * (sa * x + ca * y), z: 0 }));
}

const CURL = ['curl', 'curl', 'curl', 'curl'];
const POINT = { fingers: ['ext', 'curl', 'curl', 'curl'], thumb: 'in' };
const PINCH = { fingers: ['bent', 'ext', 'ext', 'ext'], thumb: 'pinch' };
const cases = [
  ['open', {}],
  ['fist', { fingers: CURL, thumb: 'in' }],
  ['point', POINT],
  ['two', { fingers: ['ext', 'ext', 'curl', 'curl'], thumb: 'in' }],
  ['three', { fingers: ['ext', 'ext', 'ext', 'curl'], thumb: 'in' }],
  ['pinch', PINCH],
  ['pinch', { fingers: ['bent', 'curl', 'curl', 'curl'], thumb: 'pinch' }],
  ['thumbsup', { fingers: CURL, thumb: 'out' }],
  ['pinky', { fingers: ['curl', 'curl', 'curl', 'ext'], thumb: 'in' }],
  ['pinky', { fingers: ['curl', 'curl', 'curl', 'ext'], thumb: 'out' }], // 🤙 shaka counts too
];
for (const angle of [0, 0.5, -0.5, 1.2, Math.PI]) { // rotation invariance
  for (const [want, cfg] of cases) assert.equal(classifyHand(hand({ ...cfg, angle })).gesture, want, `${want} at ${angle} rad`);
}

const events = [];
const g = new GestureController({
  video: null, overlay: null,
  onCursor: (c) => events.push(['cursor', c]), onTap: (x, y) => events.push(['tap', x, y]),
  onDrag: (phase, x, y, dx, dy) => events.push(['drag', phase, dx, dy]), onPalm: (dx, dy) => events.push(['palm', dx, dy]),
  onZoom: (f) => events.push(['zoom', f]), onAction: (a) => events.push(['action', a]),
});
const of = (kind) => events.filter((e) => e[0] === kind);
let t = 0;
const feed = (cfg, frames, move = () => ({})) => { for (let k = 0; k < frames; k++) g.process([hand({ ...cfg, ...move(k) })], (t += 33)); };

// 1. pointing moves a visible cursor; the cursor is mirrored (camera +x = screen left)
feed(POINT, 10, (k) => ({ cx: 0.6 - 0.01 * k }));
const cur = of('cursor').map((e) => e[1]);
assert.ok(cur.at(-1).visible && cur.at(-1).x > cur[0].x, 'cursor follows the hand, mirrored');

// 2. a quick pinch is a tap (click) at the cursor; no drag
events.length = 0;
feed(PINCH, 5);
feed(POINT, 4);
assert.equal(of('tap').length, 1, 'quick pinch = one tap');
assert.equal(of('drag').length, 0, 'no drag for a quick pinch');

// 3. a held, moving pinch is a drag: start, moves, end (and no tap)
events.length = 0;
feed(PINCH, 25, (k) => ({ cy: 0.6 - 0.006 * k }));
feed(POINT, 4);
const phases = of('drag').map((e) => e[1]);
assert.equal(phases[0], 'start');
assert.equal(phases.at(-1), 'end');
assert.ok(phases.filter((p) => p === 'move').length > 5, 'drag moves');
assert.ok(of('drag').filter((e) => e[1] === 'move').reduce((s, e) => s + e[3], 0) < -0.02, 'raising the hand drags upward');
assert.equal(of('tap').length, 0, 'a drag is not a tap');

// 4. open palm motion -> palm events; held fist -> exactly one toggle-play; held two fingers -> cycle-mode
events.length = 0;
feed({}, 20, (k) => ({ cx: 0.4 + 0.01 * k }));
assert.ok(of('palm').length > 5 && of('palm').reduce((s, e) => s + e[1], 0) < -0.05, 'mirrored palm motion');
feed({ fingers: CURL, thumb: 'in' }, 40);
feed({ fingers: ['ext', 'ext', 'curl', 'curl'], thumb: 'in' }, 40);
assert.deepEqual(of('action').map((e) => e[1]), ['toggle-play', 'cycle-mode']);

// 5. two open hands moving apart -> zoom in (factor < 1)
events.length = 0;
for (let k = 0; k < 15; k++) g.process([hand({ cx: 0.4 - 0.008 * k, s: 0.08 }), hand({ cx: 0.6 + 0.008 * k, s: 0.08 })], (t += 33));
const zoom = of('zoom').reduce((s, e) => s * e[1], 1);
assert.ok(zoom < 0.8, `spreading two hands zooms in, got factor ${zoom}`);

// 6. losing the hand hides the cursor
events.length = 0;
for (let k = 0; k < 8; k++) g.process([], (t += 33));
assert.equal(of('cursor').at(-1)[1].visible, false);
// 7. air writing: holding the little finger toggles draw mode once; then the fingertip is reported every frame
events.length = 0;
const PINKY = { fingers: ['curl', 'curl', 'curl', 'ext'], thumb: 'in' };
feed(PINKY, 40);
assert.deepEqual(of('action').map((e) => e[1]), ['toggle-draw'], 'holding 🤙 fires toggle-draw once');
const frames = [];
g.onDraw = (f) => frames.push(f);
g.setMode('draw');
feed(PINKY, 40); // still held after the switch: must NOT toggle straight back
assert.equal(of('action').length, 1, 'the toggling pose is locked until released');
frames.length = 0;
feed(POINT, 12, (k) => ({ cx: 0.6 - 0.01 * k }));
const pen = frames.filter((f) => f.gesture === 'point');
assert.ok(pen.length >= 9, 'pointing = pen down within two frames');
assert.ok(pen.at(-1).tip.x > pen[0].tip.x, 'fingertip is mirrored like the camera view');
assert.ok(pen.every((f) => f.tip.x >= 0 && f.tip.x <= 1 && f.tip.y >= 0 && f.tip.y <= 1), 'tip in view coordinates');
frames.length = 0;
feed(PINCH, 4);
feed(POINT, 3);
assert.equal(frames.filter((f) => f.tap).length, 1, 'a quick pinch is one tap in draw mode');
feed({ fingers: CURL, thumb: 'in' }, 30);
assert.ok(frames.some((f) => f.gesture === 'fist'), 'fist reported (eraser)');
assert.equal(of('action').length, 1, 'fist does not toggle play while writing');
feed(PINKY, 40);
assert.deepEqual(of('action').map((e) => e[1]), ['toggle-draw', 'toggle-draw'], 'holding 🤙 again leaves air writing');
for (let k = 0; k < 4; k++) g.process([], (t += 33));
assert.equal(frames.at(-1).visible, false, 'losing the hand lifts the pen');
console.log('gesture tests passed');
