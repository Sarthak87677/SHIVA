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
];
for (const angle of [0, 0.5, -0.5, 1.2, Math.PI]) { // rotation invariance
  for (const [want, cfg] of cases) assert.equal(classifyHand(hand({ ...cfg, angle })).gesture, want, `${want} at ${angle} rad`);
}

const events = [];
const g = new GestureController({
  video: null, overlay: null, onState: () => {},
  onRotate: (dx, dy) => events.push(['rotate', dx, dy]), onZoom: (f) => events.push(['zoom', f]), onAction: (a) => events.push(['action', a]),
});
let t = 0;
for (let k = 0; k < 40; k++) g.process(hand({ fingers: CURL, thumb: 'in' }), (t += 33)); // held fist
for (let k = 0; k < 20; k++) g.process(hand({ cx: 0.4 + 0.01 * k }), (t += 33)); // palm moves toward image +x (user's left)
for (let k = 0; k < 20; k++) g.process(hand({ ...PINCH, cy: 0.7 - 0.01 * k }), (t += 33)); // pinch raised
for (let k = 0; k < 14; k++) g.process(hand({ ...POINT, cx: 0.7 - 0.03 * k }), (t += 33)); // swipe to the user's right
const actions = events.filter((e) => e[0] === 'action').map((e) => e[1]);
assert.deepEqual(actions, ['toggle-play', 'swipe-right'], 'a held fist fires once; a pointing swipe fires once');
const rot = events.filter((e) => e[0] === 'rotate').reduce((s, e) => s + e[1], 0);
assert.ok(rot < -0.05, `mirrored rotation should be negative, got ${rot}`);
const zoom = events.filter((e) => e[0] === 'zoom').reduce((s, e) => s * e[1], 1);
assert.ok(zoom < 0.9, `raising a pinch zooms in, got factor ${zoom}`);
console.log('gesture tests passed');
