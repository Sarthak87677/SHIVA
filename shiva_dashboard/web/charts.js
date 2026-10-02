// SHIVA-1 dashboard :: small SVG chart kit (no dependencies).
// Charts redraw at the container's pixel width so text stays 11 px; every chart
// has a legend, a hover tooltip and an equivalent table view (see app.js).

const NS = 'http://www.w3.org/2000/svg';

export const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

/** Compact number formatting: 3 significant digits, exponent for very large/small magnitudes. */
export function fmt(v, digits = 3) {
  if (v === null || v === undefined || !Number.isFinite(v)) return '–';
  const a = Math.abs(v);
  if (a === 0) return '0';
  if (a >= 1e5 || a < 1e-3) return v.toExponential(Math.max(0, digits - 1)).replace('e+', 'e');
  return String(Number(v.toPrecision(digits)));
}

export const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

function svgEl(tag, attrs = {}, parent = null) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) if (v !== undefined && v !== null) e.setAttribute(k, v);
  if (parent) parent.appendChild(e);
  return e;
}

// ---------------------------------------------------------------- tooltip
const tipEl = () => document.getElementById('tooltip');

export function showTip(html, x, y) {
  const t = tipEl();
  t.innerHTML = html;
  t.hidden = false;
  const r = t.getBoundingClientRect();
  let left = x + 14;
  let top = y + 14;
  if (left + r.width > innerWidth - 8) left = x - r.width - 14;
  if (top + r.height > innerHeight - 8) top = y - r.height - 14;
  t.style.left = `${Math.max(8, left)}px`;
  t.style.top = `${Math.max(8, top)}px`;
}

export function hideTip() { tipEl().hidden = true; }

export const tipRow = (color, label, value) =>
  `<div class="tt-row"><span class="tt-sw" style="background:${color}"></span>${esc(label)}` +
  `<span style="margin-left:auto;padding-left:10px">${esc(value)}</span></div>`;

// ---------------------------------------------------------------- legend
/** items: [{name, color, box?}]; with onToggle the items become toggle buttons. */
export function legend(el, items, { onToggle = null, hidden = new Set() } = {}) {
  el.innerHTML = '';
  items.forEach((it, i) => {
    const s = document.createElement('span');
    s.className = `item${onToggle ? ' toggle' : ''}${hidden.has(i) ? ' off' : ''}`;
    s.innerHTML = `<span class="sw${it.box ? ' box' : ''}" style="background:${it.color}"></span>${esc(it.name)}`;
    if (onToggle) {
      s.tabIndex = 0;
      s.setAttribute('role', 'switch');
      s.setAttribute('aria-checked', String(!hidden.has(i)));
      const fire = () => onToggle(i);
      s.addEventListener('click', fire);
      s.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fire(); } });
    }
    el.appendChild(s);
  });
}

// ---------------------------------------------------------------- table
export function renderTable(el, columns, rows) {
  const head = columns.map((c) => `<th${c.num ? ' class="num"' : ''}>${esc(c.label)}</th>`).join('');
  const body = rows.map((r) => `<tr>${r.map((v, i) => `<td${columns[i].num ? ' class="num"' : ''}>${esc(v)}</td>`).join('')}</tr>`).join('');
  el.innerHTML = `<table class="data"><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
}

// ---------------------------------------------------------------- scales
function niceTicks(lo, hi, count = 5) {
  if (!(hi > lo)) { hi = lo + 1; }
  const raw = (hi - lo) / count;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? 10 * mag;
  const start = Math.floor(lo / step) * step;
  const ticks = [];
  for (let v = start; v <= hi + step * 0.5; v += step) ticks.push(Number(v.toFixed(12)));
  return ticks;
}

function logTicks(lo, hi) {
  const a = Math.floor(Math.log10(lo));
  const b = Math.ceil(Math.log10(hi));
  const ticks = [];
  const mult = b - a <= 2 ? [1, 2, 5] : [1];
  for (let e = a; e <= b; e++) for (const m of mult) ticks.push(m * 10 ** e);
  return ticks.filter((t) => t >= lo * 0.999 && t <= hi * 1.001);
}

const superscript = (n) => String(n).replace(/[-0-9]/g, (c) => '⁻⁰¹²³⁴⁵⁶⁷⁸⁹'['-0123456789'.indexOf(c)]);
const fmtLogTick = (v) => {
  const e = Math.log10(v);
  return Math.abs(e - Math.round(e)) < 1e-9 && Math.abs(e) >= 3 ? `10${superscript(Math.round(e))}` : fmt(v, 2);
};

/** Re-run draw(width) whenever the container width changes. */
function mount(el, draw) {
  let last = 0;
  const run = () => {
    const w = Math.round(el.clientWidth);
    if (w > 0 && w !== last) { last = w; draw(w); }
  };
  if (el._ro) el._ro.disconnect();
  el._ro = new ResizeObserver(run);
  el._ro.observe(el);
  el._redraw = () => { last = 0; run(); };
  run();
}

// ---------------------------------------------------------------- horizontal grouped bars
/**
 * categories: [{label, info}], series: [{name, color, values}], values in [0, max].
 */
export function hBarChart(el, { categories, series, max = 1, format = (v) => fmt(v, 2) }) {
  mount(el, (W) => {
    el.innerHTML = '';
    const barH = 8;
    const gap = 2;
    const rowH = series.length * (barH + gap) + 10;
    const longest = Math.max(...categories.map((c) => String(c.label).length));
    const padL = Math.min(W * 0.48, Math.max(90, longest * 6.3 + 14)); // fit the longest label
    const padR = 12;
    const padT = 4;
    const padB = 22;
    const H = padT + categories.length * rowH + padB;
    const x = (v) => padL + (Math.max(0, Math.min(max, v ?? 0)) / max) * (W - padL - padR);
    const svg = svgEl('svg', { viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: 'img' }, el);
    const grid = svgEl('g', { class: 'grid' }, svg);
    for (const t of niceTicks(0, max, 4)) {
      if (t > max + 1e-9) continue;
      svgEl('line', { x1: x(t), x2: x(t), y1: padT, y2: H - padB }, grid);
      svgEl('text', { x: x(t), y: H - 6, 'text-anchor': 'middle' }, svg).textContent = format(t);
    }
    svgEl('line', { class: 'axis', x1: padL, x2: padL, y1: padT, y2: H - padB }, svg);
    categories.forEach((c, i) => {
      const y0 = padT + i * rowH + 5;
      const label = svgEl('text', { class: 'lbl', x: padL - 8, y: y0 + (series.length * (barH + gap)) / 2 + 3, 'text-anchor': 'end' }, svg);
      label.textContent = c.label;
      series.forEach((s, j) => {
        const v = s.values[i];
        const w = Math.max(0, x(v) - padL);
        // rounded data end, flat at the baseline
        const yb = y0 + j * (barH + gap);
        const r = Math.min(2, w / 2);
        const d = `M${padL},${yb}h${w - r}a${r},${r} 0 0 1 ${r},${r}v${barH - 2 * r}a${r},${r} 0 0 1 ${-r},${r}h${-(w - r)}z`;
        if (w > 0) svgEl('path', { d, fill: s.color }, svg);
      });
      const hit = svgEl('rect', { class: 'hit', x: 0, y: padT + i * rowH, width: W, height: rowH }, svg);
      hit.addEventListener('mousemove', (e) => {
        const rows = series.map((s) => tipRow(s.color, s.name, format(s.values[i]))).join('');
        showTip(`<b>${esc(c.label)}</b>${c.info ? `<div class="tt-muted" style="margin:2px 0 6px">${esc(c.info)}</div>` : ''}${rows}`, e.clientX, e.clientY);
      });
      hit.addEventListener('mouseleave', hideTip);
    });
  });
}

// ---------------------------------------------------------------- line chart (linear or log axes)
/**
 * x: number[]; series: [{name, color, values, dash?}]; hidden: Set of series indices.
 */
export function lineChart(el, opts) {
  const { x, series, logX = false, logY = false, height = 240, xLabel = '', yLabel = '', xFormat = (v) => fmt(v),
    yFormat = (v) => fmt(v), hidden = new Set(), yDomain = null, integerX = false } = opts;
  mount(el, (W) => {
    el.innerHTML = '';
    const padL = 46;
    const padR = 14;
    const padT = 20;
    const padB = 34;
    const H = height;
    const vis = series.map((s, i) => i).filter((i) => !hidden.has(i));
    const ok = (v) => Number.isFinite(v) && (!logY || v > 0);
    let ys = [];
    for (const i of vis) ys = ys.concat(series[i].values.filter(ok));
    if (!ys.length) ys = [logY ? 1 : 0, logY ? 10 : 1];
    let [y0, y1] = yDomain ?? [Math.min(...ys), Math.max(...ys)];
    if (!logY && !yDomain) { const pad = (y1 - y0 || Math.abs(y1) || 1) * 0.08; y0 -= pad; y1 += pad; }
    if (logY && !yDomain) { y0 /= 1.4; y1 *= 1.4; }
    const yt = logY ? logTicks(y0, y1) : niceTicks(y0, y1, 4);
    if (!logY && !yDomain) { y0 = Math.min(y0, yt[0]); y1 = Math.max(y1, yt[yt.length - 1]); }
    const xs0 = Math.min(...x);
    const xs1 = Math.max(...x);
    const tx = logX ? (v) => Math.log10(v) : (v) => v;
    const ty = logY ? (v) => Math.log10(v) : (v) => v;
    const sx = (v) => padL + ((tx(v) - tx(xs0)) / (tx(xs1) - tx(xs0) || 1)) * (W - padL - padR);
    const sy = (v) => H - padB - ((ty(v) - ty(y0)) / (ty(y1) - ty(y0) || 1)) * (H - padT - padB);
    const svg = svgEl('svg', { viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: 'img' }, el);
    const grid = svgEl('g', { class: 'grid' }, svg);
    for (const t of yt) {
      if (t < y0 - 1e-12 || t > y1 + 1e-12) continue;
      svgEl('line', { x1: padL, x2: W - padR, y1: sy(t), y2: sy(t) }, grid);
      svgEl('text', { x: padL - 6, y: sy(t) + 3.5, 'text-anchor': 'end' }, svg).textContent = logY ? fmtLogTick(t) : yFormat(t);
    }
    let xt = logX ? logTicks(xs0, xs1) : niceTicks(xs0, xs1, Math.min(6, x.length));
    if (integerX) xt = xt.filter((t) => Number.isInteger(t));
    for (const t of xt) {
      if (t < xs0 - 1e-12 || t > xs1 + 1e-12) continue;
      svgEl('text', { x: sx(t), y: H - padB + 15, 'text-anchor': 'middle' }, svg).textContent = logX ? fmtLogTick(t) : xFormat(t);
    }
    svgEl('line', { class: 'axis', x1: padL, x2: W - padR, y1: H - padB, y2: H - padB }, svg);
    if (yLabel) svgEl('text', { x: padL - 6, y: 10, 'text-anchor': 'start' }, svg).textContent = yLabel;
    if (xLabel) svgEl('text', { x: W - padR, y: H - 4, 'text-anchor': 'end' }, svg).textContent = xLabel;
    const clipId = `clip${Math.random().toString(36).slice(2)}`;
    const defs = svgEl('defs', {}, svg);
    const cp = svgEl('clipPath', { id: clipId }, defs);
    svgEl('rect', { x: padL, y: padT - 4, width: W - padL - padR, height: H - padT - padB + 8 }, cp);
    const lines = svgEl('g', { 'clip-path': `url(#${clipId})` }, svg);
    for (const i of vis) {
      const s = series[i];
      let d = '';
      let pen = false;
      s.values.forEach((v, k) => {
        if (!ok(v)) { pen = false; return; }
        d += `${pen ? 'L' : 'M'}${sx(x[k]).toFixed(1)},${sy(v).toFixed(1)}`;
        pen = true;
      });
      svgEl('path', { class: 'line', d, stroke: s.color, 'stroke-dasharray': s.dash }, lines);
      if (x.length <= 12) {
        s.values.forEach((v, k) => { if (ok(v)) svgEl('circle', { class: 'marker', cx: sx(x[k]), cy: sy(v), r: 4, fill: s.color }, lines); });
      }
    }
    const cross = svgEl('line', { class: 'crosshair', y1: padT, y2: H - padB, visibility: 'hidden' }, svg);
    const dots = svgEl('g', { visibility: 'hidden' }, svg);
    const hit = svgEl('rect', { class: 'hit', x: padL, y: padT, width: W - padL - padR, height: H - padT - padB }, svg);
    hit.addEventListener('mousemove', (e) => {
      const r = svg.getBoundingClientRect();
      const px = ((e.clientX - r.left) / r.width) * W;
      let k = 0;
      let best = Infinity;
      x.forEach((xv, j) => { const dd = Math.abs(sx(xv) - px); if (dd < best) { best = dd; k = j; } });
      cross.setAttribute('x1', sx(x[k]));
      cross.setAttribute('x2', sx(x[k]));
      cross.setAttribute('visibility', 'visible');
      dots.innerHTML = '';
      dots.setAttribute('visibility', 'visible');
      const rows = [];
      for (const i of vis) {
        const v = series[i].values[k];
        if (ok(v)) svgEl('circle', { class: 'marker', cx: sx(x[k]), cy: sy(v), r: 4, fill: series[i].color }, dots);
        rows.push(tipRow(series[i].color, series[i].name, ok(v) ? yFormat(v) : '–'));
      }
      showTip(`<b>${esc(xLabel || 'x')} = ${esc(xFormat(x[k]))}</b>${rows.join('')}`, e.clientX, e.clientY);
    });
    hit.addEventListener('mouseleave', () => {
      hideTip();
      cross.setAttribute('visibility', 'hidden');
      dots.setAttribute('visibility', 'hidden');
    });
  });
}
