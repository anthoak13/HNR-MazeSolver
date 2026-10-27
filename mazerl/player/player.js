/* mazerl maze player: NYT-style animated maze episodes in plain SVG + JS.
 *
 * This file is a single expression that evaluates to {mount: function}.
 * The Python side inlines it inside an IIFE for every render, so it defines no
 * globals and several players can live on one page (or in one notebook).
 *
 * Clock: `t` is measured in moves. t = k means move k has just finished and the
 * agent sits on path[k]; between k and k+1 the agent glides toward path[k+1].
 */
(function () {
  'use strict';

  var NS = 'http://www.w3.org/2000/svg';
  var S = 10; // viewBox units per cell
  var COL = {
    red: '#d9472b',
    startStroke: '#e2694f',
    wall: '#dcdcdc',
    weak: '#f3f3f3',
    dotIn: '#a6a6a6',
    dotOut: '#d3d3d3',
    tint: '#f6dfd9',
    flag: '#262626',
    ink: '#121212'
  };
  var MINUS = '−';
  var DEFAULTS = {
    speed: 6, autoplay: true, loop: true, tint_outside: 'auto', callouts: true,
    show_score: true, labels: true, trail_length: null
  };
  var START_HOLD = 0.8;  // seconds paused on the first frame when (re)starting
  var END_HOLD = 2.0;    // seconds paused on the last frame before looping
  var CALLOUT_SECS = 1.8;

  // ---------------------------------------------------------------- helpers
  function el(tag, attrs, parent) {
    var e = document.createElementNS(NS, tag);
    if (attrs) for (var k in attrs) if (attrs[k] != null) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  function div(cls, parent, text) {
    var e = document.createElement('div');
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    if (parent) parent.appendChild(e);
    return e;
  }
  function clamp(x, a, b) { return x < a ? a : (x > b ? b : x); }
  function ease(f) { return 0.55 * f + 0.45 * f * f * (3 - 2 * f); }
  function easeOut(f) { return 1 - (1 - f) * (1 - f); }
  function num(v) { return (typeof v === 'number' && isFinite(v)) ? v : null; }
  function merge() {
    var o = {};
    for (var i = 0; i < arguments.length; i++) {
      var a = arguments[i];
      if (a) for (var k in a) if (a[k] !== undefined && a[k] !== null) o[k] = a[k];
    }
    return o;
  }
  function fmtValue(v) {
    if (num(v) === null) return '';
    var a = Math.abs(v);
    var s = a !== 0 && a < 0.001 ? a.toPrecision(2) : String(Math.round(a * 1000) / 1000);
    return (v < 0 ? MINUS : (v > 0 ? '+' : '')) + s;
  }
  function fmtScore(v) {
    if (num(v) === null) return '–';
    var r = Math.round(v * 100) / 100;
    return (r < 0 ? MINUS : '') + Math.abs(r).toFixed(2);
  }
  function cellKey(rc) { return rc[0] + ',' + rc[1]; }
  function sameCell(a, b) { return a && b && a[0] === b[0] && a[1] === b[1]; }

  function calloutLines(ev) {
    var v = fmtValue(ev.value);
    var vs = v ? ' ' + v : '';
    switch (ev.type) {
      case 'outside': return ['Penalty for', 'stepping outside' + vs];
      case 'probe': return ['Probe' + vs];
      case 'through_wall': return ['Broke through', 'wall' + vs];
      case 'flag': return ['Flag' + vs];
      case 'new_cell': return null;
      default: return [String(ev.type).replace(/_/g, ' ') + vs];
    }
  }
  // when (in moves) an event's effects start showing: probes at the nudge peak,
  // everything else as the agent arrives in the cell
  function fireTime(ev) { return ev.t - (ev.type === 'probe' ? 0.5 : 0.3); }

  // ------------------------------------------------------- data preparation
  function prepare(ep, extraDefaults) {
    var o = merge(DEFAULTS, extraDefaults, ep.options);
    var grid = ep.grid || [[0]];
    var rows = grid.length, cols = grid[0].length;
    var opens = {}; // "r,c" -> earliest move index of a through_wall into that cell
    var agents = (ep.agents || []).map(function (a, i) {
      var path = (a.path && a.path.length) ? a.path : [ep.start || [0, 0]];
      var n = path.length;
      var events = (a.events || []).filter(function (ev) { return ev && num(ev.t) !== null; })
        .slice().sort(function (x, y) { return x.t - y.t; });
      var probeDir = {}, outside = {}, through = {};
      events.forEach(function (ev) {
        if (ev.type === 'outside') outside[ev.t] = true;
        if (ev.type === 'through_wall') {
          through[ev.t] = true;
          if (ev.cell) {
            var k = cellKey(ev.cell);
            if (!(k in opens) || ev.t < opens[k].t) opens[k] = { t: ev.t, cell: ev.cell };
          }
        }
        if (ev.type === 'probe' && ev.cell && ev.t >= 1 && ev.t - 1 < n) {
          var p = path[ev.t - 1];
          var dr = clamp(ev.cell[0] - p[0], -1, 1), dc = clamp(ev.cell[1] - p[1], -1, 1);
          if (dr || dc) probeDir[ev.t] = [dr, dc];
        }
      });
      var callouts = [], pulses = [];
      events.forEach(function (ev) {
        var lines = calloutLines(ev);
        var firstOutside = ev.type !== 'outside' || !outside[ev.t - 1];
        var c = { ev: ev, fire: fireTime(ev), lines: lines };
        if (lines && firstOutside) callouts.push(c);
        if (ev.type !== 'new_cell' && firstOutside) pulses.push(c);
      });
      return {
        idx: i, path: path, n: n, score: a.score || [], events: events, label: a.label,
        probeDir: probeDir, outside: outside, through: through, callouts: callouts, pulses: pulses
      };
    });
    var T = 0;
    agents.forEach(function (a) { T = Math.max(T, a.n - 1); });
    return { ep: ep, o: o, grid: grid, rows: rows, cols: cols, agents: agents, opens: opens, T: T };
  }

  // agent position (in cell units, fractional) at time t; `trail` = without probe nudges
  function agentPos(a, t, trail) {
    var path = a.path, n = a.n;
    if (t <= 0 || n === 1) return [path[0][0], path[0][1]];
    var k = Math.floor(t), f = t - k;
    if (k >= n - 1) return [path[n - 1][0], path[n - 1][1]];
    var p = path[k], q = path[k + 1];
    if (sameCell(p, q)) {
      var d = a.probeDir[k + 1];
      if (!d || trail) return [p[0], p[1]];
      var s = 0.3 * Math.sin(Math.PI * f);
      return [p[0] + d[0] * s, p[1] + d[1] * s];
    }
    if (a.through[k + 1]) f = clamp((f - 0.3) / 0.7, 0, 1); // pause while the wall cracks
    var e = ease(f);
    return [p[0] + (q[0] - p[0]) * e, p[1] + (q[1] - p[1]) * e];
  }
  function cellAt(a, t) {
    var k = clamp(Math.floor(t + 0.5), 0, a.n - 1);
    return a.path[k];
  }

  // --------------------------------------------------------------- the scene
  function Scene(parent, P, fallbackPx) {
    var rows = P.rows, cols = P.cols, grid = P.grid, o = P.o, ep = P.ep;
    var W = cols * S, H = rows * S;
    var multi = P.agents.length > 1;
    var svg = el('svg', {
      viewBox: '0 0 ' + W + ' ' + H, 'class': 'mzp-svg', role: 'img',
      'aria-label': ep.title || 'Maze episode', preserveAspectRatio: 'xMidYMid meet'
    }, parent);
    var fontEls = []; // [element, baseUnits, minPx]

    function ppu() {
      var w = svg.getBoundingClientRect().width;
      return (w > 0 ? w : fallbackPx) / W;
    }
    function fontUnits(base, minPx) { return Math.max(base, minPx / ppu()); }
    function code(r, c) {
      if (r < 0 || c < 0 || r >= rows || c >= cols) return -1;
      return grid[r][c];
    }
    function wallish(r, c) { var k = code(r, c); return k === 1 || k === 2; }
    function cx(c) { return (c + 0.5) * S; }
    function cy(r) { return (r + 0.5) * S; }

    el('rect', { x: 0, y: 0, width: W, height: H, fill: '#ffffff' }, svg);

    // pink tint over the outside band
    var gTint = el('g', { opacity: o.tint_outside === 'always' ? 1 : 0 }, svg);
    // one path per fill colour: abutting squares in a single path rasterise without seams
    function sq(r, c) { return 'M' + c * S + ' ' + r * S + 'h' + S + 'v' + S + 'h' + (-S) + 'Z'; }
    var anyOutside = false, tintD = '';
    for (var r = 0; r < rows; r++) for (var c = 0; c < cols; c++) {
      if (grid[r][c] !== 3) continue;
      anyOutside = true;
      tintD += sq(r, c);
    }
    if (tintD) el('path', { d: tintD, fill: COL.tint }, gTint);

    // dots and walls
    var gDots = el('g', null, svg), wallD = '';
    var opens = {}; // key -> {g, gray, pale, dot, flash, crack, frags, t} for weak spots that open
    for (r = 0; r < rows; r++) for (c = 0; c < cols; c++) {
      var v = grid[r][c], key = r + ',' + c;
      if (v === 0 || v === 3) {
        if (!(key in P.opens)) el('circle', { cx: cx(c), cy: cy(r), r: 0.55, fill: v === 3 ? COL.dotOut : COL.dotIn }, gDots);
      } else if ((v === 1 || v === 2) && !(key in P.opens)) {
        wallD += sq(r, c); // weak spots that never open share the wall path
      }
    }
    if (wallD) el('path', { d: wallD, fill: COL.wall }, svg);
    var gWeak = el('g', null, svg);
    for (r = 0; r < rows; r++) for (c = 0; c < cols; c++) {
      key = r + ',' + c;
      if (!(grid[r][c] === 2 || key in P.opens)) continue;
      var g = el('g', null, gWeak);
      var x0 = c * S - (wallish(r, c - 1) ? 0.12 : 0), y0 = r * S - (wallish(r - 1, c) ? 0.12 : 0);
      var x1 = (c + 1) * S + (wallish(r, c + 1) ? 0.12 : 0), y1 = (r + 1) * S + (wallish(r + 1, c) ? 0.12 : 0);
      var wasWall = grid[r][c] === 1 || grid[r][c] === 2;
      var gray = wasWall && key in P.opens ? el('rect', { x: x0, y: y0, width: x1 - x0, height: y1 - y0, fill: COL.wall }, g) : null;
      var pale = grid[r][c] === 2 ? el('rect', { x: c * S + 1.05, y: r * S + 1.05, width: S - 2.1, height: S - 2.1, fill: COL.weak }, g) : null;
      var info = { g: g, gray: gray, pale: pale, r: r, c: c, t: key in P.opens ? P.opens[key].t : null };
      if (info.t !== null) {
        info.dot = el('circle', { cx: cx(c), cy: cy(r), r: 0.55, fill: COL.dotIn, opacity: 0 }, g);
        info.flash = el('rect', {
          x: c * S, y: r * S, width: S, height: S, fill: 'none', stroke: COL.red,
          'stroke-width': 0.6, opacity: 0
        }, g);
        info.crack = el('path', {
          d: 'M' + (c * S + 2.2) + ' ' + (r * S + 2.6) + ' L' + (c * S + 4.6) + ' ' + (r * S + 4.8) +
            ' L' + (c * S + 4.1) + ' ' + (r * S + 6.2) + ' L' + (c * S + 7.4) + ' ' + (r * S + 7.8) +
            ' M' + (c * S + 4.6) + ' ' + (r * S + 4.8) + ' L' + (c * S + 7.2) + ' ' + (r * S + 3.0),
          fill: 'none', stroke: '#9a9a9a', 'stroke-width': 0.45, 'stroke-linecap': 'round',
          'stroke-linejoin': 'round', opacity: 0
        }, g);
        info.frags = [[-1, -1], [1, -0.6], [-0.7, 1], [0.9, 0.9]].map(function (d) {
          return {
            d: d, e: el('rect', {
              x: -1.1, y: -1.1, width: 2.2, height: 2.2, fill: COL.wall, opacity: 0
            }, g)
          };
        });
        opens[key] = info;
      }
    }

    // labels (OUTSIDE / GOAL / AGENT)
    var gLabels = el('g', { 'class': 'mzp-label' }, svg);
    function label(text, x, y, anchor, base) {
      var e = el('text', {
        x: x, y: y, 'text-anchor': anchor || 'start', 'font-size': base,
        fill: COL.ink, stroke: '#ffffff', 'stroke-width': base * 0.35, 'paint-order': 'stroke',
        'stroke-linejoin': 'round', 'font-weight': 600, 'letter-spacing': '0.02em', 'dominant-baseline': 'middle'
      }, gLabels);
      e.textContent = text;
      fontEls.push([e, base, 10]);
      return e;
    }
    var agentLabel = null;
    if (o.labels) {
      if (anyOutside && rows >= 3 && grid[rows - 1][0] === 3 && grid[rows - 2][0] === 3) {
        label('OUTSIDE', 0.4 * S, (rows - 1) * S, 'start', 4.4);
      }
      if (ep.flag) {
        var fr = ep.flag[0], fc = ep.flag[1];
        if (fc >= 3) label('GOAL', cx(fc) - 0.62 * S, cy(fr) + 0.04 * S, 'end', 4.4);
        else label('GOAL', cx(fc) + 0.62 * S, cy(fr) + 0.04 * S, 'start', 4.4);
      }
      if (ep.start) {
        var sr = ep.start[0], sc = ep.start[1];
        agentLabel = label('AGENT', cx(sc) - 0.55 * S, sr > 0 ? cy(sr) - 0.78 * S : cy(sr) + 0.85 * S, 'start', 4.4);
      }
    }

    // trails, start, flag, effects, agents, callouts (back to front)
    var gTrails = el('g', {
      fill: 'none', stroke: COL.red, 'stroke-width': 0.26 * S,
      'stroke-linecap': 'round', 'stroke-linejoin': 'round'
    }, svg);
    var trailOpacity = multi ? 0.5 : 0.7;
    var gMarks = el('g', null, svg);
    if (ep.start) {
      el('circle', {
        cx: cx(ep.start[1]), cy: cy(ep.start[0]), r: 0.29 * S, fill: '#ffffff',
        stroke: COL.startStroke, 'stroke-width': 0.065 * S
      }, gMarks);
    }
    if (ep.flag) {
      var fx = cx(ep.flag[1]), fy = cy(ep.flag[0]);
      el('circle', { cx: fx, cy: fy, r: 0.43 * S, fill: COL.flag }, gMarks);
      el('line', {
        x1: fx - 0.13 * S, y1: fy - 0.22 * S, x2: fx - 0.13 * S, y2: fy + 0.25 * S,
        stroke: '#ffffff', 'stroke-width': 0.055 * S, 'stroke-linecap': 'round'
      }, gMarks);
      el('path', {
        d: 'M' + (fx - 0.13 * S) + ' ' + (fy - 0.23 * S) + ' L' + (fx + 0.22 * S) + ' ' + (fy - 0.09 * S) +
          ' L' + (fx - 0.13 * S) + ' ' + (fy + 0.06 * S) + ' Z', fill: '#ffffff'
      }, gMarks);
    }
    var gFx = el('g', null, svg), gHalo = el('g', null, svg), gAgents = el('g', null, svg);
    var gCallouts = el('g', { 'class': 'mzp-callouts' }, svg);

    var A = P.agents.map(function (a) {
      return {
        a: a,
        trail: el('polyline', { 'stroke-opacity': trailOpacity, points: '' }, gTrails),
        steady: el('circle', { r: 0.52 * S, fill: COL.red, 'fill-opacity': 0.13, stroke: COL.red, 'stroke-opacity': 0.3, 'stroke-width': 0.05 * S, opacity: 0 }, gHalo),
        pulse: el('circle', { r: 0.5 * S, fill: COL.red, 'fill-opacity': 0.12, stroke: COL.red, 'stroke-opacity': 0.55, 'stroke-width': 0.06 * S, opacity: 0 }, gHalo),
        bump: el('circle', { r: 1, fill: 'none', stroke: COL.red, 'stroke-width': 0.06 * S, opacity: 0 }, gFx),
        dot: el('circle', { r: 0.28 * S, fill: COL.red }, gAgents),
        steadyOp: 0,
        callout: null, calloutRef: null,
        trailCacheK: -1, trailCache: ''
      };
    });

    var GAP = 1.7 * S; // vertical distance between callout text and the agent
    function calloutLayout(lines, r, c) {
      var fs = fontUnits(4.3, 11), lh = fs * 1.18;
      var longest = 0;
      lines.forEach(function (l) { longest = Math.max(longest, l.length); });
      var w = longest * fs * 0.54, h = lines.length * lh;
      var ax = cx(c), ay = cy(r);
      var above = ay - GAP - h >= 0.1 * S;
      var x0 = clamp(ax - 1.25 * S, 0.25 * S, W - 0.25 * S - w);
      var yTop = above ? ay - GAP - h : Math.min(ay + GAP, H - h - 0.1 * S);
      return { lines: lines, fs: fs, lh: lh, w: w, h: h, ax: ax, ay: ay, above: above, x0: x0, yTop: yTop };
    }
    function boxesOverlap(a, b) {
      var pad = 0.3 * S;
      return a.x0 - pad < b.x0 + b.w && b.x0 - pad < a.x0 + a.w &&
        a.yTop - pad < b.yTop + b.h && b.yTop - pad < a.yTop + a.h;
    }
    function buildCallout(L) {
      var lines = L.lines, fs = L.fs, lh = L.lh, w = L.w, h = L.h, ax = L.ax, ay = L.ay;
      var above = L.above, x0 = L.x0, yTop = L.yTop;
      var g = el('g', { opacity: 0 }, gCallouts);
      var text = el('text', {
        x: x0, y: yTop, 'font-size': fs, fill: COL.ink, 'font-weight': 500,
        stroke: '#ffffff', 'stroke-width': fs * 0.32, 'paint-order': 'stroke', 'stroke-linejoin': 'round'
      }, g);
      lines.forEach(function (l, i) {
        var ts = el('tspan', { x: x0, dy: i === 0 ? fs * 0.95 : lh }, text);
        ts.textContent = l;
      });
      // curved arrow from the text to the agent's halo, as on p.3 of the article
      var sx = clamp(ax - 0.85 * S, x0 + 0.3 * S, x0 + Math.max(0.3 * S, w - 0.3 * S));
      var sy = above ? yTop + h + 0.3 * S : yTop - 0.3 * S;
      var side = Math.abs(sx - ax) < 0.25 * S ? 0 : (sx < ax ? -1 : 1);
      var ex = ax + side * 0.36 * S, ey = ay + (above ? -1 : 1) * (side ? 0.5 : 0.62) * S;
      if (Math.abs(sy - ey) > 0.35 * S) {
        var qx = sx, qy = side ? ey : (sy + ey) / 2;
        el('path', {
          d: 'M' + sx + ' ' + sy + ' Q' + qx + ' ' + qy + ' ' + ex + ' ' + ey,
          fill: 'none', stroke: COL.ink, 'stroke-width': 0.055 * S, 'stroke-linecap': 'round'
        }, g);
        var tx = ex - qx, ty = ey - qy, tl = Math.sqrt(tx * tx + ty * ty) || 1;
        tx /= tl; ty /= tl;
        var hl = 0.22 * S, ca = Math.cos(0.5), sa = Math.sin(0.5);
        var h1x = ex - hl * (tx * ca - ty * sa), h1y = ey - hl * (ty * ca + tx * sa);
        var h2x = ex - hl * (tx * ca + ty * sa), h2y = ey - hl * (ty * ca - tx * sa);
        el('path', {
          d: 'M' + h1x + ' ' + h1y + ' L' + ex + ' ' + ey + ' L' + h2x + ' ' + h2y,
          fill: 'none', stroke: COL.ink, 'stroke-width': 0.055 * S, 'stroke-linecap': 'round', 'stroke-linejoin': 'round'
        }, g);
      }
      return { g: g, L: L };
    }

    function trailPoints(st, t) {
      var a = st.a, path = a.path;
      var k = clamp(Math.floor(t), 0, a.n - 1);
      var from = 0;
      if (num(o.trail_length) !== null) from = Math.max(0, k - Math.floor(o.trail_length));
      if (st.trailCacheK !== k || st.trailFrom !== from) {
        var pts = [], last = null;
        for (var i = from; i <= k; i++) {
          var p = path[i];
          if (last && sameCell(last, p)) continue;
          pts.push(cx(p[1]).toFixed(2) + ',' + cy(p[0]).toFixed(2));
          last = p;
        }
        st.trailCache = pts.join(' ');
        st.trailCacheK = k; st.trailFrom = from;
      }
      var cur = agentPos(a, t, true);
      return st.trailCache + ' ' + cx(cur[1]).toFixed(2) + ',' + cy(cur[0]).toFixed(2);
    }

    var tintOp = o.tint_outside === 'always' ? 1 : 0;
    var lastPpu = 0;

    function update(t, mps, dt, nowMs) {
      var secs = function (moves) { return moves / mps; };
      var smooth = 1 - Math.exp(-dt / 0.22);

      // keep label/callout text legible when the maze is drawn small
      var p = ppu();
      if (Math.abs(p - lastPpu) > 1e-3) {
        lastPpu = p;
        fontEls.forEach(function (f) {
          var fs = fontUnits(f[1], f[2]);
          f[0].setAttribute('font-size', fs);
          f[0].setAttribute('stroke-width', fs * 0.35);
        });
        A.forEach(function (st) { if (st.callout) { gCallouts.removeChild(st.callout.g); st.callout = null; st.calloutRef = null; } });
      }

      // weak spots opening
      for (var key in opens) {
        var w = opens[key];
        var age = secs(t - (w.t - 1));
        var prog = clamp((age - 0.06) / 0.4, 0, 1);
        if (w.gray) w.gray.setAttribute('opacity', 1 - prog);
        if (w.pale) w.pale.setAttribute('opacity', 1 - prog);
        w.dot.setAttribute('opacity', prog);
        var fl = age >= 0 && age < 0.6 ? 1 - age / 0.6 : 0;
        w.crack.setAttribute('opacity', age >= 0 && age < 0.25 ? 1 : (age >= 0.25 && age < 0.45 ? 1 - (age - 0.25) / 0.2 : 0));
        w.flash.setAttribute('opacity', fl * 0.8);
        if (fl > 0) {
          var grow = 1 + 0.25 * easeOut(age / 0.6);
          var ccx = cx(w.c), ccy = cy(w.r), hs = S * grow / 2;
          w.flash.setAttribute('x', ccx - hs); w.flash.setAttribute('y', ccy - hs);
          w.flash.setAttribute('width', 2 * hs); w.flash.setAttribute('height', 2 * hs);
        }
        var fa = age >= 0.05 && age < 0.65 ? (age - 0.05) / 0.6 : -1;
        w.frags.forEach(function (fr) {
          if (fa < 0) { fr.e.setAttribute('opacity', 0); return; }
          var dist = 0.25 * S + 0.55 * S * easeOut(fa);
          var x = cx(w.c) + fr.d[0] * dist, y = cy(w.r) + fr.d[1] * dist;
          var sc = 1 - 0.5 * fa;
          fr.e.setAttribute('transform', 'translate(' + x.toFixed(2) + ' ' + y.toFixed(2) + ') rotate(' + (fa * 140 * fr.d[0]).toFixed(1) + ') scale(' + sc.toFixed(3) + ')');
          fr.e.setAttribute('opacity', (1 - fa).toFixed(3));
        });
      }

      var anyOut = false;
      var shown = [];
      A.forEach(function (st) {
        var a = st.a;
        var pos = agentPos(a, t, false);
        var x = cx(pos[1]), y = cy(pos[0]);
        st.dot.setAttribute('cx', x.toFixed(2));
        st.dot.setAttribute('cy', y.toFixed(2));
        st.trail.setAttribute('points', trailPoints(st, t));

        var here = cellAt(a, t);
        if (code(here[0], here[1]) === 3 && !(cellKey(here) in opens)) anyOut = true;

        // steady halo while the agent keeps collecting the outside penalty
        var kk = Math.floor(t + 0.3);
        var lastK = a.n - 1;
        var steadyOn = t > 0 && (a.outside[kk] || (kk > lastK && a.outside[lastK]));
        st.steadyOp += ((steadyOn ? 1 : 0) - st.steadyOp) * smooth;
        if (st.steadyOp > 0.01) {
          var breathe = 1 + 0.07 * Math.sin(nowMs / 1000 * Math.PI * 1.6);
          st.steady.setAttribute('cx', x.toFixed(2)); st.steady.setAttribute('cy', y.toFixed(2));
          st.steady.setAttribute('r', (0.52 * S * breathe).toFixed(2));
          st.steady.setAttribute('opacity', st.steadyOp.toFixed(3));
        } else st.steady.setAttribute('opacity', 0);

        // pulse ring for the latest discrete event
        var pulse = null;
        for (var i = a.pulses.length - 1; i >= 0; i--) if (a.pulses[i].fire <= t) { pulse = a.pulses[i]; break; }
        var pa = pulse ? secs(t - pulse.fire) : 99;
        var pdur = pulse && pulse.ev.type === 'flag' ? 1.4 : 0.9;
        if (pulse && pa < pdur) {
          var f = pa / pdur;
          st.pulse.setAttribute('cx', x.toFixed(2)); st.pulse.setAttribute('cy', y.toFixed(2));
          st.pulse.setAttribute('r', (S * (0.36 + (pulse.ev.type === 'flag' ? 0.8 : 0.45) * easeOut(f))).toFixed(2));
          st.pulse.setAttribute('opacity', (1 - f).toFixed(3));
        } else st.pulse.setAttribute('opacity', 0);

        // probe: ring where the agent touched the wall
        if (pulse && pulse.ev.type === 'probe' && pulse.ev.cell && pa < 0.55) {
          var src = a.path[clamp(pulse.ev.t - 1, 0, a.n - 1)];
          var bx = (cx(src[1]) + cx(pulse.ev.cell[1])) / 2, by = (cy(src[0]) + cy(pulse.ev.cell[0])) / 2;
          var bf = pa / 0.55;
          st.bump.setAttribute('cx', bx.toFixed(2)); st.bump.setAttribute('cy', by.toFixed(2));
          st.bump.setAttribute('r', (S * (0.12 + 0.35 * easeOut(bf))).toFixed(2));
          st.bump.setAttribute('opacity', (0.9 * (1 - bf)).toFixed(3));
        } else st.bump.setAttribute('opacity', 0);

        // floating callout
        if (!o.callouts) return;
        var co = null;
        for (i = a.callouts.length - 1; i >= 0; i--) if (a.callouts[i].fire <= t) { co = a.callouts[i]; break; }
        var ca = co ? secs(t - co.fire) : 99;
        var persist = co && co.ev.type === 'flag';
        var vis = co && (persist || ca < CALLOUT_SECS);
        var L = null;
        if (vis) {
          var ecell = co.ev.type === 'probe' ? a.path[clamp(co.ev.t - 1, 0, a.n - 1)] : a.path[clamp(co.ev.t, 0, a.n - 1)];
          L = st.calloutRef === co && st.callout ? st.callout.L : calloutLayout(co.lines, ecell[0], ecell[1]);
          // earlier agents win when callouts would overlap
          for (var j = 0; j < shown.length; j++) if (boxesOverlap(shown[j], L)) { vis = false; break; }
          if (vis) shown.push(L);
        }
        if (!vis) {
          if (st.callout) { gCallouts.removeChild(st.callout.g); st.callout = null; st.calloutRef = null; }
          return;
        }
        if (st.calloutRef !== co) {
          if (st.callout) gCallouts.removeChild(st.callout.g);
          st.callout = buildCallout(L);
          st.calloutRef = co;
        }
        // the callout travels with its agent (kept inside the maze bounds)
        var tp = agentPos(a, t, true), CL = st.callout.L;
        var ox = clamp(cx(tp[1]) - CL.ax, 0.1 * S - CL.x0, W - 0.1 * S - CL.w - CL.x0);
        var oy = clamp(cy(tp[0]) - CL.ay, 0.05 * S - CL.yTop, H - 0.05 * S - CL.h - CL.yTop);
        st.callout.g.setAttribute('transform', 'translate(' + ox.toFixed(2) + ' ' + oy.toFixed(2) + ')');
        var op = clamp(ca / 0.12, 0, 1);
        if (!persist) op *= clamp((CALLOUT_SECS - ca) / 0.4, 0, 1);
        st.callout.g.setAttribute('opacity', op.toFixed(3));
      });

      if (o.tint_outside === 'auto') {
        tintOp += ((anyOut ? 1 : 0) - tintOp) * (1 - Math.exp(-dt / 0.3));
        gTint.setAttribute('opacity', tintOp.toFixed(3));
      } else if (o.tint_outside === 'never') gTint.setAttribute('opacity', 0);

      if (agentLabel) agentLabel.setAttribute('opacity', clamp(1 - t / 1.5, 0, 1).toFixed(3));
    }

    return { svg: svg, update: update, P: P };
  }

  function scoreAt(a, t) {
    if (!a.score.length) return null;
    var k = clamp(Math.floor(t + 1e-6), 0, a.score.length - 1);
    return a.score[k];
  }
  function scoreLine(P, t, prefix) {
    var ag = P.agents;
    if (!ag.length) return '';
    if (ag.length === 1) return prefix + fmtScore(scoreAt(ag[0], t));
    return ag.map(function (a, i) {
      return (a.label || ('Agent ' + (i + 1))) + ' ' + fmtScore(scoreAt(a, t));
    }).join('  ·  ');
  }

  var ICON_PLAY = '<svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true"><path d="M4 2.5 L13 8 L4 13.5 Z" fill="currentColor"/></svg>';
  var ICON_PAUSE = '<svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true"><rect x="3.5" y="2.5" width="3" height="11" fill="currentColor"/><rect x="9.5" y="2.5" width="3" height="11" fill="currentColor"/></svg>';
  var ICON_REPLAY = '<svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true"><path d="M8 2.5 A5.5 5.5 0 1 1 2.6 6.8" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/><path d="M1.2 3.2 L3.2 7.4 L6.6 4.6 Z" fill="currentColor"/></svg>';

  // ---------------------------------------------------------------- mount
  function mount(root, payload) {
    if (!root || root.__mzpMounted) return;
    root.__mzpMounted = true;
    var gridMode = payload.mode === 'grid';
    var stage = root.querySelector('.mzp-stage') || div('mzp-stage', root);
    var fb = stage.querySelector('.mzp-fallback');
    if (fb) stage.removeChild(fb);

    var widthPx = payload.width || 520;
    var cols = Math.max(1, payload.cols || 3);
    var eps = payload.episodes || [];
    var extra = gridMode ? { labels: false, callouts: false } : null;
    var scenes = [], scoreEls = [];
    var cellPx = gridMode ? widthPx / Math.min(cols, eps.length || 1) : widthPx;

    if (gridMode) {
      stage.className = 'mzp-stage mzp-grid' + (cols > 2 ? ' mzp-grid-many' : '');
      stage.style.gridTemplateColumns = 'repeat(' + Math.min(cols, Math.max(1, eps.length)) + ', minmax(0, 1fr))';
    }
    eps.forEach(function (ep) {
      var P = prepare(ep, extra);
      var holder = stage;
      if (gridMode) {
        holder = div('mzp-cell', stage);
        if (ep.title) div('mzp-cell-title', holder, ep.title);
      }
      scenes.push(Scene(holder, P, cellPx));
      if (gridMode) {
        scoreEls.push(P.o.show_score && P.agents.length ? div('mzp-cell-score', holder, '') : null);
        if (ep.caption) div('mzp-cell-caption', holder, ep.caption);
      }
    });
    if (!scenes.length) return;

    var o0 = scenes[0].P.o;
    var T = 0;
    scenes.forEach(function (s) { T = Math.max(T, s.P.T); });
    var base = num(o0.speed) && o0.speed > 0 ? o0.speed : 6;
    var mult = 1;
    var t = 0, playing = !!o0.autoplay && T > 0, hold = START_HOLD, endHold = 0;
    var loop = !!o0.loop;
    function mps() { return base * mult; }

    // controls
    var bar = div('mzp-controls', null);
    var caption = root.querySelector('.mzp-caption');
    if (caption && caption.parentNode === root) root.insertBefore(bar, caption);
    else root.appendChild(bar);
    var btn = document.createElement('button');
    btn.type = 'button'; btn.className = 'mzp-play';
    bar.appendChild(btn);
    var scrub = document.createElement('input');
    scrub.type = 'range'; scrub.className = 'mzp-scrub';
    scrub.min = 0; scrub.max = Math.max(T, 1); scrub.step = 'any'; scrub.value = 0;
    scrub.setAttribute('aria-label', 'Time');
    bar.appendChild(scrub);
    var sel = document.createElement('select');
    sel.className = 'mzp-speed';
    sel.setAttribute('aria-label', 'Playback speed');
    [0.5, 1, 2, 4].forEach(function (m) {
      var op = document.createElement('option');
      op.value = m; op.textContent = m + '×';
      if (m === 1) op.selected = true;
      sel.appendChild(op);
    });
    bar.appendChild(sel);
    var stepEl = document.createElement('span');
    stepEl.className = 'mzp-step';
    bar.appendChild(stepEl);
    var scoreEl = null;
    if (!gridMode && o0.show_score && scenes[0].P.agents.length) {
      scoreEl = document.createElement('span');
      scoreEl.className = 'mzp-score' + (scenes[0].P.agents.length > 1 ? ' mzp-score-multi' : '');
      bar.appendChild(scoreEl);
    }
    if (T <= 0) bar.style.display = 'none';

    var btnState = '';
    function syncButton() {
      var s = playing ? 'pause' : (t >= T && T > 0 ? 'replay' : 'play');
      if (s === btnState) return;
      btnState = s;
      btn.innerHTML = s === 'pause' ? ICON_PAUSE : (s === 'replay' ? ICON_REPLAY : ICON_PLAY);
      btn.setAttribute('aria-label', s === 'pause' ? 'Pause' : (s === 'replay' ? 'Replay' : 'Play'));
    }
    function toggle() {
      if (T <= 0) return;
      if (playing) { playing = false; }
      else {
        if (t >= T) { t = 0; hold = START_HOLD; }
        playing = true; endHold = 0;
      }
      syncButton();
    }
    btn.addEventListener('click', toggle);
    scenes.forEach(function (s) { s.svg.addEventListener('click', toggle); });
    var scrubbing = false;
    scrub.addEventListener('input', function () {
      scrubbing = true;
      t = clamp(parseFloat(scrub.value) || 0, 0, T);
      playing = false; hold = 0; endHold = 0;
      syncButton();
      for (var i = 0; i < scenes.length; i++) scenes[i].update(t, mps(), 0.016, performance.now());
      syncControls();
    });
    scrub.addEventListener('change', function () { scrubbing = false; });
    sel.addEventListener('change', function () { mult = parseFloat(sel.value) || 1; });

    var lastStep = -1, lastScoreT = -1;
    function syncControls() {
      if (!scrubbing) scrub.value = t;
      var k = Math.min(T, Math.floor(t + 1e-6));
      if (k !== lastStep) {
        lastStep = k;
        stepEl.textContent = 'Step ' + k + ' / ' + T;
        if (scoreEl) scoreEl.textContent = scoreLine(scenes[0].P, t, 'Score ');
        scenes.forEach(function (s, i) {
          if (scoreEls[i]) scoreEls[i].textContent = scoreLine(s.P, t, 'Score ');
        });
      }
    }

    var visible = true;
    if (typeof IntersectionObserver !== 'undefined') {
      try {
        new IntersectionObserver(function (entries) {
          entries.forEach(function (e) { visible = e.isIntersecting; });
        }).observe(root);
      } catch (e) { /* ignore */ }
    }
    var last = null, wasConnected = false;
    function frame(now) {
      if (root.isConnected) wasConnected = true;
      else if (wasConnected) return; // output cleared: stop the loop
      var dt = last === null ? 0 : Math.min(0.1, (now - last) / 1000);
      last = now;
      if (playing && visible) {
        if (t <= 0 && hold > 0) hold -= dt;
        else if (t >= T) {
          endHold += dt;
          if (endHold >= END_HOLD) {
            endHold = 0;
            if (loop) { t = 0; hold = START_HOLD; }
            else { playing = false; syncButton(); }
          }
        } else t = Math.min(T, t + dt * mps());
      }
      if (visible) {
        for (var i = 0; i < scenes.length; i++) scenes[i].update(t, mps(), dt, now);
        syncControls();
      }
      requestAnimationFrame(frame);
    }
    syncButton();
    // draw the first frame synchronously so screenshots / static exports are not blank
    var now0 = (typeof performance !== 'undefined' ? performance.now() : 0);
    for (var i = 0; i < scenes.length; i++) scenes[i].update(0, mps(), 0, now0);
    syncControls();
    requestAnimationFrame(frame);

    // small hook for tests / debugging: root.__mzp.seek(t)
    root.__mzp = {
      seek: function (tt) {
        t = clamp(tt, 0, T); playing = false; hold = 0; endHold = 0; syncButton();
        for (var i = 0; i < scenes.length; i++) scenes[i].update(t, mps(), 10, performance.now());
        syncControls();
      },
      play: function () { if (!playing) toggle(); },
      pause: function () { if (playing) toggle(); },
      get t() { return t; },
      T: T
    };
  }

  return { mount: mount, _prepare: prepare, _agentPos: agentPos, _fmtValue: fmtValue };
})()
