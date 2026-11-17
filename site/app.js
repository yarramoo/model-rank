/* Parameter budget: filter open-weight models by a limit on total or active
   parameters and rank what fits by a public benchmark. No dependencies. */
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const SVG_NS = "http://www.w3.org/2000/svg";
  const SRC_TEXT = {
    hub: "Counted from the published weights",
    stated: "The maker's stated figure",
    name: "Nominal size from the model's name",
    description: "As described in the models.dev catalogue",
    third_party: "Third-party figure, not checked against the maker",
    estimate: "Estimated from the model's configuration",
  };
  const APPROX = new Set(["third_party", "estimate"]);

  const state = { mode: "total", limit: 32e9, bench: null, arch: "all", since: "", q: "", view: "rank", fitOnly: true };
  let DATA = null;
  let BENCH = new Map();
  let hot = null; // id of the model highlighted in both chart and table
  let view = null; // last computed view, reused by hover handlers
  let geom = null; // chart geometry for pointer maths

  /* ---------- small helpers ---------- */

  function el(tag, attrs, ...kids) {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (v === null || v === undefined || v === false) continue;
      if (k === "class") node.className = v;
      else if (k === "text") node.textContent = v;
      else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
      else node.setAttribute(k, v === true ? "" : v);
    }
    for (const kid of kids) if (kid !== null && kid !== undefined) node.append(kid);
    return node;
  }
  function svg(tag, attrs, text) {
    const node = document.createElementNS(SVG_NS, tag);
    for (const [k, v] of Object.entries(attrs || {})) if (v !== null && v !== undefined) node.setAttribute(k, v);
    if (text !== undefined) node.textContent = text;
    return node;
  }
  const trim = (n) => String(Number(n.toPrecision(3)));

  function fmtParams(n) {
    if (n === null || n === undefined) return "unknown";
    if (!isFinite(n)) return "no limit";
    if (n >= 1e12) return trim(n / 1e12) + "T";
    if (n >= 1e9) return trim(n / 1e9) + "B";
    return trim(n / 1e6) + "M";
  }
  function splitParams(n) {
    if (!isFinite(n)) return ["∞", ""];
    const s = fmtParams(n);
    return [s.slice(0, -1), s.slice(-1)];
  }
  function parseLimit(text) {
    const t = String(text).trim().toLowerCase();
    if (t === "" || t === "∞" || t === "inf" || t === "any" || t === "none" || t === "no limit") return Infinity;
    const m = t.match(/^(\d+(?:[.,]\d+)?)\s*([mbt])?$/);
    if (!m) return null;
    const num = parseFloat(m[1].replace(",", "."));
    if (!(num > 0)) return null;
    return num * { m: 1e6, b: 1e9, t: 1e12 }[m[2] || "b"];
  }
  function fmtScore(v, bench) {
    if (v === null || v === undefined) return "";
    if (bench && bench.unit === "Elo") return String(Math.round(v));
    return v.toFixed(1);
  }
  const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  const scoreText = (entry, bench) => (entry.est ? "~" : "") + fmtScore(entry.v, bench);
  function fmtDate(iso) {
    if (!iso) return "";
    const dateOnly = iso.length <= 10;
    const d = new Date(dateOnly ? iso + "T00:00:00Z" : iso);
    if (isNaN(d)) return iso;
    // Release dates are calendar dates; build times are shown in the reader's own time zone.
    return dateOnly
      ? `${d.getUTCDate()} ${MONTHS[d.getUTCMonth()]} ${d.getUTCFullYear()}`
      : `${d.getDate()} ${MONTHS[d.getMonth()]} ${d.getFullYear()}`;
  }
  function hostOf(url) {
    try { return new URL(url).hostname.replace(/^www\./, ""); } catch { return ""; }
  }
  const paramOf = (m) => (state.mode === "active" ? m.params_active : m.params_total);
  const srcOf = (m) => (state.mode === "active" ? m.params_active_src : m.params_total_src);
  const modeWord = () => (state.mode === "active" ? "active" : "total");

  /* ---------- state <-> URL ---------- */

  function readHash() {
    const p = new URLSearchParams(location.hash.slice(1));
    if (p.get("mode") === "active" || p.get("mode") === "total") state.mode = p.get("mode");
    if (p.has("limit")) { const v = parseLimit(p.get("limit")); if (v !== null) state.limit = v; }
    if (p.has("bench")) state.bench = p.get("bench");
    if (["all", "dense", "moe"].includes(p.get("arch"))) state.arch = p.get("arch");
    if (p.has("since")) state.since = p.get("since");
    if (p.has("q")) state.q = p.get("q");
    state.view = p.get("view") === "all" ? "all" : "rank";
    state.fitOnly = p.get("fit") !== "all";
    if (!["", "6", "12"].includes(state.since)) state.since = "";
  }
  function writeHash() {
    const p = new URLSearchParams();
    p.set("limit", isFinite(state.limit) ? fmtParams(state.limit) : "none");
    p.set("mode", state.mode);
    if (state.bench) p.set("bench", state.bench);
    if (state.arch !== "all") p.set("arch", state.arch);
    if (state.since) p.set("since", state.since);
    if (state.q) p.set("q", state.q);
    if (state.view === "all") p.set("view", "all");
    if (!state.fitOnly) p.set("fit", "all");
    history.replaceState(null, "", "#" + p.toString());
  }

  /* ---------- computing the view ---------- */

  function sinceDate() {
    if (!state.since) return null;
    const d = new Date(DATA.generated_at);
    d.setUTCMonth(d.getUTCMonth() - Number(state.since));
    return d.toISOString().slice(0, 10);
  }

  function compute() {
    const bench = BENCH.get(state.bench) || null;
    const since = sinceDate();
    const q = state.q.trim().toLowerCase();
    const base = DATA.models.filter((m) => {
      if (state.arch !== "all" && m.arch !== state.arch) return false;
      if (since && (!m.released || m.released < since)) return false;
      if (q && !(m.name + " " + m.org + " " + (m.hf_id || "")).toLowerCase().includes(q)) return false;
      return true;
    });
    const score = (m) => (bench && m.scores[bench.id] ? m.scores[bench.id].v : null);
    const dir = bench && bench.higher_is_better === false ? 1 : -1;

    const plotted = [];   // has a size and a score: shown on the chart
    const ranked = [];    // plotted and within the limit
    const fitsUnscored = [];
    const unknownSize = [];
    let over = 0;
    for (const m of base) {
      const p = paramOf(m);
      const s = score(m);
      if (p === null || p === undefined) { if (s !== null) unknownSize.push(m); continue; }
      if (s === null) { if (p <= state.limit) fitsUnscored.push(m); continue; }
      plotted.push(m);
      if (p <= state.limit) ranked.push(m); else over++;
    }
    ranked.sort((a, b) => dir * (score(a) - score(b)) || paramOf(a) - paramOf(b) || a.name.localeCompare(b.name));
    fitsUnscored.sort((a, b) => paramOf(b) - paramOf(a));
    unknownSize.sort((a, b) => dir * (score(a) - score(b)));
    const fits = (m) => { const p = paramOf(m); return p !== null && p !== undefined && p <= state.limit; };
    // One-benchmark table: scored models, either only those under the limit or all of them.
    const rankRows = state.fitOnly ? ranked
      : plotted.concat(unknownSize).sort((a, b) => dir * (score(a) - score(b)) || (paramOf(a) || 0) - (paramOf(b) || 0) || a.name.localeCompare(b.name));
    // All-benchmarks table: every model with at least one score of any kind.
    const fitsAny = base.filter((m) => Object.keys(m.scores).length && (!state.fitOnly || fits(m)));
    fitsAny.sort((a, b) => {
      const sa = score(a), sb = score(b);
      if (sa !== null && sb !== null) return dir * (sa - sb) || (paramOf(a) || 0) - (paramOf(b) || 0);
      if (sa !== null) return -1;
      if (sb !== null) return 1;
      return Object.keys(b.scores).length - Object.keys(a.scores).length || a.name.localeCompare(b.name);
    });
    return { bench, base, plotted, ranked, fitsUnscored, unknownSize, over, score, dir, fitsAny, rankRows, fits };
  }

  /* ---------- header controls ---------- */

  function renderHeader() {
    const [num, unit] = splitParams(state.limit);
    const input = $("limit-input");
    if (document.activeElement !== input) input.value = num;
    sizeInput();
    $("limit-unit").textContent = unit;
    for (const r of document.querySelectorAll('input[name="mode"]')) r.checked = r.value === state.mode;
    renderFitLabel();
    $("mode-note").textContent = state.mode === "active"
      ? "Active parameters are the weights used for each token. They equal the total for a dense model and are far lower for a mixture-of-experts model, so this limit tracks compute per token. Memory still follows the total."
      : "Total parameters are every weight in the model, which sets how much memory it needs to load.";
    document.title = `${isFinite(state.limit) ? "Under " + fmtParams(state.limit) : "Any size"}, ${modeWord()} parameters: open-weight model ranking`;
  }

  function renderFitLabel() {
    $("fit-only-label").textContent = isFinite(state.limit)
      ? `Only models under ${fmtParams(state.limit)} ${modeWord()}`
      : "Only models under the limit (no limit set)";
    $("fit-only").checked = state.fitOnly;
  }

  function sizeInput() {
    const input = $("limit-input");
    input.style.width = `calc(${Math.max(1, input.value.length)}ch + 0.12em)`;
  }

  function renderBenchNote(v) {
    const node = $("bench-note");
    node.textContent = "";
    const b = v.bench;
    if (!b) { node.textContent = "No benchmark data is available in this build."; return; }
    const parts = [];
    if (b.description) parts.push(b.description);
    if (b.source_id === "benchlm") { if (b.est_count) parts.push(`${b.est_count} of ${b.count} scores here are BenchLM estimates.`); }
    else if (b.self_count === 0) parts.push(`All ${b.count} scores were measured independently of the makers.`);
    else if (b.self_count === b.count) parts.push(`All ${b.count} scores are the makers' own reported results, so test settings differ between models.`);
    else parts.push(`${b.self_count} of ${b.count} scores are the makers' own reported results; the rest were measured independently.`);
    if (b.latest) parts.push(`Latest result ${fmtDate(b.latest)}.`);
    node.append(parts.join(" ") + " ");
    if (b.url) node.append(el("a", { href: b.url, rel: "noopener", text: "About this benchmark" }));
  }

  /* ---------- chart ---------- */

  function domainFor() {
    // Stable across filters: every model with a size in the current mode.
    let lo = Infinity, hi = 0;
    // Stable across filters and benchmarks: every model with a size and at least one score.
    for (const m of DATA.models) { const p = paramOf(m); if (p && Object.keys(m.scores).length) { lo = Math.min(lo, p); hi = Math.max(hi, p); } }
    if (!isFinite(lo)) { lo = 1e8; hi = 1e12; }
    return [Math.log10(lo) - 0.15, Math.log10(hi) + 0.2];
  }
  function niceTicks(lo, hi, n) {
    const span = hi - lo || 1;
    const raw = span / n;
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const step = [1, 2, 2.5, 5, 10].map((k) => k * mag).find((s) => s >= raw) || raw;
    const out = [];
    for (let t = Math.ceil(lo / step) * step; t <= hi + 1e-9; t += step) out.push(Number(t.toFixed(6)));
    return out;
  }

  function renderChart(v) {
    const root = $("chart");
    const wrap = $("chart-wrap");
    const W = Math.max(300, Math.round(wrap.clientWidth));
    const H = Math.round(Math.min(430, Math.max(250, W * 0.42)));
    const M = { l: 46, r: 14, t: 46, b: 38 };
    const pw = W - M.l - M.r, ph = H - M.t - M.b;
    root.setAttribute("viewBox", `0 0 ${W} ${H}`);
    root.setAttribute("width", W);
    root.setAttribute("height", H);
    root.textContent = "";

    const [xlo, xhi] = domainFor();
    const x = (p) => M.l + ((Math.log10(p) - xlo) / (xhi - xlo)) * pw;
    const xInv = (px) => Math.pow(10, xlo + ((px - M.l) / pw) * (xhi - xlo));
    const b = v.bench;
    const scores = v.plotted.map(v.score);
    let ylo = Math.min(...scores), yhi = Math.max(...scores);
    if (!scores.length) { ylo = 0; yhi = 1; }
    const pad = (yhi - ylo || Math.abs(yhi) * 0.1 || 1) * 0.12;
    ylo -= pad; yhi += pad;
    if (b && b.unit !== "Elo" && ylo < 0) ylo = 0;
    const y = (s) => M.t + ph - ((s - ylo) / (yhi - ylo)) * ph;

    const limitX = isFinite(state.limit) ? Math.min(M.l + pw, Math.max(M.l, x(state.limit))) : M.l + pw;

    // Over-the-limit zone
    if (limitX < M.l + pw - 0.5) {
      root.append(svg("rect", { class: "over-zone", x: limitX, y: M.t, width: M.l + pw - limitX, height: ph }));
      if (M.l + pw - limitX > 110) root.append(svg("text", { class: "over-label", x: M.l + pw - 8, y: M.t + 18, "text-anchor": "end" }, "Over the limit"));
    }

    // Grid and axes
    for (const t of niceTicks(ylo, yhi, 5)) {
      const yy = y(t);
      if (yy < M.t - 1 || yy > M.t + ph + 1) continue;
      root.append(svg("line", { class: "grid", x1: M.l, x2: M.l + pw, y1: yy, y2: yy }));
      root.append(svg("text", { x: M.l - 8, y: yy + 4, "text-anchor": "end" }, b && b.unit === "Elo" ? String(Math.round(t)) : trim(t)));
    }
    const dense = pw > 520;
    for (let e = Math.floor(xlo); e <= Math.ceil(xhi); e++) {
      for (const k of dense ? [1, 3] : [1]) {
        const p = k * Math.pow(10, e);
        const lx = Math.log10(p);
        if (lx < xlo || lx > xhi) continue;
        root.append(svg("line", { class: "axis-line", x1: x(p), x2: x(p), y1: M.t + ph, y2: M.t + ph + 5 }));
        root.append(svg("text", { x: x(p), y: M.t + ph + 19, "text-anchor": "middle" }, fmtParams(p)));
      }
    }
    root.append(svg("line", { class: "axis-line", x1: M.l, x2: M.l + pw, y1: M.t + ph, y2: M.t + ph }));
    root.append(svg("text", { class: "axis-title", x: M.l + pw, y: H - 4, "text-anchor": "end" }, `${state.mode === "active" ? "Active" : "Total"} parameters (log scale)`));
    if (b) root.append(svg("text", { class: "axis-title", x: 4, y: 12 }, `${b.name}${b.unit && b.unit !== "index" ? " (" + b.unit + ")" : ""}`));

    // Frontier: best score at or below each size
    const pts = v.plotted.map((m) => ({ m, px: x(paramOf(m)), py: y(v.score(m)), s: v.score(m), p: paramOf(m) })).sort((a, c) => a.p - c.p);
    if (pts.length > 1) {
      let best = null, d = "";
      for (const pt of pts) {
        if (best === null) { best = pt.s; d = `M${pt.px.toFixed(1)} ${pt.py.toFixed(1)}`; continue; }
        const improves = v.dir < 0 ? pt.s > best : pt.s < best;
        if (improves) { best = pt.s; d += ` H${pt.px.toFixed(1)} V${pt.py.toFixed(1)}`; }
      }
      d += ` H${(M.l + pw).toFixed(1)}`;
      root.append(svg("path", { class: "frontier", d }));
    }

    // Points (over-limit first so in-budget marks sit on top)
    const inBudget = new Set(v.ranked.map((m) => m.id));
    const ordered = pts.slice().sort((a, c) => Number(inBudget.has(a.m.id)) - Number(inBudget.has(c.m.id)));
    for (const pt of ordered) {
      const cls = `pt ${pt.m.arch === "moe" ? "moe" : "dense"}${inBudget.has(pt.m.id) ? "" : " is-over"}${hot === pt.m.id ? " is-hot" : ""}`;
      const mark = pt.m.arch === "moe"
        ? svg("rect", { class: cls, x: pt.px - 4.6, y: pt.py - 4.6, width: 9.2, height: 9.2, transform: `rotate(45 ${pt.px.toFixed(1)} ${pt.py.toFixed(1)})` })
        : svg("circle", { class: cls, cx: pt.px, cy: pt.py, r: 5 });
      mark.dataset.id = pt.m.id;
      root.append(mark);
    }

    // Direct labels for the top three that fit
    const boxes = [];
    const hit = (bx) => boxes.some((o) => bx.x < o.x + o.w && bx.x + bx.w > o.x && bx.y < o.y + o.h && bx.y + bx.h > o.y);
    for (const pt of pts) boxes.push({ x: pt.px - 6, y: pt.py - 6, w: 12, h: 12 });
    for (const m of v.ranked.slice(0, 3)) {
      const pt = pts.find((q) => q.m.id === m.id);
      const label = m.name.length > 28 ? m.name.slice(0, 27) + "…" : m.name;
      const node = svg("text", { class: "pt-label", x: 0, y: -100 }, label);
      root.append(node);
      const w = node.getComputedTextLength() + 6, hgt = 15;
      const tries = [
        { x: pt.px - w / 2, y: pt.py - 9 - hgt, a: "middle", tx: pt.px },
        { x: pt.px + 9, y: pt.py - hgt / 2, a: "start", tx: pt.px + 9 },
        { x: pt.px - 9 - w, y: pt.py - hgt / 2, a: "end", tx: pt.px - 9 },
        { x: pt.px - w / 2, y: pt.py + 9, a: "middle", tx: pt.px },
      ];
      const ok = tries.find((t) => t.x >= M.l && t.x + w <= W - 2 && t.y >= 2 && t.y + hgt <= M.t + ph && !hit({ x: t.x, y: t.y, w, h: hgt }));
      if (!ok) { node.remove(); continue; }
      boxes.push({ x: ok.x, y: ok.y, w, h: hgt });
      node.setAttribute("x", ok.tx);
      node.setAttribute("y", ok.y + 11.5);
      node.setAttribute("text-anchor", ok.a);
    }

    if (!pts.length) {
      root.append(svg("text", { class: "chart-empty", x: M.l + pw / 2, y: M.t + ph / 2, "text-anchor": "middle" }, b ? "No scored models match these filters." : "No benchmark data yet."));
    }

    // Limit handle (drawn last so it stays on top)
    const g = svg("g", {
      class: "limit-handle", tabindex: "0", role: "slider",
      "aria-label": `Limit on ${modeWord()} parameters`,
      "aria-valuemin": Math.round(xInv(M.l)), "aria-valuemax": Math.round(xInv(M.l + pw)),
      "aria-valuenow": isFinite(state.limit) ? Math.round(state.limit) : Math.round(xInv(M.l + pw)),
      "aria-valuetext": isFinite(state.limit) ? fmtParams(state.limit) : "No limit",
    });
    if (dragging) g.classList.add("is-dragging");
    g.append(svg("rect", { class: "hit", x: limitX - 20, y: M.t - 34, width: 40, height: ph + 34 }));
    g.append(svg("line", { class: "limit-line", x1: limitX, x2: limitX, y1: M.t - 8, y2: M.t + ph }));
    g.append(svg("rect", { class: "limit-foot", x: limitX - 5, y: M.t + ph - 3, width: 10, height: 6, rx: 2 }));
    // Pill: "◀ 32B limit ▶", sized to its text and kept inside the chart
    const label = svg("text", { class: "limit-text", x: 0, y: 0, "text-anchor": "middle" }, isFinite(state.limit) ? `${fmtParams(state.limit)} limit` : "No limit");
    g.append(label);
    root.append(g);
    const tw = label.getComputedTextLength() || 60;
    const gw = tw + 44, gh = 28;
    const cx = Math.max(gw / 2 + 2, Math.min(W - gw / 2 - 2, limitX));
    const py = M.t - 8 - gh;
    g.insertBefore(svg("rect", { class: "limit-grip", x: cx - gw / 2, y: py, width: gw, height: gh, rx: gh / 2 }), label);
    label.setAttribute("x", cx);
    label.setAttribute("y", py + gh / 2 + 4.5);
    const ay = py + gh / 2, lx = cx - gw / 2 + 11, rx = cx + gw / 2 - 11;
    g.append(svg("path", { class: "limit-arrow", d: `M${lx - 4} ${ay} L${lx + 2} ${ay - 5} L${lx + 2} ${ay + 5} Z` }));
    g.append(svg("path", { class: "limit-arrow", d: `M${rx + 4} ${ay} L${rx - 2} ${ay - 5} L${rx - 2} ${ay + 5} Z` }));

    geom = { M, pw, ph, W, H, x, xInv, pts, limitX };
    g.addEventListener("pointerdown", startDrag);
    g.addEventListener("keydown", keyLimit);
  }

  function snap(p) { return Number(p.toPrecision(2)); }

  function limitFromEvent(ev) {
    const rect = $("chart").getBoundingClientRect();
    const px = ((ev.clientX - rect.left) / rect.width) * geom.W;
    if (px >= geom.M.l + geom.pw - 3) return Infinity;
    return snap(geom.xInv(Math.max(geom.M.l, px)));
  }
  let dragging = false, raf = 0;
  function startDrag(ev) {
    ev.preventDefault();
    dragging = true;
    hideTip();
    ev.currentTarget.classList.add("is-dragging");
    const move = (e) => {
      const v = limitFromEvent(e);
      if (v === state.limit) return;
      state.limit = v;
      if (!raf) raf = requestAnimationFrame(() => { raf = 0; render(); });
    };
    const up = () => {
      dragging = false;
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
      window.removeEventListener("pointercancel", up);
      render();
      const h = document.querySelector(".limit-handle");
      if (h) h.focus({ preventScroll: true });
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
    window.addEventListener("pointercancel", up);
    move(ev);
  }
  function keyLimit(ev) {
    const steps = { ArrowLeft: -1, ArrowDown: -1, ArrowRight: 1, ArrowUp: 1, PageDown: -5, PageUp: 5 };
    const edgeLo = geom.xInv(geom.M.l), edgeHi = geom.xInv(geom.M.l + geom.pw);
    let next = null;
    if (ev.key in steps) {
      const cur = Math.log10(isFinite(state.limit) ? state.limit : edgeHi);
      let cand = Math.pow(10, cur + steps[ev.key] * (ev.shiftKey ? 0.1 : 0.025));
      // keep moving until the 2-significant-figure value changes
      let guard = 0;
      while (snap(cand) === state.limit && guard++ < 20) cand *= Math.pow(10, Math.sign(steps[ev.key]) * 0.01);
      next = cand >= edgeHi ? Infinity : snap(Math.max(edgeLo, cand));
    } else if (ev.key === "Home") next = snap(edgeLo);
    else if (ev.key === "End") next = Infinity;
    if (next === null) return;
    ev.preventDefault();
    state.limit = next;
    render();
    const h = document.querySelector(".limit-handle");
    if (h) h.focus({ preventScroll: true });
  }

  /* ---------- tooltip + cross-highlight ---------- */

  function setHot(id) {
    if (hot === id) return;
    hot = id;
    for (const n of document.querySelectorAll("#chart .pt")) n.classList.toggle("is-hot", n.dataset.id === id);
    for (const r of document.querySelectorAll("#tbody tr")) r.classList.toggle("is-hot", r.dataset.id === id);
  }
  function hideTip() { $("tooltip").hidden = true; }
  function showTip(pt) {
    const tip = $("tooltip");
    const m = pt.m, b = view.bench;
    tip.textContent = "";
    tip.append(
      el("span", { class: "tt-score", text: `${scoreText(m.scores[b.id], b)}${b.unit === "%" ? "%" : ""}` }),
      el("span", { class: "tt-name", text: m.name }),
      el("span", { class: "tt-row", text: m.org + (m.released ? `, released ${fmtDate(m.released)}` : "") }),
      el("span", { class: "tt-row", text: m.arch === "moe" ? `${fmtParams(m.params_total)} total, ${m.params_active ? fmtParams(m.params_active) + " active" : "active unknown"}` : `${fmtParams(m.params_total)} parameters, dense` }),
      el("span", { class: "tt-row", text: m.scores[b.id].est ? `BenchLM estimate, 90% range ${fmtScore(m.scores[b.id].lo, b)} to ${fmtScore(m.scores[b.id].hi, b)}`
        : b.source_id === "benchlm" ? `Composite by BenchLM, 90% range ${fmtScore(m.scores[b.id].lo, b)} to ${fmtScore(m.scores[b.id].hi, b)}`
        : m.scores[b.id].self ? "Score reported by the maker" : "Score measured independently" }),
    );
    tip.hidden = false;
    const wrap = $("chart-wrap").getBoundingClientRect();
    const scale = wrap.width / geom.W;
    const tw = tip.offsetWidth, th = tip.offsetHeight;
    let left = pt.px * scale + 12, top = pt.py * scale - th - 10;
    if (left + tw > wrap.width) left = pt.px * scale - tw - 12;
    if (left < 0) left = 0;
    if (top < 0) top = pt.py * scale + 12;
    tip.style.left = left + "px";
    tip.style.top = top + "px";
  }
  function onChartMove(ev) {
    if (dragging || !geom || !geom.pts.length) return;
    const rect = $("chart").getBoundingClientRect();
    const px = ((ev.clientX - rect.left) / rect.width) * geom.W;
    const py = ((ev.clientY - rect.top) / rect.height) * geom.H;
    let best = null, bd = 30 * 30;
    for (const pt of geom.pts) {
      const d = (pt.px - px) ** 2 + (pt.py - py) ** 2;
      if (d < bd) { bd = d; best = pt; }
    }
    if (best) { setHot(best.m.id); showTip(best); } else { setHot(null); hideTip(); }
  }

  /* ---------- table ---------- */

  function sizeCell(m, which) {
    const val = which === "active" ? m.params_active : m.params_total;
    const src = which === "active" ? m.params_active_src : m.params_total_src;
    const td = el("td", { class: which === state.mode ? "is-limit" : "" });
    if (val === null || val === undefined) { td.textContent = "unknown"; td.classList.add("muted"); return td; }
    if (which === "active" && m.arch !== "moe") { td.textContent = "same"; td.classList.add("muted"); td.title = "Dense model: every parameter is used for each token"; return td; }
    td.textContent = (APPROX.has(src) ? "~" : "") + fmtParams(val);
    td.title = (SRC_TEXT[src] || "") + (m.params_note && (src === "stated" || src === "third_party") ? ` (${m.params_note})` : "");
    return td;
  }

  function modelCell(m, entry) {
    const link = m.url ? el("a", { href: m.url, rel: "noopener", text: m.name }) : document.createTextNode(m.name);
    const notes = [m.org];
    if (m.unlisted) notes.push(`listed by ${m.unlisted} only`);
    if (entry && entry.label) notes.push(`listed as “${entry.label}”`);
    return el("td", { class: "col-model" },
      el("span", { class: "model-name" }, link),
      el("span", { class: "tag" + (m.arch === "moe" ? " moe" : ""), text: m.arch === "moe" ? "MoE" : "Dense" }),
      el("span", { class: "model-sub", text: notes.join(", ") }));
  }

  /* The all-benchmarks table: one row per model that fits, one column per benchmark. */
  function renderMatrix(v) {
    const thead = $("thead"), tbody = $("tbody");
    const rows = v.fitsAny;
    const cols = DATA.benchmarks.filter((b) => rows.some((m) => m.scores[b.id]));
    const lim = isFinite(state.limit) ? `under ${fmtParams(state.limit)} ${modeWord()} parameters` : "at any size";
    const under = rows.filter(v.fits).length;
    $("ranking-summary").textContent = state.fitOnly
      ? `${rows.length} models ${lim} have at least one score, across ${cols.length} benchmarks.`
      : `${rows.length} models with at least one score, ${under} of them ${lim}, across ${cols.length} benchmarks.`;
    $("matrix-key").hidden = !rows.length;
    if (!rows.length) return false;

    const head = el("tr", {},
      el("th", { class: "col-rank", scope: "col", text: "#" }),
      el("th", { class: "col-model", scope: "col", text: "Model" }),
      el("th", { scope: "col", class: state.mode === "total" ? "is-limit" : "", text: "Total" }),
      el("th", { scope: "col", class: state.mode === "active" ? "is-limit" : "", text: "Active" }));
    const range = new Map();
    for (const b of cols) {
      const vals = rows.filter((m) => m.scores[b.id]).map((m) => m.scores[b.id].v);
      range.set(b.id, [Math.min(...vals), Math.max(...vals), vals.length]);
      const th = el("th", { scope: "col", class: "col-bench" + (b.id === state.bench ? " is-sort" : "") },
        el("button", { type: "button", title: `${b.name}. ${b.description || ""} Select to rank by it.`, text: b.short || b.name,
          "aria-pressed": String(b.id === state.bench), onclick: () => { state.bench = b.id; syncFilters(); render(); } }),
        el("span", { class: "bench-count", text: `${vals.length} of ${rows.length}` }));
      head.append(th);
    }
    thead.append(head);

    rows.forEach((m, i) => {
      const tr = el("tr", {},
        el("td", { class: "col-rank", text: v.score(m) !== null ? String(i + 1) : "" }),
        modelCell(m, v.bench && m.scores[v.bench.id]),
        sizeCell(m, "total"), sizeCell(m, "active"));
      if (!v.fits(m)) tr.classList.add("is-over-row");
      for (const b of cols) {
        const e = m.scores[b.id];
        const td = el("td", { class: "cell" + (b.id === state.bench ? " is-sort" : "") });
        if (e) {
          const [lo, hi] = range.get(b.id);
          const t = hi > lo ? (e.v - lo) / (hi - lo) : 1;
          const share = b.higher_is_better === false ? 1 - t : t;
          td.style.setProperty("--heat", (6 + share * 34).toFixed(0) + "%");
          td.classList.add("has-score");
          if (e.self) td.classList.add("is-self");
          td.textContent = scoreText(e, b);
          td.title = `${b.name}: ${e.self ? "reported by the maker" : e.est ? "BenchLM estimate" : "measured independently"}${e.label ? `, listed as “${e.label}”` : ""}`;
        }
        tr.append(td);
      }
      tr.dataset.id = m.id;
      if (hot === m.id) tr.classList.add("is-hot");
      tr.addEventListener("pointerenter", () => setHot(m.id));
      tr.addEventListener("pointerleave", () => setHot(null));
      tbody.append(tr);
    });
    return true;
  }

  function renderTable(v) {
    const b = v.bench;
    const thead = $("thead"), tbody = $("tbody");
    thead.textContent = "";
    tbody.textContent = "";
    const empty = $("empty");
    const scroll = $("table-scroll");
    const all = state.view === "all";
    for (const r of document.querySelectorAll('input[name="view"]')) r.checked = r.value === state.view;
    $("table").classList.toggle("matrix", all);
    $("matrix-key").hidden = true;
    if (all && renderMatrix(v)) {
      scroll.hidden = false;
      empty.hidden = true;
      $("unscored").hidden = true;
      return;
    }

    const summary = $("ranking-summary");
    const scoredTotal = v.ranked.length + v.over;
    const rows = v.rankRows;
    const lim = isFinite(state.limit) ? `under ${fmtParams(state.limit)} ${modeWord()} parameters` : "at any size";
    summary.textContent = !b ? ""
      : state.fitOnly ? `${v.ranked.length} of ${scoredTotal} scored models fit ${lim}, ranked by ${b.name}.`
      : `${rows.length} scored models, ${v.ranked.length} of them ${lim}, ranked by ${b.name}.`;

    if (!rows.length || all) {
      scroll.hidden = true;
      empty.hidden = false;
      empty.textContent = "";
      if (!b) {
        empty.append(el("strong", { text: "No benchmark data yet" }), "The data build has not produced any scores. Check the pipeline log for the sources that failed.");
      } else if (scoredTotal > 0) {
        const smallest = v.plotted.reduce((a, m) => (paramOf(m) < paramOf(a) ? m : a));
        empty.append(
          el("strong", { text: `Nothing with a ${b.name} score fits ${lim}` }),
          `The smallest scored model is ${smallest.name} at ${fmtParams(paramOf(smallest))}.`,
          el("br"),
          el("button", { type: "button", text: `Raise the limit to ${fmtParams(snap(paramOf(smallest) * 1.0001))}`, onclick: () => { state.limit = Math.max(paramOf(smallest), snap(paramOf(smallest) * 1.0001)); render(); } }),
        );
        if (state.mode === "total") empty.append(el("button", { type: "button", text: "Count active parameters instead", onclick: () => { state.mode = "active"; render(); } }));
      } else {
        empty.append(el("strong", { text: "No scored models match these filters" }), "Clear the search or widen the release window, or rank by a benchmark with more results.",
          el("br"), el("button", { type: "button", text: "Clear filters", onclick: () => { state.q = ""; state.arch = "all"; state.since = ""; syncFilters(); render(); } }));
        if (state.since) empty.append(el("button", { type: "button", text: "Show any release date", onclick: () => { state.since = ""; syncFilters(); render(); } }));
      }
    } else {
      scroll.hidden = false;
      empty.hidden = true;
      const head = el("tr", {},
        el("th", { class: "col-rank", scope: "col", text: "#" }),
        el("th", { class: "col-model", scope: "col", text: "Model" }),
        el("th", { scope: "col", class: state.mode === "total" ? "is-limit" : "", text: "Total" }),
        el("th", { scope: "col", class: state.mode === "active" ? "is-limit" : "", text: "Active" }),
        el("th", { scope: "col", class: "is-sort", text: b.name + (b.unit === "%" ? ", %" : "") }),
        el("th", { scope: "col", text: "Measured by" }),
        el("th", { scope: "col", text: "Released" }),
      );
      thead.append(head);
      const vals = rows.map(v.score);
      const top = Math.max(...vals), bottom = Math.min(...vals);
      const base = b.unit === "Elo" ? bottom - (top - bottom || 1) * 0.25 : 0;
      rows.forEach((m, i) => {
        const s = m.scores[b.id];
        const pct = Math.max(2, Math.min(100, ((s.v - base) / ((top - base) || 1)) * 100));
        const scoreCell = el("td", { class: "is-sort" },
          el("span", { class: "score-cell" }, el("span", { text: scoreText(s, b) }), el("span", { class: "score-bar", "aria-hidden": "true" }, el("i", { style: `width:${pct.toFixed(1)}%` }))));
        const by = el("td", { class: s.self ? "" : "muted" });
        const who = s.self ? "The maker" : (hostOf(s.src) || "Independent") + (s.est ? ", estimate" : "");
        if (s.src) by.append(el("a", { href: s.src, rel: "noopener", text: who })); else by.textContent = who;
        const tr = el("tr", {},
          el("td", { class: "col-rank", text: String(i + 1) }),
          modelCell(m, s),
          sizeCell(m, "total"),
          sizeCell(m, "active"),
          scoreCell,
          by,
          el("td", { class: "muted", text: m.released ? fmtDate(m.released) : "" }),
        );
        if (!v.fits(m)) tr.classList.add("is-over-row");
        tr.dataset.id = m.id;
        if (hot === m.id) tr.classList.add("is-hot");
        tr.addEventListener("pointerenter", () => setHot(m.id));
        tr.addEventListener("pointerleave", () => setHot(null));
        tbody.append(tr);
      });
    }

    // What the ranking leaves out, and why
    const det = $("unscored"), list = $("unscored-list");
    list.textContent = "";
    const notes = [];
    if (v.unknownSize.length && state.fitOnly) notes.push(`${v.unknownSize.length} scored ${v.unknownSize.length === 1 ? "model has" : "models have"} no known ${state.mode === "active" ? "active " : ""}parameter count`);
    if (v.fitsUnscored.length) notes.push(`${v.fitsUnscored.length} more ${v.fitsUnscored.length === 1 ? "model fits" : "models fit"} the limit but ${v.fitsUnscored.length === 1 ? "has" : "have"} no ${b ? b.name : ""} score`);
    det.hidden = !notes.length;
    $("unscored-summary").textContent = "Left out: " + notes.join("; ") + ".";
    for (const m of state.fitOnly ? v.unknownSize : []) {
      list.append(el("li", {}, m.url ? el("a", { href: m.url, rel: "noopener", text: m.name }) : m.name, ` scored ${fmtScore(v.score(m), b)}, size unknown`));
    }
    for (const m of v.fitsUnscored.slice(0, 80)) {
      list.append(el("li", {}, m.url ? el("a", { href: m.url, rel: "noopener", text: m.name }) : m.name, ` ${fmtParams(paramOf(m))}, no score`));
    }
    if (v.fitsUnscored.length > 80) list.append(el("li", { text: `and ${v.fitsUnscored.length - 80} more` }));
  }

  /* ---------- footer ---------- */

  function renderSources() {
    const extra = DATA.counts.from_leaderboards_only ? ` ${DATA.counts.from_leaderboards_only} of them are known only from a leaderboard, with the size taken from the name.` : "";
    $("updated").textContent = `Data built ${fmtDate(DATA.generated_at)}. ${DATA.counts.models} open-weight models listed, ${DATA.counts.with_size} with a known size, ${DATA.counts.with_any_score} with at least one score.${extra}`;
    const ul = $("sources");
    ul.textContent = "";
    for (const s of DATA.sources) {
      const status = s.status === "ok" ? `Read ${fmtDate(s.fetched_at)}.`
        : s.status === "stale" ? `Stale since ${fmtDate(s.fetched_at)}.`
        : s.status === "skipped" ? "Not used in this build."
        : "Could not be read in this build; the pipeline log has the reason.";
      const li = el("li", {},
        el("strong", {}, el("a", { href: s.url, rel: "noopener", text: s.name })),
        `${s.role} ${s.licence}. `,
        el("span", { class: s.status === "ok" ? "" : "stale", text: status }),
        s.status === "ok" || s.status === "failed" || /^Skipped for this build/.test(s.detail) ? "" : ` ${s.detail}`,
      );
      ul.append(li);
    }
  }

  /* ---------- wiring ---------- */

  function syncFilters() {
    $("bench").value = state.bench || "";
    $("arch").value = state.arch;
    for (const r of document.querySelectorAll('input[name="since"]')) r.checked = r.value === state.since;
    $("q").value = state.q;
  }

  function render() {
    view = compute();
    renderHeader();
    renderBenchNote(view);
    renderChart(view);
    renderTable(view);
    writeHash();
  }

  function init(data) {
    DATA = data;
    BENCH = new Map(data.benchmarks.map((b) => [b.id, b]));
    readHash();
    if (!BENCH.has(state.bench)) state.bench = data.default_benchmark || (data.benchmarks[0] && data.benchmarks[0].id) || null;

    const sel = $("bench");
    const groups = [["broad", "Whole-model comparisons"], ["task", "Single benchmarks"]];
    for (const [key, label] of groups) {
      const members = data.benchmarks.filter((b) => (b.group || "task") === key);
      if (!members.length) continue;
      const og = el("optgroup", { label });
      for (const b of members) og.append(el("option", { value: b.id, text: `${b.name} (${b.sized_count} models)` }));
      sel.append(og);
    }
    for (const r of document.querySelectorAll('input[name="view"]')) r.addEventListener("change", () => { state.view = r.value; render(); });
    for (const r of document.querySelectorAll('input[name="since"]')) r.addEventListener("change", () => { state.since = r.value; render(); });
    $("fit-only").addEventListener("change", (e) => { state.fitOnly = e.target.checked; render(); });

    const input = $("limit-input");
    const commit = () => {
      const v = parseLimit(input.value);
      if (v !== null) state.limit = v;
      input.blur();
      render();
    };
    input.addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); commit(); } });
    input.addEventListener("change", commit);
    input.addEventListener("input", () => {
      sizeInput();
      const v = parseLimit(input.value);
      if (v !== null && v !== state.limit) { state.limit = v; view = compute(); renderFitLabel(); renderBenchNote(view); renderChart(view); renderTable(view); writeHash(); $("limit-unit").textContent = splitParams(v)[1]; }
    });
    input.addEventListener("focus", () => input.select());
    for (const r of document.querySelectorAll('input[name="mode"]')) r.addEventListener("change", () => { state.mode = r.value; render(); });
    sel.addEventListener("change", () => { state.bench = sel.value; render(); });
    $("arch").addEventListener("change", (e) => { state.arch = e.target.value; render(); });
    $("q").addEventListener("input", (e) => { state.q = e.target.value; render(); });

    const chart = $("chart");
    chart.addEventListener("pointermove", onChartMove);
    chart.addEventListener("pointerleave", () => { setHot(null); hideTip(); });
    new ResizeObserver(() => { if (view) renderChart(view); }).observe($("chart-wrap"));
    window.addEventListener("hashchange", () => { readHash(); syncFilters(); render(); });

    syncFilters();
    renderSources();
    render();
  }

  fetch("data/models.json", { cache: "no-cache" })
    .then((r) => { if (!r.ok) throw new Error("HTTP " + r.status); return r.json(); })
    .then(init)
    .catch((err) => {
      const box = $("empty");
      box.hidden = false;
      box.textContent = "";
      box.append(el("strong", { text: "The model data could not be loaded" }),
        `data/models.json did not load (${err.message}). If you opened this page straight from disk, serve the folder instead: python3 -m http.server --directory public`);
      $("table-scroll").hidden = true;
    });
})();
