"use strict";
const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const st = { book: null, sym: "BTC", interval: "1h", state: null, quotes: [], lastPx: {} };
const BOOK_COLORS = { confident: "--accent", active: "--accent2", rotation: "--warn" };

async function api(p, o = {}) {
  const r = await fetch(p, { credentials: "same-origin", ...o });
  const j = await r.json().catch(() => ({ error: r.statusText }));
  if (!r.ok) throw new Error(j.error || r.statusText);
  return j;
}
const n = (v, d = 2) => v == null ? "—" : Number(v).toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
const px = (v) => v == null ? "—" : n(v, v >= 1000 ? 2 : v >= 1 ? 4 : 5);
const usd = (v, d = 2) => v == null ? "—" : (v < 0 ? "-$" : "$") + n(Math.abs(v), d);
const sUsd = (v, d = 2) => v == null ? "—" : (v >= 0 ? "+$" : "-$") + n(Math.abs(v), d);
const pct = (v, d = 1) => v == null ? "—" : (v * 100).toFixed(d) + "%";
const sPct = (v, d = 2) => v == null ? "—" : (v >= 0 ? "+" : "") + (v * 100).toFixed(d) + "%";
const cls = (v) => v == null || Math.abs(v) < 1e-12 ? "" : v > 0 ? "up" : "down";
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
function set(id, text, c) { const e = document.getElementById(id); e.textContent = text; e.className = c || ""; }
function banner(m) { const b = $("#banner"); b.hidden = !m; b.textContent = m || ""; }
function toast(m) { const t = $("#toast"); t.textContent = m; t.hidden = false; clearTimeout(t._h); t._h = setTimeout(() => t.hidden = true, 3500); }
function err(e) { banner(/unauthor/i.test(e.message) ? "Session expired: reopen the URL with ?token=" : "Backend: " + e.message); }
function seg(el, cb) { el.addEventListener("click", (e) => { const b = e.target.closest("button"); if (!b) return;
  [...el.children].forEach((x) => x.classList.toggle("on", x === b)); cb(b); }); }

function mkChart(el, opts = {}) {
  return LightweightCharts.createChart(el, { autoSize: true,
    layout: { background: { type: "solid", color: css("--panel") }, textColor: css("--muted"), fontFamily: "JetBrains Mono", fontSize: 11 },
    grid: { vertLines: { color: css("--border") + "80" }, horzLines: { color: css("--border") + "80" } },
    rightPriceScale: { borderColor: css("--border") }, timeScale: { borderColor: css("--border"), timeVisible: true, secondsVisible: false },
    crosshair: { mode: 0 }, ...opts });
}
const eqChart = mkChart($("#eq-chart"));
const eqLines = {};
const bhLine = eqChart.addLineSeries({ color: css("--muted"), lineWidth: 1, lineStyle: 2, title: "buy&hold", priceLineVisible: false });
const pxChart = mkChart($("#px-chart"));
const candles = pxChart.addCandlestickSeries({ upColor: css("--up"), downColor: css("--down"), borderVisible: false, wickUpColor: css("--up"), wickDownColor: css("--down") });
const vol = pxChart.addHistogramSeries({ priceFormat: { type: "volume" }, priceScaleId: "" });
vol.priceScale().applyOptions({ scaleMargins: { top: 0.82, bottom: 0 } });
let accChart = null, accLine = null, accBase = null;

// ------------------------------------------------------------------ state + KPIs
async function loadState() {
  const s = await api("/api/state"); st.state = s; banner(s.live ? null : "Engine heartbeat is stale: the trading engine may not be running.");
  const names = Object.keys(s.books);
  if (!st.book || !s.books[st.book]) st.book = names.includes("active") ? "active" : names[0];
  const segEl = $("#book-seg");
  if (segEl.children.length !== names.length) {
    segEl.innerHTML = names.map((b) => `<button data-b="${b}">${b.toUpperCase()}${s.books[b].kind === "rotation" ? " · long-only" : " · ±" + s.books[b].threshold}</button>`).join("");
  }
  [...segEl.children].forEach((b) => b.classList.toggle("on", b.dataset.b === st.book));
  const lp = $("#live-pill");
  lp.textContent = s.live ? `● ENGINE LIVE · ${Math.round(s.heartbeat_age)}s` : "ENGINE DOWN"; lp.className = "pill " + (s.live ? "live" : "halted");
  const b = s.books[st.book]; if (!b) return;
  set("k-eq", usd(b.equity)); set("k-start", `start ${usd(b.start_capital, 0)} · ${b.start.slice(5, 16).replace("T", " ")} UTC`);
  set("k-pnl", sUsd(b.pnl), cls(b.pnl)); set("k-ret", sPct(b.ret, 3), cls(b.ret));
  set("k-alpha", sPct(b.ret - b.bh, 3), cls(b.ret - b.bh)); set("k-bh", `buy&hold 20 coins ${sPct(b.bh)}`);
  set("k-btc", sPct(b.btc), cls(b.btc)); set("k-hours", `${n(b.hours, 1)} h running`);
  set("k-gross", n(b.gross) + "× gross"); set("k-net", `net ${sPct(b.net, 1)} · ${b.n_long}L / ${b.n_short}S`);
  set("k-trades", String(b.trades)); set("k-fees", `fees ${usd(b.fees)} · realized ${sUsd(b.realized)}`);
  set("k-dd", sPct(b.drawdown), b.drawdown < -0.02 ? "down" : ""); set("k-status", b.halted ? "HALTED: " + b.halted : "trading");
  $("#pos-book").textContent = st.book; $("#resume-btn").hidden = !(s.kill || Object.values(s.books).some((x) => x.halted));
}
function tickClock() {
  const now = new Date(); $("#clock").textContent = now.toISOString().slice(11, 19) + " UTC";
  if (st.state) { const m = Math.max(0, (new Date(st.state.next_decision) - now) / 1000);
    $("#next-pill").textContent = `next decision in ${Math.floor(m / 60)}m ${String(Math.floor(m % 60)).padStart(2, "0")}s`; }
}
seg($("#book-seg"), (b) => { st.book = b.dataset.b; loadAll(); });

// ------------------------------------------------------------------ equity
async function loadEquity() {
  const e = await api("/api/equity"); let any = false;
  for (const [name, d] of Object.entries(e)) {
    if (!eqLines[name]) eqLines[name] = eqChart.addLineSeries({ color: css(BOOK_COLORS[name] || "--warn"), lineWidth: 2, title: name, priceLineVisible: false });
    eqLines[name].setData(d.time.map((t, i) => ({ time: t, value: d.equity[i] }))); any = any || d.time.length > 1;
  }
  const first = e.active || Object.values(e)[0];
  if (first) bhLine.setData(first.time.map((t, i) => ({ time: t, value: first.bh[i] })));
  $("#eq-empty").hidden = any;
  $("#eq-legend").innerHTML = Object.keys(e).map((k) => `<span style="color:${css(BOOK_COLORS[k] || "--warn")}">━ ${k}</span>`).join(" ") + ` <span>┅ buy&hold</span>`;
}

// ------------------------------------------------------------------ accuracy
async function loadAccuracy() {
  const a = await api("/api/accuracy");
  set("k-acc", a.n_scored ? pct(a.models.ens.acc) : "—", a.n_scored ? (a.models.ens.acc > a.up_rate ? "up" : "down") : "");
  set("k-accn", a.n_scored ? `${a.n_scored} scored · always-up ${pct(a.up_rate)}` : `${a.n_preds} made, scoring after 5h`);
  const tag = $("#acc-tag"); tag.textContent = `${a.n_preds} preds · ${a.n_hours}h`;
  if (!a.n_scored) { $("#acc-body").innerHTML = `<div class="empty">${a.n_preds} predictions logged so far. Each is scored when its 4-hour outcome is known (first scores ~5h after launch).</div>`; return; }
  const m = a.models, ens = m.ens.acc;
  const conf = Object.entries(a.confident).map(([k, v]) => `<tr><td>ens · |p−0.5|&gt;${k}</td><td class="r">${v.n}</td><td class="r">${v.acc == null ? "—" : pct(v.acc)}</td><td></td></tr>`).join("");
  $("#acc-body").innerHTML = `
    <div class="accbig"><b class="${ens > a.up_rate ? "up" : "down"}">${pct(ens)}</b><span class="muted">ensemble accuracy on ${a.n_scored} scored calls</span></div>
    <table class="tbl"><thead><tr><th>Model</th><th class="r">n</th><th class="r">Accuracy</th><th class="r">AUC</th></tr></thead><tbody>
      <tr><td>always up (baseline)</td><td class="r">${a.n_scored}</td><td class="r">${pct(a.up_rate)}</td><td class="r">0.500</td></tr>
      ${Object.entries(m).map(([k, v]) => `<tr><td>${k}</td><td class="r">${a.n_scored}</td><td class="r ${v.acc > a.up_rate ? "up" : "down"}">${pct(v.acc)}</td><td class="r">${v.auc == null ? "—" : n(v.auc, 3)}</td></tr>`).join("")}
      ${conf}</tbody></table>
    <div class="ph mt"><h2>Calibration</h2><span class="muted">bar = predicted · line = actual up-rate</span></div>
    ${(a.calibration || []).map((c) => `<div class="cal"><span>${esc(c.bin)}</span><div class="bars"><i class="pred" style="width:${(c.pred * 100).toFixed(0)}%"></i><i class="act" style="width:${(c.actual * 100).toFixed(0)}%"></i></div><span class="r">${pct(c.actual, 0)} (${c.n})</span></div>`).join("")}
    <div class="acc-note">Noise band at this sample size: ±${pct(a.noise_band)} around 50% (wider in practice: coins move together within an hour). The backtest edge was about +3 pts (52.6% vs 49.4%), so it takes thousands of scored calls to confirm or reject it.</div>`;
  if (!accChart) { accChart = mkChart($("#acc-chart"));
    accLine = accChart.addLineSeries({ color: css("--accent"), lineWidth: 2, title: "cumulative accuracy", priceLineVisible: false });
    accBase = accChart.addLineSeries({ color: css("--muted"), lineWidth: 1, lineStyle: 2, title: "50%", priceLineVisible: false }); }
  accLine.setData(a.by_hour.map((h) => ({ time: h.time, value: h.cum_acc * 100 })));
  accBase.setData(a.by_hour.map((h) => ({ time: h.time, value: 50 })));
  $("#hours tbody").innerHTML = a.by_hour.slice().reverse().map((h) => `<tr><td>${h.t.slice(0, 16)}</td><td class="r">${h.n}</td><td class="r">${pct(h.up, 0)}</td><td class="r ${h.acc >= 0.5 ? "up" : "down"}">${pct(h.acc, 0)}</td><td class="r ${cls(h.ret)}">${(h.ret * 1e4).toFixed(0)} bp</td></tr>`).join("");
}

// ------------------------------------------------------------------ price chart + model markers
async function loadBars() {
  const d = await api(`/api/bars?symbol=${st.sym}&interval=${st.interval}`);
  candles.setData(d.candles.map((c) => ({ time: c.time, open: c.open, high: c.high, low: c.low, close: c.close })));
  vol.setData(d.candles.map((c) => ({ time: c.time, value: c.volume, color: (c.close >= c.open ? css("--up") : css("--down")) + "50" })));
  const times = new Set(d.candles.map((c) => c.time));
  candles.setMarkers(st.interval !== "1h" ? [] : d.preds.filter((p) => times.has(p.time) && Math.abs(p.p - 0.5) > 0.02).map((p) => {
    const upCall = p.p > 0.5, right = p.up == null ? null : (p.up === 1) === upCall;
    return { time: p.time, position: upCall ? "belowBar" : "aboveBar", shape: upCall ? "arrowUp" : "arrowDown",
             color: css(upCall ? "--up" : "--down") + (p.up == null ? "88" : ""), text: `${(p.p * 100).toFixed(1)}%${right == null ? "" : right ? " ✓" : " ✗"}` };
  }));
  $("#px-title").textContent = `${st.sym}/USDT`;
}
seg($("#px-seg"), (b) => { st.interval = b.dataset.i; loadBars().catch(err); });
function pick(sym) { st.sym = sym; loadBars().catch(err); quoteLine(); }
function quoteLine() {
  const q = st.quotes.find((x) => x.symbol === st.sym);
  $("#px-quote").innerHTML = q ? `<span class="${cls(q.chg24)}">${px(q.last)} ${sPct(q.chg24)}</span> · bid ${px(q.bid)} / ask ${px(q.ask)} · model P(up 4h) <b>${q.p == null ? "—" : pct(q.p)}</b>` : "";
}

// ------------------------------------------------------------------ watchlist + signals
async function loadQuotes() {
  const q = await api("/api/quotes"); const prev = st.lastPx; st.quotes = q; st.lastPx = {};
  q.forEach((x) => st.lastPx[x.symbol] = x.last);
  $("#watch tbody").innerHTML = q.slice().sort((a, b) => (b.p ?? 0) - (a.p ?? 0)).map((x) => {
    const f = prev[x.symbol] && prev[x.symbol] !== x.last ? (x.last > prev[x.symbol] ? "flash-up" : "flash-down") : "";
    const pos = x.pos[st.book] || 0;
    return `<tr class="click ${f}" data-s="${x.symbol}"><td><b>${x.symbol}</b></td><td class="r">${px(x.last)}</td><td class="r ${cls(x.chg24)}">${sPct(x.chg24)}</td>
      <td class="r ${x.p == null ? "" : x.p > 0.5 ? "up" : "down"}">${x.p == null ? "—" : pct(x.p)}</td>
      <td class="r">${pos ? `<span class="poschip ${pos > 0 ? "L" : "S"}">${pos > 0 ? "LONG" : "SHORT"}</span>` : "·"}</td></tr>`; }).join("");
  quoteLine();
  const thr = st.state?.books[st.book]?.threshold ?? 0.04, withP = q.filter((x) => x.p != null).sort((a, b) => b.p - a.p);
  const span = Math.max(0.06, ...withP.map((x) => Math.abs(x.p - 0.5)));
  $("#signals").innerHTML = withP.map((x) => { const d = x.p - 0.5, w = Math.abs(d) / span * 50, conf = Math.abs(d) > thr;
    return `<div class="sig"><span>${x.symbol}</span><div class="bar"><i style="${d >= 0 ? `left:50%;width:${w}%` : `right:50%;width:${w}%`};background:${css(d >= 0 ? "--up" : "--down")}${conf ? "" : "55"}"></i>
      <span class="thr" style="left:${50 + thr / span * 50}%"></span><span class="thr" style="left:${50 - thr / span * 50}%"></span></div>
      <span class="r ${conf ? cls(d) : "muted"}">${pct(x.p)}</span></div>`; }).join("") || `<div class="empty">No predictions yet.</div>`;
  $("#sig-t").textContent = `dashed = ${st.book} threshold ±${thr}`;
}
$("#watch").addEventListener("click", (e) => { const tr = e.target.closest("tr[data-s]"); if (tr) pick(tr.dataset.s); });

// ------------------------------------------------------------------ positions / trades / preds / log / research
async function loadPositions() {
  const p = await api(`/api/positions?book_name=${st.book}`);
  $("#pos-empty").hidden = p.length > 0;
  $("#pos-sum").textContent = p.length ? `unrealized ${sUsd(p.reduce((a, r) => a + r.upl, 0))}` : "";
  $("#pos tbody").innerHTML = p.map((r) => `<tr class="click" data-s="${r.symbol}"><td><b>${r.symbol}</b></td>
    <td><span class="poschip ${r.qty > 0 ? "L" : "S"}">${r.side}</span></td><td class="r">${usd(Math.abs(r.value), 0)}</td><td class="r">${sPct(r.weight, 1)}</td>
    <td class="r">${px(r.avg)}</td><td class="r">${px(r.last)}</td><td class="r ${cls(r.upl)}">${sUsd(r.upl)}</td>
    <td class="r ${cls(r.uplpc)}">${sPct(r.uplpc)}</td><td class="r ${cls(r.chg24)}">${sPct(r.chg24)}</td></tr>`).join("");
}
$("#pos tbody").addEventListener("click", (e) => { const tr = e.target.closest("tr[data-s]"); if (tr) pick(tr.dataset.s); });
async function loadTrades() {
  const t = await api(`/api/trades?book_name=${st.book}`);
  $("#trades-empty").hidden = t.length > 0;
  $("#trades tbody").innerHTML = t.map((r) => `<tr><td>${esc(r.time.slice(5, 19).replace("T", " "))}</td><td><b>${esc(r.symbol)}</b></td>
    <td class="${r.side === "BUY" ? "up" : "down"}">${r.side}</td><td class="r">${n(r.qty, 4)}</td><td class="r">${px(r.price)}</td>
    <td class="r">${usd(r.notional)}</td><td class="r">${usd(r.fee)}</td><td class="r ${cls(r.realized)}">${r.realized ? sUsd(r.realized) : "—"}</td>
    <td class="r">${r.p_ens == null ? "—" : pct(r.p_ens)}</td><td class="muted">${esc(r.reason)}</td></tr>`).join("");
}
async function loadPreds() {
  const p = await api("/api/predictions?limit=500");
  $("#preds tbody").innerHTML = p.map((r) => { const call = r.p_ens > 0.5 ? "UP" : "DOWN", done = r.ret != null, ok = done && ((r.up === 1) === (r.p_ens > 0.5));
    return `<tr><td>${r.t.slice(0, 16)}</td><td><b>${r.symbol}</b></td><td class="r">${pct(r.p_xgb)}</td><td class="r">${pct(r.p_lgbm)}</td>
      <td class="r"><b>${pct(r.p_ens)}</b></td><td class="${call === "UP" ? "up" : "down"}">${call}</td>
      <td class="r ${done ? cls(r.ret) : ""}">${done ? (r.ret * 1e4).toFixed(0) + " bp" : "pending"}</td>
      <td class="${done ? (ok ? "up" : "down") : "muted"}">${done ? (ok ? "✓ right" : "✗ wrong") : "…"}</td></tr>`; }).join("");
}
async function loadLog() { const l = await api("/api/log?lines=300"); const c = $("#log"); c.textContent = l.join("\n"); c.scrollTop = c.scrollHeight; }
async function loadResearch() { const r = await api("/api/research"); $("#research").innerHTML = r.map((x) => marked.parse(x.text)).join("<hr>") || "<p class='muted'>No reports.</p>"; }

$("#tabs").addEventListener("click", (e) => { const b = e.target.closest("button"); if (!b) return;
  $$("#tabs button").forEach((x) => x.classList.toggle("on", x === b)); $$(".tab").forEach((t) => t.hidden = t.dataset.tab !== b.dataset.tab);
  ({ trades: loadTrades, preds: loadPreds, hours: loadAccuracy, log: loadLog, research: loadResearch })[b.dataset.tab]().catch(err); });
$("#kill-btn").onclick = async () => { if (!confirm("KILL: flatten all virtual positions and stop trading?")) return; await api("/api/kill", { method: "POST" }); toast("kill sent: the engine flattens on its next tick"); };
$("#resume-btn").onclick = async () => { if (!confirm("Resume trading?")) return; await api("/api/resume", { method: "POST" }); toast("resume sent"); };

function loadAll() { loadState().then(() => Promise.all([loadQuotes(), loadPositions(), loadTrades(), loadEquity(), loadAccuracy()])).catch(err); }
loadAll(); loadBars().catch(err);
setInterval(tickClock, 1000);
setInterval(() => loadState().catch(err), 5000);
setInterval(() => loadQuotes().catch(err), 4000);
setInterval(() => loadPositions().catch(err), 5000);
setInterval(() => loadTrades().catch(err), 20000);
setInterval(() => loadEquity().catch(err), 60000);
setInterval(() => loadAccuracy().catch(err), 60000);
setInterval(() => loadBars().catch(err), 30000);
setInterval(() => { if (!$("[data-tab=log].tab").hidden) loadLog().catch(err); if (!$("[data-tab=preds].tab").hidden) loadPreds().catch(err); }, 15000);
