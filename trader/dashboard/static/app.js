"use strict";
const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();

const st = { book: null, eqPeriod: "1M", pxSym: "SPY", pxTf: "1D", posFilter: "all", sortK: "value", sortDir: -1,
             positions: [], lastPx: {}, trendWt: {}, overview: null, jobRunning: false };

// ------------------------------------------------------------------ helpers
async function api(path, opts = {}) {
  const r = await fetch(path, { credentials: "same-origin", ...opts });
  const j = await r.json().catch(() => ({ error: r.statusText }));
  if (!r.ok) throw new Error(j.detail || j.error || r.statusText);
  return j;
}
const n = (v, d = 2) => v == null ? "—" : Number(v).toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
const usd = (v, d = 2) => v == null ? "—" : (v < 0 ? "-$" : "$") + n(Math.abs(v), d);
const sUsd = (v, d = 2) => v == null ? "—" : (v >= 0 ? "+" : "-") + "$" + n(Math.abs(v), d);
const pct = (v, d = 2) => v == null ? "—" : (v * 100).toFixed(d) + "%";
const sPct = (v, d = 2) => v == null ? "—" : (v >= 0 ? "+" : "") + (v * 100).toFixed(d) + "%";
const cls = (v) => v == null || v === 0 ? "" : v > 0 ? "up" : "down";
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
function set(id, text, c) { const el = document.getElementById(id); el.textContent = text; el.className = c || ""; }
function toast(msg, ms = 3500) { const t = $("#toast"); t.textContent = msg; t.hidden = false; clearTimeout(t._h); t._h = setTimeout(() => t.hidden = true, ms); }
function banner(msg) { const b = $("#banner"); b.hidden = !msg; b.textContent = msg || ""; }
function etOffset(ts) {
  const d = new Date(ts * 1000);
  return (new Date(d.toLocaleString("en-US", { timeZone: "America/New_York" })) - new Date(d.toLocaleString("en-US", { timeZone: "UTC" }))) / 1000;
}
const toET = (ts) => ts + etOffset(ts);
function seg(el, onPick) {
  el.addEventListener("click", (e) => {
    const b = e.target.closest("button"); if (!b) return;
    [...el.children].forEach((x) => x.classList.toggle("on", x === b)); onPick(b);
  });
}

// ------------------------------------------------------------------ charts
function mkChart(el) {
  return LightweightCharts.createChart(el, {
    autoSize: true,
    layout: { background: { type: "solid", color: css("--panel") }, textColor: css("--muted"), fontFamily: "JetBrains Mono", fontSize: 11 },
    grid: { vertLines: { color: css("--border") + "80" }, horzLines: { color: css("--border") + "80" } },
    rightPriceScale: { borderColor: css("--border") }, timeScale: { borderColor: css("--border"), timeVisible: true },
    crosshair: { mode: 0 },
  });
}
const eqChart = mkChart($("#eq-chart"));
const eqSeries = eqChart.addAreaSeries({ lineColor: css("--accent"), topColor: css("--accent") + "55", bottomColor: css("--accent") + "05", lineWidth: 2 });
const pxChart = mkChart($("#px-chart"));
const candles = pxChart.addCandlestickSeries({ upColor: css("--up"), downColor: css("--down"), borderVisible: false, wickUpColor: css("--up"), wickDownColor: css("--down") });
const vol = pxChart.addHistogramSeries({ priceFormat: { type: "volume" }, priceScaleId: "" });
vol.priceScale().applyOptions({ scaleMargins: { top: 0.82, bottom: 0 } });
let rsChart = null;

// ------------------------------------------------------------------ overview / KPIs / risk
async function loadOverview() {
  const o = await api(`/api/overview?book=${st.book}`);
  st.overview = o; banner(null);
  const a = o.account, x = o.exposure, r = o.risk;
  const mp = $("#mode-pill"); mp.textContent = o.mode.toUpperCase() + " MODE"; mp.className = "pill " + o.mode;
  set("k-equity", usd(a.equity)); set("k-acct", `paper · ${a.status.replace("AccountStatus.", "")}`);
  set("k-day", sUsd(o.day_pnl), cls(o.day_pnl)); set("k-daypct", sPct(o.day_pnl_pct), cls(o.day_pnl_pct));
  set("k-upl", sUsd(o.book_upl), cls(o.book_upl)); set("k-upltoday", "today " + sUsd(o.book_upl_today), cls(o.book_upl_today));
  set("k-gross", n(x.gross) + "×"); set("k-ls", `L ${pct(x.long, 0)} · S ${pct(-x.short, 0)}`);
  set("k-net", sPct(x.net, 1), cls(x.net));
  set("k-dd", pct(r.drawdown), r.drawdown < -0.05 ? "down" : ""); set("k-peak", "peak " + usd(r.peak_equity, 0));
  set("k-npos", x.n_long + x.n_short); set("k-nls", `${x.n_long} long · ${x.n_short} short`);
  set("k-bp", usd(a.buying_power, 0)); set("k-cash", "cash " + usd(a.cash, 0));

  const m = o.market, mk = $("#market-pill");
  const target = new Date(m.is_open ? m.next_close : m.next_open), mins = Math.max(0, (target - Date.now()) / 60000);
  const h = Math.floor(mins / 60), mm = Math.floor(mins % 60);
  mk.textContent = m.is_open ? `● OPEN · closes in ${h}h ${mm}m` : `CLOSED · opens in ${h}h ${mm}m`;
  mk.className = "pill " + (m.is_open ? "live" : "closed");

  const L = r.limits, dayLoss = Math.max(0, -(o.day_pnl_pct || 0));
  const bars = [["Gross leverage", x.gross, L.max_gross_leverage, "×"], ["Net exposure", Math.abs(x.net), L.max_net_exposure_frac, "%"],
                ["Largest position", x.max_name, L.max_position_frac, "%"], ["Day loss", dayLoss, L.daily_loss_frac, "%"],
                ["Drawdown", -r.drawdown, L.max_drawdown_frac, "%"]];
  $("#risk-bars").innerHTML = bars.map(([lbl, v, lim, u]) => {
    const f = Math.min(1, (v || 0) / lim), c = f > 0.9 ? "bad" : f > 0.7 ? "warn" : "";
    const fmt = (z) => u === "×" ? n(z) + "×" : pct(z, 1);
    return `<div class="rbar"><div class="lbl"><span>${lbl}</span><span>${fmt(v || 0)} / ${fmt(lim)}</span></div>
            <div class="track"><div class="fill ${c}" style="width:${(f * 100).toFixed(1)}%"></div></div></div>`;
  }).join("");
  const rs = $("#risk-state");
  rs.textContent = r.halted ? "HALTED" : "ARMED"; rs.className = "tag " + (r.halted ? "bad" : "ok");
  $("#resume-btn").hidden = !r.halted;
  if (r.halted) { mp.textContent = "HALTED"; mp.className = "pill halted"; }
  const sc = o.scheduler, md = o.model;
  $("#sys-kv").innerHTML = `
    <dt>Halt reason</dt><dd title="${esc(r.halt_reason)}">${esc(r.halt_reason || "—")}</dd>
    <dt>Scheduler</dt><dd>${sc.enabled ? "ON" : "OFF"} <button id="sched-toggle">${sc.enabled ? "disable" : "enable"}</button></dd>
    <dt>Next daily run</dt><dd>${sc.next ? new Date(sc.next).toLocaleString("en-US", { timeZone: "America/New_York", weekday: "short", hour: "2-digit", minute: "2-digit" }) + " ET" : "—"}</dd>
    <dt>Last run</dt><dd>${sc.last_run || "—"}</dd>
    <dt>Model</dt><dd>${md ? esc(md.file) : "none: train first"}</dd>
    <dt>Model data through</dt><dd>${md ? esc(md.data_end) : "—"}</dd>
    <dt>Sleeves</dt><dd>ML ${pct(o.sleeve_alloc.stock_ml, 0)} · Trend ${pct(o.sleeve_alloc.etf_trend, 0)}</dd>`;
  $("#sched-toggle").onclick = async () => { await api(`/api/scheduler?enabled=${!sc.enabled}`, { method: "POST" }); loadOverview(); };

  $("#sleeves").innerHTML = [["ml", "Stock ML (long/short)"], ["trend", "Cross-asset trend"]].map(([k, name]) => {
    const s = o.sleeves[k] || { long: 0, short: 0, n: 0, upl_today: 0 }, tot = s.long - s.short || 1;
    return `<div class="sl"><div class="top"><span><span class="chip ${k}">${k.toUpperCase()}</span> ${name}</span>
      <span class="${cls(s.upl_today)}">${sUsd(s.upl_today, 0)}</span></div>
      <div class="top muted"><span>${s.n} pos · L ${pct(s.long, 0)} / S ${pct(-s.short, 0)}</span><span>net ${sPct(s.long + s.short, 1)}</span></div>
      <div class="lsbar"><div class="l" style="width:${(s.long / tot * 100).toFixed(1)}%"></div><div class="s" style="width:${(-s.short / tot * 100).toFixed(1)}%"></div></div></div>`;
  }).join("");

  const j = o.job; $("#job-dot").className = j.running ? "run" : "";
  if (j.running && !st.jobRunning) pollJob();
}

// ------------------------------------------------------------------ equity
async function loadEquity() {
  const e = await api(`/api/equity?period=${st.eqPeriod}`);
  const pts = e.points.map((p) => ({ time: toET(p.time), value: p.equity }));
  $("#eq-empty").hidden = pts.length > 1;
  eqSeries.setData(pts); eqChart.timeScale().fitContent();
}
seg($("#eq-seg"), (b) => { st.eqPeriod = b.textContent; loadEquity().catch(err); });

// ------------------------------------------------------------------ price chart
async function loadBars() {
  const d = await api(`/api/bars?symbol=${encodeURIComponent(st.pxSym)}&tf=${st.pxTf}`);
  candles.setData(d.map((b) => ({ time: toET(b.time), open: b.open, high: b.high, low: b.low, close: b.close })));
  vol.setData(d.map((b) => ({ time: toET(b.time), value: b.volume, color: (b.close >= b.open ? css("--up") : css("--down")) + "55" })));
  pxChart.timeScale().fitContent(); $("#px-title").textContent = st.pxSym;
  pxChart.applyOptions({ timeScale: { timeVisible: st.pxTf !== "1D" } });
}
function pickSymbol(sym) { st.pxSym = sym; loadBars().catch(err); refreshQuoteLine(); window.scrollTo({ top: $("#px-chart").getBoundingClientRect().top + scrollY - 140, behavior: "smooth" }); }
seg($("#px-seg"), (b) => { st.pxTf = b.textContent; loadBars().catch(err); });
function refreshQuoteLine() {
  const q = st.lastPx[st.pxSym];
  $("#px-quote").innerHTML = q ? `<span class="${cls(q.chg)}">${n(q.last)} ${sPct(q.chg)}</span> · bid ${n(q.bid)} / ask ${n(q.ask)}` : "";
}

// ------------------------------------------------------------------ watchlist
async function loadQuotes() {
  const syms = new Set();
  st.positions.forEach((p) => syms.add(p.symbol)); syms.add(st.pxSym);
  const [etf, extra] = await Promise.all([api("/api/quotes"), syms.size ? api(`/api/quotes?symbols=${[...syms].slice(0, 400).join(",")}`) : []]);
  const prev = { ...st.lastPx };
  [...etf, ...extra].forEach((q) => st.lastPx[q.symbol] = q);
  $("#watch tbody").innerHTML = etf.map((q) => {
    const p = prev[q.symbol], f = p && q.last !== p.last ? (q.last > p.last ? "flash-up" : "flash-down") : "";
    const w = st.trendWt[q.symbol];
    return `<tr class="click ${f}" data-s="${q.symbol}"><td>${q.symbol}</td><td class="r">${n(q.last)}</td>
      <td class="r ${cls(q.chg)}">${sPct(q.chg)}</td><td class="r ${cls(w)}">${w == null ? "—" : sPct(w, 1)}</td></tr>`;
  }).join("");
  refreshQuoteLine();
}
$("#watch").addEventListener("click", (e) => { const tr = e.target.closest("tr[data-s]"); if (tr) pickSymbol(tr.dataset.s); });

// ------------------------------------------------------------------ positions
async function loadPositions() { st.positions = await api(`/api/positions?book=${st.book}`); renderPositions(); }
function renderPositions() {
  const q = $("#pos-search").value.trim().toUpperCase(), f = st.posFilter;
  let rows = st.positions.filter((p) => (f === "all" || p.side === f || p.sleeve === f) && (!q || p.symbol.includes(q)));
  rows.sort((a, b) => { const x = a[st.sortK], y = b[st.sortK];
    if (typeof x === "string") return st.sortDir * x.localeCompare(y);
    return st.sortDir * ((st.sortK === "value" || st.sortK === "weight" ? Math.abs(x ?? 0) - Math.abs(y ?? 0) : (x ?? -1e18) - (y ?? -1e18))); });
  $("#pos-empty").hidden = rows.length > 0;
  $("#pos tbody").innerHTML = rows.map((p) => `<tr class="click" data-s="${p.symbol}">
    <td><b>${p.symbol}</b> <span class="${p.qty > 0 ? "up" : "down"}">${p.side === "LONG" ? "L" : "S"}</span></td>
    <td><span class="chip ${p.sleeve}">${p.sleeve.toUpperCase()}</span></td>
    <td class="r ${cls(p.qty)}">${n(p.qty, 0)}</td><td class="r">${n(p.avg)}</td><td class="r">${n(p.last)}</td>
    <td class="r">${usd(p.value, 0)}</td><td class="r">${pct(p.weight, 1)}</td>
    <td class="r ${cls(p.upl)}">${sUsd(p.upl, 0)}</td><td class="r ${cls(p.uplpc)}">${sPct(p.uplpc)}</td>
    <td class="r ${cls(p.today)}">${sPct(p.today)}</td></tr>`).join("");
  $$("#pos th").forEach((th) => th.classList.toggle("sorted", th.dataset.k === st.sortK));
}
$("#pos thead").addEventListener("click", (e) => { const k = e.target.closest("th")?.dataset.k; if (!k) return;
  st.sortDir = st.sortK === k ? -st.sortDir : -1; st.sortK = k; renderPositions(); });
$("#pos tbody").addEventListener("click", (e) => { const tr = e.target.closest("tr[data-s]"); if (tr) pickSymbol(tr.dataset.s); });
seg($("#pos-filter"), (b) => { st.posFilter = b.dataset.f; renderPositions(); });
$("#pos-search").addEventListener("input", renderPositions);

// ------------------------------------------------------------------ targets
async function loadTargets() {
  const t = await api("/api/targets"), eq = st.overview?.account.equity || 1;
  st.trendWt = {}; t.filter((x) => x.sleeve === "trend").forEach((x) => st.trendWt[x.symbol] = x.target / eq);
  $("#tgt-count").textContent = t.length ? `${t.length} names` : "none yet";
  const top = [...t.slice(-8).reverse(), ...t.slice(0, 8)], mx = Math.max(1, ...top.map((x) => Math.abs(x.target)));
  $("#targets").innerHTML = top.map((x) => { const w = Math.abs(x.target) / mx * 50;
    return `<div class="tg"><span>${x.symbol}</span><div class="bar"><i style="${x.target >= 0 ? `left:50%;width:${w}%;background:var(--up)` : `right:50%;width:${w}%;background:var(--down)`}"></i></div>
      <span class="r ${cls(x.target)}">${sUsd(x.target, 0)}</span></div>`; }).join("") || `<div class="empty">Targets appear after the first daily run.</div>`;
}

// ------------------------------------------------------------------ orders / runs / research
async function loadOrders() {
  const o = await api(`/api/orders?book=${st.book}`);
  $("#orders-empty").hidden = o.length > 0;
  $("#orders tbody").innerHTML = o.map((r) => `<tr><td>${esc(String(r.time).replace("T", " ").slice(0, 19))}</td><td><b>${esc(r.symbol)}</b></td>
    <td class="${r.side === "BUY" ? "up" : "down"}">${r.side}</td><td class="r">${n(r.qty, 0)}</td><td class="r">${r.filled_qty == null ? "—" : n(r.filled_qty, 0)}</td>
    <td class="r">${n(r.price)}</td><td class="r">${usd(r.notional, 0)}</td><td class="muted">${esc(r.type)}</td>
    <td class="${/fill/i.test(r.status) ? "up" : /reject|error|cancel/i.test(r.status) ? "down" : ""}">${esc(r.status)}</td></tr>`).join("");
}
async function loadRuns() {
  const r = await api("/api/runs");
  $("#runs tbody").innerHTML = r.map((x) => `<tr><td>${x.date}</td><td>${x.mode}</td><td class="r">${usd(x.equity, 0)}</td>
    <td class="r">${n(x.gross_lev)}×</td><td class="r ${cls(x.net_exposure)}">${sPct(x.net_exposure, 1)}</td><td class="r">${x.n_targets}</td>
    <td class="r">${x.n_orders}</td><td class="r">${usd(x.order_notional, 0)}</td><td>${x.data_through}</td><td class="muted">${esc(x.model)}</td></tr>`).join("")
    || `<tr><td colspan="10" class="empty">No runs yet.</td></tr>`;
}
async function loadResearch() {
  const r = await api("/api/research");
  if (r.report) $("#rs-report").innerHTML = marked.parse(r.report);
  if (!rsChart) rsChart = mkChart($("#rs-chart"));
  const colors = { combined: css("--text"), stock_ml: css("--accent2"), etf_trend: css("--warn"),
                   baseline_momentum_12_1: css("--muted"), baseline_reversal_5d: "#5a6475" };
  (rsChart._s || []).forEach((s) => rsChart.removeSeries(s)); rsChart._s = [];
  Object.entries(r.curves).forEach(([k, pts]) => {
    const s = rsChart.addLineSeries({ color: colors[k] || css("--accent"), lineWidth: k === "combined" ? 3 : 1, title: k, priceLineVisible: false });
    s.setData(pts); rsChart._s.push(s);
  });
  rsChart.timeScale().fitContent();
}

// ------------------------------------------------------------------ tabs / jobs / controls
$("#tabs").addEventListener("click", (e) => {
  const b = e.target.closest("button"); if (!b) return;
  $$("#tabs button").forEach((x) => x.classList.toggle("on", x === b));
  $$(".tab").forEach((t) => t.hidden = t.dataset.tab !== b.dataset.tab);
  ({ runs: loadRuns, research: loadResearch, orders: loadOrders, console: pollJob })[b.dataset.tab]?.().catch?.(err);
});
async function pollJob() {
  const j = await api("/api/job"); st.jobRunning = j.running;
  $("#job-name").textContent = j.name ? `${j.name} · started ${j.started?.replace("T", " ").slice(0, 19)} · ${j.running ? "RUNNING" : "exit " + j.exit_code}` : "no job run yet";
  const c = $("#console"), atBottom = c.scrollTop + c.clientHeight >= c.scrollHeight - 20;
  c.textContent = j.log.join("\n"); if (atBottom) c.scrollTop = c.scrollHeight;
  $("#job-dot").className = j.running ? "run" : "";
  if (j.running) setTimeout(() => pollJob().catch(err), 1500);
  else if (j.name) { refreshAll(); }
}
$("#jobs-btn").onclick = () => $("#jobs-list").classList.toggle("open");
document.addEventListener("click", (e) => { if (!e.target.closest(".menu")) $("#jobs-list").classList.remove("open"); });
$("#jobs-list").addEventListener("click", async (e) => {
  const b = e.target.closest("button[data-job]"); if (!b) return;
  $("#jobs-list").classList.remove("open");
  if (b.dataset.confirm && !confirm(b.dataset.confirm)) return;
  try { await api(`/api/job/${b.dataset.job}`, { method: "POST" }); toast(`started: ${b.textContent}`);
    $$("#tabs button").find((x) => x.dataset.tab === "console").click(); } catch (e2) { toast(e2.message, 6000); }
});
$("#kill-btn").onclick = async () => {
  if (!confirm("KILL SWITCH: halt trading and close ALL paper positions?")) return;
  const r = await api("/api/kill", { method: "POST" }); toast(r.flattened_paper ? "halted + flattening paper book" : "halted"); refreshAll();
};
$("#resume-btn").onclick = async () => { if (confirm("Remove the halt and allow trading again?")) { await api("/api/resume", { method: "POST" }); refreshAll(); } };
seg($("#book-seg"), (b) => { st.book = b.dataset.book; try { localStorage.setItem("book", st.book); } catch {} refreshAll(); });

// ------------------------------------------------------------------ loops
function err(e) { banner(/unauthor/i.test(e.message) ? "Session expired: reopen the URL with ?token=" : `Backend: ${e.message}`); }
function refreshAll() { loadOverview().then(() => Promise.all([loadPositions(), loadTargets(), loadOrders(), loadEquity()])).catch(err); }
setInterval(() => { $("#clock").textContent = new Date().toLocaleTimeString("en-US", { timeZone: "America/New_York", hour12: false }) + " ET"; }, 1000);

(async function init() {
  let saved = null; try { saved = localStorage.getItem("book"); } catch {}
  try { const o = await api("/api/overview?book=paper"); st.book = saved || (o.mode === "shadow" ? "shadow" : "paper"); }
  catch (e) { st.book = saved || "paper"; err(e); }
  $$("#book-seg button").forEach((b) => b.classList.toggle("on", b.dataset.book === st.book));
  refreshAll(); loadBars().catch(err); loadQuotes().catch(err); loadResearch().catch(() => {});
  setInterval(() => loadOverview().catch(err), 5000);
  setInterval(() => loadPositions().catch(err), 5000);
  setInterval(() => loadQuotes().catch(err), 4000);
  setInterval(() => loadOrders().catch(err), 10000);
  setInterval(() => loadEquity().catch(err), 60000);
  setInterval(() => loadTargets().catch(err), 60000);
  setInterval(() => { if (st.pxTf === "5Min") loadBars().catch(err); }, 30000);
})();
