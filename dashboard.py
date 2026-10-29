"""
Live training dashboard for pretrain.py runs

Serves one page on the local network that refreshes every 10 seconds: GPU load,
the current run's progress and ETA, loss and throughput curves for every run in
a group, a runs table, and each run's config and latest sample. It only reads
the files pretrain.py writes (config.json, log.jsonl, samples.txt); no extra
dependencies.

    python dashboard.py                      # http://<this machine's IP>:8000
    python dashboard.py --host 127.0.0.1     # this machine only
"""

import argparse
import json
import math
import shutil
import socket
import subprocess
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
ACTIVE_SECONDS = 600  # a run whose log changed this recently counts as running
MAX_POINTS = 400  # per curve sent to the page


def load_variants() -> dict:
    try:
        from ablate import VARIANTS

        return VARIANTS
    except Exception:
        return {}


def read_log(path: Path):
    train, val = [], []
    with open(path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # blank, or a line being written right now
            (val if "val_loss" in rec else train).append(rec)
    return train, val


def thin(points, limit=MAX_POINTS):
    if len(points) <= limit:
        return points
    step = len(points) / limit
    out = [points[int(i * step)] for i in range(limit)]
    out[-1] = points[-1]
    return out


def latest_sample(path: Path):
    if not path.exists():
        return None
    chunks = path.read_text(errors="replace").split("--- step ")
    if len(chunks) < 2:
        return None
    header, _, body = chunks[-1].partition("\n")
    return {"header": f"step {header.strip()}", "text": body.strip()[:4000]}


def is_run(d: Path) -> bool:
    return any((d / f).exists() for f in ("config.json", "variant.json", "log.jsonl"))


def run_state(run_dir: Path, overrides):
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        cfg_path = run_dir / "variant.json"
    config = json.loads(cfg_path.read_text()) if cfg_path.exists() else None
    log = run_dir / "log.jsonl"
    train, val = read_log(log) if log.exists() else ([], [])

    tokens_per_step = max_steps = total = None
    if config:
        t = config["train"]
        micro, block = t["micro_batch_size"], t["block_size"]
        tokens_per_step = max(1, t["tokens_per_step"] // (micro * block)) * micro * block
        total = t.get("total_tokens")
        if total:
            max_steps = max(1, int(total) // tokens_per_step)

    if (run_dir / "final.pt").exists():
        status = "done"
    elif not train:
        status = "starting" if log.exists() and time.time() - log.stat().st_mtime < ACTIVE_SECONDS else "queued"
    elif time.time() - log.stat().st_mtime < ACTIVE_SECONDS:
        status = "running"
    else:
        status = "stopped"

    last = train[-1] if train else {}
    recent = [r["tok_per_s"] for r in train[-5:] if "tok_per_s" in r]
    tok_s = sum(recent) / len(recent) if recent else None
    step = last.get("step", 0)
    eta = None
    if status == "running" and max_steps and tok_s:
        eta = (max_steps - step) * tokens_per_step / tok_s / 3600

    return {
        "name": run_dir.name,
        "status": status,
        "overrides": overrides,
        "config": config,
        "step": step,
        "max_steps": max_steps,
        "tokens": last.get("tokens", 0),
        "total_tokens": total,
        "loss": last.get("lm_loss", last.get("loss")),
        "val_loss": val[-1]["val_loss"] if val else None,
        "best_val": min(r["val_loss"] for r in val) if val else None,
        "tok_s": tok_s,
        "train_hours": last.get("train_hours"),
        "eta_hours": eta,
        "curves": {
            "train": thin([[r["tokens"], r.get("lm_loss", r.get("loss"))] for r in train]),
            "val": [[r["tokens"], r["val_loss"]] for r in val],
            "speed": thin([[r["tokens"], r["tok_per_s"]] for r in train if "tok_per_s" in r]),
        },
        "sample": latest_sample(run_dir / "samples.txt"),
        "updated": log.stat().st_mtime if log.exists() else None,
    }


def collect_groups():
    variants = load_variants()
    found = {}
    if RUNS.exists():
        for d in RUNS.iterdir():
            if not d.is_dir():
                continue
            if is_run(d):
                found.setdefault("runs", []).append(d)
            else:
                found[d.name] = [r for r in d.iterdir() if r.is_dir() and is_run(r)]

    groups = []
    for name, dirs in found.items():
        ablation = name.startswith("ablation-")
        by_name = {d.name: d for d in dirs}
        if ablation:
            order = list(variants) + sorted(n for n in by_name if n not in variants)
        else:
            order = [d.name for d in sorted(dirs, key=lambda d: d.stat().st_ctime)]
        runs = []
        for run_name in order:
            overrides = variants.get(run_name) if ablation else None
            if run_name in by_name:
                runs.append(run_state(by_name[run_name], overrides))
            elif ablation:
                runs.append({"name": run_name, "status": "queued", "overrides": overrides, "config": None,
                             "curves": {"train": [], "val": [], "speed": []}})
        # Colors follow the run, never its rank: baseline is the neutral reference,
        # the rest take the 8 categorical slots in a fixed order, then fold into "other"
        slot = 0
        for run in runs:
            if run["name"] == "baseline":
                run["slot"] = "base"
            else:
                run["slot"] = slot if slot < 8 else "other"
                slot += 1
        updated = max((r.get("updated") or 0 for r in runs), default=0)
        groups.append({"name": name, "runs": runs, "updated": updated})
    groups.sort(key=lambda g: g["updated"], reverse=True)
    return groups


def gpu_state():
    fields = "name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,clocks.sm"
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return None
    if not out:
        return None
    values = [v.strip() for v in out.splitlines()[0].split(",")]

    def num(v):
        try:
            return float(v)
        except ValueError:
            return None

    name, util, used, total, temp, power, clock = values
    return {"name": name, "util": num(util), "mem_used": num(used), "mem_total": num(total),
            "temp": num(temp), "power": num(power), "clock": num(clock)}


def jobs_state():
    try:
        out = subprocess.run(["systemctl", "--user", "list-units", "llm-*", "--all", "--no-legend", "--plain"],
                             capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    jobs = []
    for line in out.splitlines():
        parts = line.split(None, 4)
        if len(parts) >= 4 and parts[0] != "llm-dashboard.service":
            jobs.append({"unit": parts[0].removesuffix(".service"), "active": parts[2], "sub": parts[3]})
    return jobs


def clean(value):
    """JSON has no NaN/Infinity; a diverged loss must not break the page"""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    return value


def state():
    return clean({
        "host": socket.gethostname(),
        "time": time.time(),
        "gpu": gpu_state(),
        "jobs": jobs_state(),
        "disk_free_gb": shutil.disk_usage(ROOT).free / 1e9,
        "groups": collect_groups(),
    })


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            body, ctype = PAGE.encode(), "text/html; charset=utf-8"
        elif self.path == "/api/state":
            body, ctype = json.dumps(state()).encode(), "application/json"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LLM training</title>
<style>
:root { color-scheme: light dark; }
.viz-root {
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb;
  --text-primary: #0b0b0b; --text-secondary: #52514e; --text-muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,0.10);
  --series-1: #2a78d6; --series-2: #eb6834; --series-3: #1baf7a; --series-4: #eda100;
  --series-5: #e87ba4; --series-6: #008300; --series-7: #6250d6; --series-8: #e34948;
  --series-base: #0b0b0b; --series-other: #898781;
  --accent: #2a78d6; --meter-track: #cde2fb;
  --good: #0ca30c; --warning: #fab219; --serious: #ec835a; --critical: #d03b3b;
  --hover: rgba(11,11,11,0.04);
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) .viz-root {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19;
    --text-primary: #f0efec; --text-secondary: #c3c2b7; --text-muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
    --series-1: #3987e5; --series-2: #d95926; --series-3: #199e70; --series-4: #c98500;
    --series-5: #d55181; --series-6: #008300; --series-7: #9085e9; --series-8: #e66767;
    --series-base: #f0efec; --series-other: #898781;
    --accent: #3987e5; --meter-track: #184f95; --hover: rgba(255,255,255,0.05);
  }
}
:root[data-theme="dark"] .viz-root {
  color-scheme: dark;
  --page: #0d0d0d; --surface: #1a1a19;
  --text-primary: #f0efec; --text-secondary: #c3c2b7; --text-muted: #898781;
  --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
  --series-1: #3987e5; --series-2: #d95926; --series-3: #199e70; --series-4: #c98500;
  --series-5: #d55181; --series-6: #008300; --series-7: #9085e9; --series-8: #e66767;
  --series-base: #f0efec; --series-other: #898781;
  --accent: #3987e5; --meter-track: #184f95; --hover: rgba(255,255,255,0.05);
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--text-primary);
  font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1280px; margin: 0 auto; padding: 20px 20px 48px; transition: opacity .2s; }
body.stale main { opacity: .55; }
h1 { font-size: 20px; margin: 0; font-weight: 650; }
h2 { font-size: 15px; margin: 0 0 12px; font-weight: 600; }
.top { display: flex; align-items: baseline; gap: 16px; flex-wrap: wrap; margin-bottom: 14px; }
.top .meta { color: var(--text-secondary); }
.top .spacer { flex: 1; }
button, select { font: inherit; color: var(--text-primary); background: var(--surface);
  border: 1px solid var(--border); border-radius: 8px; padding: 5px 10px; cursor: pointer; }
button:hover, select:hover { background: var(--hover); }
button:focus-visible, select:focus-visible, [tabindex]:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.filters { display: flex; align-items: center; gap: 16px; flex-wrap: wrap; margin-bottom: 16px; }
.filters label { color: var(--text-secondary); display: flex; gap: 8px; align-items: center; }
.stale-note { display: none; color: var(--text-secondary); }
body.stale .stale-note { display: inline; }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 16px; margin-bottom: 16px; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin-bottom: 16px; }
.tile { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 12px 14px; }
.tile .label { color: var(--text-secondary); font-size: 13px; }
.tile .value { font-size: 22px; font-weight: 600; margin-top: 2px; }
.tile .sub { color: var(--text-muted); font-size: 12px; margin-top: 2px; }
.meter { height: 6px; border-radius: 3px; background: var(--meter-track); margin-top: 8px; overflow: hidden; }
.meter > div { height: 100%; border-radius: 3px; background: var(--accent); }
.meter.warn > div { background: var(--warning); }
.current-head { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; margin-bottom: 4px; }
.current-head .name { font-size: 17px; font-weight: 600; }
.chips { display: flex; gap: 6px; flex-wrap: wrap; }
.chip { font-size: 12px; color: var(--text-secondary); border: 1px solid var(--border); border-radius: 999px; padding: 1px 8px; }
.progress-line { display: flex; justify-content: space-between; color: var(--text-secondary); font-size: 13px; margin-top: 10px; }
.current .meter { height: 8px; }
.current .tiles { margin: 14px 0 0; }
.current .tile { border: none; background: transparent; padding: 0; }
.status { display: inline-flex; align-items: center; gap: 5px; font-size: 13px; color: var(--text-secondary); white-space: nowrap; }
.status .icon { font-size: 11px; }
.status.running .icon { color: var(--good); }
.status.stopped .icon { color: var(--serious); }
.status.done .icon, .status.queued .icon, .status.starting .icon { color: var(--text-muted); }
.legend { display: flex; flex-wrap: wrap; gap: 6px 14px; margin-bottom: 12px; }
.legend button { border: none; background: none; padding: 2px 4px; display: inline-flex; align-items: center; gap: 7px; color: var(--text-primary); }
.legend button[aria-pressed="false"] { color: var(--text-muted); }
.legend button[aria-pressed="false"] .key { opacity: .3; }
.key { display: inline-block; width: 16px; height: 2px; border-radius: 1px; }
.charts { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
.charts .wide { grid-column: 1 / -1; }
@media (max-width: 900px) { .charts { grid-template-columns: 1fr; } }
.chart-card { margin: 0; }
.chart-head { display: flex; justify-content: space-between; align-items: baseline; gap: 12px; }
.chart-head .note { color: var(--text-muted); font-size: 12px; }
.chart-head button { font-size: 12px; padding: 2px 8px; }
.plot { position: relative; min-height: 120px; }
svg text { fill: var(--text-muted); font-size: 11px; font-variant-numeric: tabular-nums; }
.grid { stroke: var(--grid); stroke-width: 1; }
.axis { stroke: var(--axis); stroke-width: 1; }
.line { fill: none; stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }
.line.base { stroke-width: 2.5; }
.dot { stroke: var(--surface); stroke-width: 2; }
.cross { stroke: var(--axis); stroke-width: 1; }
.overlay { fill: transparent; cursor: crosshair; }
.overlay:focus { outline: none; }
.tooltip { position: absolute; pointer-events: none; background: var(--surface); border: 1px solid var(--border);
  border-radius: 8px; padding: 8px 10px; box-shadow: 0 4px 16px rgba(0,0,0,.18); min-width: 170px; z-index: 2; }
.tooltip .tt-x { color: var(--text-secondary); font-size: 12px; margin-bottom: 4px; }
.tooltip .row { display: flex; align-items: center; gap: 8px; font-size: 12px; white-space: nowrap; }
.tooltip .row strong { font-variant-numeric: tabular-nums; min-width: 52px; }
.tooltip .row span:last-child { color: var(--text-secondary); }
.empty { color: var(--text-muted); padding: 24px 0; margin: 0; }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
th, td { text-align: left; padding: 7px 10px; border-bottom: 1px solid var(--grid); white-space: nowrap; }
th { color: var(--text-secondary); font-weight: 500; font-size: 13px; }
td.num, th.num { text-align: right; }
tbody tr { cursor: pointer; }
tbody tr:hover { background: var(--hover); }
tbody tr.selected { background: var(--hover); box-shadow: inset 3px 0 0 var(--accent); }
.table-scroll { overflow-x: auto; }
.better { color: var(--good); }
.detail-grid { display: grid; grid-template-columns: 1.3fr 1fr; gap: 16px; }
@media (max-width: 900px) { .detail-grid { grid-template-columns: 1fr; } }
pre { margin: 0; white-space: pre-wrap; word-break: break-word; font: 12.5px/1.5 ui-monospace, SFMono-Regular, Menlo, monospace;
  background: var(--page); border: 1px solid var(--border); border-radius: 8px; padding: 10px 12px; max-height: 360px; overflow: auto; }
.sample { font-family: system-ui, -apple-system, "Segoe UI", sans-serif; font-size: 14px; }
.subtle { color: var(--text-secondary); font-size: 13px; margin: 0 0 8px; }
</style>
</head>
<body class="viz-root">
<main>
  <header class="top">
    <h1>LLM training</h1>
    <span class="meta" id="host"></span>
    <span class="meta" id="updated"></span>
    <span class="stale-note">· can't reach the dashboard server, retrying</span>
    <span class="spacer"></span>
    <button id="theme" type="button">Theme: auto</button>
  </header>
  <div class="filters">
    <label>Run group <select id="group"></select></label>
    <span id="jobs" class="chips"></span>
  </div>
  <section class="tiles" id="gpu" aria-label="Machine"></section>
  <section class="card current" id="current" aria-label="Current run"></section>
  <section class="card">
    <h2>Curves</h2>
    <div class="legend" id="legend" aria-label="Runs shown in the charts"></div>
    <div class="charts">
      <div class="chart-card wide" id="chart-val"></div>
      <div class="chart-card" id="chart-train"></div>
      <div class="chart-card" id="chart-speed"></div>
    </div>
  </section>
  <section class="card">
    <h2>Runs</h2>
    <div class="table-scroll"><table id="runs"></table></div>
  </section>
  <section class="card" id="detail"></section>
</main>
<script>
"use strict";
const POLL_MS = 10000;
let state = null, groupName = null, selectedRun = null;
const hidden = new Set();
const tableMode = {};

function h(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k === "class") e.className = v;
    else if (k === "text") e.textContent = v;
    else if (k === "style") e.style.cssText = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v === true ? "" : v);
  }
  for (const kid of kids.flat()) if (kid != null) e.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
  return e;
}
const SVGNS = "http://www.w3.org/2000/svg";
function s(tag, attrs = {}) {
  const e = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  return e;
}

const fmtTokens = n => n == null ? "–" : n >= 1e9 ? (n / 1e9).toFixed(2) + "B" : n >= 1e6 ? (n / 1e6).toFixed(n >= 1e8 ? 0 : 1) + "M" : n >= 1e3 ? (n / 1e3).toFixed(0) + "K" : String(Math.round(n));
const fmtLoss = v => v == null ? "–" : v.toFixed(3);
const fmtInt = v => v == null ? "–" : Math.round(v).toLocaleString();
const fmtHours = v => v == null ? "–" : v < 1 ? Math.max(1, Math.round(v * 60)) + " min" : v.toFixed(1) + " h";
const color = run => run.slot === "base" ? "var(--series-base)" : run.slot === "other" ? "var(--series-other)" : `var(--series-${run.slot + 1})`;

function statusEl(status) {
  const icons = { running: "●", done: "✓", stopped: "■", queued: "○", starting: "◌" };
  return h("span", { class: `status ${status}` }, h("span", { class: "icon", "aria-hidden": "true", text: icons[status] || "·" }), status);
}

function overrideText(o) {
  if (!o) return [];
  const out = [];
  for (const section of Object.values(o)) for (const [k, v] of Object.entries(section)) out.push(`${k}=${Array.isArray(v) ? "[" + v.join(",") + "]" : v}`);
  return out;
}

function niceTicks(lo, hi, n) {
  if (!(hi > lo)) hi = lo + 1;
  const span = hi - lo, raw = span / n, mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(st => span / st <= n) || 10 * mag;
  const ticks = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) ticks.push(+v.toFixed(10));
  return ticks;
}

function nearest(points, x) {
  let lo = 0, hi = points.length - 1;
  while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (points[mid][0] < x) lo = mid; else hi = mid; }
  return Math.abs(points[lo][0] - x) <= Math.abs(points[hi][0] - x) ? points[lo] : points[hi];
}

// One chart card: header with a chart/table toggle, then an SVG line chart with a crosshair tooltip
function renderChart(id, title, note, runs, key, opt) {
  const card = document.getElementById(id);
  const series = runs.filter(r => r.curves && r.curves[key].length).map(r => ({ run: r, points: r.curves[key] }));
  const visible = series.filter(sr => !hidden.has(sr.run.name));
  const asTable = !!tableMode[id];
  const toggle = h("button", { type: "button", onclick: () => { tableMode[id] = !asTable; render(); } }, asTable ? "Chart" : "Table");
  const box = h("div", { class: "plot" });
  card.replaceChildren(h("div", { class: "chart-head" }, h("h2", { text: title }), h("span", { class: "note", text: note }), toggle), box);
  if (!visible.length) { box.append(h("p", { class: "empty", text: "No data yet" })); return; }
  if (asTable) { box.append(chartTable(visible, opt)); return; }

  const W = Math.max(320, box.clientWidth), H = opt.height || 240, m = { l: 52, r: 16, t: 10, b: 28 };
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const xMax = Math.max(...visible.map(sr => sr.points[sr.points.length - 1][0])) || 1;
  const ys = [];
  for (const sr of visible) {
    const from = sr.points[sr.points.length - 1][0] * (opt.clipFrac || 0);
    for (const [x, y] of sr.points) if (x >= from && y != null) ys.push(y);
  }
  if (!ys.length) { box.append(h("p", { class: "empty", text: "No data yet" })); return; }
  let yLo = Math.min(...ys), yHi = Math.max(...ys);
  const pad = (yHi - yLo) * 0.08 || Math.abs(yHi) * 0.05 || 1;
  yLo = opt.zeroBase ? 0 : yLo - pad; yHi += pad;
  const yt = niceTicks(yLo, yHi, 5);
  const X = x => m.l + (x / xMax) * iw;
  const Y = y => m.t + (1 - (y - yLo) / (yHi - yLo)) * ih;

  const svg = s("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": title });
  const clipId = "clip-" + id;
  const defs = s("defs"), clip = s("clipPath", { id: clipId });
  clip.append(s("rect", { x: m.l, y: m.t - 6, width: iw + 6, height: ih + 6 }));
  defs.append(clip); svg.append(defs);
  for (const t of yt) {
    if (t < yLo || t > yHi) continue;
    svg.append(s("line", { class: "grid", x1: m.l, x2: W - m.r, y1: Y(t), y2: Y(t) }));
    const label = s("text", { x: m.l - 8, y: Y(t) + 4, "text-anchor": "end" });
    label.textContent = opt.fmtTick(t);
    svg.append(label);
  }
  svg.append(s("line", { class: "axis", x1: m.l, x2: W - m.r, y1: m.t + ih, y2: m.t + ih }));
  for (const t of niceTicks(0, xMax, Math.max(2, Math.floor(iw / 90)))) {
    if (t > xMax) continue;
    const label = s("text", { x: X(t), y: H - 8, "text-anchor": "middle" });
    label.textContent = fmtTokens(t);
    svg.append(label);
  }
  const lines = s("g", { "clip-path": `url(#${clipId})` });
  // Baseline last so the neutral reference line sits on top
  const ordered = [...visible].sort((a, b) => (a.run.slot === "base") - (b.run.slot === "base"));
  for (const sr of ordered) {
    const pts = sr.points.filter(p => p[1] != null);
    const d = pts.map((p, i) => (i ? "L" : "M") + X(p[0]).toFixed(1) + " " + Y(p[1]).toFixed(1)).join("");
    const path = s("path", { d, class: "line" + (sr.run.slot === "base" ? " base" : "") });
    path.style.stroke = color(sr.run);
    lines.append(path);
    const last = pts[pts.length - 1];
    const dot = s("circle", { class: "dot", cx: X(last[0]), cy: Y(last[1]), r: 4 });
    dot.style.fill = color(sr.run);
    lines.append(dot);
  }
  svg.append(lines);

  const cross = s("line", { class: "cross", y1: m.t, y2: m.t + ih, visibility: "hidden" });
  const hoverDots = s("g", { visibility: "hidden" });
  svg.append(cross, hoverDots);
  const overlay = s("rect", { class: "overlay", x: m.l, y: m.t, width: iw, height: ih, tabindex: 0,
    "aria-label": `${title}: use the arrow keys to read values` });
  svg.append(overlay);
  const tip = h("div", { class: "tooltip", style: "visibility:hidden" });
  box.append(svg, tip);

  const xs = [...new Set(visible.flatMap(sr => sr.points.map(p => p[0])))].sort((a, b) => a - b);
  let cursor = xs.length - 1;
  function show(i) {
    cursor = Math.max(0, Math.min(xs.length - 1, i));
    const x = xs[cursor];
    cross.setAttribute("x1", X(x)); cross.setAttribute("x2", X(x)); cross.setAttribute("visibility", "visible");
    hoverDots.replaceChildren();
    const rows = [];
    for (const sr of visible) {
      const p = nearest(sr.points, x);
      const span = sr.points.length > 1 ? (sr.points[sr.points.length - 1][0] - sr.points[0][0]) / (sr.points.length - 1) : xMax;
      if (p[1] == null || Math.abs(p[0] - x) > span) continue;  // this run has no point near here
      rows.push({ run: sr.run, value: p[1] });
      const dot = s("circle", { class: "dot", cx: X(p[0]), cy: Y(p[1]), r: 4 });
      dot.style.fill = color(sr.run);
      hoverDots.append(dot);
    }
    hoverDots.setAttribute("visibility", "visible");
    rows.sort((a, b) => opt.higherIsBetter ? b.value - a.value : a.value - b.value);
    tip.replaceChildren(h("div", { class: "tt-x", text: `${fmtTokens(x)} tokens` }),
      ...rows.map(r => h("div", { class: "row" }, h("span", { class: "key", style: `background:${color(r.run)}` }),
        h("strong", { text: opt.fmt(r.value) }), h("span", { text: r.run.name }))));
    tip.style.visibility = "visible";
    const left = X(x) + 14;
    tip.style.left = (left + 200 > W ? X(x) - 14 - tip.offsetWidth : left) + "px";
    tip.style.top = m.t + "px";
  }
  function hide() { cross.setAttribute("visibility", "hidden"); hoverDots.setAttribute("visibility", "hidden"); tip.style.visibility = "hidden"; }
  overlay.addEventListener("pointermove", e => {
    const rect = svg.getBoundingClientRect();
    const xv = ((e.clientX - rect.left) - m.l) / iw * xMax;
    let best = 0;
    for (let i = 1; i < xs.length; i++) if (Math.abs(xs[i] - xv) < Math.abs(xs[best] - xv)) best = i;
    show(best);
  });
  overlay.addEventListener("pointerleave", hide);
  overlay.addEventListener("focus", () => show(cursor));
  overlay.addEventListener("blur", hide);
  overlay.addEventListener("keydown", e => {
    if (e.key === "ArrowLeft") { show(cursor - 1); e.preventDefault(); }
    if (e.key === "ArrowRight") { show(cursor + 1); e.preventDefault(); }
  });
}

function chartTable(visible, opt) {
  let xs = [...new Set(visible.flatMap(sr => sr.points.map(p => p[0])))].sort((a, b) => a - b);
  if (xs.length > 25) { const step = xs.length / 24; xs = [...Array.from({ length: 24 }, (_, i) => xs[Math.floor(i * step)]), xs[xs.length - 1]]; }
  const head = h("tr", {}, h("th", { text: "Tokens" }), ...visible.map(sr => h("th", { class: "num", text: sr.run.name })));
  const body = xs.map(x => h("tr", {}, h("td", { text: fmtTokens(x) }), ...visible.map(sr => {
    const p = nearest(sr.points, x);
    return h("td", { class: "num", text: p[0] === x ? opt.fmt(p[1]) : "–" });
  })));
  return h("div", { class: "table-scroll" }, h("table", {}, h("thead", {}, head), h("tbody", {}, body)));
}

function tile(label, value, sub, meter) {
  const t = h("div", { class: "tile" }, h("div", { class: "label", text: label }), h("div", { class: "value", text: value }));
  if (sub) t.append(h("div", { class: "sub", text: sub }));
  if (meter != null) t.append(h("div", { class: "meter" + (meter > 0.9 ? " warn" : ""), role: "meter",
    "aria-valuemin": 0, "aria-valuemax": 100, "aria-valuenow": Math.round(meter * 100) }, h("div", { style: `width:${Math.min(100, meter * 100)}%` })));
  return t;
}

function renderMachine() {
  const g = state.gpu, el = document.getElementById("gpu");
  const tiles = [];
  if (g) {
    tiles.push(tile("GPU utilization", g.util == null ? "–" : g.util + "%", g.name, g.util == null ? null : g.util / 100));
    tiles.push(tile("GPU memory", g.mem_used == null ? "–" : (g.mem_used / 1024).toFixed(2) + " GB",
      g.mem_total ? `of ${(g.mem_total / 1024).toFixed(1)} GB` : null, g.mem_total ? g.mem_used / g.mem_total : null));
    const hot = g.temp >= 87 ? " · running hot" : "";
    tiles.push(tile("GPU temperature", g.temp == null ? "–" : g.temp + " °C", (g.clock ? `${g.clock} MHz` : "") + hot));
    tiles.push(tile("GPU power", g.power == null ? "–" : g.power.toFixed(0) + " W", null));
  } else {
    tiles.push(tile("GPU", "not found", "nvidia-smi unavailable"));
  }
  tiles.push(tile("Disk free", state.disk_free_gb.toFixed(0) + " GB", "checkpoints and data"));
  el.replaceChildren(...tiles);
}

function renderCurrent(runs) {
  const el = document.getElementById("current");
  const run = runs.find(r => r.status === "running") || runs.find(r => r.status === "starting") ||
    [...runs].reverse().find(r => r.status !== "queued");
  if (!run) { el.replaceChildren(h("h2", { text: "Current run" }), h("p", { class: "empty", text: "Nothing has started in this group yet" })); return; }
  const done = runs.filter(r => r.status === "done").length;
  const frac = run.max_steps ? run.step / run.max_steps : 0;
  const base = runs.find(r => r.name === "baseline");
  el.replaceChildren(
    h("div", { class: "current-head" }, h("span", { class: "name", text: run.name }), statusEl(run.status),
      h("span", { class: "chips" }, overrideText(run.overrides).map(t => h("span", { class: "chip", text: t })))),
    h("div", { class: "subtle", text: `Run ${runs.indexOf(run) + 1} of ${runs.length} in this group · ${done} done` }),
    h("div", { class: "progress-line" }, h("span", { text: `Step ${fmtInt(run.step)} of ${fmtInt(run.max_steps)}` }),
      h("span", { text: `${fmtTokens(run.tokens)} of ${fmtTokens(run.total_tokens)} tokens · ${(frac * 100).toFixed(1)}%` })),
    h("div", { class: "meter", role: "progressbar", "aria-valuemin": 0, "aria-valuemax": 100, "aria-valuenow": Math.round(frac * 100) },
      h("div", { style: `width:${frac * 100}%` })),
    h("div", { class: "tiles" },
      tile("Train loss", fmtLoss(run.loss), null),
      tile("Val loss", fmtLoss(run.val_loss), base && base !== run && base.val_loss != null && run.val_loss != null
        ? `baseline ${fmtLoss(base.val_loss)} at ${fmtTokens(base.curves.val[base.curves.val.length - 1][0])}` : null),
      tile("Tokens/s", fmtInt(run.tok_s), null),
      tile("Elapsed", fmtHours(run.train_hours), null),
      tile("Time left", run.status === "running" ? fmtHours(run.eta_hours) : "–", null)));
}

function renderLegend(runs) {
  const withData = runs.filter(r => r.curves && (r.curves.train.length || r.curves.val.length));
  document.getElementById("legend").replaceChildren(...withData.map(r => h("button", {
    type: "button", "aria-pressed": hidden.has(r.name) ? "false" : "true",
    onclick: () => { hidden.has(r.name) ? hidden.delete(r.name) : hidden.add(r.name); render(); },
  }, h("span", { class: "key", style: `background:${color(r)}` }), r.name)));
}

function renderRuns(runs) {
  const base = runs.find(r => r.name === "baseline");
  const head = h("tr", {}, ...["Run", "Status", "Progress", "Train loss", "Val loss", "vs baseline", "Tokens/s", "Time", "Changes"]
    .map((t, i) => h("th", { class: i >= 2 && i <= 7 ? "num" : null, text: t })));
  const rows = runs.map(r => {
    let delta = null;
    if (base && r !== base && r.val_loss != null && base.curves.val.length && r.curves.val.length) {
      // Compare at the same token count: the baseline's value nearest this run's latest eval
      const x = r.curves.val[r.curves.val.length - 1][0];
      const b = nearest(base.curves.val, x);
      if (Math.abs(b[0] - x) <= x * 0.05) delta = r.val_loss - b[1];
    }
    const pct = r.max_steps ? `${Math.min(100, (r.step / r.max_steps) * 100).toFixed(0)}%` : "–";
    const tr = h("tr", { tabindex: 0, class: r.name === selectedRun ? "selected" : null,
      onclick: () => { selectedRun = r.name; render(); },
      onkeydown: e => { if (e.key === "Enter") { selectedRun = r.name; render(); } } },
      h("td", {}, h("span", { class: "key", style: `background:${color(r)};margin-right:8px;vertical-align:middle` }), r.name),
      h("td", {}, statusEl(r.status)),
      h("td", { class: "num", text: pct }),
      h("td", { class: "num", text: fmtLoss(r.loss) }),
      h("td", { class: "num", text: fmtLoss(r.val_loss) }),
      h("td", { class: "num" + (delta != null && delta < 0 ? " better" : ""), text: delta == null ? "–" : (delta > 0 ? "+" : "−") + Math.abs(delta).toFixed(3) }),
      h("td", { class: "num", text: fmtInt(r.tok_s) }),
      h("td", { class: "num", text: fmtHours(r.train_hours) }),
      h("td", { text: overrideText(r.overrides).join(" · ") || (r.name === "baseline" ? "reference" : "–") }));
    return tr;
  });
  document.getElementById("runs").replaceChildren(h("thead", {}, head), h("tbody", {}, rows));
}

function renderDetail(runs) {
  const el = document.getElementById("detail");
  const run = runs.find(r => r.name === selectedRun);
  if (!run) { el.replaceChildren(); return; }
  const sample = run.sample;
  el.replaceChildren(h("h2", { text: `${run.name}: latest sample and config` }),
    h("div", { class: "detail-grid" },
      h("div", {}, h("p", { class: "subtle", text: sample ? `Generated at ${sample.header}` : "No sample yet" }),
        h("pre", { class: "sample", text: sample ? sample.text : "" })),
      h("div", {}, h("p", { class: "subtle", text: "Model and training settings" }),
        h("pre", { text: run.config ? JSON.stringify(run.config, null, 2) : "Not started yet" }))));
}

function render() {
  if (!state) return;
  document.getElementById("host").textContent = state.host + (state.gpu ? ` · ${state.gpu.name}` : "");
  document.getElementById("updated").textContent = "updated " + new Date(state.time * 1000).toLocaleTimeString();
  document.getElementById("jobs").replaceChildren(...state.jobs.map(j => h("span", { class: "chip", text: `${j.unit}: ${j.sub}` })));
  const select = document.getElementById("group");
  const names = state.groups.map(g => g.name);
  if (!names.includes(groupName)) groupName = names[0] || null;
  select.replaceChildren(...names.map(n => h("option", { value: n, selected: n === groupName }, n)));
  renderMachine();
  const group = state.groups.find(g => g.name === groupName);
  const runs = group ? group.runs : [];
  if (!runs.some(r => r.name === selectedRun)) {
    const cur = runs.find(r => r.status === "running") || runs.find(r => r.status !== "queued");
    selectedRun = cur ? cur.name : null;
  }
  renderCurrent(runs);
  renderLegend(runs);
  renderChart("chart-val", "Validation loss", "lower is better · held-out tokens", runs, "val",
    { fmt: fmtLoss, fmtTick: v => v.toFixed(2), height: 280 });
  renderChart("chart-train", "Training loss", "per optimizer step", runs, "train",
    { fmt: fmtLoss, fmtTick: v => v.toFixed(1), clipFrac: 0.12 });
  renderChart("chart-speed", "Throughput", "tokens per second", runs, "speed",
    { fmt: fmtInt, fmtTick: v => v >= 1000 ? (v / 1000).toFixed(v % 1000 ? 1 : 0) + "K" : String(v), zeroBase: true, higherIsBetter: true });
  renderRuns(runs);
  renderDetail(runs);
}

async function refresh() {
  try {
    const res = await fetch("/api/state", { cache: "no-store" });
    if (!res.ok) throw new Error(res.status);
    state = await res.json();
    document.body.classList.remove("stale");
    render();
  } catch (err) {
    document.body.classList.add("stale");
  }
}

document.getElementById("group").addEventListener("change", e => { groupName = e.target.value; selectedRun = null; render(); });
const themes = ["auto", "light", "dark"];
let theme = localStorage.getItem("llm-dash-theme") || "auto";
function applyTheme() {
  if (theme === "auto") delete document.documentElement.dataset.theme; else document.documentElement.dataset.theme = theme;
  document.getElementById("theme").textContent = "Theme: " + theme;
}
document.getElementById("theme").addEventListener("click", () => {
  theme = themes[(themes.indexOf(theme) + 1) % themes.length];
  localStorage.setItem("llm-dash-theme", theme);
  applyTheme();
});
applyTheme();
let resizeTimer;
window.addEventListener("resize", () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(render, 150); });
refresh();
setInterval(refresh, POLL_MS);
</script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description="Live training dashboard")
    parser.add_argument("--host", default="0.0.0.0", help="0.0.0.0 serves the local network; 127.0.0.1 this machine only")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Dashboard on http://{socket.gethostname()}:{args.port} (listening on {args.host})")
    server.serve_forever()


if __name__ == "__main__":
    main()
