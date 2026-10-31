// Headless browser check of the built site: backend selection, 100 generated tokens,
// tokens/s, and a screenshot. Runs against `vite preview` (run `npm run build:local` first).
//
//   node scripts/browser-check.mjs            # CPU (WASM), then WebGPU
//   node scripts/browser-check.mjs wasm       # CPU only
//
// Chrome chooses the backend from ?backend=; the page reports what it actually started.

import { spawn } from 'node:child_process';
import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { launchChrome } from './chrome.mjs';

const WEB_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const OUT = join(WEB_ROOT, 'test', 'out');
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const GPU_FLAGS = [
  '--enable-unsafe-webgpu',
  '--enable-features=Vulkan',
  '--use-vulkan=native',
  '--use-angle=vulkan',
  '--ignore-gpu-blocklist',
];

async function startPreview(port = 4310 + Math.floor(Math.random() * 40)) {
  const proc = spawn(
    process.execPath,
    [join(WEB_ROOT, 'node_modules', 'vite', 'bin', 'vite.js'), 'preview', '--port', String(port), '--strictPort', '--host', '127.0.0.1'],
    { cwd: WEB_ROOT, stdio: 'ignore' },
  );
  const url = `http://127.0.0.1:${port}`;
  for (let i = 0; i < 100; i++) {
    try {
      if ((await fetch(url + '/')).ok) return { url, stop: () => proc.kill() };
    } catch {
      /* not up yet */
    }
    await sleep(200);
  }
  proc.kill();
  throw new Error('vite preview did not start (did you run `npm run build:local`?)');
}

/** Runs in the page. Waits for the model, then generates and returns what the UI shows. */
async function inPage(steps) {
  const demo = window.__demo;
  if (!demo) throw new Error('the page did not expose __demo');
  const deadline = Date.now() + 120_000;
  while (!demo.engine()) {
    if (Date.now() > deadline) throw new Error('the model did not load in 120 s');
    await new Promise((r) => setTimeout(r, 250));
  }
  const e = demo.engine();
  const started = performance.now();
  const result = await demo.generate('Once upon a time', steps);
  return {
    backend: e.backend,
    runtime: e.runtime,
    threads: e.threads,
    gpu: e.gpu,
    note: e.note,
    chip: document.getElementById('chip-text').textContent,
    statBackend: document.getElementById('stat-backend').textContent,
    isolated: crossOriginIsolated,
    steps: result.generated,
    tokensPerSecond: result.tokensPerSecond,
    wallSeconds: (performance.now() - started) / 1000,
    text: result.text,
    tokenIds: result.tokenIds ? result.tokenIds.slice(0, 10) : null,
  };
}

mkdirSync(OUT, { recursive: true });
const only = process.argv.slice(2);
const wanted = only.length ? only : ['wasm', 'webgpu'];
const preview = await startPreview();
const results = {};
let failed = false;

try {
  for (const backend of wanted) {
    const chrome = await launchChrome(backend === 'webgpu' ? GPU_FLAGS : []);
    try {
      await chrome.goto(`${preview.url}/?backend=${backend}`);
      const r = await chrome.run(inPage, 100);
      results[backend] = { browser: await chrome.version(), ...r, console: chrome.consoleLines.slice(-6) };
      await chrome.screenshot(join(OUT, `ui-${backend}.png`));
      console.log(backend, JSON.stringify({ ...results[backend], text: undefined }));
      if (backend === 'webgpu' && r.backend !== 'webgpu') failed = true;
      if (backend === 'wasm' && r.backend !== 'wasm') failed = true;
    } catch (err) {
      failed = true;
      results[backend] = { error: String(err.message ?? err), console: chrome.consoleLines.slice(-10) };
      console.log(backend, 'FAILED', JSON.stringify(results[backend]));
      try {
        await chrome.screenshot(join(OUT, `ui-${backend}-failed.png`));
      } catch {
        /* page may be gone */
      }
    } finally {
      await chrome.close();
    }
  }
} finally {
  preview.stop();
}

writeFileSync(join(OUT, 'browser.json'), JSON.stringify(results, null, 2) + '\n');
console.log(`Wrote ${join(OUT, 'browser.json')}`);
process.exit(failed ? 1 : 0);
