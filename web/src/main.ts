import '@fontsource-variable/outfit/index.css';
import './style.css';
import { config, MODEL } from './config';
import {
  createSession,
  saveCachedBackendChoice,
  type BackendName,
  type BackendPreference,
  type CreatedSession,
} from './backend';
import { clearModelCache, loadModel } from './modelLoader';
import { generate } from './generate';

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

const chip = $<HTMLDivElement>('backend-chip');
const chipText = $<HTMLSpanElement>('chip-text');
const whyButton = $<HTMLButtonElement>('why-btn');
const promptEl = $<HTMLTextAreaElement>('prompt');
const generateButton = $<HTMLButtonElement>('generate');
const stopButton = $<HTMLButtonElement>('stop');
const promptSpan = $<HTMLSpanElement>('story-prompt');
const storyText = $<HTMLSpanElement>('story-text');
const loader = $<HTMLDivElement>('loader');
const loaderTitle = $<HTMLHeadingElement>('loader-title');
const loaderBar = $<HTMLDivElement>('loader-bar');
const loaderTrack = $<HTMLDivElement>('loader-track');
const loaderText = $<HTMLDivElement>('loader-text');
const loaderRetry = $<HTMLButtonElement>('loader-retry');
const errorBox = $<HTMLDivElement>('error');
const statBackend = $<HTMLElement>('stat-backend');
const statSpeed = $<HTMLElement>('stat-speed');
const statTokens = $<HTMLElement>('stat-tokens');
const statPrefill = $<HTMLElement>('stat-prefill');
const statRuntime = $<HTMLElement>('stat-runtime');
const statGpu = $<HTMLElement>('stat-gpu');
const statThreads = $<HTMLElement>('stat-threads');
const statBenchmark = $<HTMLDivElement>('stat-benchmark');
const backendNote = $<HTMLDivElement>('backend-note');
const backendSelect = $<HTMLSelectElement>('backend-select');
const temperature = $<HTMLInputElement>('temperature');
const temperatureValue = $<HTMLOutputElement>('temperature-value');
const topK = $<HTMLInputElement>('top-k');
const topKValue = $<HTMLOutputElement>('top-k-value');
const topP = $<HTMLInputElement>('top-p');
const topPValue = $<HTMLOutputElement>('top-p-value');

let engine: CreatedSession | null = null;
let controller: AbortController | null = null;
let loading = false;
let modelBytes: Uint8Array | null = null;

const formatBytes = (bytes: number) => `${(bytes / 1e6).toFixed(1)} MB`;

function setChip(state: 'loading' | BackendName | 'error', text: string) {
  chip.className = `chip ${state}`;
  chipText.textContent = text;
}

function backendLabel(created: CreatedSession): string {
  return created.backend === 'webgpu' ? 'WebGPU' : `CPU · ${created.threads} thread${created.threads === 1 ? '' : 's'}`;
}

function setBusy(busy: boolean) {
  generateButton.disabled = busy || !engine;
  stopButton.disabled = !busy;
  promptEl.disabled = busy;
  backendSelect.disabled = busy || !engine;
}

function showError(message: string) {
  errorBox.hidden = false;
  errorBox.textContent = message;
}

function clearError() {
  errorBox.hidden = true;
  errorBox.textContent = '';
}

function preferenceFromUrl(): BackendPreference {
  const value = new URLSearchParams(location.search).get('backend');
  return value === 'wasm' || value === 'webgpu' ? value : 'auto';
}

function applyEngineToUI(created: CreatedSession) {
  setChip(created.backend, backendLabel(created));
  statBackend.textContent = backendLabel(created);
  statRuntime.textContent = created.runtime;
  statGpu.textContent = created.gpu || '-';
  statThreads.textContent = String(created.threads);
  backendSelect.value = created.backend;

  if (created.benchmark) {
    const b = created.benchmark;
    statBenchmark.hidden = false;
    statBenchmark.textContent = `WebGPU ${Math.round(b.webgpuTokPerSec)} tok/s · CPU ${Math.round(b.wasmTokPerSec)} tok/s → using ${created.backend === 'webgpu' ? 'WebGPU' : 'CPU'}`;
    const details = `Auto-selected ${created.backend === 'webgpu' ? 'WebGPU' : 'CPU'} (WebGPU ${b.webgpuTokPerSec.toFixed(1)} tok/s, CPU ${b.wasmTokPerSec.toFixed(1)} tok/s). Timing step took ${b.durationMs.toFixed(0)} ms${b.cached ? ' (cached)' : ''}.`;
    backendNote.textContent = created.note ? `${details} ${created.note}` : details;
  } else {
    statBenchmark.hidden = true;
    statBenchmark.textContent = '';
    backendNote.textContent = created.note || (created.backend === 'webgpu' ? 'WebGPU is in use.' : 'CPU is in use.');
  }

  whyButton.hidden = created.backend === 'webgpu';
}

async function boot() {
  if (loading) return;
  loading = true;
  clearError();
  loader.hidden = false;
  loaderRetry.hidden = true;
  loaderTitle.textContent = 'Loading the model';
  loaderBar.style.width = '0%';
  loaderTrack.setAttribute('aria-valuenow', '0');
  loaderText.textContent = 'Checking the browser cache';
  setChip('loading', 'Loading model');

  try {
    const { bytes, fromCache } = await loadModel(config.modelUrl, config.modelVersion, (loaded, total, cached) => {
      const percent = total ? Math.min(100, (loaded / total) * 100) : 0;
      loaderBar.style.width = `${percent}%`;
      loaderTrack.setAttribute('aria-valuenow', String(Math.round(percent)));
      loaderText.textContent = total
        ? `${cached ? "From this browser's cache: " : 'Downloading model: '}${formatBytes(loaded)} of ${formatBytes(total)}`
        : `${cached ? "From this browser's cache: " : 'Downloading model: '}${formatBytes(loaded)}`;
    });

    modelBytes = bytes;
    loaderText.textContent = fromCache ? `Using the copy in this browser (${formatBytes(bytes.byteLength)})` : `${formatBytes(bytes.byteLength)} downloaded`;
    loaderTitle.textContent = 'Starting the runtime';

    engine = await createSession(bytes, preferenceFromUrl(), config.ortExternalBase, (message) => {
      loaderText.textContent = message;
    });

    loader.hidden = true;
    applyEngineToUI(engine);
    generateButton.disabled = false;
  } catch (err) {
    console.error(err);
    loaderTitle.textContent = 'Could not start';
    loaderText.textContent = (err as Error).message;
    loaderRetry.hidden = false;
    setChip('error', 'Model failed');
    showError(`The model could not be loaded: ${(err as Error).message}`);
    statBackend.textContent = '-';
  } finally {
    loading = false;
  }
}

function samplingOptions() {
  return {
    temperature: Number(temperature.value),
    topK: Number(topK.value),
    topP: Number(topP.value),
  };
}

function updateControls() {
  temperatureValue.textContent = Number(temperature.value).toFixed(2);
  topKValue.textContent = topK.value;
  topPValue.textContent = Number(topP.value).toFixed(2);
}

async function start() {
  if (!engine || controller) return;
  clearError();
  const prompt = promptEl.value.trim() ? promptEl.value : 'Once upon a time';
  controller = new AbortController();
  stopButton.textContent = 'Stop';
  setBusy(true);
  promptSpan.textContent = prompt;
  storyText.textContent = '';
  statSpeed.textContent = '-';
  statTokens.textContent = '0';
  statPrefill.textContent = '-';
  setChip(engine.backend, `${backendLabel(engine)} · generating`);

  try {
    const result = await generate(engine.ort, engine.session, prompt, {
      maxNewTokens: MODEL.maxNewTokens,
      sampling: samplingOptions,
      signal: controller.signal,
      onToken: (text) => {
        storyText.textContent = text;
      },
      onStats: (tokens, perSecond) => {
        statTokens.textContent = String(tokens);
        statSpeed.textContent = `${perSecond.toFixed(1)} tokens/s`;
      },
      onPrefill: (seconds) => {
        statPrefill.textContent = `${(seconds * 1000).toFixed(0)} ms`;
      },
    });
    storyText.textContent = result.text;
    statSpeed.textContent = `${result.tokensPerSecond.toFixed(1)} tokens/s`;
    statTokens.textContent = `${result.generated}${result.stopped ? ' (stopped)' : ''}`;
    if (result.generated === 0 && !result.stopped) {
      showError('The model ended the story immediately. Try a different prompt or a higher temperature.');
    }
  } catch (err) {
    console.error(err);
    showError(`Generation failed: ${(err as Error).message}`);
  } finally {
    controller = null;
    setBusy(false);
    if (engine) setChip(engine.backend, backendLabel(engine));
  }
}

function stop() {
  controller?.abort();
  stopButton.disabled = true;
  stopButton.textContent = 'Stopping...';
}

generateButton.addEventListener('click', start);
stopButton.addEventListener('click', stop);
promptEl.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) {
    event.preventDefault();
    void start();
  }
});
loaderRetry.addEventListener('click', () => void boot());
whyButton.addEventListener('click', () => {
  $<HTMLDetailsElement>('engine-details').open = true;
  backendNote.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
});
for (const input of [temperature, topK, topP]) input.addEventListener('input', updateControls);

backendSelect.addEventListener('change', async () => {
  const targetBackend = backendSelect.value as BackendName;
  if (!engine || engine.backend === targetBackend || loading || !modelBytes) return;

  setBusy(true);
  backendSelect.disabled = true;
  setChip('loading', `Switching to ${targetBackend === 'webgpu' ? 'WebGPU' : 'CPU'}...`);
  clearError();

  try {
    const oldEngine = engine;
    const prevBenchmark = oldEngine.benchmark;
    await oldEngine.session.release();

    engine = await createSession(modelBytes, targetBackend, config.ortExternalBase, (message) => {
      setChip('loading', message);
    });

    const bench = prevBenchmark ? { ...prevBenchmark, choice: targetBackend } : null;
    engine.benchmark = bench;
    saveCachedBackendChoice(targetBackend, engine.gpu, bench);
    applyEngineToUI(engine);
  } catch (err) {
    console.error('Failed to switch backend', err);
    showError(`Failed to switch backend: ${(err as Error).message}`);
    if (engine) {
      applyEngineToUI(engine);
    }
  } finally {
    setBusy(false);
  }
});

temperature.value = '0.8';
topK.value = '50';
topP.value = '1';
updateControls();
setBusy(false);
void boot();

// Exposed for the browser tests and for anyone poking at the console.
Object.assign(window, {
  __demo: {
    engine: () => engine,
    benchmark: () => engine?.benchmark ?? null,
    switchBackend: async (backend: BackendName) => {
      if (backendSelect.value !== backend) {
        backendSelect.value = backend;
        backendSelect.dispatchEvent(new Event('change'));
      }
    },
    generate: async (prompt: string, maxNewTokens = 20) => {
      if (!engine) throw new Error('model is not loaded yet');
      return generate(engine.ort, engine.session, prompt, {
        maxNewTokens,
        sampling: samplingOptions,
        signal: new AbortController().signal,
        onToken: () => {},
        onStats: () => {},
      });
    },
    clearModelCache,
  },
});
