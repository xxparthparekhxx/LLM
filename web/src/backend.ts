// Picks an onnxruntime-web build and creates the inference session.
//
// When preference is 'auto' and WebGPU is available, it times a short warm-up
// plus 12 decode steps on WebGPU and on WASM, picks the faster backend,
// disposes the other session, and caches the choice in localStorage keyed by
// model version, ORT version, and GPU adapter description.
//
// Order of preference for explicit backends:
//   1. WebGPU through onnxruntime-web's JSPI build (bundled, ~17 MiB .wasm). Needs WebAssembly JSPI (Chrome / Edge 137+).
//   2. WebGPU through the asyncify build. Its .wasm is 28 MiB, over Cloudflare's 25 MiB per-file limit,
//      so it is loaded from config.ortExternalBase (default: jsDelivr).
//   3. CPU through the plain WebAssembly build (bundled). Multi-threaded only when the page is cross-origin
//      isolated (COOP + COEP headers); see threads().

import { ORT_BASE, ORT_VERSION, config } from './config';
import { generate } from './generate';

export type OrtModule = typeof import('onnxruntime-web');
export type Session = import('onnxruntime-web').InferenceSession;

export type BackendName = 'webgpu' | 'wasm';
export type Runtime = 'jspi' | 'asyncify' | 'wasm';
export type BackendPreference = 'auto' | BackendName;

export interface BenchmarkResult {
  webgpuTokPerSec: number;
  wasmTokPerSec: number;
  choice: BackendName;
  durationMs: number;
  cached?: boolean;
}

export interface CreatedSession {
  ort: OrtModule;
  session: Session;
  backend: BackendName;
  runtime: Runtime;
  threads: number;
  /** Why a GPU backend was not used, if it was not */
  note: string | null;
  gpu: string | null;
  benchmark?: BenchmarkResult | null;
}

interface GpuNavigator {
  gpu?: { requestAdapter(): Promise<{ info?: { vendor?: string; architecture?: string; description?: string } } | null> };
}

export async function probeWebGpu(): Promise<{ ok: boolean; note: string | null; gpu: string | null }> {
  const gpu = (navigator as unknown as GpuNavigator).gpu;
  if (!gpu) return { ok: false, note: 'This browser does not expose WebGPU (navigator.gpu is missing).', gpu: null };
  try {
    const adapter = await gpu.requestAdapter();
    if (!adapter) return { ok: false, note: 'WebGPU is present but no GPU adapter is available (blocklisted GPU or driver).', gpu: null };
    const i = adapter.info;
    return { ok: true, note: null, gpu: [i?.vendor, i?.architecture, i?.description].filter(Boolean).join(' ') || null };
  } catch (err) {
    return { ok: false, note: `WebGPU adapter request failed: ${(err as Error).message}`, gpu: null };
  }
}

export function hasJspi(): boolean {
  const wa = WebAssembly as unknown as Record<string, unknown>;
  return typeof wa.Suspending === 'function' && typeof wa.promising === 'function';
}

export const MAX_CPU_THREADS = 8;

/** Threads for the CPU runtime. Without cross-origin isolation there are no threads at all. */
export function cpuThreads(hardwareConcurrency: number, isolated: boolean): number {
  if (!isolated) return 1;
  const cores = Number.isFinite(hardwareConcurrency) && hardwareConcurrency > 0 ? Math.floor(hardwareConcurrency) : 2;
  return Math.max(1, Math.min(cores - 1, MAX_CPU_THREADS));
}

export function getAutoBackendCacheKey(gpuDescription: string | null): string {
  return `tinystories_auto_backend:${config.modelVersion}:${ORT_VERSION}:${gpuDescription || 'unknown'}`;
}

export function saveCachedBackendChoice(
  choice: BackendName,
  gpuDescription: string | null,
  prevBench?: BenchmarkResult | null,
) {
  try {
    const key = getAutoBackendCacheKey(gpuDescription);
    const data: BenchmarkResult = {
      choice,
      webgpuTokPerSec: prevBench?.webgpuTokPerSec ?? 0,
      wasmTokPerSec: prevBench?.wasmTokPerSec ?? 0,
      durationMs: prevBench?.durationMs ?? 0,
      cached: true,
    };
    localStorage.setItem(key, JSON.stringify(data));
  } catch (err) {
    console.warn('Could not save backend choice to localStorage', err);
  }
}

export async function benchmarkSession(ort: OrtModule, session: Session): Promise<number> {
  const prompt = 'Once upon a time';
  // Warm-up: prefill + 1 decode step
  await generate(ort, session, prompt, {
    maxNewTokens: 1,
    sampling: () => ({ temperature: 0, topK: 50, topP: 1 }),
    signal: new AbortController().signal,
    onToken: () => {},
    onStats: () => {},
  });

  // Timed: 12 decode steps
  const res = await generate(ort, session, prompt, {
    maxNewTokens: 12,
    sampling: () => ({ temperature: 0, topK: 50, topP: 1 }),
    signal: new AbortController().signal,
    onToken: () => {},
    onStats: () => {},
  });
  return res.tokensPerSecond;
}

export async function createSession(
  bytes: Uint8Array,
  preference: BackendPreference,
  externalBase: string,
  onStep: (message: string) => void,
): Promise<CreatedSession> {
  const notes: string[] = [];
  let gpu: string | null = null;

  if (preference === 'auto') {
    const probe = await probeWebGpu();
    gpu = probe.gpu;
    if (!probe.ok) {
      if (probe.note) notes.push(probe.note);
      onStep('Starting the CPU (WebAssembly) runtime');
      const created = await tryRuntime('wasm', bytes, externalBase);
      return { ...created, note: notes.join(' ') || null, gpu: null, benchmark: null };
    }

    const cacheKey = getAutoBackendCacheKey(gpu);
    let cached: BenchmarkResult | null = null;
    try {
      const raw = localStorage.getItem(cacheKey);
      if (raw) cached = JSON.parse(raw);
    } catch {}

    if (cached && (cached.choice === 'webgpu' || cached.choice === 'wasm')) {
      const targetRuntime = cached.choice === 'webgpu' ? (hasJspi() ? 'jspi' : 'asyncify') : 'wasm';
      onStep(targetRuntime === 'wasm' ? 'Starting the CPU runtime (cached choice)' : 'Starting the WebGPU runtime (cached choice)');
      const created = await tryRuntime(targetRuntime, bytes, externalBase);
      return { ...created, note: null, gpu, benchmark: { ...cached, cached: true } };
    }

    // Benchmark both WebGPU and WASM
    const t0 = performance.now();
    let gpuCreated: Awaited<ReturnType<typeof tryRuntime>> | null = null;
    let webgpuTokPerSec = 0;
    try {
      const gpuRuntime = hasJspi() ? 'jspi' : 'asyncify';
      onStep('Benchmarking WebGPU...');
      gpuCreated = await tryRuntime(gpuRuntime, bytes, externalBase);
      webgpuTokPerSec = await benchmarkSession(gpuCreated.ort, gpuCreated.session);
    } catch (err) {
      console.warn('WebGPU benchmark failed, falling back to CPU', err);
      notes.push(`WebGPU failed: ${(err as Error).message ?? err}`);
      if (gpuCreated) {
        try { await gpuCreated.session.release(); } catch {}
      }
      onStep('Starting the CPU (WebAssembly) runtime');
      const wasmCreated = await tryRuntime('wasm', bytes, externalBase);
      return { ...wasmCreated, note: notes.join(' '), gpu, benchmark: null };
    }

    let wasmCreated: Awaited<ReturnType<typeof tryRuntime>> | null = null;
    let wasmTokPerSec = 0;
    try {
      onStep('Benchmarking CPU (WebAssembly)...');
      wasmCreated = await tryRuntime('wasm', bytes, externalBase);
      wasmTokPerSec = await benchmarkSession(wasmCreated.ort, wasmCreated.session);
    } catch (err) {
      console.warn('CPU benchmark failed, using WebGPU', err);
      notes.push(`CPU failed: ${(err as Error).message ?? err}`);
      if (wasmCreated) {
        try { await wasmCreated.session.release(); } catch {}
      }
      return { ...gpuCreated, note: notes.join(' '), gpu, benchmark: null };
    }

    const durationMs = performance.now() - t0;
    const choice: BackendName = webgpuTokPerSec >= wasmTokPerSec ? 'webgpu' : 'wasm';
    const benchmark: BenchmarkResult = {
      choice,
      webgpuTokPerSec,
      wasmTokPerSec,
      durationMs,
      cached: false,
    };

    try {
      localStorage.setItem(cacheKey, JSON.stringify(benchmark));
    } catch {}

    if (choice === 'webgpu') {
      await wasmCreated.session.release();
      return { ...gpuCreated, note: notes.length ? notes.join(' ') : null, gpu, benchmark };
    } else {
      await gpuCreated.session.release();
      return { ...wasmCreated, note: notes.length ? notes.join(' ') : null, gpu, benchmark };
    }
  }

  // Explicit backend preference
  const attempts: Runtime[] = [];
  if (preference === 'webgpu') {
    const probe = await probeWebGpu();
    gpu = probe.gpu;
    if (!probe.ok) notes.push(probe.note!);
    else if (hasJspi()) attempts.push('jspi');
    else if (externalBase) attempts.push('asyncify');
    else notes.push('WebGPU is available, but this browser has no WebAssembly JSPI and the external WebGPU runtime is disabled.');
  } else {
    attempts.push('wasm');
  }

  let lastError: unknown = null;
  for (const runtime of attempts) {
    try {
      onStep(runtime === 'wasm' ? 'Starting the CPU (WebAssembly) runtime' : 'Starting the WebGPU runtime');
      const created = await tryRuntime(runtime, bytes, externalBase);
      return { ...created, note: notes.length ? notes.join(' ') : null, gpu, benchmark: null };
    } catch (err) {
      lastError = err;
      console.warn(`onnxruntime-web (${runtime}) failed`, err);
      notes.push(`${runtime === 'wasm' ? 'CPU' : 'WebGPU'} runtime failed: ${(err as Error).message ?? err}`);
    }
  }
  throw lastError instanceof Error ? lastError : new Error(String(lastError));
}

export async function tryRuntime(runtime: Runtime, bytes: Uint8Array, externalBase: string) {
  const isolated = typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated;
  // onnxruntime-web resolves file names against this with new URL(), so it must be absolute.
  const localBase = new URL(ORT_BASE, self.location.href).href;
  let ort: OrtModule;
  let ep: 'webgpu' | 'wasm';
  let threads: number;

  if (runtime === 'jspi') {
    ort = (await import('onnxruntime-web/jspi')) as OrtModule;
    ort.env.wasm.wasmPaths = localBase;
    ep = 'webgpu';
    threads = 1;
  } else if (runtime === 'asyncify') {
    ort = (await import('onnxruntime-web/webgpu')) as OrtModule;
    ort.env.wasm.wasmPaths = externalBase;
    ep = 'webgpu';
    threads = 1;
  } else {
    ort = (await import('onnxruntime-web/wasm')) as OrtModule;
    ort.env.wasm.wasmPaths = localBase;
    ep = 'wasm';
    threads = cpuThreads(navigator.hardwareConcurrency, isolated);
  }

  ort.env.wasm.numThreads = threads;
  ort.env.wasm.proxy = false;
  if (ep === 'webgpu') (ort.env as unknown as { webgpu: Record<string, unknown> }).webgpu.powerPreference = 'high-performance';

  const session = await ort.InferenceSession.create(bytes, {
    executionProviders: [ep],
    graphOptimizationLevel: 'all',
    ...(ep === 'webgpu'
      ? {
          preferredOutputLocation: {
            last_logits: 'cpu',
            logits: 'cpu',
            present_keys: 'gpu-buffer',
            present_values: 'gpu-buffer',
          },
        }
      : {}),
  });

  return { ort, session, backend: ep, runtime, threads };
}
