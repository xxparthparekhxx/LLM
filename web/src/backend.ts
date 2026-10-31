// Picks an onnxruntime-web build and creates the inference session.
//
// Order of preference (the same chain as the chess page):
//   1. WebGPU through onnxruntime-web's JSPI build (bundled, ~17 MiB .wasm). Needs WebAssembly JSPI (Chrome / Edge 137+).
//   2. WebGPU through the asyncify build. Its .wasm is 28 MiB, over Cloudflare's 25 MiB per-file limit,
//      so it is loaded from config.ortExternalBase (default: jsDelivr).
//   3. CPU through the plain WebAssembly build (bundled). Multi-threaded only when the page is cross-origin
//      isolated (COOP + COEP headers); see threads().
// WebGL is skipped: it cannot be trusted with this graph's dynamic cache dimensions, and the CPU path is
// the useful fallback.
// A failure in 1 or 2 falls through to the next option and is reported in the UI.

import { ORT_BASE } from './config';

export type OrtModule = typeof import('onnxruntime-web');
export type Session = import('onnxruntime-web').InferenceSession;

export type BackendName = 'webgpu' | 'wasm';
export type Runtime = 'jspi' | 'asyncify' | 'wasm';
export type BackendPreference = 'auto' | BackendName;

export interface CreatedSession {
  ort: OrtModule;
  session: Session;
  backend: BackendName;
  runtime: Runtime;
  threads: number;
  /** Why a GPU backend was not used, if it was not */
  note: string | null;
  gpu: string | null;
}

interface GpuNavigator {
  gpu?: { requestAdapter(): Promise<{ info?: { vendor?: string; architecture?: string; description?: string } } | null> };
}

async function probeWebGpu(): Promise<{ ok: boolean; note: string | null; gpu: string | null }> {
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

function hasJspi(): boolean {
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

export async function createSession(
  bytes: Uint8Array,
  preference: BackendPreference,
  externalBase: string,
  onStep: (message: string) => void,
): Promise<CreatedSession> {
  const notes: string[] = [];
  const attempts: Runtime[] = [];
  let gpu: string | null = null;

  if (preference !== 'wasm') {
    const probe = await probeWebGpu();
    gpu = probe.gpu;
    if (!probe.ok) notes.push(probe.note!);
    else if (hasJspi()) attempts.push('jspi');
    else if (externalBase) attempts.push('asyncify');
    else notes.push('WebGPU is available, but this browser has no WebAssembly JSPI and the external WebGPU runtime is disabled.');
  }
  if (preference !== 'webgpu') attempts.push('wasm');

  let lastError: unknown = null;
  for (const runtime of attempts) {
    try {
      onStep(runtime === 'wasm' ? 'Starting the CPU (WebAssembly) runtime' : 'Starting the WebGPU runtime');
      const created = await tryRuntime(runtime, bytes, externalBase);
      return { ...created, note: notes.length ? notes.join(' ') : null, gpu };
    } catch (err) {
      lastError = err;
      console.warn(`onnxruntime-web (${runtime}) failed`, err);
      notes.push(`${runtime === 'wasm' ? 'CPU' : 'WebGPU'} runtime failed: ${(err as Error).message ?? err}`);
    }
  }
  throw lastError instanceof Error ? lastError : new Error(String(lastError));
}

async function tryRuntime(runtime: Runtime, bytes: Uint8Array, externalBase: string) {
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
  });

  return { ort, session, backend: ep, runtime, threads };
}
