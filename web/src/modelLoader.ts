// Downloads the ONNX file with progress and keeps it in the Cache API, so a returning
// visitor does not download ~67 MiB again. The cache name contains the model version
// (VITE_MODEL_VERSION); older versions are deleted.

const CACHE_PREFIX = 'tinystories-model-';

export interface LoadedModel {
  bytes: Uint8Array;
  fromCache: boolean;
}

export async function loadModel(
  url: string,
  version: string,
  onProgress: (loaded: number, total: number | undefined, fromCache: boolean) => void,
): Promise<LoadedModel> {
  const cacheName = CACHE_PREFIX + version;
  let cache: Cache | null = null;
  try {
    cache = await caches.open(cacheName);
    for (const name of await caches.keys()) {
      if (name.startsWith(CACHE_PREFIX) && name !== cacheName) await caches.delete(name);
    }
  } catch {
    cache = null; // Cache API unavailable (private mode in some browsers): just download
  }

  if (cache) {
    const hit = await cache.match(url);
    if (hit) {
      const buf = new Uint8Array(await hit.arrayBuffer());
      if (buf.byteLength > 1_000_000) {
        onProgress(buf.byteLength, buf.byteLength, true);
        return { bytes: buf, fromCache: true };
      }
      await cache.delete(url); // truncated or wrong entry
    }
  }

  const res = await fetch(url, { mode: 'cors', credentials: 'omit' });
  if (!res.ok) throw new Error(`Could not download the model: HTTP ${res.status} from ${url}`);
  const headerLength = Number(res.headers.get('content-length') ?? res.headers.get('x-linked-size') ?? 0);
  const total = headerLength > 0 ? headerLength : undefined;
  if (!res.body) {
    const buf = new Uint8Array(await res.arrayBuffer());
    await store(cache, url, buf);
    return { bytes: buf, fromCache: false };
  }

  const reader = res.body.getReader();
  let received = 0;
  let buffer = total ? new Uint8Array(total) : null;
  const chunks: Uint8Array[] = [];
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (buffer && received + value.byteLength <= buffer.byteLength) buffer.set(value, received);
    else {
      // size header was missing or wrong (for example a compressed transfer): fall back to collecting chunks
      if (buffer) {
        chunks.push(buffer.slice(0, received));
        buffer = null;
      }
      chunks.push(value);
    }
    received += value.byteLength;
    onProgress(received, total, false);
  }
  let bytes: Uint8Array;
  if (buffer && received === buffer.byteLength) bytes = buffer;
  else if (buffer) bytes = buffer.slice(0, received);
  else {
    bytes = new Uint8Array(received);
    let off = 0;
    for (const c of chunks) {
      bytes.set(c, off);
      off += c.byteLength;
    }
  }
  if (bytes.byteLength < 1_000_000) throw new Error(`The downloaded file is only ${bytes.byteLength} bytes; is the model URL right? (${url})`);
  await store(cache, url, bytes);
  return { bytes, fromCache: false };
}

async function store(cache: Cache | null, url: string, bytes: Uint8Array): Promise<void> {
  if (!cache) return;
  try {
    await cache.put(
      url,
      new Response(bytes as BodyInit, { headers: { 'content-type': 'application/octet-stream', 'content-length': String(bytes.byteLength) } }),
    );
  } catch (err) {
    console.warn('Could not cache the model (storage full or blocked); it will be downloaded again next time.', err);
  }
}

export async function clearModelCache(): Promise<boolean> {
  try {
    let any = false;
    for (const name of await caches.keys()) {
      if (name.startsWith(CACHE_PREFIX)) any = (await caches.delete(name)) || any;
    }
    return any;
  } catch {
    return false;
  }
}
