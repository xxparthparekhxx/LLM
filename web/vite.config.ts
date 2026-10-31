import { createReadStream, existsSync, readFileSync, readdirSync, statSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { defaultClientConditions, defineConfig, type Plugin } from 'vite';

const root = dirname(fileURLToPath(import.meta.url));
const MODEL_DIR = resolve(root, 'model');
const ORT_DIST = resolve(root, 'node_modules', 'onnxruntime-web', 'dist');
const ORT_VERSION: string = JSON.parse(readFileSync(resolve(root, 'node_modules', 'onnxruntime-web', 'package.json'), 'utf8')).version;
// The two runtimes the site ships: JSPI (WebGPU) and plain WebAssembly (CPU). The version is in the
// path, so the files can be cached forever. The 28 MiB asyncify / jsep builds are not copied.
const ORT_PREFIX = `/ort/${ORT_VERSION}/`;
const ORT_FILES: Record<string, string> = {
  'ort-wasm-simd-threaded.jspi.mjs': 'text/javascript',
  'ort-wasm-simd-threaded.jspi.wasm': 'application/wasm',
  'ort-wasm-simd-threaded.mjs': 'text/javascript',
  'ort-wasm-simd-threaded.wasm': 'application/wasm',
};
const CF_LIMIT = 25 * 1024 * 1024; // Cloudflare static assets: 25 MiB per file

// Same isolation headers as public/_headers, so dev and preview behave like production
// (WebAssembly threads need cross-origin isolation).
const isolationHeaders = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
};

/**
 * Serves model/* at /model/* in `vite` and `vite preview`. The model is deliberately not under
 * public/, so it never gets copied into dist/ (it is over Cloudflare's per-file limit and lives on
 * Hugging Face in production).
 */
function serveLocalModel(): Plugin {
  const handler = (req: { url?: string }, res: import('node:http').ServerResponse, next: () => void) => {
    const url = (req.url ?? '').split('?')[0]!;
    if (!url.startsWith('/model/')) return next();
    const file = join(MODEL_DIR, decodeURIComponent(url.slice('/model/'.length)));
    if (!file.startsWith(MODEL_DIR) || !existsSync(file) || !statSync(file).isFile()) {
      res.statusCode = 404;
      res.end(`Not found: ${url}. Create it with ../export_onnx.py (see README.md).`);
      return;
    }
    for (const [k, v] of Object.entries(isolationHeaders)) res.setHeader(k, v);
    res.setHeader('Content-Type', 'application/octet-stream');
    res.setHeader('Content-Length', String(statSync(file).size));
    res.setHeader('Cache-Control', 'no-cache');
    createReadStream(file).pipe(res);
  };
  return {
    name: 'serve-local-model',
    configureServer: (server) => void server.middlewares.use(handler),
    configurePreviewServer: (server) => void server.middlewares.use(handler),
  };
}

/**
 * Puts the onnxruntime-web runtime files at /ort/<version>/ in dev, preview and the production build,
 * unprocessed. (Vite would otherwise try to transform the Emscripten glue in dev.)
 */
function ortRuntime(): Plugin {
  const handler = (req: { url?: string }, res: import('node:http').ServerResponse, next: () => void) => {
    const url = (req.url ?? '').split('?')[0]!;
    if (!url.startsWith(ORT_PREFIX)) return next();
    const name = url.slice(ORT_PREFIX.length);
    const type = ORT_FILES[name];
    if (!type) {
      res.statusCode = 404;
      res.end('Not found');
      return;
    }
    // The runtime's threads start as dedicated workers from the .mjs file, and a worker script must carry
    // the same COEP header as the page that starts it.
    for (const [k, v] of Object.entries(isolationHeaders)) res.setHeader(k, v);
    res.setHeader('Content-Type', type);
    res.setHeader('Content-Length', String(statSync(join(ORT_DIST, name)).size));
    res.setHeader('Cache-Control', 'no-cache');
    createReadStream(join(ORT_DIST, name)).pipe(res);
  };
  return {
    name: 'ort-runtime',
    configureServer: (server) => void server.middlewares.use(handler),
    configurePreviewServer: (server) => void server.middlewares.use(handler),
    generateBundle() {
      for (const name of Object.keys(ORT_FILES)) {
        this.emitFile({ type: 'asset', fileName: `ort/${ORT_VERSION}/${name}`, source: readFileSync(join(ORT_DIST, name)) });
      }
    },
  };
}

/** Fails the build if anything in dist/ would not upload to Cloudflare static assets. */
function checkDistLimits(): Plugin {
  return {
    name: 'check-dist-limits',
    apply: 'build',
    closeBundle() {
      const dist = resolve(root, 'dist');
      if (!existsSync(dist)) return;
      const bad: string[] = [];
      const walk = (dir: string) => {
        for (const name of readdirSync(dir)) {
          const p = join(dir, name);
          const st = statSync(p);
          if (st.isDirectory()) walk(p);
          else if (st.size > CF_LIMIT) bad.push(`${p.slice(dist.length + 1)} (${(st.size / 1048576).toFixed(1)} MiB)`);
        }
      };
      walk(dist);
      if (existsSync(join(dist, 'model'))) bad.push('model/ (the network must not be part of the site)');
      if (bad.length) throw new Error(`dist/ has files Cloudflare static assets would reject or that do not belong:\n  ${bad.join('\n  ')}`);
    },
  };
}

export default defineConfig({
  plugins: [serveLocalModel(), ortRuntime(), checkDistLimits()],
  define: { __ORT_VERSION__: JSON.stringify(ORT_VERSION), __ORT_BASE__: JSON.stringify(ORT_PREFIX) },
  resolve: {
    // Use onnxruntime-web's builds that keep the Emscripten glue (.mjs) and the .wasm as separate files,
    // so worker threads start from the glue file rather than from our own worker bundle.
    conditions: ['onnxruntime-web-use-extern-wasm', ...defaultClientConditions],
  },
  optimizeDeps: { exclude: ['onnxruntime-web'] },
  worker: { format: 'es' },
  build: {
    target: 'es2022',
    chunkSizeWarningLimit: 1500,
  },
  server: { headers: isolationHeaders },
  preview: { headers: isolationHeaders },
});
