// Build-time configuration (Vite env variables, see web/.env.example).

/** onnxruntime-web version the app is built against (read from node_modules at build time). */
export const ORT_VERSION = __ORT_VERSION__;

/** Where the bundled onnxruntime-web runtime files (JSPI and plain WASM) are served from. */
export const ORT_BASE = __ORT_BASE__;

const env = import.meta.env;

export const config = {
  /** Where the ONNX file is fetched from. A relative path is served by the dev server from web/model/. */
  modelUrl: env.VITE_MODEL_URL || '/model/model_fp16.onnx',

  /** Part of the Cache API cache name; change it when you publish a new ONNX file. */
  modelVersion: env.VITE_MODEL_VERSION || '1',

  /**
   * onnxruntime-web's WebGPU build for browsers without JSPI ships a 28 MiB .wasm, over Cloudflare's
   * 25 MiB per-file limit, so it is not part of the site. Browsers that have WebGPU but not JSPI load
   * it from here instead. Set VITE_ORT_EXTERNAL_BASE to an empty string to disable that path.
   */
  ortExternalBase:
    env.VITE_ORT_EXTERNAL_BASE !== undefined
      ? env.VITE_ORT_EXTERNAL_BASE
      : `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`,
};

/** Architecture of the checkpoint, from its config.json. */
export const MODEL = {
  params: 33_086_848,
  layers: 8,
  kvHeads: 2,
  headDim: 64,
  heads: 6,
  dModel: 384,
  vocab: 50304,
  context: 512,
  maxNewTokens: 300,
};
