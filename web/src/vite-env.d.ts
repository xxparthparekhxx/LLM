/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** URL of the ONNX file. Default: /model/model_fp16.onnx (served by the dev server from web/model/). */
  readonly VITE_MODEL_URL?: string;
  /** Bump to make returning visitors download the model again (it is part of the cache name). */
  readonly VITE_MODEL_VERSION?: string;
  /** Where the large onnxruntime-web WebGPU runtime (asyncify build) is loaded from. Empty = do not use it. */
  readonly VITE_ORT_EXTERNAL_BASE?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}

/** onnxruntime-web version from node_modules, injected by vite.config.ts */
declare const __ORT_VERSION__: string;
/** URL prefix the bundled onnxruntime-web runtime files are served from, e.g. /ort/1.31.1/ */
declare const __ORT_BASE__: string;
