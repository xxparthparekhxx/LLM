// Cached token-by-token generation with the exported ONNX graph.
//
// One prefill call for the prompt, then one small call per new token that passes
// the previous present key/value tensors back as past. At the 512-token context
// limit the cache is dropped and the last 256 tokens are prefilled again, which is
// what Transformer.generate does in llm/model.py.

import { EOT, decode, encode } from './tokenizer';
import { sampleToken, type SamplingOptions } from './sampling';
import { MODEL } from './config';
import type { OrtModule, Session } from './backend';

export interface GenerateOptions {
  maxNewTokens: number;
  sampling: () => SamplingOptions;
  signal: AbortSignal;
  onToken: (text: string, generatedTokens: number) => void;
  onStats: (tokens: number, tokensPerSecond: number) => void;
  onPrefill?: (seconds: number) => void;
}

interface Cache {
  keys: import('onnxruntime-web').Tensor;
  values: import('onnxruntime-web').Tensor;
}

function emptyCache(ort: OrtModule) {
  return new ort.Tensor('float32', new Float32Array(0), [MODEL.layers, 1, MODEL.kvHeads, 0, MODEL.headDim]);
}

function causalMask(length: number, total: number, queryOffset: number): Uint8Array {
  const mask = new Uint8Array(length * total);
  for (let q = 0; q < length; q++) {
    const lastKey = queryOffset + q;
    for (let k = 0; k <= lastKey && k < total; k++) mask[q * total + k] = 1;
  }
  return mask;
}

async function run(
  ort: OrtModule,
  session: Session,
  ids: number[],
  positions: number[],
  past: Cache | null,
  mask: Uint8Array,
  total: number,
): Promise<{ logits: Float32Array; cache: Cache }> {
  const length = ids.length;
  const feeds = {
    input_ids: new ort.Tensor('int64', BigInt64Array.from(ids, BigInt), [1, length]),
    position_ids: new ort.Tensor('int64', BigInt64Array.from(positions, BigInt), [1, length]),
    past_keys: past ? past.keys : emptyCache(ort),
    past_values: past ? past.values : emptyCache(ort),
    attention_mask: new ort.Tensor('bool', mask, [length, total]),
  };
  const out = await session.run(feeds, ['last_logits', 'present_keys', 'present_values']);
  // onnxruntime-web returns an object keyed by output name; onnxruntime-node returns an array.
  const pick = (index: number, name: string) =>
    (Array.isArray(out) ? out[index] : (out as unknown as Record<string, import('onnxruntime-web').Tensor>)[name])!;
  return {
    logits: pick(0, 'last_logits').data as Float32Array,
    cache: { keys: pick(1, 'present_keys'), values: pick(2, 'present_values') },
  };
}

export interface GenerateResult {
  text: string;
  generated: number;
  tokenIds: number[];
  tokensPerSecond: number;
  stopped: boolean;
}

export async function generate(
  ort: OrtModule,
  session: Session,
  prompt: string,
  options: GenerateOptions,
): Promise<GenerateResult> {
  const promptIds = [EOT, ...encode(prompt)].slice(-(MODEL.context - 1));
  const started = performance.now();
  let { logits, cache } = await run(
    ort,
    session,
    promptIds,
    promptIds.map((_, i) => i),
    null,
    causalMask(promptIds.length, promptIds.length, 0),
    promptIds.length,
  );
  options.onPrefill?.((performance.now() - started) / 1000);

  let position = promptIds.length;
  const generated: number[] = [];
  const decodeStart = performance.now();
  let stopped = false;

  for (let i = 0; i < options.maxNewTokens; i++) {
    if (options.signal.aborted) {
      stopped = true;
      break;
    }
    const next = sampleToken(logits, options.sampling());
    if (next === EOT) break;
    generated.push(next);

    // Decode all generated tokens again and hold back a trailing incomplete UTF-8
    // character, so the page never shows a replacement character mid-character.
    const text = decode(generated).replace(/\ufffd+$/, '');
    options.onToken(text, generated.length);
    options.onStats(generated.length, generated.length / ((performance.now() - decodeStart) / 1000));

    if (position >= MODEL.context) {
      // Context full: slide the window and re-prefill from the latest half.
      const tail = promptIds.concat(generated).slice(-(MODEL.context >> 1));
      ({ logits, cache } = await run(
        ort,
        session,
        tail,
        tail.map((_, j) => j),
        null,
        causalMask(tail.length, tail.length, 0),
        tail.length,
      ));
      position = tail.length;
    } else {
      ({ logits, cache } = await run(
        ort,
        session,
        [next],
        [position],
        cache,
        new Uint8Array(position + 1).fill(1),
        position + 1,
      ));
      position += 1;
    }
  }

  const seconds = (performance.now() - decodeStart) / 1000;
  return {
    text: decode(generated),
    generated: generated.length,
    tokenIds: generated,
    tokensPerSecond: seconds > 0 ? generated.length / seconds : 0,
    stopped,
  };
}
