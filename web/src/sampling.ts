// Next-token sampling, following Transformer.generate / sample_next in llm/model.py:
// repetition penalty is not exposed (the page keeps it at 1, as in sample.py's default),
// temperature 0 is greedy, then top-k, softmax, top-p.

export interface SamplingOptions {
  temperature: number;
  topK: number;
  topP: number;
}

export const DEFAULTS: SamplingOptions = { temperature: 0.8, topK: 50, topP: 1.0 };

/** Index of the largest value; the first one on a tie, like torch.argmax. */
export function argmax(values: Float32Array | number[]): number {
  let best = 0;
  for (let i = 1; i < values.length; i++) if (values[i]! > values[best]!) best = i;
  return best;
}

/**
 * Threshold for top-k: the k-th largest logit. A small min-heap keeps this
 * O(vocab log k) instead of sorting all 50,304 logits for every token.
 */
function kthLargest(values: Float32Array, k: number): number {
  const heap: number[] = [];
  const up = (i: number) => {
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (heap[p]! <= heap[i]!) break;
      [heap[p], heap[i]] = [heap[i]!, heap[p]!];
      i = p;
    }
  };
  const down = (i: number) => {
    for (;;) {
      const l = i * 2 + 1;
      const r = l + 1;
      let m = i;
      if (l < heap.length && heap[l]! < heap[m]!) m = l;
      if (r < heap.length && heap[r]! < heap[m]!) m = r;
      if (m === i) break;
      [heap[m], heap[i]] = [heap[i]!, heap[m]!];
      i = m;
    }
  };
  for (let i = 0; i < values.length; i++) {
    const v = values[i]!;
    if (heap.length < k) {
      heap.push(v);
      up(heap.length - 1);
    } else if (v > heap[0]!) {
      heap[0] = v;
      down(0);
    }
  }
  return heap[0]!;
}

/**
 * Draw one token from (vocab,) logits. `random` is injectable so the tests can
 * pin the draw; the page passes Math.random.
 */
export function sampleToken(logits: Float32Array, options: SamplingOptions, random: () => number = Math.random): number {
  const { temperature, topK, topP } = options;
  if (!(temperature > 0)) return argmax(logits);

  const vocab = logits.length;
  const scaled = new Float32Array(vocab);
  for (let i = 0; i < vocab; i++) scaled[i] = logits[i]! / temperature;

  let threshold = -Infinity;
  if (topK > 0 && topK < vocab) threshold = kthLargest(scaled, Math.floor(topK));

  // Softmax over the surviving logits, in float64 so the cumulative top-p sum is stable.
  let max = -Infinity;
  for (let i = 0; i < vocab; i++) if (scaled[i]! >= threshold && scaled[i]! > max) max = scaled[i]!;
  const probs = new Float64Array(vocab);
  let sum = 0;
  for (let i = 0; i < vocab; i++) {
    if (scaled[i]! < threshold) {
      probs[i] = 0;
      continue;
    }
    const p = Math.exp(scaled[i]! - max);
    probs[i] = p;
    sum += p;
  }

  const order = new Uint32Array(vocab);
  for (let i = 0; i < vocab; i++) order[i] = i;
  const kept: number[] = [];
  let cumulative = 0;
  if (topP < 1) {
    const sorted = Array.from(order).sort((a, b) => probs[b]! - probs[a]!);
    for (const i of sorted) {
      if (probs[i]! === 0) break;
      // sample_next drops a token when the mass *before* it already exceeds top_p
      if (cumulative > topP) break;
      kept.push(i);
      cumulative += probs[i]! / sum;
    }
    if (kept.length === 0) kept.push(argmax(logits));
  } else {
    for (let i = 0; i < vocab; i++) if (probs[i]! > 0) kept.push(i);
  }

  let total = 0;
  for (const i of kept) total += probs[i]!;
  if (!(total > 0)) return argmax(logits);

  let r = random() * total;
  for (const i of kept) {
    r -= probs[i]!;
    if (r <= 0) return i;
  }
  return kept[kept.length - 1]!;
}
