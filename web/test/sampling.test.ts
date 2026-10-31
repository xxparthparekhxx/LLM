// The sampler mirrors sample_next in llm/model.py. These tests pin the parts that
// can be checked without torch: greedy, top-k, top-p and the draw itself.
import { describe, expect, it } from 'vitest';
import { argmax, sampleToken, type SamplingOptions } from '../src/sampling';

const options = (overrides: Partial<SamplingOptions> = {}): SamplingOptions => ({
    temperature: 1,
    topK: 0,
    topP: 1,
    ...overrides,
});

describe('argmax', () => {
    it('returns the first maximum', () => {
        expect(argmax([1, 5, 5, 2])).toBe(1);
    });
});

describe('sampleToken', () => {
    it('is greedy at temperature 0, whatever the filters say', () => {
        const logits = new Float32Array([0, 4, 1]);
        expect(sampleToken(logits, options({ temperature: 0, topK: 1 }), () => 0.99)).toBe(1);
    });

    it('top-k = 1 always picks the best token', () => {
        const logits = new Float32Array([3, 2, 1]);
        for (const r of [0, 0.25, 0.5, 0.99]) expect(sampleToken(logits, options({ topK: 1 }), () => r)).toBe(0);
    });

    it('never picks outside the top-k set', () => {
        const logits = new Float32Array([10, 9, 1, 0]);
        for (let r = 0; r < 1; r += 0.01) expect(sampleToken(logits, options({ topK: 2 }), () => r)).toBeLessThan(2);
    });

    it('top-p keeps the smallest set that covers the mass', () => {
        // The first token has ~99.99% of the mass, so a 0.5 top-p leaves only it.
        const logits = new Float32Array([10, 0, 0]);
        for (const r of [0, 0.5, 0.999]) expect(sampleToken(logits, options({ topP: 0.5 }), () => r)).toBe(0);
    });

    it('draws in proportion to the probabilities', () => {
        // Two equal logits: half the mass each, so the draw decides.
        const logits = new Float32Array([0, 0]);
        expect(sampleToken(logits, options(), () => 0)).toBe(0);
        expect(sampleToken(logits, options(), () => 0.49)).toBe(0);
        expect(sampleToken(logits, options(), () => 0.51)).toBe(1);
        expect(sampleToken(logits, options(), () => 0.99)).toBe(1);
    });

    it('handles -Infinity from a top-k mask without NaN', () => {
        const logits = new Float32Array([2, -Infinity, -Infinity]);
        expect(sampleToken(logits, options({ topK: 1 }), () => 0.9)).toBe(0);
    });
});
