// Headless end-to-end check and speed measurement.
//
//   npx vitest run test/bench.test.ts --reporter=verbose
//
// Loads the shipped fp16 graph, generates with the same JS tokenizer, sampler and
// cache as the page, and compares the token ids with torch's greedy continuation
// (test/fixtures/generation.json). Then it does the same through the WebAssembly
// build of onnxruntime-web, which is what the CPU fallback in the browser runs,
// and prints tokens/s.
import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import type { OrtModule, Session } from '../src/backend';
import { generate } from '../src/generate';
import { EOT, encode } from '../src/tokenizer';
import fixture from './fixtures/generation.json';

const WEB_ROOT = resolve(import.meta.dirname, '..');
const MODEL = resolve(WEB_ROOT, 'model', 'model_fp16.onnx');

const greedy = {
    temperature: 0,
    topK: 50,
    topP: 1,
};

function options(maxNewTokens: number) {
    return {
        maxNewTokens,
        sampling: () => greedy,
        signal: new AbortController().signal,
        onToken: () => {},
        onStats: () => {},
    };
}

describe.skipIf(!existsSync(MODEL))('ONNX generation', () => {
    it('matches torch token for token (onnxruntime-node, CPU)', async () => {
        const ort = (await import('onnxruntime-node')) as unknown as OrtModule;
        const session = (await ort.InferenceSession.create(MODEL)) as unknown as Session;

        expect([EOT, ...encode(fixture.prompt)]).toEqual(fixture.prompt_ids);
        const result = await generate(ort, session, fixture.prompt, options(fixture.max_new_tokens));
        console.log(`onnxruntime-node fp16: ${result.generated} tokens, ${result.tokensPerSecond.toFixed(1)} tokens/s`);
        console.log(`  ${result.text}`);
        expect(result.tokenIds).toEqual(fixture.token_ids);
        expect(result.text).toBe(fixture.text);
    });

    it('matches torch through onnxruntime-web WASM and reports tokens/s', async () => {
        const ort = (await import('onnxruntime-web/wasm')) as unknown as OrtModule;
        // Node has no crossOriginIsolated, so the browser's CPU fallback single-threaded
        // path is what is measured here.
        ort.env.wasm.numThreads = 1;
        ort.env.wasm.wasmPaths = resolve(WEB_ROOT, 'node_modules', 'onnxruntime-web', 'dist') + '/';
        const session = (await ort.InferenceSession.create(readFileSync(MODEL), {
            executionProviders: ['wasm'],
            graphOptimizationLevel: 'all',
        })) as unknown as Session;

        const result = await generate(ort, session, fixture.prompt, options(fixture.max_new_tokens));
        console.log(`onnxruntime-web WASM fp16: ${result.generated} tokens, ${result.tokensPerSecond.toFixed(1)} tokens/s`);
        expect(result.tokenIds).toEqual(fixture.token_ids);

        const speed = await generate(ort, session, fixture.prompt, options(100));
        console.log(`onnxruntime-web WASM fp16, 100 tokens: ${speed.generated} tokens, ${speed.tokensPerSecond.toFixed(1)} tokens/s`);

        const fp32 = resolve(WEB_ROOT, 'model', 'model.onnx');
        if (existsSync(fp32)) {
            const s32 = (await ort.InferenceSession.create(readFileSync(fp32), {
                executionProviders: ['wasm'],
                graphOptimizationLevel: 'all',
            })) as unknown as Session;
            const r32 = await generate(ort, s32, fixture.prompt, options(100));
            console.log(`onnxruntime-web WASM fp32, 100 tokens: ${r32.generated} tokens, ${r32.tokensPerSecond.toFixed(1)} tokens/s`);
        }
        expect(speed.tokensPerSecond).toBeGreaterThan(0);
    });
});
