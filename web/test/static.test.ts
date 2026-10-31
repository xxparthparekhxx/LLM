// Cheap checks on the static pieces that cannot be exercised without a browser.
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';

const WEB_ROOT = resolve(import.meta.dirname, '..');
const read = (p: string) => readFileSync(resolve(WEB_ROOT, p), 'utf8');

describe('index.html and the UI code agree', () => {
    const html = read('index.html');
    const ids = new Set([...html.matchAll(/\bid="([^"]+)"/g)].map((m) => m[1]!));

    it('has every element id the UI looks up', () => {
        const used = [...read('src/main.ts').matchAll(/\$\s*<[^>]*>\s*\('([^']+)'\)/g)].map((m) => m[1]!);
        expect(used.length).toBeGreaterThan(15);
        expect([...new Set(used)].filter((id) => !ids.has(id))).toEqual([]);
    });

    it('has no duplicate ids', () => {
        const all = [...html.matchAll(/\bid="([^"]+)"/g)].map((m) => m[1]!);
        expect(all.length).toBe(new Set(all).size);
    });

    it('loads the entry module', () => {
        expect(html).toContain('src="/src/main.ts"');
    });
});

describe('deploy config', () => {
    it('wrangler.jsonc is assets-only and serves dist/ on llm.parthparekhdev.com', () => {
        const text = read('wrangler.jsonc')
            .split('\n')
            .map((l) => l.replace(/^\s*\/\/.*$/, ''))
            .join('\n');
        const cfg = JSON.parse(text) as Record<string, any>;
        expect(cfg.assets.directory).toBe('./dist');
        expect(cfg.main).toBeUndefined(); // no Worker script: static assets only
        expect(cfg.routes).toEqual([{ pattern: 'llm.parthparekhdev.com', custom_domain: true }]);
    });

    it('_headers sets cross-origin isolation and caching', () => {
        const h = read('public/_headers');
        expect(h).toMatch(/Cross-Origin-Opener-Policy: same-origin/);
        expect(h).toMatch(/Cross-Origin-Embedder-Policy: require-corp/);
        expect(h).toMatch(/\/assets\/\*\n\s+Cache-Control: public, max-age=31536000, immutable/);
        expect(h).toMatch(/\/ort\/\*\n\s+Cache-Control: public, max-age=31536000, immutable/);
    });

    it('production points at the Hugging Face file and dev defaults to the local one', () => {
        expect(read('.env.production')).toMatch(/VITE_MODEL_URL=https:\/\/huggingface\.co\/xxparthparekhxx\/tinystories-33m\/resolve\//);
        expect(read('src/config.ts')).toContain("'/model/model_fp16.onnx'");
    });

    it('git ignores the exported model and node_modules', () => {
        expect(read('.gitignore')).toMatch(/^node_modules\/$/m);
        expect(readFileSync(resolve(WEB_ROOT, '..', '.gitignore'), 'utf8')).toMatch(/^\*\.onnx$/m);
    });
});
