import { defineConfig } from 'vitest/config';

export default defineConfig({
  define: { __ORT_VERSION__: JSON.stringify('1.31.1'), __ORT_BASE__: JSON.stringify('/ort/1.31.1/') },
  test: {
    environment: 'node',
    include: ['test/**/*.test.ts'],
    testTimeout: 300_000,
  },
});
