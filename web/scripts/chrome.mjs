// Minimal headless Chrome driver over the DevTools protocol (Node's built-in WebSocket, no dependencies).
// Used by the browser check; not part of the site.

import { spawn } from 'node:child_process';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

export const CHROME = process.env.CHROME ?? '/usr/bin/google-chrome-stable';

export async function launchChrome(args = []) {
  const port = 9600 + Math.floor(Math.random() * 300);
  const profile = mkdtempSync(join(tmpdir(), 'tinystories-chrome-'));
  const proc = spawn(
    CHROME,
    [
      '--headless=new',
      '--no-first-run',
      '--no-default-browser-check',
      '--hide-scrollbars',
      `--remote-debugging-port=${port}`,
      `--user-data-dir=${profile}`,
      ...args,
      'about:blank',
    ],
    { stdio: 'ignore' },
  );

  let target;
  for (let i = 0; i < 100 && !target; i++) {
    try {
      const list = await (await fetch(`http://127.0.0.1:${port}/json`)).json();
      target = list.find((t) => t.type === 'page');
    } catch {
      await sleep(200);
    }
  }
  if (!target) {
    proc.kill();
    throw new Error(`Chrome did not start (${CHROME})`);
  }

  const ws = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((resolve, reject) => {
    ws.onopen = resolve;
    ws.onerror = () => reject(new Error('could not connect to Chrome DevTools'));
  });
  let nextId = 0;
  const pending = new Map();
  const listeners = [];
  ws.onmessage = ({ data: raw }) => {
    const message = JSON.parse(raw);
    if (message.id !== undefined) {
      pending.get(message.id)?.(message);
      pending.delete(message.id);
    } else {
      for (const l of listeners) l(message);
    }
  };
  const call = (method, params = {}) =>
    new Promise((resolve, reject) => {
      const id = ++nextId;
      pending.set(id, (m) => (m.error ? reject(new Error(`${method}: ${m.error.message}`)) : resolve(m.result)));
      ws.send(JSON.stringify({ id, method, params }));
    });

  const consoleLines = [];
  listeners.push((m) => {
    if (m.method === 'Runtime.consoleAPICalled') {
      consoleLines.push(`${m.params.type}: ${m.params.args.map((a) => a.value ?? a.description ?? '').join(' ')}`);
    } else if (m.method === 'Runtime.exceptionThrown') {
      consoleLines.push(`exception: ${m.params.exceptionDetails.exception?.description ?? m.params.exceptionDetails.text}`);
    }
  });
  await call('Runtime.enable');
  await call('Page.enable');

  return {
    call,
    consoleLines,
    async version() {
      return (await fetch(`http://127.0.0.1:${port}/json/version`).then((r) => r.json())).Browser;
    },
    async goto(url) {
      await call('Page.navigate', { url });
      for (let i = 0; i < 100; i++) {
        const { result } = await call('Runtime.evaluate', { expression: 'document.readyState', returnByValue: true });
        if (result.value === 'complete') return;
        await sleep(100);
      }
      throw new Error(`${url} did not finish loading`);
    },
    async run(fn, arg) {
      const expression = `(${fn.toString()})(${JSON.stringify(arg ?? null)})`;
      const { result, exceptionDetails } = await call('Runtime.evaluate', {
        expression,
        awaitPromise: true,
        returnByValue: true,
      });
      if (exceptionDetails) throw new Error(`page function failed: ${exceptionDetails.exception?.description ?? exceptionDetails.text}`);
      return result.value;
    },
    async screenshot(path) {
      const { data } = await call('Page.captureScreenshot', { format: 'png' });
      const { writeFileSync } = await import('node:fs');
      writeFileSync(path, Buffer.from(data, 'base64'));
    },
    async close() {
      try {
        ws.close();
      } catch {
        /* already closed */
      }
      proc.kill();
      await sleep(300);
      rmSync(profile, { recursive: true, force: true });
    },
  };
}
