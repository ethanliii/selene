#!/usr/bin/env node
// SELENE screenshot driver: headless Chrome over the DevTools protocol, no npm dependencies
// (Node >= 22 for the global WebSocket).
//
// Usage
//   node scripts/capture-screenshots.mjs [spec.json] [--only name,name] [--base URL] [--out DIR] [--lenient]
//
//   spec.json   defaults to scripts/screenshots.json
//   --only      run only the "scenes" (a step with a `url` plus the steps after it) that contain one of these shot names
//   --base      override spec.baseUrl (default http://127.0.0.1:8000, i.e. `make demo`)
//   --out       override spec.outDir (raw PNG frames; default .tools/screenshots)
//   --lenient   exit 0 even when a waitFor timed out or the page logged errors
//
//   CHROME_PATH overrides the browser binary (default: Google Chrome on macOS, else google-chrome / chromium on PATH).
//
// `make screenshots` builds the UI, starts the API on :8000, runs this script, then
// scripts/build-readme-images.py turns the PNGs into docs/images/*.jpg and docs/images/demo.gif.
//
// Spec
//   { "baseUrl": "...", "outDir": "...", "width": 1600, "height": 900, "scale": 2,
//     "steps": [ { "name": "hero", "url": "/ops?demo=1&t=243000", "waitFor": "<js expr>", "waitForTimeout": 60000,
//                  "js": "<js run after waitFor>", "wait": 3000, "then": [ {"js": "...", "wait": 500, "waitFor": "..."} ],
//                  "probe": "<js expr logged before the shot>", "noShot": false,
//                  "width": 1600, "height": 900, "scale": 2 } ] }
//   Per step: navigate (url, relative to baseUrl) -> poll waitFor -> eval js -> sleep wait -> run `then` actions in
//   order -> log probe -> screenshot <outDir>/<name>.png (unless noShot). Page state carries over between steps, so
//   a step without a url continues where the previous one left off.
//   A `then` action may also carry real pointer input for the 3D view (points are [x, y] CSS px or a JS expression
//   returning [x, y]):  {"drag": {"button": "right", "from": [x, y] | "<expr>", "to": [x, y], "steps": 16}}
//                       {"wheel": {"at": [x, y] | "<expr>", "deltaY": -100, "times": 5}}
//   Console errors, uncaught exceptions and waitFor timeouts are collected; any of them makes the exit code 2.
import { spawn, spawnSync } from 'node:child_process';
import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { setTimeout as sleep } from 'node:timers/promises';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, '..');

// ---------- arguments ----------
const argv = process.argv.slice(2);
const opt = { spec: path.join(HERE, 'screenshots.json'), only: null, base: null, out: null, lenient: false };
for (let i = 0; i < argv.length; i++) {
  const a = argv[i];
  if (a === '--only') opt.only = new Set(argv[++i].split(',').map((s) => s.trim()).filter(Boolean));
  else if (a === '--base') opt.base = argv[++i];
  else if (a === '--out') opt.out = argv[++i];
  else if (a === '--lenient') opt.lenient = true;
  else if (a === '-h' || a === '--help') {
    console.log(readFileSync(fileURLToPath(import.meta.url), 'utf8').split('\n').filter((l) => l.startsWith('//')).map((l) => l.slice(3)).join('\n'));
    process.exit(0);
  } else opt.spec = path.resolve(a);
}

const spec = JSON.parse(readFileSync(opt.spec, 'utf8'));
const baseUrl = (opt.base ?? process.env.SELENE_URL ?? spec.baseUrl ?? 'http://127.0.0.1:8000').replace(/\/$/, '');
const outDir = path.resolve(ROOT, opt.out ?? spec.outDir ?? '.tools/screenshots');
mkdirSync(outDir, { recursive: true });
const W = spec.width ?? 1600;
const H = spec.height ?? 900;
const SCALE = spec.scale ?? 1;

function findChrome() {
  if (process.env.CHROME_PATH) return process.env.CHROME_PATH;
  const mac = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
  if (existsSync(mac)) return mac;
  for (const bin of ['google-chrome', 'google-chrome-stable', 'chromium', 'chromium-browser']) {
    const r = spawnSync('which', [bin], { encoding: 'utf8' });
    if (r.status === 0 && r.stdout.trim()) return r.stdout.trim();
  }
  throw new Error('Chrome not found: set CHROME_PATH');
}

// ---------- step selection (--only works on whole scenes) ----------
let steps = spec.steps;
if (opt.only) {
  const scenes = [];
  for (const s of steps) {
    if (s.url || scenes.length === 0) scenes.push([]);
    scenes[scenes.length - 1].push(s);
  }
  steps = scenes.filter((sc) => sc.some((s) => opt.only.has(s.name))).flat();
  if (steps.length === 0) throw new Error(`--only matched no steps: ${[...opt.only].join(', ')}`);
}

// ---------- browser ----------
const profile = path.join(outDir, `.chrome-profile-${process.pid}`);
const chrome = spawn(findChrome(), [
  '--headless=new', '--remote-debugging-port=0', '--disable-gpu', '--use-angle=swiftshader', '--enable-unsafe-swiftshader',
  `--window-size=${W},${H}`, `--user-data-dir=${profile}`, '--no-first-run', '--no-default-browser-check', '--hide-scrollbars',
  '--disable-background-timer-throttling', '--disable-renderer-backgrounding', '--disable-backgrounding-occluded-windows',
  '--mute-audio', 'about:blank',
], { stdio: ['ignore', 'ignore', 'pipe'] });

function cleanup() {
  try { chrome.kill('SIGKILL'); } catch { /* already gone */ }
  try { rmSync(profile, { recursive: true, force: true }); } catch { /* best effort */ }
}
process.on('SIGINT', () => { cleanup(); process.exit(130); });

const wsUrl = await new Promise((resolve, reject) => {
  let buf = '';
  chrome.stderr.on('data', (d) => {
    buf += d.toString();
    const m = buf.match(/DevTools listening on (ws:\/\/\S+)/);
    if (m) resolve(m[1]);
  });
  chrome.on('exit', (c) => reject(new Error(`chrome exited (${c})\n${buf}`)));
  setTimeout(() => reject(new Error(`no DevTools URL from chrome\n${buf}`)), 20000);
});

const ws = new WebSocket(wsUrl);
await new Promise((r, j) => { ws.onopen = r; ws.onerror = j; });

let seq = 0;
let sessionId = null;
const pending = new Map();
const loadWaiters = [];
const logLines = [];
const problems = [];
const log = (s) => logLines.push(s);
// Benign noise: SwiftShader perf notes and favicon misses are not page errors.
const benign = (s) => /GPU stall|GL Driver Message|favicon\.ico|WebGL-0x/.test(s);

ws.onmessage = (ev) => {
  const msg = JSON.parse(ev.data);
  if (msg.id && pending.has(msg.id)) {
    const { res, rej } = pending.get(msg.id);
    pending.delete(msg.id);
    if (msg.error) rej(new Error(JSON.stringify(msg.error)));
    else res(msg.result);
    return;
  }
  const p = msg.params;
  if (msg.method === 'Runtime.consoleAPICalled') {
    const text = p.args.map((a) => a.value ?? a.description ?? JSON.stringify(a)).join(' ');
    log(`[console.${p.type}] ${text}`);
    if (p.type === 'error' && !benign(text)) problems.push(`console.error: ${text.slice(0, 300)}`);
  } else if (msg.method === 'Runtime.exceptionThrown') {
    const d = p.exceptionDetails;
    const text = `${d.text} ${d.exception?.description ?? ''} @${d.url ?? ''}:${d.lineNumber}`;
    log(`[exception] ${text}`);
    problems.push(`exception: ${text.slice(0, 300)}`);
  } else if (msg.method === 'Log.entryAdded') {
    const e = p.entry;
    const text = `${e.text} ${e.url ?? ''}`;
    log(`[log.${e.source}.${e.level}] ${text}`);
    if (e.level === 'error' && !benign(text)) problems.push(`log.${e.source}: ${text.slice(0, 300)}`);
  } else if (msg.method === 'Page.loadEventFired') {
    loadWaiters.splice(0).forEach((r) => r());
  }
};

function send(method, params = {}, withSession = true) {
  const id = ++seq;
  const payload = { id, method, params };
  if (withSession && sessionId) payload.sessionId = sessionId;
  ws.send(JSON.stringify(payload));
  return new Promise((res, rej) => pending.set(id, { res, rej }));
}

const { targetId } = await send('Target.createTarget', { url: 'about:blank' }, false);
({ sessionId } = await send('Target.attachToTarget', { targetId, flatten: true }, false));
await send('Page.enable');
await send('Runtime.enable');
await send('Log.enable');

let metrics = null;
async function setViewport(w, h, scale) {
  const key = `${w}x${h}@${scale}`;
  if (metrics === key) return;
  metrics = key;
  await send('Emulation.setDeviceMetricsOverride', { width: w, height: h, deviceScaleFactor: scale, mobile: false });
}

async function evaluate(expression, label) {
  const r = await send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true });
  if (r.exceptionDetails) {
    const text = `${r.exceptionDetails.text} ${r.exceptionDetails.exception?.description ?? ''}`;
    log(`[eval-error ${label}] ${text}`);
    problems.push(`eval error in ${label}: ${text.slice(0, 300)}`);
  }
  return r.result?.value;
}

async function waitFor(expr, timeout, label) {
  const t0 = Date.now();
  while (Date.now() - t0 < timeout) {
    if (await evaluate(`(()=>{try{return !!(${expr})}catch(e){return false}})()`, label)) {
      log(`waitFor ${label}: ${Date.now() - t0} ms`);
      return true;
    }
    await sleep(250);
  }
  log(`waitFor ${label}: TIMEOUT after ${timeout} ms: ${expr}`);
  problems.push(`waitFor timed out in ${label}: ${expr.slice(0, 200)}`);
  return false;
}

// A point is [x, y] in CSS px or a JS expression that returns [x, y] (e.g. the screen position of a label).
async function point(p, label) {
  const v = Array.isArray(p) ? p : await evaluate(p, label);
  if (!Array.isArray(v) || v.length < 2 || !v.every(Number.isFinite)) {
    problems.push(`bad point in ${label}: ${JSON.stringify(p).slice(0, 200)} -> ${JSON.stringify(v)}`);
    return null;
  }
  return v;
}

const mouse = (type, x, y, extra = {}) => send('Input.dispatchMouseEvent', { type, x, y, ...extra });

// Real pointer input (the 3D view's orbit controls ignore synthetic DOM events): right-drag pans, left-drag orbits.
async function drag(d, label) {
  const from = await point(d.from, label), to = await point(d.to, label);
  if (!from || !to) return;
  const button = d.button ?? 'left';
  const buttons = button === 'right' ? 2 : button === 'middle' ? 4 : 1;
  const n = d.steps ?? 16;
  await mouse('mouseMoved', from[0], from[1]);
  await sleep(100);
  await mouse('mousePressed', from[0], from[1], { button, buttons, clickCount: 1 });
  for (let i = 1; i <= n; i++) {
    const x = from[0] + ((to[0] - from[0]) * i) / n, y = from[1] + ((to[1] - from[1]) * i) / n;
    await mouse('mouseMoved', x, y, { button, buttons });
    await sleep(16);
  }
  await mouse('mouseReleased', to[0], to[1], { button, buttons: 0, clickCount: 1 });
}

// Each wheel notch dollies the orbit camera by ~5 % toward its target (the magnitude of deltaY is ignored).
async function wheel(wh, label) {
  const at = await point(wh.at, label);
  if (!at) return;
  await mouse('mouseMoved', at[0], at[1]);
  await sleep(100);
  for (let i = 0; i < (wh.times ?? 1); i++) {
    await mouse('mouseWheel', at[0], at[1], { deltaX: 0, deltaY: wh.deltaY ?? -100 });
    await sleep(wh.interval ?? 60);
  }
}

async function act(a, label) {
  if (a.waitFor) await waitFor(a.waitFor, a.waitForTimeout ?? 30000, label);
  if (a.js) await evaluate(a.js, label);
  if (a.drag) await drag(a.drag, label);
  if (a.wheel) await wheel(a.wheel, label);
  await sleep(a.wait ?? 0);
}

// ---------- run ----------
const t0All = Date.now();
for (const step of steps) {
  const w = step.width ?? W, h = step.height ?? H, scale = step.scale ?? SCALE;
  await setViewport(w, h, scale);
  log(`--- step ${step.name} (${w}x${h}@${scale}x) ${step.url ?? ''}`);
  process.stdout.write(`  ${step.name} ... `);
  let preId = null;
  if (step.preJs) ({ identifier: preId } = await send('Page.addScriptToEvaluateOnNewDocument', { source: step.preJs }));
  if (step.url) {
    const url = /^https?:/.test(step.url) ? step.url : baseUrl + step.url;
    const loaded = new Promise((r) => loadWaiters.push(r));
    await send('Page.navigate', { url });
    await Promise.race([loaded, sleep(15000)]);
  }
  await act({ waitFor: step.waitFor, waitForTimeout: step.waitForTimeout ?? 60000, js: step.js, wait: step.wait ?? 2000 }, step.name);
  for (const [i, a] of (step.then ?? []).entries()) await act(a, `${step.name}.then[${i}]`);
  if (step.probe) {
    const v = await evaluate(step.probe, `${step.name}.probe`);
    log(`[probe ${step.name}] ${typeof v === 'string' ? v : JSON.stringify(v)}`);
  }
  if (preId) await send('Page.removeScriptToEvaluateOnNewDocument', { identifier: preId });
  if (step.noShot) { console.log('ok'); continue; }
  const shot = await send('Page.captureScreenshot', { format: 'png', captureBeyondViewport: false });
  const file = path.join(outDir, `${step.name}.png`);
  writeFileSync(file, Buffer.from(shot.data, 'base64'));
  log(`saved ${file}`);
  console.log(path.relative(ROOT, file));
}

const logFile = path.join(outDir, 'capture.console.log');
writeFileSync(logFile, logLines.join('\n') + '\n');
for (const l of logLines.filter((l) => l.startsWith('[probe'))) console.log(l.slice(0, 400));
console.log(`done in ${((Date.now() - t0All) / 1000).toFixed(0)} s; full log: ${path.relative(ROOT, logFile)}`);
try { await send('Browser.close', {}, false); } catch { /* ignore */ }
cleanup();
if (problems.length) {
  console.error(`\n${problems.length} problem(s):\n  ` + problems.join('\n  '));
  process.exit(opt.lenient ? 0 : 2);
}
process.exit(0);
