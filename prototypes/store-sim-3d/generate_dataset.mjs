// Headless dataset generator for the AisleLens 3D store simulator.
//
// Walks a virtual shopper to random poses in front of shelves, re-stocks the
// shelves and changes the lighting every few frames, and writes one RealSense-
// style RGB-D frame per pose:
//   out/rgb/00001.png     8-bit colour image
//   out/depth/00001.png   16-bit depth in millimetres (0 = no data), aligned to colour
//   out/mask/00001.png    R = class id (SKU), G*256+B = instance id
//   out/labels/00001.json boxes, SKU, barcode, price, camera intrinsics and pose
//   out/catalog.json      the class list
//
// Usage:
//   npm i playwright          (once; uses your installed Chrome/Chromium)
//   node generate_dataset.mjs --frames 500 --out ./dataset --camera d435
//
// Options:
//   --frames N        number of frames (default 200)
//   --out DIR         output folder (default ./dataset)
//   --camera NAME     d455 (default), d435 or glasses
//   --no-noise        write clean depth instead of RealSense-style noisy depth
//   --restock-every N new planogram every N frames (default 50)
//   --seed N          base random seed (default 1)
//   --three PATH      local three.module.min.js (v0.160.0) if the machine is offline
//   --gpu             use the real GPU instead of software rendering
//   --browser NAME    chrome, msedge or chromium (default: first one that works,
//                     in that order; chromium needs `npx playwright install chromium`)
import { chromium } from 'playwright';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const args = process.argv.slice(2);
const opt = (name, def) => { const i = args.indexOf('--' + name); return i >= 0 ? args[i + 1] : def; };
const flag = name => args.includes('--' + name);
const frames = +opt('frames', 200), out = path.resolve(opt('out', './dataset'));
const camera = opt('camera', 'd455'), restockEvery = +opt('restock-every', 50), seed = +opt('seed', 1);
const here = path.dirname(fileURLToPath(import.meta.url));
const html = path.join(here, 'index.html');

for (const d of ['rgb', 'depth', 'mask', 'labels']) fs.mkdirSync(path.join(out, d), { recursive: true });

const launchArgs = flag('gpu') ? ['--enable-gpu', '--ignore-gpu-blocklist'] : ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'];
// Prefer the Chrome or Edge already on the machine, so `npm i playwright` is enough.
const wanted = opt('browser');
let browser;
for (const channel of wanted ? [wanted] : ['chrome', 'msedge', 'chromium']) {
  try {
    browser = await chromium.launch({ args: launchArgs, ...(channel === 'chromium' ? {} : { channel }) });
    console.log(`Using ${channel}`);
    break;
  } catch (e) {
    if (wanted) throw e;
  }
}
if (!browser) throw new Error('No browser found. Install Chrome or Edge, or run: npx playwright install chromium');
const page = await browser.newPage({ viewport: { width: 640, height: 400 } });
page.on('pageerror', e => console.error('page error:', e.message));
const three = opt('three');
if (three) await page.route('https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.min.js', r => r.fulfill({ path: three, contentType: 'application/javascript' }));
await page.goto(pathToFileURL(html).href);
await page.waitForFunction(() => window.sim && window.sim.ready, null, { timeout: 180000 });
// pause the interactive render loop so only dataset frames are rendered
await page.evaluate(cam => { document.getElementById('start').hidden = true; window.sim.paused = true; window.sim.setPreset(cam); }, camera);
fs.writeFileSync(path.join(out, 'catalog.json'), JSON.stringify(await page.evaluate(() => window.sim.catalog()), null, 1));

const t0 = Date.now();
for (let i = 0; i < frames; i++) {
  const s = seed * 100003 + i;
  if (i % restockEvery === 0) await page.evaluate(k => window.sim.restock(k), s);
  const f = await page.evaluate(async ({ s, noise }) => {
    window.sim.randomLighting(s);
    const pose = window.sim.randomPose(s);
    const r = await window.sim.capture({ noise });
    r.labels.pose_sampled = pose;
    return r;
  }, { s, noise: !flag('no-noise') });
  const id = String(i + 1).padStart(5, '0');
  f.labels.frame = id;
  fs.writeFileSync(path.join(out, 'rgb', id + '.png'), Buffer.from(f.rgb, 'base64'));
  fs.writeFileSync(path.join(out, 'depth', id + '.png'), Buffer.from(f.depth, 'base64'));
  fs.writeFileSync(path.join(out, 'mask', id + '.png'), Buffer.from(f.mask, 'base64'));
  fs.writeFileSync(path.join(out, 'labels', id + '.json'), JSON.stringify(f.labels, null, 1));
  const rate = (i + 1) / ((Date.now() - t0) / 1000);
  console.log(`frame ${id}: ${f.labels.objects.length} products, ${rate.toFixed(2)} frames/s`);
}
await browser.close();
console.log(`Done. ${frames} frames in ${out}`);
