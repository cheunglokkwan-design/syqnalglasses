// Headless dataset generator for the AisleLens 3D store simulator.
//
// Walks a virtual shopper to random poses in front of shelves, re-stocks the
// shelves and changes the lighting every few frames, sometimes puts a product
// in the shopper's hand, and writes one RealSense-style RGB-D frame per pose:
//   out/rgb/00001.png     8-bit colour image (.jpg with --jpg)
//   out/depth/00001.png   16-bit depth in millimetres (0 = no data), aligned to colour
//   out/mask/00001.png    R = class id (SKU) or 255 = price tag, G*256+B = instance id
//   out/labels/00001.json products and price tags with boxes, SKU, barcode, price,
//                         camera intrinsics and pose, planogram id
//   out/catalog.json      the class list
//   out/gallery/AL-001/front.jpg ...  clean reference photos (with --refs)
//
// Usage:
//   npm i playwright          (once; uses your installed Chrome or Edge)
//   node generate_dataset.mjs --frames 500 --out ./dataset --camera d435
//
// Options:
//   --frames N        number of frames (default 200)
//   --out DIR         output folder (default ./dataset)
//   --camera NAME     d455 (default), d435 or glasses
//   --no-noise        write clean depth instead of RealSense-style noisy depth
//   --restock-every N new planogram every N frames (default 50)
//   --seed N          base random seed (default 1); use different seeds for train and test sets
//   --held-ratio F    fraction of frames with a product in the shopper's hand (default 0.35)
//   --jpg             write RGB as JPEG (about 5x smaller than PNG)
//   --skip LIST       comma list of outputs to leave out: depth, mask
//   --refs            also render the reference gallery (8 views per SKU) into out/gallery
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
const heldRatio = +opt('held-ratio', 0.35);
const skip = (opt('skip', '') || '').split(',').filter(Boolean);
const ext = flag('jpg') ? 'jpg' : 'png';
const here = path.dirname(fileURLToPath(import.meta.url));
const html = path.join(here, 'index.html');

for (const d of ['rgb', 'labels', ...['depth', 'mask'].filter(d => !skip.includes(d))]) fs.mkdirSync(path.join(out, d), { recursive: true });

const launchArgs = flag('gpu') ? ['--enable-gpu', '--ignore-gpu-blocklist', ...(process.platform === 'win32' ? ['--use-angle=d3d11'] : [])] : ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'];
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
console.log('Renderer:', await page.evaluate(() => { const gl = document.querySelector('canvas').getContext('webgl2'); const d = gl && gl.getExtension('WEBGL_debug_renderer_info'); return d ? gl.getParameter(d.UNMASKED_RENDERER_WEBGL) : 'unknown'; }));
const catalog = await page.evaluate(() => window.sim.catalog());
fs.writeFileSync(path.join(out, 'catalog.json'), JSON.stringify(catalog, null, 1));

if (flag('refs')) {
  const views = await page.evaluate(() => window.sim.refViews());
  for (const p of catalog) {
    const dir = path.join(out, 'gallery', p.sku);
    fs.mkdirSync(dir, { recursive: true });
    for (const v of views) {
      const b = await page.evaluate(({ sku, v }) => window.sim.renderReference(sku, v, 320), { sku: p.sku, v });
      fs.writeFileSync(path.join(dir, v + '.jpg'), Buffer.from(b, 'base64'));
    }
  }
  console.log(`Gallery: ${catalog.length} SKUs x ${views.length} views in ${path.join(out, 'gallery')}`);
}

const t0 = Date.now();
for (let i = 0; i < frames; i++) {
  const s = seed * 100003 + i;
  const planogram = seed * 100003 + Math.floor(i / restockEvery) * restockEvery;
  if (i % restockEvery === 0) await page.evaluate(k => window.sim.restock(k), planogram);
  const f = await page.evaluate(async ({ s, noise, hold, format, skip }) => {
    window.sim.dropHeld();
    window.sim.randomLighting(s);
    // step back from the shelf when holding something, so the item is not inside the shelf
    const pose = window.sim.randomPose(s, hold ? { minDist: 0.95 } : {});
    const h = hold ? window.sim.holdItem(s + 7) : null;
    const r = await window.sim.capture({ noise, format, skip });
    r.labels.pose_sampled = pose;
    r.labels.held_sampled = h;
    return r;
  }, { s, noise: !flag('no-noise'), hold: ((s * 2654435761) >>> 0) / 4294967296 < heldRatio, format: ext, skip });
  const id = String(i + 1).padStart(5, '0');
  f.labels.frame = id;
  f.labels.planogram = planogram;
  f.labels.seed = seed;
  f.labels.rgb_file = `rgb/${id}.${ext}`;
  fs.writeFileSync(path.join(out, 'rgb', `${id}.${ext}`), Buffer.from(f.rgb, 'base64'));
  if (f.depth) fs.writeFileSync(path.join(out, 'depth', id + '.png'), Buffer.from(f.depth, 'base64'));
  if (f.mask) fs.writeFileSync(path.join(out, 'mask', id + '.png'), Buffer.from(f.mask, 'base64'));
  fs.writeFileSync(path.join(out, 'labels', id + '.json'), JSON.stringify(f.labels));
  const rate = (i + 1) / ((Date.now() - t0) / 1000);
  const nTags = f.labels.objects.filter(o => o.type === 'price_tag').length;
  if (i % 10 === 0 || i === frames - 1) console.log(`frame ${id}: ${f.labels.objects.length - nTags} products, ${nTags} tags${f.labels.held_sampled ? ', held ' + f.labels.held_sampled.sku : ''}, ${rate.toFixed(2)} frames/s`);
}
await browser.close();
console.log(`Done. ${frames} frames in ${out}`);
