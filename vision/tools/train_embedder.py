"""Build the reference gallery and fine-tune the embedding head.

1. Embeds every gallery render (data/sim/test/gallery/<SKU>/*.jpg) with frozen DINOv2.
2. Cuts labelled product crops out of the training frames and embeds them.
3. Trains one linear layer (CLS+mean features -> 256-d) with a normalised-softmax
   (proxy) loss so crops land next to their own SKU's gallery photos.
   Every 6th SKU is held out of training entirely, to check that a product the head
   never saw still works from its reference photos alone ("new SKU onboarding").

Writes runs/embedder/{gallery.npz, head.pt, heldout_skus.json}.

Usage: python vision/tools/train_embedder.py --sim data/sim --out runs/embedder
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aislelens_vision.embedder import Embedder, Gallery  # noqa: E402


def crops_from_split(split: Path, per_sku: int, min_px: int, seed: int = 0):
    rng = random.Random(seed)
    by = defaultdict(list)
    for lf in sorted((split / "labels").glob("*.json")):
        L = json.loads(lf.read_text(encoding="utf-8"))
        for o in L["objects"]:
            if o["type"] != "product" or o["visible_pixels"] < min_px:
                continue
            x, y, w, h = o["bbox_xywh"]
            if o["visible_pixels"] < 0.35 * w * h:  # mostly hidden behind something
                continue
            by[o["sku"]].append((split / L["rgb_file"], (x, y, x + w, y + h), bool(o.get("held"))))
    out = []
    for sku, items in by.items():
        held = [t for t in items if t[2]]
        rest = [t for t in items if not t[2]]
        rng.shuffle(rest)
        out += [(sku, *t[:2]) for t in (held + rest)[:per_sku]]
    return out


def load_crops(items):
    cache, crops = {}, []
    for _, path, (x0, y0, x1, y1) in items:
        if path not in cache:
            cache.clear()
            cache[path] = Image.open(path).convert("RGB")
        im = cache[path]
        px, py = (x1 - x0) * 0.04, (y1 - y0) * 0.04
        crops.append(im.crop((max(0, x0 - px), max(0, y0 - py), min(im.width, x1 + px), min(im.height, y1 + py))))
    return crops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", default="data/sim")
    ap.add_argument("--gallery", default=None, help="default: <sim>/test/gallery")
    ap.add_argument("--out", default="runs/embedder")
    ap.add_argument("--per-sku", type=int, default=40)
    ap.add_argument("--min-px", type=int, default=2500)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=60)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    emb = Embedder()
    t = time.time()
    gal = Gallery.build(emb, a.gallery or Path(a.sim) / "test" / "gallery")
    gal.save(out / "gallery.npz")
    print(f"gallery: {len(gal.skus)} reference images, {len(set(gal.skus))} SKUs, {time.time() - t:.0f}s")

    skus = sorted(set(gal.skus.tolist()))
    heldout = skus[5::6]
    (out / "heldout_skus.json").write_text(json.dumps(heldout), encoding="utf-8")
    train_skus = [s for s in skus if s not in heldout]
    items = [it for it in crops_from_split(Path(a.sim) / "train", a.per_sku, a.min_px) if it[0] in train_skus]
    t = time.time()
    feats = emb.features(load_crops(items))
    print(f"train crops: {len(items)} from {len(set(i[0] for i in items))} SKUs, {time.time() - t:.0f}s")

    # training set = shelf crops + gallery renders of the training SKUs
    gmask = np.isin(gal.skus, train_skus)
    X = torch.from_numpy(np.concatenate([feats, gal.feats[gmask]]).astype(np.float32))
    names = [i[0] for i in items] + gal.skus[gmask].tolist()
    cls = {s: k for k, s in enumerate(train_skus)}
    y = torch.tensor([cls[s] for s in names])
    X = torch.nn.functional.normalize(X, dim=1)  # Embedder.project does the same before the head

    head = torch.nn.Linear(X.shape[1], a.dim, bias=False)
    proxies = torch.nn.Parameter(torch.randn(len(train_skus), a.dim) * 0.1)
    opt = torch.optim.AdamW(list(head.parameters()) + [proxies], lr=2e-3, weight_decay=1e-4)
    for ep in range(a.epochs):
        perm = torch.randperm(len(X))
        tot = 0.0
        for i in range(0, len(X), 256):
            b = perm[i:i + 256]
            z = torch.nn.functional.normalize(head(X[b]), dim=1)
            p = torch.nn.functional.normalize(proxies, dim=1)
            loss = torch.nn.functional.cross_entropy(z @ p.T / 0.05, y[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        if ep % 10 == 0 or ep == a.epochs - 1:
            print(f"epoch {ep:3d} loss {tot / len(X):.3f}")
    torch.save(head.state_dict(), out / "head.pt")
    print(f"saved {out / 'head.pt'} (held-out SKUs: {', '.join(heldout)})")


if __name__ == "__main__":
    main()
