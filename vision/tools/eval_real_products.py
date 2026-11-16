"""Identification on REAL photos: the Grocery Store Dataset (Klasson et al., MIT licence).

81 products (fruit, vegetables, and packaged juice / milk / yoghurt), photographed with a
phone in real Swedish grocery stores, plus one catalogue ("iconic") photo per product.

Three settings, each scored on the 2,485 in-store test photos:
  1. one catalogue photo per product, DINOv2 off the shelf  (a new product added from one photo)
  2. the 2,640 in-store training photos as the gallery, off the shelf
  3. as 2, with an embedding head trained on the training photos (written to --out/head.pt)

Usage: python vision/tools/eval_real_products.py --data data/real/raw/grocerystore/dataset --out runs/embedder_real
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aislelens_vision.embedder import Embedder  # noqa: E402


def read_list(root: Path, name: str):
    out = []
    for line in (root / name).read_text(encoding="utf-8").splitlines():
        if line.strip():
            p, fine, coarse = [x.strip() for x in line.split(",")]
            out.append((root / p, int(fine), int(coarse)))
    return out


def feats(emb: Embedder, paths, batch=32):
    out = []
    for i in range(0, len(paths), batch):
        out.append(emb.features([Image.open(p).convert("RGB") for p in paths[i:i + batch]], batch=batch))
    return np.concatenate(out)


def topk(q, g, gy, qy, k=5):
    """Per-class max similarity, top-1 and top-k accuracy."""
    sims = q @ g.T
    classes = np.unique(gy)
    per = np.stack([sims[:, gy == c].max(1) for c in classes], 1)
    order = classes[np.argsort(-per, 1)[:, :k]]
    return float(np.mean(order[:, 0] == qy)), float(np.mean([y in o for y, o in zip(qy, order)]))


def norm(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/real/raw/grocerystore/dataset")
    ap.add_argument("--out", default="runs/embedder_real")
    ap.add_argument("--model", default="facebook/dinov2-small")
    ap.add_argument("--epochs", type=int, default=60)
    a = ap.parse_args()
    root, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    classes = list(csv.DictReader((root / "classes.csv").open(encoding="utf-8")))
    name = {int(c["Class ID (int)"]): c["Class Name (str)"] for c in classes}
    coarse_name = {int(c["Coarse Class ID (int)"]): c["Coarse Class Name (str)"] for c in classes}
    packaged = {int(c["Class ID (int)"]) for c in classes if "Packages" in c["Iconic Image Path (str)"]}
    iconic = [(root / c["Iconic Image Path (str)"].lstrip("/"), int(c["Class ID (int)"])) for c in classes]
    train, test = read_list(root, "train.txt"), read_list(root, "test.txt")

    emb = Embedder(a.model)
    t = time.time()
    Fi = feats(emb, [p for p, _ in iconic]); yi = np.array([c for _, c in iconic])
    Ftr = feats(emb, [p for p, _, _ in train]); ytr = np.array([f for _, f, _ in train])
    Fte = feats(emb, [p for p, _, _ in test]); yte = np.array([f for _, f, _ in test])
    print(f"embedded {len(iconic) + len(train) + len(test)} real photos in {time.time() - t:.0f}s on {emb.device}")
    pk = np.isin(yte, list(packaged))
    R = {"dataset": "Grocery Store Dataset (Klasson et al., MIT)", "model": a.model, "test_photos": len(test),
         "products": len(classes), "packaged_products": len(packaged)}

    def score(tag, q, g, gy):
        t1, t5 = topk(q, g, gy, yte)
        p1, _ = topk(q[pk], g, gy, yte[pk])
        f1, _ = topk(q[~pk], g, gy, yte[~pk])
        R[tag] = {"top1": t1, "top5": t5, "top1_packaged": p1, "top1_produce": f1}
        print(f"{tag:34s} top-1 {t1:.1%}  top-5 {t5:.1%}  packaged {p1:.1%}  produce {f1:.1%}")

    score("1_catalogue_photo_off_the_shelf", norm(Fte), norm(Fi), yi)
    score("2_store_photos_off_the_shelf", norm(Fte), norm(np.concatenate([Ftr, Fi])), np.concatenate([ytr, yi]))

    # 3. train a linear head (same recipe as tools/train_embedder.py) on the training photos
    torch.manual_seed(0)
    X = torch.nn.functional.normalize(torch.from_numpy(np.concatenate([Ftr, Fi]).astype(np.float32)), dim=1)
    y = torch.from_numpy(np.concatenate([ytr, yi]))
    head = torch.nn.Linear(X.shape[1], 256, bias=False)
    prox = torch.nn.Parameter(torch.randn(len(classes), 256) * 0.1)
    opt = torch.optim.AdamW(list(head.parameters()) + [prox], lr=2e-3, weight_decay=1e-4)
    for ep in range(a.epochs):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), 256):
            b = perm[i:i + 256]
            z = torch.nn.functional.normalize(head(X[b]), dim=1)
            loss = torch.nn.functional.cross_entropy(z @ torch.nn.functional.normalize(prox, dim=1).T / 0.05, y[b])
            opt.zero_grad(); loss.backward(); opt.step()
    torch.save(head.state_dict(), out / "head.pt")
    with torch.no_grad():
        P = lambda F: torch.nn.functional.normalize(head(torch.nn.functional.normalize(torch.from_numpy(F.astype(np.float32)), dim=1)), dim=1).numpy()
        score("3_store_photos_trained_head", P(Fte), P(np.concatenate([Ftr, Fi])), np.concatenate([ytr, yi]))
        score("3b_catalogue_photo_trained_head", P(Fte), P(Fi), yi)
    (out / "grocery_store_report.json").write_text(json.dumps(R, indent=1), encoding="utf-8")
    print(f"saved {out / 'grocery_store_report.json'}")


if __name__ == "__main__":
    main()
