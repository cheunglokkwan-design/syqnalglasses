"""Score the vision pipeline on simulator frames against the targets in docs/vision-model-report.md.

Stages are scored on their own (with perfect boxes from the labels) and end to end
(with the trained detector), so a weak stage is easy to spot:

  detector    AP50 for product / price_tag / held_item
  identify    shelf top-1 / top-5 (and for SKUs the embedder head never saw),
              held item: barcode read rate, top-1, wrong-answer rate, "turn the pack" rate
  tags        exact price read on tags at least --min-tag-h pixels tall
  end to end  held item right SKU, item under the centre of view right SKU, latency

The visual-match acceptance threshold is tuned on the validation split so that
confident wrong answers stay at or under --max-wrong.

Usage:
  python vision/tools/evaluate.py --test data/sim/test --valid data/sim/valid \
      --embedder runs/embedder --detector runs/detector/checkpoint_best_total.pth --out runs/eval
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aislelens_vision.catalog import Catalog  # noqa: E402
from aislelens_vision.embedder import Embedder, Gallery  # noqa: E402
from aislelens_vision.perception import TagReader, read_ean13  # noqa: E402
from aislelens_vision.pipeline import Accept, Pipeline  # noqa: E402

TARGETS = {"held_top1": 0.90, "held_top1_barcode": 0.98, "shelf_top1": 0.80, "tag_read": 0.95,
           "wrong_rate": 0.02, "latency_ms": 300, "detector_map50": 0.85}


def frames(split: Path):
    for lf in sorted((split / "labels").glob("*.json")):
        L = json.loads(lf.read_text(encoding="utf-8"))
        yield L, split / L["rgb_file"], (split / "depth" / f"{lf.stem}.png")


def crop(im, box, pad=0.04):
    x0, y0, x1, y1 = box
    px, py = (x1 - x0) * pad, (y1 - y0) * pad
    return im.crop((max(0, x0 - px), max(0, y0 - py), min(im.width, x1 + px), min(im.height, y1 + py)))


def xyxy(o):
    x, y, w, h = o["bbox_xywh"]
    return (x, y, x + w, y + h)


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def ap50(preds, gts):
    """VOC-style all-point AP. preds: [(img, score, box)], gts: {img: [box]}."""
    npos = sum(len(v) for v in gts.values())
    if npos == 0:
        return None
    used = {k: [False] * len(v) for k, v in gts.items()}
    tp, fp = [], []
    for img, _, box in sorted(preds, key=lambda p: -p[1]):
        best, bj = 0.0, -1
        for j, g in enumerate(gts.get(img, [])):
            v = iou(box, g)
            if v > best:
                best, bj = v, j
        if best >= 0.5 and not used[img][bj]:
            used[img][bj] = True
            tp.append(1); fp.append(0)
        else:
            tp.append(0); fp.append(1)
    tp, fp = np.cumsum(tp), np.cumsum(fp)
    rec = tp / npos
    prec = tp / np.maximum(tp + fp, 1e-9)
    mrec = np.concatenate([[0], rec, [1]])
    mpre = np.concatenate([[0], prec, [0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def gt_boxes(L, min_product_px=300, min_tag_px=60):
    g = defaultdict(list)
    for o in L["objects"]:
        if o["type"] == "price_tag":
            if o["visible_pixels"] >= min_tag_px:
                g["price_tag"].append(xyxy(o))
        elif o.get("held"):
            g["held_item"].append(xyxy(o))
        elif o["visible_pixels"] >= min_product_px:
            g["product"].append(xyxy(o))
    return g


def identification_set(split: Path, min_px: int, max_shelf_per_frame: int):
    """Crops with known SKU: held items and well-visible shelf packs."""
    held, shelf = [], []
    for L, rgb, _ in frames(split):
        im = Image.open(rgb).convert("RGB")
        prods = [o for o in L["objects"] if o["type"] == "product"]
        for o in prods:
            if o.get("held"):
                held.append((o["sku"], crop(im, xyxy(o)), o.get("held_view")))
        ok = [o for o in prods if not o.get("held") and o["visible_pixels"] >= min_px
              and o["visible_pixels"] >= 0.35 * o["bbox_xywh"][2] * o["bbox_xywh"][3]]
        ok.sort(key=lambda o: -o["visible_pixels"])
        shelf += [(o["sku"], crop(im, xyxy(o)), None) for o in ok[:max_shelf_per_frame]]
    return held, shelf


def tune_accept(emb, gal, valid: Path, max_wrong: float, min_px: int) -> Accept:
    held, shelf = identification_set(valid, min_px, 6)
    items = held + shelf
    tops = gal.search(emb.embed([c for _, c, _ in items]), k=2)
    rows = [(t[0][1], t[0][1] - t[1][1], t[0][0] == sku) for (sku, _, _), t in zip(items, tops)]
    best = (Accept(0.0, 0.0), -1.0)
    for ms in np.arange(0.30, 0.95, 0.01):
        for mm in (0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12):
            acc = [r for r in rows if r[0] >= ms and r[1] >= mm]
            wrong = sum(1 for r in acc if not r[2]) / len(rows)
            cover = len(acc) / len(rows)
            if wrong <= max_wrong and cover > best[1]:
                best = (Accept(float(round(ms, 2)), float(mm)), cover)
    print(f"tuned on {len(rows)} validation crops: min_sim {best[0].min_sim}, min_margin {best[0].min_margin}, "
          f"answers {best[1]:.1%} without asking")
    return best[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="data/sim/test")
    ap.add_argument("--valid", default="data/sim/valid")
    ap.add_argument("--embedder", default="runs/embedder")
    ap.add_argument("--no-head", action="store_true", help="score the off-the-shelf DINOv2 without the trained head")
    ap.add_argument("--detector", default=None, help="RF-DETR checkpoint; omit to skip detector and end-to-end scoring")
    ap.add_argument("--detector-size", default="nano")
    ap.add_argument("--catalog", default=None)
    ap.add_argument("--out", default="runs/eval")
    ap.add_argument("--min-shelf-px", type=int, default=2500)
    ap.add_argument("--min-tag-h", type=int, default=22)
    ap.add_argument("--max-wrong", type=float, default=0.02)
    ap.add_argument("--max-frames", type=int, default=0)
    a = ap.parse_args()
    test, out = Path(a.test), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cat = Catalog.load(a.catalog or (test / "catalog.json"))
    edir = Path(a.embedder)
    emb = Embedder(head_path=None if a.no_head else edir / "head.pt")
    gal = Gallery.load(edir / "gallery.npz").ready(emb)
    heldout = set(json.loads((edir / "heldout_skus.json").read_text())) if (edir / "heldout_skus.json").exists() else set()
    R = {"embedder": "dinov2-small" + ("" if a.no_head else " + trained head"), "targets": TARGETS}

    accept = tune_accept(emb, gal, Path(a.valid), a.max_wrong, a.min_shelf_px) if Path(a.valid).exists() else Accept()
    R["accept"] = accept.__dict__

    # ---------------- identification with perfect boxes
    t = time.time()
    held, shelf = identification_set(test, a.min_shelf_px, 8)
    pipe = Pipeline(cat, None, emb, gal, accept=accept)
    sh_top = gal.search(emb.embed([c for _, c, _ in shelf]), k=5)
    s1 = [tp[0][0] == s for (s, _, _), tp in zip(shelf, sh_top)]
    s5 = [s in [x for x, _ in tp] for (s, _, _), tp in zip(shelf, sh_top)]
    ho = [ok for (s, _, _), ok in zip(shelf, s1) if s in heldout]
    R["shelf"] = {"crops": len(shelf), "top1": float(np.mean(s1)), "top5": float(np.mean(s5)),
                  "top1_new_skus": float(np.mean(ho)) if ho else None, "new_sku_crops": len(ho)}
    hres = defaultdict(int)
    hv = defaultdict(lambda: [0, 0])
    for sku, c, view in held:
        bc = read_ean13(c)
        r = pipe.identify(c, try_barcode=True)
        hres["n"] += 1
        hres["barcode_read"] += bool(bc)
        if r.get("sku") is None:
            hres["asked_to_turn"] += 1
        elif r["sku"] == sku:
            hres["right"] += 1
            hres["right_barcode" if r["method"] == "barcode" else "right_visual"] += 1
        else:
            hres["wrong"] += 1
        hv[view][0] += 1
        hv[view][1] += r.get("sku") == sku
    n = max(1, hres["n"])
    R["held"] = {"items": hres["n"], "top1": hres["right"] / n, "wrong_rate": hres["wrong"] / n,
                 "asked_to_turn_rate": hres["asked_to_turn"] / n, "barcode_read_rate": hres["barcode_read"] / n,
                 "top1_when_barcode_read": (hres["right_barcode"] / hres["barcode_read"]) if hres["barcode_read"] else None,
                 "by_view": {k: {"items": v[0], "top1": v[1] / v[0]} for k, v in hv.items()}}
    print(f"identification scored in {time.time() - t:.0f}s")

    # ---------------- tag reading with perfect boxes
    t = time.time()
    reader = TagReader()
    tg = defaultdict(int)
    for L, rgb, _ in frames(test):
        tags = [o for o in L["objects"] if o["type"] == "price_tag" and o["bbox_xywh"][3] >= a.min_tag_h]
        if not tags:
            continue
        im = Image.open(rgb).convert("RGB")
        for o in tags:
            r = reader.read(im, xyxy(o))
            hb = "22-29px" if o["bbox_xywh"][3] < 30 else "30-44px" if o["bbox_xywh"][3] < 45 else "45px+"
            tg["n"] += 1
            tg["n_" + hb] += 1
            tg["price_right"] += r["price_text"] == o["price_text"]
            tg["right_" + hb] += r["price_text"] == o["price_text"]
            tg["price_wrong"] += r["price_text"] is not None and r["price_text"] != o["price_text"]
            tg["promo_right"] += r["promo"] == o["promo"]
    R["tags"] = {"tags": tg["n"], "min_height_px": a.min_tag_h, "price_read": tg["price_right"] / max(1, tg["n"]),
                 "price_misread": tg["price_wrong"] / max(1, tg["n"]), "promo_flag": tg["promo_right"] / max(1, tg["n"]),
                 "by_height": {hb: {"tags": tg["n_" + hb], "price_read": tg["right_" + hb] / tg["n_" + hb]}
                               for hb in ("22-29px", "30-44px", "45px+") if tg["n_" + hb]}}
    print(f"tags scored in {time.time() - t:.0f}s")

    # ---------------- detector + end to end
    if a.detector:
        from aislelens_vision.detector import RFDETRDetector
        det = RFDETRDetector(a.detector, size=a.detector_size)
        pipe = Pipeline(cat, det, emb, gal, tag_reader=reader, accept=accept)
        preds, gts = defaultdict(list), defaultdict(dict)
        e2e = defaultdict(int)
        lat, det_lat = [], []
        for k, (L, rgb, dpath) in enumerate(frames(test)):
            if a.max_frames and k >= a.max_frames:
                break
            im = Image.open(rgb).convert("RGB")
            depth = np.asarray(Image.open(dpath)).astype(np.uint16) if dpath.exists() else None
            t0 = time.perf_counter()
            dets = det(im)
            det_lat.append((time.perf_counter() - t0) * 1000)
            for d in dets:
                preds[d["label"]].append((k, d["score"], (d["x0"], d["y0"], d["x1"], d["y1"])))
            for c, boxes in gt_boxes(L).items():
                gts[c][k] = boxes
            res = pipe.analyze(im, depth, read_tags=True)
            lat.append(res["timings_ms"]["total_ms"])
            gh = [o for o in L["objects"] if o.get("held")]
            if not gh and res["mode"] == "looking" and res["item"]:
                # ground truth for the pack under the centre of view, from the label mask boxes
                W, H = L["camera"]["width"], L["camera"]["height"]
                under = [o for o in L["objects"] if o["type"] == "product" and o["bbox_xywh"][0] <= W / 2 <= o["bbox_xywh"][0] + o["bbox_xywh"][2]
                         and o["bbox_xywh"][1] <= H / 2 <= o["bbox_xywh"][1] + o["bbox_xywh"][3]]
                if under:
                    g = min(under, key=lambda o: o["bbox_xywh"][2] * o["bbox_xywh"][3])
                    it = res["item"]
                    e2e["look_n"] += 1
                    e2e["look_right"] += it.get("sku") == g["sku"]
                    e2e["look_wrong"] += it.get("sku") not in (None, g["sku"])
                    if it.get("sku") == g["sku"]:
                        e2e["look_price_n"] += 1
                        e2e["look_price_tag"] += it.get("price_source") == "shelf tag"
                        e2e["look_price_right"] += abs(it.get("price_myr", -1) - g["price_myr"]) < 0.005
            if gh:
                e2e["held_n"] += 1
                it = res["item"]
                if res["mode"] != "held" or not it:
                    e2e["held_missed"] += 1  # the held item was not found as the held item
                elif it.get("sku") == gh[0]["sku"]:
                    e2e["held_right"] += 1
                elif it.get("sku") is None:
                    e2e["held_asked"] += 1
                else:
                    e2e["held_wrong"] += 1
        for c in ("product", "price_tag", "held_item"):
            for kk in range(len(lat)):
                gts[c].setdefault(kk, [])
        aps = {c: ap50(preds[c], gts[c]) for c in ("product", "price_tag", "held_item")}
        hn = max(1, e2e["held_n"])
        R["detector"] = {"ap50": aps, "map50": float(np.mean([v for v in aps.values() if v is not None])),
                         "median_ms": statistics.median(det_lat), "frames": len(lat)}
        R["end_to_end"] = {"held_frames": e2e["held_n"], "held_top1": e2e["held_right"] / hn,
                           "held_asked_to_turn": e2e["held_asked"] / hn, "held_wrong": e2e["held_wrong"] / hn,
                           "held_missed": e2e["held_missed"] / hn,
                           "looking_frames": e2e["look_n"], "looking_top1": e2e["look_right"] / max(1, e2e["look_n"]),
                           "looking_wrong": e2e["look_wrong"] / max(1, e2e["look_n"]),
                           "looking_price_from_tag": e2e["look_price_tag"] / max(1, e2e["look_price_n"]),
                           "looking_price_right": e2e["look_price_right"] / max(1, e2e["look_price_n"]),
                           "latency_ms": {"median": statistics.median(lat), "p90": float(np.percentile(lat, 90))}}

    (out / "report.json").write_text(json.dumps(R, indent=1), encoding="utf-8")
    print(json.dumps(R, indent=1))


if __name__ == "__main__":
    main()
