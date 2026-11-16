"""Run the pipeline on one image (optionally with its depth PNG) and print the verdict.

    python vision/analyze.py data/sim/test/rgb/00004.jpg --depth data/sim/test/depth/00004.png
    python vision/analyze.py photo.jpg --allergies peanuts,milk --save annotated.jpg
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import serve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--depth", default=None)
    ap.add_argument("--allergies", default="")
    ap.add_argument("--save", default=None, help="write the image with the chosen item and tags boxed")
    ap.add_argument("--json", action="store_true", help="print the full result")
    a, rest = ap.parse_known_args()
    sa = serve.argparse.ArgumentParser()
    for k, v in [("--catalog", str(serve.HERE / "data" / "catalog.json")), ("--embedder", str(serve.ROOT / "runs" / "embedder")),
                 ("--detector", "auto"),
                 ("--detector-size", "nano"),
                 ("--accept", str(serve.ROOT / "runs" / "eval" / "report.json"))]:
        sa.add_argument(k, default=v)
    sa.add_argument("--no-head", action="store_true")
    sa.add_argument("--detector-resolution", type=int, default=None)
    pipe, info = serve.build_pipeline(sa.parse_args(rest))
    pipe.profile = serve.Profile(allergies=[x for x in a.allergies.split(",") if x])
    img = Image.open(a.image).convert("RGB")
    depth = np.asarray(Image.open(a.depth)).astype(np.uint16) if a.depth and Path(a.depth).exists() else None
    res = pipe.analyze(img, depth, identify_shelf=6)
    if a.json:
        print(json.dumps(res, indent=1, default=float))
    v, it = res.get("verdict") or {}, res.get("item") or {}
    print(f"\nMode: {res['mode']}  |  detector: {info['detector']}")
    print(f"Lens: {v.get('headline', 'no product in view')}")
    for line in v.get("lines", []):
        print("   ", line)
    if it:
        print(f"Item: {it.get('title', '?')} via {it.get('method')} (confidence {it.get('confidence')}), top-5 {it.get('top5')}")
    print(f"Tags read: {[t['price_text'] for t in res['price_tags']]}")
    print(f"Time: {res['timings_ms']}")
    if a.save:
        d = ImageDraw.Draw(img)
        for t in res["price_tags"]:
            d.rectangle([t["x0"], t["y0"], t["x1"], t["y1"]], outline="yellow", width=2)
        if it.get("box"):
            d.rectangle(it["box"], outline="lime" if it.get("sku") else "red", width=4)
            d.text((it["box"][0] + 4, max(0, it["box"][1] - 14)), v.get("headline", ""), fill="lime")
        img.save(a.save)


if __name__ == "__main__":
    main()
