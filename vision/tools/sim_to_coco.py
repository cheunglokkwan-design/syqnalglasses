"""Convert simulator frames into the COCO (Roboflow-style) layout RF-DETR trains on.

    data/sim/train, data/sim/valid, data/sim/test   (from generate_dataset.mjs)
      -> data/coco/{train,valid,test}/_annotations.coco.json + images (hard links, no extra disk)

Classes: product, price_tag, held_item. Tiny slivers of packs at the frame edge
or deep behind others are left out (fewer than --min-product-px visible pixels),
since nobody would expect the glasses to name them.

Usage: python vision/tools/sim_to_coco.py --sim data/sim --out data/coco
(folders named train_* are merged into train)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

CATS = [{"id": 1, "name": "product", "supercategory": "none"},
        {"id": 2, "name": "price_tag", "supercategory": "none"},
        {"id": 3, "name": "held_item", "supercategory": "none"}]


def convert(split_dirs: list[Path], out_dir: Path, min_product_px: int, min_tag_px: int) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    images, anns, counts = [], [], {"product": 0, "price_tag": 0, "held_item": 0}
    files = [(d, lf) for d in split_dirs for lf in sorted((d / "labels").glob("*.json"))]
    for i, (split_dir, lf) in enumerate(files):
        L = json.loads(lf.read_text(encoding="utf-8"))
        src = split_dir / L.get("rgb_file", f"rgb/{lf.stem}.png")
        name = f"{split_dir.name}_{src.name}"
        dst = out_dir / name
        if not dst.exists():
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
        cam = L["camera"]
        images.append({"id": i, "file_name": name, "width": cam["width"], "height": cam["height"]})
        for o in L["objects"]:
            if o["type"] == "price_tag":
                if o["visible_pixels"] < min_tag_px:
                    continue
                cid, key = 2, "price_tag"
            elif o.get("held"):
                cid, key = 3, "held_item"
            else:
                if o["visible_pixels"] < min_product_px:
                    continue
                cid, key = 1, "product"
            x, y, w, h = o["bbox_xywh"]
            anns.append({"id": len(anns), "image_id": i, "category_id": cid, "bbox": [x, y, w, h], "area": w * h,
                         "iscrowd": 0, "segmentation": [], "sku": o["sku"]})
            counts[key] += 1
    coco = {"info": {"description": "AisleLens simulator: " + ", ".join(d.name for d in split_dirs)}, "categories": CATS,
            "images": images, "annotations": anns}
    (out_dir / "_annotations.coco.json").write_text(json.dumps(coco), encoding="utf-8")
    return {"images": len(images), **counts}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", default="data/sim")
    ap.add_argument("--out", default="data/coco")
    ap.add_argument("--min-product-px", type=int, default=300)
    ap.add_argument("--min-tag-px", type=int, default=60)
    a = ap.parse_args()
    sim = Path(a.sim)
    for split in ("train", "valid", "test"):
        # train also takes extra folders such as train_scan
        dirs = sorted(d for d in sim.iterdir() if d.is_dir() and (d.name == split or d.name.startswith(split + "_")))
        if dirs:
            print(split, [d.name for d in dirs], convert(dirs, Path(a.out) / split, a.min_product_px, a.min_tag_px))


if __name__ == "__main__":
    main()
