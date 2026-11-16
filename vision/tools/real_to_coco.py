"""Convert downloaded real-image datasets into RF-DETR's COCO layout.

Two detectors are trained because each public set labels only one kind of thing:
  data/real/coco_products   SKU-110K shelf photos, class "product" (non-commercial licence:
                            prototype and research use only, replace before a commercial launch)
  data/real/coco_tags       Open Food Facts price-tag-detection, class "price_tag" (CC BY-SA 4.0)

Images are re-saved as JPEG with the long side at most --max-side pixels to save disk.
Each source's validation split is cut in half: valid (for training checks) and test (for scoring).

Usage: python vision/tools/real_to_coco.py --raw data/real/raw --out data/real
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image


def save_img(data: bytes, path: Path, max_side: int) -> tuple[int, int, float]:
    im = Image.open(io.BytesIO(data)).convert("RGB")
    s = min(1.0, max_side / max(im.size))
    if s < 1:
        im = im.resize((round(im.width * s), round(im.height * s)), Image.BILINEAR)
    im.save(path, quality=90)
    return im.width, im.height, s


def write(split_dir: Path, name: str, images: list, anns: list):
    coco = {"info": {"description": name}, "categories": [{"id": 1, "name": name, "supercategory": "none"}],
            "images": images, "annotations": anns}
    (split_dir / "_annotations.coco.json").write_text(json.dumps(coco), encoding="utf-8")


def rows(files):
    for f in files:
        pf = pq.ParquetFile(f)
        for g in range(pf.num_row_groups):
            yield from pf.read_row_group(g).to_pylist()


def box_from(b, W, H, fmt):
    """Return COCO [x, y, w, h] in pixels of the saved image size (W, H)."""
    if fmt == "yxyx_norm":
        y0, x0, y1, x1 = b
        x0, x1, y0, y1 = x0 * W, x1 * W, y0 * H, y1 * H
    elif fmt == "xyxy_px":
        x0, y0, x1, y1 = b
    else:  # xywh_px
        x0, y0, w, h = b
        x1, y1 = x0 + w, y0 + h
    x0, y0 = max(0.0, x0), max(0.0, y0)
    x1, y1 = min(float(W), x1), min(float(H), y1)
    return [round(x0, 1), round(y0, 1), round(x1 - x0, 1), round(y1 - y0, 1)]


def convert(src_files: dict, out: Path, name: str, fmt: str, max_side: int, scale_boxes: bool):
    stats = {}
    for split, files in src_files.items():
        d = out / split
        d.mkdir(parents=True, exist_ok=True)
        images, anns = [], []
        for i, r in enumerate(rows(files)):
            img = r.get("image") or {}
            data = img.get("bytes") if isinstance(img, dict) else img
            if not data:
                continue
            fn = f"{split}_{i:05d}.jpg"
            W, H, s = save_img(data, d / fn, max_side)
            images.append({"id": i, "file_name": fn, "width": W, "height": H})
            objs = r.get("objects") or {}
            for b in objs.get("bbox") or []:
                if scale_boxes:
                    b = [v * s for v in b]
                x, y, w, h = box_from(b, W, H, fmt)
                if w >= 2 and h >= 2:
                    anns.append({"id": len(anns), "image_id": i, "category_id": 1, "bbox": [x, y, w, h], "area": w * h, "iscrowd": 0})
        write(d, name, images, anns)
        stats[split] = {"images": len(images), "boxes": len(anns)}
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/real/raw")
    ap.add_argument("--out", default="data/real")
    ap.add_argument("--max-side", type=int, default=1280)
    ap.add_argument("--only", choices=["tags", "products"], default=None)
    a = ap.parse_args()
    raw, out = Path(a.raw), Path(a.out)

    import pyarrow as pa
    tmp = out / "_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    if a.only != "products":
        tags(raw, out, tmp, a.max_side, pa)
    if a.only != "tags":
        products(raw, out, tmp, a.max_side, pa)
    for f in tmp.glob("*.parquet"):
        f.unlink()
    tmp.rmdir()


def tags(raw, out, tmp, max_side, pa):
    # OFF train -> train; OFF val -> valid + test (first half / second half)
    tag_val = list(rows([raw / "pricetags" / "val.parquet"]))
    for nm, part in (("valid", tag_val[: len(tag_val) // 2]), ("test", tag_val[len(tag_val) // 2:])):
        pq.write_table(pa.Table.from_pylist(part), tmp / f"tags_{nm}.parquet")
    print("price tags", convert({"train": [raw / "pricetags" / "train.parquet"], "valid": [tmp / "tags_valid.parquet"],
                                 "test": [tmp / "tags_test.parquet"]}, out / "coco_tags", "price_tag", "yxyx_norm", max_side, False))


def products(raw, out, tmp, max_side, pa):
    # SKU-110K boxes are pixel x, y, w, h (checked by drawing them)
    sku = sorted((raw / "sku110k").glob("train-*.parquet"))
    val = list(rows(sorted((raw / "sku110k").glob("validation-*.parquet"))))
    for nm, part in (("valid", val[: len(val) // 2]), ("test", val[len(val) // 2:])):
        pq.write_table(pa.Table.from_pylist(part), tmp / f"sku_{nm}.parquet")
    print("products", convert({"train": sku, "valid": [tmp / "sku_valid.parquet"], "test": [tmp / "sku_test.parquet"]},
                              out / "coco_products", "product", "xywh_px", max_side, True))


if __name__ == "__main__":
    main()
