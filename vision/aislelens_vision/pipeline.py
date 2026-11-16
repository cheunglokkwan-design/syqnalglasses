"""The full glance-to-verdict loop: detect, pick the item, identify it, read the price, decide.

    frame (RGB, optional depth)
      -> detector: products, price tags, held item
      -> depth: drop the far aisle, pick the item in the hand
      -> barcode first (ZXing); if none, embedding search (DINOv2 + gallery)
      -> confidence check: unsure -> ask the shopper to turn the pack
      -> price tag OCR, matched to the product by the name on the tag
      -> verdict: price vs other stores, promo check, allergens, sugar
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from PIL import Image

from .catalog import Catalog
from .perception import TagReader, box_depth_m, pick_held, read_ean13, tag_for_item
from .verdict import Profile, verdict


@dataclass
class Accept:
    """Visual match is accepted when top-1 similarity and the gap to top-2 are both high enough.

    Tuned on the validation split by tools/evaluate.py so the wrong-answer rate stays under target.
    """
    min_sim: float = 0.55
    min_margin: float = 0.02


def _crop(img: Image.Image, d: dict, pad: float = 0.04) -> Image.Image:
    px, py = d["w"] * pad, d["h"] * pad
    return img.crop((max(0, d["x0"] - px), max(0, d["y0"] - py), min(img.width, d["x1"] + px), min(img.height, d["y1"] + py)))


class Pipeline:
    def __init__(self, catalog: Catalog, detector, embedder, gallery, tag_reader: TagReader | None = None,
                 profile: Profile | None = None, accept: Accept | None = None, far_m: float = 4.0):
        self.catalog, self.detector, self.embedder, self.gallery = catalog, detector, embedder, gallery
        self.tags = tag_reader
        self.profile = profile or Profile()
        self.accept = accept or Accept()
        self.far_m = far_m

    # ------------------------------------------------------------ identification
    def identify(self, crop: Image.Image, try_barcode: bool = True, timings: dict | None = None) -> dict:
        timings = timings if timings is not None else {}
        if try_barcode:
            t = time.perf_counter()
            code = read_ean13(crop)
            timings["barcode_ms"] = timings.get("barcode_ms", 0) + (time.perf_counter() - t) * 1000
            if code:
                p = self.catalog.by_barcode.get(code)
                if p:
                    return {"sku": p.sku, "method": "barcode", "confidence": 0.99, "barcode": code, "top5": [(p.sku, 1.0)]}
                return {"sku": None, "method": "barcode", "status": "unknown_barcode", "barcode": code, "top5": []}
        t = time.perf_counter()
        top = self.gallery.search(self.embedder.embed([crop]), k=5)[0]
        timings["embed_ms"] = timings.get("embed_ms", 0) + (time.perf_counter() - t) * 1000
        return self._decide(top)

    def _decide(self, top: list[tuple[str, float]]) -> dict:
        s1 = top[0][1]
        s2 = top[1][1] if len(top) > 1 else 0.0
        ok = s1 >= self.accept.min_sim and (s1 - s2) >= self.accept.min_margin
        return {"sku": top[0][0] if ok else None, "method": "visual", "confidence": round(s1, 3), "margin": round(s1 - s2, 3),
                "status": "ok" if ok else "unsure", "top5": [(s, round(v, 3)) for s, v in top]}

    # ------------------------------------------------------------ tags
    def read_tags(self, img: Image.Image, dets: list[dict], max_tags: int = 6, timings: dict | None = None) -> list[dict]:
        if self.tags is None:
            return []
        timings = timings if timings is not None else {}
        t = time.perf_counter()
        tags = sorted((d for d in dets if d["label"] == "price_tag" and d["h"] >= 14), key=lambda d: -d["w"] * d["h"])[:max_tags]
        out = []
        for d in tags:
            r = self.tags.read(img, (d["x0"], d["y0"], d["x1"], d["y1"]))
            p = self.catalog.match_name(r["name_text"]) if r["name_text"] else None
            out.append({**{k: d[k] for k in ("x0", "y0", "x1", "y1", "w", "h", "score")}, **r, "sku": p.sku if p else None})
        timings["ocr_ms"] = (time.perf_counter() - t) * 1000
        return out

    # ------------------------------------------------------------ whole frame
    def analyze(self, img: Image.Image, depth_mm: np.ndarray | None = None, identify_shelf: int = 0,
                read_tags: bool = True, basket_total: float = 0.0) -> dict:
        timings: dict = {}
        t0 = time.perf_counter()
        img = img.convert("RGB")
        W, H = img.size
        dets = self.detector(img)
        timings["detect_ms"] = (time.perf_counter() - t0) * 1000
        for d in dets:
            d["depth_m"] = box_depth_m(depth_mm, (d["x0"], d["y0"], d["x1"], d["y1"]))
        dets = [d for d in dets if d["depth_m"] is None or d["depth_m"] <= self.far_m]

        target = pick_held(dets, W, H)
        mode = "held" if target else None
        if target is None:  # what the shopper is looking at: the product under the centre of view
            cx, cy = W / 2, H / 2
            under = [d for d in dets if d["label"] == "product" and d["x0"] <= cx <= d["x1"] and d["y0"] <= cy <= d["y1"]]
            if under:
                target, mode = min(under, key=lambda d: d["w"] * d["h"]), "looking"

        result = {"mode": mode, "detections": len(dets), "item": None, "verdict": None, "image_size": [W, H],
                  "boxes": [{"label": d["label"], "score": round(d["score"], 3), "depth_m": d.get("depth_m"),
                             "box": [round(d[k]) for k in ("x0", "y0", "x1", "y1")]} for d in dets]}
        if target is not None:
            crop = _crop(img, target)
            ident = self.identify(crop, try_barcode=(mode == "held" or target["h"] > H * 0.3), timings=timings)
            result["item"] = {"box": [round(target[k]) for k in ("x0", "y0", "x1", "y1")], "depth_m": target.get("depth_m"),
                              "detector_label": target["label"], **ident}

        tags = self.read_tags(img, dets, timings=timings) if read_tags else []
        result["price_tags"] = tags

        item = result["item"]
        if item and item.get("sku"):
            p = self.catalog.by_sku[item["sku"]]
            # the tag naming this product, else (for a pack on the shelf) the tag below it
            tag = next((t for t in tags if t["sku"] == p.sku and t["price_text"]), None)
            if tag is None and mode == "looking":
                tag = tag_for_item(target, [t for t in tags if t["price_text"]])
                if tag is not None and tag["sku"] not in (None, p.sku):
                    tag = None
            # a tag price far from the known price is more likely a misread or the neighbour's tag
            if tag is not None and not (0.5 * p.price_myr <= float(tag["price_text"]) <= 1.5 * p.price_myr):
                tag = None
            price, src = (float(tag["price_text"]), "shelf tag") if tag else (p.price_myr, self.catalog.here_store)
            if tag is not None:
                item["tag_box"] = [round(tag[k]) for k in ("x0", "y0", "x1", "y1")]
            item.update({"title": p.title, "size": p.size, "price_myr": price, "price_source": src})
            result["verdict"] = verdict(p, price, src, self.profile, promo_tag=tag["promo"] if tag else None,
                                        basket_total=basket_total)
        elif item and item.get("status") in ("unsure", "unknown_barcode"):
            cand = ", ".join(self.catalog.by_sku[s].title for s, _ in item["top5"][:3]) if item["top5"] else ""
            msg = "Barcode not in the catalogue" if item["status"] == "unknown_barcode" else "Not sure: turn the pack to show the barcode"
            result["verdict"] = {"headline": msg, "lines": [f"Could be: {cand}"] if cand else [], "alerts": [], "compare": None}

        if identify_shelf:
            prods = sorted((d for d in dets if d["label"] == "product" and d is not target), key=lambda d: -d["w"] * d["h"])[:identify_shelf]
            if prods:
                t = time.perf_counter()
                tops = self.gallery.search(self.embedder.embed([_crop(img, d) for d in prods]), k=3)
                timings["shelf_embed_ms"] = (time.perf_counter() - t) * 1000
                result["shelf"] = [{"box": [round(d[k]) for k in ("x0", "y0", "x1", "y1")], **self._decide(tp)} for d, tp in zip(prods, tops)]
        timings["total_ms"] = (time.perf_counter() - t0) * 1000
        result["timings_ms"] = {k: round(v, 1) for k, v in timings.items()}
        return result
