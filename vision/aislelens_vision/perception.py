"""Barcode, price-tag OCR and depth helpers.

- Barcode: ZXing-C++ (Apache-2.0), EAN-13 only. Tried first whenever a pack is close.
- OCR: RapidOCR, which runs PaddleOCR's detection and recognition models through
  ONNX Runtime (Apache-2.0). Same models as PaddleOCR, much lighter to install.
- Depth: RealSense-style 16-bit millimetre depth. Used only to pick the item in
  the hand and to drop far-aisle detections, never as a model input, because the
  final glasses probably will not have a depth sensor.
"""
from __future__ import annotations

import re

import numpy as np
from PIL import Image

PRICE_RE = re.compile(r"(?<!\d)(\d{1,3})[.,](\d{2})(?!\d)")


# ---------------------------------------------------------------- barcode
def read_ean13(img: Image.Image) -> str | None:
    """Return a valid EAN-13 from the image (tries 1x and 2x), or None."""
    import zxingcpp
    arr = np.asarray(img.convert("L"))
    for scale in (1, 2):
        a = arr if scale == 1 else np.asarray(img.convert("L").resize((img.width * 2, img.height * 2), Image.BICUBIC))
        try:
            res = zxingcpp.read_barcodes(a, formats=zxingcpp.BarcodeFormat.EAN13, try_rotate=True, try_downscale=True)
        except TypeError:  # older binding signature
            res = zxingcpp.read_barcodes(a, formats=zxingcpp.BarcodeFormat.EAN13)
        for r in res:
            t = r.text.strip()
            if len(t) == 13 and t.isdigit() and ean13_ok(t):
                return t
        if min(img.size) > 400:
            break
    return None


def ean13_ok(code: str) -> bool:
    d = [int(c) for c in code]
    return (10 - sum(v * (3 if i % 2 else 1) for i, v in enumerate(d[:12])) % 10) % 10 == d[12]


# ---------------------------------------------------------------- OCR
class TagReader:
    """Reads a shelf-edge price tag: price, PROMO flag and (when big enough) the name line.

    Tags are small in a glasses frame (20-50 px tall), too blurry for text detection, so
    the price is read by running only the recogniser on the price area of the tag layout
    (big number, right of centre), three slightly different crops, and voting. Promo tags
    are recognised by their red background. Each store chain needs its own tag layout here.
    """

    PRICE_REGIONS = [(0.36, 0.26, 0.99, 0.80), (0.30, 0.22, 1.0, 0.84), (0.42, 0.30, 0.97, 0.76)]

    def __init__(self, min_conf: float = 0.5):
        from rapidocr_onnxruntime import RapidOCR
        self.ocr = RapidOCR()
        self.min_conf = min_conf

    def _rec(self, crop: Image.Image, h: int = 48) -> tuple[str, float]:
        crop = crop.resize((max(8, int(crop.width * h / max(1, crop.height))), h), Image.BICUBIC)
        res, _ = self.ocr(np.asarray(crop.convert("RGB")), use_det=False, use_cls=False, use_rec=True)
        return (res[0][0], float(res[0][1])) if res else ("", 0.0)

    @staticmethod
    def parse_price(txt: str) -> str | None:
        t = re.sub(r"[^0-9.,:]", "", txt)
        m = PRICE_RE.search(t.replace(":", "."))
        if m:
            return f"{int(m.group(1))}.{m.group(2)}"
        d = re.sub(r"\D", "", t)  # big digits sometimes lose the decimal point: "790" -> 7.90
        return f"{int(d[:-2])}.{d[-2:]}" if 3 <= len(d) <= 5 else None

    def read(self, img: Image.Image, box) -> dict:
        x0, y0, x1, y1 = box
        w, h = x1 - x0, y1 - y0
        votes: dict[str, float] = {}
        for rx0, ry0, rx1, ry1 in self.PRICE_REGIONS:
            txt, conf = self._rec(img.crop((x0 + rx0 * w, y0 + ry0 * h, x0 + rx1 * w, y0 + ry1 * h)))
            p = self.parse_price(txt)
            if p and conf >= self.min_conf:
                votes[p] = votes.get(p, 0.0) + conf
        price = max(votes, key=votes.get) if votes else None
        conf = round(votes[price] / len(self.PRICE_REGIONS), 3) if price else 0.0
        a = np.asarray(img.crop((x0 + 0.05 * w, y0 + 0.3 * h, x0 + 0.35 * w, y0 + 0.7 * h)).convert("RGB")).reshape(-1, 3).mean(0)
        promo = bool(a[0] > 1.4 * a[1] and a[0] > 90)
        name = ""
        if h >= 48:  # the name line is only legible on big (close) tags
            name, _ = self._rec(img.crop((x0 + 0.02 * w, y0 + 0.02 * h, x1 - 0.02 * w, y0 + 0.22 * h)), 32)
        return {"price_text": price, "price_conf": conf, "promo": promo, "name_text": name}


def tag_for_item(item: dict, tags: list[dict]) -> dict | None:
    """The price tag that belongs to a product facing: on the shelf edge just below the
    pack, at or left of it (tags sit at the left end of each run of facings)."""
    best, best_d = None, 1e9
    for t in tags:
        if not (item["y1"] - 0.25 * item["h"] <= t["y0"] <= item["y1"] + 0.9 * item["h"]):
            continue
        if t["x0"] > item["x1"] or t["x1"] < item["x0"] - 3.5 * item["w"]:
            continue
        dx = max(0.0, item["x0"] - t["x1"]) + 0.3 * abs(t["y0"] - item["y1"])
        if dx < best_d:
            best, best_d = t, dx
    return best


# ---------------------------------------------------------------- depth
def box_depth_m(depth_mm: np.ndarray | None, box_xyxy) -> float | None:
    """Median depth of the central half of a box, ignoring holes (0 = no data)."""
    if depth_mm is None:
        return None
    x0, y0, x1, y1 = [int(round(v)) for v in box_xyxy]
    w, h = x1 - x0, y1 - y0
    cx0, cx1 = max(0, x0 + w // 4), min(depth_mm.shape[1], x1 - w // 4)
    cy0, cy1 = max(0, y0 + h // 4), min(depth_mm.shape[0], y1 - h // 4)
    if cx1 <= cx0 or cy1 <= cy0:
        return None
    v = depth_mm[cy0:cy1, cx0:cx1]
    v = v[v > 0]
    return float(np.median(v)) / 1000.0 if v.size >= 10 else None


def pick_held(dets: list[dict], img_w: int, img_h: int, max_hold_m: float = 0.85) -> dict | None:
    """Choose the item in the shopper's hand.

    Order of evidence: the detector's held_item class, then the nearest large
    product within arm's reach (depth), then the largest box near the bottom
    centre when depth has no data (e.g. a D455 closer than its 0.52 m minimum).
    """
    area = img_w * img_h
    held = [d for d in dets if d["label"] == "held_item"]
    if held:
        return max(held, key=lambda d: d["score"] * (d["w"] * d["h"]))
    near = [d for d in dets if d["label"] in ("product", "held_item") and d.get("depth_m") is not None
            and d["depth_m"] < max_hold_m and d["w"] * d["h"] > 0.03 * area]
    if near:
        return min(near, key=lambda d: d["depth_m"])
    big = [d for d in dets if d["label"] == "product" and d["w"] * d["h"] > 0.08 * area
           and d.get("depth_m") is None and d["y1"] > img_h * 0.55]
    return max(big, key=lambda d: d["w"] * d["h"]) if big else None
