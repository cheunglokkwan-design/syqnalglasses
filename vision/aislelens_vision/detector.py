"""Product / price tag / held item detector (RF-DETR, Apache-2.0).

Three classes, as in docs/vision-model-report.md: product, price_tag, held_item.
The detector only finds boxes; which exact product it is comes from the
embedder (or the barcode).
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image

CLASSES = ["product", "price_tag", "held_item"]


def _det(label, score, x0, y0, x1, y1, **extra):
    return {"label": label, "score": float(score), "x0": float(x0), "y0": float(y0), "x1": float(x1), "y1": float(y1),
            "w": float(x1 - x0), "h": float(y1 - y0), **extra}


class RFDETRDetector:
    def __init__(self, weights: str | Path, size: str = "nano", resolution: int | None = None,
                 thresholds: dict | None = None, device: str | None = None):
        import rfdetr
        cls = {"nano": rfdetr.RFDETRNano, "small": rfdetr.RFDETRSmall, "medium": rfdetr.RFDETRMedium,
               "base": rfdetr.RFDETRBase}[size]
        kw = {"pretrain_weights": str(weights)}
        if resolution:
            kw["resolution"] = resolution
        if device:
            kw["device"] = device
        self.model = cls(**kw)
        self.names = list(self.model.class_names)
        self.thresholds = {"product": 0.35, "price_tag": 0.3, "held_item": 0.4, **(thresholds or {})}
        try:
            self.model.optimize_for_inference()
        except Exception:
            pass

    def __call__(self, img: Image.Image) -> list[dict]:
        r = self.model.predict(img, threshold=min(self.thresholds.values()), include_source_image=False)
        out = []
        for (x0, y0, x1, y1), c, s in zip(r.xyxy, r.class_id, r.confidence):
            name = self.names[int(c)] if int(c) < len(self.names) else str(c)
            if name in self.thresholds and s >= self.thresholds[name]:
                out.append(_det(name, s, x0, y0, x1, y1))
        return out


class GroundTruthDetector:
    """Uses the simulator's labels as perfect detections, to test later stages on their own."""

    def __init__(self, min_pixels: int = 400):
        self.min_pixels = min_pixels
        self.labels = None

    def set_labels(self, labels: dict):
        self.labels = labels

    def __call__(self, img: Image.Image) -> list[dict]:
        out = []
        for o in self.labels["objects"]:
            if o["visible_pixels"] < (40 if o["type"] == "price_tag" else self.min_pixels):
                continue
            x, y, w, h = o["bbox_xywh"]
            label = "price_tag" if o["type"] == "price_tag" else ("held_item" if o.get("held") else "product")
            out.append(_det(label, 1.0, x, y, x + w, y + h, gt_sku=o["sku"]))
        return out
