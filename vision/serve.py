"""Local vision server: the "brain" the glasses (or the 3D simulator, or a RealSense) talk to.

    python vision/serve.py            # http://127.0.0.1:8770

POST /analyze   {"rgb": <base64 JPEG/PNG>, "depth": <base64 16-bit PNG, mm, optional>,
                 "basket_total": <RM, optional>, "profile": {"allergies": [...], "avoid_high_sugar": bool}}
             -> {"mode", "item": {sku, title, price_myr, price_source, method, confidence, ...},
                 "verdict": {headline, lines, alerts, compare}, "price_tags": [...], "timings_ms": {...}}
POST /basket    {"sku": "AL-011", "price_myr": 11.9}   add to the running basket
DELETE /basket  {"sku": "AL-011"}                      remove one
GET  /basket    running total
GET  /health    which models are loaded

Only listens on 127.0.0.1. In the 3D store, press V ("Ask AisleLens") to send the current camera frame here.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from aislelens_vision.catalog import Catalog  # noqa: E402
from aislelens_vision.embedder import Embedder, Gallery  # noqa: E402
from aislelens_vision.perception import TagReader  # noqa: E402
from aislelens_vision.pipeline import Accept, Pipeline  # noqa: E402
from aislelens_vision.verdict import Basket, Profile  # noqa: E402


def build_pipeline(a) -> tuple[Pipeline, dict]:
    cat = Catalog.load(a.catalog)
    edir = Path(a.embedder)
    head = None if a.no_head else edir / "head.pt"
    emb = Embedder(head_path=head)
    gal = Gallery.load(edir / "gallery.npz").ready(emb)
    accept = Accept()
    acc_file = Path(a.accept)
    if acc_file.exists():
        accept = Accept(**json.loads(acc_file.read_text())["accept"])
    from aislelens_vision.detector import load_detectors
    if a.detector == "auto":
        real = [ROOT / "runs" / d / "checkpoint_best_total.pth" for d in ("det_products", "det_tags")]
        paths = [str(p) for p in real if p.exists()] or [str(ROOT / "runs" / "detector" / "checkpoint_best_total.pth")]
    else:
        paths = [p for p in a.detector.split(",") if p]
    det = load_detectors(paths, size=a.detector_size, resolution=a.detector_resolution)
    if det is not None:
        det_name = f"RF-DETR {a.detector_size}: " + ", ".join(Path(p).parent.name + "/" + Path(p).name for p in paths if Path(p).exists())
    else:
        from aislelens_vision.detector import _det

        def det(img):  # no trained detector yet: treat the centre of view as the product
            w, h = img.size
            return [_det("product", 0.5, w * 0.3, h * 0.2, w * 0.7, h * 0.85)]
        det_name = "none (centre crop fallback; train one with tools/train_detector.py)"
    pipe = Pipeline(cat, det, emb, gal, tag_reader=TagReader(), accept=accept)
    info = {"detector": det_name, "embedder": "DINOv2-small" + ("" if a.no_head else " + head"),
            "gallery_skus": len(gal.uniq), "catalog": len(cat), "accept": accept.__dict__, "ocr": "RapidOCR (PaddleOCR models)",
            "barcode": "ZXing-C++"}
    return pipe, info


def decode_image(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--catalog", default=str(HERE / "data" / "catalog.json"))
    ap.add_argument("--embedder", default=str(ROOT / "runs" / "embedder"))
    ap.add_argument("--no-head", action="store_true")
    ap.add_argument("--detector", default="auto",
                    help="RF-DETR checkpoints, comma separated; auto = the real product + tag models if trained, else the simulator model")
    ap.add_argument("--detector-size", default="nano")
    ap.add_argument("--detector-resolution", type=int, default=None, help="the resolution the checkpoints were trained at, e.g. 640")
    ap.add_argument("--accept", default=str(ROOT / "runs" / "eval" / "report.json"),
                    help="report.json from tools/evaluate.py, for the tuned 'turn the pack' threshold")
    a = ap.parse_args()

    t = time.time()
    pipe, info = build_pipeline(a)
    basket = Basket(pipe.catalog)
    lock = threading.Lock()
    print(f"Models loaded in {time.time() - t:.0f}s: {json.dumps(info)}")

    class H(BaseHTTPRequestHandler):
        def _send(self, code, obj):
            body = json.dumps(obj, default=float).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self):
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Allow-Private-Network", "true")
            self.end_headers()

        def _json(self):
            n = int(self.headers.get("Content-Length", 0))
            return json.loads(self.rfile.read(n) or b"{}")

        def do_GET(self):
            if self.path == "/health":
                return self._send(200, {"ok": True, **info})
            if self.path == "/basket":
                return self._send(200, basket.summary())
            self._send(404, {"error": "not found"})

        def do_DELETE(self):
            if self.path == "/basket":
                return self._send(200, basket.remove(self._json().get("sku", "")))
            self._send(404, {"error": "not found"})

        def do_POST(self):
            try:
                req = self._json()
                if self.path == "/basket":
                    p = pipe.catalog.by_sku.get(req.get("sku", ""))
                    if not p:
                        return self._send(400, {"error": "unknown sku"})
                    return self._send(200, basket.add(p, float(req.get("price_myr", p.price_myr)), int(req.get("qty", 1))))
                if self.path != "/analyze":
                    return self._send(404, {"error": "not found"})
                img = decode_image(req["rgb"]).convert("RGB")
                depth = np.asarray(decode_image(req["depth"])).astype(np.uint16) if req.get("depth") else None
                prof = req.get("profile") or {}
                with lock:  # one frame at a time; the models are not thread-safe
                    pipe.profile = Profile(allergies=prof.get("allergies", []), avoid_high_sugar=bool(prof.get("avoid_high_sugar")))
                    res = pipe.analyze(img, depth, basket_total=float(req.get("basket_total") or basket.total))
                res["basket"] = basket.summary()
                self._send(200, res)
            except Exception as e:  # report, keep serving
                self._send(500, {"error": f"{type(e).__name__}: {e}"})

        def log_message(self, fmt, *args):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", a.port), H)
    print(f"AisleLens vision server on http://127.0.0.1:{a.port}  (Ctrl+C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
