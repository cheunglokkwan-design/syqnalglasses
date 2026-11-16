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

Listens on 127.0.0.1 unless --lan. The phone app is served at /app/. In the 3D store, press V ("Ask AisleLens") to send the current camera frame here.
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


CATALOGS = {  # --catalog real | sim: the catalogue, its embedding gallery and its product photos
    "real": {"catalog": HERE / "data" / "real_catalog.json", "embedder": ROOT / "runs" / "embedder_real",
             "photos": ROOT / "data" / "real" / "gallery", "head": False},
    "sim": {"catalog": HERE / "data" / "catalog.json", "embedder": ROOT / "runs" / "embedder",
            "photos": ROOT / "data" / "sim" / "test" / "gallery", "head": True},
}


def build_pipeline(a) -> tuple[Pipeline, dict]:
    preset = CATALOGS.get(a.catalog)
    if preset:
        a.catalog, a.photos = str(preset["catalog"]), str(preset["photos"])
        if a.embedder is None:
            a.embedder = str(preset["embedder"])
        if not preset["head"]:
            a.no_head = True
    a.embedder = a.embedder or str(ROOT / "runs" / "embedder")
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
    info = {"catalog_file": Path(a.catalog).name, "here_store": cat.here_store,
            "detector": det_name, "embedder": "DINOv2-small" + ("" if a.no_head else " + head"),
            "gallery_skus": len(gal.uniq), "catalog": len(cat), "accept": accept.__dict__, "ocr": "RapidOCR (PaddleOCR models)",
            "barcode": "ZXing-C++"}
    return pipe, info


def decode_image(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--catalog", default="real", help="real (the 20 real products), sim (the 3D store), or a catalogue file")
    ap.add_argument("--embedder", default=None, help="folder with gallery.npz (and head.pt)")
    ap.add_argument("--photos", default=None, help="product photo folders (<photos>/<sku>/*.jpg), shown in the app")
    ap.add_argument("--lan", action="store_true", help="listen on the local network (phone, Jetson), not only this computer")
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

    photos = Path(a.photos) if a.photos else None
    app_dir = ROOT / "app"

    def photo_url(sku):
        if photos and (photos / sku).is_dir() and any((photos / sku).iterdir()):
            return f"/image/{sku}"
        return None

    def product_card(p, price=None, source=None):
        nut = p.nutrition_per_100g() or {}
        return {"sku": p.sku, "title": p.title, "size": p.size, "price_myr": price if price is not None else p.price_myr,
                "price_source": source or pipe.catalog.here_store, "allergens": p.allergens,
                "sugars_g": nut.get("sugars_g"), "image": photo_url(p.sku)}

    def enrich(res):
        it = res.get("item")
        if it and it.get("sku"):
            p = pipe.catalog.by_sku[it["sku"]]
            it["image"] = photo_url(p.sku)
            res["alternatives"] = [product_card(q) for q in pipe.catalog.alternatives(p)]
        if it and it.get("top5"):
            res["candidates"] = [{**product_card(pipe.catalog.by_sku[s]), "score": sc} for s, sc in it["top5"][:3]
                                 if s in pipe.catalog.by_sku]
        return res

    class H(BaseHTTPRequestHandler):
        def _file(self, path: Path, ctype: str, cache: str = "max-age=300"):
            data = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", cache)
            self.end_headers()
            self.wfile.write(data)

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
            if self.path.startswith("/image/") and photos:
                sku = self.path.split("/")[2].split("?")[0]
                files = sorted((photos / sku).glob("*.jp*g")) if (photos / sku).is_dir() and "." not in sku else []
                if files:
                    return self._file(files[0], "image/jpeg")
            if self.path == "/" or self.path.startswith("/app"):
                # the phone app, served here too for testing on this computer (http://localhost:<port>/app/)
                rel = self.path.split("?")[0][len("/app"):].lstrip("/") or "index.html"
                f = (app_dir / rel).resolve()
                if app_dir.resolve() in f.parents and f.is_file():
                    ct = {".html": "text/html", ".js": "text/javascript", ".css": "text/css", ".json": "application/json",
                          ".svg": "image/svg+xml", ".png": "image/png"}.get(f.suffix, "application/octet-stream")
                    return self._file(f, ct + ("; charset=utf-8" if ct.startswith("text") else ""), "no-cache")
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
                if self.path == "/verdict":  # the shopper tapped or said which product it is
                    p = pipe.catalog.by_sku.get(req.get("sku", ""))
                    if not p:
                        return self._send(400, {"error": "unknown sku"})
                    from aislelens_vision.verdict import verdict as make_verdict
                    prof = req.get("profile") or {}
                    v = make_verdict(p, p.price_myr, pipe.catalog.here_store, Profile(allergies=prof.get("allergies", [])),
                                     basket_total=basket.total)
                    res = {"mode": "confirmed", "item": {**product_card(p), "method": "confirmed by you", "confidence": 1.0},
                           "verdict": v, "alternatives": [product_card(q) for q in pipe.catalog.alternatives(p)],
                           "basket": basket.summary(), "timings_ms": {"total_ms": 0}}
                    return self._send(200, res)
                if self.path != "/analyze":
                    return self._send(404, {"error": "not found"})
                img = decode_image(req["rgb"]).convert("RGB")
                depth = np.asarray(decode_image(req["depth"])).astype(np.uint16) if req.get("depth") else None
                prof = req.get("profile") or {}
                with lock:  # one frame at a time; the models are not thread-safe
                    pipe.profile = Profile(allergies=prof.get("allergies", []), avoid_high_sugar=bool(prof.get("avoid_high_sugar")))
                    res = pipe.analyze(img, depth, basket_total=float(req.get("basket_total") or basket.total))
                res["basket"] = basket.summary()
                self._send(200, enrich(res))
            except Exception as e:  # report, keep serving
                self._send(500, {"error": f"{type(e).__name__}: {e}"})

        def log_message(self, fmt, *args):
            pass

    host = "0.0.0.0" if a.lan else "127.0.0.1"
    srv = ThreadingHTTPServer((host, a.port), H)
    print(f"AisleLens vision server on http://{host}:{a.port}  app: http://localhost:{a.port}/app/  (Ctrl+C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
