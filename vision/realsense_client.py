"""Use a real Intel RealSense camera with the AisleLens vision server, or record store frames.

Frames are colour-aligned depth in the same format as the 3D simulator (16-bit PNG,
millimetres, 0 = no data), so the pipeline and tools/evaluate.py treat sim and real
frames the same way.

    pip install pyrealsense2 opencv-python
    python vision/serve.py                         # in one terminal
    python vision/realsense_client.py              # live: SPACE = ask AisleLens, R = record on/off, Q = quit
    python vision/realsense_client.py --record data/store/run1   # save every 10th frame while walking

NOT YET TESTED ON HARDWARE: written against the pyrealsense2 API without a camera attached.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image


def png16(depth: np.ndarray) -> bytes:
    b = io.BytesIO()
    Image.fromarray(depth.astype(np.uint16)).save(b, format="PNG")
    return b.getvalue()


def jpg(rgb: np.ndarray) -> bytes:
    b = io.BytesIO()
    Image.fromarray(rgb).save(b, format="JPEG", quality=90)
    return b.getvalue()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:8770")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--record", default=None, help="folder to save rgb/, depth/, labels/ (camera intrinsics only)")
    ap.add_argument("--every", type=int, default=10, help="when recording, keep every Nth frame")
    a = ap.parse_args()

    import cv2
    import pyrealsense2 as rs

    pipe, cfg = rs.pipeline(), rs.config()
    cfg.enable_stream(rs.stream.color, a.width, a.height, rs.format.rgb8, a.fps)
    cfg.enable_stream(rs.stream.depth, a.width if a.width <= 1280 else 1280, a.height if a.height <= 720 else 720, rs.format.z16, a.fps)
    prof = pipe.start(cfg)
    dev = prof.get_device()
    scale = dev.first_depth_sensor().get_depth_scale()  # metres per unit; usually 0.001
    align = rs.align(rs.stream.color)
    intr = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    cam = {"model": dev.get_info(rs.camera_info.name), "width": intr.width, "height": intr.height,
           "fx": intr.fx, "fy": intr.fy, "cx": intr.ppx, "cy": intr.ppy, "depth_units": "uint16 millimetres, 0 = no data"}
    print("Camera:", json.dumps(cam))

    out = Path(a.record) if a.record else None
    recording = out is not None
    if out:
        for d in ("rgb", "depth", "labels"):
            (out / d).mkdir(parents=True, exist_ok=True)
    n = saved = 0
    try:
        while True:
            fs = align.process(pipe.wait_for_frames())
            c, d = fs.get_color_frame(), fs.get_depth_frame()
            if not c or not d:
                continue
            rgb = np.asanyarray(c.get_data())
            depth = np.asanyarray(d.get_data()).astype(np.float32) * scale * 1000.0  # -> millimetres
            depth = np.clip(depth, 0, 65535).astype(np.uint16)
            n += 1
            if recording and out and n % a.every == 0:
                saved += 1
                fid = f"{saved:05d}"
                (out / "rgb" / f"{fid}.jpg").write_bytes(jpg(rgb))
                (out / "depth" / f"{fid}.png").write_bytes(png16(depth))
                (out / "labels" / f"{fid}.json").write_text(json.dumps({"frame": fid, "rgb_file": f"rgb/{fid}.jpg", "camera": cam,
                                                                         "timestamp": time.time(), "objects": []}))
            view = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.putText(view, f"SPACE ask  R record({'on' if recording else 'off'}, {saved})  Q quit", (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("AisleLens RealSense", view)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            if k == ord("r") and out:
                recording = not recording
            if k == ord(" "):
                body = json.dumps({"rgb": base64.b64encode(jpg(rgb)).decode(), "depth": base64.b64encode(png16(depth)).decode()}).encode()
                req = urllib.request.Request(a.server + "/analyze", body, {"Content-Type": "application/json"})
                try:
                    res = json.loads(urllib.request.urlopen(req, timeout=30).read())
                    v, it = res.get("verdict") or {}, res.get("item") or {}
                    print(f"[{res.get('mode')}] {v.get('headline', 'no product found')} | {it.get('title', '')} "
                          f"{it.get('method', '')} {it.get('confidence', '')} | {res['timings_ms']['total_ms']:.0f} ms")
                except Exception as e:
                    print("server error:", e)
    finally:
        pipe.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
