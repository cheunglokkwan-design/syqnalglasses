"""Fine-tune RF-DETR (product, price_tag, held_item) on the simulator COCO export.

Starts from RF-DETR's COCO-pretrained weights (downloaded on first run).

CPU (works anywhere, slow):   python vision/tools/train_detector.py --epochs 4
GPU (CUDA PyTorch installed): python vision/tools/train_detector.py --size small --epochs 30 --batch 8

The best checkpoint is written to <out>/checkpoint_best_total.pth (RF-DETR's naming).
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/coco")
    ap.add_argument("--out", default="runs/detector")
    ap.add_argument("--size", default="nano", choices=["nano", "small", "medium", "base"])
    ap.add_argument("--resolution", type=int, default=None, help="square input size, multiple of 32 (nano default 384)")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--device", default=None, help="cpu or cuda (default: auto)")
    a = ap.parse_args()

    import torch
    import rfdetr
    cls = {"nano": rfdetr.RFDETRNano, "small": rfdetr.RFDETRSmall, "medium": rfdetr.RFDETRMedium, "base": rfdetr.RFDETRBase}[a.size]
    kw = {"resolution": a.resolution} if a.resolution else {}
    model = cls(**kw)
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training RF-DETR {a.size} on {device}; data {a.data}; out {a.out}")
    Path(a.out).mkdir(parents=True, exist_ok=True)
    args = dict(dataset_dir=a.data, output_dir=a.out, epochs=a.epochs, batch_size=a.batch, grad_accum_steps=a.accum,
                lr=a.lr, num_workers=a.workers, device=device, tensorboard=False, progress_bar="tqdm",
                checkpoint_interval=1, multi_scale=False, run_test=False)
    # keep only the options this RF-DETR version accepts
    try:
        from rfdetr.config import TrainConfig
        allowed = set(TrainConfig.model_fields) | {"device"}
        args = {k: v for k, v in args.items() if k in allowed}
    except Exception:
        pass
    model.train(**args)


if __name__ == "__main__":
    main()
