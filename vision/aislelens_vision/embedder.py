"""Exact-SKU identification by image embedding search.

A DINOv2 backbone (Apache-2.0) turns a product crop into a vector. Every SKU has
a few reference photos (the simulator's gallery renders; real catalogue photos
later). A crop is identified by its nearest reference vectors. New products
only need reference photos: no retraining.

An optional small projection head, trained on labelled crops (see
tools/train_embedder.py), sharpens the space for look-alike packs. It is a
single linear layer on frozen DINOv2 features, so it trains in seconds on a CPU.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

DEFAULT_MODEL = "facebook/dinov2-small"


def square_pad(img: Image.Image, fill=(216, 216, 214)) -> Image.Image:
    w, h = img.size
    s = max(w, h)
    out = Image.new("RGB", (s, s), fill)
    out.paste(img.convert("RGB"), ((s - w) // 2, (s - h) // 2))
    return out


class Embedder:
    def __init__(self, model_name: str = DEFAULT_MODEL, head_path: str | Path | None = None,
                 device: str | None = None, size: int = 224):
        from transformers import AutoModel
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        self.size = size
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.head = None
        if head_path and Path(head_path).exists():
            sd = torch.load(head_path, map_location="cpu")
            self.head = torch.nn.Linear(sd["weight"].shape[1], sd["weight"].shape[0], bias=False)
            self.head.load_state_dict(sd)
            self.head.eval()

    def _tensor(self, crops: list[Image.Image]) -> torch.Tensor:
        arr = [np.asarray(square_pad(c).resize((self.size, self.size), Image.BICUBIC), dtype=np.float32) / 255.0 for c in crops]
        x = torch.from_numpy(np.stack(arr)).permute(0, 3, 1, 2)
        return ((x - self.mean) / self.std).to(self.device)

    @torch.no_grad()
    def features(self, crops: list[Image.Image], batch: int = 32) -> np.ndarray:
        """Raw backbone features: CLS token concatenated with mean patch token."""
        out = []
        for i in range(0, len(crops), batch):
            h = self.model(pixel_values=self._tensor(crops[i:i + batch])).last_hidden_state
            out.append(torch.cat([h[:, 0], h[:, 1:].mean(1)], dim=1).cpu())
        return torch.cat(out).numpy() if out else np.zeros((0, 2 * self.model.config.hidden_size), np.float32)

    def project(self, feats: np.ndarray) -> np.ndarray:
        f = torch.nn.functional.normalize(torch.from_numpy(np.asarray(feats, dtype=np.float32)), dim=1)
        if self.head is not None:  # trained on L2-normalised features (tools/train_embedder.py)
            with torch.no_grad():
                f = self.head(f)
        return torch.nn.functional.normalize(f, dim=1).numpy()

    def embed(self, crops: list[Image.Image]) -> np.ndarray:
        return self.project(self.features(crops))


class Gallery:
    """Reference embeddings per SKU; nearest-neighbour search with per-SKU max."""

    def __init__(self, skus: list[str], feats: np.ndarray):
        self.skus = np.array(skus)
        self.feats = feats  # raw backbone features, projected at query time
        self.emb = None

    @classmethod
    def build(cls, embedder: Embedder, gallery_dir: str | Path, augment: bool = True) -> "Gallery":
        gallery_dir = Path(gallery_dir)
        skus, crops = [], []
        for d in sorted(p for p in gallery_dir.iterdir() if p.is_dir()):
            for f in sorted(d.glob("*.jpg")):
                im = Image.open(f).convert("RGB")
                views = [im]
                if augment:  # tight crop and a darker copy: shelf crops are tighter and dimmer than studio renders
                    w, h = im.size
                    views.append(im.crop((int(w * 0.12), int(h * 0.12), int(w * 0.88), int(h * 0.88))))
                    views.append(Image.eval(im, lambda v: int(v * 0.7)))
                for v in views:
                    skus.append(d.name)
                    crops.append(v)
        return cls(skus, embedder.features(crops))

    def save(self, path: str | Path):
        np.savez_compressed(path, skus=self.skus, feats=self.feats)

    @classmethod
    def load(cls, path: str | Path) -> "Gallery":
        z = np.load(path, allow_pickle=False)
        return cls(list(z["skus"]), z["feats"])

    def ready(self, embedder: Embedder) -> "Gallery":
        self.emb = embedder.project(self.feats)
        self.uniq = sorted(set(self.skus.tolist()))
        self.idx = {s: np.where(self.skus == s)[0] for s in self.uniq}
        return self

    def search(self, q: np.ndarray, k: int = 5) -> list[list[tuple[str, float]]]:
        """For each query embedding, the top-k SKUs as (sku, cosine similarity)."""
        sims = q @ self.emb.T
        per = np.stack([sims[:, self.idx[s]].max(1) for s in self.uniq], axis=1)
        order = np.argsort(-per, axis=1)[:, :k]
        return [[(self.uniq[j], float(per[i, j])) for j in row] for i, row in enumerate(order)]


def save_meta(path: Path, meta: dict):
    path.write_text(json.dumps(meta, indent=1), encoding="utf-8")
