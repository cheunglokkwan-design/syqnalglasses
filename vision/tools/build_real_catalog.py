"""Build the real-product catalogue and photo gallery for the first 20 products.

Inputs:  vision/data/real_products.json   the 20 products we chose
         vision/data/real_prices.json     Jaya Grocer (here) and Lotus's (elsewhere) prices
         data/real/products/off_matches.json + photos  from tools/fetch_off_products.py
         data/real/photos/RP-xx/*.jpg     optional: your own phone photos (always trusted)
Outputs: vision/data/real_catalog.json    the catalogue serve.py loads with --catalog
         data/real/gallery/RP-xx/*.jpg    reference photos for the embedding gallery
         runs/embedder_real/gallery.npz   gallery features (DINOv2, no head)

Open Food Facts matches are only used where we checked them by hand (VERIFIED below);
the others keep their prices but need your photos (barcode visible) before they can be
recognised. Data from Open Food Facts is ODbL / CC BY-SA: keep the source links.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "vision"))

# checked by hand on 16 Nov 2026: right brand, product and Malaysian (or matching) pack
VERIFIED = {"RP-01", "RP-02", "RP-03", "RP-04", "RP-06", "RP-07", "RP-08", "RP-10", "RP-17", "RP-18", "RP-20"}

CATEGORY = {"RP-01": "malted drinks", "RP-02": "instant noodles", "RP-03": "milk", "RP-04": "bread", "RP-05": "coffee",
            "RP-06": "biscuits", "RP-07": "snacks", "RP-08": "soy and plant drinks", "RP-09": "isotonic drinks",
            "RP-10": "canned fish", "RP-11": "spices", "RP-12": "biscuits", "RP-13": "snacks", "RP-14": "chocolate",
            "RP-15": "milk", "RP-16": "cereal", "RP-17": "cereal", "RP-18": "sauces", "RP-19": "cooking oil",
            "RP-20": "cultured milk"}

# Allergens usually declared on these packs; shown with "check the pack" until read from the label itself.
TYPICAL_ALLERGENS = {"RP-01": ["milk", "gluten"], "RP-02": ["gluten"], "RP-03": ["milk"], "RP-04": ["gluten"],
                     "RP-06": ["peanuts", "gluten", "milk"], "RP-07": ["gluten"], "RP-08": ["soy"], "RP-10": ["fish"],
                     "RP-12": ["gluten"], "RP-13": [], "RP-14": ["milk", "gluten", "soy"], "RP-15": ["milk"],
                     "RP-16": ["gluten", "milk"], "RP-17": ["gluten"], "RP-18": ["molluscs", "gluten"], "RP-20": ["milk"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--off", default=str(ROOT / "data/real/products"))
    ap.add_argument("--photos", default=str(ROOT / "data/real/photos"))
    ap.add_argument("--gallery", default=str(ROOT / "data/real/gallery"))
    ap.add_argument("--out", default=str(ROOT / "vision/data/real_catalog.json"))
    ap.add_argument("--features", default=str(ROOT / "runs/embedder_real/gallery.npz"))
    ap.add_argument("--no-features", action="store_true")
    a = ap.parse_args()

    items = json.loads((ROOT / "vision/data/real_products.json").read_text(encoding="utf-8"))
    prices = json.loads((ROOT / "vision/data/real_prices.json").read_text(encoding="utf-8"))
    off_path = Path(a.off) / "off_matches.json"
    off = {r["id"]: r.get("off") for r in json.loads(off_path.read_text(encoding="utf-8"))} if off_path.exists() else {}
    gal = Path(a.gallery)
    if gal.exists():
        shutil.rmtree(gal)
    rows = []
    for k, it in enumerate(items, start=1):
        pid = it["id"]
        pr = prices["prices"].get(pid, {})
        jaya, lot = pr.get("jaya"), pr.get("lotuss")
        here = (jaya or {}).get("promo") or (jaya or {}).get("price")
        if here is None:
            continue
        o = off.get(pid) if pid in VERIFIED else None
        photos = []
        own = Path(a.photos) / pid
        if own.exists():
            photos += sorted(p for p in own.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        if o:
            photos += [Path(a.off) / im["file"] for im in o.get("images", []) if (Path(a.off) / im["file"]).exists()]
        if photos:
            (gal / pid).mkdir(parents=True, exist_ok=True)
            for i, p in enumerate(photos):
                shutil.copy2(p, gal / pid / f"{i:02d}{p.suffix.lower()}")
        allergens = sorted(set((o or {}).get("allergens") or []) | set(TYPICAL_ALLERGENS.get(pid, [])))
        sugars = (o or {}).get("sugars_100g")
        elsewhere = []
        if lot:
            elsewhere.append({"store": "Lotus's", "kind": "store", "price_myr": lot.get("promo") or lot["price"],
                              "age_days": 0, "delivery_myr": 0.0, "free_over_myr": None,
                              "listing": lot["name"], "promo": lot.get("promo") is not None})
        rows.append({
            "class_id": k, "sku": pid, "brand": it["brand"], "name": it["name"], "category": CATEGORY.get(pid, "grocery"),
            "price_myr": float(here), "price_note": "Jaya Grocer online" + (" promo" if (jaya or {}).get("promo") else ""),
            "barcode_ean13": (o or {}).get("code") or "", "size": it["size"],
            "allergens": allergens, "allergens_source": "Open Food Facts + usual for this product: check the pack",
            "nutrition": {"sugars_g": sugars} if sugars is not None else None,
            "prices_elsewhere": elsewhere,
            "image": f"{pid}/00.jpg" if photos else None,
            "photos": len(photos),
            "source": {"prices": "vision/data/real_prices.json (16 Nov 2026)",
                       "product": (o or {}).get("url"), "licence": "Open Food Facts: ODbL data, CC BY-SA photos" if o else None},
        })
    cat = {"here_store": "Jaya Grocer", "checked": prices.get("checked"), "products": rows}
    Path(a.out).write_text(json.dumps(cat, indent=1, ensure_ascii=False), encoding="utf-8")
    with_photos = sum(1 for r in rows if r["photos"])
    print(f"catalogue: {len(rows)} products, {with_photos} with photos, {sum(1 for r in rows if r['barcode_ean13'])} with barcodes -> {a.out}")
    need = [f"{r['sku']} {r['brand']} {r['name']}" for r in rows if not r["photos"]]
    if need:
        print("need your photos:", "; ".join(need))
    if not a.no_features and with_photos:
        from aislelens_vision.embedder import Embedder, Gallery
        emb = Embedder()
        g = Gallery.build(emb, gal)
        Path(a.features).parent.mkdir(parents=True, exist_ok=True)
        g.save(a.features)
        print(f"gallery features: {len(g.skus)} photos of {len(set(g.skus.tolist()))} products -> {a.features}")


if __name__ == "__main__":
    main()
