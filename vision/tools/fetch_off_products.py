"""Look up real products on Open Food Facts: barcode, name, size, allergens, nutrition, photos.

For each entry in vision/data/real_products.json, searches Open Food Facts (preferring
products sold in Malaysia that have a front photo), downloads up to --max-images photos
(front, back/ingredients, nutrition, other) into <out>/<id>/, and writes <out>/off_matches.json.

Open Food Facts photos are CC BY-SA and the database is ODbL: keep the attribution
("Open Food Facts contributors", the product page URL) with anything built from them.

Usage: python vision/tools/fetch_off_products.py --out data/real/products
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

UA = {"User-Agent": "SyqnalGlasses-prototype/0.1"}
FIELDS = "code,product_name,brands,quantity,countries_tags,image_front_url,images,allergens_tags,nutriments,url"


def get(url: str, timeout: int = 40):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _sal(q: str) -> list[dict]:
    url = "https://search.openfoodfacts.org/search?" + urllib.parse.urlencode(
        {"q": q, "page_size": 12, "fields": "code,product_name,brands,quantity,countries_tags"})
    hits = json.loads(get(url)).get("hits", [])
    for h in hits:  # brands come back as a list here
        if isinstance(h.get("brands"), list):
            h["brands"] = ", ".join(h["brands"])
    return hits


def search(q: str) -> list[dict]:
    """Search (Malaysian listings first), then fetch full records for the top hits."""
    hits = _sal(f'{q} countries_tags:"en:malaysia"') or _sal(q)
    out = []
    for h in hits[:4]:
        try:
            r = json.loads(get(f"https://world.openfoodfacts.org/api/v2/product/{h['code']}.json?fields={FIELDS}"))
            if r.get("product"):
                out.append({**r["product"], "code": h["code"]})
        except Exception:
            out.append(h)
        time.sleep(1)
    return out


def score(p: dict, brand: str) -> float:
    s = 0.0
    if "en:malaysia" in (p.get("countries_tags") or []):
        s += 3
    if p.get("image_front_url"):
        s += 2
    if brand.lower().replace("é", "e").split()[0] in (p.get("brands") or "").lower().replace("é", "e"):
        s += 2
    s += min(len(p.get("images") or {}), 10) * 0.1
    return s


def image_urls(p: dict, n: int) -> list[str]:
    """Selected photos first (front, ingredients, nutrition, packaging), then raw uploads."""
    code = p["code"]
    folder = "/".join([code[0:3], code[3:6], code[6:9], code[9:]]) if len(code) > 8 else code
    base = f"https://images.openfoodfacts.org/images/products/{folder}"
    imgs = p.get("images") or {}
    out = []
    for key, meta in imgs.items():
        if not key[0].isdigit() and isinstance(meta, dict) and "rev" in meta:  # e.g. front_en, ingredients_ms
            out.append(f"{base}/{key}.{meta['rev']}.400.jpg")
    for key in sorted((k for k in imgs if k.isdigit()), key=int):  # raw uploads: different photos of the pack
        out.append(f"{base}/{key}.400.jpg")
    seen, uniq = set(), []
    for u in out:
        if u not in seen:
            seen.add(u)
            uniq.append(u)
    return uniq[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--products", default="vision/data/real_products.json")
    ap.add_argument("--out", default="data/real/products")
    ap.add_argument("--max-images", type=int, default=6)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    items = json.loads(Path(a.products).read_text(encoding="utf-8"))
    res = []
    for it in items:
        try:
            cands = search(it["query"])
        except Exception as e:
            print(it["id"], "search failed:", e)
            res.append({**it, "off": None})
            time.sleep(3)
            continue
        cands = [c for c in cands if c.get("code")]
        best = max(cands, key=lambda c: score(c, it["brand"]), default=None)
        rec = {**it, "off": None}
        if best:
            n = best.get("nutriments") or {}
            rec["off"] = {"code": best["code"], "product_name": best.get("product_name"), "brands": best.get("brands"),
                          "quantity": best.get("quantity"), "malaysia": "en:malaysia" in (best.get("countries_tags") or []),
                          "allergens": [x.split(":", 1)[-1] for x in best.get("allergens_tags") or []],
                          "sugars_100g": n.get("sugars_100g"), "salt_100g": n.get("salt_100g"),
                          "energy_kcal_100g": n.get("energy-kcal_100g"),
                          "url": f"https://world.openfoodfacts.org/product/{best['code']}",
                          "candidates": len(cands), "images": []}
            d = out / it["id"]
            d.mkdir(exist_ok=True)
            for k, u in enumerate(image_urls(best, a.max_images)):
                try:
                    (d / f"off_{k}.jpg").write_bytes(get(u))
                    rec["off"]["images"].append({"file": f"{it['id']}/off_{k}.jpg", "src": u})
                except Exception:
                    pass
        res.append(rec)
        o = rec["off"]
        print(it["id"], it["brand"], it["name"], "->", (o["product_name"], o["brands"], o["code"], "MY" if o["malaysia"] else "", len(o["images"]), "photos") if o else "not found")
        time.sleep(3)  # stay well under Open Food Facts' rate limits
    (out / "off_matches.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print("saved", out / "off_matches.json")


if __name__ == "__main__":
    main()
