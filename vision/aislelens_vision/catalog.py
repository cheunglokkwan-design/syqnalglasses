"""Product catalogue and price data for the AisleLens simulator store.

Everything here is fictional and matches prototypes/store-sim-3d/index.html:
the 54 SKUs, their EAN-13 barcodes, shelf prices in RM, and the nutrition
numbers printed on the back of each pack. Prices at "other stores" are
generated deterministically so the verdict logic can be tested end to end.
Swap this module for the real data sources (Open Food Facts, PriceCatcher,
Open Prices) once the pipeline runs on real products.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Fictional comparison stores. "online" adds a delivery fee unless the basket
# is over the free-delivery threshold.
OTHER_STORES = [
    {"id": "mutiara", "name": "Mutiara Mart", "kind": "store"},
    {"id": "rakyat", "name": "Kedai Rakyat", "kind": "store"},
    {"id": "seri", "name": "Pasar Raya Seri", "kind": "store"},
    {"id": "belanja", "name": "BelanjaNow online", "kind": "online", "delivery_myr": 5.0, "free_over_myr": 80.0},
]
HERE_STORE = "AisleLens Market"

FOOD_CATS = {"breakfast", "snacks", "drinks", "cooking", "dairy", "produce"}

# Allergens for the fictional products, keyed by SKU (EU/Malaysia major allergen groups).
ALLERGENS = {
    "AL-007": ["gluten"], "AL-008": ["gluten"], "AL-009": ["gluten", "milk"], "AL-010": ["egg"],
    "AL-011": ["peanuts"], "AL-013": ["tree nuts", "gluten"], "AL-014": ["milk", "gluten"],
    "AL-016": ["milk"], "AL-017": ["crustaceans"], "AL-018": ["gluten"], "AL-019": ["gluten", "milk", "egg"],
    "AL-020": ["gluten", "soy"], "AL-021": ["gluten", "soy"], "AL-022": ["gluten", "milk", "soy"],
    "AL-028": ["milk"], "AL-033": ["soy", "gluten"], "AL-035": ["fish"],
    "AL-045": ["milk"], "AL-046": ["milk"], "AL-047": ["milk"], "AL-048": ["milk"], "AL-049": ["milk"],
    "AL-050": ["egg"],
}

# Products with a PROMO shelf tag in the simulator (sku id % 7 == 0). The promo
# price equals the usual price, which the verdict should call out.
def is_promo(class_id: int) -> bool:
    return class_id % 7 == 0


def _frac(key: str) -> float:
    return int(hashlib.sha1(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def round_sen(v: float) -> float:
    """Round to the nearest 5 sen, as Malaysian tills do."""
    return round(round(v * 20) / 20, 2)


@dataclass
class Product:
    class_id: int
    sku: str
    brand: str
    name: str
    category: str
    price_myr: float
    barcode_ean13: str
    size: str = ""
    allergens: list[str] = field(default_factory=list)
    # real products: measured data instead of the simulator's formulas
    nutrition: dict | None = None          # per 100 g, e.g. {"sugars_g": 12.0}
    prices_elsewhere: list | None = None   # [{"store", "kind", "price_myr", "age_days", ...}]
    image: str | None = None               # reference photo path, relative to the catalogue file
    source: dict | None = None             # where the facts came from (attribution)

    @property
    def title(self) -> str:
        return f"{self.brand} {self.name}"

    @property
    def is_food(self) -> bool:
        return self.category in FOOD_CATS

    def nutrition_per_100g(self) -> dict | None:
        """The numbers printed on the simulator's back panel (same formulas as drawBack)."""
        if self.nutrition is not None:
            return self.nutrition
        if self.source is not None or not self.is_food or self.category == "produce":
            return None
        i = self.class_id
        return {
            "energy_kcal": (180 + i * 7) % 480,
            "protein_g": round(i * 1.3 % 12, 1),
            "fat_g": round(i * 2.1 % 25, 1),
            "carbohydrate_g": round(i * 4.7 % 70, 1),
            "sugars_g": round(i * 3.1 % 30, 1),
            "sodium_mg": i * 37 % 600,
        }

    def other_store_prices(self) -> list[dict]:
        """Real prices when the catalogue has them; otherwise fictional simulator prices."""
        if self.prices_elsewhere is not None:
            return [dict(o) for o in self.prices_elsewhere]
        out = []
        for st in OTHER_STORES:
            key = f"{self.sku}:{st['id']}"
            if _frac(key + ":stock") < 0.1:
                continue  # not sold there
            factor = 0.84 + 0.28 * _frac(key)
            out.append({
                "store": st["name"], "kind": st["kind"],
                "price_myr": round_sen(self.price_myr * factor),
                "age_days": int(_frac(key + ":age") * 9),
                "delivery_myr": st.get("delivery_myr", 0.0),
                "free_over_myr": st.get("free_over_myr"),
            })
        return out


class Catalog:
    def __init__(self, products: list[Product], here_store: str = HERE_STORE, root: Path | None = None):
        self.products = products
        self.here_store = here_store
        self.root = root
        self.by_sku = {p.sku: p for p in products}
        self.by_barcode = {p.barcode_ean13: p for p in products}
        self.by_id = {p.class_id: p for p in products}

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Catalog":
        path = Path(path) if path else DATA_DIR / "catalog.json"
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        here = HERE_STORE
        if isinstance(data, dict):  # real catalogue: {"here_store": ..., "products": [...]}
            here, rows = data.get("here_store", HERE_STORE), data["products"]
        else:
            rows = data
        prods = [Product(class_id=r["class_id"], sku=r["sku"], brand=r["brand"], name=r["name"],
                         category=r["category"], price_myr=float(r["price_myr"]),
                         barcode_ean13=r["barcode_ean13"], size=r.get("size", ""),
                         allergens=r.get("allergens", ALLERGENS.get(r["sku"], [])),
                         nutrition=r.get("nutrition"), prices_elsewhere=r.get("prices_elsewhere"),
                         image=r.get("image"), source=r.get("source")) for r in rows]
        return cls(prods, here, Path(path).resolve().parent)

    def alternatives(self, p: Product, k: int = 3) -> list[Product]:
        """Other products in the same category, cheapest first."""
        return sorted((q for q in self.products if q.category == p.category and q.sku != p.sku),
                      key=lambda q: q.price_myr)[:k]

    def __len__(self) -> int:
        return len(self.products)

    def match_name(self, text: str) -> Product | None:
        """Best catalogue match for text read off a shelf tag (brand + name line)."""
        import difflib
        text = " ".join(text.lower().split())
        if len(text) < 4:
            return None
        best, best_r = None, 0.0
        for p in self.products:
            r = difflib.SequenceMatcher(None, text, p.title.lower()[: max(len(text), 8)]).ratio()
            if r > best_r:
                best, best_r = p, r
        return best if best_r >= 0.6 else None
