"""Turn an identified product and its shelf price into the short verdict the glasses show.

The verdict combines: price here vs other stores (with online delivery fees),
an honest promo check, allergen and sugar alerts against the shopper's profile,
and a running basket total.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .catalog import Catalog, Product, is_promo

HIGH_SUGAR_G = 22.5  # g per 100 g; the usual "high" cut-off on traffic-light labels
STALE_DAYS = 3


@dataclass
class Profile:
    allergies: list[str] = field(default_factory=list)   # e.g. ["peanuts", "milk"]
    avoid_high_sugar: bool = False
    budget_myr: float | None = None


def fmt(v: float) -> str:
    return f"RM {v:.2f}"


def compare_prices(p: Product, price_here: float, basket_total: float = 0.0) -> dict:
    others = []
    for o in p.other_store_prices():
        all_in = o["price_myr"]
        note = ""
        if o["kind"] == "online":
            free = o["free_over_myr"] is not None and basket_total + o["price_myr"] >= o["free_over_myr"]
            if not free:
                all_in = round(all_in + o["delivery_myr"], 2)
                note = f"incl. {fmt(o['delivery_myr'])} delivery"
            else:
                note = "free delivery on this basket"
        others.append({**o, "all_in_myr": all_in, "note": note, "stale": o["age_days"] > STALE_DAYS})
    # decide on the all-in price (online delivery included); keep the shelf price list sorted too
    others.sort(key=lambda o: (o["all_in_myr"], o["price_myr"]))
    cheapest = others[0] if others else None
    lowest_sticker = min(others, key=lambda o: o["price_myr"]) if others else None
    return {"here_myr": price_here, "others": others, "cheapest": cheapest, "lowest_sticker": lowest_sticker,
            "saving_myr": round(price_here - cheapest["all_in_myr"], 2) if cheapest else 0.0}


def verdict(p: Product, price_here: float, price_source: str, profile: Profile | None = None,
            promo_tag: bool | None = None, basket_total: float = 0.0) -> dict:
    """Return {headline, lines, alerts, compare} for one product.

    headline is the single glanceable line for the lens; lines/alerts are detail for the phone.
    """
    profile = profile or Profile()
    cmp = compare_prices(p, price_here, basket_total)
    alerts, lines = [], []

    hit = sorted(set(a.lower() for a in profile.allergies) & set(p.allergens))
    if hit:
        alerts.append(f"Contains {', '.join(hit)}")
    nut = p.nutrition_per_100g()
    if nut and nut["sugars_g"] >= HIGH_SUGAR_G:
        msg = f"High sugar: {nut['sugars_g']} g per 100 g"
        (alerts if profile.avoid_high_sugar else lines).append(msg)

    promo = is_promo(p.class_id) if promo_tag is None else promo_tag
    if promo and abs(price_here - p.price_myr) < 0.005:
        lines.append(f"PROMO tag, but {fmt(price_here)} is the usual price here")

    c = cmp["cheapest"]
    if c and cmp["saving_myr"] >= 0.10:
        where = c["store"] + (f" ({c['note']})" if c["note"] else "")
        price_line = f"{fmt(cmp['saving_myr'])} cheaper at {where}"
    else:
        price_line = "Good buy: cheapest nearby"
        ls = cmp["lowest_sticker"]
        if ls and ls["kind"] == "online" and ls["price_myr"] < price_here - 0.05:
            lines.append(f"{ls['store']} lists {fmt(ls['price_myr'])}, but {fmt(ls['all_in_myr'])} with delivery")
    if c and c["stale"]:
        lines.append(f"{c['store']} price is {c['age_days']} days old")

    headline = alerts[0] if alerts else price_line
    if alerts:
        lines.insert(0, price_line)
    lines.insert(0, f"{p.title}{' ' + p.size if p.size else ''}: {fmt(price_here)} ({price_source})")
    return {"headline": headline, "lines": lines, "alerts": alerts, "compare": cmp}


class Basket:
    """Running total of what is in the trolley, with a budget alert."""

    def __init__(self, catalog: Catalog, budget_myr: float | None = None):
        self.catalog, self.budget = catalog, budget_myr
        self.items: list[dict] = []

    def add(self, p: Product, price_myr: float, qty: int = 1, source: str = "") -> dict:
        self.items.append({"sku": p.sku, "title": p.title, "price_myr": price_myr, "qty": qty, "source": source})
        return self.summary()

    def remove(self, sku: str) -> dict:
        for i in range(len(self.items) - 1, -1, -1):
            if self.items[i]["sku"] == sku:
                del self.items[i]
                break
        return self.summary()

    @property
    def total(self) -> float:
        return round(sum(i["price_myr"] * i["qty"] for i in self.items), 2)

    def summary(self) -> dict:
        s = {"count": sum(i["qty"] for i in self.items), "total_myr": self.total, "items": list(self.items)}
        if self.budget is not None:
            s["budget_myr"] = self.budget
            s["over_budget"] = self.total > self.budget
        return s
