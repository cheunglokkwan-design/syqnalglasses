# Smart Glasses Shopping Assistant — Plan

Nov 16, 2026 · @Antony Real Betis

_Exported from the [live doc](https://claude.ai/artifact/Lvxdy5VXsGgBikzhzHmyoF) on 16 Nov 2026._

## Vision and core flow

Look at a product, get a verdict in under two seconds: is this a good buy here, and is it cheaper elsewhere? The glasses do the seeing and the short answer; the phone does the heavy lifting and the detail.

The core loop has four steps:

1. **Scan**: the camera reads a barcode (EAN-13) or recognises the pack. Fallback is voice ("Hey, Meiji milk 2 litres").
2. **Identify**: match to a product record (name, size, brand, nutrition, allergens).
3. **Compare**: unit price here vs other stores (FairPrice, Cold Storage, Sheng Siong, Giant, online) and vs alternatives on the same shelf.
4. **Decide**: a one-line verdict on the lens ("Good buy", "S$0.80 cheaper at Sheng Siong", "Contains peanuts"), with optional add-to-list or add-to-cart.

Defaults used in this plan: Singapore, prices in SGD, and a platform-neutral design so we can pick the glasses later.

## Feature set

The MVP is scan, compare and a single verdict; everything else waits until that loop feels instant. Phases refer to the roadmap below.

| Feature | What it does | Phase |
| --- | --- | --- |
| Barcode scan | Read EAN-13/UPC from the camera, hands-free | MVP |
| Product card on lens | Name, size, price, unit price (S$/100g or /L) | MVP |
| Cross-store price compare | Same item at 4 to 6 Singapore stores, cheapest highlighted | MVP |
| Verdict line | One short, glanceable recommendation | MVP |
| Shelf alternatives | Cheaper or healthier substitutes on the same shelf | MVP |
| Shopping list and running basket total | Tick items as scanned, live total vs budget | MVP |
| Weighted scoring | Ranks choices by your weights: price, health, brand, eco | v1 |
| Nutrition and allergen alerts | Nutri-Grade (A to D), sugar, Healthier Choice, allergen flags | v1 |
| Voice commands | "Compare", "add to list", "what's healthier?" | v1 |
| Price history | Is this "promo" actually lower than usual? | v1 |
| Visual recognition | Identify loose produce and packs without a visible barcode | v2 |
| Ripeness and freshness check | Look at fruit and veg: flag visible mould, bruises or wilting, and show a ripeness stage for colour-changing fruit (banana, mango, tomato). Says when the camera can't tell, e.g. avocado. Details in the vision model report. | v2 |
| Basket optimiser | Split the list across stores to minimise total cost incl. travel | v2 |
| Online checkout hand-off | Push the basket to an online store's cart | v2 |

## Jaya Grocer price check and basket total

When you scan an item at Jaya Grocer, the glasses show its shelf price beside the same item at other Malaysian stores and online, and keep a running total in RM of what is in your basket. We assemble this from existing open data and open-source parts, not from scratch. Jaya Grocer is a Malaysian chain, so this feature works in MYR and Malaysia, alongside the Singapore defaults above.

**How a scan flows**

1. The glasses camera reads the barcode with an open-source decoder.
2. The product name, size and brand come from Open Food Facts.
3. The Jaya Grocer price comes from the shelf tag (read on-device by OCR) or from Jaya Grocer's own online store.
4. Prices for the same item come from PriceCatcher (government survey of other stores), Open Prices (crowd-sourced) and online listings.
5. The verdict on the lens reads, for example: "RM 0.90 cheaper online, RM 1.20 cheaper at Mydin." The item is added to the basket and the total updates.

**What we reuse**

| Need | Open source or open data | Notes |
| --- | --- | --- |
| Barcode reading | [ZXing-C++](https://github.com/zxing-cpp/zxing-cpp/) (Apache-2.0) | Reads EAN-13; bindings for Android, iOS, WebAssembly and Python |
| Product details | [Open Food Facts](https://openfoodfacts.github.io/openfoodfacts-server/api/tutorials/scanning-barcodes) API | Name, size, nutrition by barcode; free, and we add missing Malaysian items back |
| Prices at other Malaysian stores | [PriceCatcher](https://open.dosm.gov.my/data-catalogue/pricecatcher) (DOSM / KPDN, CC BY 4.0) | Daily price records with store (premise) and item lookup tables; CSV or Parquet download, no API. Items use survey codes, so we match by name and size |
| Crowd-sourced shelf and receipt prices | [Open Prices](https://openfoodfacts.github.io/open-prices/guides/data/) (ODbL) | Prices tied to barcode and store; API plus daily exports. Our users' scans feed it, which fills gaps such as Jaya Grocer |
| Jaya Grocer's own price | Shelf-tag OCR on the phone, or Jaya Grocer's online store ([jayagrocer.com](https://says.com/my/lifestyle/you-can-now-get-your-jaya-grocer-groceries-online) and app) | Online and in-store prices may differ, so a shelf-tag reading wins when available |
| Online sellers | Supermarket web stores and marketplaces, read with an open-source scraper such as Playwright | Check each site's terms first, or prefer official affiliate or partner feeds where they exist |
| Running basket total | Our own small module on the phone, using prices above | Handles quantity, multi-buy promos, items sold by weight (from the scale label) and a budget alert |
| Checking the till | Open-source OCR (Tesseract or PaddleOCR) on the receipt photo | Compares the receipt against the basket and flags mismatches; receipts can also be shared to Open Prices |

**Limits to plan for**

- PriceCatcher covers a fixed list of everyday goods, and its store list needs checking for Jaya Grocer (its premise lookup table shows which shops are surveyed). [KPDN is adding retailer-shared prices](https://thesun.my/malaysia-news/kpdn-increases-pricecatcher-functionality-through-collaboration-with-mydin-redtick-EI13283326), starting with Mydin and Redtick.
- Online prices exclude delivery fees and minimum orders, so the basket view shows the all-in online cost next to the in-store total.
- Each price shows its source and age; anything older than a few days is flagged.

## Ideas to add later

- **Promo decoder**: turn "2 for S$5" or "30% off 2nd item" into an effective unit price, so deals are compared honestly.
- **Shrinkflation watch**: flag when pack size dropped but price did not.
- **Diet profiles**: halal, vegetarian, low-sodium, diabetic, kids; the verdict respects them.
- **Loyalty and card stacking**: factor in Linkpoints, Yuu points, credit card cashback.
- **Pantry memory**: "You bought this 4 days ago" to stop duplicates; expiry reminders.
- **Recipe mode**: scan an ingredient, get what you can make with what is already in the basket.
- **Household sharing**: a shared list that a partner can update live while you shop.
- **Accessibility**: read labels aloud for low-vision users; large-type mode.
- **Eco score**: country of origin, packaging, carbon estimate.
- **Crowd-sourced prices**: users confirm shelf prices to keep data fresh (with incentives).

## Gaps and risks

The biggest gap is price data, not the glasses: there is no open, official feed of shelf prices for Singapore supermarkets. Hardware notes below are from memory (approximate, verify before buying).

| Area | Gap or risk | Mitigation |
| --- | --- | --- |
| Price data | No public supermarket price API; scraping online stores may breach their terms; online prices can differ from in-store shelf prices | Start with simulated data; then partner feeds, store online catalogues where terms allow, CASE Price Kaki style crowd-sourcing, and receipt scanning; show "last seen" timestamps |
| Product data | Barcode-to-product mapping is incomplete for local and house brands | Open Food Facts as a base, plus our own catalogue built from scans |
| Hardware: display | Many camera glasses have no display (audio-only answers); display glasses often have no camera or limited third-party camera access | Design for both: audio-first verdict, optional lens card; phone as the fallback screen |
| Hardware: SDK access | Third-party developer access to camera and display is limited and changes fast across vendors | Keep the glasses layer thin; all logic on phone or cloud; test on one platform first |
| Latency | Camera to cloud to answer can exceed 2 s on mall Wi-Fi or weak 4G/5G indoors | Decode barcodes on-device; cache prices for the current store before entering; target under 800 ms for a cached item |
| Battery and heat | Continuous camera use drains small glasses batteries fast | Scan on demand (tap or voice), not always-on video |
| Privacy | Camera pointed at people in a store; PDPA obligations for any personal data; shopping history is sensitive | Process frames on-device, never store images, crop to the barcode region, opt-in history, clear recording indicator |
| Store policy | Some stores restrict photography or scraping | Barcode-only mode; no image upload; respectful use guidelines |
| Accuracy and trust | A wrong "cheaper elsewhere" claim destroys trust | Show source and age of every price; flag low confidence |
| Business model | Who pays: users, stores or brands? Sponsored placement would bias verdicts | Decide early; keep verdicts unsponsored and label any ads |

## Architecture

Keep the glasses thin and put the logic on the phone, so we can swap glasses vendors without a rewrite.

> **Diagram: system architecture · 3 layers.** Drawn in the [live doc](https://claude.ai/artifact/Lvxdy5VXsGgBikzhzHmyoF); not included in this Markdown export.

The phone decodes barcodes and answers from a cache it fills when you enter a store, so most scans never wait on the network. The cloud holds the catalogue and prices and runs the slower jobs such as visual matching.

## Workflow and roadmap

Build phone-first and simulation-first: the glasses come in only after the scan-to-verdict loop is fast on a phone. Week counts are a proposal for one part-time builder.

> **Diagram: roadmap · 4 phases, 3 gates (proposed timing).** Drawn in the [live doc](https://claude.ai/artifact/Lvxdy5VXsGgBikzhzHmyoF); not included in this Markdown export.

How we work each week:

1. **One data contract**: the simulator and the real app share the same product, price and store schema, so every feature is tried in the sim first.
2. **Thin vertical slices**: each week ships one end-to-end improvement (scan, lookup, verdict) rather than a layer at a time.
3. **Measure three numbers**: time to verdict, scan success rate, and price accuracy against the real shelf.
4. **Weekly store walk** from Phase 1: scan 20 items in a real store and log misses.
5. **Hardware last**: keep a glasses adapter layer so switching vendors touches one module.

## The simulation

[AisleLens Store Simulator](https://claude.ai/artifact/91yDZ8W5SmyK1NDUBbKv1t) runs in any browser and lets you walk the MVP loop end to end. All stores, brands, barcodes and prices are made up.

- **Store floor**: a fictional LionMart with 7 aisles and 31 products. Each has a valid EAN-13 barcode on Singapore's 888 prefix, nutrition per 100 g/ml, allergens and labels.
- **Glasses view**: click a product to look at it, then press Space to scan. The lens shows price, unit price, the cheapest store, promo check and the best shelf pick.
- **Phone companion**: Compare (5 stores, freshness flags, 8-week price history), Shelf picks (weighted scores), Nutrition (approximate Nutri-Grade), Basket (budget and cheapest split across stores), My weights.
- **Risk toggles**: switch to audio-only glasses, or to cloud and weak-signal pipelines to feel the latency difference (about 0.2 s vs 1 s vs 2 to 3 s, with cache fallback).
- **Edge cases built in**: bananas have no barcode (needs visual match), the kimchi barcode is not in the catalogue, the potato chip "10% off" equals its usual low, and nuts trigger an allergy alert.

The file is also saved in project files as `smart-glasses/aislelens-simulator.html`.

## Open decisions

- [ ] **Target glasses**: display glasses (lens card) or camera-only glasses (audio answers)? Decided 16 Nov 2026: stay generic and phone-first, and pick the glasses in Phase 2.
- [ ] **Region and currency**: Malaysia (MYR) is now in scope for the Jaya Grocer comparison. Is Malaysia the main market, with Singapore second, or both from the start?
- [ ] **Which real stores to start with**, and whether to begin with online catalogue prices or crowd-sourced shelf prices.
- [ ] **Who the first user is**: you only, family, or a public beta.
- [ ] **Business model**: free, subscription, or store partnerships (affects how verdicts stay neutral).

## Sources

Checked 16 Nov 2026: [PriceCatcher dataset](https://open.dosm.gov.my/data-catalogue/pricecatcher) · [PriceCatcher retailer sharing news](https://thesun.my/malaysia-news/kpdn-increases-pricecatcher-functionality-through-collaboration-with-mydin-redtick-EI13283326) · [Open Prices data](https://openfoodfacts.github.io/open-prices/guides/data/) · [Open Prices API](https://openfoodfacts.github.io/open-prices/guides/API/) · [Open Food Facts barcode scanning](https://openfoodfacts.github.io/openfoodfacts-server/api/tutorials/scanning-barcodes) · [ZXing-C++](https://github.com/zxing-cpp/zxing-cpp/) · [Jaya Grocer online store launch](https://says.com/my/lifestyle/you-can-now-get-your-jaya-grocer-groceries-online). Tesseract and PaddleOCR are named from general knowledge, not checked for this doc.
