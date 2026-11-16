# syqnalglasses

Smart glasses grocery shopping assistant. You look at a product, and the glasses give a short verdict: whether it is a good buy here, whether it is cheaper elsewhere, and whether it fits your diet. The glasses stay thin, the phone does the logic, and the cloud holds the catalogue and prices.

This repository holds the planning documents and prototypes. No app code has been written yet.

## Contents

| Path | What it is |
| --- | --- |
| [docs/plan.md](docs/plan.md) | Main plan: core flow, feature set by phase, Jaya Grocer price check and basket total, gaps and risks, architecture, roadmap, open decisions |
| [docs/vision-model-report.md](docs/vision-model-report.md) | Vision model choice: RF-DETR detector, DINOv2/v3 embedding search for exact SKU, PaddleOCR for price tags, datasets, RealSense test plan, ripeness and freshness check |
| [prototypes/aislelens-2d-simulator.html](prototypes/aislelens-2d-simulator.html) | AisleLens 2D store simulator. Open it in any browser to walk the scan, compare and verdict loop in a fictional store |

The docs are Markdown exports of living documents. The diagrams in them are only in the live versions:

- [Plan (live doc)](https://claude.ai/artifact/Lvxdy5VXsGgBikzhzHmyoF)
- [Vision model report (live doc)](https://claude.ai/artifact/DWtbWaHMzjnd9s7ujVZjwG)
- [AisleLens simulator (hosted)](https://claude.ai/artifact/91yDZ8W5SmyK1NDUBbKv1t)

## Current defaults

- Market: Singapore with prices in SGD, plus Malaysia (MYR) for the Jaya Grocer comparison. Which is the main market is still open.
- Hardware: stay generic and phone-first; the glasses are picked in Phase 2.
- Reuse open source and open data (ZXing-C++, Open Food Facts, PriceCatcher, Open Prices) rather than building from scratch.

## Roadmap

1. Phase 0: simulate (the 2D simulator is here; a 3D store simulator is in progress and will be added under `prototypes/`).
2. Phase 1: phone prototype.
3. Phase 2: glasses pilot.
4. Phase 3: real-data beta.

All stores, brands, barcodes and prices in the simulator are made up.
