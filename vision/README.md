# AisleLens vision

The "brain" behind the glasses: give it a camera frame (and depth, if the camera has it) and it says what product you are holding or looking at, reads the shelf price, and gives the verdict: cheaper elsewhere, promo check, allergens, sugar, running basket total.

It is trained and tested on the 3D store simulator ([`prototypes/store-sim-3d`](../prototypes/store-sim-3d/)) and takes RealSense frames in the same format, so the same code and the same scoring run on simulator and real store frames.

## How a frame is handled

```
frame (RGB, optional 16-bit depth in mm)
  -> RF-DETR detector ........ product, price_tag, held_item boxes
  -> depth ................... drop the far aisle; the nearest big pack within 85 cm is "in hand"
  -> ZXing barcode first ..... exact answer when the barcode is readable (pack held close)
  -> DINOv2 embedding search . otherwise: nearest reference photos in the gallery
  -> confidence check ........ unsure -> "turn the pack to show the barcode"
  -> price tag OCR ........... RapidOCR (PaddleOCR models) on the tag's price area; red tag = promo
  -> tag-to-product link ..... the tag on the shelf edge just below the pack
  -> verdict ................. price vs other stores (online incl. delivery), promo check,
                               allergens vs your profile, high sugar, basket total
```

Everything here is open source with permissive licences: RF-DETR (Apache-2.0), DINOv2 (Apache-2.0), PaddleOCR models through RapidOCR (Apache-2.0), ZXing-C++ (Apache-2.0).

| File | What it does |
| --- | --- |
| `aislelens_vision/pipeline.py` | The full loop above |
| `aislelens_vision/detector.py` | RF-DETR wrapper, plus a ground-truth detector for testing later stages alone |
| `aislelens_vision/embedder.py` | DINOv2 features, optional trained head, reference gallery search |
| `aislelens_vision/perception.py` | Barcode, price tag reader, depth helpers, tag-to-product link |
| `aislelens_vision/catalog.py` | The simulator's 54 fictional products, nutrition, allergens, other-store prices |
| `aislelens_vision/verdict.py` | Verdict text and the basket |
| `serve.py` | Local HTTP server the simulator (key V) and the RealSense client call |
| `analyze.py` | Run on one image from the command line |
| `realsense_client.py` | Live RealSense camera to the server, and store recording in the simulator's format |
| `tools/sim_to_coco.py` | Simulator labels to COCO for detector training |
| `tools/train_detector.py` | Fine-tune RF-DETR |
| `tools/train_embedder.py` | Build the gallery and train the embedding head |
| `tools/evaluate.py` | Score every stage against the targets in `docs/vision-model-report.md` |

## Set up

```bash
python -m venv .venv
.venv/Scripts/pip install torch torchvision      # or the CUDA build from pytorch.org
.venv/Scripts/pip install -r vision/requirements.txt
```

## Rebuild everything from scratch

```bash
cd prototypes/store-sim-3d && npm i playwright && cd ../..
node prototypes/store-sim-3d/generate_dataset.mjs --frames 1200 --seed 11 --camera d435 --jpg --skip depth,mask --out data/sim/train --gpu
node prototypes/store-sim-3d/generate_dataset.mjs --frames 300 --seed 12 --camera d435 --held-ratio 0.7 --jpg --skip depth,mask --out data/sim/train_scan --gpu
node prototypes/store-sim-3d/generate_dataset.mjs --frames 150 --seed 22 --camera d435 --jpg --skip depth,mask --out data/sim/valid --gpu
node prototypes/store-sim-3d/generate_dataset.mjs --frames 200 --seed 33 --camera d435 --held-ratio 0.5 --jpg --skip mask --refs --out data/sim/test --gpu
.venv/Scripts/python vision/tools/sim_to_coco.py
.venv/Scripts/python vision/tools/train_embedder.py
.venv/Scripts/python vision/tools/train_detector.py --epochs 4
.venv/Scripts/python vision/tools/evaluate.py --detector runs/detector/checkpoint_best_total.pth
```

Train, validation and test use different random seeds, so test frames have shelf layouts, lighting and poses the models never saw. `data/` and `runs/` are git-ignored; the commands above recreate them.

## Use it

```bash
.venv/Scripts/python vision/serve.py
```

Then open `prototypes/store-sim-3d/index.html`, walk to a shelf, pick something up (E) or put the ring on a pack, and press **V** (Ask AisleLens). The lens card shows the model's answer and, underneath, the simulator's ground truth with a tick or a cross.

Or on a single frame:

```bash
.venv/Scripts/python vision/analyze.py data/sim/test/rgb/00004.jpg --depth data/sim/test/depth/00004.png --allergies peanuts
```

## Results on the simulator test set

Measured on 16 Nov 2026 on the simulator test set (200 frames, shelf layouts and lighting never seen in training), with perfect boxes from the labels so each stage is scored on its own. The detector and the end-to-end loop are not scored yet. Full numbers: [`vision/results/sim_stages_2026-11-16.json`](results/sim_stages_2026-11-16.json).

| Measure | Result | Target (report) | |
| --- | --- | --- | --- |
| Shelf product, right SKU (top-1), 1,594 crops | 97.2% (top-5 99.7%) | 80% or better | Meets |
| Shelf product, SKUs the head never trained on | 94.9% (255 crops) | Photos only, no retraining | Meets |
| Held item, right SKU, 100 items | 56% | 90% or better | **Misses** |
| Held item, confident but wrong | 5% | 2% or less | **Misses** |
| Held item, "turn the pack" asked | 39% | (lower is better) | |
| Held item, barcode read / right when read | 11% / 100% | 98% with barcode visible | Barcode is right when read, but rarely readable |
| Price tag read, tags 30 px or taller | 95% (225 tags) | 95% or better | Meets |
| Price tag read, tags 22 to 29 px | 74% (204 tags) | 95% or better | **Misses** |
| Price misread (wrong price given) | 3.7% of tags | | |
| Promo tag flag | 100% | | |

Why held items are weak, and what to try first:
- The "answer or ask" threshold (similarity 0.57, margin 0.01) was tuned on validation crops that are mostly shelf packs, so it is too loose for held items. Tune a separate threshold for held items.
- The head saw few held crops (630 in training). Train it with more held and close-up crops, and crops cut tighter around the hand.
- Barcodes are only readable when the pack is close (the "scan" view). Try reading at 2x and 3x upscale on the barcode area, and ask the shopper to bring the pack closer instead of "turn the pack" when a barcode is visible but unreadable.

The detector still has to be trained on a GPU (see [docs/handover.md](../docs/handover.md)).
