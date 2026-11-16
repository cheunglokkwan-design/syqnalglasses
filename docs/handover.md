# Handover: vision pipeline (16 Nov 2026)

Branch: `vision-pipeline`. Goal of this stage: be able to use a trained vision model to recognise products in the 3D store simulator and give the shopping verdict, then take the same code to a RealSense camera in a real store.

## Train on real images in Google Colab (no local GPU needed)

Open [vision/colab/train_real_models.ipynb](https://colab.research.google.com/github/cheunglokkwan-design/syqnalglasses/blob/vision-pipeline/vision/colab/train_real_models.ipynb) in Colab, choose Runtime, then Change runtime type, then T4 GPU, then Run all. It trains a shelf-product detector (SKU-110K) and a price-tag detector (Open Food Facts photos) and saves them to Google Drive (`MyDrive/syqnal_runs`). For about an hour in total, run it on Colab with `TRAIN = 'products'` and on Kaggle with `TRAIN = 'tags'` at the same time. Copy `det_products/` and `det_tags/` into `runs/` and `vision/serve.py` uses them automatically.

The first laptop (RTX 3050, 7.7 GB RAM) runs out of RAM during training while other apps are open, so Colab is the default place to train.

## Done

**3D store simulator** ([`prototypes/store-sim-3d`](../prototypes/store-sim-3d/))
- Price tags are labelled (class 255 in the mask, `type: price_tag` in labels, with the printed price and promo flag).
- Items in the shopper's hand: a product plus a simple hand, at 0.3 to 0.7 m, showing front, back, side, a random angle, or the barcode close up ("scan").
- Clean reference photos per product (`--refs`, 8 views each) for the embedding gallery.
- JPEG output and skipping depth or mask, to keep datasets small.
- Fixed: held items were never flagged as `held` in labels (instance -1 was not handled).
- "Ask AisleLens" (key V): sends the current frame to the vision server and shows the verdict on the lens, with the simulator's ground truth and a tick or cross.

**Vision pipeline** ([`vision/`](../vision/), see its README)
- Detector wrapper (RF-DETR, classes product / price_tag / held_item), DINOv2 embedding search with a trained head, ZXing barcode, RapidOCR price-tag reader, depth logic for "item in hand", tag-to-product link, verdict (price vs 4 fictional stores incl. online delivery, promo check, allergens, high sugar), basket.
- Scripts: COCO export, detector training, embedder training, evaluation against the report's targets, local server, single-image CLI, RealSense client.

**Data generated on the first laptop** (not in git; regenerate with the commands in `vision/README.md`): train 1,200 + train_scan 300 frames, valid 150, test 200 (with depth), gallery 54 × 8. The embedder head is trained (9 products held out to test "new product from photos only").

## Not done

1. **Detector not trained yet.** CPU training ran at 0.2 steps/s (about 63 min per epoch) on the first laptop, which has no room for the CUDA build of PyTorch. It needs a GPU. Without a trained detector, `serve.py` falls back to treating the centre of the view as the product.
2. **No end-to-end score yet** (needs the detector).
3. **RealSense client untested**: written against the pyrealsense2 API without a camera.
4. Not checked in a real store; all products, prices and stores are fictional.

## Results so far (simulator test set, perfect boxes from the labels)

Measured on 16 Nov 2026 on the simulator test set (200 frames, shelf layouts and lighting never seen in training), with perfect boxes from the labels so each stage is scored on its own. The detector and the end-to-end loop are not scored yet. Full numbers: [`vision/results/sim_stages_2026-11-16.json`](../vision/results/sim_stages_2026-11-16.json).

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

## Next steps, in order

1. Set up on a machine with an NVIDIA GPU, regenerate the data, train the detector (`train_detector.py --size nano --epochs 10 --batch 8`, or `small` if the GPU has 8 GB+).
2. Run `evaluate.py --detector ...`, put the numbers in `vision/README.md` and here.
3. Try it live: `python vision/serve.py`, open the 3D store, press V on held and shelf items.
4. Weak spots to work on: barcode reads need the pack close (the "scan" gesture); price tags under about 25 px tall are unreliable; extend the gallery with real product photos before the store test.
5. RealSense: plug in, run `vision/realsense_client.py`, record a short walk, run `evaluate.py` on it after hand-labelling a few frames.
6. Store test plan: `docs/vision-model-report.md`, "Training and evaluation plan", step 4.

## RealSense and the phone

- **Android:** the RealSense SDK (librealsense) has an Android wrapper that talks to a D415/D435 over a USB OTG cable on non-rooted phones. Intel's forum notes the D435i needs extra (HID/IMU) support, and the current release notes do not list Android, so check the repo's Android docs before buying cables. The phone must power the camera, and USB 3 helps for full resolution. Not tested by us.
- **iPhone:** not supported by the SDK, as far as we know.
- **Simplest for the first store test:** RealSense into a laptop in a backpack running `realsense_client.py --record`, and replay frames through the pipeline afterwards. The phone can show live verdicts from `serve.py` over the laptop's hotspot later.
- Sources: [librealsense releases](https://github.com/realsenseai/librealsense/releases), [Intel forum: Android SDK for RealSense](https://community.intel.com/t5/Items-with-no-label/Hi-is-there-any-Android-SDK-to-connect-Realsense-depth-camera/m-p/657369).

## Jetson

A Jetson Orin Nano is a good stand-in for the glasses' on-device computer (measure detector latency there). The original Jetson Nano (2019) is stuck on JetPack 4 / Python 3.6 and can't run current RF-DETR or transformers. Neither is a good training machine; use a desktop or laptop GPU, or a cloud GPU.

## Prompt for Claude Code on the next laptop

```
Clone https://github.com/cheunglokkwan-design/syqnalglasses and check out the branch
vision-pipeline. Read docs/handover.md, vision/README.md and docs/vision-model-report.md first.

Goal: a trained vision model that recognises products in the 3D store simulator and gives
the shopping verdict, scored against the targets in the report.

1. Check the GPU (nvidia-smi) and free disk (need ~10 GB). Make a venv, install the CUDA
   build of PyTorch that matches the driver, then pip install -r vision/requirements.txt
   and "rfdetr[train]". In prototypes/store-sim-3d run: npm i playwright.
2. Regenerate the data with the commands in vision/README.md ("Rebuild everything").
3. python vision/tools/sim_to_coco.py, then vision/tools/train_embedder.py.
4. Train the detector on the GPU: vision/tools/train_detector.py --size nano --epochs 10 --batch 8
   (try --size small if there is 8 GB+ of GPU memory). Watch for a Windows dataloader
   issue: keep --workers 0 if it hangs.
5. python vision/tools/evaluate.py --detector runs/detector/checkpoint_best_total.pth
   Add the detector and end-to-end numbers to the results tables in vision/README.md and
   docs/handover.md, comparing each with the report's target. Report honestly what misses.
   Then work on the weakest stage, held items (see "Why held items are weak").
6. Start python vision/serve.py, open prototypes/store-sim-3d/index.html, and check that
   pressing V on a held item and on a shelf item gives a sensible verdict.
7. Commit to the vision-pipeline branch and push. Do not merge to main without asking me.
```
