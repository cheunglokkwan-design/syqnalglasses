# Vision model choice for shelf products

Nov 16, 2026 · @Antony Real Betis

_Exported from the [live doc](https://claude.ai/artifact/DWtbWaHMzjnd9s7ujVZjwG) on 16 Nov 2026._

## Recommendation

Do not train one big "grocery model". Use a three-stage pipeline: a small, permissively licensed detector finds products and price tags, a fine-tuned embedding model identifies the exact SKU by searching our catalogue, and OCR plus barcode decoding confirm the match and read the price. New products are added by adding catalogue photos, with no retraining.

| Job | Pick | Why | Fallback |
| --- | --- | --- | --- |
| Find products, price tags, hand-held item | [RF-DETR](https://arxiv.org/html/2511.09554v2) Nano or Small, fine-tuned on SKU-110K plus our own frames | 48.0 / 52.9 COCO AP at 2.3 / 3.5 ms on a T4, beats YOLO11 at the same speed; Apache 2.0 | D-FINE or YOLO (YOLO is AGPL-3.0, which needs a paid licence for a closed app) |
| Identify the exact product | DINOv2 or DINOv3 embedding, fine-tuned shelf-to-catalogue ([Cat2Real](https://arxiv.org/pdf/2607.09888)), vector search in the cloud | Open-set: new SKUs need only photos; fine-tuning took DINOv3 from 53.8% to 80.7% top-1 | MobileCLIP or SigLIP 2 for a smaller on-phone model |
| Read price tags and pack text | [PaddleOCR](https://www.paddleocr.ai/v3.5.0/en/update/update.html) PP-OCRv5 mobile | Runs on Android, has an English model, Apache 2.0 | Google ML Kit text recognition |
| Exact ID when the barcode is visible | ZXing-C++ (already in the plan) | Barcode = ground truth, no model needed |  |
| Hard cases only | A cloud vision-language model asked "which of these 5 candidates?" | Handles odd angles and lookalikes, too slow and costly for every frame | Ask the user to turn the pack |

The RealSense depth stream is used to cut background shelves away and measure distance to the item, not as a model input in v1.

## The problem

The glasses must answer one question fast: which exact product is the shopper looking at or holding, and what does the shelf tag say it costs. That splits into four jobs with different constraints.

- **Detect** every pack and price tag in a dense shelf frame, and the one item in the shopper's hand. Shelves hold 100+ items per frame and packs touch each other.
- **Identify** the exact SKU (brand, variant, size). Lookalikes differ only by flavour colour or "500 g" versus "1 kg", and the catalogue changes weekly, so a fixed class list will go stale.
- **Read** the shelf tag price, promo text and unit price, which vary by store layout.
- **Confirm** with the barcode whenever it is visible, since that is exact.

Constraints: under about 300 ms from glance to answer, glasses stay thin (camera + display), the phone runs detection and OCR, the cloud holds the catalogue index and prices. Open data must be reusable without a restrictive licence, because the app may be closed-source.

> **Diagram: per-frame pipeline · phone, cloud, fallback.** Drawn in the [live doc](https://claude.ai/artifact/DWtbWaHMzjnd9s7ujVZjwG); not included in this Markdown export.

A visible barcode wins outright; otherwise the embedding search answers, and only low-confidence matches go to the slow cloud model or back to the shopper.

## Candidate models

Detection is a solved, commodity problem; product identity is where accuracy is won or lost. Figures are COCO AP and NVIDIA T4 TensorRT latency from the RF-DETR paper unless noted.

| Model | Role | Accuracy / speed | Licence | Verdict |
| --- | --- | --- | --- | --- |
| [RF-DETR](https://arxiv.org/html/2511.09554v2) Nano / Small / Medium | Detector (also segmentation) | 48.0 / 52.9 / 54.7 AP at 2.3 / 3.5 / 4.4 ms; seg Nano 40.3 AP vs YOLO11-seg 30.0 | Apache 2.0 for Nano to Large ([Roboflow](https://roboflow.com/blog/rf-detr-vs-alternatives)) | Pick. Best speed and accuracy, fine-tunes well on small datasets (RF100-VL 58.6 AP Nano) |
| D-FINE N / S / M | Detector | 42.7 / 50.6 / 55.0 AP at 2.1 / 3.5 / 5.4 ms | Apache 2.0 | Fallback if RF-DETR exports badly to the phone |
| RT-DETRv2 | Detector | Below D-FINE at similar speed | Apache 2.0 | Superseded |
| YOLO11 / YOLO26 (Ultralytics) | Detector | YOLO11 N/S/M 37.1 / 44.1 / 48.3 AP at 2.2 / 3.2 / 5.1 ms | AGPL-3.0, paid licence for closed apps | Easiest tooling, but licence and accuracy both worse |
| Grounding DINO, OWLv2, YOLO-World | Open-vocabulary detector ("find the oat milk") | Zero-shot, but many times slower than RF-DETR (approximate, not benchmarked here) | Mostly Apache 2.0 | Use offline to pre-label our frames, not on device |
| DINOv2 / DINOv3 ViT, fine-tuned ([Cat2Real](https://arxiv.org/pdf/2607.09888)) | SKU embedding for catalogue search | Shelf-to-catalogue top-1 80.7% (DINOv3-L/384) and 76.3% (DINOv2-L/224) vs 53.8% / 35.5% off the shelf; top-5 94.7% | Apache 2.0 (released weights) | Pick. Open-set, new SKUs need only photos |
| CLIP / SigLIP 2 / MobileCLIP | Image + text embedding | Weaker on fine-grained SKUs untuned; text side lets us search by product name | Apache 2.0 / MIT | Distil target for an on-phone embedder; text matching against Open Food Facts names |
| [PaddleOCR](https://www.paddleocr.ai/v3.5.0/en/update/update.html) PP-OCRv5 | Price tag and pack text | Android demo since 3.0.2; dedicated English model (+11% on English) in 3.2.0 | Apache 2.0 | Pick for price tags |
| Cloud VLM (Claude, Gemini, Qwen2.5-VL) | Pick among top-5 candidates, read odd tags | Strongest on hard cases, 1 to 3 s and per-call cost | Varies | Fallback only, maybe 5% of scans |

Training a closed-set classifier over every SKU was rejected: a new product or new pack design means retraining, and a large supermarket stocks many thousands of SKUs.

## Datasets

Public data covers detection well and identity poorly for our market; Malaysian and Singaporean SKUs must come from our own capture. Check each licence before commercial use: most academic retail sets are research-only.

| Dataset | What it has | Use for us |
| --- | --- | --- |
| [SKU-110K](https://docs.ultralytics.com/datasets/detect/sku-110k) (Trax, CVPR 2019) | 11,743 dense shelf photos, 1.7M boxes, one class "object" | Pre-train the detector for packed shelves |
| [RP2K](https://ar5iv.labs.arxiv.org/html/2006.12634) | 10,385 shelf photos and 384,311 crops of 2,388 SKUs, \~500 stores | Pre-train the embedding on real shelf crops |
| [Products-10K](https://arxiv.org/abs/2008.10545) | 10,000 SKUs from JD.com, human-labelled | Pre-train the embedding on fine-grained lookalikes |
| RPC, Grocery Products, Freiburg Groceries, GroZi-120 | Checkout trays, studio-to-shelf pairs, small class sets | Extra eval sets, low priority |
| Open Food Facts | Product photos and names keyed by barcode, ODbL | Catalogue images for the search index |
| Our capture: RealSense + phone in Jaya Grocer (or SG store) | Shelf video, hand-held item clips, tag crops, with barcode as label | The real training and test set |

Our capture uses the barcode-assisted method from Cat2Real: scan the barcode, then film the pack from several angles, so every crop is labelled for free. A target of 2,000 SKUs × 10 to 20 crops each should be enough to fine-tune the embedder (our estimate); the detector needs about 500 to 1,000 annotated frames, pre-labelled by Grounding DINO and corrected by hand.

## Simulator frames and the RealSense camera

Synthetic frames are good for testing the pipeline and pre-training the detector, but cannot replace real shelf photos for product identity. Real packs have glare, crumpled film and price stickers that a renderer does not reproduce.

**From the 3D store simulator (built in the other thread):**

- Texture pack meshes with real product photos from Open Food Facts, then randomise lighting, camera height, motion blur, shelf fill and occlusion by a hand.
- Export RGB, depth and exact boxes and SKU ids per frame: free, perfect labels for detector pre-training and for end-to-end latency and UX tests.
- Mix at most about 30% synthetic into detector training and keep the test set 100% real (rule of thumb, to be tuned).

**RealSense in the store** ([RealSense spun out of Intel in July 2025](https://www.edge-ai-vision.com/2025/07/realsense-completes-spinout-from-intel-raises-50-million-to-accelerate-ai-powered-vision-for-robotics-and-biometrics/) and says it keeps the existing camera line and roadmap):

- Use a D435i or D455 for walking the aisle and a D405 for close hand-held items (it is built for short range). Mount at eye height on a cap or chest rig to mimic the glasses view.
- Record aligned RGB-D as .bag files with librealsense, so every store run can be replayed offline through each model version.
- Use depth for three things in v1: pick the nearest object as "the item in hand", drop detections from the far aisle, and estimate shelf distance to trigger a scan only when the shopper pauses.
- Feed RGB only into the models at first. The real glasses will likely have no depth sensor, so a model that depends on depth would not transfer.

## Training and evaluation plan

Five steps, each with a gate before the next. Steps 1 to 3 need only a GPU and public data; step 4 is the first store visit with the RealSense.

1. **Baseline with no training.** RF-DETR Small (COCO weights) + DINOv2 off the shelf + PaddleOCR on 200 phone photos from one store. Gate: the pipeline runs end to end and we have a number to beat.
2. **Detector fine-tune.** Classes: product, price tag, hand-held item. Pre-train on SKU-110K plus simulator frames, fine-tune on our labelled frames. Gate: mAP50 of at least 0.85 on real shelf frames.
3. **Embedder fine-tune.** DINOv2-B (or DINOv3) with triplet loss and hard negatives, Cat2Real recipe, on RP2K + Products-10K, then our barcode-labelled crops. Index catalogue photos in a vector store (FAISS or pgvector). Gate: top-1 of at least 80% and top-5 of at least 95% on held-out real crops.
4. **Store test with RealSense.** Walk 3 aisles, 20 minutes each, recorded to .bag; pick up 50 items by hand. Replay offline through every model version. Gate: the metrics below.
5. **Phone port.** Export the detector to TFLite / Core ML (int8), distil the embedder to MobileCLIP or a ViT-S, keep the full index in the cloud. Gate: latency under 300 ms on a mid-range Android phone.

| Metric | Measures | Target for the store test |
| --- | --- | --- |
| Hand-held item top-1 | Right SKU for the item in hand | ≥ 90% (≥ 98% with barcode visible) |
| Shelf product top-1 | Right SKU for a shelf pack the shopper looks at | ≥ 80% |
| Price tag read | Exact price string matches the tag | ≥ 95% |
| Wrong-answer rate | Confident answer that is wrong | ≤ 2%, otherwise ask the user to turn the pack |
| Glance to answer | Frame captured to result on the display | ≤ 300 ms median |
| New-SKU onboarding | Time to add a product nobody trained on | Photos only, no retraining |

Targets are our proposed bar, not published benchmarks, and should be revisited after step 1.

## Ripeness and freshness check

A camera can reliably flag produce that is visibly bad (mould, bruises, browning, wilting) and grade ripeness for fruit whose skin changes colour, such as bananas, mangoes and tomatoes. It cannot measure firmness, sweetness or internal rot; that needs touch, smell or a hyperspectral sensor. The feature should say "looks fresh / check this one" and a ripeness stage, never "safe to eat".

| Signal | RGB camera | RealSense depth | Verdict |
| --- | --- | --- | --- |
| Mould, rot spots, bruises, browning | Yes, visible surface defects | No | Ship: fresh vs check-this flag |
| Ripeness of colour-changing fruit (banana, mango, tomato, papaya) | Yes, stage 1 to 7 style grading | No | Ship for a short list of fruit |
| Wilting leafy greens, wrinkled skin | Partly, from texture | Partly, shape and sag | Experiment |
| Size and volume (e.g. price per fruit) | Needs a reference | Yes, from the point cloud | Nice extra |
| Avocado, kiwi, melon ripeness (skin barely changes) | Weak | No | Not with a camera; hyperspectral got 85% (avocado) to 60% (kaki) in a [2026 study](https://arxiv.org/pdf/2608.01202) |
| Firmness, sugar, internal rot | No | No | Out of scope |

**Model.** Reuse the pipeline: the RF-DETR detector finds each fruit, then a small classifier head on the same DINOv2 embedding predicts fresh / check and a ripeness stage per fruit type. No extra on-phone model, a few kilobytes of head weights per fruit.

**Data.** Public sets are mostly single fruits on plain backgrounds, so they only bootstrap the head.

| Dataset | Content | Licence | Use |
| --- | --- | --- | --- |
| [Fruit and Vegetable freshness](https://pmc.ncbi.nlm.nih.gov/articles/PMC9653939) (Kaggle) | 12,000 images, fresh vs rotten for 10 fruits and vegetables; \~70% single object | Check on Kaggle | Pre-train fresh vs rotten |
| [FruitVision](https://data.mendeley.com/datasets/xkbjx8959c) | 10,154 images, apple, banana, mango, orange, grape as fresh / rotten / formalin | CC BY-NC-ND 4.0, research only | Evaluation only |
| [Banana ripeness sets](https://data.mendeley.com/datasets/56c75hvm7r) (several on Mendeley) | Ripeness stages | Varies by set | Pre-train banana stages |
| Our store capture | RealSense + phone at the produce section, labelled by a person who handles the fruit | Ours | Real training and test set |

That 2022 study reached 73.5% AP50 and failed on overlapping items and similar colours (rotten orange vs mango), so in-store accuracy must be measured on our own frames before the feature ships. Proposed bar: at least 90% recall on "check this one" for visibly bad items, with a clear "camera can't tell" message for avocado-type fruit.

## Risks and open questions

- **Lookalike variants.** Embeddings confuse flavours and sizes; Cat2Real's weakest categories scored 51 to 61% top-1. Mitigation: OCR the size and flavour text and re-rank the top 5.
- **Dataset licences.** SKU-110K, RP2K and Products-10K are academic releases; confirm commercial terms or use them only to pre-train and ship weights trained mainly on our own data.
- **Store permission.** Filming in Jaya Grocer needs the store's consent; ask before the first visit.
- **Glasses camera.** Field of view, resolution and the lack of depth on the final glasses are unknown until the Phase 2 hardware pick; the RealSense rig only approximates them.
- Open: main market (Malaysia or Singapore) decides which store we film and which SKUs we index first.
- Open: do we want Ultralytics tooling enough to pay its commercial licence, or stay fully Apache 2.0?

## Sources

- [RF-DETR paper (arXiv 2511.09554)](https://arxiv.org/html/2511.09554v2)
- [Roboflow: RF-DETR vs alternatives, licences](https://roboflow.com/blog/rf-detr-vs-alternatives)
- [Cat2Real: catalogue-to-real product recognition (arXiv 2607.09888)](https://arxiv.org/pdf/2607.09888)
- [PaddleOCR release notes](https://www.paddleocr.ai/v3.5.0/en/update/update.html)
- [SKU-110K dataset](https://docs.ultralytics.com/datasets/detect/sku-110k)
- [RP2K dataset](https://ar5iv.labs.arxiv.org/html/2006.12634)
- [Products-10K dataset](https://arxiv.org/abs/2008.10545)
- [RealSense spinout from Intel](https://www.edge-ai-vision.com/2025/07/realsense-completes-spinout-from-intel-raises-50-million-to-accelerate-ai-powered-vision-for-robotics-and-biometrics/)
