# Syqnal Glasses: real-world plan

Nov 16, 2026

_Exported from the [live doc](https://claude.ai/artifact/Pe4TGvhKj6T8Vd8ZaLSKf2) on 16 Nov 2026. The two diagrams (architecture, milestones) are only in the live doc._

## Where we are

The pipeline recognises shelf products in our simulator 97% of the time, but it has not yet seen a real product, a real store or a real camera. Every product, brand, price and store it knows is fictional. Moving to the real world means replacing three things: the training images, the product catalogue, and the camera.

| Stage | In the simulator today (16 Nov 2026) | What the real world adds |
| --- | --- | --- |
| Find products and tags (RF-DETR) | Code ready; real-image training runs on a free Colab or Kaggle GPU | Glare, crumpled film, stickers, people and trolleys in view, motion blur |
| Name the product (DINOv2 + gallery) | 97.2% on shelf packs, 94.9% on packs it never trained on | Thousands of real SKUs, look-alike variants (flavours, sizes), worn packs |
| Item in hand | 56% right, 5% confidently wrong (target 90% and 2%) | Real hands, the real distance people hold things at |
| Barcode | Right every time it reads, but reads only 11% of held items | Curved and shiny packs; real barcodes are 37 mm wide |
| Price tag | 95% read on tags 30 px or taller | Jaya Grocer's real tag layout, promo stickers, handwritten tags |
| Prices elsewhere | 4 fictional stores | PriceCatcher, Open Prices, online stores (see the main plan) |
| Camera | Simulated RealSense D435 | Real RealSense first, then the glasses camera, which probably has no depth |

The simulator stays useful: it is free, perfectly labelled, and tests the whole loop. But the test set that decides whether we are ready must be 100% real images from a real store.

**Done on 16 Nov 2026 toward real images:**

- Real datasets downloaded and converted: 1,982 real price-tag photos and 866 real shelf photos (125,374 boxed products).
- A free-GPU training notebook for Google Colab and Kaggle trains the two real detectors in about an hour (an estimate): `vision/colab/train_real_models.ipynb` on the `vision-pipeline` branch.
- The pipeline merges the two detectors and loads Colab-trained models automatically.
- 20 real Malaysian products chosen, with Jaya Grocer and Lotus's prices (see the datasets section).
- Not yet done: the first real-image training run, real-photo scores, and gallery photos and barcodes for the 20 products.

## Real-image datasets

Only three public sources allow use beyond research: Open Food Facts photos, the Grocery Store Dataset and Open Food Facts' price-tag set. Everything else is research-only or unlicensed, and no public set shows products held in a hand or Malaysian products. Our own store capture is therefore the core dataset, and public data only gets us started.

| Dataset | What it shows | Size | Licence | Our use |
| --- | --- | --- | --- | --- |
| [Open Food Facts price-tag-detection](https://huggingface.co/datasets/openfoodfacts/price-tag-detection) | Shelf photos, price tags boxed | 2,229 images | CC BY-SA 4.0 | Train the tag detector now |
| [Grocery Store Dataset](https://github.com/marcusklasson/GroceryStoreDataset) | Phone photos in stores: produce, juice, milk, yoghurt | 5,125 images, 81 products | MIT | Test product recognition on real photos |
| [Open Food Facts product photos](https://world.openfoodfacts.org/terms-of-use) | Packs photographed by users, front and back | Millions of products | CC BY-SA (pack designs and trademarks may still apply) | Gallery photos and barcodes for real products |
| [SKU-110K](https://github.com/eg4000/SKU110K_CVPR19) | Dense supermarket shelves, every pack boxed | 11,762 images, 1.73M boxes | Academic, non-commercial only | Prototype product detector; replace before launch |
| [Products-10K](https://products-10k.github.io/challenge.html), [RPC](https://rpc-dataset.github.io/), [MVTec D2S](https://www.mvtec.com/company/research/datasets/mvtec-d2s) | Online shop photos, checkout scenes | 10k to 53k+ images | Non-commercial | Research comparisons only |
| [RP2K](https://arxiv.org/abs/2006.12634), [Freiburg Groceries](https://github.com/PhilJd/freiburg_groceries_dataset) | Shelf and home photos | 5k to 500k images | No licence found | Do not use |

Each public set labels one kind of object (products or tags), so we train two detectors and merge their boxes.

**Our own capture** (the store pilot section has the protocol): barcode-labelled clips of products turned in hand, shelf walks, and close-up tags. The target is 20 products first, then 200, then 2,000 for a beta. Boxes are pre-labelled by our detectors and corrected in CVAT or Label Studio.

**First 20 real products.** Chosen 16 Nov 2026 across pack shapes and allergens, with online prices. Jaya Grocer blocks automated reading, so its prices come from search listings and may be out of date (Jaya also prices by branch). Lotus's prices were read from the product pages on 16 Nov 2026.

| Product | Size listed | Jaya Grocer (RM) | Lotus's (RM) |
| --- | --- | --- | --- |
| Milo soft pack | 900 g | 23.20 | 19.95 (promo 18.49) |
| Maggi Curry noodles | 5 x 79 g | 5.79 | about 4.95 (sold by carton) |
| Dutch Lady full cream milk | 1 L | 8.90 (promo 7.69) | 7.95 (promo 7.65) |
| Gardenia Original Classic Jumbo | 600 g | 4.30 | 4.30 |
| Nescafe Classic jar | 200 g | 34.90 | 28.50 |
| Julie's Peanut Butter Sandwich | 360 g | 9.99 (promo 8.99) | 11.25 (promo 8.29) |
| Mamee Monster family pack | 10 x 25 g / 10 x 20 g | 5.20 | 3.99 (promo 3.75) |
| Yeo's soya bean | 1 L | 3.50 (less sugar) | 2.95 (promo 2.49) |
| 100PLUS isotonic | 1.5 L | 3.90 (promo 2.99) | 4.30 (promo 3.59) |
| Ayam Brand sardines in tomato sauce | 425 g | 10.45 | 10.65 |
| Adabi chicken and meat curry powder | 250 g | 6.90 | 5.70 (promo 5.55) |
| Hup Seng Ping Pong cream crackers | 428 g | 4.90 | 4.49 |
| Mister Potato Original | 125 g | 3.99 | 4.35 (promo 3.99) |
| Kit Kat 4 finger | 35 g | 3.60 | not listed online |
| Farm Fresh UHT full cream milk | 1 L | 6.90 | 6.65 (promo 5.49) |
| Nestle Koko Krunch | 300 g | 12.90 | 11.69 (promo 10.29) |
| Quaker Instant Oatmeal | 800 g / 1 kg | 8.99 (800 g) | 10.65 (1 kg) |
| Lee Kum Kee Premium Oyster Sauce | 510 g | 19.90 (promo 17.99) | 16.45 |
| Knife blended cooking oil | 2 kg | 15.90 | 15.45 |
| Vitagen Original | 5 x 125 ml | 5.30 | 5.59 |

## Simulator and real images together

The simulator trains and tests the plumbing; real images teach the model what real products look like; and only real store frames decide whether we are ready.

| Job | Simulator | Public real datasets | Our own store capture |
| --- | --- | --- | --- |
| Find products and tags | Pre-training, free exact boxes, rare cases on demand (hands, glare, gaps) | Dense shelf boxes (SKU-110K style) | Fine-tuning and the final test |
| Name the exact product | Pipeline tests only (fictional SKUs) | Pre-training the embedding head on many real SKUs | The real gallery and the final test |
| Item in hand | Many held poses, scan gesture | Few public sets show hands | Required: film real hands |
| Price tags | Layout and OCR tests | Little public data | Jaya Grocer's real tag layout |
| Depth logic | RealSense-style noise | None | Real RealSense recordings |
| Speed and end-to-end UX | Yes | No | Yes |

Training order, each step gated on the one before:

1. **Detector:** start from COCO weights, pre-train on shelf datasets plus simulator frames, then fine-tune on our labelled store frames. Keep simulator frames to about a third of the mix (a rule of thumb to tune).
2. **Embedding head:** train on real product crops from public datasets and from our barcode-labelled store clips. The gallery holds real photos of each product we sell to the model (front, back, sides).
3. **Tag reader:** add Jaya Grocer's tag layout, then check it on real tag crops.
4. **Thresholds:** tune the "answer or ask" thresholds on real validation frames, separately for held items and shelf packs.

To measure the gap between simulator and reality, score every model version on both test sets and track the difference per stage. A cheap way to shrink the gap: put real product photos (from our captures or Open Food Facts) on the simulator's packs, so the simulator becomes a stand-in for the real products we test with.

## Hardware path: RealSense, then phone, then glasses

Test with the RealSense on a laptop now, build the phone app next, and make Ray-Ban Meta or Oakley Meta the first glasses: they let apps take the camera stream, and they have been sold in Malaysia since 20 Oct 2026.

| Glasses | Camera stream for our app | Display | In Malaysia | Fit for us |
| --- | --- | --- | --- | --- |
| [Ray-Ban Meta, Oakley Meta](https://wearables.developer.meta.com/docs/develop/dat/build-overview/) | Yes: Wearables Device Access Toolkit 1.0, video up to 720 x 1280 at up to 30 fps, delivered to the phone app | No (audio answers) | Yes, from RM 2,069 ([launch report](https://soyacincau.com/2026/10/14/ray-ban-oakley-meta-ai-glasses-malaysia-launch-price-specs/)) | **First pick**: camera access is official; the phone already does the thinking |
| [Meta Ray-Ban Display](https://www.roadtovr.com/meta-ray-ban-display-int-launch-connect-2026/) | Yes, same toolkit | Yes | No | Later, for a lens card |
| [Android XR glasses](https://developer.android.com/develop/xr/jetpack-xr-sdk/access-hardware-projected-context) | Yes, via CameraX (experimental API) | Audio models first | No (US and Korea) | Watch; good fit once sold here |
| [Rokid Glasses](https://pcimage.com.my/?p=11265) | Unclear: photo call found, no official video stream | Yes | Yes | Test only if camera access is confirmed |
| [Brilliant Labs Halo](https://docs.brilliant.xyz/halo/halo-sdk-lua/) | Still photos only, 640 px | Yes, small | Not found | Too low resolution for packs |
| [Even Realities G2](https://hub.evenrealities.com/docs/get-started/overview) | No camera | Yes | Not checked | No |
| [Snap Specs](https://developers.snap.com/spectacles/about-spectacles-features/apis/camera-module) | Yes, up to 2016 x 1512 | Yes, AR | No | Too costly (US$2,195) and not sold here |

**What runs where.** The glasses send a frame to the phone on a tap or a pause. The phone runs barcode reading, the detector, product matching, tag reading and the verdict, and answers in about a second. The cloud keeps the full product gallery and the price feeds. Meta's frames are 720 x 1280, about the same as our RealSense test frames, so the models trained now carry over.

**RealSense on a phone.** The RealSense SDK builds an Android version that needs a USB 3 OTG port, but RealSense lists Android as "compilable, but not validated" ([Android notes](https://github.com/realsenseai/librealsense/blob/master/doc/android.md), [releases](https://github.com/realsenseai/librealsense/releases)). For the store test, plug it into a laptop instead and record with `vision/realsense_client.py`.

**Battery and heat.** No always-on video: scan only on a tap, a voice command or when the head is still. The final glasses most likely have no depth sensor, so the RGB-only fallback for finding the item in hand stays in the pipeline.

## Integration architecture

Keep the glasses thin: they capture a frame and show or speak the answer, the phone runs the pipeline we built, and the cloud holds the full catalogue, the price feeds and training.

> **Diagram: integration architecture (glasses, phone, cloud).** Drawn in the [live doc](https://claude.ai/artifact/Pe4TGvhKj6T8Vd8ZaLSKf2). Glasses send a frame to the phone and get the verdict back; the phone looks up unknown items and prices in the cloud; the RealSense test rig and the 3D simulator call the same `/analyze` API.

The phone talks to the glasses over the vendor's SDK and runs today's `vision/` code, ported (detector exported to TFLite or Core ML, embedder distilled to a small model). Unknown items fall back to a cloud search of the full gallery. Because the simulator, the RealSense rig and later the glasses all call the same `/analyze` API, every stage can be tested before the glasses arrive. Swapping glasses vendors touches only the glasses adapter.

## Store pilot

Get the store's written permission first, film products rather than people, and keep the first visit small: 3 aisles, about 2 hours, 50 products handled.

**Permission.** Ask the Jaya Grocer outlet manager (or head office) in writing before filming. Say what we capture, that nothing is published, and offer to share results. Film at quiet hours.

**Privacy (Malaysia's Personal Data Protection Act).** Point the camera at shelves and products, not people. Before any frame is stored, an on-device face and person detector blurs or drops it. No audio. Recordings are encrypted, kept only by the team, and deleted after labelling. The same rules carry into the product: frames are processed and dropped, never uploaded by default. Check the PDPA duties with someone qualified before a public beta.

**Capture protocol per visit:**

1. **Products in hand (the most valuable data):** scan the barcode with the phone, then turn the pack slowly in hand for 10 to 20 seconds in front of the RealSense. The barcode labels every frame of that clip for free.
2. **Shelf walk:** walk each aisle at eye height with the RealSense on a cap or chest mount, recording colour and depth.
3. **Price tags:** photograph tags close up, and write down the true price of 30 items for the ground truth.
4. **Reference photos:** front, back and sides of each product we handled, on a plain background (store or home).
5. **Log:** date, outlet, time, lighting notes, camera settings.

**What we measure** (targets from the vision model report): held item right 90% or more, shelf pack right 80% or more, tag price read 95% or more, confident wrong answers 2% or fewer, frame to answer 300 ms or less. We also record the same numbers on simulator frames to track the gap.

## Milestones and who does what

The glasses pilot starts only when the model passes on real store frames: held item right 90% or more and confident wrong answers 2% or fewer. Dates are a proposal for a part-time team.

> **Diagram: milestones, proposed dates Nov 2026 to Mar 2027.** Drawn in the [live doc](https://claude.ai/artifact/Pe4TGvhKj6T8Vd8ZaLSKf2): Colab training of the real detectors and held-item fixes (17 to 30 Nov), RealSense test at home (1 to 13 Dec), first Jaya Grocer visit (14 to 20 Dec), retraining on real data (21 Dec to 17 Jan), glasses camera-access test (4 to 17 Jan), phone app running the pipeline (18 Jan to 28 Feb), glasses pilot gate (1 Mar 2027).

- **Free cloud GPU (Colab or Kaggle) or the partner's machine:** detector training, retraining on real data, large dataset runs. Work arrives as pushed branches with a ready prompt (`docs/handover.md`).
- **This laptop (CPU):** code, held-item and tag-reader fixes, small model training (the embedding head), scoring, docs.
- **You, in person:** store permission and visits, the RealSense rig, buying and testing glasses.

## Risks and open decisions

The biggest risks are the glasses' camera access and the gap between simulator and real packs; both are tested cheaply before we commit money.

| Risk | Why it matters | What we do |
| --- | --- | --- |
| Glasses don't give apps the camera | No camera stream, no product recognition | Keep the phone as the brain; pick glasses only after a 1-day camera-access test |
| Simulator results don't carry over | Today's 97% is on fictional, perfectly lit packs | Real-only test set; report the sim-to-real gap per stage |
| Look-alike packs (flavours, sizes) | Wrong answers destroy trust | Barcode first; "turn the pack" when unsure; gallery photos of every variant |
| Too few real images | The embedder and tag reader underfit | Barcode-labelled clips: about 300 products per 2-hour visit (estimate) |
| Store refuses filming | No real store data | Ask early; fall back to products bought and filmed at home on a shelf mock-up |
| Privacy complaint | Cameras in a shop are sensitive | Faces blurred on device, no upload by default, clear recording light |
| Battery and heat on glasses | Continuous video drains small batteries | Scan on a tap or a pause, not always-on video |
| No depth on the final glasses | Our held-item logic leans on depth | Keep the RGB-only fallback (biggest box, lower centre) and score it separately |
| Dataset licences | Some public sets forbid commercial use | Use those only for research tests; train the product model on permissive or our own data |

Open decisions:

- [ ] Which glasses to test first (see the hardware section)
- [ ] Which Jaya Grocer outlet, and who asks for permission
- [ ] How many real products for the first gallery: 200 to start, 2,000 for a beta?
- [ ] Labelling tool: CVAT or Label Studio (both open source)
- [ ] Whether the first beta is Malaysia only

## Sources

Pages opened on 16 Nov 2026.

- Datasets: [SKU-110K](https://github.com/eg4000/SKU110K_CVPR19) and its [paper](https://arxiv.org/pdf/1904.00853) · [RP2K](https://arxiv.org/abs/2006.12634) · [Products-10K](https://products-10k.github.io/challenge.html) · [Grocery Store Dataset](https://github.com/marcusklasson/GroceryStoreDataset) · [Freiburg Groceries](https://github.com/PhilJd/freiburg_groceries_dataset) · [RPC](https://rpc-dataset.github.io/) · [MVTec D2S](https://www.mvtec.com/company/research/datasets/mvtec-d2s) · [Open Food Facts terms of use](https://world.openfoodfacts.org/terms-of-use) · [Open Prices data](https://openfoodfacts.github.io/open-prices/guides/data/) · [OFF price-tag-detection](https://huggingface.co/datasets/openfoodfacts/price-tag-detection) · [Retail-786k](https://arxiv.org/html/2309.17164v2)
- Glasses: [Meta Wearables Device Access Toolkit](https://wearables.developer.meta.com/docs/develop/dat/build-overview/) · [Meta wearables FAQ](https://developers.meta.com/wearables/faq/) · [Ray-Ban and Oakley Meta Malaysia launch](https://soyacincau.com/2026/10/14/ray-ban-oakley-meta-ai-glasses-malaysia-launch-price-specs/) · [Meta Ray-Ban Display markets](https://www.roadtovr.com/meta-ray-ban-display-int-launch-connect-2026/) · [Android XR camera access](https://developer.android.com/develop/xr/jetpack-xr-sdk/access-hardware-projected-context) · [Android XR glasses release](https://roadtovr.com/android-xr-warby-parker-gentle-monster-release-date-price/) · [Brilliant Labs Halo SDK](https://docs.brilliant.xyz/halo/halo-sdk-lua/) · [Even Realities Hub](https://hub.evenrealities.com/docs/get-started/overview) · [Snap Spectacles camera module](https://developers.snap.com/spectacles/about-spectacles-features/apis/camera-module) · [Rokid Malaysia dealer](https://pcimage.com.my/?p=11265)
- RealSense: [librealsense Android notes](https://github.com/realsenseai/librealsense/blob/master/doc/android.md) · [librealsense releases](https://github.com/realsenseai/librealsense/releases)
- Prices: Lotus's product pages, for example [Milo soft pack 900 g](https://www.lotuss.com.my/p/milo-softpack-900g-75149126); Jaya Grocer prices from search listings of jayagrocer.com (the site blocked automated reading). Full list in `vision/data/real_prices.json` in the repo.
