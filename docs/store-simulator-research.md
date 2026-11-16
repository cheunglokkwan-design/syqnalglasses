# Store simulator: options and choice

Nov 16, 2026

## Recommendation

Use our own browser simulator ([`prototypes/store-sim-3d`](../prototypes/store-sim-3d/)) now, for the walk-and-pick experience and for quick RGB-D test frames. Move to NVIDIA Isaac Sim with Replicator later, only if we need photoreal training data at scale, because it has ready-made RealSense camera models and a depth sensor simulator but needs an RTX GPU workstation.

Nothing open source today gives a "walk the aisles and pick up products" experience with RealSense-style depth and per-product labels out of the box. The closest projects are built for robot arms rather than a shopper's eye view.

## What exists

| Option | What it is | Good for us | Drawbacks |
| --- | --- | --- | --- |
| [Sari Sandbox](https://arxiv.org/html/2508.00400v1) ([code](https://github.com/upeee/sari-sandbox-env), Apache-2.0) | Unity 6 convenience store, 250 grocery items with barcodes, 3 layouts, first-person avatar with two hands, VR (Quest 2), Python API, self-checkout scanner | Closest to the 1:1 shopper experience; VR walking and picking | Windows build; RGB screenshots only, no depth or label output; small store (sari-sari shop); about 10 stars, research code |
| [MarketGen](https://arxiv.org/html/2511.21161v1) | Isaac Sim + Unreal pipeline that auto-generates supermarkets, 1,000+ goods in 10 categories, 100+ fixtures | Large, varied stores; RGB-D on robots | No public code found; built around robot tasks, not first-person shopping |
| [RoboBenchMart](https://arxiv.org/html/2511.10276v1) ([code](https://github.com/emb-ai/RoboBenchMart)) | ManiSkill3 retail benchmark, 370 products in 21 categories, ray-traced rendering | Assets on Hugging Face; good physics | Assets CC BY-NC 4.0 (no commercial use); cameras are 256 × 256 robot cams |
| [StandardSim](https://arxiv.org/abs/2202.02418) | Synthetic retail dataset (Unity), photoreal shelves with depth and segmentation | Ready-made depth + segmentation images | A fixed dataset, not a walkable simulator; not our products |
| [Isaac Sim + Replicator](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/sensors/isaacsim_sensors_camera_depth.html) | NVIDIA robotics simulator with synthetic data generation; ships RealSense D435/D455 USD camera assets and a stereo depth sensor simulator | Photoreal ray tracing, physically based depth noise, bounding boxes and masks built in | Needs an NVIDIA RTX GPU; no grocery store included, we would build or buy the scene and products |
| Habitat / AI2-THOR | Indoor embodied-AI simulators | Mature APIs, RGB-D | Homes and offices, no supermarket scenes or packaged goods |

## Why build our own browser store first

- It runs anywhere with a link, including the hosted Claude viewer, so the team can walk it on a laptop or phone today.
- We control the products: same fictional catalogue, brands, EAN-13 barcodes and RM prices as the rest of the project, so detector, barcode, OCR and price-compare code can all be tested end to end.
- Output matches the real camera: RealSense D435/D455 field of view and resolution, depth as 16-bit millimetres with the same scale as `rs2::depth_frame`, and a noise model that grows with distance squared and drops edge pixels. The code that later reads the real RealSense in the store can read simulator frames unchanged.
- Labels are exact and free: every pixel knows its SKU and instance, so we get boxes and masks for every frame without hand labelling.

## How this connects to the model work

The vision model report recommends RF-DETR for detection, a DINOv2/v3 embedding for exact SKU, PaddleOCR for tags and ZXing for barcodes. The simulator feeds that plan as follows:

1. **Pipeline and latency testing.** Run the full detect, identify, read, verify loop on simulator frames before any store visit.
2. **Detector pre-training and sanity checks.** Simulator boxes for "product" and "price tag" can be mixed with SKU-110K. Synthetic-only training will not transfer well to real shelves, so treat it as extra data, not a replacement.
3. **Depth use.** Test the "cut away the background shelf, measure distance to the held item" logic on simulator depth with RealSense-style noise.
4. **Real-store check.** Record the same kind of frames with the RealSense in the store, and compare model accuracy on sim and real frames to see how big the gap is.

## Next steps

- Replace generated packs with photographed products (front, back, sides) for the SKUs we test in the real store, so the embedding model sees real artwork.
- If the gap between sim and real is large, port the store layout to Isaac Sim and use its RealSense camera model and ray tracing for photoreal frames.
- Optional: VR mode (WebXR) for a headset walk-through.
