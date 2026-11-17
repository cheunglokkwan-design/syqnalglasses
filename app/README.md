# Syqnal Shopper (phone app)

A one-page web app: point the phone camera at a product, or hold one up, and tap **Check**. It talks to the vision server (`vision/serve.py`, on the laptop now and on the Jetson later).

- **Subtle mode** (default): a clean camera view with a faint ring and one line of advice ("RM 0.70 cheaper at Lotus's", "Contains peanuts"), spoken aloud. Product photos appear only when they help: "which one is it?" when the match is unclear, the product's photo when the match is uncertain, and cheaper or healthier picks under **More**.
- **Debug mode** (Settings, View): boxes, labels, scores, depth and timings for the team.
- **Cart** with explicit confirm: **Add**, or hold the mic and say "add this", "make that two", "remove", "cheaper", "less sugar", "show me", "what's the total". These are simple keyword commands until the LLM communicator is connected.
- **Settings:** server address, speech on or off, check automatically when the camera is still, budget, allergies.

No build step: `index.html` is the whole app.

**On-device mode (no server):** with no vision server address set, the phone scans barcodes live by itself (BarcodeDetector, or ZXing in WebAssembly on iPhone) and gives the verdict from the 20 real products. Point the camera at a barcode and the answer appears without tapping. Set a server address under Settings to use the full vision model instead.

**Open it on a phone now:** https://raw.githack.com/cheunglokkwan-design/syqnalglasses/ccr-72063d0e-ea52ph/app/index.html (served straight from this public repo over HTTPS, so the camera works).

## Try it on the laptop

```bash
.venv/Scripts/python vision/serve.py
```

Open http://localhost:8770/app/ (a localhost address may use the camera without HTTPS). **Use a photo instead** checks a picture file.

## Try it on your phone today

Phone browsers only allow the camera on HTTPS pages. A free Cloudflare quick tunnel gives the laptop server an HTTPS address:

```bash
winget install --id Cloudflare.cloudflared
```

```bash
cloudflared tunnel --url http://localhost:8770
```

It prints an address like `https://something.trycloudflare.com`. Open that address plus `/app/` on the phone. Anyone with the address can reach the server while the tunnel runs, so stop it (Ctrl+C) when you're done.

## Host it on Vercel

1. Sign in at vercel.com with GitHub, then **Add New**, **Project**, and import `cheunglokkwan-design/syqnalglasses`.
2. Set **Root Directory** to `app`, **Framework Preset** to **Other**, and leave the build command empty. Click **Deploy**.
3. Under Settings, Git, set the **Production Branch** to `vision-pipeline` (until it is merged into `main`).
4. Open the Vercel address on the phone, go to Settings, and enter the vision server's HTTPS address (the tunnel address for now; the Jetson's address later).

The Vercel page is HTTPS, so the vision server must be HTTPS too: browsers block an HTTPS page from calling a plain `http://` address.
