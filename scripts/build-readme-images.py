#!/usr/bin/env python3
"""Turn the raw PNG captures from scripts/capture-screenshots.mjs into the README images.

Usage:
    .venv/bin/python scripts/build-readme-images.py [spec.json] [--raw .tools/screenshots]

Reads the "outputs" block of the spec (default scripts/screenshots.json):
  * outputs.images: [{"from": "<shot name>", "to": "docs/images/<name>.jpg"}]
    Each PNG is downscaled (Lanczos) so its longest side is <= outputs.jpeg.maxSide and saved as a progressive JPEG at
    outputs.jpeg.quality; the quality is stepped down (not below 70) until the file fits outputs.jpeg.maxBytes.
    An optional "crop": [x0, y0, x1, y1] is given in CSS pixels of the capture viewport (spec "width") and cut from the
    full-resolution PNG first, so a cropped panel keeps the capture's device-pixel sharpness (e.g. 2x).
  * outputs.gif: {"to", "width", "colors", "maxBytes", "frames": [{"from", "ms"}]}
    An animated GIF that loops forever, each frame resized to `width` and given its own adaptive palette.

Needs Pillow (dev tooling only, not a backend dependency): `make screenshots` installs it into .venv if missing.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent


def load(raw: Path, name: str) -> Image.Image:
    path = raw / f"{name}.png"
    if not path.exists():
        sys.exit(f"missing capture {path} (run scripts/capture-screenshots.mjs first)")
    return Image.open(path).convert("RGB")


def fit(im: Image.Image, max_side: int) -> Image.Image:
    scale = max_side / max(im.size)
    if scale >= 1:
        return im
    return im.resize((round(im.width * scale), round(im.height * scale)), Image.LANCZOS)


def write_jpeg(im: Image.Image, dest: Path, quality: int, max_bytes: int) -> tuple[int, int]:
    q = quality
    while True:
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=q, optimize=True, progressive=True, subsampling="4:2:0")
        if buf.tell() <= max_bytes or q <= 70:
            break
        q -= 3
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(buf.getvalue())
    return buf.tell(), q


def write_gif(cfg: dict, raw: Path) -> None:
    dest = ROOT / cfg["to"]
    width = cfg.get("width", 960)
    colors = cfg.get("colors", 128)
    max_bytes = cfg.get("maxBytes", 6 * 1024 * 1024)
    frames_rgb = []
    for f in cfg["frames"]:
        im = load(raw, f["from"])
        frames_rgb.append(im.resize((width, round(im.height * width / im.width)), Image.LANCZOS))
    durations = [f["ms"] for f in cfg["frames"]]
    while True:
        frames = [im.quantize(colors=colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for im in frames_rgb]
        buf = io.BytesIO()
        frames[0].save(buf, "GIF", save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True, disposal=1)
        if buf.tell() <= max_bytes or colors <= 32:
            break
        colors //= 2
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(buf.getvalue())
    print(f"  {cfg['to']}: {len(frames)} frames, {width}x{frames_rgb[0].height}, {colors} colours, "
          f"{sum(durations) / 1000:.1f} s per loop, {buf.tell() / 1024:.0f} KB")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("spec", nargs="?", default=str(ROOT / "scripts" / "screenshots.json"))
    ap.add_argument("--raw", default=None, help="directory with the raw PNGs (default: spec.outDir)")
    args = ap.parse_args()
    spec = json.loads(Path(args.spec).read_text())
    raw = (ROOT / (args.raw or spec.get("outDir", ".tools/screenshots"))).resolve()
    out = spec["outputs"]
    jcfg = out.get("jpeg", {})
    for item in out.get("images", []):
        im = load(raw, item["from"])
        if "crop" in item:
            k = im.width / spec.get("width", im.width)
            x0, y0, x1, y1 = (round(v * k) for v in item["crop"])
            im = im.crop((x0, y0, x1, y1))
        im = fit(im, jcfg.get("maxSide", 1920))
        size, q = write_jpeg(im, ROOT / item["to"], jcfg.get("quality", 85), jcfg.get("maxBytes", 450 * 1024))
        print(f"  {item['to']}: {im.width}x{im.height}, q{q}, {size / 1024:.0f} KB")
    if "gif" in out:
        write_gif(out["gif"], raw)


if __name__ == "__main__":
    main()
