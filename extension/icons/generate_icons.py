#!/usr/bin/env python3
"""Generate LeakLock PNG icons for the Chrome extension."""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

OUT_DIR = Path(__file__).parent


def create_lock_icon(size: int) -> Image.Image:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Background: dark navy circle
    pad = max(1, size // 14)
    draw.ellipse(
        [pad, pad, size - pad - 1, size - pad - 1],
        fill=(15, 23, 42, 255),  # #0f172a
    )

    # Lock body dimensions
    bw = round(size * 0.44)
    bh = round(size * 0.28)
    bx = (size - bw) // 2
    by = round(size * 0.50)
    radius = max(2, size // 12)

    draw.rounded_rectangle(
        [bx, by, bx + bw, by + bh],
        radius=radius,
        fill=(255, 255, 255, 255),
    )

    # Keyhole: small dark ellipse in lock body center
    kw = max(2, round(size * 0.08))
    kx = size // 2
    ky = by + bh // 2
    draw.ellipse(
        [kx - kw, ky - kw, kx + kw, ky + kw],
        fill=(15, 23, 42, 200),
    )

    # Shackle (arc) above the lock body
    sw = round(bw * 0.55)
    sx = (size - sw) // 2
    shackle_top = round(size * 0.20)
    thick = max(2, round(size * 0.10))
    # Arc goes from bottom-left to bottom-right (180° → 0°)
    arc_bottom = by + thick
    draw.arc(
        [sx, shackle_top, sx + sw, arc_bottom],
        start=180,
        end=0,
        fill=(255, 255, 255, 255),
        width=thick,
    )

    return img


if __name__ == "__main__":
    for sz in (16, 48, 128):
        icon = create_lock_icon(sz)
        out = OUT_DIR / f"icon{sz}.png"
        icon.save(out, "PNG")
        print(f"  Created {out}")
