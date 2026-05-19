from pathlib import Path
from PIL import Image, ImageFilter, ImageChops, ImageDraw

files = [
    "Assets/Images/drone_top.png",
    "Assets/Images/rover_top.png",
    "Assets/Images/debug_bug.png",
    "Assets/Images/system_screen.png",
]

OUTLINE_ALPHA = 128

root = Path(__file__).resolve().parent.parent

for rel in files:
    p = root / rel
    # prefer .bak as source if present
    bak = p.with_suffix(p.suffix + ".bak")
    src = bak if bak.exists() else p
    if not src.exists():
        print(f"Skipping missing source: {src}")
        continue

    img = Image.open(src).convert("RGBA")
    width, height = img.size

    alpha = img.split()[3]
    # Increase dilation kernel to produce a thicker outline (was 3)
    dilated = alpha.filter(ImageFilter.MaxFilter(13))
    outline_mask = ImageChops.subtract(dilated, alpha)

    outline = Image.new("RGBA", img.size, (255, 255, 255, 0))
    outline.putalpha(outline_mask)

    # reduce opacity
    if OUTLINE_ALPHA < 255:
        a = outline.split()[3].point(lambda p: int(p * (OUTLINE_ALPHA / 255.0)))
        outline.putalpha(a)

    composed = Image.alpha_composite(outline, img)

    out_path = p.with_name(p.stem + "_outlined" + p.suffix)
    composed.save(out_path)
    print(f"Wrote outlined: {out_path}")
