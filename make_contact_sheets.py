import sys
from pathlib import Path

from PIL import Image, ImageDraw


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

for folder in [Path("runs/microscopy_demo"), Path("runs/ct_demo")]:
    names = [
        "clean.png",
        "input.png",
        "trained_prior.png",
        "wiener.png",
        "rl.png",
        "wf_hybrid.png",
        "trl_hybrid.png",
        "ct_hybrid.png",
    ]
    imgs = []
    for name in names:
        path = folder / name
        if path.exists():
            image = Image.open(path).convert("RGB")
            image.thumbnail((220, 220))
            imgs.append((name, image.copy()))
    sheet = Image.new("RGB", (240 * len(imgs), 270), "white")
    draw = ImageDraw.Draw(sheet)
    for idx, (name, image) in enumerate(imgs):
        x = idx * 240 + 10
        draw.text((x, 8), name, fill=(0, 0, 0))
        sheet.paste(image, (x, 38))
    out = folder / "contact_sheet.jpg"
    sheet.save(out, quality=92)
    print(out.resolve())
