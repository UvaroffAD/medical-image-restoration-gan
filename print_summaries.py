import json
import sys
from pathlib import Path


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

for path in [Path("runs/microscopy_demo/summary.json"), Path("runs/ct_demo/summary.json")]:
    data = json.loads(path.read_text(encoding="utf-8"))
    print(f"--- {path}")
    print("distortion:", data["distortion"])
    for row in data["results"]:
        print(
            f"{row['name']}: objects={row['objects']}, "
            f"psnr={row.get('psnr', 0):.2f}, "
            f"ssim={row.get('ssim', 0):.3f}, "
            f"iou={row.get('iou', 0):.3f}"
        )
