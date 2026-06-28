from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from uvarov_restoration.image_io import load_grayscale
from uvarov_restoration.pipeline import PipelineConfig, run_pipeline
from uvarov_restoration.report import write_html_report
from uvarov_restoration.synthetic import make_sample


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Run Uvarov restoration MVP demo.")
    parser.add_argument("--input", help="Optional input image path.")
    parser.add_argument("--preset", choices=["microscopy", "ct"], default="microscopy")
    parser.add_argument("--out", default="runs/demo")
    parser.add_argument("--methods", default="none,trained_prior,wiener,rl,wf_hybrid,trl_hybrid,ct_hybrid")
    args = parser.parse_args()

    out_dir = Path(args.out)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    config = PipelineConfig(methods=methods, min_area=22 if args.preset == "microscopy" else 30)

    if args.input:
        image = load_grayscale(args.input)
        clean = None
        mask = None
        preset = "custom"
    else:
        clean, image, mask = make_sample(args.preset)
        preset = args.preset

    summary = run_pipeline(
        image=image,
        out_dir=out_dir,
        clean=clean,
        ground_truth_mask=mask,
        preset=preset,
        config=config,
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if preset == "microscopy":
        title = "Демо: микроскопия"
    elif preset == "ct":
        title = "Демо: спектральная КТ"
    else:
        title = "Результат восстановления"
    report = write_html_report(out_dir, summary, title=title)
    print(f"Wrote {report.resolve()}")


if __name__ == "__main__":
    main()
