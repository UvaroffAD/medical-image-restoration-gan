from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .classifier import classify_distortion
from .detector import detect
from .image_io import save_grayscale, save_overlay
from .metrics import iou, summarize_restoration
from .restoration import restore


@dataclass
class PipelineConfig:
    methods: list[str] = field(default_factory=lambda: ["none", "wiener", "rl", "wf_hybrid", "trl_hybrid"])
    psf_sigma: float = 1.8
    min_area: int = 18


def run_pipeline(
    image: np.ndarray,
    out_dir: str | Path,
    clean: np.ndarray | None = None,
    ground_truth_mask: np.ndarray | None = None,
    preset: str = "custom",
    config: PipelineConfig | None = None,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = config or PipelineConfig()
    save_grayscale(out_dir / "input.png", image)
    if clean is not None:
        save_grayscale(out_dir / "clean.png", clean)
    if ground_truth_mask is not None:
        save_grayscale(out_dir / "ground_truth_mask.png", ground_truth_mask)

    distortion = classify_distortion(image)
    methods = list(dict.fromkeys(config.methods + [distortion["recommended_method"]]))
    results = []
    detect_mode = "ct" if preset == "ct" else "generic"
    model_domain = "ct" if preset == "ct" else "microscopy" if preset == "microscopy" else None
    for method in methods:
        restored = restore(image, method, psf_sigma=config.psf_sigma, domain=model_domain)
        det = detect(restored, min_area=config.min_area, mode=detect_mode)
        image_file = f"{method}.png"
        mask_file = f"{method}_mask.png"
        overlay_file = f"{method}_overlay.png"
        save_grayscale(out_dir / image_file, restored)
        save_grayscale(out_dir / mask_file, det["mask"])
        save_overlay(out_dir / overlay_file, restored, det["boxes"])
        metrics = summarize_restoration(restored, clean)
        if ground_truth_mask is not None:
            metrics["iou"] = iou(det["mask"], ground_truth_mask)
        result = {
            "name": method,
            "image_file": image_file,
            "mask_file": mask_file,
            "overlay_file": overlay_file,
            "objects": det["count"],
            "mean_area": det["mean_area"],
            **metrics,
        }
        results.append(result)

    summary = {
        "preset": preset,
        "distortion": distortion,
        "results": results,
    }
    return summary
