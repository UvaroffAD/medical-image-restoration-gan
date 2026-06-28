from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from uvarov_restoration.degradation import degrade_ct, degrade_microscopy
from uvarov_restoration.metrics import psnr, ssim
from uvarov_restoration.synthetic import ct_sample, microscopy_sample
from uvarov_restoration.trained_prior import (
    default_model_path,
    fit_linear_prior,
    save_prior,
    TrainedPrior,
)


def build_samples(domain: str, count: int, size: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    samples = []
    for idx in range(count):
        sample_seed = int(rng.integers(1, 1_000_000))
        if domain == "ct":
            clean, _ = ct_sample(size=size, seed=sample_seed)
            degraded = degrade_ct(
                clean,
                photons=float(rng.uniform(45, 140)),
                blur_sigma=float(rng.uniform(0.8, 1.5)),
                seed=sample_seed + 17,
            )
        else:
            clean, _ = microscopy_sample(
                size=size,
                objects=int(rng.integers(18, 46)),
                seed=sample_seed,
            )
            degraded = degrade_microscopy(
                clean,
                photons=float(rng.uniform(8, 90)),
                blur_sigma=float(rng.uniform(1.1, 2.4)),
                seed=sample_seed + 17,
            )
        samples.append((degraded.astype(np.float32), clean.astype(np.float32)))
    return samples


def evaluate(domain: str, model: TrainedPrior, size: int, seed: int) -> dict:
    if domain == "ct":
        clean, _ = ct_sample(size=size, seed=seed)
        degraded = degrade_ct(clean, photons=80.0, blur_sigma=1.1, seed=seed + 17)
    else:
        clean, _ = microscopy_sample(size=size, objects=32, seed=seed)
        degraded = degrade_microscopy(clean, photons=25.0, blur_sigma=1.8, seed=seed + 17)
    restored = model.predict(degraded)
    return {
        "input_psnr": psnr(clean, degraded),
        "input_ssim": ssim(clean, degraded),
        "restored_psnr": psnr(clean, restored),
        "restored_ssim": ssim(clean, restored),
    }


def train_domain(domain: str, count: int, size: int, ridge: float, seed: int, out_dir: Path) -> dict:
    print(f"Training {domain}: samples={count}, size={size}, ridge={ridge}")
    samples = build_samples(domain, count=count, size=size, seed=seed)
    weights = fit_linear_prior(samples, ridge=ridge)
    path = out_dir / f"{domain}_prior.npz"
    save_prior(path, domain=domain, weights=weights, samples=count, size=size, ridge=ridge)
    model = TrainedPrior.load(domain, path)
    metrics = evaluate(domain, model, size=size, seed=seed + 999)
    result = {
        "domain": domain,
        "path": str(path.resolve()),
        "samples": count,
        "size": size,
        "ridge": ridge,
        "weights": weights.tolist(),
        "metrics": metrics,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Train lightweight restoration prior models.")
    parser.add_argument("--domains", default="microscopy,ct")
    parser.add_argument("--samples", type=int, default=72)
    parser.add_argument("--size", type=int, default=192)
    parser.add_argument("--ridge", type=float, default=0.003)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out", default="models")
    args = parser.parse_args()

    out_dir = Path(args.out)
    results = []
    for domain in [x.strip() for x in args.domains.split(",") if x.strip()]:
        if domain not in {"microscopy", "ct"}:
            raise ValueError(f"Unsupported domain: {domain}")
        results.append(train_domain(domain, args.samples, args.size, args.ridge, args.seed, out_dir))

    report = out_dir / "training_report.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {report.resolve()}")
    for domain in ["microscopy", "ct"]:
        print(f"Default path for {domain}: {default_model_path(domain)}")


if __name__ == "__main__":
    main()
