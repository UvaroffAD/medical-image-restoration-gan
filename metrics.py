from __future__ import annotations

import math

import numpy as np


def mse(a: np.ndarray, b: np.ndarray) -> float:
    diff = np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)
    return float(np.mean(diff * diff))


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    value = mse(a, b)
    if value <= 1e-12:
        return 99.0
    return float(20.0 * math.log10(1.0 / math.sqrt(value)))


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float32)
    y = np.asarray(b, dtype=np.float32)
    c1 = 0.01**2
    c2 = 0.03**2
    ux = float(x.mean())
    uy = float(y.mean())
    vx = float(x.var())
    vy = float(y.var())
    cov = float(((x - ux) * (y - uy)).mean())
    numerator = (2 * ux * uy + c1) * (2 * cov + c2)
    denominator = (ux * ux + uy * uy + c1) * (vx + vy + c2)
    return float(numerator / denominator) if denominator else 0.0


def iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = np.asarray(mask_a) > 0
    b = np.asarray(mask_b) > 0
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0


def summarize_restoration(restored: np.ndarray, clean: np.ndarray | None = None) -> dict:
    if clean is None:
        return {}
    return {
        "mse": mse(clean, restored),
        "psnr": psnr(clean, restored),
        "ssim": ssim(clean, restored),
    }
