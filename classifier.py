from __future__ import annotations

import numpy as np

from .kernels import box_blur


def classify_distortion(image: np.ndarray) -> dict:
    arr = np.asarray(image, dtype=np.float32)
    smooth = box_blur(arr, radius=2)
    residual = arr - smooth
    noise_score = float(np.std(residual))
    gy, gx = np.gradient(smooth)
    blur_score = float(np.mean(np.sqrt(gx * gx + gy * gy)))
    col_profile = arr.mean(axis=0)
    window = max(9, (arr.shape[1] // 24) | 1)
    kernel = np.ones(window, dtype=np.float32) / float(window)
    col_smooth = np.convolve(col_profile, kernel, mode="same")
    stripe_score = float(np.std(col_profile - col_smooth))

    if stripe_score > max(0.018, noise_score * 0.42):
        label = "stripe_artifacts"
        recommended = "ct_hybrid"
    elif noise_score > 0.075:
        label = "strong_noise"
        recommended = "trl_hybrid"
    elif blur_score < 0.018:
        label = "blur"
        recommended = "trl_hybrid"
    else:
        label = "mixed"
        recommended = "wf_hybrid"

    return {
        "label": label,
        "recommended_method": recommended,
        "noise_score": noise_score,
        "blur_score": blur_score,
        "stripe_score": stripe_score,
    }
