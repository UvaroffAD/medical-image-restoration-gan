from __future__ import annotations

import math

import numpy as np

from .degradation import degrade_ct, degrade_microscopy
from .kernels import fft_convolve, gaussian_kernel


def microscopy_sample(size: int = 384, objects: int = 34, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    image = np.zeros((size, size), dtype=np.float32) + 0.035
    mask = np.zeros_like(image, dtype=np.float32)
    yy, xx = np.mgrid[0:size, 0:size]

    for _ in range(objects):
        cx = rng.uniform(24, size - 24)
        cy = rng.uniform(24, size - 24)
        rx = rng.uniform(7, 20)
        ry = rng.uniform(7, 18)
        angle = rng.uniform(0, math.pi)
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        x = (xx - cx) * cos_a + (yy - cy) * sin_a
        y = -(xx - cx) * sin_a + (yy - cy) * cos_a
        blob = ((x / rx) ** 2 + (y / ry) ** 2) <= 1.0
        ring = ((x / (rx * 0.62)) ** 2 + (y / (ry * 0.62)) ** 2) <= 1.0
        intensity = rng.uniform(0.35, 0.85)
        image[blob] += intensity * 0.55
        image[ring] += intensity * 0.25
        mask[blob] = 1.0

    image = fft_convolve(np.clip(image, 0.0, 1.0), gaussian_kernel(5, 0.75))
    image += rng.normal(0.0, 0.012, image.shape).astype(np.float32)
    image = np.clip(image, 0.0, 1.0)
    return image, mask


def ct_sample(size: int = 384, seed: int = 8) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[-1:1 : complex(size), -1:1 : complex(size)]
    torso = ((xx / 0.78) ** 2 + (yy / 0.95) ** 2) <= 1.0
    image = np.zeros((size, size), dtype=np.float32) + 0.02
    image[torso] = 0.32

    spine = ((xx / 0.13) ** 2 + ((yy - 0.45) / 0.16) ** 2) <= 1.0
    left_bone = (((xx + 0.32) / 0.08) ** 2 + ((yy - 0.1) / 0.18) ** 2) <= 1.0
    right_bone = (((xx - 0.32) / 0.08) ** 2 + ((yy - 0.1) / 0.18) ** 2) <= 1.0
    image[spine | left_bone | right_bone] = 0.78

    organ1 = (((xx + 0.26) / 0.24) ** 2 + ((yy + 0.10) / 0.18) ** 2) <= 1.0
    organ2 = (((xx - 0.30) / 0.22) ** 2 + ((yy + 0.16) / 0.18) ** 2) <= 1.0
    image[organ1 | organ2] = 0.44

    mask = np.zeros_like(image, dtype=np.float32)
    for _ in range(3):
        cx = rng.uniform(-0.35, 0.35)
        cy = rng.uniform(-0.25, 0.35)
        r = rng.uniform(0.035, 0.065)
        lesion = ((xx - cx) ** 2 + (yy - cy) ** 2) <= r * r
        image[lesion] = 0.92
        mask[lesion] = 1.0

    image = fft_convolve(image, gaussian_kernel(5, 0.85))
    return np.clip(image, 0.0, 1.0), mask


def make_sample(preset: str = "microscopy") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if preset == "ct":
        clean, mask = ct_sample()
        degraded = degrade_ct(clean)
    else:
        clean, mask = microscopy_sample()
        degraded = degrade_microscopy(clean)
    return clean, degraded, mask
