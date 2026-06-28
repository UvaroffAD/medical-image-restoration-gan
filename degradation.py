from __future__ import annotations

import numpy as np

from .kernels import fft_convolve, gaussian_kernel


def add_poisson_noise(image: np.ndarray, photons: float = 25.0, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    scaled = np.clip(image, 0.0, 1.0) * photons
    noisy = rng.poisson(scaled) / max(photons, 1.0)
    return np.clip(noisy.astype(np.float32), 0.0, 1.0)


def add_gaussian_noise(image: np.ndarray, sigma: float = 0.03, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noisy = image + rng.normal(0.0, sigma, image.shape)
    return np.clip(noisy.astype(np.float32), 0.0, 1.0)


def add_stripes(image: np.ndarray, strength: float = 0.08, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    h, w = image.shape
    columns = rng.normal(0.0, strength, (1, w)).astype(np.float32)
    low_freq = np.sin(np.linspace(0, np.pi * 6, w, dtype=np.float32))[None, :] * strength * 0.5
    return np.clip(image + columns + low_freq, 0.0, 1.0)


def degrade_microscopy(
    clean: np.ndarray,
    photons: float = 25.0,
    blur_sigma: float = 1.8,
    seed: int = 13,
) -> np.ndarray:
    blurred = fft_convolve(clean, gaussian_kernel(17, blur_sigma))
    return add_poisson_noise(blurred, photons=photons, seed=seed)


def degrade_ct(
    clean: np.ndarray,
    photons: float = 80.0,
    blur_sigma: float = 1.1,
    seed: int = 21,
) -> np.ndarray:
    blurred = fft_convolve(clean, gaussian_kernel(13, blur_sigma))
    noisy = add_poisson_noise(blurred, photons=photons, seed=seed)
    return add_stripes(noisy, strength=0.055, seed=seed + 1)
