from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .kernels import box_blur, fft_convolve, gaussian_kernel, pad_kernel_for_fft


def wiener_deconvolution(image: np.ndarray, psf: np.ndarray, balance: float = 0.015) -> np.ndarray:
    y = np.asarray(image, dtype=np.float32)
    h = np.fft.fft2(pad_kernel_for_fft(psf, y.shape))
    yf = np.fft.fft2(y)
    denom = np.abs(h) ** 2 + float(balance)
    restored = np.fft.ifft2(np.conj(h) / denom * yf).real
    return np.clip(restored.astype(np.float32), 0.0, 1.0)


def richardson_lucy(image: np.ndarray, psf: np.ndarray, iterations: int = 15, eps: float = 1e-6) -> np.ndarray:
    observed = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
    estimate = np.full_like(observed, max(float(observed.mean()), 0.05), dtype=np.float32)
    psf_mirror = psf[::-1, ::-1]
    for _ in range(max(1, int(iterations))):
        conv = fft_convolve(estimate, psf) + eps
        relative_blur = observed / conv
        estimate *= fft_convolve(relative_blur, psf_mirror)
        estimate = np.clip(estimate, 0.0, 1.0)
    return estimate


def unsharp_prior(image: np.ndarray, amount: float = 0.35, radius: int = 2) -> np.ndarray:
    smooth = box_blur(image, radius=radius)
    detail = image - smooth
    return np.clip(image + amount * detail, 0.0, 1.0)


def edge_map(image: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(image.astype(np.float32))
    mag = np.sqrt(gx * gx + gy * gy)
    if float(mag.max()) > 0:
        mag = mag / float(mag.max())
    return np.clip(mag, 0.0, 1.0)


def suppress_stripes(image: np.ndarray, strength: float = 0.85) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    column_profile = np.median(arr, axis=0, keepdims=True)
    baseline = box_blur(np.repeat(column_profile, arr.shape[0], axis=0), radius=9)
    corrected = arr - strength * (baseline - float(np.median(arr)))
    return np.clip(corrected, 0.0, 1.0)


@dataclass
class PriorRegularizer:
    """Deterministic placeholder for a future trained GAN prior."""

    detail_amount: float = 0.22
    smooth_radius: int = 2
    semantic_weight: float = 0.20

    def __call__(self, image: np.ndarray) -> np.ndarray:
        edges = edge_map(image)
        enhanced = unsharp_prior(image, amount=self.detail_amount, radius=self.smooth_radius)
        blended = image * (1.0 - self.semantic_weight * edges) + enhanced * (self.semantic_weight * edges)
        return np.clip(blended, 0.0, 1.0)


def trl_hybrid(
    image: np.ndarray,
    psf: np.ndarray | None = None,
    iterations: int = 12,
    prior: PriorRegularizer | None = None,
) -> np.ndarray:
    psf = psf if psf is not None else gaussian_kernel(17, 1.8)
    prior = prior or PriorRegularizer(detail_amount=0.16, smooth_radius=2, semantic_weight=0.22)
    observed = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
    restored = wiener_deconvolution(observed, psf, balance=0.025)
    psf_mirror = psf[::-1, ::-1]
    eps = 1e-6
    for _ in range(max(1, int(iterations))):
        conv = fft_convolve(restored, psf) + eps
        update = restored * fft_convolve(observed / conv, psf_mirror)
        restored = np.clip(0.82 * restored + 0.18 * update, 0.0, 1.0)
        restored = prior(restored)
    stabilizer = wiener_deconvolution(observed, psf, balance=0.035)
    return np.clip(0.68 * restored + 0.32 * stabilizer, 0.0, 1.0)


def wf_hybrid(image: np.ndarray, psf: np.ndarray | None = None) -> np.ndarray:
    psf = psf if psf is not None else gaussian_kernel(17, 1.8)
    low_reg = wiener_deconvolution(image, psf, balance=0.005)
    high_reg = wiener_deconvolution(image, psf, balance=0.055)
    edges = edge_map(image)
    restored = low_reg * edges + high_reg * (1.0 - edges)
    return unsharp_prior(np.clip(restored, 0.0, 1.0), amount=0.18, radius=2)


def ct_hybrid(image: np.ndarray) -> np.ndarray:
    destriped = suppress_stripes(image, strength=0.45)
    denoised = 0.68 * destriped + 0.32 * box_blur(destriped, radius=1)
    return unsharp_prior(denoised, amount=0.12, radius=3)


def restore(image: np.ndarray, method: str, psf_sigma: float = 1.8, domain: str | None = None) -> np.ndarray:
    psf = gaussian_kernel(17, psf_sigma)
    method = method.lower().strip()
    if method == "wiener":
        return wiener_deconvolution(image, psf)
    if method in {"rl", "richardson_lucy"}:
        return richardson_lucy(image, psf, iterations=15)
    if method in {"trl_hybrid", "auto_microscopy"}:
        return trl_hybrid(image, psf=psf, iterations=12)
    if method == "wf_hybrid":
        return wf_hybrid(image, psf=psf)
    if method in {"ct_hybrid", "auto_ct"}:
        return ct_hybrid(image)
    if method in {"trained_prior", "trained"}:
        from .trained_prior import predict_with_trained_prior

        return predict_with_trained_prior(image, domain=domain)
    if method == "none":
        return np.asarray(image, dtype=np.float32)
    raise ValueError(f"Unknown restoration method: {method}")
