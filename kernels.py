from __future__ import annotations

import numpy as np


def gaussian_kernel(size: int = 17, sigma: float = 2.2) -> np.ndarray:
    if size % 2 == 0:
        size += 1
    radius = size // 2
    y, x = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    kernel = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
    kernel_sum = float(kernel.sum())
    if kernel_sum <= 0:
        return np.ones((1, 1), dtype=np.float32)
    return (kernel / kernel_sum).astype(np.float32)


def motion_kernel(size: int = 15, angle: float = 0.0) -> np.ndarray:
    if size % 2 == 0:
        size += 1
    kernel = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    angle_rad = np.deg2rad(angle)
    cos_a = float(np.cos(angle_rad))
    sin_a = float(np.sin(angle_rad))
    for i in range(size):
        offset = i - center
        x = int(round(center + offset * cos_a))
        y = int(round(center + offset * sin_a))
        if 0 <= x < size and 0 <= y < size:
            kernel[y, x] = 1.0
    kernel /= max(float(kernel.sum()), 1.0)
    return kernel


def pad_kernel_for_fft(kernel: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    kh, kw = kernel.shape
    out = np.zeros(shape, dtype=np.float32)
    out[:kh, :kw] = kernel
    out = np.roll(out, -kh // 2, axis=0)
    out = np.roll(out, -kw // 2, axis=1)
    return out


def fft_convolve(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    kernel_pad = pad_kernel_for_fft(np.asarray(kernel, dtype=np.float32), image.shape)
    result = np.fft.ifft2(np.fft.fft2(image) * np.fft.fft2(kernel_pad)).real
    return np.clip(result.astype(np.float32), 0.0, 1.0)


def box_blur(image: np.ndarray, radius: int = 3) -> np.ndarray:
    radius = max(1, int(radius))
    size = radius * 2 + 1
    kernel = np.ones((size, size), dtype=np.float32) / float(size * size)
    return fft_convolve(image, kernel)
