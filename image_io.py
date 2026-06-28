from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def to_float01(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    if arr.size == 0:
        return arr
    if arr.max() > 1.5:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def to_uint8(array: np.ndarray) -> np.ndarray:
    arr = np.clip(np.asarray(array, dtype=np.float32), 0.0, 1.0)
    return (arr * 255.0 + 0.5).astype(np.uint8)


def load_grayscale(path: str | Path, max_side: int = 768) -> np.ndarray:
    image = Image.open(path)
    image = ImageOps.exif_transpose(image).convert("L")
    if max(image.size) > max_side:
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return to_float01(np.array(image))


def save_grayscale(path: str | Path, array: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(to_uint8(array), mode="L").save(path)


def overlay_boxes(base: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> Image.Image:
    img = Image.fromarray(to_uint8(base), mode="L").convert("RGB")
    if not boxes:
        return img
    from PIL import ImageDraw

    draw = ImageDraw.Draw(img)
    for x0, y0, x1, y1 in boxes:
        draw.rectangle((x0, y0, x1, y1), outline=(255, 70, 40), width=2)
    return img


def save_overlay(path: str | Path, base: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    overlay_boxes(base, boxes).save(path)
