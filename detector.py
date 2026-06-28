from __future__ import annotations

from collections import deque

import numpy as np

from .kernels import box_blur


def otsu_threshold(image: np.ndarray) -> float:
    arr = np.clip(image, 0.0, 1.0)
    hist, _ = np.histogram(arr.ravel(), bins=256, range=(0.0, 1.0))
    total = arr.size
    sum_total = float(np.dot(np.arange(256), hist))
    sum_b = 0.0
    w_b = 0
    best_var = -1.0
    threshold = 90
    for idx, count in enumerate(hist):
        w_b += int(count)
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += float(idx * count)
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f
        between = w_b * w_f * (m_b - m_f) ** 2
        if between > best_var:
            best_var = between
            threshold = idx
    return float(threshold) / 255.0


def segment_objects(image: np.ndarray, min_area: int = 18, mode: str = "generic") -> np.ndarray:
    smooth = box_blur(image, radius=1)
    if mode == "ct":
        thr = max(float(np.percentile(smooth, 96.5)), otsu_threshold(smooth) * 1.05)
        max_area = int(image.size * 0.018)
    else:
        thr = max(otsu_threshold(smooth), float(np.percentile(smooth, 65)))
        max_area = int(image.size * 0.20)
    mask = smooth > thr
    boxes = connected_components(mask, min_area=min_area)[1]
    clean = np.zeros_like(mask, dtype=np.float32)
    for x0, y0, x1, y1 in boxes:
        area = int(mask[y0 : y1 + 1, x0 : x1 + 1].sum())
        if area > max_area:
            continue
        clean[y0 : y1 + 1, x0 : x1 + 1] = np.maximum(
            clean[y0 : y1 + 1, x0 : x1 + 1], mask[y0 : y1 + 1, x0 : x1 + 1]
        )
    return clean.astype(np.float32)


def connected_components(mask: np.ndarray, min_area: int = 18) -> tuple[list[int], list[tuple[int, int, int, int]]]:
    m = np.asarray(mask).astype(bool)
    h, w = m.shape
    seen = np.zeros_like(m, dtype=bool)
    areas: list[int] = []
    boxes: list[tuple[int, int, int, int]] = []
    for y in range(h):
        for x in range(w):
            if not m[y, x] or seen[y, x]:
                continue
            q: deque[tuple[int, int]] = deque([(x, y)])
            seen[y, x] = True
            area = 0
            x0 = x1 = x
            y0 = y1 = y
            while q:
                cx, cy = q.popleft()
                area += 1
                x0 = min(x0, cx)
                x1 = max(x1, cx)
                y0 = min(y0, cy)
                y1 = max(y1, cy)
                for nx, ny in ((cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)):
                    if 0 <= nx < w and 0 <= ny < h and m[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        q.append((nx, ny))
            if area >= min_area:
                areas.append(area)
                boxes.append((x0, y0, x1, y1))
    return areas, boxes


def detect(image: np.ndarray, min_area: int = 18, mode: str = "generic") -> dict:
    mask = segment_objects(image, min_area=min_area, mode=mode)
    areas, boxes = connected_components(mask > 0, min_area=min_area)
    return {
        "mask": mask,
        "boxes": boxes,
        "count": len(boxes),
        "areas": areas,
        "mean_area": float(np.mean(areas)) if areas else 0.0,
    }
