from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .kernels import box_blur
from .restoration import edge_map, suppress_stripes, unsharp_prior


FEATURE_NAMES = [
    "bias",
    "input",
    "blur_r1",
    "blur_r3",
    "blur_r6",
    "detail_r1",
    "detail_r3",
    "edge_map",
    "destriped",
    "destriped_unsharp",
]


def default_model_path(domain: str) -> Path:
    root = Path(__file__).resolve().parent.parent
    return root / "models" / f"{domain}_prior.npz"


def feature_stack(image: np.ndarray) -> np.ndarray:
    img = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
    b1 = box_blur(img, radius=1)
    b3 = box_blur(img, radius=3)
    b6 = box_blur(img, radius=6)
    destriped = suppress_stripes(img, strength=0.45)
    destriped_unsharp = unsharp_prior(destriped, amount=0.12, radius=3)
    features = [
        np.ones_like(img, dtype=np.float32),
        img,
        b1,
        b3,
        b6,
        img - b1,
        img - b3,
        edge_map(img),
        destriped,
        destriped_unsharp,
    ]
    return np.stack(features, axis=-1).astype(np.float32)


@dataclass
class TrainedPrior:
    domain: str
    weights: np.ndarray
    metadata: dict

    @classmethod
    def load(cls, domain: str, path: str | Path | None = None) -> "TrainedPrior":
        model_path = Path(path) if path else default_model_path(domain)
        if not model_path.exists():
            raise FileNotFoundError(f"Trained model not found: {model_path}")
        data = np.load(model_path, allow_pickle=False)
        metadata = {
            "domain": str(data["domain"]),
            "samples": int(data["samples"]),
            "size": int(data["size"]),
            "ridge": float(data["ridge"]),
            "features": [str(x) for x in data["features"]],
        }
        return cls(domain=metadata["domain"], weights=data["weights"].astype(np.float32), metadata=metadata)

    def predict(self, image: np.ndarray) -> np.ndarray:
        features = feature_stack(image)
        flat = features.reshape(-1, features.shape[-1])
        restored = flat @ self.weights
        return np.clip(restored.reshape(image.shape).astype(np.float32), 0.0, 1.0)


def fit_linear_prior(samples: list[tuple[np.ndarray, np.ndarray]], ridge: float = 0.001) -> np.ndarray:
    feature_count = len(FEATURE_NAMES)
    xtx = np.zeros((feature_count, feature_count), dtype=np.float64)
    xty = np.zeros((feature_count,), dtype=np.float64)
    for degraded, clean in samples:
        features = feature_stack(degraded).reshape(-1, feature_count).astype(np.float64)
        target = clean.reshape(-1).astype(np.float64)
        xtx += features.T @ features
        xty += features.T @ target
    xtx += np.eye(feature_count, dtype=np.float64) * float(ridge)
    weights = np.linalg.solve(xtx, xty)
    return weights.astype(np.float32)


def save_prior(
    path: str | Path,
    domain: str,
    weights: np.ndarray,
    samples: int,
    size: int,
    ridge: float,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        domain=np.array(domain),
        weights=weights.astype(np.float32),
        features=np.array(FEATURE_NAMES),
        samples=np.array(samples, dtype=np.int32),
        size=np.array(size, dtype=np.int32),
        ridge=np.array(ridge, dtype=np.float32),
    )


def predict_with_trained_prior(image: np.ndarray, domain: str | None = None) -> np.ndarray:
    selected = domain if domain in {"microscopy", "ct"} else "microscopy"
    try:
        model = TrainedPrior.load(selected)
    except FileNotFoundError:
        fallback = "ct" if selected == "microscopy" else "microscopy"
        model = TrainedPrior.load(fallback)
    return model.predict(image)
