"""Sentence-to-trajectory and remix utilities."""

from __future__ import annotations

import numpy as np

from .model import ShanziModel


def sentence_path(sentence: str, model: ShanziModel) -> np.ndarray:
    """Convert a sentence to an embedding path (unknown chars are skipped)."""
    vectors: list[np.ndarray] = []
    for char in sentence:
        if char.isspace():
            continue
        try:
            vectors.append(model.embedding(char))
        except KeyError:
            continue
    if not vectors:
        raise ValueError("Sentence has no known characters in the model.")
    return np.vstack(vectors).astype(np.float32)


def smooth_path(
    points: np.ndarray,
    *,
    samples_per_segment: int = 8,
    tension: float = 0.5,
) -> np.ndarray:
    """Apply Catmull-Rom style smoothing over a discrete path."""
    if points.ndim != 2:
        raise ValueError("points must be a 2D matrix [steps, dim].")
    if samples_per_segment <= 0:
        raise ValueError("samples_per_segment must be positive.")
    if len(points) < 2:
        return points.copy()

    smoothed: list[np.ndarray] = []
    n = len(points)
    for i in range(n - 1):
        p0 = points[i - 1] if i > 0 else points[i]
        p1 = points[i]
        p2 = points[i + 1]
        p3 = points[i + 2] if i + 2 < n else points[i + 1]

        m1 = tension * (p2 - p0)
        m2 = tension * (p3 - p1)

        for t in np.linspace(0.0, 1.0, samples_per_segment, endpoint=False):
            t2 = t * t
            t3 = t2 * t
            h00 = 2 * t3 - 3 * t2 + 1
            h10 = t3 - 2 * t2 + t
            h01 = -2 * t3 + 3 * t2
            h11 = t3 - t2
            point = h00 * p1 + h10 * m1 + h01 * p2 + h11 * m2
            smoothed.append(point)

    smoothed.append(points[-1])
    return np.vstack(smoothed).astype(np.float32)


def resample_by_arclength(path: np.ndarray, n_samples: int) -> np.ndarray:
    """Resample vectors to equally spaced points along arc length."""
    if path.ndim != 2:
        raise ValueError("path must be a 2D matrix [steps, dim].")
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if len(path) == 0:
        raise ValueError("path cannot be empty.")
    if len(path) == 1:
        return np.repeat(path, n_samples, axis=0)

    deltas = np.diff(path, axis=0)
    segment_lengths = np.linalg.norm(deltas, axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    total = float(cumulative[-1])
    if total == 0.0:
        return np.repeat(path[:1], n_samples, axis=0)

    targets = np.linspace(0.0, total, n_samples)
    output = np.zeros((n_samples, path.shape[1]), dtype=np.float32)
    for dim_idx in range(path.shape[1]):
        output[:, dim_idx] = np.interp(targets, cumulative, path[:, dim_idx])
    return output


def jitter_path(path: np.ndarray, *, jitter_std: float = 0.03, seed: int | None = None) -> np.ndarray:
    """Add Gaussian jitter to each sampled point."""
    if jitter_std < 0:
        raise ValueError("jitter_std must be non-negative.")
    if jitter_std == 0.0:
        return path.copy()
    spread = float(np.std(path))
    if spread == 0.0:
        spread = 1.0
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, jitter_std * spread, size=path.shape)
    return (path + noise).astype(np.float32)


def remix_sentence(
    sentence: str,
    model: ShanziModel,
    *,
    samples_per_segment: int = 8,
    output_length: int | None = None,
    jitter_std: float = 0.03,
    seed: int | None = None,
    hanzi_only: bool = True,
) -> str:
    """
    Remix a sentence by traversing smoothed semantic space and re-decoding.
    """
    path = sentence_path(sentence, model)
    curve = smooth_path(path, samples_per_segment=samples_per_segment)
    if output_length is None:
        output_length = max(1, len([ch for ch in sentence if not ch.isspace()]))
    sampled = resample_by_arclength(curve, n_samples=output_length)
    jittered = jitter_path(sampled, jitter_std=jitter_std, seed=seed)
    return model.decode_vectors(jittered, hanzi_only=hanzi_only)
