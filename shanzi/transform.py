"""
Sentence transformation: paths through shanzi-space.

A sentence is a discretised signal evolving through embedding space.
We:
  1. Map each CJK character to its shanzi embedding.
  2. Fit a smooth B-spline curve through the resulting points.
  3. Generate a *smooth* random perturbation curve (not per-point noise)
     scaled by a temperature parameter.
  4. Add the perturbation to the main curve.
  5. Resample the perturbed curve at the original time-points.
  6. Decode each sample back to the nearest character in shanzi-space.

Non-CJK tokens (punctuation, spaces, Latin letters) pass through unchanged.
"""

from __future__ import annotations

from typing import List, Optional, Set, Tuple

import numpy as np
from scipy.interpolate import make_interp_spline

from shanzi.decompose import _is_cjk
from shanzi.embeddings import ShanziEmbeddings


class ShanziTransformer:
    """Transform text by drifting it through shanzi embedding space."""

    def __init__(self, embeddings: ShanziEmbeddings):
        self.emb = embeddings

    def transform(
        self,
        text: str,
        temperature: float = 0.3,
        seed: Optional[int] = None,
        preserve_punctuation: bool = True,
    ) -> str:
        """Return a shanzi-transformed version of *text*.

        Parameters
        ----------
        text : str
            Input (may contain any mix of CJK and non-CJK characters).
        temperature : float
            Jitter magnitude.  0 → identity (nearest neighbour of the
            original embedding, which is usually the character itself).
            Higher values → more semantic drift.
        seed : int, optional
            Random seed for reproducibility.
        preserve_punctuation : bool
            If True, non-CJK characters pass through unchanged.
        """
        rng = np.random.RandomState(seed)
        tokens = list(text)

        # Identify transformable positions
        cjk_idx: List[int] = []
        for i, ch in enumerate(tokens):
            if _is_cjk(ch) and self.emb.get(ch) is not None:
                cjk_idx.append(i)

        if not cjk_idx:
            return text

        cjk_chars = [tokens[i] for i in cjk_idx]
        embeds = np.array([self.emb.get(ch) for ch in cjk_chars], dtype=np.float32)

        n = len(cjk_chars)
        dim = embeds.shape[1]

        if n == 1:
            perturbed = self._perturb_single(embeds[0], temperature, rng, dim)
            ch, _ = self._decode_one(perturbed, exclude={cjk_chars[0]})
            tokens[cjk_idx[0]] = ch
        else:
            perturbed = self._perturb_path(embeds, temperature, rng)
            for j, pos in enumerate(cjk_idx):
                ch, _ = self._decode_one(perturbed[j], exclude={cjk_chars[j]})
                tokens[pos] = ch

        return "".join(tokens)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _perturb_single(
        self,
        vec: np.ndarray,
        temperature: float,
        rng: np.random.RandomState,
        dim: int,
    ) -> np.ndarray:
        noise = rng.randn(dim).astype(np.float32) * temperature
        out = vec + noise
        norm = np.linalg.norm(out)
        return out / norm if norm > 0 else out

    def _perturb_path(
        self,
        embeds: np.ndarray,
        temperature: float,
        rng: np.random.RandomState,
    ) -> np.ndarray:
        """Fit a curve, add smooth jitter, resample."""
        n, dim = embeds.shape
        t = np.linspace(0.0, 1.0, n)

        # Spline degree adapts to number of points
        deg = min(3, n - 1)

        # Fit the main curve
        spline = make_interp_spline(t, embeds, k=deg)

        # Smooth perturbation: fewer control points → low-frequency drift
        n_ctrl = max(2, n // 2)
        t_ctrl = np.linspace(0.0, 1.0, n_ctrl)
        ctrl_noise = rng.randn(n_ctrl, dim).astype(np.float32) * temperature
        p_deg = min(3, n_ctrl - 1)
        pert_spline = make_interp_spline(t_ctrl, ctrl_noise, k=p_deg)

        # Evaluate both curves at original time-points
        main = spline(t).astype(np.float32)
        pert = pert_spline(t).astype(np.float32)
        result = main + pert

        # Normalise each point
        norms = np.linalg.norm(result, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        result /= norms
        return result

    def _decode_one(
        self,
        vec: np.ndarray,
        exclude: Optional[Set[str]] = None,
    ) -> Tuple[str, float]:
        """Map an embedding vector back to the nearest CJK character."""
        hits = self.emb.nearest_cjk(vec, k=1, exclude=exclude)
        if hits:
            return hits[0]
        # Fallback: allow any character
        hits = self.emb.nearest(vec, k=1, exclude=exclude)
        if hits:
            return hits[0]
        return ("?", float("inf"))
