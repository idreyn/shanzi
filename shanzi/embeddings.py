"""
Shanzi embeddings: every character's meaning smeared through its components.

The core formula, applied in topological order:

    S(h) = normalise( W2V(h) + k · Σ S(part)  for part in components(h) )

where k = 0.67 (configurable).

Base embeddings W2V(h) come from one of:
  • StructuralSource — deterministic vectors seeded by Unicode codepoint
    (always available, no external data needed)
  • TextFileSource — word2vec-format text file
  • GensimSource — any gensim KeyedVectors model

The structural source is the default.  It produces meaningful geometry
because characters sharing radicals share vector subspace, and the shanzi
blending amplifies that overlap — which is exactly the point.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.spatial import cKDTree

from shanzi.decompose import Decomposer, _is_cjk, _is_cjk_primary

CACHE_DIR = Path(os.environ.get("SHANZI_CACHE", Path.home() / ".cache" / "shanzi"))


# ──────────────────────────────────────────────────────────────────────
# Base embedding sources
# ──────────────────────────────────────────────────────────────────────

class EmbeddingSource:
    """Abstract base: maps a character to a base (pre-shanzi) vector."""

    dim: int

    def get(self, char: str) -> Optional[np.ndarray]:
        raise NotImplementedError


class StructuralSource(EmbeddingSource):
    """Deterministic pseudo-random vector seeded by Unicode codepoint.

    Cheap, always available, and — crucially — consistent: the same
    character always maps to the same point.  Characters that share
    radicals share subspace structure once shanzi blending is applied.
    """

    def __init__(self, dim: int = 300):
        self.dim = dim
        self._cache: Dict[str, np.ndarray] = {}

    def get(self, char: str) -> np.ndarray:
        if char not in self._cache:
            rng = np.random.RandomState(ord(char) % (2**31 - 1))
            vec = rng.randn(self.dim).astype(np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            self._cache[char] = vec
        return self._cache[char]


class TextFileSource(EmbeddingSource):
    """Load character vectors from a word2vec-format text file.

    The file should start with ``N DIM`` on the first line, then one
    ``char v1 v2 ... vDIM`` per line.  Only single-character entries
    are loaded.
    """

    def __init__(self, path: str):
        self._vectors: Dict[str, np.ndarray] = {}
        self.dim = 0
        self._load(path)

    def _load(self, path: str):
        with open(path, "r", encoding="utf-8") as fh:
            header = fh.readline().split()
            self.dim = int(header[1])
            for line in fh:
                parts = line.rstrip().split(" ")
                word = parts[0]
                if len(word) == 1:
                    vec = np.array([float(x) for x in parts[1:]], dtype=np.float32)
                    self._vectors[word] = vec

    def get(self, char: str) -> Optional[np.ndarray]:
        return self._vectors.get(char)


class GensimSource(EmbeddingSource):
    """Load vectors via gensim (KeyedVectors).  Requires ``pip install gensim``."""

    def __init__(self, path: str):
        from gensim.models import KeyedVectors

        binary = path.endswith(".bin")
        self._kv = KeyedVectors.load_word2vec_format(path, binary=binary)
        self.dim = self._kv.vector_size

    def get(self, char: str) -> Optional[np.ndarray]:
        if char in self._kv:
            return self._kv[char].astype(np.float32)
        return None


# ──────────────────────────────────────────────────────────────────────
# Shanzi embeddings
# ──────────────────────────────────────────────────────────────────────

class ShanziEmbeddings:
    """Character embeddings with recursive component blending.

    After construction the instance exposes:
        .get(char)             → shanzi vector or None
        .nearest(vec, k=5)     → [(char, dist), ...]
        .vocab                 → set of characters with embeddings
    """

    def __init__(
        self,
        decomposer: Decomposer,
        source: Optional[EmbeddingSource] = None,
        k: float = 0.67,
        dim: int = 300,
        quiet: bool = False,
        cache: bool = True,
    ):
        self.decomposer = decomposer
        self.k = k
        self.dim = dim

        # Primary source (user-provided) + structural fallback
        self._primary = source
        self._fallback = StructuralSource(dim)
        if self._primary is not None:
            self.dim = self._primary.dim
            self._fallback = StructuralSource(self._primary.dim)

        self._shanzi: Dict[str, np.ndarray] = {}
        self._chars: List[str] = []
        self._matrix: Optional[np.ndarray] = None
        self._tree: Optional[cKDTree] = None

        loaded = False
        if cache:
            loaded = self._load_cache(quiet)
        if not loaded:
            self._build(quiet=quiet)
            if cache:
                self._save_cache(quiet)

    # ------------------------------------------------------------------
    # Caching
    # ------------------------------------------------------------------

    def _cache_key(self) -> str:
        """Deterministic hash of parameters that affect the embeddings."""
        n_chars = len(self.decomposer.all_chars())
        source_name = type(self._primary).__name__ if self._primary else "structural"
        blob = f"v2|{n_chars}|{self.dim}|{self.k}|{source_name}"
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def _cache_path(self) -> Path:
        return CACHE_DIR / f"shanzi_{self._cache_key()}.npz"

    def _load_cache(self, quiet: bool) -> bool:
        cp = self._cache_path()
        if not cp.exists():
            return False
        try:
            data = np.load(cp, allow_pickle=True)
            self._chars = list(data["chars"])
            # Stored as float16 for compactness; promote back to float32
            self._matrix = data["matrix"].astype(np.float32)
            # Re-normalise after float16 round-trip
            norms = np.linalg.norm(self._matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self._matrix /= norms
            self._shanzi = {
                ch: self._matrix[i] for i, ch in enumerate(self._chars)
            }
            self._tree = cKDTree(self._matrix)
            if not quiet:
                print(
                    f"[shanzi] loaded {len(self._chars):,} cached embeddings "
                    f"from {cp}",
                    file=sys.stderr,
                )
            return True
        except Exception as exc:
            if not quiet:
                print(f"[shanzi] cache load failed ({exc}), recomputing", file=sys.stderr)
            return False

    def _save_cache(self, quiet: bool):
        cp = self._cache_path()
        try:
            cp.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                cp,
                chars=np.array(self._chars, dtype=object),
                matrix=self._matrix.astype(np.float16),
            )
            if not quiet:
                size_mb = cp.stat().st_size / (1024 * 1024)
                print(
                    f"[shanzi] cached to {cp} ({size_mb:.1f} MB)",
                    file=sys.stderr,
                )
        except Exception as exc:
            if not quiet:
                print(f"[shanzi] cache save failed ({exc})", file=sys.stderr)

    # ------------------------------------------------------------------

    def _base(self, char: str) -> np.ndarray:
        """Get the base (pre-shanzi) embedding, falling back to structural."""
        if self._primary is not None:
            vec = self._primary.get(char)
            if vec is not None:
                return vec
        return self._fallback.get(char)

    def _build(self, quiet: bool = False):
        all_chars = self.decomposer.all_chars()
        order = self.decomposer.topological_sort(all_chars)

        if not quiet:
            print(
                f"[shanzi] computing embeddings for {len(order):,} characters "
                f"(dim={self.dim}, k={self.k}) ...",
                file=sys.stderr,
            )

        for char in order:
            base = self._base(char)
            parts = self.decomposer.get(char)
            if parts:
                part_sum = np.zeros(self.dim, dtype=np.float32)
                for p in parts:
                    if p in self._shanzi:
                        part_sum += self._shanzi[p]
                    else:
                        part_sum += self._base(p)
                vec = base + self.k * part_sum
            else:
                vec = base.copy()

            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            self._shanzi[char] = vec

        # Build nearest-neighbour index
        self._chars = list(self._shanzi.keys())
        self._matrix = np.vstack([self._shanzi[c] for c in self._chars])
        self._tree = cKDTree(self._matrix)

        if not quiet:
            print(
                f"[shanzi] done — {len(self._chars):,} vectors indexed.",
                file=sys.stderr,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def vocab(self) -> Set[str]:
        return set(self._chars)

    def get(self, char: str) -> Optional[np.ndarray]:
        return self._shanzi.get(char)

    def nearest(
        self,
        vec: np.ndarray,
        k: int = 1,
        exclude: Optional[Set[str]] = None,
    ) -> List[Tuple[str, float]]:
        """Return up to *k* (character, distance) pairs nearest to *vec*."""
        n_query = k + (len(exclude) if exclude else 0) + 10
        n_query = min(n_query, len(self._chars))
        dists, idxs = self._tree.query(vec.astype(np.float64), k=n_query)

        # cKDTree returns scalars when k=1 in older scipy
        if np.ndim(dists) == 0:
            dists = [float(dists)]
            idxs = [int(idxs)]

        results: List[Tuple[str, float]] = []
        for d, i in zip(dists, idxs):
            ch = self._chars[int(i)]
            if exclude and ch in exclude:
                continue
            results.append((ch, float(d)))
            if len(results) >= k:
                break
        return results

    def nearest_cjk(
        self,
        vec: np.ndarray,
        k: int = 1,
        exclude: Optional[Set[str]] = None,
    ) -> List[Tuple[str, float]]:
        """Like nearest() but only returns primary CJK ideograph characters.

        Filters out CJK Compatibility Ideographs to avoid duplicate glyphs.
        """
        n_query = k * 5 + 50
        n_query = min(n_query, len(self._chars))
        dists, idxs = self._tree.query(vec.astype(np.float64), k=n_query)
        if np.ndim(dists) == 0:
            dists = [float(dists)]
            idxs = [int(idxs)]
        results: List[Tuple[str, float]] = []
        for d, i in zip(dists, idxs):
            ch = self._chars[int(i)]
            if not _is_cjk_primary(ch):
                continue
            if exclude and ch in exclude:
                continue
            results.append((ch, float(d)))
            if len(results) >= k:
                break
        return results
