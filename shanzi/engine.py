"""
ShanziEngine — the top-level façade that wires everything together.

    from shanzi import ShanziEngine

    engine = ShanziEngine()
    print(engine.transform("今天天气很好"))
"""

from __future__ import annotations

import sys
from typing import Optional

from shanzi.decompose import Decomposer
from shanzi.embeddings import (
    EmbeddingSource,
    GensimSource,
    ShanziEmbeddings,
    StructuralSource,
    TextFileSource,
)
from shanzi.transform import ShanziTransformer


class ShanziEngine:
    """One object to rule them all.

    Parameters
    ----------
    k : float
        Attenuation factor for component blending (default 0.67).
    dim : int
        Embedding dimensionality when using the structural fallback
        (ignored if *embeddings_path* supplies vectors of a different size).
    embeddings_path : str, optional
        Path to a pre-trained embedding file.  Recognised formats:
        • word2vec text format (first line: ``N DIM``)
        • gensim binary (``.bin``) — requires ``pip install gensim``
    ids_path : str, optional
        Override path to the IDS decomposition file.
    quiet : bool
        Suppress progress messages.
    """

    def __init__(
        self,
        k: float = 0.67,
        dim: int = 300,
        embeddings_path: Optional[str] = None,
        ids_path: Optional[str] = None,
        quiet: bool = False,
        cache: bool = True,
    ):
        self.decomposer = Decomposer(ids_path)

        source: Optional[EmbeddingSource] = None
        if embeddings_path is not None:
            if embeddings_path.endswith(".bin"):
                source = GensimSource(embeddings_path)
            else:
                source = TextFileSource(embeddings_path)

        self.embeddings = ShanziEmbeddings(
            self.decomposer, source=source, k=k, dim=dim,
            quiet=quiet, cache=cache,
        )
        self.transformer = ShanziTransformer(self.embeddings)

    def transform(
        self,
        text: str,
        temperature: float = 0.3,
        seed: Optional[int] = None,
    ) -> str:
        """Transform *text* through shanzi-space.

        See :meth:`ShanziTransformer.transform` for parameter details.
        """
        return self.transformer.transform(text, temperature=temperature, seed=seed)

    def decompose(self, char: str):
        """Show the recursive decomposition tree for a single character."""
        return self.decomposer.get_recursive(char)

    def embedding(self, char: str):
        """Return the shanzi embedding vector for *char*, or None."""
        return self.embeddings.get(char)

    def neighbors(self, char: str, k: int = 10):
        """Return the *k* nearest characters in shanzi-space."""
        vec = self.embeddings.get(char)
        if vec is None:
            return []
        return self.embeddings.nearest_cjk(vec, k=k + 1, exclude={char})[:k]
