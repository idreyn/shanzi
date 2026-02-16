"""Model container and build pipeline for Shanzi embeddings."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

from .embeddings import build_recursive_embeddings, stack_embeddings, train_cooccurrence_embeddings
from .ids import (
    build_component_graph,
    is_character_token,
    is_hanzi,
    reachable_subgraph,
    unique_hanzi_in_text,
)


@dataclass
class ShanziModel:
    """Holds recursive embeddings and component graph metadata."""

    tokens: list[str]
    vectors: np.ndarray
    component_graph: dict[str, list[str]]
    k: float = 0.67
    metadata: dict[str, object] = field(default_factory=dict)

    _index: dict[str, int] = field(init=False, repr=False)
    _normalized_vectors: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.vectors.ndim != 2:
            raise ValueError("vectors must be a 2D matrix.")
        if len(self.tokens) != self.vectors.shape[0]:
            raise ValueError("tokens and vectors row count must match.")
        self._index = {token: idx for idx, token in enumerate(self.tokens)}
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        self._normalized_vectors = self.vectors / norms

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1])

    def embedding(self, token: str) -> np.ndarray:
        idx = self._index.get(token)
        if idx is None:
            raise KeyError(f"Token {token!r} is not in model vocabulary.")
        return self.vectors[idx]

    def character_tokens(self, *, hanzi_only: bool = True) -> list[str]:
        tokens = [token for token in self.tokens if is_character_token(token)]
        if hanzi_only:
            tokens = [token for token in tokens if is_hanzi(token)]
        return tokens

    def nearest(
        self,
        vector: np.ndarray,
        *,
        top_k: int = 1,
        token_filter: Callable[[str], bool] | None = None,
    ) -> list[tuple[str, float]]:
        if vector.ndim != 1:
            raise ValueError("vector must be 1D.")
        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        if token_filter is None:
            indices = np.arange(len(self.tokens))
        else:
            indices = np.array(
                [idx for idx, token in enumerate(self.tokens) if token_filter(token)],
                dtype=np.int64,
            )
        if len(indices) == 0:
            return []

        vec = vector.astype(np.float64, copy=False)
        norm = float(np.linalg.norm(vec))
        if norm == 0.0:
            return []
        vec = vec / norm

        scores = self._normalized_vectors[indices] @ vec
        if top_k >= len(scores):
            top_order = np.argsort(scores)[::-1]
        else:
            top_indices = np.argpartition(scores, -top_k)[-top_k:]
            top_order = top_indices[np.argsort(scores[top_indices])[::-1]]

        result: list[tuple[str, float]] = []
        for loc in top_order:
            idx = int(indices[loc])
            result.append((self.tokens[idx], float(scores[loc])))
        return result

    def decode_vectors(
        self,
        vectors: np.ndarray,
        *,
        hanzi_only: bool = True,
    ) -> str:
        chars: list[str] = []
        if hanzi_only:
            allowed = set(self.character_tokens(hanzi_only=True))
            token_filter = lambda token: token in allowed
        else:
            token_filter = lambda token: is_character_token(token)
        for row in vectors:
            nearest = self.nearest(row, top_k=1, token_filter=token_filter)
            if not nearest:
                continue
            chars.append(nearest[0][0])
        return "".join(chars)

    def save(self, path: Path) -> None:
        payload = {
            "tokens": np.array(self.tokens),
            "vectors": self.vectors.astype(np.float32),
            "k": np.array([self.k], dtype=np.float32),
            "component_graph_json": np.array(
                [json.dumps(self.component_graph, ensure_ascii=False)]
            ),
            "metadata_json": np.array([json.dumps(self.metadata, ensure_ascii=False)]),
        }
        np.savez_compressed(path, **payload)

    @classmethod
    def load(cls, path: Path) -> "ShanziModel":
        with np.load(path, allow_pickle=False) as data:
            tokens = data["tokens"].tolist()
            vectors = data["vectors"].astype(np.float32)
            k = float(data["k"][0])
            component_graph = json.loads(str(data["component_graph_json"][0]))
            metadata = json.loads(str(data["metadata_json"][0]))
        return cls(
            tokens=tokens,
            vectors=vectors,
            component_graph=component_graph,
            k=k,
            metadata=metadata,
        )


def build_model_from_corpus(
    corpus_text: str,
    decompositions: Mapping[str, str],
    *,
    dim: int = 96,
    window: int = 4,
    min_count: int = 1,
    max_vocab: int = 20000,
    k: float = 0.67,
    random_seed: int = 7,
    roots: Sequence[str] | None = None,
    include_all_characters: bool = True,
) -> ShanziModel:
    """
    Build a full Shanzi model from corpus text and IDS decompositions.
    """
    base = train_cooccurrence_embeddings(
        corpus_text,
        dim=dim,
        window=window,
        min_count=min_count,
        max_vocab=max_vocab,
    )
    graph = build_component_graph(decompositions)

    if include_all_characters:
        active_graph = graph
        selected_roots = sorted(decompositions)
    else:
        selected_roots = list(roots) if roots is not None else unique_hanzi_in_text(corpus_text)
        if not selected_roots:
            raise ValueError(
                "No Hanzi roots selected. Provide `roots` or use `include_all_characters=True`."
            )
        active_graph = reachable_subgraph(graph, selected_roots)

    recursive = build_recursive_embeddings(
        base_embeddings=base,
        component_graph=active_graph,
        k=k,
        embedding_dim=dim,
        random_seed=random_seed,
    )

    tokens, vectors = stack_embeddings(recursive)
    metadata = {
        "dim": dim,
        "window": window,
        "min_count": min_count,
        "max_vocab": max_vocab,
        "selected_roots": selected_roots,
        "include_all_characters": include_all_characters,
    }
    return ShanziModel(tokens=tokens, vectors=vectors, component_graph=active_graph, k=k, metadata=metadata)
