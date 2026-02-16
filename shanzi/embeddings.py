"""Base and recursive embedding builders."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import heapq
from typing import Mapping, Sequence

import numpy as np

from .ids import ComponentGraph


def tokenize_characters(text: str) -> list[str]:
    """Simple character tokenizer: keep all non-whitespace codepoints."""
    return [char for char in text if not char.isspace()]


def train_cooccurrence_embeddings(
    text: str,
    dim: int = 96,
    window: int = 4,
    min_count: int = 2,
    max_vocab: int = 2000,
) -> dict[str, np.ndarray]:
    """
    Train a compact static embedding from character co-occurrence.

    This is a light-weight Word2Vec alternative: PPMI + SVD.
    """
    if dim <= 0:
        raise ValueError("Embedding dim must be positive.")
    if window <= 0:
        raise ValueError("Window size must be positive.")
    if min_count <= 0:
        raise ValueError("min_count must be positive.")
    if max_vocab <= 0:
        raise ValueError("max_vocab must be positive.")

    tokens = tokenize_characters(text)
    counts = Counter(tokens)
    vocab = [token for token, freq in counts.items() if freq >= min_count]
    vocab.sort(key=lambda token: (-counts[token], token))
    vocab = vocab[:max_vocab]
    if not vocab:
        return {}

    index = {token: i for i, token in enumerate(vocab)}
    n_vocab = len(vocab)
    cooc = np.zeros((n_vocab, n_vocab), dtype=np.float64)

    for i, token in enumerate(tokens):
        token_idx = index.get(token)
        if token_idx is None:
            continue
        start = max(0, i - window)
        stop = min(len(tokens), i + window + 1)
        for j in range(start, stop):
            if i == j:
                continue
            context_idx = index.get(tokens[j])
            if context_idx is None:
                continue
            weight = 1.0 / abs(i - j)
            cooc[token_idx, context_idx] += weight

    ppmi = _ppmi_matrix(cooc)
    vectors = _svd_projection(ppmi, dim)
    vectors = _normalize_rows(vectors)
    return {token: vectors[i] for token, i in index.items()}


def _ppmi_matrix(cooc: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    total = float(cooc.sum())
    if total <= 0:
        return np.zeros_like(cooc)
    row_sum = cooc.sum(axis=1, keepdims=True)
    col_sum = cooc.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log((cooc * total + eps) / (row_sum * col_sum + eps))
    pmi[~np.isfinite(pmi)] = 0.0
    pmi = np.maximum(pmi, 0.0)
    pmi[cooc <= 0] = 0.0
    return pmi


def _svd_projection(matrix: np.ndarray, dim: int) -> np.ndarray:
    if matrix.size == 0:
        return np.zeros((0, dim), dtype=np.float64)
    u, s, _ = np.linalg.svd(matrix, full_matrices=False)
    rank = min(dim, u.shape[1], s.shape[0])
    projected = u[:, :rank] * np.sqrt(s[:rank])
    if rank == dim:
        return projected
    pad = np.zeros((projected.shape[0], dim - rank), dtype=projected.dtype)
    return np.hstack([projected, pad])


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def topological_order(component_graph: Mapping[str, Sequence[str]]) -> list[str]:
    """
    Topologically order nodes so each node appears after its dependencies.
    """
    nodes: set[str] = set(component_graph)
    for children in component_graph.values():
        nodes.update(children)

    deps: dict[str, list[str]] = {}
    reverse_adj: dict[str, list[str]] = defaultdict(list)
    indegree: dict[str, int] = {}

    for node in nodes:
        node_deps = [child for child in component_graph.get(node, []) if child != node]
        deps[node] = node_deps
        indegree[node] = len(node_deps)
        for child in node_deps:
            reverse_adj[child].append(node)

    heap = [node for node in nodes if indegree[node] == 0]
    heapq.heapify(heap)

    ordered: list[str] = []
    while heap:
        node = heapq.heappop(heap)
        ordered.append(node)
        for parent in reverse_adj[node]:
            indegree[parent] -= 1
            if indegree[parent] == 0:
                heapq.heappush(heap, parent)

    if len(ordered) != len(nodes):
        unresolved = sorted(node for node in nodes if indegree[node] > 0)
        raise ValueError(
            f"Component graph is cyclic or malformed; unresolved nodes: {unresolved[:8]}"
        )
    return ordered


def build_recursive_embeddings(
    base_embeddings: Mapping[str, np.ndarray],
    component_graph: Mapping[str, Sequence[str]],
    *,
    k: float = 0.67,
    embedding_dim: int | None = None,
    random_seed: int = 7,
    unknown_scale: float = 0.03,
) -> dict[str, np.ndarray]:
    """
    Build recursive embeddings:
        S(h) = normalize(W(h) + k * sum(S(part)))
    """
    if embedding_dim is None:
        embedding_dim = _infer_dim(base_embeddings)
    if embedding_dim <= 0:
        raise ValueError("embedding_dim must be positive.")

    order = topological_order(component_graph)
    embeddings: dict[str, np.ndarray] = {}

    for node in order:
        combined = _base_vector(
            node,
            base_embeddings=base_embeddings,
            dim=embedding_dim,
            random_seed=random_seed,
            unknown_scale=unknown_scale,
        )
        for child in component_graph.get(node, []):
            combined = combined + k * embeddings[child]
        norm = float(np.linalg.norm(combined))
        if norm > 0:
            combined = combined / norm
        embeddings[node] = combined.astype(np.float32)
    return embeddings


def _infer_dim(base_embeddings: Mapping[str, np.ndarray]) -> int:
    for vector in base_embeddings.values():
        return int(vector.shape[0])
    raise ValueError("No base embeddings available to infer embedding_dim.")


def _base_vector(
    token: str,
    *,
    base_embeddings: Mapping[str, np.ndarray],
    dim: int,
    random_seed: int,
    unknown_scale: float,
) -> np.ndarray:
    base = base_embeddings.get(token)
    if base is not None:
        if base.shape[0] != dim:
            raise ValueError(f"Inconsistent dimensions for token {token!r}.")
        return base.astype(np.float64, copy=True)
    if token.startswith("expr:"):
        return np.zeros(dim, dtype=np.float64)
    return _deterministic_noise(token, dim=dim, seed=random_seed, scale=unknown_scale)


def _deterministic_noise(token: str, *, dim: int, seed: int, scale: float) -> np.ndarray:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    token_seed = int.from_bytes(digest, "big") ^ seed
    rng = np.random.default_rng(token_seed)
    return rng.normal(0.0, scale, size=dim)


def stack_embeddings(
    embeddings: Mapping[str, np.ndarray],
    *,
    include_tokens: Sequence[str] | None = None,
) -> tuple[list[str], np.ndarray]:
    """Convert token->vector dict to aligned token list and matrix."""
    if include_tokens is None:
        tokens = sorted(embeddings)
    else:
        tokens = [token for token in include_tokens if token in embeddings]
    if not tokens:
        return [], np.zeros((0, 0), dtype=np.float32)
    matrix = np.vstack([embeddings[token] for token in tokens]).astype(np.float32)
    return tokens, matrix
