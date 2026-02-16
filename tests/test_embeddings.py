import numpy as np

from shanzi.embeddings import (
    build_recursive_embeddings,
    topological_order,
    train_cooccurrence_embeddings,
)


def test_topological_order_dependencies_first():
    graph = {
        "晒": ["日", "西"],
        "西": ["兀", "口"],
        "兀": ["一", "儿"],
        "日": [],
        "口": [],
        "一": [],
        "儿": [],
    }
    order = topological_order(graph)
    index = {node: i for i, node in enumerate(order)}
    assert index["一"] < index["兀"] < index["西"] < index["晒"]
    assert index["儿"] < index["兀"]
    assert index["日"] < index["晒"]


def test_recursive_embeddings_normalize_and_propagate():
    graph = {
        "晒": ["日", "西"],
        "西": ["兀", "口"],
        "兀": ["一", "儿"],
        "日": [],
        "口": [],
        "一": [],
        "儿": [],
    }
    base = {
        "晒": np.array([1.0, 0.0, 0.0]),
        "日": np.array([0.0, 1.0, 0.0]),
        "西": np.array([0.0, 0.0, 1.0]),
        "兀": np.array([1.0, 1.0, 0.0]),
        "口": np.array([0.0, 1.0, 1.0]),
        "一": np.array([1.0, 0.0, 1.0]),
        "儿": np.array([1.0, 1.0, 1.0]),
    }
    recursive = build_recursive_embeddings(base, graph, embedding_dim=3, k=0.67, unknown_scale=0.0)

    for vector in recursive.values():
        assert np.isclose(np.linalg.norm(vector), 1.0, atol=1e-6)

    base_sun_tan = base["晒"] / np.linalg.norm(base["晒"])
    assert not np.allclose(recursive["晒"], base_sun_tan)


def test_train_cooccurrence_embeddings_shapes():
    text = "晒西晒日日西口口口晒"
    embeddings = train_cooccurrence_embeddings(
        text=text,
        dim=8,
        window=2,
        min_count=1,
        max_vocab=20,
    )
    for token in ("晒", "西", "日", "口"):
        assert token in embeddings
        assert embeddings[token].shape == (8,)
