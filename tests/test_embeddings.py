"""Tests for shanzi.embeddings."""

import numpy as np
import pytest
from shanzi.decompose import Decomposer
from shanzi.embeddings import ShanziEmbeddings, StructuralSource


@pytest.fixture(scope="module")
def decomposer():
    return Decomposer()


@pytest.fixture(scope="module")
def embeddings(decomposer):
    return ShanziEmbeddings(decomposer, quiet=True, cache=False)


class TestStructuralSource:
    def test_deterministic(self):
        src = StructuralSource(dim=100)
        v1 = src.get("木")
        v2 = src.get("木")
        np.testing.assert_array_equal(v1, v2)

    def test_unit_norm(self):
        src = StructuralSource(dim=100)
        v = src.get("水")
        np.testing.assert_almost_equal(np.linalg.norm(v), 1.0, decimal=5)

    def test_different_chars_different_vectors(self):
        src = StructuralSource(dim=100)
        v1 = src.get("木")
        v2 = src.get("水")
        assert not np.allclose(v1, v2)


class TestShanziEmbeddings:
    def test_has_embeddings(self, embeddings):
        assert len(embeddings.vocab) > 50000

    def test_get_returns_vector(self, embeddings):
        v = embeddings.get("晒")
        assert v is not None
        assert v.shape == (300,)

    def test_vectors_normalized(self, embeddings):
        for char in ["木", "水", "火", "晒", "粒"]:
            v = embeddings.get(char)
            assert v is not None
            np.testing.assert_almost_equal(np.linalg.norm(v), 1.0, decimal=4)

    def test_nearest_returns_results(self, embeddings):
        v = embeddings.get("晒")
        results = embeddings.nearest(v, k=5)
        assert len(results) == 5
        # First result should be 晒 itself (distance ~0)
        assert results[0][0] == "晒"
        assert results[0][1] < 0.01

    def test_nearest_cjk(self, embeddings):
        v = embeddings.get("木")
        results = embeddings.nearest_cjk(v, k=5, exclude={"木"})
        assert len(results) == 5
        for ch, _ in results:
            assert ch != "木"

    def test_component_similarity(self, embeddings):
        """Characters sharing a component should be closer than random."""
        v_shai = embeddings.get("晒")  # 日+西
        v_ri = embeddings.get("日")    # component of 晒
        v_shan = embeddings.get("山")  # unrelated

        dist_related = np.linalg.norm(v_shai - v_ri)
        dist_unrelated = np.linalg.norm(v_shai - v_shan)
        assert dist_related < dist_unrelated
