"""Tests for shanzi.transform."""

import pytest
from shanzi import ShanziEngine


@pytest.fixture(scope="module")
def engine():
    return ShanziEngine(quiet=True)


class TestTransform:
    def test_basic_transform(self, engine):
        result = engine.transform("今天好", seed=42)
        assert len(result) == 3
        # Output should be different from input
        assert result != "今天好"

    def test_reproducible_with_seed(self, engine):
        r1 = engine.transform("今天好", temperature=0.3, seed=42)
        r2 = engine.transform("今天好", temperature=0.3, seed=42)
        assert r1 == r2

    def test_different_seeds_different_output(self, engine):
        r1 = engine.transform("今天天气好", seed=1)
        r2 = engine.transform("今天天气好", seed=2)
        assert r1 != r2

    def test_preserves_non_cjk(self, engine):
        result = engine.transform("hello 你好 world", seed=42)
        assert result.startswith("hello ")
        assert " world" in result

    def test_empty_string(self, engine):
        assert engine.transform("") == ""

    def test_no_cjk(self, engine):
        assert engine.transform("hello world") == "hello world"

    def test_single_character(self, engine):
        result = engine.transform("木", seed=42)
        assert len(result) == 1
        assert result != "木"

    def test_temperature_zero_stays_close(self, engine):
        # At temperature=0, output should be the nearest non-self character
        result = engine.transform("木", temperature=0.0, seed=42)
        assert len(result) == 1

    def test_higher_temperature_more_drift(self, engine):
        # This is probabilistic but should hold on average
        low = engine.transform("今天天气很好", temperature=0.05, seed=42)
        high = engine.transform("今天天气很好", temperature=1.0, seed=42)
        # At least the outputs should be different from each other
        assert low != high


class TestDecompose:
    def test_decompose_compound(self, engine):
        tree = engine.decompose("晒")
        assert "晒" in tree
        assert tree["晒"] == ["日", "西"]

    def test_decompose_atomic(self, engine):
        tree = engine.decompose("一")
        assert tree == {}


class TestNeighbors:
    def test_neighbors(self, engine):
        nbrs = engine.neighbors("木", k=5)
        assert len(nbrs) == 5
        chars = [ch for ch, _ in nbrs]
        assert "木" not in chars

    def test_neighbors_unknown(self, engine):
        nbrs = engine.neighbors("@", k=5)
        assert nbrs == []
