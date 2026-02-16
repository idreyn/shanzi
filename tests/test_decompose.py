"""Tests for shanzi.decompose."""

import pytest
from shanzi.decompose import Decomposer, _is_cjk, _is_cjk_primary


@pytest.fixture(scope="module")
def decomposer():
    return Decomposer()


class TestIsCjk:
    def test_basic_cjk(self):
        assert _is_cjk("木")
        assert _is_cjk("水")
        assert _is_cjk("晒")

    def test_non_cjk(self):
        assert not _is_cjk("A")
        assert not _is_cjk("1")
        assert not _is_cjk(" ")

    def test_extensions(self):
        assert _is_cjk("𠁤")  # Extension B

    def test_primary_excludes_compat(self):
        # U+F9F4 is a CJK Compatibility Ideograph (duplicate 林)
        assert _is_cjk(chr(0xF9F4))
        assert not _is_cjk_primary(chr(0xF9F4))
        # U+6797 is the primary 林
        assert _is_cjk_primary(chr(0x6797))


class TestDecomposer:
    def test_loads_data(self, decomposer):
        assert len(decomposer.components) > 50000

    def test_basic_decomposition(self, decomposer):
        assert decomposer.get("晒") == ["日", "西"]
        assert decomposer.get("粒") == ["米", "立"]
        assert decomposer.get("林") == ["木", "木"]
        assert decomposer.get("森") == ["木", "林"]

    def test_atomic_character(self, decomposer):
        # 一 is a single stroke, should have no decomposition
        assert decomposer.get("一") == []

    def test_recursive_decomposition(self, decomposer):
        tree = decomposer.get_recursive("晒")
        assert "晒" in tree
        assert tree["晒"] == ["日", "西"]
        assert "西" in tree

    def test_topological_sort(self, decomposer):
        order = decomposer.topological_sort({"晒", "日", "西"})
        # 日 and 西 must come before 晒
        assert order.index("日") < order.index("晒")
        assert order.index("西") < order.index("晒")

    def test_all_chars_nonempty(self, decomposer):
        chars = decomposer.all_chars()
        assert len(chars) > 50000
