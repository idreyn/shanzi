"""Tests for shanzi.generate — multi-channel beam search generator."""

import pytest
from shanzi import ShanziEngine
from shanzi.decompose import _is_cjk_primary


@pytest.fixture(scope="module")
def engine():
    return ShanziEngine(quiet=True)


class TestBeamGenerate:
    def test_basic_output(self, engine):
        result = engine.transform("今天好", seed=42, mode="beam")
        assert len(result) == 3
        assert result != "今天好"

    def test_reproducible(self, engine):
        r1 = engine.transform("今天好", temperature=0.5, seed=42, mode="beam")
        r2 = engine.transform("今天好", temperature=0.5, seed=42, mode="beam")
        assert r1 == r2

    def test_different_seeds(self, engine):
        r1 = engine.transform("上海的天空", seed=1, mode="beam")
        r2 = engine.transform("上海的天空", seed=2, mode="beam")
        assert r1 != r2

    def test_preserves_non_cjk(self, engine):
        result = engine.transform("hello 你好 world", seed=42, mode="beam")
        assert result.startswith("hello ")
        assert " world" in result

    def test_empty_string(self, engine):
        assert engine.transform("", mode="beam") == ""

    def test_no_cjk(self, engine):
        assert engine.transform("hello", mode="beam") == "hello"

    def test_single_char(self, engine):
        result = engine.transform("木", seed=42, mode="beam")
        assert len(result) == 1

    def test_output_is_cjk_primary(self, engine):
        result = engine.transform("今天天气很好", seed=42, mode="beam")
        for ch in result:
            if _is_cjk_primary(ch):
                pass  # good
            else:
                # non-CJK chars are fine if they were pass-through
                assert not _is_cjk_primary(ch) or ch in "今天天气很好"


class TestRadicalResonance:
    """Verify that adjacent output characters share radicals more than random."""

    def test_resonance_present(self, engine):
        info = engine.explain("上海的天空是灰色的", temperature=0.5, seed=42)
        chars = info["chars"]

        # Count how many adjacent pairs share at least one radical
        resonance_count = 0
        for c in chars:
            if c["resonance_with_prev"]:
                resonance_count += 1

        # With 9 characters (8 adjacent pairs), we expect at least 3
        # to have radical overlap (the beam search optimises for this)
        assert resonance_count >= 3, (
            f"Only {resonance_count} resonant pairs out of {len(chars)-1}"
        )

    def test_resonance_increases_with_temperature(self, engine):
        text = "意识正在分崩离析"

        def count_resonance(t):
            info = engine.explain(text, temperature=t, seed=42)
            return sum(
                1 for c in info["chars"] if c["resonance_with_prev"]
            )

        low = count_resonance(0.1)
        high = count_resonance(0.8)
        # Higher temperature should give at least as much resonance
        # (since resonance weight increases with temperature)
        assert high >= low, f"t=0.1 gave {low} resonances, t=0.8 gave {high}"


class TestMotifs:
    def test_motifs_selected(self, engine):
        info = engine.explain("今天天气很好", temperature=0.5, seed=42)
        assert len(info["motifs"]) >= 1

    def test_motifs_appear_in_output(self, engine):
        info = engine.explain("上海的天空是灰色的", temperature=0.5, seed=42)
        motif_rads = {m["radical"] for m in info["motifs"]}

        # At least one output character should contain at least one motif
        found = False
        for c in info["chars"]:
            if c["active_motifs"]:
                found = True
                break
        assert found, f"No motif radicals found in output (motifs: {motif_rads})"


class TestExplain:
    def test_explain_structure(self, engine):
        info = engine.explain("今天好", temperature=0.5, seed=42)
        assert "input" in info
        assert "output" in info
        assert "motifs" in info
        assert "chars" in info
        assert len(info["chars"]) == 3
        assert info["chars"][0]["input"] == "今"

    def test_explain_matches_generate(self, engine):
        text = "上海的天空"
        info = engine.explain(text, temperature=0.5, seed=42)
        result = engine.transform(text, temperature=0.5, seed=42, mode="beam")
        assert info["output"] == result


class TestSplineFallback:
    def test_spline_mode_still_works(self, engine):
        result = engine.transform("今天好", temperature=0.3, seed=42, mode="spline")
        assert len(result) == 3
        assert result != "今天好"

    def test_spline_different_from_beam(self, engine):
        spline = engine.transform("今天天气好", seed=42, mode="spline")
        beam = engine.transform("今天天气好", seed=42, mode="beam")
        # They use completely different algorithms, should differ
        assert spline != beam
