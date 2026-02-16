from shanzi.model import build_model_from_corpus


def _toy_decompositions() -> dict[str, str]:
    return {
        "晒": "⿰日西",
        "西": "⿱一儿",
        "日": "日",
        "一": "一",
        "儿": "儿",
        "森": "⿱木林",
        "林": "⿰木木",
        "木": "木",
    }


def test_default_build_includes_dictionary_characters_not_in_corpus():
    decompositions = _toy_decompositions()
    model = build_model_from_corpus(
        corpus_text="晒晒晒",
        decompositions=decompositions,
        dim=8,
        window=2,
        min_count=1,
        max_vocab=64,
        k=0.67,
    )
    assert "晒" in model.tokens
    # Absent from corpus, but present in IDS table and therefore sampleable.
    assert "森" in model.tokens
    assert "林" in model.tokens


def test_corpus_roots_mode_excludes_unreachable_dictionary_characters():
    decompositions = _toy_decompositions()
    model = build_model_from_corpus(
        corpus_text="晒晒晒",
        decompositions=decompositions,
        dim=8,
        window=2,
        min_count=1,
        max_vocab=64,
        k=0.67,
        include_all_characters=False,
    )
    assert "晒" in model.tokens
    assert "森" not in model.tokens
