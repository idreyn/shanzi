from pathlib import Path

from shanzi.ids import build_component_graph, download_ids_file, load_ids_decompositions, recursive_components
from shanzi.model import build_model_from_corpus
from shanzi.trajectory import remix_sentence


def main() -> None:
    corpus_path = Path("examples/corpus.txt")
    ids_path = Path("data/ids.txt")

    download_ids_file(ids_path)
    decompositions = load_ids_decompositions(ids_path)
    graph = build_component_graph(decompositions)

    print("晒 direct:", graph.get("晒"))
    print("晒 recursive (first 12):", recursive_components("晒", graph)[:12])

    corpus_text = corpus_path.read_text(encoding="utf-8")
    model = build_model_from_corpus(
        corpus_text,
        decompositions,
        dim=64,
        window=3,
        min_count=1,
        max_vocab=1200,
        k=0.67,
        include_all_characters=True,
    )

    sentence = "上海夜雨把语言洗亮"
    out = remix_sentence(sentence, model, output_length=len(sentence), jitter_std=0.03, seed=42)
    print("input :", sentence)
    print("shanzi:", out)


if __name__ == "__main__":
    main()
