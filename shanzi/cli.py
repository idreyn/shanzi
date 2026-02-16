"""Command line interface for the Shanzi MVP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .ids import (
    build_component_graph,
    download_ids_file,
    load_ids_decompositions,
    recursive_components,
    unique_hanzi_in_text,
)
from .model import ShanziModel, build_model_from_corpus
from .trajectory import remix_sentence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="shanzi",
        description="Recursive Hanzi embeddings + semantic trajectory remixing.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch_parser = subparsers.add_parser("fetch-ids", help="Download IDS dataset.")
    fetch_parser.add_argument("--ids-path", type=Path, default=Path("data/ids.txt"))
    fetch_parser.set_defaults(func=cmd_fetch_ids)

    components_parser = subparsers.add_parser("components", help="Show direct and recursive components.")
    components_parser.add_argument("--chars", type=str, required=True, help="Characters to inspect.")
    components_parser.add_argument("--ids-path", type=Path, default=Path("data/ids.txt"))
    components_parser.set_defaults(func=cmd_components)

    build_parser_ = subparsers.add_parser("build", help="Build a Shanzi model from corpus text.")
    build_parser_.add_argument("--corpus", type=Path, required=True)
    build_parser_.add_argument("--model-out", type=Path, default=Path("models/shanzi_model.npz"))
    build_parser_.add_argument("--ids-path", type=Path, default=Path("data/ids.txt"))
    build_parser_.add_argument("--dim", type=int, default=96)
    build_parser_.add_argument("--window", type=int, default=4)
    build_parser_.add_argument("--min-count", type=int, default=2)
    build_parser_.add_argument("--max-vocab", type=int, default=2000)
    build_parser_.add_argument("--k", type=float, default=0.67)
    build_parser_.add_argument("--seed", type=int, default=7)
    build_parser_.add_argument("--roots", type=str, default="", help="Optional explicit root characters.")
    build_parser_.add_argument(
        "--max-roots",
        type=int,
        default=0,
        help="Limit roots extracted from corpus (0 = all corpus Hanzi).",
    )
    build_parser_.add_argument(
        "--all-characters",
        action="store_true",
        help="Build embeddings for all characters in IDS table.",
    )
    build_parser_.set_defaults(func=cmd_build)

    remix_parser = subparsers.add_parser("remix", help="Generate a shanzi remix sentence.")
    remix_parser.add_argument("--model", type=Path, required=True)
    remix_parser.add_argument("--sentence", type=str, default="")
    remix_parser.add_argument("--sentence-file", type=Path, default=None)
    remix_parser.add_argument("--samples-per-segment", type=int, default=8)
    remix_parser.add_argument("--output-length", type=int, default=0)
    remix_parser.add_argument("--jitter-std", type=float, default=0.03)
    remix_parser.add_argument("--seed", type=int, default=13)
    remix_parser.add_argument(
        "--include-non-hanzi",
        action="store_true",
        help="Allow decoding to non-Hanzi character tokens.",
    )
    remix_parser.set_defaults(func=cmd_remix)

    return parser


def cmd_fetch_ids(args: argparse.Namespace) -> int:
    path = download_ids_file(args.ids_path)
    print(f"IDS downloaded: {path}")
    return 0


def cmd_components(args: argparse.Namespace) -> int:
    ids_path = download_ids_file(args.ids_path)
    decompositions = load_ids_decompositions(ids_path)
    graph = build_component_graph(decompositions)

    payload = {}
    for character in args.chars:
        payload[character] = {
            "direct": graph.get(character, []),
            "recursive": recursive_components(character, graph),
        }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    corpus_text = args.corpus.read_text(encoding="utf-8")
    ids_path = download_ids_file(args.ids_path)
    decompositions = load_ids_decompositions(ids_path)

    roots = None
    if args.roots:
        roots = list(dict.fromkeys(args.roots))
    elif not args.all_characters:
        roots = unique_hanzi_in_text(corpus_text)
        if args.max_roots > 0:
            roots = roots[: args.max_roots]

    model = build_model_from_corpus(
        corpus_text=corpus_text,
        decompositions=decompositions,
        dim=args.dim,
        window=args.window,
        min_count=args.min_count,
        max_vocab=args.max_vocab,
        k=args.k,
        random_seed=args.seed,
        roots=roots,
        include_all_characters=args.all_characters,
    )

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.model_out)
    print(
        json.dumps(
            {
                "model_path": str(args.model_out),
                "tokens": len(model.tokens),
                "dim": model.dim,
                "k": model.k,
                "roots": len(model.metadata.get("selected_roots", [])),
            },
            ensure_ascii=False,
        )
    )
    return 0


def cmd_remix(args: argparse.Namespace) -> int:
    if args.sentence_file is not None:
        sentence = args.sentence_file.read_text(encoding="utf-8").strip()
    else:
        sentence = args.sentence.strip()
    if not sentence:
        raise ValueError("Provide --sentence or --sentence-file.")

    model = ShanziModel.load(args.model)
    output_length = None if args.output_length <= 0 else args.output_length
    remix = remix_sentence(
        sentence=sentence,
        model=model,
        samples_per_segment=args.samples_per_segment,
        output_length=output_length,
        jitter_std=args.jitter_std,
        seed=args.seed,
        hanzi_only=not args.include_non_hanzi,
    )
    print(remix)
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
