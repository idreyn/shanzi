"""
Command-line interface for shanzi.

Usage
-----
    shanzi "今天天气很好"
    shanzi --temperature 0.5 --seed 42 "上海的天空是灰色的"
    shanzi decompose 晒
    shanzi neighbors 晒
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional


def _build_engine(args):
    """Lazily construct the ShanziEngine from CLI arguments."""
    from shanzi.engine import ShanziEngine

    return ShanziEngine(
        k=args.k,
        dim=args.dim,
        embeddings_path=getattr(args, "embeddings", None),
        quiet=args.quiet,
    )


def cmd_transform(args):
    engine = _build_engine(args)
    text = args.text
    result = engine.transform(text, temperature=args.temperature, seed=args.seed)
    print(result)


def cmd_decompose(args):
    engine = _build_engine(args)
    char = args.char
    if len(char) != 1:
        print(f"Expected a single character, got {len(char)}", file=sys.stderr)
        sys.exit(1)
    tree = engine.decompose(char)
    if not tree:
        print(f"{char} — atomic (no decomposition found)")
    else:
        print(json.dumps(tree, ensure_ascii=False, indent=2))


def cmd_neighbors(args):
    engine = _build_engine(args)
    char = args.char
    if len(char) != 1:
        print(f"Expected a single character, got {len(char)}", file=sys.stderr)
        sys.exit(1)
    nbrs = engine.neighbors(char, k=args.count)
    if not nbrs:
        print(f"No neighbors found for {char}")
    else:
        for ch, dist in nbrs:
            print(f"  {ch}  (dist={dist:.4f})")


def main(argv: Optional[list] = None):
    p = argparse.ArgumentParser(
        prog="shanzi",
        description="Semantic ultrasound of obscure Chinese characters.",
    )
    p.add_argument(
        "-k", type=float, default=0.67,
        help="Component attenuation factor (default: 0.67)",
    )
    p.add_argument(
        "--dim", type=int, default=300,
        help="Embedding dimensionality for structural mode (default: 300)",
    )
    p.add_argument(
        "--embeddings", "-e", type=str, default=None,
        help="Path to pre-trained embedding file (word2vec text or gensim .bin)",
    )
    p.add_argument(
        "--quiet", "-q", action="store_true",
        help="Suppress progress messages",
    )

    sub = p.add_subparsers(dest="command")

    # Default: transform
    # (also handled when no subcommand is given)

    # --- transform ---
    sp_t = sub.add_parser("transform", help="Transform text through shanzi-space")
    sp_t.add_argument("text", type=str, help="Input text")
    sp_t.add_argument(
        "-t", "--temperature", type=float, default=0.3,
        help="Jitter magnitude (default: 0.3)",
    )
    sp_t.add_argument(
        "-s", "--seed", type=int, default=None,
        help="Random seed for reproducibility",
    )
    sp_t.set_defaults(func=cmd_transform)

    # --- decompose ---
    sp_d = sub.add_parser("decompose", help="Show recursive decomposition of a character")
    sp_d.add_argument("char", type=str, help="A single Chinese character")
    sp_d.set_defaults(func=cmd_decompose)

    # --- neighbors ---
    sp_n = sub.add_parser("neighbors", help="Find nearest characters in shanzi-space")
    sp_n.add_argument("char", type=str, help="A single Chinese character")
    sp_n.add_argument(
        "-n", "--count", type=int, default=10,
        help="Number of neighbors (default: 10)",
    )
    sp_n.set_defaults(func=cmd_neighbors)

    args = p.parse_args(argv)

    if args.command is None:
        # No subcommand: treat remaining args as transform
        # Re-parse with transform defaults
        p2 = argparse.ArgumentParser(prog="shanzi")
        p2.add_argument("-k", type=float, default=0.67)
        p2.add_argument("--dim", type=int, default=300)
        p2.add_argument("--embeddings", "-e", type=str, default=None)
        p2.add_argument("--quiet", "-q", action="store_true")
        p2.add_argument("text", nargs="?", type=str, default=None)
        p2.add_argument("-t", "--temperature", type=float, default=0.3)
        p2.add_argument("-s", "--seed", type=int, default=None)
        args = p2.parse_args(argv)
        if args.text is None:
            p.print_help()
            sys.exit(0)
        args.func = cmd_transform
        args.command = "transform"

    args.func(args)


if __name__ == "__main__":
    main()
