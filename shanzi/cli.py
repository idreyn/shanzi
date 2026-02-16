"""
Command-line interface for shanzi.

Usage
-----
    shanzi "今天天气很好"
    shanzi -t 0.5 -s 42 "上海的天空是灰色的"
    shanzi decompose 晒
    shanzi neighbors 晒
    echo "一段文字" | shanzi --stdin
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

from shanzi.decompose import _is_cjk


def _build_engine(args):
    """Lazily construct the ShanziEngine from CLI arguments."""
    from shanzi.engine import ShanziEngine

    return ShanziEngine(
        k=args.k,
        dim=args.dim,
        embeddings_path=getattr(args, "embeddings", None),
        quiet=getattr(args, "quiet", False),
    )


def cmd_transform(args):
    engine = _build_engine(args)

    if getattr(args, "stdin", False):
        text = sys.stdin.read().strip()
    else:
        text = args.text

    result = engine.transform(
        text, temperature=args.temperature, seed=args.seed,
    )

    if getattr(args, "annotate", False):
        # Show input → output character mapping
        print(f"  in:  {text}")
        print(f"  out: {result}")
        print()
        for i, (orig, out) in enumerate(zip(text, result)):
            if orig == out or not _is_cjk(orig):
                continue
            orig_parts = engine.decomposer.get(orig)
            out_parts = engine.decomposer.get(out)
            orig_info = f"[{''.join(orig_parts)}]" if orig_parts else ""
            out_info = f"[{''.join(out_parts)}]" if out_parts else ""
            print(f"  {orig}{orig_info} → {out}{out_info}")
    else:
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
            parts = engine.decomposer.get(ch)
            parts_str = f" [{''.join(parts)}]" if parts else ""
            print(f"  {ch}  U+{ord(ch):04X}{parts_str}  (dist={dist:.4f})")


def _add_global_args(parser):
    """Add arguments shared across all subcommands."""
    parser.add_argument(
        "-k", type=float, default=0.67,
        help="component attenuation factor (default: 0.67)",
    )
    parser.add_argument(
        "--dim", type=int, default=300,
        help="embedding dimensionality (default: 300)",
    )
    parser.add_argument(
        "--embeddings", "-e", type=str, default=None,
        help="path to pre-trained embedding file",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="suppress progress messages",
    )


def _add_transform_args(parser):
    """Add arguments specific to the transform command."""
    parser.add_argument(
        "-t", "--temperature", type=float, default=0.3,
        help="jitter magnitude (default: 0.3)",
    )
    parser.add_argument(
        "-s", "--seed", type=int, default=None,
        help="random seed for reproducibility",
    )
    parser.add_argument(
        "-a", "--annotate", action="store_true",
        help="show input→output character mapping with decomposition",
    )
    parser.add_argument(
        "--stdin", action="store_true",
        help="read input from stdin instead of argument",
    )


def main(argv: Optional[list] = None):
    known_cmds = {"transform", "decompose", "neighbors"}
    raw = argv if argv is not None else sys.argv[1:]
    has_subcmd = any(a in known_cmds for a in raw if not a.startswith("-"))

    if has_subcmd:
        p = argparse.ArgumentParser(
            prog="shanzi",
            description="Semantic ultrasound of obscure Chinese characters.",
        )
        _add_global_args(p)
        sub = p.add_subparsers(dest="command")

        sp_t = sub.add_parser("transform", help="transform text through shanzi-space")
        sp_t.add_argument("text", nargs="?", type=str, default=None)
        _add_transform_args(sp_t)
        sp_t.set_defaults(func=cmd_transform)

        sp_d = sub.add_parser("decompose", help="show character decomposition")
        sp_d.add_argument("char", type=str)
        sp_d.set_defaults(func=cmd_decompose)

        sp_n = sub.add_parser("neighbors", help="find nearest shanzi-neighbors")
        sp_n.add_argument("char", type=str)
        sp_n.add_argument("-n", "--count", type=int, default=10)
        sp_n.set_defaults(func=cmd_neighbors)

        args = p.parse_args(raw)
        if hasattr(args, "func"):
            args.func(args)
        else:
            p.print_help()
    else:
        # Shorthand: shanzi [-t T] [-s S] "text"
        p = argparse.ArgumentParser(
            prog="shanzi",
            description="Semantic ultrasound of obscure Chinese characters.",
        )
        _add_global_args(p)
        p.add_argument("text", nargs="?", type=str, default=None,
                        help="text to transform")
        _add_transform_args(p)
        args = p.parse_args(raw)
        if args.text is None and not args.stdin:
            p.print_help()
            sys.exit(0)
        cmd_transform(args)


if __name__ == "__main__":
    main()
