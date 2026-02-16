#!/usr/bin/env python3
"""
Demo of shanzi — semantic ultrasound of obscure Chinese characters.

Run:
    python examples/demo.py
"""

from shanzi import ShanziEngine


def main():
    print("Initialising shanzi engine...")
    engine = ShanziEngine(quiet=False)
    print()

    # ── Decomposition ──────────────────────────────────────────────
    print("═══ CHARACTER DECOMPOSITION ═══")
    for char in ["晒", "粒", "想", "龍"]:
        tree = engine.decompose(char)
        if tree:
            parts_str = " → ".join(
                f"{k}=[{''.join(v)}]" for k, v in tree.items()
            )
            print(f"  {char}: {parts_str}")
        else:
            print(f"  {char}: atomic")
    print()

    # ── Neighbors ──────────────────────────────────────────────────
    print("═══ SHANZI NEIGHBORS ═══")
    print("Characters that share structural DNA cluster together:\n")
    for char, label in [("水", "water"), ("火", "fire"), ("木", "tree"),
                         ("心", "heart"), ("月", "moon"), ("山", "mountain")]:
        nbrs = engine.neighbors(char, k=5)
        nbr_str = " ".join(c for c, _ in nbrs)
        print(f"  {char} ({label:8s}) → {nbr_str}")
    print()

    # ── Transform ──────────────────────────────────────────────────
    print("═══ SHANZI TRANSFORM ═══")
    print("Drifting sentences through component-meaning space:\n")

    sentences = [
        "今天天气很好",
        "上海的天空是灰色的",
        "我在镜海之上",
        "意识正在分崩离析",
    ]

    for text in sentences:
        print(f"  原文: {text}")
        for temp in [0.1, 0.3, 0.5]:
            result = engine.transform(text, temperature=temp, seed=42)
            print(f"  t={temp}: {result}")
        print()

    # ── Multiple seeds ─────────────────────────────────────────────
    print("═══ SAME TEXT, DIFFERENT SEEDS ═══")
    text = "那些字符没有固定的意义"
    print(f"  原文: {text}\n")
    for seed in [1, 42, 137, 256, 999]:
        result = engine.transform(text, temperature=0.3, seed=seed)
        print(f"  seed={seed:3d}: {result}")
    print()

    print("Done. These characters are up there. Now they're doing something.")


if __name__ == "__main__":
    main()
