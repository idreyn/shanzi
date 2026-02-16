#!/usr/bin/env python3
"""
Demo of shanzi — semantic ultrasound of obscure Chinese characters.

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
    print("Characters sharing structural DNA cluster together:\n")
    for char, label in [("水", "water"), ("火", "fire"), ("木", "tree"),
                         ("心", "heart"), ("月", "moon"), ("山", "mountain")]:
        nbrs = engine.neighbors(char, k=5)
        nbr_str = " ".join(c for c, _ in nbrs)
        print(f"  {char} ({label:8s}) → {nbr_str}")
    print()

    # ── Beam search with explain ───────────────────────────────────
    print("═══ MULTI-CHANNEL BEAM SEARCH ═══")
    print("Each character shares radicals with its neighbours (resonance).")
    print("Motif radicals recur at polyrhythmic intervals.\n")

    sentences = [
        "上海的天空是灰色的",
        "意识正在分崩离析",
        "镜海之上没有星光",
    ]

    for text in sentences:
        info = engine.explain(text, temperature=0.5, seed=42)
        motif_str = " ".join(m["radical"] for m in info["motifs"])
        print(f"  原文: {text}")
        print(f"  山字: {info['output']}")
        print(f"  motifs: {motif_str}")

        for c in info["chars"]:
            res = "".join(c["resonance_with_prev"][:4]) or "·"
            shared = "".join(c["shared_with_input"][:3]) or "·"
            print(f"    {c['input']}→{c['output']}  ←{res:5s}  ≈{shared}")
        print()

    # ── Temperature sweep ──────────────────────────────────────────
    print("═══ TEMPERATURE SWEEP ═══")
    print("Low → semantic fidelity.  High → radical resonance.\n")

    text = "那些字符没有固定的意义"
    print(f"  原文: {text}")
    for t in [0.1, 0.3, 0.5, 0.8, 1.0]:
        result = engine.transform(text, temperature=t, seed=42)
        print(f"  t={t:.1f}: {result}")
    print()

    # ── Beam vs spline ─────────────────────────────────────────────
    print("═══ BEAM vs SPLINE ═══")
    text = "水火不容天地玄黄"
    print(f"  原文: {text}")
    print(f"  beam:   {engine.transform(text, seed=42, mode='beam')}")
    print(f"  spline: {engine.transform(text, seed=42, mode='spline')}")
    print()

    print("Those characters are up there. Now they're doing something.")


if __name__ == "__main__":
    main()
