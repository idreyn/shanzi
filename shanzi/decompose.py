"""
Character decomposition via Ideographic Description Sequences (IDS).

The CHISE / cjkvi-ids database encodes how every CJK character is built
from smaller parts using twelve structural operators:

    Binary (2 operands):  ⿰ ⿱ ⿴ ⿵ ⿶ ⿷ ⿸ ⿹ ⿺ ⿻
    Ternary (3 operands): ⿲ ⿳

Example: 晒 = ⿰日西  →  components [日, 西]
         粒 = ⿰米立  →  components [米, 立]

This module parses IDS strings, builds a directed graph of component
relationships, and provides topological ordering so that every character
is processed after all of its parts.
"""

import os
import re
from collections import defaultdict, deque
from typing import Dict, List, Optional, Set

IDS_BINARY = set("⿰⿱⿴⿵⿶⿷⿸⿹⿺⿻")
IDS_TERNARY = set("⿲⿳")
IDS_OPERATORS = IDS_BINARY | IDS_TERNARY

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def _is_cjk(ch: str) -> bool:
    """Return True if *ch* is a CJK ideograph (including extensions and compat)."""
    cp = ord(ch)
    return (
        0x4E00 <= cp <= 0x9FFF       # CJK Unified Ideographs
        or 0x3400 <= cp <= 0x4DBF    # Extension A
        or 0x20000 <= cp <= 0x2A6DF  # Extension B
        or 0x2A700 <= cp <= 0x2B73F  # Extension C
        or 0x2B740 <= cp <= 0x2B81F  # Extension D
        or 0x2B820 <= cp <= 0x2CEAF  # Extension E
        or 0x2CEB0 <= cp <= 0x2EBEF  # Extension F
        or 0xF900 <= cp <= 0xFAFF    # CJK Compatibility Ideographs
        or 0x2F800 <= cp <= 0x2FA1F  # CJK Compat. Supplement
        or 0x2E80 <= cp <= 0x2EFF   # CJK Radicals Supplement
        or 0x2F00 <= cp <= 0x2FDF   # Kangxi Radicals
        or 0x31C0 <= cp <= 0x31EF   # CJK Strokes
    )


def _is_cjk_primary(ch: str) -> bool:
    """Return True if *ch* is a CJK ideograph from the main or extension blocks.

    Excludes CJK Compatibility Ideographs (which duplicate characters from
    the main block) to avoid showing the same glyph twice in neighbor lists.
    """
    cp = ord(ch)
    return (
        0x4E00 <= cp <= 0x9FFF       # CJK Unified Ideographs
        or 0x3400 <= cp <= 0x4DBF    # Extension A
        or 0x20000 <= cp <= 0x2A6DF  # Extension B
        or 0x2A700 <= cp <= 0x2B73F  # Extension C
        or 0x2B740 <= cp <= 0x2B81F  # Extension D
        or 0x2B820 <= cp <= 0x2CEAF  # Extension E
        or 0x2CEB0 <= cp <= 0x2EBEF  # Extension F
    )


class Decomposer:
    """Decomposes Chinese characters into their structural sub-components."""

    def __init__(self, ids_path: Optional[str] = None):
        if ids_path is None:
            ids_path = os.path.join(DATA_DIR, "ids.txt")
        self.components: Dict[str, List[str]] = {}
        self._load(ids_path)

    # ------------------------------------------------------------------
    # Loading and parsing
    # ------------------------------------------------------------------

    def _load(self, path: str):
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                fields = line.split("\t")
                if len(fields) < 3:
                    continue
                char = fields[1]
                if len(char) != 1:
                    continue

                # Use the first IDS variant; strip region tags like [GJK]
                ids_raw = fields[2]
                ids_str = re.sub(r"\[.*?\]", "", ids_raw).strip()
                if not ids_str or ids_str == char:
                    continue

                parts = self._extract_components(ids_str)
                # Remove self-references, IDS operators, and non-single chars
                parts = [
                    p for p in parts
                    if p != char and len(p) == 1 and p not in IDS_OPERATORS
                ]
                if parts:
                    self.components[char] = parts

    def _extract_components(self, ids: str) -> List[str]:
        """Return the direct top-level components of an IDS string.

        When a top-level component is itself an IDS subtree (i.e. a
        component without its own codepoint), we flatten it to its leaf
        characters instead.
        """
        chars = list(ids)
        if not chars or chars[0] not in IDS_OPERATORS:
            return []

        pos = [0]

        def _skip():
            """Advance past one component; return its leaf characters."""
            if pos[0] >= len(chars):
                return []
            ch = chars[pos[0]]
            if ch in IDS_BINARY:
                pos[0] += 1
                return _skip() + _skip()
            elif ch in IDS_TERNARY:
                pos[0] += 1
                return _skip() + _skip() + _skip()
            else:
                pos[0] += 1
                return [ch]

        def _top():
            """Read one top-level component."""
            if pos[0] >= len(chars):
                return []
            ch = chars[pos[0]]
            if ch in IDS_OPERATORS:
                # Nested subtree — flatten to leaves
                return _skip()
            else:
                pos[0] += 1
                return [ch]

        arity = 2 if chars[0] in IDS_BINARY else 3
        pos[0] = 1
        result: List[str] = []
        for _ in range(arity):
            result.extend(_top())
        return result

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get(self, char: str) -> List[str]:
        """Direct sub-components of *char*, or empty list if atomic."""
        return self.components.get(char, [])

    def get_recursive(self, char: str) -> Dict[str, List[str]]:
        """Full decomposition tree rooted at *char*."""
        tree: Dict[str, List[str]] = {}
        queue = deque([char])
        seen: Set[str] = set()
        while queue:
            ch = queue.popleft()
            if ch in seen:
                continue
            seen.add(ch)
            parts = self.get(ch)
            if parts:
                tree[ch] = parts
                for p in parts:
                    if p not in seen:
                        queue.append(p)
        return tree

    def all_chars(self) -> Set[str]:
        """Every character mentioned in the decomposition graph."""
        s: Set[str] = set(self.components.keys())
        for parts in self.components.values():
            s.update(parts)
        return s

    def topological_sort(self, chars: Optional[Set[str]] = None) -> List[str]:
        """Return *chars* (default: all known) ordered leaves-first.

        Every character appears after all of its components, so shanzi
        embeddings can be computed in a single forward pass.
        """
        if chars is None:
            chars = self.all_chars()

        # Build in-degree map: in_degree[ch] = # of components of ch that
        # are in the char set (and thus must be computed first).
        in_degree: Dict[str, int] = {}
        dependents: Dict[str, List[str]] = defaultdict(list)

        for ch in chars:
            deps = [p for p in self.components.get(ch, []) if p in chars]
            in_degree[ch] = len(deps)
            for p in deps:
                dependents[p].append(ch)

        queue = deque(ch for ch in chars if in_degree.get(ch, 0) == 0)
        order: List[str] = []
        while queue:
            ch = queue.popleft()
            order.append(ch)
            for dep in dependents.get(ch, []):
                in_degree[dep] -= 1
                if in_degree[dep] == 0:
                    queue.append(dep)

        # Characters caught in cycles (rare but possible in IDS data)
        remaining = chars - set(order)
        order.extend(remaining)
        return order
