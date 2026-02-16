"""IDS parsing and Hanzi component graph utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable, Mapping
import urllib.request

DEFAULT_IDS_URL = "https://raw.githubusercontent.com/cjkvi/cjkvi-ids/master/ids.txt"

IDS_OPERATORS_ARITY: dict[str, int] = {
    "⿰": 2,
    "⿱": 2,
    "⿴": 2,
    "⿵": 2,
    "⿶": 2,
    "⿷": 2,
    "⿸": 2,
    "⿹": 2,
    "⿺": 2,
    "⿻": 2,
    "⿲": 3,
    "⿳": 3,
}

_IDS_REGION_TAG_RE = re.compile(r"\[[^\]]*\]")
_WS_RE = re.compile(r"\s+")


class IDSParseError(ValueError):
    """Raised when an IDS expression cannot be parsed."""


@dataclass(frozen=True)
class IDSNode:
    """Parsed IDS node."""

    token: str
    children: tuple["IDSNode", ...] = ()

    @property
    def is_operator(self) -> bool:
        return self.token in IDS_OPERATORS_ARITY

    def serialize(self) -> str:
        if not self.children:
            return self.token
        return self.token + "".join(child.serialize() for child in self.children)


ComponentGraph = dict[str, list[str]]


def clean_ids_expression(expression: str) -> str:
    """Remove region tags and whitespace from IDS expressions."""
    no_tags = _IDS_REGION_TAG_RE.sub("", expression)
    return _WS_RE.sub("", no_tags)


def parse_ids_expression(expression: str) -> IDSNode:
    """Parse an IDS expression into a tree."""
    cleaned = clean_ids_expression(expression)
    if not cleaned:
        raise IDSParseError("Empty IDS expression.")
    tokens = list(cleaned)
    node, offset = _parse_tokens(tokens, 0)
    if offset != len(tokens):
        trailing = "".join(tokens[offset:])
        raise IDSParseError(f"Unparsed IDS trailing tokens: {trailing!r}")
    return node


def _parse_tokens(tokens: list[str], offset: int) -> tuple[IDSNode, int]:
    if offset >= len(tokens):
        raise IDSParseError("Unexpected end of IDS expression.")
    token = tokens[offset]
    if token not in IDS_OPERATORS_ARITY:
        return IDSNode(token=token), offset + 1

    children: list[IDSNode] = []
    next_offset = offset + 1
    for _ in range(IDS_OPERATORS_ARITY[token]):
        child, next_offset = _parse_tokens(tokens, next_offset)
        children.append(child)
    return IDSNode(token=token, children=tuple(children)), next_offset


def parse_ids_lines(lines: Iterable[str]) -> dict[str, str]:
    """Parse `ids.txt` lines into a character -> expression mapping."""
    decompositions: dict[str, str] = {}
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        columns = line.split("\t")
        if len(columns) < 3:
            continue
        character = columns[1].strip()
        if not character:
            continue
        expression = clean_ids_expression(columns[2])
        if not expression:
            continue
        # Keep the first decomposition found.
        decompositions.setdefault(character, expression)
    return decompositions


def download_ids_file(target_path: Path, source_url: str = DEFAULT_IDS_URL) -> Path:
    """Download IDS data to `target_path` if it does not exist."""
    if target_path.exists():
        return target_path
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(source_url, timeout=60) as response:
        payload = response.read()
    target_path.write_bytes(payload)
    return target_path


def load_ids_decompositions(ids_path: Path) -> dict[str, str]:
    """Load IDS decomposition table from disk."""
    lines = ids_path.read_text(encoding="utf-8").splitlines()
    return parse_ids_lines(lines)


def build_component_graph(decompositions: Mapping[str, str]) -> ComponentGraph:
    """
    Build a DAG of components for characters and nested IDS expression nodes.

    Character keys map to direct components. Nested components that are not
    single characters are represented by synthetic keys, prefixed with `expr:`.
    """
    graph: ComponentGraph = {}

    def materialize(node: IDSNode) -> str:
        if not node.children:
            graph.setdefault(node.token, [])
            return node.token
        node_key = f"expr:{node.serialize()}"
        if node_key not in graph:
            graph[node_key] = [materialize(child) for child in node.children]
        return node_key

    for character, expression in decompositions.items():
        root = parse_ids_expression(expression)
        if root.is_operator:
            graph[character] = [materialize(child) for child in root.children]
        elif root.token != character:
            graph[character] = [materialize(root)]
        else:
            graph.setdefault(character, [])
        graph.setdefault(character, graph[character])
    return graph


def recursive_components(node_id: str, graph: Mapping[str, list[str]]) -> list[str]:
    """Return depth-first recursive components for one node."""
    output: list[str] = []
    seen: set[str] = set()

    def dfs(current: str) -> None:
        for child in graph.get(current, []):
            if child in seen:
                continue
            seen.add(child)
            output.append(child)
            dfs(child)

    dfs(node_id)
    return output


def reachable_subgraph(graph: Mapping[str, list[str]], roots: Iterable[str]) -> ComponentGraph:
    """Keep only nodes reachable from the given root nodes."""
    keep: set[str] = set()
    stack = list(roots)
    while stack:
        node = stack.pop()
        if node in keep:
            continue
        keep.add(node)
        stack.extend(graph.get(node, []))
    return {node: [child for child in graph.get(node, []) if child in keep] for node in keep}


def is_hanzi(char: str) -> bool:
    """Return True if a token is likely a Han ideograph."""
    if len(char) != 1:
        return False
    codepoint = ord(char)
    ranges = (
        (0x3400, 0x4DBF),    # CJK Ext A
        (0x4E00, 0x9FFF),    # CJK Unified Ideographs
        (0xF900, 0xFAFF),    # CJK Compatibility Ideographs
        (0x20000, 0x2A6DF),  # CJK Ext B
        (0x2A700, 0x2B73F),  # CJK Ext C
        (0x2B740, 0x2B81F),  # CJK Ext D
        (0x2B820, 0x2CEAF),  # CJK Ext E
        (0x2CEB0, 0x2EBEF),  # CJK Ext F
        (0x30000, 0x3134F),  # CJK Ext G/H
    )
    return any(start <= codepoint <= end for start, end in ranges)


def is_character_token(token: str) -> bool:
    """Identify decodable character-like tokens."""
    return len(token) == 1 and token not in IDS_OPERATORS_ARITY and not token.startswith("expr:")


def unique_hanzi_in_text(text: str) -> list[str]:
    """Extract unique Hanzi from text in first-seen order."""
    seen: set[str] = set()
    output: list[str] = []
    for char in text:
        if not is_hanzi(char) or char in seen:
            continue
        seen.add(char)
        output.append(char)
    return output
