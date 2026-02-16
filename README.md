# shanzi

Shanzi is a minimal, runnable prototype for:

1. Building recursive Hanzi component graphs from IDS decomposition data.
2. Computing recursive character embeddings:

   \[
   S(h) = \text{normalize}\left(W(h) + k \sum_{p \in \text{parts}(h)} S(p)\right)
   \]

   with default \(k=0.67\).
3. Treating a sentence as a path through embedding space, smoothing/resampling it,
   adding jitter, and decoding back into characters.

The result is a "semantic ultrasound" remix pipeline inspired by your prompt.

## What this MVP includes

- IDS parser for CJKVI `ids.txt` decompositions (roughly 88k entries).
- Recursive component DAG with synthetic nodes for nested IDS subexpressions.
- Word2Vec-like baseline embeddings via PPMI + SVD over character co-occurrence.
- Recursive Shanzi propagation over topological order of components.
- Sentence trajectory smoothing (Catmull-Rom style), arc-length resampling, jitter.
- Nearest-neighbor decoding back to Hanzi tokens.
- CLI and tests.

## Quickstart

```bash
python3 -m pip install -e ".[dev]"
```

### 1) Inspect components

```bash
shanzi components --chars 晒西兀 --ids-path data/ids.txt
```

Example output shape:

```json
{
  "晒": {
    "direct": ["日", "西"],
    "recursive": ["日", "西", "兀", "一", "儿", "口"]
  }
}
```

### 2) Build recursive embeddings

```bash
shanzi build \
  --corpus examples/corpus.txt \
  --model-out models/shanzi_model.npz \
  --ids-path data/ids.txt \
  --dim 64 \
  --k 0.67
```

By default this builds from Hanzi roots found in the corpus and recursively pulls
all required components. Use `--all-characters` for full-table propagation.

### 3) Remix a sentence as a semantic trajectory

```bash
shanzi remix \
  --model models/shanzi_model.npz \
  --sentence "上海夜雨把语言洗亮" \
  --samples-per-segment 8 \
  --jitter-std 0.03
```

## Python API

```python
from pathlib import Path
from shanzi.ids import download_ids_file, load_ids_decompositions
from shanzi.model import build_model_from_corpus
from shanzi.trajectory import remix_sentence

ids_path = Path("data/ids.txt")
download_ids_file(ids_path)
decompositions = load_ids_decompositions(ids_path)

corpus = Path("examples/corpus.txt").read_text(encoding="utf-8")
model = build_model_from_corpus(corpus, decompositions, dim=64, k=0.67)
print(remix_sentence("上海夜雨把语言洗亮", model))
```

## Project layout

```text
shanzi/
  ids.py          # IDS parsing + component graph
  embeddings.py   # co-occurrence embedding + recursive propagation
  model.py        # model container / save / load
  trajectory.py   # path smoothing + resampling + jitter + decode
  cli.py          # CLI entrypoint
tests/
examples/
```
