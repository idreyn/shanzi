# shanzi 山字

**Semantic ultrasound of obscure Chinese characters.**

> *Cai Duofan spoke in curlicue Shanghai slang no dictionary understood.
> She wrote in shanzi, that semantic ultrasound of obscure Chinese
> characters floating disused above the spoken language.  I'd have to
> hit my tryptamine pen, sometimes, to make sense of her messages...*
>
> — *Upon the Mirror Sea*

---

Chinese has ~106,000 characters in Unicode.  Most are alien and
unpronounceable to students, teachers, and natives alike.  They're up
there.  They're just not doing anything.

**Shanzi** puts them to work.

Each character is made of radicals and sub-components that carry wisps of
meaning.  Shanzi builds *recursive embeddings* that smear each
character's semantics through the full tree of its parts, then treats a
sentence as a **path through that semantic space** — curve-fits it, jitters
it, resamples it — and decodes back to characters that are nearby but
*structurally alien*.  The output is the written equivalent of a
tryptamine-pen hallucination: almost legible, semantically adjacent, and
deeply strange.

## Quick start

```bash
pip install -e .
shanzi transform "今天天气很好"
```

First run computes embeddings for ~89,000 characters (~12 seconds).
Subsequent runs load from cache (~1 second).

## What it does

### 1. Decompose characters into sub-components

晒 → [日, 西] → 日 is atomic; 西 → [一, 𠁤] → ...

```bash
shanzi decompose 晒
```

```json
{
  "晒": ["日", "西"],
  "西": ["一", "𠁤"],
  "𠁤": ["儿", "口"],
  "儿": ["丿", "乚"]
}
```

Uses the [CHISE/cjkvi-ids](https://github.com/cjkvi/cjkvi-ids)
Ideographic Description Sequence database (bundled).

### 2. Build shanzi embeddings

Topologically sort characters so each comes after its parts, then:

```
S(h) = normalise( W2V(h) + k · Σ S(part) )
```

where `k = 0.67` (configurable).  The base embedding `W2V(h)` is either
a pre-trained word2vec model (if you supply one) or a deterministic
structural vector seeded by Unicode codepoint.  The structural mode
needs no external data and already produces meaningful geometry: 木 (tree)
clusters near 林 (grove = tree+tree) and 森 (forest = tree+tree+tree).

```bash
shanzi neighbors 木
```

```
  林  (dist=0.8523)
  森  (dist=0.9114)
  ...
```

### 3. Transform sentences through shanzi-space

A sentence is a discretised signal through embedding space.  We:

1. Map each character to its shanzi embedding
2. Fit a **B-spline** through the sequence
3. Add **smooth random perturbation** (controlled by temperature)
4. Resample at the original time-points
5. Decode each sample to its **nearest character** in shanzi-space

```bash
shanzi transform -t 0.3 -s 42 "上海的天空是灰色的"
```

The `--temperature` (`-t`) flag controls how far the output drifts from
the input.  Low temperatures stay close; high temperatures wander into
the deep stacks of CJK Extension B/C/D/E/F.

## CLI reference

```
shanzi [options] transform [-t TEMP] [-s SEED] TEXT
shanzi [options] decompose CHAR
shanzi [options] neighbors [-n COUNT] CHAR
```

**Global options:**

| Flag | Default | Description |
|---|---|---|
| `-k` | `0.67` | Component attenuation factor |
| `--dim` | `300` | Embedding dimensionality (structural mode) |
| `-e`, `--embeddings` | — | Path to pre-trained word2vec/gensim file |
| `-q`, `--quiet` | off | Suppress progress messages |

**Transform options:**

| Flag | Default | Description |
|---|---|---|
| `-t`, `--temperature` | `0.3` | Jitter magnitude (0 = identity) |
| `-s`, `--seed` | random | Random seed for reproducibility |

## Python API

```python
from shanzi import ShanziEngine

engine = ShanziEngine(k=0.67, dim=300)

# Transform text
output = engine.transform("今天天气很好", temperature=0.3, seed=42)

# Decompose a character
tree = engine.decompose("晒")

# Find semantic neighbors
neighbors = engine.neighbors("晒", k=10)

# Get the raw shanzi embedding vector
vec = engine.embedding("晒")
```

## Using pre-trained embeddings

The structural fallback works out of the box, but you get richer
semantics by plugging in real Chinese character embeddings:

```python
# word2vec text format
engine = ShanziEngine(embeddings_path="chinese_char_vectors.txt")

# gensim binary (requires: pip install gensim)
engine = ShanziEngine(embeddings_path="chinese_vectors.bin")
```

Any word2vec-format file with single-character entries will work.
Characters missing from the external model fall back to structural
embeddings automatically.

## Caching

Computed embeddings are cached to `~/.cache/shanzi/` (~47 MB).
Override with `SHANZI_CACHE=/path/to/dir`.  Delete the cache to
force recomputation.

## How it works, in more detail

**Decomposition.**  The IDS (Ideographic Description Sequences) database
encodes how characters are built from sub-components.  Twelve structural
operators describe spatial relationships (left-right, top-bottom, surround,
etc.).  We parse the tree, extract the component characters, and build a
dependency graph.

**Topological sort.**  Characters are ordered so that every component
appears before any character that uses it.  This lets us compute shanzi
embeddings in a single forward pass.

**Shanzi formula.**  For character *h* with components *p₁, p₂, ...*:

```
S(h) = normalise( W2V(h)  +  k · [ S(p₁) + S(p₂) + ... ] )
```

The attenuation factor *k* controls how much component meaning bleeds
into the character's own embedding.  At k=0, every character stands
alone.  As k grows, characters dissolve into the meanings of their parts.

**Curve fitting.**  A sentence of *n* characters becomes *n* points in
ℝ^300.  A B-spline of degree min(3, n-1) interpolates through them,
giving a smooth continuous path.

**Jitter.**  A second, lower-frequency B-spline (fewer control points)
generates smooth random perturbation, scaled by the temperature
parameter.  This is added to the main curve.

**Resampling.**  The perturbed curve is evaluated at the original
time-points.  Each resulting vector is decoded to its nearest CJK
character in shanzi-space via a KD-tree index.

## Requirements

- Python ≥ 3.10
- numpy ≥ 1.24
- scipy ≥ 1.10
- (optional) gensim ≥ 4.0 for loading pre-trained embeddings

## License

MIT

## Acknowledgements

- Character decomposition data from [cjkvi-ids](https://github.com/cjkvi/cjkvi-ids),
  based on the [CHISE](https://www.chise.org/) IDS Database
- The concept of shanzi is from [*Upon the Mirror Sea*](https://www.royalroad.com/fiction/44188/upon-the-mirror-sea)
  by a]pod[
