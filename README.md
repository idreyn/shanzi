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
character's semantics through the full tree of its parts, then uses
**multi-channel beam search** to produce output where adjacent characters
share visible radicals — structural "chords" — while selected radical
motifs recur at polyrhythmic intervals, threading leitmotifs through the
text at different time signatures.

The output is the written equivalent of a tryptamine-pen hallucination:
semantically adjacent, structurally resonant, and deeply strange.

## Quick start

```bash
pip install -e .
shanzi "今天天气很好"
```

First run computes embeddings for ~89,000 characters (~12 seconds).
Subsequent runs load from cache (~2 seconds).

## What it does

### 1. Decompose characters into sub-components

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
Ideographic Description Sequence database (bundled, ~89K characters).

### 2. Build shanzi embeddings

Topologically sort characters leaves-first, then:

```
S(h) = normalise( W2V(h) + k · Σ S(part) )
```

where `k = 0.67`.  The base embedding is either a pre-trained word2vec
model (if you supply one) or a deterministic structural vector seeded by
Unicode codepoint.  The structural mode needs no external data and
already produces meaningful geometry: 木 (tree) clusters near 林
(grove = tree+tree) and 森 (forest = tree×3).

```bash
shanzi neighbors 木
```

### 3. Generate shanzi via multi-channel beam search

Three channels carry meaning simultaneously:

- **Semantic (melody):** shanzi embeddings preserve the input's meaning
- **Resonance (harmony):** adjacent characters share visible radicals,
  creating structural chords
- **Motif (bass line):** selected radicals recur at polyrhythmic intervals

```bash
shanzi -x -s 42 "意识正在分崩离析"
```

```
  in:  意识正在分崩离析
  out: 𩍖谙𥪃𫞼𨐳𭤯謧𬄀

  motifs:
    一  period=2.0  phase=...
    亠  period=3.0  phase=...
    丶  period=5.0  phase=...

  pos  in → out   resonance         motifs
  ───────────────────────────────────────────────────────
    0  意 → 𩍖   ←prev: ·       motifs: ·
    1  识 → 谙   ←prev: 一丶丷亠    motifs: 一
    2  正 → 𥪃   ←prev: 一丶丷亠    motifs: ·
    3  在 → 𫞼   ←prev: 一丶丷亠    motifs: 一
    4  分 → 𨐳   ←prev: 一丶丷亠    motifs: 丶
    5  崩 → 𭤯   ←prev: 一丶丿亠    motifs: 一
    6  离 → 謧   ←prev: 一丶亠     motifs: ·
    7  析 → 𬄀   ←prev: 一丶      motifs: 一
```

Notice the resonance column: a thread of shared radicals (一, 丶, 丷, 亠)
runs through six consecutive characters.  The motif 一 pulses at period 2.

The `--temperature` (`-t`) flag controls how the channels balance:

- **Low temperature** → semantics dominate, output stays close to input
- **High temperature** → structural channels take over, meaning dissolves
  into radical echoes — the deep shanzi experience

## CLI reference

```
shanzi [options] "text"
shanzi [options] transform [-t TEMP] [-s SEED] [--mode beam|spline] TEXT
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
| `-t`, `--temperature` | `0.5` | 0 = close to input, 1+ = deep shanzi |
| `-s`, `--seed` | random | Random seed for reproducibility |
| `-a`, `--annotate` | off | Show input→output mapping with decomposition |
| `-x`, `--explain` | off | Full diagnostic: motifs, schedules, resonance |
| `--mode` | `beam` | `beam` (multi-channel) or `spline` (v1 drift) |
| `--stdin` | off | Read from stdin |

## Python API

```python
from shanzi import ShanziEngine

engine = ShanziEngine(k=0.67, dim=300)

# Transform text (beam search, default)
output = engine.transform("今天天气很好", temperature=0.5, seed=42)

# Transform with full diagnostic
info = engine.explain("今天天气很好", temperature=0.5, seed=42)
# info["output"], info["motifs"], info["chars"][i]["resonance_with_prev"]

# Spline mode (v1)
output = engine.transform("今天天气很好", mode="spline", seed=42)

# Decompose, neighbors, embedding
tree = engine.decompose("晒")
neighbors = engine.neighbors("晒", k=10)
vec = engine.embedding("晒")
```

## Using pre-trained embeddings

The structural fallback works out of the box, but you get richer
semantics by plugging in real Chinese character embeddings:

```python
engine = ShanziEngine(embeddings_path="chinese_char_vectors.txt")
engine = ShanziEngine(embeddings_path="chinese_vectors.bin")  # needs gensim
```

## How it works

See [THEORY.md](THEORY.md) for the full derivation.  Summary:

**Shanzi embeddings.** Each character's embedding blends its own base
vector with attenuated copies of its components' embeddings, recursively.
This creates a space where characters sharing structural parts cluster
together.

**Multi-channel beam search.** For each position in the output, we
gather ~300 candidate characters from the embedding space and the radical
inverted index.  Each candidate is scored by:

```
score = w_sem · cos_sim(S(candidate), S(input))    // semantic fidelity
      + w_res · jaccard(components, prev_components) // radical resonance
      + w_mot · Σ schedule_weight · [motif ∈ components]  // motif presence
      + gumbel_noise · temperature                   // diversity
```

Temperature modulates the balance: low T → semantics dominate; high T →
structural channels take over and meaning dissolves into radical echoes.

**Motif scheduling.** Before generation, 2–4 radical motifs are
selected from the input text's component palette and assigned
polyrhythmic periods (2, 3, 5, 7, ...) with random phases.  The result:
motif radicals pulse through the output at different frequencies,
creating complex interference patterns — leitmotifs drifting along at
different time signatures.

## Requirements

- Python ≥ 3.10
- numpy ≥ 1.24
- scipy ≥ 1.10
- (optional) gensim ≥ 4.0 for loading pre-trained embeddings

## Caching

Computed embeddings are cached to `~/.cache/shanzi/` (~47 MB).
Override with `SHANZI_CACHE=/path/to/dir`.  Delete the cache to
force recomputation.

## License

MIT

## Acknowledgements

- Character decomposition data from [cjkvi-ids](https://github.com/cjkvi/cjkvi-ids),
  based on the [CHISE](https://www.chise.org/) IDS Database
- The concept of shanzi is from [*Upon the Mirror Sea*](https://www.royalroad.com/fiction/44188/upon-the-mirror-sea)
  by a\]pod\[
