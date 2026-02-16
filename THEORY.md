# Theory of Shanzi Generation

## What is shanzi?

Shanzi is not a cypher — it doesn't map characters one-to-one to a secret
alphabet.  It is not a substitution code.  It is a *medium*, closer to
music than to encryption: a way of writing where meaning becomes
**polyphonic**, carried simultaneously by multiple overlapping channels
that operate at different speeds.

From the novel:

> *characters with no fixed meaning, radicals echoing with sinister
> suggestion*

The key word is **echoing**.  The sub-parts of Chinese characters —
radicals, phonetic components, structural elements — begin to resonate
with each other *across* the boundaries of individual characters.  In
normal Chinese, 日 (sun) inside 晒 (to dry in the sun) is bound to that
character; it means nothing on its own within 晒.  In shanzi, the 日
leaks out.  A neighbouring character might contain 月 (moon), creating a
day/night chord.  Another might carry 明 (bright = sun + moon),
completing the triad.  These threads of sub-character meaning drift
along at different wavelengths, and the "sentence" becomes a
superposition of signals.

## Why v1 (spline jitter) is incomplete

The original implementation treats a sentence as a path through embedding
space, jitters the path with a B-spline perturbation, and decodes each
point back to the nearest character independently.

This produces obscure characters, but it doesn't produce *shanzi*,
because:

1. **No inter-character awareness.**  Each output character is selected
   in isolation.  The radicals of character 3 have no relationship to
   the radicals of characters 2 or 4.  There is no echoing.

2. **No visual threads.**  In real shanzi, you should be able to *see*
   radical motifs threading through the text — the same component
   appearing, disappearing, mutating from character to character.
   The spline approach can't produce this because it operates entirely
   in continuous embedding space, oblivious to discrete structure.

3. **No polyrhythm.**  The novel describes "leitmotifs drifting along
   at different time signatures."  This implies multiple patterns
   overlapping at different periods — radical A appearing every 2
   characters, radical B every 3, their interference pattern creating
   complex texture.  The spline approach has no mechanism for this.

## The multi-channel model

We model shanzi as a signal transmitted simultaneously on three channels:

### Channel 1: Semantic drift (the melody)

The overall meaning of the input text should be *gesturally* preserved.
If the input says "the sky is grey today," the output should, when
squinted at through component meanings, wave its hands in the direction
of sky and grey and today.

We use the shanzi embeddings `S(h)` for this: each output character
should be *near* the corresponding input character in shanzi embedding
space.  The embeddings already blend component meaning recursively,
so "near" means "structurally related, not just semantically synonymous."

### Channel 2: Radical resonance (the harmony)

Adjacent output characters should share visible sub-components, creating
structural "chords."  If character *i* contains 木 (tree), character
*i+1* might contain 林 (grove = tree+tree) or 森 (forest = tree×3) or
杜 (a surname, tree+earth) — anything where the 木 radical visually
continues into the next character.

We measure this as the **Jaccard overlap** between the recursive
component sets of adjacent characters.  This creates chains of
visual resonance: each character hands off at least one radical to
its neighbour, like voices overlapping in a round.

### Channel 3: Radical motifs (the bass line)

Before generation begins, we select 2–4 radical *motifs* drawn from
the input text's component palette.  Each motif is assigned a
**polyrhythmic schedule** — a period and phase that determine at which
positions it should appear:

```
Motif 日 (sun):  period 2, phase 0  →  positions 0, 2, 4, 6, ...
Motif 木 (tree): period 3, phase 1  →  positions 1, 4, 7, 10, ...
Motif 口 (mouth):period 5, phase 0  →  positions 0, 5, 10, ...
```

The periods are chosen from primes and near-primes (2, 3, 5, 7) to
avoid exact alignment, creating complex interference patterns.  The
schedule weight at position *p* is:

```
w(p) = (cos(2π · (p − phase) / period) + 1) / 2
```

This is a smooth pulse, maximal at scheduled positions, fading
to zero between them.  Characters that contain a motif radical at a
high-weight position get a scoring bonus.

The result: when you look at the output text, the motif radicals
appear and disappear rhythmically, like instruments entering and
leaving a score.

## The generation algorithm

### Beam search

We use beam search rather than independent per-position decoding.
This is essential: the resonance channel requires each character
to be chosen *in the context of the previous one*.

**State.**  Each beam carries the sequence of output characters so far,
a cumulative score, and the component set of the last character (for
computing resonance with the next).

**Candidates.**  For each position we gather a pool of ~300–500
candidate characters from two sources:
- **Semantic neighbors** (top-K from the KD-tree) — characters
  near the input character in shanzi embedding space.
- **Structural candidates** — characters containing a motif radical,
  sampled from the radical inverted index.

**Scoring.**  Each candidate is scored by a weighted combination:

```
score(c, pos) = w_sem  · cos_sim(S(c), S(input[pos]))     // melody
              + w_res  · jaccard(components(c), prev_comps) // harmony
              + w_mot  · Σ schedule_i.weight(pos) · [radical_i ∈ components(c)]
              + noise                                        // diversity
```

The **noise** term is Gumbel noise scaled by temperature, which is
the principled way to sample from a softmax.  At temperature 0,
the search is greedy; at higher temperatures, candidates compete
more fairly and surprising choices emerge.

**Temperature scaling.**  We modulate the channel weights by
temperature:

```
effective_semantic  = w_sem  · (1 − 0.3·T)    // loosens with temperature
effective_resonance = w_res  · (1 + 0.5·T)    // tightens with temperature
effective_motif     = w_mot  · (1 + 0.5·T)
noise_scale         = 0.15 · T
```

At low T, semantics dominates — output stays close to the input.
At high T, structural channels take over — meaning dissolves into
radical echoes, which is the deep shanzi experience described in the
novel as requiring a "tryptamine pen" to decode.

### Motif planning

Motif radicals are selected by:

1. Recursively decomposing every input character to its full
   component set.
2. Counting how many input characters contain each component.
3. Filtering to components that appear in at least 10 candidate
   characters (so the beam search has room to work).
4. Taking the top 2–4 by count.

This selects radicals that are *thematically central* to the input,
ensuring the output's visual leitmotifs are related to the input's
meaning.

### The component index

We precompute two data structures:

- `char → frozenset(components)`: for every character, its full
  recursive set of all sub-components at every depth.
- `component → [characters]`: the inverse — for every component,
  which characters contain it.

The first enables fast Jaccard computation for resonance scoring.
The second enables radical-aware candidate generation.  Both are
built once at startup using dynamic programming over the topological
order (bottom-up: each character's components = its direct parts ∪
their components recursively).

## What the output should feel like

Good shanzi reads like a familiar sentence seen through frosted glass
and synesthesia.  The characters are individually alien but collectively
suggestive.  If you know Chinese, you might catch a radical-thread and
feel a flicker of meaning — *was that 水 again? why does every other
character have 口?* — without being able to pin it to a dictionary
definition.  The meaning is in the *pattern*, not the tokens.

This is exactly what Mona describes: "characters with no fixed meaning,
radicals echoing with sinister suggestion."  The meaning isn't in any
single character.  It's in the resonance between them.
