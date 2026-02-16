"""
Multi-channel shanzi generation via beam search.

Three channels carry meaning simultaneously:

  Semantic (melody) — shanzi embeddings preserve the input's meaning.
  Resonance (harmony) — shared radicals between adjacent characters
      create visible structural "chords."
  Motif (bass line) — selected radicals recur at polyrhythmic intervals,
      threading leitmotifs through the text at different periods.

The beam search selects output characters that jointly optimise across
all three channels, producing text where the sub-character structure
carries its own parallel meaning — "radicals echoing with sinister
suggestion."

See THEORY.md for the full derivation.
"""

from __future__ import annotations

import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np

from shanzi.decompose import Decomposer, _is_cjk, _is_cjk_primary
from shanzi.embeddings import ShanziEmbeddings

# Polyrhythmic periods — primes and near-primes so that motifs
# rarely land on the same position, creating complex interference.
_PERIODS = [2, 3, 5, 7, 4, 6]


# ──────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────

@dataclass
class RadicalSchedule:
    """A radical motif that pulses through the text at a specific period."""
    radical: str
    period: float
    phase: float

    def weight_at(self, position: int) -> float:
        """Smooth cosine weighting — peaks at scheduled positions."""
        t = (position - self.phase) / self.period
        return (math.cos(2.0 * math.pi * t) + 1.0) / 2.0


@dataclass
class _Beam:
    """One hypothesis in the beam search."""
    chars: list              # output characters so far
    score: float             # cumulative score
    last_comps: frozenset    # component set of the last character


# ──────────────────────────────────────────────────────────────────────
# Generator
# ──────────────────────────────────────────────────────────────────────

class ShanziGenerator:
    """Generate shanzi text via multi-channel beam search.

    Parameters
    ----------
    embeddings : ShanziEmbeddings
    decomposer : Decomposer
    beam_width : int
        Number of hypotheses to keep at each step.
    n_semantic : int
        Semantic (KD-tree) candidates per position.
    n_radical : int
        Radical-index candidates per motif per position.
    w_semantic, w_resonance, w_motif : float
        Base channel weights (modulated by temperature at runtime).
    """

    def __init__(
        self,
        embeddings: ShanziEmbeddings,
        decomposer: Decomposer,
        beam_width: int = 24,
        n_semantic: int = 200,
        n_radical: int = 60,
        w_semantic: float = 1.0,
        w_resonance: float = 0.5,
        w_motif: float = 0.3,
        quiet: bool = False,
    ):
        self.emb = embeddings
        self.dec = decomposer
        self.beam_width = beam_width
        self.n_semantic = n_semantic
        self.n_radical = n_radical
        self.w_sem = w_semantic
        self.w_res = w_resonance
        self.w_mot = w_motif

        # component index: char → frozenset of all recursive sub-components
        self._char_comps: Dict[str, FrozenSet[str]] = {}
        # inverted index: component → list of CJK-primary chars containing it
        self._comp_to_chars: Dict[str, List[str]] = defaultdict(list)

        self._build_index(quiet)

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def _build_index(self, quiet: bool):
        """Build component sets (bottom-up DP) and inverted index."""
        if not quiet:
            print("[shanzi] building radical index ...", file=sys.stderr)

        # Bottom-up: topological order guarantees every part is computed
        # before any character that uses it.
        order = self.dec.topological_sort()
        for char in order:
            direct = self.dec.get(char)
            comps: Set[str] = set(direct)
            for part in direct:
                comps |= self._char_comps.get(part, frozenset())
            fs = frozenset(comps)
            self._char_comps[char] = fs

        # Inverted index (CJK-primary only, for candidate generation)
        vocab = self.emb.vocab  # cached set, O(1) lookup
        for char, comps in self._char_comps.items():
            if not _is_cjk_primary(char):
                continue
            if char not in vocab:
                continue
            for comp in comps:
                self._comp_to_chars[comp].append(char)

        if not quiet:
            print(
                f"[shanzi] indexed {len(self._char_comps):,} component sets, "
                f"{len(self._comp_to_chars):,} inverted entries.",
                file=sys.stderr,
            )

    # ------------------------------------------------------------------
    # Motif planning
    # ------------------------------------------------------------------

    def _plan_motifs(
        self,
        input_chars: List[str],
        n_motifs: int = 3,
    ) -> List[str]:
        """Select radical motifs from the input's component palette.

        Picks components that (a) recur across multiple input characters
        and (b) have enough candidate characters in the inverted index
        to give the beam search room to work.
        """
        counts: Counter = Counter()
        for ch in input_chars:
            for comp in self._char_comps.get(ch, frozenset()):
                counts[comp] += 1

        motifs: List[str] = []
        for comp, _count in counts.most_common(n_motifs * 5):
            if len(self._comp_to_chars.get(comp, [])) >= 20:
                motifs.append(comp)
            if len(motifs) >= n_motifs:
                break
        return motifs

    def _create_schedules(
        self,
        motifs: List[str],
        n_positions: int,
        rng: np.random.RandomState,
    ) -> List[RadicalSchedule]:
        """Assign polyrhythmic periods and random phases to motifs."""
        schedules: List[RadicalSchedule] = []
        for i, radical in enumerate(motifs):
            base_period = _PERIODS[i % len(_PERIODS)]
            # Scale period so patterns are visible within the text length
            period = max(2.0, base_period * max(1, n_positions / 10.0))
            phase = rng.uniform(0, period)
            schedules.append(RadicalSchedule(radical, period, phase))
        return schedules

    # ------------------------------------------------------------------
    # Candidate generation
    # ------------------------------------------------------------------

    def _get_candidates(
        self,
        input_char: str,
        motif_radicals: List[str],
        rng: np.random.RandomState,
    ) -> List[str]:
        """Gather candidate characters from semantic and structural sources."""
        pool: Set[str] = set()

        # Source 1: semantic neighbours from the KD-tree
        vec = self.emb.get(input_char)
        if vec is not None:
            hits = self.emb.nearest_cjk(
                vec, k=self.n_semantic, exclude={input_char},
            )
            pool.update(ch for ch, _d in hits)

        # Source 2: characters containing motif radicals
        for radical in motif_radicals:
            bucket = self._comp_to_chars.get(radical, [])
            if len(bucket) <= self.n_radical:
                pool.update(bucket)
            else:
                idxs = rng.choice(len(bucket), self.n_radical, replace=False)
                pool.update(bucket[i] for i in idxs)

        pool.discard(input_char)
        return list(pool)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _score(
        self,
        cand_vec: np.ndarray,
        cand_comps: FrozenSet[str],
        input_vec: np.ndarray,
        prev_comps: FrozenSet[str],
        schedules: List[RadicalSchedule],
        position: int,
        temperature: float,
    ) -> float:
        """Multi-channel score for placing *cand* at *position*.

        Temperature modulates the channel balance:
        - Low T  → semantics dominate (output stays close to input)
        - High T → structural channels take over (radical echoes)
        """
        # Adaptive weights
        t = temperature
        eff_sem = self.w_sem * (1.0 - 0.3 * min(t, 1.0))
        eff_res = self.w_res * (1.0 + 0.5 * t)
        eff_mot = self.w_mot * (1.0 + 0.5 * t)

        # Channel 1: semantic similarity (cosine, both unit vectors)
        sem = float(np.dot(cand_vec, input_vec))

        # Channel 2: radical resonance with previous character
        if prev_comps:
            overlap = len(cand_comps & prev_comps)
            union = len(cand_comps | prev_comps)
            res = overlap / max(union, 1)
        else:
            res = 0.0

        # Channel 3: motif presence at scheduled time
        mot = 0.0
        for sched in schedules:
            if sched.radical in cand_comps:
                mot += sched.weight_at(position)
        if schedules:
            mot /= len(schedules)

        return eff_sem * sem + eff_res * res + eff_mot * mot

    # ------------------------------------------------------------------
    # Main generation loop
    # ------------------------------------------------------------------

    def generate(
        self,
        text: str,
        temperature: float = 0.5,
        seed: Optional[int] = None,
    ) -> str:
        """Generate shanzi text via multi-channel beam search.

        Parameters
        ----------
        text : str
            Input text (may mix CJK and non-CJK characters).
        temperature : float
            Controls the balance between semantic fidelity and structural
            resonance.  0 → close to input, 1+ → deep shanzi.
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        str
            The shanzi-transformed text.
        """
        rng = np.random.RandomState(seed)
        tokens = list(text)

        # Identify transformable positions
        cjk_pos = [
            i for i, ch in enumerate(tokens)
            if _is_cjk(ch) and self.emb.get(ch) is not None
        ]
        if not cjk_pos:
            return text

        cjk_chars = [tokens[i] for i in cjk_pos]
        n = len(cjk_chars)

        # Plan motifs and schedules
        n_motifs = min(3, max(1, n // 2))
        motifs = self._plan_motifs(cjk_chars, n_motifs)
        schedules = self._create_schedules(motifs, n, rng)
        motif_rads = [s.radical for s in schedules]

        # Pre-compute per-position data
        input_vecs = [self.emb.get(ch) for ch in cjk_chars]
        candidates_per_pos: List[List[str]] = []
        for ch in cjk_chars:
            candidates_per_pos.append(
                self._get_candidates(ch, motif_rads, rng)
            )

        # Pre-compute candidate embeddings + component sets (deduplicated)
        cand_vecs: Dict[str, np.ndarray] = {}
        cand_comps: Dict[str, FrozenSet[str]] = {}
        for cands in candidates_per_pos:
            for ch in cands:
                if ch not in cand_vecs:
                    v = self.emb.get(ch)
                    if v is not None:
                        cand_vecs[ch] = v
                        cand_comps[ch] = self._char_comps.get(ch, frozenset())

        # Pre-compute base scores (semantic + motif) per (position, candidate)
        # These don't depend on the beam state so we compute them once.
        base_scores: Dict[Tuple[int, str], float] = {}
        for pos in range(n):
            iv = input_vecs[pos]
            t = temperature
            eff_sem = self.w_sem * (1.0 - 0.3 * min(t, 1.0))
            eff_mot = self.w_mot * (1.0 + 0.5 * t)
            for ch in candidates_per_pos[pos]:
                cv = cand_vecs.get(ch)
                if cv is None:
                    continue
                cc = cand_comps.get(ch, frozenset())
                sem = float(np.dot(cv, iv))
                mot = 0.0
                for sched in schedules:
                    if sched.radical in cc:
                        mot += sched.weight_at(pos)
                if schedules:
                    mot /= len(schedules)
                base_scores[(pos, ch)] = eff_sem * sem + eff_mot * mot

        # ── Beam search ───────────────────────────────────────────────
        beam = [_Beam(chars=[], score=0.0, last_comps=frozenset())]
        noise_scale = 0.15 * temperature
        eff_res = self.w_res * (1.0 + 0.5 * temperature)

        for pos in range(n):
            cands = candidates_per_pos[pos]
            if not cands:
                # No candidates — pass through input character
                for b in beam:
                    b.chars.append(cjk_chars[pos])
                continue

            expansions: List[_Beam] = []
            for b in beam:
                prev = b.last_comps
                for ch in cands:
                    bs = base_scores.get((pos, ch))
                    if bs is None:
                        continue
                    cc = cand_comps.get(ch, frozenset())

                    # Resonance with previous character (beam-dependent)
                    if prev:
                        overlap = len(cc & prev)
                        union_sz = len(cc | prev)
                        res = overlap / max(union_sz, 1)
                    else:
                        res = 0.0

                    # Gumbel noise for diversity
                    u = max(rng.random(), 1e-10)
                    noise = -math.log(-math.log(u)) * noise_scale

                    total = b.score + bs + eff_res * res + noise
                    expansions.append(
                        _Beam(chars=b.chars + [ch], score=total, last_comps=cc)
                    )

            # Keep top beam_width
            expansions.sort(key=lambda x: x.score, reverse=True)
            beam = expansions[:self.beam_width]

        if not beam:
            return text

        best = beam[0]

        # Write output characters back into token list
        for i, pos in enumerate(cjk_pos):
            if i < len(best.chars):
                tokens[pos] = best.chars[i]

        return "".join(tokens)

    # ------------------------------------------------------------------
    # Diagnostic: show what the generator planned
    # ------------------------------------------------------------------

    def explain(
        self,
        text: str,
        temperature: float = 0.5,
        seed: Optional[int] = None,
    ) -> dict:
        """Like generate() but returns a diagnostic dict.

        Includes the output text, the selected motifs, their schedules,
        and per-character resonance info.
        """
        rng = np.random.RandomState(seed)
        tokens = list(text)
        cjk_pos = [
            i for i, ch in enumerate(tokens)
            if _is_cjk(ch) and self.emb.get(ch) is not None
        ]
        if not cjk_pos:
            return {"input": text, "output": text, "motifs": [], "chars": []}

        cjk_chars = [tokens[i] for i in cjk_pos]
        n = len(cjk_chars)

        n_motifs = min(3, max(1, n // 2))
        motifs = self._plan_motifs(cjk_chars, n_motifs)
        schedules = self._create_schedules(motifs, n, rng)

        # Re-seed the rng to get the same result as generate()
        rng2 = np.random.RandomState(seed)
        output = self.generate(text, temperature=temperature, seed=seed)

        out_tokens = list(output)
        out_cjk = [out_tokens[i] for i in cjk_pos]

        char_info = []
        for pos in range(n):
            in_ch = cjk_chars[pos]
            out_ch = out_cjk[pos] if pos < len(out_cjk) else in_ch
            in_comps = self._char_comps.get(in_ch, frozenset())
            out_comps = self._char_comps.get(out_ch, frozenset())
            shared = in_comps & out_comps

            # Resonance with previous
            if pos > 0:
                prev_ch = out_cjk[pos - 1]
                prev_comps = self._char_comps.get(prev_ch, frozenset())
                resonance = sorted(out_comps & prev_comps)
            else:
                resonance = []

            # Active motifs at this position
            active = [
                s.radical for s in schedules
                if s.radical in out_comps and s.weight_at(pos) > 0.3
            ]

            char_info.append({
                "pos": pos,
                "input": in_ch,
                "output": out_ch,
                "shared_with_input": sorted(shared),
                "resonance_with_prev": resonance,
                "active_motifs": active,
            })

        return {
            "input": text,
            "output": output,
            "motifs": [
                {"radical": s.radical, "period": s.period, "phase": round(s.phase, 2)}
                for s in schedules
            ],
            "chars": char_info,
        }
