"""
shanzi — semantic ultrasound of obscure Chinese characters.

Shanzi embeddings blend a character's own meaning with attenuated copies
of its subcomponents' meanings, recursively, so every character expresses
its full "smear" over semantic space.  A sentence becomes a path through
that space; curve-fit it, jitter it, resample it, and decode back to
characters to produce text that is semantically nearby but structurally
alien — the written equivalent of a tryptamine-pen hallucination.
"""

from shanzi.engine import ShanziEngine

__version__ = "0.1.0"
__all__ = ["ShanziEngine"]
