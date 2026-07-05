"""veda_core/slm

Migration plan §8b — the SLM backend seam. Query-time SLM inference must not
be hardwired to Ollama, because a single Ollama instance serializes every SLM
call across the whole inference fleet. `call_slm` is the one choke point every
call site (IR emit, decomposer, RAG synthesis, NL answer) routes through.
"""

from ._call_slm import call_slm

__all__ = ["call_slm"]
