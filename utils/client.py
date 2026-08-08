"""Backwards-compatibility shim.

The single LLM client implementation now lives in :mod:`utils.llm_client`
so that a provider swap (Anthropic <-> Groq) is a single-file change. This
module re-exports the public names so any older import path continues to
work; new code should import from :mod:`utils.llm_client` directly.
"""

from __future__ import annotations

from utils.llm_client import (
    LLMCallResult,
    LLMClient,
    LLMClientError,
    call_llm,
)

__all__ = ["LLMCallResult", "LLMClient", "LLMClientError", "call_llm"]
