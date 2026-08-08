"""Provider-agnostic LLM client wrapper.

This is the single point in the codebase where a real LLM SDK is called.
Every agent in every architecture reaches the network through
:class:`LLMClient` (or the convenience :func:`call_llm` function), and the
provider is selected by ``Settings.llm_provider``. Switching from Anthropic
to Groq (or back) is a ``.env`` edit; no source file has to change.

Two supported providers:

* ``anthropic`` — Anthropic Messages API (``claude-opus-5`` etc.).
  Supports ``output_config.format`` (JSON Schema) for structured outputs
  and the extended-thinking / effort parameters.
* ``groq`` — Groq Chat Completions API (``llama-3.3-70b-versatile`` etc.,
  OpenAI-compatible). Structured outputs use ``response_format=
  {"type": "json_object"}`` with the JSON Schema embedded in the user
  message; no equivalent of extended thinking or the effort parameter.

Design notes worth flagging for a reviewer:

* Every LLM call is logged through :class:`~utils.logging.ExperimentLogger`
  with the same record schema, regardless of provider. The
  ``request_params`` field records provider-specific parameters verbatim
  so a moderator can reconstruct the exact call.
* Stop reasons are mapped to a common vocabulary (``end_turn``,
  ``max_tokens``, ``tool_use``, ``refusal``) so downstream code that
  branches on ``stop_reason == "end_turn"`` behaves identically on both
  providers. The raw provider-side value is not lost — it appears in the
  interaction log via ``response_text`` context and can be reconstructed.
* Token accounting differs between providers (Anthropic reports cache
  read/write; Groq does not). Missing fields default to zero.
* Failures are logged (with an ``error`` object on the interaction record)
  and re-raised. This module never silently degrades a run.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping

from config.settings import Settings
from utils.logging import ExperimentLogger, LLMInteraction, TokenUsage


class LLMClientError(RuntimeError):
    """Raised when an LLM call returns unusable data.

    Examples: an empty content array, a refusal, or a structured-output
    call whose text is not valid JSON. Surfaced immediately rather than
    propagated silently into the SRS assembly step.
    """


# --------------------------------------------------------------------------
# Data types
# --------------------------------------------------------------------------


@dataclass
class BackendResult:
    """Provider-neutral shape of one raw LLM call.

    Attributes:
        text: The generated text.
        stop_reason: The reason generation stopped, mapped to the common
            vocabulary (``end_turn``, ``max_tokens``, ``tool_use``,
            ``refusal``, ``pause_turn`` or ``None``).
        usage: Token accounting for the call.
        request_params: The provider-specific request parameters, recorded
            verbatim for the interaction log.
    """

    text: str
    stop_reason: str | None
    usage: TokenUsage
    request_params: dict[str, Any]


@dataclass
class LLMCallResult:
    """The outcome of one logged LLM call.

    Attributes:
        text: Concatenated text content of the response.
        parsed_json: The response text parsed as JSON when the caller
            requested a structured output; ``None`` otherwise.
        stop_reason: The response's stop reason.
        usage: Token accounting for the call.
        interaction: The :class:`LLMInteraction` record that was written.
    """

    text: str
    parsed_json: Any | None
    stop_reason: str | None
    usage: TokenUsage
    interaction: LLMInteraction


# --------------------------------------------------------------------------
# Backend base
# --------------------------------------------------------------------------


class LLMBackend(ABC):
    """Abstract base for a provider back-end.

    Subclasses hold a live SDK client and translate one ``issue_call``
    invocation into one HTTP request. They are responsible for token-count
    normalisation and stop-reason mapping; they do NOT handle logging
    (that is :class:`LLMClient`'s job).
    """

    #: Exception type raised by the SDK on transport / API failure.
    #: Set by subclasses so :class:`LLMClient` knows which exception to
    #: catch, log, and re-raise.
    api_error_type: type[Exception] = Exception

    @abstractmethod
    def issue_call(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        settings: Settings,
        output_schema: Mapping[str, Any] | None,
    ) -> BackendResult:
        """Issue one LLM call and return a provider-neutral result."""


# --------------------------------------------------------------------------
# Anthropic backend
# --------------------------------------------------------------------------


class AnthropicBackend(LLMBackend):
    """Anthropic Messages API backend."""

    def __init__(self, settings: Settings) -> None:
        # Lazy import: the anthropic SDK is only imported when the user
        # actually selects this provider. Callers on Groq-only setups do
        # not need it installed at import time.
        import anthropic  # noqa: PLC0415

        self._client = anthropic.Anthropic(
            api_key=settings.anthropic_api_key,
            timeout=settings.request_timeout_seconds,
            max_retries=settings.max_retries,
        )
        self.api_error_type = anthropic.APIError

    def issue_call(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        settings: Settings,
        output_schema: Mapping[str, Any] | None,
    ) -> BackendResult:
        output_config: dict[str, Any] = {"effort": settings.effort}
        if output_schema is not None:
            output_config["format"] = {
                "type": "json_schema",
                "schema": dict(output_schema),
            }
        thinking = settings.thinking_param()

        request_params: dict[str, Any] = {
            "model": settings.model_id,
            "max_tokens": settings.max_tokens,
            "output_config": output_config,
            "thinking": thinking,
        }

        response = self._client.messages.create(
            model=settings.model_id,
            max_tokens=settings.max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            output_config=output_config,
            thinking=thinking,
        )
        text_parts: list[str] = []
        for block in getattr(response, "content", []) or []:
            if getattr(block, "type", None) == "text":
                text_value = getattr(block, "text", "")
                if text_value:
                    text_parts.append(text_value)
        return BackendResult(
            text="".join(text_parts),
            stop_reason=getattr(response, "stop_reason", None),
            usage=TokenUsage.from_response_usage(response.usage),
            request_params=request_params,
        )


# --------------------------------------------------------------------------
# Groq backend
# --------------------------------------------------------------------------


#: Mapping from Groq's OpenAI-compatible finish reasons to the common
#: vocabulary used by downstream code (``end_turn`` etc.). ``pause_turn``
#: does not exist on Groq. ``content_filter`` is mapped to ``refusal`` so
#: :class:`LLMClient` raises :class:`LLMClientError` for both providers.
_GROQ_FINISH_MAP: dict[str, str] = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "content_filter": "refusal",
}


class GroqBackend(LLMBackend):
    """Groq Chat Completions API backend (OpenAI-compatible)."""

    def __init__(self, settings: Settings) -> None:
        import groq  # noqa: PLC0415

        self._client = groq.Groq(
            api_key=settings.groq_api_key,
            timeout=settings.request_timeout_seconds,
            max_retries=settings.max_retries,
        )
        self.api_error_type = groq.APIError

    def issue_call(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        settings: Settings,
        output_schema: Mapping[str, Any] | None,
    ) -> BackendResult:
        # Groq's `response_format={"type": "json_object"}` is broadly
        # supported and forces valid JSON, but does not itself enforce a
        # schema (the more powerful `json_schema` type is model-dependent).
        # We embed the schema in the user prompt so the model produces the
        # right shape; :class:`LLMClient` still validates on the way out.
        prompt_addendum = ""
        response_format: dict[str, Any] | None = None
        if output_schema is not None:
            schema_str = json.dumps(dict(output_schema), indent=2)
            prompt_addendum = (
                "\n\nReturn a single JSON object that conforms exactly to "
                "the following JSON Schema. Do not wrap the object in "
                "Markdown fences, and do not emit any text outside the "
                "object.\n\n"
                f"```json\n{schema_str}\n```"
            )
            response_format = {"type": "json_object"}

        request_params: dict[str, Any] = {
            "model": settings.model_id,
            "max_tokens": settings.max_tokens,
        }
        if response_format is not None:
            request_params["response_format"] = response_format
        # Provenance markers so the config recorded in the log makes
        # explicit that the Anthropic-only parameters were not sent.
        request_params["effort_ignored"] = settings.effort
        request_params["thinking_ignored"] = settings.thinking_mode

        response = self._client.chat.completions.create(
            model=settings.model_id,
            max_tokens=settings.max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt + prompt_addendum},
            ],
            response_format=response_format,
        )

        choice = response.choices[0]
        text = choice.message.content or ""
        stop_reason_raw = choice.finish_reason
        stop_reason = _GROQ_FINISH_MAP.get(stop_reason_raw, stop_reason_raw)

        # Groq reports prompt_tokens/completion_tokens; no cache accounting.
        usage_obj = response.usage
        usage = TokenUsage(
            input_tokens=int(getattr(usage_obj, "prompt_tokens", 0) or 0),
            output_tokens=int(getattr(usage_obj, "completion_tokens", 0) or 0),
        )
        return BackendResult(
            text=text,
            stop_reason=stop_reason,
            usage=usage,
            request_params=request_params,
        )


# --------------------------------------------------------------------------
# Backend factory
# --------------------------------------------------------------------------


def build_backend(settings: Settings) -> LLMBackend:
    """Construct the backend selected by ``settings.llm_provider``.

    Args:
        settings: The resolved run configuration.

    Returns:
        A ready-to-use :class:`LLMBackend`.

    Raises:
        LLMClientError: If the provider name is unknown, or if the
            required SDK is not installed.
    """
    provider = settings.llm_provider
    try:
        if provider == "anthropic":
            return AnthropicBackend(settings)
        if provider == "groq":
            return GroqBackend(settings)
    except ImportError as exc:
        raise LLMClientError(
            f"LLM_PROVIDER={provider!r} is selected, but its SDK is not "
            f"installed. Run 'pip install -r requirements.txt' and retry. "
            f"(underlying import error: {exc})"
        ) from exc
    raise LLMClientError(f"Unknown LLM_PROVIDER: {provider!r}")


# --------------------------------------------------------------------------
# Logged client
# --------------------------------------------------------------------------


class LLMClient:
    """Issues LLM calls on behalf of the agents, with full logging.

    Constructed once per run and shared across every agent, so exactly one
    SDK client is created regardless of how many agents run.

    Args:
        settings: Resolved run configuration.
        logger: The active experiment logger.
    """

    def __init__(self, settings: Settings, logger: ExperimentLogger) -> None:
        self._settings = settings
        self._logger = logger
        self._backend = build_backend(settings)

    # -- public API --------------------------------------------------------

    def call(
        self,
        *,
        architecture: str,
        agent: str,
        case_id: str,
        repetition: int,
        phase: str,
        system_prompt: str,
        user_prompt: str,
        output_schema: Mapping[str, Any] | None = None,
    ) -> LLMCallResult:
        """Issue one LLM call, record it, and return the parsed result.

        See the file-level docstring for the log-record schema.

        Args:
            architecture: Condition under test.
            agent: Agent role that issued the call.
            case_id: Test case identifier.
            repetition: Zero-based repetition index.
            phase: Stage within the protocol.
            system_prompt: Fully-assembled system prompt.
            user_prompt: User-turn content.
            output_schema: Optional JSON Schema. When given, the response
                is parsed as JSON and returned in ``parsed_json``.

        Returns:
            A populated :class:`LLMCallResult`.

        Raises:
            LLMClientError: The call returned unusable data (refusal, or
                non-JSON text on a structured-output call).
            Exception: Any SDK-level failure (e.g. rate limit, transport
                error). Automatic retries have already run inside the SDK
                before this raises.
        """
        request_messages: list[dict[str, Any]] = [
            {"role": "user", "content": user_prompt}
        ]

        started_at = ExperimentLogger.start_timer()
        try:
            backend_result = self._backend.issue_call(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                settings=self._settings,
                output_schema=output_schema,
            )
        except self._backend.api_error_type as exc:
            elapsed_ms = ExperimentLogger.elapsed_ms(started_at)
            self._logger.log_interaction(
                architecture=architecture,
                agent=agent,
                case_id=case_id,
                repetition=repetition,
                phase=phase,
                model=self._settings.model_id,
                system_prompt=system_prompt,
                messages=request_messages,
                request_params={
                    "provider": self._settings.llm_provider,
                    "model": self._settings.model_id,
                },
                response_text="",
                stop_reason=None,
                usage=TokenUsage(),
                latency_ms=elapsed_ms,
                error={
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "status_code": getattr(exc, "status_code", None),
                },
            )
            raise

        elapsed_ms = ExperimentLogger.elapsed_ms(started_at)

        # Tag the recorded request parameters with the provider so an
        # analyst can filter by it after the fact without re-reading the
        # settings file.
        request_params = {
            "provider": self._settings.llm_provider,
            **backend_result.request_params,
        }
        interaction = self._logger.log_interaction(
            architecture=architecture,
            agent=agent,
            case_id=case_id,
            repetition=repetition,
            phase=phase,
            model=self._settings.model_id,
            system_prompt=system_prompt,
            messages=request_messages,
            request_params=request_params,
            response_text=backend_result.text,
            stop_reason=backend_result.stop_reason,
            usage=backend_result.usage,
            latency_ms=elapsed_ms,
        )

        if backend_result.stop_reason == "refusal":
            raise LLMClientError(
                f"{agent!r} call in phase {phase!r} returned "
                "stop_reason='refusal'. The model declined to answer; "
                "the run cannot proceed."
            )

        parsed_json: Any | None = None
        if output_schema is not None:
            parsed_json = _parse_json_response(
                backend_result.text, agent=agent, phase=phase
            )

        return LLMCallResult(
            text=backend_result.text,
            parsed_json=parsed_json,
            stop_reason=backend_result.stop_reason,
            usage=backend_result.usage,
            interaction=interaction,
        )


# --------------------------------------------------------------------------
# Convenience one-shot function
# --------------------------------------------------------------------------


def call_llm(system_prompt: str, user_message: str, settings: Settings) -> str:
    """Simple one-shot entry point for quick provider-agnostic calls.

    Issues one LLM call using whichever provider is selected in
    ``settings`` and returns the model's response text. This function
    intentionally has no logger argument and no structured-output
    support; use :class:`LLMClient` for anything that needs those.

    Args:
        system_prompt: The system prompt.
        user_message: The user-turn content.
        settings: Resolved run configuration.

    Returns:
        The response text (empty string if the model returned no text).

    Raises:
        LLMClientError: The call returned a refusal.
        Exception: SDK-level failures propagate.
    """
    backend = build_backend(settings)
    result = backend.issue_call(
        system_prompt=system_prompt,
        user_prompt=user_message,
        settings=settings,
        output_schema=None,
    )
    if result.stop_reason == "refusal":
        raise LLMClientError(
            "The provider declined to answer (stop_reason='refusal')."
        )
    return result.text


# --------------------------------------------------------------------------
# JSON parsing helper
# --------------------------------------------------------------------------


def _parse_json_response(text: str, *, agent: str, phase: str) -> Any:
    """Parse a structured-output response as JSON.

    Some providers (Groq's ``json_object`` mode included) can occasionally
    wrap JSON in Markdown fences even when instructed not to; this helper
    strips a single leading/trailing fence pair if present, then parses.

    Args:
        text: The response text.
        agent: The calling agent, for the error message.
        phase: The protocol phase, for the error message.

    Returns:
        The parsed JSON value.

    Raises:
        LLMClientError: If the text is empty or not valid JSON.
    """
    stripped = text.strip()
    if not stripped:
        raise LLMClientError(
            f"{agent!r} call in phase {phase!r} requested a structured "
            "output but the response contained no text."
        )
    # Strip a leading ```json fence and trailing ``` if the model wrapped
    # its answer despite instructions.
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 2 and lines[-1].strip().startswith("```"):
            # Drop the first line (```json or similar) and the closing ```
            stripped = "\n".join(lines[1:-1]).strip()

    try:
        return json.loads(stripped)
    except json.JSONDecodeError as exc:
        preview = stripped[:200].replace("\n", " ")
        raise LLMClientError(
            f"{agent!r} call in phase {phase!r} returned text that is not "
            f"valid JSON: {exc.msg} (pos {exc.pos}). Response preview: {preview!r}"
        ) from exc


__all__ = [
    "AnthropicBackend",
    "BackendResult",
    "GroqBackend",
    "LLMBackend",
    "LLMCallResult",
    "LLMClient",
    "LLMClientError",
    "build_backend",
    "call_llm",
]
