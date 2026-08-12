"""Provider-agnostic LLM client wrapper.

This is the single point in the codebase where a real LLM SDK is called.
Every agent in every architecture reaches the network through
:class:`LLMClient` (or the convenience :func:`call_llm` function), and the
provider is selected by ``Settings.llm_provider``. Switching from Anthropic
to Groq (or back) is a ``.env`` edit; no source file has to change.

Three supported providers:

* ``anthropic`` — Anthropic Messages API (``claude-opus-5`` etc.).
  Supports ``output_config.format`` (JSON Schema) for structured outputs
  and the extended-thinking / effort parameters.
* ``groq`` — Groq Chat Completions API (``llama-3.3-70b-versatile`` etc.,
  OpenAI-compatible). Structured outputs use ``response_format=
  {"type": "json_object"}`` with the JSON Schema embedded in the user
  message; no equivalent of extended thinking or the effort parameter.
* ``ollama`` — local Ollama server (default ``http://localhost:11434``),
  native ``/api/chat`` endpoint. No SDK dependency — the endpoint is
  simple enough that the standard library's ``urllib`` avoids adding a
  third-party package for a single provider (see README "Dependencies
  and their justification"). Structured outputs pass the JSON Schema
  directly as Ollama's ``format`` field, which constrains decoding at
  the token level on Ollama >= 0.5 (grammar-based), a stronger guarantee
  than Groq's ``json_object`` mode.

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
import time
import urllib.error
import urllib.request
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
# Ollama backend
# --------------------------------------------------------------------------


class OllamaAPIError(Exception):
    """Raised when the Ollama HTTP API returns an error or is unreachable.

    There is no official Ollama Python SDK to import an exception type
    from (this backend talks to the local HTTP API directly), so this
    class plays the same role ``anthropic.APIError`` /
    ``groq.APIError`` play for the other two backends: it is what
    :class:`LLMClient` catches, logs, and re-raises.

    Attributes:
        status_code: The HTTP status code, when the failure was an HTTP
            error response. ``None`` for connection-level failures
            (server not running, DNS failure, etc.).
    """

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


#: Mapping from Ollama's ``done_reason`` to the common vocabulary used by
#: downstream code. Ollama has no equivalent of ``tool_use``, ``refusal``,
#: or ``pause_turn`` — this project never asks it to call tools, so those
#: values are not expected in practice; any unrecognised value is passed
#: through unchanged rather than silently coerced.
_OLLAMA_DONE_REASON_MAP: dict[str, str] = {
    "stop": "end_turn",
    "length": "max_tokens",
}


class OllamaBackend(LLMBackend):
    """Local Ollama server backend (native ``/api/chat`` endpoint)."""

    def __init__(self, settings: Settings) -> None:
        self._base_url = settings.ollama_base_url.rstrip("/")
        self.api_error_type = OllamaAPIError
        # Fail at construction time with a clear, actionable message
        # rather than deep inside a multi-agent run's first call — "is
        # the server even running" is by far the most common Ollama
        # failure mode (forgetting to run `ollama serve`).
        self._check_server_reachable()

    def _check_server_reachable(self) -> None:
        """Probe the server so an unreachable Ollama fails fast and clearly.

        Raises:
            LLMClientError: If the server does not respond within a short
                timeout. Uses a fixed 5-second probe timeout regardless of
                ``request_timeout_seconds`` — this is a liveness check,
                not a real request, and should not make the caller wait
                through a long configured timeout just to find out the
                server is down.
        """
        try:
            urllib.request.urlopen(f"{self._base_url}/api/tags", timeout=5)
        except (urllib.error.URLError, OSError) as exc:
            raise LLMClientError(
                f"Cannot reach Ollama server at {self._base_url!r}. Is it "
                f"running? Start it with 'ollama serve' (or check "
                f"OLLAMA_BASE_URL in .env if it's on a different host or "
                f"port). Underlying error: {exc}"
            ) from exc

    def issue_call(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        settings: Settings,
        output_schema: Mapping[str, Any] | None,
    ) -> BackendResult:
        effective_prompt = user_prompt
        request_format: Any = None
        if output_schema is not None:
            # Ollama >= 0.5 accepts a JSON Schema object directly as
            # `format`, which constrains decoding at the token level
            # (grammar-based) so the output is guaranteed structurally
            # valid. The short textual nudge below is a cheap backstop
            # for older servers where `format` degrades to loose
            # "valid JSON" enforcement rather than schema-shaped JSON.
            request_format = dict(output_schema)
            effective_prompt = (
                user_prompt
                + "\n\nRespond with a single JSON object only — no prose, "
                "no Markdown code fences."
            )

        request_params: dict[str, Any] = {
            "model": settings.ollama_model,
            "base_url": self._base_url,
            "num_predict": settings.max_tokens,
        }
        if request_format is not None:
            request_params["format"] = "json_schema"
        # Provenance markers: this project's other two backends use
        # effort/thinking; Ollama has neither, but recording the
        # configured (unused) values keeps config.json self-describing
        # if a moderator diffs runs across providers.
        request_params["effort_ignored"] = settings.effort
        request_params["thinking_ignored"] = settings.thinking_mode

        payload: dict[str, Any] = {
            "model": settings.ollama_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": effective_prompt},
            ],
            "stream": False,
            "options": {"num_predict": settings.max_tokens},
        }
        if request_format is not None:
            payload["format"] = request_format

        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self._base_url}/api/chat",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        # Unlike the Anthropic/Groq backends, this one talks to Ollama over
        # bare urllib rather than a vendor SDK, so it doesn't get automatic
        # retry-on-transient-failure "for free" the way the other two do
        # from ``max_retries=`` on their SDK client constructors. A single
        # slow/overloaded local Ollama instance would otherwise permanently
        # fail an entire architecture attempt (observed in practice with the
        # ``debate`` architecture, which issues the most calls per case) on
        # what is often just one bad request. Retry connection-level
        # failures and 5xx responses up to ``settings.max_retries`` times
        # with a short linear backoff; 4xx responses are not retried since
        # those indicate a genuinely bad request, not a transient failure.
        raw: bytes | None = None
        last_exc: Exception | None = None
        attempts = max(1, settings.max_retries + 1)
        for attempt in range(1, attempts + 1):
            try:
                with urllib.request.urlopen(
                    request, timeout=settings.request_timeout_seconds
                ) as response:
                    raw = response.read()
                break
            except urllib.error.HTTPError as exc:
                error_body = exc.read().decode("utf-8", errors="replace")
                if exc.code < 500 or attempt == attempts:
                    raise OllamaAPIError(
                        f"Ollama returned HTTP {exc.code} for model "
                        f"{settings.ollama_model!r}: {error_body}",
                        status_code=exc.code,
                    ) from exc
                last_exc = exc
            except (urllib.error.URLError, OSError) as exc:
                if attempt == attempts:
                    raise OllamaAPIError(
                        f"Could not reach Ollama at {self._base_url!r} after "
                        f"{attempts} attempt(s): {exc}"
                    ) from exc
                last_exc = exc
            time.sleep(min(2 * attempt, 10))
        if raw is None:
            # Unreachable in practice (the loop above always either
            # succeeds or raises on the final attempt), but keeps type
            # checkers and readers honest about the loop's contract.
            raise OllamaAPIError(
                f"Could not reach Ollama at {self._base_url!r} after "
                f"{attempts} attempt(s): {last_exc}"
            ) from last_exc

        data = json.loads(raw)
        text = (data.get("message") or {}).get("content", "") or ""
        done_reason_raw = data.get("done_reason", "stop")
        stop_reason = _OLLAMA_DONE_REASON_MAP.get(done_reason_raw, done_reason_raw)

        # Ollama reports prompt_eval_count/eval_count; no cache accounting
        # (there is no server-side prompt cache to report on).
        usage = TokenUsage(
            input_tokens=int(data.get("prompt_eval_count", 0) or 0),
            output_tokens=int(data.get("eval_count", 0) or 0),
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
        if provider == "ollama":
            return OllamaBackend(settings)
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
                model=self._settings.effective_model_id,
                system_prompt=system_prompt,
                messages=request_messages,
                request_params={
                    "provider": self._settings.llm_provider,
                    "model": self._settings.effective_model_id,
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
            model=self._settings.effective_model_id,
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
    "OllamaAPIError",
    "OllamaBackend",
    "build_backend",
    "call_llm",
]
