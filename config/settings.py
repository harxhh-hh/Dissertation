"""Central configuration for the multi-agent SRS generation experiment.

Every tunable parameter of the experiment lives here so that a single object
(:class:`Settings`) fully describes the configuration of a run. That object is
serialised into each run directory as ``config.json``, which is what makes a
run reproducible and auditable by a third party.

Configuration is read from environment variables, which are in turn loaded
from a local ``.env`` file (see ``.env.example``). No credential is ever
hard-coded or written to a tracked file.

Typical use::

    from config.settings import Settings

    settings = Settings.from_env()
    settings.apply_seed()
"""

from __future__ import annotations

import os
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Literal

from dotenv import load_dotenv

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

#: Repository root (the directory that contains this ``config`` package).
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent

#: Effort levels accepted by the Anthropic Messages API ``output_config``.
EffortLevel = Literal["low", "medium", "high", "xhigh", "max"]

#: Extended-thinking modes used by this project.
ThinkingMode = Literal["adaptive", "disabled"]

#: LLM providers supported by this project.
LLMProvider = Literal[
    "groq", "anthropic", "ollama", "openai", "openrouter", "gemini"
]

_VALID_EFFORT: Final[frozenset[str]] = frozenset(
    {"low", "medium", "high", "xhigh", "max"}
)
_VALID_THINKING: Final[frozenset[str]] = frozenset({"adaptive", "disabled"})
_VALID_LOG_LEVEL: Final[frozenset[str]] = frozenset(
    {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
)
_VALID_PROVIDER: Final[frozenset[str]] = frozenset(
    {"groq", "anthropic", "ollama", "openai", "openrouter", "gemini"}
)

#: Default Ollama server URL. Ollama serves an OpenAI-compatible endpoint
#: from a local process (``ollama serve``); the default port is 11434.
_DEFAULT_OLLAMA_BASE_URL: Final[str] = "http://localhost:11434"

#: Default Ollama model tag when ``OLLAMA_MODEL`` is unset. A small,
#: broadly-available default that most Ollama installs can pull quickly.
_DEFAULT_OLLAMA_MODEL: Final[str] = "mistral"

#: OpenRouter's API endpoint is fixed (unlike Ollama's, there is no
#: legitimate reason to point it elsewhere), so it is a plain constant
#: rather than a configurable Settings field.
OPENROUTER_BASE_URL: Final[str] = "https://openrouter.ai/api/v1"

#: Provider-appropriate default model IDs, used when MODEL_ID is unset.
#: Deliberately the cheap/fast tier of each provider — a run that forgets
#: to set MODEL_ID should not silently rack up cost on the priciest model.
_DEFAULT_OPENAI_MODEL: Final[str] = "gpt-4o-mini"
_DEFAULT_OPENROUTER_MODEL: Final[str] = "openai/gpt-4o-mini"
_DEFAULT_GEMINI_MODEL: Final[str] = "gemini-1.5-flash"

#: Effort levels at which the API rejects ``thinking={"type": "disabled"}``.
#: Documented behaviour of ``claude-opus-5``; validated here so the run fails
#: at start-up with a clear message rather than mid-experiment with an HTTP 400.
_EFFORT_FORBIDDING_DISABLED_THINKING: Final[frozenset[str]] = frozenset(
    {"xhigh", "max"}
)


class ConfigurationError(RuntimeError):
    """Raised when the environment does not describe a usable configuration.

    Raised eagerly at start-up so that a misconfigured run never silently
    produces partial or incomparable results.
    """


# --------------------------------------------------------------------------
# Environment parsing helpers
# --------------------------------------------------------------------------


def _require_str(name: str) -> str:
    """Return a required environment variable, or raise.

    Args:
        name: Environment variable name.

    Returns:
        The variable's value, stripped of surrounding whitespace.

    Raises:
        ConfigurationError: If the variable is missing or empty.
    """
    raw = os.environ.get(name, "").strip()
    if not raw:
        raise ConfigurationError(
            f"Required environment variable {name!r} is missing or empty. "
            f"Copy .env.example to .env and set it."
        )
    return raw


def _get_str(name: str, default: str) -> str:
    """Return an optional string environment variable, falling back to ``default``."""
    raw = os.environ.get(name, "").strip()
    return raw if raw else default


def _get_int(name: str, default: int, *, minimum: int | None = None) -> int:
    """Return an optional integer environment variable.

    Args:
        name: Environment variable name.
        default: Value used when the variable is unset or empty.
        minimum: If given, the parsed value must be greater than or equal to it.

    Returns:
        The parsed integer.

    Raises:
        ConfigurationError: If the value is not an integer, or is below ``minimum``.
    """
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigurationError(
            f"Environment variable {name!r} must be an integer, got {raw!r}."
        ) from exc
    if minimum is not None and value < minimum:
        raise ConfigurationError(
            f"Environment variable {name!r} must be >= {minimum}, got {value}."
        )
    return value


def _get_choice(name: str, default: str, allowed: frozenset[str]) -> str:
    """Return an optional environment variable constrained to a fixed set.

    Raises:
        ConfigurationError: If the value is not in ``allowed``.
    """
    value = _get_str(name, default)
    if value not in allowed:
        raise ConfigurationError(
            f"Environment variable {name!r} must be one of "
            f"{sorted(allowed)}, got {value!r}."
        )
    return value


#: Substrings that, if found (case-insensitively) in an API key value,
#: mark it as almost certainly a placeholder rather than a real credential.
#: Real API keys are opaque random tokens; none of the providers this
#: project supports issue keys containing English words like these.
_PLACEHOLDER_MARKERS: Final[tuple[str, ...]] = (
    "replace",
    "your_actual",
    "your-actual",
    "youractual",
    "your_key",
    "your-key",
    "placeholder",
    "changeme",
    "change_me",
    "change-me",
    "xxxxxxxx",
    "insert_key",
    "paste_your",
    "paste-your",
    "todo",
)


def _looks_like_placeholder(key: str) -> bool:
    """Heuristically detect an unfilled example value in an API key field.

    Catches both the exact strings shipped in ``.env.example`` (e.g.
    ``sk-ant-REPLACE-ME``) and values a user produced by copying an
    *example command* verbatim instead of substituting their real key
    (e.g. ``gsk_your_actual_key``, taken literally from an instruction
    that meant it as a placeholder to replace).

    Args:
        key: The candidate credential value (already stripped).

    Returns:
        ``True`` if the value looks like a placeholder rather than a real
        credential.
    """
    lowered = key.lower()
    return any(marker in lowered for marker in _PLACEHOLDER_MARKERS)


def _utc_timestamp_slug() -> str:
    """Return a filesystem-safe UTC timestamp, e.g. ``2026-07-26T14-05-33Z``."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Settings:
    """Immutable, fully-resolved configuration for one experiment run.

    Instances are created with :meth:`from_env` and are frozen thereafter, so
    the configuration cannot drift part-way through a run. Call
    :meth:`to_dict` to obtain a redacted, JSON-serialisable snapshot suitable
    for writing into the run directory.

    Attributes:
        llm_provider: Which back-end to use (``groq``, ``anthropic``,
            ``ollama``, ``openai``, ``openrouter``, or ``gemini``).
        anthropic_api_key: Credential for the Anthropic Messages API. Empty
            when ``llm_provider != "anthropic"``. Never logged or
            serialised; :meth:`to_dict` redacts it.
        groq_api_key: Credential for the Groq API. Empty when
            ``llm_provider != "groq"``. Never logged or serialised;
            :meth:`to_dict` redacts it.
        openai_api_key: Credential for the OpenAI API. Empty when
            ``llm_provider != "openai"``. Never logged or serialised;
            :meth:`to_dict` redacts it.
        openrouter_api_key: Credential for OpenRouter (a single key that
            proxies to many providers/models). Empty when
            ``llm_provider != "openrouter"``. Never logged or serialised;
            :meth:`to_dict` redacts it.
        gemini_api_key: Credential for the Google Gemini API. Empty when
            ``llm_provider != "gemini"``. Never logged or serialised;
            :meth:`to_dict` redacts it.
        ollama_base_url: URL of the Ollama server (typically local). Used
            only when ``llm_provider == "ollama"``. Not sensitive; NOT
            redacted in :meth:`__repr__` or :meth:`to_dict`.
        ollama_model: Model tag to pull from Ollama (e.g. ``mistral``,
            ``llama3.1``, ``qwen2.5-coder``). Used only when
            ``llm_provider == "ollama"``. Note the intentional overlap
            with ``model_id``: keeping the fields separate matches how
            Ollama users typically think about local model tags, but
            when ``provider == "ollama"`` this field wins over
            ``model_id`` for the effective model choice.
        model_id: Model identifier used by every agent in every condition,
            for every provider except Ollama (see ``ollama_model``).
        max_tokens: Upper bound on output tokens per LLM call.
        effort: Reasoning-effort level passed as ``output_config.effort``
            on Anthropic. Ignored by every other provider (recorded in
            ``config.json`` for provenance).
        thinking_mode: Extended-thinking mode (``adaptive`` or
            ``disabled``). Ignored by every provider except Anthropic
            (recorded in ``config.json`` for provenance).
        random_seed: Seed for all Python-side randomness.
        max_revision_rounds: Cap on Verification-Agent-triggered revisions.
        repetitions: Independent repetitions per (architecture, test case) pair.
        output_dir: Root directory for run artefacts.
        request_timeout_seconds: Per-request timeout passed to the SDK client.
        max_retries: SDK-level retries for transient (429/5xx) failures.
        log_level: Console log verbosity.
        run_id: Unique identifier for this run; also the run directory name.
    """

    llm_provider: LLMProvider
    anthropic_api_key: str
    groq_api_key: str
    openai_api_key: str
    openrouter_api_key: str
    gemini_api_key: str
    ollama_base_url: str
    ollama_model: str
    model_id: str
    max_tokens: int
    effort: EffortLevel
    thinking_mode: ThinkingMode
    random_seed: int
    max_revision_rounds: int
    repetitions: int
    output_dir: Path
    request_timeout_seconds: int
    max_retries: int
    log_level: str
    run_id: str = field(default_factory=_utc_timestamp_slug)

    # -- construction -------------------------------------------------------

    @classmethod
    def from_env(
        cls,
        *,
        env_file: Path | None = None,
        run_id: str | None = None,
        run_id_suffix: str | None = None,
    ) -> "Settings":
        """Build a :class:`Settings` instance from the environment.

        Loads ``.env`` from the project root (unless ``env_file`` overrides
        it), then reads and validates every configuration variable. Variables
        already present in the process environment take precedence over the
        ``.env`` file, which lets CI or a shell export override a value
        without editing files.

        Args:
            env_file: Optional explicit path to a dotenv file.
            run_id: Optional explicit run identifier, used verbatim. Defaults
                to a UTC timestamp. Supply this to group several invocations
                under one logical run. Takes precedence over
                ``run_id_suffix`` if both are given.
            run_id_suffix: Optional short, filesystem-safe slug appended to
                the default UTC-timestamp run id (e.g. a description slug),
                so the run directory reads as ``run_<timestamp>_<suffix>``
                instead of a bare timestamp — the date/time stays first so
                directory names remain chronologically sortable by name.
                Ignored if ``run_id`` is given explicitly.

        Returns:
            A validated, frozen configuration object.

        Raises:
            ConfigurationError: If any variable is missing, malformed, or
                describes a combination the API would reject.
        """
        if run_id is None and run_id_suffix:
            run_id = f"{_utc_timestamp_slug()}_{run_id_suffix}"

        dotenv_path = env_file if env_file is not None else PROJECT_ROOT / ".env"
        # override=False: a variable already exported in the shell wins.
        load_dotenv(dotenv_path, override=False)

        provider = _get_choice("LLM_PROVIDER", "groq", _VALID_PROVIDER)
        effort = _get_choice("EFFORT", "high", _VALID_EFFORT)
        thinking_mode = _get_choice("THINKING_MODE", "adaptive", _VALID_THINKING)

        # The EFFORT/THINKING interaction check applies only to the
        # Anthropic path; on Groq and Ollama these values are ignored
        # anyway (and are recorded in config.json for provenance).
        if (
            provider == "anthropic"
            and thinking_mode == "disabled"
            and effort in _EFFORT_FORBIDDING_DISABLED_THINKING
        ):
            raise ConfigurationError(
                f"THINKING_MODE='disabled' is not accepted at EFFORT={effort!r} "
                f"(the Anthropic API rejects this combination). Use EFFORT=high "
                f"or lower, or set THINKING_MODE=adaptive."
            )

        # Require only the credential for the selected provider. Store the
        # others as empty rather than requiring all of them, so a user set
        # up for one provider does not need dummy keys for the rest.
        # Ollama is keyless — it talks to a local server — so no check.
        anthropic_key = _get_str("ANTHROPIC_API_KEY", "")
        groq_key = _get_str("GROQ_API_KEY", "")
        openai_key = _get_str("OPENAI_API_KEY", "")
        openrouter_key = _get_str("OPENROUTER_API_KEY", "")
        gemini_key = _get_str("GEMINI_API_KEY", "")
        if provider == "anthropic" and not anthropic_key:
            raise ConfigurationError(
                "LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is missing "
                "or empty. Set it in .env."
            )
        if provider == "groq" and not groq_key:
            raise ConfigurationError(
                "LLM_PROVIDER=groq but GROQ_API_KEY is missing or empty. "
                "Set it in .env."
            )
        if provider == "openai" and not openai_key:
            raise ConfigurationError(
                "LLM_PROVIDER=openai but OPENAI_API_KEY is missing or "
                "empty. Set it in .env."
            )
        if provider == "openrouter" and not openrouter_key:
            raise ConfigurationError(
                "LLM_PROVIDER=openrouter but OPENROUTER_API_KEY is missing "
                "or empty. Set it in .env."
            )
        if provider == "gemini" and not gemini_key:
            raise ConfigurationError(
                "LLM_PROVIDER=gemini but GEMINI_API_KEY is missing or "
                "empty. Set it in .env."
            )
        # Placeholder detection, but only for the ACTIVE provider — an
        # unused-provider key that still holds its placeholder should not
        # block a run using the other provider. Uses a heuristic rather
        # than an exact-string match against .env.example, because a user
        # who follows an example command literally (e.g. copies
        # 'GROQ_API_KEY=gsk_your_actual_key' from an instruction verbatim
        # instead of substituting their real key) produces a placeholder
        # that isn't byte-identical to anything in .env.example but is
        # obviously not a credential either.
        if provider == "anthropic" and _looks_like_placeholder(anthropic_key):
            raise ConfigurationError(
                f"ANTHROPIC_API_KEY={anthropic_key!r} looks like a "
                "placeholder, not a real credential. Get a real key from "
                "https://console.anthropic.com/settings/keys and set it "
                "in .env."
            )
        if provider == "groq" and _looks_like_placeholder(groq_key):
            raise ConfigurationError(
                f"GROQ_API_KEY={groq_key!r} looks like a placeholder, not "
                "a real credential. Get a real key from "
                "https://console.groq.com/keys and set it in .env."
            )
        if provider == "openai" and _looks_like_placeholder(openai_key):
            raise ConfigurationError(
                f"OPENAI_API_KEY={openai_key!r} looks like a placeholder, "
                "not a real credential. Get a real key from "
                "https://platform.openai.com/api-keys and set it in .env."
            )
        if provider == "openrouter" and _looks_like_placeholder(openrouter_key):
            raise ConfigurationError(
                f"OPENROUTER_API_KEY={openrouter_key!r} looks like a "
                "placeholder, not a real credential. Get a real key from "
                "https://openrouter.ai/keys and set it in .env."
            )
        if provider == "gemini" and _looks_like_placeholder(gemini_key):
            raise ConfigurationError(
                f"GEMINI_API_KEY={gemini_key!r} looks like a placeholder, "
                "not a real credential. Get a real key from "
                "https://aistudio.google.com/apikey and set it in .env."
            )

        # Ollama configuration. Both fields have sensible defaults so a
        # user who sets LLM_PROVIDER=ollama and nothing else still gets a
        # workable configuration (talks to localhost:11434 and asks for
        # the `mistral` tag).
        ollama_base_url = _get_str("OLLAMA_BASE_URL", _DEFAULT_OLLAMA_BASE_URL)
        ollama_model = _get_str("OLLAMA_MODEL", _DEFAULT_OLLAMA_MODEL)

        output_dir_raw = _get_str("OUTPUT_DIR", "outputs")
        output_dir = Path(output_dir_raw)
        if not output_dir.is_absolute():
            output_dir = PROJECT_ROOT / output_dir

        # A provider-appropriate default for MODEL_ID, so a run that
        # forgets to set it still produces something meaningful. Real
        # runs should always set MODEL_ID explicitly for provenance.
        # For Ollama, `ollama_model` is the effective model choice; the
        # `model_id` default mirrors it so logs are still consistent.
        if provider == "anthropic":
            default_model = "claude-opus-5"
        elif provider == "groq":
            default_model = "llama-3.3-70b-versatile"
        elif provider == "openai":
            default_model = _DEFAULT_OPENAI_MODEL
        elif provider == "openrouter":
            default_model = _DEFAULT_OPENROUTER_MODEL
        elif provider == "gemini":
            default_model = _DEFAULT_GEMINI_MODEL
        else:  # provider == "ollama"
            default_model = ollama_model

        settings = cls(
            llm_provider=provider,  # type: ignore[arg-type]  # validated
            anthropic_api_key=anthropic_key,
            groq_api_key=groq_key,
            openai_api_key=openai_key,
            openrouter_api_key=openrouter_key,
            gemini_api_key=gemini_key,
            ollama_base_url=ollama_base_url,
            ollama_model=ollama_model,
            model_id=_get_str("MODEL_ID", default_model),
            max_tokens=_get_int("MAX_TOKENS", 16000, minimum=1),
            effort=effort,  # type: ignore[arg-type]  # validated against _VALID_EFFORT
            thinking_mode=thinking_mode,  # type: ignore[arg-type]  # validated above
            random_seed=_get_int("RANDOM_SEED", 42),
            max_revision_rounds=_get_int("MAX_REVISION_ROUNDS", 1, minimum=0),
            repetitions=_get_int("REPETITIONS", 1, minimum=1),
            output_dir=output_dir,
            request_timeout_seconds=_get_int("REQUEST_TIMEOUT_SECONDS", 600, minimum=1),
            max_retries=_get_int("MAX_RETRIES", 2, minimum=0),
            log_level=_get_choice("LOG_LEVEL", "INFO", _VALID_LOG_LEVEL),
            **({"run_id": run_id} if run_id else {}),
        )

        return settings

    # -- derived values -----------------------------------------------------

    @property
    def run_dir(self) -> Path:
        """Directory holding every artefact produced by this run."""
        return self.output_dir / f"run_{self.run_id}"

    @property
    def effective_model_id(self) -> str:
        """The model name actually sent to the provider on every call.

        For ``anthropic`` and ``groq`` this is ``model_id``. For
        ``ollama`` it is ``ollama_model`` — the two fields intentionally
        overlap (see the ``ollama_model`` docstring), and this property
        is the single place that resolves the overlap, so log records,
        error messages, and the backend itself never have to repeat the
        ``if provider == "ollama"`` check independently and risk
        disagreeing with each other.
        """
        if self.llm_provider == "ollama":
            return self.ollama_model
        return self.model_id

    def thinking_param(self) -> dict[str, str]:
        """Return the ``thinking`` parameter for an Anthropic Messages request."""
        return {"type": self.thinking_mode}

    def output_config_param(self) -> dict[str, str]:
        """Return the ``output_config`` parameter for an Anthropic Messages request."""
        return {"effort": self.effort}

    # -- side effects -------------------------------------------------------

    def apply_seed(self) -> None:
        """Seed Python's global RNG.

        This makes every *Python-side* random choice deterministic: test-case
        ordering, the shuffle applied when exporting anonymised documents for
        blind rating, and anonymous ID assignment.

        It does NOT make LLM output deterministic. See README, section
        "Reproducibility", for the full statement of what is and is not
        controlled.
        """
        random.seed(self.random_seed)

    def ensure_run_dir(self) -> Path:
        """Create the run directory (including parents) and return its path."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        return self.run_dir

    # -- serialisation ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable snapshot with credentials redacted.

        This snapshot is written to ``config.json`` in the run directory and is
        the authoritative record of how a run was configured. Both provider
        keys are redacted whether or not they were used, so a config file
        never leaks a key even if the run was misconfigured. Ollama fields
        (``ollama_base_url`` / ``ollama_model``) are non-sensitive and are
        included at their configured values regardless of the active
        provider — recording both the "active" and the "if I swapped" values
        makes a config snapshot self-descriptive for a moderator.
        """
        data = asdict(self)
        data["anthropic_api_key"] = "<redacted>"
        data["groq_api_key"] = "<redacted>"
        data["openai_api_key"] = "<redacted>"
        data["openrouter_api_key"] = "<redacted>"
        data["gemini_api_key"] = "<redacted>"
        data["output_dir"] = str(self.output_dir)
        data["run_dir"] = str(self.run_dir)
        return data

    def __repr__(self) -> str:  # pragma: no cover - debugging convenience
        """Return a representation that cannot leak any API key.

        Ollama fields are shown verbatim (no credential to leak); every
        provider-specific key field is redacted regardless of which
        provider is active.
        """
        return (
            f"Settings(run_id={self.run_id!r}, llm_provider={self.llm_provider!r}, "
            f"model_id={self.model_id!r}, effort={self.effort!r}, "
            f"thinking_mode={self.thinking_mode!r}, random_seed={self.random_seed}, "
            f"repetitions={self.repetitions}, "
            f"max_revision_rounds={self.max_revision_rounds}, "
            f"ollama_base_url={self.ollama_base_url!r}, "
            f"ollama_model={self.ollama_model!r}, "
            f"anthropic_api_key='<redacted>', groq_api_key='<redacted>', "
            f"openai_api_key='<redacted>', openrouter_api_key='<redacted>', "
            f"gemini_api_key='<redacted>')"
        )
