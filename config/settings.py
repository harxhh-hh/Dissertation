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
LLMProvider = Literal["groq", "anthropic"]

_VALID_EFFORT: Final[frozenset[str]] = frozenset(
    {"low", "medium", "high", "xhigh", "max"}
)
_VALID_THINKING: Final[frozenset[str]] = frozenset({"adaptive", "disabled"})
_VALID_LOG_LEVEL: Final[frozenset[str]] = frozenset(
    {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
)
_VALID_PROVIDER: Final[frozenset[str]] = frozenset({"groq", "anthropic"})

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
        llm_provider: Which back-end to use (``groq`` or ``anthropic``).
        anthropic_api_key: Credential for the Anthropic Messages API. Empty
            when ``llm_provider != "anthropic"``. Never logged or
            serialised; :meth:`to_dict` redacts it.
        groq_api_key: Credential for the Groq API. Empty when
            ``llm_provider != "groq"``. Never logged or serialised;
            :meth:`to_dict` redacts it.
        model_id: Model identifier used by every agent in every condition.
        max_tokens: Upper bound on output tokens per LLM call.
        effort: Reasoning-effort level passed as ``output_config.effort``
            on Anthropic. Ignored when ``llm_provider == "groq"``
            (recorded in ``config.json`` for provenance).
        thinking_mode: Extended-thinking mode (``adaptive`` or
            ``disabled``). Ignored when ``llm_provider == "groq"``
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
    def from_env(cls, *, env_file: Path | None = None, run_id: str | None = None) -> "Settings":
        """Build a :class:`Settings` instance from the environment.

        Loads ``.env`` from the project root (unless ``env_file`` overrides
        it), then reads and validates every configuration variable. Variables
        already present in the process environment take precedence over the
        ``.env`` file, which lets CI or a shell export override a value
        without editing files.

        Args:
            env_file: Optional explicit path to a dotenv file.
            run_id: Optional explicit run identifier. Defaults to a UTC
                timestamp. Supply this to group several invocations under one
                logical run.

        Returns:
            A validated, frozen configuration object.

        Raises:
            ConfigurationError: If any variable is missing, malformed, or
                describes a combination the API would reject.
        """
        dotenv_path = env_file if env_file is not None else PROJECT_ROOT / ".env"
        # override=False: a variable already exported in the shell wins.
        load_dotenv(dotenv_path, override=False)

        provider = _get_choice("LLM_PROVIDER", "groq", _VALID_PROVIDER)
        effort = _get_choice("EFFORT", "high", _VALID_EFFORT)
        thinking_mode = _get_choice("THINKING_MODE", "adaptive", _VALID_THINKING)

        # The EFFORT/THINKING interaction check applies only to the
        # Anthropic path; on Groq these values are ignored anyway.
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
        # other as empty rather than requiring both, so a user set up for
        # one provider does not need a dummy key for the other.
        anthropic_key = _get_str("ANTHROPIC_API_KEY", "")
        groq_key = _get_str("GROQ_API_KEY", "")
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
        # Explicit placeholder detection, but only for the ACTIVE provider —
        # an unused-provider key that still holds its placeholder should
        # not block a run using the other provider.
        if provider == "anthropic" and anthropic_key == "sk-ant-REPLACE-ME":
            raise ConfigurationError(
                "ANTHROPIC_API_KEY still holds the placeholder value from "
                ".env.example. Set a real key in .env."
            )
        if provider == "groq" and groq_key in ("gsk-REPLACE-ME", "gsk_REPLACE-ME"):
            raise ConfigurationError(
                "GROQ_API_KEY still holds the placeholder value from "
                ".env.example. Set a real key in .env."
            )

        output_dir_raw = _get_str("OUTPUT_DIR", "outputs")
        output_dir = Path(output_dir_raw)
        if not output_dir.is_absolute():
            output_dir = PROJECT_ROOT / output_dir

        # A provider-appropriate default for MODEL_ID, so a run that
        # forgets to set it still produces something meaningful. Real
        # runs should always set MODEL_ID explicitly for provenance.
        default_model = (
            "claude-opus-5" if provider == "anthropic"
            else "llama-3.3-70b-versatile"
        )

        settings = cls(
            llm_provider=provider,  # type: ignore[arg-type]  # validated
            anthropic_api_key=anthropic_key,
            groq_api_key=groq_key,
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
        never leaks a key even if the run was misconfigured.
        """
        data = asdict(self)
        data["anthropic_api_key"] = "<redacted>"
        data["groq_api_key"] = "<redacted>"
        data["output_dir"] = str(self.output_dir)
        data["run_dir"] = str(self.run_dir)
        return data

    def __repr__(self) -> str:  # pragma: no cover - debugging convenience
        """Return a representation that cannot leak either API key."""
        return (
            f"Settings(run_id={self.run_id!r}, llm_provider={self.llm_provider!r}, "
            f"model_id={self.model_id!r}, effort={self.effort!r}, "
            f"thinking_mode={self.thinking_mode!r}, random_seed={self.random_seed}, "
            f"repetitions={self.repetitions}, "
            f"max_revision_rounds={self.max_revision_rounds}, "
            f"anthropic_api_key='<redacted>', groq_api_key='<redacted>')"
        )
