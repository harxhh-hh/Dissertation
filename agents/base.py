"""Common base class for every agent in the multi-agent system.

Every agent role (orchestrator, functional, non-functional, risk,
verification) is a subclass of :class:`Agent`. The base class does three
things that would otherwise be duplicated in five files:

* Holds a reference to the shared :class:`~utils.client.LLMClient` and to
  the immutable per-run context (case identifier, repetition index,
  architecture name).
* Assembles the fully-composed system prompt (role prompt + architecture
  protocol addendum) via
  :func:`~config.prompts.compose_system_prompt`, so a subclass never has to
  remember to concatenate the two.
* Provides a small ``_call`` helper that forwards to the client with the
  agent's role name and the current phase, cutting boilerplate at the call
  sites and guaranteeing every call is attributed to the correct agent in
  the interaction log.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Mapping

from config.prompts import compose_system_prompt
from utils.llm_client import LLMCallResult, LLMClient


@dataclass(frozen=True)
class RunContext:
    """Per-invocation metadata shared by every agent in one architecture run.

    Attributes:
        architecture: The condition under test — one of ``hierarchical``,
            ``peer_to_peer``, ``debate``, ``baseline_single_prompt``.
        case_id: Identifier of the test case being processed.
        repetition: Zero-based repetition index for this (arch, case) pair.
        protocol_block: The architecture-specific protocol addendum from
            :mod:`config.prompts`; appended to every agent's role prompt.
    """

    architecture: str
    case_id: str
    repetition: int
    protocol_block: str


class Agent:
    """Base class for every specialist agent.

    Subclasses set two class variables:

    * ``role_name`` — the short identifier written to the interaction log
      (for example ``functional_requirements``).
    * ``role_system_prompt`` — the role's system prompt from
      :mod:`config.prompts` (e.g. :data:`~config.prompts.FUNCTIONAL_SYSTEM`).

    They then implement one or more high-level methods (``plan``,
    ``generate``, ``revise``, ``assess``) that call :meth:`_call`.

    Args:
        client: The shared :class:`LLMClient` for this run.
        context: The immutable per-invocation :class:`RunContext`.
    """

    #: Short identifier for this agent role. Overridden by subclasses.
    role_name: ClassVar[str] = "agent"

    #: The role's system prompt, from :mod:`config.prompts`. Overridden by
    #: subclasses.
    role_system_prompt: ClassVar[str] = ""

    def __init__(self, client: LLMClient, context: RunContext) -> None:
        self._client = client
        self._context = context
        # Assemble once per agent instance so the byte-identical string ends
        # up in the log for every call this agent issues.
        self._system_prompt: str = compose_system_prompt(
            self.role_system_prompt, context.protocol_block
        )

    # -- accessors ---------------------------------------------------------

    @property
    def context(self) -> RunContext:
        """The run context this agent was constructed with."""
        return self._context

    @property
    def system_prompt(self) -> str:
        """The fully-assembled system prompt used on every call."""
        return self._system_prompt

    # -- shared call path --------------------------------------------------

    def _call(
        self,
        *,
        phase: str,
        user_prompt: str,
        output_schema: Mapping[str, Any] | None = None,
    ) -> LLMCallResult:
        """Issue one LLM call attributed to this agent and phase.

        Args:
            phase: The stage within the protocol (``initial``, ``revision``,
                ``verification``, ...). Recorded on the interaction log.
            user_prompt: The user-turn content, assembled by the caller from
                :mod:`config.prompts`.
            output_schema: Optional JSON Schema for structured output.

        Returns:
            The client's :class:`LLMCallResult`.
        """
        return self._client.call(
            architecture=self._context.architecture,
            agent=self.role_name,
            case_id=self._context.case_id,
            repetition=self._context.repetition,
            phase=phase,
            system_prompt=self._system_prompt,
            user_prompt=user_prompt,
            output_schema=output_schema,
        )
