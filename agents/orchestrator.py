"""Orchestrator agent: parses the input and produces a planning brief."""

from __future__ import annotations

from typing import Any

from agents.base import Agent
from config import prompts


class OrchestratorAgent(Agent):
    """Reads the raw system description and produces a structured brief.

    The brief is a JSON object matching
    :data:`~config.prompts.ORCHESTRATOR_BRIEF_SCHEMA`. It is used by every
    downstream specialist so they share one view of stakeholders, scope, and
    assumptions rather than each re-deriving them independently.

    The orchestrator does not itself write requirements. Its output is
    strictly a plan.
    """

    role_name = "orchestrator"
    role_system_prompt = prompts.ORCHESTRATOR_SYSTEM

    def plan(self, description: str) -> dict[str, Any]:
        """Produce the planning brief for ``description``.

        Args:
            description: The natural-language system description supplied by
                the experimenter (or a downstream harness).

        Returns:
            The planning brief as a Python dictionary matching
            :data:`~config.prompts.ORCHESTRATOR_BRIEF_SCHEMA`. It is safe to
            index the required keys without further defensive parsing: the
            API validates the shape before returning.
        """
        result = self._call(
            phase="planning",
            user_prompt=prompts.orchestrator_user_prompt(description),
            output_schema=prompts.ORCHESTRATOR_BRIEF_SCHEMA,
        )
        # ``output_schema`` is set, so ``parsed_json`` is populated and is
        # a dict (schema root is ``type: object``).
        assert isinstance(result.parsed_json, dict), (
            "Orchestrator was requested to return a JSON object but the "
            "parsed response was not a dict; this is a logic error."
        )
        return result.parsed_json
