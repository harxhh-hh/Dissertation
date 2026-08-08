"""Functional Requirements agent."""

from __future__ import annotations

import json
from typing import Any

from agents.base import Agent
from config import prompts


class FunctionalRequirementsAgent(Agent):
    """Produces atomic, testable functional requirements in ``FR-NNN`` form.

    The agent writes only what the system **does**. Non-functional qualities
    are handled by :class:`~agents.nonfunctional_agent.NonFunctionalRequirementsAgent`;
    risks and open questions by :class:`~agents.risk_agent.RiskClarificationAgent`.
    Output is Markdown that the architecture layer concatenates into the SRS.
    """

    role_name = "functional_requirements"
    role_system_prompt = prompts.FUNCTIONAL_SYSTEM

    def generate(self, description: str, brief: dict[str, Any]) -> str:
        """Produce the initial functional-requirements section.

        Args:
            description: The natural-language system description.
            brief: The orchestrator's planning brief (already parsed).

        Returns:
            The functional-requirements section as Markdown text.
        """
        brief_json = json.dumps(brief, indent=2, sort_keys=True)
        result = self._call(
            phase="initial",
            user_prompt=prompts.functional_user_prompt(description, brief_json),
        )
        return result.text

    def revise(
        self,
        description: str,
        brief: dict[str, Any],
        previous_markdown: str,
        issues: list[dict[str, Any]],
    ) -> str:
        """Produce a revised functional-requirements section.

        Args:
            description: The natural-language system description.
            brief: The orchestrator's planning brief (already parsed).
            previous_markdown: The prior draft of this section.
            issues: Verification issues to address on this pass. Callers
                filter by ``affected_section`` before calling.

        Returns:
            The revised functional-requirements section as Markdown text.
        """
        brief_json = json.dumps(brief, indent=2, sort_keys=True)
        result = self._call(
            phase="revision",
            user_prompt=prompts.functional_revision_user_prompt(
                description, brief_json, previous_markdown, issues
            ),
        )
        return result.text
