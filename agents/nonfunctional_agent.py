"""Non-Functional Requirements agent."""

from __future__ import annotations

import json
from typing import Any

from agents.base import Agent
from config import prompts


class NonFunctionalRequirementsAgent(Agent):
    """Produces measurable non-functional requirements in ``NFR-NNN`` form.

    Categories covered typically include performance, scalability,
    availability, security, privacy, usability, accessibility,
    maintainability, portability, and regulatory compliance. Each NFR is
    required to carry a measurable criterion; when the input does not
    support a defensible number, the agent is instructed to write ``TBD``
    and defer to the Risk & Clarification agent, rather than fabricating.
    """

    role_name = "nonfunctional_requirements"
    role_system_prompt = prompts.NONFUNCTIONAL_SYSTEM

    def generate(self, description: str, brief: dict[str, Any]) -> str:
        """Produce the initial non-functional-requirements section.

        Args:
            description: The natural-language system description.
            brief: The orchestrator's planning brief (already parsed).

        Returns:
            The non-functional-requirements section as Markdown text.
        """
        brief_json = json.dumps(brief, indent=2, sort_keys=True)
        result = self._call(
            phase="initial",
            user_prompt=prompts.nonfunctional_user_prompt(description, brief_json),
        )
        return result.text

    def revise(
        self,
        description: str,
        brief: dict[str, Any],
        previous_markdown: str,
        issues: list[dict[str, Any]],
    ) -> str:
        """Produce a revised non-functional-requirements section.

        Args:
            description: The natural-language system description.
            brief: The orchestrator's planning brief (already parsed).
            previous_markdown: The prior draft of this section.
            issues: Verification issues to address on this pass. Callers
                filter by ``affected_section`` before calling.

        Returns:
            The revised non-functional-requirements section as Markdown text.
        """
        brief_json = json.dumps(brief, indent=2, sort_keys=True)
        result = self._call(
            phase="revision",
            user_prompt=prompts.nonfunctional_revision_user_prompt(
                description, brief_json, previous_markdown, issues
            ),
        )
        return result.text
