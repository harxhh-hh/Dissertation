"""Risk & Clarification agent."""

from __future__ import annotations

import json
from typing import Any

from agents.base import Agent
from config import prompts


class RiskClarificationAgent(Agent):
    """Identifies constraints, risks, ambiguities, and open questions.

    Runs after the functional and non-functional agents have produced their
    initial drafts, so it can trace ambiguities and open questions back to
    specific requirement identifiers. Output is Markdown with four fixed
    subheadings, in the order specified by the role prompt.
    """

    role_name = "risk_clarification"
    role_system_prompt = prompts.RISK_SYSTEM

    def generate(
        self,
        description: str,
        brief: dict[str, Any],
        functional_markdown: str,
        nonfunctional_markdown: str,
    ) -> str:
        """Produce the initial risk-and-clarification section.

        Args:
            description: The natural-language system description.
            brief: The orchestrator's planning brief (already parsed).
            functional_markdown: The functional-requirements draft.
            nonfunctional_markdown: The non-functional-requirements draft.

        Returns:
            The risk-and-clarification section as Markdown text.
        """
        brief_json = json.dumps(brief, indent=2, sort_keys=True)
        result = self._call(
            phase="initial",
            user_prompt=prompts.risk_user_prompt(
                description, brief_json, functional_markdown, nonfunctional_markdown
            ),
        )
        return result.text

    def revise(
        self,
        description: str,
        brief: dict[str, Any],
        functional_markdown: str,
        nonfunctional_markdown: str,
        previous_markdown: str,
        issues: list[dict[str, Any]],
    ) -> str:
        """Produce a revised risk-and-clarification section.

        Args:
            description: The natural-language system description.
            brief: The orchestrator's planning brief (already parsed).
            functional_markdown: The current functional-requirements section.
            nonfunctional_markdown: The current non-functional section.
            previous_markdown: The prior draft of the risk section.
            issues: Verification issues to address on this pass. Callers
                filter by ``affected_section`` before calling.

        Returns:
            The revised risk-and-clarification section as Markdown text.
        """
        brief_json = json.dumps(brief, indent=2, sort_keys=True)
        result = self._call(
            phase="revision",
            user_prompt=prompts.risk_revision_user_prompt(
                description,
                brief_json,
                functional_markdown,
                nonfunctional_markdown,
                previous_markdown,
                issues,
            ),
        )
        return result.text
