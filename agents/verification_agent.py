"""Verification agent: judges the draft SRS and can trigger a revision."""

from __future__ import annotations

from typing import Any

from agents.base import Agent
from config import prompts


class VerificationAgent(Agent):
    """Assesses a draft SRS against ISO/IEC/IEEE 29148-inspired criteria.

    Returns a JSON verdict matching
    :data:`~config.prompts.VERIFICATION_VERDICT_SCHEMA`, with:

    * ``verdict`` — ``"pass"`` or ``"revision_required"``.
    * ``summary`` — a short prose summary of the assessment.
    * ``issues`` — a list of specific defects, each labelled with a
      severity, category, affected section, and (where relevant) the
      requirement identifiers it points at. Downstream the architecture
      layer routes each issue back to the agent responsible for the
      affected section.

    In the debate architecture (stage 4), the same agent additionally
    plays the role of arbiter over competing proposals; that call path
    will re-use this class with a different phase label.
    """

    role_name = "verification"
    role_system_prompt = prompts.VERIFICATION_SYSTEM

    #: Phase label used when the verification agent assesses a full draft.
    #: Kept as a class constant so the architecture layer and the log always
    #: agree on the string.
    ASSESSMENT_PHASE: str = "verification"

    def assess(self, draft_srs_markdown: str, *, phase: str | None = None) -> dict[str, Any]:
        """Assess a draft SRS.

        Args:
            draft_srs_markdown: The assembled draft SRS as Markdown.
            phase: Optional override for the phase label on the log line.
                Useful when the verification agent is invoked for a
                revision round (e.g. ``"verification_round_2"``) so calls
                on the same case in the same repetition can be
                distinguished after the fact. Defaults to
                :data:`ASSESSMENT_PHASE`.

        Returns:
            The parsed verdict, guaranteed by the schema to include the
            ``verdict``, ``summary``, and ``issues`` fields.
        """
        result = self._call(
            phase=phase or self.ASSESSMENT_PHASE,
            user_prompt=prompts.verification_user_prompt(draft_srs_markdown),
            output_schema=prompts.VERIFICATION_VERDICT_SCHEMA,
        )
        assert isinstance(result.parsed_json, dict), (
            "Verification agent was requested to return a JSON object but "
            "the parsed response was not a dict; this is a logic error."
        )
        return result.parsed_json
