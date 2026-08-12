"""Pre-flight input guard for the "Try your own idea" tab.

The formal "Run experiment" tab only ever runs the 4 fixed, pre-vetted test
cases in ``test_cases/cases.py`` — no free text ever reaches an agent there.
The ad hoc tab is different: whatever a user types goes straight into the
same prompts a real generation run uses, with no check that it's actually
a system description at all. Typing something like "crack a joke" into
that field previously still triggered a full generation run — every
specialist agent dutifully invented functional/non-functional requirements
for it, because nothing ever asked "is this actually a system description?"

:class:`InputGuardAgent` is a single, cheap, boxed-in classifier call that
answers exactly that question and nothing else — it never sees or reasons
about anything downstream, and it never produces requirements itself.
"""

from __future__ import annotations

from typing import Any

from agents.base import Agent, RunContext
from config import prompts
from utils.llm_client import LLMClient


class InputGuardAgent(Agent):
    """Classifies whether free text plausibly describes a software system."""

    role_name = "input_guard"
    role_system_prompt = prompts.INPUT_GUARD_SYSTEM

    def check(self, description: str) -> dict[str, Any]:
        """Classify one piece of user-submitted text.

        Args:
            description: The raw text from the "Describe your system" field.

        Returns:
            The parsed verdict, matching
            :data:`~config.prompts.INPUT_GUARD_SCHEMA`.
        """
        result = self._call(
            phase="check",
            user_prompt=prompts.input_guard_user_prompt(description),
            output_schema=prompts.INPUT_GUARD_SCHEMA,
        )
        assert isinstance(result.parsed_json, dict), (
            "Input guard was requested to return a JSON object but the "
            "parsed response was not a dict; this is a logic error."
        )
        return result.parsed_json


def check_description(description: str, *, client: LLMClient) -> tuple[bool, str]:
    """Ask the input guard whether ``description`` is worth generating an SRS for.

    Args:
        description: The raw, unmodified text from the ad hoc tab's
            "Describe your system" field.
        client: The shared LLM client for this run.

    Returns:
        ``(is_system_description, reason)`` — ``reason`` is a short,
        human-readable explanation suitable for showing directly in the UI.
    """
    # A fresh role — reuses the baseline protocol block for the same reason
    # evaluation/rubric.py's EvaluatorAgent does: this isn't one of the four
    # architectures under test, so it shouldn't introduce a fifth
    # architecture name into the interaction log.
    context = RunContext(
        architecture="guardrail",
        case_id="adhoc-input-check",
        repetition=0,
        protocol_block=prompts.ARCHITECTURE_PROTOCOL_BASELINE,
    )
    guard = InputGuardAgent(client, context)
    verdict = guard.check(description)
    return bool(verdict.get("is_system_description", False)), str(verdict.get("reason", ""))
