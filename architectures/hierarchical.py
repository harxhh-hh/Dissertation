"""Hierarchical architecture: orchestrator delegates top-down to specialists.

Protocol (this is what defines the architecture; it is separate from the
role prompts, which are shared with every other architecture):

1. The orchestrator reads the natural-language description and produces a
   structured planning brief.
2. The functional and non-functional agents each produce their section
   *independently* from the brief and the description. They do not see each
   other's output at this stage.
3. The risk-and-clarification agent runs after them, with both drafts and
   the brief as inputs, so it can trace ambiguities and open questions to
   specific requirement identifiers.
4. The four sections are assembled into a draft SRS.
5. The verification agent assesses the draft. If the verdict is
   ``revision_required`` and the run's ``max_revision_rounds`` budget has
   not been exhausted, each specialist whose section carries at least one
   issue produces one revised version, the SRS is re-assembled, and the
   verification agent re-assesses. Otherwise the draft becomes final.

Design notes:

* Every LLM call is issued through the agents' shared client, so every one
  ends up in the interaction log. No call is made from this module directly.
* Revision is bounded by ``settings.max_revision_rounds`` and by any
  ``verdict == "pass"``, either of which terminates the loop. This makes
  the protocol guaranteed to halt.
* The module also exports :func:`format_srs_markdown`, which produces the
  final human-readable Markdown document. The format is stable and shared
  by every architecture so downstream evaluation is consistent.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from agents.base import RunContext
from agents.functional_agent import FunctionalRequirementsAgent
from agents.nonfunctional_agent import NonFunctionalRequirementsAgent
from agents.orchestrator import OrchestratorAgent
from agents.risk_agent import RiskClarificationAgent
from agents.verification_agent import VerificationAgent
from config import prompts
from config.settings import Settings
from utils.llm_client import LLMClient
from utils.logging import ExperimentLogger

#: Canonical name used in interaction log lines and directory names.
ARCHITECTURE_NAME: str = "hierarchical"


# =========================================================================
# Result types
# =========================================================================


@dataclass
class VerificationRound:
    """The outcome of one call to the verification agent.

    Attributes:
        round_index: One-based index (``1`` for the initial assessment,
            ``2`` for the first revision assessment, and so on).
        verdict: ``"pass"`` or ``"revision_required"``.
        summary: Short prose summary from the verification agent.
        issues: The raw issues list from the verdict, unchanged.
    """

    round_index: int
    verdict: str
    summary: str
    issues: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class HierarchicalResult:
    """Everything produced by one hierarchical run of one test case.

    Attributes:
        case_id: The test case identifier.
        repetition: Zero-based repetition index.
        brief: The orchestrator's planning brief.
        functional_markdown: The final functional-requirements section.
        nonfunctional_markdown: The final non-functional section.
        risk_markdown: The final risk-and-clarification section.
        verification_rounds: One entry per verification pass, in order.
        srs_markdown: The final assembled SRS document.
        revised: ``True`` if at least one revision round ran.
    """

    case_id: str
    repetition: int
    brief: dict[str, Any]
    functional_markdown: str
    nonfunctional_markdown: str
    risk_markdown: str
    verification_rounds: list[VerificationRound]
    srs_markdown: str
    revised: bool


# =========================================================================
# Public entry point
# =========================================================================


def run_hierarchical(
    description: str,
    *,
    case_id: str,
    repetition: int,
    client: LLMClient,
    settings: Settings,
    logger: ExperimentLogger,
) -> HierarchicalResult:
    """Run the hierarchical architecture end-to-end on one test case.

    Args:
        description: The natural-language system description to process.
        case_id: Identifier for the test case, recorded on every log line
            and included in the assembled SRS front matter.
        repetition: Zero-based repetition index for this (arch, case) pair.
        client: The shared LLM client for this run.
        settings: The resolved run configuration (used for the revision
            budget and to stamp the SRS front matter).
        logger: The active experiment logger (used for progress messages;
            LLM interactions are logged by the client).

    Returns:
        A :class:`HierarchicalResult` holding every intermediate artefact
        and the final SRS Markdown.
    """
    context = RunContext(
        architecture=ARCHITECTURE_NAME,
        case_id=case_id,
        repetition=repetition,
        protocol_block=prompts.ARCHITECTURE_PROTOCOL_HIERARCHICAL,
    )
    orchestrator = OrchestratorAgent(client, context)
    functional = FunctionalRequirementsAgent(client, context)
    nonfunctional = NonFunctionalRequirementsAgent(client, context)
    risk = RiskClarificationAgent(client, context)
    verification = VerificationAgent(client, context)

    # ---- 1. Orchestrator planning ---------------------------------------
    logger.info("[hierarchical/%s rep=%d] orchestrator planning", case_id, repetition)
    brief = orchestrator.plan(description)

    # ---- 2, 3. Specialist drafts ----------------------------------------
    logger.info("[hierarchical/%s rep=%d] functional requirements", case_id, repetition)
    functional_md = functional.generate(description, brief)

    logger.info("[hierarchical/%s rep=%d] non-functional requirements", case_id, repetition)
    nonfunctional_md = nonfunctional.generate(description, brief)

    logger.info("[hierarchical/%s rep=%d] risk & clarification", case_id, repetition)
    risk_md = risk.generate(description, brief, functional_md, nonfunctional_md)

    # ---- 4. Assemble the initial draft ----------------------------------
    draft_md = format_srs_markdown(
        case_id=case_id,
        repetition=repetition,
        architecture=ARCHITECTURE_NAME,
        settings=settings,
        description=description,
        brief=brief,
        functional_markdown=functional_md,
        nonfunctional_markdown=nonfunctional_md,
        risk_markdown=risk_md,
        revision_history=[],
    )

    # ---- 5. Verification and (optionally) revision ----------------------
    verification_rounds: list[VerificationRound] = []
    round_index = 1
    logger.info(
        "[hierarchical/%s rep=%d] verification round %d", case_id, repetition, round_index
    )
    verdict = verification.assess(draft_md, phase=f"verification_round_{round_index}")
    verification_rounds.append(_round_from_verdict(round_index, verdict))
    logger.info(
        "[hierarchical/%s rep=%d] verification round %d verdict=%s issues=%d",
        case_id,
        repetition,
        round_index,
        verdict["verdict"],
        len(verdict.get("issues", [])),
    )

    revised = False
    while (
        verdict["verdict"] == "revision_required"
        and round_index <= settings.max_revision_rounds
    ):
        # Fan out revisions to whichever specialists have issues to address.
        # An issue tagged ``cross_cutting`` is passed to every specialist so
        # each one can decide what part of it belongs to their section.
        fr_issues = _issues_for_section(verdict, "functional")
        nfr_issues = _issues_for_section(verdict, "nonfunctional")
        risk_issues = _issues_for_section(verdict, "risk")

        if fr_issues:
            logger.info(
                "[hierarchical/%s rep=%d] functional revision (%d issue(s))",
                case_id,
                repetition,
                len(fr_issues),
            )
            functional_md = functional.revise(description, brief, functional_md, fr_issues)
        if nfr_issues:
            logger.info(
                "[hierarchical/%s rep=%d] non-functional revision (%d issue(s))",
                case_id,
                repetition,
                len(nfr_issues),
            )
            nonfunctional_md = nonfunctional.revise(
                description, brief, nonfunctional_md, nfr_issues
            )
        if risk_issues:
            logger.info(
                "[hierarchical/%s rep=%d] risk revision (%d issue(s))",
                case_id,
                repetition,
                len(risk_issues),
            )
            risk_md = risk.revise(
                description,
                brief,
                functional_md,
                nonfunctional_md,
                risk_md,
                risk_issues,
            )

        revised = True
        round_index += 1

        draft_md = format_srs_markdown(
            case_id=case_id,
            repetition=repetition,
            architecture=ARCHITECTURE_NAME,
            settings=settings,
            description=description,
            brief=brief,
            functional_markdown=functional_md,
            nonfunctional_markdown=nonfunctional_md,
            risk_markdown=risk_md,
            revision_history=verification_rounds,
        )

        logger.info(
            "[hierarchical/%s rep=%d] verification round %d",
            case_id,
            repetition,
            round_index,
        )
        verdict = verification.assess(draft_md, phase=f"verification_round_{round_index}")
        verification_rounds.append(_round_from_verdict(round_index, verdict))
        logger.info(
            "[hierarchical/%s rep=%d] verification round %d verdict=%s issues=%d",
            case_id,
            repetition,
            round_index,
            verdict["verdict"],
            len(verdict.get("issues", [])),
        )

    # Rebuild the final SRS once more so the assessment log reflects the
    # last verification round.
    final_srs = format_srs_markdown(
        case_id=case_id,
        repetition=repetition,
        architecture=ARCHITECTURE_NAME,
        settings=settings,
        description=description,
        brief=brief,
        functional_markdown=functional_md,
        nonfunctional_markdown=nonfunctional_md,
        risk_markdown=risk_md,
        revision_history=verification_rounds,
    )

    return HierarchicalResult(
        case_id=case_id,
        repetition=repetition,
        brief=brief,
        functional_markdown=functional_md,
        nonfunctional_markdown=nonfunctional_md,
        risk_markdown=risk_md,
        verification_rounds=verification_rounds,
        srs_markdown=final_srs,
        revised=revised,
    )


# =========================================================================
# Helpers
# =========================================================================


def _issues_for_section(verdict: dict[str, Any], section: str) -> list[dict[str, Any]]:
    """Return the issues that a given specialist should address on revision.

    An issue is included if its ``affected_section`` is either the requested
    section or ``"cross_cutting"``. Cross-cutting issues are forwarded to
    every specialist so each can address the portion of the issue that
    belongs to their section.

    Args:
        verdict: The parsed verification verdict.
        section: One of ``"functional"``, ``"nonfunctional"``, ``"risk"``.

    Returns:
        The filtered list of issue objects. Empty if none apply.
    """
    result: list[dict[str, Any]] = []
    for issue in verdict.get("issues", []):
        affected = issue.get("affected_section")
        if affected == section or affected == "cross_cutting":
            result.append(issue)
    return result


def _round_from_verdict(round_index: int, verdict: dict[str, Any]) -> VerificationRound:
    """Package a verdict into a :class:`VerificationRound` for the result."""
    return VerificationRound(
        round_index=round_index,
        verdict=verdict["verdict"],
        summary=verdict.get("summary", ""),
        issues=list(verdict.get("issues", [])),
    )


# =========================================================================
# SRS Markdown assembly
# =========================================================================


def format_srs_markdown(
    *,
    case_id: str,
    repetition: int,
    architecture: str,
    settings: Settings,
    description: str,
    brief: dict[str, Any],
    functional_markdown: str,
    nonfunctional_markdown: str,
    risk_markdown: str,
    revision_history: list[VerificationRound],
) -> str:
    """Assemble a full SRS Markdown document.

    The format is architecture-agnostic on purpose: the baseline and the
    other two architectures will produce documents in the same shape, so
    downstream evaluation compares like with like.

    Args:
        case_id: Test case identifier, stamped into the front matter.
        repetition: Repetition index, stamped into the front matter.
        architecture: The generating architecture's canonical name.
        settings: The run configuration (used to record model, effort,
            thinking mode and the revision budget in the front matter).
        description: The original natural-language description; embedded
            verbatim so a reader can compare requirements to the input
            without leaving the document.
        brief: The orchestrator's planning brief.
        functional_markdown: The final functional-requirements section.
        nonfunctional_markdown: The final non-functional section.
        risk_markdown: The final risk-and-clarification section.
        revision_history: Verification rounds so far, in order.

    Returns:
        The full SRS as a single Markdown string.
    """
    brief_json = json.dumps(brief, indent=2, sort_keys=True)

    sections: list[str] = []

    sections.append(
        f"# Software Requirements Specification — {case_id}\n\n"
        "> Generated by the multi-agent SRS generation experiment. "
        "This document is machine-produced and has not been reviewed by a "
        "human requirements engineer.\n"
    )

    # Provenance block — anything a moderator would want to know about how
    # the document was produced, without having to open the interaction log.
    sections.append(
        "## Run metadata\n\n"
        f"- Test case: `{case_id}`\n"
        f"- Repetition index: `{repetition}`\n"
        f"- Architecture: `{architecture}`\n"
        f"- Run ID: `{settings.run_id}`\n"
        f"- Model: `{settings.model_id}`\n"
        f"- Effort: `{settings.effort}`\n"
        f"- Thinking mode: `{settings.thinking_mode}`\n"
        f"- Revision budget: `{settings.max_revision_rounds}` round(s)\n"
        f"- Revision rounds executed: `{max(0, len(revision_history) - 1)}`\n"
    )

    sections.append(
        "## Input description\n\n"
        f"> {description.strip().replace(chr(10), chr(10) + '> ')}\n"
    )

    sections.append(
        "## Orchestrator planning brief\n\n"
        f"- **System type:** {brief.get('system_type', '(not stated)')}\n"
        f"- **Scope summary:** {brief.get('scope_summary', '(not stated)')}\n\n"
        "### Stakeholders\n\n"
        + _format_stakeholders(brief.get("stakeholders", []))
        + "\n### Assumptions\n\n"
        + _format_bulleted_strings(brief.get("assumptions", []))
        + "\n<details>\n<summary>Full brief (JSON)</summary>\n\n"
        f"```json\n{brief_json}\n```\n\n</details>\n"
    )

    sections.append("## Functional requirements\n\n" + functional_markdown.strip() + "\n")
    sections.append(
        "## Non-functional requirements\n\n" + nonfunctional_markdown.strip() + "\n"
    )
    sections.append(
        "## Risks, ambiguities, and open questions\n\n" + risk_markdown.strip() + "\n"
    )

    if revision_history:
        sections.append(_format_verification_history(revision_history))

    return "\n".join(sections)


def _format_stakeholders(stakeholders: list[dict[str, Any]]) -> str:
    """Render the orchestrator's stakeholder list as a Markdown bulleted list."""
    if not stakeholders:
        return "_None identified._\n"
    lines: list[str] = []
    for item in stakeholders:
        role = str(item.get("role", "(role not stated)")).strip()
        responsibilities = str(item.get("responsibilities", "")).strip()
        if responsibilities:
            lines.append(f"- **{role}** — {responsibilities}")
        else:
            lines.append(f"- **{role}**")
    return "\n".join(lines) + "\n"


def _format_bulleted_strings(items: list[str]) -> str:
    """Render a list of strings as a Markdown bulleted list, or a placeholder."""
    if not items:
        return "_None stated._\n"
    return "\n".join(f"- {str(item).strip()}" for item in items) + "\n"


def _format_verification_history(rounds: list[VerificationRound]) -> str:
    """Render the verification history as an appendix section."""
    lines: list[str] = ["## Verification history\n"]
    for round_ in rounds:
        lines.append(
            f"### Round {round_.round_index} — verdict: `{round_.verdict}`\n\n"
            f"{round_.summary.strip() or '_(no summary)_'}\n"
        )
        if not round_.issues:
            lines.append("_No issues raised._\n")
            continue
        lines.append("| # | Severity | Category | Affected section | Affected IDs | Description |")
        lines.append("|---|---|---|---|---|---|")
        for idx, issue in enumerate(round_.issues, start=1):
            ids = ", ".join(issue.get("affected_ids", []) or []) or "—"
            description = (
                str(issue.get("description", "")).strip().replace("\n", " ").replace("|", "\\|")
            )
            lines.append(
                f"| {idx} | {issue.get('severity', '?')} | {issue.get('category', '?')} | "
                f"{issue.get('affected_section', '?')} | {ids} | {description} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"
