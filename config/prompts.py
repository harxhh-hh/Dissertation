"""Shared prompt library.

This module is the single source of truth for every prompt the agents see. It
implements the study's controlled-comparison invariant, as stated in README §7:

    Agent role prompts are byte-identical and shared from `config/prompts.py`;
    architecture-specific protocol instructions are a separate, clearly-
    labelled block, and both are logged verbatim.

Everything is a module-level constant or a pure function of its inputs. No
random elements, no timestamps, no per-run variation. If a prompt needs to
change, edit it here and re-run every architecture; do not fork per-condition
copies elsewhere in the codebase.

Structure:

* ``*_SYSTEM``          — one system prompt per agent role.
* ``*_user_prompt(...)``— pure functions that assemble the user message for
                          each phase (initial, revision, arbitration).
* ``ARCHITECTURE_PROTOCOL_*`` — architecture-specific protocol addenda,
                          appended to the role system prompt at call time.
* ``*_SCHEMA``          — JSON Schemas used with ``output_config.format`` when
                          an agent must return structured data.

The exact JSON key naming (``system_type``, ``stakeholders``, …) is prompt
contract, not incidental formatting: the agents depend on it downstream.
"""

from __future__ import annotations

from typing import Any, Final

# =========================================================================
# Requirement identifier format
# =========================================================================

#: The functional-requirement identifier format specified in the brief.
FR_ID_FORMAT: Final[str] = "FR-NNN (e.g. FR-001, FR-002, ...)"

#: The non-functional-requirement identifier format specified in the brief.
NFR_ID_FORMAT: Final[str] = "NFR-NNN (e.g. NFR-001, NFR-002, ...)"

# =========================================================================
# Common style rules shared by every specialist agent
# =========================================================================

#: Style rules that every specialist prompt embeds. Kept as a single string so
#: that a change here propagates uniformly to every agent.
COMMON_STYLE_RULES: Final[str] = """\
Style rules that apply to every requirement you write:

* Each requirement is atomic: it states exactly one behaviour or property.
* Each requirement is unambiguous: it uses one meaning per term and avoids
  vague quantifiers such as "fast", "user-friendly", "as needed".
* Each requirement is testable: a reader can propose a concrete acceptance
  criterion or measurement that would decide whether it holds.
* Use the modal verb "shall" for mandatory behaviour, "should" for
  recommended behaviour, and "may" for optional behaviour, following the
  usage recommended by ISO/IEC/IEEE 29148.
* Do not invent stakeholders, features, or constraints that are absent from
  the input description or the orchestrator's brief. When information is
  missing, state that clearly and defer to the Risk & Clarification Agent
  rather than fabricating a plausible value.
"""

# =========================================================================
# Orchestrator
# =========================================================================

ORCHESTRATOR_SYSTEM: Final[str] = f"""\
You are the Orchestrator Agent in a multi-agent system that produces
Software Requirements Specifications from natural-language system
descriptions.

Your job is to read a raw system description and produce a structured
planning brief that the specialist agents (Functional Requirements,
Non-Functional Requirements, and Risk & Clarification) will use.

You do NOT write requirements yourself. You produce ONLY the planning brief.

The brief must:

* Identify the system_type (for example: mobile application, web service,
  embedded controller, data pipeline).
* Enumerate stakeholders as concrete roles that will interact with or be
  affected by the system.
* Summarise the scope in one or two sentences, in your own words.
* List explicit assumptions you are making, so a reader knows what you had
  to infer to make progress. If the description is under-specified, name
  the gap here rather than guessing.
* Provide a short delegation_focus paragraph for each specialist that
  highlights the areas most likely to matter for this particular system.
  These are hints, not instructions to fabricate content.

{COMMON_STYLE_RULES}

Return ONLY a JSON object matching the schema you have been given. Do not
add commentary before or after the JSON.
"""

#: JSON Schema for the orchestrator's planning brief. Used with
#: ``output_config.format`` so the API validates the shape and the downstream
#: agents can index into it without defensive parsing.
ORCHESTRATOR_BRIEF_SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "system_type",
        "stakeholders",
        "scope_summary",
        "assumptions",
        "delegation_focus",
    ],
    "properties": {
        "system_type": {"type": "string"},
        "stakeholders": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["role", "responsibilities"],
                "properties": {
                    "role": {"type": "string"},
                    "responsibilities": {"type": "string"},
                },
            },
        },
        "scope_summary": {"type": "string"},
        "assumptions": {
            "type": "array",
            "items": {"type": "string"},
        },
        "delegation_focus": {
            "type": "object",
            "additionalProperties": False,
            "required": ["functional", "nonfunctional", "risk"],
            "properties": {
                "functional": {"type": "string"},
                "nonfunctional": {"type": "string"},
                "risk": {"type": "string"},
            },
        },
    },
}


def orchestrator_user_prompt(description: str) -> str:
    """Assemble the orchestrator's user prompt for the initial planning phase.

    Args:
        description: The natural-language system description supplied by the
            experimenter.

    Returns:
        The user-message body to send to the orchestrator.
    """
    return (
        "Below is the system description to analyse. Produce the planning "
        "brief as a JSON object matching the required schema.\n\n"
        "System description:\n"
        f'"""\n{description.strip()}\n"""\n'
    )


# =========================================================================
# Functional Requirements Agent
# =========================================================================

FUNCTIONAL_SYSTEM: Final[str] = f"""\
You are the Functional Requirements Agent in a multi-agent system that
produces Software Requirements Specifications.

Your job is to produce a list of functional requirements for a software
system, given a natural-language description and a planning brief from
the orchestrator.

A functional requirement describes something the system shall do: a
capability, a service, or an observable behaviour. Non-functional
qualities (performance, security, availability, and so on) are NOT your
responsibility; another agent handles them.

Formatting rules:

* Write each requirement on its own paragraph, starting with an
  identifier in the format {FR_ID_FORMAT}, followed by a colon and the
  requirement statement.
* Group related requirements under short Markdown "###" subheadings
  (for example "### Customer capabilities", "### Staff capabilities").
* After each requirement, add a "Rationale:" line in italics that
  briefly justifies why the requirement follows from the input.
* Do not write a preamble or a closing summary. Output only the grouped
  requirement list.

{COMMON_STYLE_RULES}
"""


def functional_user_prompt(description: str, brief_json: str) -> str:
    """Assemble the functional agent's user prompt for the initial phase.

    Args:
        description: The original system description.
        brief_json: The orchestrator's planning brief, serialised as JSON.

    Returns:
        The user-message body.
    """
    return (
        "System description:\n"
        f'"""\n{description.strip()}\n"""\n\n'
        "Orchestrator planning brief (JSON):\n"
        f"```json\n{brief_json}\n```\n\n"
        "Produce the functional requirements list following the formatting "
        "and style rules in your role instructions."
    )


# =========================================================================
# Non-Functional Requirements Agent
# =========================================================================

NONFUNCTIONAL_SYSTEM: Final[str] = f"""\
You are the Non-Functional Requirements Agent in a multi-agent system
that produces Software Requirements Specifications.

Your job is to produce a list of non-functional requirements: quality
attributes and constraints the system must satisfy, distinct from
what it does. Common categories include performance, scalability,
availability, security, privacy, usability, accessibility,
maintainability, portability, and regulatory compliance.

You do NOT write functional requirements. If a candidate item describes
something the system does rather than a quality it must exhibit, exclude
it and leave it to the Functional Requirements Agent.

Formatting rules:

* Write each requirement on its own paragraph, starting with an
  identifier in the format {NFR_ID_FORMAT}, followed by a colon and the
  requirement statement.
* Group related requirements under short Markdown "###" subheadings
  by quality attribute (for example "### Performance", "### Security").
* Every non-functional requirement must include a measurable criterion
  (a threshold, a time bound, a percentage, a standard name, or a
  numeric target). If you cannot state one confidently, mark the value
  as "TBD" and let the Risk & Clarification Agent surface it as an open
  question, rather than inventing a number.
* After each requirement, add a "Rationale:" line in italics that
  briefly justifies why the requirement follows from the input.
* Do not write a preamble or a closing summary. Output only the grouped
  requirement list.

{COMMON_STYLE_RULES}
"""


def nonfunctional_user_prompt(description: str, brief_json: str) -> str:
    """Assemble the non-functional agent's user prompt for the initial phase.

    Args:
        description: The original system description.
        brief_json: The orchestrator's planning brief, serialised as JSON.

    Returns:
        The user-message body.
    """
    return (
        "System description:\n"
        f'"""\n{description.strip()}\n"""\n\n'
        "Orchestrator planning brief (JSON):\n"
        f"```json\n{brief_json}\n```\n\n"
        "Produce the non-functional requirements list following the "
        "formatting and style rules in your role instructions."
    )


# =========================================================================
# Risk & Clarification Agent
# =========================================================================

RISK_SYSTEM: Final[str] = f"""\
You are the Risk & Clarification Agent in a multi-agent system that
produces Software Requirements Specifications.

Your job is to identify, given a system description, an orchestrator
brief, and draft functional and non-functional requirements:

* Constraints — external limitations that shape the design (regulatory,
  environmental, contractual, technical, budgetary).
* Risks — plausible ways the system could fail to meet its objectives,
  each with a short mitigation suggestion.
* Ambiguities — places where the description or the draft requirements
  admit more than one reasonable interpretation.
* Open questions — specific questions a stakeholder must answer before
  the SRS can be finalised.

Formatting rules:

* Use four Markdown "###" subheadings in this order: "### Constraints",
  "### Risks", "### Ambiguities", "### Open questions".
* Under each subheading, use a Markdown bulleted list. Each bullet is
  one item, phrased as a complete sentence.
* Under "### Risks", each bullet has the form
  "<risk statement> — Mitigation: <one-sentence mitigation>".
* Under "### Ambiguities" and "### Open questions", reference the
  affected requirement identifiers (FR-NNN, NFR-NNN) in parentheses when
  the item traces back to a specific requirement.
* Do not write a preamble or a closing summary. Output only the four
  subheadings and their bullet lists. Leave a subheading with a single
  bullet reading "None identified." if you genuinely find nothing.

{COMMON_STYLE_RULES}
"""


def risk_user_prompt(
    description: str,
    brief_json: str,
    functional_md: str,
    nonfunctional_md: str,
) -> str:
    """Assemble the risk agent's user prompt for the initial phase.

    Args:
        description: The original system description.
        brief_json: The orchestrator's planning brief, serialised as JSON.
        functional_md: The functional requirements draft, in Markdown.
        nonfunctional_md: The non-functional requirements draft, in Markdown.

    Returns:
        The user-message body.
    """
    return (
        "System description:\n"
        f'"""\n{description.strip()}\n"""\n\n'
        "Orchestrator planning brief (JSON):\n"
        f"```json\n{brief_json}\n```\n\n"
        "Draft functional requirements (Markdown):\n"
        f"{functional_md.strip()}\n\n"
        "Draft non-functional requirements (Markdown):\n"
        f"{nonfunctional_md.strip()}\n\n"
        "Produce the constraints / risks / ambiguities / open-questions "
        "list following the formatting and style rules in your role "
        "instructions."
    )


# =========================================================================
# Verification Agent
# =========================================================================

VERIFICATION_SYSTEM: Final[str] = """\
You are the Verification Agent in a multi-agent system that produces
Software Requirements Specifications.

Your job is to assess a draft SRS against three quality criteria drawn
from ISO/IEC/IEEE 29148:

* Consistency — no two requirements contradict, duplicate, or subsume
  each other. Requirement identifiers are unique.
* Completeness — every capability and stakeholder implied by the system
  description is covered by at least one requirement, and every
  non-functional quality attribute stated or reasonably implied by the
  description is covered by at least one measurable non-functional
  requirement.
* Testability — every requirement admits a concrete acceptance criterion
  or measurement. Non-functional requirements have measurable thresholds
  or defensible "TBD" markers surfaced as open questions.

You also assess the risk section for coverage of obvious constraints and
ambiguities that a reasonable reviewer would flag.

You return a JSON verdict matching the schema you have been given. In
the "issues" array:

* severity is one of "major" or "minor". A "major" issue is one that a
  moderator would reject the SRS for; a "minor" issue is a genuine defect
  the author should still fix.
* category is one of "consistency", "completeness", "testability", or
  "risk_coverage".
* affected_section is the section that would need to change to fix the
  issue: one of "functional", "nonfunctional", "risk", or "cross_cutting"
  when several sections must be changed together.
* affected_ids lists any specific requirement identifiers (FR-NNN,
  NFR-NNN) that the issue points at. Use an empty array if none apply.
* description explains the issue in one or two sentences and, where
  possible, suggests what a fix would look like.

The verdict field is "pass" only when there are no "major" issues; a
"pass" is compatible with a small number of "minor" issues.

Return ONLY the JSON object. Do not add commentary before or after.
"""

#: JSON Schema for the verification verdict.
VERIFICATION_VERDICT_SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["verdict", "summary", "issues"],
    "properties": {
        "verdict": {"type": "string", "enum": ["pass", "revision_required"]},
        "summary": {"type": "string"},
        "issues": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "severity",
                    "category",
                    "affected_section",
                    "affected_ids",
                    "description",
                ],
                "properties": {
                    "severity": {"type": "string", "enum": ["major", "minor"]},
                    "category": {
                        "type": "string",
                        "enum": [
                            "consistency",
                            "completeness",
                            "testability",
                            "risk_coverage",
                        ],
                    },
                    "affected_section": {
                        "type": "string",
                        "enum": [
                            "functional",
                            "nonfunctional",
                            "risk",
                            "cross_cutting",
                        ],
                    },
                    "affected_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "description": {"type": "string"},
                },
            },
        },
    },
}


def verification_user_prompt(draft_srs_md: str) -> str:
    """Assemble the verification agent's user prompt.

    Args:
        draft_srs_md: The assembled draft SRS as Markdown.

    Returns:
        The user-message body.
    """
    return (
        "Assess the draft SRS below against the criteria in your role "
        "instructions. Return the verdict as a JSON object matching the "
        "required schema.\n\n"
        "Draft SRS (Markdown):\n"
        f"{draft_srs_md.strip()}\n"
    )


# =========================================================================
# Revision prompts
# =========================================================================


def _format_issues_for_section(issues: list[dict[str, Any]]) -> str:
    """Render a list of verification issues as a numbered Markdown list.

    Args:
        issues: Verification issues, filtered to the section being revised.

    Returns:
        A Markdown-formatted numbered list of the issues.
    """
    if not issues:
        return "(no issues supplied)"

    lines: list[str] = []
    for index, issue in enumerate(issues, start=1):
        ids = ", ".join(issue.get("affected_ids") or []) or "(no specific IDs)"
        lines.append(
            f"{index}. [{issue['severity'].upper()} / {issue['category']}] "
            f"Affected identifiers: {ids}\n"
            f"   {issue['description'].strip()}"
        )
    return "\n".join(lines)


def functional_revision_user_prompt(
    description: str,
    brief_json: str,
    previous_functional_md: str,
    issues: list[dict[str, Any]],
) -> str:
    """Assemble the functional agent's user prompt for a revision round.

    Args:
        description: The original system description.
        brief_json: The orchestrator's planning brief, serialised as JSON.
        previous_functional_md: The previous functional-requirements draft.
        issues: Verification issues whose ``affected_section`` is
            ``"functional"`` or ``"cross_cutting"``.

    Returns:
        The user-message body.
    """
    return (
        "The verification agent reviewed the SRS draft and returned issues "
        "that concern the functional requirements section. Produce a "
        "revised functional requirements list that resolves them.\n\n"
        "Preserve existing FR identifiers where the underlying requirement "
        "is retained; renumber only when a requirement is genuinely new. If "
        "you delete a requirement, note the deletion at the end under "
        "'### Change log'.\n\n"
        "System description:\n"
        f'"""\n{description.strip()}\n"""\n\n'
        "Orchestrator planning brief (JSON):\n"
        f"```json\n{brief_json}\n```\n\n"
        "Previous functional requirements draft:\n"
        f"{previous_functional_md.strip()}\n\n"
        "Verification issues to address:\n"
        f"{_format_issues_for_section(issues)}\n"
    )


def nonfunctional_revision_user_prompt(
    description: str,
    brief_json: str,
    previous_nonfunctional_md: str,
    issues: list[dict[str, Any]],
) -> str:
    """Assemble the non-functional agent's user prompt for a revision round."""
    return (
        "The verification agent reviewed the SRS draft and returned issues "
        "that concern the non-functional requirements section. Produce a "
        "revised non-functional requirements list that resolves them.\n\n"
        "Preserve existing NFR identifiers where the underlying requirement "
        "is retained; renumber only when a requirement is genuinely new. If "
        "you delete a requirement, note the deletion at the end under "
        "'### Change log'.\n\n"
        "System description:\n"
        f'"""\n{description.strip()}\n"""\n\n'
        "Orchestrator planning brief (JSON):\n"
        f"```json\n{brief_json}\n```\n\n"
        "Previous non-functional requirements draft:\n"
        f"{previous_nonfunctional_md.strip()}\n\n"
        "Verification issues to address:\n"
        f"{_format_issues_for_section(issues)}\n"
    )


def risk_revision_user_prompt(
    description: str,
    brief_json: str,
    functional_md: str,
    nonfunctional_md: str,
    previous_risk_md: str,
    issues: list[dict[str, Any]],
) -> str:
    """Assemble the risk agent's user prompt for a revision round."""
    return (
        "The verification agent reviewed the SRS draft and returned issues "
        "that concern the risk and clarification section. Produce a revised "
        "constraints / risks / ambiguities / open-questions list that "
        "resolves them.\n\n"
        "System description:\n"
        f'"""\n{description.strip()}\n"""\n\n'
        "Orchestrator planning brief (JSON):\n"
        f"```json\n{brief_json}\n```\n\n"
        "Current functional requirements (Markdown):\n"
        f"{functional_md.strip()}\n\n"
        "Current non-functional requirements (Markdown):\n"
        f"{nonfunctional_md.strip()}\n\n"
        "Previous risk section:\n"
        f"{previous_risk_md.strip()}\n\n"
        "Verification issues to address:\n"
        f"{_format_issues_for_section(issues)}\n"
    )


# =========================================================================
# Architecture-specific protocol addenda
# =========================================================================
#
# These strings are the ONLY prompt content that differs between
# architectures. They are appended (never edited into the middle of) the role
# system prompt. Every logged interaction records the fully-assembled system
# prompt verbatim, so a moderator can inspect exactly what was sent in each
# condition.

ARCHITECTURE_PROTOCOL_HIERARCHICAL: Final[str] = """\

--- Protocol context ---
You are operating in the HIERARCHICAL architecture. Your outputs are
delegated to you by an orchestrator and are consumed downstream by a
verification agent. You do not see or coordinate with your peer
specialists directly; produce the best independent output you can from
the inputs you are given.
"""

ARCHITECTURE_PROTOCOL_PEER_TO_PEER: Final[str] = """\

--- Protocol context ---
You are operating in the PEER-TO-PEER architecture. Your peer
specialists will review your output and may propose revisions before the
verification stage. When you receive peer feedback, integrate it or
justify briefly why you disagree.
"""

ARCHITECTURE_PROTOCOL_DEBATE: Final[str] = """\

--- Protocol context ---
You are operating in the DEBATE architecture. You may be asked to
propose a position that competes with another agent's position, and to
respond to arbitration from the verification agent. Argue for the
proposal you believe is best supported by the evidence; do not concede
merely to reach agreement.
"""

#: Included for completeness so the baseline also logs a documented protocol
#: block. Baseline is a single call and technically needs no protocol note,
#: but recording one makes the log symmetric with the multi-agent conditions.
ARCHITECTURE_PROTOCOL_BASELINE: Final[str] = """\

--- Protocol context ---
You are operating in the BASELINE single-prompt condition. Produce the
complete SRS in one response; no other agents will contribute or review.
"""


def compose_system_prompt(role_system: str, protocol_block: str) -> str:
    """Combine a role prompt with an architecture-specific protocol block.

    Args:
        role_system: One of the ``*_SYSTEM`` constants above.
        protocol_block: One of the ``ARCHITECTURE_PROTOCOL_*`` constants.

    Returns:
        The system prompt to send to the model.
    """
    return f"{role_system.rstrip()}\n{protocol_block}"


# =========================================================================
# Baseline single-prompt agent
# =========================================================================
#
# The baseline is a single LLM call that must produce a complete SRS in one
# response. It is DELIBERATELY given a different system prompt: the four
# specialist prompts do not describe how to write a whole SRS, only how to
# write one section. The comparison the study cares about is between
# multi-agent orchestration and a strong single-prompt baseline, so this
# prompt bundles the essential guidance from all four specialists into one.

BASELINE_SYSTEM: Final[str] = f"""\
You are a senior requirements engineer. Given a natural-language system
description, you produce a complete Software Requirements Specification in
one response.

The SRS must contain, in this order and using these exact Markdown
subheadings under a top-level "## Requirements" section (assemble the
whole document to match the section order shown below):

## Overview

* One paragraph describing the system type and its scope.
* A bulleted list of stakeholders (each: role — responsibilities).
* A bulleted list of explicit assumptions you had to make.

## Functional requirements

* One paragraph per requirement, identifier first: {FR_ID_FORMAT}.
* Group under short "###" subheadings by capability area.
* Each ends with a "Rationale:" italic line.

## Non-functional requirements

* One paragraph per requirement, identifier first: {NFR_ID_FORMAT}.
* Group under short "###" subheadings by quality attribute.
* Every NFR must include a measurable criterion (a threshold, time
  bound, percentage, standard, or numeric target). If you cannot state
  one confidently, write "TBD" and add it to the Open questions
  subsection; do NOT invent a value.
* Each ends with a "Rationale:" italic line.

## Risks, ambiguities, and open questions

* Four "###" subheadings, in order: Constraints, Risks, Ambiguities,
  Open questions.
* Bulleted lists. Each risk bullet ends with "— Mitigation: <sentence>".
* Reference affected requirement identifiers (FR-NNN, NFR-NNN) in
  parentheses where relevant.

{COMMON_STYLE_RULES}

Output only the SRS Markdown. Do not add a preamble, a summary, or any
commentary outside the document.
"""


def baseline_user_prompt(description: str) -> str:
    """Assemble the baseline agent's user prompt.

    Args:
        description: The natural-language system description.

    Returns:
        The user-message body.
    """
    return (
        "Produce a complete SRS for the following system description, "
        "following the section structure and style rules in your role "
        "instructions.\n\n"
        "System description:\n"
        f'"""\n{description.strip()}\n"""\n'
    )


# =========================================================================
# Peer-to-peer architecture: peer-review prompts
# =========================================================================
#
# In peer-to-peer, each specialist first produces an initial draft
# (using the *same* initial prompt as hierarchical), then sees peers'
# drafts and produces a revised version. The role prompt stays identical;
# only the user message tells them peer drafts are attached.


def peer_review_functional_user_prompt(
    description: str,
    brief_json: str,
    own_draft: str,
    nonfunctional_draft: str,
    risk_draft: str,
) -> str:
    """Assemble the functional agent's peer-review user prompt.

    Args:
        description: The system description.
        brief_json: The orchestrator's planning brief, serialised as JSON.
        own_draft: This agent's initial functional-requirements draft.
        nonfunctional_draft: The peer non-functional draft.
        risk_draft: The peer risk-and-clarification draft.

    Returns:
        The user-message body.
    """
    return (
        "Your peers have produced initial drafts covering non-functional "
        "requirements and risks / clarifications. Read them, then produce a "
        "revised version of the functional requirements section that:\n\n"
        "* Removes or corrects any functional requirement that conflicts "
        "  with a peer's requirement or with a stated constraint.\n"
        "* Adds any functional requirement that a peer's output implies is "
        "  needed but is missing from your draft.\n"
        "* Preserves FR identifiers where the underlying requirement is "
        "  retained; renumber only when a requirement is new.\n"
        "* Records any deletion under a final '### Change log' subheading.\n\n"
        "System description:\n"
        f'"""\n{description.strip()}\n"""\n\n'
        "Orchestrator planning brief (JSON):\n"
        f"```json\n{brief_json}\n```\n\n"
        "Your initial functional requirements draft:\n"
        f"{own_draft.strip()}\n\n"
        "Peer draft — non-functional requirements:\n"
        f"{nonfunctional_draft.strip()}\n\n"
        "Peer draft — risks and clarifications:\n"
        f"{risk_draft.strip()}\n"
    )


def peer_review_nonfunctional_user_prompt(
    description: str,
    brief_json: str,
    own_draft: str,
    functional_draft: str,
    risk_draft: str,
) -> str:
    """Assemble the non-functional agent's peer-review user prompt.

    Args are the mirror image of :func:`peer_review_functional_user_prompt`.
    """
    return (
        "Your peers have produced initial drafts covering functional "
        "requirements and risks / clarifications. Read them, then produce a "
        "revised version of the non-functional requirements section that:\n\n"
        "* Aligns quality attributes with the functional capabilities the "
        "  peer draft actually specifies.\n"
        "* Adds any non-functional requirement implied by a peer-identified "
        "  constraint or risk that is missing from your draft.\n"
        "* Preserves NFR identifiers where the requirement is retained; "
        "  renumber only when a requirement is new.\n"
        "* Records any deletion under a final '### Change log' subheading.\n\n"
        "System description:\n"
        f'"""\n{description.strip()}\n"""\n\n'
        "Orchestrator planning brief (JSON):\n"
        f"```json\n{brief_json}\n```\n\n"
        "Your initial non-functional requirements draft:\n"
        f"{own_draft.strip()}\n\n"
        "Peer draft — functional requirements:\n"
        f"{functional_draft.strip()}\n\n"
        "Peer draft — risks and clarifications:\n"
        f"{risk_draft.strip()}\n"
    )


def peer_review_risk_user_prompt(
    description: str,
    brief_json: str,
    own_draft: str,
    functional_draft: str,
    nonfunctional_draft: str,
) -> str:
    """Assemble the risk agent's peer-review user prompt."""
    return (
        "Your peers have produced initial drafts of the functional and "
        "non-functional requirement sections. Read them, then produce a "
        "revised version of the risk / clarification section that:\n\n"
        "* Traces every ambiguity and open question to specific FR/NFR "
        "  identifiers where applicable.\n"
        "* Adds any risk or open question implied by the peer drafts that "
        "  is missing from your section.\n"
        "* Removes any risk or ambiguity that a peer draft has now "
        "  resolved.\n\n"
        "System description:\n"
        f'"""\n{description.strip()}\n"""\n\n'
        "Orchestrator planning brief (JSON):\n"
        f"```json\n{brief_json}\n```\n\n"
        "Your initial risk section:\n"
        f"{own_draft.strip()}\n\n"
        "Peer draft — functional requirements:\n"
        f"{functional_draft.strip()}\n\n"
        "Peer draft — non-functional requirements:\n"
        f"{nonfunctional_draft.strip()}\n"
    )


# =========================================================================
# Debate architecture: propose / rebut / arbitrate prompts
# =========================================================================
#
# Two independent invocations of each specialist produce competing positions
# (variance comes from LLM sampling; the prompt is otherwise identical to
# hierarchical `initial`). Each position then sees the other and produces a
# rebuttal / defence. The Verification Agent arbitrates: for each section it
# is given both final positions and produces the section that will appear in
# the final SRS, plus a JSON rationale for the choice.


def debate_rebuttal_functional_user_prompt(
    description: str,
    brief_json: str,
    own_position: str,
    other_position: str,
) -> str:
    """Assemble the functional agent's debate-rebuttal user prompt.

    Args:
        description: The system description.
        brief_json: The orchestrator's planning brief, serialised as JSON.
        own_position: The functional-requirements position this agent
            proposed in the first debate round.
        other_position: The competing functional-requirements position
            from the peer instance.

    Returns:
        The user-message body.
    """
    return (
        "You are one of two independently-invoked instances of the "
        "functional requirements agent in a structured debate. Below are "
        "your initial position and your peer's competing position. "
        "Produce a revised version of your position that:\n\n"
        "* Retains every requirement of yours that you can still defend on "
        "  the evidence.\n"
        "* Adopts any requirement from your peer that you now judge to be "
        "  clearly stronger, and explain in a '### Debate notes' subheading "
        "  why you adopted it.\n"
        "* Argues briefly, under '### Debate notes', against each of your "
        "  peer's requirements that you reject, saying why.\n"
        "* Preserves FR identifiers of your requirements that survive.\n\n"
        "Do NOT concede simply to reach agreement. If your position is "
        "sound, defend it.\n\n"
        "System description:\n"
        f'"""\n{description.strip()}\n"""\n\n'
        "Orchestrator planning brief (JSON):\n"
        f"```json\n{brief_json}\n```\n\n"
        "Your initial position:\n"
        f"{own_position.strip()}\n\n"
        "Peer's competing position:\n"
        f"{other_position.strip()}\n"
    )


def debate_rebuttal_nonfunctional_user_prompt(
    description: str,
    brief_json: str,
    own_position: str,
    other_position: str,
) -> str:
    """Assemble the non-functional agent's debate-rebuttal user prompt."""
    return (
        "You are one of two independently-invoked instances of the "
        "non-functional requirements agent in a structured debate. Below "
        "are your initial position and your peer's competing position. "
        "Produce a revised version that retains requirements you can "
        "defend, adopts any that your peer clearly argues better, and "
        "argues briefly under a '### Debate notes' subheading against "
        "requirements you reject. Preserve NFR identifiers of your "
        "requirements that survive. Do NOT concede simply to reach "
        "agreement.\n\n"
        "System description:\n"
        f'"""\n{description.strip()}\n"""\n\n'
        "Orchestrator planning brief (JSON):\n"
        f"```json\n{brief_json}\n```\n\n"
        "Your initial position:\n"
        f"{own_position.strip()}\n\n"
        "Peer's competing position:\n"
        f"{other_position.strip()}\n"
    )


def debate_rebuttal_risk_user_prompt(
    description: str,
    brief_json: str,
    functional_final: str,
    nonfunctional_final: str,
    own_position: str,
    other_position: str,
) -> str:
    """Assemble the risk agent's debate-rebuttal user prompt.

    The risk agent's debate has more inputs because its outputs typically
    depend on the FR/NFR sections. Both final specialist positions are
    provided.
    """
    return (
        "You are one of two independently-invoked instances of the risk / "
        "clarification agent in a structured debate. Below are your initial "
        "position and your peer's competing position, together with the "
        "final functional and non-functional requirement sections that "
        "resulted from their debates. Produce a revised risk / "
        "clarification section that retains items you can defend, adopts "
        "any your peer argues better, and argues briefly under a "
        "'### Debate notes' subheading against items you reject.\n\n"
        "System description:\n"
        f'"""\n{description.strip()}\n"""\n\n'
        "Orchestrator planning brief (JSON):\n"
        f"```json\n{brief_json}\n```\n\n"
        "Final functional requirements:\n"
        f"{functional_final.strip()}\n\n"
        "Final non-functional requirements:\n"
        f"{nonfunctional_final.strip()}\n\n"
        "Your initial position:\n"
        f"{own_position.strip()}\n\n"
        "Peer's competing position:\n"
        f"{other_position.strip()}\n"
    )


#: JSON schema for the arbiter's section-level ruling.
DEBATE_ARBITRATION_SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["chosen_section_markdown", "verdict", "rationale"],
    "properties": {
        "chosen_section_markdown": {
            "type": "string",
            "description": (
                "The section content that will appear in the final SRS. "
                "May be a straight selection of one position, or a "
                "synthesis of the two."
            ),
        },
        "verdict": {
            "type": "string",
            "enum": ["position_a", "position_b", "synthesis"],
        },
        "rationale": {
            "type": "string",
            "description": "One or two sentences explaining the ruling.",
        },
    },
}


def debate_arbitration_user_prompt(
    section_label: str,
    description: str,
    brief_json: str,
    position_a: str,
    position_b: str,
) -> str:
    """Assemble the verification agent's arbitration user prompt.

    Args:
        section_label: One of ``"functional"``, ``"nonfunctional"``,
            ``"risk"`` — the section under arbitration.
        description: The system description.
        brief_json: The orchestrator's planning brief, serialised as JSON.
        position_a: The final position from debate instance A.
        position_b: The final position from debate instance B.

    Returns:
        The user-message body.
    """
    return (
        f"You are arbitrating the {section_label} section of a debate "
        "between two independently-invoked specialist instances. Read "
        "their final positions, then decide the section that will appear "
        "in the SRS.\n\n"
        "You may:\n"
        "* Choose one position outright ('position_a' or 'position_b').\n"
        "* Synthesise a new section that combines the strongest items "
        "  from both ('synthesis').\n\n"
        "Return only the JSON object described in your schema.\n\n"
        "System description:\n"
        f'"""\n{description.strip()}\n"""\n\n'
        "Orchestrator planning brief (JSON):\n"
        f"```json\n{brief_json}\n```\n\n"
        "Position A (final):\n"
        f"{position_a.strip()}\n\n"
        "Position B (final):\n"
        f"{position_b.strip()}\n"
    )


# =========================================================================
# Evaluation rubric (LLM-as-judge)
# =========================================================================
#
# The evaluator is a fresh role — deliberately separated from the specialist
# agents so its judgement is not primed by their prompts. It scores the SRS
# on four dimensions drawn from the dissertation brief. All four are
# expressed so that 5 = best, 1 = worst; the "ambiguity" dimension is
# renamed "clarity" (its inverse) to preserve the invariant while remaining
# faithful to the brief's intent. See README §"Evaluation".

EVALUATOR_SYSTEM: Final[str] = """\
You are an experienced requirements engineer acting as an independent
reviewer. You do NOT know how the SRS you are about to read was
produced (which agent architecture, which prompts, how many revisions).
Judge only what is on the page.

You score the SRS on four dimensions, each from 1 (worst) to 5 (best):

* completeness — every capability and stakeholder implied by the input
  description is covered by at least one functional requirement; every
  quality attribute stated or reasonably implied is covered by at least
  one measurable non-functional requirement; risks and open questions
  are reasonably enumerated.
* consistency — no two requirements contradict, duplicate, or subsume
  each other; identifiers are unique; the risk section is consistent
  with the requirements it references.
* testability — every requirement admits a concrete acceptance criterion
  or measurement; non-functional requirements have measurable thresholds
  or a defensible "TBD" surfaced as an open question rather than an
  invented value.
* clarity — the SRS is unambiguous. A reader can determine exactly one
  interpretation of each requirement. (This is the inverse of the
  "ambiguity" dimension in the project brief: 5 = perfectly clear, 1 =
  pervasively ambiguous.)

For each dimension, produce:

* a numeric score (integer, 1-5),
* a one-or-two-sentence justification citing specific evidence from
  the SRS (mention requirement identifiers where relevant),
* up to three "issues" — short strings naming specific defects, empty
  list if none.

Also produce a short overall assessment paragraph. Do NOT produce a
composite score; the study computes aggregates externally.

Return ONLY the JSON object matching your schema.
"""

#: JSON schema for the evaluator's verdict.
EVALUATION_SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["completeness", "consistency", "testability", "clarity", "overall"],
    "properties": {
        dim: {
            "type": "object",
            "additionalProperties": False,
            "required": ["score", "justification", "issues"],
            "properties": {
                "score": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
                "justification": {"type": "string"},
                "issues": {"type": "array", "items": {"type": "string"}},
            },
        }
        for dim in ("completeness", "consistency", "testability", "clarity")
    }
    | {"overall": {"type": "string"}},
}


def evaluator_user_prompt(description: str, srs_markdown: str) -> str:
    """Assemble the evaluator's user prompt.

    Args:
        description: The original natural-language system description.
        srs_markdown: The SRS to be scored.

    Returns:
        The user-message body.
    """
    return (
        "Score the SRS below against the rubric in your role instructions. "
        "Return the JSON object described in your schema.\n\n"
        "Original system description (the requirements should trace back "
        "to this):\n"
        f'"""\n{description.strip()}\n"""\n\n'
        "SRS to score:\n"
        f"{srs_markdown.strip()}\n"
    )


#: Ordered tuple of the rubric dimension names, so downstream aggregation
#: code has a single source of truth.
EVALUATION_DIMENSIONS: Final[tuple[str, ...]] = (
    "completeness",
    "consistency",
    "testability",
    "clarity",
)
