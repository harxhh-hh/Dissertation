"""Peer-to-peer architecture: specialists review each other before verification.

Protocol:

1. The orchestrator produces a planning brief.
2. The three specialists (functional, non-functional, risk) produce their
   initial drafts *in the same way as in hierarchical* (this is the
   controlled-comparison invariant: the ``initial`` phase user prompts are
   byte-identical between hierarchical and peer-to-peer).
3. Peer-review pass: each specialist sees the other two specialists'
   drafts and produces a revised version of its own section. The role
   system prompt is unchanged; only the user message tells the agent that
   peer drafts are attached.
4. The three revised sections are assembled into a draft SRS.
5. The verification agent assesses it. If ``revision_required`` and the
   revision budget has not been exhausted, targeted revisions are dispatched
   (same routing as hierarchical), the SRS is reassembled and reassessed.

Every LLM call goes through the same shared client and lands in the
interaction log with ``architecture="peer_to_peer"``, an agent role
identical to the hierarchical case, and a phase that distinguishes
``initial``, ``peer_review``, ``revision``, and
``verification_round_<n>``.
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
from architectures.hierarchical import (
    VerificationRound,
    _issues_for_section,
    _round_from_verdict,
    format_srs_markdown,
)
from config import prompts
from config.settings import Settings
from utils.llm_client import LLMClient
from utils.logging import ExperimentLogger

#: Canonical architecture name used in interaction log lines.
ARCHITECTURE_NAME: str = "peer_to_peer"


@dataclass
class PeerToPeerResult:
    """Everything produced by one peer-to-peer run of one test case.

    Attributes:
        case_id: The test case identifier.
        repetition: Zero-based repetition index.
        brief: The orchestrator's planning brief.
        functional_markdown: The final functional-requirements section.
        nonfunctional_markdown: The final non-functional section.
        risk_markdown: The final risk-and-clarification section.
        verification_rounds: One entry per verification pass, in order.
        srs_markdown: The final assembled SRS document.
        revised: ``True`` if at least one revision round ran after
            verification (independent of the peer-review pass, which
            always runs).
    """

    case_id: str
    repetition: int
    brief: dict[str, Any]
    functional_markdown: str
    nonfunctional_markdown: str
    risk_markdown: str
    verification_rounds: list[VerificationRound]
    srs_markdown: str
    revised: bool = field(default=False)


def run_peer_to_peer(
    description: str,
    *,
    case_id: str,
    repetition: int,
    client: LLMClient,
    settings: Settings,
    logger: ExperimentLogger,
) -> PeerToPeerResult:
    """Run the peer-to-peer architecture end-to-end on one test case."""
    context = RunContext(
        architecture=ARCHITECTURE_NAME,
        case_id=case_id,
        repetition=repetition,
        protocol_block=prompts.ARCHITECTURE_PROTOCOL_PEER_TO_PEER,
    )
    orchestrator = OrchestratorAgent(client, context)
    functional = FunctionalRequirementsAgent(client, context)
    nonfunctional = NonFunctionalRequirementsAgent(client, context)
    risk = RiskClarificationAgent(client, context)
    verification = VerificationAgent(client, context)

    logger.info("[peer_to_peer/%s rep=%d] orchestrator planning", case_id, repetition)
    brief = orchestrator.plan(description)
    brief_json = json.dumps(brief, indent=2, sort_keys=True)

    # ---- Initial parallel-independent drafts (same as hierarchical) -----
    logger.info("[peer_to_peer/%s rep=%d] initial FR draft", case_id, repetition)
    functional_md = functional.generate(description, brief)

    logger.info("[peer_to_peer/%s rep=%d] initial NFR draft", case_id, repetition)
    nonfunctional_md = nonfunctional.generate(description, brief)

    logger.info("[peer_to_peer/%s rep=%d] initial risk draft", case_id, repetition)
    risk_md = risk.generate(description, brief, functional_md, nonfunctional_md)

    # ---- Peer-review pass -----------------------------------------------
    # Each specialist sees the drafts the other two produced. We snapshot
    # the initial versions before revising any of them, so the peer-review
    # step is symmetric: each agent is critiquing exactly one set of peer
    # drafts, not a moving target.
    initial_functional_md = functional_md
    initial_nonfunctional_md = nonfunctional_md
    initial_risk_md = risk_md

    logger.info("[peer_to_peer/%s rep=%d] peer-review FR", case_id, repetition)
    functional_md = _peer_review_functional(
        functional,
        description=description,
        brief_json=brief_json,
        own_draft=initial_functional_md,
        nonfunctional_draft=initial_nonfunctional_md,
        risk_draft=initial_risk_md,
    )
    logger.info("[peer_to_peer/%s rep=%d] peer-review NFR", case_id, repetition)
    nonfunctional_md = _peer_review_nonfunctional(
        nonfunctional,
        description=description,
        brief_json=brief_json,
        own_draft=initial_nonfunctional_md,
        functional_draft=initial_functional_md,
        risk_draft=initial_risk_md,
    )
    logger.info("[peer_to_peer/%s rep=%d] peer-review risk", case_id, repetition)
    risk_md = _peer_review_risk(
        risk,
        description=description,
        brief_json=brief_json,
        own_draft=initial_risk_md,
        functional_draft=initial_functional_md,
        nonfunctional_draft=initial_nonfunctional_md,
    )

    # ---- Verification (and optional targeted revision) ------------------
    verification_rounds: list[VerificationRound] = []
    round_index = 1
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
    logger.info("[peer_to_peer/%s rep=%d] verification round %d", case_id, repetition, round_index)
    verdict = verification.assess(draft_md, phase=f"verification_round_{round_index}")
    verification_rounds.append(_round_from_verdict(round_index, verdict))
    logger.info(
        "[peer_to_peer/%s rep=%d] round %d verdict=%s issues=%d",
        case_id, repetition, round_index, verdict["verdict"], len(verdict.get("issues", [])),
    )

    revised = False
    while (
        verdict["verdict"] == "revision_required"
        and round_index <= settings.max_revision_rounds
    ):
        fr_issues = _issues_for_section(verdict, "functional")
        nfr_issues = _issues_for_section(verdict, "nonfunctional")
        risk_issues = _issues_for_section(verdict, "risk")

        if fr_issues:
            functional_md = functional.revise(description, brief, functional_md, fr_issues)
        if nfr_issues:
            nonfunctional_md = nonfunctional.revise(
                description, brief, nonfunctional_md, nfr_issues
            )
        if risk_issues:
            risk_md = risk.revise(
                description, brief, functional_md, nonfunctional_md, risk_md, risk_issues
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
        logger.info("[peer_to_peer/%s rep=%d] verification round %d", case_id, repetition, round_index)
        verdict = verification.assess(draft_md, phase=f"verification_round_{round_index}")
        verification_rounds.append(_round_from_verdict(round_index, verdict))

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
    return PeerToPeerResult(
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


# --------------------------------------------------------------------------
# Peer-review helpers
# --------------------------------------------------------------------------
# Each of these issues one LLM call attributed to the correct agent with a
# fresh phase label ``peer_review``, so downstream analysis can separate the
# initial-draft cost from the peer-review cost.


def _peer_review_functional(
    agent: FunctionalRequirementsAgent,
    *,
    description: str,
    brief_json: str,
    own_draft: str,
    nonfunctional_draft: str,
    risk_draft: str,
) -> str:
    """Ask the functional agent to revise in light of peer drafts."""
    return agent._call(  # noqa: SLF001 — same package, deliberate reach-in
        phase="peer_review",
        user_prompt=prompts.peer_review_functional_user_prompt(
            description, brief_json, own_draft, nonfunctional_draft, risk_draft
        ),
    ).text


def _peer_review_nonfunctional(
    agent: NonFunctionalRequirementsAgent,
    *,
    description: str,
    brief_json: str,
    own_draft: str,
    functional_draft: str,
    risk_draft: str,
) -> str:
    """Ask the non-functional agent to revise in light of peer drafts."""
    return agent._call(  # noqa: SLF001
        phase="peer_review",
        user_prompt=prompts.peer_review_nonfunctional_user_prompt(
            description, brief_json, own_draft, functional_draft, risk_draft
        ),
    ).text


def _peer_review_risk(
    agent: RiskClarificationAgent,
    *,
    description: str,
    brief_json: str,
    own_draft: str,
    functional_draft: str,
    nonfunctional_draft: str,
) -> str:
    """Ask the risk agent to revise in light of peer drafts."""
    return agent._call(  # noqa: SLF001
        phase="peer_review",
        user_prompt=prompts.peer_review_risk_user_prompt(
            description, brief_json, own_draft, functional_draft, nonfunctional_draft
        ),
    ).text
