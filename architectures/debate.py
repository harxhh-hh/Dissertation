"""Debate architecture: competing specialists, arbitrated by verification.

Protocol (per section: functional, non-functional, risk):

1. The orchestrator produces a planning brief.
2. Two independent invocations of the specialist produce initial positions.
   The user prompt is the same one hierarchical uses at ``initial``, but
   each call is tagged with a distinct phase label (``initial_A`` /
   ``initial_B``) so the interaction log distinguishes them. Variance
   between the two positions comes from LLM sampling, not from prompt
   differences — this is deliberate: if the prompts differed, the debate
   would be measuring prompt design, not architectural interaction.
3. Rebuttal round: each position is shown the other and produces a revised
   position, retaining what it can defend and adopting what the peer
   argues better. Phases: ``rebuttal_A`` and ``rebuttal_B``.
4. The verification agent arbitrates the two final positions and produces
   the section that will appear in the SRS. Phases:
   ``arbitration_functional`` / ``arbitration_nonfunctional`` /
   ``arbitration_risk``. The arbiter may pick either position outright or
   synthesise a new one; its ruling includes a short rationale.
5. The three arbitrated sections are assembled into the draft SRS.
6. A final verification pass assesses the assembled draft. Because the
   arbitration step already applied the verification agent's judgement at
   the section level, the post-assembly verification round exists to catch
   *cross-section* problems (e.g. an NFR that traces to an FR removed in a
   different section's arbitration). Optional targeted revisions are
   dispatched like hierarchical, using the specialists that survived the
   arbitration (concretely: instance A of each specialist).

Important design notes worth flagging to a moderator:

* The debate is a compute-heavy condition (~13 LLM calls minimum for one
  case at ``MAX_REVISION_ROUNDS=0``: 1 orchestrator + 6 initial + 6
  rebuttal + 3 arbitration + 1 verification). This is expected and is
  what the compute-versus-quality analysis needs to explain.
* Interaction records carry a ``phase`` field that lets an analyst compute
  cost breakdowns by protocol phase, not just by architecture.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from agents.base import Agent, RunContext
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
ARCHITECTURE_NAME: str = "debate"


@dataclass
class DebateArbitration:
    """The arbiter's ruling on one section of the debate.

    Attributes:
        section: One of ``"functional"``, ``"nonfunctional"``, ``"risk"``.
        verdict: ``"position_a"``, ``"position_b"``, or ``"synthesis"``.
        rationale: One or two sentences explaining the ruling.
        chosen_markdown: The section text that entered the SRS.
    """

    section: str
    verdict: str
    rationale: str
    chosen_markdown: str


@dataclass
class DebateResult:
    """Everything produced by one debate run of one test case."""

    case_id: str
    repetition: int
    brief: dict[str, Any]
    functional_positions: tuple[str, str]
    nonfunctional_positions: tuple[str, str]
    risk_positions: tuple[str, str]
    arbitrations: list[DebateArbitration]
    verification_rounds: list[VerificationRound]
    srs_markdown: str
    revised: bool = field(default=False)


def run_debate(
    description: str,
    *,
    case_id: str,
    repetition: int,
    client: LLMClient,
    settings: Settings,
    logger: ExperimentLogger,
) -> DebateResult:
    """Run the debate architecture end-to-end on one test case."""
    context = RunContext(
        architecture=ARCHITECTURE_NAME,
        case_id=case_id,
        repetition=repetition,
        protocol_block=prompts.ARCHITECTURE_PROTOCOL_DEBATE,
    )
    orchestrator = OrchestratorAgent(client, context)
    # We construct two independent instances of each specialist so their
    # phase labels ``_A`` / ``_B`` are attached at the call site rather than
    # threaded through argument lists. The agents are otherwise identical.
    fr_a = FunctionalRequirementsAgent(client, context)
    fr_b = FunctionalRequirementsAgent(client, context)
    nfr_a = NonFunctionalRequirementsAgent(client, context)
    nfr_b = NonFunctionalRequirementsAgent(client, context)
    risk_a = RiskClarificationAgent(client, context)
    risk_b = RiskClarificationAgent(client, context)
    verification = VerificationAgent(client, context)

    logger.info("[debate/%s rep=%d] orchestrator planning", case_id, repetition)
    brief = orchestrator.plan(description)
    brief_json = json.dumps(brief, indent=2, sort_keys=True)

    # ---- Initial competing positions ------------------------------------
    logger.info("[debate/%s rep=%d] initial FR position A", case_id, repetition)
    fr_pos_a = _call_initial(fr_a, "initial_A", description, brief)
    logger.info("[debate/%s rep=%d] initial FR position B", case_id, repetition)
    fr_pos_b = _call_initial(fr_b, "initial_B", description, brief)

    logger.info("[debate/%s rep=%d] initial NFR position A", case_id, repetition)
    nfr_pos_a = _call_initial(nfr_a, "initial_A", description, brief)
    logger.info("[debate/%s rep=%d] initial NFR position B", case_id, repetition)
    nfr_pos_b = _call_initial(nfr_b, "initial_B", description, brief)

    # The risk agent's initial prompt requires the FR/NFR sections in
    # scope. For the initial risk positions we give each instance one of
    # the FR/NFR positions (A with A, B with B), so the two risk positions
    # start from genuinely different premises.
    logger.info("[debate/%s rep=%d] initial risk position A", case_id, repetition)
    risk_pos_a = _call_initial_risk(risk_a, "initial_A", description, brief, fr_pos_a, nfr_pos_a)
    logger.info("[debate/%s rep=%d] initial risk position B", case_id, repetition)
    risk_pos_b = _call_initial_risk(risk_b, "initial_B", description, brief, fr_pos_b, nfr_pos_b)

    # ---- Rebuttal round -------------------------------------------------
    logger.info("[debate/%s rep=%d] FR rebuttal A", case_id, repetition)
    fr_final_a = _rebut_functional(fr_a, "rebuttal_A", description, brief_json, fr_pos_a, fr_pos_b)
    logger.info("[debate/%s rep=%d] FR rebuttal B", case_id, repetition)
    fr_final_b = _rebut_functional(fr_b, "rebuttal_B", description, brief_json, fr_pos_b, fr_pos_a)

    logger.info("[debate/%s rep=%d] NFR rebuttal A", case_id, repetition)
    nfr_final_a = _rebut_nonfunctional(
        nfr_a, "rebuttal_A", description, brief_json, nfr_pos_a, nfr_pos_b
    )
    logger.info("[debate/%s rep=%d] NFR rebuttal B", case_id, repetition)
    nfr_final_b = _rebut_nonfunctional(
        nfr_b, "rebuttal_B", description, brief_json, nfr_pos_b, nfr_pos_a
    )

    # ---- Arbitration: FR and NFR first, so the risk debate rebuttals see
    # the arbitrated FR/NFR text as their canonical premise ---------------
    logger.info("[debate/%s rep=%d] arbitrate FR", case_id, repetition)
    fr_arbitration = _arbitrate(
        verification, "functional", description, brief_json, fr_final_a, fr_final_b
    )
    logger.info("[debate/%s rep=%d] arbitrate NFR", case_id, repetition)
    nfr_arbitration = _arbitrate(
        verification, "nonfunctional", description, brief_json, nfr_final_a, nfr_final_b
    )

    logger.info("[debate/%s rep=%d] risk rebuttal A", case_id, repetition)
    risk_final_a = _rebut_risk(
        risk_a, "rebuttal_A", description, brief_json,
        fr_arbitration.chosen_markdown, nfr_arbitration.chosen_markdown,
        risk_pos_a, risk_pos_b,
    )
    logger.info("[debate/%s rep=%d] risk rebuttal B", case_id, repetition)
    risk_final_b = _rebut_risk(
        risk_b, "rebuttal_B", description, brief_json,
        fr_arbitration.chosen_markdown, nfr_arbitration.chosen_markdown,
        risk_pos_b, risk_pos_a,
    )

    logger.info("[debate/%s rep=%d] arbitrate risk", case_id, repetition)
    risk_arbitration = _arbitrate(
        verification, "risk", description, brief_json, risk_final_a, risk_final_b
    )
    arbitrations = [fr_arbitration, nfr_arbitration, risk_arbitration]

    # ---- Post-arbitration verification and optional revision ------------
    functional_md = fr_arbitration.chosen_markdown
    nonfunctional_md = nfr_arbitration.chosen_markdown
    risk_md = risk_arbitration.chosen_markdown

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
    logger.info(
        "[debate/%s rep=%d] post-arbitration verification round %d",
        case_id, repetition, round_index,
    )
    verdict = verification.assess(draft_md, phase=f"verification_round_{round_index}")
    verification_rounds.append(_round_from_verdict(round_index, verdict))

    revised = False
    while (
        verdict["verdict"] == "revision_required"
        and round_index <= settings.max_revision_rounds
    ):
        fr_issues = _issues_for_section(verdict, "functional")
        nfr_issues = _issues_for_section(verdict, "nonfunctional")
        risk_issues = _issues_for_section(verdict, "risk")

        # Route revisions to the "A" instance of each specialist; the
        # architecture from this point on behaves like hierarchical.
        if fr_issues:
            functional_md = fr_a.revise(description, brief, functional_md, fr_issues)
        if nfr_issues:
            nonfunctional_md = nfr_a.revise(description, brief, nonfunctional_md, nfr_issues)
        if risk_issues:
            risk_md = risk_a.revise(
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
        logger.info(
            "[debate/%s rep=%d] verification round %d", case_id, repetition, round_index
        )
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
    return DebateResult(
        case_id=case_id,
        repetition=repetition,
        brief=brief,
        functional_positions=(fr_final_a, fr_final_b),
        nonfunctional_positions=(nfr_final_a, nfr_final_b),
        risk_positions=(risk_final_a, risk_final_b),
        arbitrations=arbitrations,
        verification_rounds=verification_rounds,
        srs_markdown=final_srs,
        revised=revised,
    )


# --------------------------------------------------------------------------
# Phase-specific call helpers
# --------------------------------------------------------------------------
#
# Each helper is a one-liner in effect, but naming the phase in one place
# means the label used in the interaction log is the label spelled here.


def _call_initial(agent: Agent, phase: str, description: str, brief: dict[str, Any]) -> str:
    """Issue the initial-draft call for a specialist, under a debate phase label.

    Reuses the initial-draft user prompt (from :mod:`config.prompts`) that
    hierarchical and peer-to-peer also use, so the prompt content is
    byte-identical across the three multi-agent conditions at ``initial``.
    """
    brief_json = json.dumps(brief, indent=2, sort_keys=True)
    if isinstance(agent, FunctionalRequirementsAgent):
        user = prompts.functional_user_prompt(description, brief_json)
    elif isinstance(agent, NonFunctionalRequirementsAgent):
        user = prompts.nonfunctional_user_prompt(description, brief_json)
    else:
        # Risk uses a different helper — see _call_initial_risk.
        raise AssertionError(
            f"_call_initial does not handle {type(agent).__name__}; use _call_initial_risk"
        )
    return agent._call(phase=phase, user_prompt=user).text  # noqa: SLF001


def _call_initial_risk(
    agent: RiskClarificationAgent,
    phase: str,
    description: str,
    brief: dict[str, Any],
    functional_md: str,
    nonfunctional_md: str,
) -> str:
    """Issue the initial risk-draft call under a debate phase label."""
    brief_json = json.dumps(brief, indent=2, sort_keys=True)
    user = prompts.risk_user_prompt(description, brief_json, functional_md, nonfunctional_md)
    return agent._call(phase=phase, user_prompt=user).text  # noqa: SLF001


def _rebut_functional(
    agent: FunctionalRequirementsAgent,
    phase: str,
    description: str,
    brief_json: str,
    own_position: str,
    other_position: str,
) -> str:
    """Issue the functional rebuttal call under a debate phase label."""
    return agent._call(  # noqa: SLF001
        phase=phase,
        user_prompt=prompts.debate_rebuttal_functional_user_prompt(
            description, brief_json, own_position, other_position
        ),
    ).text


def _rebut_nonfunctional(
    agent: NonFunctionalRequirementsAgent,
    phase: str,
    description: str,
    brief_json: str,
    own_position: str,
    other_position: str,
) -> str:
    """Issue the non-functional rebuttal call under a debate phase label."""
    return agent._call(  # noqa: SLF001
        phase=phase,
        user_prompt=prompts.debate_rebuttal_nonfunctional_user_prompt(
            description, brief_json, own_position, other_position
        ),
    ).text


def _rebut_risk(
    agent: RiskClarificationAgent,
    phase: str,
    description: str,
    brief_json: str,
    functional_final: str,
    nonfunctional_final: str,
    own_position: str,
    other_position: str,
) -> str:
    """Issue the risk rebuttal call under a debate phase label."""
    return agent._call(  # noqa: SLF001
        phase=phase,
        user_prompt=prompts.debate_rebuttal_risk_user_prompt(
            description, brief_json, functional_final, nonfunctional_final,
            own_position, other_position,
        ),
    ).text


def _arbitrate(
    verification: VerificationAgent,
    section: str,
    description: str,
    brief_json: str,
    position_a: str,
    position_b: str,
) -> DebateArbitration:
    """Ask the verification agent to arbitrate one section of the debate.

    Returns a populated :class:`DebateArbitration` including the section
    text that will appear in the SRS.
    """
    phase = f"arbitration_{section}"
    result = verification._call(  # noqa: SLF001
        phase=phase,
        user_prompt=prompts.debate_arbitration_user_prompt(
            section, description, brief_json, position_a, position_b
        ),
        output_schema=prompts.DEBATE_ARBITRATION_SCHEMA,
    )
    assert isinstance(result.parsed_json, dict), (
        "Arbitration was requested to return a JSON object but the parsed "
        "response was not a dict; this is a logic error."
    )
    return DebateArbitration(
        section=section,
        verdict=result.parsed_json["verdict"],
        rationale=result.parsed_json.get("rationale", ""),
        chosen_markdown=result.parsed_json["chosen_section_markdown"],
    )
