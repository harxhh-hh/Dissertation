"""Layer 2 of the grounded scoring stack: a constrained checklist grader.

This is the ONLY place an LLM touches the grounded-scoring pipeline, and it
is deliberately boxed in: the agent below never ranks, never compares
architectures, and never produces a holistic verdict. It answers atomic
yes/partial/no questions about individual facts and individual
requirements (see ``GROUNDED_GRADER_SYSTEM`` in ``config/prompts.py``).
The actual scoring — coverage %, faithfulness %, composite, winner — is
pure arithmetic in :mod:`evaluation.scoring`, over the structured judgments
this module returns.

Follows the same construction pattern as ``evaluation.rubric.EvaluatorAgent``:
a fresh :class:`~agents.base.Agent` subclass with its own role prompt, no
knowledge of which architecture produced the SRS it is grading.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from agents.base import Agent, RunContext
from config import prompts
from evaluation.grounding_schema import DomainKB, KBFact, RequirementRecord
from utils.llm_client import LLMClient


class GroundedGraderAgent(Agent):
    """Constrained fact-checklist grader — atomic judgments only."""

    role_name = "grounded_grader"
    role_system_prompt = prompts.GROUNDED_GRADER_SYSTEM

    def grade(
        self, requirements: list[RequirementRecord], facts: list[KBFact],
    ) -> dict[str, Any]:
        """Grade one SRS's requirements against one domain's KB facts.

        Args:
            requirements: Requirements parsed out of the SRS under review.
            facts: The domain's gold facts.

        Returns:
            The parsed verdict, matching
            :data:`~config.prompts.GROUNDED_GRADER_SCHEMA`.
        """
        requirements_json = json.dumps(
            [{"id": r.id, "type": r.type, "section": r.section, "statement": r.statement}
             for r in requirements],
            indent=2,
        )
        kb_facts_json = json.dumps(
            [{"id": f.id, "section": f.section, "category": f.category, "statement": f.statement}
             for f in facts],
            indent=2,
        )
        result = self._call(
            phase="grade",
            user_prompt=prompts.grounded_grader_user_prompt(requirements_json, kb_facts_json),
            output_schema=prompts.GROUNDED_GRADER_SCHEMA,
        )
        assert isinstance(result.parsed_json, dict), (
            "Grounded grader was requested to return a JSON object but the "
            "parsed response was not a dict; this is a logic error."
        )
        return result.parsed_json


@dataclass
class FactCoverageResult:
    fact_id: str
    present: str  # "yes" | "partial" | "no"
    evidence: str = ""


@dataclass
class RequirementFinding:
    requirement_id: str
    contradicts_kb: bool
    evidence: str = ""


@dataclass
class GroundedGradeResult:
    """Structured output of one grounded-grading pass over one SRS."""

    case_id: str
    architecture: str
    repetition: int
    domain: str
    fact_coverage: list[FactCoverageResult] = field(default_factory=list)
    requirement_findings: list[RequirementFinding] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "architecture": self.architecture,
            "repetition": self.repetition,
            "domain": self.domain,
            "fact_coverage": [
                {"fact_id": f.fact_id, "present": f.present, "evidence": f.evidence}
                for f in self.fact_coverage
            ],
            "requirement_findings": [
                {
                    "requirement_id": r.requirement_id,
                    "contradicts_kb": r.contradicts_kb,
                    "evidence": r.evidence,
                }
                for r in self.requirement_findings
            ],
        }


def grade_grounding(
    *,
    case_id: str,
    architecture: str,
    repetition: int,
    requirements: list[RequirementRecord],
    domain_kb: DomainKB,
    client: LLMClient,
) -> GroundedGradeResult:
    """Run the grounded grader on one SRS and return a structured result.

    Missing ids in the model's response (it was asked to cover every fact
    and every requirement, but is not guaranteed to) are backfilled as
    ``"no"`` / ``contradicts_kb=False`` with an explanatory evidence
    string, rather than silently dropped — a missing judgment must not
    quietly inflate a coverage or faithfulness score.

    Args:
        case_id: Test case identifier.
        architecture: Condition that produced the SRS being graded.
        repetition: Zero-based repetition index.
        requirements: Requirements parsed out of the SRS.
        domain_kb: The domain's knowledge base (any status — callers
            decide whether a ``"draft"`` KB is acceptable to grade
            against; this function does not gate on status).
        client: The shared LLM client.

    Returns:
        A populated :class:`GroundedGradeResult`.
    """
    context = RunContext(
        architecture="evaluation",
        case_id=case_id,
        repetition=repetition,
        protocol_block=prompts.ARCHITECTURE_PROTOCOL_BASELINE,
    )
    grader = GroundedGraderAgent(client, context)
    verdict = grader.grade(requirements, domain_kb.facts)

    fact_ids = {f.id for f in domain_kb.facts}
    req_ids = {r.id for r in requirements}

    coverage_by_id: dict[str, FactCoverageResult] = {}
    for row in verdict.get("fact_coverage", []) or []:
        fid = str(row.get("fact_id", ""))
        if fid not in fact_ids:
            continue
        coverage_by_id[fid] = FactCoverageResult(
            fact_id=fid,
            present=str(row.get("present", "no")),
            evidence=str(row.get("evidence", "")),
        )
    for fid in fact_ids - coverage_by_id.keys():
        coverage_by_id[fid] = FactCoverageResult(
            fact_id=fid, present="no", evidence="(grader returned no judgment for this fact)",
        )

    findings_by_id: dict[str, RequirementFinding] = {}
    for row in verdict.get("requirement_findings", []) or []:
        rid = str(row.get("requirement_id", ""))
        if rid not in req_ids:
            continue
        findings_by_id[rid] = RequirementFinding(
            requirement_id=rid,
            contradicts_kb=bool(row.get("contradicts_kb", False)),
            evidence=str(row.get("evidence", "")),
        )
    for rid in req_ids - findings_by_id.keys():
        findings_by_id[rid] = RequirementFinding(
            requirement_id=rid, contradicts_kb=False,
            evidence="(grader returned no judgment for this requirement)",
        )

    return GroundedGradeResult(
        case_id=case_id, architecture=architecture, repetition=repetition,
        domain=domain_kb.domain,
        fact_coverage=[coverage_by_id[fid] for fid in sorted(coverage_by_id)],
        requirement_findings=[findings_by_id[rid] for rid in sorted(findings_by_id)],
    )
