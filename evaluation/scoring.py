"""Layer 3 of the grounded scoring stack: deterministic aggregation.

No LLM call anywhere in this module. Every number here is arithmetic over
the structured outputs of Layer 1 (:mod:`evaluation.linter`) and Layer 2
(:mod:`evaluation.grounded_grader`) — coverage, faithfulness and quality are
weighted and summed in Python, and so is the ranking. The winner is a
``sorted()`` call, not a model's opinion.

Three scores per (architecture, case, repetition):

* **coverage** — weighted fraction of the domain's KB facts the SRS
  addresses (``yes`` = 1, ``partial`` = 0.5, ``no`` = 0), weighted by
  :attr:`~evaluation.grounding_schema.KBFact.weight`.
* **faithfulness** — fraction of requirements that do NOT contradict a KB
  fact. This is the direct, measured inverse of hallucination.
* **quality** — the Layer-1 linter's domain-independent quality score
  (ambiguity, testability, structure, consistency).

``composite`` is their weighted sum; see :data:`DEFAULT_WEIGHTS`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from evaluation.grounded_grader import GroundedGradeResult
from evaluation.grounding_schema import DomainKB
from evaluation.linter import LintResult

#: Composite weights. Grounding (coverage + faithfulness) is weighted more
#: heavily than generic document quality on purpose — the whole point of
#: this pipeline is to rank architectures on how well-grounded their output
#: is, not to re-derive the pre-existing LLM-as-judge rubric.
DEFAULT_WEIGHTS: dict[str, float] = {
    "coverage": 0.40,
    "faithfulness": 0.35,
    "quality": 0.25,
}

_PRESENT_VALUE: dict[str, float] = {"yes": 1.0, "partial": 0.5, "no": 0.0}


@dataclass
class ArchitectureScore:
    """One (architecture, case, repetition)'s grounded score."""

    architecture: str
    case_id: str
    repetition: int
    domain: str
    coverage: float
    faithfulness: float
    quality: float
    kb_status: str
    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))

    @property
    def composite(self) -> float:
        w = self.weights
        return (
            w["coverage"] * self.coverage
            + w["faithfulness"] * self.faithfulness
            + w["quality"] * self.quality
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "architecture": self.architecture,
            "case_id": self.case_id,
            "repetition": self.repetition,
            "domain": self.domain,
            "coverage": round(self.coverage, 4),
            "faithfulness": round(self.faithfulness, 4),
            "quality": round(self.quality, 4),
            "composite": round(self.composite, 4),
            "kb_status": self.kb_status,
            "weights": dict(self.weights),
        }


def compute_coverage(grade: GroundedGradeResult, domain_kb: DomainKB) -> float:
    """Weighted fraction of KB facts the SRS addresses, in ``[0, 1]``."""
    weight_by_fact = {f.id: f.weight for f in domain_kb.facts}
    total_weight = sum(weight_by_fact.values())
    if total_weight <= 0:
        return 0.0
    earned = sum(
        weight_by_fact.get(row.fact_id, 0.0) * _PRESENT_VALUE.get(row.present, 0.0)
        for row in grade.fact_coverage
    )
    return earned / total_weight


def compute_faithfulness(grade: GroundedGradeResult) -> float:
    """Fraction of requirements that do not contradict the KB, in ``[0, 1]``.

    A document with no parsed requirements at all has nothing to be
    unfaithful about, so it scores 1.0 here — the linter's structure
    check is what penalises a document with no extractable requirements,
    not this score.
    """
    total = len(grade.requirement_findings)
    if total == 0:
        return 1.0
    contradictions = sum(1 for r in grade.requirement_findings if r.contradicts_kb)
    return 1.0 - (contradictions / total)


def compute_architecture_score(
    *,
    architecture: str,
    case_id: str,
    repetition: int,
    domain_kb: DomainKB,
    lint_result: LintResult,
    grade_result: GroundedGradeResult,
    weights: dict[str, float] | None = None,
) -> ArchitectureScore:
    """Combine Layer 1 + Layer 2 outputs into one deterministic score."""
    return ArchitectureScore(
        architecture=architecture,
        case_id=case_id,
        repetition=repetition,
        domain=domain_kb.domain,
        coverage=compute_coverage(grade_result, domain_kb),
        faithfulness=compute_faithfulness(grade_result),
        quality=lint_result.quality_score,
        kb_status=domain_kb.status,
        weights=dict(weights or DEFAULT_WEIGHTS),
    )


@dataclass
class CaseWinner:
    """The ranked scoreboard for one test case, winner first."""

    case_id: str
    scores: list[ArchitectureScore]

    @property
    def winner(self) -> ArchitectureScore | None:
        return self.scores[0] if self.scores else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "winner": self.winner.architecture if self.winner else None,
            "scores": [s.to_dict() for s in self.scores],
        }


def rank_case(scores: list[ArchitectureScore]) -> CaseWinner:
    """Sort one case's architecture scores, best first.

    Ties broken alphabetically by architecture name so the ranking is
    reproducible from the same inputs every time.
    """
    ranked = sorted(scores, key=lambda s: (-s.composite, s.architecture))
    return CaseWinner(case_id=scores[0].case_id if scores else "", scores=ranked)


@dataclass
class ArchitectureSummary:
    """One architecture's aggregate performance across every case scored."""

    architecture: str
    mean_composite: float
    mean_coverage: float
    mean_faithfulness: float
    mean_quality: float
    cases_won: int
    cases_scored: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "architecture": self.architecture,
            "mean_composite": round(self.mean_composite, 4),
            "mean_coverage": round(self.mean_coverage, 4),
            "mean_faithfulness": round(self.mean_faithfulness, 4),
            "mean_quality": round(self.mean_quality, 4),
            "cases_won": self.cases_won,
            "cases_scored": self.cases_scored,
        }


@dataclass
class OverallResult:
    """The ranked scoreboard across every case in a run, winner first."""

    case_winners: list[CaseWinner]
    summaries: list[ArchitectureSummary]

    @property
    def winner(self) -> ArchitectureSummary | None:
        return self.summaries[0] if self.summaries else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "winner": self.winner.architecture if self.winner else None,
            "summaries": [s.to_dict() for s in self.summaries],
            "case_winners": [c.to_dict() for c in self.case_winners],
        }


def compute_overall(all_scores: list[ArchitectureScore]) -> OverallResult:
    """Rank every case, then rank architectures overall by mean composite.

    Args:
        all_scores: Every :class:`ArchitectureScore` produced for a run
            (all architectures, all cases, all repetitions).

    Returns:
        A populated :class:`OverallResult`. Empty input yields an empty
        result rather than raising.
    """
    if not all_scores:
        return OverallResult(case_winners=[], summaries=[])

    by_case: dict[str, list[ArchitectureScore]] = {}
    for s in all_scores:
        by_case.setdefault(s.case_id, []).append(s)
    case_winners = [rank_case(scores) for scores in by_case.values()]
    case_winners.sort(key=lambda c: c.case_id)

    by_arch: dict[str, list[ArchitectureScore]] = {}
    for s in all_scores:
        by_arch.setdefault(s.architecture, []).append(s)
    wins_by_arch: dict[str, int] = {}
    for cw in case_winners:
        if cw.winner is not None:
            wins_by_arch[cw.winner.architecture] = wins_by_arch.get(cw.winner.architecture, 0) + 1

    summaries: list[ArchitectureSummary] = []
    for arch, scores in by_arch.items():
        n = len(scores)
        summaries.append(ArchitectureSummary(
            architecture=arch,
            mean_composite=sum(s.composite for s in scores) / n,
            mean_coverage=sum(s.coverage for s in scores) / n,
            mean_faithfulness=sum(s.faithfulness for s in scores) / n,
            mean_quality=sum(s.quality for s in scores) / n,
            cases_won=wins_by_arch.get(arch, 0),
            cases_scored=n,
        ))
    summaries.sort(key=lambda s: (-s.mean_composite, s.architecture))

    return OverallResult(case_winners=case_winners, summaries=summaries)
