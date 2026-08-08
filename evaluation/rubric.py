"""LLM-as-judge evaluation rubric.

This module implements the automated evaluation used to compare the four
conditions of the study. It is deliberately kept simple and separate from
the specialist agents so its judgement is not primed by their prompts.

Scope and known limitations, worth stating up front:

* **The evaluator is an LLM.** LLM-as-judge is a defensible approximation
  under time pressure, but it is not a substitute for human raters. The
  brief calls for blind human rating, and :mod:`evaluation.export_for_raters`
  supports that workflow. LLM scores from this module are best treated as
  a preliminary signal that lets you compare architectures quickly; the
  final dissertation results should combine or replace them with human
  scores.
* **Same model, same provider.** By default the evaluator uses the same
  ``MODEL_ID`` as the generators. A model may prefer its own outputs;
  using a different family for evaluation would strengthen the design.
  See README §Evaluation for how to swap the evaluator model out.

Four dimensions, all scored 1 (worst) to 5 (best):

* completeness
* consistency
* testability
* clarity — the inverse of the "ambiguity" dimension in the project
  brief; renamed so higher is always better.

Every evaluator call is logged like any other LLM interaction, with
``architecture="evaluation"``, ``agent="evaluator"``, and a phase that
identifies which artefact was scored.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agents.base import Agent, RunContext
from config import prompts
from config.settings import Settings
from utils.llm_client import LLMClient
from utils.logging import ExperimentLogger

#: Fixed dimension names, mirrored from :mod:`config.prompts`.
DIMENSIONS: tuple[str, ...] = prompts.EVALUATION_DIMENSIONS

#: Filename that :func:`evaluate_run` writes to the run directory holding
#: the flat scores table.
EVALUATION_FILENAME: str = "evaluation.json"


class EvaluatorAgent(Agent):
    """LLM-as-judge evaluator.

    Deliberately not a subclass of any specialist agent; it has its own
    role prompt and no knowledge of how the SRS it is scoring was
    produced.
    """

    role_name = "evaluator"
    role_system_prompt = prompts.EVALUATOR_SYSTEM

    def score(self, description: str, srs_markdown: str) -> dict[str, Any]:
        """Score an SRS document against the rubric.

        Args:
            description: The original natural-language description; supplied
                so the evaluator can judge completeness against the input.
            srs_markdown: The full SRS Markdown to score.

        Returns:
            The parsed verdict as a dictionary matching
            :data:`~config.prompts.EVALUATION_SCHEMA`.
        """
        result = self._call(
            phase="score",
            user_prompt=prompts.evaluator_user_prompt(description, srs_markdown),
            output_schema=prompts.EVALUATION_SCHEMA,
        )
        assert isinstance(result.parsed_json, dict), (
            "Evaluator was requested to return a JSON object but the "
            "parsed response was not a dict; this is a logic error."
        )
        return result.parsed_json


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------


@dataclass
class EvaluationRecord:
    """One row of the evaluation table.

    Attributes:
        case_id: Test case identifier.
        architecture: Condition that produced the SRS.
        repetition: Zero-based repetition index.
        scores: Mapping from dimension name to integer score (1-5).
        justifications: Mapping from dimension name to justification text.
        issues: Mapping from dimension name to a list of short defect strings.
        overall: The evaluator's overall assessment paragraph.
    """

    case_id: str
    architecture: str
    repetition: int
    scores: dict[str, int] = field(default_factory=dict)
    justifications: dict[str, str] = field(default_factory=dict)
    issues: dict[str, list[str]] = field(default_factory=dict)
    overall: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary form."""
        return {
            "case_id": self.case_id,
            "architecture": self.architecture,
            "repetition": self.repetition,
            "scores": dict(self.scores),
            "justifications": dict(self.justifications),
            "issues": {k: list(v) for k, v in self.issues.items()},
            "overall": self.overall,
        }

    @classmethod
    def from_verdict(
        cls,
        *,
        case_id: str,
        architecture: str,
        repetition: int,
        verdict: dict[str, Any],
    ) -> "EvaluationRecord":
        """Build one record from an evaluator verdict.

        Args:
            case_id: Test case identifier.
            architecture: Condition that produced the SRS.
            repetition: Zero-based repetition index.
            verdict: Parsed evaluator verdict.

        Returns:
            A populated :class:`EvaluationRecord`.
        """
        record = cls(
            case_id=case_id, architecture=architecture, repetition=repetition,
            overall=str(verdict.get("overall", "")),
        )
        for dim in DIMENSIONS:
            block = verdict.get(dim, {}) or {}
            record.scores[dim] = int(block.get("score", 0))
            record.justifications[dim] = str(block.get("justification", ""))
            record.issues[dim] = [str(i) for i in (block.get("issues", []) or [])]
        return record


# --------------------------------------------------------------------------
# Entry points
# --------------------------------------------------------------------------


def score_srs(
    *,
    case_id: str,
    architecture: str,
    repetition: int,
    description: str,
    srs_markdown: str,
    client: LLMClient,
) -> EvaluationRecord:
    """Score one SRS and return an :class:`EvaluationRecord`.

    Args:
        case_id: Test case identifier.
        architecture: Condition that produced the SRS.
        repetition: Zero-based repetition index.
        description: The original natural-language description.
        srs_markdown: The SRS Markdown to score.
        client: The shared LLM client.

    Returns:
        A populated :class:`EvaluationRecord`.
    """
    # The evaluator is a fresh role — it uses a dedicated protocol block
    # (the baseline block, chosen because the evaluator itself does not
    # participate in any of the four conditions and reusing the baseline
    # block avoids introducing a fifth architecture name into the
    # interaction log). The ``architecture`` attribution on the record
    # tracks the SRS being scored, not the evaluator's own protocol.
    context = RunContext(
        architecture="evaluation",
        case_id=case_id,
        repetition=repetition,
        protocol_block=prompts.ARCHITECTURE_PROTOCOL_BASELINE,
    )
    evaluator = EvaluatorAgent(client, context)
    verdict = evaluator.score(description, srs_markdown)
    return EvaluationRecord.from_verdict(
        case_id=case_id, architecture=architecture,
        repetition=repetition, verdict=verdict,
    )


def write_evaluation(records: list[EvaluationRecord], output_path: Path) -> None:
    """Write a list of evaluation records to a JSON file.

    Args:
        records: The records to persist.
        output_path: File path to write to. Parent directories are created.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps([r.to_dict() for r in records], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


# --------------------------------------------------------------------------
# Simple aggregation helpers used by the report generator
# --------------------------------------------------------------------------


def aggregate_by_architecture(
    records: list[EvaluationRecord],
) -> dict[str, dict[str, float]]:
    """Return per-architecture mean scores, one entry per dimension.

    Args:
        records: All evaluation records for the run.

    Returns:
        Nested mapping ``{architecture: {dimension: mean_score}}``.
        Architectures that have no scored SRS are absent from the mapping.
    """
    by_arch: dict[str, dict[str, list[int]]] = {}
    for r in records:
        by_arch.setdefault(r.architecture, {})
        for dim in DIMENSIONS:
            by_arch[r.architecture].setdefault(dim, []).append(r.scores.get(dim, 0))
    means: dict[str, dict[str, float]] = {}
    for arch, dim_scores in by_arch.items():
        means[arch] = {
            dim: (sum(vs) / len(vs)) if vs else 0.0 for dim, vs in dim_scores.items()
        }
    return means
