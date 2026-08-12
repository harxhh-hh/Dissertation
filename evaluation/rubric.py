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
from evaluation.linter import LintIssue, LintResult, lint_srs
from evaluation.srs_parser import parse_requirements
from utils.llm_client import LLMClient
from utils.logging import ExperimentLogger

#: Fixed dimension names, mirrored from :mod:`config.prompts`.
DIMENSIONS: tuple[str, ...] = prompts.EVALUATION_DIMENSIONS

#: Filename that :func:`evaluate_run` writes to the run directory holding
#: the flat scores table.
EVALUATION_FILENAME: str = "evaluation.json"

#: Enum words from *other* structured-output schemas in this prompt
#: library (the verification verdict, the debate arbiter's verdict).
#: Smaller local models occasionally echo a field/enum value from a
#: different part of their context into the wrong field instead of
#: generating real content - the same failure mode already observed and
#: handled in ``architectures/debate.py:_arbitrate()``. If "overall"
#: comes back as one of these (or is otherwise too short to be a real
#: summary sentence), the whole verdict is suspect: every case seen so
#: far also had a numeric score inconsistent with its own justification
#: text (e.g. score=1 next to a strongly positive justification).
_DEGENERATE_OVERALL_VALUES = {"pass", "revision_required", "synthesis", "position_a", "position_b"}
_MIN_OVERALL_LENGTH = 15

#: Attempts (including the first) before giving up and falling back to a
#: linter-derived proxy for a dimension the LLM kept returning a
#: corrupted verdict for.
_MAX_SCORE_ATTEMPTS = 3

#: Deterministic proxy used only once every retry has been exhausted.
#: "clarity" is documented elsewhere in this module as the literal
#: inverse of ambiguity, so ambiguity_score is an exact substitute, not
#: an approximation. "completeness" has no exact mechanical analogue -
#: structure_score (are the required sections present and non-empty) is
#: used as a clearly-labelled, degraded proxy rather than trusting a
#: score the model's own justification contradicts.
_LLM_DIMENSION_LINT_PROXY = {"clarity": "ambiguity", "completeness": "structure"}


def _verdict_looks_corrupted(verdict: dict[str, Any]) -> bool:
    """Detect the cross-schema field-echo failure mode in an evaluator verdict."""
    overall = str(verdict.get("overall", "")).strip().lower()
    return overall in _DEGENERATE_OVERALL_VALUES or len(overall) < _MIN_OVERALL_LENGTH


def _dimension_from_lint(
    score_0_1: float, lint_issues: list[LintIssue], category: str,
) -> tuple[int, str, list[str]]:
    """Convert one Layer-1 lint sub-score into a rubric-scale (1-5) triple.

    Args:
        score_0_1: The linter's ``[0, 1]`` sub-score for this dimension
            (``LintResult.testability_score`` or ``.consistency_score``).
        lint_issues: The full issue list from the same :class:`LintResult`;
            filtered here down to ``category``.
        category: The :class:`~evaluation.linter.LintIssue` category to
            filter to — ``"testability"`` or ``"consistency"``.

    Returns:
        ``(score, justification, issue_strings)``, in the same shape
        :class:`EvaluationRecord` stores for its LLM-judged dimensions,
        so the two sources are indistinguishable to any downstream code.
    """
    score_1_5 = max(1, min(5, round(1 + 4 * score_0_1)))
    relevant = [i for i in lint_issues if i.category == category]
    if relevant:
        justification = (
            f"Deterministic lint: {len(relevant)} issue(s) found "
            f"({score_0_1:.0%} of requirements pass this check)."
        )
    else:
        justification = f"Deterministic lint: no issues found ({score_0_1:.0%} pass)."
    issue_strings = [
        f"{i.requirement_id or '(document)'}: {i.message}" for i in relevant[:5]
    ]
    return score_1_5, justification, issue_strings


class EvaluatorAgent(Agent):
    """LLM-as-judge evaluator.

    Deliberately not a subclass of any specialist agent; it has its own
    role prompt and no knowledge of how the SRS it is scoring was
    produced.
    """

    role_name = "evaluator"
    role_system_prompt = prompts.EVALUATOR_SYSTEM

    def score(
        self,
        description: str,
        srs_markdown: str,
        *,
        architecture: str,
        logger: ExperimentLogger,
    ) -> dict[str, Any]:
        """Score an SRS document against the rubric.

        Retries up to :data:`_MAX_SCORE_ATTEMPTS` times if the verdict shows
        the cross-schema field-echo failure mode (see
        :func:`_verdict_looks_corrupted`). Returns whatever the last
        attempt produced if every attempt is corrupted; the caller falls
        back to a deterministic proxy in that case (see :func:`score_srs`).

        Args:
            description: The original natural-language description; supplied
                so the evaluator can judge completeness against the input.
            srs_markdown: The full SRS Markdown to score.
            architecture: The architecture whose SRS is being scored, for
                log messages only (``self.context.architecture`` is always
                the fixed ``"evaluation"`` protocol placeholder).
            logger: Run logger, so a retry or an exhausted-retries fallback
                is always recorded, never silent.

        Returns:
            The parsed verdict as a dictionary matching
            :data:`~config.prompts.EVALUATION_SCHEMA`.
        """
        verdict: dict[str, Any] = {}
        for attempt in range(1, _MAX_SCORE_ATTEMPTS + 1):
            result = self._call(
                phase="score" if attempt == 1 else f"score_retry_{attempt - 1}",
                user_prompt=prompts.evaluator_user_prompt(description, srs_markdown),
                output_schema=prompts.EVALUATION_SCHEMA,
            )
            assert isinstance(result.parsed_json, dict), (
                "Evaluator was requested to return a JSON object but the "
                "parsed response was not a dict; this is a logic error."
            )
            verdict = result.parsed_json
            if not _verdict_looks_corrupted(verdict):
                return verdict
            logger.warning(
                "[evaluator/%s case=%s rep=%d] attempt %d/%d returned a "
                "corrupted verdict (overall=%r) - retrying",
                architecture, self._context.case_id, self._context.repetition,
                attempt, _MAX_SCORE_ATTEMPTS, verdict.get("overall"),
            )
        logger.warning(
            "[evaluator/%s case=%s rep=%d] all %d attempts returned a "
            "corrupted verdict - falling back to linter-derived "
            "completeness/clarity",
            architecture, self._context.case_id, self._context.repetition,
            _MAX_SCORE_ATTEMPTS,
        )
        return verdict


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

        Only fills :data:`~config.prompts.LLM_EVALUATION_DIMENSIONS`
        (completeness, clarity) — the verdict no longer carries
        testability/consistency at all, since the model is never asked
        for them (see the module note in ``config/prompts.py``). Call
        :meth:`apply_lint` afterwards to fill those two deterministically.

        Args:
            case_id: Test case identifier.
            architecture: Condition that produced the SRS.
            repetition: Zero-based repetition index.
            verdict: Parsed evaluator verdict.

        Returns:
            A populated :class:`EvaluationRecord`, missing the
            deterministic dimensions until :meth:`apply_lint` runs.
        """
        record = cls(
            case_id=case_id, architecture=architecture, repetition=repetition,
            overall=str(verdict.get("overall", "")),
        )
        for dim in prompts.LLM_EVALUATION_DIMENSIONS:
            block = verdict.get(dim, {}) or {}
            record.scores[dim] = int(block.get("score", 0))
            record.justifications[dim] = str(block.get("justification", ""))
            record.issues[dim] = [str(i) for i in (block.get("issues", []) or [])]
        return record

    def apply_lint(self, lint_result: LintResult) -> None:
        """Fill the testability and consistency dimensions deterministically.

        These two are mechanical enough that :mod:`evaluation.linter`
        already checks them exactly (measurable acceptance criteria,
        unique/resolvable requirement ids) — see the module note in
        ``config/prompts.py`` for why they're no longer asked of the LLM.

        Args:
            lint_result: The Layer-1 lint result for the same SRS this
                record scores.
        """
        for dim, score_0_1 in (
            ("testability", lint_result.testability_score),
            ("consistency", lint_result.consistency_score),
        ):
            score, justification, issues = _dimension_from_lint(
                score_0_1, lint_result.issues, dim,
            )
            self.scores[dim] = score
            self.justifications[dim] = justification
            self.issues[dim] = issues

    def apply_llm_dimension_fallback(self, lint_result: LintResult) -> None:
        """Overwrite completeness/clarity with a linter-derived proxy.

        Only called when :func:`score_srs` sees the LLM verdict for these
        two dimensions kept showing the cross-schema field-echo failure
        mode after every retry (see :func:`_verdict_looks_corrupted`) - a
        deterministic proxy that is honestly labelled as degraded is safer
        than trusting a score the model's own justification contradicts.

        Args:
            lint_result: The Layer-1 lint result for the same SRS this
                record scores.
        """
        self.overall = (
            "(LLM evaluator verdict discarded as corrupted after "
            f"{_MAX_SCORE_ATTEMPTS} attempts — completeness/clarity below are "
            "a deterministic linter-derived fallback, not an LLM judgement.)"
        )
        for dim, lint_category in _LLM_DIMENSION_LINT_PROXY.items():
            score_0_1 = getattr(lint_result, f"{lint_category}_score")
            score_1_5 = max(1, min(5, round(1 + 4 * score_0_1)))
            self.scores[dim] = score_1_5
            self.justifications[dim] = (
                f"LLM evaluator verdict was corrupted after "
                f"{_MAX_SCORE_ATTEMPTS} attempts (see run log); falling back "
                f"to a deterministic proxy from linter {lint_category}_score "
                f"({score_0_1:.0%})."
            )
            self.issues[dim] = ["LLM evaluator response discarded — see justification."]


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
    logger: ExperimentLogger,
) -> EvaluationRecord:
    """Score one SRS and return an :class:`EvaluationRecord`.

    Args:
        case_id: Test case identifier.
        architecture: Condition that produced the SRS.
        repetition: Zero-based repetition index.
        description: The original natural-language description.
        srs_markdown: The SRS Markdown to score.
        client: The shared LLM client.
        logger: Run logger, passed through to :meth:`EvaluatorAgent.score`
            so a retry or a corrupted-verdict fallback is always recorded.

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
    verdict = evaluator.score(
        description, srs_markdown, architecture=architecture, logger=logger,
    )
    record = EvaluationRecord.from_verdict(
        case_id=case_id, architecture=architecture,
        repetition=repetition, verdict=verdict,
    )

    # Testability and consistency are mechanical enough to check without a
    # second LLM call — see the module note in config/prompts.py.
    requirements = parse_requirements(srs_markdown)
    lint_result = lint_srs(srs_markdown, requirements)
    record.apply_lint(lint_result)

    if _verdict_looks_corrupted(verdict):
        record.apply_llm_dimension_fallback(lint_result)

    return record


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
