"""End-to-end experiment harness.

Runs every (architecture × test case × repetition) combination, then
optionally evaluates every SRS with the LLM-as-judge rubric and produces
the anonymised rater export.

Usage::

    python run_experiment.py                 # runs everything
    python run_experiment.py --skip-evaluation
    python run_experiment.py --skip-rater-export
    python run_experiment.py --architectures baseline_single_prompt hierarchical
    python run_experiment.py --cases TC-01_restaurant_app TC-04_smart_home

Cost warning:

    A default run at ``REPETITIONS=1``, ``MAX_REVISION_ROUNDS=1`` on the
    four test cases is roughly 4 baseline + 4×hierarchical + 4×peer_to_peer
    + 4×debate + (optional) 16 evaluations = **~120 LLM calls total** at
    ``claude-opus-5``. Ballpark **~£3–£6**. Higher REPETITIONS scales
    linearly. Print the plan and require the user to type "yes" before
    starting.

Outputs land in the standard run directory, one file per artefact:

    outputs/run_<id>/
      config.json
      llm_interactions.jsonl
      run.log
      run_summary.json
      srs_<case>_<architecture>_rep<n>.md
      brief_<case>_<architecture>_rep<n>.json            (multi-agent only)
      verification_<case>_<architecture>_rep<n>.json     (multi-agent only)
      arbitration_<case>_debate_rep<n>.json              (debate only)
      evaluation.json                                    (unless skipped)
      rater_export_<id>/                                 (unless skipped)
      run_manifest.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from architectures.debate import DebateResult, run_debate
from architectures.hierarchical import HierarchicalResult, run_hierarchical
from architectures.peer_to_peer import PeerToPeerResult, run_peer_to_peer
from baseline_single_prompt import BaselineResult, run_baseline
from config.settings import ConfigurationError, Settings
from evaluation.export_for_raters import SrsArtifact, export_for_raters
from evaluation.grounded_grader import grade_grounding
from evaluation.knowledge_base import detect_domain, get_domain_kb
from evaluation.linter import lint_srs
from evaluation.rubric import (
    DIMENSIONS as EVALUATION_DIMENSIONS,
    EvaluationRecord,
    aggregate_by_architecture,
    score_srs,
    write_evaluation,
)
from evaluation.scoring import (
    ArchitectureScore,
    OverallResult,
    compute_architecture_score,
    compute_overall,
)
from evaluation.srs_parser import parse_requirements
from test_cases.cases import TEST_CASES, TestCase, get_case
from utils.llm_client import LLMClient, LLMClientError
from utils.logging import ExperimentLogger

#: The four conditions under test. Order matters for the manifest and the
#: report tables; keep baseline first so it appears as the reference
#: column when tables are read left to right.
ALL_ARCHITECTURES: tuple[str, ...] = (
    "baseline_single_prompt",
    "hierarchical",
    "peer_to_peer",
    "debate",
)


@dataclass
class RunOutcome:
    """One (architecture × case × repetition) invocation's outcome.

    Attributes:
        architecture: The condition that ran.
        case_id: The test case identifier.
        repetition: Zero-based repetition index.
        srs_path: Path to the persisted SRS Markdown. ``None`` on failure.
        error: Populated when the run failed (with the exception's
            string form); ``None`` on success.
    """

    architecture: str
    case_id: str
    repetition: int
    srs_path: Path | None = None
    error: str | None = None


@dataclass
class RunManifest:
    """Summary manifest of an experiment run, written as ``run_manifest.json``.

    Attributes:
        run_id: The run identifier.
        settings: Redacted settings snapshot.
        outcomes: One entry per (architecture, case, repetition) attempted.
        evaluation_path: Path to the evaluation output, if written.
        rater_export_path: Path to the rater export directory, if written.
        grounded_scores_path: Path to the grounded-scoring output (KB
            coverage/faithfulness + composite ranking), if written.
    """

    run_id: str
    settings: dict[str, Any] = field(default_factory=dict)
    outcomes: list[dict[str, Any]] = field(default_factory=list)
    evaluation_path: str | None = None
    rater_export_path: str | None = None
    grounded_scores_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable snapshot."""
        return {
            "run_id": self.run_id,
            "settings": self.settings,
            "outcomes": self.outcomes,
            "evaluation_path": self.evaluation_path,
            "rater_export_path": self.rater_export_path,
            "grounded_scores_path": self.grounded_scores_path,
        }


# =========================================================================
# Argument parsing
# =========================================================================


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: The argument list to parse, or ``None`` to use ``sys.argv``.

    Returns:
        The parsed namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    # Both plural and singular flag names are accepted so scripted
    # invocations that pass a single value read naturally
    # (``--architecture hierarchical`` instead of ``--architectures``).
    parser.add_argument(
        "--architectures", "--architecture",
        nargs="+", choices=ALL_ARCHITECTURES,
        default=list(ALL_ARCHITECTURES),
        help="Restrict the run to a subset of architectures.",
    )
    parser.add_argument(
        "--cases", "--case",
        nargs="+",
        default=[c.case_id for c in TEST_CASES],
        help=(
            "Restrict the run to specific test cases. Accepts either full "
            "case IDs (e.g. 'TC-01_restaurant_app') or unique substrings "
            "of them (e.g. 'restaurant_app')."
        ),
    )
    parser.add_argument(
        "--skip-evaluation", action="store_true",
        help="Skip the LLM-as-judge scoring step.",
    )
    parser.add_argument(
        "--skip-rater-export", action="store_true",
        help="Skip the anonymised export for human raters.",
    )
    parser.add_argument(
        "--skip-grounding", action="store_true",
        help=(
            "Skip grounded scoring (knowledge-base coverage/faithfulness "
            "+ deterministic lint). Cases whose domain has no knowledge "
            "base are skipped automatically even when this is not set."
        ),
    )
    parser.add_argument(
        "--yes", action="store_true",
        help="Skip the interactive confirmation prompt.",
    )
    return parser.parse_args(argv)


# =========================================================================
# Runner
# =========================================================================


def _persist_multi_agent_artefacts(
    *,
    run_dir: Path,
    architecture: str,
    result: HierarchicalResult | PeerToPeerResult | DebateResult,
) -> Path:
    """Write SRS, brief, verification, and (if debate) arbitration files.

    Returns the path to the SRS Markdown.
    """
    slug = f"{result.case_id}_{architecture}_rep{result.repetition}"

    srs_path = run_dir / f"srs_{slug}.md"
    srs_path.write_text(result.srs_markdown, encoding="utf-8")

    (run_dir / f"brief_{slug}.json").write_text(
        json.dumps(result.brief, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (run_dir / f"verification_{slug}.json").write_text(
        json.dumps(
            [
                {
                    "round_index": r.round_index,
                    "verdict": r.verdict,
                    "summary": r.summary,
                    "issues": r.issues,
                }
                for r in result.verification_rounds
            ],
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if isinstance(result, DebateResult):
        (run_dir / f"arbitration_{slug}.json").write_text(
            json.dumps(
                [
                    {
                        "section": a.section,
                        "verdict": a.verdict,
                        "rationale": a.rationale,
                    }
                    for a in result.arbitrations
                ],
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return srs_path


def _persist_baseline_artefact(run_dir: Path, result: BaselineResult) -> Path:
    """Write the baseline SRS and return its path."""
    slug = f"{result.case_id}_baseline_single_prompt_rep{result.repetition}"
    srs_path = run_dir / f"srs_{slug}.md"
    srs_path.write_text(result.srs_markdown, encoding="utf-8")
    return srs_path


def _run_one(
    *,
    architecture: str,
    case: TestCase,
    repetition: int,
    client: LLMClient,
    settings: Settings,
    logger: ExperimentLogger,
) -> Path:
    """Run one (architecture × case × repetition), persist artefacts.

    Returns the SRS path. Any exception from the run propagates so the
    caller can record the failure and continue with the next combination.
    """
    run_dir = settings.run_dir
    if architecture == "baseline_single_prompt":
        result = run_baseline(
            case.description, case_id=case.case_id, repetition=repetition,
            client=client, settings=settings, logger=logger,
        )
        return _persist_baseline_artefact(run_dir, result)
    if architecture == "hierarchical":
        result = run_hierarchical(
            case.description, case_id=case.case_id, repetition=repetition,
            client=client, settings=settings, logger=logger,
        )
    elif architecture == "peer_to_peer":
        result = run_peer_to_peer(
            case.description, case_id=case.case_id, repetition=repetition,
            client=client, settings=settings, logger=logger,
        )
    elif architecture == "debate":
        result = run_debate(
            case.description, case_id=case.case_id, repetition=repetition,
            client=client, settings=settings, logger=logger,
        )
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    return _persist_multi_agent_artefacts(
        run_dir=run_dir, architecture=architecture, result=result
    )


def _resolve_case_id(user_input: str) -> str:
    """Resolve a user-supplied case name to a canonical case_id.

    Accepts either the exact canonical ID (e.g. ``TC-01_restaurant_app``)
    or any unique substring of one (e.g. ``restaurant_app``, ``TC-04``).

    Args:
        user_input: The value the user passed on the command line.

    Returns:
        The canonical case identifier.

    Raises:
        SystemExit: If the value matches nothing, or matches more than
            one case (with a message listing the candidates).
    """
    known = [c.case_id for c in TEST_CASES]
    if user_input in known:
        return user_input
    matches = [k for k in known if user_input in k]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise SystemExit(
            f"error: no test case matches {user_input!r}. Known: {known}"
        )
    raise SystemExit(
        f"error: {user_input!r} is ambiguous; matches {matches}. Use a "
        "more specific value or the full case_id."
    )


def _print_plan(architectures: list[str], cases: list[str], repetitions: int) -> None:
    """Print a summary of the planned run and rough cost bounds to stdout."""
    combos = len(architectures) * len(cases) * repetitions
    print()
    print(f"Plan:")
    print(f"  architectures : {len(architectures)}  ({', '.join(architectures)})")
    print(f"  test cases    : {len(cases)}  ({', '.join(cases)})")
    print(f"  repetitions   : {repetitions}")
    print(f"  total runs    : {combos}")
    print(f"  each 'debate' run is ~13 LLM calls;")
    print(f"  each 'peer_to_peer' run is ~8; hierarchical ~5–8; baseline 1.")
    print()


def _log_evaluation_summary(eval_records: list[EvaluationRecord], logger: ExperimentLogger) -> None:
    """Log each architecture's mean rubric score and the leader, to the narrative log.

    A no-op with fewer than two scored architectures — nothing to rank.
    Mirrors ui/app.py's ``_render_rubric_summary`` so the UI banner and the
    log line always agree on the same numbers and the same winner.
    """
    if not eval_records:
        return
    means = aggregate_by_architecture(eval_records)
    if len(means) < 2:
        return

    max_total = len(EVALUATION_DIMENSIONS) * 5
    totals = {arch: sum(dims.values()) for arch, dims in means.items()}
    ranked = sorted(totals, key=lambda a: (-totals[a], a))

    logger.info("=== rubric scores by architecture ===")
    for arch in ranked:
        breakdown = " ".join(f"{d}={means[arch][d]:.1f}" for d in EVALUATION_DIMENSIONS)
        logger.info("  %-24s total=%.1f/%d  (%s)", arch, totals[arch], max_total, breakdown)
    logger.info("📋 Rubric leader: %s (%.1f/%d)", ranked[0], totals[ranked[0]], max_total)


def _log_grounded_summary(overall: OverallResult, logger: ExperimentLogger) -> None:
    """Log each architecture's grounded composite score and the winner.

    A no-op with fewer than two scored architectures — nothing to rank.
    Mirrors ui/app.py's ``_render_grounded_winner`` so the UI banner and
    the log line always agree.
    """
    if len(overall.summaries) < 2:
        return

    logger.info("=== grounded scores by architecture ===")
    for s in overall.summaries:
        logger.info(
            "  %-24s composite=%.2f coverage=%.0f%% faithfulness=%.0f%% "
            "quality=%.0f%% won=%d/%d",
            s.architecture, s.mean_composite, s.mean_coverage * 100,
            s.mean_faithfulness * 100, s.mean_quality * 100,
            s.cases_won, s.cases_scored,
        )
    winner = overall.summaries[0]
    logger.info("🏆 Grounded winner: %s (composite %.2f)", winner.architecture, winner.mean_composite)


def run_matrix(
    *,
    settings: Settings,
    architectures: list[str],
    cases: list[TestCase],
    client: LLMClient,
    logger: ExperimentLogger,
    skip_evaluation: bool = False,
    skip_rater_export: bool = False,
    skip_grounding: bool = False,
) -> tuple[RunManifest, list[RunOutcome]]:
    """Run the full (architecture x case x repetition) matrix and post-steps.

    This is the reusable core of the experiment harness: everything
    ``main()`` does after the CLI has resolved its arguments into concrete
    values and opened a logger/client, up to (but not including) printing
    the final human-readable summary line. Factored out so a non-CLI caller
    (the Streamlit UI) can drive the exact same logic — same artefact
    layout, same logging, same manifest — without going through argparse
    or the interactive confirmation prompt.

    Args:
        settings: Resolved run configuration.
        architectures: Architecture names to run (subset of
            :data:`ALL_ARCHITECTURES`).
        cases: Test cases to run.
        client: The shared LLM client for this run.
        logger: The active experiment logger.
        skip_evaluation: If ``True``, skip the LLM-as-judge scoring step.
        skip_rater_export: If ``True``, skip the anonymised rater export.
        skip_grounding: If ``True``, skip grounded scoring entirely. When
            ``False`` (the default), grounded scoring still only runs for
            cases whose domain has a knowledge base registered under
            ``knowledge_bases/`` — cases with no matching domain are
            skipped with a log line, never guessed at.

    Returns:
        A tuple of the written :class:`RunManifest` and the list of
        per-combination :class:`RunOutcome` (including failed ones).
    """
    manifest = RunManifest(run_id=settings.run_id, settings=settings.to_dict())
    outcomes: list[RunOutcome] = []
    artefacts_by_key: dict[tuple[str, str, int], Path] = {}

    # ---- Main matrix ------------------------------------------------
    for architecture in architectures:
        for case in cases:
            for repetition in range(settings.repetitions):
                logger.info(
                    "=== %s / %s / rep %d ===", architecture, case.case_id, repetition
                )
                try:
                    srs_path = _run_one(
                        architecture=architecture, case=case, repetition=repetition,
                        client=client, settings=settings, logger=logger,
                    )
                except (LLMClientError, Exception) as exc:  # noqa: BLE001
                    logger.error(
                        "Run FAILED: %s / %s / rep %d: %s",
                        architecture, case.case_id, repetition, exc, exc_info=True,
                    )
                    outcomes.append(RunOutcome(
                        architecture=architecture, case_id=case.case_id,
                        repetition=repetition, error=str(exc),
                    ))
                    continue
                outcomes.append(RunOutcome(
                    architecture=architecture, case_id=case.case_id,
                    repetition=repetition, srs_path=srs_path,
                ))
                artefacts_by_key[(architecture, case.case_id, repetition)] = srs_path

    manifest.outcomes = [asdict(o) | ({"srs_path": str(o.srs_path)} if o.srs_path else {})
                          for o in outcomes]

    # ---- Evaluation --------------------------------------------------
    if not skip_evaluation and artefacts_by_key:
        logger.info("=== evaluation ===")
        eval_records = _run_evaluation(
            artefacts_by_key=artefacts_by_key, cases=cases,
            client=client, logger=logger,
        )
        eval_path = settings.run_dir / "evaluation.json"
        write_evaluation(eval_records, eval_path)
        manifest.evaluation_path = str(eval_path)
        _log_evaluation_summary(eval_records, logger)

    # ---- Grounded scoring ---------------------------------------------
    if not skip_grounding and artefacts_by_key:
        logger.info("=== grounded scoring ===")
        grounded_scores = _run_grounded_scoring(
            artefacts_by_key=artefacts_by_key, cases=cases,
            client=client, logger=logger,
        )
        if grounded_scores:
            grounded_path = settings.run_dir / "grounded_scores.json"
            overall = compute_overall(grounded_scores)
            _log_grounded_summary(overall, logger)
            grounded_path.write_text(
                json.dumps(
                    {
                        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "scores": [s.to_dict() for s in grounded_scores],
                        "overall": overall.to_dict(),
                    },
                    indent=2, sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            manifest.grounded_scores_path = str(grounded_path)
        else:
            logger.info(
                "No case in this run has a registered knowledge-base "
                "domain; grounded scoring produced nothing."
            )

    # ---- Rater export ------------------------------------------------
    if not skip_rater_export and artefacts_by_key:
        logger.info("=== rater export ===")
        artefacts = [
            SrsArtifact(
                case_id=case_id, architecture=arch, repetition=rep,
                srs_path=path,
                description=get_case(case_id).description,
            )
            for (arch, case_id, rep), path in artefacts_by_key.items()
        ]
        export_result = export_for_raters(artefacts, settings=settings)
        manifest.rater_export_path = str(export_result.export_dir)

    # ---- Manifest ----------------------------------------------------
    (settings.run_dir / "run_manifest.json").write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return manifest, outcomes


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a Unix-style exit code."""
    args = _parse_args(argv)
    try:
        settings = Settings.from_env()
    except ConfigurationError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    try:
        cases = [get_case(_resolve_case_id(c)) for c in args.cases]
    except SystemExit:
        raise
    architectures = list(args.architectures)

    _print_plan(architectures, args.cases, settings.repetitions)
    if not args.yes:
        try:
            response = input("Proceed? [type 'yes' to run]: ").strip().lower()
        except EOFError:
            response = ""
        if response != "yes":
            print("Aborted by user.")
            return 130

    settings.apply_seed()

    with ExperimentLogger.for_run(settings) as logger:
        client = LLMClient(settings, logger)
        _manifest, outcomes = run_matrix(
            settings=settings, architectures=architectures, cases=cases,
            client=client, logger=logger,
            skip_evaluation=args.skip_evaluation,
            skip_rater_export=args.skip_rater_export,
            skip_grounding=args.skip_grounding,
        )

    failed = sum(1 for o in outcomes if o.error is not None)
    print(f"\nDone: {len(outcomes) - failed}/{len(outcomes)} runs succeeded.")
    return 0 if failed == 0 else 1


def _run_evaluation(
    *,
    artefacts_by_key: dict[tuple[str, str, int], Path],
    cases: Iterable[TestCase],
    client: LLMClient,
    logger: ExperimentLogger,
) -> list[EvaluationRecord]:
    """Run the LLM-as-judge scorer on every produced SRS.

    Failures are logged and skipped so a single bad score does not abort
    the whole evaluation pass.
    """
    case_by_id = {c.case_id: c for c in cases}
    records: list[EvaluationRecord] = []
    for (architecture, case_id, repetition), srs_path in artefacts_by_key.items():
        try:
            srs_markdown = srs_path.read_text(encoding="utf-8")
            record = score_srs(
                case_id=case_id, architecture=architecture, repetition=repetition,
                description=case_by_id[case_id].description,
                srs_markdown=srs_markdown, client=client,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Evaluation FAILED for %s / %s / rep %d: %s",
                architecture, case_id, repetition, exc, exc_info=True,
            )
            continue
        records.append(record)
    return records


def _run_grounded_scoring(
    *,
    artefacts_by_key: dict[tuple[str, str, int], Path],
    cases: Iterable[TestCase],
    client: LLMClient,
    logger: ExperimentLogger,
) -> list[ArchitectureScore]:
    """Run the grounded-scoring stack (parse -> lint -> grade -> score) on
    every produced SRS whose case's domain has a registered knowledge base.

    Cases with no matching domain are skipped with an info log line, not an
    error — most cases will have no KB until more domains are researched,
    and that must never look like a failure. Failures scoring an
    individual artefact are logged and skipped, same policy as
    :func:`_run_evaluation`, so one bad grade does not abort the run.
    """
    case_by_id = {c.case_id: c for c in cases}
    scores: list[ArchitectureScore] = []
    for (architecture, case_id, repetition), srs_path in artefacts_by_key.items():
        case = case_by_id[case_id]
        domain = detect_domain(case_id=case_id, description=case.description)
        if domain is None:
            logger.info(
                "Grounded scoring SKIPPED for %s / %s / rep %d: no "
                "registered knowledge-base domain matches this case.",
                architecture, case_id, repetition,
            )
            continue
        domain_kb = get_domain_kb(domain)
        if domain_kb is None or not domain_kb.facts:
            logger.info(
                "Grounded scoring SKIPPED for %s / %s / rep %d: domain "
                "%r has no usable knowledge base.",
                architecture, case_id, repetition, domain,
            )
            continue
        try:
            srs_markdown = srs_path.read_text(encoding="utf-8")
            requirements = parse_requirements(srs_markdown)
            lint_result = lint_srs(srs_markdown, requirements)
            grade_result = grade_grounding(
                case_id=case_id, architecture=architecture, repetition=repetition,
                requirements=requirements, domain_kb=domain_kb, client=client,
            )
            score = compute_architecture_score(
                architecture=architecture, case_id=case_id, repetition=repetition,
                domain_kb=domain_kb, lint_result=lint_result, grade_result=grade_result,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Grounded scoring FAILED for %s / %s / rep %d: %s",
                architecture, case_id, repetition, exc, exc_info=True,
            )
            continue
        scores.append(score)
    return scores


if __name__ == "__main__":
    sys.exit(main())


# Keep ``Callable`` referenced for future extension; documents that
# ``_run_one``'s dispatch table could be lifted to a mapping later.
_ = Callable
