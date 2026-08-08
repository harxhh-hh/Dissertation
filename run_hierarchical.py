"""Stage-2 driver: run one test case through the hierarchical architecture.

Usage::

    python run_hierarchical.py

Prerequisites:

* ``.env`` present at the project root (copy from ``.env.example``) with a
  real ``ANTHROPIC_API_KEY``.
* Dependencies installed (``pip install -r requirements.txt``).

Output:

* A single SRS Markdown file at
  ``outputs/run_<UTC timestamp>/srs_<case_id>_rep0.md``.
* All the standard run artefacts alongside it: ``config.json``,
  ``llm_interactions.jsonl``, ``run.log``, ``run_summary.json``.

This driver is deliberately small: it exists to demonstrate that the
hierarchical architecture produces a complete SRS end-to-end for a single
test case. The full experiment harness (all architectures × all test cases
× ``REPETITIONS``) lands in stage 5 as ``run_experiment.py``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Final

from architectures.hierarchical import HierarchicalResult, run_hierarchical
from config.settings import ConfigurationError, Settings
from utils.llm_client import LLMClient, LLMClientError
from utils.logging import ExperimentLogger

# --------------------------------------------------------------------------
# Fixed inputs for stage 2
# --------------------------------------------------------------------------

#: Test case identifier used in the log lines and the SRS front matter.
CASE_ID: Final[str] = "TC-01_restaurant_app"

#: The natural-language description supplied by the supervisor as the
#: stage-2 acceptance test. Kept as a constant so that the exact string
#: reaches every subsequent stage unaltered.
CASE_DESCRIPTION: Final[str] = (
    "Design a mobile app for a small restaurant chain. Customers should be "
    "able to browse the menu, place orders, and pay through the app. "
    "Restaurant staff should be able to view incoming orders and mark them "
    "as completed. Managers need access to daily sales reports and the "
    "ability to update the menu."
)


def main() -> int:
    """Entry point. Returns a Unix-style exit code (``0`` on success)."""
    try:
        settings = Settings.from_env()
    except ConfigurationError as exc:
        # Configuration errors are the caller's problem to fix, not a bug in
        # the codebase; print a plain message rather than a traceback.
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    settings.apply_seed()

    with ExperimentLogger.for_run(settings) as logger:
        client = LLMClient(settings, logger)

        logger.info(
            "Stage-2 driver: hierarchical architecture on %s (repetition 0)",
            CASE_ID,
        )
        try:
            result = run_hierarchical(
                CASE_DESCRIPTION,
                case_id=CASE_ID,
                repetition=0,
                client=client,
                settings=settings,
                logger=logger,
            )
        except LLMClientError as exc:
            logger.error("Run failed: %s", exc)
            return 1

        _persist_artefacts(result, settings, logger)

    return 0


def _persist_artefacts(
    result: HierarchicalResult,
    settings: Settings,
    logger: ExperimentLogger,
) -> None:
    """Write the SRS, the brief, and the verification verdicts to disk.

    All paths are under the run directory. Artefacts:

    * ``srs_<case>_rep<n>.md`` — the assembled SRS document.
    * ``brief_<case>_rep<n>.json`` — the orchestrator's planning brief.
    * ``verification_<case>_rep<n>.json`` — one record per verification
      round for post-hoc analysis.
    """
    run_dir: Path = settings.run_dir
    slug = f"{result.case_id}_rep{result.repetition}"

    srs_path = run_dir / f"srs_{slug}.md"
    srs_path.write_text(result.srs_markdown, encoding="utf-8")
    logger.info("Wrote SRS to %s (%d chars)", srs_path, len(result.srs_markdown))

    brief_path = run_dir / f"brief_{slug}.json"
    brief_path.write_text(
        json.dumps(result.brief, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    verification_path = run_dir / f"verification_{slug}.json"
    verification_path.write_text(
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

    final_verdict = (
        result.verification_rounds[-1].verdict
        if result.verification_rounds
        else "(none)"
    )
    logger.info(
        "Hierarchical run finished: revised=%s, rounds=%d, final verdict=%s",
        result.revised,
        len(result.verification_rounds),
        final_verdict,
    )


if __name__ == "__main__":
    sys.exit(main())
