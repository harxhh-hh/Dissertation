"""Post-run analysis: produce a Markdown report from a run's artefacts.

The report consumes:

* ``llm_interactions.jsonl`` — every LLM call (used for cost, latency,
  and per-phase call counts).
* ``evaluation.json``       — LLM-as-judge scores (if the evaluation
  step was not skipped).
* ``run_manifest.json``     — the list of (architecture, case, repetition)
  outcomes.

It produces ``analysis_report.md`` inside the same run directory. The
report is deliberately factual: tables of scores, tables of cost and
latency, and short summaries. Interpretation and dissertation-worthy
narrative should be written by the researcher, not by this script.

Usage::

    python -m evaluation.generate_report outputs/run_<id>
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from config.prompts import EVALUATION_DIMENSIONS


# --------------------------------------------------------------------------
# Data structures
# --------------------------------------------------------------------------


@dataclass
class ArchStats:
    """Aggregate statistics for one architecture over a run.

    Attributes:
        architecture: Condition name.
        call_count: Number of LLM calls issued for this architecture,
            excluding evaluator calls.
        failed_calls: Subset of ``call_count`` that raised.
        input_tokens: Sum of input tokens (cached + uncached) across calls.
        output_tokens: Sum of output tokens.
        latency_ms: Sum of per-call wall-clock latency in milliseconds.
        scores_by_dim: Dimension → list of individual document scores.
    """

    architecture: str
    call_count: int = 0
    failed_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    scores_by_dim: dict[str, list[int]] = field(default_factory=dict)


# --------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSON-lines file into a list of decoded objects."""
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _load_json(path: Path) -> Any:
    """Load a JSON file, or ``None`` if it does not exist."""
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------


def _summarise(run_dir: Path) -> dict[str, ArchStats]:
    """Aggregate interactions and evaluation records by architecture.

    Args:
        run_dir: The run directory to analyse.

    Returns:
        Mapping ``{architecture: ArchStats}``. The evaluator's own calls
        are excluded from the cost/latency tallies.
    """
    stats: dict[str, ArchStats] = {}

    for record in _load_jsonl(run_dir / "llm_interactions.jsonl"):
        arch = record.get("architecture", "unknown")
        if arch == "evaluation":
            # Evaluator calls are billed to the analysis, not to any of
            # the four conditions. Excluded from cost comparison.
            continue
        s = stats.setdefault(arch, ArchStats(architecture=arch))
        s.call_count += 1
        if record.get("error") is not None:
            s.failed_calls += 1
        usage = record.get("usage") or {}
        s.input_tokens += int(usage.get("total_input_tokens", 0))
        s.output_tokens += int(usage.get("output_tokens", 0))
        s.latency_ms += float(record.get("latency_ms", 0.0))

    evaluations = _load_json(run_dir / "evaluation.json") or []
    for record in evaluations:
        arch = record.get("architecture", "unknown")
        s = stats.setdefault(arch, ArchStats(architecture=arch))
        scores = record.get("scores") or {}
        for dim in EVALUATION_DIMENSIONS:
            if dim in scores:
                s.scores_by_dim.setdefault(dim, []).append(int(scores[dim]))

    return stats


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def _fmt_float(value: float, digits: int = 2) -> str:
    """Format a float to a fixed number of digits, empty string on NaN."""
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def _mean_stdev(values: list[int]) -> tuple[float, float]:
    """Return (mean, sample stdev). Stdev is 0 for lists of length ≤ 1."""
    if not values:
        return (0.0, 0.0)
    mean = statistics.fmean(values)
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    return (mean, stdev)


def _render_scores_table(stats: dict[str, ArchStats]) -> str:
    """Render the per-architecture score table as Markdown."""
    architectures = sorted(stats)
    dims = list(EVALUATION_DIMENSIONS)
    header = "| Architecture | " + " | ".join(f"{d} (mean ± sd)" for d in dims) + " | n |"
    sep = "|---" * (len(dims) + 2) + "|"
    rows = [header, sep]
    for arch in architectures:
        s = stats[arch]
        n = min((len(s.scores_by_dim.get(d, [])) for d in dims), default=0)
        cells = []
        for dim in dims:
            mean, sd = _mean_stdev(s.scores_by_dim.get(dim, []))
            cells.append(
                f"{_fmt_float(mean, 2)} ± {_fmt_float(sd, 2)}"
                if s.scores_by_dim.get(dim)
                else "—"
            )
        rows.append(f"| {arch} | " + " | ".join(cells) + f" | {n} |")
    return "\n".join(rows)


def _render_cost_table(stats: dict[str, ArchStats]) -> str:
    """Render the per-architecture cost/latency table as Markdown."""
    header = (
        "| Architecture | calls | failed | in tokens | out tokens | total tokens | total s | s / call |"
    )
    sep = "|---" * 8 + "|"
    rows = [header, sep]
    for arch in sorted(stats):
        s = stats[arch]
        total = s.input_tokens + s.output_tokens
        total_s = s.latency_ms / 1000.0
        per_call = total_s / s.call_count if s.call_count else 0.0
        rows.append(
            f"| {arch} | {s.call_count} | {s.failed_calls} | {s.input_tokens} | "
            f"{s.output_tokens} | {total} | {_fmt_float(total_s, 1)} | "
            f"{_fmt_float(per_call, 2)} |"
        )
    return "\n".join(rows)


def _render_case_matrix(evaluations: list[dict[str, Any]]) -> str:
    """Render a case × architecture matrix of mean total scores.

    Total score here is the sum of the four dimension scores (max 20).
    Cells hold the mean across repetitions; the ``n`` column shows the
    total number of scored documents that fed into that architecture's
    row so a reader can see the sample size at a glance.
    """
    if not evaluations:
        return "_No evaluation records found; rerun without --skip-evaluation._\n"

    totals: dict[tuple[str, str], list[int]] = defaultdict(list)
    for record in evaluations:
        arch = record["architecture"]
        case = record["case_id"]
        total = sum(int(record["scores"].get(d, 0)) for d in EVALUATION_DIMENSIONS)
        totals[(arch, case)].append(total)

    architectures = sorted({k[0] for k in totals})
    cases = sorted({k[1] for k in totals})

    header = "| Architecture | " + " | ".join(cases) + " | n |"
    sep = "|---" * (len(cases) + 2) + "|"
    rows = [header, sep]
    for arch in architectures:
        arch_n = 0
        cells: list[str] = []
        for case in cases:
            values = totals.get((arch, case), [])
            arch_n += len(values)
            if values:
                mean = statistics.fmean(values)
                cells.append(_fmt_float(mean, 1))
            else:
                cells.append("—")
        rows.append(f"| {arch} | " + " | ".join(cells) + f" | {arch_n} |")
    rows.append("")
    rows.append("_Cell values are mean total scores out of 20 (sum of the four rubric dimensions)._")
    return "\n".join(rows)


def generate_report(run_dir: Path) -> Path:
    """Write ``analysis_report.md`` inside ``run_dir`` and return its path.

    Args:
        run_dir: Path to a directory produced by :mod:`run_experiment`.

    Returns:
        Path to the newly-written report.

    Raises:
        FileNotFoundError: If ``run_dir`` does not exist.
    """
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    stats = _summarise(run_dir)
    evaluations = _load_json(run_dir / "evaluation.json") or []
    config = _load_json(run_dir / "config.json") or {}
    manifest = _load_json(run_dir / "run_manifest.json") or {}

    lines: list[str] = []
    lines.append(f"# Experiment analysis — run `{config.get('run_id', run_dir.name)}`\n")
    lines.append(
        "_This report is generated automatically from the run's artefacts._ "
        "_Cost and latency exclude evaluator calls; scores are LLM-as-judge_ "
        "_unless the run also includes human ratings._\n"
    )

    lines.append("## Configuration\n")
    lines.append("| Setting | Value |")
    lines.append("|---|---|")
    for key in (
        "model_id", "effort", "thinking_mode", "max_tokens",
        "max_revision_rounds", "repetitions", "random_seed",
    ):
        lines.append(f"| `{key}` | `{config.get(key, '?')}` |")
    lines.append("")

    lines.append("## Rubric scores by architecture\n")
    lines.append(_render_scores_table(stats))
    lines.append("")

    lines.append("## Rubric total by case (mean total out of 20)\n")
    lines.append(_render_case_matrix(evaluations))
    lines.append("")

    lines.append("## Cost and latency by architecture\n")
    lines.append(_render_cost_table(stats))
    lines.append("")

    failed = [o for o in manifest.get("outcomes", []) if o.get("error")]
    if failed:
        lines.append("## Failed runs\n")
        lines.append("| Architecture | Case | Rep | Error |")
        lines.append("|---|---|---|---|")
        for o in failed:
            err = str(o.get("error", "")).replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {o['architecture']} | {o['case_id']} | {o['repetition']} | {err} |")
        lines.append("")

    lines.append("## Interpretation guidance\n")
    lines.append(
        "* Between-architecture differences under two runs per condition "
        "are indicative, not conclusive. If a difference matters, raise "
        "`REPETITIONS` and re-run before reporting a claim.\n"
        "* LLM-as-judge scores are subject to same-model bias when the "
        "evaluator uses the same model family as the generators. Prefer "
        "human rater scores (see `rater_export_*/`) as the primary "
        "measure and treat this table as a sanity check.\n"
        "* Cost and latency are strictly comparable across conditions; "
        "the numbers in the cost table are wall-clock, not billable API "
        "time, so treat them as approximate but consistent.\n"
    )
    report_path = run_dir / "analysis_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Path to outputs/run_<id>.")
    args = parser.parse_args(argv)
    try:
        path = generate_report(args.run_dir)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
