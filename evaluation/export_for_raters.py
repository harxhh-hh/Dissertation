"""Anonymised export of generated SRS documents for blind human rating.

The dissertation calls for blind evaluation by human raters. This module
takes the artefacts of a completed experiment run and produces a package
that a rater can score without knowing which architecture produced which
document. The mapping from anonymised identifier to source condition is
written to a separate file that the rater does NOT receive.

Structure of an export:

``rater_export/``
    ``README.md``             — cover letter and instructions for raters.
    ``documents/``            — the anonymised SRS files (DOC-01.md, ...).
    ``scoring_sheet.csv``     — one row per document with empty score cells.
    ``mapping.json``          — anonymised ID → (case, architecture, rep).
                                Kept out of the ``documents/`` folder so
                                it is not sent to raters by accident.

Anonymisation strips the run-metadata block from each SRS. The stripped
copy still shows the input description (raters need it to judge
completeness) and the requirements themselves, but no architecture or
model identifier remains.

The anonymised order is deterministic given the run's ``RANDOM_SEED``, so
re-running the export against the same run reproduces the same
identifier assignments.
"""

from __future__ import annotations

import csv
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from config import prompts
from config.settings import Settings

#: Regex used to strip the run-metadata block from an SRS. The block is
#: everything between the ``## Run metadata`` header and the following
#: top-level header. Matches the format written by
#: :func:`architectures.hierarchical.format_srs_markdown`.
_METADATA_BLOCK_RE = re.compile(
    r"^##\s+Run metadata\s*\n.*?(?=^##\s)", re.MULTILINE | re.DOTALL
)


@dataclass
class SrsArtifact:
    """One SRS produced by the experiment, as input to the export.

    Attributes:
        case_id: The test case identifier.
        architecture: The condition that produced the SRS.
        repetition: Zero-based repetition index.
        srs_path: Path to the SRS Markdown file.
        description: The original natural-language description (embedded
            in the anonymised copy so raters can judge completeness).
    """

    case_id: str
    architecture: str
    repetition: int
    srs_path: Path
    description: str


@dataclass
class ExportEntry:
    """One row of the rater export mapping.

    Attributes:
        anonymous_id: The rater-facing identifier (``DOC-01``, ...).
        case_id: Source test case identifier.
        architecture: Source architecture.
        repetition: Source repetition index.
        source_path: Original SRS path in the run directory.
        exported_path: Anonymised copy in the export directory.
    """

    anonymous_id: str
    case_id: str
    architecture: str
    repetition: int
    source_path: Path
    exported_path: Path

    def to_mapping_dict(self) -> dict[str, str | int]:
        """Return a dict for the private ``mapping.json``."""
        return {
            "anonymous_id": self.anonymous_id,
            "case_id": self.case_id,
            "architecture": self.architecture,
            "repetition": self.repetition,
            "source_path": str(self.source_path),
            "exported_path": str(self.exported_path),
        }


@dataclass
class ExportResult:
    """Outcome of an export.

    Attributes:
        export_dir: The root directory of the export.
        entries: One entry per exported document.
    """

    export_dir: Path
    entries: list[ExportEntry] = field(default_factory=list)


def export_for_raters(
    artefacts: Iterable[SrsArtifact],
    *,
    settings: Settings,
    export_root: Path | None = None,
) -> ExportResult:
    """Produce an anonymised, seed-shuffled export package.

    Args:
        artefacts: The generated SRS artefacts to include. Typically one
            per (case, architecture, repetition) combination.
        settings: The run configuration. ``random_seed`` controls the
            shuffle; ``run_id`` is used in the default export directory
            name so successive exports do not overwrite each other.
        export_root: Directory under which the ``rater_export`` folder
            will be created. Defaults to ``settings.run_dir``.

    Returns:
        A populated :class:`ExportResult` with paths to every artefact.
    """
    root = export_root if export_root is not None else settings.run_dir
    export_dir = root / f"rater_export_{settings.run_id}"
    documents_dir = export_dir / "documents"
    documents_dir.mkdir(parents=True, exist_ok=True)

    # Materialise, sort deterministically, then shuffle under the run's
    # seed. Sorting first makes the shuffle output reproducible even if
    # the caller supplies artefacts in different orders across runs.
    materialised = sorted(
        artefacts,
        key=lambda a: (a.case_id, a.architecture, a.repetition),
    )
    rng = random.Random(settings.random_seed)
    order = list(range(len(materialised)))
    rng.shuffle(order)

    entries: list[ExportEntry] = []
    for anon_index, source_index in enumerate(order, start=1):
        artefact = materialised[source_index]
        anonymous_id = f"DOC-{anon_index:02d}"
        exported_path = documents_dir / f"{anonymous_id}.md"

        source_text = artefact.srs_path.read_text(encoding="utf-8")
        anonymised = _anonymise_srs(source_text, anonymous_id)
        exported_path.write_text(anonymised, encoding="utf-8")

        entries.append(
            ExportEntry(
                anonymous_id=anonymous_id,
                case_id=artefact.case_id,
                architecture=artefact.architecture,
                repetition=artefact.repetition,
                source_path=artefact.srs_path,
                exported_path=exported_path,
            )
        )

    (export_dir / "mapping.json").write_text(
        json.dumps(
            {"run_id": settings.run_id, "entries": [e.to_mapping_dict() for e in entries]},
            indent=2, sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    _write_scoring_sheet(export_dir / "scoring_sheet.csv", entries)
    (export_dir / "README.md").write_text(_rater_instructions(entries), encoding="utf-8")

    return ExportResult(export_dir=export_dir, entries=entries)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _anonymise_srs(source_text: str, anonymous_id: str) -> str:
    """Strip the run-metadata block and replace the title with the anonymous ID.

    Args:
        source_text: The original SRS Markdown.
        anonymous_id: The document's rater-facing identifier.

    Returns:
        The anonymised Markdown.
    """
    # Replace the SRS title's case identifier with the anonymous one.
    # First line of every SRS is ``# Software Requirements Specification — <case>``.
    stripped_title = re.sub(
        r"^# Software Requirements Specification[^\n]*",
        f"# Software Requirements Specification — {anonymous_id}",
        source_text,
        count=1,
    )
    # Remove the run metadata block. If the file for some reason has no
    # metadata block (older format, hand-edited), the sub is a no-op.
    without_metadata = _METADATA_BLOCK_RE.sub("", stripped_title, count=1)
    return without_metadata.rstrip() + "\n"


def _write_scoring_sheet(path: Path, entries: list[ExportEntry]) -> None:
    """Write an empty scoring sheet for the raters to fill in.

    One row per document, with a column per rubric dimension left blank.
    """
    dimensions = list(prompts.EVALUATION_DIMENSIONS)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["anonymous_id", *[f"{d}_score" for d in dimensions], "overall_comment"]
        )
        for entry in entries:
            writer.writerow([entry.anonymous_id] + [""] * (len(dimensions) + 1))


def _rater_instructions(entries: list[ExportEntry]) -> str:
    """Compose the rater-facing README.

    Written in the second person and deliberately terse: raters do not
    need the study's methodology, only what to score and how.
    """
    dimensions = list(prompts.EVALUATION_DIMENSIONS)
    dimension_bullets = "\n".join(f"- {d}" for d in dimensions)
    return (
        "# Rater Instructions\n\n"
        "Thank you for helping evaluate this set of Software Requirements "
        "Specification (SRS) documents.\n\n"
        f"You have {len(entries)} documents to score. Each is in the "
        "`documents/` folder, named `DOC-NN.md`. They are presented in a "
        "randomised order and their identifiers have been anonymised.\n\n"
        "## What to do\n\n"
        "For each document, open the file and read:\n\n"
        "1. The **Input description** section — this is the natural-"
        "language description the SRS is supposed to satisfy.\n"
        "2. The requirements, risks, and open questions that follow.\n\n"
        "Then open `scoring_sheet.csv` and fill in one row per document. "
        "Score each of the four dimensions from **1 (worst)** to **5 "
        "(best)**:\n\n"
        f"{dimension_bullets}\n\n"
        "Definitions:\n\n"
        "- **completeness** — every capability and stakeholder implied by "
        "the description is covered; every reasonably-implied quality "
        "attribute has a measurable non-functional requirement.\n"
        "- **consistency** — no two requirements contradict, duplicate, "
        "or subsume each other; identifiers are unique.\n"
        "- **testability** — every requirement admits a concrete "
        "acceptance criterion or measurement.\n"
        "- **clarity** — the SRS is unambiguous; a reader can determine "
        "one interpretation of each requirement.\n\n"
        "Add a short overall comment per document in the "
        "`overall_comment` column.\n\n"
        "Please do NOT compare notes with other raters before scoring, "
        "and please do NOT try to guess which document came from which "
        "system.\n"
    )
