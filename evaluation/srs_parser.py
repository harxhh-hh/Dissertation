"""Parse a generated SRS Markdown document into structured requirements.

The four architectures share :func:`architectures.hierarchical.format_srs_markdown`,
but the *content* inside the ``## Functional requirements`` and
``## Non-functional requirements`` sections is free-form prose written by an
LLM against a style guide (see ``config/prompts.py``), not a machine-checked
schema. In practice the model doesn't follow the style guide's exact
formatting: it sometimes puts the ``FR-NNN:`` id in a ``####`` heading and
sometimes in plain text, sometimes writes the rationale on its own line and
sometimes inline on the same paragraph, sometimes wraps the statement in
Markdown italics and sometimes doesn't.

This parser is deliberately tolerant of that drift rather than assuming the
ideal format, because rejecting real output would make the whole downstream
scoring pipeline unusable on real runs. It extracts three things per
requirement, on a best-effort basis:

* the id (``FR-NNN`` / ``NFR-NNN``, which also determines ``type``),
* the ``###`` subsection heading it was grouped under (``section``),
* the statement text and, if present, the "Rationale:" text — separated
  even when they were written as a single run-on paragraph.

Nothing here calls an LLM; this is pure text processing so it behaves
identically no matter which architecture or provider produced the document.
"""

from __future__ import annotations

import re

from evaluation.grounding_schema import RequirementRecord

TOP_SECTION_RE = re.compile(r"\n##(?!#)[ \t]+(?P<heading>[^\n]+)\n")
SUBSECTION_RE = re.compile(r"\n###(?!#)[ \t]+(?P<heading>[^\n]+)\n")
_ID_RE = re.compile(
    r"(?:^|\n)[ \t]{0,3}(?:[-*+][ \t]+|\d+[.)][ \t]+)?#{0,4}[ \t]*\**_*[ \t]*"
    r"(?P<id>(?:FR|NFR)-\d+)\**_*"
    r"(?:[ \t]*\([^)\n]{0,60}\))?[ \t]*:[ \t]*"
)
_RATIONALE_SPLIT_RE = re.compile(r"(?i)\brationale\s*:\s*")
_EMPHASIS_RE = re.compile(r"^[_*#\s]+|[_*\s]+$")


def split_by_heading(markdown: str, pattern: re.Pattern[str]) -> list[tuple[str, str]]:
    """Split ``markdown`` at every heading matched by ``pattern``.

    Returns a list of ``(heading, body)`` pairs. Text before the first
    matched heading (if any) is dropped — callers only care about content
    grouped under a heading.
    """
    text = "\n" + markdown
    matches = list(pattern.finditer(text))
    out: list[tuple[str, str]] = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        out.append((m.group("heading").strip(), text[start:end]))
    return out


def _clean(text: str) -> str:
    """Strip Markdown emphasis/heading wrappers and surrounding whitespace."""
    return _EMPHASIS_RE.sub("", text.strip()).strip()


def _extract_from_subsection(section_heading: str, body: str) -> list[RequirementRecord]:
    """Extract every ``FR-NNN:``/``NFR-NNN:`` requirement out of one subsection body."""
    ids = list(_ID_RE.finditer(body))
    records: list[RequirementRecord] = []
    for i, m in enumerate(ids):
        start = m.end()
        end = ids[i + 1].start() if i + 1 < len(ids) else len(body)
        block = body[start:end]
        parts = _RATIONALE_SPLIT_RE.split(block, maxsplit=1)
        statement = _clean(parts[0])
        rationale = _clean(parts[1]) if len(parts) > 1 else ""
        if not statement:
            continue
        req_id = m.group("id")
        req_type = "nonfunctional" if req_id.upper().startswith("NFR") else "functional"
        records.append(RequirementRecord(
            id=req_id, section=section_heading, type=req_type,
            statement=statement, rationale=rationale,
        ))
    return records


def _matches_top_heading(heading: str, *, must_contain: str, must_not_contain: str = "") -> bool:
    h = heading.lower()
    if must_not_contain and must_not_contain in h:
        return False
    return must_contain in h


def parse_requirements(srs_markdown: str) -> list[RequirementRecord]:
    """Parse every functional and non-functional requirement out of an SRS.

    Args:
        srs_markdown: The full SRS Markdown, as produced by
            ``format_srs_markdown`` (any of the four architectures).

    Returns:
        All :class:`~evaluation.grounding_schema.RequirementRecord`\\ s found,
        in document order. Best-effort: a document with no parseable
        requirements returns an empty list rather than raising.
    """
    top_sections = split_by_heading(srs_markdown, TOP_SECTION_RE)
    records: list[RequirementRecord] = []
    for heading, body in top_sections:
        is_functional = _matches_top_heading(
            heading, must_contain="functional requirements", must_not_contain="non-functional"
        )
        is_nonfunctional = _matches_top_heading(heading, must_contain="non-functional requirements")
        if not (is_functional or is_nonfunctional):
            continue
        subsections = split_by_heading(body, SUBSECTION_RE)
        if not subsections:
            # No ### grouping used — treat the whole body as one subsection.
            subsections = [("(ungrouped)", body)]
        for sub_heading, sub_body in subsections:
            records.extend(_extract_from_subsection(sub_heading, sub_body))
    return records
