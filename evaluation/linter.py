"""Layer 1 of the grounded scoring stack: a deterministic SRS linter.

Pure Python, no LLM call, and — this is the point — no domain knowledge
either. It checks mechanical properties any SRS should have regardless of
whether it describes a restaurant, a cafe, or a smart-home hub, drawn from
IEEE/ISO/IEC 29148's requirement-quality characteristics (unambiguous,
verifiable/testable, complete, consistent) and the "weak phrase" checklists
common in requirements-quality tooling:

* **Ambiguity** — flags vague quantifiers and weasel words the style guide
  in ``config/prompts.py`` explicitly tells the generating agents to avoid
  ("fast", "user-friendly", "as appropriate", ...).
* **Testability** — a requirement is countable only if it carries a
  measurable criterion (a number, a unit, a percentage, a named standard)
  or an explicit ``TBD`` deferral; bare "shall" statements with no way to
  check them fail this.
* **Structure** — the required top-level sections exist and are non-empty.
* **Consistency** — no duplicate requirement ids, and every FR-NNN/NFR-NNN
  referenced from the risk section actually exists.

Swapping the domain (restaurant → cafe) never changes this module, because
it is never told what the domain is.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

from evaluation.grounding_schema import RequirementRecord
from evaluation.srs_parser import TOP_SECTION_RE, split_by_heading

IssueSeverity = Literal["major", "minor"]
IssueCategory = Literal["ambiguity", "testability", "structure", "consistency"]

#: Weak/ambiguous phrases the style guide asks agents to avoid. Word-boundary
#: matched, case-insensitive. Kept short and high-precision on purpose — a
#: linter that flags too eagerly gets ignored.
_AMBIGUOUS_PHRASES: tuple[str, ...] = (
    "fast", "user-friendly", "user friendly", "robust", "as appropriate",
    "as needed", "as necessary", "as required", "easy to use", "intuitive",
    "seamless", "seamlessly", "efficient", "efficiently", "reasonable",
    "adequate", "flexible", "several", "many", "some", "simple", "clean",
    "elegant", "quickly", "clearly", "properly", "significant", "minimal",
    "optimal", "state-of-the-art", "world-class", "high-quality",
    "good", "better", "best", "modern", "and/or", "etc.", "etc,", "tbd",
    "timely", "normal", "if practical", "if possible", "be able to",
)
_AMBIGUOUS_RE = re.compile(
    r"\b(" + "|".join(re.escape(p) for p in _AMBIGUOUS_PHRASES) + r")\b",
    re.IGNORECASE,
)

#: A requirement counts as "measurable" if it contains a number+unit,
#: a percentage, or a reference to a named standard.
_MEASURABLE_RE = re.compile(
    r"(\d+(\.\d+)?\s*(ms|milliseconds?|s|secs?|seconds?|mins?|minutes?|"
    r"hours?|hrs?|days?|%|percent|kb|mb|gb|requests?|users?|connections?))"
    r"|(\bISO\b|\bIEC\b|\bIEEE\b|\bWCAG\b|\bPCI[- ]DSS\b|\bGDPR\b|\bOWASP\b)",
    re.IGNORECASE,
)
_TBD_RE = re.compile(r"\bTBD\b", re.IGNORECASE)

#: A functional requirement is testable (in the IEEE 29148 sense — a
#: reader can write a pass/fail test case for it) if it commits to a
#: mandatory, unhedged action via one of these modal verbs. Non-functional
#: requirements are held to the stricter numeric/standard bar above,
#: matching the style guide's explicit "every NFR needs a measurable
#: criterion" rule in ``config/prompts.py``.
_MODAL_RE = re.compile(r"\b(shall|must|will)\b", re.IGNORECASE)

#: Top-level sections every SRS produced by ``format_srs_markdown`` must have.
_REQUIRED_TOP_SECTIONS: tuple[str, ...] = (
    "functional requirements",
    "non-functional requirements",
    "risks",
)

_REQ_ID_MENTION_RE = re.compile(r"\b(?:FR|NFR)-\d+\b")


@dataclass
class LintIssue:
    severity: IssueSeverity
    category: IssueCategory
    requirement_id: str | None
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "category": self.category,
            "requirement_id": self.requirement_id,
            "message": self.message,
        }


@dataclass
class LintResult:
    """Deterministic quality scores, each in ``[0, 1]`` where 1 is best."""

    ambiguity_score: float
    testability_score: float
    structure_score: float
    consistency_score: float
    issues: list[LintIssue] = field(default_factory=list)

    @property
    def quality_score(self) -> float:
        """Equal-weighted mean of the four sub-scores."""
        return (
            self.ambiguity_score + self.testability_score
            + self.structure_score + self.consistency_score
        ) / 4.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "ambiguity_score": round(self.ambiguity_score, 4),
            "testability_score": round(self.testability_score, 4),
            "structure_score": round(self.structure_score, 4),
            "consistency_score": round(self.consistency_score, 4),
            "quality_score": round(self.quality_score, 4),
            "issues": [i.to_dict() for i in self.issues],
        }


def _check_ambiguity(reqs: list[RequirementRecord]) -> tuple[float, list[LintIssue]]:
    if not reqs:
        return 1.0, []
    issues: list[LintIssue] = []
    flagged = 0
    for r in reqs:
        hits = sorted({m.group(1).lower() for m in _AMBIGUOUS_RE.finditer(r.statement)})
        if hits:
            flagged += 1
            issues.append(LintIssue(
                severity="minor", category="ambiguity", requirement_id=r.id,
                message=f"vague wording: {', '.join(hits)}",
            ))
    return 1.0 - (flagged / len(reqs)), issues


def _check_testability(reqs: list[RequirementRecord]) -> tuple[float, list[LintIssue]]:
    if not reqs:
        return 1.0, []
    issues: list[LintIssue] = []
    passed = 0
    for r in reqs:
        if r.type == "nonfunctional":
            ok = bool(_MEASURABLE_RE.search(r.statement) or _TBD_RE.search(r.statement))
            reason = ("no measurable acceptance criterion (number, unit, "
                       "percentage, or named standard) and no TBD deferral")
        else:
            ok = bool(_MODAL_RE.search(r.statement))
            reason = 'no mandatory modal verb ("shall"/"must"/"will") to test against'
        if ok:
            passed += 1
        else:
            issues.append(LintIssue(
                severity="major", category="testability", requirement_id=r.id,
                message=reason,
            ))
    return passed / len(reqs), issues


def _check_structure(srs_markdown: str) -> tuple[float, list[LintIssue]]:
    top_sections = split_by_heading(srs_markdown, TOP_SECTION_RE)
    headings_lower = [h.lower() for h, _ in top_sections]
    bodies_by_heading = {h.lower(): b for h, b in top_sections}
    issues: list[LintIssue] = []
    present = 0
    for required in _REQUIRED_TOP_SECTIONS:
        match = next((h for h in headings_lower if required in h), None)
        if match is None:
            issues.append(LintIssue(
                severity="major", category="structure", requirement_id=None,
                message=f"missing required section containing {required!r}",
            ))
            continue
        if not bodies_by_heading[match].strip():
            issues.append(LintIssue(
                severity="major", category="structure", requirement_id=None,
                message=f"section {match!r} is present but empty",
            ))
            continue
        present += 1
    return present / len(_REQUIRED_TOP_SECTIONS), issues


def _check_consistency(
    srs_markdown: str, reqs: list[RequirementRecord],
) -> tuple[float, list[LintIssue]]:
    issues: list[LintIssue] = []
    seen: dict[str, int] = {}
    for r in reqs:
        seen[r.id] = seen.get(r.id, 0) + 1
    duplicates = {rid: n for rid, n in seen.items() if n > 1}
    for rid, n in duplicates.items():
        issues.append(LintIssue(
            severity="major", category="consistency", requirement_id=rid,
            message=f"requirement id used {n} times",
        ))

    known_ids = set(seen)
    mentioned = set(_REQ_ID_MENTION_RE.findall(srs_markdown))
    dangling = mentioned - known_ids
    for rid in sorted(dangling):
        issues.append(LintIssue(
            severity="minor", category="consistency", requirement_id=rid,
            message="referenced elsewhere in the document but never defined",
        ))

    total_checks = len(reqs) + len(mentioned) or 1
    penalty = sum(n - 1 for n in duplicates.values()) + len(dangling)
    score = max(0.0, 1.0 - penalty / total_checks)
    return score, issues


def lint_srs(srs_markdown: str, requirements: list[RequirementRecord]) -> LintResult:
    """Run every Layer-1 check and return the combined :class:`LintResult`.

    Args:
        srs_markdown: The full SRS Markdown document.
        requirements: The requirements already parsed out of it by
            :func:`evaluation.srs_parser.parse_requirements` (passed in
            rather than re-parsed here, so caller and linter always agree
            on the same requirement set).

    Returns:
        A populated :class:`LintResult`.
    """
    ambiguity_score, ambiguity_issues = _check_ambiguity(requirements)
    testability_score, testability_issues = _check_testability(requirements)
    structure_score, structure_issues = _check_structure(srs_markdown)
    consistency_score, consistency_issues = _check_consistency(srs_markdown, requirements)
    return LintResult(
        ambiguity_score=ambiguity_score,
        testability_score=testability_score,
        structure_score=structure_score,
        consistency_score=consistency_score,
        issues=[*ambiguity_issues, *testability_issues, *structure_issues, *consistency_issues],
    )
