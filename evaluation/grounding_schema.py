"""Shared record schema for grounded, deterministic architecture comparison.

Two record types, both keyed by ``(section, id)`` so that matching an SRS
against a knowledge base is a lookup, never a document diff:

* :class:`RequirementRecord` — one atomic requirement extracted from a
  generated SRS (see :mod:`evaluation.srs_parser`).
* :class:`KBFact` — one atomic, sourced ground-truth fact for a domain,
  authored once and injected byte-identical into every architecture.

:class:`DomainKB` groups a domain's facts and carries a ``status`` so an
unapproved (freshly researched) knowledge base can never silently be
treated as ground truth — see the ``draft`` / ``approved`` gate in
``ui/app.py`` and the project's Gate A workflow.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

RequirementType = Literal["functional", "nonfunctional"]
GroundingStatus = Literal["grounded", "inferred", "unknown"]
KBStatus = Literal["draft", "approved"]
FactCategory = Literal["functional", "nonfunctional", "risk"]


@dataclass(frozen=True)
class KBFact:
    """One atomic, sourced ground-truth fact for a domain.

    Attributes:
        id: Stable identifier, e.g. ``KB-RESTAURANT-012``.
        section: Human-readable grouping that mirrors the SRS section
            structure (e.g. ``"Payments"``, ``"Food safety"``).
        category: Which SRS section this fact should be checked against.
        weight: Relative importance used by the coverage score; ``1.0`` is
            the default, higher values matter more.
        statement: The fact itself, phrased as a checkable claim.
        source_title: Human-readable name of the source (a standard, a
            regulation, a published spec).
        source_url: URL of the source, shown to the user at Gate A.
    """

    id: str
    section: str
    category: FactCategory
    statement: str
    source_title: str
    source_url: str
    weight: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "section": self.section,
            "category": self.category,
            "statement": self.statement,
            "source_title": self.source_title,
            "source_url": self.source_url,
            "weight": self.weight,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "KBFact":
        return cls(
            id=str(data["id"]),
            section=str(data["section"]),
            category=str(data["category"]),  # type: ignore[arg-type]
            statement=str(data["statement"]),
            source_title=str(data["source_title"]),
            source_url=str(data["source_url"]),
            weight=float(data.get("weight", 1.0)),
        )


@dataclass
class DomainKB:
    """A domain's curated knowledge base: facts plus provenance and status.

    Attributes:
        domain: Canonical domain key, e.g. ``"restaurant"``.
        status: ``"draft"`` until a human approves it at Gate A;
            grounded scoring refuses to run against a ``"draft"`` KB.
        facts: The domain's gold facts.
        matched_case_ids: Test-case IDs this domain applies to, used for
            deterministic domain detection on the four built-in cases.
        keywords: Fallback keyword list used to detect this domain for
            ad hoc / unknown cases when no case_id match exists.
        researched_at: ISO-8601 timestamp of when the draft was produced,
            for the Gate A review screen.
    """

    domain: str
    status: KBStatus = "draft"
    facts: list[KBFact] = field(default_factory=list)
    matched_case_ids: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    researched_at: str = ""

    def facts_by_category(self, category: FactCategory) -> list[KBFact]:
        return [f for f in self.facts if f.category == category]

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "status": self.status,
            "facts": [f.to_dict() for f in self.facts],
            "matched_case_ids": list(self.matched_case_ids),
            "keywords": list(self.keywords),
            "researched_at": self.researched_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DomainKB":
        return cls(
            domain=str(data["domain"]),
            status=str(data.get("status", "draft")),  # type: ignore[arg-type]
            facts=[KBFact.from_dict(f) for f in data.get("facts", [])],
            matched_case_ids=list(data.get("matched_case_ids", [])),
            keywords=list(data.get("keywords", [])),
            researched_at=str(data.get("researched_at", "")),
        )


@dataclass
class RequirementRecord:
    """One atomic requirement extracted from a generated SRS document.

    Attributes:
        id: The requirement's own identifier as written by the agent,
            e.g. ``FR-003`` or ``NFR-011``.
        section: The ``###`` subheading the requirement was grouped
            under in the source Markdown (e.g. ``"Payments"``).
        type: ``"functional"`` or ``"nonfunctional"``, from which
            top-level ``##`` section it was parsed out of.
        statement: The requirement statement, verbatim.
        rationale: The agent's own "Rationale:" line, if present.
        grounding: Set by the Layer-2 grader: ``"grounded"`` when it
            matches a KB fact, ``"inferred"`` when it doesn't and the
            grader found no contradiction, ``"unknown"`` pre-grading.
        kb_ref: The :class:`KBFact` id this requirement was matched to,
            if any.
    """

    id: str
    section: str
    type: RequirementType
    statement: str
    rationale: str = ""
    grounding: GroundingStatus = "unknown"
    kb_ref: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "section": self.section,
            "type": self.type,
            "statement": self.statement,
            "rationale": self.rationale,
            "grounding": self.grounding,
            "kb_ref": self.kb_ref,
        }
