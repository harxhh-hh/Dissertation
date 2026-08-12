"""Domain knowledge-base registry.

Loads every ``*.json`` file under ``knowledge_bases/`` into a
:class:`~evaluation.grounding_schema.DomainKB` and provides deterministic
domain detection: an exact ``case_id`` match for the four built-in test
cases, falling back to a keyword match in the free-text description for
ad hoc cases. No LLM call and no fuzzy matching — a case either matches a
known domain or it doesn't, and "it doesn't" is exactly the signal that
should trigger the HITL gate in the UI (see ``ui/app.py``), never a guess.

A KB with ``status == "draft"`` is a knowledge base that has been
researched but not yet approved by a human at Gate A (see the project's
grounding workflow). It is loaded and can be inspected or used for a
provisional/preview run, but every caller that runs grounded scoring for
real must check :meth:`DomainKB.status` and label results accordingly —
this module does not hide or refuse draft KBs, it only reports status
honestly so nothing downstream can silently treat a draft as ground truth.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from evaluation.grounding_schema import DomainKB

#: Directory holding one ``<domain>.json`` file per domain, relative to the
#: repository root (this file lives in ``evaluation/``, so go up one level).
KNOWLEDGE_BASE_DIR: Path = Path(__file__).resolve().parent.parent / "knowledge_bases"

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def load_all(directory: Path = KNOWLEDGE_BASE_DIR) -> dict[str, DomainKB]:
    """Load every domain KB JSON file in ``directory``.

    Args:
        directory: Directory to scan for ``*.json`` files. Defaults to the
            repository's ``knowledge_bases/`` directory.

    Returns:
        Mapping from domain key to :class:`DomainKB`. An empty mapping if
        the directory doesn't exist yet or holds no KB files.
    """
    if not directory.exists():
        return {}
    kbs: dict[str, DomainKB] = {}
    for path in sorted(directory.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        kb = DomainKB.from_dict(data)
        kbs[kb.domain] = kb
    return kbs


def get_domain_kb(domain: str, directory: Path = KNOWLEDGE_BASE_DIR) -> DomainKB | None:
    """Load a single domain's KB by name, or ``None`` if it doesn't exist."""
    return load_all(directory).get(domain)


def detect_domain(
    *, case_id: str, description: str, directory: Path = KNOWLEDGE_BASE_DIR,
) -> str | None:
    """Deterministically resolve a test case to a known domain, if any.

    Resolution order:

    1. Exact ``case_id`` membership in a KB's ``matched_case_ids`` — this
       is how the four built-in test cases resolve, with zero ambiguity.
    2. Keyword overlap between the description and a KB's ``keywords``
       list — used for ad hoc / "Try your own idea" cases. Requires at
       least one whole-word match; the KB with the most matched keywords
       wins, ties broken by domain name for determinism.

    Args:
        case_id: The test case identifier (may be an ad hoc synthetic id).
        description: The case's free-text system description.
        directory: Knowledge-base directory to search.

    Returns:
        The matched domain key, or ``None`` if nothing matches — the
        caller (the UI) is responsible for treating ``None`` as "trigger
        the HITL gate", not for guessing further.
    """
    kbs = load_all(directory)
    for kb in kbs.values():
        if case_id in kb.matched_case_ids:
            return kb.domain

    desc_tokens = _tokenize(description)
    if not desc_tokens:
        return None

    best_domain: str | None = None
    best_score = 0
    for domain in sorted(kbs):
        kb = kbs[domain]
        kw_tokens = {kw.lower() for kw in kb.keywords}
        score = sum(1 for kw in kw_tokens if kw in desc_tokens or kw in description.lower())
        if score > best_score:
            best_score = score
            best_domain = domain
    return best_domain if best_score > 0 else None


def list_domains(directory: Path = KNOWLEDGE_BASE_DIR) -> list[str]:
    """Return every known domain key, sorted."""
    return sorted(load_all(directory))


def save_domain_kb(kb: DomainKB, directory: Path = KNOWLEDGE_BASE_DIR) -> Path:
    """Write ``kb`` to ``<directory>/<domain>.json`` and return the path.

    Used by the Gate A / Gate B research workflow to persist a freshly
    drafted KB (status ``"draft"``) or to flip an approved KB's status.
    Always writes the full object, so approving a KB is just loading it,
    setting ``status = "approved"``, and calling this again.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{kb.domain}.json"
    path.write_text(json.dumps(kb.to_dict(), indent=2) + "\n", encoding="utf-8")
    return path
