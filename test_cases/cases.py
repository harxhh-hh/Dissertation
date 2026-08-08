"""Test cases: the natural-language system descriptions used as inputs.

Each :class:`TestCase` bundles a stable identifier, a short human-readable
title, and the description text that is fed to every architecture and to
the baseline.

Keeping the cases as immutable module-level constants (rather than reading
them from a file at run time) has two upsides:

* The exact text used for a run is fixed at import time and is captured
  in the interaction log verbatim, so nothing about the input can drift
  between runs.
* A moderator can diff two runs by inspecting only ``config.json`` and
  the two artefact trees; no additional input files need to be consulted.

The identifiers are filesystem-safe (letters, digits, underscore, hyphen)
because they end up in output-file names.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final, Iterator


@dataclass(frozen=True)
class TestCase:
    """One natural-language input, ready to feed to an architecture.

    Attributes:
        case_id: Filesystem-safe identifier. Appears in interaction log
            lines and output file names.
        title: Human-readable title, used in report headings.
        description: The description text sent to the model, verbatim.
    """

    case_id: str
    title: str
    description: str


#: The four test cases from the project brief.
TEST_CASES: Final[tuple[TestCase, ...]] = (
    TestCase(
        case_id="TC-01_restaurant_app",
        title="Restaurant chain mobile app",
        description=(
            "Design a mobile app for a small restaurant chain. Customers "
            "should be able to browse the menu, place orders, and pay "
            "through the app. Restaurant staff should be able to view "
            "incoming orders and mark them as completed. Managers need "
            "access to daily sales reports and the ability to update the "
            "menu."
        ),
    ),
    TestCase(
        case_id="TC-02_project_management",
        title="Web-based project management tool",
        description=(
            "Develop a web-based project management tool for a software "
            "development team. The tool should allow users to create "
            "projects, assign tasks to team members, track progress, and "
            "generate reports. It should also support file sharing and "
            "integration with version control systems like Git."
        ),
    ),
    TestCase(
        case_id="TC-03_elearning_platform",
        title="University e-learning platform",
        description=(
            "Create an e-learning platform for a university. The platform "
            "should enable instructors to create and manage courses, "
            "upload learning materials, and conduct online assessments. "
            "Students should be able to enroll in courses, access course "
            "content, participate in discussions, and submit assignments."
        ),
    ),
    TestCase(
        case_id="TC-04_smart_home",
        title="Smart home automation system",
        description=(
            "Build a smart home automation system that allows users to "
            "control and monitor various devices through a mobile app. "
            "The system should support devices such as lights, "
            "thermostats, security cameras, and door locks. It should "
            "also provide energy consumption insights and allow users to "
            "create custom automation routines."
        ),
    ),
)

# Enforce that identifiers are actually filesystem-safe. This is a
# defensive check: a typo above would otherwise land in a file path.
_ID_RE: Final = re.compile(r"^[A-Za-z0-9_\-]+$")
for _case in TEST_CASES:
    assert _ID_RE.match(_case.case_id), (
        f"Test case identifier {_case.case_id!r} contains characters that "
        "are not filesystem-safe."
    )


def iter_cases() -> Iterator[TestCase]:
    """Yield the test cases in declaration order.

    Wrapping the tuple in a function documents that other stages should
    iterate through this helper rather than importing :data:`TEST_CASES`
    directly, so a future change that filters or shuffles cases has one
    place to land.
    """
    yield from TEST_CASES


def get_case(case_id: str) -> TestCase:
    """Look up one test case by identifier.

    Args:
        case_id: The case identifier as it appears in the log.

    Returns:
        The matching :class:`TestCase`.

    Raises:
        KeyError: If no case with that identifier exists.
    """
    for case in TEST_CASES:
        if case.case_id == case_id:
            return case
    raise KeyError(f"No test case with case_id={case_id!r}")
