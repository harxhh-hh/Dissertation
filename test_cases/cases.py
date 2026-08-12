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


#: The four test cases from the project brief, plus six additional cases
#: added to broaden domain coverage for the architecture comparison.
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
    TestCase(
        case_id="TC-05_telehealth_platform",
        title="Telehealth consultation platform",
        description=(
            "Build a telehealth platform connecting patients with doctors "
            "for virtual consultations. Patients should be able to book "
            "appointments, join secure video calls, and view their "
            "prescriptions and visit history. Doctors need to write and "
            "send e-prescriptions, review a patient's medical history "
            "before a call, and document consultation notes. The "
            "platform should also support insurance verification and "
            "billing."
        ),
    ),
    TestCase(
        case_id="TC-06_mobile_banking_app",
        title="Mobile banking app",
        description=(
            "Develop a mobile banking app for a retail bank's customers. "
            "Customers should be able to check account balances, "
            "transfer money between accounts, pay bills, and deposit "
            "checks by photographing them. New customers should be able "
            "to open an account and verify their identity entirely "
            "within the app. The bank's fraud team needs to review "
            "flagged transactions and freeze accounts when necessary."
        ),
    ),
    TestCase(
        case_id="TC-07_ecommerce_marketplace",
        title="Online seller marketplace",
        description=(
            "Create an online marketplace where independent sellers can "
            "list and sell products directly to consumers. Sellers "
            "should be able to create storefronts, manage inventory, and "
            "fulfil orders. Buyers should be able to search and filter "
            "products, read and leave reviews, and track shipments. The "
            "platform should also handle payment processing between "
            "buyers and sellers and provide a way to resolve disputes."
        ),
    ),
    TestCase(
        case_id="TC-08_ride_sharing_platform",
        title="Ride-sharing platform",
        description=(
            "Build a ride-sharing app that connects passengers with "
            "nearby drivers. Passengers should be able to request a "
            "ride, see the driver's estimated arrival time, track the "
            "trip on a live map, and pay automatically at the end of the "
            "trip. Drivers should be able to accept or decline ride "
            "requests, navigate to pickup and drop-off locations, and "
            "view their earnings. The platform should also let "
            "passengers rate drivers and flag safety concerns."
        ),
    ),
    TestCase(
        case_id="TC-09_donation_platform",
        title="Charity donation and crowdfunding platform",
        description=(
            "Develop an online donation platform for charities to run "
            "fundraising campaigns. Campaign organisers should be able "
            "to create a campaign page with a fundraising goal, post "
            "updates, and receive donations from supporters. Donors "
            "should be able to give one-off or recurring donations, "
            "choose to donate anonymously, and download a receipt for "
            "tax purposes. The platform should also give charities a "
            "dashboard to track total funds raised and donor activity."
        ),
    ),
    TestCase(
        case_id="TC-10_recruitment_platform",
        title="Applicant tracking system",
        description=(
            "Create an applicant tracking system for a company's HR team "
            "to manage hiring. Recruiters should be able to post job "
            "openings, review incoming applications, and move candidates "
            "through stages of the hiring pipeline. Candidates should be "
            "able to create a profile, apply to open roles, upload a "
            "resume, and schedule interviews. The system should also "
            "support running background checks on shortlisted candidates "
            "and collecting structured interview feedback from hiring "
            "panels."
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
