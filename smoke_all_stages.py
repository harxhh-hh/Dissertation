"""Offline smoke test for the whole pipeline.

Verifies stages 2-5 end-to-end WITHOUT any real network call. Every LLM
interaction goes through a :class:`FakeClient` that mimics the
:class:`~utils.llm_client.LLMClient` contract and returns canned but
architecture-aware responses. This lets us catch:

* Import errors after a refactor (all modules load).
* Contract mismatches between the client and its callers.
* Regressions in the SRS Markdown format.
* Regressions in the interaction-log schema (architecture / agent /
  phase attribution stays correct).
* Rater-export invariants (anonymisation, deterministic shuffle,
  mapping kept out of the documents folder).
* Report-generator invariants (all four conditions, cost table
  excludes evaluator).
* Security invariant: no API-key sentinel leaks into any artefact.

This smoke test does NOT verify that the Groq or Anthropic backends
actually work against a live API. That is what the live-run test
(``run_experiment.py`` against a real key) is for.
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

import architectures.debate as debate_mod  # noqa: E402
import architectures.hierarchical as hier_mod  # noqa: E402
import architectures.peer_to_peer as p2p_mod  # noqa: E402
import baseline_single_prompt as base_mod  # noqa: E402
from config import prompts  # noqa: E402
from config.settings import Settings  # noqa: E402
from evaluation.export_for_raters import SrsArtifact, export_for_raters  # noqa: E402
from evaluation.generate_report import generate_report  # noqa: E402
from evaluation.rubric import score_srs, write_evaluation, aggregate_by_architecture  # noqa: E402
from test_cases.cases import TEST_CASES, get_case  # noqa: E402
from utils.llm_client import LLMCallResult  # noqa: E402
from utils.logging import ExperimentLogger, TokenUsage  # noqa: E402

failures: list[str] = []


def check(label: str, condition: bool) -> None:
    """Print PASS/FAIL for one invariant and record failures."""
    print(f"  [{'PASS' if condition else 'FAIL'}] {label}")
    if not condition:
        failures.append(label)


# --------------------------------------------------------------------------
# Fake LLM client
# --------------------------------------------------------------------------


class FakeClient:
    """Records every call and returns responses appropriate to the phase.

    Matches the :class:`~utils.llm_client.LLMClient` public contract so it
    is a drop-in for the agents. Deliberately not a subclass, so the
    smoke test cannot accidentally pass because of inherited behaviour.
    """

    def __init__(self, logger: ExperimentLogger, settings: Settings) -> None:
        self._logger = logger
        self._settings = settings
        self._verification_counters: dict[tuple[str, str, int], int] = {}
        self.calls: list[dict[str, Any]] = []

    def call(self, *, architecture, agent, case_id, repetition, phase,
             system_prompt, user_prompt, output_schema=None) -> LLMCallResult:
        text, parsed = self._response_for(
            architecture, agent, phase, output_schema, case_id, repetition
        )
        usage = TokenUsage(input_tokens=100, output_tokens=200,
                           cache_read_input_tokens=500)
        interaction = self._logger.log_interaction(
            architecture=architecture, agent=agent, case_id=case_id,
            repetition=repetition, phase=phase, model=self._settings.model_id,
            system_prompt=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            request_params={"provider": self._settings.llm_provider,
                            "model": self._settings.model_id,
                            "max_tokens": self._settings.max_tokens},
            response_text=text, stop_reason="end_turn", usage=usage,
            latency_ms=1.0,
        )
        self.calls.append({
            "architecture": architecture, "agent": agent, "phase": phase,
            "case_id": case_id, "repetition": repetition,
        })
        return LLMCallResult(
            text=text, parsed_json=parsed, stop_reason="end_turn",
            usage=usage, interaction=interaction,
        )

    def _response_for(self, architecture, agent, phase, schema, case_id, rep):
        if agent == "orchestrator":
            brief = {
                "system_type": "test system",
                "stakeholders": [
                    {"role": "User", "responsibilities": "Uses the system."},
                ],
                "scope_summary": "Test scope.",
                "assumptions": ["Payments handled externally."],
                "delegation_focus": {
                    "functional": "capabilities",
                    "nonfunctional": "performance",
                    "risk": "constraints",
                },
            }
            return json.dumps(brief), brief

        if agent == "baseline":
            body = (
                "## Overview\nThe system is a test system.\n\n"
                "## Functional requirements\n### Core\n"
                "FR-001: The system shall serve a request.\n"
                "*Rationale:* baseline test.\n\n"
                "## Non-functional requirements\n### Performance\n"
                "NFR-001: The system shall respond within 2 s at p95.\n"
                "*Rationale:* baseline test.\n\n"
                "## Risks, ambiguities, and open questions\n"
                "### Constraints\n- None identified.\n"
                "### Risks\n- Downtime — Mitigation: monitor.\n"
                "### Ambiguities\n- None identified.\n"
                "### Open questions\n- None identified.\n"
            )
            return body, None

        if agent == "functional_requirements":
            return (
                "### Core capabilities\n"
                "FR-001: The system shall serve a request.\n"
                "*Rationale:* covers the entry point.\n"
            ), None

        if agent == "nonfunctional_requirements":
            return (
                "### Performance\n"
                "NFR-001: The system shall respond within 2 s at p95.\n"
                "*Rationale:* interactivity.\n"
            ), None

        if agent == "risk_clarification":
            return (
                "### Constraints\n- None identified.\n\n"
                "### Risks\n- Load spike — Mitigation: autoscale.\n\n"
                "### Ambiguities\n- None identified.\n\n"
                "### Open questions\n- None identified.\n"
            ), None

        if agent == "verification":
            if phase.startswith("arbitration"):
                verdict = {
                    "chosen_section_markdown":
                        "### Arbitrated section\n"
                        "FR-001: The system shall serve a request (arbitrated).\n"
                        "*Rationale:* arbiter picked this.\n"
                        if "functional" in phase else
                        "### Arbitrated NFR section\n"
                        "NFR-001: The system shall respond within 2 s at p95 (arbitrated).\n"
                        "*Rationale:* arbiter picked this.\n"
                        if "nonfunctional" in phase else
                        "### Arbitrated risk section\n"
                        "### Constraints\n- None identified.\n\n"
                        "### Risks\n- Load spike — Mitigation: autoscale.\n\n"
                        "### Ambiguities\n- None identified.\n\n"
                        "### Open questions\n- None identified.\n",
                    "verdict": "position_a",
                    "rationale": "Position A slightly clearer.",
                }
                return json.dumps(verdict), verdict
            key = (architecture, case_id, rep)
            self._verification_counters[key] = self._verification_counters.get(key, 0) + 1
            round_no = self._verification_counters[key]
            if round_no == 1 and architecture == "hierarchical":
                v = {
                    "verdict": "revision_required",
                    "summary": "Ordering coverage weak.",
                    "issues": [{
                        "severity": "major", "category": "completeness",
                        "affected_section": "functional",
                        "affected_ids": ["FR-001"],
                        "description": "Ordering is not covered.",
                    }],
                }
            else:
                v = {"verdict": "pass", "summary": "Looks fine.", "issues": []}
            return json.dumps(v), v

        if agent == "evaluator":
            verdict = {
                "completeness": {"score": 4, "justification": "Covers core.", "issues": []},
                "consistency": {"score": 5, "justification": "No contradictions.", "issues": []},
                "testability": {"score": 4, "justification": "Most items testable.", "issues": []},
                "clarity": {"score": 5, "justification": "Reads clearly.", "issues": []},
                "overall": "Acceptable baseline-quality SRS.",
            }
            return json.dumps(verdict), verdict

        raise AssertionError(f"unhandled agent={agent} phase={phase}")


# --------------------------------------------------------------------------
# Setup
# --------------------------------------------------------------------------
tmpdir = Path(tempfile.mkdtemp(prefix="stage_all_smoke_"))

# Exercise the provider-aware settings path with a Groq configuration
# (matching the user's current .env). The smoke test bypasses real API
# calls via FakeClient, so the sentinel below never leaves this process.
os.environ.update({
    "LLM_PROVIDER": "groq",
    "GROQ_API_KEY": "gsk-SECRET-DO-NOT-LEAK",
    "ANTHROPIC_API_KEY": "",
    "MODEL_ID": "llama-3.3-70b-versatile",
    "MAX_TOKENS": "8000",
    "EFFORT": "high",
    "THINKING_MODE": "adaptive",
    "RANDOM_SEED": "42",
    "MAX_REVISION_ROUNDS": "1",
    "REPETITIONS": "1",
    "OUTPUT_DIR": str(tmpdir),
    "LOG_LEVEL": "WARNING",
})
settings = Settings.from_env(env_file=Path("/nonexistent"), run_id="allstages")
settings.apply_seed()

print("\n1. Test cases module")
case_ids = [c.case_id for c in TEST_CASES]
check("10 test cases defined", len(TEST_CASES) == 10)
check("all case IDs unique", len(set(case_ids)) == 10)
check("get_case looks up by id",
      get_case("TC-01_restaurant_app").title == "Restaurant chain mobile app")

print("\n2. Provider-aware Settings")
check("Settings.llm_provider is 'groq'", settings.llm_provider == "groq")
check("Groq key stored", settings.groq_api_key == "gsk-SECRET-DO-NOT-LEAK")
check("Anthropic key empty when unused", settings.anthropic_api_key == "")
check("__repr__ redacts groq key",
      "SECRET-DO-NOT-LEAK" not in repr(settings) and "<redacted>" in repr(settings))
check("to_dict redacts both keys",
      settings.to_dict()["groq_api_key"] == "<redacted>"
      and settings.to_dict()["anthropic_api_key"] == "<redacted>")

print("\n3. All four architectures run end-to-end")
srs_paths: dict[tuple[str, str, int], Path] = {}
with ExperimentLogger.for_run(settings) as logger:
    client = FakeClient(logger, settings)
    case = get_case("TC-01_restaurant_app")

    # Baseline
    base_res = base_mod.run_baseline(
        case.description, case_id=case.case_id, repetition=0,
        client=client, settings=settings, logger=logger,
    )
    check("baseline produced SRS", "# Software Requirements" in base_res.srs_markdown)
    check("baseline SRS carries architecture metadata",
          "baseline_single_prompt" in base_res.srs_markdown)

    # Hierarchical (with revision path)
    hier_res = hier_mod.run_hierarchical(
        case.description, case_id=case.case_id, repetition=0,
        client=client, settings=settings, logger=logger,
    )
    check("hierarchical revised at least once", hier_res.revised is True)
    check("hierarchical had exactly 2 verification rounds",
          len(hier_res.verification_rounds) == 2)
    check("hierarchical SRS contains all headers",
          all(h in hier_res.srs_markdown for h in [
              "## Functional requirements", "## Non-functional requirements",
              "## Risks, ambiguities, and open questions",
              "## Verification history",
          ]))

    # Peer-to-peer
    p2p_res = p2p_mod.run_peer_to_peer(
        case.description, case_id=case.case_id, repetition=0,
        client=client, settings=settings, logger=logger,
    )
    check("peer_to_peer produced SRS", "# Software Requirements" in p2p_res.srs_markdown)
    check("peer_to_peer used peer_review phase",
          any(c["architecture"] == "peer_to_peer" and c["phase"] == "peer_review"
              for c in client.calls))

    # Debate
    deb_res = debate_mod.run_debate(
        case.description, case_id=case.case_id, repetition=0,
        client=client, settings=settings, logger=logger,
    )
    check("debate produced SRS", "# Software Requirements" in deb_res.srs_markdown)
    check("debate ran 6 initial position calls (2 per specialist)",
          sum(1 for c in client.calls
              if c["architecture"] == "debate"
              and c["phase"] in ("initial_A", "initial_B")) == 6)
    check("debate ran 6 rebuttal calls",
          sum(1 for c in client.calls
              if c["architecture"] == "debate"
              and c["phase"] in ("rebuttal_A", "rebuttal_B")) == 6)
    check("debate ran 3 arbitrations",
          sum(1 for c in client.calls
              if c["architecture"] == "debate" and c["phase"].startswith("arbitration_")) == 3)
    check("3 debate arbitrations recorded",
          len(deb_res.arbitrations) == 3
          and {a.section for a in deb_res.arbitrations}
          == {"functional", "nonfunctional", "risk"})

    print("\n4. Persist artefacts, then run rater export + evaluation")

    run_dir = settings.run_dir
    for arch, res in [
        ("baseline_single_prompt", base_res), ("hierarchical", hier_res),
        ("peer_to_peer", p2p_res), ("debate", deb_res),
    ]:
        slug = f"{res.case_id}_{arch}_rep{res.repetition}"
        p = run_dir / f"srs_{slug}.md"
        p.write_text(res.srs_markdown, encoding="utf-8")
        srs_paths[(arch, res.case_id, res.repetition)] = p

    artefacts = [SrsArtifact(
        case_id=case_id, architecture=arch, repetition=rep,
        srs_path=p, description=get_case(case_id).description,
    ) for (arch, case_id, rep), p in srs_paths.items()]
    export = export_for_raters(artefacts, settings=settings)
    check("rater export folder exists", export.export_dir.is_dir())
    check("rater export README exists", (export.export_dir / "README.md").is_file())
    check("rater export mapping.json exists",
          (export.export_dir / "mapping.json").is_file())
    check("rater export scoring_sheet.csv exists",
          (export.export_dir / "scoring_sheet.csv").is_file())
    check("mapping.json outside documents dir (not accidentally shared)",
          not (export.export_dir / "documents" / "mapping.json").exists())
    check("all 4 documents exported",
          len(list((export.export_dir / "documents").glob("DOC-*.md"))) == 4)
    check("anonymised docs do not contain architecture identifiers",
          not any(("hierarchical" in p.read_text() or "debate" in p.read_text()
                   or "peer_to_peer" in p.read_text()
                   or "baseline_single_prompt" in p.read_text())
                  for p in (export.export_dir / "documents").glob("DOC-*.md")))
    check("anonymised docs contain input description",
          all("## Input description" in p.read_text()
              for p in (export.export_dir / "documents").glob("DOC-*.md")))

    export2 = export_for_raters(artefacts, settings=settings,
                                 export_root=tmpdir / "second")
    m1 = json.loads((export.export_dir / "mapping.json").read_text())
    m2 = json.loads((export2.export_dir / "mapping.json").read_text())
    check("shuffle is deterministic under the seed",
          [e["anonymous_id"] + e["case_id"] + e["architecture"]
           for e in m1["entries"]]
          == [e["anonymous_id"] + e["case_id"] + e["architecture"]
              for e in m2["entries"]])

    records = []
    for (arch, cid, rep), p in srs_paths.items():
        rec = score_srs(
            case_id=cid, architecture=arch, repetition=rep,
            description=get_case(cid).description,
            srs_markdown=p.read_text(),
            client=client,
        )
        records.append(rec)
    write_evaluation(records, run_dir / "evaluation.json")
    check("evaluation.json written", (run_dir / "evaluation.json").is_file())
    check("evaluation covers all 4 SRS", len(records) == 4)
    agg = aggregate_by_architecture(records)
    check("aggregate has 4 architectures", len(agg) == 4)
    for arch in ("baseline_single_prompt", "hierarchical", "peer_to_peer", "debate"):
        check(f"aggregate contains {arch}", arch in agg)
        check(f"aggregate {arch} has all dimensions",
              set(agg[arch]) == set(prompts.EVALUATION_DIMENSIONS))

    (run_dir / "run_manifest.json").write_text(json.dumps({
        "run_id": settings.run_id, "settings": settings.to_dict(),
        "outcomes": [
            {"architecture": arch, "case_id": cid, "repetition": rep,
             "srs_path": str(p), "error": None}
            for (arch, cid, rep), p in srs_paths.items()
        ],
    }, indent=2, sort_keys=True), encoding="utf-8")

print("\n5. Report generator")
report_path = generate_report(run_dir)
check("analysis_report.md written", report_path.is_file())
report = report_path.read_text()
check("report contains a scores table header", "Rubric scores by architecture" in report)
check("report contains cost/latency table", "Cost and latency by architecture" in report)
check("report mentions all 4 architectures",
      all(a in report for a in
          ("baseline_single_prompt", "hierarchical", "peer_to_peer", "debate")))
check("report excludes evaluator calls from cost table",
      "| evaluation " not in report)

print("\n6. Security invariants")
def _iter_artefact_texts():
    for p in run_dir.rglob("*"):
        if p.is_file() and p.suffix in (".md", ".json", ".jsonl", ".log", ".csv"):
            try:
                yield p, p.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                pass

leaks = [str(p) for p, text in _iter_artefact_texts() if "SECRET-DO-NOT-LEAK" in text]
check("API key does not appear in ANY artefact", leaks == [])

records = list(logger.iter_interactions())
by_arch = {}
for r in records:
    by_arch.setdefault(r["architecture"], []).append(r)
check("evaluation calls are tagged architecture='evaluation'",
      len(by_arch.get("evaluation", [])) == 4)
check("baseline architecture recorded", "baseline_single_prompt" in by_arch)
for arch in ("hierarchical", "peer_to_peer", "debate"):
    check(f"{arch}: every call carries composed system prompt (protocol block)",
          all("--- Protocol context ---" in r["system_prompt"] for r in by_arch[arch]))

banned = ["hierarchical", "peer-to-peer", "debate", "baseline"]
for role, txt in [
    ("ORCHESTRATOR_SYSTEM", prompts.ORCHESTRATOR_SYSTEM),
    ("FUNCTIONAL_SYSTEM", prompts.FUNCTIONAL_SYSTEM),
    ("NONFUNCTIONAL_SYSTEM", prompts.NONFUNCTIONAL_SYSTEM),
    ("RISK_SYSTEM", prompts.RISK_SYSTEM),
    ("VERIFICATION_SYSTEM", prompts.VERIFICATION_SYSTEM),
    ("EVALUATOR_SYSTEM", prompts.EVALUATOR_SYSTEM),
]:
    for term in banned:
        check(f"{role} does not mention {term!r}", term not in txt.lower())

for header in ("## Overview", "## Functional requirements",
               "## Non-functional requirements",
               "## Risks, ambiguities, and open questions"):
    check(f"BASELINE_SYSTEM instructs the {header!r} section",
          header in prompts.BASELINE_SYSTEM)

with (export.export_dir / "scoring_sheet.csv").open() as fh:
    reader = csv.reader(fh)
    headers = next(reader)
check("scoring_sheet.csv column headers",
      headers == ["anonymous_id"]
      + [f"{d}_score" for d in prompts.EVALUATION_DIMENSIONS]
      + ["overall_comment"])

one_anon = next((export.export_dir / "documents").glob("DOC-*.md")).read_text()
check("anonymised docs do NOT contain '## Run metadata'",
      "## Run metadata" not in one_anon)
check("anonymised docs DO contain '## Input description'",
      "## Input description" in one_anon)

print("\n7. Provider-config validation")
# Missing groq key with provider=groq must fail eagerly.
os.environ["GROQ_API_KEY"] = ""
try:
    Settings.from_env(env_file=Path("/nonexistent"))
    check("missing groq key rejected", False)
except Exception as exc:
    check("missing groq key rejected",
          "GROQ_API_KEY" in str(exc) or "groq" in str(exc).lower())

# Swap to anthropic with a real key and confirm settings load.
os.environ["LLM_PROVIDER"] = "anthropic"
os.environ["ANTHROPIC_API_KEY"] = "sk-ant-fake-for-swap-test"
os.environ["MODEL_ID"] = "claude-opus-5"
try:
    s2 = Settings.from_env(env_file=Path("/nonexistent"))
    check("provider swap to anthropic loads cleanly",
          s2.llm_provider == "anthropic" and s2.anthropic_api_key.startswith("sk-ant-"))
except Exception as exc:
    check(f"provider swap to anthropic loads cleanly (got: {exc!r})", False)

print(f"\n{'ALL CHECKS PASSED' if not failures else f'FAILURES ({len(failures)}): ' + str(failures)}")
sys.exit(1 if failures else 0)
