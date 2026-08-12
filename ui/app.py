"""Streamlit dashboard for the multi-agent SRS generation experiment.

Wraps :func:`run_experiment.run_matrix` with a browser UI: pick a provider
(Anthropic, Groq, or a local Ollama server) and model right in the sidebar,
pick architectures, test cases, and repetitions, watch live logs while a
*real* run executes, then browse and view every generated SRS as rendered
HTML instead of raw Markdown.

Run with::

    streamlit run ui/app.py

Every run started from this UI makes real calls to the selected provider —
there is no mocked/offline mode here. (``smoke_all_stages.py`` is the
separate, non-UI developer tool that uses a fake client; it is
intentionally not wired into this dashboard.)

Provider selection: by default the dashboard uses whatever ``.env``
already specifies (``LLM_PROVIDER`` and its credentials). The sidebar lets
you override the provider, model, and credentials for the *next run only*
— those values live in this browser session's memory for the lifetime of
the Streamlit process and are never written to ``.env`` or to disk.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

import streamlit as st
import streamlit.components.v1 as components

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.input_guard import check_description  # noqa: E402
from config.prompts import EVALUATION_DIMENSIONS  # noqa: E402
from config.settings import ConfigurationError, Settings  # noqa: E402
from evaluation.generate_report import generate_report  # noqa: E402
from evaluation.knowledge_base import detect_domain  # noqa: E402
from run_experiment import ALL_ARCHITECTURES, RunOutcome, run_matrix  # noqa: E402
from test_cases.cases import TEST_CASES, TestCase, get_case  # noqa: E402
from ui.render_html import render_srs_html  # noqa: E402
from ui.log_bridge import StreamlitLogHandler  # noqa: E402
from ui.log_view import render_log_panel  # noqa: E402
from ui.phase_tracker import PhaseTrackingLogHandler  # noqa: E402
from utils.llm_client import LLMClient  # noqa: E402
from utils.logging import ExperimentLogger  # noqa: E402

st.set_page_config(
    page_title="SRS Multi-Agent Experiment Runner",
    page_icon="🧪",
    layout="wide",
)

#: Estimated LLM calls per single (architecture, case, repetition) run,
#: mirrors the note in run_experiment.py's _print_plan().
_CALLS_PER_RUN = {
    "baseline_single_prompt": 1,
    "hierarchical": 7,
    "peer_to_peer": 8,
    "debate": 13,
}

#: Human-readable labels for the four ARCHITECTURES (collaboration
#: patterns), shown via multiselect's format_func. Not to be confused
#: with the five AGENTS (Orchestrator, Functional Requirements,
#: Non-Functional Requirements, Risk & Clarification, Verification) —
#: the same five agents run inside hierarchical/peer_to_peer/debate every
#: time; these four options control how they hand work to each other,
#: not who does the work.
_ARCHITECTURE_LABELS: dict[str, str] = {
    "baseline_single_prompt": "Baseline (single prompt)",
    "hierarchical": "Hierarchical (top-down)",
    "peer_to_peer": "Peer-to-peer (mutual review)",
    "debate": "Debate (competing + arbitrated)",
}

#: Longer descriptions for the help tooltip next to the architecture picker.
_ARCHITECTURE_HELP = (
    "These are the four **collaboration patterns** being compared — not "
    "the agents themselves. The same five agents (Orchestrator, "
    "Functional Requirements, Non-Functional Requirements, Risk & "
    "Clarification, Verification) run inside hierarchical / peer_to_peer "
    "/ debate every time; only how they hand work to each other changes:\n\n"
    "- **Baseline** — one LLM call writes the whole SRS in one shot. "
    "No agents, no collaboration — the control group.\n"
    "- **Hierarchical** — Orchestrator plans, specialists work "
    "independently, Verification checks the result and can send it back "
    "for one revision round.\n"
    "- **Peer-to-peer** — specialists read each other's drafts and "
    "revise their own before verification.\n"
    "- **Debate** — two independent copies of each specialist propose "
    "competing requirements; Verification arbitrates between them."
)


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


def _load_settings() -> Settings | None:
    """Load Settings from .env, surfacing a configuration error in the UI."""
    try:
        return Settings.from_env()
    except ConfigurationError as exc:
        st.error(f"Configuration error: {exc}")
        return None


@contextmanager
def _env_overrides(values: Mapping[str, str]) -> Iterator[None]:
    """Temporarily set process environment variables, then restore them.

    ``Settings.from_env()`` reads directly from ``os.environ`` (via
    ``load_dotenv(..., override=False)``, so anything already set here wins
    over ``.env``). Scoping the mutation to just the ``Settings.from_env()``
    call — rather than leaving it set for the rest of the process — means a
    provider/key typed into the UI for one run never leaks into a later run
    that meant to use the ``.env`` default, and doesn't linger in the
    process environment (visible to e.g. a crash dump) longer than needed.
    """
    previous = {key: os.environ.get(key) for key in values}
    try:
        for key, value in values.items():
            os.environ[key] = value
        yield
    finally:
        for key, prior in previous.items():
            if prior is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = prior


def _resolve_settings(
    overrides: Mapping[str, str], *, silent: bool = False, run_id_suffix: str | None = None,
) -> Settings | None:
    """Build Settings with ``overrides`` applied on top of the environment.

    Args:
        overrides: Environment variable values to use for this resolution
            only (see :func:`_env_overrides`); typically the UI's
            provider/model/credential form fields.
        silent: If ``True``, swallow a :class:`ConfigurationError` and
            return ``None`` without rendering ``st.error`` — used for the
            live sidebar preview, which re-resolves on every keystroke and
            would otherwise flash an error while the user is still filling
            in the form.
        run_id_suffix: Forwarded to :meth:`Settings.from_env` — a short
            slug appended to the run's timestamp-based id, so the run
            directory name carries a hint of what the run was about
            instead of being a bare timestamp.

    Returns:
        A validated :class:`Settings`, or ``None`` if the (overridden)
        configuration is invalid or incomplete.
    """
    with _env_overrides(overrides):
        try:
            return Settings.from_env(run_id_suffix=run_id_suffix)
        except ConfigurationError as exc:
            if not silent:
                st.error(f"Configuration error: {exc}")
            return None


#: Directory name (relative to a Settings.output_dir) ad hoc runs are
#: redirected into — mirrors the constant used at the "Try your own idea"
#: run-trigger site below, duplicated here only as a literal because that
#: site computes it via dataclasses.replace() rather than a named constant.
_ADHOC_SUBDIR = "adhoc"


@dataclasses.dataclass(frozen=True)
class _BrowsableRun:
    """One run directory the "Browse past runs" tab can show.

    Attributes:
        path: The run directory itself.
        is_adhoc: Whether this came from ``outputs/adhoc/`` (a "Try your
            own idea" run) rather than a formal experiment run — shown as
            a badge so the two are never mistaken for each other, even
            though they're now browsable from the same list.
    """

    path: Path
    is_adhoc: bool


def _list_run_dirs(output_dir: Path) -> list[_BrowsableRun]:
    """All run directories — formal and ad hoc — newest-first.

    Formal runs live directly under ``output_dir``; ad hoc runs live under
    ``output_dir/adhoc/`` (see the "Try your own idea" tab, which redirects
    there specifically so ad hoc artefacts never mix into the formal
    result set on disk — that separation is preserved here, this just
    makes both halves visible from one browser instead of only the first).

    Run directory names are ``run_<UTC timestamp>[_<slug>]`` — the
    timestamp always comes first, so sorting by name is still
    chronological regardless of whether a descriptive suffix follows it.
    """
    runs: list[_BrowsableRun] = []
    if output_dir.is_dir():
        runs.extend(
            _BrowsableRun(p, is_adhoc=False)
            for p in output_dir.iterdir() if p.is_dir() and p.name.startswith("run_")
        )
    adhoc_dir = output_dir / _ADHOC_SUBDIR
    if adhoc_dir.is_dir():
        runs.extend(
            _BrowsableRun(p, is_adhoc=True)
            for p in adhoc_dir.iterdir() if p.is_dir() and p.name.startswith("run_")
        )
    return sorted(runs, key=lambda r: r.path.name, reverse=True)


def _format_run_label(run: _BrowsableRun) -> str:
    """Turn a run directory name into a compact, readable selectbox label.

    ``run_2026-08-10T16-15-06Z_hotel_booking_platform`` becomes
    ``🧪 2026-08-10 16:15 — hotel booking platform`` (ad hoc) or
    ``▶️ 2026-08-10 16:15 — 4arch full_matrix`` (formal); falls back to the
    raw directory name for anything that doesn't match the expected shape
    (e.g. a run directory from before this naming scheme existed).
    """
    m = re.match(r"^run_(\d{4}-\d{2}-\d{2})T(\d{2})-(\d{2})-\d{2}Z(?:_(.+))?$", run.path.name)
    icon = "🧪" if run.is_adhoc else "▶️"
    if not m:
        return f"{icon} {run.path.name}"
    date, hour, minute, suffix = m.groups()
    label = f"{icon} {date} {hour}:{minute}"
    if suffix:
        label += f" — {suffix.replace('_', ' ')}"
    return label


def _srs_files(run_dir: Path) -> list[Path]:
    return sorted(run_dir.glob("srs_*.md"))


def _load_interactions(jsonl_path: Path) -> list[dict[str, Any]]:
    """Flatten llm_interactions.jsonl into rows suitable for st.dataframe.

    Keeps only the fields useful for a scan-at-a-glance table (full detail,
    including the exact prompts, stays in the raw file — offered as a
    download alongside the table).
    """
    rows: list[dict[str, Any]] = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        usage = record.get("usage") or {}
        error = record.get("error") or {}
        rows.append({
            "timestamp": record.get("timestamp"),
            "architecture": record.get("architecture"),
            "agent": record.get("agent"),
            "case_id": record.get("case_id"),
            "repetition": record.get("repetition"),
            "phase": record.get("phase"),
            "model": record.get("model"),
            "stop_reason": record.get("stop_reason"),
            "in_tokens": usage.get("total_input_tokens", usage.get("input_tokens", 0)),
            "out_tokens": usage.get("output_tokens", 0),
            "latency_ms": record.get("latency_ms"),
            "error": error.get("message", ""),
        })
    return rows


def _view_srs(md_path: Path, *, key_prefix: str) -> None:
    """Render one SRS file's HTML view + download buttons, inside an expander."""
    markdown_text = md_path.read_text(encoding="utf-8")
    with st.expander(f"📄 {md_path.stem}"):
        html_doc = render_srs_html(markdown_text, title=md_path.stem)
        components.html(html_doc, height=650, scrolling=True)
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "Download .md", data=markdown_text, file_name=md_path.name,
                mime="text/markdown", key=f"{key_prefix}_md",
            )
        with col2:
            st.download_button(
                "Download .html", data=html_doc, file_name=md_path.stem + ".html",
                mime="text/html", key=f"{key_prefix}_html",
            )


def _slugify_case_name(text: str) -> str:
    """Turn free text into a short, filesystem-safe case identifier.

    Used only by the "Try your own idea" tab: ad hoc descriptions aren't
    drawn from the fixed ``test_cases.py`` list, so an id has to be
    invented on the spot. The result feeds directly into output file
    names (``srs_<case_id>_<architecture>_rep0.md``), so it must satisfy
    the same ``[A-Za-z0-9_-]`` constraint the formal test cases do.
    """
    base = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower()).strip("_")
    return base[:40] or "custom"


def _formal_run_slug(architectures: list[str], case_ids: list[str]) -> str:
    """Build a short, readable slug for a formal run's directory name.

    A formal run can cover many (architecture, case) combinations at
    once, so unlike the ad hoc tab there's no single description to slug
    — this summarises shape instead of content: the full 4×4 matrix reads
    as ``full_matrix``; a single case reads as e.g. ``2arch_restaurant_app``
    (its own case_id, minus the ``TC-0N_`` numbering, which is filesystem-
    noise once it's sitting inside a dated run directory); anything else
    falls back to plain counts, e.g. ``3arch_2case``.
    """
    if len(architectures) == len(ALL_ARCHITECTURES) and len(case_ids) == len(TEST_CASES):
        return "full_matrix"
    if len(case_ids) == 1:
        case_slug = re.sub(r"^TC-\d+_", "", case_ids[0])
        return f"{len(architectures)}arch_{case_slug}"
    return f"{len(architectures)}arch_{len(case_ids)}case"


def _domain_gate(*, gate_key: str, cases: list[TestCase], grounding_requested: bool) -> bool:
    """Pre-flight HITL check: is every case's domain covered by a knowledge base?

    Runs *before* generation starts, not mid-run — ``run_matrix`` already
    starts real, billed LLM calls the moment it's invoked, and Streamlit's
    blocking execution model has no clean way to pause a call already in
    flight. Checking domains up front sidesteps that entirely: nothing is
    generated until this returns ``True``.

    Args:
        gate_key: A key unique to the calling tab (e.g. ``"formal_gate"``),
            used to namespace this gate's session-state entries so the two
            tabs' gates never collide.
        cases: The resolved test cases the pending run would cover.
        grounding_requested: Whether the caller even asked for grounded
            scoring — if not, there is nothing to gate on.

    Returns:
        ``True`` if the caller may start generation this rerun. ``False``
        means the HITL prompt has been rendered (or the user hasn't acted
        on it yet) and the caller must not start generation.
    """
    if not grounding_requested:
        return True
    unresolved = [c for c in cases if detect_domain(case_id=c.case_id, description=c.description) is None]
    if not unresolved:
        return True
    if st.session_state.get(gate_key) == "proceed_ungrounded":
        st.session_state.pop(gate_key, None)
        return True

    plural = len(unresolved) > 1
    names = "; ".join(f"**{c.title}**" for c in unresolved)
    st.warning(
        f"🧭 Not in the knowledge base yet: {names}. There's no sourced "
        f"fact set to score {'them' if plural else 'it'} against, so "
        f"grounded coverage/faithfulness scoring can't run for "
        f"{'these cases' if plural else 'this case'} — the deterministic "
        "quality lint and (if enabled) the LLM rubric still run as normal."
    )
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Run anyway — quality-only, no grounding", key=f"{gate_key}_proceed_btn"):
            st.session_state[gate_key] = "proceed_ungrounded"
            st.rerun()
    with col2:
        if st.button("Cancel", key=f"{gate_key}_cancel_btn"):
            st.session_state.pop(gate_key, None)
            st.session_state.pop(f"{gate_key}_requested", None)
            st.info(
                "Cancelled — nothing was generated. Ask for a knowledge "
                "base to be researched for this domain; sourced facts get "
                "shown to you for approval before anything is scored "
                "against them."
            )
    return False


def _render_grounded_winner(grounded_scores_path: Path | None) -> bool:
    """Render a winner banner (2+ architectures) or a plain score line (1)
    from grounded_scores.json.

    Silently does nothing only when there's truly nothing to show — no
    case's domain had a knowledge base, so grounded scoring produced no
    records at all. A single-architecture run still has a real, useful
    score (just no competitor to rank it against), so it gets a plain
    "score" line rather than being suppressed entirely.

    Returns:
        ``True`` if anything was actually rendered — callers use this to
        decide whether :func:`_render_rubric_summary` still needs to show
        its own leaderboard, or would just be a redundant second ranking
        of the same architecture(s).
    """
    if grounded_scores_path is None:
        return False
    path = Path(grounded_scores_path)
    if not path.exists():
        return False
    data = json.loads(path.read_text(encoding="utf-8"))
    overall = data.get("overall", {})
    summaries = overall.get("summaries", [])
    if not summaries:
        return False

    if len(summaries) == 1:
        s = summaries[0]
        label = _ARCHITECTURE_LABELS.get(s["architecture"], s["architecture"])
        st.markdown(
            f"**📊 Grounded score:** {label} — composite {s['mean_composite']:.2f} · "
            f"coverage {s['mean_coverage']:.0%} · faithfulness {s['mean_faithfulness']:.0%} · "
            f"quality {s['mean_quality']:.0%}"
        )
        return True

    winner = summaries[0]
    winner_label = _ARCHITECTURE_LABELS.get(winner["architecture"], winner["architecture"])

    st.markdown(
        f"""
<div style="border:2px solid #DAA520; border-radius:10px; padding:16px 20px;
            background:rgba(218,165,32,0.08); margin-bottom:12px;">
  <div style="font-size:0.85rem; letter-spacing:0.06em; text-transform:uppercase;
              opacity:0.75;">🏆 Grounded winner — knowledge-base scored</div>
  <div style="font-size:1.4rem; font-weight:700; margin-top:2px;">{winner_label}</div>
  <div style="opacity:0.8; margin-top:2px;">
    composite {winner['mean_composite']:.2f} · coverage {winner['mean_coverage']:.0%} ·
    faithfulness {winner['mean_faithfulness']:.0%} · quality {winner['mean_quality']:.0%} ·
    won {winner['cases_won']}/{winner['cases_scored']} case(s)
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
    return True

    rows = [
        {
            "Architecture": _ARCHITECTURE_LABELS.get(s["architecture"], s["architecture"]),
            "Composite": s["mean_composite"],
            "Coverage": s["mean_coverage"],
            "Faithfulness": s["mean_faithfulness"],
            "Quality": s["mean_quality"],
            "Cases won": f"{s['cases_won']}/{s['cases_scored']}",
        }
        for s in summaries
    ]
    st.dataframe(
        rows,
        column_config={
            "Composite": st.column_config.ProgressColumn("Composite", min_value=0, max_value=1, format="%.2f"),
            "Coverage": st.column_config.ProgressColumn("Coverage", min_value=0, max_value=1, format="%.0%%"),
            "Faithfulness": st.column_config.ProgressColumn("Faithfulness", min_value=0, max_value=1, format="%.0%%"),
            "Quality": st.column_config.ProgressColumn("Quality", min_value=0, max_value=1, format="%.0%%"),
        },
        hide_index=True,
        width="stretch",
    )
    with st.expander("Per-case winners"):
        for cw in overall.get("case_winners", []):
            arch_label = _ARCHITECTURE_LABELS.get(cw["winner"], cw["winner"]) if cw["winner"] else "—"
            st.markdown(f"- **{cw['case_id']}** → {arch_label}")


def _render_rubric_summary(run_dir: Path) -> None:
    """Render a per-architecture rubric-score leaderboard from evaluation.json.

    The LLM-as-judge + deterministic-lint rubric (completeness, consistency,
    testability, clarity — see evaluation/rubric.py) runs for every case
    regardless of whether its domain has a knowledge base, unlike grounded
    scoring. So this is the ranking signal that still exists when
    :func:`_render_grounded_winner` has nothing to show — an unfamiliar
    domain (no KB yet), or a run where grounding was skipped entirely. A
    single-architecture run still shows its own score, just without
    "leader" framing — there's nothing to rank it against, but the score
    itself is real and shouldn't be hidden.
    """
    eval_path = run_dir / "evaluation.json"
    if not eval_path.exists():
        return
    try:
        records = json.loads(eval_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return

    totals_by_arch: dict[str, list[int]] = {}
    for r in records:
        arch = r.get("architecture")
        scores = r.get("scores", {})
        total = sum(int(scores.get(d, 0)) for d in EVALUATION_DIMENSIONS)
        totals_by_arch.setdefault(arch, []).append(total)

    if not totals_by_arch:
        return
    max_total = len(EVALUATION_DIMENSIONS) * 5

    if len(totals_by_arch) == 1:
        arch, vals = next(iter(totals_by_arch.items()))
        label = _ARCHITECTURE_LABELS.get(arch, arch)
        mean_total = sum(vals) / len(vals)
        st.markdown(f"**📊 Rubric score:** {label} — {mean_total:.1f}/{max_total}")
        return

    summaries = sorted(
        (
            {"architecture": arch, "mean_total": sum(vals) / len(vals), "n": len(vals)}
            for arch, vals in totals_by_arch.items()
        ),
        key=lambda s: (-s["mean_total"], s["architecture"]),
    )
    winner = summaries[0]
    winner_label = _ARCHITECTURE_LABELS.get(winner["architecture"], winner["architecture"])

    st.markdown(
        f"**📋 Rubric leader:** {winner_label} — "
        f"{winner['mean_total']:.1f}/{max_total} mean total "
        f"({' + '.join(EVALUATION_DIMENSIONS)})"
    )
    rows = [
        {
            "Architecture": _ARCHITECTURE_LABELS.get(s["architecture"], s["architecture"]),
            f"Mean total /{max_total}": round(s["mean_total"], 1),
            "n": s["n"],
        }
        for s in summaries
    ]
    st.dataframe(rows, hide_index=True, width="stretch")


# --------------------------------------------------------------------------
# Sidebar: provider & model selection
# --------------------------------------------------------------------------

st.sidebar.title("🧪 SRS Experiment Runner")

#: Loaded once for defaults (e.g. prefilling the Ollama fields with
#: whatever .env already points at) and for the "Browse past runs" tab's
#: output directory. Independent of whatever the provider form below ends
#: up overriding for the *next run*.
_env_settings = _load_settings()

st.sidebar.markdown("### Provider & model")
provider_choice = st.sidebar.selectbox(
    "Provider for the next run",
    options=["Use .env default", "anthropic", "groq", "ollama"],
    index=0,
    help=(
        "Overrides LLM_PROVIDER and its credentials for the run you start "
        "below, without editing .env. Anyone using this dashboard can "
        "point it at their own Anthropic/Groq key or their own local "
        "Ollama server this way."
    ),
)

#: Env-var overrides built from the sidebar form; applied only while
#: resolving Settings for the run started below (see _resolve_settings).
ui_overrides: dict[str, str] = {}

if provider_choice != "Use .env default":
    ui_overrides["LLM_PROVIDER"] = provider_choice

    if provider_choice == "anthropic":
        anthropic_key = st.sidebar.text_input(
            "Anthropic API key", type="password", key="ui_anthropic_key",
            placeholder="sk-ant-...",
            help="Leave blank to reuse ANTHROPIC_API_KEY from .env, if set.",
        )
        anthropic_model = st.sidebar.text_input(
            "Model ID", value="claude-opus-5", key="ui_anthropic_model",
        )
        if anthropic_key:
            ui_overrides["ANTHROPIC_API_KEY"] = anthropic_key
        if anthropic_model:
            ui_overrides["MODEL_ID"] = anthropic_model

    elif provider_choice == "groq":
        groq_key = st.sidebar.text_input(
            "Groq API key", type="password", key="ui_groq_key",
            placeholder="gsk_...",
            help="Leave blank to reuse GROQ_API_KEY from .env, if set.",
        )
        groq_model = st.sidebar.text_input(
            "Model ID", value="llama-3.3-70b-versatile", key="ui_groq_model",
        )
        if groq_key:
            ui_overrides["GROQ_API_KEY"] = groq_key
        if groq_model:
            ui_overrides["MODEL_ID"] = groq_model

    elif provider_choice == "ollama":
        default_base_url = (
            _env_settings.ollama_base_url if _env_settings else "http://localhost:11434"
        )
        default_ollama_model = (
            _env_settings.ollama_model if _env_settings else "mistral"
        )
        ollama_base_url = st.sidebar.text_input(
            "Ollama base URL", value=default_base_url, key="ui_ollama_url",
        )
        ollama_model = st.sidebar.text_input(
            "Ollama model", value=default_ollama_model, key="ui_ollama_model",
            help="Must already be pulled locally (`ollama pull <model>`).",
        )
        if ollama_base_url:
            ui_overrides["OLLAMA_BASE_URL"] = ollama_base_url
        if ollama_model:
            ui_overrides["OLLAMA_MODEL"] = ollama_model

    st.sidebar.caption(
        "Kept in this browser session's memory only — never written to "
        ".env or to disk."
    )

#: Live preview of what a run would actually use right now, given the form
#: above. Resolved silently (no st.error) since this re-runs on every
#: keystroke and the form is often mid-edit (e.g. key not typed in yet).
preview_settings = _resolve_settings(ui_overrides, silent=True)

st.sidebar.markdown("**Will use for the next run**")
if preview_settings is not None:
    st.sidebar.code(
        f"LLM_PROVIDER = {preview_settings.llm_provider}\n"
        f"model        = {preview_settings.effective_model_id}"
        + (
            f"\nollama_url    = {preview_settings.ollama_base_url}"
            if preview_settings.llm_provider == "ollama"
            else ""
        ),
        language="text",
    )
    st.sidebar.caption(
        "Every run below calls this provider for real — there is no "
        "offline/mocked mode in this dashboard."
    )
else:
    st.sidebar.warning(
        "Incomplete provider configuration — fill in the required "
        "field(s) above (or switch back to '.env default')."
    )

tab_run, tab_custom, tab_browse = st.tabs(
    ["▶️ Run experiment", "💡 Try your own idea", "📂 Browse past runs"]
)

# --------------------------------------------------------------------------
# Tab 1 — Run experiment
# --------------------------------------------------------------------------

with tab_run:
    st.subheader("Configure a run")

    architectures = st.multiselect(
        "Architectures",
        options=list(ALL_ARCHITECTURES),
        default=list(ALL_ARCHITECTURES),
        format_func=lambda a: _ARCHITECTURE_LABELS.get(a, a),
        help=_ARCHITECTURE_HELP,
    )

    case_ids = [c.case_id for c in TEST_CASES]
    case_labels = {c.case_id: f"{c.case_id} — {c.title}" for c in TEST_CASES}
    cases_selected = st.multiselect(
        "Test cases",
        options=case_ids,
        default=case_ids,
        format_func=lambda cid: case_labels[cid],
    )

    col_a, col_b = st.columns(2)
    with col_a:
        repetitions = st.number_input("Repetitions", min_value=1, max_value=20, value=1, step=1)
    with col_b:
        revision_rounds = st.number_input(
            "Revision rounds", min_value=0, max_value=5, value=1, step=1,
            help=(
                "How many times the Verification agent may send flawed "
                "sections back for a fix. 0 = fastest, but architectures "
                "never get to self-correct — only use 0 for quick checks, "
                "not for results you intend to keep. The default (1) is "
                "part of what the architectures are designed to do."
            ),
        )

    # A formal run always does the full pipeline — LLM-as-judge evaluation,
    # the anonymised rater export, and grounded knowledge-base scoring.
    # These used to be individually toggleable checkboxes; every one of
    # them defaulted to checked, and a "formal" run that skips half its
    # own scoring isn't really a formal run, so the toggles were removed
    # rather than kept as three ways to quietly produce a weaker result.
    run_evaluation = True
    run_rater_export = True
    run_grounding = True

    n_combos = len(architectures) * len(cases_selected) * int(repetitions)
    est_calls = sum(_CALLS_PER_RUN.get(a, 0) for a in architectures) * len(cases_selected) * int(repetitions)
    # +1 evaluation call and +1 grounded-grading call per generated document.
    est_calls += 2 * len(architectures) * len(cases_selected) * int(repetitions)
    provider_note = (
        f" via {preview_settings.llm_provider}/{preview_settings.effective_model_id}"
        if preview_settings is not None else ""
    )
    st.caption(
        f"Plan: {len(architectures)} architecture(s) × {len(cases_selected)} case(s) × "
        f"{int(repetitions)} repetition(s) = **{n_combos} generation run(s)**, "
        f"~{est_calls} LLM call(s) total{provider_note} "
        "(generation + LLM-as-judge evaluation + grounded scoring)."
    )
    if preview_settings is None:
        st.warning("Fix the provider configuration in the sidebar before starting a run.")

    start_clicked = st.button(
        "▶ Start real run", type="primary",
        disabled=not architectures or not cases_selected or preview_settings is None,
    )
    if start_clicked:
        st.session_state["formal_gate_requested"] = True

    if st.session_state.get("formal_gate_requested"):
        resolved_cases = [get_case(cid) for cid in cases_selected]
        gate_clear = _domain_gate(
            gate_key="formal_gate", cases=resolved_cases, grounding_requested=run_grounding,
        )
    else:
        gate_clear = False

    if gate_clear:
        st.session_state["formal_gate_requested"] = False
        # REPETITIONS folds into the same scoped-override mechanism as the
        # sidebar's provider/model/credential fields, so nothing set here
        # lingers in the process environment after Settings is resolved.
        run_overrides = dict(ui_overrides)
        run_overrides["REPETITIONS"] = str(int(repetitions))
        run_overrides["MAX_REVISION_ROUNDS"] = str(int(revision_rounds))
        settings = _resolve_settings(
            run_overrides, silent=False,
            run_id_suffix=_formal_run_slug(architectures, cases_selected),
        )
        if settings is not None:
            settings.apply_seed()

            status_label = (
                f"Running {n_combos} generation(s) via "
                f"{settings.llm_provider}/{settings.effective_model_id}…"
            )
            with st.status(status_label, expanded=True) as status:
                progress_placeholder = st.empty()
                phase_handler = PhaseTrackingLogHandler(progress_placeholder)
                with st.expander("Detailed log", expanded=False):
                    log_placeholder = st.empty()
                handler = StreamlitLogHandler(log_placeholder)
                outcomes: list[RunOutcome] = []
                run_dir: Path | None = None
                grounded_scores_path: str | None = None
                try:
                    with ExperimentLogger.for_run(settings) as logger:
                        logger.add_handler(phase_handler)
                        logger.add_handler(handler)
                        run_dir = settings.run_dir
                        client = LLMClient(settings, logger)
                        _manifest, outcomes = run_matrix(
                            settings=settings,
                            architectures=architectures,
                            cases=resolved_cases,
                            client=client,
                            logger=logger,
                            skip_evaluation=not run_evaluation,
                            skip_rater_export=not run_rater_export,
                            skip_grounding=not run_grounding,
                        )
                        grounded_scores_path = _manifest.grounded_scores_path
                except Exception as exc:  # noqa: BLE001 - surfaced to the UI, not swallowed
                    status.update(label=f"Run failed: {exc}", state="error", expanded=True)
                    st.exception(exc)
                else:
                    failed = sum(1 for o in outcomes if o.error is not None)
                    ok = len(outcomes) - failed
                    status.update(
                        label=f"Done: {ok}/{len(outcomes)} generation(s) succeeded",
                        state="complete" if failed == 0 else "error",
                        expanded=True,
                    )

            if run_dir is not None and outcomes:
                if not _render_grounded_winner(Path(grounded_scores_path) if grounded_scores_path else None):
                    _render_rubric_summary(run_dir)
                st.subheader("Results")
                for outcome in outcomes:
                    arch_label = _ARCHITECTURE_LABELS.get(outcome.architecture, outcome.architecture)
                    label = f"{arch_label} / {outcome.case_id} / rep {outcome.repetition}"
                    if outcome.error is not None:
                        st.error(f"❌ {label}: {outcome.error}")
                    elif outcome.srs_path is not None:
                        st.success(f"✅ {label}")
                        _view_srs(
                            outcome.srs_path,
                            key_prefix=f"run_{outcome.architecture}_{outcome.case_id}_{outcome.repetition}",
                        )
                st.caption(f"Artefacts written to `{run_dir}`")

# --------------------------------------------------------------------------
# Tab 2 — Try your own idea (ad hoc, outside the formal experiment)
# --------------------------------------------------------------------------
#
# Deliberately separate from "Run experiment": the formal experiment's
# fixed 4-case list exists so every architecture is compared on identical
# inputs, which is what makes the dissertation's comparison valid. This
# tab lets you generate an SRS for anything you describe, using the same
# agents/architectures, but never writes into outputs/run_<id>/ — it goes
# to outputs/adhoc/ instead, so a curious one-off never gets mistaken for
# — or pollutes — the formal result set.

with tab_custom:
    st.subheader("Describe any system — get a real SRS")
    st.caption(
        "Not part of the formal experiment. Generates a one-off SRS from "
        "whatever you describe, using the same agents and architectures "
        "as above. Saved separately under `outputs/adhoc/` so it never "
        "mixes into your dissertation's result set."
    )

    custom_description = st.text_area(
        "Describe your system",
        placeholder=(
            'e.g. "A hotel booking platform where guests can search rooms, '
            "book stays, and pay online. Front-desk staff manage check-ins "
            "and check-outs. Managers view occupancy reports and adjust "
            'room rates."'
        ),
        height=120,
        key="adhoc_description",
    )
    custom_name = st.text_input(
        "Short name (optional — auto-generated from your description if left blank)",
        key="adhoc_name",
        placeholder="e.g. hotel_booking",
    )

    adhoc_architectures = st.multiselect(
        "Architectures to compare",
        options=list(ALL_ARCHITECTURES),
        default=["baseline_single_prompt", "hierarchical"],
        key="adhoc_architectures",
        format_func=lambda a: _ARCHITECTURE_LABELS.get(a, a),
        help=(
            _ARCHITECTURE_HELP
            + "\n\nPeer-to-peer and debate take longer (more LLM calls "
            "per document) — add them once you're happy with the fast pair."
        ),
    )
    adhoc_revision_rounds = st.number_input(
        "Revision rounds", min_value=0, max_value=5, value=0, step=1,
        key="adhoc_revision_rounds",
        help=(
            "How many times Verification may send a flawed section "
            "back for a fix. Defaults to 0 here for a fast look. Set "
            "it to 1 (the formal experiment's default) if you want "
            "this ad hoc result to reflect what the real architecture "
            "actually does, self-correction included."
        ),
    )

    # Every ad hoc generation always runs the rubric + grounded scoring
    # too, same as the formal tab — these used to be individually
    # toggleable checkboxes (both defaulted to unchecked, for a fast bare
    # preview), but a "try it" that never tells you whether the result is
    # any good is a worse default than a few seconds of extra latency.
    adhoc_score = True
    adhoc_grounding = True

    adhoc_est_calls = sum(_CALLS_PER_RUN.get(a, 0) for a in adhoc_architectures)
    # +1 evaluation call and +1 grounded-grading call per generated document.
    adhoc_est_calls += 2 * len(adhoc_architectures)
    adhoc_provider_note = (
        f" via {preview_settings.llm_provider}/{preview_settings.effective_model_id}"
        if preview_settings is not None else ""
    )
    st.caption(
        f"Plan: {len(adhoc_architectures)} architecture(s) on 1 custom "
        f"case = **{len(adhoc_architectures)} generation(s)**, "
        f"~{adhoc_est_calls} LLM call(s) total{adhoc_provider_note} "
        "(generation + LLM-as-judge evaluation + grounded scoring)."
    )
    if preview_settings is None:
        st.warning("Fix the provider configuration in the sidebar before generating.")

    generate_clicked = st.button(
        "✨ Generate SRS", type="primary",
        disabled=(
            not custom_description.strip()
            or not adhoc_architectures
            or preview_settings is None
        ),
        key="adhoc_generate",
    )
    if generate_clicked:
        st.session_state["adhoc_gate_requested"] = True

    # Built unconditionally (cheap, pure text munging) so the gate below
    # can check its domain even on the rerun where "Generate SRS" itself
    # wasn't clicked — e.g. the rerun triggered by clicking the gate's own
    # "Run anyway" button.
    adhoc_slug = _slugify_case_name(custom_name or custom_description)
    adhoc_case_id = f"adhoc_{adhoc_slug}"
    adhoc_case = TestCase(
        case_id=adhoc_case_id,
        title=custom_name.strip() or custom_description.strip()[:60],
        description=custom_description.strip(),
    )

    if st.session_state.get("adhoc_gate_requested"):
        gate_clear = _domain_gate(
            gate_key="adhoc_gate", cases=[adhoc_case], grounding_requested=adhoc_grounding,
        )
    else:
        gate_clear = False

    if gate_clear:
        st.session_state["adhoc_gate_requested"] = False
        # Force exactly one repetition regardless of whatever the formal
        # experiment's REPETITIONS is set to — this is a "try it once" tool.
        run_overrides = dict(ui_overrides)
        run_overrides["REPETITIONS"] = "1"
        run_overrides["MAX_REVISION_ROUNDS"] = str(int(adhoc_revision_rounds))
        settings = _resolve_settings(run_overrides, silent=False, run_id_suffix=adhoc_slug)
        if settings is not None:
            settings.apply_seed()

            case_id = adhoc_case_id
            # Redirect to outputs/adhoc/ so this never mixes with, or gets
            # mistaken for, the formal experiment's outputs/run_<id>/ folders.
            adhoc_settings = dataclasses.replace(
                settings, output_dir=settings.output_dir / "adhoc",
            )

            status_label = (
                f"Generating via {adhoc_settings.llm_provider}/"
                f"{adhoc_settings.effective_model_id}…"
            )
            with st.status(status_label, expanded=True) as status:
                progress_placeholder = st.empty()
                phase_handler = PhaseTrackingLogHandler(progress_placeholder)
                with st.expander("Detailed log", expanded=False):
                    log_placeholder = st.empty()
                handler = StreamlitLogHandler(log_placeholder)
                outcomes: list[RunOutcome] = []
                run_dir: Path | None = None
                grounded_scores_path: str | None = None
                rejected_reason: str | None = None
                try:
                    with ExperimentLogger.for_run(adhoc_settings) as logger:
                        logger.add_handler(phase_handler)
                        logger.add_handler(handler)
                        run_dir = adhoc_settings.run_dir
                        client = LLMClient(adhoc_settings, logger)

                        # Pre-flight: does this free-text description
                        # actually describe a system? The formal tab never
                        # needs this (its 4 cases are fixed and pre-vetted);
                        # here, whatever was typed goes straight into every
                        # agent's prompt otherwise, so "crack a joke" would
                        # previously have been treated as a real system
                        # description and generated a full (nonsense) SRS.
                        status.update(label="Checking your description…", expanded=True)
                        is_valid, reason = check_description(
                            adhoc_case.description, client=client,
                        )
                        if not is_valid:
                            rejected_reason = reason
                            logger.warning("Input guard rejected description: %s", reason)
                        else:
                            _manifest, outcomes = run_matrix(
                                settings=adhoc_settings,
                                architectures=adhoc_architectures,
                                cases=[adhoc_case],
                                client=client,
                                logger=logger,
                                skip_evaluation=not adhoc_score,
                                skip_rater_export=True,
                                skip_grounding=not adhoc_grounding,
                            )
                            grounded_scores_path = _manifest.grounded_scores_path
                except Exception as exc:  # noqa: BLE001 - surfaced to the UI, not swallowed
                    status.update(label=f"Generation failed: {exc}", state="error", expanded=True)
                    st.exception(exc)
                else:
                    if rejected_reason is not None:
                        status.update(
                            label="Not a system description — nothing generated",
                            state="error", expanded=True,
                        )
                        st.error(
                            f"🛑 This doesn't look like a system description — {rejected_reason}\n\n"
                            'Try describing an actual software system, e.g. "A booking '
                            'platform where guests search rooms, reserve dates, and pay online."'
                        )
                    else:
                        failed = sum(1 for o in outcomes if o.error is not None)
                        ok = len(outcomes) - failed
                        status.update(
                            label=f"Done: {ok}/{len(outcomes)} document(s) generated",
                            state="complete" if failed == 0 else "error",
                            expanded=True,
                        )

            if run_dir is not None and outcomes:
                succeeded = [o for o in outcomes if o.error is None and o.srs_path is not None]
                failed_outcomes = [o for o in outcomes if o.error is not None]

                for o in failed_outcomes:
                    arch_label = _ARCHITECTURE_LABELS.get(o.architecture, o.architecture)
                    st.error(f"❌ {arch_label}: {o.error}")

                if not _render_grounded_winner(Path(grounded_scores_path) if grounded_scores_path else None):
                    _render_rubric_summary(run_dir)

                if len(succeeded) == 1:
                    st.subheader("Your SRS")
                    _view_srs(
                        succeeded[0].srs_path,
                        key_prefix=f"adhoc_{succeeded[0].architecture}_{case_id}",
                    )
                elif len(succeeded) > 1:
                    st.subheader(f"Compare {len(succeeded)} architectures side by side")
                    arch_tabs = st.tabs(
                        [_ARCHITECTURE_LABELS.get(o.architecture, o.architecture) for o in succeeded]
                    )
                    for arch_tab, o in zip(arch_tabs, succeeded):
                        with arch_tab:
                            _view_srs(
                                o.srs_path,
                                key_prefix=f"adhoc_{o.architecture}_{case_id}",
                            )
                st.caption(
                    f"Saved to `{run_dir}` — separate from the formal "
                    "experiment's outputs."
                )

# --------------------------------------------------------------------------
# Tab 3 — Browse past runs
# --------------------------------------------------------------------------

with tab_browse:
    st.subheader("Past runs")
    output_dir = _env_settings.output_dir if _env_settings else PROJECT_ROOT / "outputs"
    run_dirs = _list_run_dirs(output_dir)

    if not run_dirs:
        st.info(f"No runs found yet in `{output_dir}` (formal) or `{output_dir / _ADHOC_SUBDIR}` (ad hoc).")
    else:
        selected_entry = st.selectbox(
            "Run", options=run_dirs, format_func=_format_run_label,
        )
        selected_run = selected_entry.path
        if selected_entry.is_adhoc:
            st.caption("🧪 Ad hoc run — from \"Try your own idea\", not part of the formal experiment.")
        srs_files = _srs_files(selected_run)
        st.caption(f"{len(srs_files)} SRS document(s) in this run.")

        if not _render_grounded_winner(selected_run / "grounded_scores.json"):
            _render_rubric_summary(selected_run)

        report_path = selected_run / "analysis_report.md"
        col_report, _ = st.columns([1, 3])
        with col_report:
            if st.button("Generate / refresh analysis report", key="gen_report"):
                try:
                    report_path = generate_report(selected_run)
                    st.success(f"Wrote {report_path.name}")
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Report generation failed: {exc}")
        if report_path.is_file():
            with st.expander("📊 analysis_report.md", expanded=False):
                st.markdown(report_path.read_text(encoding="utf-8"))

        log_path = selected_run / "run.log"
        interactions_path = selected_run / "llm_interactions.jsonl"
        with st.expander("📜 Logs", expanded=False):
            tab_narrative, tab_interactions = st.tabs(["Narrative log", "LLM interactions"])
            with tab_narrative:
                if log_path.is_file() and log_path.stat().st_size > 0:
                    components.html(
                        render_log_panel(log_path.read_text(encoding="utf-8"), panel_id=f"log_{selected_run.name}"),
                        height=460,
                    )
                    st.download_button(
                        "Download run.log", data=log_path.read_text(encoding="utf-8"),
                        file_name="run.log", mime="text/plain", key=f"dl_runlog_{selected_run.name}",
                    )
                else:
                    st.info("No run.log for this run.")
            with tab_interactions:
                if interactions_path.is_file():
                    rows = _load_interactions(interactions_path)
                    if rows:
                        filter_text = st.text_input(
                            "Filter (agent, phase, architecture, error…)",
                            key=f"filter_{selected_run.name}",
                        )
                        filtered = (
                            [r for r in rows if filter_text.lower() in json.dumps(r, default=str).lower()]
                            if filter_text else rows
                        )
                        st.dataframe(filtered, use_container_width=True, height=380)
                        st.caption(f"{len(filtered)} / {len(rows)} interaction(s) shown.")
                    else:
                        st.info("llm_interactions.jsonl is empty.")
                    st.download_button(
                        "Download llm_interactions.jsonl",
                        data=interactions_path.read_text(encoding="utf-8"),
                        file_name="llm_interactions.jsonl", mime="application/jsonl",
                        key=f"dl_jsonl_{selected_run.name}",
                    )
                else:
                    st.info("No llm_interactions.jsonl for this run.")

        for md_path in srs_files:
            _view_srs(md_path, key_prefix=f"browse_{md_path.stem}")
