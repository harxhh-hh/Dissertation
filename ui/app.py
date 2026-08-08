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

import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Mapping

import streamlit as st
import streamlit.components.v1 as components

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import ConfigurationError, Settings  # noqa: E402
from evaluation.generate_report import generate_report  # noqa: E402
from run_experiment import ALL_ARCHITECTURES, RunOutcome, run_matrix  # noqa: E402
from test_cases.cases import TEST_CASES, get_case  # noqa: E402
from ui.render_html import render_srs_html  # noqa: E402
from ui.log_bridge import StreamlitLogHandler  # noqa: E402
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


def _resolve_settings(overrides: Mapping[str, str], *, silent: bool = False) -> Settings | None:
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

    Returns:
        A validated :class:`Settings`, or ``None`` if the (overridden)
        configuration is invalid or incomplete.
    """
    with _env_overrides(overrides):
        try:
            return Settings.from_env()
        except ConfigurationError as exc:
            if not silent:
                st.error(f"Configuration error: {exc}")
            return None


def _list_run_dirs(output_dir: Path) -> list[Path]:
    """Run directories newest-first (run_id is a UTC timestamp, so this
    is also chronological order)."""
    if not output_dir.is_dir():
        return []
    return sorted(
        (p for p in output_dir.iterdir() if p.is_dir() and p.name.startswith("run_")),
        key=lambda p: p.name,
        reverse=True,
    )


def _srs_files(run_dir: Path) -> list[Path]:
    return sorted(run_dir.glob("srs_*.md"))


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

tab_run, tab_browse = st.tabs(["▶️ Run experiment", "📂 Browse past runs"])

# --------------------------------------------------------------------------
# Tab 1 — Run experiment
# --------------------------------------------------------------------------

with tab_run:
    st.subheader("Configure a run")

    architectures = st.multiselect(
        "Architectures",
        options=list(ALL_ARCHITECTURES),
        default=list(ALL_ARCHITECTURES),
    )

    case_ids = [c.case_id for c in TEST_CASES]
    case_labels = {c.case_id: f"{c.case_id} — {c.title}" for c in TEST_CASES}
    cases_selected = st.multiselect(
        "Test cases",
        options=case_ids,
        default=case_ids,
        format_func=lambda cid: case_labels[cid],
    )

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        repetitions = st.number_input("Repetitions", min_value=1, max_value=20, value=1, step=1)
    with col_b:
        run_evaluation = st.checkbox("Run LLM-as-judge evaluation", value=True)
    with col_c:
        run_rater_export = st.checkbox("Produce anonymised rater export", value=True)

    n_combos = len(architectures) * len(cases_selected) * int(repetitions)
    est_calls = sum(_CALLS_PER_RUN.get(a, 0) for a in architectures) * len(cases_selected) * int(repetitions)
    if run_evaluation:
        est_calls += len(architectures) * len(cases_selected) * int(repetitions)
    provider_note = (
        f" via {preview_settings.llm_provider}/{preview_settings.effective_model_id}"
        if preview_settings is not None else ""
    )
    st.caption(
        f"Plan: {len(architectures)} architecture(s) × {len(cases_selected)} case(s) × "
        f"{int(repetitions)} repetition(s) = **{n_combos} generation run(s)**, "
        f"~{est_calls} LLM call(s) total{provider_note}."
    )
    if preview_settings is None:
        st.warning("Fix the provider configuration in the sidebar before starting a run.")

    start_clicked = st.button(
        "▶ Start real run", type="primary",
        disabled=not architectures or not cases_selected or preview_settings is None,
    )

    if start_clicked:
        # REPETITIONS folds into the same scoped-override mechanism as the
        # sidebar's provider/model/credential fields, so nothing set here
        # lingers in the process environment after Settings is resolved.
        run_overrides = dict(ui_overrides)
        run_overrides["REPETITIONS"] = str(int(repetitions))
        settings = _resolve_settings(run_overrides, silent=False)
        if settings is not None:
            settings.apply_seed()
            resolved_cases = [get_case(cid) for cid in cases_selected]

            status_label = (
                f"Running {n_combos} generation(s) via "
                f"{settings.llm_provider}/{settings.effective_model_id}…"
            )
            with st.status(status_label, expanded=True) as status:
                log_placeholder = st.empty()
                handler = StreamlitLogHandler(log_placeholder)
                outcomes: list[RunOutcome] = []
                run_dir: Path | None = None
                try:
                    with ExperimentLogger.for_run(settings) as logger:
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
                        )
                except Exception as exc:  # noqa: BLE001 - surfaced to the UI, not swallowed
                    status.update(label=f"Run failed: {exc}", state="error")
                    st.exception(exc)
                else:
                    failed = sum(1 for o in outcomes if o.error is not None)
                    ok = len(outcomes) - failed
                    status.update(
                        label=f"Done: {ok}/{len(outcomes)} generation(s) succeeded",
                        state="complete" if failed == 0 else "error",
                    )

            if run_dir is not None and outcomes:
                st.subheader("Results")
                for outcome in outcomes:
                    label = f"{outcome.architecture} / {outcome.case_id} / rep {outcome.repetition}"
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
# Tab 2 — Browse past runs
# --------------------------------------------------------------------------

with tab_browse:
    st.subheader("Past runs")
    output_dir = _env_settings.output_dir if _env_settings else PROJECT_ROOT / "outputs"
    run_dirs = _list_run_dirs(output_dir)

    if not run_dirs:
        st.info(f"No runs found yet in `{output_dir}`.")
    else:
        selected_run = st.selectbox(
            "Run", options=run_dirs, format_func=lambda p: p.name,
        )
        srs_files = _srs_files(selected_run)
        st.caption(f"{len(srs_files)} SRS document(s) in this run.")

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

        for md_path in srs_files:
            _view_srs(md_path, key_prefix=f"browse_{md_path.stem}")
