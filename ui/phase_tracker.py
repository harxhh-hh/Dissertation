"""Turns the raw narrative log stream into a clean, structured progress view.

``ExperimentLogger`` narrative lines follow a few fixed shapes (see the
``logger.info(...)`` call sites across ``architectures/*.py``,
``baseline_single_prompt.py``, ``run_experiment.py`` and
``evaluation/*.py``):

* ``=== <architecture> / <case_id> / rep <n> ===`` — a new generation combo
  starts.
* ``[<architecture>/<case_id> rep=<n>] <step description>`` — a named step
  within that combo (e.g. "orchestrator planning", "verification round 1").
* ``=== evaluation ===`` / ``=== grounded scoring ===`` /
  ``=== rater export ===`` — top-level pipeline stages that run once, after
  every combo has generated.
* ``Run FAILED: <architecture> / <case_id> / rep <n>: ...`` — one combo
  failed.
* ``Run <id> finished | ...`` — the whole run is done.

That's already a clean, machine-parseable structure — it's just being
displayed as a raw timestamped stream today, which reads fine to whoever
wrote the logging calls and to nobody else. :class:`RunProgress` consumes
those lines incrementally; :func:`render_progress` turns the result into a
simple nested checklist (architecture/case → its steps, then the shared
pipeline stages below), each item marked done / in progress / not started.
The raw log line is never thrown away — see
:class:`~ui.log_bridge.StreamlitLogHandler` for that — this is a second,
friendlier view onto the exact same stream.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

import streamlit as st

_COMBO_START_RE = re.compile(r"^=== (?P<architecture>\S+) / (?P<case_id>\S+) / rep (?P<rep>\d+) ===$")
_STEP_RE = re.compile(r"^\[(?P<architecture>[^/]+)/(?P<case_id>[^\]]+?) rep=(?P<rep>\d+)\] (?P<step>.+)$")
_STAGE_RE = re.compile(r"^=== (?P<stage>evaluation|grounded scoring|rater export) ===$")
_COMBO_FAILED_RE = re.compile(r"^Run FAILED: (?P<architecture>\S+) / (?P<case_id>\S+) / rep (?P<rep>\d+): (?P<reason>.+)$")
_RUN_FINISHED_RE = re.compile(r"^Run \S+ finished \|")
_RUBRIC_LEADER_RE = re.compile(r"^📋 Rubric leader: (?P<text>.+)$")
_GROUNDED_WINNER_RE = re.compile(r"^🏆 Grounded winner: (?P<text>.+)$")

#: Mirrors ui/app.py's _ARCHITECTURE_LABELS. Kept as its own copy rather
#: than imported, so this module has no dependency on app.py (which itself
#: imports from here) — avoids a circular import for a five-line dict.
_ARCHITECTURE_LABELS: dict[str, str] = {
    "baseline_single_prompt": "Baseline (single prompt)",
    "hierarchical": "Hierarchical (top-down)",
    "peer_to_peer": "Peer-to-peer (mutual review)",
    "debate": "Debate (competing + arbitrated)",
}

#: baseline_single_prompt.py logs its steps as "[baseline/... rep=N]",
#: not the canonical "baseline_single_prompt" the "=== ... ===" combo
#: marker uses (see run_experiment.py's ALL_ARCHITECTURES) — every other
#: architecture's short form already matches its canonical name. Without
#: this alias the two regexes below would key the same combo two
#: different ways and it would render as two separate, half-empty combos.
_ARCHITECTURE_ALIASES: dict[str, str] = {"baseline": "baseline_single_prompt"}


def _canonical_arch(name: str) -> str:
    return _ARCHITECTURE_ALIASES.get(name, name)

#: Top-level pipeline stages, in the fixed order run_matrix() runs them.
_STAGE_ORDER: tuple[str, ...] = ("generation", "evaluation", "grounded scoring", "rater export")
_STAGE_LABELS: dict[str, str] = {
    "generation": "Generate documents",
    "evaluation": "LLM-as-judge evaluation",
    "grounded scoring": "Grounded scoring (knowledge base)",
    "rater export": "Anonymised rater export",
}


@dataclass
class ComboProgress:
    """One (architecture, case, repetition)'s generation progress."""

    architecture: str
    case_id: str
    repetition: int
    steps: list[str] = field(default_factory=list)
    failed: str | None = None

    @property
    def key(self) -> tuple[str, str, int]:
        return (self.architecture, self.case_id, self.repetition)


@dataclass
class RunProgress:
    """Incrementally built structured view of one run's progress.

    Call :meth:`feed` once per formatted log line, in order, as they
    arrive. Purely additive state — safe to re-render from at any point.
    """

    combos: list[ComboProgress] = field(default_factory=list)
    stages_reached: list[str] = field(default_factory=lambda: ["generation"])
    finished: bool = False
    rubric_leader: str | None = None
    grounded_winner: str | None = None

    def _combo_by_key(self, key: tuple[str, str, int]) -> ComboProgress | None:
        for c in self.combos:
            if c.key == key:
                return c
        return None

    def feed(self, formatted_line: str) -> None:
        """Parse one already-formatted log line and update tracked state.

        Args:
            formatted_line: A line as produced by the same
                ``"%(asctime)s | %(levelname)-7s | %(message)s"`` formatter
                :class:`~ui.log_bridge.StreamlitLogHandler` uses — the
                ``<time> | LEVEL   | `` prefix is stripped before matching.
        """
        message = formatted_line
        parts = formatted_line.split(" | ", 2)
        if len(parts) == 3:
            message = parts[2]

        if m := _COMBO_START_RE.match(message):
            arch = _canonical_arch(m["architecture"])
            key = (arch, m["case_id"], int(m["rep"]))
            if self._combo_by_key(key) is None:
                self.combos.append(ComboProgress(arch, m["case_id"], int(m["rep"])))
            return

        if m := _STEP_RE.match(message):
            arch = _canonical_arch(m["architecture"])
            key = (arch, m["case_id"], int(m["rep"]))
            combo = self._combo_by_key(key)
            if combo is None:
                combo = ComboProgress(arch, m["case_id"], int(m["rep"]))
                self.combos.append(combo)
            step = m["step"].strip()
            if not combo.steps or combo.steps[-1] != step:
                combo.steps.append(step)
            return

        if m := _STAGE_RE.match(message):
            if m["stage"] not in self.stages_reached:
                self.stages_reached.append(m["stage"])
            return

        if m := _COMBO_FAILED_RE.match(message):
            arch = _canonical_arch(m["architecture"])
            key = (arch, m["case_id"], int(m["rep"]))
            combo = self._combo_by_key(key)
            if combo is not None:
                combo.failed = m["reason"].strip()
            return

        if m := _RUBRIC_LEADER_RE.match(message):
            self.rubric_leader = m["text"].strip()
            return

        if m := _GROUNDED_WINNER_RE.match(message):
            self.grounded_winner = m["text"].strip()
            return

        if _RUN_FINISHED_RE.match(message):
            self.finished = True
            return


def render_progress(progress: RunProgress) -> None:
    """Render a :class:`RunProgress` as a nested Streamlit checklist.

    Meant to be called repeatedly into the same ``st.empty()`` placeholder
    (via ``with placeholder.container():``) as new log lines arrive — each
    call fully replaces the placeholder's content with the latest state.
    """
    if not progress.combos:
        st.caption("Starting…")
        return

    generation_done = progress.finished or any(
        s in progress.stages_reached for s in _STAGE_ORDER[1:]
    )

    for i, combo in enumerate(progress.combos):
        is_last = i == len(progress.combos) - 1
        combo_done = combo.failed is not None or generation_done or not is_last
        icon = "❌" if combo.failed else ("✅" if combo_done else "⏳")
        arch_label = _ARCHITECTURE_LABELS.get(combo.architecture, combo.architecture)
        st.markdown(f"{icon} **{arch_label}** — `{combo.case_id}` (rep {combo.repetition})")

        if combo.steps:
            n_steps = len(combo.steps)
            lines = []
            for j, step in enumerate(combo.steps):
                is_current_step = (j == n_steps - 1) and not combo_done
                step_icon = "○" if is_current_step else "✓"
                lines.append(f"&nbsp;&nbsp;&nbsp;&nbsp;{step_icon}&nbsp; {step}")
            st.markdown("  \n".join(lines), unsafe_allow_html=True)

        if combo.failed:
            st.caption(f"⚠️ {combo.failed}")

    if len(progress.combos) > 0 and (generation_done or progress.finished):
        st.markdown("---")
        for stage in _STAGE_ORDER:
            reached = stage in progress.stages_reached
            is_current = (
                reached and stage != "generation"
                and stage == progress.stages_reached[-1] and not progress.finished
            )
            if is_current:
                icon, label_style = "⏳", ""
            elif reached:
                icon, label_style = "✅", ""
            elif progress.finished:
                # The run is over and this stage never fired — it wasn't
                # skipped mid-way, it was never part of this run's plan
                # (e.g. ad hoc runs never do the rater export). "⬜" would
                # read as "still pending"; this reads as "not part of it".
                icon, label_style = "⏭️", "color:var(--text-secondary-color, gray);"
            else:
                icon, label_style = "⬜", ""
            label = _STAGE_LABELS[stage]
            if label_style:
                st.markdown(f'{icon} <span style="{label_style}">{label} — not run this time</span>', unsafe_allow_html=True)
            else:
                st.markdown(f"{icon} {label}")

        if progress.grounded_winner or progress.rubric_leader:
            st.markdown("")
            if progress.grounded_winner:
                st.markdown(f"🏆 **Grounded winner:** {progress.grounded_winner}")
            if progress.rubric_leader:
                st.markdown(f"📋 **Rubric leader:** {progress.rubric_leader}")


class PhaseTrackingLogHandler(logging.Handler):
    """Feeds every narrative log line into a :class:`RunProgress`, live.

    Companion to :class:`~ui.log_bridge.StreamlitLogHandler` — attach both
    to the same :class:`~utils.logging.ExperimentLogger` to get the clean
    structured view (this one) and the full raw stream (that one) from a
    single run, updating together.

    Args:
        placeholder: A ``st.empty()`` placeholder to render into.
    """

    def __init__(self, placeholder: Any) -> None:
        super().__init__()
        self.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
        )
        self._placeholder = placeholder
        self._progress = RunProgress()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            line = self.format(record)
        except Exception:  # noqa: BLE001 - formatting must never break a run
            line = record.getMessage()
        self._progress.feed(line)
        with self._placeholder.container():
            render_progress(self._progress)


__all__ = ["ComboProgress", "PhaseTrackingLogHandler", "RunProgress", "render_progress"]
