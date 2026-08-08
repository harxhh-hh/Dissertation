"""Bridge between :class:`~utils.logging.ExperimentLogger` and a live Streamlit view.

``ExperimentLogger`` already writes every narrative line (``LLM call | ...``,
``=== hierarchical / TC-01... / rep 0 ===``, warnings, errors) through a
private stdlib ``logging.Logger``. Rather than teach that module anything
about Streamlit, this module supplies a plain ``logging.Handler`` that the
UI attaches via ``ExperimentLogger.add_handler()``. Because every LLM call
in this codebase is synchronous, ``emit()`` fires in the same script thread
Streamlit is already running, so writing into a placeholder here is enough
to stream updates to the browser live — no extra WebSocket plumbing needed
beyond what Streamlit's own runtime already provides.
"""

from __future__ import annotations

import logging
from typing import Any


class StreamlitLogHandler(logging.Handler):
    """Appends each formatted record to a Streamlit placeholder, live.

    Args:
        placeholder: A ``st.empty()`` placeholder (or any object exposing a
            ``.code(text, language=...)`` method) to render into.
        max_lines: Oldest lines beyond this count are dropped so a long run
            does not grow the DOM (and the page) without bound.
    """

    def __init__(self, placeholder: Any, *, max_lines: int = 500) -> None:
        super().__init__()
        self.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
        )
        self._placeholder = placeholder
        self._lines: list[str] = []
        self._max_lines = max_lines

    def emit(self, record: logging.LogRecord) -> None:
        try:
            line = self.format(record)
        except Exception:  # noqa: BLE001 - formatting must never break a run
            line = record.getMessage()
        self._lines.append(line)
        if len(self._lines) > self._max_lines:
            self._lines = self._lines[-self._max_lines :]
        self._placeholder.code("\n".join(self._lines), language="log")


__all__ = ["StreamlitLogHandler"]
