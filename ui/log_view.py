"""Renders a run's narrative log (``run.log``) as a compact, scrollable panel.

``utils/logging.py`` writes one line per narrative event in the format
``%(asctime)s | %(levelname)-8s | %(name)s | %(message)s``. This module
turns that plain-text file into a small self-contained HTML component —
color-coded by level, filterable by free-text search, toggleable by level
via chips, with jump-to-start/jump-to-end controls — meant to be embedded
via ``streamlit.components.v1.html`` inside the Browse-past-runs tab.

Typical use::

    from ui.log_view import render_log_panel
    st.components.v1.html(render_log_panel(log_text), height=460)
"""

from __future__ import annotations

import html as html_lib
import re

#: Matches the level field out of "<timestamp> | LEVEL    | <logger> | msg".
_LEVEL_RE = re.compile(r"^\S+\s*\|\s*(\w+)\s*\|")

#: Display order + accent colour per level. Anything unrecognised (e.g. a
#: multi-line traceback continuation) is grouped under the last-seen level
#: by the caller, not here.
_LEVEL_ORDER = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
_LEVEL_COLORS = {
    "DEBUG": "#8a94a6",
    "INFO": "#3b82f6",
    "WARNING": "#d97706",
    "ERROR": "#dc2626",
    "CRITICAL": "#dc2626",
}

_CSS = """
.logpanel {
  --bg: #ffffff; --fg: #1a1a1a; --muted: #5b6470; --border: #dde1e6; --bg-elevated: #f4f6f8;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  background: var(--bg); color: var(--fg); border: 1px solid var(--border); border-radius: 10px;
  overflow: hidden; display: flex; flex-direction: column; height: 100%;
}
@media (prefers-color-scheme: dark) {
  .logpanel { --bg: #12151a; --fg: #e8eaed; --muted: #9aa4b2; --border: #2a2f37; --bg-elevated: #191d24; }
}
.logpanel-toolbar {
  display: flex; flex-wrap: wrap; gap: 6px; align-items: center;
  padding: 8px 10px; border-bottom: 1px solid var(--border); background: var(--bg-elevated);
}
.logpanel-toolbar input {
  flex: 1 1 160px; min-width: 120px; padding: 5px 9px; border-radius: 7px; border: 1px solid var(--border);
  background: var(--bg); color: var(--fg); font-size: 0.82rem;
}
.level-chip {
  border: 1px solid var(--border); background: var(--bg); color: var(--muted);
  border-radius: 999px; padding: 2px 9px; font-size: 0.74rem; cursor: pointer; font-weight: 600;
  white-space: nowrap;
}
.level-chip.active { color: #fff; border-color: transparent; }
.logpanel-toolbar .spacer { flex: 1 1 auto; }
.logpanel-toolbar button.nav-btn {
  border: 1px solid var(--border); background: var(--bg); color: var(--fg);
  border-radius: 7px; padding: 4px 9px; font-size: 0.78rem; cursor: pointer;
}
.logpanel-body {
  flex: 1 1 auto; overflow-y: auto; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 0.78rem; padding: 6px 0;
}
.log-line {
  padding: 2px 12px; border-left: 3px solid transparent; white-space: pre-wrap; word-break: break-word;
}
.log-line.level-WARNING { border-left-color: #d97706; background: rgba(217,119,6,0.08); }
.log-line.level-ERROR, .log-line.level-CRITICAL { border-left-color: #dc2626; background: rgba(220,38,38,0.09); }
.log-line.hidden { display: none; }
.log-line mark { background: #ffe58a; color: #1a1a1a; border-radius: 2px; }
.logpanel-empty { padding: 24px; text-align: center; color: var(--muted); font-size: 0.85rem; }
.logpanel-status { padding: 4px 10px; font-size: 0.72rem; color: var(--muted); border-top: 1px solid var(--border); background: var(--bg-elevated); }
"""

_JS_TEMPLATE = """
(function () {
  var root = document.getElementById(%(root_id)s);
  var body = root.querySelector('.logpanel-body');
  var input = root.querySelector('.logpanel-filter');
  var status = root.querySelector('.logpanel-status');
  var lines = Array.prototype.slice.call(root.querySelectorAll('.log-line'));
  var activeLevels = new Set(%(levels_json)s);

  function applyFilter() {
    var term = input.value.trim().toLowerCase();
    var shown = 0;
    lines.forEach(function (el) {
      var level = el.dataset.level;
      var text = el.dataset.raw.toLowerCase();
      var matches = activeLevels.has(level) && (!term || text.indexOf(term) !== -1);
      el.classList.toggle('hidden', !matches);
      if (matches) {
        shown++;
        if (term) {
          var idx = el.dataset.raw.toLowerCase().indexOf(term);
          var raw = el.dataset.raw;
          el.innerHTML = escapeHtml(raw.slice(0, idx)) + '<mark>' + escapeHtml(raw.slice(idx, idx + term.length)) + '</mark>' + escapeHtml(raw.slice(idx + term.length));
        } else {
          el.textContent = el.dataset.raw;
        }
      }
    });
    status.textContent = shown + ' / ' + lines.length + ' lines shown';
  }

  function escapeHtml(s) {
    return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  input.addEventListener('input', applyFilter);
  root.querySelectorAll('.level-chip').forEach(function (chip) {
    chip.addEventListener('click', function () {
      var level = chip.dataset.level;
      if (activeLevels.has(level)) { activeLevels.delete(level); chip.classList.remove('active'); }
      else { activeLevels.add(level); chip.classList.add('active'); }
      applyFilter();
    });
  });
  var toStart = root.querySelector('.nav-start');
  var toEnd = root.querySelector('.nav-end');
  if (toStart) toStart.addEventListener('click', function () { body.scrollTop = 0; });
  if (toEnd) toEnd.addEventListener('click', function () { body.scrollTop = body.scrollHeight; });

  applyFilter();
  body.scrollTop = body.scrollHeight; // start scrolled to the most recent line
})();
"""


def render_log_panel(log_text: str, *, panel_id: str = "logpanel-1") -> str:
    """Render ``run.log`` text as a scrollable, filterable, colour-coded panel.

    Args:
        log_text: The full contents of a ``run.log`` file.
        panel_id: DOM id for this panel; vary it if more than one panel is
            embedded on the same page so their JS scopes don't collide.

    Returns:
        A self-contained HTML fragment (with its own ``<style>``/``<script>``)
        ready for ``streamlit.components.v1.html``.
    """
    lines = log_text.splitlines()
    if not lines:
        return f'<div id="{panel_id}" class="logpanel"><style>{_CSS}</style><div class="logpanel-empty">This run has no log output yet.</div></div>'

    counts: dict[str, int] = {}
    row_html: list[str] = []
    last_level = "INFO"
    for raw_line in lines:
        match = _LEVEL_RE.match(raw_line)
        if match and match.group(1) in _LEVEL_COLORS:
            last_level = match.group(1)
        level = last_level
        counts[level] = counts.get(level, 0) + 1
        escaped = html_lib.escape(raw_line)
        raw_attr = html_lib.escape(raw_line, quote=True)
        row_html.append(
            f'<div class="log-line level-{level}" data-level="{level}" data-raw="{raw_attr}">{escaped}</div>'
        )

    chips_html = "".join(
        f'<button class="level-chip active" data-level="{level}" '
        f'style="--chip-color:{_LEVEL_COLORS[level]}" '
        f'onmouseover="" data-count="{counts[level]}">{level} ({counts[level]})</button>'
        for level in _LEVEL_ORDER
        if level in counts
    )
    # Give each active chip its accent colour as a background (kept out of
    # the static CSS file since it's per-level and data-driven).
    chip_style_fix = "".join(
        f'#{panel_id} .level-chip.active[data-level="{level}"] {{ background: {color}; }}'
        for level, color in _LEVEL_COLORS.items()
    )

    import json as _json  # local import: this module has no other JSON use

    js = _JS_TEMPLATE % {
        "root_id": _json.dumps(panel_id),
        "levels_json": _json.dumps([lvl for lvl in _LEVEL_ORDER if lvl in counts]),
    }

    return f"""<div id="{panel_id}" class="logpanel">
<style>{_CSS}
{chip_style_fix}</style>
<div class="logpanel-toolbar">
  <input class="logpanel-filter" type="text" placeholder="Filter log lines&hellip;">
  {chips_html}
  <span class="spacer"></span>
  <button class="nav-btn nav-start" type="button">&#8593; Start</button>
  <button class="nav-btn nav-end" type="button">&#8595; End</button>
</div>
<div class="logpanel-body">{''.join(row_html)}</div>
<div class="logpanel-status"></div>
</div>
<script>{js}</script>
"""


__all__ = ["render_log_panel"]
