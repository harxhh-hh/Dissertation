"""Render a generated SRS Markdown document as a rich, standalone HTML page.

The experiment pipeline writes every SRS as ``srs_<case>_<arch>_rep<n>.md``
— correct for machine processing (evaluation, rater export) but tedious to
actually read. This module turns one such document into a single
self-contained ``.html`` file that behaves like a small document viewer:

* A sticky sidebar table of contents, built from the document's own
  headings, that highlights the section currently in view while scrolling.
* Every top-level (``##``) section becomes a collapsible panel, with a
  toolbar toggle to collapse or expand all of them at once.
* Requirement identifiers (``FR-001``, ``NFR-014``, ...) are rendered as
  small coloured badges instead of plain text, so they're easy to scan.
* A header card summarising run metadata (parsed from the document's own
  "Run metadata" section) plus quick stats (FR/NFR counts, word count,
  estimated reading time).
* In-page search: highlights every match, reports a count, and steps
  through hits with Next/Previous (or the ``/`` keyboard shortcut).
* A light/dark theme toggle independent of the OS preference, a
  scroll-progress bar, a back-to-top button, and print-friendly output
  (every section force-expands before printing).

Everything — CSS, JS, and content — is inlined into one file: no CDN, no
external font, no build step, so it opens correctly from a plain
``file://`` path as well as embedded in the Streamlit UI.

Typical use::

    from ui.render_html import render_srs_html
    html = render_srs_html(srs_path.read_text(), title=srs_path.stem)

Or from the command line::

    python ui/render_html.py outputs/run_.../srs_TC-01_restaurant_app_hierarchical_rep0.md
    python ui/render_html.py <path/to.md> -o <path/to.html>
"""

from __future__ import annotations

import argparse
import html as html_lib
import re
import sys
from pathlib import Path
from typing import Any

import markdown as markdown_lib

#: Markdown extensions enabled for SRS rendering: GitHub-style tables (the
#: rubric/report tables use them), fenced ``` code blocks, a heading-id +
#: toc_tokens source for the sidebar nav, sane-lists so numbered
#: requirement lists don't misparse on blank lines, and nl2br so a single
#: newline inside a paragraph still breaks visually (the agents write
#: prose that relies on that).
_MD_EXTENSIONS = ["tables", "fenced_code", "toc", "sane_lists", "nl2br"]

# --------------------------------------------------------------------------
# Content analysis: metadata card + stats + requirement badges
# --------------------------------------------------------------------------

#: Matches one "- Key: `value`" bullet line inside the "## Run metadata"
#: section every architecture writes (see baseline_single_prompt.py and
#: architectures/hierarchical.py — the format is shared verbatim).
_METADATA_LINE_RE = re.compile(r"^- ([^:\n]+):\s*`([^`]*)`", re.MULTILINE)
_METADATA_SECTION_RE = re.compile(
    r"^## Run metadata\s*\n(.*?)(?=\n##\s|\Z)", re.MULTILINE | re.DOTALL
)

#: FR-/NFR-style requirement identifiers, badge-ified wherever they appear
#: in body text (e.g. "FR-001: The system shall ..." or a cross-reference
#: "see FR-001").
_REQ_ID_RE = re.compile(r"\b(FR|NFR)-(\d{2,4})\b")

#: Human labels + a small emoji per known top-level section, purely
#: cosmetic (falls back to a generic document icon for anything else, so
#: this never has to be kept in lockstep with config/prompts.py).
#: Order matters: substring matching stops at the first hit, so more
#: specific keywords (e.g. "non-functional requirement", which otherwise
#: contains "functional requirement" as a substring) must precede the
#: more general ones they'd otherwise be shadowed by.
_SECTION_ICONS: tuple[tuple[str, str], ...] = (
    ("run metadata", "⚙️"),
    ("input description", "📝"),
    ("orchestrator", "🧭"),
    ("overview", "📋"),
    ("non-functional requirement", "⚡"),
    ("functional requirement", "✅"),
    ("risk", "🚨"),
    ("verification", "🔎"),
    ("rebuttal", "💬"),
    ("arbitration", "⚖️"),
)


def _parse_metadata(markdown_text: str) -> dict[str, str]:
    """Extract the "Run metadata" bullet list as an ordered key/value dict."""
    match = _METADATA_SECTION_RE.search(markdown_text)
    if not match:
        return {}
    return {key.strip(): value.strip() for key, value in _METADATA_LINE_RE.findall(match.group(1))}


def _compute_stats(markdown_text: str) -> dict[str, Any]:
    """Compute quick-glance stats shown in the header card."""
    n_fr = len({m.group(0) for m in _REQ_ID_RE.finditer(markdown_text) if m.group(1) == "FR"})
    n_nfr = len({m.group(0) for m in _REQ_ID_RE.finditer(markdown_text) if m.group(1) == "NFR"})
    n_words = len(markdown_text.split())
    return {
        "fr": n_fr,
        "nfr": n_nfr,
        "words": n_words,
        "reading_minutes": max(1, round(n_words / 200)),
    }


def _badge_requirement_ids(body_html: str) -> str:
    """Wrap every FR-/NFR-NNN mention in the rendered HTML with a badge span."""

    def _sub(m: re.Match[str]) -> str:
        kind, number = m.group(1), m.group(2)
        css_class = "badge-fr" if kind == "FR" else "badge-nfr"
        return f'<span class="req-badge {css_class}">{kind}-{number}</span>'

    return _REQ_ID_RE.sub(_sub, body_html)


def _icon_for_heading(heading_text: str) -> str:
    lowered = heading_text.lower()
    for keyword, icon in _SECTION_ICONS:
        if keyword in lowered:
            return icon
    return "📄"


# --------------------------------------------------------------------------
# Structural transform: body HTML -> collapsible sections + TOC
# --------------------------------------------------------------------------

_H2_SPLIT_RE = re.compile(r"(<h2\b[^>]*>.*?</h2>)", re.DOTALL)
_ID_ATTR_RE = re.compile(r'id="([^"]*)"')
_STRIP_TAGS_RE = re.compile(r"<[^>]+>")


def _split_into_sections(body_html: str) -> tuple[str, list[tuple[str, str]]]:
    """Split rendered body HTML at each ``<h2>`` boundary.

    Returns:
        A tuple of (content before the first ``<h2>``, list of
        ``(h2_tag, following_html_until_next_h2)`` pairs).
    """
    parts = _H2_SPLIT_RE.split(body_html)
    preamble = parts[0]
    sections: list[tuple[str, str]] = []
    for i in range(1, len(parts), 2):
        h2_tag = parts[i]
        content = parts[i + 1] if i + 1 < len(parts) else ""
        sections.append((h2_tag, content))
    return preamble, sections


def _render_sections(sections: list[tuple[str, str]]) -> str:
    """Wrap each (h2, content) pair as a collapsible ``<details>`` panel."""
    blocks: list[str] = []
    for h2_tag, content in sections:
        id_match = _ID_ATTR_RE.search(h2_tag)
        section_id = id_match.group(1) if id_match else ""
        heading_text = _STRIP_TAGS_RE.sub("", h2_tag).strip()
        icon = _icon_for_heading(heading_text)
        id_attr = f' id="{html_lib.escape(section_id)}"' if section_id else ""
        blocks.append(
            f'<details class="srs-section" open{id_attr}>'
            f'<summary><span class="sec-icon">{icon}</span>'
            f'<span class="sec-title">{html_lib.escape(heading_text)}</span>'
            f'<span class="sec-chevron">▸</span></summary>'
            f'<div class="srs-section-body">{content}</div>'
            f"</details>"
        )
    return "\n".join(blocks)


def _flatten_toc(tokens: list[dict[str, Any]], *, max_level: int = 3) -> str:
    """Render ``toc_tokens`` (from the ``toc`` markdown extension) as nested ``<ul>`` nav."""
    items: list[str] = []
    for token in tokens:
        level = token.get("level", 2)
        children_html = _flatten_toc(token.get("children", []), max_level=max_level)
        if level == 1:
            # The document's single <h1> title isn't a nav target — skip it
            # but keep walking into its children (the real ## sections).
            items.append(children_html)
            continue
        if level > max_level:
            continue
        name = html_lib.escape(token.get("name", ""))
        anchor = html_lib.escape(token.get("id", ""))
        items.append(
            f'<li class="toc-l{level}"><a href="#{anchor}" class="toc-link" '
            f'data-target="{anchor}">{name}</a>{children_html}</li>'
        )
    return f'<ul>{"".join(items)}</ul>' if items else ""


def _render_header(*, title: str, metadata: dict[str, str], stats: dict[str, Any]) -> str:
    """Build the header card: title, metadata pills, and quick stats."""
    pills = "".join(
        f'<span class="meta-pill"><span class="meta-key">{html_lib.escape(key)}</span>'
        f'<span class="meta-val">{html_lib.escape(value)}</span></span>'
        for key, value in metadata.items()
    )
    stat_items = [
        ("FR", stats["fr"]),
        ("NFR", stats["nfr"]),
        ("words", f'{stats["words"]:,}'),
        ("read", f'~{stats["reading_minutes"]} min'),
    ]
    stat_html = "".join(
        f'<div class="stat-chip"><span class="stat-num">{value}</span>'
        f'<span class="stat-label">{label}</span></div>'
        for label, value in stat_items
    )
    return (
        '<header class="srs-header">'
        f'<h1>{html_lib.escape(title)}</h1>'
        f'<div class="meta-pills">{pills}</div>'
        f'<div class="stat-chips">{stat_html}</div>'
        "</header>"
    )


# --------------------------------------------------------------------------
# Page shell: CSS + JS (static; content is spliced in as plain string data,
# never through str.format(), so none of the braces below need escaping)
# --------------------------------------------------------------------------

_CSS = """
:root {
  --bg: #ffffff;
  --bg-elevated: #f7f9fb;
  --fg: #1a1a1a;
  --muted: #5b6470;
  --border: #dde1e6;
  --accent: #2563eb;
  --accent-soft: #eaf1ff;
  --code-bg: #f4f6f8;
  --quote-bg: #f7f9fb;
  --table-stripe: #f7f9fb;
  --badge-fr-bg: #e6f4ea;
  --badge-fr-fg: #1e7a34;
  --badge-nfr-bg: #eef0ff;
  --badge-nfr-fg: #4338ca;
  --shadow: 0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.06);
}
:root[data-theme="dark"] {
  --bg: #12151a;
  --bg-elevated: #191d24;
  --fg: #e8eaed;
  --muted: #9aa4b2;
  --border: #2a2f37;
  --accent: #6ea8fe;
  --accent-soft: #1c2942;
  --code-bg: #1c2027;
  --quote-bg: #1a1e24;
  --table-stripe: #171b21;
  --badge-fr-bg: #163524;
  --badge-fr-fg: #6fd88f;
  --badge-nfr-bg: #23244a;
  --badge-nfr-fg: #b3b6ff;
  --shadow: 0 1px 3px rgba(0,0,0,0.4);
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #12151a;
    --bg-elevated: #191d24;
    --fg: #e8eaed;
    --muted: #9aa4b2;
    --border: #2a2f37;
    --accent: #6ea8fe;
    --accent-soft: #1c2942;
    --code-bg: #1c2027;
    --quote-bg: #1a1e24;
    --table-stripe: #171b21;
    --badge-fr-bg: #163524;
    --badge-fr-fg: #6fd88f;
    --badge-nfr-bg: #23244a;
    --badge-nfr-fg: #b3b6ff;
    --shadow: 0 1px 3px rgba(0,0,0,0.4);
  }
}
* { box-sizing: border-box; }
html, body {
  margin: 0; padding: 0;
  background: var(--bg); color: var(--fg);
}
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  line-height: 1.6;
}
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }

/* Progress bar */
#scroll-progress {
  position: fixed; top: 0; left: 0; height: 3px; width: 0%;
  background: var(--accent); z-index: 1000; transition: width 80ms linear;
}

/* Toolbar */
.toolbar {
  position: fixed; top: 10px; right: 14px; z-index: 900;
  display: flex; gap: 6px;
}
.toolbar button {
  border: 1px solid var(--border); background: var(--bg-elevated); color: var(--fg);
  border-radius: 8px; padding: 6px 10px; font-size: 0.85rem; cursor: pointer;
  box-shadow: var(--shadow);
}
.toolbar button:hover { border-color: var(--accent); }

/* Layout */
.layout { display: flex; align-items: flex-start; max-width: 1180px; margin: 0 auto; }
.sidebar {
  width: 260px; flex: 0 0 260px; position: sticky; top: 0;
  height: 100vh; overflow-y: auto; padding: 60px 14px 40px 20px;
  border-right: 1px solid var(--border);
}
.sidebar h2.toc-heading { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); margin: 0 0 10px; }
.sidebar ul { list-style: none; margin: 0; padding-left: 0; }
.sidebar ul ul { padding-left: 14px; }
.sidebar li { margin: 2px 0; }
.toc-link {
  display: block; padding: 5px 8px; border-radius: 6px; color: var(--muted);
  font-size: 0.88rem; border-left: 2px solid transparent;
}
.toc-l3 .toc-link { font-size: 0.82rem; opacity: 0.85; }
.toc-link:hover { background: var(--bg-elevated); color: var(--fg); text-decoration: none; }
.toc-link.active { color: var(--accent); border-left-color: var(--accent); background: var(--accent-soft); font-weight: 600; }

.search-box { margin-bottom: 16px; }
.search-box input {
  width: 100%; padding: 7px 10px; border-radius: 8px; border: 1px solid var(--border);
  background: var(--bg); color: var(--fg); font-size: 0.85rem;
}
.search-meta { display: flex; align-items: center; justify-content: space-between; margin-top: 6px; font-size: 0.78rem; color: var(--muted); }
.search-meta button {
  border: 1px solid var(--border); background: var(--bg-elevated); color: var(--fg);
  border-radius: 5px; padding: 1px 7px; cursor: pointer; font-size: 0.78rem;
}

.main { flex: 1 1 auto; min-width: 0; padding: 60px 40px 80px; max-width: 900px; margin: 0 auto; }

/* Header card */
.srs-header {
  background: var(--bg-elevated); border: 1px solid var(--border); border-radius: 12px;
  padding: 1.4rem 1.6rem; margin-bottom: 1.8rem;
}
.srs-header h1 { margin: 0 0 0.8rem; font-size: 1.5rem; line-height: 1.35; }
.meta-pills { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 0.9rem; }
.meta-pill {
  display: inline-flex; gap: 5px; align-items: baseline; font-size: 0.78rem;
  background: var(--bg); border: 1px solid var(--border); border-radius: 999px; padding: 3px 10px;
}
.meta-key { color: var(--muted); }
.meta-val { font-weight: 600; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
.stat-chips { display: flex; gap: 10px; flex-wrap: wrap; }
.stat-chip {
  display: flex; flex-direction: column; align-items: center; justify-content: center;
  background: var(--bg); border: 1px solid var(--border); border-radius: 10px;
  padding: 8px 14px; min-width: 64px;
}
.stat-num { font-size: 1.1rem; font-weight: 700; color: var(--accent); }
.stat-label { font-size: 0.7rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.04em; }

/* Collapsible sections */
.srs-section {
  border: 1px solid var(--border); border-radius: 10px; margin-bottom: 14px;
  background: var(--bg); overflow: hidden;
}
.srs-section > summary {
  list-style: none; cursor: pointer; padding: 0.85rem 1.1rem; font-size: 1.05rem;
  font-weight: 600; display: flex; align-items: center; gap: 0.6rem;
  background: var(--bg-elevated);
}
.srs-section > summary::-webkit-details-marker { display: none; }
.sec-chevron { margin-left: auto; color: var(--muted); transition: transform 120ms ease; font-size: 0.85rem; }
.srs-section[open] > summary .sec-chevron { transform: rotate(90deg); }
.srs-section-body { padding: 1rem 1.3rem 1.3rem; }
.srs-section-body h3 { color: var(--accent); margin-top: 1.3rem; }
.srs-section-body h3:first-child { margin-top: 0; }

p, ul, ol { margin: 0.7rem 0; }
ul, ol { padding-left: 1.4rem; }
li { margin: 0.25rem 0; }
blockquote {
  margin: 1rem 0; padding: 0.6rem 1rem; background: var(--quote-bg);
  border-left: 3px solid var(--accent); color: var(--muted);
}
code {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 0.9em;
  background: var(--code-bg); padding: 0.15em 0.35em; border-radius: 4px;
}
pre { background: var(--code-bg); border: 1px solid var(--border); border-radius: 6px; padding: 0.9rem 1rem; overflow-x: auto; }
pre code { background: none; padding: 0; }
table { border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: 0.92rem; }
th, td { border: 1px solid var(--border); padding: 0.5rem 0.7rem; text-align: left; vertical-align: top; }
th { background: var(--code-bg); font-weight: 600; }
tr:nth-child(even) td { background: var(--table-stripe); }
hr { border: none; border-top: 1px solid var(--border); margin: 2rem 0; }
strong { font-weight: 600; }
details { border: 1px solid var(--border); border-radius: 8px; padding: 0.4rem 0.7rem; margin: 0.6rem 0; }
details summary { cursor: pointer; }

/* Requirement badges */
.req-badge {
  display: inline-block; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 0.82em; font-weight: 700; padding: 0.05em 0.5em; border-radius: 5px; letter-spacing: 0.01em;
}
.badge-fr { background: var(--badge-fr-bg); color: var(--badge-fr-fg); }
.badge-nfr { background: var(--badge-nfr-bg); color: var(--badge-nfr-fg); }

/* Search highlight */
mark.search-hit { background: #ffe58a; color: #1a1a1a; border-radius: 3px; padding: 0 1px; }
mark.search-hit.current { background: #ff9d2e; }

/* Back to top */
#back-to-top {
  position: fixed; bottom: 20px; right: 20px; z-index: 900;
  border: 1px solid var(--border); background: var(--bg-elevated); color: var(--fg);
  border-radius: 999px; width: 40px; height: 40px; cursor: pointer; box-shadow: var(--shadow);
  opacity: 0; pointer-events: none; transition: opacity 150ms ease;
  font-size: 1rem;
}
#back-to-top.visible { opacity: 1; pointer-events: auto; }

/* Mobile sidebar */
.sidebar-toggle { display: none; }
@media (max-width: 900px) {
  .sidebar {
    position: fixed; left: 0; top: 0; z-index: 950; background: var(--bg);
    width: 78vw; max-width: 300px; transform: translateX(-100%);
    transition: transform 160ms ease; box-shadow: var(--shadow);
  }
  .sidebar.open { transform: translateX(0); }
  .sidebar-toggle {
    display: inline-flex; position: fixed; top: 10px; left: 14px; z-index: 960;
    border: 1px solid var(--border); background: var(--bg-elevated); color: var(--fg);
    border-radius: 8px; padding: 6px 10px; cursor: pointer; box-shadow: var(--shadow);
  }
  .main { padding: 60px 20px 60px; }
}

@media print {
  .toolbar, .sidebar, .sidebar-toggle, #back-to-top, #scroll-progress { display: none !important; }
  .main { max-width: 100%; padding: 0; }
  .srs-section { break-inside: avoid; border: none; }
  .srs-section > summary { background: none; }
}
"""

_JS = """
(function () {
  var root = document.documentElement;

  // ---- Theme toggle (independent of OS preference, session-local) ----
  var themeBtn = document.getElementById('theme-toggle');
  function applyTheme(mode) {
    if (mode === 'light' || mode === 'dark') { root.setAttribute('data-theme', mode); }
    else { root.removeAttribute('data-theme'); }
    themeBtn.textContent = mode === 'dark' ? '\\u2600\\uFE0F Light' : mode === 'light' ? '\\uD83C\\uDF19 Dark' : '\\uD83C\\uDF13 Auto';
  }
  var savedTheme = null;
  try { savedTheme = localStorage.getItem('srs-theme'); } catch (e) {}
  applyTheme(savedTheme || 'auto');
  themeBtn.addEventListener('click', function () {
    var current = root.getAttribute('data-theme') || 'auto';
    var next = current === 'auto' ? 'dark' : current === 'dark' ? 'light' : 'auto';
    applyTheme(next);
    try { localStorage.setItem('srs-theme', next); } catch (e) {}
  });

  // ---- Collapse / expand all ----
  var sections = Array.prototype.slice.call(document.querySelectorAll('.srs-section'));
  var collapseBtn = document.getElementById('collapse-toggle');
  collapseBtn.addEventListener('click', function () {
    var anyOpen = sections.some(function (s) { return s.open; });
    sections.forEach(function (s) { s.open = !anyOpen; });
    collapseBtn.textContent = anyOpen ? '\\u25B8 Expand all' : '\\u25BE Collapse all';
  });

  // ---- Sidebar toggle (mobile) ----
  var sidebar = document.querySelector('.sidebar');
  var sidebarToggle = document.getElementById('sidebar-toggle');
  if (sidebarToggle) {
    sidebarToggle.addEventListener('click', function () { sidebar.classList.toggle('open'); });
  }

  // ---- TOC clicks: scroll within this document, never let the browser
  // handle the fragment natively. This page is normally embedded via
  // Streamlit's components.v1.html(), which loads it into an iframe via
  // `srcdoc` — some browsers resolve a bare `href="#id"` against the
  // *embedding* page's URL in that context instead of this document,
  // which reads as the click "redirecting to the Streamlit tab" instead
  // of jumping to the section. Doing the scroll ourselves sidesteps that
  // entirely, regardless of the browser's srcdoc fragment-resolution quirk.
  document.querySelectorAll('.toc-link').forEach(function (a) {
    a.addEventListener('click', function (e) {
      var id = a.dataset.target;
      var target = id && document.getElementById(id);
      if (target) {
        e.preventDefault();
        var section = target.closest('.srs-section') || target;
        if (section.tagName === 'DETAILS' && !section.open) { section.open = true; }
        target.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
      if (sidebar) { sidebar.classList.remove('open'); }
    });
  });

  // ---- Scroll progress + back-to-top ----
  var progress = document.getElementById('scroll-progress');
  var backToTop = document.getElementById('back-to-top');
  function onScroll() {
    var doc = document.documentElement;
    var scrollable = doc.scrollHeight - doc.clientHeight;
    var pct = scrollable > 0 ? (doc.scrollTop / scrollable) * 100 : 0;
    progress.style.width = pct + '%';
    backToTop.classList.toggle('visible', doc.scrollTop > 400);
  }
  document.addEventListener('scroll', onScroll, { passive: true });
  onScroll();
  backToTop.addEventListener('click', function () {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  });

  // ---- Active-section highlighting in the sidebar ----
  var headings = Array.prototype.slice.call(document.querySelectorAll('.srs-section[id], .srs-section-body h3[id]'));
  var links = {};
  document.querySelectorAll('.toc-link').forEach(function (a) { links[a.dataset.target] = a; });
  if ('IntersectionObserver' in window && headings.length) {
    var observer = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        var link = links[entry.target.id];
        if (!link) return;
        if (entry.isIntersecting) {
          Object.keys(links).forEach(function (id) { links[id].classList.remove('active'); });
          link.classList.add('active');
        }
      });
    }, { rootMargin: '-15% 0px -70% 0px', threshold: 0 });
    headings.forEach(function (h) { observer.observe(h); });
  }

  // ---- Search: highlight matches, step through with Next/Prev ----
  var searchInput = document.getElementById('search-input');
  var searchCount = document.getElementById('search-count');
  var searchPrev = document.getElementById('search-prev');
  var searchNext = document.getElementById('search-next');
  var hits = [];
  var currentHit = -1;

  function clearHighlights() {
    document.querySelectorAll('mark.search-hit').forEach(function (mark) {
      var parent = mark.parentNode;
      parent.replaceChild(document.createTextNode(mark.textContent), mark);
      parent.normalize();
    });
    hits = [];
    currentHit = -1;
  }

  function highlight(term) {
    clearHighlights();
    if (!term) { searchCount.textContent = ''; return; }
    var lower = term.toLowerCase();
    var walker = document.createTreeWalker(
      document.querySelector('.main'), NodeFilter.SHOW_TEXT,
      { acceptNode: function (node) {
          if (!node.nodeValue.toLowerCase().includes(lower)) return NodeFilter.FILTER_REJECT;
          if (node.parentNode.closest('script,style,mark')) return NodeFilter.FILTER_REJECT;
          return NodeFilter.FILTER_ACCEPT;
        } }
    );
    var textNodes = [];
    var n;
    while ((n = walker.nextNode())) textNodes.push(n);
    textNodes.forEach(function (node) {
      var text = node.nodeValue;
      var lowerText = text.toLowerCase();
      var frag = document.createDocumentFragment();
      var last = 0, idx;
      while ((idx = lowerText.indexOf(lower, last)) !== -1) {
        frag.appendChild(document.createTextNode(text.slice(last, idx)));
        var mark = document.createElement('mark');
        mark.className = 'search-hit';
        mark.textContent = text.slice(idx, idx + term.length);
        frag.appendChild(mark);
        hits.push(mark);
        last = idx + term.length;
      }
      frag.appendChild(document.createTextNode(text.slice(last)));
      node.parentNode.replaceChild(frag, node);
    });
    searchCount.textContent = hits.length ? ('1 / ' + hits.length) : 'no matches';
    if (hits.length) goToHit(0);
  }

  function goToHit(i) {
    if (!hits.length) return;
    if (currentHit >= 0 && hits[currentHit]) hits[currentHit].classList.remove('current');
    currentHit = ((i % hits.length) + hits.length) % hits.length;
    var hit = hits[currentHit];
    hit.classList.add('current');
    var section = hit.closest('.srs-section');
    if (section && !section.open) section.open = true;
    hit.scrollIntoView({ behavior: 'smooth', block: 'center' });
    searchCount.textContent = (currentHit + 1) + ' / ' + hits.length;
  }

  var debounceTimer;
  searchInput.addEventListener('input', function () {
    clearTimeout(debounceTimer);
    var value = searchInput.value.trim();
    debounceTimer = setTimeout(function () { highlight(value); }, 150);
  });
  searchNext.addEventListener('click', function () { goToHit(currentHit + 1); });
  searchPrev.addEventListener('click', function () { goToHit(currentHit - 1); });
  searchInput.addEventListener('keydown', function (e) {
    if (e.key === 'Enter') { e.preventDefault(); goToHit(e.shiftKey ? currentHit - 1 : currentHit + 1); }
  });
  document.addEventListener('keydown', function (e) {
    if (e.key === '/' && document.activeElement !== searchInput) {
      var tag = (document.activeElement && document.activeElement.tagName) || '';
      if (tag !== 'INPUT' && tag !== 'TEXTAREA') { e.preventDefault(); searchInput.focus(); }
    }
  });

  // ---- Print: force every section open, keep it that way ----
  window.addEventListener('beforeprint', function () {
    sections.forEach(function (s) { s.open = true; });
  });
})();
"""


def _build_page(*, title: str, header: str, toc: str, preamble: str, sections_html: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html_lib.escape(title)}</title>
<style>
{_CSS}
</style>
</head>
<body>
<div id="scroll-progress"></div>
<button id="sidebar-toggle" class="sidebar-toggle" aria-label="Toggle table of contents">&#9776;</button>
<div class="toolbar">
  <button id="collapse-toggle" type="button">&#9662; Collapse all</button>
  <button id="theme-toggle" type="button">&#127763; Auto</button>
  <button id="print-btn" type="button" onclick="window.print()">&#128424; Print</button>
</div>
<div class="layout">
  <nav class="sidebar">
    <h2 class="toc-heading">Contents</h2>
    <div class="search-box">
      <input id="search-input" type="text" placeholder="Search this document&hellip; (press /)" autocomplete="off">
      <div class="search-meta">
        <span id="search-count"></span>
        <span>
          <button id="search-prev" type="button" title="Previous match">&uarr;</button>
          <button id="search-next" type="button" title="Next match">&darr;</button>
        </span>
      </div>
    </div>
    {toc}
  </nav>
  <main class="main">
    {header}
    {preamble}
    {sections_html}
  </main>
</div>
<button id="back-to-top" type="button" aria-label="Back to top">&#8593;</button>
<script>
{_JS}
</script>
</body>
</html>
"""


def render_srs_html(markdown_text: str, *, title: str) -> str:
    """Convert one SRS Markdown document into a rich, standalone HTML page.

    See the module docstring for the full feature list (sticky TOC with
    active-section tracking, collapsible sections, requirement badges,
    in-page search, theme toggle, print support).

    Args:
        markdown_text: The raw Markdown, as written by the pipeline
            (``srs_*.md``).
        title: Page ``<title>`` and header ``<h1>``, typically the source
            file's stem (e.g. ``srs_TC-01_restaurant_app_hierarchical_rep0``).

    Returns:
        A complete, self-contained HTML document as a string — safe to
        write directly to a ``.html`` file or hand to a browser component.
    """
    metadata = _parse_metadata(markdown_text)
    stats = _compute_stats(markdown_text)

    converter = markdown_lib.Markdown(
        extensions=_MD_EXTENSIONS,
        extension_configs={"toc": {"permalink": False}},
    )
    body_html = converter.convert(markdown_text)
    toc_tokens = getattr(converter, "toc_tokens", [])

    body_html = _badge_requirement_ids(body_html)
    preamble, sections = _split_into_sections(body_html)

    return _build_page(
        title=title,
        header=_render_header(title=title, metadata=metadata, stats=stats),
        toc=_flatten_toc(toc_tokens),
        preamble=preamble,
        sections_html=_render_sections(sections),
    )


def render_srs_file(md_path: Path, *, out_path: Path | None = None) -> Path:
    """Render one SRS ``.md`` file on disk to a sibling (or given) ``.html`` file.

    Args:
        md_path: Path to the source Markdown file.
        out_path: Destination path. Defaults to ``md_path`` with its
            suffix swapped to ``.html``.

    Returns:
        The path the HTML was written to.
    """
    markdown_text = md_path.read_text(encoding="utf-8")
    html_doc = render_srs_html(markdown_text, title=md_path.stem)
    destination = out_path if out_path is not None else md_path.with_suffix(".html")
    destination.write_text(html_doc, encoding="utf-8")
    return destination


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("markdown_path", type=Path, help="Path to an srs_*.md file.")
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Output .html path (default: same name, .html extension).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: render one Markdown file to HTML and print the path."""
    args = _parse_args(argv)
    if not args.markdown_path.is_file():
        print(f"error: no such file: {args.markdown_path}", file=sys.stderr)
        return 2
    out_path = render_srs_file(args.markdown_path, out_path=args.output)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
