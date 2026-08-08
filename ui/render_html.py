"""Render a generated SRS Markdown document as a clean, standalone HTML page.

The experiment pipeline writes every SRS as ``srs_<case>_<arch>_rep<n>.md``
— correct for machine processing (evaluation, rater export) but tedious to
read as raw Markdown. This module converts one such document into a single
self-contained ``.html`` file: no external stylesheet, font, or script, so
it opens correctly from a plain ``file://`` path as well as from the
Streamlit UI's "view" button.

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
import sys
from pathlib import Path

import markdown as md_lib

#: Markdown extensions enabled for SRS rendering: GitHub-style tables (the
#: rubric/report tables use them), fenced ``` code blocks, a table of
#: contents anchor for every heading (used by the in-page nav), and
#: sane-lists so numbered requirement lists don't misparse on blank lines.
_MD_EXTENSIONS = ["tables", "fenced_code", "toc", "sane_lists", "nl2br"]

_PAGE_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
{css}
</style>
</head>
<body>
<article class="srs">
{body}
</article>
</body>
</html>
"""

# Self-contained CSS: no CDN, no external font. Respects the reader's OS
# theme via prefers-color-scheme rather than assuming light mode, since
# this file is opened standalone as often as it is embedded in the UI.
_CSS = """
:root {
  --bg: #ffffff;
  --fg: #1a1a1a;
  --muted: #5b6470;
  --border: #dde1e6;
  --accent: #2563eb;
  --code-bg: #f4f6f8;
  --quote-bg: #f7f9fb;
  --table-stripe: #f7f9fb;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #14171c;
    --fg: #e8eaed;
    --muted: #9aa4b2;
    --border: #2a2f37;
    --accent: #6ea8fe;
    --code-bg: #1c2027;
    --quote-bg: #1a1e24;
    --table-stripe: #191d23;
  }
}
* { box-sizing: border-box; }
html, body {
  margin: 0;
  padding: 0;
  background: var(--bg);
  color: var(--fg);
}
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
    Helvetica, Arial, sans-serif;
  line-height: 1.6;
}
.srs {
  max-width: 860px;
  margin: 0 auto;
  padding: 2.5rem 1.5rem 4rem;
}
h1, h2, h3, h4 {
  line-height: 1.3;
  font-weight: 600;
  scroll-margin-top: 1rem;
}
h1 {
  font-size: 1.9rem;
  border-bottom: 2px solid var(--border);
  padding-bottom: 0.5rem;
  margin-top: 0;
}
h2 {
  font-size: 1.4rem;
  margin-top: 2.2rem;
  border-bottom: 1px solid var(--border);
  padding-bottom: 0.3rem;
}
h3 {
  font-size: 1.15rem;
  margin-top: 1.6rem;
  color: var(--accent);
}
h4 { font-size: 1rem; margin-top: 1.2rem; }
p, ul, ol { margin: 0.7rem 0; }
ul, ol { padding-left: 1.4rem; }
li { margin: 0.25rem 0; }
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }
blockquote {
  margin: 1rem 0;
  padding: 0.6rem 1rem;
  background: var(--quote-bg);
  border-left: 3px solid var(--accent);
  color: var(--muted);
}
code {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 0.9em;
  background: var(--code-bg);
  padding: 0.15em 0.35em;
  border-radius: 4px;
}
pre {
  background: var(--code-bg);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 0.9rem 1rem;
  overflow-x: auto;
}
pre code { background: none; padding: 0; }
table {
  border-collapse: collapse;
  width: 100%;
  margin: 1rem 0;
  font-size: 0.92rem;
}
th, td {
  border: 1px solid var(--border);
  padding: 0.5rem 0.7rem;
  text-align: left;
  vertical-align: top;
}
th { background: var(--code-bg); font-weight: 600; }
tr:nth-child(even) td { background: var(--table-stripe); }
hr { border: none; border-top: 1px solid var(--border); margin: 2rem 0; }
strong { font-weight: 600; }
em.rationale, p > em:only-child { color: var(--muted); }
"""


def render_srs_html(markdown_text: str, *, title: str) -> str:
    """Convert one SRS Markdown document into a standalone HTML page.

    Args:
        markdown_text: The raw Markdown, as written by the pipeline
            (``srs_*.md``).
        title: Page ``<title>``, typically the source file's stem (e.g.
            ``srs_TC-01_restaurant_app_hierarchical_rep0``).

    Returns:
        A complete, self-contained HTML document as a string — safe to
        write directly to a ``.html`` file or hand to a browser component.
    """
    body_html = md_lib.markdown(markdown_text, extensions=_MD_EXTENSIONS)
    return _PAGE_TEMPLATE.format(
        title=html_lib.escape(title), css=_CSS.strip(), body=body_html
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
