"""Browser UI for the multi-agent SRS generation experiment.

Kept separate from the core pipeline (``agents/``, ``architectures/``,
``config/``, ``utils/``) so the dashboard can be skipped entirely (no
``streamlit`` or ``markdown`` install needed) by anyone only running the
CLI (``run_experiment.py``) or the smoke test. This package imports the
core pipeline; the core pipeline never imports this package.
"""
