# -*- coding: utf-8 -*-
"""Shared path definitions for the revision scripts (import as the first project import).

HERE    revision/            the scripts (flat, like submitted_v1/validation/)
REPO    repository root
BASE    submitted_v1/        submitted code and outputs: Workflow_cantera.py, CanteraResult/, validation/Result/
RESULT  revision/Result/     every CSV / TXT / PNG written by the revision scripts (committed)
NOTES   revision/notes/      working notes (not committed)
SOURCES revision/sources/    third-party PDFs / data files (not committed; see SOURCES.md)
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
BASE = os.path.join(REPO, "submitted_v1")
RESULT = os.path.join(HERE, "Result"); NOTES = os.path.join(HERE, "notes"); SOURCES = os.path.join(HERE, "sources")
os.makedirs(RESULT, exist_ok=True)
for _p in (BASE, HERE):
    if _p not in sys.path: sys.path.insert(0, _p)
