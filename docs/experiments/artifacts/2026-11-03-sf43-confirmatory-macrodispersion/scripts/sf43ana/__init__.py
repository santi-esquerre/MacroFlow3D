"""sf43ana: frozen analysis package of SF-43 (Gate 4 core: estimands, error budgets, robust sign, Holm, classes).

Run from the directory that contains this package::

    python3 -m sf43ana run --prereg P.json --campaign C --reduced R --gate5 G.json|none --out O
    python3 -W error -m unittest discover -s . -t .

Imports ``mcstat`` (SF-42 statistics library, unchanged) and ``sf42red`` (reduced-table format) from the SF-42 artifact scripts
directory, located relative to this file or by the environment variable ``SF42_SCRIPTS``.
stdlib + numpy + scipy only (host stack: Python 3.11 / numpy 1.26 / scipy 1.11).
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
SF42_SCRIPTS = os.environ.get("SF42_SCRIPTS") or os.path.normpath(
    os.path.join(_HERE, "..", "..", "..", "2026-10-30-sf42-confirmatory-tooling", "scripts"))
if SF42_SCRIPTS not in sys.path:
    sys.path.append(SF42_SCRIPTS)
import mcstat  # noqa: E402,F401  (fails loudly if the SF-42 scripts are not found)
