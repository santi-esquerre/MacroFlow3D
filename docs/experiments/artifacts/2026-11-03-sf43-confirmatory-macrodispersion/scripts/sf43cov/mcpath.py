"""Makes the SF-42 ``mcstat`` package importable (it lives in the SF-42 artifact, which is not modified).  Import for the side effect."""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(_HERE)  # .../2026-11-03-sf43-confirmatory-macrodispersion/scripts
ARTIFACTS = os.path.dirname(os.path.dirname(SCRIPTS))
MCSTAT_ROOT = os.path.join(ARTIFACTS, "2026-10-30-sf42-confirmatory-tooling", "scripts")
if os.path.isdir(os.path.join(MCSTAT_ROOT, "mcstat")) and MCSTAT_ROOT not in sys.path:
    sys.path.insert(0, MCSTAT_ROOT)
