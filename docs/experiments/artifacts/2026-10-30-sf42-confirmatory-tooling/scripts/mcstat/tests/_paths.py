"""Locations of committed SF-39 evidence relative to this artifact (A = the SF-39 artifact root)."""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(os.path.dirname(HERE))
R = os.path.dirname(SCRIPTS)
ARTIFACTS = os.path.dirname(R)
A = os.path.join(ARTIFACTS, "2026-10-27-sf39-macrodispersion-study")
REPO = os.path.dirname(os.path.dirname(os.path.dirname(ARTIFACTS)))
HEAVY = os.environ.get("MCSTAT_HEAVY") == "1"
