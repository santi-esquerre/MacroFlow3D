"""The ONLY place that names campaign-engine keys added by the engine node after the base (``sf43camp/plan.py`` validates the rest).

If the engine node renames a field, change it here.  Fields emitted: top-level ``executor``; per arm ``primary``, ``no_reestimation``,
``hmax_div_first`` ({n, divs} plus the planning extras ``est_seconds_extra`` and ``raw_gb_factor`` that the base engine ignores) and, in the
smoke only, ``lane``; smoke-only ``determinism_pairs``, ``background``, ``background_wait_minutes``.
"""
from collections import OrderedDict

from . import design as D

EXECUTOR = "sf43camp.runner_executor:RunnerExecutor"


def top_level():
    return OrderedDict([("executor", EXECUTOR)])


def arm_fields(e):
    """Engine flags of one arm entry (``build.Entry``)."""
    f = OrderedDict()
    f["primary"] = "primary" in e.row.flags                  # production arms: the arms whose plane arrays survive pruning (H21)
    f["no_reestimation"] = "reest" not in e.row.flags        # Gate 5, ladder and every other arm outside blinded_reestimation.arms
    if e.lane:
        f["lane"] = e.lane
    return f


def consumer_fields(e):
    """Keys added to the consumer dict of an arm whose first n positions run the three h_max passes."""
    return OrderedDict([("hmax_div_first", OrderedDict([("n", e.hm_n), ("divs", list(D.HMAX_FIRST["divs"]))]))])


def smoke_fields(pairs, campaign_dir, wait_minutes, art_rel):
    """Smoke-only top-level fields.  ``pairs``: [(id, arm_on_lane_A, arm_on_lane_B)]."""
    run_cov = "%s/scripts/sf43cov/run_cov.sh" % art_rel
    return OrderedDict([
        ("determinism_pairs", [[a, b] for _, a, b in pairs]),
        ("background", [OrderedDict([("name", "coverage"), ("nice", 19), ("argv", ["bash", run_cov, "{campaign_dir}/coverage", "--workers", "32"])])]),
        ("background_wait_minutes", wait_minutes),
    ])
