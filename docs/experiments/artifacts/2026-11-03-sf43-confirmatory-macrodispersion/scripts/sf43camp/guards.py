"""Pure decision functions: budget, disk, projection, backoff, blinded re-sizing factor."""
import math


def budget_ok(used_h, in_flight_h, unit_h, budget_h):
    return used_h + in_flight_h + unit_h <= budget_h


def disk_ok(free_gb, min_free_gb, unit_raw_gb):
    return free_gb >= min_free_gb + unit_raw_gb


def oom_backoff_s(n):
    """10 min doubling to 60 min; n = consecutive OOMs of the arm including this one (n >= 1)."""
    return min(600.0 * 2 ** (n - 1), 3600.0)


def projection(st, now, used_h, lanes):
    """Remaining plan at the measured per-arm rates: lane-seconds, GPU-h (one GPU per lane-hour) and completion epoch."""
    rem = sum(a.rate() * len(a.pending()) for a in st.arms.values() if not a.halted)
    lanes = max(1, lanes)
    return {"remaining_lane_h": rem / 3600.0, "projected_gpu_h": used_h + rem / 3600.0, "completion": now + rem / lanes}


def sd_upper80(sd, n):
    """80 % upper confidence bound of a standard deviation from n values: sd sqrt((n-1)/chi2_{0.2, n-1}) (as mcstat.sizing.sd_upper)."""
    from scipy import stats
    return sd * math.sqrt((n - 1) / stats.chi2.ppf(0.2, n - 1))


def resize(sd_obs, n_obs, sd_plan_ub, n_conf, max_factor, cap):
    """Blinded re-sizing: factor = max over estimators of clip((sd_obs_ub / sd_plan_ub)^2, 1, max_factor); returns (factor, n_new)
    with n_new = min(ceil(factor n_conf), cap) (never below n_conf).  Takes standard deviations only."""
    f = 1.0
    for e, ub in sd_plan_ub.items():
        if e in sd_obs:
            f = max(f, min(max_factor, (sd_upper80(sd_obs[e], n_obs) / ub) ** 2))
    return f, max(n_conf, min(math.ceil(f * n_conf), cap))
