"""Pure decision functions: budget, disk, projection, backoff, blinded re-sizing (H9)."""
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


def n_required(sd_obs, n_obs, hw_target):
    """H9 precision-target N at the 80 % upper bound of the SD: ceil((t_{0.95, n-1} sd_UB80 / hw_target)^2)."""
    from scipy import stats
    return int(math.ceil((stats.t.ppf(0.95, n_obs - 1) * sd_upper80(sd_obs, n_obs) / hw_target) ** 2))


def resize(sd_obs, n_obs, hw_target, n_conf, max_factor, cap):
    """Blinded re-sizing (H9): N_req = max over the arm's estimators of n_required; n_new = min(floor(max_factor n_conf), max(n_conf, N_req), cap)
    (upward only, at most max_factor, cap = the arm's n_max / seed range).  Takes standard deviations and counts only; returns (N_req, n_new)."""
    req = [n_required(sd_obs[e], n_obs, hw) for e, hw in hw_target.items() if e in sd_obs]
    n_req = max(req, default=n_conf)
    return n_req, max(n_conf, min(int(math.floor(max_factor * n_conf)), n_req, cap))
