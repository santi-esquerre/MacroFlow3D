"""Reference implementations for the tests (independent of mcstat's own code paths)."""
from scipy import optimize, stats


def t_ppf(p, nu):
    """Student-t quantile by root finding on the (accurate) cdf: identical on every scipy version, unlike ``stats.t.ppf``, which is
    off by up to 1.4e-9 relative at some degrees of freedom on scipy 1.11."""
    return optimize.brentq(lambda q: stats.t.cdf(q, nu) - p, 0.0 if p >= 0.5 else -80.0, 80.0 if p >= 0.5 else 0.0,
                           xtol=1e-15, rtol=4 * 2.220446049250313e-16, maxiter=500)
