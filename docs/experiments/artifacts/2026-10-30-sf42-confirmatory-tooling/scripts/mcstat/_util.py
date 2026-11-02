"""Low-level helpers shared by the mcstat modules (rank quantiles, seed-bootstrap counts).

Everything is vectorized over leading (replicate) axes: statistics live on the LAST axis, any leading axes are
independent replicates (the coverage simulator passes thousands of them at once).

Rank rule (Davison and Hinkley, 1997, section 5.3.2).  For a sorted bootstrap sample of size ``B`` the upper
``tau``-quantile ("value exceeded with probability tau") is the order statistic of rank ``ceil((1 - tau)(B + 1))`` and the lower
``tau``-quantile the one of rank ``floor(tau (B + 1))``.  When that rank does not exist (``tau < 1 / (B + 1)``) the tail is not
resolvable by ``B`` resamples and the helpers return ``+inf`` (upper quantile) or ``-inf`` (lower quantile): every
confidence limit built on them is then the trivial one, never an optimistic one.
"""
from __future__ import annotations

import numpy as np
from scipy import stats as _st

_EPS = 1e-9  # guards floor/ceil against products such as 0.95 * 2000 = 1900.0000000000002


def as_tails(tails):
    """-> (1-D float array of tails in (0, 1), was_scalar)."""
    arr = np.asarray(tails, dtype=float)
    scalar = arr.ndim == 0
    arr = np.atleast_1d(arr)
    if arr.ndim != 1:
        raise ValueError("tails must be a scalar or a 1-D array")
    if np.any(~(arr > 0.0)) or np.any(arr >= 1.0):
        raise ValueError("tail levels must lie in (0, 1)")
    return arr, scalar


def as_levels(a):
    """Like ``as_tails`` but for a test level ``a`` of the robust limit, which may equal 1: -> (1-D array in (0, 1], was_scalar)."""
    arr = np.asarray(a, dtype=float)
    scalar = arr.ndim == 0
    arr = np.atleast_1d(arr)
    if arr.ndim != 1:
        raise ValueError("levels must be a scalar or a 1-D array")
    if np.any(~(arr > 0.0)) or np.any(arr > 1.0):
        raise ValueError("levels must lie in (0, 1]")
    return arr, scalar


def _squeeze(out, scalar):
    return out[..., 0] if scalar else out


def ucb_sorted(sorted_vals, tails):
    """Upper ``tau``-quantile of a sorted (ascending, last axis) bootstrap sample, per tail.

    Output shape: ``sorted_vals.shape[:-1] + (T,)`` (``T`` dropped when ``tails`` is a scalar); ``+inf`` where the rank does not exist.
    """
    tau, scalar = as_tails(tails)
    B = sorted_vals.shape[-1]
    k = np.ceil((1.0 - tau) * (B + 1) - _EPS)
    valid = k <= B
    k = np.clip(k, 1, B).astype(int)
    out = np.take(sorted_vals, k - 1, axis=-1)
    out = np.where(valid, out, np.inf)
    return _squeeze(out, scalar)


def lcb_sorted(sorted_vals, tails):
    """Lower ``tau``-quantile of a sorted bootstrap sample, per tail; ``-inf`` where the rank does not exist."""
    tau, scalar = as_tails(tails)
    B = sorted_vals.shape[-1]
    k = np.floor(tau * (B + 1) + _EPS)
    valid = k >= 1
    k = np.clip(k, 1, B).astype(int)
    out = np.take(sorted_vals, k - 1, axis=-1)
    out = np.where(valid, out, -np.inf)
    return _squeeze(out, scalar)


def boot_counts(n, B, rng):
    """Resampling-count matrix ``W`` (B, n): ``W[b, i]`` = times seed ``i`` is drawn in resample ``b`` (rows sum to ``n``).

    ``mean* = x @ W.T / n`` gives the ``B`` resample means of any array ``x`` (..., n) at once, and arms that share ``W`` share the
    resampled seed indices (paired arms).  ``rng`` is a ``numpy.random.Generator``.
    """
    if n < 2:
        raise ValueError("a bootstrap over seeds needs at least 2 seeds")
    idx = rng.integers(0, n, size=(B, n))
    flat = (np.arange(B)[:, None] * n + idx).ravel()
    return np.bincount(flat, minlength=B * n).reshape(B, n).astype(float)


def bootstrap_means(arrays, B, rng, share_draws=False, inner_ndim=0):
    """Means over the resampled seeds (LAST axis) of one or several arrays that share the SAME resample indices.

    ``arrays``: tuple of arrays of identical shape ``(*lead, *inner, n)``; ``lead`` are independent replicates, the
    ``inner_ndim`` axes before the seed axis (e.g. the levels of a ladder) belong to one replicate and share its draws.
    Returns a tuple of arrays ``(*lead, *inner, B)``.  ``share_draws=False`` (default) draws independent indices for every
    replicate (a Python loop over replicates: about 1 ms each for B = 2000, n = 40); ``share_draws=True`` uses one index matrix for
    the whole batch (one matrix product, several times faster): every replicate's limit is still valid, but replicates are dependent
    through the shared draws, which inflates the Monte Carlo variance of a miss-rate estimate over the batch (negligible with a few
    hundred replicates per call and many calls).
    """
    first = arrays[0]
    n = first.shape[-1]
    lead = first.shape[:first.ndim - 1 - inner_ndim]
    if share_draws or not lead:
        W = boot_counts(n, B, rng)
        return tuple(a @ W.T / n for a in arrays)
    nrep = int(np.prod(lead))
    flat = [a.reshape((nrep,) + a.shape[len(lead):]) for a in arrays]
    outs = [np.empty((nrep,) + f.shape[1:-1] + (B,)) for f in flat]
    for i in range(nrep):
        W = boot_counts(n, B, rng).T / n
        for o, f in zip(outs, flat):
            o[i] = f[i] @ W
    return tuple(o.reshape(lead + o.shape[1:]) for o in outs)


def get_rng(rng=None, seed=None):
    """``rng`` if given, else ``np.random.default_rng(seed)``.  One of the two is required: an implicit entropy-seeded stream
    would make a confirmatory computation irreproducible, and an implicit fixed seed would silently reuse one stream."""
    if rng is not None:
        return rng
    if seed is None:
        raise ValueError("pass rng= (a numpy Generator) or seed= (an integer): bootstrap work must be reproducible")
    return np.random.default_rng(seed)


def t_isf(tail, dof):
    """Student-t upper quantile ``q`` with ``P(T > q) = tail`` (``tail`` and ``dof`` broadcast), refined to double precision.

    ``scipy.stats.t.isf`` / ``ppf`` are accurate to about 1e-9 relative on some scipy versions (the host's scipy 1.11 returns a
    quantile whose survival function misses the level by up to 2e-10), while ``t.sf`` is accurate on all of them.  Two Newton steps
    on ``sf(q) - tail`` (derivative ``-pdf``) make the quantile, and every interval built on it, identical across scipy versions to
    about 1e-14 relative.  NaN / infinite degrees of freedom pass through.
    """
    tail = np.asarray(tail, dtype=float)
    dof = np.asarray(dof, dtype=float)
    q = np.asarray(_st.t.isf(tail, dof), dtype=float)
    with np.errstate(all="ignore"):
        for _ in range(2):
            step = (_st.t.sf(q, dof) - tail) / _st.t.pdf(q, dof)
            q = np.where(np.isfinite(step), q + step, q)
    return q if q.ndim else float(q)
