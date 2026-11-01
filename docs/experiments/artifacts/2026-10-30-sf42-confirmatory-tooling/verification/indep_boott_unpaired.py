# SF-42 R5b: independent re-implementation (imports nothing from mcstat; numpy + scipy only) of the unpaired-contrast miss rate of the
# bootstrap-t and Student upper bounds of |Delta| under the stress mixture (n 20, R 20000, B 2000, Delta = 5 s.e.), to check that the
# bootstrap-t over-miss found by the coverage simulation is not an implementation defect.  Run: python3 -I indep_boott_unpaired.py 7
# Independent check (no mcstat import): miss rate of the bootstrap-t and Student upper bounds of |Delta| for the
# unpaired contrast (E_W style: x = Delta + Z1, y = Z2, n each, Delta = 5 sqrt(2/n)) under the stress mixture.
import math, sys, numpy as np
from scipy import stats
p, v = 0.0646345578406386, 0.3314838129156588           # mixture weight / within variance (scenario definition)
q = 1 - p; x1 = math.sqrt((1 - v) * q / p); x2 = -math.sqrt((1 - v) * p / q)
def draw(rng, shape):
    return np.sqrt(v) * rng.standard_normal(shape) + np.where(rng.random(shape) < p, x1, x2)
rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 1)
z = draw(rng, 2_000_000); m = z.mean(); s = z.std()
print("mixture check: mean %.4f sd %.4f skew %.3f exkurt %.3f" % (m, s, stats.skew(z), stats.kurtosis(z)))
n, R, B = 20, 20000, 2000
tails = np.array([0.00833333, 0.00416667, 0.0025])
delta = 5 * math.sqrt(2 / n)
k = np.floor(tails * (B + 1)).astype(int)                 # rank of the lower t-quantile of T* among B sorted values (1-based)
miss_bt = np.zeros(len(tails), int); miss_st = np.zeros(len(tails), int)
tq = stats.t.ppf(1 - tails, 2 * n - 2)                    # Student (pooled df, equal n; Welch df differs slightly)
for b0 in range(0, R, 50):
    r = 50
    x = delta + draw(rng, (r, n)); y = draw(rng, (r, n))
    d = x.mean(1) - y.mean(1); se = np.sqrt(x.var(1, ddof=1) / n + y.var(1, ddof=1) / n)
    # Student UCB of |Delta| (miss when the upper bound is below Delta)
    ust = np.abs(d)[:, None] + tq[None, :] * se[:, None]
    miss_st += (ust < delta).sum(0)
    ix = rng.integers(0, n, (r, B, n)); iy = rng.integers(0, n, (r, B, n))
    xs = np.take_along_axis(x[:, None, :].repeat(B, 1), ix, 2); ys = np.take_along_axis(y[:, None, :].repeat(B, 1), iy, 2)
    ds = xs.mean(2) - ys.mean(2); ses = np.sqrt(xs.var(2, ddof=1) / n + ys.var(2, ddof=1) / n)
    T = np.sort((ds - d[:, None]) / ses, axis=1)
    lo = T[:, k - 1]                                       # lower t-quantile of T*
    hi = T[:, B - k]                                       # upper t-quantile of T*
    ucl = d[:, None] - lo * se[:, None]; lcl = d[:, None] - hi * se[:, None]
    U = np.maximum(ucl, -lcl)
    miss_bt += (U < delta).sum(0)
print("n", n, "R", R, "B", B)
for t, mb, ms in zip(tails, miss_bt, miss_st):
    print("tail %.5f  boott misses %d rate/tail %.2f   student misses %d rate/tail %.2f   (nominal %.0f)" % (t, mb, mb / R / t, ms, ms / R / t, R * t))
