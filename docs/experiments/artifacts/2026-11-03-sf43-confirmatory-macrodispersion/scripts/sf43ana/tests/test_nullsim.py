"""Null simulation of the Gate 4 pipeline: family-wise rejection rate of the robust-sign hypotheses under a true alpha at the null.

Synthetic per-seed data with the budget structure of the design at the four amplitudes (E_h ladders with the production level in the
middle at 0.25 / 1 and the finest at 2.25 / 4 (with an extra coarse triple at 4), E_W widths (2x2 with the interaction at 2.25 / 4),
E_int passes), Student sampling and Student-studentized components, Holm over the fixed family of 8 (4 robust-sign + 4 Gate 5 with
p = 1), the decisions taken by evaluating RL at the Holm levels.  Fast settings (R replicates, coarse lattice); the pre-registration
runs the same code with its own sizes.  Scenarios:
  at_null     every arm's mean equals null_value(sigma^2) (no bias): the sharp null;
  at_zero     every arm's mean 0 (inside the null);
  biased      true alpha at the null, production biased UP by a discretization error of order 1.5, a width error and an integration
              error (about 7 standard errors in total): sampling alone would reject, the budget must hold the rate;
  biased_skew the same with skewed (centred gamma) per-seed noise;
  mixed       alpha = null + large at 2.25 and 4, null at 0.25 and 1 (biased): rejections at the large amplitudes are allowed, a
              rejection at a true-null amplitude is a false rejection (the power side is checked too).
"""
import copy
import unittest

import numpy as np
from scipy import stats

import sf43ana  # noqa: F401
from mcstat.budget import RobustLimit, StudentMean, default_lattice
from sf43ana import config
from sf43ana.gate4 import Design, RobustTest
from sf43ana.holm import holm

CFG = copy.deepcopy(config.DEFAULTS)
M = 8
LEVELS = [0.05 / (M - j) for j in range(M)]
SPEC = {
    "0.25": dict(sigma2=0.25, comps=["E_h", "E_W", "E_int"], res=[5, 10, 20], ip=1, wf=1, wc=None),
    "1": dict(sigma2=1.0, comps=["E_h", "E_W", "E_int"], res=[5, 10, 20], ip=1, wf=1, wc=None),
    "2.25": dict(sigma2=2.25, comps=["E_h", "E_W", "E_WxH", "E_int"], res=[4, 8, 16], ip=2, wf=1, wc=0),
    "4": dict(sigma2=4.0, comps=["E_h", "E_W", "E_WxH", "E_int"], res=[4, 8, 16, 32], ip=3, wf=2, wc=1),
}
S, N, NW, NH = 0.003, 60, 60, 30  # per-seed sd of the production arm; seeds per arm


def noise(rng, shape, dist):
    z = rng.standard_normal(shape)
    if dist == "skew":
        z = (rng.gamma(3.0, 1.0, shape) - 3.0) / np.sqrt(3.0)
    return z


def make_design(rng, R, spec, alpha_star, biases, dist):
    """Design with a batch axis of R replicates: levels (R, N, K) on common seeds, wide / h_max arms independent."""
    bh0, bw, bi = biases  # production discretization error, width error (production width vs converged), integration error
    res, ip = np.asarray(spec["res"], float), spec["ip"]
    bh = lambda r: bh0 * (res[ip] / r) ** 1.5  # noqa: E731  (error of order 1.5)
    common = 0.8 * S * noise(rng, (R, N, 1), dist)
    levels = np.stack([alpha_star + bw + bh(r) + common[..., 0] + 0.3 * S * noise(rng, (R, N), dist) for r in res], axis=-1)
    d = Design(levels[..., ip], spec["comps"], levels=levels, res=list(res), ip=ip)
    cw = 0.8 * S * noise(rng, (R, NW, 1), dist)
    wide = lambda k: alpha_star + bh(res[k]) + cw[..., 0] + 0.3 * S * noise(rng, (R, NW), dist)  # noqa: E731
    d.wide_fine = wide(spec["wf"])
    d.prod_fine, d.prod_fine_paired = levels[..., spec["wf"]], True
    if spec["wc"] is not None:
        d.wide_coarse, d.prod_coarse = wide(spec["wc"]), levels[..., spec["wc"]]
    ch = 0.8 * S * noise(rng, (R, NH, 1), dist)
    base = alpha_star + ch[..., 0]
    d.h8 = base + bi + 0.2 * S * noise(rng, (R, NH), dist)
    d.h16 = base + bi / 2.0 + 0.2 * S * noise(rng, (R, NH), dist)
    return d


def simulate(scenario, R=200, seed=0, true_up=(), sampling="student", B_sampling=None):
    """-> dict(rate, se, n_false, per-amplitude rejection rates, naive rate).  ``true_up``: amplitudes with a truly positive alpha."""
    rng = np.random.default_rng(seed)
    dist = "skew" if scenario == "biased_skew" else "normal"
    se = S / np.sqrt(N)
    biased = scenario in ("biased", "biased_skew", "mixed")
    lattice = default_lattice(per_decade=20)
    tabs, ps, nulls, naive = {}, {}, {}, {}
    for amp, spec in SPEC.items():
        null = 7.992e-5 * spec["sigma2"]
        nulls[amp] = null
        a_star = 0.0 if scenario == "at_zero" else null + (0.02 if amp in true_up else 0.0)
        biases = (3.0 * se, 3.0 * se, 1.0 * se) if biased else (0.0, 0.0, 0.0)
        for est in ("alpha_TP", "alpha_TAP"):
            d = make_design(rng, R, spec, a_star, biases, dist)
            t = RobustTest(d, null, CFG, "%s/%s" % (amp, est), lattice=lattice, sampling=sampling, B_sampling=B_sampling, B_budget=4000, rng=rng)
            tabs[(amp, est)] = np.asarray(t.rl.lower(np.array(LEVELS)))  # (R, M) exact RL at every possible Holm level
            ps[(amp, est)] = np.asarray(t.p_value()[0])
            naive[(amp, est)] = np.asarray(StudentMean(d.prod).lower(0.05)) > null
    false, rej = np.zeros(R, bool), {a: np.zeros(R, bool) for a in SPEC}
    for r in range(R):
        hyps = []
        for amp in SPEC:
            hyps.append({"id": "robust:" + amp, "testable": True, "p": max(ps[(amp, e)][r] for e in ("alpha_TP", "alpha_TAP")),
                         "exact": lambda lv, amp=amp, r=r: all(tabs[(amp, e)][r, int(np.argmin(np.abs(np.array(LEVELS) - lv)))] > nulls[amp]
                                                                for e in ("alpha_TP", "alpha_TAP"))})
        hyps += [{"id": "gate5:" + a, "testable": False, "p": 1.0} for a in SPEC]
        res, _ = holm(hyps, 0.05)
        for h in res[:4]:
            amp = h["id"].split(":")[1]
            rej[amp][r] = h["rejected"]
            if h["rejected"] and amp not in true_up:
                false[r] = True
    rate = float(false.mean())
    nv = np.zeros(R, bool)
    for amp in SPEC:
        if amp not in true_up:
            nv |= naive[(amp, "alpha_TP")] & naive[(amp, "alpha_TAP")]
    k = int(false.sum())
    return {"scenario": scenario, "R": R, "rate": rate, "mc_se": float(np.sqrt(rate * (1 - rate) / R)),
            "ucl95_clopper_pearson": float(stats.beta.ppf(0.95, k + 1, R - k)) if k < R else 1.0, "n_false": k, "rejection_by_amplitude": {a: float(v.mean()) for a, v in rej.items()},
            "naive_sampling_only_rate": float(nv.mean())}


class NullSimulation(unittest.TestCase):
    R = 200
    sampling, B_sampling = "student", None

    def check(self, scenario, **kw):
        out = simulate(scenario, self.R, sampling=self.sampling, B_sampling=self.B_sampling, **kw)
        print("\nnull simulation (%s sampling limit) %s: %s" % (self.sampling, scenario, out))
        self.assertLessEqual(out["rate"], 0.05, out)
        self.assertLessEqual(out["ucl95_clopper_pearson"], 0.05, out)  # R = 200 resolves 5 % when no false rejection occurs (rule of three: 1.5 %)
        return out

    def test_at_null(self):
        self.check("at_null", seed=1)

    def test_at_zero(self):
        self.check("at_zero", seed=2)

    def test_biased_production_is_held_by_the_budget(self):
        out = self.check("biased", seed=3)
        self.assertGreater(out["naive_sampling_only_rate"], 0.5)  # the test is not vacuous: sampling alone would reject

    def test_biased_skewed(self):
        self.check("biased_skew", seed=4)

    def test_mixed_power_and_no_false_rejection(self):
        out = self.check("mixed", seed=5, true_up=("2.25", "4"))
        self.assertGreater(out["rejection_by_amplitude"]["2.25"], 0.9)  # power where alpha is truly above the null
        self.assertGreater(out["rejection_by_amplitude"]["4"], 0.9)


class NullSimulationBootstrapT(NullSimulation):
    """The same scenarios with the pre-registered production sampling limit: the bootstrap-t lower limit (B + 1 = 3200 resolves the
    smallest Holm tail 0.05 / 16 with 10 ranks; the pre-registration uses B = 33599)."""
    sampling, B_sampling = "bootstrap-t", 3199


if __name__ == "__main__":
    unittest.main()
