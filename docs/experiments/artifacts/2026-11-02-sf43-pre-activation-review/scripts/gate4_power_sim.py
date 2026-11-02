#!/usr/bin/env python3
"""SF-43 pre-activation review: Gate 4 quantitative analysis (read-only; numpy, scipy and the SF-42 `mcstat` package).

Sub-commands (markdown tables on stdout; nothing is written; nothing needs the host or a GPU):
  components  section 1: SF-39 budget components of the SF-43 estimand (alpha_TP on planes d in [25, 94] lambda, position basis)
              from the SF-39 analysis/reduced/ tables: C1 estimand, C2 paired and narrow-ladder differences, C3 width contrasts,
              C4 h_max passes, C5 particle share of the paired differences; then a check of every constant the simulations use.
  pipeline    section 2: robust-sign pipeline of decision G5.4 run with mcstat's own classes (StudentMean, LadderBound,
              e_w / e_int / e_interaction with method='student', RobustLimit): P(RL > 0), E[RL], mean B_k, order-branch use;
              Holm (G5.6) over the three sign p-values with the four Gate-5 p-values fixed at 1 or 0.  About 3 min.
  mde         section 3: smallest true alpha_T (fraction of the planning mean mu) with P(RL > 0) >= 0.8.  At fixed biases RL moves
              1:1 with alpha, so MDE = mu - q_0.2(RL).  Includes the D-alloc designs found by `alloc`.
  forced-pL   section 4: E_h with production at the finest level, order branch forced at p_L in {0.5, 1, 1.5, 2} or the
              fallback ('fb'); resolved-bias cases at sigma^2 = 4 (true order 1 / 2) through LadderBound.
  numerical   section 7: reachability of the G5.7 "numerical" class for a pre-registered fraction f (normal approximation).
  alloc       section 2 D-alloc: displacement-basis grid search at G5.2 planning inputs; mcstat planning-model check (C10).  About 6 min.

Scenario model (inputs are SF-39 / SF-42 values rounded as in the report; `components` checks them against the data):
  production arm x = mu + e(h_p) + dW + SD_p Z (N seeds).  E_h ladder on the first n_l production seeds; S1 level errors of order
  2 anchored on the SF-39 production-minus-coarse-twin mean D_tw.  Per-seed ladder noise: sigma^2 0.25 / 1 per-seed C h^2 with the
  measured corr(D_c, D_f); sigma^2 2.25 / 4 'nf' (independent per-level noise: equal consecutive SDs, corr -0.5) or 'o2' (coarser
  pair x4, corr 0).  K = 3 (0.25, 1): width arm against the production arm.  K = 4 (2.25, 4): 2 x 2 width x lambda/h arms, E_W at
  the coarse resolution plus the interaction.  E_int from the SF-39 h_max passes.  Scenarios: S1; S1w5 / S1w10 production width
  deficit of 5 / 10 % of mu; S2 non-monotone level errors with range |adverse SF-39 90 % limit of D_tw| (fallback forced).
  Displacement-basis ladder and interaction SDs are taken equal to the position-basis ones (conservative).

Seeds:
  pipeline / mde at sigma^2 0.25 and 1: N + n_ladder + n_width + len(scenario), as in the report (identical at R 400, B 4000).
  pipeline / mde at sigma^2 2.25 and 4: crc32("design|amp|scenario|noise").  The report's section 2-3 values at 2.25 / 4 and its
  Holm table came from Python's salted hash() of the same labels, which cannot be replayed; this script reproduces them within
  Monte Carlo error (binomial SE about 0.024 at R = 400), not bit for bit.  Largest gap: sigma^2 4, D-spec, S1, o2 prints
  P(RL > 0 at 0.05/7) 0.610 and Holm 0.710 / 0.915 / 0.980 here, 0.688 and 0.777 / 0.925 / 0.985 in the report.
  forced-pL 5 and 7, numerical default_rng(3), alloc and the D-alloc rows of mde N*7 + n_ladder*3 + n_width (+1 for S1w10) at
  R 300, B 3000: as in the report.
"""
import argparse
import csv
import functools
import itertools
import math
import os
import sys
import zlib
from collections import OrderedDict

import numpy as np
from scipy import stats

A7, A3 = 0.05 / 7, 0.05 / 3  # Holm worst level (m = 7); level of the first sign test after four Gate-5 rejections
bd = sizing = holm = abs_upper_bound = welch_groups = None  # mcstat, bound by load_mcstat()


def load_mcstat(path):
    global bd, sizing, holm, abs_upper_bound, welch_groups
    if not path or not os.path.isdir(os.path.join(path, "mcstat")):
        sys.exit("need --sf42-scripts <dir that contains mcstat/> (or the SF42_SCRIPTS environment variable)")
    sys.path.insert(0, os.path.abspath(path))
    from mcstat import budget, intervals, multiplicity
    from mcstat import sizing as sizing_mod
    bd, sizing, holm = budget, sizing_mod, multiplicity.holm
    abs_upper_bound, welch_groups = intervals.abs_upper_bound, intervals.welch_groups


AMPS = ("s025", "s1", "s225", "s4")
SIG2 = {"s025": "0.25", "s1": "1", "s225": "2.25", "s4": "4"}
# planning mean / per-realization SD of alpha_TP on [25, 94] lambda: pos = C1 (at 0.25 the displacement mean is used, the position
# mean being unresolved), disp = SF-42 R3 verification/noise_summary.csv; sdi = SD of the h/16 - h/8 pass difference (C4)
AMP = {"s025": dict(K=3, mean={"pos": 5.093e-4, "disp": 5.093e-4}, sd={"pos": 4.476e-3, "disp": 1.206e-4}, sdi=1.441e-12),
       "s1": dict(K=3, mean={"pos": 6.237e-3, "disp": 7.217e-3}, sd={"pos": 1.103e-2, "disp": 1.121e-3}, sdi=1.505e-10),
       "s225": dict(K=4, mean={"pos": 2.833e-2, "disp": 2.946e-2}, sd={"pos": 1.514e-2, "disp": 4.073e-3}, sdi=1.280e-6),
       "s4": dict(K=4, mean={"pos": 6.648e-2, "disp": 6.538e-2}, sd={"pos": 1.918e-2, "disp": 1.688e-2}, sdi=9.749e-8)}
DISP_R3 = {"s025": (5.093330294e-4, 1.205564082e-4, 25), "s1": (7.216874402e-3, 1.120574325e-3, 50),
           "s225": (2.946473445e-2, 4.07294397e-3, 62), "s4": (6.538089597e-2, 1.687814483e-2, 71)}  # mean, SD, n (SF-42 R3)
DTW = {"s025": (1.208e-5, 3.270e-5, 2.472e-5), "s1": (4.015e-5, 3.550e-4, 1.774e-4),
       "s225": (-2.719e-4, 1.180e-3, -6.671e-4), "s4": (-3.357e-4, 2.198e-3, -1.186e-3)}  # D_tw mean, per-seed SD, adverse 90 % limit
NDTW = {"s025": 20, "s1": 20, "s225": 26, "s4": 20}
RHO_C = {"s025": 0.999846, "s1": 0.990019}  # per-seed corr(D_c, D_f) of the narrow ladders (C2)
SCEN = ("S1", "S1w5", "S1w10", "S2")
SCEN_DW = {"S1": 0.0, "S1w5": -0.05, "S1w10": -0.10, "S2": 0.0}
# GPU-h per 100 realizations: SF-42 note, J4 per-arm table (M_plan basis); 2.25 ladder A = 4/8/16 (416x96 from J1B), B = 8/16/20
COST = {"s025": dict(prod=0.479, lad=0.169 + 2.800, wide=1.451, hm=0.797),
        "s1": dict(prod=0.490, lad=0.171 + 2.888, wide=1.495, hm=0.808),
        "s225": dict(prod=1.703, ladA=0.143 + 0.360, ladB=0.360 + 3.121, wide=0.360 + 0.903 + 0.551 + 1.612, hm=2.620),
        "s4": dict(prod=3.401, lad=0.199 + 0.586, wide=0.199 + 0.344 + 0.586 + 1.751, hm=4.548)}
# design -> (basis, {amp: (N production, n ladder, n per width arm, n h_max, ladder at 2.25)})
DESIGNS = OrderedDict([
    ("D-spec", ("pos", {"s025": (1100, 300, 1100, 50, None), "s1": (1650, 300, 1650, 50, None),
                        "s225": (110, 110, 110, 50, "B"), "s4": (30, 30, 30, 30, None)})),
    ("D-spec 4/8/16", ("pos", {"s225": (110, 110, 110, 50, "A")})),
    ("D-disp", ("disp", {"s025": (100, 100, 100, 30, None), "s1": (100, 100, 100, 30, None),
                         "s225": (100, 100, 100, 30, "A"), "s4": (100, 100, 100, 30, None)})),
    ("D-lean", ("disp", {"s025": (100, 50, 50, 30, None), "s1": (100, 50, 50, 30, None),
                         "s225": (100, 50, 50, 30, "A"), "s4": (50, 50, 30, 30, None)})),
    ("D-lean(pos)", ("pos", {"s025": (1100, 50, 50, 30, None), "s1": (1650, 50, 50, 30, None),
                             "s225": (110, 50, 50, 30, "B")})),
])
ALLOC = {"s025": (50, 30, 30), "s1": (30, 30, 30), "s225": (75, 50, 30), "s4": (100, 100, 50)}  # alloc result, target 0.95
ALLOC_GRID = {"s025": ([30, 50, 75, 100], [30, 50, 100], [30, 50, 100]), "s1": ([30, 50, 75, 100], [30, 50, 100], [30, 50, 100]),
              "s225": ([30, 50, 75, 100], [30, 50, 100], [30, 50, 100]),
              "s4": ([30, 50, 75, 100, 150], [30, 50, 100, 150], [30, 50, 100, 150])}


def ladder_name(a, lad):
    return {"s025": "5/10/20", "s1": "5/10/20", "s4": "8/16/32"}.get(a) or {"A": "4/8/16", "B": "8/16/20"}[lad]


def cost(a, N, nl, nw, ni, lad=None):
    c = COST[a]
    return (N * c["prod"] + nl * (c["lad" + lad] if a == "s225" else c["lad"]) + nw * c["wide"] + ni * c["hm"]) / 100.0


def noises(a):
    return ("-",) if AMP[a]["K"] == 3 else ("nf", "o2")


# ------------------------------------------------------------------------------------------------------------------------------
# Pipeline cells
# ------------------------------------------------------------------------------------------------------------------------------
class ForcedLadder:
    """E_h with the order branch forced at p_L (inputs at t/2, G5.4) or the fallback ('fb'), Student bounds as
    mcstat.budget.LadderBound(method='student'); check_forced_ladder() asserts the identity."""
    name = "E_h"

    def __init__(self, lev, res, ip, pL):
        self.res = np.asarray(res, float)
        self.ip, self.pL = ip, pL
        K = lev.shape[-1]
        self.r = self.res[1] / self.res[0]
        self.rmin = float((self.res[1:] / self.res[:-1]).min())
        j = ip + 1 if ip < K - 1 else ip - 1
        self.finer = ip < K - 1
        self.adj = welch_groups([lev[..., max(ip, j)] - lev[..., min(ip, j)]])
        self.pairs = [welch_groups([lev[..., b] - lev[..., a]]) for a, b in ((0, 1), (0, 2), (1, 2))]

    def bound(self, t):
        t = np.asarray(t, float)
        if self.pL == "fb":
            return 3.0 * np.max(np.stack([abs_upper_bound(d, s, t / 3, nu) for d, s, nu in self.pairs], -2), axis=-2) / (self.rmin - 1)
        dU = abs_upper_bound(self.adj[0], self.adj[1], t / 2, self.adj[2])
        rp = self.r ** self.pL
        return 1.25 * dU * (rp / (rp - 1) if self.finer else 1 / (rp - 1))


def check_forced_ladder():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((50, 30, 3)) * 1e-3 + np.array([3e-3, 1e-3, 0.0])
    t = np.array([1e-3, 1e-2])
    det = bd.LadderBound(x, (8., 16., 32.), 2, method="student", B=4000, rng=rng, share_draws=True).details(t)
    out = ~det["order_branch_used"]
    d1 = np.abs(det["order_branch_value"] - ForcedLadder(x, (8., 16., 32.), 2, 0.5).bound(t))[out]
    d2 = np.abs(det["fallback_value"] - ForcedLadder(x, (8., 16., 32.), 2, "fb").bound(t))
    assert d1.size > 0 and d1.max() <= 1e-15 and d2.max() <= 1e-15, "ForcedLadder differs from LadderBound"


def _finish(mu, x, comps, ladder, K):
    rl = bd.RobustLimit(bd.StudentMean(x), comps)
    RL = rl.lower(np.array([A7, A3]))
    out = dict(mu=mu, RL7=RL[:, 0], RL3=RL[:, 1], cb=rl.component_bounds(A7), samp=mu - bd.StudentMean(x).lower(A7 / 2),
               pv=rl.p_value().p, names=[getattr(c, "name", "?") for c in comps])
    if hasattr(ladder, "details"):
        det = ladder.details(A7 / (2 * K))
        out["branch"], out["pL"] = det["order_branch_used"], det["p_L"]
    return out


def cell_k3(a, basis, D, sn, R, B):
    """sigma^2 0.25 / 1 (K = 3): ladder 5/10/20 (production in the middle), width arm against production, E_int."""
    N, nl, nw, ni, _ = D
    P = AMP[a]
    mu, sdp = P["mean"][basis], P["sd"][basis]
    m, s, lim = DTW[a]
    rng = np.random.default_rng(N + nl + nw + len(sn))
    Ec = -m / 3
    e = np.array([4 * Ec, Ec, Ec / 4]) if sn != "S2" else np.array([abs(lim) / 2, -abs(lim) / 2, abs(lim) / 2])
    x = mu + e[1] + SCEN_DW[sn] * mu + sdp * rng.standard_normal((R, N))
    xl = x[:, :nl]
    z1 = rng.standard_normal((R, nl))
    z2 = rng.standard_normal((R, nl))
    rho = RHO_C[a]
    d01 = e[1] - e[0] + s * z1
    d12 = e[2] - e[1] + (s / 4) * (rho * z1 + math.sqrt(1 - rho ** 2) * z2)
    lad = bd.LadderBound(np.stack([xl - d01, xl, xl + d12], -1), (5., 10., 20.), 1, method="student", B=B, rng=rng,
                         share_draws=True)
    di = P["sdi"] * rng.standard_normal((R, ni))
    comps = [lad, bd.e_w(mu + e[1] + sdp * rng.standard_normal((R, nw)), x, method="student"),
             bd.e_int(di, np.zeros_like(di), method="student")]
    return _finish(mu, x, comps, lad, 3)


def ladder_spec(a, noise, lad, sn):
    m, s, lim = DTW[a]
    if a == "s225" and lad == "A":  # 4/8/16, production finest
        e1 = -m / 3
        e, res, ip, s12 = [16 * e1, 4 * e1, e1], (4., 8., 16.), 2, s
        s01, rho = (s, -0.5) if noise == "nf" else (4 * s, 0.0)
    elif a == "s225":  # 8/16/20, production in the middle; no constant ratio
        e1 = -m / 3
        e, res, ip, s01 = [4 * e1, e1, e1 * (16 / 20.) ** 2], (8., 16., 20.), 1, s
        s12, rho = (s, -0.5) if noise == "nf" else (0.48 * s / 3.79 * 1.66, 0.0)
    else:  # sigma^2 = 4: 8/16/32, production finest
        e1 = -m / 3
        e, res, ip, s12 = [16 * e1, 4 * e1, e1], (8., 16., 32.), 2, s
        s01, rho = (s, -0.5) if noise == "nf" else (4 * s, 0.0)
    if sn == "S2":
        X = abs(lim)
        e = [X / 2, -X / 2, X / 2] if ip == 1 else [-X / 2, X / 2, -X / 2]
    return np.array(e, float), res, ip, s01, s12, rho


def cell_k4(a, basis, D, sn, noise, R, B, seed, forced=None, e_over=None):
    """sigma^2 2.25 / 4 (K = 4): ladder, 2 x 2 width x lambda/h arms (E_W at the coarse resolution + interaction), E_int."""
    N, nl, nw, ni, lad = D
    P = AMP[a]
    mu, sdp = P["mean"][basis], P["sd"][basis]
    rng = np.random.default_rng(seed)
    e, res, ip, s01, s12, rho = ladder_spec(a, noise, lad, sn)
    if e_over is not None:
        e = np.array(e_over, float)
    dW = SCEN_DW[sn] * mu
    x = mu + e[ip] + dW + sdp * rng.standard_normal((R, N))
    xl = x[:, :nl]
    z1 = rng.standard_normal((R, nl))
    z2 = rng.standard_normal((R, nl))
    d01 = e[1] - e[0] + s01 * z1
    d12 = e[2] - e[1] + s12 * (rho * z1 + math.sqrt(max(0.0, 1 - rho ** 2)) * z2)
    if ip == 2:
        m2 = xl
        m1 = m2 - d12
        m0 = m1 - d01
    else:
        m1 = xl
        m0 = m1 - d01
        m2 = m1 + d12
    lev = np.stack([m0, m1, m2], -1)
    if forced is None:
        ladder = bd.LadderBound(lev, res, ip, method="student", B=B, rng=rng, share_draws=True)
    else:
        ladder = ForcedLadder(lev, res, ip, forced)
    s = DTW[a][1]
    q = s if noise == "nf" else (0.48 * s if a == "s225" else 4 * s)  # per-seed SD of the 2 x 2 resolution step
    pc = dW + sdp * rng.standard_normal((R, nw))
    wc = sdp * rng.standard_normal((R, nw))
    pf = pc + q * rng.standard_normal((R, nw))
    wf = wc + q * rng.standard_normal((R, nw))
    di = P["sdi"] * rng.standard_normal((R, ni))
    comps = [ladder, bd.e_w(wc, pc, method="student"), bd.e_interaction(wf, wc, pf, pc, method="student"),
             bd.e_int(di, np.zeros_like(di), method="student")]
    return _finish(mu, x, comps, ladder, 4)


def cell_alloc(a, N, nl, nw, ni, noise, dW, R, B, seed):
    """C8: displacement basis at G5.2 planning inputs (mean at its 80 % lower bound, SDs at their 80 % upper bound)."""
    mean, sd, n = DISP_R3[a]
    mu, sdp = float(sizing.mean_lower(mean, sd, n)), float(sizing.sd_upper(sd, n))
    m, s, _ = DTW[a]
    s = s * float(sizing.sd_upper(1, NDTW[a]))
    rng = np.random.default_rng(seed)
    K = AMP[a]["K"]
    if K == 3:
        Ec = -m / 3
        e, res, ip, s01, s12, rho = np.array([4 * Ec, Ec, Ec / 4]), (5., 10., 20.), 1, s, s / 4, RHO_C[a]
    else:
        e1 = -m / 3
        e, res, ip, s12 = np.array([16 * e1, 4 * e1, e1]), ((4., 8., 16.) if a == "s225" else (8., 16., 32.)), 2, s
        s01, rho = (s, -0.5) if noise == "nf" else (4 * s, 0.0)
    x = mu + e[ip] + dW * mu + sdp * rng.standard_normal((R, N))
    xl = x[:, :nl]
    z1 = rng.standard_normal((R, nl))
    z2 = rng.standard_normal((R, nl))
    d01 = e[1] - e[0] + s01 * z1
    d12 = e[2] - e[1] + s12 * (rho * z1 + math.sqrt(max(0, 1 - rho ** 2)) * z2)
    if ip == 2:
        m2 = xl
        m1 = m2 - d12
        m0 = m1 - d01
    else:
        m1 = xl
        m0 = m1 - d01
        m2 = m1 + d12
    comps = [bd.LadderBound(np.stack([m0, m1, m2], -1), res, ip, method="student", B=B, rng=rng, share_draws=True)]
    if K == 3:
        comps.append(bd.e_w(mu + e[ip] + sdp * rng.standard_normal((R, nw)), x, method="student"))
    else:
        q = s if noise == "nf" else (0.48 * s if a == "s225" else 4 * s)
        pc = dW * mu + sdp * rng.standard_normal((R, nw))
        wc = sdp * rng.standard_normal((R, nw))
        pf = pc + q * rng.standard_normal((R, nw))
        wf = wc + q * rng.standard_normal((R, nw))
        comps += [bd.e_w(wc, pc, method="student"), bd.e_interaction(wf, wc, pf, pc, method="student")]
    di = AMP[a]["sdi"] * rng.standard_normal((R, ni))
    comps.append(bd.e_int(di, np.zeros_like(di), method="student"))
    RL = bd.RobustLimit(bd.StudentMean(x), comps).lower(np.array([A7, A3]))
    return dict(mu=mu, P7=(RL[:, 0] > 0).mean(), P3=(RL[:, 1] > 0).mean(), RL7=RL[:, 0], RL3=RL[:, 1])


def run_cells(designs, scens, R, B):
    cells = {}
    for dn, (basis, dd) in designs.items():
        for a, D in dd.items():
            for sn in scens:
                if AMP[a]["K"] == 3:
                    cells[(dn, a, sn, "-")] = cell_k3(a, basis, D, sn, R, B)
                    continue
                for nz in ("nf", "o2"):
                    seed = zlib.crc32(("%s|%s|%s|%s" % (dn, a, sn, nz)).encode())
                    cells[(dn, a, sn, nz)] = cell_k4(a, basis, D, sn, nz, R, B, seed)
    return cells


def mde(rl, mu):
    return 1.0 - np.quantile(rl, 0.2) / mu


# ------------------------------------------------------------------------------------------------------------------------------
# Sub-commands
# ------------------------------------------------------------------------------------------------------------------------------
def cmd_pipeline(R, B):
    cells = run_cells(DESIGNS, SCEN, R, B)
    print("## Section 2: P(RL > 0) at a = 0.05/7 (Holm worst level); R = %d, B = %d\n" % (R, B))
    print("| sigma^2 | design | basis | ladder | noise | GPU-h | S1 | S1w5 | S1w10 | S2 | P at 0.05/3 (S1) | E[RL]/mu (S1) "
          "| mean B_k/mu at 0.05/7 (S1) | order branch (S1) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for a in AMPS:
        for dn, (basis, dd) in DESIGNS.items():
            if a not in dd:
                continue
            D = dd[a]
            for nz in noises(a):
                c = {sn: cells[(dn, a, sn, nz)] for sn in SCEN}
                s1 = c["S1"]
                mu = s1["mu"]
                bk = ", ".join(["samp %.3f" % (s1["samp"].mean() / mu)] +
                               ["%s %.3f" % (nm, v) for nm, v in zip(s1["names"], s1["cb"].mean(0) / mu)])
                print("| %s | %s | %s | %s | %s | %.1f | %s | %.3f | %.2f | %s | %.2f |" % (
                    SIG2[a], dn, basis, ladder_name(a, D[4]), nz, cost(a, *D),
                    " | ".join("%.3f" % (c[sn]["RL7"] > 0).mean() for sn in SCEN),
                    (s1["RL3"] > 0).mean(), s1["RL7"].mean() / mu, bk, s1["branch"].mean()))
    print("\n## Holm (G5.6): P(reject) of the sign tests at sigma^2 1 / 2.25 / 4; Gate-5 p-values fixed at 1 or 0\n")
    print("| design | scenario | noise (2.25, 4) | Gate 5 none | J3B-like (0.25, 1, 2.25 pass; 4 fails) | Gate 5 all |")
    print("|---|---|---|---|---|---|")
    for dn in ("D-spec", "D-disp", "D-lean"):
        for sn in SCEN:
            for nz in ("nf", "o2"):
                pv = np.stack([cells[(dn, "s1", sn, "-")]["pv"], cells[(dn, "s225", sn, nz)]["pv"],
                               cells[(dn, "s4", sn, nz)]["pv"]], -1)
                res = []
                for g5 in ((1., 1., 1., 1.), (0., 0., 0., 1.), (0., 0., 0., 0.)):
                    rej = np.array([holm(np.concatenate([pv[i], g5]), 0.05)[1][:3] for i in range(len(pv))])
                    res.append("%.3f / %.3f / %.3f" % tuple(rej.mean(0)))
                print("| %s | %s | %s | %s |" % (dn, sn, nz, " | ".join(res)))


def cmd_mde(R, B, Ra, Ba):
    designs = OrderedDict((k, v) for k, v in DESIGNS.items() if k != "D-lean(pos)")
    cells = run_cells(designs, ("S1",), R, B)
    print("## Section 3: MDE, S1 (fraction of the planning mean mu, and in lambda)\n")
    print("| sigma^2 | design | ladder | noise | MDE/mu at 0.05/7 | at 0.05/3 | MDE [lambda] at 0.05/7 | at 0.05/3 |")
    print("|---|---|---|---|---|---|---|---|")
    for a in AMPS:
        for dn, (basis, dd) in designs.items():
            if a not in dd:
                continue
            for nz in noises(a):
                c = cells[(dn, a, "S1", nz)]
                m7, m3 = mde(c["RL7"], c["mu"]), mde(c["RL3"], c["mu"])
                print("| %s | %s | %s | %s | %.2f | %.2f | %.2e | %.2e |" % (SIG2[a], dn, ladder_name(a, dd[a][4]), nz, m7, m3,
                                                                           m7 * c["mu"], m3 * c["mu"]))
        N, nl, nw = ALLOC[a]
        nz = "o2" if AMP[a]["K"] == 4 else "nf"
        c = cell_alloc(a, N, nl, nw, 30, nz, 0.0, Ra, Ba, N * 7 + nl * 3 + nw)
        m7, m3 = mde(c["RL7"], c["mu"]), mde(c["RL3"], c["mu"])
        print("| %s | D-alloc N%d / l%d / w%d (planning inputs, R %d) | %s | %s | %.2f | %.2f | %.2e | %.2e |" % (
            SIG2[a], N, nl, nw, Ra, ladder_name(a, "A"), nz, m7, m3, m7 * c["mu"], m3 * c["mu"]))


def cmd_forced(R, B):
    check_forced_ladder()
    print("## Section 4: E_h/mu at a = 0.05/7, mean [10 %%, 90 %%]; P(RL > 0) at 0.05/7 / 0.05/3; E[RL]/mu (S1, seed 5, R = %d)\n" % R)
    print("| sigma^2 | design | noise | p_L 0.5 | p_L 1 | p_L 1.5 | p_L 2 | fallback |")
    print("|---|---|---|---|---|---|---|---|")
    designs = (("s4", "D-spec", "pos", (30, 30, 30, 30, None)), ("s4", "D-disp", "disp", (100, 100, 100, 30, None)),
               ("s4", "D-lean", "disp", (50, 50, 30, 30, None)), ("s225", "D-spec 4/8/16", "pos", (110, 110, 110, 50, "A")),
               ("s225", "D-disp 4/8/16", "disp", (100, 100, 100, 30, "A")), ("s225", "D-lean 4/8/16", "disp", (100, 50, 50, 30, "A")))
    for a, dn, basis, D in designs:
        for nz in ("nf", "o2"):
            cols = []
            for pL in (0.5, 1.0, 1.5, 2.0, "fb"):
                c = cell_k4(a, basis, D, "S1", nz, R, B, 5, forced=pL)
                eh = c["cb"][:, 0] / c["mu"]
                cols.append("%.3f [%.3f, %.3f]; P %.3f / %.3f; RL %.2f" % (eh.mean(), *np.quantile(eh, [0.1, 0.9]),
                            (c["RL7"] > 0).mean(), (c["RL3"] > 0).mean(), c["RL7"].mean() / c["mu"]))
            print("| %s | %s | %s | %s |" % (SIG2[a], dn, nz, " | ".join(cols)))
    print("\n## Section 4 (S3): sigma^2 = 4, true order 1 / 2, production biased low by 2 / 5 / 10 %% of mu (LadderBound, seed 7, "
          "R = %d)\n" % R)
    print("| design | order | e(h_p)/mu | noise | order branch | p_L median (finite) | P(p_L < 1 given branch) | E_h/mu "
          "| P(RL > 0) at 0.05/7 / 0.05/3 | E[RL]/mu |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for dn, basis, D in (("D-spec", "pos", (30, 30, 30, 30, None)), ("D-disp", "disp", (100, 100, 100, 30, None))):
        mu0 = AMP["s4"]["mean"][basis]
        for order in (1.0, 2.0):
            for frac in (0.02, 0.05, 0.10):
                e32 = -frac * mu0
                for nz in ("nf", "o2"):
                    c = cell_k4("s4", basis, D, "S1", nz, R, B, 7, e_over=[e32 * (4 ** order), e32 * (2 ** order), e32])
                    br, pl = c["branch"], c["pL"]
                    fin = np.isfinite(pl)
                    print("| %s | %.0f | -%.2f | %s | %.2f | %s | %s | %.3f | %.3f / %.3f | %.2f |" % (
                        dn, order, frac, nz, br.mean(), ("%.2f" % np.median(pl[fin])) if fin.any() else "none",
                        ("%.2f" % np.mean(pl[br] < 1)) if br.any() else "-", c["cb"][:, 0].mean() / c["mu"],
                        (c["RL7"] > 0).mean(), (c["RL3"] > 0).mean(), c["RL7"].mean() / c["mu"]))


NUM_DESIGNS = (  # label, sigma^2, mu, SD_p, N, K, SD_w, n_w, SD of the ladder step used by Richardson, Richardson factor, n_l
    ("D-spec pos", "0.25", 5.093e-4, 4.476e-3, 1100, 3, 4.476e-3, 1100, 8.2e-6, 4 / 3, 300),
    ("D-spec pos", "1", 6.237e-3, 1.103e-2, 1650, 3, 1.103e-2, 1650, 8.9e-5, 4 / 3, 300),
    ("D-spec pos", "2.25", 2.833e-2, 1.514e-2, 110, 4, 1.514e-2, 110, 1.180e-3, 1 / 3, 110),
    ("D-spec pos", "4", 6.648e-2, 1.918e-2, 30, 4, 1.918e-2, 30, 2.198e-3, 1 / 3, 30),
    ("D-disp N100", "0.25", 5.093e-4, 1.206e-4, 100, 3, 1.206e-4, 100, 8.2e-6, 4 / 3, 100),
    ("D-disp N100", "1", 7.217e-3, 1.121e-3, 100, 3, 1.121e-3, 100, 8.9e-5, 4 / 3, 100),
    ("D-disp N100", "2.25", 2.946e-2, 4.073e-3, 100, 4, 4.073e-3, 100, 1.180e-3, 1 / 3, 100),
    ("D-disp N100", "4", 6.538e-2, 1.688e-2, 100, 4, 1.688e-2, 100, 2.198e-3, 1 / 3, 100),
    ("D-alloc disp", "4", 6.538e-2, 1.688e-2, 50, 4, 1.688e-2, 100, 2.198e-3, 1 / 3, 50))


def cmd_numerical():
    """P(production 90 % CI excludes 0 and UCL_95(bias-corrected) < f * production mean); K = 3: corrected = wide arm - e_h
    (independent of production); K = 4: corrected = production - e_h - (prod_f - wide_f) (shares the production noise)."""
    rng = np.random.default_rng(3)
    R = 200000
    z = stats.norm.isf(0.05)
    print("## Section 7: P(numerical) for g = true corrected/production ratio 0 / 0.25 / 1 (g = 1: physical; g = 0: fully numerical)\n")
    print("| design | sigma^2 | SE_corr/mu | f_min (P >= 0.8 at g = 0) | f = 0.25 | f = 0.5 | f = 0.75 |")
    print("|---|---|---|---|---|---|---|")
    for lab, amp, mu, sdp, N, K, sdw, nw, sdl, rf, nl in NUM_DESIGNS:
        sep = sdp / math.sqrt(N)
        seh = sdl * rf / math.sqrt(nl)
        if K == 3:
            se_c, cov = math.sqrt(sdw ** 2 / nw + seh ** 2), 0.0
        else:
            se_c, cov = math.sqrt(sep ** 2 + 2 * sdw ** 2 / nw + seh ** 2), sep ** 2

        def pnum(f, g):
            zp = rng.standard_normal(R)
            zo = rng.standard_normal(R)
            ap = mu + sep * zp
            ac = g * mu + (cov / sep) * zp + math.sqrt(max(se_c ** 2 - cov ** 2 / sep ** 2, 0)) * zo
            return np.mean((ap - z * sep > 0) & (ac + z * se_c < f * ap))

        fmin = next((f for f in np.arange(0.02, 1.5, 0.01) if pnum(f, 0.0) >= 0.8), float("nan"))
        vals = ["%.2f / %.2f / %.2f" % tuple(pnum(f, g) for g in (0.0, 0.25, 1.0)) for f in (0.25, 0.5, 0.75)]
        print("| %s | %s | %.3f | %.2f | %s |" % (lab, amp, se_c / mu, fmin, " | ".join(vals)))


def cmd_alloc(Ra, Ba, R, B):
    print("## D-alloc: cheapest displacement-basis designs with P(RL > 0 at 0.05/7) >= target under S1 and S1w10 (planning inputs; "
          "noise o2 at 2.25 / 4; ladder 4/8/16 at 2.25; n_hmax 30; R = %d, B = %d)\n" % (Ra, Ba))
    print("| sigma^2 | target | N | n_ladder | n_width | GPU-h | S1: P at 0.05/7 / 0.05/3, MDE/mu | S1w10: P, MDE/mu |")
    print("|---|---|---|---|---|---|---|---|")
    for a in AMPS:
        rows = []
        nz = "o2" if AMP[a]["K"] == 4 else "nf"
        for N, nl, nw in itertools.product(*ALLOC_GRID[a]):
            if nl > N:
                continue
            s1 = cell_alloc(a, N, nl, nw, 30, nz, 0.0, Ra, Ba, N * 7 + nl * 3 + nw)
            w10 = cell_alloc(a, N, nl, nw, 30, nz, -0.10, Ra, Ba, N * 7 + nl * 3 + nw + 1)
            rows.append((cost(a, N, nl, nw, 30, "A"), N, nl, nw, s1, w10))
        rows.sort(key=lambda r: r[:4])
        for tgt in (0.8, 0.95):
            ok = [r for r in rows if min(r[4]["P7"], r[5]["P7"]) >= tgt]
            if not ok:
                print("| %s | %.2f | none in the grid | | | | | |" % (SIG2[a], tgt))
                continue
            c, N, nl, nw, s1, w10 = ok[0]
            print("| %s | %.2f | %d | %d | %d | %.2f | %.3f / %.3f, %.2f | %.3f / %.3f, %.2f |" % (
                SIG2[a], tgt, N, nl, nw, c, s1["P7"], s1["P3"], mde(s1["RL7"], s1["mu"]), w10["P7"], w10["P3"],
                mde(w10["RL7"], w10["mu"])))
    print("\n## C10: mcstat planning model (sizing.gate4_power, a = 0.05/7) against the simulated pipeline; sigma^2 = 4, D-spec, S1\n")
    print("| noise | planning power, E_h as order branch p = 2 | planning power, E_h as fallback | simulated P(RL > 0) |")
    print("|---|---|---|---|")
    mu, sdp, n, sdD = AMP["s4"]["mean"]["pos"], AMP["s4"]["sd"]["pos"], 30, DTW["s4"][1]
    for nz, sd_range in (("nf", sdD), ("o2", math.sqrt(17.0) * sdD)):  # per-seed SD of m32 - m8 under each noise model
        ew = dict(n=n, sd=math.sqrt(2) * sdp, slack=0.0)
        ewh = dict(n=n, sd=math.sqrt(2) * (sdD if nz == "nf" else 4 * sdD), slack=0.0)
        ei = dict(n=n, sd=2 * 9.7e-8, slack=0.0)
        order2 = dict(n=n, sd=1.25 * sdD / 3, slack=1.25 * 3.36e-4 / 3)
        fallb = dict(n=n, sd=3 * sd_range, slack=3 * 1.68e-3)
        sim = cell_k4("s4", "pos", DESIGNS["D-spec"][1]["s4"], "S1", nz, R, B, zlib.crc32(("D-spec|s4|S1|%s" % nz).encode()))
        print("| %s | %.3f | %.3f | %.3f |" % (nz, sizing.gate4_power(mu, A7, n, sdp, [order2, ew, ewh, ei]),
                                              sizing.gate4_power(mu, A7, n, sdp, [fallb, ew, ewh, ei]), (sim["RL7"] > 0).mean()))


# ------------------------------------------------------------------------------------------------------------------------------
# components (C1-C5)
# ------------------------------------------------------------------------------------------------------------------------------
NARROW = {"s025": ("n05", "n10", "n20"), "s1": ("n05", "n10", "n20"), "s225": ("n08", "n16", "n32"), "s4": ("n16", "n32", "n64")}
TWIN_PAIR = {"s025": "5->10", "s1": "5->10", "s225": "8->16", "s4": "16->32"}
WIDTH_CASES = (("s025", "s025_prod", "s025_n10", "25.6 vs 12.8 lambda, lambda/h 10"),
               ("s1", "s1_prod", "s1_n10", "25.6 vs 12.8 lambda, lambda/h 10"),
               ("s225", "s225_prod", "s225_n16", "24 vs 8 lambda, lambda/h 16"),
               ("s4", "s4_prod", "s4_n32", "12 vs 4 lambda, lambda/h 32"),
               ("s4", "s4_w24", "s4_tw_first30", "24 vs 12 lambda, lambda/h 16 (F10)"),
               ("s4", "s4_w12", "s4_n16_first30", "12 vs 4 lambda, lambda/h 16 (F10)"),
               ("s4", "s4_w12", "s4_tw_first30", "12 vs 12 lambda replication (seeds 7201+ vs 5001+)"),
               ("s4", "s4_w24", "s4_w12", "24 (7101+) vs 12 (7201+) lambda, lambda/h 16"))


@functools.lru_cache(maxsize=None)
def read_rows(red, name):
    with open(os.path.join(red, name + ".csv"), newline="") as f:
        return tuple(csv.DictReader(f))


@functools.lru_cache(maxsize=None)
def estimand(red, setname):
    """C1: per realization, half the OLS slope of sigma_T^2 / ell^2 against d = (x1 - x0) / ell over d in [25, 94]."""
    meta = read_rows(red, setname)[0]
    ell, x0 = float(meta["ell"]), float(meta["x0"])
    series = OrderedDict()
    for r in read_rows(red, "sigma_series_" + setname):
        series.setdefault(int(r["seed"]), []).append((float(r["x1"]), float(r["sigma_T2"])))
    out = OrderedDict()
    for seed, v in series.items():
        v = np.array(v)
        d = (v[:, 0] - x0) / ell
        m = (d >= 25 * (1 - 1e-9)) & (d <= 94 * (1 + 1e-9))
        dd, y = d[m], v[m, 1] / ell ** 2
        dc = dd - dd.mean()
        out[seed] = 0.5 * float(((y - y.mean()) @ dc) / (dc @ dc))
    return out


def ci90(x):
    n = len(x)
    m, s = float(np.mean(x)), float(np.std(x, ddof=1))
    h = stats.t.isf(0.05, n - 1) * s / math.sqrt(n)
    return m, s, n, m - h, m + h


def flux_rows(red, name):
    return {int(r["seed"]): r for r in read_rows(red, name) if r["method"] == "flux"}


def cmd_components(red, sf42):
    if not red or not os.path.isdir(red):
        sys.exit("components needs --sf39-reduced <SF-39 analysis/reduced dir>")
    print("## Section 1: budget components of the SF-43 estimand (position basis; C1-C5); % mu = percent of the planning mean\n")
    print("| sigma^2 | mu pos (n) | mu disp | SD pos / disp | D_tw pair: mean [90 % CI] (% mu); per-seed SD (% mu) "
          "| narrow D_c / D_f: mean (SD) | corr(D_c, D_f); per-seed order median [IQR]; same sign | skew / exkurt (prod) "
          "| h/16 - h/8 mean (SD) | var(D_tw) / particle |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    got = {}
    for a in AMPS:
        mu = AMP[a]["mean"]["pos"]
        P, T = estimand(red, a + "_prod"), estimand(red, a + "_tw")
        pv = np.array(list(P.values()))
        dm, ds, dn, dlo, dhi = ci90(np.array([P[k] - T[k] for k in P if k in T]))
        L = [estimand(red, "%s_%s" % (a, x)) for x in NARROW[a]]
        X = np.array([[lv[k] for lv in L] for k in L[0] if k in L[1] and k in L[2]])
        Dc, Df = X[:, 1] - X[:, 0], X[:, 2] - X[:, 1]
        with np.errstate(all="ignore"):
            order = np.log2(np.abs(Dc) / np.abs(Df))
        rho = float(np.corrcoef(Dc, Df)[0, 1])
        hm = {h: {k: float(r["tp.alpha_TP"]) for k, r in flux_rows(red, "%s_A_hmax%d" % (a, h)).items()} for h in (8, 16)}
        dh = np.array([hm[16][k] - hm[8][k] for k in sorted(set(hm[8]) & set(hm[16]))])
        fl = {nm: flux_rows(red, "%s_%s" % (a, nm)) for nm in ("prod", "tw")}
        kp = sorted(set(fl["prod"]) & set(fl["tw"]))
        dtp = np.array([float(fl["prod"][k]["tp.alpha_TP"]) - float(fl["tw"][k]["tp.alpha_TP"]) for k in kp])
        part = sum(np.mean([(float(fl[nm][k]["split.alpha_TP.A"]) - float(fl[nm][k]["split.alpha_TP.B"])) ** 2 / 4 for k in kp])
                   for nm in ("prod", "tw"))
        print("| %s | %.4e (%d) | %.4e | %.4e / %.4e | %s: %.3e [%.3e, %.3e] (%+.2f %%); %.3e (%.2f %%) | %.3e (%.3e) / %.3e (%.3e) "
              "| %.4f; %.2f [%.2f, %.2f]; %d/%d | %.2f / %.2f | %.3e (%.3e) | %.2f |" % (
                  SIG2[a], pv.mean(), len(pv), DISP_R3[a][0], pv.std(ddof=1), DISP_R3[a][1], TWIN_PAIR[a], dm, dlo, dhi,
                  100 * dm / mu, ds, 100 * ds / mu, Dc.mean(), Dc.std(ddof=1), Df.mean(), Df.std(ddof=1), rho,
                  np.median(order), *np.percentile(order, [25, 75]), int(np.sum(np.sign(Dc) == np.sign(Df))), len(Dc),
                  stats.skew(pv, bias=False), stats.kurtosis(pv, bias=False), dh.mean(), dh.std(ddof=1), dtp.var(ddof=1) / part))
        got[a] = dict(pos_mean=pv.mean(), pos_sd=pv.std(ddof=1), dtw_mean=dm, dtw_sd=ds,
                      dtw_lim=dlo if abs(dlo) > abs(dhi) else dhi, rho=rho, sdi=dh.std(ddof=1))
    print("\n## C3: width contrasts in the SF-43 estimand (Welch, 90 %); % of mu\n")
    print("| sigma^2 | case | wide: mean (SD, n) | reference: mean (SD, n) | difference [90 % CI] | % of mu [CI] | SE/mu |")
    print("|---|---|---|---|---|---|---|")
    for a, wide, ref, lab in WIDTH_CASES:
        w, r = np.array(list(estimand(red, wide).values())), np.array(list(estimand(red, ref).values()))
        va, vb = w.var(ddof=1) / len(w), r.var(ddof=1) / len(r)
        se = math.sqrt(va + vb)
        nu = (va + vb) ** 2 / (va ** 2 / (len(w) - 1) + vb ** 2 / (len(r) - 1))
        d = w.mean() - r.mean()
        h = stats.t.isf(0.05, nu) * se
        mu = AMP[a]["mean"]["pos"]
        print("| %s | %s | %.4e (%.3e, %d) | %.4e (%.3e, %d) | %.3e [%.3e, %.3e] | %+.1f [%+.1f, %+.1f] | %.1f %% |" % (
            SIG2[a], lab, w.mean(), w.std(ddof=1), len(w), r.mean(), r.std(ddof=1), len(r), d, d - h, d + h,
            100 * d / mu, 100 * (d - h) / mu, 100 * (d + h) / mu, 100 * se / mu))
    print("\n## Check: constants used by the simulations against the data (relative tolerance 1e-3; corr 1e-5)\n")
    print("| sigma^2 | quantity | from data | constant | match |")
    print("|---|---|---|---|---|")
    bad = 0
    for a in AMPS:
        g = got[a]
        checks = [("D_tw mean", g["dtw_mean"], DTW[a][0], 1e-3), ("D_tw per-seed SD", g["dtw_sd"], DTW[a][1], 1e-3),
                  ("D_tw adverse 90 % limit", g["dtw_lim"], DTW[a][2], 1e-3), ("SD pos", g["pos_sd"], AMP[a]["sd"]["pos"], 1e-3),
                  ("SD of h/16 - h/8", g["sdi"], AMP[a]["sdi"], 1e-3)]
        if a != "s025":  # at 0.25 the simulation uses the displacement mean on purpose
            checks.append(("mean pos", g["pos_mean"], AMP[a]["mean"]["pos"], 1e-3))
        if a in RHO_C:
            checks.append(("corr(D_c, D_f)", g["rho"], RHO_C[a], 1e-5))
        for name, v, c, tol in checks:
            ok = math.isclose(v, c, rel_tol=tol)
            bad += not ok
            print("| %s | %s | %.6e | %.6e | %s |" % (SIG2[a], name, v, c, "ok" if ok else "MISMATCH"))
    ns = os.path.join(os.path.dirname(os.path.abspath(sf42)), "verification", "noise_summary.csv") if sf42 else ""
    if ns and os.path.isfile(ns):
        with open(ns, newline="") as f:
            rows = [r for r in csv.reader(f) if r and not r[0].startswith("#")]
        hdr = rows[0]
        for r in rows[1:]:
            d = dict(zip(hdr, r))
            a = d["set"]
            if d["estimator"] == "alpha_TP" and d["basis"] == "disp" and a in DISP_R3:
                for name, v, c, tol in (("disp mean (SF-42 R3)", float(d["mean"]), DISP_R3[a][0], 1e-9),
                                        ("disp SD (SF-42 R3)", float(d["sd"]), DISP_R3[a][1], 1e-9),
                                        ("disp mean, rounded", float(d["mean"]), AMP[a]["mean"]["disp"], 1e-3),
                                        ("disp SD, rounded", float(d["sd"]), AMP[a]["sd"]["disp"], 1e-3)):
                    ok = math.isclose(v, c, rel_tol=tol)
                    bad += not ok
                    print("| %s | %s | %.6e | %.6e | %s |" % (SIG2[a], name, v, c, "ok" if ok else "MISMATCH"))
    else:
        print("| - | displacement constants | not checked (SF-42 verification/noise_summary.csv not found) | | |")
    if bad:
        sys.exit("%d constant(s) do not match the data" % bad)


def main(argv=None):
    ap = argparse.ArgumentParser(description="SF-43 pre-activation review: Gate 4 analysis (read-only).")
    ap.add_argument("command", choices=("components", "pipeline", "mde", "forced-pL", "numerical", "alloc"))
    ap.add_argument("--sf42-scripts", default=os.environ.get("SF42_SCRIPTS"),
                    help="SF-42 scripts directory that contains mcstat/ (default: $SF42_SCRIPTS)")
    ap.add_argument("--sf39-reduced", help="SF-39 analysis/reduced/ directory (components)")
    ap.add_argument("--R", type=int, help="replicates per cell (default 400; alloc and the D-alloc rows 300)")
    ap.add_argument("--B", type=int, help="LadderBound bootstrap size (default 4000; alloc and the D-alloc rows 3000)")
    args = ap.parse_args(argv)
    R, B, Ra, Ba = args.R or 400, args.B or 4000, args.R or 300, args.B or 3000
    if args.command == "components":
        cmd_components(args.sf39_reduced, args.sf42_scripts)
    elif args.command == "numerical":
        cmd_numerical()
    else:
        load_mcstat(args.sf42_scripts)
        if args.command == "pipeline":
            cmd_pipeline(R, B)
        elif args.command == "mde":
            cmd_mde(R, B, Ra, Ba)
        elif args.command == "forced-pL":
            cmd_forced(R, B)
        else:
            cmd_alloc(Ra, Ba, R, B)


if __name__ == "__main__":
    main()
