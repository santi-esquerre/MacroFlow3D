#!/usr/bin/env python3
"""SF-38 threshold C (RR6): the medium A/B equivalence test, computed from the per-realization primary statistics.

  ab_decision.py S1_PRIMARY.csv S4_PRIMARY.csv [--lambda1 0.0390625] [--lambda4 0.125]
  ab_decision.py --selftest

S1_PRIMARY.csv / S4_PRIMARY.csv: `ab_primary.csv` of the A/B application for sigma^2 = 1 (2048x256x256, ell = 10/256) and
sigma^2 = 4 (3328x256x256, ell = 32/256).  Exactly ten paired realizations (the same seeds in both media) per amplitude.
Input grammar (the first line is the header, `#` lines are skipped, names are case-insensitive):
  long form : columns `seed` (or `r`), `medium` (`a`|`b`), `S_T`, `S_tau`, `V1`, `theta_2`, `theta_3`   (20 rows)
  wide form : columns `seed` (or `r`), and `<stat>_a`, `<stat>_b` for the five statistics                 (10 rows)

Threshold C, verbatim in the activation bitacora row (t_{0.95,9} = 1.833, N = 10, sample standard deviation n-1):
  ratio statistics S_T, S_tau, V1:  R = ln(Pbar_a / Pbar_b), CI R +- 1.833 SE_R,
      SE_R = sd_r(P_a,r / Pbar_a - P_b,r / Pbar_b) / sqrt(10); margins +-ln(1.05) (S_T), +-ln(1.025) (S_tau), +-ln(1.05) (V1).
  S_T floor: if Pbar_b < 0.05 lambda or Pbar_a <= 0, S_T is tested on D_r = P_a,r - P_b,r, CI Dbar +- 1.833 sd(D)/sqrt(10),
      margin +-2.5e-3 lambda.  Units: S_T = [sigma_T^2(X_m*) - sigma_T^2(X_m*/2)] / ((m*/2) lambda) has the dimension of a
      length, and the box is the unit of length of every quantity of the A/B (lambda = ell in box units: 0.0390625 for
      sigma^2 = 1, 0.125 for sigma^2 = 4), so both the trigger 0.05 lambda and the margin 2.5e-3 lambda are in box units.
  S_tau or V1 with Pbar_a <= 0 or Pbar_b <= 0: inconclusive.
  tilts theta_2, theta_3 (each component): D_r = P_a,r - P_b,r, CI Dbar +- 1.833 sd(D)/sqrt(10), margin +-0.005.
  class: equivalent iff the CI lies inside the margin ([-m, m], closed), non-equivalent iff the CI lies entirely outside
  (hi < -m or lo > m), inconclusive otherwise.  Medium (a) is selected iff all ten primary statistics (both amplitudes)
  are equivalent; otherwise (b).
Output (stdout, machine readable; every line starts with its token):
  C_INPUT amp= file= n_pairs= seeds=
  C_D amp= stat= seed= P_a= P_b= D=             (every paired difference, all statistics)
  C_STAT amp= stat= test=ratio|floor|diff|nonpositive Pbar_a= Pbar_b= est= se= ci_lo= ci_hi= margin= class=
  C_AMP amp= equivalent=k/5 non_equivalent= inconclusive=
  C_VERDICT selected=a|b equivalent=k/10 non_equivalent= inconclusive=
Exit 0: a selection was computed; 2: unreadable / malformed input (no selection).  Stdlib only.
"""
import csv
import math
import os
import sys

T95_9 = 1.833
N_REAL = 10
STATS = ["S_T", "S_tau", "V1", "theta_2", "theta_3"]
LN_MARGIN = {"S_T": math.log(1.05), "S_tau": math.log(1.025), "V1": math.log(1.05)}
TILT_MARGIN = 0.005
FLOOR_TRIGGER = 0.05  # x lambda
FLOOR_MARGIN = 2.5e-3  # x lambda
LAMBDA_DEFAULT = {1: 0.0390625, 4: 0.125}


class InputError(Exception):
    pass


def mean(v):
    return sum(v) / len(v)


def sd(v):
    m = mean(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))


def classify(lo, hi, margin):
    if lo >= -margin and hi <= margin:
        return "equivalent"
    if hi < -margin or lo > margin:
        return "non-equivalent"
    return "inconclusive"


def diff_test(pa, pb, margin):
    d = [x - y for x, y in zip(pa, pb)]
    dbar = mean(d)
    se = sd(d) / math.sqrt(len(d))
    lo, hi = dbar - T95_9 * se, dbar + T95_9 * se
    return dbar, se, lo, hi, classify(lo, hi, margin)


def ratio_test(pa, pb, margin):
    ma, mb = mean(pa), mean(pb)
    r = math.log(ma / mb)
    x = [a / ma - b / mb for a, b in zip(pa, pb)]
    se = sd(x) / math.sqrt(len(x))
    lo, hi = r - T95_9 * se, r + T95_9 * se
    return r, se, lo, hi, classify(lo, hi, margin)


def test_stat(stat, pa, pb, lam):
    """-> dict(test, est, se, lo, hi, margin, cls)"""
    ma, mb = mean(pa), mean(pb)
    if stat in ("theta_2", "theta_3"):
        est, se, lo, hi, cls = diff_test(pa, pb, TILT_MARGIN)
        return dict(test="diff", est=est, se=se, lo=lo, hi=hi, margin=TILT_MARGIN, cls=cls)
    if stat == "S_T" and (mb < FLOOR_TRIGGER * lam or ma <= 0):
        m = FLOOR_MARGIN * lam
        est, se, lo, hi, cls = diff_test(pa, pb, m)
        return dict(test="floor", est=est, se=se, lo=lo, hi=hi, margin=m, cls=cls)
    if ma <= 0 or mb <= 0:  # S_tau or V1 (S_T cannot get here: ma <= 0 or mb < 0.05 lambda went to the floor)
        nan = float("nan")
        return dict(test="nonpositive", est=nan, se=nan, lo=nan, hi=nan, margin=LN_MARGIN[stat], cls="inconclusive")
    est, se, lo, hi, cls = ratio_test(pa, pb, LN_MARGIN[stat])
    return dict(test="ratio", est=est, se=se, lo=lo, hi=hi, margin=LN_MARGIN[stat], cls=cls)


def read_primary(path):
    """-> (seeds sorted, {stat: ([P_a...], [P_b...])}) with the realizations ordered by seed."""
    try:
        with open(path, newline="") as f:
            lines = [ln for ln in f.read().splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    except OSError as e:
        raise InputError("cannot read %s: %s" % (path, e))
    if not lines:
        raise InputError("%s: empty" % path)
    rows = list(csv.reader(lines))
    head = [h.strip().lower() for h in rows[0]]
    body = rows[1:]
    lstats = {s.lower(): s for s in STATS}

    def col(name):
        if name not in head:
            raise InputError("%s: missing column '%s' (header %s)" % (path, name, head))
        return head.index(name)

    skey = "seed" if "seed" in head else ("r" if "r" in head else None)
    if skey is None:
        raise InputError("%s: needs a 'seed' (or 'r') column" % path)
    si = head.index(skey)

    def num(x, what):
        try:
            return float(x)
        except ValueError:
            raise InputError("%s: %s: '%s' is not a number" % (path, what, x))

    vals = {s: {} for s in STATS}  # stat -> medium -> {seed: value}
    for s in STATS:
        vals[s] = {"a": {}, "b": {}}
    if "medium" in head:
        mi = head.index("medium")
        ci = {s: col(s.lower()) for s in STATS}
        for row in body:
            if len(row) != len(head):
                raise InputError("%s: ragged row %s" % (path, row))
            med = row[mi].strip().lower()
            if med not in ("a", "b"):
                raise InputError("%s: medium must be a or b, got '%s'" % (path, row[mi]))
            seed = row[si].strip()
            for s in STATS:
                if seed in vals[s][med]:
                    raise InputError("%s: duplicate (seed %s, medium %s)" % (path, seed, med))
                vals[s][med][seed] = num(row[ci[s]], "%s/%s" % (s, med))
    else:
        for s in STATS:
            ia, ib = col(s.lower() + "_a"), col(s.lower() + "_b")
            for row in body:
                if len(row) != len(head):
                    raise InputError("%s: ragged row %s" % (path, row))
                seed = row[si].strip()
                if seed in vals[s]["a"]:
                    raise InputError("%s: duplicate seed %s" % (path, seed))
                vals[s]["a"][seed] = num(row[ia], s + "_a")
                vals[s]["b"][seed] = num(row[ib], s + "_b")
    seeds = sorted(vals[STATS[0]]["a"], key=lambda x: (len(x), x))
    for s in STATS:
        for med in ("a", "b"):
            if set(vals[s][med]) != set(seeds):
                raise InputError("%s: realization (seed) sets differ between media / statistics" % path)
    if len(seeds) != N_REAL:
        raise InputError("%s: %d paired realizations, exactly %d required" % (path, len(seeds), N_REAL))
    out = {s: ([vals[s]["a"][k] for k in seeds], [vals[s]["b"][k] for k in seeds]) for s in STATS}
    return seeds, out


def fmt(x):
    return "nan" if x != x else "%.12e" % x


def decide(files, lams):
    """files: {amp: path}; lams: {amp: lambda} -> (lines, selected)"""
    lines = []
    ncls = {"equivalent": 0, "non-equivalent": 0, "inconclusive": 0}
    for amp in (1, 4):
        seeds, data = read_primary(files[amp])
        lines.append("C_INPUT amp=%d file=%s n_pairs=%d lambda=%.17g seeds=%s" % (amp, files[amp], len(seeds), lams[amp],
                                                                                 ",".join(seeds)))
        acls = {"equivalent": 0, "non-equivalent": 0, "inconclusive": 0}
        for s in STATS:
            pa, pb = data[s]
            for k, seed in enumerate(seeds):
                lines.append("C_D amp=%d stat=%s seed=%s P_a=%s P_b=%s D=%s" % (amp, s, seed, fmt(pa[k]), fmt(pb[k]),
                                                                              fmt(pa[k] - pb[k])))
            t = test_stat(s, pa, pb, lams[amp])
            acls[t["cls"]] += 1
            ncls[t["cls"]] += 1
            lines.append("C_STAT amp=%d stat=%s test=%s Pbar_a=%s Pbar_b=%s est=%s se=%s ci_lo=%s ci_hi=%s margin=%s class=%s"
                         % (amp, s, t["test"], fmt(mean(pa)), fmt(mean(pb)), fmt(t["est"]), fmt(t["se"]), fmt(t["lo"]),
                            fmt(t["hi"]), fmt(t["margin"]), t["cls"]))
        lines.append("C_AMP amp=%d equivalent=%d/5 non_equivalent=%d inconclusive=%d"
                     % (amp, acls["equivalent"], acls["non-equivalent"], acls["inconclusive"]))
    sel = "a" if ncls["equivalent"] == 2 * len(STATS) else "b"
    lines.append("C_VERDICT selected=%s equivalent=%d/%d non_equivalent=%d inconclusive=%d"
                 % (sel, ncls["equivalent"], 2 * len(STATS), ncls["non-equivalent"], ncls["inconclusive"]))
    return lines, sel


# ----------------------------------------------------------------------------------------------------------------
def selftest():
    import tempfile
    fails = []

    def expect(name, cond):
        if not cond:
            fails.append(name)
            print("SELFTEST FAIL %s" % name)

    def close(a, b, rel=1e-9, ab=1e-15):
        return abs(a - b) <= max(rel * abs(b), ab)

    u = [1.0] * 5 + [-1.0] * 5  # sd(u) = sqrt(10/9), mean 0
    sq10 = math.sqrt(10.0)

    # 1. ratio, equivalent: P_a = 2 (1 + 0.01 u), P_b = 2 -> R = 0, SE = 0.01/3, CI = +-1.833*0.01/3
    pa = [2 * (1 + 0.01 * x) for x in u]
    pb = [2.0] * 10
    t = test_stat("S_tau", pa, pb, 0.04)
    expect("ratio_est", close(t["est"], 0.0, ab=1e-14))
    expect("ratio_se", close(t["se"], 0.01 / 3))
    expect("ratio_ci", close(t["hi"], T95_9 * 0.01 / 3) and close(t["lo"], -T95_9 * 0.01 / 3))
    expect("ratio_margin_stau", close(t["margin"], math.log(1.025)))
    expect("ratio_equiv", t["test"] == "ratio" and t["cls"] == "equivalent")
    # 1b. same statistic with S_tau margin ln(1.025) = 0.02469: R = 0.03 -> inconclusive/non-equivalent edge
    # 2. ratio, non-equivalent: P_a = 2.2 (1 + 0.01 u), P_b = 2 -> R = ln 1.1 = 0.0953102
    t = test_stat("V1", [2.2 * (1 + 0.01 * x) for x in u], pb, 0.04)
    expect("noneq_est", close(t["est"], math.log(1.1)))
    expect("noneq_lo", close(t["lo"], math.log(1.1) - T95_9 * 0.01 / 3))
    expect("noneq_cls", t["cls"] == "non-equivalent" and close(t["margin"], math.log(1.05)))
    # 3. ratio, inconclusive: P_a = 2.0816 (1 + 0.05 u), P_b = 2 -> R = ln 1.0408, SE = 0.05/3, CI straddles ln 1.05
    t = test_stat("S_T", [2.0816 * (1 + 0.05 * x) for x in u], pb, 0.04)
    expect("incl_est", close(t["est"], math.log(1.0408)))
    expect("incl_se", close(t["se"], 0.05 / 3))
    expect("incl_cls", t["test"] == "ratio" and t["cls"] == "inconclusive"
           and t["lo"] > -t["margin"] and t["hi"] > t["margin"])
    # 3b. the SE uses the sample sd (n-1): u has sum of squares 10, sd = sqrt(10/9), SE = 0.05 sqrt(10/9)/sqrt(10) = 0.05/3
    expect("sd_n_minus_1", close(sd(u), math.sqrt(10.0 / 9.0)))
    # 4. S_T floor switch (lambda = 0.04: trigger 0.05 lambda = 0.002, margin 2.5e-3 lambda = 1e-4)
    lam = 0.04
    pb_small = [0.001] * 10  # Pbar_b = 0.001 < 0.002
    pa_small = [0.00105 + 0.00005 * x for x in u]
    t = test_stat("S_T", pa_small, pb_small, lam)
    expect("floor_test", t["test"] == "floor" and close(t["margin"], 1e-4))
    expect("floor_est", close(t["est"], 5e-5) and close(t["se"], 5e-5 / 3))
    expect("floor_ci", close(t["lo"], 5e-5 - T95_9 * 5e-5 / 3) and close(t["hi"], 5e-5 + T95_9 * 5e-5 / 3))
    expect("floor_equiv", t["cls"] == "equivalent")
    t = test_stat("S_T", [0.0015 + 0.00005 * x for x in u], pb_small, lam)
    expect("floor_noneq", t["test"] == "floor" and t["cls"] == "non-equivalent")
    # at Pbar_b = 0.0021 (>= 0.002) the ratio path is used
    t = test_stat("S_T", [0.0021 * (1 + 0.01 * x) for x in u], [0.0021] * 10, lam)
    expect("ratio_above_floor", t["test"] == "ratio")
    # Pbar_a <= 0 -> floor for S_T, inconclusive for S_tau / V1
    t = test_stat("S_T", [-0.01 + 0.0001 * x for x in u], [0.5] * 10, lam)
    expect("floor_pa_nonpos", t["test"] == "floor" and t["cls"] == "non-equivalent")
    t = test_stat("S_tau", [-0.01 + 0.0001 * x for x in u], [0.5] * 10, lam)
    expect("stau_pa_nonpos", t["test"] == "nonpositive" and t["cls"] == "inconclusive")
    t = test_stat("V1", [0.5] * 10, [-1.0 + 0.1 * x for x in u], lam)
    expect("v1_pb_nonpos", t["test"] == "nonpositive" and t["cls"] == "inconclusive")
    # 5. tilts (margin 0.005, paired difference)
    t = test_stat("theta_2", [0.002 + 0.001 * x for x in u], [0.0] * 10, lam)
    expect("tilt_equiv", t["test"] == "diff" and close(t["est"], 0.002) and close(t["se"], 0.001 / 3)
           and t["cls"] == "equivalent" and close(t["margin"], 0.005))
    t = test_stat("theta_3", [0.0075 + 0.001 * x for x in u], [0.0] * 10, lam)
    expect("tilt_noneq", t["cls"] == "non-equivalent" and close(t["lo"], 0.0075 - T95_9 * 0.001 / 3))
    t = test_stat("theta_3", [0.004 + 0.003 * x for x in u], [0.0] * 10, lam)
    expect("tilt_incl", t["cls"] == "inconclusive" and close(t["se"], 0.001) and close(t["hi"], 0.004 + T95_9 * 0.001))
    # inside is closed: CI exactly [-m, m] is equivalent; just outside entirely is non-equivalent
    expect("classify_edges", classify(-0.005, 0.005, 0.005) == "equivalent" and classify(0.0050001, 0.01, 0.005) == "non-equivalent"
           and classify(-0.01, -0.0050001, 0.005) == "non-equivalent" and classify(-0.001, 0.0051, 0.005) == "inconclusive")

    # 6. full decision through CSV files (long and wide forms) and the selection rule
    def write_long(path, perturb=None, lam=0.1, seeds=range(4001, 4011), drop=None):
        with open(path, "w") as f:
            f.write("# synthetic\nseed,medium,S_T,S_tau,V1,theta_2,theta_3\n")
            for k, sd_ in enumerate(seeds):
                x = u[k] if k < 10 else 0.0
                base = {"S_T": 2.0 * lam, "S_tau": 3.0, "V1": 0.5, "theta_2": 0.001, "theta_3": -0.001}
                for med in ("a", "b"):
                    row = []
                    for s in STATS:
                        v = base[s] * (1 + 0.005 * x) if s not in ("theta_2", "theta_3") else base[s] + 0.0005 * x
                        if med == "a" and perturb and s == perturb[0]:
                            v = perturb[1](v, x)
                        row.append(v)
                    if drop and (sd_, med) == drop:
                        continue
                    f.write("%d,%s,%s\n" % (sd_, med, ",".join("%.17g" % v for v in row)))

    def write_wide(path):
        with open(path, "w") as f:
            f.write("seed," + ",".join("%s_a,%s_b" % (s, s) for s in STATS) + "\n")
            for k in range(10):
                row = []
                for s in STATS:
                    b = {"S_T": 0.2, "S_tau": 3.0, "V1": 0.5, "theta_2": 0.001, "theta_3": -0.001}[s]
                    row += [b, b]
                f.write("%d,%s\n" % (4001 + k, ",".join("%.17g" % v for v in row)))

    with tempfile.TemporaryDirectory() as d:
        p1, p4 = os.path.join(d, "s1.csv"), os.path.join(d, "s4.csv")
        write_long(p1, lam=0.0390625)
        write_long(p4, lam=0.125)
        lines, sel = decide({1: p1, 4: p4}, LAMBDA_DEFAULT)
        expect("all_equivalent_selects_a", sel == "a" and any(l.startswith("C_VERDICT selected=a equivalent=10/10") for l in lines))
        expect("lines_c_d", sum(1 for l in lines if l.startswith("C_D ")) == 2 * 5 * 10)
        expect("lines_c_stat", sum(1 for l in lines if l.startswith("C_STAT ")) == 10)
        # one non-equivalent statistic (S_tau of a shifted by +10 %) -> b
        write_long(p4, perturb=("S_tau", lambda v, x: v * 1.1), lam=0.125)
        lines, sel = decide({1: p1, 4: p4}, LAMBDA_DEFAULT)
        expect("one_noneq_selects_b", sel == "b" and any("C_STAT amp=4 stat=S_tau" in l and "class=non-equivalent" in l for l in lines)
               and any(l.startswith("C_VERDICT selected=b equivalent=9/10 non_equivalent=1 inconclusive=0") for l in lines))
        # one inconclusive statistic (theta_2 difference 0.0049 +- 0.004: CI [0.00246, 0.00734] straddles the margin) -> b
        write_long(p4, perturb=("theta_2", lambda v, x: v + 0.0049 + 0.004 * x), lam=0.125)
        lines, sel = decide({1: p1, 4: p4}, LAMBDA_DEFAULT)
        expect("tilt_shift_inconclusive_selects_b", sel == "b" and any("C_STAT amp=4 stat=theta_2" in l and "class=inconclusive" in l
                                                                       for l in lines))
        # wide form
        write_wide(p1)
        write_wide(p4)
        lines, sel = decide({1: p1, 4: p4}, LAMBDA_DEFAULT)
        expect("wide_form_zero_difference", sel == "a")
        # input errors: 9 realizations; seed sets that differ between media; non-numeric
        write_long(p4, lam=0.125, drop=(4010, "b"))
        try:
            decide({1: p1, 4: p4}, LAMBDA_DEFAULT)
            expect("missing_pair_rejected", False)
        except InputError:
            expect("missing_pair_rejected", True)
        write_long(p4, lam=0.125, seeds=range(4001, 4010))
        try:
            decide({1: p1, 4: p4}, LAMBDA_DEFAULT)
            expect("nine_rejected", False)
        except InputError:
            expect("nine_rejected", True)
        try:
            read_primary(os.path.join(d, "absent.csv"))
            expect("absent_rejected", False)
        except InputError:
            expect("absent_rejected", True)
    expect("t_value", T95_9 == 1.833 and N_REAL == 10)
    if fails:
        print("SELFTEST FAILED: %s" % ", ".join(fails))
        return 1
    print("SELFTEST OK ab_decision")
    return 0


def main(argv):
    if len(argv) == 2 and argv[1] == "--selftest":
        return selftest()
    lams = dict(LAMBDA_DEFAULT)
    pos = []
    i = 1
    while i < len(argv):
        if argv[i] in ("--lambda1", "--lambda4") and i + 1 < len(argv):
            try:
                lams[1 if argv[i] == "--lambda1" else 4] = float(argv[i + 1])
            except ValueError:
                print("ab_decision.py: bad number %r" % argv[i + 1], file=sys.stderr)
                return 2
            i += 2
        elif argv[i].startswith("--"):
            print("ab_decision.py: bad argument %r" % argv[i], file=sys.stderr)
            return 2
        else:
            pos.append(argv[i])
            i += 1
    if len(pos) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    try:
        lines, _ = decide({1: pos[0], 4: pos[1]}, lams)
    except InputError as e:
        print("C_ERROR %s" % e)
        return 2
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
