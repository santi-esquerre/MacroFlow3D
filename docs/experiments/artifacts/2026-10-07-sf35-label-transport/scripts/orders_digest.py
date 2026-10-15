#!/usr/bin/env python3
"""SF-35 convergence digest of one amplitude (matrix part): tables, observed orders, verdicts B and C.

  orders_digest.py --eps E raw/matrix_eps<E>/N*.json [--out analysis/matrix_eps<E>.md]
                   [--planes-csv PATH] [--expected-ns 32,64,128,256]
  orders_digest.py --selftest

Reads the driver `--summary` JSON of `inlet_slab --construct trace` (the primary source; keys metrics.trace,
residual, backflow, growth, transport, planes, ladder, timing, memory, status).  The `/usr/bin/time -v` file of a
point (logs/<part>/N<N>.time, found by replacing raw/ by logs/ and .json by .time) supplies the host max RSS.
A point whose stage failed (inlet_backflow, darcy_failed) has no tracing data: it is listed with its status and every
verdict that needs it is `not_evaluable`.  Deterministic; standard library only.
"""
import argparse
import csv
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (rel, fnum, get, load_json, md_table, n_of, num, order, parse_time_file, write_out)  # noqa: E402

RULE_TEXT = """\
Verdict rules (pre-registered, SF-35 spec "Acceptance thresholds", implemented literally):
- B. Round trip <= 1e-8 on every plane at the primary (h/8, 1e-8) for every matrix point; non_ok = 0; at 32 and 64 the
  ladder gives max|dpsi| between the primary and (h/32, 1e-10) <= 1e-8.
- C. Convergence per amplitude (32/64/128/256). e_v, e_i1, e_i2, e_div, r_F each decrease on every refinement and show
  observed order >= 1.8 on 128 -> 256: `accept`; any of them changing < 10 % between 128 and 256 while above 1e-12:
  `reject` (floor); otherwise `unresolved`. Amplitude verdict: `accept` iff all five quantities are `accept`, else
  `reject` if any is `reject`, else `unresolved`. Blocking: eps = 0.5 and 1.0 must be `accept`; eps = 1.5 must be
  `accept` unless the stage reports `inlet_backflow` (then recorded; owner decision).
- Implementation reading: `decrease on every refinement` = q(2N) < q(N) for every consecutive pair of the expected grid
  list; the observed order on the last pair is log2(q(128)/q(256)); the relative change is |q(256) - q(128)| / |q(128)|;
  a non-finite value or a missing grid makes the quantity `not_evaluable` (never silently `unresolved`)."""

QUANTS = ("e_v", "e_i1", "e_i2", "e_div", "r_F")


# ------------------------------------------------------------------------------------------
class Point:
    """One matrix point (one driver JSON)."""

    def __init__(self, path, js):
        self.path = path
        self.js = js
        self.N = n_of(path, js)
        self.status = get(js, "status", default="unknown")
        self.exit_code = get(js, "exit_code", default=None)
        self.traced = "planes" in js and "metrics" in js and "trace" in js.get("metrics", {})
        m = get(js, "metrics", "trace", default={}) or {}
        self.m = {k: num(m.get(k)) for k in ("e_v", "e_i1", "e_i2", "e_div", "min_c", "p0.1", "p1", "p5", "p50")}
        self.m["r_F"] = num(get(js, "residual", "r_F"))
        self.m["r_out"] = num(get(js, "residual", "r_out"))
        self.bf = get(js, "backflow", default={}) or {}
        self.tr = get(js, "transport", default={}) or {}
        self.growth = get(js, "growth", default={}) or {}
        self.planes = get(js, "planes", default=[]) or []
        self.ladder = get(js, "ladder", default=[]) or []
        self.timing_total = num(get(js, "timing", "total"))
        self.peak_dev = num(get(js, "memory", "peak_device_bytes"))
        tf = path.replace(os.sep + "raw" + os.sep, os.sep + "logs" + os.sep)
        tf = tf[:-5] + ".time" if tf.endswith(".json") else tf + ".time"
        self.time = parse_time_file(tf)

    # per-plane round trip: JSON transport.roundtrip_per_plane (planes 1..N), else planes[].roundtrip
    def roundtrips(self):
        rt = self.tr.get("roundtrip_per_plane")
        if rt:
            return [num(x) for x in rt]
        return [num(p.get("roundtrip")) for p in self.planes if p.get("j", 0) > 0]

    def max_roundtrip(self):
        g = num(self.growth.get("max_roundtrip"))
        rts = self.roundtrips()
        if rts:
            # NaN counts as a failure: propagate it
            return float("nan") if any(math.isnan(x) for x in rts) else max(rts)
        return g

    def non_ok(self):
        v = self.tr.get("non_ok")
        return None if v is None else int(v)

    def ladder_entry(self, div, tol):
        for e in self.ladder:
            if int(e.get("hmax_div", -1)) == div and abs(num(e.get("tol_requested")) - tol) <= 1e-3 * tol:
                return e
        return None


def e_i_plane(p):
    a, b = num(p.get("e_i1")), num(p.get("e_i2"))
    return math.sqrt((a * a + b * b) / 2.0)


# ------------------------------------------------------------------------------------------
def verdict_quantity(values, ns, expected):
    """Literal C verdict for one quantity.  values: dict N -> q.  Returns (verdict, detail dict)."""
    d = {"order_last": float("nan"), "relchange_last": float("nan"), "decreasing": None}
    missing = [n for n in expected if n not in values]
    if len(expected) < 2:
        return "not_evaluable", dict(d, why="fewer than two expected grids")
    if missing:
        return "not_evaluable", dict(d, why="missing N=" + ",".join(str(m) for m in missing))
    seq = [num(values[n]) for n in expected]
    if any(not math.isfinite(x) for x in seq):
        return "not_evaluable", dict(d, why="non-finite value")
    dec = all(seq[i + 1] < seq[i] for i in range(len(seq) - 1))
    last_o = order(seq[-2], seq[-1])
    rel = abs(seq[-1] - seq[-2]) / abs(seq[-2]) if seq[-2] != 0.0 else float("inf")
    d.update(order_last=last_o, relchange_last=rel, decreasing=dec)
    if dec and not math.isnan(last_o) and last_o >= 1.8:
        return "accept", d
    if rel < 0.10 and seq[-1] > 1e-12:
        return "reject", d
    return "unresolved", d


def aggregate(verdicts):
    if any(v == "not_evaluable" for v in verdicts):
        return "not_evaluable"
    if all(v == "accept" for v in verdicts):
        return "accept"
    if any(v == "reject" for v in verdicts):
        return "reject"
    return "unresolved"


def verdict_b(points, expected):
    """Returns (verdict, per-point rows)."""
    rows, states = [], []
    for n in expected:
        p = points.get(n)
        if p is None:
            rows.append((n, "missing", "-", "-", "-", "not_evaluable"))
            states.append("not_evaluable")
            continue
        if not p.traced:
            rows.append((n, p.status, "-", "-", "-", "not_evaluable"))
            states.append("not_evaluable")
            continue
        rt = p.max_roundtrip()
        no = p.non_ok()
        ok_rt = (not math.isnan(rt)) and rt <= 1e-8
        ok_no = no == 0
        lad = ""
        ok_lad = True
        if n in (32, 64):
            e = p.ladder_entry(32, 1e-10)
            if e is None:
                lad = "ladder missing"
                ok_lad = None
            else:
                dd = max(num(get(e, "labeldiff", "max_dpsi1")), num(get(e, "labeldiff", "max_dpsi2")))
                lad = fnum(dd)
                ok_lad = (not math.isnan(dd)) and dd <= 1e-8
        else:
            lad = "n/a"
        if ok_lad is None:
            st = "not_evaluable" if (ok_rt and ok_no) else "fail"
        else:
            st = "pass" if (ok_rt and ok_no and ok_lad) else "fail"
        states.append(st)
        rows.append((n, p.status, fnum(rt), "-" if no is None else no, lad, st))
    if any(s == "fail" for s in states):
        v = "fail"
    elif all(s == "pass" for s in states):
        v = "pass"
    else:
        v = "not_evaluable"
    return v, rows


# ------------------------------------------------------------------------------------------
def build_report(eps, points, expected, planes_csv=None):
    L = []
    ns = sorted(points)
    L.append("# SF-35 matrix digest, eps = %s (sigma_Y = %s)" % (eps, eps))
    L.append("")
    L.append("```text")
    L.append(RULE_TEXT)
    L.append("```")
    L.append("")
    L.append("Grids present: %s; expected for the verdicts: %s." % (ns, expected))
    L.append("")
    traced = [n for n in ns if points[n].traced]
    # ---- stage status ---------------------------------------------------------------------
    L.append("## Status per grid")
    L.append("")
    L.append(md_table(["N", "status", "exit_code", "stage"],
                      [(n, points[n].status, points[n].exit_code,
                        "traced" if points[n].traced else "no tracing data (stage/trace not completed)")
                       for n in ns]))
    L.append("")
    if not traced:
        L.append("No point has tracing data: every verdict is `not_evaluable`.")
        L.append("")
    # ---- main table -----------------------------------------------------------------------
    cols = traced

    def row(label, fn):
        return [label] + [fn(points[n]) for n in cols]

    if cols:
        L.append("## Quantities per grid")
        L.append("")
        hdr = ["quantity"] + ["N=%d" % n for n in cols]
        rows = []
        for q in ("e_v", "e_i1", "e_i2", "e_div", "r_F", "r_out", "min_c", "p0.1", "p1", "p5", "p50"):
            rows.append(row(q, lambda p, q=q: fnum(p.m[q])))
        rows.append(row("max_roundtrip", lambda p: fnum(p.max_roundtrip())))
        rows.append(row("non_ok", lambda p: p.non_ok()))
        rows.append(row("status", lambda p: p.status))
        rows.append(row("inlet_vmin", lambda p: fnum(p.bf.get("inlet_vmin"))))
        rows.append(row("vD1_min", lambda p: fnum(p.bf.get("vD1_min"))))
        rows.append(row("vD1_nonpos", lambda p: p.bf.get("vD1_nonpos")))
        rows.append(row("seed_backflow", lambda p: p.bf.get("seed_backflow")))
        rows.append(row("backflow_enc", lambda p: p.bf.get("backflow_enc")))
        rows.append(row("landings_discarded", lambda p: p.bf.get("landings_discarded")))
        rows.append(row("min_g1_hat", lambda p: fnum(p.bf.get("min_g1_hat"))))
        rows.append(row("status_hist (ok,seed_bf,cap,stag,uflow,landfail,invalid_k)",
                        lambda p: ",".join(str(x) for x in p.bf.get("status_hist", [])) or "-"))
        rows.append(row("e_v_N/e_v_1", lambda p: fnum(p.growth.get("e_v_N_over_e_v_1"))))
        rows.append(row("e_i_N/e_i_1", lambda p: fnum(p.growth.get("e_i_N_over_e_i_1"))))
        rows.append(row("wall (s, driver total)", lambda p: fnum(p.timing_total, "%.2f")))
        rows.append(row("wall (s, /usr/bin/time)", lambda p: fnum(p.time["wall_s"], "%.1f")))
        rows.append(row("max RSS (kB)", lambda p: fnum(p.time["max_rss_kB"], "%.0f")))
        rows.append(row("peak device (bytes)", lambda p: fnum(p.peak_dev, "%.0f")))
        L.append(md_table(hdr, rows))
        L.append("")
        # ---- observed orders --------------------------------------------------------------
        L.append("## Observed orders log2(q_N / q_2N)")
        L.append("")
        pairs = [(a, b) for a, b in zip(cols[:-1], cols[1:]) if b == 2 * a]
        if pairs:
            orows = []
            for q in QUANTS:
                orows.append([q] + [fnum(order(points[a].m[q], points[b].m[q]), "%.2f") for a, b in pairs])
            L.append(md_table(["quantity"] + ["%d->%d" % pr for pr in pairs], orows))
        else:
            L.append("No consecutive pair of grids (N, 2N) available: orders not_evaluable.")
        L.append("")
        # ---- per-plane growth -------------------------------------------------------------
        L.append("## Per-plane growth along x1 (E: recorded, no threshold)")
        L.append("")
        L.append("`e_i_j = sqrt((e_i1_j^2 + e_i2_j^2)/2)` (as the GROWTH line). Full lists: " +
                 (("`%s`" % rel(planes_csv)) if planes_csv else "(csv not written)") + ".")
        L.append("")
        for n in cols:
            p = points[n]
            sel = sorted(set([0, 1, n // 4, n // 2, (3 * n) // 4, n]))
            byj = {int(pl["j"]): pl for pl in p.planes}
            rr = []
            for j in sel:
                pl = byj.get(j)
                if pl is None:
                    rr.append((j, "-", "-", "-", "-"))
                else:
                    rr.append((j, fnum(pl.get("x1"), "%.4f"), fnum(pl.get("e_v")), fnum(e_i_plane(pl)),
                               fnum(pl.get("roundtrip"))))
            L.append("N = %d" % n)
            L.append("")
            L.append(md_table(["j", "x1", "e_v_j", "e_i_j", "roundtrip_j"], rr))
            L.append("")
        # ---- ladder -----------------------------------------------------------------------
        L.append("## Integration ladder (B)")
        L.append("")
        lrows = []
        for n in cols:
            p = points[n]
            for e in p.ladder:
                lrows.append((n, "h/%s" % e.get("hmax_div"), fnum(e.get("tol_requested"), "%.0e"),
                              fnum(get(e, "labeldiff", "max_dpsi1")), fnum(get(e, "labeldiff", "max_dpsi2")),
                              fnum(e.get("max_roundtrip")), e.get("status")))
        if lrows:
            L.append(md_table(["N", "hmax", "tol", "max|dpsi1| vs primary", "max|dpsi2| vs primary",
                               "max_roundtrip", "status"], lrows))
        else:
            L.append("No ladder entries in the inputs (the campaign runs --ladder at N = 32 and 64).")
        L.append("")
    # ---- verdicts -------------------------------------------------------------------------
    L.append("## Verdict B (integration certificate)")
    L.append("")
    vb, brows = verdict_b(points, expected)
    L.append(md_table(["N", "status", "max_roundtrip (every plane)", "non_ok",
                       "ladder max|dpsi| (primary vs h/32,1e-10; N=32,64)", "point"], brows))
    L.append("")
    L.append("**B verdict: %s**" % vb)
    L.append("")
    L.append("## Verdict C (convergence per amplitude)")
    L.append("")
    crow, vs = [], []
    for q in QUANTS:
        vals = {n: points[n].m[q] for n in traced}
        v, d = verdict_quantity(vals, ns, expected)
        vs.append(v)
        crow.append((q, fnum(vals.get(expected[-2])) if len(expected) > 1 else "-",
                     fnum(vals.get(expected[-1])), fnum(d["order_last"], "%.2f"),
                     fnum(d["relchange_last"], "%.3f"), d["decreasing"], v, d.get("why", "")))
    L.append(md_table(["quantity", "q(%s)" % (expected[-2] if len(expected) > 1 else "-"), "q(%s)" % expected[-1],
                       "order last pair", "rel change last pair", "decreases on every refinement", "verdict", "note"],
                      crow))
    L.append("")
    va = aggregate(vs)
    L.append("**Amplitude verdict (C): %s**" % va)
    L.append("")
    statuses = {points[n].status for n in ns}
    if "inlet_backflow" in statuses:
        L.append("Note: the stage reported `inlet_backflow` at least once; for eps = 1.5 this is recorded and "
                 "left to the owner decision (not a failure by itself); other amplitudes: blocking.")
        L.append("")
    return "\n".join(L), vb, va


def write_planes_csv(points, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["N", "j", "x1", "e_v_j", "e_i1_j", "e_i2_j", "e_i_j", "min_c_j", "vD1_min_j", "vD1_nonpos_j",
                    "roundtrip_j", "nfev_j", "nfev_rt_j", "non_ok_j", "backflow_enc_j", "landings_discarded_j"])
        for n in sorted(points):
            for pl in points[n].planes:
                w.writerow([n, pl.get("j"), pl.get("x1"), pl.get("e_v"), pl.get("e_i1"), pl.get("e_i2"),
                            "%.17g" % e_i_plane(pl), pl.get("min_c"), pl.get("vD1_min"), pl.get("vD1_nonpos"),
                            pl.get("roundtrip"), pl.get("nfev"), pl.get("nfev_rt"), pl.get("non_ok"),
                            pl.get("backflow_enc"), pl.get("landings_discarded")])


# ------------------------------------------------------------------------------------------
def selftest():
    exp = [32, 64, 128, 256]
    ok = True

    def chk(name, got, want):
        nonlocal ok
        good = got == want
        ok &= good
        print("selftest %-44s %s (got %s, want %s)" % (name, "ok" if good else "FAIL", got, want))

    h2 = {32: 1.0, 64: 0.25, 128: 0.0625, 256: 0.015625}   # order 2
    chk("accept (order 2, decreasing)", verdict_quantity(h2, list(h2), exp)[0], "accept")
    flat = {32: 1.0, 64: 0.5, 128: 0.2, 256: 0.19}           # change 5 % < 10 %, > 1e-12
    chk("reject (floor)", verdict_quantity(flat, list(flat), exp)[0], "reject")
    slow = {32: 1.0, 64: 0.5, 128: 0.25, 256: 0.15}          # order 0.74, change 40 %
    chk("unresolved (slow)", verdict_quantity(slow, list(slow), exp)[0], "unresolved")
    nondec = {32: 1.0, 64: 0.25, 128: 0.3, 256: 0.01}        # not monotone; last order large
    chk("unresolved (non-monotone)", verdict_quantity(nondec, list(nondec), exp)[0], "unresolved")
    tiny = {32: 1e-9, 64: 9.5e-10, 128: 9.5e-14, 256: 9.4e-14}  # change 1 % but 256 value below 1e-12, order 0.0
    chk("unresolved (floor below 1e-12)", verdict_quantity(tiny, list(tiny), exp)[0], "unresolved")
    chk("not_evaluable (missing N)", verdict_quantity({32: 1.0}, [32], exp)[0], "not_evaluable")
    chk("not_evaluable (nan)", verdict_quantity({32: 1, 64: float("nan"), 128: 1, 256: 1}, exp, exp)[0],
        "not_evaluable")
    chk("aggregate accept", aggregate(["accept"] * 5), "accept")
    chk("aggregate reject", aggregate(["accept", "reject", "unresolved", "accept", "accept"]), "reject")
    chk("aggregate unresolved", aggregate(["accept", "unresolved", "accept", "accept", "accept"]), "unresolved")
    chk("aggregate not_evaluable", aggregate(["accept", "not_evaluable", "accept", "accept", "accept"]),
        "not_evaluable")
    print("selftest %s" % ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", help="driver --summary JSON of the points of one amplitude (N<N>.json)")
    ap.add_argument("--eps", default="?", help="amplitude label for the report (0.5 | 1 | 1.5)")
    ap.add_argument("--out", help="markdown output path (also printed)")
    ap.add_argument("--planes-csv", help="CSV of every plane of every point (default: next to the first JSON)")
    ap.add_argument("--expected-ns", default="32,64,128,256", help="grid list the verdicts require (default 32,64,128,256)")
    ap.add_argument("--selftest", action="store_true", help="check the verdict logic on synthetic series and exit")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if not a.files:
        ap.error("no input JSON")
    expected = [int(x) for x in a.expected_ns.split(",") if x]
    points = {}
    for f in a.files:
        try:
            js = load_json(f)
        except (OSError, ValueError) as ex:
            print("WARNING: cannot read %s: %s" % (f, ex), file=sys.stderr)
            continue
        p = Point(f, js)
        points[p.N] = p
    if not points:
        print("ERROR: no readable input", file=sys.stderr)
        return 2
    csv_path = a.planes_csv
    if csv_path is None and any(p.planes for p in points.values()):
        first = a.files[0]
        csv_path = os.path.join(os.path.dirname(first), "planes_eps%s.csv" % a.eps)
    if csv_path and any(p.planes for p in points.values()):
        write_planes_csv(points, csv_path)
    else:
        csv_path = None
    text, vb, va = build_report(a.eps, points, expected, csv_path)
    write_out(text, a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
