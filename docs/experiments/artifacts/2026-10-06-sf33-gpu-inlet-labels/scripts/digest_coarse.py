#!/usr/bin/env python3
"""SF-33 N8': per-log digest of the coarse correction and the linear solves of `inlet_slab` logs.

Per log: STAGE_END lines; GMRES its per Newton solve (LINEAR lines: count, total, max, median, linear time total);
COARSE build lines (K, count, t_assembly / t_lu / t_cond mean and max, rcond_est min, kl/ku); COARSE apply lines
(applications total, t_apply_avg and t_host_solve_avg mean); host max RSS from a sibling `<log stem>.time`
(`/usr/bin/time -v`) when present.  Text only, no interpretation.  Usage: digest_coarse.py LOG... [--out md]
"""
import argparse
import os
import re
import statistics as st


def kv(line):
    return dict(t.split("=", 1) for t in line.split() if "=" in t)


def fnum(s):
    return float(s[:-2]) if s.endswith("ms") else float(s.rstrip("s"))


def digest(path):
    lin, bld, app, ends = [], [], [], []
    with open(path) as f:
        for raw in f:
            s = raw.strip()
            if s.startswith("LINEAR "):
                d = kv(s)
                lin.append((int(d["its"]), fnum(d["t"]), d.get("status", "?")))
            elif s.startswith("COARSE build"):
                bld.append(kv(s))
            elif s.startswith("COARSE apply"):
                app.append(kv(s))
            elif s.startswith("STAGE_END"):
                ends.append(s)
    rss = None
    tf = os.path.splitext(path)[0] + ".time"
    if os.path.exists(tf):
        m = re.search(r"Maximum resident set size \(kbytes\): (\d+)", open(tf).read())
        rss = int(m.group(1)) if m else None
    L = [f"### {os.path.basename(os.path.dirname(path))}/{os.path.basename(path)}", ""]
    L += [f"- {e}" for e in ends]
    if lin:
        its = [x[0] for x in lin]
        L.append(f"- LINEAR solves={len(its)} its_total={sum(its)} its_max={max(its)} its_median={st.median(its):g} "
                 f"t_linear_total={sum(x[1] for x in lin):.2f}s non_converged="
                 f"{sum(1 for x in lin if x[2] != 'converged')}")
    if bld:
        def col(k):
            return [fnum(b[k]) for b in bld if k in b]
        ta, tl, tc = col("t_assembly"), col("t_lu"), col("t_cond")
        rc = [float(b["rcond_est"]) for b in bld if "rcond_est" in b]
        L.append(f"- COARSE build count={len(bld)} K={bld[0].get('K')} kl={bld[0].get('kl')} ku={bld[0].get('ku')} "
                 f"t_assembly mean={st.mean(ta):.3f}s max={max(ta):.3f}s | t_lu mean={st.mean(tl):.3f}s "
                 f"max={max(tl):.3f}s total={sum(tl):.1f}s | t_cond mean={st.mean(tc):.3f}s | rcond_est min={min(rc):.3e}")
    if app:
        n = sum(int(a["applications"]) for a in app)
        ta = [fnum(a["t_apply_avg"]) for a in app]
        th = [fnum(a["t_host_solve_avg"]) for a in app]
        L.append(f"- COARSE apply applications_total={n} t_apply_avg mean={st.mean(ta):.3f}ms max={max(ta):.3f}ms | "
                 f"t_host_solve_avg mean={st.mean(th):.3f}ms max={max(th):.3f}ms")
    if rss is not None:
        L.append(f"- host max RSS (/usr/bin/time -v) = {rss} kB = {rss / 1024 / 1024:.2f} GiB")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    text = "\n".join(digest(p) for p in a.logs)
    print(text)
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(text)


if __name__ == "__main__":
    main()
