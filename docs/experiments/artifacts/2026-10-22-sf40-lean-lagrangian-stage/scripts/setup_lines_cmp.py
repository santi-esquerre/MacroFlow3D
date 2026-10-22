#!/usr/bin/env python3
"""SF-40 criterion A helper (RR4, LEAN set): compare driver-log lines selected by LITERAL LINE PREFIX.

  setup_lines_cmp.py NEW.log REF.log PREFIX [PREFIX ...]
  setup_lines_cmp.py --selftest

Selected lines: those that begin with one of the PREFIXes (character-exact, whitespace included; the SETUP prefixes are
`SETUP   SF-18:`, `SETUP   SF-19:` and `SETUP   SF-19 corrector`, three spaces after SETUP, from
`out_block("SETUP ", summary())` with the summary's two-space indent).  The same selection is applied to both logs, file
order is preserved, and the only masked fields are the timing fields of lines_cmp.py (imported, not edited).  Every
PREFIX must select at least one line in both logs.  Prints `IDENTICAL <n_lines> (per-prefix counts ...)` or
`DIFF <n> (n_new= n_ref=)` followed by up to 5 differing pairs; exit 0 IDENTICAL, 1 DIFF, 2 unreadable input.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lines_cmp import mask  # noqa: E402  (verbatim SF-37 module, unmodified)


def select(path, prefixes):
    out, counts = [], {p: 0 for p in prefixes}
    with open(path, errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            for p in prefixes:
                if line.startswith(p):
                    out.append(mask(line))
                    counts[p] += 1
                    break
    return out, counts


def compare(new_path, ref_path, prefixes):
    new, cn = select(new_path, prefixes)
    ref, cr = select(ref_path, prefixes)
    diffs = []
    for i in range(max(len(new), len(ref))):
        a = new[i] if i < len(new) else "<missing>"
        b = ref[i] if i < len(ref) else "<missing>"
        if a != b:
            diffs.append((i + 1, a, b))
    empty = [p for p in prefixes if cn[p] == 0 or cr[p] == 0]
    if not diffs and not empty:
        return 0, "IDENTICAL %d (%s)" % (len(new), ", ".join("%r x%d" % (p, cn[p]) for p in prefixes))
    msg = "DIFF %d (n_new=%d n_ref=%d)" % (len(diffs), len(new), len(ref))
    if empty:
        msg += " prefixes with no line in one log: %s" % ", ".join(repr(p) for p in empty)
    for i, a, b in diffs[:5]:
        msg += "\n  line %d\n    new: %s\n    ref: %s" % (i, a[:300], b[:300])
    return 1, msg


def selftest():
    import tempfile
    P = ["SETUP   SF-18:", "SETUP   SF-19:", "SETUP   SF-19 corrector"]
    ref = [
        "SETUP STAGE field=sf18 eps=1 N=64 status=ok",
        "SETUP   SF-18: raw_mean=1.0e-3 raw_variance=1.0 applied_scale=1.0 final_variance=1.0 active_modes=3",
        "SETUP   SF-19: mg_levels=4 converged=1 G=(1.0, 1.0, 1.0) mean_flux=(1,0,0) div_max=1e-12 div_rms=1e-13",
        "SETUP   SF-19 corrector x: status=converged its=12 rel_residual=1.0e-11 final_residual=1.0e-11",
        "SETUP   SF-19 corrector y: status=converged its=13 rel_residual=1.0e-11 final_residual=1.0e-11",
        "SETUP   SF-19 corrector z: status=converged its=14 rel_residual=1.0e-11 final_residual=1.0e-11",
        "SETUP   inlet: vmin=1 v_rms=2",
        "SETUP stage eps=1 built in 1.234s (grad_lnk=spline)",
    ]
    new = [ref[1], "MEM_PHASE phase=F ...", ref[2], ref[3], ref[4], ref[5], "SETUP   inlet: vmin=n/a v_rms=n/a",
           "SETUP stage eps=1 built in 9.9s (grad_lnk=spline)"]
    fails = []
    with tempfile.TemporaryDirectory() as d:
        def w(name, lines):
            p = os.path.join(d, name)
            with open(p, "w") as f:
                f.write("\n".join(lines) + "\n")
            return p
        rp, np_ = w("ref.log", ref), w("new.log", new)
        rc, msg = compare(np_, rp, P)
        if rc != 0 or not msg.startswith("IDENTICAL 5"):
            fails.append("identical:" + msg)
        bad = list(new)
        bad[3] = bad[3].replace("its=12", "its=11")
        rc, msg = compare(w("bad.log", bad), rp, P)
        if rc != 1 or "its=11" not in msg:
            fails.append("diff_value:" + msg)
        rc, msg = compare(w("short.log", new[:3]), rp, P)
        if rc != 1:
            fails.append("diff_missing:" + msg)
        rc, msg = compare(w("reorder.log", [new[2], new[1], new[0]] + new[3:]), rp, P)
        if rc != 1:
            fails.append("diff_order:" + msg)
        sp = list(new)
        sp[0] = sp[0].replace("SETUP   SF-18:", "SETUP  SF-18:")  # whitespace matters
        rc, msg = compare(w("ws.log", sp), rp, P)
        if rc != 1:
            fails.append("diff_whitespace:" + msg)
        t = list(new)
        t[0] = ref[1]
        t.append("SETUP   SF-19: mg_levels=4 converged=1 G=(1.0, 1.0, 1.0) mean_flux=(1,0,0) div_max=1e-12 div_rms=1e-13")
        rc, msg = compare(w("extra.log", t), rp, P)
        if rc != 1:
            fails.append("diff_extra:" + msg)
        rc, msg = compare(rp, rp, ["SETUP   NOPE"])
        if rc != 1:
            fails.append("empty_prefix:" + msg)
        # timing masks come from lines_cmp (built in <t>s, t=)
        ma = w("ma.log", ["SETUP   SF-19: t=1.5s x"])
        mb = w("mb.log", ["SETUP   SF-19: t=2.5s x"])
        rc, msg = compare(ma, mb, ["SETUP   SF-19:"])
        if rc != 0:
            fails.append("mask:" + msg)
    if fails:
        for f in fails:
            print("SELFTEST FAIL " + f)
        print("SELFTEST FAILED")
        return 1
    print("SELFTEST OK setup_lines_cmp")
    return 0


def main(argv):
    if len(argv) >= 2 and argv[1] == "--selftest":
        return selftest()
    if len(argv) < 4:
        print("usage: setup_lines_cmp.py NEW.log REF.log PREFIX [PREFIX ...]", file=sys.stderr)
        return 2
    try:
        rc, msg = compare(argv[1], argv[2], argv[3:])
    except OSError as e:
        print("UNREADABLE %s" % e)
        return 2
    print(msg)
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv))
