#!/usr/bin/env python3
"""SF-40 criterion C instrument (RR3, C-MEMPHASE): checks the MEM_GEOM / MEM_PHASE lines of a lean-stage log.

  closed_form.py LOG [--run NAME] [--incomplete] [--extra-scratch name=bytes,...] [--probe-device-used BYTES]
  closed_form.py --selftest

Per phase line (F, S1, S2, S3, T):
  * exact_bytes == cufft_work + sum(items)                                           (identity)
  * every item equals its geometric size rule (C-MEMPHASE; n-arrays 8 n; grown face buffers `pcg_r`, `pcg_p`,
    `corrector`, `mg_l0_x`, `h_tilde` 8 m; `mg_l<l>_{b,r,coef}` 8 n_l; spectra 16 (nx/2+1) ny nz; scratch items with
    s(n_l) from `cub_sum_bytes`); items whose size is a capacity, not a geometric rule (`sf18_reduce_blocks`,
    `labels_*`, `tracker_*`) must be present-or-optional as listed and are only required to be non-negative;
  * every required item is present; any unknown item name -> FAIL naming it (`--extra-scratch` registers explicitly
    disclosed extra fixed-size `scratch_<name>` items, C-MEMPHASE);
  * S2 equals the S+ closed form  8 [7 n + 4 sum_{l>=1} n_l] + 32 (m - n) + S_scr  (+ 8 n if bhat=device), plus the
    reported cufft_work (0 for S2);
  * S2 is the largest phase.
Output lines (parsed by the campaign scripts):
  C_VERDICT run=<r> phase=<p> expected=<b> reported=<b> diff=<b> PASS|FAIL|NOT_EVALUABLE
  C_ITEM run= phase= item= expected= reported= FAIL        (failing items only)
  C_LARGEST run= S2= max_other= PASS|FAIL|NOT_EVALUABLE
  C_RUN run= PASS|FAIL|NOT_EVALUABLE
  C_INFO ... (rounded sums, device_used excess, halving consistency)
Exit status: 0 iff C_RUN is PASS; 1 FAIL; 3 NOT_EVALUABLE; 2 unreadable input.  Stdlib only; all integers are Python ints
(no 32-bit overflow).
"""
import re
import sys

PHASES = ["F", "S1", "S2", "S3", "T"]
MIB2 = 2 * 1024 * 1024


def parse_kv(line):
    """Leading tag removed; `key=value` tokens (value without spaces) -> dict of strings."""
    kv = {}
    for tok in line.split()[1:]:
        if "=" in tok:
            k, v = tok.split("=", 1)
            kv[k] = v
    return kv


def parse_items(s):
    out = []
    if s in ("", "-"):
        return out
    for part in s.split(","):
        name, _, val = part.rpartition(":")
        out.append((name, int(val)))
    return out


def ints(s):
    return [int(x) for x in s.split(",") if x != ""]


def parse(text):
    """-> (geom dict | None, {phase: [dict,...]})"""
    geom = None
    phases = {}
    for line in text.splitlines():
        if line.startswith("MEM_GEOM "):
            kv = parse_kv(line)
            geom = {
                "pid": kv.get("pid"),
                "nx": int(kv["nx"]), "ny": int(kv["ny"]), "nz": int(kv["nz"]), "n": int(kv["n"]),
                "levels": int(kv["levels"]), "level_cells": ints(kv["level_cells"]),
                "coarsest": kv.get("coarsest", ""), "face_max": int(kv["face_max"]),
                "bhat": kv.get("bhat", "mapped"), "bhat_host_bytes": int(kv.get("bhat_host_bytes", "0")),
                "cub": ints(kv["cub_sum_bytes"]),
            }
        elif line.startswith("MEM_PHASE "):
            kv = parse_kv(line)
            p = {
                "phase": kv["phase"], "exact": int(kv["exact_bytes"]), "cufft": int(kv.get("cufft_work", "0")),
                "device_used": int(kv["device_used"]) if "device_used" in kv else None,
                "seconds": float(kv["seconds"]) if "seconds" in kv else None,
                "items": parse_items(kv.get("items", "")),
            }
            phases.setdefault(p["phase"], []).append(p)
    return geom, phases


def item_rules(geom, phase, extra_scratch):
    """-> (required {name: size}, optional_free_prefixes, free_required {name}, alternatives)"""
    n, m = geom["n"], geom["face_max"]
    nx, ny, nz = geom["nx"], geom["ny"], geom["nz"]
    hs = (nx // 2 + 1) * ny * nz
    nl, s = geom["level_cells"], geom["cub"]
    req, free_req, free_prefix = {}, set(), []
    alt = None
    if phase == "F":
        req = {"y": 8 * n, "sf18_spectrum": 16 * hs, "sf18_mode_counter": 8}
        free_req = {"sf18_reduce_blocks"}
    elif phase == "S1":
        req = {"lnk": 8 * n, "coef_logk": 8 * n, "prefilter_spectrum": 16 * hs}
    elif phase == "S2":
        for l, c in enumerate(nl):
            req["mg_l%d_x" % l] = 8 * (m if l == 0 else c)
            for k in ("b", "r", "coef"):
                req["mg_l%d_%s" % (l, k)] = 8 * c
            req["scratch_mg_mean_zero_l%d" % l] = s[l] + 8
        req["pcg_r"] = req["pcg_p"] = req["corrector"] = 8 * m
        if geom["bhat"] == "device":
            req["bhat_device"] = 8 * n
        req["scratch_pcg_d_scalar"] = 8
        req["scratch_pcg_mean_zero"] = s[0] + 8
        req["scratch_rhs_mean_zero"] = s[0] + 8
        req["scratch_rhs_means"] = 32
        req["scratch_reduce_blocks"] = 2048
        for name, b in extra_scratch.items():
            req[name] = b
    elif phase == "S3":
        req = {"h_tilde": 8 * m, "coef_pot": 8 * n, "prefilter_spectrum": 16 * hs, "coef_logk": 8 * n}
        free_prefix = ["labels_"]
    elif phase == "T":
        alt = {"coef_pot": 8 * n, "coef_logk": 8 * n, "coef_interleaved": 16 * n}
        free_prefix = ["labels_", "tracker_"]
    return req, free_req, free_prefix, alt


def s2_closed_form(geom, extra_scratch):
    n, m = geom["n"], geom["face_max"]
    nl, s = geom["level_cells"], geom["cub"]
    scr = 8 + 2 * (s[0] + 8) + 32 + sum(x + 8 for x in s) + 2048 + sum(extra_scratch.values())
    cf = 8 * (7 * n + 4 * sum(nl[1:])) + 32 * (m - n) + scr
    if geom["bhat"] == "device":
        cf += 8 * n
    return cf, scr


def check_geom(geom):
    msgs = []
    nx, ny, nz = geom["nx"], geom["ny"], geom["nz"]
    if geom["n"] != nx * ny * nz:
        msgs.append("n=%d != nx*ny*nz=%d" % (geom["n"], nx * ny * nz))
    m = max((nx + 1) * ny * nz, nx * (ny + 1) * nz, nx * ny * (nz + 1))
    if geom["face_max"] != m:
        msgs.append("face_max=%d != max face count %d" % (geom["face_max"], m))
    if len(geom["level_cells"]) != geom["levels"] or len(geom["cub"]) != geom["levels"]:
        msgs.append("levels=%d but level_cells/cub_sum_bytes have %d/%d entries"
                    % (geom["levels"], len(geom["level_cells"]), len(geom["cub"])))
    if geom["level_cells"] and geom["level_cells"][0] != geom["n"]:
        msgs.append("level_cells[0]=%d != n" % geom["level_cells"][0])
    return msgs


def evaluate(text, run="run", complete=True, extra_scratch=None, probe_device_used=None):
    """-> (list of output lines, status 'PASS'|'FAIL'|'NOT_EVALUABLE')"""
    extra_scratch = extra_scratch or {}
    out = []
    geom, phases = parse(text)
    if geom is None:
        for p in PHASES:
            out.append("C_VERDICT run=%s phase=%s expected=- reported=- diff=- NOT_EVALUABLE" % (run, p))
        out.append("C_RUN run=%s NOT_EVALUABLE (no MEM_GEOM line)" % run)
        return out, "NOT_EVALUABLE"
    bad = False
    nev = False
    for msg in check_geom(geom):
        out.append("C_ITEM run=%s phase=GEOM item=%s expected=- reported=- FAIL" % (run, msg.replace(" ", "_")))
        bad = True
    nl = geom["level_cells"]
    # informational: halving consistency of the hierarchy
    if geom["nx"] % (1 << (geom["levels"] - 1)) == 0 and geom["ny"] % (1 << (geom["levels"] - 1)) == 0 \
            and geom["nz"] % (1 << (geom["levels"] - 1)) == 0:
        want = [(geom["nx"] >> l) * (geom["ny"] >> l) * (geom["nz"] >> l) for l in range(geom["levels"])]
        out.append("C_INFO run=%s level_cells_halving=%s" % (run, "consistent" if want == nl else "NOT_consistent"))
    exacts = {}
    first_dev = None
    for ph in PHASES:
        lst = phases.get(ph, [])
        if not lst:
            if complete:
                out.append("C_VERDICT run=%s phase=%s expected=- reported=- diff=- FAIL" % (run, ph))
                out.append("C_ITEM run=%s phase=%s item=phase_line_missing expected=- reported=- FAIL" % (run, ph))
                bad = True
            else:
                out.append("C_VERDICT run=%s phase=%s expected=- reported=- diff=- NOT_EVALUABLE" % (run, ph))
                nev = True
            continue
        if len(lst) > 1:
            out.append("C_ITEM run=%s phase=%s item=duplicate_phase_lines(%d) expected=1 reported=%d FAIL"
                       % (run, ph, len(lst), len(lst)))
            bad = True
        P = lst[-1]
        exacts[ph] = P["exact"]
        if first_dev is None and P["device_used"] is not None:
            first_dev = (ph, P["device_used"])
        req, free_req, free_prefix, alt = item_rules(geom, ph, extra_scratch)
        got = {}
        fail_phase = False
        expected_sum = P["cufft"]
        for name, b in P["items"]:
            if name in got:
                out.append("C_ITEM run=%s phase=%s item=%s expected=- reported=%d FAIL" % (run, ph, name + "_duplicate", b))
                fail_phase = True
                continue
            got[name] = b
        # known items
        for name, b in got.items():
            if name in req:
                exp = req[name]
            elif name in free_req:
                exp = b
                if b <= 0:
                    exp = 1
            elif alt is not None and name in alt:
                exp = alt[name]
            elif any(name.startswith(pr) and len(name) > len(pr) for pr in free_prefix):
                exp = b
            else:
                out.append("C_ITEM run=%s phase=%s item=%s(unknown) expected=- reported=%d FAIL" % (run, ph, name, b))
                fail_phase = True
                expected_sum += 0
                continue
            expected_sum += exp
            if b != exp:
                out.append("C_ITEM run=%s phase=%s item=%s expected=%d reported=%d FAIL" % (run, ph, name, exp, b))
                fail_phase = True
        for name, sz in req.items():
            if name not in got:
                out.append("C_ITEM run=%s phase=%s item=%s(missing) expected=%d reported=- FAIL" % (run, ph, name, sz))
                expected_sum += sz
                fail_phase = True
        for name in free_req:
            if name not in got:
                out.append("C_ITEM run=%s phase=%s item=%s(missing) expected=>0 reported=- FAIL" % (run, ph, name))
                fail_phase = True
        if alt is not None:
            if not ("coef_interleaved" in got or ("coef_pot" in got and "coef_logk" in got)):
                out.append("C_ITEM run=%s phase=T item=coef_pot+coef_logk_or_coef_interleaved(missing) "
                           "expected=%d reported=- FAIL" % (run, 16 * geom["n"]))
                expected_sum += 16 * geom["n"]
                fail_phase = True
        total_items = sum(got.values()) + P["cufft"]
        if P["exact"] != total_items:
            out.append("C_ITEM run=%s phase=%s item=exact_bytes==cufft_work+sum(items) expected=%d reported=%d FAIL"
                       % (run, ph, total_items, P["exact"]))
            fail_phase = True
        if ph == "S2":
            cf, scr = s2_closed_form(geom, extra_scratch)
            expected = cf + P["cufft"]
            out.append("C_INFO run=%s S2_closed_form=%d S_scr=%d cufft_work=%d bhat=%s"
                       % (run, cf, scr, P["cufft"], geom["bhat"]))
        else:
            expected = expected_sum
        diff = P["exact"] - expected
        if diff != 0:
            fail_phase = True
        out.append("C_VERDICT run=%s phase=%s expected=%d reported=%d diff=%d %s"
                   % (run, ph, expected, P["exact"], diff, "FAIL" if fail_phase else "PASS"))
        bad = bad or fail_phase
        rounded = sum(((b + MIB2 - 1) // MIB2) * MIB2 for b in got.values()) \
            + ((P["cufft"] + MIB2 - 1) // MIB2) * MIB2
        info = "C_INFO run=%s phase=%s exact=%d sum_rounded_2MiB_per_alloc=%d seconds=%s device_used=%s" % (
            run, ph, P["exact"], rounded, P["seconds"], P["device_used"])
        if P["device_used"] is not None and first_dev is not None:
            info += " dev_minus_first_phase_minus_exact=%d(first=%s)" % (
                P["device_used"] - first_dev[1] - P["exact"], first_dev[0])
        if P["device_used"] is not None and probe_device_used is not None:
            info += " dev_minus_probe_minus_exact=%d" % (P["device_used"] - probe_device_used - P["exact"])
        out.append(info)
    if "S2" in exacts:
        others = [v for k, v in exacts.items() if k != "S2"]
        mo = max(others) if others else 0
        ok = exacts["S2"] >= mo
        out.append("C_LARGEST run=%s S2=%d max_other=%d %s" % (run, exacts["S2"], mo, "PASS" if ok else "FAIL"))
        bad = bad or not ok
    else:
        out.append("C_LARGEST run=%s S2=- max_other=- NOT_EVALUABLE" % run)
        if complete:
            bad = True
        else:
            nev = True
    status = "FAIL" if bad else ("NOT_EVALUABLE" if nev else "PASS")
    out.append("C_RUN run=%s %s" % (run, status))
    return out, status


# ---------------------------------------------------------------------------------------------------------------
# selftest
# ---------------------------------------------------------------------------------------------------------------
def synth(nx, ny, nz, levels, bhat="mapped", mutate=None, drop_phase=None):
    n = nx * ny * nz
    m = max((nx + 1) * ny * nz, nx * (ny + 1) * nz, nx * ny * (nz + 1))
    nl = [(nx >> l) * (ny >> l) * (nz >> l) for l in range(levels)]
    s = [1024 + 16 * ((k + 255) // 256) for k in nl]
    hs = (nx // 2 + 1) * ny * nz
    geom = {"n": n, "face_max": m, "nx": nx, "ny": ny, "nz": nz, "level_cells": nl, "cub": s, "bhat": bhat}
    items = {
        "F": [("y", 8 * n), ("sf18_spectrum", 16 * hs), ("sf18_reduce_blocks", 8192), ("sf18_mode_counter", 8)],
        "S1": [("lnk", 8 * n), ("coef_logk", 8 * n), ("prefilter_spectrum", 16 * hs)],
        "S3": [("h_tilde", 8 * m), ("coef_pot", 8 * n), ("prefilter_spectrum", 16 * hs), ("coef_logk", 8 * n),
               ("labels_x", 5000)],
        "T": [("coef_pot", 8 * n), ("coef_logk", 8 * n), ("labels_x", 5000), ("tracker_pos", 650 * 100)],
    }
    s2 = []
    for l, c in enumerate(nl):
        s2.append(("mg_l%d_x" % l, 8 * (m if l == 0 else c)))
        for k in ("b", "r", "coef"):
            s2.append(("mg_l%d_%s" % (l, k), 8 * c))
        s2.append(("scratch_mg_mean_zero_l%d" % l, s[l] + 8))
    s2 += [("pcg_r", 8 * m), ("pcg_p", 8 * m), ("corrector", 8 * m)]
    if bhat == "device":
        s2.append(("bhat_device", 8 * n))
    s2 += [("scratch_pcg_d_scalar", 8), ("scratch_pcg_mean_zero", s[0] + 8), ("scratch_rhs_mean_zero", s[0] + 8),
           ("scratch_rhs_means", 32), ("scratch_reduce_blocks", 2048)]
    items["S2"] = s2
    cuf = {"F": 1000, "S1": 0, "S2": 0, "S3": 777, "T": 0}
    if mutate:
        mutate(items, cuf)
    lines = ["MEM_GEOM pid=4242 nx=%d ny=%d nz=%d n=%d levels=%d level_cells=%s coarsest=%dx%dx%d face_max=%d "
             "bhat=%s bhat_host_bytes=%d cub_sum_bytes=%s"
             % (nx, ny, nz, n, levels, ",".join(map(str, nl)), nx >> (levels - 1), ny >> (levels - 1),
                nz >> (levels - 1), m, bhat, 8 * n if bhat == "mapped" else 0, ",".join(map(str, s)))]
    dev = 500 * 2**20
    for ph in PHASES:
        if ph == drop_phase:
            continue
        it = items[ph]
        ex = cuf[ph] + sum(b for _, b in it)
        lines.append("MEM_PHASE phase=%s exact_bytes=%d cufft_work=%d device_used=%d seconds=1.5 items=%s"
                     % (ph, ex, cuf[ph], dev + ex + 123456, ",".join("%s:%d" % x for x in it)))
    return "\n".join(lines) + "\n", geom


def selftest():
    fails = []

    def expect(name, cond):
        if not cond:
            fails.append(name)
            print("SELFTEST FAIL %s" % name)

    # geometries: small cube (mapped and device), box, and the 3328x384x384 geometry (values > 2^32)
    for (nx, ny, nz, lv) in [(32, 32, 32, 3), (64, 16, 16, 3), (3328, 384, 384, 7)]:
        for bhat in ("mapped", "device"):
            txt, g = synth(nx, ny, nz, lv, bhat)
            lines, st = evaluate(txt, "r")
            expect("pass_%dx%dx%d_%s" % (nx, ny, nz, bhat), st == "PASS")
            if (nx, ny, nz) == (3328, 384, 384):
                expect("n_big", g["n"] == 490733568 and g["face_max"] == 3328 * 385 * 384)
                expect("gt32bit", any(int(re.search(r"reported=(\d+)", l).group(1)) > 2**32
                                      for l in lines if l.startswith("C_VERDICT")))
                s2 = [l for l in lines if l.startswith("C_VERDICT") and "phase=S2" in l][0]
                expect("s2_diff0", " diff=0 PASS" in s2)
    # failure cases
    base = (64, 16, 16, 3)

    def bad_size(items, cuf):
        items["S2"] = [(a, b + 8 if a == "corrector" else b) for a, b in items["S2"]]

    def missing_scr(items, cuf):
        items["S2"] = [x for x in items["S2"] if x[0] != "scratch_rhs_means"]

    def unknown(items, cuf):
        items["S2"].append(("mystery_buffer", 64))

    def not_largest(items, cuf):
        items["T"].append(("tracker_huge", 10**12))

    def ident(items, cuf):  # exact != cufft + sum(items): emulate by editing the line afterwards
        pass

    for name, mut, key in [("bad_size", bad_size, "item=corrector"), ("missing_scratch", missing_scr, "scratch_rhs_means(missing)"),
                           ("unknown", unknown, "mystery_buffer(unknown)"), ("s2_not_largest", not_largest, "C_LARGEST")]:
        txt, _ = synth(*base, mutate=mut)
        lines, st = evaluate(txt, "r")
        expect("fail_" + name, st == "FAIL" and any(key in l for l in lines))
    # identity violation
    txt, _ = synth(*base)
    txt2 = re.sub(r"(phase=S1 exact_bytes=)(\d+)", lambda m: m.group(1) + str(int(m.group(2)) + 8), txt)
    lines, st = evaluate(txt2, "r")
    expect("fail_identity", st == "FAIL" and any("exact_bytes==cufft_work+sum(items)" in l for l in lines))
    # wrong m (face_max inconsistent)
    txt3 = re.sub(r"face_max=\d+", "face_max=5", txt)
    lines, st = evaluate(txt3, "r")
    expect("fail_face_max", st == "FAIL")
    # --extra-scratch registered
    def extra(items, cuf):
        items["S2"].append(("scratch_foo", 40))
    txt4, _ = synth(*base, mutate=extra)
    expect("extra_unknown", evaluate(txt4, "r")[1] == "FAIL")
    expect("extra_registered", evaluate(txt4, "r", extra_scratch={"scratch_foo": 40})[1] == "PASS")
    # truncated run (OOM in S3): not evaluable when incomplete, FAIL when it claims completion
    txt5, _ = synth(*base, drop_phase="S3")
    txt5 = "\n".join(l for l in txt5.splitlines() if not l.startswith("MEM_PHASE phase=T")) + "\n"
    expect("oom_nev", evaluate(txt5, "r", complete=False)[1] == "NOT_EVALUABLE")
    expect("oom_complete_fail", evaluate(txt5, "r", complete=True)[1] == "FAIL")
    expect("no_geom", evaluate("nothing\n", "r")[1] == "NOT_EVALUABLE")
    # missing required T alternative
    def noT(items, cuf):
        items["T"] = [("labels_x", 1)]
    expect("T_missing_coef", evaluate(synth(*base, mutate=noT)[0], "r")[1] == "FAIL")
    def tinter(items, cuf):
        n = 64 * 16 * 16
        items["T"] = [("coef_interleaved", 16 * n), ("tracker_a", 10)]
    expect("T_interleaved", evaluate(synth(*base, mutate=tinter)[0], "r")[1] == "PASS")
    # probe-relative excess line is produced
    lines, _ = evaluate(synth(*base)[0], "r", probe_device_used=500 * 2**20)
    expect("probe_info", any("dev_minus_probe_minus_exact=123456" in l for l in lines))
    if fails:
        print("SELFTEST FAILED: %s" % ", ".join(fails))
        return 1
    print("SELFTEST OK closed_form")
    return 0


def main(argv):
    if len(argv) >= 2 and argv[1] == "--selftest":
        return selftest()
    if len(argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    path, run, complete, extra, probe = argv[1], "run", True, {}, None
    i = 2
    while i < len(argv):
        a = argv[i]
        if a == "--run" and i + 1 < len(argv):
            run = argv[i + 1]
            i += 2
        elif a == "--incomplete":
            complete = False
            i += 1
        elif a == "--extra-scratch" and i + 1 < len(argv):
            for part in argv[i + 1].split(","):
                k, _, v = part.partition("=")
                extra[k] = int(v)
            i += 2
        elif a == "--probe-device-used" and i + 1 < len(argv):
            probe = int(argv[i + 1])
            i += 2
        else:
            print("closed_form.py: bad argument %r" % a, file=sys.stderr)
            return 2
    try:
        with open(path, errors="replace") as f:
            text = f.read()
    except OSError as e:
        print("UNREADABLE %s" % e)
        return 2
    lines, st = evaluate(text, run, complete, extra, probe)
    print("\n".join(lines))
    return {"PASS": 0, "FAIL": 1, "NOT_EVALUABLE": 3}[st]


if __name__ == "__main__":
    sys.exit(main(sys.argv))
