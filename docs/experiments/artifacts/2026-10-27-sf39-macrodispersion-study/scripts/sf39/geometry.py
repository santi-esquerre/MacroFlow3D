"""SF-39 C-GEOM: arm table (preregistration section 1), plane lists (section 2), runner-parser mirror.

CLI (PYTHONPATH must contain the directory that holds the `sf39` package):
  python3 -m sf39.geometry flags <arm>     runner arguments of the arm, except --out, --seed0, --nr (supplied by run_sf39.sh)
  python3 -m sf39.geometry planes <P1..P7|arm>   the plane string (printed "%.17g", comma-joined, no spaces)
  python3 -m sf39.geometry check           the seven sha256 values of section 2 + every arm against the runner's parser; rc != 0 on failure
  python3 -m sf39.geometry arms            one arm id per line (production, ladders, width, checks)

Plane strings are generated here at run time and checked, never hand-edited.  The runner parser constraints that can fail
before any CUDA work (EnsembleRunner.cu parse_ensemble_args / find_seed_on_face) are mirrored in `validate`.
"""

import hashlib
import math
import sys
from collections import OrderedDict, namedtuple

DEFAULT_WINDOW = "0.1,0.9,0.1,0.9"
NP_SIDE = 100
TOL = "1e-8"

# list id -> sha256 of the ASCII plane string (preregistration section 2, normative)
PLANE_SHA256 = OrderedDict([
    ("P1", "0bf748196fff9c61acca5f9db5973eaf350f4364a8f3d313e66cf50b563340fa"),
    ("P2", "87a4811e81e0cbf31c4a7d459084663f58e14ec9e40236845c1204a367c0f5a0"),
    ("P3", "e7e54bb4e957ad293beb93bcb2bde8ab7162172e64c1e6f3e94e2b3125fed2eb"),
    ("P4", "b03effae1b1c6172d6a13d12efca504a93a2ec2939c1a9dde7690615d953059f"),
    ("P5", "592dee43813642071a4e92173b02be03c718afb66b8916e5c60a31dc7a90bcd1"),
    ("P6", "2f0d0048c93883df69448f5d59a0ed981f384d589261c6091848865da2e696a3"),
    ("P7", "fb48abd16a4fc39aa6217e99360357c869053d7b44acd645ba94c90b62b7ff83"),
])
PLANE_COUNT = {"P1": 197, "P2": 199, "P3": 95, "P4": 197, "P5": 199, "P6": 95, "P7": 95}

Arm = namedtuple("Arm", "id grid lh eps ell_str window hmax plist")

W12 = "0.36666666666666664,0.63333333333333336,0.36666666666666664,0.63333333333333336"
W24 = "0.3,0.7,0.3,0.7"


def _arm(aid, nx, ny, lh, eps, ell_str, plist, hmax="8", window=DEFAULT_WINDOW):
    return Arm(aid, (nx, ny, ny), lh, eps, ell_str, window, hmax, plist)


def _build():
    t = OrderedDict()
    rows = [
        # production A (hmax 8,16,32) / B (8) per amplitude
        _arm("s1_A", 2048, 256, 10, "1", "0.0390625", "P1", "8,16,32"),
        _arm("s1_B", 2048, 256, 10, "1", "0.0390625", "P1"),
        _arm("s1_n05", 1024, 64, 5, "1", "0.078125", "P4"),
        _arm("s1_n10", 2048, 128, 10, "1", "0.078125", "P4"),
        _arm("s1_n20", 4096, 256, 20, "1", "0.078125", "P4"),
        _arm("s1_tw", 1024, 128, 5, "1", "0.0390625", "P1"),
        _arm("s225_A", 3328, 384, 16, "1.5", "0.041666666666666664", "P2", "8,16,32"),
        _arm("s225_B", 3328, 384, 16, "1.5", "0.041666666666666664", "P2"),
        _arm("s225_n08", 1664, 64, 8, "1.5", "0.125", "P5"),
        _arm("s225_n16", 3328, 128, 16, "1.5", "0.125", "P5"),
        _arm("s225_n32", 6656, 256, 32, "1.5", "0.125", "P5"),
        _arm("s225_tw", 1664, 192, 8, "1.5", "0.041666666666666664", "P2"),
        _arm("s4_A", 3328, 384, 32, "2", "0.083333333333333329", "P3", "8,16,32"),
        _arm("s4_B", 3328, 384, 32, "2", "0.083333333333333329", "P3"),
        _arm("s4_n16", 1664, 64, 16, "2", "0.25", "P6"),
        _arm("s4_n32", 3328, 128, 32, "2", "0.25", "P6"),
        _arm("s4_n64", 6656, 256, 64, "2", "0.25", "P6"),
        _arm("s4_tw", 1664, 192, 16, "2", "0.083333333333333329", "P3"),
        _arm("s4_w24", 1664, 384, 16, "2", "0.041666666666666664", "P7", window=W24),
        _arm("s4_w12", 1664, 192, 16, "2", "0.083333333333333329", "P3", window=W12),
        # continuation of the halted s4_w12 (owner decision 2026-10-28): same settings, seeds 7215-7231; never a check arm
        _arm("s4_w12c", 1664, 192, 16, "2", "0.083333333333333329", "P3", window=W12),
    ]
    for a in rows:
        t[a.id] = a
    # check arms (never data): every narrow size, twin and width geometry, hmax 8; the s4_A grid with the multi-pass list
    # (chk_s4_A) and with the single pass (chk_s4_A1) for the byte check
    for a in rows:
        if a.id.endswith("_A") and a.id != "s4_A" or a.id.endswith("_B") or a.id == "s4_w12c":
            continue
        h = a.hmax if a.id == "s4_A" else "8"
        t["chk_" + a.id] = a._replace(id="chk_" + a.id, hmax=h)
    t["chk_s4_A1"] = t["s4_A"]._replace(id="chk_s4_A1", hmax="8")
    return t


ARMS = _build()
SOLVE_CHECK_ARMS = [k for k in ARMS if k.startswith("chk_") and k not in ("chk_s4_A", "chk_s4_A1", "chk_s4_B")]


def fmt17(x):
    return "%.17g" % x


def plane_values(nx, ny, lh):
    """Section 2 algorithm, verbatim (doubles)."""
    ell = lh / ny
    L1 = nx / ny
    x0 = 5.0 * ell
    L = L1 - 10.0 * ell
    Xa = x0 + L / 2.0
    Xb = L1 - 5.0 * ell
    mmax = (nx - 10 * lh) // lh
    cand = [(x0 + m * ell, False) for m in range(1, mmax + 1)]
    cand += [(x0 + L / 4.0, False), (Xa, True), (Xb, True)]
    cand.sort(key=lambda c: c[0])
    kept = []  # [value, canonical]
    for x, canon in cand:
        if kept and x - kept[-1][0] <= 1e-12 * max(1.0, abs(x)):
            if canon and not kept[-1][1]:
                kept[-1] = [x, True]  # keep Xa / Xb when involved in the merge
            continue
        kept.append([x, canon])
    return [k[0] for k in kept]


def plane_string(nx, ny, lh):
    return ",".join(fmt17(x) for x in plane_values(nx, ny, lh))


def plane_list_of(pid_or_arm):
    """-> (list id, plane string) for P1..P7 or an arm id."""
    if pid_or_arm in PLANE_SHA256:
        pid = pid_or_arm
        arm = next(a for a in ARMS.values() if a.plist == pid and not a.id.startswith("chk_"))
    else:
        arm = ARMS[pid_or_arm]
        pid = arm.plist
    return pid, plane_string(arm.grid[0], arm.grid[1], arm.lh)


def ell_of(arm):
    return arm.lh / arm.grid[1]


def flags(arm_id):
    a = ARMS[arm_id]
    nx, ny, nz = a.grid
    return ["--medium", "b", "--grid", "%d,%d,%d" % a.grid, "--sigma2", "1", "--eps", a.eps, "--ell", a.ell_str,
            "--window", a.window, "--np-side", str(NP_SIDE), "--planes", plane_string(nx, ny, a.lh),
            "--times", "auto", "--tol", TOL, "--hmax-div", a.hmax, "--fused-splines", "--single-launch",
            "--twin", "on", "--pollock-bridge", "on"]


def _round_half_away(v):
    return math.floor(v + 0.5) if v >= 0 else -math.floor(-v + 0.5)


def validate(arm_id):
    """Mirror of the runner's usage errors reachable before CUDA work; returns a list of violations (empty = accepted)."""
    a = ARMS[arm_id]
    nx, ny, nz = a.grid
    bad = []
    if min(a.grid) < 4:
        bad.append("grid extents must be >= 4")
    if ny != nz:
        bad.append("ny must equal nz")
    if nx % 2 or ny % 2:
        bad.append("grid extents must be even")
    ell = float(a.ell_str)
    if not ell > 0:
        bad.append("ell must be > 0")
    if abs(ell - ell_of(a)) > 0:
        bad.append("ell string %s != lambda/h / ny = %s" % (a.ell_str, fmt17(ell_of(a))))
    if fmt17(ell_of(a)) != a.ell_str:
        bad.append("ell string is not the %.17g of lambda/h / ny")
    L1 = nx / ny
    x0 = 5.0 * ell
    if not (0.0 <= x0 < L1):
        bad.append("x0 outside [0, L1)")
    if not L1 > 10.0 * ell:
        bad.append("L1 must exceed 10 ell")
    w = [float(s) for s in a.window.split(",")]
    if len(w) != 4 or not (w[0] < w[1]) or not (w[2] < w[3]):
        bad.append("window needs w2a < w2b and w3a < w3b")
    pl = [float(s) for s in plane_string(nx, ny, a.lh).split(",")]
    for i, p in enumerate(pl):
        if not p > x0 or (i > 0 and not p > pl[i - 1]):
            bad.append("planes must be strictly increasing and > x0 (index %d)" % i)
            break
    if not pl[-1] < L1:
        bad.append("last plane must be < L1")
    M = NP_SIDE
    for k, (wa, wb) in enumerate(((w[0], w[1]), (w[2], w[3]))):
        for m in range(M):
            x = wa + (wb - wa) * ((m + 0.5) / M)
            v = ny * x
            if not abs(v - _round_half_away(v)) > 1e-6:
                bad.append("RR12 seed on a transverse face: axis x%d m=%d n_T*x=%s" % (k + 2, m, fmt17(v)))
                break
    hs = a.hmax.split(",")
    if len(set(hs)) != len(hs) or not all(h.isdigit() and 1 <= int(h) <= 1000000 for h in hs):
        bad.append("hmax-div list invalid")
    return bad


def check():
    """Seven sha256 values, plane counts, identical strings within a list, every arm against the parser. -> list of failures."""
    fails = []
    for pid, want in PLANE_SHA256.items():
        members = [a for a in ARMS.values() if a.plist == pid]
        strings = set(plane_string(a.grid[0], a.grid[1], a.lh) for a in members)
        if len(strings) != 1:
            fails.append("%s: members give %d different strings" % (pid, len(strings)))
            continue
        s = next(iter(strings))
        got = hashlib.sha256(s.encode("ascii")).hexdigest()
        n = s.count(",") + 1
        ok = got == want and n == PLANE_COUNT[pid]
        print("%s %s P=%d sha256=%s %s" % (pid, ",".join(a.id for a in members if not a.id.startswith("chk_")), n, got,
                                           "OK" if ok else "MISMATCH (want %s, P=%d)" % (want, PLANE_COUNT[pid])))
        if not ok:
            fails.append("%s: sha256/count mismatch" % pid)
    for aid in ARMS:
        v = validate(aid)
        if v:
            fails.append("%s: %s" % (aid, "; ".join(v)))
    return fails


def main(argv):
    if len(argv) >= 2 and argv[0] == "flags" and argv[1] in ARMS:
        print(" ".join(flags(argv[1])))
        return 0
    if len(argv) >= 2 and argv[0] == "planes" and (argv[1] in PLANE_SHA256 or argv[1] in ARMS):
        sys.stdout.write(plane_list_of(argv[1])[1] + "\n")
        return 0
    if argv == ["arms"]:
        print("\n".join(ARMS))
        return 0
    if argv == ["check"]:
        fails = check()
        for f in fails:
            print("FAIL " + f)
        print("GEOMETRY_CHECK %s" % ("FAILED n=%d" % len(fails) if fails else "OK"))
        return 1 if fails else 0
    sys.stderr.write("usage: python3 -m sf39.geometry flags <arm> | planes <P1..P7|arm> | arms | check\n")
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
