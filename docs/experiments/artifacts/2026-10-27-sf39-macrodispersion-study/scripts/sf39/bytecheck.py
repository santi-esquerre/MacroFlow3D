"""SF-39 RR3(iii) / F12 byte check: one realization run with --hmax-div 8,16,32 (multi) vs --hmax-div 8 (single).

  python3 -m sf39.bytecheck <multi_out> <single_out>

Per realization directory r_* (the sets must agree), bytes of:
  root:        field.json, flow_stats.json, solver_b.json, pollock/*                      (multi root vs single root)
  pass files:  <multi>/hmax_8/{*.npy, moments_*.csv, exits_*.csv, track.json, twin/*}      vs the same names at the single root
  series:      <multi_out>/hmax_8/stats/*, stats_twin/*                                    vs <single_out>/stats/*, stats_twin/*
  status.json: key by key; each single-pass key is read at its multi-pass location:
               tracker_plane_pass, tracker_time_pass, Np, seed_backflow <- hmax_8/status.json ; pollock <- the root status.json
Excluded (differ by construction): DONE, timings.json.  Exit 0 iff identical (every difference is printed); 2 on usage error.
"""

import fnmatch
import json
import os
import sys

STATUS_PASS_KEYS = ("tracker_plane_pass", "tracker_time_pass", "Np", "seed_backflow")
STATUS_ROOT_KEYS = ("pollock",)
EXCLUDED = ("DONE", "timings.json")


class Checker:
    def __init__(self):
        self.n_ok = 0
        self.diffs = []

    def diff(self, msg):
        self.diffs.append(msg)
        print("DIFF " + msg)

    def ok(self):
        self.n_ok += 1

    def files(self, label, a, b):
        """Byte comparison of two files (either may be absent)."""
        ea, eb = os.path.isfile(a), os.path.isfile(b)
        if not (ea and eb):
            self.diff("%s: missing (%s: %s, %s: %s)" % (label, a, "present" if ea else "ABSENT", b, "present" if eb else "ABSENT"))
        elif _read(a) != _read(b):
            self.diff("%s: bytes differ (%s vs %s)" % (label, a, b))
        else:
            self.ok()

    def sets(self, label, da, db, patterns=None, recursive=False):
        """Compare the file sets of two directories (names filtered by patterns) and the bytes of each common file."""
        la, lb = _names(da, patterns, recursive), _names(db, patterns, recursive)
        if not os.path.isdir(da) or not os.path.isdir(db):
            self.diff("%s: directory missing (%s %s, %s %s)" % (label, da, os.path.isdir(da), db, os.path.isdir(db)))
            return
        if not la and not lb:
            self.diff("%s: no files in either directory (%s, %s)" % (label, da, db))
            return
        for n in sorted(set(la) ^ set(lb)):
            self.diff("%s: file set differs: %s only in %s" % (label, n, da if n in la else db))
        for n in sorted(set(la) & set(lb)):
            self.files("%s/%s" % (label, n), os.path.join(da, n), os.path.join(db, n))

    def status(self, label, multi_r, single_r):
        try:
            s = _json(os.path.join(single_r, "status.json"))
            mr = _json(os.path.join(multi_r, "status.json"))
            mp = _json(os.path.join(multi_r, "hmax_8", "status.json"))
        except (OSError, ValueError) as e:
            self.diff("%s: status.json unreadable: %s" % (label, e))
            return
        for k in sorted(s):
            if k in STATUS_PASS_KEYS:
                src, where = mp, "hmax_8/status.json"
            elif k in STATUS_ROOT_KEYS:
                src, where = mr, "status.json"
            else:
                self.diff("%s: single-pass key %r has no multi-pass mapping" % (label, k))
                continue
            if k not in src:
                self.diff("%s: key %r absent from multi-pass %s" % (label, k, where))
            elif src[k] != s[k]:
                self.diff("%s: key %r differs (multi %s: %r, single: %r)" % (label, k, where, src[k], s[k]))
            else:
                self.ok()
        for k in STATUS_PASS_KEYS + STATUS_ROOT_KEYS:
            if k not in s:
                self.diff("%s: key %r absent from the single-pass status.json" % (label, k))


def _read(p):
    with open(p, "rb") as f:
        return f.read()


def _json(p):
    with open(p, "rb") as f:
        return json.loads(f.read().decode("utf-8"))


def _names(d, patterns, recursive):
    out = []
    if not os.path.isdir(d):
        return out
    for root, dirs, files in os.walk(d):
        if not recursive:
            dirs[:] = []
        for fn in files:
            rel = os.path.relpath(os.path.join(root, fn), d)
            if os.path.basename(rel) in EXCLUDED:
                continue
            if patterns is None or any(fnmatch.fnmatchcase(rel, p) for p in patterns):
                out.append(rel)
    return sorted(out)


def realizations(out):
    d = os.path.join(out, "realizations")
    if not os.path.isdir(d):
        return []
    return sorted(x for x in os.listdir(d) if x.startswith("r_") and not x.endswith(".tmp") and os.path.isdir(os.path.join(d, x)))


def bytecheck(multi, single):
    c = Checker()
    rm, rs = realizations(multi), realizations(single)
    if not rm or rm != rs:
        c.diff("realization directories differ or are absent: multi %s, single %s" % (rm, rs))
    for r in sorted(set(rm) & set(rs)):
        M = os.path.join(multi, "realizations", r)
        S = os.path.join(single, "realizations", r)
        for f in ("field.json", "flow_stats.json", "solver_b.json"):
            c.files("%s/%s" % (r, f), os.path.join(M, f), os.path.join(S, f))
        c.sets("%s/pollock" % r, os.path.join(M, "pollock"), os.path.join(S, "pollock"), recursive=True)
        c.sets("%s/hmax_8 vs root" % r, os.path.join(M, "hmax_8"), S,
               patterns=["*.npy", "moments_*.csv", "exits_*.csv", "track.json"])
        c.sets("%s/hmax_8/twin vs root twin" % r, os.path.join(M, "hmax_8", "twin"), os.path.join(S, "twin"), recursive=True)
        c.status("%s/status.json" % r, M, S)
    c.sets("hmax_8/stats vs stats", os.path.join(multi, "hmax_8", "stats"), os.path.join(single, "stats"))
    c.sets("hmax_8/stats_twin vs stats_twin", os.path.join(multi, "hmax_8", "stats_twin"), os.path.join(single, "stats_twin"))
    return c


def main(argv):
    if len(argv) != 2 or not all(os.path.isdir(a) for a in argv):
        sys.stderr.write("usage: python3 -m sf39.bytecheck <multi_out> <single_out>\n")
        return 2
    c = bytecheck(*argv)
    if c.diffs:
        print("BYTECHECK DIFF n=%d (identical items: %d)" % (len(c.diffs), c.n_ok))
        return 1
    print("BYTECHECK IDENTICAL items=%d" % c.n_ok)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
