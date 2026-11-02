"""Admissible seed-lattice sizes M for the SF-43 planning grids and sources (SF-42 seed-lattice rule).

Usage (from the scripts root):  python3 -m seedlattice.plan [--out-dir DIR]   (no --out-dir: markdown to stdout;
with it: writes seedlattice_plan.md and seedlattice_plan.json)

Arms are derived from the SF-43 specification (in-scope items 2-4, 7, 9): box 104 lambda (208 for the distance arms), grid
nx x ny x ny = (L lh) x (W lh) x (W lh).  The source is a centred square of side S lambda per amplitude, the same in every arm
of that amplitude (decision 2026-10-29 G4): S = 19.2 at sigma^2 0.25, 1, 2.25 and 9.6 at 4, so the transverse window in box
fractions is (W - S)/(2W) .. (W + S)/(2W), written as the exact short decimal strings the runner will receive.  One M per amplitude
keeps seeds per lambda of source constant across its arms, so M is searched among those admissible on EVERY constraining arm.
Reference density: SF-39's M = 100 over 0.8 * (25.6 | 25.6 | 24 | 12) lambda at sigma^2 (0.25 | 1 | 2.25 | 4).
"""
import argparse
import json
import os
from collections import OrderedDict
from decimal import Decimal
from fractions import Fraction

from .rule import admissible_all, on_face

SOURCE = OrderedDict([("0.25", "19.2"), ("1", "19.2"), ("2.25", "19.2"), ("4", "9.6")])      # S in lambda
SF39_SOURCE = {"0.25": "20.48", "1": "20.48", "2.25": "19.2", "4": "9.6"}                     # 0.8 * (25.6, 25.6, 24, 12)
SF39_M = 100
PROD = {"0.25": (24, 10), "1": (24, 10), "2.25": (24, 16), "4": (12, 32)}                     # (width, lambda/h)
RANGE_FACTORS = (Fraction(1, 2), Fraction(5, 2))                                               # searched M in [0.5, 2.5] x M_ref
TARGET_FACTORS = (Fraction(3, 5), Fraction(1), Fraction(2))                                    # 0.6x, 1x, 2x of SF-39's density
FULL_PLANE = "0,1,0,1"


def window_string(W, S):
    """Centred window (W - S)/(2W) .. (W + S)/(2W) as exact short decimals (both axes); refuses inexact decimals."""
    W, S = Decimal(str(W)), Decimal(str(S))
    out = []
    for num in (W - S, W + S):
        q = Fraction(num) / Fraction(2 * W)
        s = format(num / (2 * W), "f")
        if "." in s:
            s = s.rstrip("0").rstrip(".")
        if Fraction(Decimal(s)) != q:
            raise ValueError("window fraction %s is not an exact short decimal" % q)
        out.append(s)
    return "%s,%s,%s,%s" % (out[0], out[1], out[0], out[1])


def build_arms():
    """List of dicts: amp, family, label, W, lh, L, nx, ny, window, constrains."""
    spec = []  # (amp, family, W, lh, L, constrains)

    def add(amp, family, W, lh, L=104, constrains=True):
        spec.append((amp, family, W, lh, L, constrains))

    for amp in ("0.25", "1"):
        W, lh = PROD[amp]
        add(amp, "production", W, lh)
        for l in (5, 20):
            add(amp, "E_h", W, l)
        add(amp, "E_W", 48, lh)
        add(amp, "gate5", W, lh)                    # 1040x240x240 (same grid as production)
    W, lh = PROD["2.25"]
    add("2.25", "production", W, lh)
    for l in (8, 4, 20):                            # 8/16 plus 4 or 20 (chosen later: both listed)
        add("2.25", "E_h", W, l)
    for w in (24, 48):
        for l in (8, 10):
            add("2.25", "E_W", w, l)
    add("2.25", "gate5", 24, 12)                    # 1248x288x288
    W, lh = PROD["4"]
    add("4", "production", W, lh)
    for l in (8, 16):
        add("4", "E_h", W, l)
    for w in (12, 24):
        for l in (8, 16):
            add("4", "E_W", w, l)
    add("4", "gate5", 12, 16)                       # 1664x192x192
    add("4", "pilot", 12, 24, constrains=False)     # SF-42 grid pilot 2496x288x288 (feasibility only)
    for amp in ("1", "2.25"):                       # distance arms: boxes of 208 lambda, width 24, production lambda/h
        add(amp, "distance", 24, PROD[amp][1], L=208)
    arms = []
    for amp, family, W, lh, L, constrains in spec:
        ny, nx = W * lh, L * lh
        arms.append(OrderedDict(amp=amp, family=family, W=W, lh=lh, L=L, nx=nx, ny=ny,
                                window=window_string(W, SOURCE[amp]), constrains=constrains,
                                label="%s %s %dx%dx%d" % (amp, family, nx, ny, ny)))
    return arms


def m_ref(amp):
    """SF-39 seeds per lambda of source times the new source side."""
    return Fraction(SF39_M) / Fraction(Decimal(SF39_SOURCE[amp])) * Fraction(Decimal(SOURCE[amp]))


def nearest(adm, target):
    """Admissible M nearest to target (ties: smaller)."""
    return min(adm, key=lambda m: (abs(m - target), m)) if adm else None


def compact(ms):
    out, i = [], 0
    while i < len(ms):
        j = i
        while j + 1 < len(ms) and ms[j + 1] == ms[j] + 1:
            j += 1
        out.append(str(ms[i]) if i == j else "%d-%d" % (ms[i], ms[j]))
        i = j + 1
    return ", ".join(out)


def variants(amp, con):
    """Constraining arm sets: at 2.25 the E_h third level is lh 4 OR lh 20 (chosen later), so both sets are evaluated."""
    if amp != "2.25":
        return OrderedDict([("all arms", con)])
    return OrderedDict([
        ("E_h third level lh=4 (lh=20 not run)", [a for a in con if not (a["family"] == "E_h" and a["lh"] == 20)]),
        ("E_h third level lh=20 (lh=4 not run)", [a for a in con if not (a["family"] == "E_h" and a["lh"] == 4)])])


def flag(adm, ref, S):
    out = OrderedDict()
    for f in TARGET_FACTORS:
        M = nearest(adm, f * ref)
        out["%gx" % float(f)] = None if M is None else OrderedDict(
            M=M, target=float(f * ref), seeds_per_lambda=float(M / S), density_vs_sf39=float(M / ref))
    return out


def per_amplitude(arms):
    res = OrderedDict()
    for amp in SOURCE:
        mine = [a for a in arms if a["amp"] == amp]
        con = [a for a in mine if a["constrains"]]
        ref = m_ref(amp)
        S = Fraction(Decimal(SOURCE[amp]))
        lo = -int(-RANGE_FACTORS[0] * ref // 1)     # ceil
        hi = int(RANGE_FACTORS[1] * ref // 1)       # floor
        rng = range(lo, hi + 1)
        vs = OrderedDict()
        for name, arm_set in variants(amp, con).items():
            groups = OrderedDict()                  # window -> sorted unique ny
            for a in arm_set:
                groups.setdefault(a["window"], set()).add(a["ny"])
            adm = [M for M in rng if not any(on_face(ny, w, M)[0] for w, nys in groups.items() for ny in nys)]
            full = admissible_all(sorted({a["ny"] for a in arm_set}), FULL_PLANE, rng)
            flagged = flag(adm, ref, S)
            pairs = [(w, ny) for w, nys in groups.items() for ny in sorted(nys)]
            ok = {pr: {M for M in rng if not on_face(pr[1], pr[0], M)[0]} for pr in pairs}
            diag = [OrderedDict(window=w, ny=ny, own_admissible=len(ok[(w, ny)]),
                                intersection_without_it=len(set.intersection(*[ok[q] for q in pairs if q != (w, ny)])))
                    for (w, ny) in pairs] if len(pairs) > 1 else []
            vs[name] = OrderedDict(conflict_diagnosis=diag,
                ny_by_window=OrderedDict((w, sorted(n)) for w, n in groups.items()),
                admissible_on_every_arm=adm, flagged=flagged, admissible_full_plane_same_grids=full,
                per_arm_check=[OrderedDict(
                    label=a["label"], window=a["window"],
                    flagged_passes={k: (v is not None and not on_face(a["ny"], a["window"], v["M"])[0])
                                    for k, v in flagged.items() if v is not None}) for a in arm_set])
        prod = next(a for a in con if a["family"] == "production")
        padm = [M for M in rng if not on_face(prod["ny"], prod["window"], M)[0]]
        res[amp] = OrderedDict(
            source_lambda=float(S), m_ref_sf39_density=float(ref), m_range=[lo, hi], variants=vs,
            production_arm_only=OrderedDict(label=prod["label"], admissible=padm, flagged=flag(padm, ref, S)))
    return res


def flag_rows(flagged):
    rows = ["| target | M | seeds/lambda | density vs SF-39 |", "|---|---|---|---|"]
    for k, v in flagged.items():
        rows.append("| %s SF-39 density | none admissible | | |" % k if v is None else
                    "| %s SF-39 density (M = %.1f) | **%d** | %.3f | %.3f |" % (
                        k, v["target"], v["M"], v["seeds_per_lambda"], v["density_vs_sf39"]))
    return rows


def markdown(arms, res):
    L = ["# Seed-lattice plan for the SF-43 planning grids (SF-42 seed-lattice rule)", "",
         "Generated by `python3 -m seedlattice.plan`; RR12 predicted by `seedlattice.rule` (verified against the production "
         "`find_seed_on_face` by `seedlattice_probe`). Window = centred source of side S lambda, box fractions as the runner "
         "receives them; one `M` per amplitude must be admissible on every constraining arm of that amplitude.", "",
         "## Arms", "", "| sigma^2 | family | grid | W (lambda) | lambda/h | window | constrains M |", "|---|---|---|---|---|---|---|"]
    for a in arms:
        L.append("| %s | %s | %dx%dx%d | %d | %d | `%s` | %s |" % (
            a["amp"], a["family"], a["nx"], a["ny"], a["ny"], a["W"], a["lh"], a["window"],
            "yes" if a["constrains"] else "no (pilot)"))
    for amp, r in res.items():
        L += ["", "## sigma^2 = %s" % amp, "",
              "SF-39 density gives M_ref = %.4g for the source S = %g lambda; searched M in [%d, %d]." % (
                  r["m_ref_sf39_density"], r["source_lambda"], r["m_range"][0], r["m_range"][1])]
        for name, v in r["variants"].items():
            L += ["", "### %s" % name, "",
                  "- windows and ny: " + "; ".join("`%s` ny in {%s}" % (w, ", ".join(map(str, n)))
                                                   for w, n in v["ny_by_window"].items()),
                  "- admissible on every arm (%d values): %s" % (len(v["admissible_on_every_arm"]),
                                                                 compact(v["admissible_on_every_arm"]) or "none"),
                  "- full plane `[0,1]^2` on the same grids (%d values; rule v2(M) >= v2(ny)): %s" % (
                      len(v["admissible_full_plane_same_grids"]), compact(v["admissible_full_plane_same_grids"]) or "none"),
                  ""] + flag_rows(v["flagged"])
            if len(v["admissible_on_every_arm"]) < 5:
                L += ["", "Conflict diagnosis (this variant admits fewer than 5 values): per (window, ny), the values it admits alone in the "
                      "range and the size of the intersection of all the others.", "",
                      "| window | ny | admitted alone | intersection without this pair |", "|---|---|---|---|"]
                L += ["| `%s` | %d | %d | %d |" % (d["window"], d["ny"], d["own_admissible"], d["intersection_without_it"])
                      for d in v["conflict_diagnosis"]]
        p = r["production_arm_only"]
        L += ["", "### Production arm alone (%s): for the single-configuration particle-count pilot" % p["label"], "",
              "- admissible (%d values): %s" % (len(p["admissible"]), compact(p["admissible"]) or "none"), ""] + flag_rows(p["flagged"])
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()
    arms = build_arms()
    res = per_amplitude(arms)
    md = markdown(arms, res)
    if a.out_dir:
        os.makedirs(a.out_dir, exist_ok=True)
        with open(os.path.join(a.out_dir, "seedlattice_plan.md"), "w") as f:
            f.write(md)
        with open(os.path.join(a.out_dir, "seedlattice_plan.json"), "w") as f:
            json.dump(OrderedDict(arms=arms, amplitudes=res), f, indent=1)
            f.write("\n")
    else:
        print(md)


if __name__ == "__main__":
    main()
