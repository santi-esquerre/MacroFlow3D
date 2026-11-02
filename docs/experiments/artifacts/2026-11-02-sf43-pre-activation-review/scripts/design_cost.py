"""SF-43 pre-activation review: GPU-h, two-stage schedule and deadline fit of the candidate designs.

Inputs are the planning seconds per realization of the SF-42 cost audit (own time on the V100, measured where the grid, sigma^2 and M
were measured; SF-42 note section G and `pilots/`; values copied here, see `README.md`). No GPU, no data; numpy only.

Usage: python3 design_cost.py            # prints the tables of section 4 of the review
"""
import math

# (seconds per realization, planning) by (grid, sigma^2, M); 'h3' = three h_max passes in one realization
SEC = {
    ("1040x240", "0.25"): 26.4, ("1040x240", "1"): 26.8,
    ("520x120", "0.25"): 12.0, ("520x120", "1"): 11.5,
    ("2080x480", "0.25"): 138.1, ("2080x480", "1"): 142.0,
    ("1040x480", "0.25"): 79.3, ("1040x480", "1"): 81.5,
    ("h3_1040x240", "0.25"): 72.1, ("h3_1040x240", "1"): 72.5,
    ("par2_1040x240", "0.25"): 12.8, ("par2_1040x240", "1"): 13.8,
    ("g7_cell_mean", "0.25"): 15.7, ("g7_cell_mean", "1"): 16.3,
    ("2080x240", "1"): 54.0,
    # sigma^2 2.25 at M = 100 (production measured at 100; other arms measured at 86/100, scaled by M^2 tracing where needed)
    ("1664x384", "2.25"): 54.0, ("416x96", "2.25"): 5.1, ("832x192", "2.25"): 14.5, ("832x384", "2.25"): 30.0,
    ("1040x240", "2.25"): 29.0, ("1040x480", "2.25"): 85.1, ("h3_1664x384", "2.25"): 98.6,
    ("1248x288", "2.25"): 37.5, ("par2_1248x288", "2.25"): 19.2, ("3328x384_L208", "2.25"): 102.1,
    # sigma^2 4 at M = 100
    ("3328x384", "4"): 131.5, ("832x96", "4"): 9.7, ("1664x192", "4"): 26.9, ("416x48", "4"): 4.0,
    ("832x192", "4"): 15.1, ("1664x384", "4"): 49.7, ("h3_3328x384", "4"): 220.8, ("par2_1664x192", "4"): 17.5,
    ("3328x192_L208", "4"): 52.0,
}
# memory class from the own-PID peaks of SF-42 (MiB): near-empty GPU (>= ~28 GB) / half (~14-16 GB) / small
NEAR_EMPTY = {"2080x480", "3328x384", "h3_3328x384", "3328x384_L208"}
HALF = {"1040x480", "1664x384", "h3_1664x384"}


def arm(fam, amp, grid, n, prio):
    return dict(fam=fam, amp=amp, grid=grid, n=n, prio=prio, sec=SEC[(grid, amp)],
                heavy=(grid in NEAR_EMPTY or grid in HALF), near_empty=grid in NEAR_EMPTY)


def design(prod=(150, 150, 150, 150), eh=100, hmax=30, ew=150, g5=(150, 150, 150, 250), g7=60, dist=100, d4=0):
    """Arms of a design; shared realizations (CRN reuse) are counted once (see the review, section 4)."""
    P025, P1, P225, P4 = prod
    A = []
    for amp, p, gn in (("0.25", P025, g5[0]), ("1", P1, g5[1])):
        A += [arm("production", amp, "1040x240", max(p, gn), 1),          # Gate-5 consumer twin = production seeds
              arm("E_h", amp, "520x120", eh, 1), arm("E_h", amp, "2080x480", eh, 1),
              arm("h_max", amp, "h3_1040x240", hmax, 1),
              arm("E_W", amp, "1040x480", ew, 1),
              arm("gate5_par2", amp, "par2_1040x240", gn, 1)]
        if g7:
            A.append(arm("g7_par2", amp, "g7_cell_mean", 8 * g7 - g7, 3))  # one cell is the Gate-5 Par2 arm
    A += [arm("distance", "1", "2080x240", dist, 3)]
    amp = "2.25"
    A += [arm("production", amp, "1664x384", P225, 1),
          arm("E_h", amp, "416x96", eh, 1), arm("E_h", amp, "832x192", max(eh, ew), 1),  # (24, 8) = E_h rung = E_W cell
          arm("h_max", amp, "h3_1664x384", hmax, 1),
          arm("E_W", amp, "832x384", ew, 1), arm("E_W", amp, "1040x240", ew, 1), arm("E_W", amp, "1040x480", ew, 1),
          arm("gate5_consumer", amp, "1248x288", g5[2], 1), arm("gate5_par2", amp, "par2_1248x288", g5[2], 1),
          arm("distance", amp, "3328x384_L208", dist, 3)]
    amp = "4"
    A += [arm("production", amp, "3328x384", P4, 1),
          arm("E_h", amp, "832x96", max(eh, ew), 1),                      # (12, 8) cell shared
          arm("E_h", amp, "1664x192", max(eh, ew, g5[3]), 1),              # (12, 16) cell and Gate-5 consumer shared
          arm("E_h_diag", amp, "416x48", eh, 2),
          arm("h_max", amp, "h3_3328x384", hmax, 1),
          arm("E_W", amp, "832x192", ew, 1), arm("E_W", amp, "1664x384", ew, 1),
          arm("gate5_par2", amp, "par2_1664x192", g5[3], 1)]
    if d4:
        A.append(arm("distance", amp, "3328x192_L208", d4, 3))
    return A


def gpu_h(A):
    return sum(a["n"] * a["sec"] for a in A) / 3600.0


def two_stage(A, margin=1.0, cap_h=30.0):
    """Stage 1: two lanes on the heavy queue (longest first, nothing ends after the cap); stage 2: one lane."""
    heavy = sorted((a["sec"] * margin for a in A if a["heavy"] for _ in range(a["n"])), reverse=True)
    light = sum(a["sec"] * margin * a["n"] for a in A if not a["heavy"])
    lanes, left, cap = [0.0, 0.0], 0.0, cap_h * 3600
    for d in heavy:
        k = 0 if lanes[0] <= lanes[1] else 1
        if lanes[k] + d <= cap:
            lanes[k] += d
        else:
            left += d
    s1 = max(lanes)
    s2 = left + light
    return dict(s1_h=s1 / 3600, s1_gpuh=2 * s1 / 3600, s2_h=s2 / 3600, gpuh=(2 * s1 + s2) / 3600, wall_h=(s1 + s2) / 3600)


VARIANTS = {
    "A recommended": design(),
    "B conservative": design(prod=(100, 100, 100, 100), eh=80, hmax=20, ew=100, g5=(150, 150, 150, 205), g7=30, dist=60),
    "C ambitious": design(prod=(300, 300, 300, 200), eh=150, hmax=40, ew=300, g5=(300, 300, 200, 300), g7=60, dist=150, d4=100),
}

if __name__ == "__main__":
    print("variant | GPU-h planning | +5% reserve | heavy GPU-h | stage-1 h / GPU-h (x1.0, x1.1, x1.32) | stage-2 h (x1.0, x1.1, x1.32) | wall h (x1.0, x1.1, x1.32)")
    for name, A in VARIANTS.items():
        g = gpu_h(A)
        hv = sum(a["n"] * a["sec"] for a in A if a["heavy"]) / 3600
        r = [two_stage(A, m) for m in (1.0, 1.101, 1.32)]
        print("%s | %.1f | %.1f | %.1f | %s | %s | %s" % (
            name, g, 1.05 * g, hv,
            ", ".join("%.1f/%.1f" % (x["s1_h"], x["s1_gpuh"]) for x in r),
            ", ".join("%.1f" % x["s2_h"] for x in r),
            ", ".join("%.1f" % x["wall_h"] for x in r)))
    print()
    print("per family (variant A), GPU-h planning:")
    fam = {}
    for a in VARIANTS["A recommended"]:
        fam[a["fam"]] = fam.get(a["fam"], 0.0) + a["n"] * a["sec"] / 3600
    for k, v in sorted(fam.items(), key=lambda kv: -kv[1]):
        print("  %-15s %.2f" % (k, v))
