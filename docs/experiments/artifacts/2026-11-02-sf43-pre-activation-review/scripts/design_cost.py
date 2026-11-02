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
    # extra passes of a 3-pass h_max realization (3-pass minus 1-pass), on production seeds
    ("h3x_1040x240", "0.25"): 72.1 - 26.4, ("h3x_1040x240", "1"): 72.5 - 26.8, ("h3x_1664x384", "2.25"): 98.6 - 54.0,
    ("h3x_3328x384", "4"): 220.8 - 131.5,
    # (48, 12) width cell at 2.25: not measured (4.14e8 cells, ~25 GB): scaled from 1040x480x480 by cells, x1.5 for safety
    ("1248x576", "2.25"): 85.1 * (4.14 / 2.40) * 1.0,
    # positive control: sigma^2 = 0.04 on 1040x240x240 (cost of the sigma^2 0.25 production grid)
    ("1040x240_pc", "0.04"): 26.4,
}
# memory class from the own-PID peaks of SF-42 (MiB): near-empty GPU (>= ~28 GB) / half (~14-16 GB) / small
NEAR_EMPTY = {"2080x480", "3328x384", "h3x_3328x384", "3328x384_L208", "1248x576"}
HALF = {"1040x480", "1664x384", "h3x_1664x384"}


def arm(fam, amp, grid, n, prio):
    return dict(fam=fam, amp=amp, grid=grid, n=n, prio=prio, sec=SEC[(grid, amp)],
                heavy=(grid in NEAR_EMPTY or grid in HALF), near_empty=grid in NEAR_EMPTY)


def design(prod=(200, 200, 200, 200), eh=100, hmax=30, ew=150, g5=(200, 200, 200, 250), g7=60, dist=100, d4=100, pc=30):
    """Arms of a design (review section 4, decision H9); shared realizations are counted once:
    - the h_max passes run on the first `hmax` production seeds (`--hmax-div 8,16,32`), so only the extra passes are added;
    - the Gate-5 consumer twin at 0.25 and 1 is the production arm (uniform weights of the same runs);
    - at 2.25 the (24, 8) width cell is the lambda/h 8 rung and the (24, 12) cell is the Gate-5 consumer arm (production seeds);
    - at 4 the (12, 8) and (12, 16) width cells are the ladder rungs, the (12, 16) rung is also the Gate-5 consumer arm;
    - G7 runs all 8 cells of the half-fraction (the matched Gate-5 configuration is outside it)."""
    P025, P1, P225, P4 = prod
    A = []
    for amp, p, gn in (("0.25", P025, g5[0]), ("1", P1, g5[1])):
        A += [arm("production", amp, "1040x240", max(p, gn), 1),
              arm("h_max_extra", amp, "h3x_1040x240", hmax, 1),
              arm("E_h", amp, "520x120", eh, 1), arm("E_h", amp, "2080x480", eh, 1),
              arm("E_W", amp, "1040x480", ew, 1),
              arm("gate5_par2", amp, "par2_1040x240", gn, 1)]
        if g7:
            A.append(arm("g7_par2", amp, "g7_cell_mean", 8 * g7, 3))
    A += [arm("distance", "1", "2080x240", dist, 3)]
    if pc:
        A.append(arm("positive_control", "0.04", "1040x240_pc", pc, 2))
    amp = "2.25"
    A += [arm("production", amp, "1664x384", P225, 1),
          arm("h_max_extra", amp, "h3x_1664x384", hmax, 1),
          arm("E_h", amp, "416x96", eh, 1), arm("E_h", amp, "832x192", max(eh, ew), 1),
          arm("E_W", amp, "832x384", ew, 1), arm("E_W", amp, "1248x576", ew, 1),
          arm("gate5_consumer", amp, "1248x288", max(g5[2], ew), 1), arm("gate5_par2", amp, "par2_1248x288", g5[2], 1),
          arm("distance", amp, "3328x384_L208", dist, 3)]
    amp = "4"
    A += [arm("production", amp, "3328x384", P4, 1),
          arm("h_max_extra", amp, "h3x_3328x384", hmax, 1),
          arm("E_h", amp, "832x96", max(eh, ew), 1),
          arm("E_h", amp, "1664x192", max(eh, ew, g5[3]), 1),
          arm("E_h", amp, "416x48", eh, 1),
          arm("E_W", amp, "832x192", ew, 1), arm("E_W", amp, "1664x384", ew, 1),
          arm("gate5_par2", amp, "par2_1664x192", g5[3], 1)]
    if d4:
        A.append(arm("distance", amp, "3328x192_L208", d4, 3))
    return A


def gpu_h(A):
    return sum(a["n"] * a["sec"] for a in A) / 3600.0


def two_stage(A, margin=1.0, cap_h=30.0):
    """Lane A (first GPU, >= 30000 MiB free) runs the near-empty-GPU grids, then whatever remains; lane B (second GPU, >= 17000 MiB)
    runs only half-class heavy grids (longest first, balanced against lane A's heavy load) and releases its GPU as soon as no
    half-class unit remains or at the 30-h cap (decision H14). Returns lane-B hours (= two-GPU stage), wall and GPU-h."""
    near = sum(a["sec"] * margin * a["n"] for a in A if a["near_empty"])
    half = sorted((a["sec"] * margin for a in A if a["heavy"] and not a["near_empty"] for _ in range(a["n"])), reverse=True)
    light = sum(a["sec"] * margin * a["n"] for a in A if not a["heavy"])
    la, lb, cap = near, 0.0, cap_h * 3600
    for d in half:
        if lb + d <= min(la, cap) or lb + d <= la:
            lb += d
        elif la <= lb and lb + d > cap:
            la += d
        else:
            if lb + d <= cap and lb <= la:
                lb += d
            else:
                la += d
    s1 = min(lb, cap)                       # two GPUs held while lane B works
    wall = max(la, lb) + light              # lane A finishes its heavy load, then all light units (lane B gone)
    return dict(s1_h=s1 / 3600, s1_gpuh=2 * s1 / 3600, s2_h=(wall - s1) / 3600, gpuh=(s1 + wall) / 3600, wall_h=wall / 3600)


VARIANTS = {
    "A recommended": design(),
    "B conservative": design(prod=(200, 200, 200, 150), eh=60, hmax=20, ew=100, g5=(200, 200, 200, 250), g7=30, dist=60, d4=0),
    "C ambitious": design(prod=(300, 300, 300, 250), eh=150, hmax=40, ew=300, g5=(300, 300, 300, 300), g7=60, dist=150, d4=150),
}

if __name__ == "__main__":
    print("variant | GPU-h planning | +5% reserve | heavy GPU-h | two-GPU stage h (x1.0, x1.1, x1.32) | GPU-h held (x1.0, x1.1, x1.32) | wall h (x1.0, x1.1, x1.32) | one-GPU wall h (x1.0, x1.1, x1.32)")
    for name, A in VARIANTS.items():
        g = gpu_h(A)
        hv = sum(a["n"] * a["sec"] for a in A if a["heavy"]) / 3600
        r = [two_stage(A, m) for m in (1.0, 1.101, 1.32)]
        print("%s | %.1f | %.1f | %.1f | %s | %s | %s | %s" % (
            name, g, 1.05 * g, hv,
            ", ".join("%.1f" % x["s1_h"] for x in r), ", ".join("%.1f" % x["gpuh"] for x in r),
            ", ".join("%.1f" % x["wall_h"] for x in r), ", ".join("%.1f" % (g * m) for m in (1.0, 1.101, 1.32))))
    print()
    print("per family (variant A), GPU-h planning:")
    fam = {}
    for a in VARIANTS["A recommended"]:
        fam[a["fam"]] = fam.get(a["fam"], 0.0) + a["n"] * a["sec"] / 3600
    for k, v in sorted(fam.items(), key=lambda kv: -kv[1]):
        print("  %-15s %.2f" % (k, v))
