"""THE design table of SF-43 variant A (owner decision H9 of 2026-11-02): data only, no logic beyond tiny derivations.

Everything the plan, the pre-registration, the seed registry and the smoke test are built from is here.  Sources of every number:

* geometry, sizes, priorities, memory classes: decision ``2026-11-02-sf43-pre-activation-decisions`` (H5-H13, H21) and the review
  section 4 (``design_cost.py``); sources 19.2 / 9.6 lambda and ``M`` 128 / 100: H10; Par2 horizon 3.6, ``dt``, ``cg_rtol``: H12;
* ``SEC`` (planning seconds per realization) and the sharing rules: ``design_cost.py`` (SF-42 cost audit, own time on the V100);
* ``PEAK_MIB``: own-PID GPU peaks of the SF-42 pilots (``pilots/*/RUN.json``);
* planning standard deviations (displacement basis, SF-42 R3 ``verification/noise_summary.csv``) and precision targets (specification item 6).
"""
from collections import OrderedDict, namedtuple

VARIANT = "A"
CAMPAIGN = "sf43-confirmatory-A"
SCHEMA_VERSION = 1
CUTOFF_UTC = "2026-11-08T19:00:00Z"        # H13: 16:00 local
GRACE_MINUTES = 60
STAGE1_MAX_HOURS = 30
BUDGET_GPU_H = 150
DISK_MIN_FREE_GB = 100                      # host: 548 GB free; plane arrays of the primary arms (about 80 GB) are retained
PAR2_TRANSPORT_SEED = 420800001             # transport.seed base B: the unit's seed is B + 1000 seed0 (SF-42 J3 convention)
SEED_BASE_FORBIDDEN = [
    # (first, last, why): any SF-43 field seed or derived Par2 transport seed must avoid these
    (1, 99999, "every earlier campaign (SF-35..SF-39: 3001-3005, 3101-3103, 4001-4010, 4242, 4901, 5001-5101, 6001-6103, 7001-7010, 7101+, 7201-7231, 9001-9021)"),
    (240000, 240999, "SF-40 note"),
    (420000, 420999, "SF-42 pilots and cross-checks (J1 420601+, J2 420701+, J3 420801+, J1B 420901+, spectral 420001/2, MMS/identity 4202xx)"),
]
TRANSPORT_FORBIDDEN = [
    (PAR2_TRANSPORT_SEED, PAR2_TRANSPORT_SEED + 1000 * 99, "SF-42 J3/J3B Par2 transport series 420800001 + 1000 r"),
    (123456789, 123456789 + 1000 * 99, "SF-39 Par2 pilot transport series"),
    (123476789, 123476789 + 1000 * 99, "SF-39 Par2 extension transport series"),
]

# amplitude key -> (sigma^2, label, --eps with --sigma2 1, lattice M, source side S in lambda)
AMPS = OrderedDict([
    ("s025", (0.25, "0.25", 0.5, 128, "19.2")),
    ("s1", (1.0, "1", 1.0, 128, "19.2")),
    ("s225", (2.25, "2.25", 1.5, 100, "19.2")),
    ("s4", (4.0, "4", 2.0, 100, "9.6")),
    ("s004", (0.04, "0.04", 0.2, 128, "19.2")),     # positive control
])
D_LO, D_HI, D_FAR = 25, 94, 194                      # travel windows in lambda
WINDOWS = [[25, 94], [25, 59.5], [59.5, 94]]         # production-box reductions (main, h1, h2)
WINDOWS_DIST = [[25, 94], [94, 194]]                 # distance arms (main, far)

# planning seconds per realization (design_cost.py SEC; M = 128 at 0.25 / 1, 100 at 2.25 / 4)
SEC = {
    ("1040x240", "0.25"): 26.4, ("1040x240", "1"): 26.8, ("520x120", "0.25"): 12.0, ("520x120", "1"): 11.5,
    ("2080x480", "0.25"): 138.1, ("2080x480", "1"): 142.0, ("1040x480", "0.25"): 79.3, ("1040x480", "1"): 81.5,
    ("par2_1040x240", "0.25"): 12.8, ("par2_1040x240", "1"): 13.8, ("g7_cell_mean", "0.25"): 15.7, ("g7_cell_mean", "1"): 16.3,
    ("2080x240", "1"): 54.0,
    ("1664x384", "2.25"): 54.0, ("416x96", "2.25"): 5.1, ("832x192", "2.25"): 14.5, ("832x384", "2.25"): 30.0,
    ("1248x288", "2.25"): 37.5, ("par2_1248x288", "2.25"): 19.2, ("3328x384_L208", "2.25"): 102.1,
    ("1248x576", "2.25"): 85.1 * (4.14 / 2.40),    # not measured (4.14e8 cells): scaled from 1040x480x480 by cells
    ("3328x384", "4"): 131.5, ("832x96", "4"): 9.7, ("1664x192", "4"): 26.9, ("416x48", "4"): 4.0, ("832x192", "4"): 15.1,
    ("1664x384", "4"): 49.7, ("par2_1664x192", "4"): 17.5, ("3328x192_L208", "4"): 52.0,
    ("1040x240_pc", "0.04"): 26.4,
}
# extra seconds of a 3-pass (h_max 8,16,32) realization over the 1-pass one, on the first 30 production seeds
H3_EXTRA = {"s025": 72.1 - 26.4, "s1": 72.5 - 26.8, "s225": 98.6 - 54.0, "s4": 220.8 - 131.5}
H3_RAW_FACTOR = 2.1                                   # raw output of a 3-pass realization (SF-42 pilots)
HMAX_FIRST = {"n": 30, "divs": [8, 16, 32]}

# own-PID GPU peaks (MiB) of SF-42 pilots by (runner, nx, ny); None = not measured (estimate in PEAK_EST)
PEAK_MIB = {
    ("c", 520, 120): 1757, ("c", 416, 96): 1655, ("c", 832, 96): 1701, ("c", 832, 192): 2275, ("c", 1040, 240): 4997,
    ("c", 1664, 192): 5077, ("c", 1248, 288): 7517, ("c", 832, 384): 7595, ("c", 2080, 240): 7445,
    ("p", 1040, 240): 8157, ("p", 1248, 288): 13673, ("p", 1664, 192): 8335,
}
PEAK_EST = {("c", 416, 48): 1655, ("c", 3328, 192): 10200}   # smaller than 416x96 / twice 1664x192; measured by the smoke test
MEM_FREE = {"near": 30000, "half": 17000}                    # near-empty / half-class units (H13, H14)
PLANES_RAW_GB_M128 = 0.231                                   # raw output per realization at M = 128, 70 planes (95-231 MB, review 3.g)

Row = namedtuple("Row", "id amp fam run W lh L n grp prio sec mem nmax flags")


def row(id, amp, fam, run, W, lh, L, n, grp, prio, sec, mem=None, nmax=None, flags=()):
    return Row(id, amp, fam, run, W, lh, L, n, grp, prio, sec, mem, nmax or n, tuple(flags))


P, E = ("primary", "reest", "hm3"), ("reest",)
ROWS = [
    # ---- sigma^2 0.25: CRN group g_s025 (prod = Gate 5 consumer twin; ladder 5/10/20; width 48; Gate 5 Par2)
    row("s025_prod", "s025", "production", "c", 24, 10, 104, 200, "g_s025", 1, ("1040x240", "0.25"), nmax=400, flags=P),
    row("s025_eh5", "s025", "E_h", "c", 24, 5, 104, 100, "g_s025", 1, ("520x120", "0.25")),
    row("s025_eh20", "s025", "E_h", "c", 24, 20, 104, 100, "g_s025", 1, ("2080x480", "0.25"), mem="near"),
    row("s025_w48", "s025", "E_W", "c", 48, 10, 104, 150, "g_s025", 1, ("1040x480", "0.25"), mem="half", nmax=300, flags=E),
    row("s025_g5p", "s025", "gate5_par2", "p", 24, 10, 104, 200, "g_s025", 1, ("par2_1040x240", "0.25"), flags=("g5", "snap")),
    # ---- sigma^2 1
    row("s1_prod", "s1", "production", "c", 24, 10, 104, 200, "g_s1", 1, ("1040x240", "1"), nmax=400, flags=P),
    row("s1_eh5", "s1", "E_h", "c", 24, 5, 104, 100, "g_s1", 1, ("520x120", "1")),
    row("s1_eh20", "s1", "E_h", "c", 24, 20, 104, 100, "g_s1", 1, ("2080x480", "1"), mem="near"),
    row("s1_w48", "s1", "E_W", "c", 48, 10, 104, 150, "g_s1", 1, ("1040x480", "1"), mem="half", nmax=300, flags=E),
    row("s1_g5p", "s1", "gate5_par2", "p", 24, 10, 104, 200, "g_s1", 1, ("par2_1040x240", "1"), flags=("g5", "snap")),
    row("s1_dist", "s1", "distance", "c", 24, 10, 208, 100, "g_d_s1", 3, ("2080x240", "1"), flags=("dist",)),
    # ---- sigma^2 2.25 (prod lambda/h 16; ladder 4/8/16; width 2x2 of widths 24/48 at lambda/h 8/12)
    row("s225_prod", "s225", "production", "c", 24, 16, 104, 200, "g_s225", 1, ("1664x384", "2.25"), mem="half", nmax=400, flags=P),
    row("s225_eh4", "s225", "E_h", "c", 24, 4, 104, 150, "g_s225", 1, ("416x96", "2.25")),
    row("s225_eh8", "s225", "E_h", "c", 24, 8, 104, 150, "g_s225", 1, ("832x192", "2.25")),            # also the (24, 8) width cell
    row("s225_w48_8", "s225", "E_W", "c", 48, 8, 104, 150, "g_s225", 1, ("832x384", "2.25"), nmax=300, flags=E),
    row("s225_w48_12", "s225", "E_W", "c", 48, 12, 104, 150, "g_s225", 1, ("1248x576", "2.25"), mem="near", nmax=300, flags=E),
    row("s225_c12", "s225", "gate5_consumer", "c", 24, 12, 104, 200, "g_s225", 1, ("1248x288", "2.25"), flags=("g5c",)),  # (24, 12) cell, E_h gate level
    row("s225_g5p", "s225", "gate5_par2", "p", 24, 12, 104, 200, "g_s225", 1, ("par2_1248x288", "2.25"), flags=("g5", "snap")),
    row("s225_dist", "s225", "distance", "c", 24, 16, 208, 100, "g_d_s225", 3, ("3328x384_L208", "2.25"), mem="near", flags=("dist",)),
    # ---- sigma^2 4 (prod lambda/h 32, width 12; ladder 4/8/16/32; width 2x2 of 12/24 at 8/16)
    row("s4_prod", "s4", "production", "c", 12, 32, 104, 200, "g_s4", 1, ("3328x384", "4"), mem="near", nmax=400, flags=P),
    row("s4_l4", "s4", "E_h", "c", 12, 4, 104, 100, "g_s4", 1, ("416x48", "4")),
    row("s4_l8", "s4", "E_h", "c", 12, 8, 104, 150, "g_s4", 1, ("832x96", "4")),                      # also the (12, 8) width cell
    row("s4_l16", "s4", "gate5_consumer", "c", 12, 16, 104, 250, "g_s4", 1, ("1664x192", "4"), flags=("g5c",)),  # E_h rung, (12, 16) cell, Gate 5 twin
    row("s4_w24_8", "s4", "E_W", "c", 24, 8, 104, 150, "g_s4", 1, ("832x192", "4"), nmax=300, flags=E),
    row("s4_w24_16", "s4", "E_W", "c", 24, 16, 104, 150, "g_s4", 1, ("1664x384", "4"), mem="half", nmax=300, flags=E),
    row("s4_g5p", "s4", "gate5_par2", "p", 12, 16, 104, 250, "g_s4", 1, ("par2_1664x192", "4"), flags=("g5", "snap")),
    row("s4_dist", "s4", "distance", "c", 12, 16, 208, 100, "g_d_s4", 3, ("3328x192_L208", "4"), flags=("dist",)),
    # ---- positive control (priority 2)
    row("pc_s004", "s004", "positive_control", "c", 24, 10, 104, 30, "g_pc", 2, ("1040x240_pc", "0.04"), flags=("pc",)),
]
PRIMARY_FAMILY = ["robust:s025", "robust:s1", "robust:s225", "robust:s4", "gate5:s025", "gate5:s1", "gate5:s225", "gate5:s4"]

# G7: 2^(4-1) resolution-IV fraction I = GIDR at sigma^2 0.25 and 1, Par2 on 1040x240x240 (lambda/h 10, width 24), 60 seeds per cell.
# g: +1 SF-18 / -1 Par2 random-spectral (own seed block: whole plot, unpaired); i: +1 window / -1 full plane; d: +1 dt/2 / -1 dt;
# r: +1 cg_rtol 1e-10 / -1 1e-6; the 8 cells have g i d r = +1.
G7_AMPS = ("s025", "s1")
G7_SEEDS = 60
G7_RS_MODES = 10000                                   # SF-39 used 10000 modes for Par2's own generator

# planning standard deviation of alpha_TP (displacement basis) with its n: SF-42 R3 noise_summary.csv (alpha_TAP identical to 4 digits)
SD_PLAN = {"s025": (1.205564082e-4, 25), "s1": (1.120574325e-3, 50), "s225": (4.07294397e-3, 62), "s4": (1.687814483e-2, 71)}
# precision targets (specification item 6): 90 % half-width of alpha_TP (lambda); reported, not extended after the fact
TARGETS = {"s025": 2.5e-4, "s1": 6e-4, "s225": 2.8e-3, "s4": 6.6e-3}
# first-order alpha_T of the window [25, 94] in an unbounded medium: 7.992e-5 sigma^2 (firstorder_window.py); H3 null c = this value
NULL_PER_SIGMA2 = 7.992e-5
BLINDED = {"fraction": 0.25, "max_factor": 2.0, "level": 0.90}   # G5.2 / H9
FAILURE = {"halt_rate": 0.05, "min_for_rate": 20}                # engine arm halt; expected rates pre-registered in the note
RESERVE_FRAC = {"default": 0.05, "s4": 0.10}                     # H13
RETAIN_FIRST, REVIEW_POSITIONS = 5, (0.4, 0.7, 1.0)              # retained seeds per consumer arm: first 5 + 3 for the human review

# seed blocks: group -> start (count = largest n_max of the group; reserves follow the block)
SEED_START = OrderedDict([
    ("g_s025", 431001), ("g_s1", 432001), ("g_s225", 433001), ("g_s4", 434001), ("g_pc", 435001),
    ("g_d_s1", 436001), ("g_d_s225", 436201), ("g_d_s4", 436401),
    ("g7_s025_S", 437001), ("g7_s025_R", 437101), ("g7_s1_S", 437201), ("g7_s1_R", 437301),
])
# smoke: own block 439xxx; one group per amplitude (all arms of the amplitude share the start)
SMOKE_START = OrderedDict([
    ("smoke_s025", 439001), ("smoke_s1", 439101), ("smoke_s225", 439201), ("smoke_s4", 439301), ("smoke_s004", 439401),
    ("smoke_g7_S", 439501), ("smoke_g7_R", 439601), ("smoke_detc", 439901), ("smoke_detp", 439911),
])
SMOKE = {
    "n": 2, "n_slow": 1, "slow_s": 100.0,           # 2 realizations per arm, 1 above ~100 s (never the production arms)
    "reest_arm": "s025_prod", "reest_n_conf": 10, "reest_n_max": 14, "reest_fraction": 0.6, "reest_hw_scale": 0.1,
    "hmax_first_n": 2, "unit": 2, "stage1_max_hours": 4, "cutoff_minutes": 360, "background_wait_minutes": 300,
    "budget_factor": 2.6, "budget_min": 3,
    "campaign_suffix": "_smoke", "det_consumer": ("s4", 416, 48, 4, 12), "det_par2": ("s025", 1040, 240, 10, 24),
}
COVERAGE_ARGS = ["--workers", "32"]
