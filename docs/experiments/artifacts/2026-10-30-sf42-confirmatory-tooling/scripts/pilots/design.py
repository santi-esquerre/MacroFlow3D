"""SF-42 R4 design pilots: run tables (J1 grid feasibility, J2 particle count, J3 Gate 5 pairing) and the command / config builders.

Everything here is derived from the SF-43 planning geometry (box 104 lambda; grid nx x ny x ny = (104 lh) x (W lh) x (W lh) unless stated),
the amplitude's fixed source S (19.2 lambda at sigma^2 0.25 / 1 / 2.25, 9.6 at 4, centred, ``seedlattice.plan``) and the seed-lattice rule
(``seedlattice.rule``).  Consumer flags are those of SF-39's production arms (``sf39.geometry flags``) with the SF-42 plane list
(d = 25 .. 94 lambda from x0 = 5 lambda, spacing 1 lambda, the SF-39 spacing).  Par2 configs use the SF-42 keys
(``stochastic.field_source: sf18_medium_b``, ``transport.inject_window``, ``analysis.plane_crossings``) and SF-39's Par2 time-step rule
scaled to the grid (``par2_time``).  Seeds: SF-42 pilot ranges 420601-420699 (J1), 420701-420799 (J2), 420801-420899 (J3), 420901-420999 (J1B).
"""
import math
from collections import OrderedDict, namedtuple

from seedlattice import plan as splan
from seedlattice import rule

EPS = OrderedDict([("0.25", 0.5), ("1", 1.0), ("2.25", 1.5), ("4", 2.0)])   # --eps with --sigma2 1 (sigma^2_lnK = eps^2)
D_LO, D_HI = 25, 94                                                        # travel window in lambda (SF-43 item 1)
PAR2_SEED = 420800001                                                      # transport.seed of the pairing pilot (+1000 r in the pipeline)
PAR2_NP = 10000
PAR2_CG_RTOL = "1.0e-10"
PAR2_CG_MAX = 4000

Run = namedtuple("Run", "id job code nx ny lh amp seed0 nr M hz", defaults=(1.2,))   # code: consumer or par2; hz: Par2 horizon factor


def fmt(x):
    return "%.17g" % x


def lam_over_h_width(nx, ny, lh):
    """-> (box length, width) in lambda; both must be integers (SF-43 grids)."""
    if nx % lh or ny % lh:
        raise ValueError("grid %dx%d not a whole number of lambda at lh=%d" % (nx, ny, lh))
    return nx // lh, ny // lh


def ell(ny, lh):
    return lh / ny


def L1(nx, ny):
    """The consumer's L1 = nx * h with h = 1.0 / ny (DirichletLongBoxStage)."""
    return nx * (1.0 / ny)


def planes(nx, ny, lh):
    """Plane strings (``%.17g``) x0 + d ell, d = 25 .. 94 (the SF-39 spacing of 1 lambda), x0 = 5 ell."""
    e = ell(ny, lh)
    x0 = 5.0 * e
    L, _ = lam_over_h_width(nx, ny, lh)
    if D_HI + 5 >= L:
        raise ValueError("box of %d lambda too short for the window (x0 + 94 lambda must be inside)" % L)
    return [fmt(x0 + d * e) for d in range(D_LO, D_HI + 1)]


def window(nx, ny, lh, amp):
    """Centred source window of the amplitude (exact short decimals, both axes)."""
    _, W = lam_over_h_width(nx, ny, lh)
    if W < float(splan.SOURCE[amp]):
        raise ValueError("source %s lambda wider than the box width %d" % (splan.SOURCE[amp], W))
    return splan.window_string(W, splan.SOURCE[amp])


def target_M(amp, factor=1.0):
    """SF-39 seeds per lambda of source times S, times ``factor`` (0.6 / 1 / 2)."""
    return float(splan.m_ref(amp)) * factor


def pick_M(ny, win, target, exclude=()):
    """Admissible M (seed-lattice rule on this grid and window) nearest to ``target`` among [0.5, 2.5] x target (ties: smaller),
    skipping the values in ``exclude`` (so that three targets give three distinct lattice sizes)."""
    lo = max(1, int(0.5 * target))
    hi = min(rule.M_MAX, int(math.ceil(2.5 * target)))
    adm = [m for m in rule.admissible(ny, win, range(lo, hi + 1)) if m not in exclude]
    return splan.nearest(adm, target)


def mg_levels(nx, ny, nz):
    """Par2 multigrid depth: the rule of ``box_auto_mg_levels`` (LongBoxStage.cu) with a coarsest extent of at least 8, which reproduces
    SF-39's Par2 choices (6 levels at 2048x256x256, 5 at 2496x288x288)."""
    def can(n):
        return n % 2 == 0 and n // 2 >= 8 and (n // 2) % 2 == 0
    lv, a, b, c = 1, nx, ny, nz
    while can(a) and can(b) and can(c):
        a, b, c, lv = a // 2, b // 2, c // 2, lv + 1
    return lv


def par2_time(nx, ny, lh, hz=1.2):
    """SF-39 Par2 rule scaled to the grid.  SF-39: dt = 12.5 with dx = 5 (U_nom dt = 0.0244140625 dx, U_nom = K_g J = 100/10240) and
    n_steps = ceil(1.2 (Lx - 10 lambda) / (U_nom dt)).  Here (box units, K_g = 1, head drop L1, J = 1): dt = 0.0244140625 dx,
    n_steps = ceil(hz (L1 - 10 ell) / dt) with hz = 1.2 (the SF-39 rule; J3B uses 3.6 to test the horizon)."""
    dx = 1.0 / ny
    dt = 0.0244140625 * dx
    n = int(math.ceil(hz * (L1(nx, ny) - 10.0 * ell(ny, lh)) / dt))
    return dt, n


def consumer_args(run, out):
    """Arguments of ``lagrangian_ensemble`` for a consumer run (flags of SF-39's production arms)."""
    w = window(run.nx, run.ny, run.lh, run.amp)
    return ["--medium", "b", "--grid", "%d,%d,%d" % (run.nx, run.ny, run.ny), "--sigma2", "1", "--eps", "%g" % EPS[run.amp],
            "--ell", fmt(ell(run.ny, run.lh)), "--window", w, "--np-side", str(run.M),
            "--planes", ",".join(planes(run.nx, run.ny, run.lh)), "--times", "auto", "--tol", "1e-8", "--hmax-div", "8",
            "--fused-splines", "--single-launch", "--twin", "on", "--pollock-bridge", "on",
            "--seed0", str(run.seed0), "--nr", str(run.nr), "--out", out]


def par2_yaml(run):
    """Par2 configuration of a run on the SF-18 medium (b) field (same seeds as the consumer), window injection, plane crossings."""
    w = [float(x) for x in window(run.nx, run.ny, run.lh, run.amp).split(",")][:2]
    e = ell(run.ny, run.lh)
    dt, n_steps = par2_time(run.nx, run.ny, run.lh, run.hz)
    pl = ", ".join(planes(run.nx, run.ny, run.lh))
    return """# SF-42 R4 Par2 pilot {id}: sigma^2 {amp}, grid {nx}x{ny}x{ny}, lambda/h {lh} (generated by pilots.design.par2_yaml)
run_mode: ensemble
grid: {{nx: {nx}, ny: {ny}, nz: {ny}, dx: {dx}}}
stochastic:
  sigma2: 1.0
  corr_length: {ell}
  n_modes: 1000
  covariance_type: 1
  seed: 42
  K_mean: 1.0
  field_source: sf18_medium_b
  sf18_medium_b: {{sigma2: 1, eps: {eps:g}, seed0: {seed0}}}
flow:
  solver: mg_cg
  mg_levels: {mgl}
  mg_pre_smooth: 4
  mg_post_smooth: 4
  mg_coarse_iters: 50
  mg_max_cycles: 20
  cg_max_iter: {cgmax}
  cg_rtol: {rtol}
  cg_check_every: 10
  rtol: 1.0e-6
  pin: {{mode: off}}
  bc:
    west:   {{ type: dirichlet, value: {head} }}
    east:   {{ type: dirichlet, value: 0.0 }}
    south:  {{ type: periodic }}
    north:  {{ type: periodic }}
    bottom: {{ type: periodic }}
    top:    {{ type: periodic }}
transport:
  n_particles: {npart}
  dt: {dt}
  n_steps: {nst}
  porosity: 1.0
  diffusion: 0.0
  alpha_l: 0.0
  alpha_t: 0.0
  seed: {tseed}
  output_every: {nst}
  snapshot_every: 0
  inject_x: {x0}
  inject_window: [{w2a}, {w2b}, {w2a}, {w2b}]
analysis:
  macrodispersion: {{enabled: true, NR: {nr}, lambda: {ell}, vmean_norm: 1.0, sample_every: 1000, var_estimator: biased}}
  plane_crossings:
    enabled: true
    planes: [{planes}]
diagnostics:
  velocity_field: false
output:
  output_dir: out
  save_K: false
  save_head: false
  save_velocity: false
  save_particles: false
""".format(id=run.id, amp=run.amp, nx=run.nx, ny=run.ny, lh=run.lh, dx=fmt(1.0 / run.ny), ell=fmt(e), eps=EPS[run.amp],
           seed0=run.seed0, mgl=mg_levels(run.nx, run.ny, run.ny), cgmax=PAR2_CG_MAX, rtol=PAR2_CG_RTOL, head=fmt(L1(run.nx, run.ny)),
           npart=PAR2_NP, dt=fmt(dt), nst=n_steps, tseed=PAR2_SEED, x0=fmt(5.0 * e), w2a=repr(w[0]), w2b=repr(w[1]), nr=run.nr,
           planes=pl)


# ----------------------------------------------------------------------------------------------------------------------
# Job tables
# ----------------------------------------------------------------------------------------------------------------------
# J1 (nx, ny, lh, amp): the highest sigma^2 that SF-43 plans on the grid; see the experiment note for the SF-43 use of each.
J1_GRIDS = [(520, 120, 5, "1"), (832, 96, 8, "4"), (832, 192, 8, "4"), (1040, 240, 10, "2.25"), (1040, 480, 10, "2.25"),
            (2080, 480, 20, "2.25"), (2496, 288, 24, "4"), (1248, 288, 12, "2.25"), (1664, 192, 16, "4")]
J1_PAR2 = [(1040, 240, 10, "1"), (1248, 288, 12, "2.25"), (1664, 192, 16, "4")]     # Gate 5 grids (Par2)
J1_SEED0 = 420601                                                                  # + index in J1_GRIDS (Par2: seed of the consumer run of the grid)

# J1B (R6, audit AUD-GH-3): the SF-43 planning grids that neither SF-39 nor J1 ran, one realization each at the highest sigma^2 planned on the grid:
# 416x96x96 (E_h rung lh 4 at 2.25), 832x384x384 (E_W 2x2 at 2.25), 2080x240x240 (distance arm at 1).  Consumer only (none is a Gate 5 grid).
J1B_GRIDS = [(416, 96, 4, "2.25"), (832, 384, 8, "2.25"), (2080, 240, 10, "1")]
J1B_M = {"1": 128}                                                                 # the only M admissible on every SF-43 arm at 0.25 and 1 (as J3_M)
J1B_SEED0 = 420901                                                                 # + index; reserved range 420901-420999

J2_CONFIG = OrderedDict([("0.25", (1040, 240, 10)), ("1", (1040, 240, 10)), ("2.25", (1664, 384, 16)), ("4", (3328, 384, 32))])
J2_SEED0 = OrderedDict([("0.25", 420701), ("1", 420711), ("2.25", 420721), ("4", 420731)])
J2_FACTORS = (0.6, 1.0, 2.0)

J3_GRIDS = OrderedDict([("0.25", (1040, 240, 10)), ("1", (1040, 240, 10)), ("2.25", (1248, 288, 12)), ("4", (1664, 192, 16))])
J3_M = {"0.25": 128, "1": 128}      # the only M admissible on all SF-43 arms at these amplitudes (Finding (a)); 2.25 / 4: nearest SF-39 density
J3_SEED0 = OrderedDict([("0.25", 420801), ("1", 420811), ("2.25", 420821), ("4", 420831)])


def gid(nx, ny, amp, tag=""):
    return "%s_%dx%d%s" % ("s" + amp.replace(".", ""), nx, ny, tag)


def j1_runs():
    runs = []
    for i, (nx, ny, lh, amp) in enumerate(J1_GRIDS):
        w = window(nx, ny, lh, amp)
        M = pick_M(ny, w, target_M(amp))
        if M is None:
            raise ValueError("no admissible M for %dx%d %s" % (nx, ny, amp))
        runs.append(Run("c_" + gid(nx, ny, amp), "J1", "consumer", nx, ny, lh, amp, J1_SEED0 + i, 1, M))
    for nx, ny, lh, amp in J1_PAR2:
        seed = next(J1_SEED0 + i for i, g in enumerate(J1_GRIDS) if g[:3] == (nx, ny, lh))
        runs.append(Run("p_" + gid(nx, ny, amp), "J1", "par2", nx, ny, lh, amp, seed, 1, 0))
    return runs


def j1b_runs():
    runs = []
    for i, (nx, ny, lh, amp) in enumerate(J1B_GRIDS):
        M = J1B_M.get(amp) or pick_M(ny, window(nx, ny, lh, amp), target_M(amp))
        if M is None:
            raise ValueError("no admissible M for %dx%d %s" % (nx, ny, amp))
        runs.append(Run("c_" + gid(nx, ny, amp), "J1B", "consumer", nx, ny, lh, amp, J1B_SEED0 + i, 1, M))
    return runs


def j2_runs(nr=6):
    runs = []
    for amp, (nx, ny, lh) in J2_CONFIG.items():
        w = window(nx, ny, lh, amp)
        used = []
        for f in J2_FACTORS:
            M = pick_M(ny, w, target_M(amp, f), used)
            if M is None:
                raise ValueError("no admissible M for %s at %gx" % (amp, f))
            used.append(M)
            runs.append(Run("c_%s_M%d" % ("s" + amp.replace(".", ""), M), "J2", "consumer", nx, ny, lh, amp, J2_SEED0[amp], nr, M))
    return runs


def j3_runs(nr=8, M_by_amp=None):
    runs = []
    for amp, (nx, ny, lh) in J3_GRIDS.items():
        w = window(nx, ny, lh, amp)
        M = (M_by_amp or {}).get(amp) or J3_M.get(amp) or pick_M(ny, w, target_M(amp))
        if M is None:
            raise ValueError("no admissible M for %s" % amp)
        runs.append(Run("c_" + gid(nx, ny, amp), "J3", "consumer", nx, ny, lh, amp, J3_SEED0[amp], nr, M))
        runs.append(Run("p_" + gid(nx, ny, amp), "J3", "par2", nx, ny, lh, amp, J3_SEED0[amp], nr, 0))
    return runs


J3B_HZ = 3.6   # 3 x the SF-39 horizon factor: J3's Par2 sets are censored at the horizon (finding of J3), J3B repeats Par2 only with a longer one


def j3b_runs(nr=8):
    """Par2 only, the J3 seeds, grids and windows of amplitudes 1, 2.25, 4 with the horizon factor 3.6 (sensitivity of the J3 pairing to the
    Par2 time horizon; the consumer tables of J3 are reused)."""
    return [r._replace(id=r.id + "_h3", job="J3B", hz=J3B_HZ) for r in j3_runs(nr) if r.code == "par2" and r.amp != "0.25"]


JOBS = {"J1": j1_runs, "J1B": j1b_runs, "J2": j2_runs, "J3": j3_runs, "J3B": j3b_runs}
