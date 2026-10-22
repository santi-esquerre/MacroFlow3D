#!/usr/bin/env python3
"""Byte model of the Lagrangian production path (memory audit of 2026-10-22).

Stdlib only, deterministic, no input files. Prints every number of the model
tables of docs/experiments/2026-10-22-lagrangian-memory-audit.md (sections 4,
6, 7, 8, 9). Each array is annotated with the code line (at da1c185) that
allocates it; "nominal" B/cell counts every array as n = nx ny nz cells (the
convention of the planning session), "exact" adds the CompactMAC (+1) faces,
the half-spectrum (nx/2 + 1) and the (N + 1) vertex planes.

Assumptions (named constants below, nothing else is tuned):
  CONTEXT_MIB   500   planning assumption for the CUDA context (see section 4 of
                      the note for the values implied by the measurements).
  LIMIT_MIB     30000 owner decision 2026-10-22 (V100-PCIE-32GB has 32510 MiB).
  CUFFT_WORK_IS_SPECTRUM  each cuFFT 3D R2C/C2R plan reports a work area equal to
                      one half-spectrum (measured at 256^3 in the SF-30 closure-gate
                      summary.json: 270532608 B = 2 x 135266304 B; extrapolated to
                      other sizes, which the SF-37 peak distribution confirms at 512^3).
Usage: python3 memory_model.py
"""

MIB = 2 ** 20
CONTEXT_MIB = 500
LIMIT_MIB = 30000
REAL = 8   # sizeof(real) = sizeof(double)
CPLX = 16  # sizeof(cufftDoubleComplex)


def mg_levels_cells(nx, ny, nz, levels):
    """MGHierarchy rule (src/multigrid/mg_types.hpp:75-98): halve each extent, stop at < 2."""
    out = []
    a, b, c = nx, ny, nz
    for l in range(levels):
        out.append((a, b, c))
        if l < levels - 1:
            a, b, c = a // 2, b // 2, c // 2
            if a < 2 or b < 2 or c < 2:
                break
    return out


def slab_auto_mg_levels(n):
    """src/physics/streamfunctions/inlet_slab/SlabProductionSetup.cu:269-277 (cubic N)."""
    levels, m = 1, n
    while m % 2 == 0 and m // 2 >= 4 and (m // 2) % 2 == 0:
        m //= 2
        levels += 1
    return levels


def coarse_tail(levels):
    """32 * sum_{l=1}^{levels-1} 8^-l  (x, b, r, coefficient on the coarse levels), B/cell."""
    return 32.0 * sum(8.0 ** -l for l in range(1, levels))


def mib(nbytes):
    return nbytes / MIB


class Grid:
    def __init__(self, nx, ny, nz):
        self.nx, self.ny, self.nz = nx, ny, nz
        self.n = nx * ny * nz

    def mac(self):  # CompactMAC face counts, AffinePeriodicFlowSolver.cu:25-36
        return ((self.nx + 1) * self.ny * self.nz, self.nx * (self.ny + 1) * self.nz,
                self.nx * self.ny * (self.nz + 1))

    def half_spectrum(self):  # (nx/2 + 1) ny nz, PeriodicTricubicBSpline.cu:249-251
        return (self.nx // 2 + 1) * self.ny * self.nz

    def full_vertices(self, nplanes=None):  # (N + 1) N^2, InletSlabGrid.cuh:79-81 (cubic N)
        assert self.nx == self.ny == self.nz
        return (self.nx + 1) * self.ny * self.nz


def sf19_workspace_bytes(g, levels):
    """AffinePeriodicFlowWorkspace::prepare, AffinePeriodicFlowSolver.cu:630-666 (exact bytes).
    The per-level MeanZeroWorkspace (CUB temporary storage, a few KiB) and reduce_blocks
    (256 reals) are not modeled."""
    n = g.n
    mu, mv, mw = g.mac()
    d = {}
    d["correctors w_x,w_y,w_z (.cuh:283)"] = 3 * n * REAL
    d["h_tilde (.cuh:284)"] = n * REAL
    d["rhs_a, rhs_b (.cuh:289-290)"] = 2 * n * REAL
    d["flux_u,flux_v,flux_w (.cuh:295-297, CompactMAC)"] = (mu + mv + mw) * REAL
    d["divergence (.cuh:301)"] = n * REAL
    d["PCG r,z,p,Ap (pcg.cuh:50-54)"] = 4 * n * REAL
    d["PCG b_hat = projected_rhs (pcg.cuh:224)"] = n * REAL
    lv = mg_levels_cells(g.nx, g.ny, g.nz, levels)
    d["MG level 0 x,b,r,coefficient (mg_types.hpp:27-50)"] = 4 * lv[0][0] * lv[0][1] * lv[0][2] * REAL
    d["MG coarse levels 1.. x,b,r,coefficient"] = sum(4 * a * b * c * REAL for a, b, c in lv[1:])
    return d


def total(d):
    return sum(d.values())


def bpc(nbytes, n):
    return nbytes / n


def line(label, nbytes, n, extra=""):
    print(f"  {label:<62s} {bpc(nbytes, n):9.3f} B/cell {mib(nbytes):12.1f} MiB {extra}")


def main():
    print("== constants ==")
    print(f"MIB = {MIB}; CONTEXT_MIB = {CONTEXT_MIB} (planning assumption); LIMIT_MIB = {LIMIT_MIB}"
          f" (owner decision 2026-10-22); sizeof(real) = {REAL}; sizeof(cufftDoubleComplex) = {CPLX}")

    # ------------------------------------------------------------------ cells
    print("\n== cell counts ==")
    cube = Grid(512, 512, 512)
    long_box = Grid(2048, 256, 256)
    lean = Grid(3328, 384, 384)
    print(f"512^3            = {cube.n:>14,d}")
    print(f"2048x256x256     = {long_box.n:>14,d}  (equal to 512^3: {long_box.n == cube.n})")
    print(f"3328x384x384     = {lean.n:>14,d}  ({lean.n:.4e})")
    bdd = {}
    for lh in (16, 32):
        bdd[lh] = (204.8 * lh) * (25.6 * lh) * (25.6 * lh)
        print(f"B&dD 204.8x25.6x25.6 lambda at lambda/h = {lh:2d}: {bdd[lh]:.4e} cells"
              f" ({round(204.8 * lh * 10) / 10:g} x {round(25.6 * lh * 10) / 10:g} x"
              f" {round(25.6 * lh * 10) / 10:g} cells)")
    print(f"MiB per (B/cell) at 512^3 = {cube.n / MIB:.4f}; at 3328x384x384 = {lean.n / MIB:.4f}")
    L512 = slab_auto_mg_levels(512)
    L384 = slab_auto_mg_levels(384)
    print(f"slab_auto_mg_levels(512) = {L512} (coarsest {mg_levels_cells(512,512,512,L512)[-1]});"
          f" slab_auto_mg_levels(384) = {L384}")
    lv_lean = mg_levels_cells(lean.nx, lean.ny, lean.nz, L384)
    print(f"MG levels of 3328x384x384 with {L384} levels: {lv_lean}")
    print(f"coarse tail 32*sum 8^-l: levels 8 (coarse 1..7) = {coarse_tail(8):.5f};"
          f" levels 7 (1..6) = {coarse_tail(7):.5f}; limit = {32/7:.5f}")

    # ------------------------------------------------------------- SF-19 phase
    print("\n== section 4: SF-19 AffinePeriodicFlowWorkspace at 512^3 (exact bytes) ==")
    ws = sf19_workspace_bytes(cube, L512)
    for k, v in ws.items():
        line(k, v, cube.n)
    ws_total = total(ws)
    line("workspace total (canonical 156.6)", ws_total, cube.n)
    mu, mv, mw = cube.mac()
    persist = {
        "K_dev (SlabProductionSetup.cu:479,486)": cube.n * REAL,
        "lnK_dev (SlabProductionSetup.cu:479,485)": cube.n * REAL,
        "u,v,w velocity (SlabProductionSetup.cu:498-499)": (mu + mv + mw) * REAL,
    }
    for k, v in persist.items():
        line(k, v, cube.n)
    flow_phase = ws_total + total(persist)
    line("SF-19 solve phase total (canonical ~196.7)", flow_phase, cube.n)
    nominal_ws = 24 + 8 + 16 + 32 + 32 + 8 + 32 + coarse_tail(L512)
    nominal_flow = nominal_ws + 8 + 8 + 24
    print(f"  nominal (all arrays = n): workspace {nominal_ws:.3f}, SF-19 phase {nominal_flow:.3f} B/cell"
          f" (CompactMAC +1/nx adds {(ws_total + total(persist)) / cube.n - nominal_flow:.3f})")

    # ---------------------------------------------------------------- prefilter
    print("\n== section 4: stage phases at 512^3 (exact bytes; cuFFT work area = 1 half-spectrum per plan) ==")
    spec = cube.half_spectrum() * CPLX          # one half-spectrum (SF-28 prefilter / SF-18 generator)
    work_plan = spec                            # CUFFT_WORK_IS_SPECTRUM
    n = cube.n
    # field phase: SF-18 generator (SlabProductionSetup.cu:317-319, PeriodicGaussianField.cu:283,315-317)
    gen_peak = n * REAL + spec + work_plan
    gen_peak_no_y = spec + work_plan
    # field phase: spectral vertex evaluation (SlabProductionSetup.cu:216-250; y and the generator
    # workspace are still alive inside SlabFieldSource::gaussian, :317-323)
    vertex_eval = n * REAL + spec + 6 * n * CPLX
    print("  [field] SF-18 generator: y_out + half-spectrum + one Z2D work area:")
    line("    generator peak (y_out 8 + spectrum + work area)", gen_peak, n,
         "(the '~24 B/cell' of the planning text)")
    line("    generator peak without y_out (spectrum + work area)", gen_peak_no_y, n,
         "(the '16 B/cell' of the planning text)")
    line("  [field] spectral_periodic_evaluate: y + gen spectrum + in,F,ov,og[3] (6 complex n)", vertex_eval, n,
         "(Z2Z work area not modeled)")
    line("  [flow]  SF-19 solve phase", flow_phase, n)
    sh_ws = n * REAL + spec
    sy_ws = n * REAL + spec
    lnK = n * REAL
    pref1 = ws_total + lnK + sh_ws + 2 * work_plan
    pref2 = ws_total + lnK + sh_ws + sy_ws + 2 * work_plan
    line("  [prefilter 1] SF-19 workspace + lnK + sh_ws + 2 plans (:562-563)", pref1, n)
    line("  [prefilter 2] SF-19 workspace + lnK + sh_ws + sy_ws + 2 plans (:564)", pref2, n,
         "<- exact-byte stage peak (canonical ~213)")
    full = cube.full_vertices()
    vert7 = 7 * full * REAL                     # dpx,dpy,dpz,val,gx,gy,gz (:592)
    inputs = 5 * full * REAL                    # q, lnk, grad_lnk[3] (InletSlabGrid.cuh:142-165)
    reference = 3 * full * REAL                 # vD[3] (InletSlabGrid.cuh:186-199)
    steps56 = sh_ws + sy_ws + vert7 + inputs + reference
    line("  [steps v-vi] sh_ws + sy_ws + 7 vertex arrays + inputs(5) + reference(3)", steps56, n,
         "(canonical ~154; u0, vperp_in, p1, p2 are O(N^2) and omitted)")
    persistent_stage = inputs + reference
    tracker_coef = 2 * n * REAL
    track = persistent_stage + tracker_coef
    line("  [tracking] stage inputs+reference kept by the driver", persistent_stage, n,
         "(read by nobody during tracking)")
    line("  [tracking] + tracker coefficients coef_pot, coef_logk (LagrangianTracker.cu:237,256-257)",
         track, n)

    # -------------------------------------------------- per-particle tracker bytes
    print("\n== section 6: tracker device bytes per particle (LagrangianTrackerImpl.cuh:63-79) ==")
    pass_stats = 7 * 8 + 2 * 8 + 4 * 4          # LagrangianTrackerTypes.hpp:22-33
    P = T = 7                                   # SF-37 production: planes and times lists of 7
    per_particle = 3 * 8 + 2 * pass_stats + P * (2 + 1 + 1) * 8 + T * (3 + 1) * 8
    print(f"  sizeof(PassStats) = {pass_stats} B (7 long long, 2 double, 4 int)")
    print(f"  per particle, P = T = 7, no covectors: seeds 24 + 2 x PassStats {2*pass_stats}"
          f" + planes 7 x 32 = {P*32} + times 7 x 32 = {T*32}  =  {per_particle} B"
          f" (covectors would add {P*6*8 + T*6*8} B)")
    for npart in (10_000, 65_536):
        print(f"  {npart:>6d} particles: {npart * per_particle / 1e6:8.2f} MB device")
    regs = 255
    regs_alloc = -(-regs // 8) * 8              # allocation granularity 8 regs/thread (256 regs per warp)
    threads_sm = 65536 // regs_alloc
    sms = 80                                    # Tesla V100 datasheet (not queried)
    print(f"  255 regs/thread -> {regs_alloc} allocated -> {threads_sm} threads/SM (register file 65536)"
          f" -> {threads_sm * sms} resident threads on {sms} SMs (V100 datasheet)")
    print(f"  load at 1e4 particles: {10_000 / (threads_sm * sms):.2f} of the resident threads;"
          f" 65536 particles = {65_536 / (threads_sm * sms):.2f} waves")
    print(f"  stack 1376 B/thread (plane_kernel<5>) x {threads_sm * sms} threads ="
          f" {1376 * threads_sm * sms / 1e6:.1f} MB local memory")

    # ------------------------------------------------------------- Par2 baseline
    print("\n== section 7: Par2 baseline ensemble path, 2048x256x256, mg_cg, mg_levels 6 (EnsembleRunner.cu:156-170) ==")
    par2 = Grid(2048, 256, 256)
    lv6 = mg_levels_cells(par2.nx, par2.ny, par2.nz, 6)
    par2_mg = sum(4 * a * b * c * REAL for a, b, c in lv6)
    pad = 3 * (par2.nx + 1) * (par2.ny + 1) * (par2.nz + 1) * REAL
    comps = [
        ("K_field", par2.n * REAL), ("head_field", par2.n * REAL),
        ("FlowWorkspace residual,aux1,aux2 (never read)", 3 * par2.n * REAL),
        ("FlowWorkspace rhs", par2.n * REAL), ("FlowWorkspace MG hierarchy (6 levels)", par2_mg),
        ("FlowWorkspace PCG r,z,p,Ap (solver mg_cg -> PCG_MG)", 4 * par2.n * REAL),
        ("StochasticWorkspace logK", par2.n * REAL), ("PaddedVelocityField U,V,W", pad),
    ]
    tot = 0
    for k, v in comps:
        line(k, v, par2.n)
        tot += v
    line("total (Par2_Core particle arrays and mode arrays not modeled)", tot, par2.n)
    line("  of which dead FlowWorkspace buffers residual+aux1+aux2", 3 * par2.n * REAL, par2.n,
         f"= {mib(3 * par2.n * REAL) / 1024:.2f} GiB")
    line("VelocityDiagnostics when diagnostics.velocity_field (3 ScalarField)", 3 * par2.n * REAL, par2.n)

    # --------------------------------------------------------- reconciliation
    print("\n== section 4: reconciliation with the measured peaks (512^3, nvidia-smi, 200 ms, MiB) ==")
    base_model = mib(flow_phase)
    peak_model = mib(pref2)
    plan = mib(work_plan)
    print(f"  model: SF-19 plateau {base_model:.1f} MiB; one cuFFT plan work area {plan:.1f} MiB;"
          f" two plans {2 * plan:.1f} MiB; exact stage peak {peak_model:.1f} MiB")
    print(f"  model + CONTEXT_MIB = {base_model + CONTEXT_MIB:.1f} MiB (planning text: 25716, not reproduced)")
    modal = 25921
    ovh = modal - base_model
    print(f"  SF-37 modal value {modal} MiB (50/72 runs) - plateau = implied unmodeled overhead {ovh:.1f} MiB")
    for label, obs in (("plateau", [25921, 25925, 25929]), ("one plan", [26951, 26953]), ("two plans", [27981])):
        shift = {"plateau": 0.0, "one plan": plan, "two plans": 2 * plan}[label]
        pred = base_model + ovh + shift
        print(f"  {label:<10s} predicted {pred:9.1f} MiB; observed {obs}; "
              f"delta {[round(o - pred, 1) for o in obs]}")
    print(f"  max observed 27981 MiB = {27981 * MIB / cube.n:.1f} B/cell incl. context"
          f" (canonical 218.6); model exact peak {pref2 / cube.n:.1f} B/cell + overhead {ovh * MIB / cube.n:.1f} B/cell")
    field_meas = 14825
    print(f"  field phase: model {mib(vertex_eval):.1f} MiB (spectral vertex evaluation, no Z2Z work area);"
          f" SF-35 measured max {field_meas} -> difference {field_meas - mib(vertex_eval):.1f} MiB")
    track_meas = 11652825088 / MIB
    print(f"  tracking plateau: model {mib(track):.1f} MiB; driver PhaseLog (cudaMemGetInfo, phase edges)"
          f" {track_meas:.1f} MiB (11652825088 B) -> difference {track_meas - mib(track):.1f} MiB")
    n256 = 256 ** 3
    flow256 = flow_phase / cube.n * n256
    meas256 = 3769
    ovh256 = meas256 - mib(flow256)
    print(f"  256^3: model SF-19 plateau {mib(flow256):.1f} MiB; SF-35 J2a measured stage max {meas256}"
          f" (SF-36 G: 3771) -> implied overhead {ovh256:.1f} MiB")
    slope = (ovh - ovh256) / (cube.n - n256)    # MiB per cell
    icpt = ovh256 - slope * n256
    print(f"  two-point overhead model (indicative, 2 points): {icpt:.1f} MiB + {slope * MIB:.3f} B/cell x cells"
          f" -> {icpt + slope * lean.n:.1f} MiB at 3328x384x384")
    print(f"  host RSS (/usr/bin/time -v, SF-37 v2 512^3): 25824868 kB = {25824868 * 1024 / cube.n:.1f} B/cell;"
          f" v3b (keep_face_fluxes): 28977484 kB = {28977484 * 1024 / cube.n:.1f} B/cell;"
          f" src host arrays cell_ + vtx_ + vgrad_[3] = {(n * REAL + 4 * full * REAL) / n:.1f} B/cell")
    print(f"  host RSS scaled linearly to 3328x384x384 at 197.0 B/cell: {197.0 * lean.n / 2**30:.1f} GiB"
          " (host RAM of the V100 machine is not recorded in the repository)")

    # ----------------------------------------------------------- bitwise levels
    print("\n== section 8: bitwise memory levels (device B/cell at the SF-19 PCG peak, context excluded) ==")
    tail = coarse_tail(L384)                    # 3328x384x384: 7 levels
    cur_ws = 24 + 8 + 16 + 32 + 40 + 32 + tail  # nominal workspace (canonical 156.6)
    cur_total = cur_ws + 8 + 8 + 24
    o1 = dict([("K", 8), ("MG L0 coefficient copy of K", 8), ("MG L0 x,b,r", 24), ("MG coarse levels", tail),
               ("PCG r,z,p,Ap", 32), ("b_hat", 8), ("active corrector", 8)])
    o2 = dict([("K as MG L0 coefficient", 8), ("MG L0 x,b,r", 24), ("MG coarse levels", tail),
               ("PCG r,z,p,Ap", 32), ("b_hat", 8), ("active corrector", 8)])
    s = dict([("K as MG L0 coefficient", 8), ("MG L0 x,r", 16), ("shared buffer z / Ap / MG L0 b", 8),
              ("MG coarse levels", tail), ("PCG r,p", 16), ("b_hat", 8), ("active corrector", 8)])
    sp = dict([(k, v) for k, v in s.items() if k != "b_hat"])
    sp["b_hat (pinned mapped host memory)"] = 0
    levels = [("current (nominal)", None, cur_total), ("O1", o1, sum(o1.values())),
              ("O2", o2, sum(o2.values())), ("S", s, sum(s.values())), ("S+", sp, sum(sp.values()))]
    print(f"  current nominal: workspace {cur_ws:.2f} + K 8 + lnK 8 + u,v,w 24 = {cur_total:.2f}"
          f" (coarse tail for {L384} levels = {tail:.4f})")
    for name, comp, val in levels[1:]:
        print(f"  {name:<3s} = {val:7.3f} B/cell  " + " + ".join(f"{k} {v:g}" if v != int(v) else f"{k} {int(v)}"
                                                                for k, v in comp.items()))
    print("  max cells within LIMIT_MIB with CONTEXT_MIB of context:")
    avail = (LIMIT_MIB - CONTEXT_MIB) * MIB
    for name, val in (("current, exact peak (~212.7)", pref2 / cube.n), ("current, SF-19 phase (196.7)", flow_phase / cube.n),
                      ("O1", levels[1][2]), ("O2", levels[2][2]), ("S", levels[3][2]), ("S+", levels[4][2])):
        print(f"    {name:<30s} {val:8.3f} B/cell -> {avail / val:.4e} cells")
    print("  b_hat touches per corrector (pcg.cuh:321-324,325-331,411,455): copy(write) 1 + project(sum read 1,"
          " subtract read+write 2) + recompute_true_residual reads (1 initial + one per check_every = 10 iterations)")
    bytes_pass = lean.n * REAL
    for iters in (20, 30, 200):
        recomputes = 1 + iters // 10
        traffic_passes = 1 + 1 + 2 + recomputes
        total_gb = 3 * traffic_passes * bytes_pass / 1e9
        print(f"    {iters:>3d} PCG iterations: {recomputes} recomputes -> {traffic_passes} array traffic passes per corrector;"
              f" 3 correctors move {total_gb:6.1f} GB over PCIe at 3328x384x384"
              f" -> {total_gb / 12:5.1f} s at 12 GB/s, {total_gb / 10:5.1f} s at 10 GB/s (assumed, not measured)")

    # ------------------------------------------------------------- feasibility
    print("\n== section 9: feasibility at 3328x384x384 (MiB = cells x B/cell / 2^20 + CONTEXT_MIB) ==")
    names = [("current, exact stage peak", pref2 / cube.n), ("current, SF-19 phase", flow_phase / cube.n)] + \
            [(nm, val) for nm, _, val in levels[1:]]
    for nm, val in names:
        m = lean.n * val / MIB
        print(f"  {nm:<28s} {val:8.3f} B/cell: {m:10.1f} MiB + {CONTEXT_MIB} = {m + CONTEXT_MIB:10.1f} MiB"
              f"  ({'fits' if m + CONTEXT_MIB <= LIMIT_MIB else 'does not fit'} in {LIMIT_MIB})")
    sp_val = levels[4][2]
    sp_mib = lean.n * sp_val / MIB
    print("  S+ sensitivity to the unmodeled overhead:")
    for o in (CONTEXT_MIB, 750, 1000, icpt + slope * lean.n):
        print(f"    overhead {o:8.1f} MiB -> {sp_mib + o:9.1f} MiB (margin to {LIMIT_MIB}: {LIMIT_MIB - sp_mib - o:8.1f} MiB)")
    print("  lambda extents of 3328x384x384 (cells / (lambda/h)):")
    for lh, s2 in ((8, "1"), (16, "2.25"), (32, "4")):
        print(f"    lambda/h {lh:2d} (sigma^2 {s2:>4s}): {3328 / lh:6.1f} x {384 / lh:5.1f} x {384 / lh:5.1f} lambda")
    print("  Beaudoin & de Dreuzy domain 204.8 x 25.6 x 25.6 lambda:")
    for lh in (16, 32):
        cells = bdd[lh]
        print(f"    lambda/h {lh}: {cells:.4e} cells; S+ needs {cells * sp_val / MIB + CONTEXT_MIB:12.1f} MiB"
              f" (= {cells * sp_val / MIB / 1024:.1f} GiB) vs {LIMIT_MIB}; "
              f"{cells / (avail / sp_val):.2f}x the S+ max cell count")
    print("  FFT phases (transient, additive to the persistent arrays of the phase):")
    pref_own = (n * REAL + spec + 2 * work_plan) / n
    print(f"    SF-28 prefilter of one field: coefficients 8 + half-spectrum {spec / n:.3f} + 2 work areas"
          f" {2 * work_plan / n:.3f} = {pref_own:.3f} B/cell")
    print(f"    SF-18 generator: y_out 8 + half-spectrum {spec / n:.3f} + work area {work_plan / n:.3f}"
          f" = {gen_peak / n:.3f} B/cell")
    lean_pref2 = lnK / n + 8 + sy_ws / n + 2 * work_plan / n
    print(f"    lean second prefilter (SF-19 workspace freed, sh spectrum freed): lnK 8 + sh coefficients 8 +"
          f" sy_ws {sy_ws / n:.3f} + 2 work areas {2 * work_plan / n:.3f} = {lean_pref2:.3f} B/cell"
          f" (< S+ {sp_val:.2f})")
    print(f"    lean second prefilter with the sh spectrum kept: {lean_pref2 + spec / n:.3f} B/cell")
    post_pcg = 8 + 8 + 8 + 24 + 8
    print(f"    post-PCG velocity phase if PCG/MG buffers are released: K 8 + h_tilde 8 + upload scratch 8 +"
          f" u,v,w 24 + divergence 8 = {post_pcg} B/cell (< S+)")
    nothing_released = 32 + 8 + (32 + tail) + 8 + 8 + 24 + 8
    print(f"    same phase if nothing is released (PCG 32 + b_hat 8 + MG incl. K {32 + tail:.2f} + h_tilde 8 +"
          f" scratch 8 + u,v,w 24 + divergence 8): {nothing_released:.2f} B/cell (> S+; releasing or reusing"
          " the dead buffers is a design condition)")


if __name__ == "__main__":
    main()
