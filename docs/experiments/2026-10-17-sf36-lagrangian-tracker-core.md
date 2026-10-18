# SF-36: Lagrangian-label tracker core on the cube production stage

- Date: 2026-10-17 (activation 19:48Z; V100 campaign 22:33Z-23:17Z)
- Status: `complete — implementation claim established under the owner-amended criteria; three pre-registered criteria (A2's I, E, D at one case) not met as written and amended by the owner after measurement (disclosed)`
- Increment: [`SF-36-lagrangian-tracker-core.md`](../plans/active/lester-eq14/increments/SF-36-lagrangian-tracker-core.md)
- Decision record: [`2026-10-17-lagrangian-consumer-sigma4.md`](../decisions/2026-10-17-lagrangian-consumer-sigma4.md) (items 2, 4, 9)
- Theory: [`lester-2023-key-claims.md`](../theory/lester-2023-key-claims.md) §5 (sources A and B), §6, Rule 3
- Artifacts: [`artifacts/2026-10-17-sf36-lagrangian-tracker-core/`](artifacts/2026-10-17-sf36-lagrangian-tracker-core/README.md)
  (`scripts/`, `logs/`, `raw/`, `analysis/`)

Abbreviations (under `A = artifacts/2026-10-17-sf36-lagrangian-tracker-core/`): `CF` = `A/raw/controls/controls_full.log`
(64^3 `--full`), `CD` = `A/raw/controls/controls_diag.log`, `GH` = `A/raw/controls/gpu_host_full.log`, `BC` =
`A/analysis/bytecmp.md`, `MEM` = `A/analysis/mem.md`, `COST` = `A/analysis/cost.md`, `ST` = `A/analysis/stress.md`, `BIT`
= the increment's bitácora (16^3/32^3 measurements before the amendment). Every number below is read from one of these.

## Question

The increment's exact Goal: `Construir el tracker lagrangiano de producción sobre el stage del cubo (trazado hacia
adelante DP5(4) con tolerancia controlada sobre el spline v_s = k_s grad Phi_s, etiquetas por partícula, transporte de
covectores y divergencia acumulada, momentos por plano y por tiempo) y verificarlo con controles exactos, equivalencia
con el trazador host de SF-30 y la identidad c = v_s exp(-int div v_s dt), sin cambiar ningún bit de las salidas de
SF-35.`

The single claim allowed (spec): the implemented forward tracker integrates the intended equations — positions (SF-30
DP5(4) arclength with Hénon landing), travel time, the accumulated divergence `I = int div v_s dt` and covectors
(`d(grad psi)/ds = -(grad u)^T grad psi`). No claim about plume convergence, the production `lambda/h`, `alpha_L` or
`alpha_T` (SF-37, SF-39).

## Hypothesis

Pre-registered thresholds A-J (spec at `040e3ce`, activation row). Derived expectations (spec "Expected numerical
effect"): exact controls at roundoff; `lester2021` one-period return `~1e-9-1e-8` at `tol = 1e-8`; GPU vs host 0 to
`1e-12`; covector FD defect of order 2 in `delta`; identity residual `<= 1e-8` at `tol = 1e-10`. The last expectation
was wrong (see Result): it ignored that the C^2 spline's Hessian is only C^0.

## Build / environment

Integrated head `e81b118` (delivery branch `science/lester-sf36-lagrangian-tracker-core`; base `040e3ce`), V100
`Tesla V100-PCIE-32GB` GPU 0, CUDA 11.4.152, GCC 9.3.1, preset `v100-release`, mirror `scripts/remote --increment SF-36`
synced from clean worktrees. Base references (`~/sf36_base_refs`, orchestrator job `sf36-base-refs`) built from
`040e3ce` on the same mirror before any SF-36 source reached it. Code: `src/physics/particles/lagrangian_labels/`
(tracker, `--fmad=false`), `SlabTraceDevice.cuh` (verbatim move of the SF-35 device integrator), `SlabProductionSetup`
(P2, `require_inlet_labels`), `evaluate_point_hessian` (SF-28 spline, additive), driver `inlet_slab --track forward`.

## Config(s)

Controls: `k = 1`; layered `k = exp(0.5 cos 2 pi x1)`; synthetic failure field (A3); `lester2021` eps 1; `generic3d` eps
0.25; SF-18 Gaussian (`sigma2 = 1` normalized, `ell = 1/4`, seed 3001) eps 0.5, 1.0, 2.0; a C^inf analytic field with
exact derivatives (E1: `Phi = x1 + 0.05 sin 2 pi x2 cos 2 pi x1 + 0.05 sin 2 pi x3 sin 2 pi x1`,
`k = exp(0.5 cos 2 pi (x1 + x2 + x3))`). Defaults `tol = 1e-8`, `h_max = h/8`, `min_step = 1e-13`. Cost runs: `M = 256`
(65536 particles), planes and times {0.25, 0.5, 1, 2, 4}.

## Commands

`A/scripts/run_sf36.sh build/v100-release <part>` as detached jobs `sf36-controls`, `sf36-bytecmp`, `sf36-mem`,
`sf36-cost`, `sf36-stress` (exact commands in `A/logs/commands/`, job logs in `A/logs/jobs/`); digests by
`A/scripts/digest_sf36.py` (byte-identical to the host's). Fast tests: `ctest -R 'lagrangian|inlet_slab|interpolation'`.

## Outputs inspected

### A. Exact controls (`CF`, 64^3)

- A1 `k = 1`: transverse displacement, `tau - (X - X0)`, covector change and `|I|` all `0.000e+00` (640 records).
- A2 layered: transverse positions `9.7e-17`, covectors `1.2e-13`, `tau` vs an independent adaptive quadrature
  `5.65e-11` relative (threshold `1e-9`; 32^3 `4.77e-10`, 16^3 `4.6e-9` per `BIT`); `I` vs quadrature `7.4e-4`
  relative (**not met** as pre-registered, `1e-9`; RECORD after the amendment).
- A3 synthetic failure field: plane-pass histogram `88 40 32 32 0 32 32`, time-pass `120 40 32 32 0 0 32`, exactly as
  derived from the construction; `seed_backflow = 104` with zero weights there; `backflow_encounters = 24` per pass;
  `landings_discarded` asserted as a lower bound (retry count not derivable).

### B. Closed streamlines (`CF`)

`lester2021` eps 1, 64^3: one-period transverse return `3.97e-11` (threshold `10 tol = 1e-7`), 256/256 ok.

### C. GPU vs host (`GH`, 64^3)

Single-plane forward traces vs the host `trace_to_plane`, 4096 particles each on `k = 1`, `generic3d` 0.25, gauss 0.5,
gauss 2.0: max `|dx| = 0.000e+00`, statuses and `nfev` identical on every particle; `tau` (passive) differs by at most
`2.84e-14` (gauss 2.0).

### D. Covectors vs finite differences of the flow map (`CF`, 64^3, 256 seeds, `tol = 1e-12`, `h/32`)

| field, `tau` | pre-registered `delta` (8,4,2)e-3 | amended `delta` (1,0.5,0.25)e-3 |
|---|---|---|
| generic3d, 0.5 | met (orders >= 1.9) | PASS (>= 1.9) |
| generic3d, 2 | met | PASS |
| gauss 1.0, 0.5 | met | PASS |
| gauss 1.0, 2 | **not met** (orders 0.73 / 0.97 for a, 1.27 / 0.89 for b; RMS defect 5.2 -> 1.6) | PASS (orders 2.0) |

The pre-registered ladder is pre-asymptotic for the stretched flow at `tau = 2`; with smaller `delta` the defect falls
at order 2 with no `h_max` floor.

### E. Identity `a x b = v_s exp(-I)` (`CF`, `CD`)

- E1 (amended, blocking): C^inf analytic field, `tol = 1e-10`, `h_max = 1/128`: max residual `5.29e-11` (`<= 1e-8`).
  The covector and `I` equations and their implementation are right.
- E2 (RECORD): spline fields at 64^3, `tol = 1e-10`, `h/8`: `6.68e-6` (generic3d 0.25), `4.61e-5` (gauss 1.0), `1.54e-3`
  (gauss 2.0). As pre-registered (`<= 1e-8`) **not met** (16^3: `1.99e-4`, `8.77e-4`, `BIT`).
- `h_max` ladder at `tol = 1e-12` (64^3; max residual at `h/8, h/16, h/32, h/64`): generic3d `6.68e-6 -> 5.78e-8`
  (observed orders 2.78, 1.93, 2.14); gauss 1.0 `1.87e-5 -> 8.70e-7` (1.23, 1.56, 1.63); gauss 2.0 `1.00e-3 -> 3.31e-5`
  (1.88, 1.51, 1.53); the C^inf field stays at the roundoff floor (`1.6e-13`-`4.3e-13`, no order).
- Accuracy of `I` (DIAG-3, 64^3): `|I(h/8) - I(h/64)|` max `6.7e-6`, `1.3e-5`, `2.4e-5` against RMS `I` of `5.6e-4`,
  `1.4e-2`, `5.1e-2` (generic3d, gauss 1.0, gauss 2.0).

### F. SF-35 byte identity and regression (`BC`)

Head vs base (`040e3ce`) on the same mirror: `--construct trace` at 32/64 x eps 0.5/1.0 x host/GPU and eps 2.0 GPU at
64/128 — 10/10 cases `psi1.npy`, `psi2.npy` byte-identical, CASE/PLANE/BACKFLOW/GROWTH/RESIDUAL/TRANSPORT_STATUS/STATUS
lines identical (timings masked), exit codes equal (21 where the base has 21). `--production` 32^3 eps 0.25: 102/102
CASE/ORACLE/ORACLE_SUMMARY/EXTRA lines identical to the base and to SF-33's `ladder_0.25_c4/N32.log`. Full ctest and the
SF-27 trio (orchestrator job `sf36-ctest-full-bytecmp`, tree `fc3e5b4`, `logs/jobs/sf36-ctest-full-bytecmp.log`): 37/37
tests passed (2881.9 s), the 9 `inlet_slab_*` tests among them; `config_pspta_small`,
`config_streamfunctions_homogeneous` and `config_streamfunctions_continuation` outputs byte-identical to the base build
(manifests identical after removing time/date/duration lines). `git diff 040e3ce..` is empty on the frozen stack,
`src/multigrid`, `apps/closure_gate` and `src/physics/particles/{pspta,par2_adapter,streamline_tracker}`, and has no
removed line in `PeriodicTricubicBSpline.cuh`.

### G. P2 memory (`MEM`)

`--construct trace` 256^3 eps 2 GPU: head device peak `3771 MiB` (nvidia-smi, 200 ms) against the base `6239 MiB`, and
inside the probe band `3769 +- 2 %`.

### H, I. Spline Hessian and determinism

H (`interpolation_hessian`, part of the ctest suite): exact to `<= 2.5e-14` on cubic coefficient lattices, value and
gradient bitwise equal to `evaluate_point` (host and device). I (`GH`): two launches bitwise identical, block 64 vs 256
bitwise identical, positions/`tau`/`I`/statuses/`nfev` bitwise identical with covectors on vs off, device storage pointers
stable across `track`.

### J. Cost (`COST`)

65536 particles to `x1 = 4` and `t = 4`, every run `ok`:

| N | eps | plane pass [s] | time pass [s] | fevals / particle / unit length | device peak [MiB] |
|---|---|---|---|---|---|
| 64 | 1 | 1.25 (1.86 with covectors) | 1.58 (2.27) | 3286 | 651 (785) |
| 128 | 1 | 3.18 (4.11) | 4.03 (5.36) | 6556 | 891 (957) |
| 256 | 1 | 10.21 (11.78) | 11.83 (14.55) | 13096 | 3771 |
| 256 | 2 | 14.00 (15.50) | 18.97 (23.00) | 15359 | 3771 |

The work per unit length doubles with `N`: the step is capped by `h_max = h/8`, not by `tol`. The 256^3 device peak
is the stage's (3771 MiB, as in G); the tracker adds little.

### Stress (informative, `ST`)

Inlet-backflow field far outside the study's range (`sigma_Y^2 = 16`, `ell = 0.15`, 16^3, default settings): terminated
with exit 22 (`track_fail`) after 1228 s; plane pass 254 `cap_exceeded` and 2 ok (`nfev` `1.75e10`), time pass 256 ok;
at the plane `x1 = 0.5` the mean travel time is 93 and `ev_rms` overflows to `2.07e121` (particles lingering in slow
regions accumulate a huge `I`).

## Result

Classification per `docs/AGENTS.md`.

- **Established (confirmed in runs, within the caveats):** the forward tracker integrates the intended equations. Its
  positions reproduce the SF-30 host integrator bit for bit (C), the exact controls hold at roundoff (A1, A2 positions/
  covectors/`tau`, A3, B), the covector law agrees with finite differences of the flow map at order 2 (D, amended
  `delta`), and the identity `a x b = v exp(-I)` holds to `5e-11` where the field is smooth (E1). SF-35 outputs are
  byte-identical (F) and P2 lowers the 256^3 peak from 6239 to 3771 MiB (G).
- **Not met as pre-registered (recorded):** A2's `I` (`7.4e-4` vs `1e-9`), E on the spline fields (`6.7e-6`-`1.5e-3` vs
  `1e-8` at 64^3), and D at (gauss 1.0, `tau = 2`) with the pre-registered `delta`. The owner amended these criteria
  after measurement (spec, "Owner amendment"); the amended criteria pass.
- **Confirmed by the diagnostics (cause of the misses):** on the C^2 tricubic spline the Hessian is only C^0, so the
  passive `I` and covectors carry an error that falls with `h_max` at order ~1.2-2.8 and does not fall with `tol`; on a
  C^inf field the same code is at the roundoff floor. This is a property of the representation, not a defect of the
  tracker. The `delta` miss of D is pre-asymptotic finite differencing of a strongly stretched map.
- **Consequences for SF-37 (recorded, not established here):**
  - The Lagrangian `e_v = 1 - exp(-I)` that SF-37 V4 uses carries an `I` error of about 1 % (generic3d) to 0.05 %
    (gauss 2.0) of RMS `I` at 64^3 and `h/8`.
  - Covector norms (V5) are accurate to the E2 level.
  - RMS of `e_v` is not robust when particles linger near stagnation (stress run); percentiles are safer.
  - The tracker's cost is dominated by the `h_max` cap, about `1.3e4` field evaluations per particle and unit length
    at 256^3.

## Caveats

- One realization (seed 3001) for the Gaussian cases, `ell = 1/4`; controls at 64^3 at most (C, D, E, A on the V100).
- The amended criteria were chosen after the measurements (E1's field, D's `delta`); the original verdicts stay
  recorded as not met in the spec.
- The fast-tier (ctest) D runs at 32^3 with `h/16` and 16 seeds (wall time); the acceptance D is the 64^3 `--full` run.
- `tau` and `I` are passive: they are not in the error norm, so their accuracy is set by the step sequence.
- The stress field is not physically meaningful at 16^3; it only exercises termination and accounting.
- A pre-existing bug was found and left alone: `analysis::read_timeseries_csv` reads one extra all-zero row because
  `CsvTimeSeriesWriter` writes two header lines. It is relevant for SF-38's `AnalysisRunner` path.

## Next step

SF-37 (verification of the plume statistics on the cube at `sigma^2` in {1, 2.25, 4}) and SF-38 (long domain) become
eligible once this increment is merged and marked `done` on the default branch.
