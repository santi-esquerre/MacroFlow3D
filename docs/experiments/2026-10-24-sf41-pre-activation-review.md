# SF-41 pre-activation review of the acceptance thresholds A-H (code reading, adversarial reviews, design emulations)

- Date: 2026-10-24
- Status: `complete — design review before activation; NOT SF-41 evidence; no GPU run, no build; the emulations are numpy on CPU`
- Increment: [`SF-41-medium-b-lagrangian-stage.md`](../plans/active/lester-eq14/increments/SF-41-medium-b-lagrangian-stage.md) (State `pending`
  at the time of this review; base `master a3d98aec3136b26b826b55663e089b1dd89888c1`)
- Decision fed: [`2026-10-24-sf41-sf39-planning-decisions.md`](../decisions/2026-10-24-sf41-sf39-planning-decisions.md), addendum E8-E12
- Precedent: decision 2026-10-23 D21 (SF-38 threshold A amended before activation for an established cause)
- Theory: [`lester-2023-key-claims.md`](../theory/lester-2023-key-claims.md) (Rule 1: no `alpha_L` / `alpha_T` statement is made here)
- Artifact: [`artifacts/2026-10-24-sf41-pre-activation-review/`](artifacts/2026-10-24-sf41-pre-activation-review/README.md)

## Question

Before the activation row pre-registers SF-41's thresholds verbatim, does any threshold fail, or turn out unmeasurable, for an
established reason, as SF-38's threshold A would have (D21)? Which readings must the activation row supply, and which need a
specification change (owner decision, docs-only governance PR)?

## Method

1. Code reading at `a3d98ae` by five read-only agents (forward tracker, Pollock, `solve_head` / PCG / MG / `DirichletMedium`, the SF-28
   spline and its tests, the SF-36/37/38 evidence), each with file:line citations; the decisive facts were re-read by the orchestrator.
2. Two independent adversarial reviews of the orchestrator's draft (numerics lens: A, B, C1-C3, F; code-path and process lens: C4, C5,
   D, E, G, H), asked to refute each finding and to find new ones.
3. Design emulations in numpy (the artifact): the non-periodic not-a-knot spline, the DP5(4) tracker step sequence, an RT0 (Pollock)
   emulation, round-off probes. They predict mathematics, not GPU results; they are not SF-41 evidence.

Classes: **PF** predictable failure or unmeasurable as written (specification change); **RR** resolved by an explicit reading or a design
choice fixed before activation; **RISK** residual risk recorded.

## Findings

Code facts at `a3d98ae` (*confirmed in code*) and design-emulation numbers (*emulation*, artifact `analysis/`; commands in the artifact
README). "Resolution" names where the specification now states it (decision addendum E8-E12).

| thr | class | finding | evidence | resolution |
|---|---|---|---|---|
| C4 | PF | "particles started with a reversed head drop end `exit_inlet`; the Pollock non-periodic mode gives the same statuses" cannot be measured: a seed with non-positive flux returns `seed_backflow` untraced in both tracker passes, and both the tracker and the Pollock lattice API throw when no seed has positive flux | `LagrangianTrackerDevice.cuh:485-489, 639-642`; `LagrangianTracker.cu:105-107`; `LagrangianTrackerImpl.cuh:109`; `PollockComparator.cu:293` | E8: C4a outlet (`k = 1`), C4b inlet on the harmonic lobe field with exact stream function; *emulation* (`c4_lobe.txt`): `a = 0.5`, `X0 = 0.1`, 64 x 64 lattice filtered by `v1 >= 0.05`, `|psi0 - psi_sep| >= 0.02`: 2560 seeds (768 inlet, 1792 downstream) on both grids, 0 disagreements exact vs DP5-on-spline vs RT0 (Pollock wrong only within 4.6e-4 of the separatrix) |
| D | PF | `DirichletMedium`'s head is a local freed before return, no accessor; the spec lists the file read-only | `DirichletMedium.cu:165, 294-295`; `DirichletMedium.cuh:84-107` | E9: additive opt-in `keep_host_head`; G covers the flag-off bytes |
| D | PF (rule) | "(local)" for 128x32x32 and 512x64x64 conflicts with the owner's rule of no local grid above `32^3` | owner rule (2026-10-22) | E10: V100 |
| A ladder | RISK -> RR | near-face not-a-knot convergence is pre-asymptotic at 16/32/64; the verdict depends on the field: natural field `sin(2 pi 1.1 x + 0.3) + 0.5 cos(2 pi 0.6 x)` Hessian min order 1.7697 (`L1 = 1`), value 3.7879 (`L1 = 2`); 27-58 of 300 random 3-mode fields fail the gates (seeded, four amplitude laws); a `y/z`-dominated field passes but hardly tests the `x` closure (C1: `x` share 0.02) | `a_ladder.txt` (a), (b), (b2) | E11: field `exp(x1) sin(3 x1) (1 + 0.05 sin(2 pi x2) cos(2 pi x3))` (`x` share 0.95), predicted min orders 3.9024 / 4.3659 / 2.9042 / 3.3858 / 1.9154 (value max/RMS, gradient max/RMS, Hessian); recorded `x`-only sub-ladder to 128 |
| A CPU vs GPU | PF (if all quantities gated) -> RR | the not-a-knot end rows amplify a round-off difference between two prefilter algorithms by about 33 at `c_{-2}` and 8 at `c_{-1}`: full-pipeline gradient up to 1.9e-13 and Hessian up to 1.6e-12 at `h = 1/32` (C1 field), and even the SF-28 periodic FFT-vs-tridiagonal pair gives a full-pipeline Hessian of 3.2e-13 (SF-28 gated value and gradient only) | `a_ladder.txt` (f) | E10: gate the full-pipeline value and the evaluation-only value/gradient/Hessian at `1e-13`; record coefficients and full-pipeline derivatives |
| A reproduction | RR | through the prefilter the Hessian of a cubic carries round-off growing like `nx^2` (7.5e-13 to 2.9e-11); value <= 4.4e-15, gradient <= 2.7e-13 | `a_ladder.txt` (d) | E10: value and gradient gated (`1e-12`), Hessian recorded |
| A partition of unity | RR | value <= 5.3e-15 inside `[0, L1]` (gate `1e-14`); gradient 2.9e-14 to 3.9e-13 (SF-28's `1e-13` gradient gate would fail at fine `h`) | `a_ladder.txt` (e) | E10: value gated, gradient and margin recorded |
| A margins | RR | fixed physical margin points leave the evaluation domain on finer grids (`-h16/8 = -h64/2`) | `a_ladder.txt` (e) | E10: margin points at fixed `x/h` per grid, outside the ladder set |
| B | RISK | the interior difference is a round-off floor growing like `eps (lambda/h)^2` for the Hessian: 2.7e-14 (K-like spectrum) to 3.4e-12 (linearized head, the smoothest) at 1664x128x128 (gate `1e-11`; worst-case margin 2.9); per-knot end-effect factor 0.2679-0.2680, floor at about knot 15; at `lambda/h = 8` the `2 lambda` rule would not suffice (0.268^16) | `b_roundoff.txt` | none needed at `lambda/h = 32`; FFT in `y/z` kept (larger margin) |
| C1, C5 | RR | `f = 0` and `I = 0` exactly need `G1 = 1` on dyadic `h`, `L1`; with `k = 1` the linear initial guess is the exact discrete head, `r_0 = 0` exactly and the PCG returns at once (`rel := 0`, the `DirichletMedium` convention) | `solve_head.cu:41-42`; `pcg.cuh:125-132`; `DirichletMedium.cu:215`; exact dyadic arithmetic (review 1) | E10/E12 |
| C2 | RISK -> RR | a periodic `ln k` spline of a non-periodic profile fails (`tau` 1.347e-9 at `L1 = 4`); the SF-38 F2 profile phase-shifted fails (2.442e-9); the candidate `0.5 cos(2 pi 0.6 x1 + 0.3)` with both splines not-a-knot gives 5.023e-11; the emulation reproduces SF-38 F2 / SF-36 exactly (4.771e-10, 4.620e-9) | `c2_tau.txt` | E10/E12: profile, planes, `X0 = 0` fixed; `ln k` on the non-periodic geometry |
| C2 | RISK -> RR | cyclic line solves in `y/z` leave transverse round-off on `y/z`-constant data (spurious transverse Hessian 1.87e-11 relative against 2.32e-12 with FFT); C2's transverse and covector gates (`1e-12`) must not see it | `a_ladder.txt` (0); SF-38 F2 covectors 2.985e-14 with the SF-28 cuFFT path | E12: device prefilter = `x` line solve + FFT in `y/z` |
| C3 | PF (as literally applied) -> RR | on the C1/C2 fields every transverse shift is trivially exact (vacuous); on a written transverse field with amplitudes 0.5 / 0.3 the per-point relative gate of the SF-38 F3 method fails where `|v|` is about 0.05 (1.10e-14) | `c3_shift.txt` (V0) | E10: amplitudes 0.1 / 0.1 / 0.2, `|v|` in [0.2, 2.2]: max abs 1.1e-15, max rel 3.7e-15 (V3); dyadic points bitwise |
| E | PF (policy) | the pre-authorized fallback addresses only the solve phase; a non-solve phase over the limit would have led to R3 | spec text | E10: placement defect, corrected in SF-41 |
| E | RR | lean (b) solve peak `8 [7 n + 4 sum n_l]` bytes (60.571 B/cell): `pcg_solve` reads `b` once, no true-residual recompute, no `b_hat`; `K` must survive the velocity phase; MG coarsest 104x4x4 at 6656x256x256 (more PCG iterations, time only) | `pcg.cuh:100-190`; `velocity_from_head.cu:601-691`; `LongBoxStage.cu:32-43` | E12 |
| F | RISK low -> RR | SF-37 V4 (same `lambda/h` range): 45/45 cells strictly decreasing, last-pair orders 1.996-2.007, no floor; F left particles, weighting, order convention and blocks open | SF-37 artifact `analysis/v4v5/v4_cells.csv` | E10: `M = 100` (with `M = 256` the window `[0.1, 0.9]^2` puts 52 seeds per row exactly on transverse faces at `ny` 128 and 256, rejected by the RR12 seed-face assertion; found while writing this note), flux-weighted RMS, SF-37 RR4, 4 blocks |
| G | RR | the A/B application always runs both media; the runner's `DONE` signature and file set must not change for medium (a); the tracker's 7-slot status histogram must not grow | `lagrangian_medium_ab_main.cu:540-757`; `EnsembleRunner.cu:252-253, 900-909`; `LagrangianTrackerTypes.hpp:84` | E10/E12 |
| H | RR | `e2_b.csv`'s `non_ok` is a per-run total; exits go to a new file; the 20 SF-38 files are committed; Pollock exits through `x = 0` are impossible on medium (b) up to solver error (M-matrix: `U[0] > 0`; RT0 cannot reach a face with `u >= 0` moving upstream) | `lagrangian_medium_ab_main.cu:459-468`; `PollockTracker.cuh:262-272` | E10: expected 0 exits, 20/20 identical; `e2_a.csv` environment control |
| H | RR | Pollock histograms have 16 slots and clamp codes `>= 16`; the periodic repack drops `U[nx]` | `PollockComparator.cu:294-298`; `FaceFluxRepack.cuh:14-19` | E12: exit codes in `0..15`, additive non-periodic face view |
| process | RR | the validation commands had no base build / base references and built with `exec` (no GPU lock); the local `ctest -R 'interpolation|lagrangian'` would run existing cases above `32^3` | spec text; `scripts/remote` | E10: base build and references from a clean tree at the base, detached builds, local work `<= 32^3` |

## Outcome

Owner decisions (2026-10-24, the orchestrator's recommendations): E8 (C4 replaced by two exact controls), E9 (`DirichletMedium` opt-in
head copy), E10 (clarifications written into the specification), E11 (A's ladder field fixed by an exact emulation); E12 records the
design items fixed before activation. The specification's Goal is unchanged; no threshold value was changed; C4's reversed-drop clause
and D's reference access were replaced because they could not be met or measured.

## Caveats

- The emulations reproduce the mathematics of the chosen closure and integrator in double precision on the CPU (no FMA); GPU round-off
  can differ by small factors; margins quoted here are design margins, not measurements.
- The B surrogate fields (linearized head of an SF-18-like field, nonlinear drivers) bracket, but are not, the SF-19 potential of seed
  6001.
- Nothing here is acceptance evidence of SF-41; every threshold is measured by SF-41's own pre-registered campaign.
