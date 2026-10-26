# SF-41: medium (b) for the Lagrangian consumer, lean Dirichlet stage, non-periodic-in-x1 spline and x-exit handling

- Date: 2026-10-25 (first evidence job `sf41-basebuild` started 2026-10-24T23:44:20Z; last job `sf41-final` finished 2026-10-25T04:22:29Z)
- Status: `measurement complete; every pre-registered part ran to completion; the verdict per criterion is in "Result"; the increment State is set by the orchestrator after FINAL_AUDIT and by human review, not by this note`
- Increment: [`SF-41-medium-b-lagrangian-stage.md`](../plans/active/lester-eq14/increments/SF-41-medium-b-lagrangian-stage.md)
- Decision records: [`2026-10-24-sf41-sf39-planning-decisions.md`](../decisions/2026-10-24-sf41-sf39-planning-decisions.md) (owner decisions E1-E12), parent records
  [`2026-10-17-lagrangian-consumer-sigma4.md`](../decisions/2026-10-17-lagrangian-consumer-sigma4.md) (items 2, 4, 7; stop rules R3, R4) and
  [`2026-10-23-sf38-activation-decisions.md`](../decisions/2026-10-23-sf38-activation-decisions.md) (D1, D4, D9, D11); pre-activation review (design emulations, numpy, not SF-41 evidence):
  [`2026-10-24-sf41-pre-activation-review.md`](2026-10-24-sf41-pre-activation-review.md)
- Predecessor note (structure; medium (b) selected there): [`2026-10-23-sf38-long-domain-infrastructure.md`](2026-10-23-sf38-long-domain-infrastructure.md)
- Theory: [`lester-2023-key-claims.md`](../theory/lester-2023-key-claims.md) (Rule 1: `alpha_T` is neither accepted as physical by default nor presupposed zero; **this note makes no `alpha_L` / `alpha_T` statement of any kind and does not read any dispersion number as physical or numerical**; `D_P`, `sigma_T^2`, the tilts `theta` and the V3b class are recorded inputs only), [`beaudoin-de-dreuzy-2013-key-claims.md`](../theory/beaudoin-de-dreuzy-2013-key-claims.md) (the long, transversally periodic Dirichlet-in-`x` configuration the medium (b) reproduces)
- Artifacts: [`artifacts/2026-10-25-sf41-medium-b-lagrangian-stage/`](artifacts/2026-10-25-sf41-medium-b-lagrangian-stage/README.md) (`preregistration.md`, `scripts/`, `raw/`, `analysis/`)
- Commits: base `master 2cf48c19dac641e1ce8c823c5cf2d6165ac66757` (PR #64 merged); activation A0 `fa3f253d0e90143a910de092258ecc4cb1dbaf8d` (committed 2026-10-24T20:43:21-03:00 = 2026-10-24T23:43:21Z, *derived*;
  pre-registration `preregistration.md` committed there, sha256 `8f424fa01d3eaf75743b7361ab0e6e508bd47b7da209be844d4fb33ccabd29d8`, **re-computed at the base of this note (`sha256sum` of the working-tree file and of `git show fa3f253:...`) and equal to the value in the activation row**);
  **source head H1 `269c658687816788fdaf65f9d5b1307d87376b6a`** (integration I1 of the accepted nodes; `SF41_COMMIT=269c658...` in every head `run.env`; every head job echoes `TREE=269c658687816788fdaf65f9d5b1307d87376b6a`,
  `R:job_logs/sf41-{build,controls,cross,prodgrid,evladder,abrerun,final}.log`, line 5 of each); imported-evidence commit `f91ace2cc9094f9a3d1da625c8acc855c27ccf84` plus `9a48e29` (adds the 346 gitignored files the first import omitted; this note is based on `9a48e29`). The mirror had no `.git` (`TREE_git=absent` in `run.env`); the source identity of the head parts is the
  tree echo plus `TREE_src_digest=821e4feec2e8895d392c754c049da0a1044172a453097b5fb8f5c099f8168243` (the same digest in the superseded build attempt 1 and in attempt 2: `R:sf41_campaign/build/results.txt`, `R:sf41_campaign_superseded/build_attempt1_tmpfull/results.txt`).
  Integrated commits (`git log --reverse 2cf48c1..269c658`; the original node commits are from the orchestrator's map. They exist in the local object store as unpublished worker-branch commits (`git cat-file -t` = `commit` for all 17), not on the published branch; I compared each with its H1 commit by `git show <commit> | git patch-id --stable`: **15 of 17 patch-ids are equal**; the two that differ, `043dae5` (N4, H1 `ca0edca`) and `bc869a3` (N5, H1 `fb6021c`), are the two for which the orchestrator reports context-only conflict resolutions in I1 (that explanation is the orchestrator's; I did not diff them further)):

| H1 commit | content (subject) | original node commit | patch-id (`--stable`) equal to H1's |
|---|---|---|---|
| `fa3f253` | activation: State active, pre-registration, thresholds A-H verbatim | A0 | - (not a cherry-pick) |
| `949d53a` | N1 non-periodic-in-x1 tricubic B-spline (C-SPLINE) and threshold-A program | N1 `3dd6f8e` | yes |
| `d97a4b5` | N2 `pcg_solve_external` and `DirichletMedium` `keep_host_head` | N2 `25ea1cd` | yes |
| `9964aa3` | N2 `DirichletLongBoxStage` (lean stage, flow part) and its tests | N2 `ef7401e` | yes |
| `ed8fb6a` | N2c 4-level D regression in the fast tier, V100 `--diag` mode | N2c `d1b9ff8` | yes |
| `63b79ff` | N2c empty reference head (stale `DirichletMedium` object) made visible; D harness no longer vacuous | N2c `164682d` | yes |
| `ca0edca` | N4 stage spline phase S3 (non-periodic spline of `f` and `ln k`, fused option), `--memory-model` | N4 `043dae5` | **no** |
| `63875c3` | N3a Pollock core and comparator non-periodic in x (exit codes 8 / 9) | N3a `01d3fc2` | yes |
| `70c16dc` | N3a tests, lobe reference (C4b), ctest `lagrangian_pollock_nonperiodic` | N3a `3a00fe0` | yes |
| `fd7a953` | N3b non-periodic field functors and x-exit handling in the forward tracker | N3b `486e45e` | yes |
| `6a9a81f` | N3b controls program C1-C4 (`dirichlet_controls_tests`) | N3b `3d1bda8` | yes |
| `616583a` | N1b P-CROSS program (threshold B) and ctest fast tier | N1b `fab6414` | yes |
| `3a3bcf1` | N6 A/B medium (b) Pollock on the x-non-periodic faces; `exits_b.csv` | N6 `574a501` | yes |
| `fb6021c` | N5 stage exposes the SF-18 report and a tracker `T` mem line | N5 `bc869a3` | **no** |
| `7207d3c` | N5 `lagrangian_ensemble --medium b`, multi-pass `--hmax-div`, `flow_stats.json` | N5 `2dee9f2` | yes |
| `a64b328` | N5 ctest `lagrangian_ensemble_medium_b` | N5 `7ce48f4` | yes |
| `2faf289` | N7 campaign tooling (`run_sf41.sh`, verdict scripts A-H, F bootstrap matrix, README) | N7 `4989fdf` | yes |
| `269c658` | N7 corrective: D memory recorded, not gated | N7 `0256bb0` | yes |

- Host/GPU (`R:sf41_campaign/controls/run.env`, identical in the six head parts and in the build part: `HOST`, `GPU_NAME`, `NVIDIA_DRIVER`, `CUDA_VERSION_NVIDIA_SMI`, `NVCC`, `GCC`, `TREE_src_digest`, `SF41_COMMIT` compared line by line, checked by the author of this note): `localhost.localdomain`, `Tesla V100-PCIE-32GB` (banner `32510MiB`), `CUDA_VISIBLE_DEVICES=0`, NVIDIA driver `470.239.06`, `nvidia-smi` banner CUDA 11.4, `nvcc` `11.4`,
  `gcc (GCC) 9.3.1 20200408 (Red Hat 9.3.1-2)`; host RAM `134590115840` B (`free -b`, e.g. `R:sf41_campaign/prodgrid/E4.free`). The `nvidia-smi` banner and the sampler clocks are host-local time, three hours behind the UTC stamps (*derived*, table "Binary md5" below; the E4 sampler's first sample
  `2026/10/24 23:26:54.9` against the E3 end `2026-10-25T02:26:49Z` of `walls.tsv`).
- Binaries (`build/v100-release`, preset `v100-release`): head md5 (`R:sf41_campaign/build/binaries.md5`, 50 binaries, and the nine `BIN_MD5_*` of the head `run.env`) and base md5 (`R:sf41_base_refs/run.env`) in the table of section "Parts" below (`ANA/parts.md`); the final R-set item `IDENTICAL binary:md5_differs head=ded79b8a... base=aae63d28...` confirms head != base build for `inlet_slab`.

Abbreviations: `A` = `artifacts/2026-10-25-sf41-medium-b-lagrangian-stage/`; `R:` = `A/raw/`; `ANA` = `A/analysis/` (`parts.md`, `A.md`, `B.md`, `C.md`, `D.md`, `E.md`, `F.md`, `G.md`, `H.md`, written by `A/scripts/note_tables.py`; `bootstrap_blocks_f.*` is the committed N7 matrix, untouched);
`PRE` = `A/preregistration.md` (RR0-RR12, C-SPLINE, C-STAGE, C-EXIT, C-RUNNER-B, C-AB); `SPEC` = the increment specification; `BIT` = its bitacora rows. Every number below is copied from a cited committed file under `R:` or computed by `A/scripts/note_tables.py` from `raw/`
(outputs in `ANA`, regenerated byte-identically by `python3 A/scripts/note_tables.py A/raw A/analysis`; the tables below are verbatim copies of those outputs). Labels: *measured* = read from a raw file; *derived* = arithmetic on measured values (formula given);
*not in the imported evidence* = absent from `raw/`. 1 MiB = 1048576 B. Box units: the transverse period is 1, `lambda = ell`. Verdict-script names (`a_verdict.py` ... `h_verdict.py`, `g_compare.py`) are those of `A/scripts/`.

## Question

The increment's exact Goal: `Llevar el consumidor lagrangiano al medio (b) de Dirichlet en x: stage austero (solve_head con la ubicación de buffers de SF-40, cabeza bit a bit igual a la de SF-38, pico de memoria de dispositivo medido <= 30000 MiB en 3328x384x384 y 6656x256x256), spline cúbico no periódico en x1 del potencial, salida explícita por las caras x en el tracker y en el comparador Pollock, e integración en el runner de ensemble; verificado con controles exactos, representación cruzada contra el spline periódico y escalera de e_v, sin cambio de bytes en los caminos periódicos.`

Claims allowed (SPEC "Scientific or engineering intent"): (a) a lean medium (b) stage whose head is bit for bit SF-38's `DirichletMedium` head and whose measured device peak fits 3328x384x384 and 6656x256x256 within 30000 MiB; (b) a cubic B-spline non-periodic in `x1` that satisfies the SF-28 unit contracts and coincides with the periodic SF-28 spline in the interior
up to an end effect that decays from the faces; (c) explicit `x`-exit handling in the tracker and the Pollock comparator for non-periodic media with the periodic paths byte-identical; (d) the consumer on medium (b) passes exact controls and its Lagrangian `e_v` converges under grid refinement at `sigma^2` 1 and 4; (e) the ensemble runner runs medium (b) with the Gate 5 outputs.
Not allowed: any `alpha` estimate (SF-39; Rule 1), any change of SF-18/SF-19/SF-28 numerics or of the periodic paths' outputs, any claim about the production `lambda/h` beyond the transfer argument of E2.

## Hypothesis

Nature of the change (new representation and stage, verified by exact controls, cross-representation and refinement): SPEC "Expected numerical effect": no change in existing outputs (periodic spline, cube and medium (a) paths, closure gate, inlet slab, Par2: byte-identical); the SF-38 A/B outputs of medium (b) change only for particles that leave through `x = 0` (expected none, counted);
interior coincidence with the periodic representation up to an end effect decaying geometrically from the faces with the per-knot factor `2 - sqrt(3)` (*derivation*); `e_v` of order about 2 in `h` as in SF-37 V4. Pre-registered thresholds (SPEC "Acceptance thresholds", copied verbatim into the activation row of the bitacora; reading rules in `PRE`):

| criterion | pre-registered rule (abridged; the verbatim text is the activation row) |
|---|---|
| A spline contracts | non-periodic spline: node interpolation `<= 1e-12`; partition of unity `<= 1e-14` abs; reproduction of cubics in `x1` (value, gradient `<= 1e-12`); order ladder `f = exp(x1) sin(3 x1) (1 + 0.05 S)` on 32x16x16, 64x32x32, 128x64x64: min orders value `>= 3.8`, gradient `>= 2.8` (max and RMS), Hessian `>= 1.8` (max); CPU vs GPU `<= 1e-13`; Hessian-path bitwise; no allocation; shift `<= 1e-14`; evaluation domain `[-h/2, L1 + h/2)` |
| B cross-representation | SF-18 `sigma^2 = 4`, 1664x128x128, seed 6001: non-periodic vs periodic spline of the SF-19 `h_tilde`, `>= 2 lambda` from the faces: value, gradient, Hessian `<= 1e-11`; decay factor (expected ~0.268), `ln k` recorded |
| C exact controls | C1 `k = 1`, C2 layered `k(x1)` (`tau` `<= 1e-9`, transverse and covectors `<= 1e-12`), C3 shift `<= 1e-14` on a written transverse field, C4a outlet and C4b inlet exits (counts exact), C5 solver (`rel <= 1e-10`, balance `<= 1e-8`) |
| D lean stage bitwise | head byte for byte equal to `DirichletMedium`'s and `B_SOLVER` line identical at 128x32x32, 512x64x64, 3328x256x256 (mapped-host path also at the two small sizes) |
| E production feasibility | seed 4001, E1 2048x256x256 `eps` 1; E2, E3 3328x384x384 `eps` 1.5, 2; E4 6656x256x256 `eps` 2: solve `rel <= 1e-10`, balance `<= 1e-8`, PID peak `<= 30000` MiB (mapped-host head fallback if needed), every non-solve phase `<=` the solve phase; exits, wall, RSS, per-phase bytes recorded |
| F `e_v` ladder | seeds 6101-6103, `eps` 1 and 2, `lambda/h` 16-128 (384x32x32 to 3072x256x256): RMS `e_v` strictly decreasing and last-pair order `>= 1.8` per (eps, seed, plane); `D_P`, `SE_D`, V3b recorded |
| G regression | full V100 `ctest` green; SF-27 trio and SF-38 20-run R set byte-identical to base binaries; medium (a) `lagrangian_ensemble` run and `lagrangian_medium_ab` run byte-identical (`timings.json` excluded); the two named periodic interpolation tests unchanged and green |
| H recorded | SF-38 A/B re-run with the exit-aware Pollock (both amplitudes, seeds 4001-4010): `x = 0` exit counts; `e2_b.csv` byte-identical where zero (expected zero exits and 20 of 20 identical files); `C_STAT` recomputed where not |

Failure policy (SPEC): A, B, C, D or G not met = defect (BLOCKING); E per its policy (C0 fallback, mapped-host head, R3 for the owner; a non-solve phase above the solve phase is a placement defect); F not met at an amplitude = the consumer is not defensible on medium (b) there (stop, owner);
H with exits is recorded (the SF-38 decision is not re-opened). Pre-activation design predictions (SPEC / `PRE` / owner decisions E8-E12; numpy emulations, not SF-41 evidence), compared with the measurements in each subsection: A minimum orders 3.9024 / 4.3659 / 2.9042 / 3.3858 / 1.9154 and x-only 64->128 orders 3.98 / 2.98 / 1.99;
B decay factor about 0.268 and an interior round-off floor (Hessian 2.2e-12 worst case); C2 `tau` about 5e-11; C3 about 1.1e-15 absolute and 3.7e-15 relative; C4b 768 `exit_inlet` and 1792 downstream; the lean-stage solve peak 60.57 B/cell; H zero `x = 0` exits and 20 of 20 identical `e2_b.csv`.

## Build / environment

- Base build: job `sf41-basebuild` (`cmake --preset v100-release` then `cmake --build build/v100-release -j`), `[283/283]`, `BUILD_EXIT=0`, `ctest -N` `Total Tests: 46`, tree `2cf48c1` (`TREE=2cf48c1 BASE=2cf48c19dac641e1ce8c823c5cf2d6165ac66757`), md5 of the four base binaries copied to `~/sf41_base_refs/bin` (`R:job_logs/sf41-basebuild.log`).
- Head build (attempt 2, `TMPDIR=$HOME/sf41_tmp`): job `sf41-build` = `run_sf41.sh build/v100-release build`: `rm -rf build/v100-release && cmake --preset v100-release && cmake --build ... -j`; part record `R:sf41_campaign/build/`: `results.txt` `BUILD_VERDICT PASS (binaries recorded; the head parts verify them)`, `BUILD_BINARIES n=50`, `ctest_N.log` `Total Tests: 52` and `ctest_names.txt` (52 names), `clean_build.log` ending `[303/303]`, `walls.tsv` `clean_build 311` s, `PART_EXIT` 0, `binaries.md5` (50 binaries; its md5 for the nine binaries of the table below equal the head `BIN_MD5_*`); the same facts are echoed in `R:job_logs/sf41-build.log`.
  The six extra ctest entries are `interpolation_nonperiodic_x`, `lagrangian_cross_representation`, `lagrangian_dirichlet_long_box_stage`, `lagrangian_pollock_nonperiodic`, `lagrangian_dirichlet_controls`, `lagrangian_ensemble_medium_b` (`git diff 2cf48c1 269c658 -- CMakeLists.txt` adds exactly these six `add_test` lines plus two new library sources and six new test executables; checked by the author of this note).
  Every head part verified the md5 of 50 binaries before running (`BINARIES_CHECK=ok(50 binaries, md5file_sha256=82751b5ebafa457d)` in the six head `run.env`; the prefix equals the `sha256sum` of `R:sf41_campaign/build/binaries.md5` (checked by the author of this note) and the `BUILD_BINARIES sha256=82751b5ebafa457d772421f...` of the build part).
- Nothing else ran on the GPU during a sampled or exclusive run: 42 `.precheck` files in `raw/`, all `precheck: no compute process on GPU 0` (`find raw -name '*.precheck'`, checked by the author of this note); one job at a time on GPU 0. No `resumed.txt` exists (`find raw -name resumed.txt` finds none): no run was re-run.
- Integrity: `sha256sum -c --ignore-missing sha256.txt` in each of the nine part directories (`sf41_base_refs`, `sf41_campaign/{build,controls,cross,prodgrid,evladder,abrerun,final}` and the superseded attempt-1 directory) reports 0 non-OK lines (section "Evidence re-verification"); `.npy` files are absent by design (862 listed in the per-part `sha256.txt`; *derived* as the sum of `.npy` lines: prodgrid 68, final 193, baserefs 193, evladder 408, other parts 0; `find raw -name '*.npy'` finds none).
- Orchestrator checks (not in `raw/`, reported as received): the tarball sha256 and transfer record of the README section "Evidence import".
- Tooling (committed at H1): `A/scripts/run_sf41.sh` (parts `build`, `baserefs`, `controls`, `cross`, `prodgrid`, `evladder`, `abrerun`, `final`), `a_verdict.py` ... `h_verdict.py`, `g_compare.py`, `sf41lib.py`, the SF-38 tools copied byte for byte (`pid_peak.py`, `mem_verdict.py`, `json_strip_cmp.py`, `lines_cmp.py`, `d_checks.py`, `ab_decision.py`), `make_bootstrap_blocks_f.py`, described in `A/README.md`.

## Config(s)

- **A (RR3)**: `interpolation_nonperiodic_x_tests --full` and `compute-sanitizer --tool memcheck interpolation_nonperiodic_x_tests --full --only domain`.
- **B (RR4)**: `sf41_cross_representation_tests --only B` (SF-18 Gaussian, `sigma^2 = 1` with `normalize_variance`, `eps = 2`, `ell = 0.25`, `lambda/h = 32`, seed 6001, 1664x128x128, `L1 = 13`; `B_CONFIG` line).
- **C (RR5)**: `dirichlet_controls_tests --full` (C1-C4, tracker unfused and fused, Pollock), `pollock_nonperiodic_tests --only C4`, `dirichlet_long_box_stage_tests --only C5`.
- **D (RR6)**: `dirichlet_long_box_stage_tests --only D` (sampled), plus the closed-form memory model runs `dirichlet_long_box_stage_tests --memory-model 3328,384,384 --verify-cufft` and the `6656,256,256` analogue.
- **E (RR7), verbatim** (`R:job_logs/sf41-prodgrid.log`, `+ [E1]` ... lines): `lagrangian_ensemble --medium b --grid <G> --eps <E> --ell <L> --seed0 4001 --nr 1 --window 0.1,0.9,0.1,0.9 --np-side 100 --planes auto --times auto --tol 1e-8 --hmax-div 8 --fused-splines --single-launch --twin on --pollock-bridge on --out <dir>`
  with `(G, E, L)` = E1 `(2048,256,256; 1; 0.0390625)`, E2 `(3328,384,384; 1.5; 0.041666666666666664)`, E3 `(3328,384,384; 2; 0.08333333333333333)`, E4 `(6656,256,256; 2; 0.25)`; `--x0` not passed (default `5 lambda` for medium (b); the `RESOLVED` line of E4 shows `x0=1.25`).
- **F (RR8), verbatim** (`R:job_logs/sf41-evladder.log`): `lagrangian_ensemble --medium b --grid <384,32,32 | 768,64,64 | 1536,128,128 | 3072,256,256> --eps <1|2> --ell 0.5 --seed0 <6101|6102|6103> --nr 1 --window 0.1,0.9,0.1,0.9 --np-side 100 --planes 3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5 --times auto --tol 1e-8 --hmax-div 8 --fused-splines --single-launch --twin off --pollock-bridge on`
  on 384x32x32, 768x64x64, 1536x128x128, 3072x256x256 (24 runs, names `F_e<eps>_s<seed>_g<ny>`).
- **G (RR9)**: full `ctest --test-dir build/v100-release --output-on-failure`; the SF-27 trio and the SF-38 20-run R set with head and base binaries (`R:sf41_base_refs/`); `g_runner_a`: `lagrangian_ensemble --grid 512,64,64 --eps 2 --ell 0.125 --seed0 4001 --nr 2 --window 0.1,0.9,0.1,0.9 --np-side 100 --planes auto --times auto --tol 1e-8 --hmax-div 8 --fused-splines --single-launch --twin on --pollock-bridge on`;
  `g_ab`: `lagrangian_medium_ab --grid 512,64,64 --ell 0.125 --eps 2 --seed0 4001 --nr 2`.
- **H (RR10), verbatim** (`R:job_logs/sf41-abrerun.log`): `lagrangian_medium_ab --grid 2048,256,256 --ell 0.0390625 --eps 1 --seed0 4001 --nr 10` (`ab_s1`) and `lagrangian_medium_ab --grid 3328,256,256 --ell 0.125 --eps 2 --seed0 4001 --nr 10` (`ab_s4`).

## Commands

All head jobs are detached `scripts/remote --increment SF-41 run <job> -- "..."` + `wait` calls, one at a time on the shared mirror `~/MacroFlow3D-SF-41` (the exact command strings are line 4 of each log in `R:job_logs/`; each is `TMPDIR=$HOME/sf41_tmp SF41_COMMIT=269c658... bash A/scripts/run_sf41.sh build/v100-release <part>` after `echo TREE=269c658...`).
The job windows, parts, GPU-hours, per-run walls and binary md5 are in the following tables (`ANA/parts.md`, verbatim). GPU-hours are the wall hours of the part (one GPU, one job at a time; *derived* from `RECORDED part_seconds`), as decision E1-E3 asks.

#### Jobs (every file of `raw/job_logs/`, status from `raw/job_status/`; `window` = finished minus started, derived)

| job | evidence | started (UTC) | finished (UTC) | window s | finished exit | status STATE/EXIT_CODE | TREE echo |
|---|---|---|---|---|---|---|---|
| sf41-abrerun | yes | 2026-10-25T02:39:21Z | 2026-10-25T03:25:13Z | 2752 | 0 | succeeded/0 | 269c658687816788fdaf65f9d5b1307d87376b6a |
| sf41-basebuild | yes | 2026-10-24T23:44:20Z | 2026-10-24T23:47:11Z | 171 | 0 | succeeded/0 | 2cf48c1 BASE=2cf48c19dac641e1ce8c823c5cf2d6165ac66757 |
| sf41-baserefs | yes | 2026-10-25T01:47:25Z | 2026-10-25T01:51:01Z | 216 | 0 | succeeded/0 | 0256bb0 SCRIPTS_ONLY_FOR_BASEREFS BASE=2cf48c1 |
| sf41-build | yes | 2026-10-25T02:10:44Z | 2026-10-25T02:16:10Z | 326 | 0 | succeeded/0 | 269c658687816788fdaf65f9d5b1307d87376b6a |
| sf41-controls | yes | 2026-10-25T02:16:40Z | 2026-10-25T02:20:03Z | 203 | 0 | succeeded/0 | 269c658687816788fdaf65f9d5b1307d87376b6a |
| sf41-cross | yes | 2026-10-25T02:20:40Z | 2026-10-25T02:21:14Z | 34 | 0 | succeeded/0 | 269c658687816788fdaf65f9d5b1307d87376b6a |
| sf41-evladder | yes | 2026-10-25T02:29:56Z | 2026-10-25T02:38:52Z | 536 | 0 | succeeded/0 | 269c658687816788fdaf65f9d5b1307d87376b6a |
| sf41-final | yes | 2026-10-25T03:25:41Z | 2026-10-25T04:22:29Z | 3408 | 0 | succeeded/0 | 269c658687816788fdaf65f9d5b1307d87376b6a |
| sf41-pre-e | no (pre-check) | 2026-10-25T01:30:22Z | 2026-10-25T01:40:58Z | 636 | 0 | succeeded/0 | - |
| sf41-pre-n1-full | no (pre-check) | 2026-10-25T00:08:34Z | 2026-10-25T00:09:32Z | 58 | 0 | succeeded/0 | - |
| sf41-pre-n1b | no (pre-check) | 2026-10-25T00:41:29Z | 2026-10-25T00:41:50Z | 21 | 0 | succeeded/0 | - |
| sf41-pre-n2 | no (pre-check) | 2026-10-25T00:21:20Z | 2026-10-25T00:25:49Z | 269 | 0 | succeeded/0 | - |
| sf41-pre-n2c-clean | no (pre-check) | 2026-10-25T01:24:11Z | 2026-10-25T01:28:59Z | 288 | 0 | succeeded/0 | - |
| sf41-pre-n2c-d | no (pre-check) | 2026-10-25T01:16:33Z | 2026-10-25T01:18:49Z | 136 | 0 | succeeded/0 | - |
| sf41-pre-n2c-diag | no (pre-check) | 2026-10-25T01:12:30Z | 2026-10-25T01:15:07Z | 157 | 0 | succeeded/0 | - |
| sf41-pre-n3b | no (pre-check) | 2026-10-25T00:52:43Z | 2026-10-25T00:57:52Z | 309 | 0 | succeeded/0 | - |
| sf41-prodgrid | yes | 2026-10-25T02:21:43Z | 2026-10-25T02:29:19Z | 456 | 0 | succeeded/0 | 269c658687816788fdaf65f9d5b1307d87376b6a |

#### Parts (start = `run.env` DATE_UTC; PART_EXIT; runs = rows of `exit_codes.tsv`; part_seconds / part_runs_seconds = `RECORDED` line of `results.txt`)

| part | start (UTC) | PART_EXIT | runs | non-zero rc | part_seconds | part_runs_seconds | SF41_COMMIT | BINARIES_CHECK |
|---|---|---|---|---|---|---|---|---|
| baserefs | 2026-10-25T01:47:26Z | 0 | 25 | 21 | 215 | 196 | 2cf48c1 | base binaries: md5 recorded only |
| build | 2026-10-25T02:10:45Z | 0 | 2 | - | 325 | 311 | 269c658 | not_applicable(build part) |
| controls | 2026-10-25T02:16:44Z | 0 | 8 | - | 203 | 172 | 269c658 | ok(50 binaries, md5file_sha256=82751b5ebafa457d) |
| cross | 2026-10-25T02:20:45Z | 0 | 1 | - | 33 | 15 | 269c658 | ok(50 binaries, md5file_sha256=82751b5ebafa457d) |
| prodgrid | 2026-10-25T02:21:47Z | 0 | 4 | - | 456 | 424 | 269c658 | ok(50 binaries, md5file_sha256=82751b5ebafa457d) |
| evladder | 2026-10-25T02:30:01Z | 0 | 24 | - | 535 | 450 | 269c658 | ok(50 binaries, md5file_sha256=82751b5ebafa457d) |
| abrerun | 2026-10-25T02:39:26Z | 0 | 2 | - | 2751 | 2728 | 269c658 | ok(50 binaries, md5file_sha256=82751b5ebafa457d) |
| final | 2026-10-25T03:25:45Z | 0 | 26 | 21 | 3407 | 3380 | 269c658 | ok(50 binaries, md5file_sha256=82751b5ebafa457d) |
| build attempt 1 (superseded, not counted) | 2026-10-25T02:02:40Z | 1 | 2 | 1 | 325 | 314 | 269c658 | - |

derived (GPU-hours = wall hours: one GPU, one job at a time): sum of `part_seconds` over baserefs + build (attempt 2) + the six head parts = 7925 s = 2.201 h; sum of `part_runs_seconds` = 7676 s = 2.132 h; adding the base build job window (`sf41-basebuild`, 2026-10-24T23:44:20Z to 2026-10-24T23:47:11Z = 171 s) gives 8096 s = 2.249 h; the superseded build attempt 1 (325 s) is not counted.

| part | GPU-hours (part_seconds / 3600, derived) |
|---|---|
| baserefs | 0.060 |
| build | 0.090 |
| controls | 0.056 |
| cross | 0.009 |
| prodgrid | 0.127 |
| evladder | 0.149 |
| abrerun | 0.764 |
| final | 0.946 |
| build attempt 1 (superseded, not counted) | 0.090 |

#### Per-run wall seconds (`walls.tsv`; the 24 runs of `evladder` are summarised in F.md)

| part | run | wall s | end (UTC) |
|---|---|---|---|
| controls | A_full | 5 | 2026-10-25T02:17:03Z |
| controls | A_memcheck | 10 | 2026-10-25T02:17:15Z |
| controls | C_full | 12 | 2026-10-25T02:17:29Z |
| controls | C4_pollock | 4 | 2026-10-25T02:17:36Z |
| controls | C5 | 4 | 2026-10-25T02:17:42Z |
| controls | D | 130 | 2026-10-25T02:19:56Z |
| controls | MEMMODEL_a | 3 | 2026-10-25T02:19:59Z |
| controls | MEMMODEL_b | 4 | 2026-10-25T02:20:03Z |
| cross | B | 15 | 2026-10-25T02:21:14Z |
| prodgrid | E1 | 35 | 2026-10-25T02:22:37Z |
| prodgrid | E2 | 112 | 2026-10-25T02:24:34Z |
| prodgrid | E3 | 131 | 2026-10-25T02:26:49Z |
| prodgrid | E4 | 146 | 2026-10-25T02:29:19Z |
| abrerun | ab_s1 | 1110 | 2026-10-25T02:58:11Z |
| abrerun | ab_s4 | 1618 | 2026-10-25T03:25:13Z |
| final | ctest | 3182 | 2026-10-25T04:19:01Z |
| final | trio_config_pspta_small | 5 | 2026-10-25T04:19:06Z |
| final | trio_config_streamfunctions_homogeneous | 3 | 2026-10-25T04:19:09Z |
| final | trio_config_streamfunctions_continuation | 3 | 2026-10-25T04:19:13Z |
| final | r_trace_N32_e0.5_host | 6 | 2026-10-25T04:19:19Z |
| final | r_trace_N32_e0.5_gpu | 5 | 2026-10-25T04:19:24Z |
| final | r_trace_N32_e1.0_host | 6 | 2026-10-25T04:19:30Z |
| final | r_trace_N32_e1.0_gpu | 6 | 2026-10-25T04:19:36Z |
| final | r_trace_N64_e0.5_host | 16 | 2026-10-25T04:19:52Z |
| final | r_trace_N64_e0.5_gpu | 6 | 2026-10-25T04:19:58Z |
| final | r_trace_N64_e1.0_host | 17 | 2026-10-25T04:20:16Z |
| final | r_trace_N64_e1.0_gpu | 8 | 2026-10-25T04:20:24Z |
| final | r_trace_N64_e2.0_gpu | 8 | 2026-10-25T04:20:32Z |
| final | r_prod | 17 | 2026-10-25T04:20:49Z |
| final | r_fwd_smoke_g3d_n16 | 4 | 2026-10-25T04:20:53Z |
| final | r_fwd_g3d_n32_cov | 4 | 2026-10-25T04:20:58Z |
| final | r_fwd_gauss_n64_e1_cov | 7 | 2026-10-25T04:21:05Z |
| final | r_fwd_gauss_n64_e2 | 7 | 2026-10-25T04:21:12Z |
| final | r_fwd_gauss_n64_e2_tol | 7 | 2026-10-25T04:21:19Z |
| final | r_fwd_gauss_n40_e1 | 5 | 2026-10-25T04:21:25Z |
| final | r_fwd_gauss_n80_e2 | 6 | 2026-10-25T04:21:31Z |
| final | r_neutral_gauss_n64_e1_cov | 7 | 2026-10-25T04:21:38Z |
| final | r_neutral_gauss_n64_e2 | 7 | 2026-10-25T04:21:45Z |
| final | r_neutral_gauss_n64_e2_tol | 8 | 2026-10-25T04:21:53Z |
| final | g_runner_a | 16 | 2026-10-25T04:22:12Z |
| final | g_ab | 14 | 2026-10-25T04:22:29Z |

#### Binary md5 (head: `controls/run.env`, identical in every head part: yes; base: `sf41_base_refs/run.env`)

| binary | head md5 | base md5 |
|---|---|---|
| dirichlet_controls_tests | dce34ae54d23dff7a2becd7442fca778 | - |
| dirichlet_long_box_stage_tests | 13b99e8cf39e52134a1066a17cbaa9e0 | - |
| inlet_slab | ded79b8ad5ad019d81e0dc8bea1d980b | aae63d2868be1627e80c16a3a41eb479 |
| interpolation_nonperiodic_x_tests | 7e37f248d7391bf9d23902df1b182240 | - |
| lagrangian_ensemble | 0c8caaf6b31c15fe9abf3558b6a8050d | 59ac27b53610b881b14fe730128080ba |
| lagrangian_medium_ab | f5a80df650ab6d086dfc66a61e8a729c | a53800587a2b5b75e5e289f6f5cf37ee |
| macroflow3d_pipeline | 516d516dc759a12bace29b0d50908e70 | fba69968c801cc651695c80b971da5a0 |
| pollock_nonperiodic_tests | e08de5d9cc2c39050a817a1a44e8b33e | - |
| sf41_cross_representation_tests | d54b7f0c655b9f8eae2fa6b4194bcf3a | - |

`build/binaries.md5` (attempt-2 build part) lists 50 binaries; its md5 for the 9 binaries above equal the `BIN_MD5_*` of the head `run.env`.

Host: `localhost.localdomain`, GPU `Tesla V100-PCIE-32GB`, driver 470.239.06, CUDA (nvidia-smi) 11.4, nvcc 11.4, gcc (GCC) 9.3.1 20200408 (Red Hat 9.3.1-2); `TREE_src_digest=821e4feec2e8895d392c754c049da0a1044172a453097b5fb8f5c099f8168243` (`TREE_git=absent`).
derived clock offset: `run.env` DATE_UTC 2026-10-25T02:16:44Z minus the `nvidia-smi` banner time 2026-10-24 23:16:55 = 3.00 h.

## Outputs inspected

### 0. Completeness and job outcomes

Job exits (`finished exit=0` lines and `STATE=succeeded/EXIT_CODE=0` in `R:job_status/`): `sf41-basebuild`, `sf41-baserefs`, `sf41-build`, `sf41-controls`, `sf41-cross`, `sf41-prodgrid`, `sf41-evladder`, `sf41-abrerun`, `sf41-final` all 0 (the first table above). Every part's `PART_EXIT` is `0`
(`baserefs`, `build`, `controls`, `cross`, `prodgrid`, `evladder`, `abrerun`, `final`; including the attempt-2 `build` part, `R:sf41_campaign/build/PART_EXIT`, with `PART_DONE part=build rc=0` and `PART_SCRIPT_EXIT=0` in its job log). The non-zero return codes in the table are the three R-set runs that exit 21 in base and head alike
(`r_trace_N32_e1.0_host`, `r_trace_N32_e1.0_gpu`, `r_trace_N64_e2.0_gpu`; same set and codes in `R:sf41_base_refs/exit_codes.tsv` and `R:sf41_campaign/final/exit_codes.tsv`, table in section G); the failed build attempt 1 (`PART_EXIT` 1) is superseded. The `sf41-pre-*` jobs are **not evidence** (Disclosures).

### A. Spline contracts (met)

`SPEC` A: all gated sub-checks on the V100 in the full tier. `a_verdict.py` re-run on `R:sf41_campaign/controls`: `A_VERDICT run=A_full exit=0 program_all_pass=1 gated_pass=79 gated_fail=0 inconsistent=0 missing=0 memcheck_exit=0 memcheck_errors=0 PASS` (identical to the campaign line in `results.txt`).
Pre-activation predictions for the ladder orders (3.9024 / 4.3659 / 2.9042 / 3.3858 / 1.9154, owner decision E11) and for the x-only 64->128 orders (3.98 / 2.98 / 1.99) are in the comparison tables; the measured minima reproduce them within -0.0124 to +0.0045 and the gates are met with margins 0.10-0.57.
Reading of the tightest gated margins (*derived*, last column): CPU vs GPU Hessian (evaluation only) `6.021e-14` against `1e-13` (margin 1.66), the partition of unity `3.109e-15` against `1e-14` (3.22), cubic reproduction gradient `1.005e-13` against `1e-12` (9.95); recorded Hessian of the reproduction `2.517e-12` / `7.386e-12` at 128x16x16
(emulated 1.3e-12 to 2.9e-11, inside) and `7.139e-13` (cubic, host) / `1.928e-12` (quadratic, host) at 64x16x16 (emulated 7.5e-13 to 7.7e-12: the cubic host value is just below the lower end, the quadratic one inside). Recorded gradient partition of unity `1.075e-13` at 128x64x64 (emulated 2.9e-14 to 3.9e-13: inside);
full-pipeline CPU vs GPU at 64x32x32 (`h = 1/32`): gradient up to `3.833e-14`, Hessian up to `9.555e-13` (emulated up to 1.9e-13 and 1.6e-12: below). The measured/emulated error ratios of the ladder are 0.976-0.992 for the max norms and 1.059-1.094 for the RMS norms (table below; recorded only, the cause is not examined).

Tier `full`; checks in `A_full.log`: gated 79 (PASS 79, FAIL 0), recorded 54; `A_VERDICT` line of the log: `A_VERDICT all_pass=1 checks_pass=79 checks_fail=0`.

#### Gated checks grouped by name (worst row over boxes and sides; `min` for orders, `max` otherwise)

| check | rows | worst | value | worst at (box/side) | gate | verdict | margin (derived) |
|---|---|---|---|---|---|---|---|
| memory_report_formula_mismatch | 1 | max | 0.000e+00 | 32x16x16/device | 0.000e+00 | PASS | - |
| node_interp_rel | 4 | max | 1.776e-15 | 32x16x16/host | 1.000e-12 | PASS | 562.91 |
| pou_value_abs | 6 | max | 3.109e-15 | 32x16x16/host | 1.000e-14 | PASS | 3.22 |
| repro_cubic_value_rel | 4 | max | 1.331e-15 | 128x16x16/host | 1.000e-12 | PASS | 751.41 |
| repro_cubic_gradient_rel | 4 | max | 5.291e-14 | 128x16x16/host | 1.000e-12 | PASS | 18.90 |
| repro_quad_value_rel | 4 | max | 1.938e-15 | 64x16x16/device | 1.000e-12 | PASS | 515.91 |
| repro_quad_gradient_rel | 4 | max | 1.005e-13 | 128x16x16/host | 1.000e-12 | PASS | 9.95 |
| ladder_min_order_ge_val_max | 2 | min | 3.902e+00 | 32x16x16..64x32x32..128x64x64/host | 3.800e+00 | PASS | 1.03 |
| ladder_min_order_ge_val_rms | 2 | min | 4.364e+00 | 32x16x16..64x32x32..128x64x64/host | 3.800e+00 | PASS | 1.15 |
| ladder_min_order_ge_grad_max | 2 | min | 2.909e+00 | 32x16x16..64x32x32..128x64x64/host | 2.800e+00 | PASS | 1.04 |
| ladder_min_order_ge_grad_rms | 2 | min | 3.373e+00 | 32x16x16..64x32x32..128x64x64/host | 2.800e+00 | PASS | 1.20 |
| ladder_min_order_ge_hess_max | 2 | min | 1.918e+00 | 32x16x16..64x32x32..128x64x64/host | 1.800e+00 | PASS | 1.07 |
| cpugpu_full_value_rel | 2 | max | 2.126e-15 | 64x32x32/both | 1.000e-13 | PASS | 47.04 |
| cpugpu_evalonly_value_rel | 2 | max | 5.002e-16 | 32x16x16/both | 1.000e-13 | PASS | 199.91 |
| cpugpu_evalonly_gradient_rel | 2 | max | 3.366e-15 | 64x32x32/both | 1.000e-13 | PASS | 29.71 |
| cpugpu_evalonly_hessian_rel | 2 | max | 6.021e-14 | 64x32x32/both | 1.000e-13 | PASS | 1.66 |
| hessian_path_bitwise_mismatches | 2 | max | 0.000e+00 | 64x32x32/host | 0.000e+00 | PASS | - |
| noalloc_eval_free_bytes_delta | 1 | max | 0.000e+00 | 64x32x32/device | 0.000e+00 | PASS | - |
| prefilter_repeat_workspace_reallocations | 1 | max | 0.000e+00 | 64x32x32/device | 0.000e+00 | PASS | - |
| shift_value_maxabs | 4 | max | 0.000e+00 | 32x16x16/host | 1.000e-14 | PASS | - |
| shift_gradient_maxabs | 4 | max | 0.000e+00 | 32x16x16/host | 1.000e-14 | PASS | - |
| shift_hessian_maxabs | 4 | max | 0.000e+00 | 32x16x16/host | 1.000e-14 | PASS | - |
| domain_margin_nonfinite_or_outside | 4 | max | 0.000e+00 | 32x16x16/host | 0.000e+00 | PASS | - |
| domain_margin_vs_independent_value_rel | 4 | max | 8.318e-16 | 64x32x32/host | 1.000e-12 | PASS | 1202.16 |
| domain_margin_vs_independent_gradient_rel | 4 | max | 1.454e-15 | 64x32x32/device | 1.000e-12 | PASS | 687.73 |
| domain_outside_reported_without_read | 4 | max | 0.000e+00 | 32x16x16/host | 0.000e+00 | PASS | - |
| domain_half_open_violations | 2 | max | 0.000e+00 | 32x16x16/host | 0.000e+00 | PASS | - |

margin = gate / worst value for upper-bound gates, worst value / gate for order gates; `-` for exact-zero gates or zero values.

#### Order ladder `f = exp(x1) sin(3 x1) (1 + 0.05 S)` on 32x16x16, 64x32x32, 128x64x64: minimum over the pairs 16->32, 32->64 (`A_LADDER_ORDER`, host; device equal: yes)

| quantity | predicted (PRE, A_ladder emulation) | measured min | measured - predicted (derived) | gate | margin over gate (derived) |
|---|---|---|---|---|---|
| val_max | 3.9024 | 3.901686 | -0.0007 | 3.8 | 0.1017 |
| val_rms | 4.3659 | 4.363914 | -0.0020 | 3.8 | 0.5639 |
| grad_max | 2.9042 | 2.908731 | +0.0045 | 2.8 | 0.1087 |
| grad_rms | 3.3858 | 3.373425 | -0.0124 | 2.8 | 0.5734 |
| hess_max | 1.9154 | 1.917651 | +0.0023 | 1.8 | 0.1177 |

Per pair (host):

| pair | val_max | val_rms | grad_max | grad_rms | hess_max |
|---|---|---|---|---|---|
| 16->32 | 3.901686 | 4.363914 | 2.908731 | 3.373425 | 1.917651 |
| 32->64 | 3.975464 | 4.428238 | 2.968985 | 3.435354 | 1.970022 |

#### Recorded: x-only sub-ladder `g = exp(x1) sin(3 x1)`, `L1 = 2`, boxes `nx x 4 x 4` with 16/32/64/128 cells per unit length (host; device equal: yes)

| pair | val_max | val_rms | grad_max | grad_rms | hess_max |
|---|---|---|---|---|---|
| 16->32 | 3.900789 | 4.372389 | 2.905061 | 3.380338 | 1.915694 |
| 32->64 | 3.96186 | 4.480636 | 2.962639 | 3.458511 | 1.966126 |
| 64->128 | 3.98698 | 4.517801 | 2.985752 | 3.511095 | 1.986263 |

Predicted 64->128 orders (SPEC) vs measured: val_max 3.98 vs 3.98698; grad_max 2.98 vs 2.985752; hess_max 1.99 vs 1.986263.

#### Recorded: measured over emulated error per level (`A_EMUL`, host; device lines equal the host lines: yes)

| quantity | level h | measured | emulated | measured / emulated |
|---|---|---|---|---|
| val_max | 1/16 | 2.147676e-03 | 2.175690e-03 | 0.987124 |
| val_max | 1/32 | 1.436959e-04 | 1.455010e-04 | 0.987594 |
| val_max | 1/64 | 9.135043e-06 | 9.360220e-06 | 0.975943 |
| val_rms | 1/16 | 3.141618e-04 | 2.951800e-04 | 1.064306 |
| val_rms | 1/32 | 1.525754e-05 | 1.431620e-05 | 1.065753 |
| val_rms | 1/64 | 7.086828e-07 | 6.477280e-07 | 1.094106 |
| grad_max | 1/16 | 1.191266e-01 | 1.201610e-01 | 0.991392 |
| grad_max | 1/32 | 1.586330e-02 | 1.605180e-02 | 0.988257 |
| grad_max | 1/64 | 2.026002e-03 | 2.060180e-03 | 0.983410 |
| grad_rms | 1/16 | 2.169646e-02 | 2.047930e-02 | 1.059434 |
| grad_rms | 1/32 | 2.093566e-03 | 1.959180e-03 | 1.068593 |
| grad_rms | 1/64 | 1.935272e-04 | 1.797730e-04 | 1.076509 |
| hess_max | 1/16 | 3.990829e+00 | 4.022630e+00 | 0.992094 |
| hess_max | 1/32 | 1.056313e+00 | 1.066380e+00 | 0.990560 |
| hess_max | 1/64 | 2.696230e-01 | 2.730160e-01 | 0.987572 |

derived: measured/emulated over the 15 host rows: min 0.975943, max 1.094106.

#### Recorded (ungated) checks (`A_CHECK ... verdict=RECORDED`, host and device rows)

| check | rows | worst | value | worst at (box/side) |
|---|---|---|---|---|
| pou_gradient_abs_ps | 6 | max | 1.075e-13 | 128x64x64/host |
| pou_value_abs_margin | 6 | max | 2.220e-15 | 64x32x32/host |
| pou_gradient_abs_margin | 6 | max | 8.096e-14 | 128x64x64/host |
| repro_cubic_hessian_rel | 4 | max | 2.517e-12 | 128x16x16/host |
| repro_quad_hessian_rel | 4 | max | 7.386e-12 | 128x16x16/host |
| cpugpu_coeff_diff_rel | 2 | max | 3.458e-14 | 64x32x32/both |
| cpugpu_full_gradient_rel_interior | 2 | max | 5.239e-15 | 64x32x32/both |
| cpugpu_full_hessian_rel_interior | 2 | max | 1.797e-13 | 64x32x32/both |
| cpugpu_full_gradient_rel_nearface | 2 | max | 1.836e-14 | 64x32x32/both |
| cpugpu_full_hessian_rel_nearface | 2 | max | 3.716e-13 | 64x32x32/both |
| cpugpu_full_gradient_rel_margin | 2 | max | 3.833e-14 | 64x32x32/both |
| cpugpu_full_hessian_rel_margin | 2 | max | 9.555e-13 | 64x32x32/both |
| shift_memcmp_mismatched_pairs | 4 | max | 0.000e+00 | 32x16x16/host |

`A_memcheck.log` (compute-sanitizer memcheck, `--only domain`): `A_VERDICT all_pass=1 checks_pass=18 checks_fail=0`; ERROR SUMMARY: 0 errors.
`A_MEMORY` (32x16x16): `A_MEMORY box=32x16x16 coefficient_bytes=73728 spectrum_bytes=82944 line_table_bytes=224 cufft_work_area_bytes=165888 total_device_bytes=322784`.
No-allocation record: `A_NOALLOC free_before=33560199168 free_after=33560199168` (`cudaMemGetInfo` free bytes before / after).

### B. Cross-representation (met)

`SPEC` B: `<= 1e-11` for value, gradient and Hessian at points `>= 2 lambda` from both faces (104608 points = `1e5` lattice points plus the cell centres of three `x1` lines inside `[0.5, L1 - 0.5]`). `b_verdict.py` re-run: `B_VERDICT run=B exit=0 h_tilde_value=9.936224e-16 h_tilde_gradient=2.112960e-14 h_tilde_hessian=9.723778e-13 gate=1e-11 PASS`.
The Hessian margin is `1e-11 / 9.723778e-13 = 10.28` (*derived*); the design emulation gave a worst-case interior floor 2.2e-12 (margin 4.6) and about 3e-14 for a K-like spectrum: the measured `9.72e-13` lies between them. Recorded: the end effect decays with the per-knot factor 0.267938-0.267958 in all 24 fits
(`2 - sqrt(3) = 0.267949`, *derived formula*; SPEC "expected ~0.268"), for `h_tilde` and for `ln k`; the nearest-cell difference at the faces and the interior floor are in the table.

`B_CONFIG`: `tier=RR4 grid=1664x128x128 ell=0.25 lambda_over_h=32 eps=2 seed=6001 L1=13`.

#### Correctors (`B_SOLVE`) and metrics (`B_METRIC`; points = lattice 1e5 + cell centres of three x1 lines inside `[0.5, L1 - 0.5]`)

| corrector | converged | iterations | initial | final | rel |
|---|---|---|---|---|---|
| 1 | 1 | 30 | 1.194356e+06 | 8.365622e-07 | 7.004294e-13 |
| 2 | 1 | 30 | 1.075872e+06 | 2.996660e-07 | 2.785331e-13 |
| 3 | 1 | 30 | 1.191288e+06 | 2.564746e-07 | 2.152919e-13 |

G=7.187359468623e-01,2.459384430307e-02,2.171976215826e-02 mg_levels=6

| field | quantity | rel difference | gate | verdict | points |
|---|---|---|---|---|---|
| h_tilde | value | 9.936224e-16 | 1e-11 | PASS | 104608 |
| h_tilde | gradient | 2.112960e-14 | 1e-11 | PASS | 104608 |
| h_tilde | hessian | 9.723778e-13 | 1e-11 | PASS | 104608 |
| lnK | value | 7.595407e-16 | none | RECORDED | 104608 |
| lnK | gradient | 1.568518e-14 | none | RECORDED | 104608 |
| lnK | hessian | 9.189621e-13 | none | RECORDED | 104608 |

log verdict line: `B_VERDICT all_pass=1`.

#### Recorded: per-knot decay factor of the end effect (`B_DECAY`, fitted on knots above 1e3 times the interior floor)

| field | line (j,k) | face | quantity | factor | knots used | floor |
|---|---|---|---|---|---|---|
| h_tilde | 0,0 | west | gradient | 0.267955 | 9 | 3.404e-15 |
| h_tilde | 0,0 | east | gradient | 0.267947 | 7 | 3.404e-15 |
| h_tilde | 0,0 | west | hessian | 0.267958 | 9 | 2.477e-13 |
| h_tilde | 0,0 | east | hessian | 0.267949 | 8 | 2.477e-13 |
| h_tilde | 64,32 | west | gradient | 0.267953 | 11 | 2.975e-15 |
| h_tilde | 64,32 | east | gradient | 0.267955 | 11 | 2.975e-15 |
| h_tilde | 64,32 | west | hessian | 0.267943 | 11 | 2.040e-13 |
| h_tilde | 64,32 | east | hessian | 0.267951 | 11 | 2.040e-13 |
| h_tilde | 37,101 | west | gradient | 0.267951 | 11 | 3.349e-15 |
| h_tilde | 37,101 | east | gradient | 0.267952 | 11 | 3.349e-15 |
| h_tilde | 37,101 | west | hessian | 0.267950 | 11 | 2.250e-13 |
| h_tilde | 37,101 | east | hessian | 0.267952 | 11 | 2.250e-13 |
| lnK | 0,0 | west | gradient | 0.267943 | 13 | 2.718e-15 |
| lnK | 0,0 | east | gradient | 0.267944 | 13 | 2.718e-15 |
| lnK | 0,0 | west | hessian | 0.267949 | 12 | 1.884e-13 |
| lnK | 0,0 | east | hessian | 0.267938 | 13 | 1.884e-13 |
| lnK | 64,32 | west | gradient | 0.267944 | 13 | 2.304e-15 |
| lnK | 64,32 | east | gradient | 0.267949 | 13 | 2.304e-15 |
| lnK | 64,32 | west | hessian | 0.267956 | 13 | 1.499e-13 |
| lnK | 64,32 | east | hessian | 0.267939 | 13 | 1.499e-13 |
| lnK | 37,101 | west | gradient | 0.267947 | 13 | 2.389e-15 |
| lnK | 37,101 | east | gradient | 0.267955 | 13 | 2.389e-15 |
| lnK | 37,101 | west | hessian | 0.267952 | 12 | 1.737e-13 |
| lnK | 37,101 | east | hessian | 0.267944 | 12 | 1.737e-13 |

derived: h_tilde gradient factor over 6 fits: mean 0.267952, min 0.267947, max 0.267955.
derived: h_tilde hessian factor over 6 fits: mean 0.267950, min 0.267943, max 0.267958.
derived: lnK gradient factor over 6 fits: mean 0.267947, min 0.267943, max 0.267955.
derived: lnK hessian factor over 6 fits: mean 0.267946, min 0.267938, max 0.267956.

`2 - sqrt(3)` = 0.267949 (derived formula; SPEC "~0.268").

#### Recorded: relative difference profile `B_PROFILE` (9984 rows in the log; per knot distance from the face; first knot = nearest cell centre)

| field | line | face | knots per face | gradient @knot0 | hessian @knot0 | gradient @knot8 | hessian @knot8 | gradient median knots 64.. | hessian median knots 64.. |
|---|---|---|---|---|---|---|---|---|---|
| h_tilde | 0,0 | east | 832 | 2.598e-08 | 2.832e-06 | 6.890e-13 | 7.528e-11 | 3.822e-15 | 2.351e-13 |
| h_tilde | 0,0 | west | 832 | 2.747e-07 | 1.531e-05 | 7.301e-12 | 4.070e-10 | 3.098e-15 | 2.598e-13 |
| h_tilde | 37,101 | east | 832 | 5.277e-06 | 2.942e-04 | 1.402e-10 | 7.817e-09 | 3.414e-15 | 2.331e-13 |
| h_tilde | 37,101 | west | 832 | 3.769e-06 | 2.101e-04 | 1.002e-10 | 5.583e-09 | 3.253e-15 | 2.139e-13 |
| h_tilde | 64,32 | east | 832 | 3.396e-06 | 1.893e-04 | 9.024e-11 | 5.031e-09 | 2.968e-15 | 2.185e-13 |
| h_tilde | 64,32 | west | 832 | 5.155e-06 | 2.874e-04 | 1.370e-10 | 7.636e-09 | 2.976e-15 | 1.894e-13 |
| lnK | 0,0 | east | 832 | 3.075e-05 | 1.733e-03 | 8.170e-10 | 4.604e-08 | 2.631e-15 | 1.933e-13 |
| lnK | 0,0 | west | 832 | 2.394e-05 | 1.349e-03 | 6.363e-10 | 3.586e-08 | 2.787e-15 | 1.815e-13 |
| lnK | 37,101 | east | 832 | 1.906e-05 | 1.074e-03 | 5.064e-10 | 2.854e-08 | 2.428e-15 | 1.736e-13 |
| lnK | 37,101 | west | 832 | 1.750e-05 | 9.863e-04 | 4.650e-10 | 2.621e-08 | 2.344e-15 | 1.778e-13 |
| lnK | 64,32 | east | 832 | 2.877e-05 | 1.621e-03 | 7.645e-10 | 4.308e-08 | 2.244e-15 | 1.553e-13 |
| lnK | 64,32 | west | 832 | 2.396e-05 | 1.350e-03 | 6.367e-10 | 3.588e-08 | 2.352e-15 | 1.433e-13 |

`RECORDED B_PROFILE` line of `results.txt`: `B_PROFILE rows=9984 (raw in the log)`.

### C. Exact controls on medium (b) (met)

`c_verdict.py` re-run: `C1_VERDICT exit=0 gated_pass=36 PASS`, `C2_VERDICT ... 18`, `C3_VERDICT ... 18`, `C4_VERDICT ... 72`, `C5_VERDICT exit=0 rows=4 PASS`, `C_VERDICT PASS`. Against the design predictions: C2 `tau` `5.023e-11` (prediction about 5e-11; gate `1e-9`, margin 19.91, *derived*) for both grids and both paths;
C3 random-point maxima `6.661e-16` / `8.882e-16` absolute and `6.944e-16` / `6.360e-16` relative (prediction 1.1e-15 and 3.7e-15: below), dyadic shifts bitwise (2048 of 2048); C4b 2560 included seeds, 768 `exit_inlet`, 1792 downstream (predicted exactly 768 / 1792), the tracker (unfused and fused) and the Pollock
non-periodic mode agree with the exact stream-function classification for every seed (mismatches 0) on both grids, `psi` drift of the reference integration `1.95e-13`; C4a 256 of 256 particles `exit_outlet` in every pass and mode, 3072 rejected steps whose stages lay outside the evaluation domain were exercised (`C4a_stage_outside_domain_exercised_rejected_steps`).
The reversed head drop (recorded, no threshold) raises `lagrangian_labels: no injected seed has positive flux` in the tracker (both paths), as the specification states. C5 has its tightest gate on C2 at 64x32x32 (`rel = 7.970487e-11` against `1e-10`).

`C_full.log`: 144 `C_CHECK` rows, all `verdict=PASS`: yes; log line `C_VERDICT all_pass=1`.

#### C1 (gated rows grouped by name; worst row; grids/paths in the worst-at column)

| check | rows | worst | value | worst at (grid/path) | gate | verdict | margin (derived) |
|---|---|---|---|---|---|---|---|
| C1_f_samples_zero_exactly | 2 | max | 0.000e+00 | 64x16x16/host | <=0e+00 | PASS | - |
| C1_coefficients_zero_exactly | 2 | max | 0.000e+00 | 64x16x16/host | <=0e+00 | PASS | - |
| C1_v_equals_G1_e1 | 2 | max | 0.000e+00 | 64x16x16/host | <=0e+00 | PASS | - |
| C1_particles_not_ok | 4 | max | 0.000e+00 | 64x16x16/unfused | ==0 | PASS | - |
| C1_transverse_displacement | 4 | max | 0.000e+00 | 64x16x16/unfused | <=1e-12 | PASS | - |
| C1_tau_vs_dx1_over_G1_rel | 4 | max | 0.000e+00 | 64x16x16/unfused | <=1e-09 | PASS | - |
| C1_x1_at_times_vs_X0_plus_G1_t_rel | 4 | max | 0.000e+00 | 64x16x16/unfused | <=1e-09 | PASS | - |
| C1_abs_I | 4 | max | 0.000e+00 | 64x16x16/unfused | <=1e-14 | PASS | - |
| C1_covectors_constant | 4 | max | 0.000e+00 | 64x16x16/unfused | <=1e-12 | PASS | - |
| C1_exit_statuses | 4 | max | 0.000e+00 | 64x16x16/unfused | ==0 | PASS | - |
| C1_fused_equals_unfused_bitwise | 2 | max | 0.000e+00 | 64x16x16/fused | ==0 | PASS | - |

#### C2 (gated rows grouped by name; worst row; grids/paths in the worst-at column)

| check | rows | worst | value | worst at (grid/path) | gate | verdict | margin (derived) |
|---|---|---|---|---|---|---|---|
| C2_particles_not_ok | 4 | max | 0.000e+00 | 64x32x32/unfused | ==0 | PASS | - |
| C2_transverse_positions_constant | 4 | max | 5.551e-17 | 128x32x32/unfused | <=1e-12 | PASS | 18014.40 |
| C2_covectors_constant | 4 | max | 7.494e-16 | 128x32x32/unfused | <=1e-12 | PASS | 1334.48 |
| C2_tau_vs_quadrature_rel | 4 | max | 5.023e-11 | 64x32x32/unfused | <=1e-09 | PASS | 19.91 |
| C2_fused_equals_unfused_bitwise | 2 | max | 0.000e+00 | 64x32x32/fused | ==0 | PASS | - |

#### C3 (gated rows grouped by name; worst row; grids/paths in the worst-at column)

| check | rows | worst | value | worst at (grid/path) | gate | verdict | margin (derived) |
|---|---|---|---|---|---|---|---|
| C3_random_shift_plus_1_violations_1e-14 | 6 | max | 0.000e+00 | 64x16x16/host | ==0 | PASS | - |
| C3_random_fused_equals_unfused_eval_bitwise | 2 | max | 0.000e+00 | 64x16x16/fused | ==0 | PASS | - |
| C3_dyadic_shift_plus_1_violations_1e-14 | 6 | max | 0.000e+00 | 64x16x16/host | ==0 | PASS | - |
| C3_dyadic_fused_equals_unfused_eval_bitwise | 2 | max | 0.000e+00 | 64x16x16/fused | ==0 | PASS | - |
| C3_domain_report_outside_false_NaN_no_read | 2 | max | 0.000e+00 | 64x16x16/host+fused | ==0 | PASS | - |

#### C4a (gated rows grouped by name; worst row; grids/paths in the worst-at column)

| check | rows | worst | value | worst at (grid/path) | gate | verdict | margin (derived) |
|---|---|---|---|---|---|---|---|
| C4a_particles_not_exit_outlet | 6 | max | 0.000e+00 | 64x16x16/unfused | ==0 | PASS | - |
| C4a_first_plane_time_not_reached_by_all | 6 | max | 0.000e+00 | 64x16x16/unfused | ==0 | PASS | - |
| C4a_reached_records_nonfinite_or_after_exit_not_NaN | 6 | max | 0.000e+00 | 64x16x16/unfused | ==0 | PASS | - |
| C4a_exit_counts_inexact | 6 | max | 0.000e+00 | 64x16x16/unfused | ==0 | PASS | - |
| C4a_fused_equals_unfused_bitwise | 2 | max | 0.000e+00 | 64x16x16/fused | ==0 | PASS | - |
| C4a_straddle_statuses | 4 | max | 0.000e+00 | 64x16x16/unfused | ==0 | PASS | - |
| C4a_plane_on_L1_statuses | 4 | max | 0.000e+00 | 64x16x16/unfused | ==0 | PASS | - |
| C4a_plane_just_beyond_statuses | 4 | max | 0.000e+00 | 64x16x16/unfused | ==0 | PASS | - |
| C4a_stage_outside_domain_hmax_2h_statuses | 4 | max | 0.000e+00 | 64x16x16/unfused | ==0 | PASS | - |
| C4a_stage_outside_domain_exercised_rejected_steps | 2 | max | 3.072e+03 | 64x16x16/fused+unfused | >0 | PASS | - |
| C4a_track_json_and_exit_files | 2 | max | 0.000e+00 | 64x16x16/unfused | ==0 | PASS | - |

#### C4b (gated rows grouped by name; worst row; grids/paths in the worst-at column)

| check | rows | worst | value | worst at (grid/path) | gate | verdict | margin (derived) |
|---|---|---|---|---|---|---|---|
| C4b_plane_pass_misclassified_seeds | 6 | max | 0.000e+00 | 64x32x32/unfused | ==0 | PASS | - |
| C4b_time_pass_misclassified_seeds | 6 | max | 0.000e+00 | 64x32x32/unfused | ==0 | PASS | - |
| C4b_plane_counts_inexact | 6 | max | 0.000e+00 | 64x32x32/unfused | ==0 | PASS | - |
| C4b_time_counts_inexact | 6 | max | 0.000e+00 | 64x32x32/unfused | ==0 | PASS | - |
| C4b_fused_equals_unfused_bitwise | 2 | max | 0.000e+00 | 64x32x32/fused | ==0 | PASS | - |

#### C2 `tau` against the independent quadrature (pre-activation emulation: about 5e-11; gate 1e-9)

| grid | path | tau rel error | gate |
|---|---|---|---|
| 64x32x32 | unfused | 5.022843e-11 | <=1e-09 |
| 64x32x32 | fused | 5.022843e-11 | <=1e-09 |
| 128x32x32 | unfused | 5.022803e-11 | <=1e-09 |
| 128x32x32 | fused | 5.022803e-11 | <=1e-09 |

#### Recorded `C_RECORD` lines (no threshold)

```
C_RECORD name=C2_I_vs_quadrature grid=64x32x32 path=unfused max_abs_err=1.466e-07 max_abs_Iref=8.659e-04 q_exact=0.98677408375415
C_RECORD name=C2_I_vs_quadrature grid=64x32x32 path=fused max_abs_err=1.466e-07 max_abs_Iref=8.659e-04 q_exact=0.98677408375415
C_RECORD name=C2_I_vs_quadrature grid=128x32x32 path=unfused max_abs_err=1.481e-06 max_abs_Iref=9.678e-04 q_exact=0.943348864398513
C_RECORD name=C2_I_vs_quadrature grid=128x32x32 path=fused max_abs_err=1.481e-06 max_abs_Iref=9.678e-04 q_exact=0.943348864398513
C_RECORD name=C3_random_shift grid=64x16x16 path=host max_abs=6.661e-16 max_rel=6.944e-16 bitwise_equal_points=918 of 4096
C_RECORD name=C3_random_shift grid=64x16x16 path=device max_abs=6.661e-16 max_rel=6.944e-16 bitwise_equal_points=918 of 4096
C_RECORD name=C3_random_shift grid=64x16x16 path=fused max_abs=6.661e-16 max_rel=6.944e-16 bitwise_equal_points=918 of 4096
C_RECORD name=C3_random_host_vs_device_v grid=64x16x16 path=device max_abs=4.441e-16
C_RECORD name=C3_dyadic_shift grid=64x16x16 path=host max_abs=0.000e+00 max_rel=0.000e+00 bitwise_equal_points=2048 of 2048
C_RECORD name=C3_dyadic_shift grid=64x16x16 path=device max_abs=0.000e+00 max_rel=0.000e+00 bitwise_equal_points=2048 of 2048
C_RECORD name=C3_dyadic_shift grid=64x16x16 path=fused max_abs=0.000e+00 max_rel=0.000e+00 bitwise_equal_points=2048 of 2048
C_RECORD name=C3_dyadic_host_vs_device_v grid=64x16x16 path=device max_abs=2.220e-16
C_RECORD name=C3_random_shift grid=128x16x16 path=host max_abs=8.882e-16 max_rel=6.360e-16 bitwise_equal_points=920 of 4096
C_RECORD name=C3_random_shift grid=128x16x16 path=device max_abs=8.882e-16 max_rel=6.360e-16 bitwise_equal_points=920 of 4096
C_RECORD name=C3_random_shift grid=128x16x16 path=fused max_abs=8.882e-16 max_rel=6.360e-16 bitwise_equal_points=920 of 4096
C_RECORD name=C3_random_host_vs_device_v grid=128x16x16 path=device max_abs=2.220e-16
C_RECORD name=C3_dyadic_shift grid=128x16x16 path=host max_abs=0.000e+00 max_rel=0.000e+00 bitwise_equal_points=2048 of 2048
C_RECORD name=C3_dyadic_shift grid=128x16x16 path=device max_abs=0.000e+00 max_rel=0.000e+00 bitwise_equal_points=2048 of 2048
C_RECORD name=C3_dyadic_shift grid=128x16x16 path=fused max_abs=0.000e+00 max_rel=0.000e+00 bitwise_equal_points=2048 of 2048
C_RECORD name=C3_dyadic_host_vs_device_v grid=128x16x16 path=device max_abs=2.220e-16
C_RECORD name=C4b_reference grid=64x32x32 path=reference included=2560 inlet=768 outlet=1792 plane_reaching=1792 T=3 psi_drift=1.95e-13
C_RECORD name=C4b_counts grid=64x32x32 path=unfused included=2560 plane:inlet=768 reached=1792 outlet=0 time:inlet=768 outlet=1792
C_RECORD name=C4b_counts grid=64x32x32 path=fused included=2560 plane:inlet=768 reached=1792 outlet=0 time:inlet=768 outlet=1792
C_RECORD name=C4b_reference grid=128x32x32 path=reference included=2560 inlet=768 outlet=1792 plane_reaching=1792 T=5.5 psi_drift=1.95e-13
C_RECORD name=C4b_counts grid=128x32x32 path=unfused included=2560 plane:inlet=768 reached=1792 outlet=0 time:inlet=768 outlet=1792
C_RECORD name=C4b_counts grid=128x32x32 path=fused included=2560 plane:inlet=768 reached=1792 outlet=0 time:inlet=768 outlet=1792
C_RECORD name=C4_reversed_head grid=64x16x16 path=unfused outcome="exception: lagrangian_labels: no injected seed has positive flux"
C_RECORD name=C4_reversed_head grid=64x16x16 path=fused outcome="exception: lagrangian_labels: no injected seed has positive flux"
```

#### C3 against the design emulation (max abs about 1.1e-15, max rel about 3.7e-15 on random points)

| grid | path | max abs | max rel | bitwise-equal points (of 4096) |
|---|---|---|---|---|
| 64x16x16 | host | 6.661e-16 | 6.944e-16 | 918 |
| 64x16x16 | device | 6.661e-16 | 6.944e-16 | 918 |
| 64x16x16 | fused | 6.661e-16 | 6.944e-16 | 918 |
| 128x16x16 | host | 8.882e-16 | 6.360e-16 | 920 |
| 128x16x16 | device | 8.882e-16 | 6.360e-16 | 920 |
| 128x16x16 | fused | 8.882e-16 | 6.360e-16 | 920 |

#### C4a Pollock non-periodic checks (`pollock_nonperiodic_tests --only C4`, `C4_pollock.log`)

| grid | case | mode | exit_inlet | exit_outlet | reached | expected | mismatches | verdict |
|---|---|---|---|---|---|---|---|---|
| 64x16x16 | C4a | plane | 0 | 256 | 256 | inlet:0,outlet:256,reached:256 | - | PASS |
| 64x16x16 | C4a | time | 0 | 256 | 256 | inlet:0,outlet:256,reached:256 | - | PASS |
| 128x16x16 | C4a | plane | 0 | 256 | 256 | inlet:0,outlet:256,reached:256 | - | PASS |
| 128x16x16 | C4a | time | 0 | 256 | 256 | inlet:0,outlet:256,reached:256 | - | PASS |
| 64x32x32 | C4b | plane | 768 | 0 | 1792 | inlet:768,outlet:0,reached:1792 | 0 | PASS |
| 64x32x32 | C4b | time | 768 | 1792 | 0 | inlet:768,outlet:1792,reached:0 | 0 | PASS |
| 128x32x32 | C4b | plane | 768 | 0 | 1792 | inlet:768,outlet:0,reached:1792 | 0 | PASS |
| 128x32x32 | C4b | time | 768 | 1792 | 0 | inlet:768,outlet:1792,reached:0 | 0 | PASS |

log line: `C4_VERDICT method=pollock PASS (22 checks, 0 failed)`.

#### C5 solver on the C1 / C2 media through the lean stage (`C5.log`)

| case | size | converged | iterations | initial | final | rel | balance | max head rel diff | pass |
|---|---|---|---|---|---|---|---|---|---|
| C1 | 64x16x16 | 1 | 0 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 1 |
| C1 | 128x16x16 | 1 | 0 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 1 |
| C2 | 64x32x32 | 1 | 10 | 3.775619e+02 | 3.009352e-08 | 7.970487e-11 | 2.067940e-13 | 1.404551e-13 | 1 |
| C2 | 128x32x32 | 1 | 20 | 5.206390e+02 | 1.143529e-12 | 2.196396e-15 | 8.709027e-14 | 6.676744e-16 | 1 |

re-run verdict lines are in the note (`c_verdict.py`).

### D. Lean stage bitwise (met)

`d_verdict.py --run D` re-run: `D_VERDICT run=D exit=0 sizes=[128x32x32:head=1,bsolver=1,lines_equal=1,mapped=1; 512x64x64:head=1,bsolver=1,lines_equal=1,mapped=1; 3328x256x256:head=1,bsolver=1,lines_equal=1,mapped=na] PASS`.
The lean stage's head equals `DirichletMedium`'s head byte for byte (`memcmp` of the host arrays, `differing=0` in every `DIAG` line) at the three sizes, the `B_SOLVER` lines are string-equal, and the mapped-host head is identical at the two small sizes; at 3328x256x256 both lines equal SF-38's committed C0 line field for field
(`D_C0_COMPARE ... identical_fields=11/11`, the environment control). The mapped-host re-run of a production grid was not needed (section E). The lean-stage solve-phase exact bytes equal the closed form at every size (`S2_closed_form` checks): 60.563 / 60.570 / 60.571 B/cell for the device head (52.563 / 52.570 for the mapped head) against the SPEC estimate 60.571 B/cell at 7 levels.
The process-level PID peak `19755` MiB and device-wide peak `19759` MiB of the D process are recorded, not gated (threshold D is bitwise only; decision of the N7 corrective `269c658`): the D process holds the SF-38 `DirichletMedium` reference run and the lean stage one after the other, and the evidence does not separate their peaks (SF-38 reported a PID peak of 19745 MiB for its C0 run of medium (b)); the lean stage's own bytes are the `MEM_PHASE` rows (12598.854 MiB at 3328x256x256).

Compared by `d_verdict.py` from the `D.log` lines; campaign lines copied from `results.txt`:

```
D_VERDICT run=D exit=0 sizes=[128x32x32:head=1,bsolver=1,lines_equal=1,mapped=1; 512x64x64:head=1,bsolver=1,lines_equal=1,mapped=1; 3328x256x256:head=1,bsolver=1,lines_equal=1,mapped=na] PASS
```

#### `B_SOLVER` fields (reference `DirichletMedium`; the lean stage line is string-equal: column 2)

| size | lean line equal | iterations | initial | final | rel | qbar | Q_in | Q_out | balance | max_div |
|---|---|---|---|---|---|---|---|---|---|---|
| 128x32x32 | yes | 30 | 1.434840e+05 | 6.496539e-08 | 4.527710e-13 | 1.756410271345e+00 | 1.756410271345e+00 | 1.756410271344e+00 | 6.552326e-13 | 1.888051e-10 |
| 512x64x64 | yes | 40 | 9.307687e+05 | 1.604320e-05 | 1.723651e-11 | 1.853115550126e+00 | 1.853115550126e+00 | 1.853115550200e+00 | 4.027468e-11 | 1.336864e-08 |
| 3328x256x256 | yes | 50 | 6.724494e+06 | 6.526221e-04 | 9.705149e-11 | 1.002469787644e+00 | 1.002469787644e+00 | 1.002469787885e+00 | 2.404573e-10 | 1.034116e-07 |

`D_C0_COMPARE size=3328x256x256 identical_fields=11/11 differing=[-] (environment control vs SF-38 C0 line)`

#### Lean-stage `MEM_PHASE` exact bytes in the D process (order of the block: 128 device, 128 mapped, 512 device, 512 mapped, 3328 device; B/cell = S2 / cells, derived)

| size | head | cells | S2 exact B | S2 MiB | S2 B/cell | max phase | phases (exact bytes) |
|---|---|---|---|---|---|---|---|
| 128x32x32 | device head | 131072 | 7938056 | 7.570 | 60.563 | S2 | F=0,S1=0,S2=7938056,V=5337088,H=1048576 |
| 128x32x32 | mapped-host head | 131072 | 6889480 | 6.570 | 52.563 | S2 | F=0,S1=0,S2=6889480,V=5337088,H=1048576 |
| 512x64x64 | device head | 2097152 | 127025160 | 121.141 | 60.570 | S2 | F=0,S1=0,S2=127025160,V=84516864,H=16777216 |
| 512x64x64 | mapped-host head | 2097152 | 110247944 | 105.141 | 52.570 | S2 | F=0,S1=0,S2=110247944,V=84516864,H=16777216 |
| 3328x256x256 | device head | 218103808 | 13210855432 | 12598.854 | 60.571 | S2 | F=0,S1=0,S2=13210855432,V=8739364864,H=1744830464 |

#### `D.log` program checks: 40 `CHECK` lines, PASS 40, other 0; `D_CHECK` lines and the closed-form S2 checks

```
DIAG D_128x32x32_device_vs_ref sizes=131072/131072(expected 131072) differing=0 first=-1 (i=-1 j=-1 k=-1) maxabs=0.000e+00
CHECK D_128x32x32_S2_closed_form PASS  S2=7938056 closed=7938056 (60.563 B/cell, scratch=8)
DIAG D_128x32x32_mapped_vs_ref sizes=131072/131072(expected 131072) differing=0 first=-1 (i=-1 j=-1 k=-1) maxabs=0.000e+00
CHECK D_128x32x32_mapped_S2_closed_form PASS  S2=6889480 closed=6889480 (52.563 B/cell, scratch=8)
D_CHECK size=128x32x32 head_identical=1 bsolver_identical=1 mapped_identical=1
DIAG D_512x64x64_device_vs_ref sizes=2097152/2097152(expected 2097152) differing=0 first=-1 (i=-1 j=-1 k=-1) maxabs=0.000e+00
CHECK D_512x64x64_S2_closed_form PASS  S2=127025160 closed=127025160 (60.570 B/cell, scratch=8)
DIAG D_512x64x64_mapped_vs_ref sizes=2097152/2097152(expected 2097152) differing=0 first=-1 (i=-1 j=-1 k=-1) maxabs=0.000e+00
CHECK D_512x64x64_mapped_S2_closed_form PASS  S2=110247944 closed=110247944 (52.570 B/cell, scratch=8)
D_CHECK size=512x64x64 head_identical=1 bsolver_identical=1 mapped_identical=1
DIAG D_3328x256x256_device_vs_ref sizes=218103808/218103808(expected 218103808) differing=0 first=-1 (i=-1 j=-1 k=-1) maxabs=0.000e+00
CHECK D_3328x256x256_S2_closed_form PASS  S2=13210855432 closed=13210855432 (60.571 B/cell, scratch=8)
D_CHECK size=3328x256x256 head_identical=1 bsolver_identical=1 mapped_identical=na
```

`D_MEM run=D pid_peak_mib=19755 dev_peak_mib=19759 samples=631 sampler=OK reasons=- (recorded, not gated)`
`run=D exit=0 wall_s=128.26 host_max_rss_kb=10873636 free_b_before=[Mem: 134590115840 3132481536 22223757312 382025728 109233876992 130434596864]`
`run=D pid_peak_mib=19755 dev_peak_mib=19759 samples=631 max_exact_mib=12598.854 tree_pid=144387 status=OK reasons=-`

### E. Production feasibility (met at the four geometries)

`e_verdict.py` re-run: `E_VERDICT ALL PASS (E1=PASS,E2=PASS,E3=PASS,E4=PASS)`; every run `solve=primary` (`rel <= 1e-10`, no C0 fallback needed), balance `<= 4.2e-11` (E_RUN field, recomputed from the printed `Q_in`, `Q_out`), exit 0, `device_head` (no mapped-host re-run was triggered: PID peak below 30000 MiB in all four, and no `E*_mapped` file exists).
The maximum exact-bytes phase is the solve phase S2 in all four (*derived*: `every phase <= S2` column), at 60.571 B/cell (SPEC estimate 60.57; the closed-form memory model of the `controls` part equals the measured F, S2, V, H and S3 bytes exactly in E2, E3 and E4); the PID peak exceeds S2 by 497.9-507.6 MiB (CUDA context and allocator overhead, *derived*).
The tightest run is E2/E3 at 28855 MiB (`1145` MiB below the gate, *derived*; SF-38 reported 28885 MiB for the medium (a) stage of the same geometry). The sampler gates (`>= 5` samples, PID peak `<=` device-wide peak) hold in all four. Exits and statuses: no tracker non-ok in either pass, no `exit_inlet` anywhere, `exit_outlet` only in the time pass (3695 / 4142 / 3957 / 1910 of 10000 particles)
and, in the Pollock bridge, 3560 / 4157 / 3961 / 1896 (the two comparators count different quantities and are not compared by any threshold; both are reported as found). Inlet backflow is encountered (E2: 78 plane / 72 time; E3: 843 / 674) without any exit or non-ok particle.

Settings echoed by each log (`RESOLVED` line), e.g. E4: `RESOLVED grid=6656x256x256 L1=26 lambda=0.25 planes(auto)=1.5,1.75,2.25,3.25,5.25,9.25,17.25,24.75 times(auto)=n*lambda/2 up to (L1-10 lambda)/qbar, qbar measured per realization medium=b x0=1.25`

#### Solve, memory sampler, wall (verdict `E_RUN` fields: solve mode, PID peak against 30000 MiB; sampler `pid_peak.py`)

| run | grid | eps | ell | solve | converged | iterations | rel | balance | max_div | fallback | PID peak MiB | device-wide peak MiB | samples | sampler status | wall s | host max RSS kB | stage s | E_RUN state |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E1 | 2048x256x256 | 1 | 0.0390625 | primary | 1 | 20 | 3.409160e-11 | 2.348053e-11 | 3.922635e-09 | 0 | 8251 | 8255 | 160 | OK | 33.69 | 4858940 | 13.870 | PASS |
| E2 | 3328x384x384 | 1.5 | 0.041666666666666664 | primary | 1 | 40 | 1.725197e-11 | 5.513091e-12 | 1.522198e-08 | 0 | 28855 | 28859 | 536 | OK | 110.28 | 16021444 | 67.512 | PASS |
| E3 | 3328x384x384 | 2 | 0.08333333333333333 | primary | 1 | 50 | 1.607008e-12 | 6.261729e-12 | 1.112806e-07 | 0 | 28855 | 28859 | 627 | OK | 129.16 | 16021512 | 74.824 | PASS |
| E4 | 6656x256x256 | 2 | 0.25 | primary | 1 | 90 | 1.263955e-11 | 4.164330e-11 | 6.374646e-08 | 0 | 25701 | 25705 | 703 | OK | 143.81 | 14314552 | 90.314 | PASS |

#### Per-phase exact bytes (`MEM_PHASE`; MiB = bytes / 1048576, derived; B/cell = S2 / cells, derived; context+overhead = PID peak - S2 MiB, derived)

| run | max phase | S2 exact B | S2 MiB | S2 B/cell | phases (MiB): F, S1, S2, V, H, S3, T, B | PID peak - S2 MiB | headroom to 30000 MiB | every phase <= S2 |
|---|---|---|---|---|---|---|---|---|
| E1 | S2 | 8129757192 | 7753.141 | 60.571 | F=3074.002,S1=0.000,S2=7753.141,V=5129.508,H=1024.000,S3=6178.062,T=2157.136,B=4100.500 | 497.859 | 21749.0 | yes |
| E2 | S2 | 29724424712 | 28347.420 | 60.571 | F=11236.502,S1=0.000,S2=28347.420,V=18742.883,H=3744.000,S3=22545.096,T=7585.045,B=14986.875 | 507.580 | 1145.0 | yes |
| E3 | S2 | 29724424712 | 28347.420 | 60.571 | F=11236.502,S1=0.000,S2=28347.420,V=18742.883,H=3744.000,S3=22545.096,T=7531.333,B=14986.875 | 507.580 | 1145.0 | yes |
| E4 | S2 | 26421710856 | 25197.707 | 60.571 | F=9986.002,S1=0.000,S2=25197.707,V=16667.508,H=3328.000,S3=20056.098,T=6703.183,B=13325.500 | 503.293 | 4299.0 | yes |

#### Closed-form memory model (`controls` MEMMODEL_a / MEMMODEL_b, printed before the runs) against the measured exact bytes (derived comparison)

| run | model | F | S2 | V | H | S3 |
|---|---|---|---|---|---|---|
| E2 | MEMMODEL_a | F equal | S2 equal | V equal | H equal | S3 equal |
| E3 | MEMMODEL_a | F equal | S2 equal | V equal | H equal | S3 equal |
| E4 | MEMMODEL_b | F equal | S2 equal | V equal | H equal | S3 equal |

#### Tracking, exits and non-ok (`STATUS`, `TEXIT`, `TSTATUS`, `POLLOCK`, `BACKFLOW` lines of the logs)

| run | tracker non_ok plane | non_ok time | seed_backflow | exit_inlet plane | exit_outlet plane | exit_inlet time | exit_outlet time | TSTATUS time hist | Pollock exit_inlet plane | Pollock exit_outlet plane | Pollock exit_inlet time | Pollock exit_outlet time | Pollock weighted_not_active | inlet_vmin | backflow_enc plane | backflow_enc time |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E1 | 0 | 0 | 0 | 0 | 0 | 0 | 3695 | 6305,0,0,0,0,0,0 | 0 | 0 | 0 | 3560 | 3560 | 3.441734e-02 | 0 | 0 |
| E2 | 0 | 0 | 0 | 0 | 0 | 0 | 4142 | 5858,0,0,0,0,0,0 | 0 | 0 | 0 | 4157 | 4157 | 5.287898e-03 | 78 | 72 |
| E3 | 0 | 0 | 0 | 0 | 0 | 0 | 3957 | 6043,0,0,0,0,0,0 | 0 | 0 | 0 | 3961 | 3961 | 2.267710e-03 | 843 | 674 |
| E4 | 0 | 0 | 0 | 0 | 0 | 0 | 1910 | 8090,0,0,0,0,0,0 | 0 | 0 | 0 | 1896 | 1896 | 1.023875e-02 | 0 | 0 |

#### `flow_stats.json` values (`FLOWSTATS` line; `theta_2`, `theta_3` = tilt of the interior mean face-flux vector)

| run | qbar | x_faces_nonpositive | speed/qbar percentiles (0, 0.1, 1, 50 %) | interior planes | mean_u | mean_v | mean_w | theta_2 | theta_3 |
|---|---|---|---|---|---|---|---|---|---|
| E1 | 1.181348938810e+00 | 0/134283264 | 1.818938e-02,7.232117e-02,1.337163e-01,8.356550e-01 | 50..1998 | 1.181348938834e+00 | -1.135347269871e-03 | 2.249141258075e-03 | -9.610597538112e-04 | 1.903873162811e-03 |
| E2 | 1.449649782898e+00 | 2230/490881024 | 1.995558e-03,1.497091e-02,3.909465e-02,6.689606e-01 | 80..3248 | 1.449649782893e+00 | -1.365187113382e-03 | 6.605289720408e-03 | -9.417355322050e-04 | 4.556441208865e-03 |
| E3 | 1.899552783704e+00 | 173142/490881024 | 3.666271e-04,2.540001e-03,9.565589e-03,4.944097e-01 | 160..3168 | 1.899552783710e+00 | 6.483659625399e-03 | 2.938580790799e-02 | 3.413242581245e-03 | 1.546862205015e-02 |
| E4 | 1.480233352885e+00 | 0/436273152 | 4.282347e-04,1.667436e-03,7.329138e-03,5.343201e-01 | 320..6336 | 1.480233352925e+00 | -4.711448413967e-02 | -4.585585028417e-02 | -3.181835032224e-02 | -3.096889389965e-02 |

Mapped-host re-run: files with `mapped` in the E part directory: none; `E_RUN` lines carry `(device_head)` only.

```
E_RUN run=E1 exit=0 solve=primary rel=3.409160e-11 balance=2.370176e-11 pid_peak_mib=8251 limit_mib=30000 mem=OK phases_le_S2=1 state=PASS
E_VERDICT id=E1 grid=2048x256x256 PASS (device_head)
E_RUN run=E2 exit=0 solve=primary rel=1.725197e-11 balance=5.518605e-12 pid_peak_mib=28855 limit_mib=30000 mem=OK phases_le_S2=1 state=PASS
E_VERDICT id=E2 grid=3328x384x384 PASS (device_head)
E_RUN run=E3 exit=0 solve=primary rel=1.607008e-12 balance=6.317254e-12 pid_peak_mib=28855 limit_mib=30000 mem=OK phases_le_S2=1 state=PASS
E_VERDICT id=E3 grid=3328x384x384 PASS (device_head)
E_RUN run=E4 exit=0 solve=primary rel=1.263955e-11 balance=4.188526e-11 pid_peak_mib=25701 limit_mib=30000 mem=OK phases_le_S2=1 state=PASS
E_VERDICT id=E4 grid=6656x256x256 PASS (device_head)
E_VERDICT ALL PASS (E1=PASS,E2=PASS,E3=PASS,E4=PASS)
```

### F. Lagrangian `e_v` on medium (b) (met at both amplitudes)

`f_verdict.py` re-run on `R:sf41_campaign/evladder`: `F_VERDICT eps=1 cells_met=42/42 runs_ok=12/12 MET`, `F_VERDICT eps=2 cells_met=42/42 runs_ok=12/12 MET`, `F_VERDICT ALL MET`. The RMS `e_v` decreases strictly on every refinement in all 84 (eps, seed, plane) cells, with last-pair orders 1.9920-2.0263 (`eps = 1`) and 1.9959-2.0040 (`eps = 2`), all `>= 1.8`
(SF-37 V4 on the cube: 1.996-2.007 in 45 of 45 cells, no floor at `lambda/h = 128`, SPEC). The earlier pairs give 1.9762-2.0928. Wall time of the 24 runs 450 s in total. **The `D_P` rows cannot be recomputed from `raw/`** (the per-particle `weights.npy` files are absent by design): the re-run prints `F_DP ... not_evaluable` and `F_V3B_SUMMARY ... not_evaluable=42`;
the `D_P`, `SE_D` and V3b numbers below are the campaign's `F_DP` lines of `R:sf41_campaign/evladder/results.txt` (computed on the V100 with the committed bootstrap matrix), and my recomputation of the V3b class from those printed numbers agrees with the printed class in 84 of 84 cells (*derived*). Recorded only; the numbers are not interpreted (Rule 1).

Rungs: `ny` = 32, 64, 128, 256 (`lambda/h` = 16, 32, 64, 128; `ell` 0.5), planes `x1` = 3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5 (`RECORDED F_PLANES`).

```
F_VERDICT eps=1 cells_met=42/42 runs_ok=12/12 MET
F_VERDICT eps=2 cells_met=42/42 runs_ok=12/12 MET
F_VERDICT ALL MET
```

#### Observed order of the RMS `e_v` ladder per amplitude (`F_CELL`; order of a pair = log2(e_N / e_2N); last pair = 128 -> 256; 42 cells = 3 seeds x 14 planes)

| eps | cells | strictly decreasing | MET | min order 32->64 | max order 32->64 | min order 64->128 | max order 64->128 | min last-pair order (cell) | max last-pair order (cell) | min e_v at ny=256 | max e_v at ny=256 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 42 | 42 | 42 | 2.0001 | 2.0928 | 1.9945 | 2.0435 | 1.9920 (seed 6103 plane 1) | 2.0263 (seed 6101 plane 9) | 2.90e-05 | 1.15e-04 |
| 2 | 42 | 42 | 42 | 1.9762 | 2.0220 | 1.9956 | 2.0063 | 1.9959 (seed 6103 plane 7) | 2.0040 (seed 6101 plane 9) | 7.11e-05 | 4.76e-04 |

RMS `e_v` of the first and last plane per seed (`F_CELL`; four rungs ny = 32, 64, 128, 256):

| eps | seed | plane | x1 | e_v at ny=32 / 64 / 128 / 256 | orders |
|---|---|---|---|---|---|
| 1 | 6101 | 0 | 3 | 2.878377e-03 / 7.167827e-04 / 1.786441e-04 / 4.467375e-05 | 2.0056 / 2.0044 / 1.9996 |
| 1 | 6101 | 13 | 9.5 | 4.832941e-03 / 1.184314e-03 / 2.954404e-04 / 7.338998e-05 | 2.0288 / 2.0031 / 2.0092 |
| 1 | 6102 | 0 | 3 | 2.781594e-03 / 6.948803e-04 / 1.730313e-04 / 4.333496e-05 | 2.0011 / 2.0057 / 1.9974 |
| 1 | 6102 | 13 | 9.5 | 6.610420e-03 / 1.634113e-03 / 4.043280e-04 / 1.013403e-04 | 2.0162 / 2.0149 / 1.9963 |
| 1 | 6103 | 0 | 3 | 4.856242e-03 / 1.211155e-03 / 3.033636e-04 / 7.566049e-05 | 2.0035 / 1.9973 / 2.0034 |
| 1 | 6103 | 13 | 9.5 | 5.402744e-03 / 1.328329e-03 / 3.310511e-04 / 8.221937e-05 | 2.0241 / 2.0045 / 2.0095 |
| 2 | 6101 | 0 | 3 | 7.897289e-03 / 1.984870e-03 / 4.965732e-04 / 1.241984e-04 | 1.9923 / 1.9990 / 1.9994 |
| 2 | 6101 | 13 | 9.5 | 2.121238e-02 / 5.300103e-03 / 1.326211e-03 / 3.309887e-04 | 2.0008 / 1.9987 / 2.0025 |
| 2 | 6102 | 0 | 3 | 7.399537e-03 / 1.855418e-03 / 4.639829e-04 / 1.160390e-04 | 1.9957 / 1.9996 / 1.9995 |
| 2 | 6102 | 13 | 9.5 | 3.036325e-02 / 7.614222e-03 / 1.903636e-03 / 4.758619e-04 | 1.9956 / 1.9999 / 2.0001 |
| 2 | 6103 | 0 | 3 | 1.752398e-02 / 4.350705e-03 / 1.087634e-03 / 2.715606e-04 | 2.0100 / 2.0001 / 2.0018 |
| 2 | 6103 | 13 | 9.5 | 2.646440e-02 / 6.540733e-03 / 1.628007e-03 / 4.073098e-04 | 2.0165 / 2.0063 / 1.9989 |

#### The 24 runs (exit code, wall s from `walls.tsv`, `B_SOLVER`, `STATUS` lines)

| run | rc | wall s | PCG iterations | rel | balance | non_ok plane | non_ok time | exit_inlet plane | exit_outlet plane | exit_inlet time | exit_outlet time | seed_backflow |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F_e1_s6101_g32 | 0 | 5 | 40 | 8.712231e-13 | 9.820473e-13 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| F_e1_s6101_g64 | 0 | 6 | 40 | 1.928585e-12 | 8.023446e-13 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| F_e1_s6101_g128 | 0 | 12 | 40 | 4.059601e-12 | 1.844127e-12 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| F_e1_s6101_g256 | 0 | 47 | 40 | 8.081399e-12 | 1.771732e-12 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| F_e1_s6102_g32 | 0 | 5 | 30 | 6.497745e-11 | 1.146443e-10 | 0 | 0 | 0 | 0 | 0 | 279 | 0 |
| F_e1_s6102_g64 | 0 | 6 | 40 | 2.862392e-14 | 1.676735e-13 | 0 | 0 | 0 | 0 | 0 | 277 | 0 |
| F_e1_s6102_g128 | 0 | 12 | 40 | 7.647650e-14 | 5.847627e-13 | 0 | 0 | 0 | 0 | 0 | 276 | 0 |
| F_e1_s6102_g256 | 0 | 48 | 40 | 1.703020e-13 | 2.629606e-12 | 0 | 0 | 0 | 0 | 0 | 276 | 0 |
| F_e1_s6103_g32 | 0 | 5 | 40 | 6.518864e-14 | 6.043527e-14 | 0 | 0 | 0 | 0 | 0 | 117 | 0 |
| F_e1_s6103_g64 | 0 | 7 | 40 | 1.678768e-13 | 3.450564e-14 | 0 | 0 | 0 | 0 | 0 | 111 | 0 |
| F_e1_s6103_g128 | 0 | 11 | 40 | 3.615968e-13 | 9.712946e-14 | 0 | 0 | 0 | 0 | 0 | 111 | 0 |
| F_e1_s6103_g256 | 0 | 49 | 40 | 7.481393e-13 | 3.290102e-13 | 0 | 0 | 0 | 0 | 0 | 109 | 0 |
| F_e2_s6101_g32 | 0 | 6 | 50 | 1.329922e-11 | 9.832689e-12 | 0 | 0 | 0 | 0 | 0 | 2200 | 0 |
| F_e2_s6101_g64 | 0 | 7 | 50 | 5.069808e-11 | 7.550515e-12 | 0 | 0 | 0 | 0 | 0 | 2207 | 0 |
| F_e2_s6101_g128 | 0 | 13 | 60 | 2.280992e-12 | 6.473766e-13 | 0 | 0 | 0 | 0 | 0 | 2204 | 0 |
| F_e2_s6101_g256 | 0 | 55 | 60 | 4.795790e-12 | 7.380071e-13 | 0 | 0 | 0 | 0 | 0 | 2204 | 0 |
| F_e2_s6102_g32 | 0 | 5 | 40 | 3.213466e-11 | 1.051728e-10 | 0 | 0 | 0 | 0 | 0 | 540 | 0 |
| F_e2_s6102_g64 | 0 | 7 | 40 | 7.668435e-11 | 1.928310e-09 | 0 | 0 | 0 | 0 | 0 | 484 | 0 |
| F_e2_s6102_g128 | 0 | 12 | 50 | 4.870740e-13 | 3.696217e-12 | 0 | 0 | 0 | 0 | 0 | 482 | 0 |
| F_e2_s6102_g256 | 0 | 50 | 50 | 8.223086e-13 | 5.679360e-12 | 0 | 0 | 0 | 0 | 0 | 482 | 0 |
| F_e2_s6103_g32 | 0 | 6 | 50 | 8.374818e-12 | 1.489147e-11 | 0 | 0 | 0 | 0 | 0 | 2288 | 0 |
| F_e2_s6103_g64 | 0 | 7 | 50 | 3.662682e-11 | 5.130068e-11 | 0 | 0 | 0 | 0 | 0 | 2260 | 0 |
| F_e2_s6103_g128 | 0 | 13 | 50 | 8.776535e-11 | 6.402827e-11 | 0 | 0 | 0 | 0 | 0 | 2252 | 0 |
| F_e2_s6103_g256 | 0 | 56 | 60 | 3.683051e-11 | 1.226788e-11 | 0 | 0 | 0 | 0 | 0 | 2249 | 0 |

derived: wall of the 24 runs: min 5 s, max 56 s, sum 450 s.

#### Recorded (no threshold): `D_P = sigma_T^2(Pollock) - sigma_T^2(consumer)` from the campaign `F_DP` lines of `results.txt` (computed on the V100 from the per-particle `.npy` arrays, absent from `raw/`; 84 lines)

| eps | cells | concordant (campaign) | discordant (campaign) | recomputed class agrees | median |D_P| ny=32/64/128/256 | max |D_P| ny=32/64/128/256 | consumer sigma_T^2 range at ny=256 |
|---|---|---|---|---|---|---|---|
| 1 | 42 | 21 | 21 | 42 / 42 | 5.07e-05 / 1.36e-05 / 1.86e-05 / 2.80e-06 | 3.31e-04 / 1.82e-04 / 1.33e-04 / 8.61e-06 | 2.9053e-02 - 7.5889e-02 |
| 2 | 42 | 28 | 14 | 42 / 42 | 2.69e-04 / 5.72e-05 / 3.73e-05 / 6.38e-06 | 1.06e-03 / 3.94e-04 / 2.73e-04 / 3.06e-05 | 1.5400e-02 - 9.8512e-02 |

V3b recomputation (derived, RR8: concordant iff |D_P| decreases on every refinement and |D_P(ny=256)| <= max(0.05 sigma_T^2(consumer, ny=256), 2 SE_D(ny=256))): agrees with the campaign label on 84 of 84 cells. Recorded only; no `alpha` content.

`F_V3B_SUMMARY` lines of the campaign: `F_V3B_SUMMARY eps=1 concordant=21 discordant=21 not_evaluable=0 (of 42 cells; non-blocking)`; `F_V3B_SUMMARY eps=2 concordant=28 discordant=14 not_evaluable=0 (of 42 cells; non-blocking)`.
derived: max `csv_max_rel` (recomputed var_x2/var_x3 against the programs' moments CSVs) over all F_DP lines: 2.410e-14.
Lines in the campaign `results.txt` that are not `D_P` rows: 0.

### G. Regression (met)

`g_compare.py verdict R:sf41_campaign/final/results.txt --ctest-times R:sf41_campaign/final/ctest_times.tsv` re-run: `G_VERDICT PASS`. Full V100 `ctest`: 52 of 52 passed, 3182 s (SF-38 note: 46 of 46 in 3065 s; the six new entries take 71.0 s together, *derived* from `ctest_times.tsv`);
the SF-27 trio (172 `IDENTICAL trio:` items) and the SF-38 R set (269 `IDENTICAL r:` items plus one `IDENTICAL binary:md5_differs` item confirming head != base) byte-identical, and the `IDENTICAL g_runner_a:rc` / `g_ab:rc` lines of G(ii) and G(iii), counted separately; `runner_medium_a` 67 of 67 identical (DONE files included, `timings.json` excluded), `medium_ab` 22 of 22 identical with exactly the two allowed extras `realizations/r_000{0,1}/exits_b.csv`;
`interpolation_periodic_tricubic` (4.04 s) and `interpolation_hessian` (2.86 s) Passed. RR9 (iv), the emptiness of the diff `2cf48c1..269c658` for the periodic spline files, is **confirmed** (by the orchestrator, and re-run by me): `git diff --stat 2cf48c1 269c658 -- <the four files>` prints nothing, and `git rev-parse <commit>:<file>` gives identical blob ids at `2cf48c1` and `269c658`: `tests/interpolation/periodic_tricubic_bspline_tests.cu` `3a6256e3295409e6d16523312e6eb165a5da5819`, `tests/interpolation/spline_hessian_tests.cu` `3100a33938bd6a12788e33301dbfa08f70261789`, `src/numerics/interpolation/PeriodicTricubicBSpline.cuh` `556b97199abfb7d364ab6e757d03b10f97480055`, `src/numerics/interpolation/PeriodicTricubicBSpline.cu` `33d9504dd21503c71aacaaf556f3166b4c44cfa6`.

```
G_CTEST exit=0 tests=52 failed=0 wall_s=3182 PASS (summary: 100% tests passed, 0 tests failed out of 52)
G_TRIO IDENTICAL (SF-27 trio byte-compare clean vs /home/sesquerre/sf41_base_refs/trio; manifest keys timestamp build.git_hash dropped)
G_RSET IDENTICAL (R set byte-identical to /home/sesquerre/sf41_base_refs/r; SF-40 compare rules)
G_BYTES name=runner_medium_a files_ref=67 files_cand=67 identical=67 diff=0 missing=0 extra_allowed=0 extra_unexpected=0 excluded=2/2 IDENTICAL
G_BYTES name=medium_ab files_ref=22 files_cand=24 identical=22 diff=0 missing=0 extra_allowed=2 extra_unexpected=0 excluded=2/2 IDENTICAL
G_BYTES_EXTRA name=medium_ab path=realizations/r_0000/exits_b.csv
G_BYTES_EXTRA name=medium_ab path=realizations/r_0001/exits_b.csv
G_VERDICT PASS
```

| item | count (derived from `results.txt` lines) |
|---|---|
| `IDENTICAL trio:` items (SF-27 trio) | 172 |
| `IDENTICAL r:` items (SF-38 20-run R set) | 269 |
| `IDENTICAL binary:md5_differs` item (R set, head != base binary) | 1 |
| R set total (`r:` + `binary:`) | 270 |
| `IDENTICAL g_runner_a:` items (G(ii), the rc line) | 1 |
| `IDENTICAL g_ab:` items (G(iii), the rc line) | 1 |
| all `IDENTICAL` lines | 444 |
| DIFF / MISSING / EXTRA lines | 0 |
| `ctest_times.tsv` rows | 52 |
| rows `Passed` | 52 |
| sum of per-test seconds | 3181.58 |

binary line: `IDENTICAL binary:md5_differs head=ded79b8ad5ad019d81e0dc8bea1d980b base=aae63d2868be1627e80c16a3a41eb479`

#### Named ctest entries (`ctest_times.tsv`)

| test | status | seconds |
|---|---|---|
| streamfunction_operator_tests | Passed | 1350.08 |
| interpolation_periodic_tricubic | Passed | 4.04 |
| interpolation_hessian | Passed | 2.86 |
| interpolation_nonperiodic_x | Passed | 3.87 |
| inlet_slab_trace_gpu | Passed | 33.33 |
| lagrangian_tracker_gpu_host | Passed | 13.83 |
| lagrangian_cross_representation | Passed | 5.73 |
| lagrangian_dirichlet_long_box_stage | Passed | 10.04 |
| lagrangian_pollock_nonperiodic | Passed | 3.38 |
| lagrangian_dirichlet_controls | Passed | 6.21 |
| lagrangian_ensemble | Passed | 48.28 |
| lagrangian_ensemble_medium_b | Passed | 41.76 |
| lagrangian_medium_ab | Passed | 62.23 |

Five longest tests: streamfunction_operator_tests 1350.08 s; streamfunction_picard_fixed 429.78 s; streamfunction_gmres 262.86 s; streamfunction_continuation 163.78 s; streamfunction_anderson 144.79 s.

#### Base reference runs (`sf41_base_refs`) and head runs of the byte-compare sets

`BASEREFS_RECORDED r_runs=20 trio_runs=3 g_runs=2 recorded_in_exit_codes=25 unexpected_exit_codes=0`
base reference runs in `exit_codes.tsv`: 25; non-zero (run: rc) in the base part: r_trace_N32_e1.0_gpu: 21, r_trace_N32_e1.0_host: 21, r_trace_N64_e2.0_gpu: 21; in the head `final` part: r_trace_N32_e1.0_gpu: 21, r_trace_N32_e1.0_host: 21, r_trace_N64_e2.0_gpu: 21 (26 head runs); same non-zero set and codes: yes; every head run name except `ctest` also a base run: yes (derived).

### H. SF-38 A/B re-run with the exit-aware Pollock (recorded; zero exits, 20 of 20 identical)

`h_verdict.py` re-run: `H_VERDICT ALL class=OK_ZERO_EXITS_IDENTICAL exits_inlet=0 exits_outlet=0 runs_exit_ok=1 OK`. Zero medium (b) particles crossed `x = 0` in any of the 20 realizations and any plane (derivation in SPEC In scope 3: Dirichlet heads make the discrete operator an irreducible M-matrix, positive inlet face fluxes), and `e2_b.csv` is byte-identical to the committed SF-38 file in 20 of 20
realizations; the control `e2_a.csv` is identical in 20 of 20; `ab_primary.csv` of both amplitudes equals SF-38's. With zero exits the `C_STAT` lines were not recomputed (`ab_decision.py` was not needed); the SF-38 selection of medium (b) is not re-decided here. Every per-plane count of `H_EXITS_BY_PLANE` is 0 (8 planes for `s1`, 7 for `s4`). This agrees with the post-hoc reading of the SF-38 note (P6, "plausibly zero, not measured").

```
H_VERDICT amp=s1 run_exit=0 realizations=10 e2_b_identical=10/10 e2_a_control_identical=10/10 exit_inlet_total=0 exit_outlet_total=0 class=OK_ZERO_EXITS_IDENTICAL
RECORDED H_AB_PRIMARY amp=s1 ab_primary.csv_vs_SF38=IDENTICAL
H_VERDICT amp=s4 run_exit=0 realizations=10 e2_b_identical=10/10 e2_a_control_identical=10/10 exit_inlet_total=0 exit_outlet_total=0 class=OK_ZERO_EXITS_IDENTICAL
RECORDED H_AB_PRIMARY amp=s4 ab_primary.csv_vs_SF38=IDENTICAL
H_VERDICT ALL class=OK_ZERO_EXITS_IDENTICAL exits_inlet=0 exits_outlet=0 runs_exit_ok=1 OK
```

| amp | r | Np | planes | x=0 exits (all planes) | outlet exits (all planes) | e2_b.csv vs SF-38 | e2_a.csv vs SF-38 (control) | class |
|---|---|---|---|---|---|---|---|---|
| s1 | 0 | 10000 | 8 | 0 | 0 | IDENTICAL | IDENTICAL | OK_ZERO_EXITS_IDENTICAL |
| s1 | 1 | 10000 | 8 | 0 | 0 | IDENTICAL | IDENTICAL | OK_ZERO_EXITS_IDENTICAL |
| s1 | 2 | 10000 | 8 | 0 | 0 | IDENTICAL | IDENTICAL | OK_ZERO_EXITS_IDENTICAL |
| s1 | 3 | 10000 | 8 | 0 | 0 | IDENTICAL | IDENTICAL | OK_ZERO_EXITS_IDENTICAL |
| s1 | 4 | 10000 | 8 | 0 | 0 | IDENTICAL | IDENTICAL | OK_ZERO_EXITS_IDENTICAL |
| s1 | 5 | 10000 | 8 | 0 | 0 | IDENTICAL | IDENTICAL | OK_ZERO_EXITS_IDENTICAL |
| s1 | 6 | 10000 | 8 | 0 | 0 | IDENTICAL | IDENTICAL | OK_ZERO_EXITS_IDENTICAL |
| s1 | 7 | 10000 | 8 | 0 | 0 | IDENTICAL | IDENTICAL | OK_ZERO_EXITS_IDENTICAL |
| s1 | 8 | 10000 | 8 | 0 | 0 | IDENTICAL | IDENTICAL | OK_ZERO_EXITS_IDENTICAL |
| s1 | 9 | 10000 | 8 | 0 | 0 | IDENTICAL | IDENTICAL | OK_ZERO_EXITS_IDENTICAL |
| s4 | 0 | 10000 | 7 | 0 | 0 | IDENTICAL | IDENTICAL | OK_ZERO_EXITS_IDENTICAL |
| s4 | 1 | 10000 | 7 | 0 | 0 | IDENTICAL | IDENTICAL | OK_ZERO_EXITS_IDENTICAL |
| s4 | 2 | 10000 | 7 | 0 | 0 | IDENTICAL | IDENTICAL | OK_ZERO_EXITS_IDENTICAL |
| s4 | 3 | 10000 | 7 | 0 | 0 | IDENTICAL | IDENTICAL | OK_ZERO_EXITS_IDENTICAL |
| s4 | 4 | 10000 | 7 | 0 | 0 | IDENTICAL | IDENTICAL | OK_ZERO_EXITS_IDENTICAL |
| s4 | 5 | 10000 | 7 | 0 | 0 | IDENTICAL | IDENTICAL | OK_ZERO_EXITS_IDENTICAL |
| s4 | 6 | 10000 | 7 | 0 | 0 | IDENTICAL | IDENTICAL | OK_ZERO_EXITS_IDENTICAL |
| s4 | 7 | 10000 | 7 | 0 | 0 | IDENTICAL | IDENTICAL | OK_ZERO_EXITS_IDENTICAL |
| s4 | 8 | 10000 | 7 | 0 | 0 | IDENTICAL | IDENTICAL | OK_ZERO_EXITS_IDENTICAL |
| s4 | 9 | 10000 | 7 | 0 | 0 | IDENTICAL | IDENTICAL | OK_ZERO_EXITS_IDENTICAL |

planes (x1) of amplitude s1: 0.234375,0.2734375,0.3515625,0.5078125,0.8203125,1.4453125,2.6953125,5.1953125

planes (x1) of amplitude s4: 0.75,0.875,1.125,1.625,2.625,4.625,8.625

derived: realizations 20; x = 0 exits total 0; e2_b identical 20 of 20; e2_a control identical 20 of 20.

Per-run walls (`walls.tsv`): ab_s1 1110 s; ab_s4 1618 s.

### Recorded (no threshold)

- **GPU time per part (decision E1-E3)**: the first "Parts" table and the GPU-hours table above (`RECORDED part_seconds`). All evidence parts plus the base build window: 8096 s = 2.249 h (*derived*); the `abrerun` (2751 s) and `final` (3407 s) parts are 76 % of that total (*derived*, (2751 + 3407) / 8096 = 0.7606).
- **Medium (b) memory** (tables of section E, *measured*): exact bytes of the solve phase 8129757192 / 29724424712 / 29724424712 / 26421710856 B for E1-E4 (7753.141 / 28347.420 / 28347.420 / 25197.707 MiB); PID peaks 8251 / 28855 / 28855 / 25701 MiB; the stage build `seconds` 13.870 / 67.512 / 74.824 / 90.314 s; host max RSS 4.86 / 16.02 / 16.02 / 14.31 GB (kB column, *derived* unit conversion); wall 33.69 / 110.28 / 129.16 / 143.81 s.
- **Flow statistics (`flow_stats.json`)** in the table of section E: fraction of `x` faces with `u <= 0` is 0 for E1 and E4, 2230 of 490881024 (4.5e-6, *derived*) for E2 and 173142 of 490881024 (3.5e-4, *derived*) for E3; the interior tilt `theta_2`, `theta_3` of E4 is -3.18e-2 / -3.10e-2, of E3 3.41e-3 / 1.55e-2, of E2 -9.42e-4 / 4.56e-3, of E1 -9.61e-4 / 1.90e-3 (single realizations; recorded only).
- **Tracker versus Pollock exit counts in the time pass** (section E): differences -135, +15, +4, -14 particles (E1-E4; Pollock minus tracker, *derived*).
- **B decay factors**, **A measured-minus-emulated errors**, **V3b classifications** and **H exits**: in the tables of A, B, F and H (the four items the RR12 closing rule asks to report explicitly).

## Evidence re-verification

All commands were run from the repository root on the worktree at the evidence commit, outside any build or GPU; each exited 0. The comparison "found verbatim" is `sort | comm -12` of the verdict output against the part's campaign `results.txt` (checked by the author of this note).

| command (README "Re-running a verdict from `raw/`") | result | verbatim in the campaign `results.txt` |
|---|---|---|
| `python3 $S/a_verdict.py $R/controls` | `A_VERDICT run=A_full exit=0 program_all_pass=1 gated_pass=79 gated_fail=0 inconsistent=0 missing=0 memcheck_exit=0 memcheck_errors=0 PASS` | 98 of 98 lines |
| `python3 $S/c_verdict.py $R/controls` | `C_VERDICT PASS` (`C1`-`C5` PASS: 36, 18, 18, 72 gated, 4 rows) | 38 of 38 |
| `python3 $S/d_verdict.py $R/controls --run D` | `D_VERDICT run=D exit=0 sizes=[...] PASS` | 36 of 36 |
| `python3 $S/b_verdict.py $R/cross --run B` | `B_VERDICT run=B exit=0 ... gate=1e-11 PASS` | 37 of 38 (see below) |
| `python3 $S/e_verdict.py $R/prodgrid` | `E_VERDICT ALL PASS (E1=PASS,E2=PASS,E3=PASS,E4=PASS)` | 113 of 113 |
| `python3 $S/f_verdict.py $R/evladder` | `F_VERDICT ALL MET` | 87 of 95 (see below) |
| `python3 $S/h_verdict.py $R/abrerun` | `H_VERDICT ALL class=OK_ZERO_EXITS_IDENTICAL exits_inlet=0 exits_outlet=0 runs_exit_ok=1 OK` | 45 of 45 |
| `python3 $S/g_compare.py verdict $R/final/results.txt --ctest-times $R/final/ctest_times.tsv` | `G_VERDICT PASS` | 1 of 1 |
| `bash $S/selftest_all.sh` | twelve `SELFTEST OK`, `SELFTEST_ALL failing_scripts=0` | - |

Differences from the campaign output (reported, not fixed): (i) `b_verdict.py` prints `RECORDED B_DECAY_SUMMARY field=h_tilde quantity=hessian n=6 mean_factor=0.267950 ...` where the campaign line has `mean_factor=0.267951` (the other three summary lines agree; a recorded, ungated mean that rounds at the last printed digit);
(ii) `f_verdict.py` prints 6 `RECORDED F_DP ... not_evaluable (unreadable: ... weights.npy ...)` lines and 2 `RECORDED F_V3B_SUMMARY ... not_evaluable=42` lines instead of the campaign's 84 `D_P` rows and the summaries `concordant=21 discordant=21` / `concordant=28 discordant=14`, because the per-particle arrays are excluded from `raw/`; the 84 `F_CELL` lines and the three `F_VERDICT` lines are verbatim.
The verdict scripts and the campaign agree on every gated line.

Integrity: `sha256sum -c --ignore-missing sha256.txt` inside each of the nine part directories of the rebased `raw/` (non-OK lines counted with `grep -vc ': OK'`; listed / OK / non-OK): `sf41_base_refs` 565 / 372 / 0; `build` 8 / 8 / 0; `abrerun` 258 / 258 / 0; `controls` 33 / 33 / 0; `cross` 4 / 4 / 0; `evladder` 984 / 576 / 0; `final` 572 / 379 / 0; `prodgrid` 196 / 128 / 0;
`sf41_campaign_superseded/build_attempt1_tmpfull` 8 / 8 / 0. The OK counts are lower than the listed counts by the absent `.npy` entries (193, 408, 193, 68 for baserefs, evladder, final, prodgrid, *derived* as listed minus OK; those files are *not in the imported evidence*). `raw/` holds 1861 regular files and 360 directories (`find raw -type f | wc -l`, `-type d`), the README's tarball file count. The pre-registration sha256 of the working-tree file and of `git show fa3f253:...` is `8f424fa01d3eaf75743b7361ab0e6e508bd47b7da209be844d4fb33ccabd29d8` in both.
`python3 A/scripts/note_tables.py --selftest` prints `note_tables selftest: OK`; two successive runs of `python3 A/scripts/note_tables.py A/raw A/analysis` leave `git status --porcelain A/analysis` unchanged (no diff). `bash scripts/hooks/check-lester-increments.sh`: `Lester increment harness: OK (42 increments, ready=SF-34, nonterminal=SF-41)`.

## Disclosures / deviations

Each item says what I verified against files in the repository and what I could not.

- **Agent model** (orchestrator fact; **not verifiable** from the repository): workers and the integrator ran with the locally configured (uncommitted) agent definitions, model `sonnet`, effort `high`, instead of the checked-in `opus` / `medium`; the orchestrator ran as Claude Opus 5.5. Local builds only through `mf3d-build`, local tests `<= 32^3` by explicit `-R`.
  The author of this note is such a worker; this note itself ran no build, no `ctest`, no GPU and no campaign script (only the verdict scripts' and `note_tables.py`'s text parsing). No threshold depends on it.
- **Build attempt 1 failed (infrastructure; verified)**: `R:sf41_campaign_superseded/build_attempt1_tmpfull/` (`DATE_UTC=2026-10-25T02:02:40Z`, `PART_EXIT` 1, `BUILD_VERDICT FAIL missing_binaries=[... 9 binaries ...]`); `clean_build.log` contains `fatal error: error writing to /tmp/....s: No queda espacio en el dispositivo` (lines 39, 46, 53) and `/tmp/ccKi7hJ5.s:128076: Error: pseudo-operador desconocido: «.qu»` (line 70).
  Its record was moved unchanged to `~/sf41_campaign_superseded/build_attempt1_tmpfull` and the job re-run (attempt 2) with `TMPDIR=$HOME/sf41_tmp`, visible in the command of the seven head job logs (`grep TMPDIR` finds it in `sf41-{build,controls,cross,prodgrid,evladder,abrerun,final}.log` and in none of the other logs). Attempt 1 and attempt 2 record the same `TREE_src_digest`: no source change. The attempt-1 `/tmp` size (2.0 GB) is an orchestrator fact, not in `raw/`.
- **Import omission, corrected after the first version of this note (orchestrator fact, verified against the rebased `raw/`)**: the orchestrator's first import commit `f91ace2` omitted 346 files whose paths match `.gitignore` (`build/`: the 14 files of `raw/sf41_campaign/build/`; `output*/`: the 332 trio output files under `raw/sf41_base_refs/trio/` and `raw/sf41_campaign/final/trio/`); commit `9a48e29` added them (`git add -f`) and amended the README "Evidence import". My first draft reported a missing attempt-2 build record and an unreproducible file count; both were consequences of that omission and are withdrawn.
  After `9a48e29`, `find raw -type f | wc -l` gives 1861 (and 360 directories), which is the README's tarball file count; `raw/sf41_campaign/build/` is present and used in this note; the counts of "Evidence re-verification" are recomputed from it.
- **Stale-object hazard found in a pre-check (verified in the pre-check log, not evidence)**: `sf41-pre-n2` (`TREE=ef7401e`, incremental build on the shared mirror, `PRECHECK_NOT_EVIDENCE` in its command) printed `D_CHECK size=128x32x32 head_identical=0 bsolver_identical=1 mapped_identical=0`, and the same `head_identical=0` at 512x64x64 and 3328x256x256. The root cause (a stale `DirichletMedium.cu.o` kept by an incremental build, compared vacuously on size mismatch) is the orchestrator's and the commit subjects `ed8fb6a` / `63b79ff` (N2c) say so;
  the clean-build pre-check `sf41-pre-n2c-clean` ends with `VERDICT mode=D checks_pass=40 checks_fail=0` (not evidence). Process after that: every head build is clean and every part verifies the binaries' md5 (`BINARIES_CHECK` in `run.env`, verified above).
- **Pre-checks are not evidence (verified)**: eight `sf41-pre-*` jobs between 2026-10-25T00:08:34Z and 01:40:58Z ran intermediate node trees (`sf41-pre-n1-full`, `-n2`, `-n3b`, `-n1b`, `-n2c-diag`, `-n2c-d`, `-n2c-clean`, `-e`; table of jobs). One of them, `sf41-pre-e` (`TREE=7ce48f4`, `CLEAN_BUILD`), ran an ad hoc script `bash ~/sf41_pre_e.sh` that is not under `scripts/` of the repository (verified from its command line); that it was copied by `scp` and that its PID-peak reading was wrong is an orchestrator fact **not verifiable from `raw/`**. No number of this note comes from a pre-check.
- **Correctives before any evidence run (verified from the commit subjects)**: N2c (`ed8fb6a`, `63b79ff`: BLOCKING D finding at >= 4 MG levels / Release, root cause the stale object per the orchestrator); N7 round 1 gated D's memory, corrected by `269c658` ("D memory recorded, not gated (threshold D is bitwise only)"); the D memory is recorded in `D_MEM` and is not part of `D_VERDICT`.
- **Base binaries and the order of `sf41-baserefs` (verified; deviation from the SPEC command order)**: the base binaries were built by `sf41-basebuild` (2026-10-24T23:44:20Z, `TREE=2cf48c1`), before any pre-check job. The SPEC "Validation commands" list `baserefs` right after `basebuild`, "before any SF-41 source reaches the mirror"; `sf41-baserefs` actually started at 2026-10-25T01:47:25Z,
  after the pre-checks, with the command `echo TREE=0256bb0 SCRIPTS_ONLY_FOR_BASEREFS BASE=2cf48c1`, i.e. on a mirror tree containing SF-41 sources but using only the committed scripts and the base binaries copied to `~/sf41_base_refs/bin`. The md5 of those binaries in `R:sf41_base_refs/run.env` equal the md5 printed by the base build job (`inlet_slab aae63d28...`, `macroflow3d_pipeline fba69968...`, `lagrangian_ensemble 59ac27b5...`, `lagrangian_medium_ab a5380058...`).
  RR1 as written in `PRE` ("base binaries are built from a clean tree ... before any SF-41 source reaches the mirror ... They produce the G references (job `sf41-baserefs`)") is met; the SPEC wording about the references is not. Verified with git: `git diff --stat 0256bb0 269c658 -- docs/experiments/artifacts/2026-10-25-sf41-medium-b-lagrangian-stage/scripts` is empty (the scripts `sf41-baserefs` ran, tree `0256bb0`, equal H1's) and `git diff --stat 2cf48c1 269c658 -- 'apps/*.yaml'` is empty (the run configurations are unchanged between base and head). I do not judge whether this matters; the byte-compare items of G compare base-binary outputs with head-binary outputs.
- **Activation time stamp (orchestrator finding; verified with git)**: the activation bitacora row is stamped `2026-10-25T00:30Z` and the SPEC `Started:` field says `2026-10-25T00:30Z on base 2cf48c1`, but the activation commit `fa3f253` (which contains the pre-registration) is dated 2026-10-24T20:43:21-03:00 = 2026-10-24T23:43:21Z (`git log -1 --format=%cI fa3f253`), about 47 minutes earlier. This note cites the commit time as the activation time. What RR0 needs holds:
  every remote job started after that commit (first `sf41-basebuild` 2026-10-24T23:44:20Z, the first pre-check finished 2026-10-25T00:09:32Z, the first evidence head job `sf41-build` started 2026-10-25T02:10:44Z, all in `R:job_logs/`). The SPEC is not edited here; the orchestrator records the correction in a new bitacora row.
- **`D_P` / V3b not re-evaluable from `raw/`** (verified, section F and "Evidence re-verification"): their per-particle inputs are excluded by the pack; the tables use the campaign's printed lines.
- **Host clocks**: `nvidia-smi` banner and sampler timestamps are host-local time, three hours behind UTC (*derived*: banner 2026-10-24 23:16:55 against `DATE_UTC=2026-10-25T02:16:44Z` of the same part, offset 3.00 h; consistent with the E4 sampler start).
- **Differences from the plan found in the raw evidence** (reported as found, nothing resolved): (i) the `balance` printed by `E_RUN` is recomputed by `e_verdict.py` from the 13-digit `Q_in`, `Q_out` and differs in the third significant digit from the solver's own `balance` (E1 2.370176e-11 against 2.348053e-11; both `<= 1e-8` by orders of magnitude); (ii) the tracker and Pollock time-pass exit counts differ (section E);
  (iii) the F runs have no balance gate; the largest F balance is `1.928310e-09` (`F_e2_s6102_g64`) and the largest F solver `rel` `8.776535e-11` (`F_e2_s6103_g128`); (iv) the F Pollock `sigma_T^2` changes between `ny = 128` and `ny = 256` by 7.7e-5 in `F_e1_s6102` plane 0 (`5.646064e-02` to `5.653722e-02`) while the consumer's changes by 3.5e-7, so `D_P` changes sign there; recorded, not analysed;
  (v) the A ladder RMS errors are 6-9 % above the design emulation (the max-norm errors 0.8-2.4 % below; the cause is not examined).
- **Not in the imported evidence**: the per-particle `.npy` arrays; the base build's `ctest -N` name list (only the count 46); the tarball and its transfer; the pre-check outputs other than their logs.

## Result

Per criterion (SPEC thresholds A-H), with the decisive numbers:

| criterion | verdict | decisive numbers |
|---|---|---|
| A spline contracts | **met** | 79 of 79 gated checks pass on host and device, memcheck 0 errors; ladder minima 3.901686 / 4.363914 / 2.908731 / 3.373425 / 1.917651 (gates 3.8 / 3.8 / 2.8 / 2.8 / 1.8; predicted 3.9024 / 4.3659 / 2.9042 / 3.3858 / 1.9154); x-only 64->128 3.98698 / 2.985752 / 1.986263; node `1.776e-15`; PoU `3.109e-15`; CPU/GPU Hessian `6.021e-14` |
| B cross-representation | **met** | value `9.936e-16`, gradient `2.113e-14`, Hessian `9.724e-13` (gate `1e-11`); decay factors 0.267938-0.267958 (24 fits) |
| C exact controls | **met** | 144 of 144 gated rows pass; C2 `tau` `5.023e-11`; C3 max abs `8.882e-16`; C4b 768 / 1792 exact on both grids, mismatches 0; C5 `rel` `7.970487e-11` worst |
| D lean stage bitwise | **met** | head, `B_SOLVER` identical at 128x32x32, 512x64x64, 3328x256x256, mapped-host identical at the two small sizes, C0 line 11 of 11 fields; 60.571 B/cell at 7 levels |
| E production feasibility | **met** (E1-E4, device head) | rel `<= 3.41e-11`, balance `<= 4.2e-11`; PID peak 8251 / 28855 / 28855 / 25701 MiB (`<= 30000`); max phase = S2 in all; no exits at `x = 0` |
| F `e_v` ladder | **met** (both amplitudes) | 84 of 84 cells strictly decreasing, last-pair order 1.9920-2.0263 (`eps` 1), 1.9959-2.0040 (`eps` 2); `D_P` / V3b recorded (21 / 21 and 28 / 14 concordant / discordant) |
| G regression | **met** (campaign items and RR9 (iv)) | ctest 52 of 52 (3182 s); trio 172 and R set 269 + 1 `IDENTICAL` items; `runner_medium_a` 67 of 67, `medium_ab` 22 of 22 plus two allowed `exits_b.csv`; periodic tests Passed; periodic spline files: identical blob ids at `2cf48c1` and `269c658` |
| H recorded | **zero exits, 20 of 20 identical** | `x = 0` exits 0 in 20 realizations; `e2_b.csv` and control `e2_a.csv` identical 20 of 20; `C_STAT` not recomputed |

Answer to the Goal's claims: the lean stage's head is bit for bit SF-38's, its measured device peak is `<= 30000` MiB at 3328x384x384 (28855 MiB) and at 6656x256x256 (25701 MiB); the non-periodic spline, the explicit `x` exits in the tracker and in the Pollock comparator, and the runner integration passed the exact controls, the cross-representation test and the `e_v` ladder at both amplitudes;
the periodic paths are byte-identical in the regression sets. No `alpha_L` / `alpha_T` statement follows from any number here; this note does not decide the State, and the SF-38 medium selection is not re-decided by H.

## Caveats

- **Memory margin**: E2 and E3 peak at 28855 MiB, 1145 MiB (3.8 %) below the gate; the sampler is 200 ms and one run per geometry (run-to-run variation not characterized; E2 and E3 have the same exact bytes and the same PID peak). The device-head placement was sufficient; the pre-authorized mapped-host fallback was not exercised at a production grid (only at the two small D sizes).
- **One realization per E run, three seeds per F rung and one seed in A/B-type controls**: F's ladder is three realizations per amplitude at `ell = 0.5` (transverse period `2 lambda`); the production `lambda/h` and width are not tested here (SF-39 re-verifies them, E2).
- **D memory** is the process peak including the SF-38 reference; no threshold uses it.
- **`D_P` and V3b** are the campaign's printed values; they could not be recomputed from `raw/`; the V3b label is non-blocking and its recomputation from the printed numbers agrees in 84 of 84 cells.
- **Tracker and Pollock time-pass exit counts** differ by up to 135 particles (E1); no threshold compares them, and the cause is not examined here.
- **Tilt of the medium (b) flow in fixed axes** (E3, E4 `theta_2`, `theta_3` up to 3.2e-2 in a single realization) is recorded; SF-38 P4 measured rms 0.0227 at `sigma^2 = 4`; how SF-39 treats it is not decided here.
- **Agent model and attempt-2 build part record**: see Disclosures.
- No `alpha` claim: the E, F and H quantities are recorded as pre-registered inputs only, never interpreted (Rule 1).

## Next step

Measurement is complete; the orchestrator performs FINAL_AUDIT on the integrated commit and records the State; this is a human-review increment (`awaiting_review` after the audit). After closure (merged and visible on the default branch) `SF-39` becomes eligible by the advancement rule. Items for the owner (not decisions): the memory margin at E2/E3; the activation time stamp correction (new bitacora row); the `sf41-baserefs` ordering relative to the SPEC text.
