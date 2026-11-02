I've finished the audit. Nothing under ROOT or the main repo was modified, and nothing ran on the remote host or a GPU.

**The test run, and two things you should know about it:**
- **Results.**
  - `scripts/tests/remote_gpu_select_test.sh`: ALL PASS (12 cases).
  - `scripts/tests/remote_blinded_test.sh`: ALL PASS (49 checks, 0 FAIL).
  - SF-42 suite with `python3 -B -W error -m unittest discover`: `Ran 369 tests in 53.194s, OK (skipped=1)`, exit 0. The `summary: FAIL` line in the log comes from a spectral test that exercises the failure path.
- **Temp location.** `analysis-exec/` did not exist and read-only mode forbids `mkdir`. I pointed `TMPDIR` at the existing scratchpad root instead; the tests created and removed their own temp dirs there.
- **Stray `__pycache__` in ROOT.** Two directories exist that I did not create: `scripts/mcstat/__pycache__` (10:44:16, before my test run) and `scripts/seedlattice/__pycache__` (10:46:12, while my run was in the spectral tests). My run used `-B` and `PYTHONDONTWRITEBYTECODE=1`, so another agent sharing ROOT probably wrote them. I left them in place.

**Files cited below.** All paths are under ROOT = `/tmp/claude-1000/-home-sesquerre-Projects-MacroFlow3D/0b813f1d-3625-4dbd-9e7d-1066997d5f71/scratchpad/master-check`. Line numbers refer to these files.
- [remote] `ROOT/scripts/remote`; [env] `ROOT/scripts/remote.env`; [tsel] `ROOT/scripts/tests/remote_gpu_select_test.sh`; [tbl] `ROOT/scripts/tests/remote_blinded_test.sh`
- [sf39] `ROOT/docs/experiments/artifacts/2026-10-27-sf39-macrodispersion-study/scripts/run_sf39.sh`
- [S42] `ROOT/docs/experiments/artifacts/2026-10-30-sf42-confirmatory-tooling/scripts/` (design.py, run.py, sf42red, mcstat, recompute, reconcile live here)
- [ER] `ROOT/src/physics/particles/lagrangian_labels/EnsembleRunner.cu`; [ERh] the matching `.cuh`; [DLB] `ROOT/src/physics/particles/lagrangian_labels/DirichletLongBoxStage.cu`; [LT] `ROOT/src/physics/particles/lagrangian_labels/LagrangianTracker.cu`
- [PE] `ROOT/src/runtime/ensemble/EnsembleRunner.cu` (the Par2 pipeline); [CY] `ROOT/src/io/config/ConfigYaml.cpp`; [CV] `ROOT/src/io/config/ConfigValidator.hpp`
- [P2] `/home/sesquerre/Projects/MacroFlow3D/src/external/Par2_Core` (same commit a1eecaef as master's pin; the snapshot's submodule directory is empty)

## 1. scripts/remote

**How `run` takes the slot and picks a GPU**
- **Validation.** `REMOTE_GPU_WAIT` and `REMOTE_GPU_MIN_FREE_MIB` must be integers. `--gpu <n>` is checked against `nvidia-smi` before ssh ([remote]:427-434). Defaults are 0 and 0 ([env]:15,17); the lock root is `~/.macroflow3d-remote` ([env]:14).
- **Start of the launcher.** It truncates the job log (`: >"$log_file"`, :606), writes `waiting_gpu` (:607) and sets one deadline of `SECONDS + gpu_wait` (:608).
- **Slot.** `exec 9>"$lock_root/gpu.lock"`, then `flock -w remaining` (or `flock -n` when the wait is 0) (:523-534). Failure writes `failed` with exit **75** and logs the current holder (:610-617, `describe_holders` :500-517).
- **Holder file.** `$lock_root/gpu.holder` holds `HOLDER_JOB/SESSION/INCREMENT/SINCE/GPU`. It is written once with GPU=pending and again after the choice (:490-498, :618, :625) and removed on exit only if this session owns it (:483-488). The file is informational; the flock is the authority (:480-482).
- **Choice.** Five `nvidia-smi` samples about 1 s apart (:536-543). `gpu_select` (:388-418) keeps GPUs whose minimum free memory over the samples (total − used) is at least the threshold, then takes the lowest mean utilisation, ties to more free memory, then the lower index.
  - Exit codes: 0 chosen, 1 none eligible (re-sample every 30 s until the deadline, then exit 75: :586-593, :620-624), 2 or a sampling failure gives exit **70** (:565, :569).
  - The log gets a per-GPU snapshot and `chose GPU n (reason)` (:572-583).
- **How the GPU reaches the job.** The helper itself does `export CUDA_VISIBLE_DEVICES="$gpu"` (:626), writes `running` (:627), logs `GPU=… (CUDA_VISIBLE_DEVICES=…)` (:631), then runs `bash -lc "$(cat command_file)"` in the mirror (:636-640). The final state is `succeeded` or `failed` with the command's exit code (:646-652).
- **Launch.** Over ssh, through tmux. It first kills any live session with the **same job name** (:658-661, `kill_session_snippet` :253-272).

**`--increment`.** Parsed before the subcommand (:12-27). It selects mirror `~/MacroFlow3D-<id>`, state root `~/.macroflow3d-remote/macroflow3d-<id>` and tmux prefix `macroflow3d-<id>` ([env]:4-12). The session name is `<prefix>-<job>` (:141-143). Logs, status, commands and launchers live under the state root ([env]:22-25).

**What a watcher sees**
- **States:** `waiting_gpu` (:529, :588, :607), `running` (:627), `succeeded` (:647), `failed` with EXIT_CODE 70, 75 or the command's code (:556, :615, :622, :649), `cancelled`/130 (trap :595-603; `cancel` rewrites the status file :785-800), and `unknown` when there is no status file (:289).
- **`status`** prints job, session, state, exit_code, tmux_active, gpu, working_dir, log, status, command and updated_at (:669-689).
- **`wait`** returns 0 on succeeded, EXIT_CODE on failed, 130 on cancelled, and keeps polling on anything else (:745-771).
- **`tail`** follows by default, or use `--lines N --no-follow` (:691-722).
- **`cancel`** sends TERM, then KILL, to the whole process group (:253-272, :799).

**Pitfalls for a 6-day single job:**
- `wait` never checks `TMUX_ACTIVE`. A launcher killed by SIGKILL or a host reboot stays `running` forever, so a watcher must treat `state=running` with `tmux_active=0` as dead.
- `UPDATED_AT` is only written at start and end, so there is no heartbeat.
- Resubmitting the same job name kills a live session (:659) and truncates the old log (:606). That log is the evidence the reconciliation reads, so every `--resume` submission needs a new job name.
- The SF-43 mirror cannot be synced while the job runs ([runbook] `ROOT/docs/runbooks/remote-v100.md`:97-99). All code, including the frozen analysis, is fixed at launch.
- `run_sf43.sh` must not exit with 70, 75, 97 or 130, which the helper already uses.

**Minimal N=2 design (`REMOTE_GPU_COUNT`, default 1, meaning "up to N")**
- **[env]:** add `: "${REMOTE_GPU_COUNT:=1}"`. Document it in the usage text (:93-106) and the runbook.
- **`gpu_select`:** add an optional third argument `cnt` (default 1). Collect the eligible GPUs in index order, then pick the best remaining one `cnt` times with the same comparison as today. Print `CHOSEN i` (unchanged when cnt=1) or `CHOSEN i,j`. Exit 0 if at least one is chosen, 1 if none, 2 on error.
- **Launcher:** pass `$gpu_count` to `gpu_select` (:567). Append `, up to N` to the reason only when N>1. Everything downstream already works with a comma list: the parse at :582, the export at :626 (giving `CUDA_VISIBLE_DEVICES=1,0`), `GPU=1,0` in the status file and `HOLDER_GPU=1,0`.
- **`run_command`:** require `REMOTE_GPU_COUNT` to be 1 or 2. Refuse `--gpu <n>` together with a count above 1. Add `gpu_count=N` to the blinded-log detail (:661) only when N≠1.
- **Only one GPU eligible:** take it at once. Never wait for a second GPU and never fail with 75 for that reason. With none eligible, behaviour is today's (wait until the deadline, then 75). The job learns how many it got by counting the entries in `CUDA_VISIBLE_DEVICES`.
- **Releasing the second GPU:** no helper change needed. The flock is one project-wide slot held for the whole job, the helper only arbitrates at launch, and colleagues never read our holder file. Lane B simply exits (its CUDA context frees the memory) and `run_sf43.sh` logs `LANE_B_RELEASED gpu=…`. `status` will keep showing `1,0`, which is cosmetic; I'd skip updating it.
- **Lanes must narrow the list themselves.** Both runners always use device 0 (`CudaContext ctx(0)`: `ROOT/apps/lagrangian_ensemble/lagrangian_ensemble_main.cu`:34, `ROOT/apps/macroflow3d_pipeline.cu`:46). Each lane therefore has to set `CUDA_VISIBLE_DEVICES` to its one entry from the helper's list.
- **Default 1 is unchanged** if the cnt=1 path yields the same `CHOSEN` line, the same exit codes, the same log lines and the same blinded detail. The 12 existing cases keep calling with two arguments.

**Governance blocker.** A two-GPU stage contradicts decision `ROOT/docs/decisions/2026-10-27-remote-single-gpu.md`:21 (one GPU at a time) and :31 (commands do not set `CUDA_VISIBLE_DEVICES` or run multi-GPU programs). It also contradicts `ROOT/AGENTS.md`:198-203, [runbook]:101 ("There is no multi-GPU exception") and the "one GPU" wording of G10 in `ROOT/docs/decisions/2026-10-29-sf39-remediation-sf42-sf43.md`:190. A new owner decision record is needed, and it should explicitly allow lanes to narrow the helper-granted list.

**Tests.**
- Existing: [tsel] covers `gpu_select` on canned CSV (12 cases, :25-43). [tbl] runs `ssh` as a local `eval` (:22); `run` is only checked for its submission guard (:113-124).
- No test executes the generated launcher today.
- To add in [tsel], with a count argument: two eligible ordered by utilisation (`CHOSEN 1,0`); the tie rule; one eligible (`CHOSEN 1`, rc 0); a memory-ineligible second GPU; none eligible (rc 1); the 12 existing cases unchanged.
- To add as a new `scripts/tests/remote_gpu_count_test.sh` in the fake-host style of [tbl]:
  - Stubs on PATH for `nvidia-smi` (canned CSV, uuid and compute-apps queries), `tmux` (runs the launcher synchronously) and `sleep`; the real `flock`; the state root in a temp dir.
  - The command file is `echo CVD=$CUDA_VISIBLE_DEVICES`.
  - Scenarios: count 1 (exact old log lines, CVD=1, status GPU=1); count 2 with two eligible (CVD=1,0, status, holder); count 2 with one eligible; none eligible with `REMOTE_GPU_WAIT=0` (failed, 75); pin plus count 2 refused before any ssh; an invalid count refused.
- Effort: about 8–12 worker-hours, including docs.

## 2. Runners per arm type — every option needed exists

**Consumer arms** (production, ladder, width, twin, Gate 5 consumer). Template from [S42] `pilots/design.py`:99-106 and `ROOT/docs/experiments/artifacts/2026-10-30-sf42-confirmatory-tooling/pilots/J3/c_s1_1040x240/cmd.txt`:
`lagrangian_ensemble --medium b --grid NX,NY,NY --sigma2 1 --eps E --ell LH/NY --window W --np-side M --planes <x0+d·ell, d=25..94> --times auto --tol 1e-8 --hmax-div 8 --fused-splines --single-launch --twin on --pollock-bridge on --seed0 S --nr N --out DIR`

| Need | Where it is supported |
|---|---|
| Accepted options | [ER]:1456-1462 |
| Amplitude via `--eps` 0.5/1/1.5/2 with `--sigma2 1` | design.py:16 |
| `--x0` defaults to 5·ell for medium b | [ER]:1546-1549 |
| Explicit planes: strictly increasing, > x0 | [ER]:1643-1645 |
| Planes must be < L1 for medium b | [ER]:1634-1636 |
| h_max passes 8/16/32 in one stage build | `--hmax-div 8,16,32` ([ER]:1569-1586), outputs under `hmax_<d>/` ([ERh]:45-51) |
| Twin (uniform-weight) series | `--twin on` ([ER]:805-817) |
| Per-particle crossings | planes_x23 / planes_tau / planes_I ([LT]:507-509) |
| Displacement basis and flux/unif/expI weights | computed by sf42red (reduce.py:311-318, 352); the runner's CSVs hold positions only ([LT]:524-530) |
| Distance window [94,194] in a ~208-λ box | supported: last plane at 199 λ < 208 λ; reduce takes `--d-window` ([S42] sf42red/__main__.py:29) |

**Par2** (`stdbuf -oL macroflow3d_pipeline config.yaml`; template design.py:109-175, J3/J3B `config.yaml`):
- `field_source: sf18_medium_b` with `{sigma2: 1, eps, seed0}`; the per-realization seed is seed0 + r ([PE]:127). Validation requires `K_mean` exactly 1, an even grid and `dx = 1/ny` exactly ([CV]:101-120).
- `inject_window` is supported ([CY]:300-308, [CV]:178-191, [PE]:824-836).
- Plane crossings: planes increasing, first > `inject_x`, last < Lx, non-periodic x ([CV]:193-215). A 208-λ box is fine.
- `cg_rtol` is read at [CY]:192 and used by the PCG-MG solve (`ROOT/src/physics/flow/solve_head.cuh`:121).
- Horizon 3.6: `dt = 0.0244140625/ny`, `n_steps = ceil(3.6·(L1 − 10·ell)/dt)` (design.py:89-96, J3B_HZ :256).

**The four factorial factors:**

| Factor | How to set it | Watch out for |
|---|---|---|
| G (Par2 random-spectral generator) | Omit `field_source` (validator message [CV]:91-94); the field seed becomes `stochastic.seed + r` ([PE]:392) | `stochastic.sigma2` and `n_modes` become live keys. design.py writes `sigma2: 1.0` and `n_modes: 1000`; they must become σ² and the pre-registered mode count (SF-39 used 10000) |
| I (full-plane injection) | Omit `inject_window` (the historical full-plane call) | — |
| D (dt/2) | Halve `dt`, double `n_steps` | Config keys only |
| R (cg_rtol 1e-6) | Set `cg_rtol: 1.0e-6` | Config key only |

**Gaps.** None of these needs a runner change.
1. design.py hardcodes d = 25..94 (:17, :46-53), the `cg_rtol` value (:19), `NR` and horizon handling, and has no G/I/D/R or snapshot options. A small SF-43 config generator is needed.
2. A Par2 head solve that does not converge only prints a warning and still exits 0 ([PE]:416-420). Verdicts must be parsed from `Converged=YES|NO iters=…` (regex already in [S42] pilots/run.py:125-138).
3. Par2 has no per-realization restart: all `NR` realizations run in one process ([PE]:352-948), so the unit of work is the whole invocation. Every invocation also writes about 100 dense r=0 snapshots ([PE]:328, `IOScheduler.hpp`:134-137, `output_layout.hpp`:69-71). That is roughly 60 MB of CSV per invocation (10k particles, precision 15), and `prune` does not delete it. Use `NR` batches rather than NR=1.
4. The Par2 time-window secondary in sf42red needs regular snapshots (`analysis.snapshots.enabled`; reduce.py:15-17, 553-575). The pilots never enabled them, and they cost extra disk.
5. A consumer exit 4 stops the invocation at the failing realization ([ER]:1729-1731), and seed0 is part of the DONE signature ([ER]:1441). The remaining seeds need a new segment directory, as SF-39 did with `s4_w12c`.

## 3. `run_sf39.sh`: what to reuse and what is missing

**Structure.**
- Contract and resume rules RR0: :1-49. The final exit codes " 0 2 3 4 7 " are never re-run (:90); anything else re-runs with identical settings (:42-47).
- Helpers `res`, `mk`, `tsv_rc`, `tsv_col`, `safe_rm`: :99-126. Signal handling (`kill_tree`, `on_signal`, one global `CHILD`): :128-143. `run_in` with `/usr/bin/time -v`: :145-152.
- `case_run`: :161-222. It writes `exit_codes.tsv` (last record wins), `walls.tsv`, `resumed.txt` and `logs/<id>.log` + `.time`, and greps result lines into the job log cut at **200 characters** (:220).
- md5 gate `check_binaries`: :225-244. `src_rev` without `git -C`: :249-260. `write_env`: :262-289. `part_begin`/`part_finish`/`sha_part`: :291-367 (`.build_digest` refuses a stale part directory, `PART_DONE` marker). Note that `results.txt` is truncated on every invocation (:309).
- `run_arm`: :383-419 (exit 3/4 adds the arm to `halted_arms.txt`; incomplete counts). `par2_run` re-runs from scratch into a cleaned directory (:466-495). `solver_verdict`: :503-510. The dry-run pattern: :59, :99-106, :174-178, :296-301.
- There is **no reduction or pruning** in it (only the `sf39.reduce smoke` checks at :605-615).

**Reusable for `run_sf43.sh`:** the helpers, the signal pattern, `case_run`, the md5 gate with `.build_digest`, `write_env`, the part bookkeeping, `sha_part`, the dry-run pattern and the bash 4.2 conventions (:49). The fake-binary functional tests in `ROOT/docs/experiments/artifacts/2026-10-27-sf39-macrodispersion-study/scripts/sf39/tests/test_runner_functional.py` (and `fixtures_runner.py`) are a ready pattern for the new tests.

**Missing, all new:**
- A priority, seed-interleaved queue.
- Two lanes sharing it. The host has bash 4.2 (no `wait -n`), so use `flock` around claims and per-lane subshells. A single global `CHILD` handles only one child.
- The wall-clock cutoff, budget guard with STOPPED marker, and disk guard.
- OOM detection and requeue. Today exit 1 is reported as `UNEXPECTED_EXIT_CODE` and only re-run on the next invocation (:414-417).
- Reserve-seed replacement and segment continuation.
- Per-arm halt rules beyond RR6.
- Periodic reduction and pruning, the in-job blinded re-estimation, the in-job final analysis, and DONE/STOPPED markers.
- Resume state: campaign start time, claims and OOM counters.
- A per-case log header for reconciliation. Two lanes writing the same job log interleave lines, which breaks reconcile's header-based mapping.

**How a CUDA OOM shows up.** Always exit code 1, which is shared with every other exception, so classify by matching the log text:
- Consumer: `lagrangian_ensemble: ERROR: CUDA error at <file>:<line> - out of memory` (`ROOT/src/runtime/cuda_check.cuh`:12-17, main :36-38); cuFFT `- code 2` (`PeriodicGaussianField.cu`:31-37); cuBLAS at context creation `- code 1|3` (`CudaContext.cu`:9-14).
- Par2: `ERROR: … out of memory`, or `Failed to allocate RNG states|corner velocity buffers|drift correction buffers|status array|wrap counter arrays` ([P2] src/transport_engine.cu:89,207,250,451,472), or `std::bad_alloc` ([P2] src/io.cu:50-85).
- The kernel's host-RAM OOM killer gives 137.
- Disk full surfaces as `write failed` / `cannot write` with exit 1 ([LT]:430-463). It must go to the disk guard, not to a retry.

**Solver verdicts.**
- Consumer verdicts: converged, converged_balance_above_1e-8, fallback (rel ≤ 1e-8 and balance ≤ 1e-8, the C0 rule), not_converged_beyond_fallback, non_finite ([DLB]:225-239).
- The `B_SOLVER` line is printed ([ER]:640, [DLB]:247-252). An unusable solve prints `STAGE_FAILED` and exits **4** ([ER]:651-657), and leaves no files (`fail()` removes the temp directory: :503-507). The failing index equals the number of DONE realizations.
- Fallback-accepted realizations complete normally with `"verdict": "fallback"` in `solver_b.json` ([ER]:965-985), which is what the no-fallback sensitivity needs.
- Other consumer codes: 3 = seed on a cell face, 7 = option mismatch, 2 = usage ([ERh]:72-80).

## 4. Analysis tooling (SF-42 scripts)

**Already provided**
- `mcstat`: API list in `__init__.py`:35-60. It covers:
  - the robust limit, its p-value and intersection-union (`budget.py`:328-414)
  - `E_h` (`LadderBound` :190-320) and `E_int`/`E_W`/interaction with seed-pairing groups (:115-188)
  - Gate 5 TOST and power (`gate5.py`:90-206), the split-plot factorial (`splitplot.py`:130-177) and Holm (`multiplicity.py`:12)
  - sizing and joint allocation, the blinded re-estimation that returns SDs only (`blinded.py`:44-91), the cost model and the coverage simulator
- `sf42red reduce`: per-realization plane (TP/TAP/LP) and time-family rows, flux/unif/expI weights, pos/disp bases, split halves. The `.planes.csv` output carries **every** plane's covariances (reduce.py:396-406), so sub-windows such as the stationarity halves and [94,194] can be recomputed after pruning with `window_planes`/`estimators` (:230-264).
- `seedlattice` gives admissible M (128 at σ² 0.25 and 1, none common at 2.25, 100 at 4).

**Must be written**
- An arm/seed loader that understands segments, replacements and paired groups.
- The Gate 4 assembly per estimator and amplitude, including the `numerical` class's bias-corrected limit.
- The Gate 5 and G7 assembly.
- The free decomposition: Pollock-twin estimates are not reduced by sf42red. After pruning only `pollock_*moments_planes.csv` remains, which has no cov23 ([ER]:236-245), so only TP and LP are possible from it.
- B&dD eq. 6 fit (not found anywhere), stationarity and exp(−I) readings, failure rates with worst-case imputation, the no-fallback sensitivity.
- A per-realization solver-verdict table (G8 requires one; the sf42red COLUMNS have no verdicts, reduce.py:101-105).
- `summary.json`, tables, a PARTIAL label after cutoff, and a manifest.
- Figures: there is no plotting code, and the host stack is "numpy and scipy only" per the SF-42 README, so generate them off-host.

**`recompute`** is SF-39-specific: SF-39 output root, artifact and committed column names (`recompute/sf39.py`:1-44, 243, 283, 360, 474-483). A new `recompute/sf43.py` is needed.
- The consumer `moments_planes.csv` files (flux and twin) support TP and LP only.
- TAP and the displacement basis need arrays.
- Par2 crossing CSVs support everything.
- Still to verify: whether the tracker's per-plane "contributing" set equals sf42red's per-plane set.

**`reconcile`.** The `B_SOLVER` rule is generic. Mapping lines to (arm, seed) depends on SF-39's log grammar (`solver.py`:46-50; the arm token allows only `[A-Za-z0-9_]`) and on the `<runs>/<arm>/realizations/r_*/solver_b.json` layout (:179-211). `--check-sf39` is SF-39-only (:39-40). There is no Par2 parsing and no input for an SF-43 failure ledger.

**`sf42red prune` and periodic pruning: partial.**
- **What works:** it can prune any arm directory whose realizations are all DONE (non-DONE refuses, :138-139). It hashes every `.npy` and logs before deleting (:155-183), and deletes only `*.npy` (:185-189).
- **What blocks small-unit pruning:**
  - Each directory can be pruned only **once, when final**. The table must cover every realization in the directory (:136-146), reduce needs the `.npy` files (reduce.py:343-346), and `verify` re-hashes all inputs (:86-92).
  - The G8 minimum is applied **per directory**, fixed at 5 in the CLI (:34, :151-154; `__main__.py`:56; a lower value is "tests only"). Small units would retain min(5, n) each.
- **Fix:** a disclosed `--min-retained` CLI option, with SF-43 enforcing at least 5 per logical arm through the pre-registered retained list. Alternatively use coarse segments of about 100 or more realizations.
- It never touches Par2 CSV snapshots, and it was never run on a real campaign (SF-42 note `ROOT/docs/experiments/2026-10-30-sf42-confirmatory-tooling.md`:1580-1581).

## 5. Disk

**Formula (consumer, per realization):** `bytes ≈ M²·[H·(50 + 32P + 32T) + (42 + 24P + 24T)]`.
- First term: tracker arrays, H = number of h_max passes ([LT]:502-511, f8 except two 1-byte status arrays).
- Second term: Pollock arrays ([ER]:997-1006).
- P = number of planes. T = resolved snapshot times ≈ 2·(L/λ − 10)/qbar; measured T = 180 / 159 / 130 / ~98 at σ² 0.25 / 1 / 2.25 / 4, and 335 in the 208-λ box at σ² 1 (pilot logs).
- Checked against the J3 manifests: the arrays reduce reads come to 131.62 / 120.55 / 64.22 / 54.46 MB per realization, and the formula gives 120.6 MB at σ² 1.

**Per realization:**

| Configuration | Size |
|---|---|
| σ² 0.25 / 1, M=128 | 230.9 / 211.6 MB (Pollock is about 43%) |
| σ² 2.25 / 4, M=100 | 112.9 / 95.0 MB |
| h_max arm at σ² 1 (H=3) | 453 MB |
| Distance arm, σ² 1, P=170, T=335 | 465 MB |

- Par2 per realization: crossings about 20–30 KB plus stats about 10 KB, plus about 60 MB of r0 snapshots per invocation.
- Total with the planning N of `cost_projection_rows.csv` (M_plan): about **1.77 TB** of raw per-particle output. The last recorded host free disk was 601 GB (`SF-39-macrodispersion-study.md`:324). Pruning during the run is mandatory.

## 6. Effort (worker-hours) and risk

| Component | Hours | Risk |
|---|---|---|
| `run_sf43.sh` plus a Python core (queue, ledger, config generator), fake-binary tests, a small host smoke run | 45–60 | High: complex failure semantics on bash 4.2; one ~6-day shot |
| `prune` `--min-retained` change plus tests | 1–2 | Low |
| Analysis package plus `recompute/sf43`, reconcile extension, end-to-end synthetic test | 70–95 | Medium-high: frozen before data; the coverage re-run at final n is about 42 slot-hours |
| Two-GPU helper change, tests and docs | 8–12 | Low technically; needs an owner decision |
| Dry-run mode (built in from the start) | 4–6 | Low; does not exercise failures or OOM |

**Open design rules the pre-registration must fix:**
- Replacing a failed seed while keeping pairing across ladder, h_max and Gate 5 arms.
- Whether the final analysis runs after STOPPED.
- During the blinded phase, the job may not print outcome numbers; the freeze row and `blinded stop` must come before anyone reads `summary.json`.

### Critical Files for Implementation
- /tmp/claude-1000/-home-sesquerre-Projects-MacroFlow3D/0b813f1d-3625-4dbd-9e7d-1066997d5f71/scratchpad/master-check/scripts/remote
- /tmp/claude-1000/-home-sesquerre-Projects-MacroFlow3D/0b813f1d-3625-4dbd-9e7d-1066997d5f71/scratchpad/master-check/docs/experiments/artifacts/2026-10-27-sf39-macrodispersion-study/scripts/run_sf39.sh
- /tmp/claude-1000/-home-sesquerre-Projects-MacroFlow3D/0b813f1d-3625-4dbd-9e7d-1066997d5f71/scratchpad/master-check/docs/experiments/artifacts/2026-10-30-sf42-confirmatory-tooling/scripts/sf42red/prune.py
- /tmp/claude-1000/-home-sesquerre-Projects-MacroFlow3D/0b813f1d-3625-4dbd-9e7d-1066997d5f71/scratchpad/master-check/docs/experiments/artifacts/2026-10-30-sf42-confirmatory-tooling/scripts/pilots/design.py
- /tmp/claude-1000/-home-sesquerre-Projects-MacroFlow3D/0b813f1d-3625-4dbd-9e7d-1066997d5f71/scratchpad/master-check/docs/experiments/artifacts/2026-10-30-sf42-confirmatory-tooling/scripts/mcstat/budget.py

Not checked: the B&dD eq. 6 formula, host matplotlib, and whether the tracker's per-plane sets match sf42red's (this decides whether recompute can reach 1e-10). The largest risk is governance: the two-GPU stage cannot be used until the owner supersedes decision 2026-10-27 items 1 and 3.