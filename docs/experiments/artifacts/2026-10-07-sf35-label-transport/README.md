# SF-35 artifact: label transport by backward streamline tracing

Evidence of increment SF-35 (the tracing constructor of the inlet-anchored Darcy labels on the `x1`-slab; decision
`docs/decisions/2026-10-07-label-transport-constructor.md`). This README says what is where, how to regenerate it,
which verdict rules the digests implement, and holds the job table that node N7 fills. It is not a result summary:
results live in `analysis/` and, once written, in the experiment note
`docs/experiments/2026-10-07-sf35-label-transport.md`.

- Code (not in this directory): `src/physics/streamfunctions/inlet_slab/` (`SlabLabelTransport`, `SlabMetrics`),
  driver `apps/inlet_slab/` (`inlet_slab --construct trace`), fast tests `tests/inlet_slab/`
  (`ctest -R inlet_slab`).
- Increment spec: `docs/plans/active/lester-eq14/increments/SF-35-label-transport-constructor.md`.
- Pattern: the SF-33 artifact `docs/experiments/artifacts/2026-10-06-sf33-gpu-inlet-labels/`.

## Layout

| path | content | committed |
|---|---|---|
| `scripts/` | campaign runner `run_campaign.sh` and digests; [`scripts/README.md`](scripts/README.md) documents every script | yes |
| `logs/<part>/` | driver logs `<name>.log`, `/usr/bin/time -v` files `<name>.time`, `run.env` (commit, host, date, GPU), `exit_codes.tsv`; remote job logs in `logs/jobs/<job>.log` | yes (text) |
| `raw/<part>/` | driver `--summary` JSON per run; `raw/matrix_eps<E>/planes_eps<E>.csv` (every plane of every point) | yes (text) |
| `analysis/` | digests written by the scripts (`controls.md`, `xcons.md`, `regress.md`, `matrix_eps05|10|15.md`, `info_sigma_eps.md`, `gpu.md`, `cost_256.md`) | yes (text) |
| `exports/` | saved labels (`psi1.npy`, `psi2.npy`, `labels.json`), control inputs | **no** (gitignored) |
| `COMMIT` | commit hash of the synced tree, written by N7 before `scripts/remote sync` (the mirror has no `.git`); absent -> `unknown` in `run.env` | N7 decides |

Local smoke of the scripts (not evidence): `run_campaign.sh build/wsl-debug smoke` writes `logs/smoke/`,
`raw/smoke/`, `analysis/smoke_*` and `exports/smoke/`; all of them are gitignored.

## Regeneration

Run from the repository root; `A=docs/experiments/artifacts/2026-10-07-sf35-label-transport`. Each line is one detached
V100 job (`scripts/remote --increment SF-35 run <job> -- "<command>"`, then `scripts/remote --increment SF-35 wait <job>`);
the digests run at the end of each part, so `analysis/*.md` is a by-product of the job.

```bash
bash $A/scripts/run_campaign.sh build/v100-release controls           # A1-A4, orders A2 at 16/32/64
bash $A/scripts/run_campaign.sh build/v100-release xcons              # D
bash $A/scripts/run_campaign.sh build/v100-release regress            # G (32^3 production vs SF-33 log)
bash $A/scripts/run_campaign.sh build/v100-release matrix 05          # B, C, E, H at eps 0.5
bash $A/scripts/run_campaign.sh build/v100-release matrix 10          # eps 1.0
bash $A/scripts/run_campaign.sh build/v100-release matrix 15          # eps 1.5
bash $A/scripts/run_campaign.sh build/v100-release info               # E: sigma2 2.25 eps 1 vs sigma2 1 eps 1.5
bash $A/scripts/run_campaign.sh build/v100-release gpu                # F, only if the GPU tracer exists
```

`THREADS` (default 32) is the host thread count of the tracer; the labels are bitwise independent of it. No part
changes any setting after a failure: exit codes 0 / 21 (`roundtrip_exceeded`) / 19 (`trace_fail`) /
17 (`inlet_backflow`) / 18 (`darcy_failed`) are recorded in the logs and `exit_codes.tsv`, never retried.

## Orchestrator reading rules (activation bitacora row of 2026-10-15T17:05Z, quoted)

1. criterion D's 1e-6 relative is evaluated against the FULL-PRECISION SF-33 values in
   `raw/ladder_0.25_c4/N{32,64}.json` and `raw/prod32_05/N32.json` of the SF-33 artifact (`metrics.i1o4.e_psi*`), not
   the printed 4 digits;
2. `--construct trace` exit codes 0 ok / 17 `inlet_backflow` / 18 `darcy_failed` / 19 `trace_fail` as specified, and
   `roundtrip_exceeded` gets the distinct new code 21 (20 is `not_run`; the `--production` meaning of 19 is unchanged);
3. per-plane metrics: `e_v_j = RMS_j|c - vD| / v_rms` (global `v_rms`), `e_i_j = RMS_j(vD . g_i)/(v_rms RMS_j|g_i|)`,
   `min_c_j`, `vD1_min_j`, `vD1_nonpos_j` over the N^2 vertices of plane j = 0..N with the global stencils; GROWTH uses
   `e_i_j = sqrt((e_i1_j^2 + e_i2_j^2)/2)`;
4. G's byte-compare trio = `config_pspta_small`, `config_streamfunctions_homogeneous`,
   `config_streamfunctions_continuation` vs base references built from `f8bec23` on the SF-35 mirror (job
   `sf35-base-refs`), as in SF-27/SF-33 (not part of this artifact's scripts: orchestrator job
   `sf35-ctest-full-bytecmp`);
5. G's 32^3 regression masks only the timing fields (`t=`, `t_trace=`, `t_labels=`) (`regress_diff.py` also masks
   `t_lu=`, `t_apply=`, `t_fact=`, `t_lin=` if present on those lines; none is expected on the compared lines);
6. the constructor entry point is a template on the direction-field functor (as `compute_oracle`) with a
   `ProductionStage` overload, so the A4 synthetic field and the GPU backend share it;
7. A2's order `>= 3.5` is evaluated on both pairs 16->32 and 32->64; a 16->32 miss with a 32->64 pass is reported as
   pre-asymptotic (SF-33 N0 precedent on the outlet one-sided stencils), disclosed, not tuned.

Ladder settings fixed before the runs: primary `(h/8, 1e-8)`; ladder `(h/16, 1e-8)`, `(h/16, 1e-10)`,
`(h/32, 1e-10)`; `max_roundtrip = 1e-8`; `min_step 1e-13`, `max_arclength 100` (driver defaults; the scripts pass no
integration option).

## Verdict rules (spec "Acceptance thresholds", verbatim)

Pre-registered verbatim in the activation bitacora row before any run. The digests compute the verdicts by code:
`controls_digest.py` (A), `orders_digest.py` (B, C; the literal reading of C is printed at the top of each
`matrix_eps*.md`), `xcons_digest.py` (D), `regress_diff.py` (G, production lines), `labels_diff.py` (E, informative
comparison), `gpu_equiv.py` (F), `cost_digest.py` (H). The G items about `git diff` on the frozen stack, the SF-27 trio
byte-compare and the full V100 `ctest` are orchestrator jobs and are not computed here.


Pre-registered verbatim in the activation bitácora row before any run; the ladder settings `(h/8, 1e-8)` primary,
`(h/16, 1e-8)`, `(h/16, 1e-10)`, `(h/32, 1e-10)` are fixed before the runs. No threshold is tuned afterwards.

- **A. Positive controls** (fast tests at 16/32; V100 adds 64). A1 `k = 1`: feet `= (x2, x3)` and labels affine to
  `<= 1e-13` on every plane. A2 `k = exp(0.5 cos 2 pi x2)`: `|psi_i(j, m2, m3) - psi_i(0, m2, m3)| <= 1e-10` on every
  plane; `e_v` and `r_F` of these labels decrease 16 -> 32 (-> 64) with observed order `>= 3.5`. A3 `control2d`
  `eps = 0.5`: `|psi2 - x3| <= 1e-10` on every plane. A4 synthetic failure field: status histogram,
  `landings_discarded`, NaN labels and `trace_fail` exactly as constructed.
- **B. Integration certificate.** Round trip `<= 1e-8` on every plane at `(h/8, 1e-8)` for every matrix point;
  `non_ok = 0`; at 32 and 64 the ladder gives max `|dpsi|` between the primary and `(h/32, 1e-10)` `<= 1e-8`.
- **C. Convergence per amplitude** (32/64/128/256). `e_v`, `e_i1`, `e_i2`, `e_div`, `r_F` each decrease on every
  refinement and show observed order `>= 1.8` on 128 -> 256: `accept`; any of them changing `< 10 %` between 128 and
  256 while above 1e-12: `reject` (floor); otherwise `unresolved`. Rationale for 1.8 on `r_F`: the traced labels are
  the exact labels of the spline flow, whose consistency with the spectral `k` is O(h^2); a 4th-order threshold would
  presuppose an exact Darcy flow. Blocking: `eps = 0.5` and `1.0` must be `accept`; `eps = 1.5` must be `accept`
  unless the stage reports `inlet_backflow` (then recorded; owner decision). `r_out`, `min_c`, percentiles recorded.
- **D. Cross-construction.** The `--solve` runs converge (`r_F <= 1e-10`) and `e_psi / e_psi1 / e_psi2` (elliptic vs
  traced, C-i4 normalization) reproduce SF-33 within `1e-6` relative: 32^3 `eps = 0.25` `1.250e-2 / 1.250e-2 /
  1.140e-2`; 64^3 `eps = 0.25` `1.555e-3 / 1.483e-3 / 1.555e-3`; 32^3 `eps = 0.5` `3.837e-2 / 3.837e-2 / 2.756e-2`
  (`analysis/ladder_orders_025_c4.md`, `analysis/prod32_05.md`, `logs/ladder_0.25_c4/N{32,64}.log` of the SF-33
  artifact); `a_psi1`, `a_psi2` observed order 32 -> 64 `>= 2.5` (SF-33: 3.07 / 2.87 — the two constructions agree
  faster than either agrees with `vD`).
- **E. Recorded, no threshold.** Per-plane `e_v_j`, `e_i_j`, `roundtrip_j`, `nfev_j` and the `GROWTH` line for every
  point; the `BACKFLOW` line for every point, tabulated vs `eps` and `N`; the `--sigma2 2.25 --eps 1` vs `--sigma2 1
  --eps 1.5` label difference at 32^3 (expected `<= 1e-12`).
- **F. GPU tracer (non-blocking).** With `--fmad=false`: max `|dfoot|`, `|dpsi|` host vs device `<= 1e-12` at 16^3
  and 32^3 on {`k = 1`, `generic3d` 0.25, `gauss` 0.5}; per-plane `nfev` and status histograms identical (otherwise the
  number of streamlines with a different step sequence is reported; the 1e-12 bound remains the criterion); two
  launches bitwise identical; no device allocation in `trace` after `prepare`; on V100 at 128^3 (and 256^3)
  `eps = 0.5`: `<= 1e-12` vs host, wall time recorded; the `--fmad=true` difference reported as INFO.
- **G. Regression.** CASE/ORACLE/ORACLE_SUMMARY/EXTRA lines of the `--production` 32^3 `eps = 0.25` run identical to
  SF-33's `N32.log` (timings excluded); `git diff <base>..<head>` empty on the frozen stack, `src/numerics`,
  `src/multigrid`, `src/physics/particles`, `apps/closure_gate`; SF-27 trio byte-compare clean; full V100 `ctest`
  green; the 6 SF-33 tests unchanged.
- **H.** Wall time, host RSS and peak device memory at 256^3 recorded per amplitude.
- **Gate 3A mapping.** `r_F`, `r_out` = residuals of the equation and outlet rows evaluated at the constructed labels
  (necessary condition; never an acceptance quantity alone); `e_v`, `e_i`, `e_div`, `|c|` percentiles as in
  `SlabMetrics.cuh`; denominator regularization: none; gauge: none (labels anchored by D-1); "Newton failures" ->
  tracing status histogram and round trips; grid convergence = C; positive controls (A) precede acceptance (C).


## Job table (filled by N7; one row per detached job)

Per-increment mirror `~/MacroFlow3D-SF-35` (`scripts/remote --increment SF-35`); jobs are serialized (one job per GPU,
no overlapping sync). Fill: job name, kind, GPU, start/end UTC, exit, command, outputs. The remote job logs are copied
to `logs/jobs/<job>.log`.

| job | kind | GPU | start (UTC) | end (UTC) | exit | command / outputs |
|---|---|---|---|---|---|---|
| `sf35-build` | build + ctest `-R inlet_slab` | | | | | |
| `sf35-controls` | GPU | | | | | `run_campaign.sh build/v100-release controls`; `logs/controls/`, `raw/controls/`, `analysis/controls.md` |
| `sf35-xcons` | GPU | | | | | `... xcons`; `logs/xcons/`, `raw/xcons/`, `analysis/xcons.md` |
| `sf35-regress-prod32` | GPU | | | | | `... regress`; `logs/regress/`, `analysis/regress.md` |
| `sf35-matrix-eps05` | GPU | | | | | `... matrix 05`; `logs/matrix_eps05/`, `raw/matrix_eps05/`, `analysis/matrix_eps05.md` |
| `sf35-matrix-eps10` | GPU | | | | | `... matrix 10` |
| `sf35-matrix-eps15` | GPU | | | | | `... matrix 15` |
| `sf35-info` | GPU | | | | | `... info`; `analysis/info_sigma_eps.md` |
| `sf35-gpu-equiv` | GPU (only if the GPU tracer is accepted) | | | | | `... gpu`; `analysis/gpu.md` |

## Retrieval (text only; no exports)

```bash
scripts/remote --increment SF-35 exec -- "cd docs/experiments/artifacts/2026-10-07-sf35-label-transport && \
    tar czf /tmp/sf35x.tgz logs raw analysis"
scp v100:/tmp/sf35x.tgz . && tar xzf sf35x.tgz -C docs/experiments/artifacts/2026-10-07-sf35-label-transport
scp v100:~/.macroflow3d-remote/macroflow3d-SF-35/logs/<job>.log \
    docs/experiments/artifacts/2026-10-07-sf35-label-transport/logs/jobs/
```

Caution: `scripts/remote sync` (`rsync --delete`) removes gitignored host-only files (`exports/`) from the mirror;
the matrix exports are only needed by `info`/`gpu` within their own part, so no save/restore is required between parts.

## Post-hoc diagnostics (not pre-registered)

`scripts/label_diagnostics.py` (python3 + numpy only) was written after the campaign verdicts, on the orchestrator's
exploratory checks of the eps 1.0 / 1.5 label exports. It is **not** part of the pre-registered thresholds A-H and
changes no verdict. It reports, per eps: (1) a periodic-wrap check of the periodic parts `U1`, `U2`
(`max |U(next) - U| * N` along `x1`, `x2`, `x3`); (2) pointwise self-convergence of the labels on common vertices
(`psi(N)` vs `psi(2N)[::2,::2,::2]`: max, RMS, per-plane max, observed orders); (3) the concentration of the
non-convergence of the cross product `c = grad psi1 x grad psi2` (2nd-order central differences, interior planes,
`c(N)` vs `c(2N)` on common vertices, normalized by `RMS|c(2N)|`).

Inputs: label exports kept outside the mirror at `~/sf35_exports_keep/matrix_eps{10,15}/N{32,64,128,256}/{psi1,psi2}.npy`
on the V100 host (the eps 0.5 exports were lost by an `rsync --delete` sync; their JSON/logs are retained here). Job
`sf35c-labeldiag`:

```bash
python3 docs/experiments/artifacts/2026-10-07-sf35-label-transport/scripts/label_diagnostics.py ~/sf35_exports_keep \
    --out docs/experiments/artifacts/2026-10-07-sf35-label-transport/analysis/label_diagnostics.md
```

Output: `analysis/label_diagnostics.md`; job log `logs/jobs/sf35c-labeldiag.log`. Local check of the script (not
committed, not evidence): synthetic smooth labels with an added `h^2` term at `N = 16/32/64` give no wrap jumps and
observed order 2.00.
