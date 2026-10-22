# SF-40 campaign tooling (V100 jobs)

Artifact of increment SF-40 (`docs/plans/active/lester-eq14/increments/SF-40-lean-lagrangian-stage.md`). The normative
reading rules are `preregistration.md` (RR0-RR12, C-LEAN-PLACEMENT, C-SYNC, C-MEMPHASE, C-TRACKER-L6); this directory
holds only the scripts that execute them. The scripts never run locally on a GPU: the orchestrator launches them as
detached `scripts/remote --increment SF-40 run <job> -- "..."` jobs, one at a time. **Output of any `SF40_DRYRUN=1` run
is NOT evidence** (tiny settings, separate roots, labelled in `run.env` and `results.txt`).

## Layout

| File | Role |
|---|---|
| `scripts/run_sf40.sh` | the campaign runner: `bash scripts/run_sf40.sh <bin_dir> <part>` |
| `scripts/closed_form.py` | criterion C (RR3, C-MEMPHASE): identity, item size rules, S+ closed form of S2, S2 largest; `--selftest` |
| `scripts/pid_peak.py` | criterion D (RR5): per-PID peak from the raw nvidia-smi files + the three gates; `--selftest` |
| `scripts/mem_verdict.py` | `mem` part verdicts: `C_*`, `D_VERDICT`, `D3328_VERDICT`, `E_VERDICT`, `RECORDED`; `--selftest` |
| `scripts/setup_lines_cmp.py` | LEAN SETUP comparison selected by literal line prefix (imports the mask of `lines_cmp.py`); `--selftest` |
| `scripts/l6_timings.py` | `l6_timings.tsv` and `RECORDED L6_SPEEDUP` lines; `--selftest` |
| `scripts/make_zeros_npy.py` | stdlib writer of the zeros `.npy` of the driver smoke; `--selftest` |
| `scripts/json_strip_cmp.py`, `scripts/lines_cmp.py` | SF-37 tools, copied byte-identical (below) |

Provenance of the copied scripts (source `docs/experiments/artifacts/2026-10-19-sf37-lagrangian-cube-verification/scripts/`):

```
3f55dc1f92753170356992006ece7fe7acbaf007ac73bd06b2d131ea806fe654  json_strip_cmp.py
4f82a56e47d208fb1345952a9516acdf1779cfae1c2327fb417d1b6ad812a708  lines_cmp.py
```

Everything is stdlib python3 (no numpy) and bash 4.2-safe (the V100 host).

## Runner conventions

`<bin_dir>` holds `inlet_slab` and `macroflow3d_pipeline` (part `baserefs`: the BASE build; other parts:
`build/v100-release` of the audited head). The test executables and `ctest` of `controls` / `final` come from
`$SF40_BUILD_DIR` (default `<bin_dir>`). The mirror (repository) root is derived from the script path.

Environment: `SF40_OUT_ROOT` (default `~/sf40_campaign`; dry `~/sf40_dryrun`), `SF40_BASE_REFS` (default
`~/sf40_base_refs`; dry `$SF40_OUT_ROOT/base_refs`), `SF40_COMMIT`, `SF40_BUILD_DIR`, `SF40_DRYRUN=1`, `PYTHON`.

Per part, in `$SF40_OUT_ROOT/<part>/` (for `baserefs`, in `$SF40_BASE_REFS`): `exit_codes.tsv` (`run<TAB>rc`),
`walls.tsv` (`run<TAB>wall_s<TAB>end_utc`), `run.env` (COMMIT, binaries and md5, host, `CUDA_VISIBLE_DEVICES` and GPU name,
NVIDIA driver, CUDA version from the `nvidia-smi` banner, nvcc), `sha256.txt`, `resumed.txt`, `results.txt`. A run `R`
writes `R/` (particles/labels), `R.json` (summary, outside the particles directory), `R.log`, `R.time` (`/usr/bin/time -v`),
`R.free` (`free -b` before), and, when sampled, `R.apps.csv`, `R.dev.csv`, `R.pid_tree`.

Resume (SF-37 RR12): a recorded rc = finished, never re-run; partial outputs without rc are removed and the run is re-run
with identical settings (listed in `resumed.txt`); nothing is re-run because of its result. Comparisons and verdicts are
recomputed on every invocation from the finished runs. The part's exit status is non-zero iff a blocking item failed.
`results.txt` lines `IDENTICAL|DIFF|MISSING|SKIP <item>` are counted; `RECORDED ...` and verdict lines are reported.

## Parts

### `baserefs` (BASE binaries; job J0 `sf40-base-refs`)
* RR4 A set at the default stage into `$SF40_BASE_REFS/a/<case>/` (+ `.json .log .time`): 27 cases
  `N{64,128,256} x eps{1,1.5,2} x seed{3101,3102,3103}` (`N<n>_e<10|15|20>_s<seed>`), `N512_e10_s3102`, `N512_e20_s3102`,
  `cov` (N 64, eps 1, seed 3101, `--covectors`), `pol` (N 64, eps 2, seed 3102, `--comparator pollock`); common arguments
  `--track forward --sigma2 1 --ell 0.25 --seed S --n N --eps E --np-side 256 --planes 0.25,0.5,1,2,4,8,16
  --times 0.25,0.5,1,2,4,8,16 --tol 1e-8 --hmax-div 8 --max-arclength 1000 --save-particles <run>/ --summary <run>.json`.
* RR8 default-path set into `$SF40_BASE_REFS/r/{trace,fwd,neutral}/` and `r/prod32.log`: the SF-37 RR13 commands
  (`--construct trace --threads 32` 32/64 x eps 0.5/1.0 x host/gpu and 64 eps 2 gpu; `--production --n 32 --eps 0.25
  --oracle-ladder`; the seven `--track forward` cases of seed 3001; the three opt-in cases `gauss_n64_e1_cov`,
  `gauss_n64_e2`, `gauss_n64_e2_tol` with `--comparator pollock --save-vd <dir>/vd --save-seed-velocity`).
* SF-27 trio into `$SF40_BASE_REFS/trio/<cfg>/` (each config from its own empty working directory).
* Par2 config twice, `par2_a` and `par2_b` (empty working directories, `/usr/bin/time -v`, `timeout 4 h`, memory sampled),
  and the determinism comparison par2_b vs par2_a (RR7 rule) in `results.txt` (`BASEREFS_PAR2_DETERMINISTIC` or
  `BASEREFS_PAR2_NONDETERMINISTIC`, the latter blocking). The base binaries are copied to `$SF40_BASE_REFS/bin/`.

### `bytecmp` (HEAD)
Re-runs, with the head binary, set LEAN (the 31 cases + `--stage lean --fused-splines --single-launch`), set L6 (the 31
cases + `--fused-splines --single-launch`) and the RR8 set, and compares with `$SF40_BASE_REFS`. Per case, every item
must be `IDENTICAL`:
* every reference file of the particles directory exists in the head directory and is `cmp`-identical, except `*.json`
  (`json_strip_cmp.py`, only `**.timings_s` dropped; `labels.json` of the trace cases also `transport.seconds_*`, the
  SF-37 timing keys), and the head directory has no extra file (`:fileset`);
* lines (`lines_cmp.py`, timing masks only): `FIELD TRACK BACKFLOW TPLANE TTIME TSTATUS STATUS`; `pol` also
  `PTRACK PPLANE PTIME PSTATUS PFAIL`; LEAN SETUP lines only `SETUP   SF-18:`, `SETUP   SF-19:`, `SETUP   SF-19 corrector`
  (three spaces after SETUP: `out_block("SETUP ", summary())` plus the two-space indent of
  `ProductionStageReport::summary()`), selected identically in both logs, order preserved (`setup_lines_cmp.py`); L6 and
  R: every `SETUP` line with the `built in <t>s` mask;
* R opt-in cases additionally `PTRACK PPLANE PTIME PSTATUS PFAIL SEEDV` and every `pollock_*`, `seed_v.npy`, `vd/*` file;
  R trace cases: `CASE PLANE BACKFLOW GROWTH RESIDUAL TRANSPORT_STATUS STATUS`; `prod32`: `CASE ORACLE ORACLE_SUMMARY EXTRA`;
* the exit code equals the base `Exit status`.
The part ends with `BYTECMP_ALL_IDENTICAL` or `BYTECMP_DIFF n=<k>`; a `binary:md5_differs` line proves head != base build.
Not scripted (orchestrator, local): `git diff --numstat 9485beb..<head>` of RR8 and the `cuobjdump -sass` check of C-TRACKER-L6.

### `l6` (HEAD, current stage)
N 256, eps 2, seed 3102, RR4 setting, four variants `default`, `fused` (`--fused-splines`), `single` (`--single-launch`),
`both`. `l6_timings.tsv` has per variant rc, wall, `timings_s` (plane/time/single kernel, download, moments), nfev from the
`TSTATUS` lines, total kernel seconds and the speedup versus `default`; `RECORDED L6_SPEEDUP` lines repeat it. Each
non-default variant is byte-compared with the default run (all files, `FIELD`, `SETUP`, `TRACK`... lines, exit code).

### `controls` (HEAD)
`long_box_tests --full`, `projected_pcg_external_tests --full`, `affine_periodic_flow_lean_tests --full`,
`long_box_tests --cufft-size 3328,384,384` (`CONTROL_VERDICT`: exit 0, `ALL_PASS`, no `CHECK ... FAIL`); driver smoke
`--track forward --stage lean --grid 128,16,16 --cells <zeros> --eps 1 --np-side 8 --planes 0.25,0.5,1 --times 0.5,1
--tol 1e-8 --hmax-div 8 --max-arclength 1000` (MEM_GEOM, five MEM_PHASE lines and `closed_form.py` PASS); sampler smoke
(`--context-probe 5` and a 256x32x32 lean run, both with the RR5 sampler; `SAMPLER_SMOKE ... status=OK|NOT_EVALUABLE`);
early-warning lean run `--grid 1664,192,192 --sigma2 1 --eps 2 --ell 0.08333333333333333 --seed 4001 --np-side 10
--planes 0.5 --max-arclength 1000` (with `--stage lean --fused-splines --single-launch`), whose SF-19 corrector lines,
MEM lines and `C_*` lines are `RECORDED` (RR9).

### `mem` (HEAD, RR5)
Runs `probe1..probe3` (`--context-probe 10`), `probe_warm` (`--context-probe 10 --warm`; if rejected: "warm probe
unavailable"), `d512` (LEAN case N 512, eps 2, seed 3102), `d512_bhatdev` (same + `--bhat device`), `d3328_i`
(`--eps 1.5 --ell 16h`) and `d3328_ii` (`--eps 2 --ell 32h`) with the literal RR5 arguments (the script refuses `--bhat`
at 3328). Every run: `free -b` before, no other compute process on the GPU (`.precheck`; otherwise `GPU_BUSY`, not run, no
rc), sampler `nvidia-smi --query-compute-apps=timestamp,pid,used_memory --format=csv,noheader,nounits -lms 200` +
`--query-gpu=timestamp,memory.used -i $GPU -lms 200`, `/usr/bin/time -v`. `mem_verdict.py` then writes the lines below.

### `par2` (HEAD)
`macroflow3d_pipeline apps/config_pipeline_par2.yaml` once from an empty working directory (timeout 4 h, sampled), compared
with `$SF40_BASE_REFS/par2_a` by the RR7 rule: every file `cmp`-identical, same file set, exit code, except
`manifest.json`, compared by `json_strip_cmp.py` dropping the explicit list below. PID/device peaks of base and head are
`RECORDED PAR2_PEAK`.

### `final` (HEAD build dir)
`ctest --test-dir <build> --output-on-failure` (`CTEST_EXIT`, `CTEST_WALL_S`, compare with SF-37's 2901.55 s), the SF-27
trio from empty working directories versus `$SF40_BASE_REFS/trio/<cfg>` (RR7 rule; `TRIO_ALL_IDENTICAL` or `TRIO_DIFF`),
md5 of the binaries (`RECORDED binaries md5`, `run.env`).

## manifest.json key list (RR7)

`ManifestWriter::write` (`src/io/writers/ManifestWriter.hpp`) writes, besides the config echo, the run time
(`timestamp`, `std::time(nullptr)`) and the build identity (`build.git_hash`, `git rev-parse --short HEAD` at configure time;
different for a base and a head build by construction). The dropped list is exactly `timestamp` and `build.git_hash`
(`MANIFEST_DROPS` in `run_sf40.sh`); `git_hash` is not a time/date field strictly, it is listed because RR7 cannot hold
without it, and the base and head values are printed (`RECORDED manifest_git_hash`). `gpu.*`, `build.*` others, grid,
stochastic, flow and transport keys are compared.

## Job plan (RR11, one job at a time on the shared mirror)

1. J0 `sf40-base-refs`: from a clean detached worktree whose tree differs from `9485beb` only under `docs/`: build, then
   `SF40_DRYRUN=1 bash .../run_sf40.sh <base bin_dir> baserefs` (into `~/sf40_dryrun`, NOT evidence), then
   `bash .../run_sf40.sh <base bin_dir> baserefs` (real, `~/sf40_base_refs`).
2. At H1 (integrated head): `sf40-build` (detached), `sf40-dry` (every part with `SF40_DRYRUN=1`, NOT evidence),
   then `sf40-controls`, `sf40-bytecmp`, `sf40-l6`, `sf40-mem`, `sf40-par2`, `sf40-final`, each
   `bash .../run_sf40.sh build/v100-release <part>` with `SF40_COMMIT=<head>`.

## Reading the results

* Byte items: `IDENTICAL`/`DIFF`/`MISSING <item>`; `BYTECMP_ALL_IDENTICAL`, `PAR2_ALL_IDENTICAL`, `TRIO_ALL_IDENTICAL`,
  `L6_VARIANTS_ALL_IDENTICAL` mean criterion A / G / R byte parts hold; any `DIFF`/`MISSING` means not met (RR0).
* `C_VERDICT run= phase= expected= reported= diff= PASS|FAIL|NOT_EVALUABLE` per phase (diff must be 0), `C_ITEM ... FAIL`
  lists failing items (unknown item names included), `C_LARGEST` S2 largest, `C_RUN` overall; `CS_*` is the same check for
  the level-S run `d512_bhatdev` (recorded, not criterion C). `C_INFO` lines carry the 2 MiB-rounded sums and
  `device_used - device_used(first phase) - exact` (and `- probe device_used - exact`), the completeness instrument of RR3.
  A run that stopped in phase P gives `NOT_EVALUABLE` for P and later ("consequence of D").
* `D_VERDICT run=d512 pid_peak_mib= E_max_mib= C_mib= limit_mib=` (limit `E_max*1.03 + C`, `C` = min PID peak of the three
  probes) with `D_ALT` (`(E_max + C)*1.03`, not the verdict); `D3328_VERDICT run= exit= pid_peak_mib= limit_mib=30000
  exit_class=ok|memory|other`; `NOT_EVALUABLE` when a sampler gate fails (< 5 samples, PID peak below `max exact_bytes/2^20`
  or above the device-wide peak, PID mismatch with the process tree): re-run with identical settings, disclosed.
* `E_VERDICT run= correctors=<status,its,rel_res x3> converged= levels= coarsest= PASS|FAIL`: PASS iff three `converged`
  correctors with `rel_residual <= 1e-10`, `SF-19 ... converged=1`, `levels=7`, `coarsest=52x6x6`.
* `RECORDED ...`: host RSS, `free -b`, phase seconds, exact bytes per phase, S vs S+ times, driver/runtime versions,
  maximum cells with exit 0, L6 speedups, early-warning corrector lines. Selftests: `python3 scripts/<name>.py --selftest`.

## Notes and limits (reported to the orchestrator)

* The `AffinePeriodicFlowWorkspace::memory_report()` categories mapped onto the S2 line (C-MEMPHASE) have no pre-registered
  line format; `closed_form.py` ignores tokens outside `items=` (not checked).
* `device_used(stage start)` of RR3 is not a printed quantity; `closed_form.py` prints the excess against the first phase
  line (as specified for L5) and against the probe `device_used`.
* The driver-smoke and sampler-smoke argument sets, the `--warm` probe flag and the `--cells` .npy shape `(nz, ny, nx)` are
  assumptions on interfaces owned by other nodes; a rejection is recorded, not worked around.
* Dry mode runs the test executables without `--full` and shrinks `--cufft-size`/early-warning (cannot be verified locally).

## Raw evidence (N8, imported by the orchestrator from the V100 host)

`raw/` is the text subset of the outputs of the SF-40 V100 jobs, extracted unchanged from
`sf40_evidence.tar.gz` (sha256 `b6b1f0a6b9b4b6df07332dec141daaa4ff9eb5c052f3bad5fd3a290f59c879f2`, 1469 files, 17 MB). Layout
(the V100 relative layout):

| Directory | Source on the V100 host | Content |
|---|---|---|
| `raw/sf40_base_refs/` | `~/sf40_base_refs` (job `sf40-j0b-base-refs`) | base references: A set (`a/`), R set (`r/`), SF-27 trio, Par2 base twice, `results.txt`, `run.env` |
| `raw/sf40_campaign/{controls,bytecmp,l6,mem,par2,final}/` | `~/sf40_campaign` (job `sf40-campaign`) | per part: `exit_codes.tsv`, `walls.tsv`, `run.env`, `sha256.txt`, `results.txt`, run logs, `.time`, `.free`, samplers |
| `raw/sf40_campaign/sass/calls_extract.txt` | `~/sf40_campaign/sass` (job `sf40-build`) | CALL targets per function of the `inlet_slab` cubin (C-TRACKER-L6) |
| `raw/job_logs/*.log` | `~/.macroflow3d-remote/macroflow3d-SF-40/logs/` (renamed to `job_logs/` by the pack command) | `sf40-j0a-build-dry`, `sf40-j0b-base-refs`, `sf40-build`, `sf40-dry`, `sf40-campaign` |

Excluded from `raw/` (listed, with their sha256, in the per-part `sha256.txt` manifests): every `*.npy`, `*.cubin`,
`*.cubin.sass`, the base binaries `sf40_base_refs/bin/*`, the Par2 bulk output directories (`*/output/*`), and any other file
larger than 2 MB (the list `sf40_evidence.excluded_large` was empty). The dry-run tree `~/sf40_dryrun` (jobs `sf40-j0a-build-dry`
dry part and `sf40-dry`) is NOT evidence and is not included; only the job logs of those jobs are.

The SF-27 trio output directories (`output_pspta_small/`, `output_streamfunctions_*/`, 332 files under `sf40_base_refs/trio/` and
`sf40_campaign/final/trio/`) match the repository rule `output*/` of `.gitignore` and were committed with `git add -f`, so that `raw/` equals the tarball.

Pack (one short `scripts/remote --increment SF-40 exec -- "<script>"` call on the V100 host, 2026-10-22T22:12Z, after job
`sf40-campaign` finished with exit 0); the script verbatim:

```bash
set -o pipefail; cd ~ || exit 2
S=sf40_campaign/sass
awk '/^\.text\.[^ ]*:$/{f=$1} /CALL/{if (match($0,/`\([^)]*\)/)) t=substr($0,RSTART,RLENGTH); else t="(no-target)"; print f" -> "t}' $S/inlet_slab.2.sm_70.cubin.sass | sort | uniq -c | sort -rn > $S/calls_extract.txt
find sf40_campaign sf40_base_refs -type f ! -name '*.npy' ! -name '*.cubin' ! -name '*.cubin.sass' ! -path 'sf40_base_refs/bin/*' ! -path '*/output/*' -size -2M | LC_ALL=C sort > ~/sf40_evidence.list
find sf40_campaign sf40_base_refs -type f ! -name '*.npy' ! -name '*.cubin' ! -name '*.cubin.sass' ! -path 'sf40_base_refs/bin/*' ! -path '*/output/*' -size +2M -printf '%p %s\n' | LC_ALL=C sort > ~/sf40_evidence.excluded_large
ls .macroflow3d-remote/macroflow3d-SF-40/logs/*.log >> ~/sf40_evidence.list
tar --transform 's,^\.macroflow3d-remote/macroflow3d-SF-40/logs,job_logs,' -czf ~/sf40_evidence.tar.gz -T ~/sf40_evidence.list
echo FILES=$(wc -l < ~/sf40_evidence.list); echo EXCLUDED_LARGE:; cat ~/sf40_evidence.excluded_large; wc -l < $S/calls_extract.txt; ls -l ~/sf40_evidence.tar.gz; sha256sum ~/sf40_evidence.tar.gz
```

Transfer (local, from the clean sync worktree):

```bash
scripts/remote --increment SF-40 exec -- 'echo B64_BEGIN; base64 -w 0 ~/sf40_evidence.tar.gz; echo; echo B64_END' 2>/dev/null > b64.txt
sed -n '/^B64_BEGIN$/,/^B64_END$/p' b64.txt | sed '1d;$d' | tr -d '\n' | base64 -d > sf40_evidence.tar.gz
sha256sum sf40_evidence.tar.gz   # b6b1f0a6b9b4b6df07332dec141daaa4ff9eb5c052f3bad5fd3a290f59c879f2 on both hosts
```

Re-verification:

```bash
# 1. the tarball: sha256 above; extracting it into an empty directory gives a tree with `diff -r` empty against raw/
mkdir /tmp/sf40_chk && tar -xzf sf40_evidence.tar.gz -C /tmp/sf40_chk && diff -r /tmp/sf40_chk raw
# 2. every raw file that is in a manifest matches it (1427 files; the other manifest lines are the excluded files, 42 raw
#    files are bookkeeping outside the manifests: exit_codes.tsv, walls.tsv, run.env, results.txt, l6_timings.tsv,
#    sha256.txt, calls_extract.txt, job logs)
for d in sf40_base_refs sf40_campaign/{controls,bytecmp,l6,mem,par2,final}; do (cd raw/$d && sha256sum -c --ignore-missing sha256.txt | grep -vc ': OK'); done   # 0 each
# 3. derived tables (committed in analysis/, regenerate byte-identically)
python3 scripts/note_tables.py --selftest
python3 scripts/note_tables.py raw analysis
```

`scripts/note_tables.py` reads only `raw/` and writes `analysis/{feasibility.csv,feasibility.md,pid_plateaus.md,completeness.md,summary.md}`
(summary.md: byte-comparison counts, per-phase exact bytes and the S+ closed form, D/E/RSS tables, S vs S+, F controls, L6, Par2, ctest).
The experiment note `docs/experiments/2026-10-22-sf40-lean-lagrangian-stage.md` cites these files.
