# scripts/regress - live byte-compare regression harness

Purpose: let every cleanup / refactor / performance PR prove that its outputs are
byte-identical to a build of the PR's merge-base. It is the reusable form of the
SF-37..SF-41 "criterion R / threshold G" campaign scripts (provenance headers in
each file: `docs/experiments/artifacts/2026-10-25-sf41-medium-b-lagrangian-stage/scripts/`
at commit `3452a7d`; the frozen originals are not modified).

Files:

- `run.sh [--list] <bin_dir> <out_dir>` - runs the core case set with the four
  binaries of `<bin_dir>` (`macroflow3d_pipeline`, `inlet_slab`,
  `lagrangian_ensemble`, `lagrangian_medium_ab`); configs come from the `apps/` of
  the tree containing the script. Refuses an existing `<out_dir>`. `--list` prints
  every command and runs nothing.
- `compare.sh <ref_out> <head_out> [--ref-filter '<glob>=<sed-expr>']...` - case by
  case comparison; last line `REGRESS IDENTICAL` or `REGRESS DIFF n=<k>`; exit 0 iff
  identical.
- `selftest.sh` - no GPU, seconds: python helpers + `compare.sh` on synthetic trees.
- `json_strip_cmp.py`, `lines_cmp.py` - verbatim helper copies.

## Case set (edit the arrays in `run.sh` to change it)

| case id | what |
|---|---|
| `pipe_<cfg>` | `macroflow3d_pipeline apps/<cfg>.yaml` in its own empty cwd, for `PIPE_CASES`: `config_par2_small` (random field + heterogeneous MG-PCG head + velocity + Par2), `config_streamfunctions_homogeneous`, `config_streamfunctions_continuation` |
| `r_*` (20) | the `inlet_slab` R set (`r_items` / `r_args` / `r_expected` of SF-40/41) |
| `g_runner_a` | `lagrangian_ensemble` medium (a), SF-41 G(ii) arguments |
| `g_ab` | `lagrangian_medium_ab`, SF-41 G(iii) arguments |
| `runner_b` | `lagrangian_ensemble --medium b`, 512x64x64, eps 1, 2 realizations, twin and Pollock bridge on |
| `runner_b_analysis` | `macroflow3d_pipeline analysis_config.yaml` (`run_mode: analysis_only`) on a copy of the `runner_b` tree, the AnalysisRunner path documented by the runner |

Outputs in `<out_dir>`: `exit_codes.tsv` (`id<TAB>rc`), `walls.tsv`, `binaries.md5`,
`env.txt` (source tree id from `REGRESS_SOURCE_TREE`, GPU, nvcc, gcc), `log/<id>.log`,
the case trees (`pipe/`, `r/`, `g_*`, `runner_b*`, `work/` empty-cwd scratch, not
compared) and the final `REGRESS_DONE` marker. `run.sh` exits non-zero if a case
returned an exit code outside its expected list (still recorded; the reference is
usable, the unexpected code is reported).

Comparison rules (SF-40/41 G semantics): exit codes equal; trees file by file with
`cmp`, strict file set, `timings.json` ignored, `manifest.json` through
`json_strip_cmp.py --drop timestamp --drop build.git_hash` (floats compared
exactly); R set: files of the label/particle dirs (`*.json` with `**.timings_s`, and
for trace also `transport.seconds_*` dropped) and driver-log lines with the
`t=`/`built in` timing fields masked. A head case absent in ref is a DIFF; a case
only in ref is `NOT_COMPARED` (reported, not a failure); an output without
`REGRESS_DONE` is a DIFF. `n=<k>` counts differing items (files, exit codes,
file sets, missing cases/trees).

## Protocol for a PR (V100; the orchestrator runs every remote step)

Local runs are limited to `bash scripts/regress/selftest.sh` and tiny smokes: the
R set and the runners exceed local limits.

1. Base reference, once per base commit (reuse when
   `~/regress/refs/<base-short-sha>/out` already contains `REGRESS_DONE`):
   - `git worktree add --detach <dir> $(git merge-base HEAD master)` and
     `git submodule update --init` there;
   - `scripts/remote --increment <id> sync` from that worktree (per-PR mirror);
   - `scripts/remote --increment <id> run base-build -- "cmake --preset v100-release && mkdir -p $HOME/.cache/mf3d-tmp && export TMPDIR=$HOME/.cache/mf3d-tmp && cmake --build build/v100-release -j"` then `wait`
     (`TMPDIR`: the V100 `/tmp` is a 2 GB partition that a parallel nvcc build fills, "No space left on device" in device-link);
   - `scripts/remote --increment <id> run base-regress -- "REGRESS_SOURCE_TREE=<base sha> bash scripts/regress/run.sh build/v100-release ~/regress/refs/<base-short-sha>/out"` then `wait`.
     (Each `scripts/remote run` is one job on one GPU; the cases run serially.)
2. Head: sync the PR tree to the same mirror, build `v100-release` the same way (same `mkdir -p $HOME/.cache/mf3d-tmp && export TMPDIR=$HOME/.cache/mf3d-tmp` before `cmake --build`), then
   `scripts/remote --increment <id> run head-regress -- "REGRESS_SOURCE_TREE=<head sha> bash scripts/regress/run.sh build/v100-release ~/regress/heads/<head-short-sha>/out && bash scripts/regress/compare.sh ~/regress/refs/<base-short-sha>/out ~/regress/heads/<head-short-sha>/out"`.
   The last line of the log is the verdict. `compare.sh` prints a NOTE when the two
   `binaries.md5` are identical (it then cannot detect a code change).
3. Never mix GPUs/hosts between ref and head: byte identity is only meaningful on the
   same device and toolchain.

## Declaring an expected difference

A PR that intentionally changes bytes (e.g. drops a retired config key from `effective_config.yaml`, changes a
log line) declares it in its description and in the compare command via
`--ref-filter '<glob>=<sed-expr>'` (repeatable). The sed expression is applied to
every file whose path relative to the out dir matches the bash glob (`*` also
matches `/`), in both trees, before comparing:

    compare.sh REF HEAD --ref-filter 'log/pipe_*.log=s/^version=.*$//'

A case retired by the PR needs no filter (it is `NOT_COMPARED`). A case added by the PR
is a DIFF ("absent in ref"): the base cannot run it, so declare it in the PR description
and read the remaining cases. Keep filters minimal and list them in the PR.
