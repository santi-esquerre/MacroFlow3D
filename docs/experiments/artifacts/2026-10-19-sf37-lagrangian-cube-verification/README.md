# SF-37 artifact: verification of the Lagrangian consumer on the cube (sigma^2 in {1, 2.25, 4}), blind O3

Evidence of increment SF-37 (tolerance ladder V1, grid ladder V2, SF-35 gridded labels V3a, Pollock comparator V3b,
Lagrangian divergence V4, V5, production lambda/h, blind robust metric O3 of the gridded labels). This README says what
is where, where each number comes from and how to regenerate it. It is not a result summary: results live in
`analysis/` and in the experiment note of the increment.

- Increment spec: `docs/plans/active/lester-eq14/increments/SF-37-lagrangian-cube-verification.md` (thresholds,
  pre-registered reading rules RR1-RR15, bitácora).
- Decision: `docs/decisions/2026-10-17-lagrangian-consumer-sigma4.md`. Tracker under test: SF-36
  (`src/physics/particles/lagrangian_labels/`, `inlet_slab --track forward`; evidence
  `docs/experiments/artifacts/2026-10-17-sf36-lagrangian-tracker-core/`).
- Pattern: the SF-36 artifact above and the SF-35 artifact `docs/experiments/artifacts/2026-10-07-sf35-label-transport/`.

## Provenance

| item | value | source |
|---|---|---|
| increment base | `master 44d4697` (`44d4697107b65d4b67ff84b4f84e3adfc2379e68`); activation commit `b27ab81` (`b27ab8101e6e879854103940ec2b96b53d10195f`) | `logs/jobs/sf37-base-refs.log` line 5 (`BASE_TREE_COMMIT`, `ACT_COMMIT`) |
| measured source tree | H1 `d866c40` (`d866c40875cc52b3bdef7c7c9921b00b504a2350`): every head-binary part (build, dryrun, bytecmp, v2, science, final) | `FINAL_TREE_COMMIT` line of `logs/jobs/sf37-final.log`; `COMMIT=` in the env lines of `logs/jobs/sf37-{bytecmp,v2,science}.log`, `raw/*/run.env`, `analysis/run.env` |
| delivery head before this import | `530bc76` (H1 plus orchestrator bitácora rows; docs/evidence only, no source change) | git history |
| base binary | `~/sf37_base_refs/bin/inlet_slab` md5 `fef042097f08dc725831f3df7ab975b5`, built from `44d4697` by job `sf37-base-refs`; used for the label constructions (jobs `sf37-labels-eps10`, `sf37-labels-o3`; mirror tree `610cfc4`, `SF37_COMMIT=610cfc471b9ee8b50e22eb0a5932ec97e57a6d14`) and as the criterion-R reference (`sf37-bytecmp`) | `logs/jobs/sf37-base-refs.log` (md5sum lines), `logs/jobs/sf37-labels-eps10.log` (`BIN_MD5`, `COMMIT`), `logs/jobs/sf37-build.log` (md5 of both binaries) |
| head binary | `build/v100-release/inlet_slab` md5 `a2ecce53bdeda469b41593f706dd7028` (`BIN_MD5` in every part env line of `sf37-bytecmp`, `sf37-v2`, `sf37-science`); `macroflow3d_pipeline` md5 `d0112ce8497b751303543c7de60a9e94` (base: `08e41b62f93d516464a9926cec1ad3e9`) | `logs/jobs/sf37-build.log`, `logs/jobs/sf37-final.log` (md5sum lines) |
| hardware | GPU 0 Tesla V100-PCIE-32GB (`CUDA_VISIBLE_DEVICES=0`), host `localhost.localdomain`, strictly sequential jobs | `logs/jobs/*.log` line 2, `raw/*/run.env` (`GPU=`) |
| analysis toolchain | python 3.11.7, numpy 1.26.4, scipy 1.11.4 | header lines of `analysis/*/*.csv`, `analysis/*/*summary.md`, `logs/jobs/sf37-audit.log` |
| bootstrap matrix | committed file `analysis/bootstrap_blocks.npy` sha256 `c7abb02bbc9516f5dfd82a4781b020bc70641471f9f6d9368774fc00905dc513` (file bytes; `analysis/*/input_manifest_*.csv`); sha256 of the int64 index matrix bytes `16c1a5b9d9fe062576d5f18fc9d54127db76472c7fbe64493201d8cffc66af39` (`bootstrap_sha256=` in the digest CSV headers, `scripts/sf37lib.py`) | as cited |
| SF-37 mirror | `~/MacroFlow3D-SF-37` on the V100 host (synced without `.git`; the hash is passed by `SF37_COMMIT` / `SF37_FINAL_COMMIT`) | `logs/commands/*.command` |

Jobs (all `STATE=succeeded`, `EXIT_CODE=0`; start = first line of `logs/jobs/<job>.log`, end = `UPDATED_AT` of
`logs/status/<job>.env`; all GPU 0; strictly sequential on the shared mirror):

| job | what | tree / binary | start (UTC) | end (UTC) | exit |
|---|---|---|---|---|---|
| `sf37-base-refs` | build of `44d4697`, base binaries, criterion-R references (`scripts/sf37_base_refs.sh`) | `44d4697` | 2026-10-19T13:33:50Z | 2026-10-19T13:48:37Z | 0 |
| `sf37-labels-eps10` | part `labels-eps10`: gridded labels eps = 1, seeds 3101..3103, N 64/128/256 | base binary, `610cfc4` | 2026-10-19T14:03:33Z | 2026-10-19T14:31:46Z | 0 |
| `sf37-labels-o3` | part `labels-o3`: gridded labels eps = 1.5 and 2 (blind O3 input), 24 cases | base binary, `610cfc4` | 2026-10-19T14:32:43Z | 2026-10-20T15:13:29Z | 0 |
| `sf37-build` | `cmake --preset v100-release` + build of H1; md5 of both binaries | H1 `d866c40` | 2026-10-20T19:37:32Z | 2026-10-20T19:38:52Z | 0 |
| `sf37-dryrun` | `SF37_DRYRUN=1` smoke of the runner (NOT evidence; output tree `~/sf37_dryrun` not imported) | H1 | 2026-10-20T19:39:48Z | 2026-10-20T19:47:02Z | 0 |
| `sf37-bytecmp` | part `bytecmp` (criterion R: head vs base) | H1 | 2026-10-20T19:48:46Z | 2026-10-20T19:51:26Z | 0 |
| `sf37-v2` | part `v2` (grid ladder, V4/V5 inputs) | H1 | 2026-10-20T19:51:56Z | 2026-10-20T21:52:25Z | 0 |
| `sf37-science` | parts `v3a`, `v3b`, `v1`, `digest` in one job (per-part tee logs in `logs/parts/`) | H1 | 2026-10-21T13:57:58Z | 2026-10-22T07:56:16Z | 0 |
| `sf37-final` | FINAL_AUDIT: head build, full V100 `ctest` (38/38 passed, wall 2902 s), SF-27 trio byte-compare against the base (`scripts/sf37_final.sh`) | H1 | 2026-10-22T07:58:19Z | 2026-10-22T08:46:56Z | 0 |
| `sf37-audit` | orchestrator independent recomputation (`scripts/audit_recompute.py`, see `analysis/audit/`) | H1 outputs | 2026-10-22T08:49:15Z | 2026-10-22T08:50:36Z | 0 |

`sf37-science` ran the parts v3a, v3b, v1 and digest as one job after the local chain that was driving them died with
the orchestrator session (GPU idle 2026-10-20T21:52Z..2026-10-21T13:58Z); see the bitácora row 2026-10-22T09:15Z of the
increment spec. The inner part windows are the `PART_BEGIN` / `PART_EXIT` lines of `logs/jobs/sf37-science.log`.

## Layout

`A` = this directory. Everything below is text copied unmodified from the host (no `.npy`, no file above 10 MB).

| path | content | source on the host |
|---|---|---|
| `scripts/` | campaign code: `run_sf37.sh` (runner, one part per detached job), `sf37lib.py`, `digest_v1v2.py`, `digest_v4v5.py`, `digest_v3.py`, `o3.py`, `lines_cmp.py`, `json_strip_cmp.py`, `make_bootstrap_blocks.py`, self-tests `selftest_sf37.py`, `selftest_o3_v3.py`; orchestrator jobs `sf37_base_refs.sh` (job `sf37-base-refs`), `sf37_final.sh` (job `sf37-final`), `audit_recompute.py` (job `sf37-audit`, executed from its embedded copy; byte-identical to it) | repository |
| `analysis/bootstrap_blocks.npy` | pre-registered block-bootstrap index matrix (committed before the campaign; never regenerated by the digests) | repository |
| `analysis/{v1v2,v4v5,v3,o3}/` | digests of the campaign: CSV tables, `*_summary.md`, `input_manifest_*.csv` (path + sha256 of every file read), `criterionC_s31xx.{md,_planes.csv}` | `~/sf37_campaign/analysis/` (part `digest` of `run_sf37.sh`) |
| `analysis/{v1v2,v4v5,v3,o3}.{log,time}`, `analysis/exit_codes.tsv`, `analysis/run.env` | digest stdout/stderr, `/usr/bin/time -v` files, return codes (all 0), run environment of the part `digest` | same |
| `analysis/audit/` | **orchestrator independent recomputation** (not the committed digests): `audit_summary.txt`, `audit.time`, `v1_cells.csv`, `v2_cells.csv`, `lambda_h_cells.csv`, `v3b_cells.csv`, `v4_cells.csv`, `o3_cells.csv`, `v3a_spot.csv`; produced by `scripts/audit_recompute.py` (`AUDIT_SCRIPT_SHA256=ebafc94f330d7bd99e01320a08aa0859102d9910e504487c3b41967fcdd3856c`, `logs/jobs/sf37-audit.log`) run as `audit_recompute.py ~/sf37_campaign analysis/bootstrap_blocks.npy ~/sf37_audit --o3 --v3a`; written from the spec text without reading the digest code; compared cell by cell with the digests by the orchestrator | `~/sf37_audit/` |
| `raw/labels/` | `raw/eps{E}_s{S}/N{N}.json` (driver `--summary`), `logs/eps{E}_s{S}/N{N}.{log,time}`, `exports/eps{E}_s{S}/N{N}/labels.json`, `exit_codes_<part>.tsv`, `run_<part>.env`, `sha256_<part>.txt` (host manifest, includes the non-imported `.npy`), `preflight_<part>/`; E in {10, 15, 20}, S in {3101, 3102, 3103}, N in {64, 128, 256, 512} | `~/sf37_campaign/labels/` |
| `raw/v2/`, `raw/v1/`, `raw/v3a/`, `raw/v3b/` | one directory per run (`moments_planes.csv`, `moments_time.csv`, `track.json`, ...), plus `<run>.json`, `<run>.log`, `<run>.time`, `.mem.gz` samplers where present, `exit_codes.tsv`, `run.env`, `sha256.txt`, `walls.tsv` | `~/sf37_campaign/<part>/` |
| `raw/bytecmp/` | criterion R: `results.txt` (`BYTECMP_ALL_IDENTICAL`), `exit_codes.tsv`, `run.env`, `sha256.txt`, base/head `trace`, `fwd`, `neutral` comparison directories | `~/sf37_campaign/bytecmp/` |
| `logs/jobs/<job>.log`, `logs/status/<job>.env`, `logs/commands/<job>.command` | remote job records of every SF-37 job (`scripts/remote`) | `~/.macroflow3d-remote/macroflow3d-SF-37/{logs,status,commands}/` |
| `logs/parts/` | per-part tee logs of the job `sf37-science` (`sf37-science-{v3a,v3b,v1,digest}.log`) | `~/sf37_campaign/jobs/` |

Host output root: `${SF37_OUT_ROOT:-$HOME/sf37_campaign}/<part>/`; base references: `${SF37_BASE_REFS:-$HOME/sf37_base_refs}`.

## Integrity of the import

The evidence was staged as one tarball (sha256 `a19e7bfa3388d32c2427e32994334713ee0c741608b8d7d5beecfe63d91acaa2`) with a
`MANIFEST.sha256` of 1888 files (all verified), then copied byte for byte. Check an imported part against the host
manifest from the part directory (only the imported subset exists here), e.g.

```bash
cd $A/raw/v2 && while read -r h p; do [ -f "$p" ] && echo "$h  $p"; done < sha256.txt | sha256sum -c -
```

(the manifest line format is `<sha256>  <relative path>`; `labels` has `sha256_labels-eps10.txt` and
`sha256_labels-o3.txt`). The lines of non-imported files (all `.npy`) are skipped.

## Regeneration

From the repository root of the synced mirror (`~/MacroFlow3D-SF-37`), `A=docs/experiments/artifacts/2026-10-19-sf37-lagrangian-cube-verification`.
Each line is one detached V100 job; serialize them (shared mirror and GPU). The label parts use the BASE binary
directory (`~/sf37_base_refs/bin`, produced by `sf37_base_refs.sh` from a clean `44d4697` tree), the other parts the head
build.

```bash
scripts/remote --increment SF-37 run sf37-base-refs -- "bash sf37_base_refs.sh"   # script copied to the mirror root, tree 44d4697
scripts/remote --increment SF-37 run sf37-labels-eps10 -- "SF37_COMMIT=<hash> bash $A/scripts/run_sf37.sh ~/sf37_base_refs/bin labels-eps10"
scripts/remote --increment SF-37 run sf37-labels-o3 -- "SF37_COMMIT=<hash> bash $A/scripts/run_sf37.sh ~/sf37_base_refs/bin labels-o3"
scripts/remote --increment SF-37 exec -- "cmake --preset v100-release && cmake --build build/v100-release -j"
scripts/remote --increment SF-37 run sf37-bytecmp -- "SF37_COMMIT=<hash> bash $A/scripts/run_sf37.sh build/v100-release bytecmp"
scripts/remote --increment SF-37 run sf37-v2 -- "SF37_COMMIT=<hash> bash $A/scripts/run_sf37.sh build/v100-release v2"
scripts/remote --increment SF-37 run sf37-science -- "<loop of logs/commands/sf37-science.command: parts v3a v3b v1 digest>"
scripts/remote --increment SF-37 run sf37-final -- "export SF37_FINAL_COMMIT=<hash>; <body of scripts/sf37_final.sh>"
```

Exact command strings are in `logs/commands/*.command`. Each `scripts/remote wait <job>` follows its `run`. To regenerate
the digests from a populated output root (no GPU):

```bash
for pair in digest_v1v2:v1v2 digest_v4v5:v4v5 digest_v3:v3 o3:o3; do
  python3 $A/scripts/${pair%%:*}.py "$SF37_OUT_ROOT" "$SF37_OUT_ROOT/analysis/${pair#*:}" --design full
done
```

The committed `raw/` subset is not a complete output root (the `.npy` particle/label/vD files are absent), so the
digests cannot be regenerated from this directory alone; their `input_manifest_*.csv` list the sha256 of every file read.

## Not committed

All of these stay on the V100 host under `~/sf37_campaign` (or the paths given) and are not in the repository:

- every `.npy` output: particle files, label exports (`psi1.npy`, `psi2.npy`, about 1.08 GB per 512^3 field), `vD`
  exports (`--save-vd`), seed-velocity files, `labels.npy` of the runs;
- the dry-run tree `~/sf37_dryrun` (job `sf37-dryrun`, not evidence);
- the base references `~/sf37_base_refs` (binaries, SF-27 trio, `--construct trace` and `--track forward` references);
- the head references `~/sf37_head_refs` (trio of job `sf37-final`) and the audit job directory `~/sf37_audit_job`;
- the host manifests list the `.npy` files by sha256 (`raw/*/sha256*.txt`), so a re-transfer from the host can be verified.
