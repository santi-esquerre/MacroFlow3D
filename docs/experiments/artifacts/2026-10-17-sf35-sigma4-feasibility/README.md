# Artifact: SF-35 sigma_Y^2 = 4 feasibility probes (2026-10-17)

Evidence for [`../../2026-10-17-sf35-sigma4-feasibility.md`](../../2026-10-17-sf35-sigma4-feasibility.md).
Exploratory probes run after the SF-35 FINAL_AUDIT and before the owner decision of 2026-10-17
(`docs/decisions/2026-10-17-lagrangian-consumer-sigma4.md`). They are **not pre-registered** and are
not acceptance evidence of any increment.

## Provenance

- Probe branch (local, not pushed, not for merge): `worktree-agent-a53f621a4de3d5aaf`, head `d7b74d9`,
  base `747adc7` (SF-35 PR #53 head). The complete diff is [`probe.patch`](probe.patch)
  (`git diff 747adc7 d7b74d9`): `--save-field` (unit-amplitude `Y_cells.npy`), the release of dead
  `N^3` temporaries in `build_production_stage` ("P2") with the `MF_PROBE_KEEP_TEMPS` control that
  restores the old lifetimes, and the job scripts `tools/probe_sf35/*`.
- Binary: built by job `J0-build` (start 2026-10-17T13:55:56Z) from source commit `e1e376e`
  (committed 13:55:23Z); `git diff e1e376e d7b74d9 -- apps src CMakeLists.txt` is empty, so every
  probe ran the source of `d7b74d9`. V100 (`Tesla V100-PCIE-32GB`), CUDA 11.4.152, GCC 9.3.1, preset
  `v100-release`; `ctest -R inlet_slab` 9/9 in the same job.
- Remote: `scripts/remote --increment SF-35-feas` (mirror `~/MacroFlow3D-SF-35-feas`), outputs under
  `~/sf35_feas_probe/` on the V100 host. Label exports (`exports/`, `J1/L_*`, `controls/L_*`,
  up to 1.08 GB per field at 512^3) and the saved fields (`Y256/`, `controls/Y*/`) stay on the host
  and are not committed.

## Layout

| path | content |
|---|---|
| `probe.patch` | full probe diff (source + scripts) |
| `logs/jobs/*.log` | `scripts/remote` job logs: `J0-build` (build + `ctest -R inlet_slab`), `J0b-controls` (failed, exit 126: scripts not executable; fixed by `aaed917`), `J0c-controls` (P1/P2/roll controls), `J1a-n128`, `J1b-n256` (anchor shift), `J2a-n256`, `J2b-n512` (refinement) |
| `logs/status/*.env`, `logs/commands/*` | job status files and exact commands |
| `raw/controls/` | P1/P2 controls at 64^3 and 128^3: driver logs, `--summary` JSON (per-plane metrics), `/usr/bin/time -v`, exit codes, extracted comparison lines |
| `raw/J1/` | anchor-shift runs (`N128_ctrl`, `N128_k24`, `N128_k16`, `N256_k48`, `N256_k32`) |
| `raw/J2/` | 256^3 and 512^3 runs; `N*.mem.gz` = nvidia-smi `memory.used` samples (MiB, ~200 ms, device-wide); `label_diagnostics_eps20.md` = SF-35 `label_diagnostics.py` on the eps 2 exports 128/256/512 |
| `scripts/planes.py` | per-plane order digest (copy of `tools/probe_sf35/planes.py`) |
| `scripts/memory_phases.py` | per-phase device-memory digest of `raw/J2` |
| `analysis/planes_anchor0.md` | per-plane orders at the original anchor, 128 -> 256 and 256 -> 512 (post-hoc digest of committed JSON) |
| `analysis/memory_phases.md` | device memory by phase (field / stage / trace) at 256^3 and 512^3 |

Exit code 21 in `*.rc` = `roundtrip_exceeded` (SF-35 driver convention); 0 = ok.

## Regenerate the digests

```bash
A=docs/experiments/artifacts/2026-10-17-sf35-sigma4-feasibility
python3 $A/scripts/planes.py $A/raw/controls/N128_new.json $A/raw/J2/N256.json
python3 $A/scripts/planes.py $A/raw/J2/N256.json $A/raw/J2/N512.json
python3 $A/scripts/memory_phases.py $A/raw/J2
```
