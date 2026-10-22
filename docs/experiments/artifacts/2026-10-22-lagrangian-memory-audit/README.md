# Artifacts of the Lagrangian memory audit (2026-10-22)

Note: [`../../2026-10-22-lagrangian-memory-audit.md`](../../2026-10-22-lagrangian-memory-audit.md). Code analysis plus existing measurements; no new runs except the two read-only V100 queries under `raw/`.

| file | what it is |
|---|---|
| `scripts/memory_model.py` | stdlib byte model of the stage, tracker, Par2 path, bitwise levels O1/O2/S/S+ and the feasibility table; no inputs; constants `CONTEXT_MIB = 500` (planning assumption) and `LIMIT_MIB = 30000` (owner decision) are named |
| `analysis/memory_model.txt` | exact stdout of `memory_model.py` |
| `scripts/tabulate_sf37_measurements.py` | re-tabulates versioned SF-37 data: `MEM_PEAK` distribution of the 512^3 runs, timings, evaluation counts, PCG iterations, host RSS (reads `../2026-10-19-sf37-lagrangian-cube-verification/{logs,raw}`) |
| `analysis/sf37_measurements.txt` | its stdout |
| `scripts/lh_interpolation.py` | power-law interpolation of the SF-37 post-hoc grid errors to `lambda/h = 11.28` (reads `../2026-10-19-sf37-lagrangian-cube-verification/analysis/posthoc/lh10.txt`) |
| `analysis/lh_interpolation.txt` | its stdout |
| `raw/cuobjdump_res_usage.txt` | stdout of the read-only query 1 (registers and stack of the Lagrangian kernels in the SF-37 campaign binary) |
| `raw/nvidia_smi_visible_devices.txt` | stdout of the read-only query 2 (binary timestamp; `nvidia-smi` listing with `CUDA_VISIBLE_DEVICES=1`) |
| `SHA256SUMS` | sha256 of every file above |

## Reproduce (local, light, seconds)

```bash
cd docs/experiments/artifacts/2026-10-22-lagrangian-memory-audit
python3 scripts/memory_model.py                 | diff - analysis/memory_model.txt
python3 scripts/tabulate_sf37_measurements.py   | diff - analysis/sf37_measurements.txt
python3 scripts/lh_interpolation.py             | diff - analysis/lh_interpolation.txt
sha256sum -c SHA256SUMS
# post-hoc source of lh_interpolation.py, regenerated from the versioned CSVs:
python3 ../2026-10-19-sf37-lagrangian-cube-verification/scripts/posthoc/lh10.py ../2026-10-19-sf37-lagrangian-cube-verification | head -12 \
  | diff - <(head -12 ../2026-10-19-sf37-lagrangian-cube-verification/analysis/posthoc/lh10.txt)
```

## Remote queries (read-only, run 2026-10-22 from the orchestrated worktree of node N1)

Both through the mirror of the SF-37 campaign (`~/MacroFlow3D-SF-37`); no `sync`, no `run`, no build, nothing written on the remote. The SSH client printed a post-quantum key-exchange warning on stderr for each call; it is not part of the files (`2>/dev/null`).

```bash
scripts/remote --increment SF-37 exec -- 'cuobjdump -res-usage build/v100-release/inlet_slab 2>&1 | grep -B1 -A1 -iE "lagrangian|plane_kernel|time_kernel" | head -80' > raw/cuobjdump_res_usage.txt
scripts/remote --increment SF-37 exec -- 'ls -la --time-style=full-iso build/v100-release/inlet_slab; CUDA_VISIBLE_DEVICES=1 nvidia-smi --query-gpu=index,name,memory.total --format=csv' > raw/nvidia_smi_visible_devices.txt
```

- The mangled kernel names were readable with the filter as given (no adaptation was needed); lines cut by `grep -B1 -A1` carry the name on the `Function` line and the numbers on the following line.
- The binary is the SF-37 campaign binary: `build/v100-release/inlet_slab`, 2026-10-20 16:38:52 -0300, built from the SF-37 measured tree H1 `d866c40` (job `sf37-build`, md5 in `../2026-10-19-sf37-lagrangian-cube-verification/logs/jobs/sf37-build.log`). The sources `src/`, `apps/`, `CMakeLists.txt` of `d866c40` and of `da1c185` are identical; the binary is not a build of `da1c185`.
- `nvidia-smi` lists both GPUs (index 0 and 1, Tesla V100-PCIE-32GB, 32510 MiB each) although `CUDA_VISIBLE_DEVICES=1` is set: it ignores that variable.
