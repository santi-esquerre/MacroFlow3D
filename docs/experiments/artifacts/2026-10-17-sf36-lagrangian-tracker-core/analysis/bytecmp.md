# SF-36 byte comparison against the base-commit references (criterion F)

Run environment (`run.env`): COMMIT=e81b11833412149027643293d7e11ff989643189; HOST=localhost.localdomain; DATE_UTC=2026-10-17T22:37:34Z; BUILD=/home/sesquerre/MacroFlow3D-SF-36/build/v100-release; PART=bytecmp; DRYRUN=0; BASE_REFS=/home/sesquerre/sf36_base_refs; CUDA_VISIBLE_DEVICES=0; GPU=Tesla V100-PCIE-32GB

Verdicts are mechanical: `PASS` = psi1/psi2 `cmp`-identical (or n/a), selected lines identical after masking the timing fields, and equal exit codes (`?` = the base exit code was not recorded for that case).

| case | npy | lines | rc base/head | verdict | note |
|---|---|---|---|---|---|
| N32_e0.5_host | IDENTICAL | IDENTICAL | 0/0 | PASS |  |
| N32_e0.5_gpu | IDENTICAL | IDENTICAL | 0/0 | PASS |  |
| N32_e1.0_host | IDENTICAL | IDENTICAL | 21/21 | PASS |  |
| N32_e1.0_gpu | IDENTICAL | IDENTICAL | 21/21 | PASS |  |
| N64_e0.5_host | IDENTICAL | IDENTICAL | 0/0 | PASS |  |
| N64_e0.5_gpu | IDENTICAL | IDENTICAL | 0/0 | PASS |  |
| N64_e1.0_host | IDENTICAL | IDENTICAL | 0/0 | PASS |  |
| N64_e1.0_gpu | IDENTICAL | IDENTICAL | 0/0 | PASS |  |
| N64_e2.0_gpu | IDENTICAL | IDENTICAL | 21/21 | PASS |  |
| N128_e2.0_gpu | IDENTICAL | IDENTICAL | 21/21 | PASS |  |
| prod32_vs_base | n/a | IDENTICAL | ?/0 | PASS |  |
| prod32_vs_sf33 | n/a | IDENTICAL | ?/0 | PASS | (ref /home/sesquerre/MacroFlow3D-SF-36/docs/experiments/artifacts/2026-10-06-sf33-gpu-inlet-labels/logs/ladder_0.25_c4/N32.log) |

Mismatches: 0; skipped: 0.

## Exit codes of the head runs

| run | exit code |
|---|---|
| N32_e0.5_host | 0 |
| N32_e0.5_gpu | 0 |
| N32_e1.0_host | 21 |
| N32_e1.0_gpu | 21 |
| N64_e0.5_host | 0 |
| N64_e0.5_gpu | 0 |
| N64_e1.0_host | 0 |
| N64_e1.0_gpu | 0 |
| N64_e2.0_gpu | 21 |
| N128_e2.0_gpu | 21 |
| prod32 | 0 |

### N128_e2.0_gpu.linecmp

```
    IDENTICAL 135
```

### N32_e0.5_gpu.linecmp

```
    IDENTICAL 39
```

### N32_e0.5_host.linecmp

```
    IDENTICAL 39
```

### N32_e1.0_gpu.linecmp

```
    IDENTICAL 39
```

### N32_e1.0_host.linecmp

```
    IDENTICAL 39
```

### N64_e0.5_gpu.linecmp

```
    IDENTICAL 71
```

### N64_e0.5_host.linecmp

```
    IDENTICAL 71
```

### N64_e1.0_gpu.linecmp

```
    IDENTICAL 71
```

### N64_e1.0_host.linecmp

```
    IDENTICAL 71
```

### N64_e2.0_gpu.linecmp

```
    IDENTICAL 71
```

### prod32_vs_base.linecmp

```
    IDENTICAL 102
```

### prod32_vs_sf33.linecmp

```
    IDENTICAL 102
```
