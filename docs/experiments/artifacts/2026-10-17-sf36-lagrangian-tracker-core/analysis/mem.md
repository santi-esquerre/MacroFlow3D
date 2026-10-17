# SF-36 device peak (criterion G)

Run environment (`run.env`): COMMIT=e81b11833412149027643293d7e11ff989643189; HOST=localhost.localdomain; DATE_UTC=2026-10-17T22:40:19Z; BUILD=/home/sesquerre/MacroFlow3D-SF-36/build/v100-release; PART=mem; DRYRUN=0; BASE_REFS=/home/sesquerre/sf36_base_refs; CUDA_VISIBLE_DEVICES=0; GPU=Tesla V100-PCIE-32GB

## Result lines

```
HEAD_PEAK_MIB=3771   (N=256, nvidia-smi memory.used, 200 ms, physical GPU 0)
BASE_PEAK_MIB=6239   (max of /home/sesquerre/sf36_base_refs/mem/N256_e2.mem; orchestrator recorded 6239)
PROBE_REF_MIB=3769   (P2 probe; band +-2 % = 3693.6..3844.4)
G_VERDICT head<=base: yes ; head within 2 % of probe 3769: yes
```

Sampler `N256_e2.mem`: 2859 samples, max 3771 MiB, min 0 MiB.
- `N256_e2.log`: `MEMORY peak_device_bytes=3664773120 free_min=30424956928 total_device=34089730048 workspaces_bytes=2299055151 (peak/free: cudaMemGetInfo polls before/after every phase, device-wide)`
- `N256_e2.log`: `STATUS roundtrip_exceeded`

## Exit codes

| run | exit code |
|---|---|
| N256_e2 | 21 |
