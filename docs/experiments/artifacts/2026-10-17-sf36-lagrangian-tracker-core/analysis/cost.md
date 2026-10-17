# SF-36 cost of the forward tracker (criterion J, recorded, no threshold)

Run environment (`run.env`): COMMIT=e81b11833412149027643293d7e11ff989643189; HOST=localhost.localdomain; DATE_UTC=2026-10-17T22:50:36Z; BUILD=/home/sesquerre/MacroFlow3D-SF-36/build/v100-release; PART=cost; DRYRUN=0; BASE_REFS=/home/sesquerre/sf36_base_refs; CUDA_VISIBLE_DEVICES=0; GPU=Tesla V100-PCIE-32GB

fevals/particle/length = nfev(plane pass) / (n_ok(plane pass) x X_last); fevals/particle/time = nfev(time pass) / (n_ok(time pass) x T_last).  `n_ok` is `hist[0]` of the TSTATUS line of that pass; `X_last`, `T_last` the last entries of the TRACK `planes=` / `times=` lists.  Device peak = max of the nvidia-smi 200 ms sampler (whole GPU) and `peak_device_bytes` of the driver MEMORY line.

| run | Np | status | wall [h:mm:ss] | driver total [s] | tracker [s] | nfev plane | fevals/part/len | nfev time | fevals/part/time | backflow enc | device peak MiB (smi / drv) | host max RSS kB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| N64_e1_cov | 65536 | ok | 0:08.23 | 6.4 | plane 1.86 / time 2.27 / dl 0.03 / mom 0.03 | 861334862 | 3285.7 | 839253068 | 3201.5 | 0 | 785 / 785 | 682792 |
| N64_e1_plain | 65536 | ok | 0:07.05 | 5.1 | plane 1.25 / time 1.58 / dl 0.02 / mom 0.03 | 861334862 | 3285.7 | 839253068 | 3201.5 | 0 | 651 / 651 | 656200 |
| N64_e2_cov | 65536 | ok | 0:10.28 | 8.2 | plane 2.35 / time 3.51 / dl 0.03 / mom 0.03 | 1009678376 | 3851.6 | 1038538244 | 3961.7 | 0 | 785 / 785 | 684172 |
| N64_e2_plain | 65536 | ok | 0:08.03 | 6.3 | plane 1.62 / time 2.36 / dl 0.02 / mom 0.03 | 1009678376 | 3851.6 | 1038538244 | 3961.7 | 0 | 651 / 651 | 656308 |
| N128_e1_cov | 65536 | ok | 0:15.07 | 13.2 | plane 4.11 / time 5.36 / dl 0.03 / mom 0.03 | 1718603336 | 6556.0 | 1672446752 | 6379.9 | 0 | 957 / 957 | 1013904 |
| N128_e1_plain | 65536 | ok | 0:12.62 | 10.9 | plane 3.18 / time 4.03 / dl 0.02 / mom 0.03 | 1718603336 | 6556.0 | 1672446752 | 6379.9 | 0 | 891 / 823 | 997580 |
| N128_e2_cov | 65536 | ok | 0:19.61 | 17.7 | plane 5.52 / time 8.48 / dl 0.03 / mom 0.03 | 2015148236 | 7687.2 | 2071766630 | 7903.2 | 0 | 957 / 957 | 1012336 |
| N128_e2_plain | 65536 | ok | 0:16.36 | 14.3 | plane 4.23 / time 6.39 / dl 0.02 / mom 0.03 | 2015148236 | 7687.2 | 2071766630 | 7903.2 | 0 | 891 / 823 | 1012444 |
| N256_e1_cov | 65536 | ok | 0:44.23 | 42.3 | plane 11.78 / time 14.55 / dl 0.04 / mom 0.03 | 3433060916 | 13096.1 | 3339750134 | 12740.1 | 0 | 3771 / 2143 | 3769272 |
| N256_e1_plain | 65536 | ok | 0:40.14 | 38.0 | plane 10.21 / time 11.83 / dl 0.04 / mom 0.03 | 3433060916 | 13096.1 | 3339750134 | 12740.1 | 0 | 3771 / 2009 | 3769280 |
| N256_e2_cov | 65536 | ok | 0:56.45 | 54.3 | plane 15.50 / time 23.00 / dl 0.03 / mom 0.04 | 4026126752 | 15358.5 | 4138737188 | 15788.0 | 0 | 3771 / 2143 | 3769152 |
| N256_e2_plain | 65536 | ok | 0:50.94 | 48.8 | plane 14.00 / time 18.97 / dl 0.02 / mom 0.03 | 4026126752 | 15358.5 | 4138737188 | 15788.0 | 0 | 3771 / 2009 | 3769284 |

## Exit codes

| run | exit code |
|---|---|
| N64_e1_plain | 0 |
| N64_e1_cov | 0 |
| N64_e2_plain | 0 |
| N64_e2_cov | 0 |
| N128_e1_plain | 0 |
| N128_e1_cov | 0 |
| N128_e2_plain | 0 |
| N128_e2_cov | 0 |
| N256_e1_plain | 0 |
| N256_e1_cov | 0 |
| N256_e2_plain | 0 |
| N256_e2_cov | 0 |

## Result lines

```
  COST N64_e1_plain rc=0 device_peak_MiB=651 max_rss_kB=656200 wall=0:07.05
  COST N64_e1_cov rc=0 device_peak_MiB=785 max_rss_kB=682792 wall=0:08.23
  COST N64_e2_plain rc=0 device_peak_MiB=651 max_rss_kB=656308 wall=0:08.03
  COST N64_e2_cov rc=0 device_peak_MiB=785 max_rss_kB=684172 wall=0:10.28
  COST N128_e1_plain rc=0 device_peak_MiB=891 max_rss_kB=997580 wall=0:12.62
  COST N128_e1_cov rc=0 device_peak_MiB=957 max_rss_kB=1013904 wall=0:15.07
  COST N128_e2_plain rc=0 device_peak_MiB=891 max_rss_kB=1012444 wall=0:16.36
  COST N128_e2_cov rc=0 device_peak_MiB=957 max_rss_kB=1012336 wall=0:19.61
  COST N256_e1_plain rc=0 device_peak_MiB=3771 max_rss_kB=3769280 wall=0:40.14
  COST N256_e1_cov rc=0 device_peak_MiB=3771 max_rss_kB=3769272 wall=0:44.23
  COST N256_e2_plain rc=0 device_peak_MiB=3771 max_rss_kB=3769284 wall=0:50.94
  COST N256_e2_cov rc=0 device_peak_MiB=3771 max_rss_kB=3769152 wall=0:56.45
```
