Lines of `ensemble/results.txt` (verbatim, paths shortened where marked `...`).

```
D_KILL run=DB_run1 rc=9 killed_by_script=1 r0_status_ok=1 r1_stage_line=1 r1_status_line=0 CONFIRMED
D_BYTES name=DB_vs_DA files_ref=99 files_cand=99 identical=99 diff=0 missing=0 extra=0 excluded=3/3 IDENTICAL
D_BYTES name=DC_vs_DA_r0000 files_ref=32 files_cand=32 identical=32 diff=0 missing=0 extra=0 excluded=3/1 IDENTICAL
D_CSV name=analysis_only_vs_in_process:analysis/macrodispersion.csv ref=... cand=... rows=208/208 cols=7 max_rel=0.000e+00 max_abs_at_zero=0.000e+00 byte_equal=1 within=1 PASS
D_CSV name=analysis_only_vs_in_process:ensemble/ensemble_timeseries.csv ref=... cand=... rows=208/208 cols=8 max_rel=0.000e+00 max_abs_at_zero=0.000e+00 byte_equal=1 within=1 PASS
D_TTIME name=DA_stats_vs_TTIME realizations=3 rows=624 max_rel_moments=4.859e-13 max_rel_t=3.200e-10 active_equal=1 PASS
D_VERDICT PASS (D-A, D-B restart, D-C, analysis_only reproduction all met)
RECORDED run=DA exit=0 wall_s=589 host_max_rss_kb=31355600
RECORDED run=DB_run2 exit=0 wall_s=394 host_max_rss_kb=31355300
RECORDED run=DC exit=0 wall_s=201 host_max_rss_kb=31325176
```

Per-realization wall (`RECORDED D_WALL`, seconds, from `timings.json`).

| run | r | total | stage | prepare | track | bridge_repack | bridge_track | write |
|---|---|---|---|---|---|---|---|---|
| DA | 0 | 196.468 | 164.822 | 0.038 | 20.989 | 8.816 | 0.283 | 0.165 |
| DA | 1 | 194.111 | 161.502 | 0.035 | 20.551 | 10.117 | 0.284 | 0.195 |
| DA | 2 | 194.881 | 166.631 | 0.035 | 20.539 | 6.158 | 0.285 | 0.164 |
| DB | 0 | 194.839 | 164.518 | 0.037 | 21.012 | 7.547 | 0.273 | 0.177 |
| DB | 1 | 195.749 | 167.530 | 0.036 | 20.616 | 6.039 | 0.279 | 0.165 |
| DB | 2 | 194.713 | 161.602 | 0.035 | 20.628 | 10.672 | 0.293 | 0.165 |
| DC | 0 | 197.617 | 165.687 | 0.049 | 21.051 | 9.105 | 0.280 | 0.179 |

keys of `timings.json` of D-A r=0 (imported file): commit, mem_lines, pollock_timings_s, r, seconds, seed, stage_status, tracker_timings_s
