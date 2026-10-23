Per part: start (`run.env` DATE_UTC), `PART_EXIT`, number of runs in `exit_codes.tsv`, non-zero rcs, `part_seconds` and `part_runs_seconds` (RECORDED lines of `results.txt`).

| part | start (UTC) | PART_EXIT | runs | non-zero rc | part_seconds | part_runs_seconds |
|---|---|---|---|---|---|---|
| baserefs | 2026-10-23T18:21:02Z | 0 | 23 | 21 | 177 | 164 |
| controls | 2026-10-23T18:34:32Z | 0 | 7 | - | 144 | 126 |
| prodgrid | 2026-10-23T18:36:56Z | 0 | 3 | - | 494 | 469 |
| bpilot | 2026-10-23T18:45:10Z | 0 | 1 | - | 170 | 155 |
| ab | 2026-10-23T18:48:00Z | 0 | 2 | - | 2627 | 2607 |
| ensemble | 2026-10-23T19:31:47Z | 0 | 5 | 9 | 1573 | 1546 |
| fieldonly | 2026-10-23T19:58:00Z | 0 | 3 | - | 38 | 19 |
| par2pilot | 2026-10-23T19:58:38Z | 0 | 16 | - | 4474 | 4215 |
| final | 2026-10-23T21:14:13Z | 0 | 24 | 21 | 3246 | 3231 |

derived: sum of `part_seconds` = 12943 s = 3.595 h; sum of `part_runs_seconds` = 12532 s = 3.481 h (GPU 0, one job at a time).
