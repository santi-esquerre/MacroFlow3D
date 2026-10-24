B verdict lines (copied from `prodgrid/results.txt`): correctors are `its,rel_residual` for x;y;z.

| run | grid | exit | correctors (its,rel) x; y; z | mg_levels/auto | PID peak MiB | device-wide peak MiB | samples | wall s | host max RSS kB | TIMING r=0 s | B_VERDICT |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B1 | 2048x256x256 | 0 | converged,20,3.294e-14; converged,20,2.592e-14; converged,20,2.804e-14 | 7/7 | 8257 | 8263 | 262 | 53.93 | 9021988 | 52.088 | PASS |
| B2 | 3328x384x384 | 0 | converged,30,8.367e-14; converged,30,6.495e-14; converged,30,7.528e-14 | 7/7 | 28885 | 28895 | 1022 | 209.11 | 31325308 | 207.139 | PASS |
| B3 | 3328x384x384 | 0 | converged,30,8.941e-12; converged,30,8.486e-12; converged,30,4.614e-12 | 7/7 | 28885 | 28895 | 976 | 200.71 | 31325124 | 198.898 | PASS |

Stage, backflow and status lines (copied from `B*.log`).

| run | STAGE status | stage s | inlet_vmin | labels_available | seed_backflow | backflow_enc plane | backflow_enc time | TSTATUS plane hist | TSTATUS time hist | TWIN planes/times | STATUS |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B1 | ok | 38.786 | 4.054640e-02 | 1 | 0 | 0 | 0 | 10000,0,0,0,0,0,0 | 10000,0,0,0,0,0,0 | 10/410 | ok tracker_non_ok_plane=0 tracker_non_ok_time=0 seed_backflow=0 pollock_weighted_not_active=0 |
| B2 | ok | 175.256 | 5.043988e-03 | 1 | 0 | 72 | 75 | 10000,0,0,0,0,0,0 | 10000,0,0,0,0,0,0 | 10/416 | ok tracker_non_ok_plane=0 tracker_non_ok_time=0 seed_backflow=0 pollock_weighted_not_active=0 |
| B3 | inlet_backflow | 166.465 | -4.297129e-02 | 0 | 0 | 1117 | 1011 | 10000,0,0,0,0,0,0 | 10000,0,0,0,0,0,0 | 9/208 | ok tracker_non_ok_plane=0 tracker_non_ok_time=0 seed_backflow=0 pollock_weighted_not_active=0 |

MEM_PHASE lines (copied from `B*.log`; MiB = bytes / 1048576, derived).

| run | phase | exact_bytes | exact MiB | device_used B | device_used MiB | seconds |
|---|---|---|---|---|---|---|
| B1 | F | 3223324680 | 3074.002 | 2666528768 | 2543.000 | 0.873793 |
| B1 | S1 | 5371854848 | 5123.000 | 3740270592 | 3567.000 | 4.241701 |
| B1 | S2 | 8146667625 | 7769.268 | 8664383488 | 8263.000 | 28.361181 |
| B1 | S3 | 6584535016 | 6279.502 | 4952424448 | 4723.000 | 4.307281 |
| B1 | T | 2418951048 | 2306.891 | 2937061376 | 2801.000 | 0.073570 |
| B2 | F | 11782326280 | 11236.502 | 8372879360 | 7985.000 | 3.595685 |
| B2 | S1 | 19636420608 | 18726.750 | 12298747904 | 11729.000 | 17.724351 |
| B2 | S2 | 29765452393 | 28386.547 | 30298603520 | 28895.000 | 128.284927 |
| B2 | S3 | 23775677416 | 22674.253 | 16440623104 | 15679.000 | 21.863574 |
| B2 | T | 8193545144 | 7813.974 | 8712617984 | 8309.000 | 0.083368 |
| B3 | F | 11782326280 | 11236.502 | 8372879360 | 7985.000 | 3.797751 |
| B3 | S1 | 19636420608 | 18726.750 | 12298747904 | 11729.000 | 17.907992 |
| B3 | S2 | 29765452393 | 28386.547 | 30298603520 | 28895.000 | 126.386810 |
| B3 | S3 | 23572512768 | 22480.500 | 16235102208 | 15483.000 | 15.751644 |
| B3 | T | 7923258824 | 7556.208 | 8439988224 | 8049.000 | 0.037765 |

derived: PID memory in the tracking and bridge phase. The sampler clock origin is the first sample of the PID (an approximation of the process start); `late` = samples more than 5 s after the stage build time `STAGE seconds`.

| run | stage s | PID peak MiB | samples within 1 MiB of the peak | late samples | late max MiB |
|---|---|---|---|---|---|
| B1 | 38.786 | 8257 | 140 | 44 | 4725 |
| B2 | 175.256 | 28885 | 632 | 134 | 15611 |
| B3 | 166.465 | 28885 | 622 | 135 | 15611 |
