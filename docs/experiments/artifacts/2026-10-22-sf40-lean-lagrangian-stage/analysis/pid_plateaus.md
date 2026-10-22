# PID time series (200 ms) of the D runs against exact bytes + context

Source: `sf40_campaign/mem/<run>.apps.csv` filtered to the PID of `<run>.pid_tree` (= `MEM_GEOM pid=`).
Context C = 487 MiB (`C_CONTEXT`, `mem/results.txt`). expected(phase) = exact_bytes / 2^20 + C.
The series is split structurally (windows below); per phase the sample nearest to expected inside its window is reported.

## d512

samples 1409, PID peak 8249 MiB.

| phase | window | exact MiB | expected = exact + C | max sample in window | nearest sample (#, first ts) | delta MiB | within 64 MiB |
|---|---|---:|---:|---:|---|---:|---|
| F | before the host gap | 3080.00 | 3567.00 | 2541 | 2541 (x4, 18:04:39.251) | -1026.00 | no |
| S1 | gap .. S2 plateau | 5132.00 | 5619.00 | 3565 | 3565 (x5, 18:04:44.678) | -2054.00 | no |
| S2 | plateau | 7761.27 | 8248.27 | 8249 | 8249 (x141, 18:04:45.684) | +0.73 | yes |
| S3 | after the plateau | 6417.50 | 6904.50 | - | 7223 (x1, 18:05:14.130) | +318.50 | no |
| T | after the plateau | 2350.00 | 2837.00 | - | 2843 (x1, 18:05:21.410) | +6.00 | yes |

Highest distinct sampled levels before the S2 plateau: 3565, 2541, 2539 MiB; after it: 7223, 2977, 2843 MiB.
S2 plateau: 8249 MiB x 141 samples (18:04:45.684 .. ), S2 expected 8248.27 MiB, delta +0.73 MiB.
Last run of >= 30 samples (tracker phase): 2977 MiB x 1194 samples, T expected 2837.00 MiB, delta +140.00 MiB.

Run-length encoding (value MiB x samples @ first timestamp):

```
51 x1@18:04:38.445  351 x1@18:04:38.647  417 x1@18:04:38.848  2539 x1@18:04:39.050  2541 x4@18:04:39.251  489 x22@18:04:40.055  1513 x1@18:04:44.477  3565 x5@18:04:44.678  8249 x141@18:04:45.684  7223 x1@18:05:14.130  1515 x14@18:05:14.332  2539 x3@18:05:17.148  2797 x8@18:05:17.778  749 x9@18:05:19.390  2797 x1@18:05:21.209  2843 x1@18:05:21.410  2977 x1194@18:05:21.611  615 x1@18:09:21.778
```

One 8m buffer = 8 * face_max / 2^20 = 1026.00 MiB; sampled levels within 1 MiB of (S2 plateau - 8m): 7223.

## d3328_i

samples 847, PID peak 28885 MiB.

| phase | window | exact MiB | expected = exact + C | max sample in window | nearest sample (#, first ts) | delta MiB | within 64 MiB |
|---|---|---:|---:|---:|---|---:|---|
| F | before the host gap | 11236.50 | 11723.50 | 11735 | 11735 (x1, 18:14:16.279) | +11.50 | yes |
| S1 | gap .. S2 plateau | 18726.75 | 19213.75 | 17623 | 17623 (x1, 18:14:39.527) | -1590.75 | no |
| S2 | plateau | 28386.55 | 28873.55 | 28885 | 28885 (x584, 18:14:40.162) | +11.45 | yes |
| S3 | after the plateau | 22674.25 | 23161.25 | - | 8313 (x48, 18:16:58.016) | -14848.25 | no |
| T | after the plateau | 7686.71 | 8173.71 | - | 8173 (x33, 18:16:42.007) | -0.71 | yes |

Highest distinct sampled levels before the S2 plateau: 17623, 11735, 11725 MiB; after it: 8313, 8173, 7987 MiB.
S2 plateau: 28885 MiB x 584 samples (18:14:40.162 .. ), S2 expected 28873.55 MiB, delta +11.45 MiB.
Last run of >= 30 samples (tracker phase): 8313 MiB x 48 samples, T expected 8173.71 MiB, delta +139.29 MiB.

Run-length encoding (value MiB x samples @ first timestamp):

```
303 x1@18:14:15.475  367 x1@18:14:15.676  431 x1@18:14:15.877  7979 x1@18:14:16.078  11735 x1@18:14:16.279  7981 x16@18:14:16.481  489 x78@18:14:19.696  4233 x5@18:14:35.539  11725 x14@18:14:36.710  17623 x1@18:14:39.527  28885 x584@18:14:40.162  4243 x6@18:16:38.150  7987 x12@18:16:39.518  8173 x33@18:16:42.007  685 x37@18:16:48.665  8173 x7@18:16:56.270  8313 x48@18:16:58.016  615 x1@18:17:07.668
```

One 8m buffer = 8 * face_max / 2^20 = 3753.75 MiB; sampled levels within 1 MiB of (S2 plateau - 8m): none.

## d3328_ii

samples 865, PID peak 28885 MiB.

| phase | window | exact MiB | expected = exact + C | max sample in window | nearest sample (#, first ts) | delta MiB | within 64 MiB |
|---|---|---:|---:|---:|---|---:|---|
| F | before the host gap | 11236.50 | 11723.50 | 11735 | 11735 (x1, 18:17:15.364) | +11.50 | yes |
| S1 | gap .. S2 plateau | 18726.75 | 19213.75 | 17623 | 17623 (x1, 18:17:38.876) | -1590.75 | no |
| S2 | plateau | 28386.55 | 28873.55 | 28885 | 28885 (x588, 18:17:39.321) | +11.45 | yes |
| S3 | after the plateau | 22480.50 | 22967.50 | - | 25131 (x1, 18:19:38.112) | +2163.50 | no |
| T | after the plateau | 7492.43 | 7979.43 | - | 7977 (x29, 18:19:43.419) | -2.43 | yes |

Highest distinct sampled levels before the S2 plateau: 17623, 17397, 11735 MiB; after it: 25131, 11731, 8117 MiB.
S2 plateau: 28885 MiB x 588 samples (18:17:39.321 .. ), S2 expected 28873.55 MiB, delta +11.45 MiB.
Last run of >= 30 samples (tracker phase): 8117 MiB x 51 samples, T expected 7979.43 MiB, delta +137.57 MiB.

Run-length encoding (value MiB x samples @ first timestamp):

```
51 x1@18:17:14.359  353 x1@18:17:14.560  419 x1@18:17:14.761  461 x1@18:17:14.962  7979 x1@18:17:15.163  11735 x1@18:17:15.364  7981 x17@18:17:15.565  489 x78@18:17:18.981  4233 x5@18:17:34.701  11725 x14@18:17:35.866  17397 x1@18:17:38.676  17623 x1@18:17:38.876  28885 x588@18:17:39.321  25131 x1@18:19:38.112  4243 x1@18:19:38.334  7987 x21@18:19:38.646  11731 x1@18:19:43.176  7977 x29@18:19:43.419  489 x45@18:19:49.243  7977 x5@18:19:58.517  8117 x51@18:19:59.968  615 x1@18:20:10.225
```

One 8m buffer = 8 * face_max / 2^20 = 3753.75 MiB; sampled levels within 1 MiB of (S2 plateau - 8m): 25131.

