# Completeness instrument (RR3): device_used against exact bytes

Source: `sf40_campaign/mem/{d512,d3328_i,d3328_ii}.log` (`MEM_PHASE` lines) and `sf40_campaign/mem/results.txt`
(`PROBE ... device_used_bytes`). The stage-start `device_used` is not printed; the context probe value 514850816 B
is used as `start` (same process creation sequence, nothing else allocated).

- `pre` = device_used - start - exact_bytes (as pre-registered).
- `adj` = device_used - start - (exact_bytes - cufft_work) (excludes the cuFFT plan work areas, which no longer
  exist when `LongBoxStage.cu` samples `device_used` right after the SF-18 / SF-28 call returns; S2 and T have
  cufft_work = 0, so pre = adj there).
- `rounded excess` = sum over allocations rounded up to 2 MiB minus exact (`C_INFO ... sum_rounded_2MiB_per_alloc`).

## d512

| phase | exact B | cufft_work B | device_used B | pre B | adj B | adj MiB | rounded excess B |
|---|---:|---:|---:|---:|---:|---:|---:|
| F | 3229616136 | 1077936128 | 2668625920 | -1075841032 | 2095096 | 2.00 | 4192248 |
| S1 | 5381292032 | 2155872256 | 3742367744 | -2153775104 | 2097152 | 2.00 | 0 |
| S2 | 8138281074 | 0 | 8655994880 | 2862990 | 2862990 | 2.73 | 59486094 |
| S3 | 6729240552 | 2155872256 | 5090836480 | -2153254888 | 2617368 | 2.50 | 520216 |
| T | 2464157784 | 0 | 2983198720 | 4190120 | 4190120 | 4.00 | 10481576 |

## d3328_i

| phase | exact B | cufft_work B | device_used B | pre B | adj B | adj MiB | rounded excess B |
|---|---:|---:|---:|---:|---:|---:|---:|
| F | 11782326280 | 3928227840 | 8372879360 | -3924297736 | 3930104 | 3.75 | 7862264 |
| S1 | 19636420608 | 7856455680 | 12298747904 | -7852523520 | 3932160 | 3.75 | 3407872 |
| S2 | 29765452393 | 0 | 30298603520 | 18300311 | 18300311 | 17.45 | 56049047 |
| S3 | 23775677416 | 7856455680 | 16440623104 | -7849905128 | 6550552 | 6.25 | 3929112 |
| T | 8060101832 | 0 | 8578400256 | 3447608 | 3447608 | 3.29 | 18127672 |

## d3328_ii

| phase | exact B | cufft_work B | device_used B | pre B | adj B | adj MiB | rounded excess B |
|---|---:|---:|---:|---:|---:|---:|---:|
| F | 11782326280 | 3928227840 | 8372879360 | -3924297736 | 3930104 | 3.75 | 7862264 |
| S1 | 19636420608 | 7856455680 | 12298747904 | -7852523520 | 3932160 | 3.75 | 3407872 |
| S2 | 29765452393 | 0 | 30298603520 | 18300311 | 18300311 | 17.45 | 56049047 |
| S3 | 23572512768 | 7856455680 | 16235102208 | -7852261376 | 4194304 | 4.00 | 3670016 |
| T | 7856377176 | 0 | 8372879360 | 1651368 | 1651368 | 1.57 | 18428584 |

