# SF-43 Gate 5 coverage (sf43cov)

Criterion (G5.3): one-sided miss rate with 95% one-sided Clopper-Pearson UCL at most 1.25 x tail, >= 175 expected misses at the smallest tail. Cells complete: 76 of 76; core-seconds 75609; seed 20261103; plan hash `816619bcc2cf771e`; B = 33599; family m = 8; reference t.

Tails used: hall and boott at the Holm levels 0.00625, 0.007143, 0.008333, 0.01, 0.0125, 0.01667, 0.025, 0.05; boott_half at their halves.

## Qualification (normal, stress, stress-; both sides; every tail used)

| procedure | n=100 | n=150 | n=200 | n=250 | n=300 | n=400 | n=500 |
|---|---|---|---|---|---|---|---|
| hall | no (1.59) | no (1.31) | QUALIFIED (1.17) | QUALIFIED (1.19) | QUALIFIED (1.19) | QUALIFIED (1.17) | QUALIFIED (1.12) |
| boott | no (1.82) | no (1.39) | QUALIFIED (1.20) | QUALIFIED (1.19) | QUALIFIED (1.18) | QUALIFIED (1.18) | QUALIFIED (1.12) |
| boott_half | no (2.36) | no (1.57) | no (1.39) | no (1.35) | no (1.30) | no (1.26) | QUALIFIED (1.18) |

Entry: verdict (largest UCL / tail over the tails used). First qualified procedure by n (the selection rule): 100: None, 150: None, 200: hall, 250: hall, 300: hall, 400: hall, 500: hall

## False-fail probability of the criterion for an exactly nominal procedure

- R = 28000: 0.083 per (cell, distribution, row); at most 0.41 over the six rows of a procedure at one n if independent.
- R = 56000: 0.083 per (cell, distribution, row); at most 0.41 over the six rows of a procedure at one n if independent.

## Failing rows (procedures hall and boott, judged cells)

- boott/hi n=100: 10 failing (tail, dist): 0.003125 stress 2.36; 0.003571 stress 2.27; 0.004167 stress 2.16; 0.005 stress 2.01
- boott/hi n=150: 8 failing (tail, dist): 0.003125 stress 1.52; 0.003571 normal 1.26; 0.003571 stress 1.49; 0.004167 normal 1.29
- boott/hi n=200: 3 failing (tail, dist): 0.003125 stress 1.39; 0.003571 normal 1.25; 0.003571 stress 1.31
- boott/hi n=250: 4 failing (tail, dist): 0.003125 normal 1.26; 0.003125 stress 1.35; 0.003571 stress 1.32; 0.004167 stress 1.28
- boott/hi n=400: 1 failing (tail, dist): 0.003571 normal 1.25
- boott/lo n=100: 9 failing (tail, dist): 0.003125 stress- 2.34; 0.003571 stress- 2.18; 0.004167 stress- 2.04; 0.005 stress- 1.93
- boott/lo n=150: 8 failing (tail, dist): 0.003125 stress- 1.57; 0.003571 stress- 1.52; 0.004167 stress- 1.51; 0.005 stress- 1.46
- boott/lo n=200: 3 failing (tail, dist): 0.003125 stress- 1.33; 0.003571 stress- 1.31; 0.004167 stress- 1.30
- boott/lo n=300: 2 failing (tail, dist): 0.003125 normal 1.30; 0.003571 normal 1.25
- boott/lo n=400: 1 failing (tail, dist): 0.003125 stress- 1.26
- hall/hi n=100: 9 failing (tail, dist): 0.003125 stress 1.95; 0.003571 stress 1.89; 0.004167 stress 1.78; 0.005 stress 1.68
- hall/hi n=150: 5 failing (tail, dist): 0.003125 normal 1.25; 0.003125 stress 1.33; 0.003571 normal 1.26; 0.003571 stress 1.31
- hall/hi n=200: 3 failing (tail, dist): 0.003125 normal 1.25; 0.003125 stress 1.28; 0.003571 normal 1.26
- hall/hi n=250: 4 failing (tail, dist): 0.003125 normal 1.26; 0.003125 stress 1.28; 0.003571 normal 1.28; 0.003571 stress 1.27
- hall/lo n=100: 8 failing (tail, dist): 0.003125 stress- 1.86; 0.003571 stress- 1.85; 0.004167 stress- 1.73; 0.005 stress- 1.64
- hall/lo n=150: 5 failing (tail, dist): 0.003125 stress- 1.38; 0.003571 stress- 1.35; 0.004167 stress- 1.32; 0.005 stress- 1.34
- hall/lo n=300: 2 failing (tail, dist): 0.003125 normal 1.27; 0.003571 normal 1.25

## Reported, non-gating: heavier stress (skewness 2.73, excess kurtosis 7.59) and its mirror

- n=150 mix:2.73:7.59 hall/hi: max rate/tail 0.92, max UCL/tail 0.95
- n=150 mix:2.73:7.59 hall/lo: max rate/tail 0.94, max UCL/tail 0.97
- n=150 mix:2.73:7.59 boott/hi: max rate/tail 1.14, max UCL/tail 1.28
- n=150 mix:2.73:7.59 boott/lo: max rate/tail 0.93, max UCL/tail 0.96
- n=200 mix:2.73:7.59 hall/hi: max rate/tail 0.92, max UCL/tail 0.95
- n=200 mix:2.73:7.59 hall/lo: max rate/tail 0.96, max UCL/tail 0.99
- n=200 mix:2.73:7.59 boott/hi: max rate/tail 0.89, max UCL/tail 0.93
- n=200 mix:2.73:7.59 boott/lo: max rate/tail 0.95, max UCL/tail 0.98
- n=300 mix:2.73:7.59 hall/hi: max rate/tail 0.97, max UCL/tail 1.00
- n=300 mix:2.73:7.59 hall/lo: max rate/tail 0.95, max UCL/tail 0.99
- n=300 mix:2.73:7.59 boott/hi: max rate/tail 0.94, max UCL/tail 0.98
- n=300 mix:2.73:7.59 boott/lo: max rate/tail 0.95, max UCL/tail 0.98
- n=400 mix:2.73:7.59 hall/hi: max rate/tail 0.98, max UCL/tail 1.02
- n=400 mix:2.73:7.59 hall/lo: max rate/tail 0.98, max UCL/tail 1.01
- n=400 mix:2.73:7.59 boott/hi: max rate/tail 0.97, max UCL/tail 1.01
- n=400 mix:2.73:7.59 boott/lo: max rate/tail 0.97, max UCL/tail 1.01
- n=150 mix:2.73:7.59- hall/hi: max rate/tail 0.93, max UCL/tail 0.96
- n=150 mix:2.73:7.59- hall/lo: max rate/tail 0.92, max UCL/tail 0.95
- n=150 mix:2.73:7.59- boott/hi: max rate/tail 0.93, max UCL/tail 0.96
- n=150 mix:2.73:7.59- boott/lo: max rate/tail 1.08, max UCL/tail 1.22
- n=200 mix:2.73:7.59- hall/hi: max rate/tail 0.95, max UCL/tail 0.98
- n=200 mix:2.73:7.59- hall/lo: max rate/tail 0.94, max UCL/tail 0.97
- n=200 mix:2.73:7.59- boott/hi: max rate/tail 0.94, max UCL/tail 0.97
- n=200 mix:2.73:7.59- boott/lo: max rate/tail 0.91, max UCL/tail 0.94
- n=300 mix:2.73:7.59- hall/hi: max rate/tail 0.95, max UCL/tail 0.98
- n=300 mix:2.73:7.59- hall/lo: max rate/tail 0.96, max UCL/tail 0.99
- n=300 mix:2.73:7.59- boott/hi: max rate/tail 0.95, max UCL/tail 0.98
- n=300 mix:2.73:7.59- boott/lo: max rate/tail 0.94, max UCL/tail 0.98
- n=400 mix:2.73:7.59- hall/hi: max rate/tail 0.97, max UCL/tail 1.01
- n=400 mix:2.73:7.59- hall/lo: max rate/tail 0.98, max UCL/tail 1.02
- n=400 mix:2.73:7.59- boott/hi: max rate/tail 0.97, max UCL/tail 1.01
- n=400 mix:2.73:7.59- boott/lo: max rate/tail 0.97, max UCL/tail 1.00
