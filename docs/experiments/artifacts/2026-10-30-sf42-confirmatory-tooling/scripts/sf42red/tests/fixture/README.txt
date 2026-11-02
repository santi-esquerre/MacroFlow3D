Tiny real output of the consumer runner (labels.npy and the wall-clock timings.json removed), used by test_fixture.py:
  lagrangian_ensemble --medium b --grid 128,16,16 --ell 0.25 --eps 1 --seed0 420501 --nr 2 --np-side 10 --twin on
      --pollock-bridge off --planes 2.25,3.25,4.25,5.25,6.25 --out <dir>
(build wsl-debug of the SF-42 base 3164e4f; realization 1 has 7 outlet exits inside the snapshots, the first one in (4.25, 4.375]).
stats_twin/ is the uniform-weight series, in the format of the Par2 per-realization time series.
