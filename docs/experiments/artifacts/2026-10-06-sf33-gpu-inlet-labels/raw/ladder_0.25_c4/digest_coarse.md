### ladder_0.25_c4/N32.log

- STAGE_END field=gaussian eps=0.25 N=32 cand=i1o4 status=converged its=11 r_F=1.097e-14 r_out=4.407e-16 -> accepted
- LINEAR solves=11 its_total=218 its_max=58 its_median=13 t_linear_total=1.50s non_converged=0
- COARSE build count=11 K=4096 kl=639 ku=639 t_assembly mean=0.026s max=0.028s | t_lu mean=0.588s max=0.641s total=6.5s | t_cond mean=0.034s | rcond_est min=1.606e-05
- COARSE apply applications_total=229 t_apply_avg mean=6.024ms max=6.553ms | t_host_solve_avg mean=5.125ms max=5.610ms
- host max RSS (/usr/bin/time -v) = 690604 kB = 0.66 GiB

### ladder_0.25_c4/N64.log

- STAGE_END field=gaussian eps=0.25 N=64 cand=i1o4 status=linear_failure its=4 r_F=5.539e-02 r_out=1.640e-04 -> FAILED
- STAGE_END field=gaussian eps=0.125 N=64 cand=i1o4 status=converged its=9 r_F=1.979e-14 r_out=4.248e-16 -> accepted
- STAGE_END field=gaussian eps=0.25 N=64 cand=i1o4 status=linear_failure its=4 r_F=2.138e-02 r_out=5.026e-05 -> FAILED
- STAGE_END field=gaussian eps=0.1875 N=64 cand=i1o4 status=converged its=8 r_F=6.406e-14 r_out=6.735e-16 -> accepted
- STAGE_END field=gaussian eps=0.25 N=64 cand=i1o4 status=converged its=9 r_F=5.102e-14 r_out=1.036e-15 -> accepted
- LINEAR solves=34 its_total=3914 its_max=1100 its_median=36 t_linear_total=214.78s non_converged=2
- COARSE build count=34 K=16384 kl=1279 ku=1279 t_assembly mean=0.139s max=0.160s | t_lu mean=11.601s max=14.031s total=394.4s | t_cond mean=0.349s | rcond_est min=2.488e-06
- COARSE apply applications_total=3971 t_apply_avg mean=51.718ms max=83.157ms | t_host_solve_avg mean=49.608ms max=81.099ms
- host max RSS (/usr/bin/time -v) = 1168220 kB = 1.11 GiB

### ladder_0.25_c4/N128.log

- STAGE_END field=gaussian eps=0.25 N=128 cand=i1o4 status=linear_failure its=3 r_F=4.923e-01 r_out=1.010e-03 -> FAILED
- STAGE_END field=gaussian eps=0.125 N=128 cand=i1o4 status=converged its=9 r_F=7.938e-14 r_out=8.620e-16 -> accepted
- STAGE_END field=gaussian eps=0.25 N=128 cand=i1o4 status=linear_failure its=2 r_F=9.158e-01 r_out=1.008e-03 -> FAILED
- STAGE_END field=gaussian eps=0.1875 N=128 cand=i1o4 status=linear_failure its=3 r_F=3.575e-02 r_out=8.027e-05 -> FAILED
- STAGE_END field=gaussian eps=0.15625 N=128 cand=i1o4 status=linear_failure its=3 r_F=1.645e-02 r_out=1.876e-05 -> FAILED
- STAGE_END field=gaussian eps=0.140625 N=128 cand=i1o4 status=linear_failure its=3 r_F=8.119e-03 r_out=4.497e-06 -> FAILED
- STAGE_END field=gaussian eps=0.25 N=128 cand=i1o4 status=linear_failure its=2 r_F=9.158e-01 r_out=1.008e-03 (final attempt) -> FAILED
- LINEAR solves=25 its_total=4449 its_max=900 its_median=100 t_linear_total=1904.56s non_converged=6
- COARSE build count=25 K=65536 kl=2559 ku=2559 t_assembly mean=1.039s max=1.225s | t_lu mean=312.732s max=372.839s total=7818.3s | t_cond mean=3.508s | rcond_est min=3.477e-09
- COARSE apply applications_total=4503 t_apply_avg mean=397.636ms max=434.480ms | t_host_solve_avg mean=389.196ms max=426.151ms
- host max RSS (/usr/bin/time -v) = 5021912 kB = 4.79 GiB
