"""mcstat: statistics and error-budget library of the SF-42 confirmatory tooling (decisions G4, G5, G7 of
docs/decisions/2026-10-29-sf39-remediation-sf42-sf43.md).

stdlib + numpy + scipy only; identical results on the host stack (Python 3.11 / numpy 1.26 / scipy 1.11) and the local one
(3.13 / 2.5 / 1.18).  Run the tests from the directory that contains this package::

    python3 -W error -m unittest discover -s . -t .

Modules (each function documents its formula and the decision clause it implements):

* ``intervals``    per-realization slope estimators (G4); Student, bootstrap-t, BCa, Welch, paired intervals and one-sided limits
* ``budget``       robust lower limit and error budget (G5.4): ``StudentMean`` / ``BootstrapTMean``, ``LadderBound`` (E_h),
                   ``e_int``, ``e_w``, ``e_interaction``, ``RobustLimit`` (RL, p-value), ``intersection_union``
* ``gate5``        paired log-ratio equivalence (G5.5): estimate, verdict, TOST p-value, analytic power, sample size
* ``splitplot``    2^(4-1) split-plot attribution factorial (G7): design generator, analysis, aliases
* ``multiplicity`` Holm step-down (G5.6)
* ``sizing``       sample-size rules (G5.2) and the Gate 4 power model / joint allocation
* ``blinded``      blinded variance re-estimation (G5.2): outputs standard deviations only
* ``costmodel``    GPU-second cost model calibrated on SF-39 walls, with projection of a study plan
* ``coverage``     coverage simulator (G5.3): per-tail miss rates of the sampling limits, the budget bounds, the whole robust-sign
                   pipeline and the Gate 5 TOST under the normal and the stress family; ``python3 -m mcstat.coverage {stress,plan,
                   bench,run,summarize}``; ``run_coverage.sh <out_dir> [--workers N]`` is the detached host job.  Helpers:
                   ``covdist`` (distributions, ``stress_moments.json`` from the SF-39 tables), ``covproc`` (procedures, tail levels)

Vectorization: samples are on the LAST axis, leading axes are independent replicates, so the interval / limit functions can be
called on thousands of replicates at once (coverage simulator, SF-42 node N6b).  A one-sided limit takes ``tail`` as a scalar or a
1-D array (the output then has a trailing tail axis).  Bootstrap work requires ``rng=`` (a ``numpy.random.Generator``) or ``seed=``.

API at a glance (``x``: (..., n) samples; every function documents its formula and decision clause)::

    intervals.ols_slope(d, y)                         alpha_T_from_slope(slope)         alpha_L_from_slope(slope, U)
    intervals.mean_crossing_speed(d, t_mean, method)  orthogonal_variance(var2, var3, cov23, drift2, drift3)
    intervals.student_interval(x, level, sided)       student_limit(x, tail, side)      paired_interval(x, y, level, sided)
    intervals.welch(a, b, level, sided, c1, c2)       welch_from_summary(m1, s1, n1, m2, s2, n2, level, sided)
    intervals.welch_contrast(m1, s1, n1, m2, s2, n2, c1, c2, level, sided)             welch_limit(a, b, tail, side)
    intervals.bootstrap_t(x, B, rng, seed, share_draws) -> BootT   bootstrap_t_limit(bt, tail, side)   bootstrap_t_interval(...)
    intervals.bca_interval(x, level, sided, B, seed, statistic)    margin_multiple(lo, hi, m)
    budget.StudentMean(x) / BootstrapTMean(x, B, rng, seed)        .lower(tails) .upper(tails)   (sampling limits L_e)
    budget.LadderBound(levels, resolutions, ip, p_min, p_max, safety, fallback_factor, B, rng, seed)   (E_h; .bound(t), .details(t))
    budget.e_int(h16, h8) / e_w(wide, prod) / e_interaction(wide_f, wide_c, prod_f, prod_c)            (ContrastBound; .bound(t))
    budget.RobustLimit(sampling, components, lattice)   .lower(a)  .component_bounds(a)  .p_value()  .report(a)
    budget.intersection_union(*p)                     is_non_increasing_in_tail(bound_fn, tails)
    gate5.gate5_paired(L, P, level, margin_ratio)     tost_power(n, sigma_e, delta, theta, alpha)    gate5_power(n, cv_L, cv_P, corr, alpha, ratio)
    gate5.gate5_sample_size(cv_L, cv_P, corr, alpha, power, ratio)   gate5_conservative_inputs(sd_L, sd_P, mean_L, mean_P, n_pilot, corr)
    splitplot.design(seeds_plus, seeds_minus)         analyze(data, level, log_bootstrap, seed)      aliases(word)   cells_of(g)
    multiplicity.holm(p, alpha)
    sizing.sd_upper / mean_lower / corr_lower         n_for_halfwidth / n_absolute_target / n_relative_target / clamp_resize
    sizing.gate4_power(alpha, a, n_prod, sd_prod, comps)   allocate_gate4(alpha_plan, a, sd_prod, cost_prod, comps, power)
    blinded.blinded_sds(arms)   blinded_reestimate(arms, n_orig, half_widths)   blinded_gate5(L, P, plan_mean_L, plan_mean_P, alpha, n_orig)
    costmodel.sf39_points(A) -> (points, excluded)    CostModel(points).predict(arm) / .project(plan) / .with_points(new) / .summary()
    coverage.cp_upper(k, R)   coverage.make_cell / profile_full / init_plan / run / summarize / bench / project
    covproc.holm_levels() / sampling_tails() / component_tails()      covdist.get_dist("normal" | "stress" | "stress-")
"""
