/**
 * @file SlabNewtonKrylov.cu
 * @brief SF-33 N2: Newton-Krylov with line search and amplitude continuation on the inlet slab
 *        (see SlabNewtonKrylov.cuh for the contract and the prototype correspondence).
 */

#include "SlabNewtonKrylov.cuh"

#include "../../../runtime/cuda_check.cuh"

#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <deque>
#include <stdexcept>

namespace macroflow3d {
namespace streamfunctions {
namespace inlet_slab {

namespace {

std::string fmt(const char* f, ...) __attribute__((format(printf, 1, 2)));
std::string fmt(const char* f, ...) {
    char buf[1024];
    va_list ap;
    va_start(ap, f);
    std::vsnprintf(buf, sizeof(buf), f, ap);
    va_end(ap);
    return std::string(buf);
}

double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

real merit_of(real r_F, real r_out) {
    return std::sqrt(r_F * r_F + r_out * r_out);
}

void check_forcing_config(const SlabNewtonConfig& cfg) {
    if (cfg.forcing == SlabForcing::fixed)
        return;
    if (cfg.forcing != SlabForcing::ew)
        throw std::invalid_argument("SlabNewtonKrylov::solve: unknown forcing policy");
    const SlabEwConfig& e = cfg.ew;
    if (!(e.gamma > 0.0 && e.gamma <= 1.0) || !(e.alpha > 1.0 && e.alpha <= 2.0) ||
        !(e.eta0 > 0.0 && e.eta0 < 1.0) || !(e.eta_max > 0.0 && e.eta_max < 1.0) ||
        !(cfg.gmres.tol > 0.0) || cfg.gmres.tol > e.eta_max)
        throw std::invalid_argument("SlabNewtonKrylov::solve: Eisenstat-Walker parameters out of "
                                    "range (0 < gamma <= 1, 1 < alpha <= 2, 0 < eta0, eta_max < 1, "
                                    "0 < gmres.tol <= eta_max)");
}

/// Forcing term of Newton step k (0-based within one solve() call); see SlabNewtonKrylov.cuh.
/// m = merit at the start of step k, m_prev / eta_prev = merit at the start of / forcing term of
/// step k - 1 (unused for k = 0).
real forcing_term(const SlabNewtonConfig& cfg, int k, real m, real m_prev, real eta_prev) {
    if (cfg.forcing == SlabForcing::fixed)
        return cfg.gmres.tol;
    const SlabEwConfig& e = cfg.ew;
    real eta = e.eta0;
    if (k >= 1) {
        const real ratio = m / m_prev;
        eta = e.gamma * std::pow(ratio, e.alpha);
        const real safeguard = e.gamma * std::pow(eta_prev, e.alpha);
        if (safeguard > 0.1)
            eta = std::fmax(eta, safeguard);
    }
    eta = std::fmin(eta, e.eta_max);
    eta = std::fmax(eta, cfg.gmres.tol); // eta_min = lin_tol
    // oversolving guard: no tighter than needed to bring the merit to the Newton tolerance
    eta = std::fmax(eta, 0.5 * cfg.tol / m);
    return eta;
}

} // namespace

void slab_stdout_logger(const std::string& line) {
    std::fputs(line.c_str(), stdout);
    std::fputc('\n', stdout);
    std::fflush(stdout);
}

void SlabNewtonKrylov::prepare(CudaContext& ctx, const InletSlabGrid& grid, int restart,
                               int max_newton, int max_gmres_iterations) {
    require_valid_grid(grid, "SlabNewtonKrylov::prepare");
    if (max_newton < 1)
        throw std::invalid_argument("SlabNewtonKrylov::prepare: max_newton must be >= 1");
    grid_ = grid;
    rws_.prepare(grid);
    jws_.prepare(grid);
    prec_.prepare(ctx, grid);
    gmres_.prepare(grid, restart, max_gmres_iterations);
    U1_.resize(grid.full_size());
    U2_.resize(grid.full_size());
    for (DeviceBuffer<real>* b : {&E_, &Et_, &xt_, &dx_, &rhs_, &xconv_})
        b->resize(grid.unknown_size());
    max_newton_reserved_ = max_newton;
    ctx.synchronize();
}

void SlabNewtonKrylov::prepare_coarse(CudaContext& ctx, int profiles, SlabCoarseAssembly assembly,
                                      SlabCoarseFactor factor) {
    if (!prepared())
        throw std::logic_error("SlabNewtonKrylov::prepare_coarse: call prepare() first");
    coarse_.prepare(ctx, grid_, profiles, assembly, factor);
}

std::size_t SlabNewtonKrylov::allocated_bytes() const {
    std::size_t b = rws_.allocated_bytes() + jws_.allocated_bytes() + prec_.allocated_bytes() +
                    gmres_.allocated_bytes() + coarse_.allocated_bytes();
    for (const DeviceBuffer<real>* v : {&U1_, &U2_, &E_, &Et_, &xt_, &dx_, &rhs_, &xconv_})
        b += v->capacity() * sizeof(real);
    return b;
}

std::vector<const void*> SlabNewtonKrylov::buffer_pointers() const {
    std::vector<const void*> p = {U1_.data(),           U2_.data(),      E_.data(),   Et_.data(),
                                  xt_.data(),           dx_.data(),      rhs_.data(), xconv_.data(),
                                  rws_.partials_data(), rws_.sums_data()};
    for (const auto& v :
         {prec_.buffer_pointers(), gmres_.buffer_pointers(), coarse_.buffer_pointers()})
        p.insert(p.end(), v.begin(), v.end());
    return p;
}

SlabResidualNorms SlabNewtonKrylov::residual_norms(CudaContext& ctx, const SlabStageInputs& inputs,
                                                   DeviceSpan<const real> x) {
    if (!prepared())
        throw std::logic_error("SlabNewtonKrylov: not prepared");
    assemble_full_planes(ctx, grid_, x, inputs, span(U1_), span(U2_));
    SlabResidualNorms nr;
    evaluate_residual(ctx, grid_, inputs, cspan(U1_), cspan(U2_), span(E_), rws_, &nr);
    return nr;
}

SlabNewtonReport SlabNewtonKrylov::solve(CudaContext& ctx, const SlabStageInputs& inputs,
                                         DeviceSpan<real> x, const std::string& label,
                                         const SlabNewtonConfig& cfg, const SlabLogger& log) {
    const auto t0 = std::chrono::steady_clock::now();
    if (!prepared())
        throw std::logic_error("SlabNewtonKrylov::solve: not prepared");
    if (x.size() != grid_.unknown_size())
        throw std::invalid_argument("SlabNewtonKrylov::solve: x has the wrong size");
    if (cfg.max_iterations > max_newton_reserved_)
        throw std::invalid_argument("SlabNewtonKrylov::solve: max_iterations exceeds the prepared "
                                    "reservation");
    if (!gmres_.prepared_for(grid_, cfg.gmres.restart))
        throw std::invalid_argument("SlabNewtonKrylov::solve: GMRES restart exceeds the prepared "
                                    "basis");
    inputs.check(grid_);
    check_forcing_config(cfg);
    if (cfg.coarse != SlabCoarseMode::off && !coarse_.prepared_for(grid_))
        throw std::invalid_argument("SlabNewtonKrylov::solve: coarse correction requested but "
                                    "prepare_coarse() was not called");
    if (cfg.psitc.enabled &&
        (!(cfg.psitc.mu0 >= 0.0) || !(cfg.psitc.mu_max >= cfg.psitc.mu0) ||
         !std::isfinite(cfg.psitc.mu_max) || cfg.psitc.max_retries < 0 ||
         !(cfg.psitc.retry_factor > 1.0) || !std::isfinite(cfg.psitc.retry_factor) ||
         !(cfg.psitc.h_ref >= 0.0) || !std::isfinite(cfg.psitc.h_ref)))
        throw std::invalid_argument("SlabNewtonKrylov::solve: Psi-tc parameters out of range "
                                    "(0 <= mu0 <= mu_max < inf, max_retries >= 0, "
                                    "retry_factor > 1, 0 <= h_ref < inf)");

    SlabNewtonReport rep;
    rep.forcing = cfg.forcing;
    rep.psitc = cfg.psitc.enabled;
    rep.hist_r_F.reserve(static_cast<std::size_t>(max_newton_reserved_) + 1);
    rep.hist_r_out.reserve(static_cast<std::size_t>(max_newton_reserved_) + 1);
    rep.steps.reserve(static_cast<std::size_t>(max_newton_reserved_));
    SlabReduction& red = gmres_.reduction();
    const char* lab = label.c_str();

    const DeviceSpan<const real> xc(x.data(), x.size());
    SlabResidualNorms nr = residual_norms(ctx, inputs, xc);
    real r_F = nr.r_F, r_out = nr.r_out;
    real m = merit_of(r_F, r_out);
    rep.hist_r_F.push_back(r_F);
    rep.hist_r_out.push_back(r_out);
    log(fmt("  NEWTON %s it=%2d r_F=%.3e r_out=%.3e", lab, 0, r_F, r_out));
    auto done = [&](SlabSolveStatus st) {
        // Documented sync: x (and every pending copy) is complete when solve() returns.
        MACROFLOW3D_CUDA_CHECK(cudaStreamSynchronize(ctx.cuda_stream()));
        rep.status = st;
        rep.r_F = r_F;
        rep.r_out = r_out;
        rep.seconds = seconds_since(t0);
        return rep;
    };
    if (!std::isfinite(m))
        return done(SlabSolveStatus::nan_inf);
    auto converged = [&]() { return r_F <= cfg.tol && r_out <= cfg.tol; };

    const bool psitc = cfg.psitc.enabled;
    real cur_mu = 0.0; // shift of the operator currently handed to GMRES (0: plain J, bitwise N7a)
    const SlabGmres::Operator opA = [&](DeviceSpan<const real> in, DeviceSpan<real> out) {
        jws_.apply(ctx, grid_, in, out);
        if (cur_mu != 0.0)
            slab_add_pseudo_time_shift(ctx, grid_, inputs, cur_mu, in, out);
    };
    const SlabGmres::Operator opPA = [&](DeviceSpan<const real> in, DeviceSpan<real> out) {
        prec_.apply(ctx, grid_, in, out);
    };
    const bool use_coarse = cfg.coarse != SlabCoarseMode::off;
    const SlabGmres::Operator opM = [&](DeviceSpan<const real> in, DeviceSpan<real> out) {
        if (use_coarse)
            coarse_.apply(ctx, grid_, cfg.coarse, opA, opPA, in, out);
        else
            prec_.apply(ctx, grid_, in, out);
    };

    SlabSolveStatus status = SlabSolveStatus::maxit;
    real m_prev = 0.0, eta_prev = 0.0;
    const real m_start = m; // SER reference: merit of the start state of this solve() call
    // SF-33 C4: grid-scaled SER reference shift (== mu0 exactly when h == h_ref or h_ref == 0)
    const real mu0_eff = psitc_effective_mu0(cfg.psitc, grid_.h);
    SlabGmresConfig gcfg = cfg.gmres; // tol overwritten per step with the forcing term
    for (int it = 1; it <= cfg.max_iterations; ++it) {
        if (converged()) {
            status = SlabSolveStatus::converged;
            break;
        }
        // base state of this step (U1_, U2_ = full arrays of x)
        assemble_full_planes(ctx, grid_, xc, inputs, span(U1_), span(U2_));
        jws_.prepare_base(ctx, grid_, inputs, cspan(U1_), cspan(U2_));
        slab_scale_copy(ctx, -1.0, cspan(E_), span(rhs_));
        const real eta = forcing_term(cfg, it - 1, m, m_prev, eta_prev);
        gcfg.tol = eta;
        real mu = 0.0;
        if (psitc) // switched evolution relaxation, clamped to [0, mu_max]
            mu = std::fmin(std::fmax(mu0_eff * (m / m_start), 0.0), cfg.psitc.mu_max);
        SlabNewtonStepRecord step;
        step.eta = eta;
        step.mu_ser = mu;
        rep.its = it;

        real lam = 1.0;
        bool accepted = false, lin_failed = false;
        real rFn = 0.0, ron = 0.0, mn = 0.0;
        for (int attempt = 0;; ++attempt) {
            if (attempt > 0) // the line search overwrote U1_, U2_ with trial states
                assemble_full_planes(ctx, grid_, xc, inputs, span(U1_), span(U2_));
            const SlabPrecFactorReport fr =
                prec_.factor(ctx, grid_, inputs, cspan(U1_), cspan(U2_), mu);
            cur_mu = mu;
            step.mu = mu;
            step.t_fact += fr.seconds;
            step.prec_singular_modes = fr.singular_modes;
            bool coarse_ok = true;
            if (use_coarse && fr.singular_modes == 0) {
                // SF-33 N7c probe: Galerkin coarse matrix of the SAME shifted operator
                const SlabCoarseBuildReport cr = coarse_.build(ctx, grid_, opA);
                coarse_ok = cr.zero_pivots == 0;
                step.t_fact += cr.t_assembly + cr.t_lu + cr.t_cond;
                log(fmt("    COARSE build mode=%s profiles=%d K=%d t_assembly=%.3fs t_lu=%.3fs "
                        "t_cond=%.3fs norm1=%.3e inv_norm1_est=%.3e rcond_est=%.3e "
                        "min|U_kk|=%.3e max|U_kk|=%.3e zero_pivots=%d mu=%.3e%s",
                        to_string(cfg.coarse), cr.profiles, cr.K, cr.t_assembly, cr.t_lu, cr.t_cond,
                        cr.norm1, cr.inv_norm1_est, cr.rcond_est, cr.min_abs_u, cr.max_abs_u,
                        cr.zero_pivots, mu,
                        cr.assembly == SlabCoarseAssembly::direct &&
                                cr.factor == SlabCoarseFactor::dense
                            ? ""
                            : fmt(" assembly=%s factor=%s applications=%d color_period=%d "
                                  "kl=%d ku=%d",
                                  to_string(cr.assembly), to_string(cr.factor), cr.applications,
                                  cr.color_period, cr.kl, cr.ku)
                                  .c_str()));
            }
            const auto tl0 = std::chrono::steady_clock::now();
            if (fr.singular_modes == 0 && coarse_ok) {
                step.linear = gmres_.solve(ctx, opA, opM, cspan(rhs_), span(dx_), gcfg);
            } else {
                step.linear = SlabGmresReport();
                step.linear.status = SlabLinearStatus::not_run;
            }
            const double t_lin = seconds_since(tl0);
            step.t_lin += t_lin;
            if (use_coarse && coarse_.applications() > 0)
                log(fmt("    COARSE apply applications=%d t_apply_avg=%.3fms "
                        "t_host_solve_avg=%.3fms",
                        coarse_.applications(),
                        1e3 * coarse_.apply_seconds() / coarse_.applications(),
                        1e3 * coarse_.host_solve_seconds() / coarse_.applications()));
            const SlabGmresReport& L = step.linear;
            step.linear_iterations_all += L.iterations;
            step.linear_its_solves.push_back(L.iterations);
            rep.linear_iterations_total += L.iterations;
            if (L.iterations > rep.linear_iterations_max)
                rep.linear_iterations_max = L.iterations;
            rep.last_linear_status = L.status;
            log(fmt("    LINEAR gmres+%s its=%d rel=%.1e t=%.2fs cycles=%d reorth=%d rec=%.1e "
                    "status=%s t_fact=%.2fs singular_modes=%d%s",
                    cfg.prec_name.c_str(), L.iterations, L.rel_residual, t_lin, L.cycles,
                    L.reorthogonalizations, L.rel_recurrence, to_string(L.status), fr.seconds,
                    fr.singular_modes, psitc ? fmt(" mu=%.3e", mu).c_str() : ""));
            if (L.status != SlabLinearStatus::converged) {
                log(fmt("  NEWTON %s it=%2d linear_failure: GMRES status=%s rel=%.1e lits=%d "
                        "prec singular modes=%d (step not taken) eta=%.2e%s",
                        lab, it, to_string(L.status), L.rel_residual, L.iterations,
                        fr.singular_modes, eta, psitc ? fmt(" mu=%.3e", mu).c_str() : ""));
                lin_failed = true;
                break;
            }
            step.dx_max = red.maxabs_host(ctx, cspan(dx_));
            step.dx_l2 = red.nrm2_host(ctx, cspan(dx_));

            lam = 1.0;
            while (lam >= cfg.lambda_min) {
                slab_axpby(ctx, 1.0, xc, lam, cspan(dx_), span(xt_));
                assemble_full_planes(ctx, grid_, cspan(xt_), inputs, span(U1_), span(U2_));
                SlabResidualNorms tn;
                evaluate_residual(ctx, grid_, inputs, cspan(U1_), cspan(U2_), span(Et_), rws_, &tn);
                rFn = tn.r_F;
                ron = tn.r_out;
                mn = merit_of(rFn, ron);
                if (std::isfinite(mn) && mn < (1.0 - cfg.armijo * lam) * m) {
                    accepted = true;
                    break;
                }
                lam *= 0.5;
            }
            if (accepted)
                break;
            // Psi-tc: on line-search failure raise mu (x retry_factor, clamped to mu_max) and
            // re-solve the step, at most max_retries times, before declaring linesearch-fail.
            const real mu_next =
                psitc ? std::fmin(mu * cfg.psitc.retry_factor, cfg.psitc.mu_max) : mu;
            if (!psitc || attempt >= cfg.psitc.max_retries || !(mu_next > mu))
                break;
            log(fmt("  NEWTON %s it=%2d line search failed (min lambda 1/1024): merit %.3e -> "
                    "%.3e at mu=%.3e; Psi-tc retry %d/%d with mu=%.3e",
                    lab, it, m, mn, mu, attempt + 1, cfg.psitc.max_retries, mu_next));
            mu = mu_next;
            step.mu_retries.push_back(mu);
            ++rep.linesearch_retries;
        }
        cur_mu = 0.0;
        const SlabGmresReport& L = step.linear;
        if (lin_failed) {
            rep.steps.push_back(step);
            status = SlabSolveStatus::linear_failure;
            break;
        }
        if (!accepted) {
            rep.steps.push_back(step);
            log(fmt("  NEWTON %s it=%2d line search failed (min lambda 1/1024): merit %.3e -> "
                    "%.3e  [gmres+%s rel=%.1e lits=%d |dx|max=%.2e |dx|2=%.2e] eta=%.2e%s",
                    lab, it, m, mn, cfg.prec_name.c_str(), L.rel_residual, L.iterations,
                    step.dx_max, step.dx_l2, eta,
                    psitc ? fmt(" mu=%.3e retries=%zu", mu, step.mu_retries.size()).c_str() : ""));
            status = SlabSolveStatus::linesearch_fail;
            break;
        }
        MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(x.data(), xt_.data(), x.size() * sizeof(real),
                                               cudaMemcpyDeviceToDevice, ctx.cuda_stream()));
        MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(E_.data(), Et_.data(), E_.size() * sizeof(real),
                                               cudaMemcpyDeviceToDevice, ctx.cuda_stream()));
        r_F = rFn;
        r_out = ron;
        m_prev = m;
        eta_prev = eta;
        m = mn;
        step.r_F = r_F;
        step.r_out = r_out;
        step.lambda = lam;
        rep.steps.push_back(step);
        rep.hist_r_F.push_back(r_F);
        rep.hist_r_out.push_back(r_out);
        log(fmt("  NEWTON %s it=%2d r_F=%.3e r_out=%.3e lambda=%.4g |dx|max=%.2e lin=gmres+%s "
                "rel=%.1e lits=%d t_lin=%.1fs |dx|2=%.2e t_fact=%.2fs eta=%.2e%s",
                lab, it, r_F, r_out, lam, step.dx_max, cfg.prec_name.c_str(), L.rel_residual,
                L.iterations, step.t_lin, step.dx_l2, step.t_fact, eta,
                psitc ? fmt(" mu=%.3e retries=%zu", step.mu, step.mu_retries.size()).c_str() : ""));
        const std::size_t nh = rep.hist_r_F.size();
        const std::size_t w = static_cast<std::size_t>(cfg.stagnation_window);
        if (w >= 2 && nh >= w) {
            const real m_first = merit_of(rep.hist_r_F[nh - w], rep.hist_r_out[nh - w]);
            if (m > cfg.stagnation_factor * m_first && !converged()) {
                status = SlabSolveStatus::stagnation;
                break;
            }
        }
    }
    if (converged())
        status = SlabSolveStatus::converged;
    return done(status);
}

SlabContinuationReport SlabNewtonKrylov::solve_with_continuation(
    CudaContext& ctx, real eps, const StageInputProvider& provider, DeviceSpan<real> x,
    const SlabContinuationConfig& ccfg, const SlabNewtonConfig& ncfg, const SlabLogger& log) {
    const auto t0 = std::chrono::steady_clock::now();
    if (!prepared())
        throw std::logic_error("SlabNewtonKrylov::solve_with_continuation: not prepared");
    if (x.size() != grid_.unknown_size())
        throw std::invalid_argument("SlabNewtonKrylov::solve_with_continuation: x has the wrong "
                                    "size");
    if (!(eps > 0.0) || !std::isfinite(eps))
        throw std::invalid_argument("SlabNewtonKrylov::solve_with_continuation: eps must be > 0");
    const int N = grid_.n;
    const char* field = ccfg.field.c_str();
    const char* cand = ccfg.cand.c_str();

    std::deque<real> todo;
    for (real e : ccfg.ladder)
        if (e < eps - 1e-12)
            todo.push_back(e);
    todo.push_back(eps);

    SlabContinuationReport rep;
    real e_conv = 0.0;
    bool have_conv = false;
    std::vector<std::string> taken;
    const DeviceSpan<real> xconv = span(xconv_);
    auto copy_vec = [&](DeviceSpan<real> dst, DeviceSpan<const real> src) {
        MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(dst.data(), src.data(), dst.size() * sizeof(real),
                                               cudaMemcpyDeviceToDevice, ctx.cuda_stream()));
    };
    auto label_of = [&](real e) { return fmt("%s:%g:%d:%s", field, e, N, cand); };
    // SF-33 C4: the effective SER reference shift is printed on every STAGE line (Psi-tc on only)
    const std::string psitc_tag =
        ncfg.psitc.enabled
            ? fmt(" psitc_mu0_eff=%.6e (mu0=%g h_ref=%.6g h=%.6g)",
                  psitc_effective_mu0(ncfg.psitc, grid_.h), ncfg.psitc.mu0, ncfg.psitc.h_ref,
                  grid_.h)
            : std::string();

    while (!todo.empty()) {
        const real e = todo.front();
        const SlabStageInputs& in = provider(e);
        if (have_conv)
            copy_vec(x, xconv);
        else
            slab_fill(ctx, 0.0, x);
        log(fmt("STAGE field=%s eps=%g N=%d cand=%s (from eps=%g, %s)%s", field, e, N, cand,
                e_conv, have_conv ? "warm start" : "u=0", psitc_tag.c_str()));
        SlabStageRecord st;
        st.eps = e;
        st.from_eps = e_conv;
        st.warm_start = have_conv;
        st.newton = solve(ctx, in, x, label_of(e), ncfg, log);
        const real mrt = st.newton.merit();
        st.accepted = st.newton.status == SlabSolveStatus::converged ||
                      (std::isfinite(mrt) && mrt <= ccfg.stage_ok);
        log(fmt("  STAGE_END field=%s eps=%g N=%d cand=%s status=%s its=%d r_F=%.3e r_out=%.3e "
                "-> %s",
                field, e, N, cand, to_string(st.newton.status), st.newton.its, st.newton.r_F,
                st.newton.r_out, st.accepted ? "accepted" : "FAILED"));
        taken.push_back(fmt("%g%s", e, st.accepted ? "" : "(fail)"));
        rep.final_newton = st.newton;
        const bool ok = st.accepted;
        rep.stages.push_back(std::move(st));
        if (ok) {
            e_conv = e;
            have_conv = true;
            copy_vec(xconv, DeviceSpan<const real>(x.data(), x.size()));
            todo.pop_front();
        } else if (rep.bisections < ccfg.max_bisections) {
            ++rep.bisections;
            todo.push_front(0.5 * (e_conv + e));
            log(fmt("  CONTINUATION bisection %d/%d: retry from eps=%g at eps=%g", rep.bisections,
                    ccfg.max_bisections, e_conv, todo.front()));
        } else {
            rep.floor_reached = true;
            // SF-33 N7a log fix: the message names the stage that failed; the state actually
            // reported is announced after the final attempt (which has its own STAGE_END).
            log(fmt("  CONTINUATION gave up after %d bisections at the failed stage eps=%g",
                    rep.bisections, e));
            if (std::fabs(e - eps) <= 1e-12) {
                log(fmt("  CONTINUATION reporting the failed state at eps=%g (status=%s)", e,
                        to_string(rep.final_newton.status)));
            } else {
                const SlabStageInputs& tin = provider(eps);
                if (have_conv)
                    copy_vec(x, xconv);
                else
                    slab_fill(ctx, 0.0, x);
                log(fmt("STAGE field=%s eps=%g N=%d cand=%s (final attempt from eps=%g)%s",
                        field, eps, N, cand, e_conv, psitc_tag.c_str()));
                SlabStageRecord fs;
                fs.eps = eps;
                fs.from_eps = e_conv;
                fs.warm_start = have_conv;
                fs.final_attempt = true;
                fs.newton = solve(ctx, tin, x, label_of(eps), ncfg, log);
                const real fm = fs.newton.merit();
                fs.accepted = fs.newton.status == SlabSolveStatus::converged ||
                              (std::isfinite(fm) && fm <= ccfg.stage_ok);
                log(fmt("  STAGE_END field=%s eps=%g N=%d cand=%s status=%s its=%d r_F=%.3e "
                        "r_out=%.3e (final attempt) -> %s",
                        field, eps, N, cand, to_string(fs.newton.status), fs.newton.its,
                        fs.newton.r_F, fs.newton.r_out, fs.accepted ? "accepted" : "FAILED"));
                log(fmt("  CONTINUATION reporting the state of the final attempt at eps=%g "
                        "(status=%s)",
                        eps, to_string(fs.newton.status)));
                rep.final_newton = fs.newton;
                rep.stages.push_back(std::move(fs));
                taken.push_back(fmt("%g(final)", eps));
            }
            break;
        }
    }
    rep.eps_accepted = have_conv ? e_conv : 0.0;
    std::string path;
    for (std::size_t i = 0; i < taken.size(); ++i) {
        if (i > 0)
            path += "->";
        path += taken[i];
    }
    rep.path = path;
    log(fmt("PATH field=%s eps=%g N=%d cand=%s continuation path: %s", field, eps, N, cand,
            path.c_str()));
    rep.status = rep.floor_reached ? SlabSolveStatus::continuation_floor : rep.final_newton.status;
    MACROFLOW3D_CUDA_CHECK(cudaStreamSynchronize(ctx.cuda_stream())); // documented: x complete
    rep.seconds = seconds_since(t0);
    return rep;
}

} // namespace inlet_slab
} // namespace streamfunctions
} // namespace macroflow3d
