/**
 * @file slab_newton_tests.cu
 * @brief SF-33 N2: fast contract tests of the inlet-slab GMRES, the per-mode preconditioner P-A,
 * Newton with line search and the amplitude continuation (SlabGmres.cuh,
 * SlabModePreconditioner.cuh, SlabNewtonKrylov.cuh). Standalone ctest runner (printed
 * [PASS]/[FAIL] checks, exit code), grids N = 12 and 16.
 *
 * Cases (acceptance items of the SF-33 N2 task):
 *   1  P-A at k = 1, u = 0, N = 12 and 16: ||P^-1 (J x) - x|| / ||x|| <= 1e-12, x gaussian
 *      (J = the N1 JVP); the prototype's `--k1check` lin0 identity.
 *   1b P-A exact (<= 1e-12) for an x1-only state (k(x1), u(x1)): every JVP coefficient is
 *      plane-constant there, so this checks the full plane-averaged assembly beyond lin0.
 *   2  GMRES: (a) k = 1 system with P-A converges in 1 inner iteration to true residual <= 1e-13;
 *      (b) frozen Jacobian at the perturbed exact pair (N1 fixture: exact pair + 1e-2 RMS(u)
 *      gaussian noise, seed 3301), b = -E(u), P-A at that state: true relative residual <= 1e-12
 *      with restart 200 (3 cycles) and 400 (1 cycle); recurrence and recomputed true residual
 *      agree at every cycle end to 1e-10 relative to ||b||; iteration counts printed. The
 *      default restart 50 is run for the record (not gated): it exits with `stagnation`.
 *   3  Newton on the exact-pair problem (k = exp(0.7 sin 2 pi x1), inlet (Phi(x3), 0),
 *      v_perp = 0) from x = 0, N = 12 and 16: converged, r_F <= 1e-13, r_out <= 1e-13 in <= 8
 *      iterations, max |u - u_exact| <= 1e-11; inlet data + band-limited random noise (max 1e-2,
 *      transverse modes |m| <= 2) started from the unperturbed exact pair: converged. For the
 *      record (not gated, N = 12): the perturbed problems from x = 0 (band-limited and per-vertex
 *      noise) do NOT converge; their distinct exit statuses are printed.
 *   4  Continuation mechanics on 12^3, k = exp(eps f), f = generic3d of
 *      apps/closure_gate/closure_fields.hpp evaluated analytically at the vertices with its
 *      hand-derived analytic gradient (checked against a central FD of f to 1e-8), affine inlet
 *      labels (u0 = 0), v_perp = 0, v_rms = 1 (mechanics only: these inputs are NOT Darcy labels;
 *      no metric claim): ladder to eps = 0.5 (STAGE / PATH printed); then max_newton = 1 and
 *      max_bisections = 1: exactly one bisection, `continuation_floor`, PATH asserted.
 *   5  Statuses: distinct enum strings; forced exits nan_inf (NaN start), linear_failure (GMRES cap
 *      1), linesearch-fail (lambda_min > 1, no trial allowed), stagnation (factor 0, window 2),
 *      maxit (1 iteration); one LINEAR line per Newton step that ran a linear solve.
 *   6  No allocation after prepare: workspace bytes and buffer pointers unchanged across a
 *      Newton solve and a continuation; cudaMemGetInfo free memory unchanged (after a warm-up).
 */

#include "apps/closure_gate/closure_fields.hpp"
#include "src/core/DeviceBuffer.cuh"
#include "src/core/DeviceSpan.cuh"
#include "src/core/Scalar.hpp"
#include "src/physics/streamfunctions/inlet_slab/InletSlabGrid.cuh"
#include "src/physics/streamfunctions/inlet_slab/SlabGmres.cuh"
#include "src/physics/streamfunctions/inlet_slab/SlabJacobianVectorProduct.cuh"
#include "src/physics/streamfunctions/inlet_slab/SlabModePreconditioner.cuh"
#include "src/physics/streamfunctions/inlet_slab/SlabNewtonKrylov.cuh"
#include "src/physics/streamfunctions/inlet_slab/SlabResidual.cuh"
#include "src/physics/streamfunctions/inlet_slab/SlabSolverTypes.cuh"
#include "src/runtime/cuda_check.cuh"
#include "src/runtime/CudaContext.cuh"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <functional>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

using namespace macroflow3d;
namespace sl = macroflow3d::streamfunctions::inlet_slab;

namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kTwoPi = 2.0 * kPi;

struct TestReport {
    bool overall_pass = true;
    int checks = 0;
    void check(bool cond, const std::string& name, const std::string& detail = "") {
        ++checks;
        std::printf("[%s] %s%s%s\n", cond ? "PASS" : "FAIL", name.c_str(),
                    detail.empty() ? "" : "  ", detail.c_str());
        std::fflush(stdout);
        overall_pass = overall_pass && cond;
    }
};

std::string fmtd(const char* f, double a) {
    char b[200];
    std::snprintf(b, sizeof(b), f, a);
    return b;
}

void upload(DeviceBuffer<real>& b, const std::vector<real>& h) {
    b.resize(h.size());
    MACROFLOW3D_CUDA_CHECK(
        cudaMemcpy(b.data(), h.data(), h.size() * sizeof(real), cudaMemcpyHostToDevice));
}

std::vector<real> download(const real* d, std::size_t n) {
    std::vector<real> h(n);
    MACROFLOW3D_CUDA_CHECK(cudaMemcpy(h.data(), d, n * sizeof(real), cudaMemcpyDeviceToHost));
    return h;
}

DeviceSpan<const real> cspan(const DeviceBuffer<real>& b) {
    return DeviceSpan<const real>(b.data(), b.size());
}
DeviceSpan<real> mspan(DeviceBuffer<real>& b) {
    return DeviceSpan<real>(b.data(), b.size());
}

real norm2(const std::vector<real>& a) {
    real s = 0.0;
    for (real v : a)
        s += v * v;
    return std::sqrt(s);
}
real rms(const std::vector<real>& a) {
    return norm2(a) / std::sqrt(static_cast<real>(a.size()));
}
real rel_diff(const std::vector<real>& a, const std::vector<real>& ref) {
    std::vector<real> d(a.size());
    for (std::size_t i = 0; i < a.size(); ++i)
        d[i] = a[i] - ref[i];
    return norm2(d) / norm2(ref);
}
std::vector<real> gaussian(std::mt19937_64& rng, std::size_t n) {
    std::normal_distribution<real> nd(0.0, 1.0);
    std::vector<real> a(n);
    for (auto& x : a)
        x = nd(rng);
    return a;
}

using F3 = std::function<real(real, real, real)>;

std::vector<real> sample_full(const sl::InletSlabGrid& g, const F3& f) {
    std::vector<real> a(g.full_size());
    for (int j = 0; j <= g.n; ++j)
        for (int m2 = 0; m2 < g.n; ++m2)
            for (int m3 = 0; m3 < g.n; ++m3)
                a[g.full_index(j, m2, m3)] = f(g.coord(j), g.coord(m2), g.coord(m3));
    return a;
}

struct CaseSpec {
    F3 lnk, gl0, gl1, gl2, u1, u2;
};

real phi_x3(real x3) {
    return 0.3 * std::sin(kTwoPi * x3) + 0.1 * std::cos(2.0 * kTwoPi * x3);
}

/// Exact discrete pair of N0/N1: k = exp(0.7 sin 2 pi x1), u1 = Phi(x3), u2 = 0, v_perp = 0.
CaseSpec exact_pair_spec() {
    CaseSpec s;
    s.lnk = [](real x1, real, real) { return 0.7 * std::sin(kTwoPi * x1); };
    s.gl0 = [](real x1, real, real) { return 0.7 * kTwoPi * std::cos(kTwoPi * x1); };
    s.gl1 = [](real, real, real) { return 0.0; };
    s.gl2 = [](real, real, real) { return 0.0; };
    s.u1 = [](real, real, real x3) { return phi_x3(x3); };
    s.u2 = [](real, real, real) { return 0.0; };
    return s;
}

CaseSpec k1_spec() {
    CaseSpec s;
    auto zero = [](real, real, real) { return 0.0; };
    s.lnk = s.gl0 = s.gl1 = s.gl2 = s.u1 = s.u2 = zero;
    return s;
}

/// Fills stage inputs (allocated for g) from a spec; returns the host unknown vector of the spec's
/// u (planes 1..N). vperp = 0, v_rms = 1.
std::vector<real> fill_inputs(CudaContext& ctx, const sl::InletSlabGrid& g, sl::SlabStageInputs& in,
                              const CaseSpec& s) {
    in.allocate(g);
    in.field = "test";
    upload(in.lnk, sample_full(g, s.lnk));
    sl::fill_q_from_lnk(ctx, g, in);
    const F3* gl[3] = {&s.gl0, &s.gl1, &s.gl2};
    for (int k = 0; k < 3; ++k)
        upload(in.grad_lnk[k], sample_full(g, *gl[k]));
    const auto hU1 = sample_full(g, s.u1);
    const auto hU2 = sample_full(g, s.u2);
    const std::size_t np = g.plane_size(), nf = g.field_size();
    upload(in.u0[0], std::vector<real>(hU1.begin(), hU1.begin() + np));
    upload(in.u0[1], std::vector<real>(hU2.begin(), hU2.begin() + np));
    upload(in.vperp_in[0], std::vector<real>(np, 0.0));
    upload(in.vperp_in[1], std::vector<real>(np, 0.0));
    in.v_rms = 1.0;
    std::vector<real> u(2 * nf, 0.0);
    for (std::size_t i = 0; i < nf; ++i) {
        u[i] = hU1[np + i];
        u[nf + i] = hU2[np + i];
    }
    ctx.synchronize();
    return u;
}

/// Direct operator harness (residual, JVP, P-A) on one state.
struct OpHarness {
    sl::InletSlabGrid g;
    sl::SlabStageInputs in;
    sl::SlabResidualWorkspace rws;
    sl::SlabJvpWorkspace jws;
    sl::SlabModePreconditioner prec;
    DeviceBuffer<real> U1, U2, uvec, E, a, b, c;
    std::vector<real> u_spec;

    void build(CudaContext& ctx, int N, const CaseSpec& s) {
        g = sl::InletSlabGrid::make(N);
        u_spec = fill_inputs(ctx, g, in, s);
        U1.resize(g.full_size());
        U2.resize(g.full_size());
        for (auto* v : {&uvec, &E, &a, &b, &c})
            v->resize(g.unknown_size());
        rws.prepare(g);
        jws.prepare(g);
        prec.prepare(ctx, g);
        ctx.synchronize();
    }
    /// Freeze JVP + factor P-A at u; returns E(u).
    std::vector<real> freeze(CudaContext& ctx, const std::vector<real>& u,
                             int* singular = nullptr) {
        upload(uvec, u);
        sl::assemble_full_planes(ctx, g, cspan(uvec), in, mspan(U1), mspan(U2));
        sl::evaluate_residual(ctx, g, in, cspan(U1), cspan(U2), mspan(E), rws, nullptr);
        jws.prepare_base(ctx, g, in, cspan(U1), cspan(U2));
        const auto fr = prec.factor(ctx, g, in, cspan(U1), cspan(U2));
        if (singular)
            *singular = fr.singular_modes;
        ctx.synchronize();
        return download(E.data(), E.size());
    }
};

// ------------------------------------------------------------------------------------------------
// 1. P-A at k = 1, u = 0
// ------------------------------------------------------------------------------------------------
void case_pa_k1(TestReport& rep, CudaContext& ctx) {
    for (int N : {12, 16}) {
        OpHarness hs;
        hs.build(ctx, N, k1_spec());
        int singular = -1;
        hs.freeze(ctx, hs.u_spec, &singular);
        std::mt19937_64 rng(4400 + N);
        const auto x = gaussian(rng, hs.g.unknown_size());
        upload(hs.a, x);
        hs.jws.apply(ctx, hs.g, cspan(hs.a), mspan(hs.b));
        hs.prec.apply(ctx, hs.g, cspan(hs.b), mspan(hs.c));
        ctx.synchronize();
        const auto z = download(hs.c.data(), hs.c.size());
        const real err = rel_diff(z, x);
        std::printf("[INFO] PA_K1 N=%d |P^-1 J x - x|/|x| = %.3e  (singular modes %d, modes %d, "
                    "P-A bytes %zu)\n",
                    N, err, singular, hs.prec.n_modes(), hs.prec.allocated_bytes());
        rep.check(err <= 1e-12 && singular == 0,
                  "P-A k = 1, u = 0, N = " + std::to_string(N) +
                      ": ||P^-1 (J x) - x|| / ||x|| <= 1e-12",
                  fmtd("%.3e", err));
    }
}

/// 1b. P-A exactness beyond lin0: for k = k(x1) and a base state u = u(x1) every pointwise JVP
/// coefficient is constant on each plane (g_i, H_i, B, c, S_i, grad ln k and q depend on x1
/// only, with nonzero B and S-linearization terms), so the plane average is exact and
/// M^-1 J = I to roundoff. This checks the FULL plane-averaged assembly (grad ln k, S-terms,
/// first-derivative x1 coefficients, outlet coefficients at a non-affine state).
void case_pa_x1_state(TestReport& rep, CudaContext& ctx) {
    CaseSpec s;
    s.lnk = [](real x1, real, real) {
        return 0.5 * std::sin(kTwoPi * x1) + 0.2 * std::cos(2.0 * kTwoPi * x1);
    };
    s.gl0 = [](real x1, real, real) {
        return 0.5 * kTwoPi * std::cos(kTwoPi * x1) - 0.4 * kTwoPi * std::sin(2.0 * kTwoPi * x1);
    };
    s.gl1 = [](real, real, real) { return 0.0; };
    s.gl2 = [](real, real, real) { return 0.0; };
    s.u1 = [](real x1, real, real) { return 0.05 + 0.1 * std::sin(kPi * x1); };
    s.u2 = [](real x1, real, real) { return 0.08 * x1 * x1 - 0.03 * x1; };
    for (int N : {12, 16}) {
        OpHarness hs;
        hs.build(ctx, N, s);
        int singular = -1;
        hs.freeze(ctx, hs.u_spec, &singular);
        const auto coef = hs.prec.download_coefficients(ctx);
        real maxS =
            0.0; // a coefficient that vanishes at k = 1, u = 0: dF1 / d(dg1[0]) (grad ln k, S)
        for (int j = 1; j < N; ++j)
            maxS = std::fmax(maxS, std::fabs(coef[static_cast<std::size_t>(j - 1) * 36 + 0]));
        std::mt19937_64 rng(4450 + N);
        const auto x = gaussian(rng, hs.g.unknown_size());
        upload(hs.a, x);
        hs.jws.apply(ctx, hs.g, cspan(hs.a), mspan(hs.b));
        hs.prec.apply(ctx, hs.g, cspan(hs.b), mspan(hs.c));
        ctx.synchronize();
        const real err = rel_diff(download(hs.c.data(), hs.c.size()), x);
        std::printf("[INFO] PA_X1STATE N=%d |P^-1 J x - x|/|x| = %.3e (singular %d; max |coef "
                    "dF1/d(d1 u1)| on the equation planes = %.3e)\n",
                    N, err, singular, maxS);
        rep.check(err <= 1e-12 && singular == 0 && maxS > 1e-2,
                  "P-A exact for an x1-only state (k(x1), u(x1)), N = " + std::to_string(N) +
                      ": ||P^-1 (J x) - x|| / ||x|| <= 1e-12",
                  fmtd("%.3e", err));
    }
}

// ------------------------------------------------------------------------------------------------
// 2. GMRES
// ------------------------------------------------------------------------------------------------
void case_gmres(TestReport& rep, CudaContext& ctx) {
    // (a) k = 1
    {
        OpHarness hs;
        hs.build(ctx, 12, k1_spec());
        hs.freeze(ctx, hs.u_spec);
        sl::SlabGmres gm;
        gm.prepare(hs.g, 50);
        std::mt19937_64 rng(4501);
        upload(hs.a, gaussian(rng, hs.g.unknown_size()));
        const sl::SlabGmres::Operator A = [&](DeviceSpan<const real> i, DeviceSpan<real> o) {
            hs.jws.apply(ctx, hs.g, i, o);
        };
        const sl::SlabGmres::Operator M = [&](DeviceSpan<const real> i, DeviceSpan<real> o) {
            hs.prec.apply(ctx, hs.g, i, o);
        };
        sl::SlabGmresConfig cfg;
        const auto r = gm.solve(ctx, A, M, cspan(hs.a), mspan(hs.b), cfg);
        std::printf("[INFO] GMRES k=1 N=12: status=%s its=%d cycles=%d true rel=%.3e rec=%.3e\n",
                    sl::to_string(r.status), r.iterations, r.cycles, r.rel_residual,
                    r.rel_recurrence);
        rep.check(r.status == sl::SlabLinearStatus::converged && r.iterations == 1 &&
                      r.rel_residual <= 1e-13,
                  "GMRES + P-A on the k = 1 system: 1 inner iteration, true residual <= 1e-13",
                  "its " + std::to_string(r.iterations) + fmtd(" rel %.3e", r.rel_residual));
    }
    // (b) perturbed exact pair
    {
        OpHarness hs;
        hs.build(ctx, 12, exact_pair_spec());
        std::mt19937_64 rng(3301);
        const real rms_ex = rms(hs.u_spec);
        const auto noise = gaussian(rng, hs.g.unknown_size());
        std::vector<real> u(hs.u_spec);
        for (std::size_t i = 0; i < u.size(); ++i)
            u[i] += 1e-2 * rms_ex * noise[i];
        const auto E = hs.freeze(ctx, u);
        std::vector<real> rhs(E.size());
        for (std::size_t i = 0; i < E.size(); ++i)
            rhs[i] = -E[i];
        upload(hs.a, rhs);
        const sl::SlabGmres::Operator A = [&](DeviceSpan<const real> i, DeviceSpan<real> o) {
            hs.jws.apply(ctx, hs.g, i, o);
        };
        const sl::SlabGmres::Operator M = [&](DeviceSpan<const real> i, DeviceSpan<real> o) {
            hs.prec.apply(ctx, hs.g, i, o);
        };
        // restart 50 (the default) is run for the record only: P-A is a weak preconditioner on
        // this rough state (gaussian per-vertex noise) and GMRES(50) exits with `stagnation`; the
        // gated runs use restart 200 (several cycles: recurrence vs true checked at each) and 400.
        for (int restart : {50, 200, 400}) {
            const bool gated = restart != 50;
            sl::SlabGmres gm;
            gm.prepare(hs.g, restart);
            sl::SlabGmresConfig cfg;
            cfg.restart = restart;
            const auto r = gm.solve(ctx, A, M, cspan(hs.a), mspan(hs.b), cfg);
            // independent host-side check of the returned x: ||b - J x|| / ||b||
            hs.jws.apply(ctx, hs.g, cspan(hs.b), mspan(hs.c));
            ctx.synchronize();
            const auto Jx = download(hs.c.data(), hs.c.size());
            const real indep = rel_diff(Jx, rhs);
            real worst = 0.0;
            std::string cyc;
            for (std::size_t k = 0; k < r.cycle_true.size(); ++k) {
                const real d = std::fabs(r.cycle_true[k] - r.cycle_recurrence[k]);
                worst = std::fmax(worst, d);
                char b[96];
                std::snprintf(b, sizeof(b), " [true %.3e rec %.3e]", r.cycle_true[k],
                              r.cycle_recurrence[k]);
                cyc += b;
            }
            std::printf("[INFO] GMRES perturbed exact pair N=12 restart=%d: status=%s its=%d "
                        "cycles=%d reorth=%d true rel=%.3e (host recheck %.3e); per cycle:%s\n",
                        restart, sl::to_string(r.status), r.iterations, r.cycles,
                        r.reorthogonalizations, r.rel_residual, indep, cyc.c_str());
            char det[160];
            std::snprintf(det, sizeof(det),
                          "its %d cycles %d rel %.3e host %.3e max|rec-true| %.3e", r.iterations,
                          r.cycles, r.rel_residual, indep, worst);
            if (!gated) {
                std::printf("[INFO] GMRES(50) at the perturbed exact pair (not gated): status=%s "
                            "true rel=%.3e after %d its; max|rec-true| %.3e\n",
                            sl::to_string(r.status), r.rel_residual, r.iterations, worst);
                continue;
            }
            rep.check(r.status == sl::SlabLinearStatus::converged && r.rel_residual <= 1e-12 &&
                          indep <= 1e-12,
                      "GMRES + P-A, frozen J at the perturbed exact pair, restart " +
                          std::to_string(restart) + ": true relative residual <= 1e-12",
                      det);
            rep.check(worst <= 1e-10 && !r.cycle_true.empty(),
                      "GMRES restart " + std::to_string(restart) +
                          ": recurrence and true residual agree at every cycle end to 1e-10 "
                          "(relative to ||b||)",
                      det);
        }
    }
}

// ------------------------------------------------------------------------------------------------
// Logger capturing lines (and echoing them)
// ------------------------------------------------------------------------------------------------
struct Capture {
    std::vector<std::string> lines;
    sl::SlabLogger logger() {
        return [this](const std::string& l) {
            lines.push_back(l);
            sl::slab_stdout_logger(l);
        };
    }
    int count(const std::string& needle) const {
        int c = 0;
        for (const auto& l : lines)
            if (l.find(needle) != std::string::npos)
                ++c;
        return c;
    }
};

// ------------------------------------------------------------------------------------------------
// 3. Newton on the exact pair
// ------------------------------------------------------------------------------------------------
void case_newton_exact_pair(TestReport& rep, CudaContext& ctx) {
    for (int N : {12, 16}) {
        const auto g = sl::InletSlabGrid::make(N);
        sl::SlabStageInputs in;
        const auto u_ex = fill_inputs(ctx, g, in, exact_pair_spec());
        sl::SlabNewtonKrylov nk;
        nk.prepare(ctx, g);
        DeviceBuffer<real> x(g.unknown_size());
        upload(x, std::vector<real>(g.unknown_size(), 0.0));
        sl::SlabNewtonConfig cfg;
        Capture cap;
        const auto r =
            nk.solve(ctx, in, mspan(x), "exact_pair:0.7:" + std::to_string(N), cfg, cap.logger());
        const auto xh = download(x.data(), x.size());
        real maxerr = 0.0;
        for (std::size_t i = 0; i < xh.size(); ++i)
            maxerr = std::fmax(maxerr, std::fabs(xh[i] - u_ex[i]));
        std::string lits;
        for (const auto& s : r.steps)
            lits += " " + std::to_string(s.linear.iterations);
        char det[220];
        std::snprintf(det, sizeof(det),
                      "status %s its %d r_F %.3e r_out %.3e max|u-u_ex| %.3e GMRES its/step:%s",
                      sl::to_string(r.status), r.its, r.r_F, r.r_out, maxerr, lits.c_str());
        rep.check(r.status == sl::SlabSolveStatus::converged && r.r_F <= 1e-13 &&
                      r.r_out <= 1e-13 && r.its <= 8,
                  "Newton exact pair N = " + std::to_string(N) +
                      " from x = 0: converged, r_F, r_out <= 1e-13, <= 8 iterations",
                  det);
        rep.check(maxerr <= 1e-11,
                  "Newton exact pair N = " + std::to_string(N) + ": max |u - u_exact| <= 1e-11",
                  det);
        rep.check(cap.count("LINEAR gmres+P-A") == static_cast<int>(r.steps.size()),
                  "Newton exact pair N = " + std::to_string(N) +
                      ": one LINEAR line per Newton step",
                  std::to_string(cap.count("LINEAR gmres+P-A")) + " lines, " +
                      std::to_string(r.steps.size()) + " steps");

        // inlet data perturbed by band-limited random noise of max amplitude 1e-2 (both fields):
        // random coefficients of the transverse modes |m2|, |m3| <= 2, rescaled to max |.| = 1e-2.
        std::mt19937_64 rng(4600 + N);
        const auto u0_ex0 = download(in.u0[0].data(), in.u0[0].size());
        const auto u0_ex1 = download(in.u0[1].data(), in.u0[1].size());
        for (int f = 0; f < 2; ++f) {
            auto u0 = f == 0 ? u0_ex0 : u0_ex1;
            std::vector<real> nz(u0.size(), 0.0);
            std::normal_distribution<real> nd(0.0, 1.0);
            for (int a = -2; a <= 2; ++a)
                for (int b = -2; b <= 2; ++b) {
                    const real ca = nd(rng), sa = nd(rng);
                    for (int m2 = 0; m2 < g.n; ++m2)
                        for (int m3 = 0; m3 < g.n; ++m3) {
                            const real th = kTwoPi * (a * g.coord(m2) + b * g.coord(m3));
                            nz[g.plane_index(m2, m3)] += ca * std::cos(th) + sa * std::sin(th);
                        }
                }
            real mx = 0.0;
            for (real v : nz)
                mx = std::fmax(mx, std::fabs(v));
            for (std::size_t i = 0; i < u0.size(); ++i)
                u0[i] += 1e-2 * nz[i] / mx;
            upload(in.u0[f], u0);
        }
        // Gated: Newton started from the unperturbed exact pair (continuation in the inlet data).
        upload(x, u_ex);
        Capture cap2;
        const auto r2 =
            nk.solve(ctx, in, mspan(x), "exact_pair_noisy_inlet:0.7:" + std::to_string(N), cfg,
                     cap2.logger());
        std::string lits2;
        for (const auto& s : r2.steps)
            lits2 += " " + std::to_string(s.linear.iterations);
        std::snprintf(det, sizeof(det), "status %s its %d r_F %.3e r_out %.3e GMRES its/step:%s",
                      sl::to_string(r2.status), r2.its, r2.r_F, r2.r_out, lits2.c_str());
        rep.check(r2.status == sl::SlabSolveStatus::converged && r2.r_F <= 1e-13 &&
                      r2.r_out <= 1e-13 &&
                      cap2.count("LINEAR gmres+P-A") == static_cast<int>(r2.steps.size()),
                  "Newton exact pair N = " + std::to_string(N) +
                      ", inlet data + band-limited 1e-2 random noise, started from the "
                      "unperturbed exact pair: converged (r_F, r_out <= 1e-13)",
                  det);

        if (N == 12) {
            // For the record (NOT gated): the same perturbed problem started from x = 0. The
            // unperturbed exact pair converges from x = 0 only inside its symmetric
            // (x2-independent) invariant subspace; breaking the symmetry exposes the strong
            // nonlinearity of the large inlet deformation (|d3 Phi| up to ~3): with exact linear
            // solves (restart 400) Newton stagnates as well. The exit must be a DISTINCT status.
            upload(x, std::vector<real>(g.unknown_size(), 0.0));
            const sl::SlabLogger quiet = [](const std::string&) {};
            const auto r0 = nk.solve(ctx, in, mspan(x), "noisy_from_zero", cfg, quiet);
            std::printf("[INFO] Newton exact pair N = 12, band-limited inlet noise 1e-2, from "
                        "x = 0 (not gated): status=%s its=%d r_F=%.3e last linear status=%s\n",
                        sl::to_string(r0.status), r0.its, r0.r_F,
                        sl::to_string(r0.last_linear_status));
            // For the record (NOT gated): per-vertex (white) gaussian noise of std 1e-2 on the
            // inlet data, from x = 0.
            std::mt19937_64 rng2(4650);
            for (int f = 0; f < 2; ++f) {
                auto u0 = f == 0 ? u0_ex0 : u0_ex1;
                const auto wn = gaussian(rng2, u0.size());
                for (std::size_t i = 0; i < u0.size(); ++i)
                    u0[i] += 1e-2 * wn[i];
                upload(in.u0[f], u0);
            }
            upload(x, std::vector<real>(g.unknown_size(), 0.0));
            const auto r3 = nk.solve(ctx, in, mspan(x), "white", cfg, quiet);
            std::printf("[INFO] Newton exact pair N = 12, per-vertex gaussian inlet noise 1e-2 "
                        "(not gated): status=%s its=%d r_F=%.3e last linear status=%s\n",
                        sl::to_string(r3.status), r3.its, r3.r_F,
                        sl::to_string(r3.last_linear_status));
        }
    }
}

// ------------------------------------------------------------------------------------------------
// 4. generic3d continuation mechanics
// ------------------------------------------------------------------------------------------------
using closure_gate::analytic_log_conductivity;
using closure_gate::AnalyticField;

/// Hand-derived analytic gradient of generic3d f(X, Y, Z) (closure_fields.hpp):
/// f = cos A + cos B + 0.8 cos C + 0.6 sin D, A = 2pi(X+Y), B = 2pi(X+Z) + 0.7,
/// C = 2pi(X-Y+Z) + 1.3, D = 2pi(2X+Y-Z).
void generic3d_grad(real X, real Y, real Z, real gr[3]) {
    const real A = kTwoPi * (X + Y);
    const real B = kTwoPi * (X + Z) + 0.7;
    const real C = kTwoPi * (X - Y + Z) + 1.3;
    const real D = kTwoPi * (2.0 * X + Y - Z);
    gr[0] = kTwoPi * (-std::sin(A) - std::sin(B) - 0.8 * std::sin(C) + 1.2 * std::cos(D));
    gr[1] = kTwoPi * (-std::sin(A) + 0.8 * std::sin(C) + 0.6 * std::cos(D));
    gr[2] = kTwoPi * (-std::sin(B) - 0.8 * std::sin(C) - 0.6 * std::cos(D));
}

struct Generic3dProvider {
    sl::InletSlabGrid g;
    sl::SlabStageInputs in;
    CudaContext* ctx = nullptr;
    std::vector<real> f, gf[3];
    int calls = 0;

    void build(CudaContext& c, const sl::InletSlabGrid& grid) {
        ctx = &c;
        g = grid;
        in.allocate(g);
        in.field = "generic3d";
        in.N = g.n;
        f = sample_full(g, [](real x1, real x2, real x3) {
            return analytic_log_conductivity(AnalyticField::generic3d, x1, x2, x3);
        });
        for (int k = 0; k < 3; ++k)
            gf[k] = sample_full(g, [k](real x1, real x2, real x3) {
                real gr[3];
                generic3d_grad(x1, x2, x3, gr);
                return gr[k];
            });
        const std::vector<real> zp(g.plane_size(), 0.0);
        for (int f2 = 0; f2 < 2; ++f2) {
            upload(in.u0[f2], zp);
            upload(in.vperp_in[f2], zp);
        }
        in.v_rms = 1.0;
        (*this)(0.25); // sizes every buffer before any allocation measurement
    }
    const sl::SlabStageInputs& operator()(real amp) {
        ++calls;
        std::vector<real> t(f.size());
        for (std::size_t i = 0; i < f.size(); ++i)
            t[i] = amp * f[i];
        upload(in.lnk, t);
        sl::fill_q_from_lnk(*ctx, g, in);
        for (int k = 0; k < 3; ++k) {
            for (std::size_t i = 0; i < f.size(); ++i)
                t[i] = amp * gf[k][i];
            upload(in.grad_lnk[k], t);
        }
        in.eps = amp;
        ctx->synchronize();
        return in;
    }
};

void case_continuation(TestReport& rep, CudaContext& ctx) {
    const auto g = sl::InletSlabGrid::make(12);
    // analytic gradient vs central FD of f (h = 1e-6) on a set of points
    {
        real worst = 0.0;
        std::mt19937_64 rng(4700);
        std::uniform_real_distribution<real> ud(0.0, 1.0);
        const real hh = 1e-6;
        for (int p = 0; p < 200; ++p) {
            const real X = ud(rng), Y = ud(rng), Z = ud(rng);
            real gr[3];
            generic3d_grad(X, Y, Z, gr);
            const auto fA = [](real a, real b, real c) {
                return analytic_log_conductivity(AnalyticField::generic3d, a, b, c);
            };
            const real fd[3] = {(fA(X + hh, Y, Z) - fA(X - hh, Y, Z)) / (2 * hh),
                                (fA(X, Y + hh, Z) - fA(X, Y - hh, Z)) / (2 * hh),
                                (fA(X, Y, Z + hh) - fA(X, Y, Z - hh)) / (2 * hh)};
            for (int k = 0; k < 3; ++k)
                worst = std::fmax(worst, std::fabs(fd[k] - gr[k]));
        }
        rep.check(worst <= 1e-8, "generic3d analytic gradient vs central FD (h = 1e-6, 200 points)",
                  fmtd("max abs diff %.3e", worst));
    }
    Generic3dProvider prov;
    prov.build(ctx, g);
    const sl::StageInputProvider provider = [&](real amp) -> const sl::SlabStageInputs& {
        return prov(amp);
    };
    sl::SlabNewtonKrylov nk;
    nk.prepare(ctx, g);
    DeviceBuffer<real> x(g.unknown_size());
    sl::SlabNewtonConfig ncfg;
    sl::SlabContinuationConfig ccfg;
    ccfg.field = "generic3d";
    {
        Capture cap;
        const auto r =
            nk.solve_with_continuation(ctx, 0.5, provider, mspan(x), ccfg, ncfg, cap.logger());
        std::printf("[INFO] continuation generic3d N=12 eps=0.5: status=%s path=%s bisections=%d "
                    "stages=%zu\n",
                    sl::to_string(r.status), r.path.c_str(), r.bisections, r.stages.size());
        rep.check(r.status == sl::SlabSolveStatus::converged && r.path == "0.25->0.5" &&
                      r.bisections == 0 && cap.count("STAGE field=generic3d") == 2 &&
                      cap.count("PATH field=generic3d eps=0.5 N=12 cand=i1o4 continuation "
                                "path: 0.25->0.5") == 1,
                  "continuation generic3d 12^3 to eps = 0.5: ladder 0.25 -> 0.5, both stages "
                  "accepted, STAGE / PATH lines in the prototype format",
                  "path " + r.path);
        int steps = 0;
        for (const auto& s : r.stages)
            steps += static_cast<int>(s.newton.steps.size());
        rep.check(cap.count("LINEAR gmres+P-A") == steps,
                  "continuation: one LINEAR line per Newton step",
                  std::to_string(cap.count("LINEAR gmres+P-A")) + " / " + std::to_string(steps));
    }
    {
        Capture cap;
        sl::SlabNewtonConfig n1 = ncfg;
        n1.max_iterations = 1;
        sl::SlabContinuationConfig c1 = ccfg;
        c1.max_bisections = 1;
        const auto r =
            nk.solve_with_continuation(ctx, 0.5, provider, mspan(x), c1, n1, cap.logger());
        std::printf("[INFO] forced failure (max_newton = 1, bisect = 1): status=%s path=%s "
                    "bisections=%d\n",
                    sl::to_string(r.status), r.path.c_str(), r.bisections);
        const std::string expect = "0.25(fail)->0.125(fail)->0.5(final)";
        rep.check(r.status == sl::SlabSolveStatus::continuation_floor && r.bisections == 1 &&
                      cap.count("CONTINUATION bisection") == 1 &&
                      cap.count("CONTINUATION bisection 1/1: retry from eps=0 at eps=0.125") == 1 &&
                      cap.count("CONTINUATION gave up after 1 bisections") == 1 &&
                      r.path == expect &&
                      cap.count("PATH field=generic3d eps=0.5 N=12 cand=i1o4 continuation path: " +
                                expect) == 1,
                  "continuation forced failure: exactly one bisection, continuation_floor, PATH " +
                      expect,
                  "status " + std::string(sl::to_string(r.status)) + " path " + r.path);
    }
}

// ------------------------------------------------------------------------------------------------
// 5. statuses
// ------------------------------------------------------------------------------------------------
void case_statuses(TestReport& rep, CudaContext& ctx) {
    {
        std::set<std::string> s;
        const sl::SlabSolveStatus all[] = {
            sl::SlabSolveStatus::not_run,         sl::SlabSolveStatus::converged,
            sl::SlabSolveStatus::linesearch_fail, sl::SlabSolveStatus::stagnation,
            sl::SlabSolveStatus::maxit,           sl::SlabSolveStatus::linear_failure,
            sl::SlabSolveStatus::nan_inf,         sl::SlabSolveStatus::continuation_floor};
        for (auto v : all)
            s.insert(sl::to_string(v));
        std::set<std::string> l;
        const sl::SlabLinearStatus lall[] = {
            sl::SlabLinearStatus::not_run,        sl::SlabLinearStatus::converged,
            sl::SlabLinearStatus::max_iterations, sl::SlabLinearStatus::stagnation,
            sl::SlabLinearStatus::breakdown,      sl::SlabLinearStatus::nonfinite};
        for (auto v : lall)
            l.insert(sl::to_string(v));
        rep.check(s.size() == 8 && l.size() == 6 && s.count("linesearch-fail") &&
                      s.count("linear_failure") && s.count("nan_inf") &&
                      s.count("continuation_floor") && s.count("stagnation") && s.count("maxit"),
                  "status enums: all strings distinct (8 nonlinear, 6 linear)");
    }
    const auto g = sl::InletSlabGrid::make(12);
    Generic3dProvider prov;
    prov.build(ctx, g);
    const sl::SlabStageInputs& in = prov(0.25);
    sl::SlabNewtonKrylov nk;
    nk.prepare(ctx, g);
    DeviceBuffer<real> x(g.unknown_size());
    const std::vector<real> zero(g.unknown_size(), 0.0);
    auto run = [&](const sl::SlabNewtonConfig& cfg, const std::string& label, Capture& cap) {
        upload(x, zero);
        return nk.solve(ctx, in, mspan(x), label, cfg, cap.logger());
    };
    {
        std::vector<real> bad(zero);
        bad[17] = std::nan("");
        upload(x, bad);
        Capture cap;
        const auto r = nk.solve(ctx, in, mspan(x), "nan", sl::SlabNewtonConfig{}, cap.logger());
        rep.check(r.status == sl::SlabSolveStatus::nan_inf && r.steps.empty(),
                  "forced nan_inf: NaN in the start vector -> nan_inf, no step attempted",
                  sl::to_string(r.status));
    }
    {
        sl::SlabNewtonConfig cfg;
        cfg.gmres.max_iterations = 1;
        Capture cap;
        const auto r = run(cfg, "lincap", cap);
        rep.check(r.status == sl::SlabSolveStatus::linear_failure &&
                      r.last_linear_status == sl::SlabLinearStatus::max_iterations &&
                      cap.count("LINEAR gmres+P-A") == 1 && cap.count("(step not taken)") == 1 &&
                      r.hist_r_F.size() == 1,
                  "forced linear_failure: GMRES cap 1 -> linear_failure (linear status "
                  "max_iterations), step not taken",
                  std::string(sl::to_string(r.status)) + " / " +
                      sl::to_string(r.last_linear_status));
    }
    {
        sl::SlabNewtonConfig cfg;
        cfg.lambda_min = 2.0; // no trial step allowed: exercises the line-search failure exit
        Capture cap;
        const auto r = run(cfg, "ls", cap);
        rep.check(r.status == sl::SlabSolveStatus::linesearch_fail &&
                      cap.count("line search failed") == 1 && cap.count("LINEAR gmres+P-A") == 1,
                  "forced linesearch-fail (lambda_min = 2): distinct status and log line",
                  sl::to_string(r.status));
    }
    {
        sl::SlabNewtonConfig cfg;
        cfg.stagnation_window = 2;
        cfg.stagnation_factor = 0.0;
        Capture cap;
        const auto r = run(cfg, "stag", cap);
        rep.check(r.status == sl::SlabSolveStatus::stagnation && r.its == 1,
                  "forced stagnation (window 2, factor 0): stagnation after the first step",
                  sl::to_string(r.status));
    }
    {
        sl::SlabNewtonConfig cfg;
        cfg.max_iterations = 1;
        Capture cap;
        const auto r = run(cfg, "maxit", cap);
        rep.check(r.status == sl::SlabSolveStatus::maxit && r.its == 1 &&
                      cap.count("LINEAR gmres+P-A") == 1,
                  "forced maxit (max_iterations = 1): maxit", sl::to_string(r.status));
    }
}

// ------------------------------------------------------------------------------------------------
// 6. no allocation after prepare
// ------------------------------------------------------------------------------------------------
void case_no_allocation(TestReport& rep, CudaContext& ctx) {
    const auto g = sl::InletSlabGrid::make(12);
    Generic3dProvider prov;
    prov.build(ctx, g);
    const sl::StageInputProvider provider = [&](real amp) -> const sl::SlabStageInputs& {
        return prov(amp);
    };
    sl::SlabNewtonKrylov nk;
    nk.prepare(ctx, g);
    DeviceBuffer<real> x(g.unknown_size());
    sl::SlabNewtonConfig ncfg;
    sl::SlabContinuationConfig ccfg;
    ccfg.field = "generic3d";
    const sl::SlabLogger quiet = [](const std::string&) {};
    // warm-up (lazy module loading of every kernel)
    nk.solve_with_continuation(ctx, 0.5, provider, mspan(x), ccfg, ncfg, quiet);
    ctx.synchronize();
    const std::size_t bytes0 = nk.allocated_bytes();
    const auto ptr0 = nk.buffer_pointers();
    // cudaMemGetInfo is device-wide: other processes sharing the GPU can change it. Up to three
    // measurement windows are taken; the secondary check passes if one window shows no change
    // (the primary check is the workspace bytes / buffer pointers, which no other process affects).
    bool free_same = false;
    std::string attempts;
    for (int attempt = 0; attempt < 3 && !free_same; ++attempt) {
        std::size_t free0 = 0, free1 = 0, total = 0;
        ctx.synchronize();
        MACROFLOW3D_CUDA_CHECK(cudaMemGetInfo(&free0, &total));
        const auto r = nk.solve_with_continuation(ctx, 0.5, provider, mspan(x), ccfg, ncfg, quiet);
        upload(x, std::vector<real>(g.unknown_size(), 0.0));
        const auto r2 = nk.solve(ctx, prov(0.25), mspan(x), "alloc", ncfg, quiet);
        ctx.synchronize();
        MACROFLOW3D_CUDA_CHECK(cudaMemGetInfo(&free1, &total));
        free_same = free0 == free1;
        attempts += " [" + std::to_string(free0) + " -> " + std::to_string(free1) + ", " +
                    sl::to_string(r.status) + "/" + sl::to_string(r2.status) + "]";
    }
    const std::size_t bytes1 = nk.allocated_bytes();
    const auto ptr1 = nk.buffer_pointers();
    std::printf("[INFO] allocation: Newton-Krylov workspace %zu bytes at N = 12 (GMRES basis "
                "formula 2(m+1)N^3*8 = %zu, P-A band formula %zu); free memory windows:%s\n",
                bytes0, sl::SlabGmres::basis_bytes(12, 50),
                sl::SlabModePreconditioner::band_bytes(12), attempts.c_str());
    rep.check(bytes0 == bytes1 && ptr0 == ptr1,
              "no allocation after prepare: workspace bytes and every buffer pointer unchanged "
              "across a continuation and a Newton solve");
    rep.check(free_same,
              "no allocation after prepare: cudaMemGetInfo free memory unchanged "
              "(secondary; device-wide)",
              attempts);
}

} // namespace

int main() {
    TestReport rep;
    CudaContext ctx(0);
    const auto t0 = std::chrono::steady_clock::now();
    try {
        std::printf("=== SF-33 N2: (1) P-A at k = 1, u = 0 ===\n");
        case_pa_k1(rep, ctx);
        std::printf("=== SF-33 N2: (1b) P-A exact for an x1-only state ===\n");
        case_pa_x1_state(rep, ctx);
        std::printf("=== SF-33 N2: (2) GMRES ===\n");
        case_gmres(rep, ctx);
        std::printf("=== SF-33 N2: (3) Newton on the exact pair ===\n");
        case_newton_exact_pair(rep, ctx);
        std::printf("=== SF-33 N2: (4) continuation mechanics (generic3d, 12^3) ===\n");
        case_continuation(rep, ctx);
        std::printf("=== SF-33 N2: (5) statuses ===\n");
        case_statuses(rep, ctx);
        std::printf("=== SF-33 N2: (6) no allocation after prepare ===\n");
        case_no_allocation(rep, ctx);
    } catch (const std::exception& e) {
        std::printf("[FAIL] unexpected exception: %s\n", e.what());
        rep.overall_pass = false;
    }
    const double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("\n=== inlet_slab_newton: %d checks, %s (%.1f s) ===\n", rep.checks,
                rep.overall_pass ? "PASS" : "FAIL", t);
    return rep.overall_pass ? 0 : 1;
}
