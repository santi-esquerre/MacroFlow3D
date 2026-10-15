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
 *      Newton solve and a continuation (cudaMemGetInfo delta printed as [INFO] only: device-wide).
 *
 * SF-33 N7a (linear forcing; the default policy is now `ew`):
 *   - cases 3 and 4 (and the forced linear_failure of case 5) run their N2 fixtures with
 *     `forcing = fixed`; the r_F / r_out histories of the exact pair (N = 12, 16, from x = 0) and
 *     of both stages of the generic3d ladder must equal BITWISE the histories recorded with the
 *     pre-N7a code (head f5b050f, hexadecimal literals below; recorded on sm_86 / CUDA 13.4);
 *   - 3e: `ew` on the exact pair from x = 0: converged, r_F, r_out <= 1e-13 in <= 15 iterations;
 *     every eta_k printed, eta_min <= eta_k <= max(eta_max, 0.5 tol / m_k), and equal to the
 *     Eisenstat-Walker choice-2 value recomputed from the recorded merit history;
 *   - 4e: `ew` on the generic3d ladder to 0.5: converged (PATH 0.25->0.5) and total GMRES
 *     iterations <= the `fixed` total (both printed);
 *   - 4: the forced continuation failure prints the final attempt's own STAGE_END line and the
 *     `CONTINUATION reporting the state of the final attempt` line;
 *   - 5e: `ew` with an unreachable eta (eta0 = eta_max = 1e-6, GMRES cap 1): linear_failure, step
 *     not taken, eta printed on the NEWTON line.
 *
 * SF-33 N7b (pseudo-transient continuation, cfg.psitc; library default off):
 *   - the pre-N7a literal histories (cases 3, 4) are compared with a 1e-10 RELATIVE tolerance
 *     (absolute floor 1e-14 for the roundoff-level entries below the Newton tolerance 1e-13): the
 *     literals were recorded on sm_86 and other platforms (V100 / CUDA 11.4) differ at roundoff;
 *     strict bitwise checks are kept only between runs inside this process;
 *   1s P-A for the SHIFTED operator: ||P_mu^-1 (J + mu D) x - x|| / ||x|| <= 1e-12 for mu in
 *      {0, 1, 100} at k = 1, u = 0 (N = 12, 16) and at the x1-only state of 1b (N = 12); the shift
 *      kernel itself against a host evaluation of mu q_v / h^2 x (equation rows) and 0 (outlet);
 *   3p `psitc off` vs `psitc on` with mu0 = 0 (both forcing fixed, exact pair N = 12): the two code
 *      paths give the same history to 1e-14 relative (bitwise expected, printed);
 *      Psi-tc on (ew) on the exact pair from x = 0, N = 12 and 16: converged, r_F, r_out <= 1e-13,
 *      max |u - u_exact| <= 1e-11; its and the mu sequence printed; mu_k = mu0 m_k / m_0
 *      recomputed from the merit history (1e-14 relative); `mu=` on every LINEAR / NEWTON line;
 *   4p Psi-tc on (ew) on the generic3d ladder to 0.5: converged (r_F, r_out <= 1e-13);
 *   5p forced line-search failure with Psi-tc (lambda_min = 2): 4 retries with mu = 4, 16, 64,
 *      100 (x4, clamped to mu_max), 5 LINEAR lines, then linesearch-fail (run with h_ref = 0, the
 *      unscaled N7b schedule, so the literal mu values stay exact).
 *
 * SF-33 C4 (grid-scaled SER reference shift mu0_eff = mu0 (h / h_ref)^2, default h_ref = 1/16):
 *   - the SER checks recompute mu_k with mu0_eff; 3p (exact pair, the N7b contract mu0 = 1) is
 *     pinned to h_ref = 0; 4p (generic3d 12^3 ladder) runs the default (mu0_eff = 16/9) and must
 *     still converge;
 *   7  psitc_effective_mu0: exactly mu0 at N = 16, exactly 1/4, 1/16, 1/64 of mu0 at N = 32, 64,
 *      128, 16/9 at N = 12 (1e-15 relative), mu0 at every N with h_ref = 0;
 *      N = 16 BITWISE identity of h_ref = 0 vs h_ref = 1/16: Psi-tc (ew) on the exact pair from
 *      x = 0 and on the generic3d stage eps = 0.25 from u = 0: r_F / r_out histories, per-step mu,
 *      mu_ser, eta, GMRES iteration counts and the final iterate x all bitwise equal;
 *      N = 12 exact pair, default h_ref: first-step mu = mu0_eff = 16/9 and the SER sequence
 *      verified (gated); the Newton outcome is RECORDED, not gated: with the larger coarse-grid
 *      shift (16/9 > 1) the exact pair from x = 0 is stopped by the stagnation rule in the slow
 *      phase (observed: its 9, r_F 1.1e-1, mu 6.5e-2) -- the same damping/stagnation interaction
 *      C4 removes on the fine grids; N < 16 is not a production grid. The STAGE lines print
 *      `psitc_mu0_eff=`.
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

#include <algorithm>
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
    // A pageable cudaMemcpy H2D runs on the legacy default stream and may return before the DMA has
    // landed; CudaContext's stream is non-blocking (no implicit ordering with the legacy stream), so
    // a kernel enqueued next on ctx.cuda_stream() could read stale data when the GPU is shared with
    // another process (observed: inlet_slab_jvp failed 11/15 runs under a concurrent GPU process).
    // Test helper: complete the copy device-wide before any stream-ordered work uses the buffer.
    MACROFLOW3D_CUDA_CHECK(cudaDeviceSynchronize());
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
    std::vector<real> freeze(CudaContext& ctx, const std::vector<real>& u, int* singular = nullptr,
                             real mu = 0.0) {
        upload(uvec, u);
        sl::assemble_full_planes(ctx, g, cspan(uvec), in, mspan(U1), mspan(U2));
        sl::evaluate_residual(ctx, g, in, cspan(U1), cspan(U2), mspan(E), rws, nullptr);
        jws.prepare_base(ctx, g, in, cspan(U1), cspan(U2));
        const auto fr = prec.factor(ctx, g, in, cspan(U1), cspan(U2), mu);
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
CaseSpec x1_state_spec() {
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
    return s;
}

void case_pa_x1_state(TestReport& rep, CudaContext& ctx) {
    const CaseSpec s = x1_state_spec();
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
// 1s. SF-33 N7b: P-A for the shifted operator J + mu D
// ------------------------------------------------------------------------------------------------
void case_pa_shift(TestReport& rep, CudaContext& ctx) {
    struct Setup {
        const char* name;
        CaseSpec spec;
        int N;
    };
    const Setup setups[] = {{"k = 1, u = 0", k1_spec(), 12},
                            {"k = 1, u = 0", k1_spec(), 16},
                            {"x1-only state", x1_state_spec(), 12}};
    for (const Setup& su : setups) {
        for (real mu : {0.0, 1.0, 100.0}) {
            OpHarness hs;
            hs.build(ctx, su.N, su.spec);
            int singular = -1;
            hs.freeze(ctx, hs.u_spec, &singular, mu);
            std::mt19937_64 rng(4480 + su.N);
            const auto x = gaussian(rng, hs.g.unknown_size());
            upload(hs.a, x);
            hs.jws.apply(ctx, hs.g, cspan(hs.a), mspan(hs.b));
            if (mu != 0.0)
                sl::slab_add_pseudo_time_shift(ctx, hs.g, hs.in, mu, cspan(hs.a), mspan(hs.b));
            hs.prec.apply(ctx, hs.g, cspan(hs.b), mspan(hs.c));
            ctx.synchronize();
            const real err = rel_diff(download(hs.c.data(), hs.c.size()), x);
            std::printf("[INFO] PA_SHIFT %s N=%d mu=%g |P_mu^-1 (J + mu D) x - x|/|x| = %.3e "
                        "(singular modes %d)\n",
                        su.name, su.N, mu, err, singular);
            rep.check(err <= 1e-12 && singular == 0,
                      std::string("P-A shifted, ") + su.name + ", N = " + std::to_string(su.N) +
                          ", mu = " + fmtd("%g", mu) +
                          ": ||P_mu^-1 ((J + mu D) x) - x|| / ||x|| <= 1e-12",
                      fmtd("%.3e", err));
        }
    }
    {
        // the shift kernel against a host evaluation (q varies with x1 in this state)
        OpHarness hs;
        hs.build(ctx, 12, x1_state_spec());
        const auto& g = hs.g;
        std::mt19937_64 rng(4490);
        const auto x = gaussian(rng, g.unknown_size());
        const auto y0 = gaussian(rng, g.unknown_size());
        upload(hs.a, x);
        upload(hs.b, y0);
        const real mu = 2.5;
        sl::slab_add_pseudo_time_shift(ctx, g, hs.in, mu, cspan(hs.a), mspan(hs.b));
        ctx.synchronize();
        const auto y = download(hs.b.data(), hs.b.size());
        const auto q = download(hs.in.q.data(), hs.in.q.size());
        const std::size_t nf = g.field_size(), np = g.plane_size();
        real worst = 0.0;
        bool outlet_untouched = true;
        for (std::size_t i = 0; i < 2 * nf; ++i) {
            const std::size_t u = i < nf ? i : i - nf;
            if (u >= nf - np) {
                outlet_untouched = outlet_untouched && y[i] == y0[i];
                continue;
            }
            const real want = y0[i] + mu * q[np + u] / (g.h * g.h) * x[i];
            worst = std::fmax(worst, std::fabs(y[i] - want) / std::fmax(std::fabs(want), 1e-300));
        }
        rep.check(worst <= 1e-14 && outlet_untouched,
                  "Psi-tc shift kernel: y += mu q_v / h^2 x on the equation rows (1e-14 "
                  "relative), outlet rows untouched",
                  fmtd("max rel %.2e", worst));
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
// SF-33 N7a: pre-N7a histories (forcing = fixed) and forcing checks
// ------------------------------------------------------------------------------------------------
struct RefHistory {
    std::vector<real> r_F, r_out;
};

/// Recorded with the pre-N7a code (head f5b050f, Release, sm_86, CUDA 13.4), `%a` of hist_r_F /
/// hist_r_out of the N2 fixtures. Deterministic reductions (SlabGmres.cuh): identical on reruns.
RefHistory pre_n7a_exact_pair(int N) {
    if (N == 12)
        return {{0x1.c0902c0b9f0fbp+1, 0x1.16d6929c4842p-1, 0x1.5c5d9140910bp-2,
                 0x1.678a56a358d6p-6, 0x1.6664f27cdaadp-13, 0x1.2fc3a65ba1dd7p-27,
                 0x1.4a57bea3b1e7cp-48},
                {0x0p+0, 0x1.b22341da17a4fp-49, 0x1.f453a41555a53p-51, 0x1.3d7e79bff3944p-51,
                 0x1.480d896470f93p-51, 0x1.2e77baf6723cp-51, 0x1.6ea69871f4b3p-51}};
    return {{0x1.6dbc168c68bb1p+2, 0x1.7c415861e3d7bp-1, 0x1.528202fbeb062p-2, 0x1.403ff206252e9p-7,
             0x1.f47d33aac176bp-16, 0x1.df1c76a6b8a94p-33, 0x1.9886ebec5f4ddp-47},
            {0x0p+0, 0x1.0aa994dc3ff0ep-49, 0x1.db28003530fdap-51, 0x1.d414751830e66p-51,
             0x1.711c4b81e0d8p-51, 0x1.94380f89fdab7p-51, 0x1.c7fe396f60808p-51}};
}

RefHistory pre_n7a_generic3d_stage(int stage) {
    if (stage == 0)
        return {{0x1.89a44a8caab62p+0, 0x1.58c8474184ac3p-4, 0x1.cdf1022e06bc8p-7,
                 0x1.0466ab0174a66p-11, 0x1.04c57a2547936p-20, 0x1.57542124b0772p-40,
                 0x1.19f87b585f116p-50},
                {0x0p+0, 0x1.90454364ae941p-53, 0x1.58023e81ec01p-52, 0x1.70a4c74515d72p-53,
                 0x1.3150bb2ad8fa1p-53, 0x1.3b7656bb3bbbdp-53, 0x1.4ee6b8ee50028p-53}};
    return {{0x1.64c14d7a362fap+0, 0x1.8ae789ccb33ccp-4, 0x1.dd1fac41e247fp-5, 0x1.c9ddc931fd165p-6,
             0x1.15e7e41d06e4p-10, 0x1.6ec4549d7956cp-17, 0x1.79afae88809a4p-33,
             0x1.69d57bb15ad2p-49},
            {0x1.4ee6b8ee50028p-53, 0x1.26fad4e7f1a3bp-51, 0x1.1d3a0d2454ae7p-51,
             0x1.07889b065e637p-51, 0x1.0ffb9aa2c166bp-51, 0x1.ddbc7ac44645bp-52,
             0x1.10b597794f8eep-51, 0x1.17aee9b91746ap-51}};
}

/// Comparison of a report's histories with a reference: equal lengths and every entry within
/// |a - b| <= rel_tol |b| + abs_floor (rel_tol = 0, abs_floor = 0: bitwise); det: sizes, first
/// mismatch beyond the tolerance and max relative difference.
bool history_close(const sl::SlabNewtonReport& r, const RefHistory& ref, real rel_tol,
                   real abs_floor, std::string& det) {
    bool same = r.hist_r_F.size() == ref.r_F.size() && r.hist_r_out.size() == ref.r_out.size();
    real maxrel = 0.0;
    int first = -1;
    const std::size_t n = std::min(r.hist_r_F.size(), ref.r_F.size());
    for (std::size_t i = 0; i < n; ++i) {
        const real a[2] = {r.hist_r_F[i], i < r.hist_r_out.size() ? r.hist_r_out[i] : -1.0};
        const real b[2] = {ref.r_F[i], i < ref.r_out.size() ? ref.r_out[i] : -1.0};
        for (int q = 0; q < 2; ++q) {
            if (!(std::fabs(a[q] - b[q]) <= rel_tol * std::fabs(b[q]) + abs_floor)) {
                same = false;
                if (first < 0)
                    first = static_cast<int>(i);
            }
            if (b[q] != 0.0)
                maxrel = std::fmax(maxrel, std::fabs(a[q] - b[q]) / std::fabs(b[q]));
        }
    }
    char buf[200];
    std::snprintf(buf, sizeof(buf),
                  "entries %zu/%zu (ref %zu/%zu), first mismatch %d, max rel %.2e (tol rel %.0e "
                  "abs %.0e)",
                  r.hist_r_F.size(), r.hist_r_out.size(), ref.r_F.size(), ref.r_out.size(), first,
                  maxrel, rel_tol, abs_floor);
    det = buf;
    return same;
}

/// Prints the eta sequence of a report; returns true iff every eta_k equals the Eisenstat-Walker
/// choice-2 value recomputed from the merit history (relative 1e-14) and lies in
/// [eta_min, max(eta_max, 0.5 tol / m_k)].
bool check_ew_etas(const sl::SlabNewtonReport& r, const sl::SlabNewtonConfig& cfg,
                   const std::string& name, std::string& det) {
    bool ok = !r.steps.empty() && r.forcing == sl::SlabForcing::ew;
    std::string seq;
    real m_prev = 0.0, eta_prev = 0.0;
    for (std::size_t k = 0; k < r.steps.size(); ++k) {
        if (k >= r.hist_r_F.size()) {
            ok = false;
            break;
        }
        const real m = std::sqrt(r.hist_r_F[k] * r.hist_r_F[k] + r.hist_r_out[k] * r.hist_r_out[k]);
        real e = cfg.ew.eta0;
        if (k >= 1) {
            e = cfg.ew.gamma * std::pow(m / m_prev, cfg.ew.alpha);
            const real sg = cfg.ew.gamma * std::pow(eta_prev, cfg.ew.alpha);
            if (sg > 0.1)
                e = std::fmax(e, sg);
        }
        e = std::fmax(std::fmax(std::fmin(e, cfg.ew.eta_max), cfg.gmres.tol), 0.5 * cfg.tol / m);
        const real eta = r.steps[k].eta;
        const real hi = std::fmax(cfg.ew.eta_max, 0.5 * cfg.tol / m);
        if (!(std::fabs(eta - e) <= 1e-14 * e) || eta < cfg.gmres.tol || eta > hi)
            ok = false;
        char b[64];
        std::snprintf(b, sizeof(b), "%s%.2e", k ? "," : "", eta);
        seq += b;
        m_prev = m;
        eta_prev = eta;
    }
    std::printf("[INFO] %s: eta sequence (forcing ew) %s\n", name.c_str(), seq.c_str());
    det = "etas " + seq;
    return ok;
}

/// Literal (sm_86) histories vs this platform: 1e-10 relative, 1e-14 absolute floor (SF-33 N7b).
constexpr real kLitRel = 1e-10;
constexpr real kLitAbs = 1e-14;

RefHistory history_of(const sl::SlabNewtonReport& r) {
    return {r.hist_r_F, r.hist_r_out};
}

/// Prints the mu sequence of a Psi-tc report (`^` = line-search retry) and checks the SER values
/// mu_k = clamp(mu0_eff m_k / m_0, 0, mu_max) against the merit history (1e-14 relative);
/// mu0_eff = mu0 (h / h_ref)^2 (SF-33 C4).
bool check_ser_mus(const sl::SlabNewtonReport& r, const sl::SlabNewtonConfig& cfg, real h,
                   const std::string& name, std::string& det) {
    const real mu0_eff = sl::psitc_effective_mu0(cfg.psitc, h); // SF-33 C4
    bool ok = r.psitc && !r.steps.empty();
    std::string seq;
    auto merit = [&](std::size_t k) {
        return std::sqrt(r.hist_r_F[k] * r.hist_r_F[k] + r.hist_r_out[k] * r.hist_r_out[k]);
    };
    for (std::size_t k = 0; k < r.steps.size() && k < r.hist_r_F.size(); ++k) {
        const real want =
            std::fmin(std::fmax(mu0_eff * (merit(k) / merit(0)), 0.0), cfg.psitc.mu_max);
        const real got = r.steps[k].mu_ser;
        if (!(std::fabs(got - want) <= 1e-14 * std::fmax(want, 1e-300)))
            ok = false;
        char b[64];
        std::snprintf(b, sizeof(b), "%s%.2e", k ? "," : "", got);
        seq += b;
        for (real mr : r.steps[k].mu_retries) {
            std::snprintf(b, sizeof(b), "^%.2e", mr);
            seq += b;
        }
    }
    std::printf("[INFO] %s: mu sequence (Psi-tc SER) %s\n", name.c_str(), seq.c_str());
    det = "mus " + seq;
    return ok;
}

int total_solves(const sl::SlabNewtonReport& r) {
    int n = 0;
    for (const auto& st : r.steps)
        n += static_cast<int>(st.linear_its_solves.size());
    return n;
}

int total_gmres(const sl::SlabContinuationReport& r) {
    int t = 0;
    for (const auto& s : r.stages)
        t += s.newton.linear_iterations_total;
    return t;
}

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
        cfg.forcing = sl::SlabForcing::fixed; // the N2 fixture (pre-N7a behaviour)
        Capture cap;
        const auto r =
            nk.solve(ctx, in, mspan(x), "exact_pair:0.7:" + std::to_string(N), cfg, cap.logger());
        {
            std::string hd;
            const bool same = history_close(r, pre_n7a_exact_pair(N), kLitRel, kLitAbs, hd);
            rep.check(same,
                      "forcing = fixed, exact pair N = " + std::to_string(N) +
                          ": r_F / r_out history equal to the pre-N7a record (sm_86 literals) "
                          "within 1e-10 relative",
                      hd);
        }
        {
            // 3e: inexact Newton (Eisenstat-Walker) on the same problem from x = 0
            upload(x, std::vector<real>(g.unknown_size(), 0.0));
            sl::SlabNewtonConfig ecfg;
            ecfg.forcing = sl::SlabForcing::ew;
            Capture cape;
            const auto re = nk.solve(ctx, in, mspan(x), "exact_pair_ew:0.7:" + std::to_string(N),
                                     ecfg, cape.logger());
            const auto xe = download(x.data(), x.size());
            real maxerr_e = 0.0;
            for (std::size_t i = 0; i < xe.size(); ++i)
                maxerr_e = std::fmax(maxerr_e, std::fabs(xe[i] - u_ex[i]));
            std::string ld;
            for (const auto& s : re.steps)
                ld += " " + std::to_string(s.linear.iterations);
            char de[260];
            std::snprintf(de, sizeof(de),
                          "status %s its %d r_F %.3e r_out %.3e max|u-u_ex| %.3e GMRES its/step:%s "
                          "(total %d)",
                          sl::to_string(re.status), re.its, re.r_F, re.r_out, maxerr_e, ld.c_str(),
                          re.linear_iterations_total);
            rep.check(re.status == sl::SlabSolveStatus::converged && re.r_F <= 1e-13 &&
                          re.r_out <= 1e-13 && re.its <= 15 && maxerr_e <= 1e-11,
                      "forcing = ew, exact pair N = " + std::to_string(N) +
                          " from x = 0: converged, r_F, r_out <= 1e-13, <= 15 iterations, max "
                          "|u - u_exact| <= 1e-11",
                      de);
            std::string ed;
            const bool eok = check_ew_etas(re, ecfg, "exact pair N = " + std::to_string(N), ed);
            rep.check(eok && cape.count(" eta=") == static_cast<int>(re.steps.size()),
                      "forcing = ew, exact pair N = " + std::to_string(N) +
                          ": every eta_k printed, eta_min <= eta_k <= max(eta_max, 0.5 tol/m_k), "
                          "equal to the EW choice-2 value",
                      ed);
            std::printf("[INFO] exact pair N = %d: GMRES total fixed %d, ew %d; Newton its fixed "
                        "%d, ew %d\n",
                        N, r.linear_iterations_total, re.linear_iterations_total, r.its, re.its);
        }
        if (N == 12) {
            // 3p (SF-33 N7b): psitc off vs psitc on with mu0 = 0 (both fixed forcing): the two
            // code paths with mu = 0 give the same history (in-process: bitwise expected)
            sl::SlabNewtonConfig c0 = cfg; // fixed, psitc off (library default)
            sl::SlabNewtonConfig c1 = cfg;
            c1.psitc.enabled = true;
            c1.psitc.mu0 = 0.0;
            const sl::SlabLogger quiet = [](const std::string&) {};
            upload(x, std::vector<real>(g.unknown_size(), 0.0));
            const auto r0 = nk.solve(ctx, in, mspan(x), "psitc_off", c0, quiet);
            upload(x, std::vector<real>(g.unknown_size(), 0.0));
            Capture capm;
            const auto r1 = nk.solve(ctx, in, mspan(x), "psitc_mu0_0", c1, capm.logger());
            std::string d14, dbit;
            const bool close = history_close(r1, history_of(r0), 1e-14, 0.0, d14);
            const bool bitwise = history_close(r1, history_of(r0), 0.0, 0.0, dbit);
            bool mus_zero = r1.psitc && !r1.steps.empty();
            for (const auto& st : r1.steps)
                mus_zero = mus_zero && st.mu == 0.0 && st.mu_retries.empty();
            std::printf("[INFO] exact pair N = 12: psitc off vs psitc on (mu0 = 0), forcing fixed: "
                        "bitwise %s (%s)\n",
                        bitwise ? "yes" : "no", dbit.c_str());
            rep.check(close && mus_zero && r0.status == r1.status && r0.its == r1.its &&
                          r0.linear_iterations_total == r1.linear_iterations_total,
                      "psitc off vs psitc on with mu0 = 0 (forcing fixed, exact pair N = 12): "
                      "same history to 1e-14 relative, same its / GMRES total, every mu = 0",
                      d14);
            // in-process bitwise reproducibility of the psitc-off fixed run (strict check)
            std::string drep;
            rep.check(history_close(r0, history_of(r), 0.0, 0.0, drep),
                      "forcing fixed, psitc off: in-process rerun bitwise equal", drep);
        }
        {
            // 3p (SF-33 N7b): Psi-tc on (ew forcing, SER mu0 = 1) on the exact pair from x = 0
            sl::SlabNewtonKrylov nkp;
            nkp.prepare(ctx, g, 50, 120);
            upload(x, std::vector<real>(g.unknown_size(), 0.0));
            sl::SlabNewtonConfig pcfg;
            pcfg.forcing = sl::SlabForcing::ew;
            pcfg.psitc.enabled = true;
            // SF-33 C4: this is the N7b contract (mu0 = 1 at every N): pinned to the unscaled
            // schedule. The grid-scaled default is exercised in case 7 (at N = 16 it is bitwise
            // this schedule; at N = 12 it is mu0_eff = 16/9, recorded there).
            pcfg.psitc.h_ref = 0.0;
            pcfg.max_iterations = 120;
            Capture capp;
            const auto rp =
                nkp.solve(ctx, in, mspan(x), "exact_pair_psitc:0.7:" + std::to_string(N), pcfg,
                          capp.logger());
            const auto xp = download(x.data(), x.size());
            real maxerr_p = 0.0;
            for (std::size_t i = 0; i < xp.size(); ++i)
                maxerr_p = std::fmax(maxerr_p, std::fabs(xp[i] - u_ex[i]));
            std::string md;
            const bool mok =
                check_ser_mus(rp, pcfg, g.h, "exact pair Psi-tc N = " + std::to_string(N), md);
            char dp[300];
            std::snprintf(dp, sizeof(dp),
                          "status %s its %d r_F %.3e r_out %.3e max|u-u_ex| %.3e GMRES total %d "
                          "retries %d",
                          sl::to_string(rp.status), rp.its, rp.r_F, rp.r_out, maxerr_p,
                          rp.linear_iterations_total, rp.linesearch_retries);
            std::printf("[INFO] exact pair Psi-tc N = %d: %s\n", N, dp);
            rep.check(rp.status == sl::SlabSolveStatus::converged && rp.r_F <= 1e-13 &&
                          rp.r_out <= 1e-13 && maxerr_p <= 1e-11,
                      "Psi-tc on (ew), exact pair N = " + std::to_string(N) +
                          " from x = 0: converged, r_F, r_out <= 1e-13, max |u - u_exact| <= "
                          "1e-11",
                      dp);
            rep.check(mok && capp.count("LINEAR gmres+P-A") == total_solves(rp) &&
                          capp.count(" mu=") == total_solves(rp) +
                                                    static_cast<int>(rp.hist_r_F.size()) - 1 +
                                                    rp.linesearch_retries,
                      "Psi-tc on, exact pair N = " + std::to_string(N) +
                          ": mu_k = mu0 m_k / m_0 (SER, merit norm) on every step, mu= on every "
                          "LINEAR / NEWTON line",
                      md);
        }
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
    ncfg.forcing = sl::SlabForcing::fixed; // the N2 fixture (pre-N7a behaviour)
    sl::SlabContinuationConfig ccfg;
    ccfg.field = "generic3d";
    int fixed_total = -1;
    {
        Capture cap;
        const auto r =
            nk.solve_with_continuation(ctx, 0.5, provider, mspan(x), ccfg, ncfg, cap.logger());
        fixed_total = total_gmres(r);
        for (int st = 0; st < 2; ++st) {
            std::string hd = "stage missing";
            const bool same = r.stages.size() == 2 &&
                              history_close(r.stages[st].newton, pre_n7a_generic3d_stage(st),
                                            kLitRel, kLitAbs, hd);
            rep.check(same,
                      "forcing = fixed, generic3d ladder stage " + std::to_string(st) +
                          ": r_F / r_out history equal to the pre-N7a record (sm_86 literals) "
                          "within 1e-10 relative",
                      hd);
        }
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
        // 4e: the same ladder with inexact Newton (Eisenstat-Walker)
        sl::SlabNewtonConfig ecfg;
        ecfg.forcing = sl::SlabForcing::ew;
        Capture cap;
        const auto r =
            nk.solve_with_continuation(ctx, 0.5, provider, mspan(x), ccfg, ecfg, cap.logger());
        const int ew_total = total_gmres(r);
        std::string its, ed_all;
        bool eok = !r.stages.empty();
        for (const auto& st : r.stages) {
            its += " " + std::to_string(st.newton.its);
            std::string ed;
            eok = check_ew_etas(st.newton, ecfg, "generic3d ew stage eps=" + fmtd("%g", st.eps),
                                ed) &&
                  eok;
            ed_all += " [" + ed + "]";
        }
        std::printf("[INFO] generic3d 12^3 ladder to 0.5: total GMRES iterations fixed %d, ew %d; "
                    "Newton its per stage (ew):%s\n",
                    fixed_total, ew_total, its.c_str());
        rep.check(r.status == sl::SlabSolveStatus::converged && r.path == "0.25->0.5" &&
                      r.final_newton.r_F <= 1e-13 && r.final_newton.r_out <= 1e-13,
                  "forcing = ew, continuation generic3d 12^3 to eps = 0.5: converged, PATH "
                  "0.25->0.5",
                  "path " + r.path + " status " + sl::to_string(r.status));
        rep.check(ew_total <= fixed_total && fixed_total > 0,
                  "forcing = ew, continuation generic3d: total GMRES iterations <= fixed total",
                  "ew " + std::to_string(ew_total) + " fixed " + std::to_string(fixed_total));
        rep.check(eok, "forcing = ew, continuation generic3d: eta_k recomputed and bounded",
                  ed_all);
    }
    {
        // 4p (SF-33 N7b): the same ladder with Psi-tc on (ew forcing)
        sl::SlabNewtonKrylov nkp;
        nkp.prepare(ctx, g, 50, 120);
        sl::SlabNewtonConfig pcfg;
        pcfg.forcing = sl::SlabForcing::ew;
        pcfg.psitc.enabled = true;
        pcfg.max_iterations = 120;
        Capture cap;
        const auto r =
            nkp.solve_with_continuation(ctx, 0.5, provider, mspan(x), ccfg, pcfg, cap.logger());
        std::string its, md_all;
        bool mok = !r.stages.empty();
        for (const auto& st : r.stages) {
            its += " " + std::to_string(st.newton.its);
            std::string md;
            mok = check_ser_mus(st.newton, pcfg, g.h,
                                "generic3d Psi-tc stage eps=" + fmtd("%g", st.eps), md) &&
                  mok;
            md_all += " [" + md + "]";
        }
        std::printf("[INFO] generic3d 12^3 ladder to 0.5, Psi-tc: status %s path %s, total GMRES "
                    "%d, Newton its per stage:%s\n",
                    sl::to_string(r.status), r.path.c_str(), total_gmres(r), its.c_str());
        rep.check(r.status == sl::SlabSolveStatus::converged && r.final_newton.r_F <= 1e-13 &&
                      r.final_newton.r_out <= 1e-13 && mok,
                  "Psi-tc on (ew), continuation generic3d 12^3 to eps = 0.5: converged, r_F, "
                  "r_out <= 1e-13, SER mu sequence verified",
                  "path " + r.path + md_all);
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
                      cap.count("STAGE_END field=generic3d eps=0.5 N=12 cand=i1o4") == 1 &&
                      cap.count("(final attempt) -> FAILED") == 1 &&
                      cap.count("CONTINUATION reporting the state of the final attempt at "
                                "eps=0.5") == 1 &&
                      r.path == expect &&
                      cap.count("PATH field=generic3d eps=0.5 N=12 cand=i1o4 continuation path: " +
                                expect) == 1,
                  "continuation forced failure: exactly one bisection, continuation_floor, final "
                  "attempt with its own STAGE_END, reporting line after it, PATH " +
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
        cfg.forcing = sl::SlabForcing::fixed; // lin_tol: unreachable in 1 iteration
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
        // 5e: ew with an unreachable forcing term within the cap: linear_failure, never a silent
        // unconverged step
        sl::SlabNewtonConfig cfg;
        cfg.forcing = sl::SlabForcing::ew;
        cfg.ew.eta0 = 1e-6;
        cfg.ew.eta_max = 1e-6;
        cfg.gmres.max_iterations = 1;
        Capture cap;
        const auto r = run(cfg, "lincap_ew", cap);
        const bool eta_ok = r.steps.size() == 1 && r.steps[0].eta == 1e-6 &&
                            !(r.steps[0].linear.rel_residual <= 1e-6);
        rep.check(r.status == sl::SlabSolveStatus::linear_failure &&
                      r.last_linear_status == sl::SlabLinearStatus::max_iterations && eta_ok &&
                      cap.count("(step not taken) eta=1.00e-06") == 1 && r.hist_r_F.size() == 1,
                  "forced linear_failure (ew, eta = 1e-6, GMRES cap 1): linear_failure, step not "
                  "taken, eta printed",
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
        // 5p (SF-33 N7b): Psi-tc retries on line-search failure: mu x4 (clamped to mu_max = 100)
        // at most 4 times, then linesearch-fail
        sl::SlabNewtonConfig cfg;
        cfg.lambda_min = 2.0;
        cfg.psitc.enabled = true;
        cfg.psitc.h_ref = 0.0; // SF-33 C4: unscaled schedule (mu0_eff = mu0 = 1 exactly at N = 12)
        Capture cap;
        const auto r = run(cfg, "ls_psitc", cap);
        const std::vector<real> want = {4.0, 16.0, 64.0, 100.0};
        const bool seq_ok = r.steps.size() == 1 && r.steps[0].mu_ser == 1.0 &&
                            r.steps[0].mu_retries == want && r.steps[0].mu == 100.0 &&
                            r.steps[0].linear_its_solves.size() == 5;
        rep.check(r.status == sl::SlabSolveStatus::linesearch_fail && seq_ok &&
                      r.linesearch_retries == 4 && cap.count("Psi-tc retry") == 4 &&
                      cap.count("Psi-tc retry 4/4 with mu=1.000e+02") == 1 &&
                      cap.count("LINEAR gmres+P-A") == 5 &&
                      cap.count("line search failed (min lambda 1/1024): merit") == 5 &&
                      r.hist_r_F.size() == 1,
                  "forced linesearch-fail with Psi-tc (lambda_min = 2): 4 retries mu = 4, 16, 64, "
                  "100 (clamped), 5 LINEAR lines, then linesearch-fail",
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
    // cudaMemGetInfo is device-wide: other processes sharing the GPU change it, so it is printed
    // as [INFO] only; the gate is the workspace bytes / buffer pointers (unaffected by others).
    std::string attempts;
    {
        std::size_t free0 = 0, free1 = 0, total = 0;
        ctx.synchronize();
        MACROFLOW3D_CUDA_CHECK(cudaMemGetInfo(&free0, &total));
        const auto r = nk.solve_with_continuation(ctx, 0.5, provider, mspan(x), ccfg, ncfg, quiet);
        upload(x, std::vector<real>(g.unknown_size(), 0.0));
        const auto r2 = nk.solve(ctx, prov(0.25), mspan(x), "alloc", ncfg, quiet);
        sl::SlabNewtonConfig pcfg = ncfg; // SF-33 N7b: the Psi-tc path allocates nothing either
        pcfg.psitc.enabled = true;
        upload(x, std::vector<real>(g.unknown_size(), 0.0));
        nk.solve(ctx, prov(0.25), mspan(x), "alloc_psitc", pcfg, quiet);
        ctx.synchronize();
        MACROFLOW3D_CUDA_CHECK(cudaMemGetInfo(&free1, &total));
        attempts += " [" + std::to_string(free0) + " -> " + std::to_string(free1) + " (delta " +
                    std::to_string(static_cast<long long>(free0) - static_cast<long long>(free1)) +
                    ", device-wide), " + sl::to_string(r.status) + "/" + sl::to_string(r2.status) +
                    "]";
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
}

// ------------------------------------------------------------------------------------------------
// 7. SF-33 C4: grid-scaled SER reference shift mu0_eff = mu0 (h / h_ref)^2
// ------------------------------------------------------------------------------------------------
/// Bitwise comparison of two Psi-tc reports (histories, per-step mu / mu_ser / eta / GMRES its).
bool reports_bitwise(const sl::SlabNewtonReport& a, const sl::SlabNewtonReport& b,
                     std::string& det) {
    bool ok = a.status == b.status && a.its == b.its && a.hist_r_F == b.hist_r_F &&
              a.hist_r_out == b.hist_r_out && a.steps.size() == b.steps.size() &&
              a.linear_iterations_total == b.linear_iterations_total &&
              a.linesearch_retries == b.linesearch_retries;
    std::string mus;
    for (std::size_t k = 0; k < a.steps.size() && k < b.steps.size(); ++k) {
        const auto& sa = a.steps[k];
        const auto& sb = b.steps[k];
        ok = ok && sa.mu == sb.mu && sa.mu_ser == sb.mu_ser && sa.eta == sb.eta &&
             sa.mu_retries == sb.mu_retries && sa.linear_its_solves == sb.linear_its_solves;
        char m[40];
        std::snprintf(m, sizeof(m), "%s%.3e", k ? "," : "", sa.mu_ser);
        mus += m;
    }
    char d[160];
    std::snprintf(d, sizeof(d), "status %s its %d GMRES total %d entries %zu mus ",
                  sl::to_string(a.status), a.its, a.linear_iterations_total, a.hist_r_F.size());
    det = d + mus;
    return ok;
}

void case_psitc_grid_scaling(TestReport& rep, CudaContext& ctx) {
    {
        sl::SlabPsitcConfig p; // default h_ref = 1/16, mu0 = 1
        p.mu0 = 1.0;
        const real e16 = sl::psitc_effective_mu0(p, sl::InletSlabGrid::make(16).h);
        const real e32 = sl::psitc_effective_mu0(p, sl::InletSlabGrid::make(32).h);
        const real e64 = sl::psitc_effective_mu0(p, sl::InletSlabGrid::make(64).h);
        const real e128 = sl::psitc_effective_mu0(p, sl::InletSlabGrid::make(128).h);
        const real e12 = sl::psitc_effective_mu0(p, sl::InletSlabGrid::make(12).h);
        sl::SlabPsitcConfig p0 = p;
        p0.h_ref = 0.0;
        p0.mu0 = 0.7;
        bool off_ok = true;
        for (int N : {12, 16, 32, 64, 128})
            off_ok = off_ok && sl::psitc_effective_mu0(p0, sl::InletSlabGrid::make(N).h) == 0.7;
        char d[200];
        std::snprintf(d, sizeof(d), "N=12 %.17g N=16 %.17g N=32 %.17g N=64 %.17g N=128 %.17g", e12,
                      e16, e32, e64, e128);
        rep.check(e16 == 1.0 && e32 == 0.25 && e64 == 1.0 / 16.0 && e128 == 1.0 / 64.0 &&
                      std::fabs(e12 - 16.0 / 9.0) <= 1e-15 * (16.0 / 9.0) &&
                      sl::SlabPsitcConfig().h_ref == 1.0 / 16.0 && off_ok,
                  "C4 psitc_effective_mu0: mu0 (h / h_ref)^2 = 1, 1/4, 1/16, 1/64 at N = 16, 32, "
                  "64, 128 (exact), 16/9 at N = 12; default h_ref = 1/16; h_ref = 0 -> mu0",
                  d);
    }
    // N = 16 bitwise identity: h_ref = 0 (unscaled) vs h_ref = 1/16 (default)
    {
        const auto g = sl::InletSlabGrid::make(16);
        sl::SlabStageInputs in;
        fill_inputs(ctx, g, in, exact_pair_spec());
        sl::SlabNewtonKrylov nk;
        nk.prepare(ctx, g, 50, 120);
        DeviceBuffer<real> x(g.unknown_size());
        sl::SlabNewtonConfig c_on;
        c_on.forcing = sl::SlabForcing::ew;
        c_on.psitc.enabled = true;
        c_on.max_iterations = 120;
        sl::SlabNewtonConfig c_off = c_on;
        c_off.psitc.h_ref = 0.0;
        const sl::SlabLogger quiet = [](const std::string&) {};
        upload(x, std::vector<real>(g.unknown_size(), 0.0));
        const auto r0 = nk.solve(ctx, in, mspan(x), "c4_href0", c_off, quiet);
        const auto x0 = download(x.data(), x.size());
        upload(x, std::vector<real>(g.unknown_size(), 0.0));
        const auto r1 = nk.solve(ctx, in, mspan(x), "c4_href16", c_on, quiet);
        const auto x1 = download(x.data(), x.size());
        std::string d;
        const bool same = reports_bitwise(r1, r0, d) && x0 == x1;
        std::printf("[INFO] C4 exact pair N = 16 (h_ref 1/16 vs 0): %s\n", d.c_str());
        rep.check(same && r1.status == sl::SlabSolveStatus::converged,
                  "C4 N = 16 exact pair Psi-tc (ew) from x = 0: h_ref = 1/16 vs h_ref = 0 bitwise "
                  "identical (histories, mu, mu_ser, eta, GMRES its, final x), converged",
                  d);
    }
    {
        const auto g = sl::InletSlabGrid::make(16);
        Generic3dProvider prov;
        prov.build(ctx, g);
        const sl::StageInputProvider provider = [&](real amp) -> const sl::SlabStageInputs& {
            return prov(amp);
        };
        sl::SlabNewtonKrylov nk;
        nk.prepare(ctx, g, 50, 120);
        DeviceBuffer<real> x(g.unknown_size());
        sl::SlabNewtonConfig c_on;
        c_on.forcing = sl::SlabForcing::ew;
        c_on.psitc.enabled = true;
        c_on.max_iterations = 120;
        sl::SlabNewtonConfig c_off = c_on;
        c_off.psitc.h_ref = 0.0;
        sl::SlabContinuationConfig ccfg;
        ccfg.field = "generic3d";
        Capture cap0, cap1;
        const auto r0 =
            nk.solve_with_continuation(ctx, 0.25, provider, mspan(x), ccfg, c_off, cap0.logger());
        const auto x0 = download(x.data(), x.size());
        const auto r1 =
            nk.solve_with_continuation(ctx, 0.25, provider, mspan(x), ccfg, c_on, cap1.logger());
        const auto x1 = download(x.data(), x.size());
        bool same = r0.status == r1.status && r0.path == r1.path &&
                    r0.stages.size() == r1.stages.size() && x0 == x1;
        std::string d_all;
        for (std::size_t s = 0; s < r0.stages.size() && s < r1.stages.size(); ++s) {
            std::string d;
            same = reports_bitwise(r1.stages[s].newton, r0.stages[s].newton, d) && same;
            d_all += " [" + d + "]";
        }
        std::printf("[INFO] C4 generic3d 16^3 eps 0.25 (h_ref 1/16 vs 0): status %s path %s%s\n",
                    sl::to_string(r1.status), r1.path.c_str(), d_all.c_str());
        rep.check(same && r1.status == sl::SlabSolveStatus::converged,
                  "C4 N = 16 generic3d stage eps = 0.25 Psi-tc (ew) from u = 0: h_ref = 1/16 vs "
                  "h_ref = 0 bitwise identical (stage histories, mu, eta, GMRES its, final x)",
                  "path " + r1.path + d_all);
        rep.check(cap1.count("psitc_mu0_eff=1.000000e+00 (mu0=1 h_ref=0.0625 h=0.0625)") ==
                          static_cast<int>(r1.stages.size()) &&
                      cap0.count("psitc_mu0_eff=1.000000e+00 (mu0=1 h_ref=0 h=0.0625)") ==
                          static_cast<int>(r0.stages.size()),
                  "C4 STAGE lines print psitc_mu0_eff (N = 16: 1 for both h_ref)");
    }
    // N = 12: effective mu0 = 16/9 on the first step; exact-pair Newton still converges
    {
        const auto g = sl::InletSlabGrid::make(12);
        sl::SlabStageInputs in;
        const auto u_ex = fill_inputs(ctx, g, in, exact_pair_spec());
        sl::SlabNewtonKrylov nk;
        nk.prepare(ctx, g, 50, 120);
        DeviceBuffer<real> x(g.unknown_size());
        upload(x, std::vector<real>(g.unknown_size(), 0.0));
        sl::SlabNewtonConfig cfg;
        cfg.forcing = sl::SlabForcing::ew;
        cfg.psitc.enabled = true;
        cfg.max_iterations = 120;
        Capture cap;
        const auto r = nk.solve(ctx, in, mspan(x), "c4_exact_pair:12", cfg, cap.logger());
        const auto xh = download(x.data(), x.size());
        real maxerr = 0.0;
        for (std::size_t i = 0; i < xh.size(); ++i)
            maxerr = std::fmax(maxerr, std::fabs(xh[i] - u_ex[i]));
        const real mu0_eff = sl::psitc_effective_mu0(cfg.psitc, g.h);
        std::string md;
        const bool mok = check_ser_mus(r, cfg, g.h, "C4 exact pair N = 12", md);
        char d[260];
        std::snprintf(d, sizeof(d),
                      "mu0_eff %.17g first mu %.17g status %s its %d r_F %.3e r_out %.3e "
                      "max|u-u_ex| %.3e GMRES total %d",
                      mu0_eff, r.steps.empty() ? -1.0 : r.steps[0].mu_ser, sl::to_string(r.status),
                      r.its, r.r_F, r.r_out, maxerr, r.linear_iterations_total);
        rep.check(!r.steps.empty() && r.steps[0].mu_ser == mu0_eff &&
                      std::fabs(mu0_eff - 16.0 / 9.0) <= 1e-15 * (16.0 / 9.0) && mok,
                  "C4 N = 12 exact pair Psi-tc (ew, default h_ref): first mu = mu0_eff = 16/9, SER "
                  "sequence verified with mu0_eff",
                  d);
        // RECORDED, not gated (see the file header): Newton outcome with mu0_eff = 16/9 at N = 12
        std::printf("[INFO] C4 exact pair N = 12, default h_ref (mu0_eff = 16/9): Newton outcome "
                    "(recorded, not gated): %s; converged to the exact pair: %s\n",
                    d, (r.status == sl::SlabSolveStatus::converged && r.r_F <= 1e-13 &&
                        r.r_out <= 1e-13 && maxerr <= 1e-11)
                           ? "yes"
                           : "no");
    }
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
        std::printf("=== SF-33 N7b: (1s) P-A for the shifted operator J + mu D ===\n");
        case_pa_shift(rep, ctx);
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
        std::printf("=== SF-33 C4: (7) grid-scaled Psi-tc reference shift ===\n");
        case_psitc_grid_scaling(rep, ctx);
    } catch (const std::exception& e) {
        std::printf("[FAIL] unexpected exception: %s\n", e.what());
        rep.overall_pass = false;
    }
    const double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("\n=== inlet_slab_newton: %d checks, %s (%.1f s) ===\n", rep.checks,
                rep.overall_pass ? "PASS" : "FAIL", t);
    return rep.overall_pass ? 0 : 1;
}
