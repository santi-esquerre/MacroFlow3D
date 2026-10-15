/**
 * @file slab_coarse_tests.cu
 * @brief SF-33 N7c (probe): contract tests of the Galerkin coarse-space correction
 *        (SlabCoarseCorrection.cuh) on N = 12. Standalone ctest runner (inlet_slab_coarse).
 *
 * Cases:
 *   1  Basis / transfer: for profiles 1 and 2, every dense basis vector v_c equals its definition
 *      (w_0 = 1, w_1 = j / N on planes 1..N of column (m2, m3), field f), prolong(e_c) = v_c, and
 *      V^T v_c equals the expected Gram column: block-diagonal per (column, field) with the
 *      profile Gram matrix G = [[N, (N+1)/2], [(N+1)/2, (N+1)(2N+1)/(6N)]] (1e-14 relative).
 *   2  E = V^T (J + mu D) V (mu = 0.7) at a 3-D state (transverse ln k variation, u != 0) equals
 *      the explicit product computed on the host from downloaded dense columns J v_c and the
 *      host definition of V, to 1e-12 relative to max |E| (profiles 1 and 2).
 *   3  k = 1, u = 0 (P-A exact there): multiplicative P-A + coarse applied to J x returns x to
 *      1e-12 (profiles 1 and 2); additive variant: ||M^-1 J x - x|| / ||x|| printed (not exact by
 *      construction: no gate beyond finiteness). The rcond estimate of E is printed.
 *   4  Host LU: random K x K (K = 1152 = 2 * 2 * 12^2) system solved to ||A x - b|| / ||b|| <=
 *      1e-12; transpose solve likewise; the Hager-Higham estimate of ||A^-1||_1 against the exact
 *      value (explicit inverse) on n = 200: est <= exact (1 + 1e-12) and est >= exact / 10;
 *      a singular matrix (two equal rows) reports zero_pivots > 0.
 *   5  Newton hook: exact pair (k = exp(0.7 sin 2 pi x1), N = 12) from x = 0 with coarse = mult
 *      and add (profiles 1): converged, r_F, r_out <= 1e-13; `COARSE build` and `COARSE apply`
 *      lines printed; coarse = off gives the history of a solver without prepare_coarse bitwise.
 *   6  Productization: colored assembly (color period p = 6 at N = 12, 8 at N = 16) gives E
 *      bitwise equal to the direct assembly (3-D state, mu = 0.7, profiles 1 and 2) with p^2 2 P
 *      operator applications; the banded host LU in the folded ordering solves E y = b like the
 *      dense LU (relative difference <= 1e-12) and its transpose solve likewise; the multiplicative
 *      preconditioner with the banded factors equals the dense one to 1e-12; rcond estimates of
 * both factorizations agree to 10 %; banded LU of a random banded matrix (n = 600, kl = 37, ku =
 * 23) against the dense LU to 1e-12.
 */

#include "src/core/DeviceBuffer.cuh"
#include "src/core/DeviceSpan.cuh"
#include "src/core/Scalar.hpp"
#include "src/physics/streamfunctions/inlet_slab/InletSlabGrid.cuh"
#include "src/physics/streamfunctions/inlet_slab/SlabCoarseCorrection.cuh"
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
    MACROFLOW3D_CUDA_CHECK(cudaDeviceSynchronize()); // legacy-stream copy landed (SF-33 C1/C2)
}

std::vector<real> download(CudaContext& ctx, const real* d, std::size_t n) {
    ctx.synchronize();
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

CaseSpec k1_spec() {
    CaseSpec s;
    auto zero = [](real, real, real) { return 0.0; };
    s.lnk = s.gl0 = s.gl1 = s.gl2 = s.u1 = s.u2 = zero;
    return s;
}

/// A genuinely 3-D state: transverse ln k variation and u != 0 (operator test only).
CaseSpec state3d_spec() {
    CaseSpec s;
    s.lnk = [](real x1, real x2, real x3) {
        return 0.3 * std::sin(kTwoPi * x2) * std::cos(kTwoPi * x3) + 0.2 * std::sin(kTwoPi * x1);
    };
    s.gl0 = [](real x1, real, real) { return 0.2 * kTwoPi * std::cos(kTwoPi * x1); };
    s.gl1 = [](real, real x2, real x3) {
        return 0.3 * kTwoPi * std::cos(kTwoPi * x2) * std::cos(kTwoPi * x3);
    };
    s.gl2 = [](real, real x2, real x3) {
        return -0.3 * kTwoPi * std::sin(kTwoPi * x2) * std::sin(kTwoPi * x3);
    };
    s.u1 = [](real x1, real x2, real x3) {
        return 0.03 * std::sin(kTwoPi * (x2 + x3)) * (1.0 + x1) + 0.01 * x1;
    };
    s.u2 = [](real x1, real x2, real x3) {
        return 0.02 * std::cos(kTwoPi * x2) * std::sin(kTwoPi * x3) * x1;
    };
    return s;
}

real phi_x3(real x3) {
    return 0.3 * std::sin(kTwoPi * x3) + 0.1 * std::cos(2.0 * kTwoPi * x3);
}

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

/// JVP + P-A frozen at the spec's own u.
struct OpHarness {
    sl::InletSlabGrid g;
    sl::SlabStageInputs in;
    sl::SlabResidualWorkspace rws;
    sl::SlabJvpWorkspace jws;
    sl::SlabModePreconditioner prec;
    DeviceBuffer<real> U1, U2, uvec, a, b, c;
    std::vector<real> u_spec;
    real mu = 0.0;

    void build(CudaContext& ctx, int N, const CaseSpec& s, real shift) {
        g = sl::InletSlabGrid::make(N);
        u_spec = fill_inputs(ctx, g, in, s);
        U1.resize(g.full_size());
        U2.resize(g.full_size());
        for (auto* v : {&uvec, &a, &b, &c})
            v->resize(g.unknown_size());
        rws.prepare(g);
        jws.prepare(g);
        prec.prepare(ctx, g);
        upload(uvec, u_spec);
        sl::assemble_full_planes(ctx, g, cspan(uvec), in, mspan(U1), mspan(U2));
        jws.prepare_base(ctx, g, in, cspan(U1), cspan(U2));
        mu = shift;
        prec.factor(ctx, g, in, cspan(U1), cspan(U2), mu);
        ctx.synchronize();
    }
    sl::SlabCoarseCorrection::Operator opA(CudaContext& ctx) {
        return [this, &ctx](DeviceSpan<const real> x, DeviceSpan<real> y) {
            jws.apply(ctx, g, x, y);
            if (mu != 0.0)
                sl::slab_add_pseudo_time_shift(ctx, g, in, mu, x, y);
        };
    }
    sl::SlabCoarseCorrection::Operator opPA(CudaContext& ctx) {
        return [this, &ctx](DeviceSpan<const real> x, DeviceSpan<real> y) {
            prec.apply(ctx, g, x, y);
        };
    }
};

/// Host definition of the basis (independent of the kernels).
struct HostBasis {
    int N, P, K;
    std::size_t np, nf;
    HostBasis(int n, int p) : N(n), P(p), K(2 * p * n * n) {
        np = static_cast<std::size_t>(n) * n;
        nf = np * n;
    }
    real w(int p, int j) const { return p == 0 ? 1.0 : static_cast<real>(j) / N; }
    std::vector<real> vec(int c) const {
        std::vector<real> v(2 * nf, 0.0);
        const int col = c / (2 * P), f = (c / P) % 2, p = c % P;
        for (int j = 1; j <= N; ++j)
            v[f * nf + (j - 1) * np + col] = w(p, j);
        return v;
    }
    std::vector<real> restrict_host(const std::vector<real>& r) const {
        std::vector<real> rc(K, 0.0);
        for (int c = 0; c < K; ++c) {
            const int col = c / (2 * P), f = (c / P) % 2, p = c % P;
            real s = 0.0;
            for (int j = 1; j <= N; ++j)
                s += w(p, j) * r[f * nf + (j - 1) * np + col];
            rc[c] = s;
        }
        return rc;
    }
};

// ------------------------------------------------------------------------------------------------
void case_basis(TestReport& rep, CudaContext& ctx) {
    const int N = 12;
    const auto g = sl::InletSlabGrid::make(N);
    for (int P : {1, 2}) {
        sl::SlabCoarseCorrection cc;
        cc.prepare(ctx, g, P);
        HostBasis hb(N, P);
        const int K = cc.K();
        DeviceBuffer<real> v(g.unknown_size()), pv(g.unknown_size()), rc(K), ec(K);
        const real G[2][2] = {{static_cast<real>(N), (N + 1) / 2.0},
                              {(N + 1) / 2.0, (N + 1) * (2.0 * N + 1) / (6.0 * N)}};
        real worst_def = 0.0, worst_pro = 0.0, worst_gram = 0.0;
        for (int c = 0; c < K; ++c) {
            cc.basis_vector(ctx, g, c, mspan(v));
            const auto hv = download(ctx, v.data(), v.size());
            worst_def = std::fmax(worst_def, rel_diff(hv, hb.vec(c)));
            std::vector<real> e(K, 0.0);
            e[c] = 1.0;
            upload(ec, e);
            upload(pv, std::vector<real>(g.unknown_size(), 0.0));
            cc.prolong_add(ctx, g, cspan(ec), mspan(pv));
            worst_pro = std::fmax(worst_pro, rel_diff(download(ctx, pv.data(), pv.size()), hv));
            cc.restrict_to_coarse(ctx, g, cspan(v), mspan(rc));
            const auto hrc = download(ctx, rc.data(), rc.size());
            const int cb = c - c % P; // first index of this (column, field) block
            for (int c2 = 0; c2 < K; ++c2) {
                real want = 0.0;
                if (c2 >= cb && c2 < cb + P)
                    want = G[c % P][c2 % P];
                worst_gram = std::fmax(worst_gram, std::fabs(hrc[c2] - want) / G[0][0]);
            }
        }
        char d[200];
        std::snprintf(d, sizeof(d), "K=%d def %.2e prolong %.2e gram %.2e", K, worst_def, worst_pro,
                      worst_gram);
        rep.check(K == 2 * P * N * N && worst_def == 0.0 && worst_pro <= 1e-15 &&
                      worst_gram <= 1e-14,
                  "coarse basis, profiles " + std::to_string(P) +
                      ": v_c = definition, prolong(e_c) = v_c, V^T V = block profile Gram",
                  d);
    }
}

// ------------------------------------------------------------------------------------------------
void case_galerkin(TestReport& rep, CudaContext& ctx) {
    const int N = 12;
    OpHarness hs;
    hs.build(ctx, N, state3d_spec(), 0.7);
    for (int P : {1, 2}) {
        sl::SlabCoarseCorrection cc;
        cc.prepare(ctx, hs.g, P);
        const auto br = cc.build(ctx, hs.g, hs.opA(ctx));
        const int K = cc.K();
        HostBasis hb(N, P);
        const auto& E = cc.galerkin_matrix();
        real emax = 0.0, worst = 0.0;
        for (real v : E)
            emax = std::fmax(emax, std::fabs(v));
        for (int c = 0; c < K; ++c) {
            upload(hs.a, hb.vec(c));
            hs.opA(ctx)(cspan(hs.a), mspan(hs.b));
            const auto col = hb.restrict_host(download(ctx, hs.b.data(), hs.b.size()));
            for (int r = 0; r < K; ++r)
                worst =
                    std::fmax(worst, std::fabs(E[static_cast<std::size_t>(r) * K + c] - col[r]));
        }
        char d[320];
        std::snprintf(d, sizeof(d),
                      "K=%d max|E - V^T A V|/max|E| = %.2e (max|E| %.3e); t_asm %.3fs t_lu %.3fs "
                      "rcond_est %.3e min/max|U_kk| %.3e/%.3e zero_pivots %d",
                      K, worst / emax, emax, br.t_assembly, br.t_lu, br.rcond_est, br.min_abs_u,
                      br.max_abs_u, br.zero_pivots);
        rep.check(worst / emax <= 1e-12 && br.zero_pivots == 0,
                  "E = V^T (J + mu D) V vs explicit host product (3-D state, mu = 0.7), profiles " +
                      std::to_string(P),
                  d);
    }
}

// ------------------------------------------------------------------------------------------------
void case_k1(TestReport& rep, CudaContext& ctx) {
    const int N = 12;
    OpHarness hs;
    hs.build(ctx, N, k1_spec(), 0.0);
    std::mt19937_64 rng(7701);
    const auto x = gaussian(rng, hs.g.unknown_size());
    upload(hs.a, x);
    hs.jws.apply(ctx, hs.g, cspan(hs.a), mspan(hs.b)); // b = J x
    for (int P : {1, 2}) {
        sl::SlabCoarseCorrection cc;
        cc.prepare(ctx, hs.g, P);
        const auto br = cc.build(ctx, hs.g, hs.opA(ctx));
        cc.apply(ctx, hs.g, sl::SlabCoarseMode::mult, hs.opA(ctx), hs.opPA(ctx), cspan(hs.b),
                 mspan(hs.c));
        const real em = rel_diff(download(ctx, hs.c.data(), hs.c.size()), x);
        cc.apply(ctx, hs.g, sl::SlabCoarseMode::add, hs.opA(ctx), hs.opPA(ctx), cspan(hs.b),
                 mspan(hs.c));
        const real ea = rel_diff(download(ctx, hs.c.data(), hs.c.size()), x);
        std::printf("[INFO] K1 profiles=%d K=%d rcond_est(E)=%.3e min/max|U_kk|=%.3e/%.3e "
                    "mult |M^-1 J x - x|/|x| = %.3e  add |M^-1 J x - x|/|x| = %.3e\n",
                    P, cc.K(), br.rcond_est, br.min_abs_u, br.max_abs_u, em, ea);
        rep.check(em <= 1e-12 && br.zero_pivots == 0,
                  "k = 1, u = 0, profiles " + std::to_string(P) +
                      ": P-A + coarse (mult) applied to J x returns x to 1e-12",
                  fmtd("%.3e", em));
        rep.check(std::isfinite(ea),
                  "k = 1, u = 0, profiles " + std::to_string(P) +
                      ": additive variant finite (value reported, not exact by construction)",
                  fmtd("%.3e", ea));
    }
}

// ------------------------------------------------------------------------------------------------
void case_host_lu(TestReport& rep) {
    std::mt19937_64 rng(7702);
    {
        const int n = 1152;
        const auto A = gaussian(rng, static_cast<std::size_t>(n) * n);
        const auto xt = gaussian(rng, n);
        std::vector<real> b(n, 0.0), bt(n, 0.0);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) {
                b[i] += A[static_cast<std::size_t>(i) * n + j] * xt[j];
                bt[j] += A[static_cast<std::size_t>(i) * n + j] * xt[i];
            }
        std::vector<real> LU = A;
        std::vector<int> piv(n);
        const auto t0 = std::chrono::steady_clock::now();
        const int zp = sl::SlabCoarseCorrection::lu_factor(n, LU.data(), piv.data());
        const double tlu =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        auto resid = [&](const std::vector<real>& x, const std::vector<real>& rhs, bool trans) {
            std::vector<real> r(n, 0.0);
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < n; ++j) {
                    const real a = trans ? A[static_cast<std::size_t>(j) * n + i]
                                         : A[static_cast<std::size_t>(i) * n + j];
                    r[i] += a * x[j];
                }
            for (int i = 0; i < n; ++i)
                r[i] -= rhs[i];
            return norm2(r) / norm2(rhs);
        };
        std::vector<real> x = b, xT = bt;
        sl::SlabCoarseCorrection::lu_solve(n, LU.data(), piv.data(), x.data());
        sl::SlabCoarseCorrection::lu_solve_transpose(n, LU.data(), piv.data(), xT.data());
        const real r1 = resid(x, b, false), r2 = resid(xT, bt, true);
        char d[200];
        std::snprintf(d, sizeof(d), "n=%d rel resid %.2e, transpose %.2e, t_lu %.3fs", n, r1, r2,
                      tlu);
        rep.check(zp == 0 && r1 <= 1e-12 && r2 <= 1e-12,
                  "host LU: random K x K system (K = 1152) and its transpose solved to 1e-12 "
                  "relative residual",
                  d);
    }
    {
        const int n = 200;
        auto A = gaussian(rng, static_cast<std::size_t>(n) * n);
        for (int i = 0; i < n; ++i)
            A[static_cast<std::size_t>(i) * n + i] += (i % 7 == 0) ? 0.05 : 3.0;
        std::vector<real> LU = A;
        std::vector<int> piv(n);
        sl::SlabCoarseCorrection::lu_factor(n, LU.data(), piv.data());
        real exact = 0.0;
        for (int c = 0; c < n; ++c) {
            std::vector<real> e(n, 0.0);
            e[c] = 1.0;
            sl::SlabCoarseCorrection::lu_solve(n, LU.data(), piv.data(), e.data());
            real s = 0.0;
            for (real v : e)
                s += std::fabs(v);
            exact = std::fmax(exact, s);
        }
        std::vector<real> work(2 * n);
        const real est =
            sl::SlabCoarseCorrection::inv_norm1_estimate(n, LU.data(), piv.data(), work.data());
        char d[160];
        std::snprintf(d, sizeof(d), "est %.6e exact %.6e ratio %.4f", est, exact, est / exact);
        rep.check(est <= exact * (1.0 + 1e-12) && est >= exact / 10.0,
                  "Hager-Higham estimate of ||A^-1||_1 within [exact/10, exact] (n = 200)", d);
    }
    {
        const int n = 50;
        auto A = gaussian(rng, static_cast<std::size_t>(n) * n);
        for (int j = 0; j < n; ++j)
            A[static_cast<std::size_t>(7) * n + j] = A[static_cast<std::size_t>(3) * n + j];
        std::vector<int> piv(n);
        const int zp = sl::SlabCoarseCorrection::lu_factor(n, A.data(), piv.data());
        // two equal rows: the last pivot is roundoff-small but rarely exactly zero; the count is
        // reported, the gate is on an exactly singular (zero column) matrix below.
        auto Z = gaussian(rng, static_cast<std::size_t>(n) * n);
        for (int i = 0; i < n; ++i)
            Z[static_cast<std::size_t>(i) * n + 5] = 0.0;
        const int zz = sl::SlabCoarseCorrection::lu_factor(n, Z.data(), piv.data());
        std::printf("[INFO] LU equal rows: zero_pivots=%d (roundoff-small pivot expected)\n", zp);
        rep.check(zz >= 1, "host LU: an exactly singular matrix (zero column) reports a zero pivot",
                  "zero_pivots=" + std::to_string(zz));
    }
}

// ------------------------------------------------------------------------------------------------
struct Capture {
    std::vector<std::string> lines;
    sl::SlabLogger logger() {
        return [this](const std::string& s) {
            lines.push_back(s);
            std::printf("    | %s\n", s.c_str());
        };
    }
    int count(const std::string& sub) const {
        int c = 0;
        for (const auto& l : lines)
            if (l.find(sub) != std::string::npos)
                ++c;
        return c;
    }
};

void case_newton_hook(TestReport& rep, CudaContext& ctx) {
    const int N = 12;
    const auto g = sl::InletSlabGrid::make(N);
    sl::SlabStageInputs in;
    const auto u_ex = fill_inputs(ctx, g, in, exact_pair_spec());
    DeviceBuffer<real> x(g.unknown_size());
    // off vs a solver without prepare_coarse: bitwise identical histories
    sl::SlabNewtonConfig cfg; // ew, psitc off (library defaults)
    std::vector<real> h_plain, h_off;
    {
        sl::SlabNewtonKrylov nk;
        nk.prepare(ctx, g);
        upload(x, std::vector<real>(g.unknown_size(), 0.0));
        const auto r = nk.solve(ctx, in, mspan(x), "plain", cfg, [](const std::string&) {});
        h_plain = r.hist_r_F;
    }
    sl::SlabNewtonKrylov nk;
    nk.prepare(ctx, g);
    nk.prepare_coarse(ctx, 1);
    {
        upload(x, std::vector<real>(g.unknown_size(), 0.0));
        const auto r = nk.solve(ctx, in, mspan(x), "off", cfg, [](const std::string&) {});
        h_off = r.hist_r_F;
    }
    rep.check(h_plain == h_off, "coarse = off: history bitwise equal to a solver without "
                                "prepare_coarse");
    for (sl::SlabCoarseMode mode : {sl::SlabCoarseMode::mult, sl::SlabCoarseMode::add}) {
        sl::SlabNewtonConfig c = cfg;
        c.coarse = mode;
        c.prec_name = std::string("P-A+CC(") + sl::to_string(mode) + ",1)";
        upload(x, std::vector<real>(g.unknown_size(), 0.0));
        Capture cap;
        const auto r = nk.solve(ctx, in, mspan(x), std::string("exact_pair_") + sl::to_string(mode),
                                c, cap.logger());
        const auto xh = download(ctx, x.data(), x.size());
        real maxerr = 0.0;
        for (std::size_t i = 0; i < xh.size(); ++i)
            maxerr = std::fmax(maxerr, std::fabs(xh[i] - u_ex[i]));
        char d[260];
        std::snprintf(d, sizeof(d),
                      "status %s its %d r_F %.3e r_out %.3e max|u-u_ex| %.3e GMRES total %d "
                      "(plain P-A: %zu Newton its)",
                      sl::to_string(r.status), r.its, r.r_F, r.r_out, maxerr,
                      r.linear_iterations_total, h_plain.size() - 1);
        rep.check(r.status == sl::SlabSolveStatus::converged && r.r_F <= 1e-13 &&
                      r.r_out <= 1e-13 && maxerr <= 1e-11 &&
                      cap.count("COARSE build") == static_cast<int>(r.steps.size()) &&
                      cap.count("COARSE apply") == static_cast<int>(r.steps.size()),
                  std::string("Newton hook coarse = ") + sl::to_string(mode) +
                      ", exact pair N = 12 from x = 0: converged, COARSE lines per step",
                  d);
    }
}

// ------------------------------------------------------------------------------------------------
void case_productization(TestReport& rep, CudaContext& ctx) {
    for (int N : {12, 16}) {
        OpHarness hs;
        hs.build(ctx, N, state3d_spec(), 0.7);
        for (int P : {1, 2}) {
            sl::SlabCoarseCorrection cd, cc, cb;
            cd.prepare(ctx, hs.g, P);
            cc.prepare(ctx, hs.g, P, sl::SlabCoarseAssembly::colored, sl::SlabCoarseFactor::dense);
            cb.prepare(ctx, hs.g, P, sl::SlabCoarseAssembly::colored, sl::SlabCoarseFactor::banded);
            const auto rd = cd.build(ctx, hs.g, hs.opA(ctx));
            const auto rcl = cc.build(ctx, hs.g, hs.opA(ctx));
            const auto rb = cb.build(ctx, hs.g, hs.opA(ctx));
            const auto Ed = cd.galerkin_dense_copy();
            const auto Ec = cc.galerkin_dense_copy();
            const auto Eb = cb.galerkin_dense_copy();
            real dmax = 0.0, dbmax = 0.0;
            for (std::size_t i = 0; i < Ed.size(); ++i) {
                dmax = std::fmax(dmax, std::fabs(Ed[i] - Ec[i]));
                dbmax = std::fmax(dbmax, std::fabs(Ed[i] - Eb[i]));
            }
            char d[360];
            std::snprintf(d, sizeof(d),
                          "K=%d p=%d applies colored %d vs direct %d; max|E_col - E_dir| = %.1e, "
                          "banded copy %.1e; t_asm direct %.3fs colored %.3fs",
                          cc.K(), rcl.color_period, rcl.applications, rd.applications, dmax, dbmax,
                          rd.t_assembly, rcl.t_assembly);
            rep.check(dmax == 0.0 && dbmax == 0.0 &&
                          rcl.applications == rcl.color_period * rcl.color_period * 2 * P,
                      "colored assembly == direct assembly bitwise, N = " + std::to_string(N) +
                          ", profiles " + std::to_string(P),
                      d);
            // banded vs dense solves through the preconditioner application
            std::mt19937_64 rng(7800 + N + P);
            const auto r = gaussian(rng, hs.g.unknown_size());
            upload(hs.a, r);
            cd.apply(ctx, hs.g, sl::SlabCoarseMode::mult, hs.opA(ctx), hs.opPA(ctx), cspan(hs.a),
                     mspan(hs.b));
            const auto zd = download(ctx, hs.b.data(), hs.b.size());
            cb.apply(ctx, hs.g, sl::SlabCoarseMode::mult, hs.opA(ctx), hs.opPA(ctx), cspan(hs.a),
                     mspan(hs.c));
            const auto zb = download(ctx, hs.c.data(), hs.c.size());
            const real dz = rel_diff(zb, zd);
            const real rr = rb.rcond_est / rd.rcond_est;
            char d2[360];
            std::snprintf(d2, sizeof(d2),
                          "mult apply banded vs dense %.2e; rcond dense %.3e banded %.3e; kl %d ku "
                          "%d; t_lu dense %.3fs banded %.3fs; min/max|U_kk| banded %.3e/%.3e",
                          dz, rd.rcond_est, rb.rcond_est, rb.kl, rb.ku, rd.t_lu, rb.t_lu,
                          rb.min_abs_u, rb.max_abs_u);
            rep.check(dz <= 1e-12 && rb.zero_pivots == 0 && rr > 0.9 && rr < 1.1,
                      "banded LU (folded ordering) == dense LU through the mult preconditioner, "
                      "N = " +
                          std::to_string(N) + ", profiles " + std::to_string(P),
                      d2);
        }
    }
    {
        // banded LU kernels vs dense on a random banded matrix (solve and transpose solve)
        const int n = 600, kl = 37, ku = 23;
        std::mt19937_64 rng(7900);
        std::normal_distribution<real> nd(0.0, 1.0);
        std::vector<real> A(static_cast<std::size_t>(n) * n, 0.0);
        const int ldab = 2 * kl + ku + 1;
        std::vector<real> AB(static_cast<std::size_t>(ldab) * n, 0.0);
        for (int j = 0; j < n; ++j)
            for (int i = std::max(0, j - ku); i <= std::min(n - 1, j + kl); ++i) {
                const real v = nd(rng);
                A[static_cast<std::size_t>(i) * n + j] = v;
                AB[static_cast<std::size_t>(j) * ldab + kl + ku + i - j] = v;
            }
        std::vector<int> pd(n), pb(n);
        std::vector<real> LU = A;
        sl::SlabCoarseCorrection::lu_factor(n, LU.data(), pd.data());
        const int zb = sl::SlabCoarseCorrection::band_lu_factor(n, kl, ku, AB.data(), pb.data());
        const auto b = gaussian(rng, n);
        auto xd = b, xb = b, td = b, tb = b;
        sl::SlabCoarseCorrection::lu_solve(n, LU.data(), pd.data(), xd.data());
        sl::SlabCoarseCorrection::band_lu_solve(n, kl, ku, AB.data(), pb.data(), xb.data());
        sl::SlabCoarseCorrection::lu_solve_transpose(n, LU.data(), pd.data(), td.data());
        sl::SlabCoarseCorrection::band_lu_solve_transpose(n, kl, ku, AB.data(), pb.data(),
                                                          tb.data());
        const real e1 = rel_diff(xb, xd), e2 = rel_diff(tb, td);
        char d[160];
        std::snprintf(d, sizeof(d), "solve %.2e transpose %.2e", e1, e2);
        rep.check(zb == 0 && e1 <= 1e-12 && e2 <= 1e-12,
                  "banded LU vs dense LU on a random banded matrix (n = 600, kl = 37, ku = 23)", d);
    }
}
} // namespace

int main() {
    TestReport rep;
    CudaContext ctx(0);
    const auto t0 = std::chrono::steady_clock::now();
    try {
        std::printf("=== SF-33 N7c: (1) coarse basis and transfers ===\n");
        case_basis(rep, ctx);
        std::printf("=== SF-33 N7c: (2) Galerkin matrix vs explicit V^T A V ===\n");
        case_galerkin(rep, ctx);
        std::printf("=== SF-33 N7c: (3) k = 1: P-A + coarse ===\n");
        case_k1(rep, ctx);
        std::printf("=== SF-33 N7c: (4) host LU and condition estimate ===\n");
        case_host_lu(rep);
        std::printf("=== SF-33 N7c: (5) Newton hook ===\n");
        case_newton_hook(rep, ctx);
        std::printf("=== SF-33 N7c: (6) colored assembly and banded LU ===\n");
        case_productization(rep, ctx);
    } catch (const std::exception& e) {
        std::printf("[FAIL] unexpected exception: %s\n", e.what());
        rep.overall_pass = false;
    }
    const double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("\n=== inlet_slab_coarse: %d checks, %s (%.1f s) ===\n", rep.checks,
                rep.overall_pass ? "PASS" : "FAIL", t);
    return rep.overall_pass ? 0 : 1;
}
