/**
 * @file slab_jvp_tests.cu
 * @brief SF-33 N1: fast contract tests of the analytic Jacobian-vector product of the inlet-slab
 * residual (SlabJacobianVectorProduct.cuh). Standalone ctest runner (printed [PASS]/[FAIL] checks,
 * exit code), every grid N = 12.
 *
 * Cases (acceptance items of the SF-33 N1 task):
 *   1  FD ladder: base = exact pair (k = exp(0.7 sin 2 pi x1), u1 = Phi(x3), u2 = 0) + 1e-2 RMS(u)
 *      gaussian noise (std::mt19937_64, fixed seed, both fields, planes 1..N); gaussian direction
 *      with RMS(v) = RMS(u); ||Jv - FD||/||FD|| with the central FD (E(u + t v) - E(u - t v))/(2t),
 *      t = 1e-3, 1e-4, 1e-5 (absolute, as the prototype's `jactest`), on the full direction and on
 *      directions restricted to the outlet planes N-5..N and the inlet-side planes 1..6. Gate:
 *      consecutive error ratios in [30, 300] and best error <= 1e-8.
 *   2  Linearity: J(a v + b w) = a J v + b J w to 1e-13 relative (a = 0.3, b = -1.7).
 *   3  k = 1, u = 0 symbol: J applied to du1 = s(x1) cos(2 pi (2 x2 + 3 x3)), du2 = 0 (and the
 *      mirrored du1 = 0, du2 = same) equals the documented linearization evaluated on the host
 *      with the N0 stencils: dF_1 = -[(d11 + d22) du1 + d23 du2], dF_2 = -[(d11 + d33) du2 +
 *      d23 du1], outlet dE_i = +(2/h) d1 du_i; to 1e-12 relative.
 *   4  Exact-pair state, direction (u1 = Phi(x3) on planes 1..N, u2 = 0): FD ladder as in 1,
 *      except that the base residual is pure roundoff (~6e-13) and the t = 1e-5 FD reaches its
 *      roundoff floor (~2e-10 relative): the last ratio is gated against the measured floor
 *      (fd_ladder doc); the literal strict verdict is printed.
 *   5  Contracts: no allocation in apply (buffer bytes + pointers; cudaMemGetInfo [INFO]), direction not mutated,
 *      bitwise-repeatable apply, std::logic_error before prepare_base / after re-prepare.
 */

#include "src/core/DeviceBuffer.cuh"
#include "src/core/DeviceSpan.cuh"
#include "src/core/Scalar.hpp"
#include "src/physics/streamfunctions/inlet_slab/InletSlabGrid.cuh"
#include "src/physics/streamfunctions/inlet_slab/SlabJacobianVectorProduct.cuh"
#include "src/physics/streamfunctions/inlet_slab/SlabResidual.cuh"
#include "src/physics/streamfunctions/inlet_slab/SlabStencils4.cuh"
#include "src/runtime/cuda_check.cuh"
#include "src/runtime/CudaContext.cuh"

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
constexpr int kN = 12;

struct TestReport {
    bool overall_pass = true;
    int checks = 0;
    void check(bool cond, const std::string& name, const std::string& detail = "") {
        ++checks;
        std::printf("[%s] %s%s%s\n", cond ? "PASS" : "FAIL", name.c_str(),
                    detail.empty() ? "" : "  ", detail.c_str());
        overall_pass = overall_pass && cond;
    }
};

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

using F3 = std::function<real(real, real, real)>;

std::vector<real> sample_full(const sl::InletSlabGrid& g, const F3& f) {
    std::vector<real> a(g.full_size());
    for (int j = 0; j <= g.n; ++j)
        for (int m2 = 0; m2 < g.n; ++m2)
            for (int m3 = 0; m3 < g.n; ++m3)
                a[g.full_index(j, m2, m3)] = f(g.coord(j), g.coord(m2), g.coord(m3));
    return a;
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

/// Stage inputs of an analytic case (copied from the N0 contract-test fixtures).
struct CaseSpec {
    F3 lnk, gl0, gl1, gl2, u1, u2;
};

/// Exact discrete pair of N0 (understanding record section 3): k = exp(0.7 sin 2 pi x1),
/// u1 = Phi(x3) = 0.3 sin 2 pi x3 + 0.1 cos 4 pi x3, u2 = 0, v_perp,in = 0.
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

CaseSpec k1_spec() {
    CaseSpec s;
    auto zero = [](real, real, real) { return 0.0; };
    s.lnk = zero;
    s.gl0 = zero;
    s.gl1 = zero;
    s.gl2 = zero;
    s.u1 = zero;
    s.u2 = zero;
    return s;
}

/// Device-side harness: stage inputs, residual and JVP workspaces, scratch arrays.
struct Harness {
    sl::InletSlabGrid g;
    sl::SlabStageInputs in;
    sl::SlabResidualWorkspace rws;
    sl::SlabJvpWorkspace jws;
    DeviceBuffer<real> U1, U2, uvec, E, dvec, Jv;
    std::vector<real> u_base; ///< host unknown vector of the case (planes 1..N)

    void build(CudaContext& ctx, int N, const CaseSpec& s) {
        g = sl::InletSlabGrid::make(N);
        in.allocate(g);
        in.field = "test";
        upload(in.lnk, sample_full(g, s.lnk));
        sl::fill_q_from_lnk(ctx, g, in);
        const F3* gl[3] = {&s.gl0, &s.gl1, &s.gl2};
        for (int k = 0; k < 3; ++k)
            upload(in.grad_lnk[k], sample_full(g, *gl[k]));
        const auto hU1 = sample_full(g, s.u1);
        const auto hU2 = sample_full(g, s.u2);
        const std::size_t np = g.plane_size();
        const std::size_t nf = g.field_size();
        upload(in.u0[0], std::vector<real>(hU1.begin(), hU1.begin() + np));
        upload(in.u0[1], std::vector<real>(hU2.begin(), hU2.begin() + np));
        upload(in.vperp_in[0], std::vector<real>(np, 0.0));
        upload(in.vperp_in[1], std::vector<real>(np, 0.0));
        in.v_rms = 1.0;
        u_base.assign(2 * nf, 0.0);
        for (std::size_t i = 0; i < nf; ++i) {
            u_base[i] = hU1[np + i];
            u_base[nf + i] = hU2[np + i];
        }
        U1.resize(g.full_size());
        U2.resize(g.full_size());
        uvec.resize(g.unknown_size());
        E.resize(g.unknown_size());
        dvec.resize(g.unknown_size());
        Jv.resize(g.unknown_size());
        rws.prepare(g);
        jws.prepare(g);
        ctx.synchronize();
    }

    std::vector<real> residual(CudaContext& ctx, const std::vector<real>& u) {
        upload(uvec, u);
        sl::assemble_full_planes(ctx, g, cspan(uvec), in, mspan(U1), mspan(U2));
        sl::evaluate_residual(ctx, g, in, cspan(U1), cspan(U2), mspan(E), rws, nullptr);
        ctx.synchronize();
        return download(E.data(), E.size());
    }

    void freeze(CudaContext& ctx, const std::vector<real>& u) {
        upload(uvec, u);
        sl::assemble_full_planes(ctx, g, cspan(uvec), in, mspan(U1), mspan(U2));
        jws.prepare_base(ctx, g, in, cspan(U1), cspan(U2));
        ctx.synchronize();
    }

    std::vector<real> jvp(CudaContext& ctx, const std::vector<real>& v) {
        upload(dvec, v);
        jws.apply(ctx, g, cspan(dvec), mspan(Jv));
        ctx.synchronize();
        return download(Jv.data(), Jv.size());
    }
};

std::vector<real> gaussian(std::mt19937_64& rng, std::size_t n) {
    std::normal_distribution<real> nd(0.0, 1.0);
    std::vector<real> a(n);
    for (auto& x : a)
        x = nd(rng);
    return a;
}

/// Keep unknown planes [jlo, jhi] (1-based, inclusive) of both fields; zero elsewhere.
std::vector<real> restrict_planes(const sl::InletSlabGrid& g, const std::vector<real>& v, int jlo,
                                  int jhi) {
    std::vector<real> r(v.size(), 0.0);
    const std::size_t nf = g.field_size();
    for (int f = 0; f < 2; ++f)
        for (int j = jlo; j <= jhi; ++j)
            for (int m2 = 0; m2 < g.n; ++m2)
                for (int m3 = 0; m3 < g.n; ++m3) {
                    const std::size_t i = f * nf + g.unknown_index(j, m2, m3);
                    r[i] = v[i];
                }
    return r;
}

/**
 * FD ladder of one direction; prints the three errors and applies the gate.
 *
 * roundoff_level < 0 (cases 1): strict gate, every consecutive ratio in [30, 300], best <= 1e-8.
 * roundoff_level >= 0 (case 4 only): ||E|| of a state whose exact residual is ZERO (the exact
 * pair), i.e. the measured roundoff of one residual evaluation near the base. The FD roundoff floor
 * at step t is then estimated as floor(t) = roundoff_level / (t ||FD||) (difference of two noisy
 * evaluations divided by 2t). A consecutive ratio outside [30, 300] is accepted only when the
 * error at the smaller step is at or below that floor (FD roundoff-limited, not a JVP defect); the
 * literal strict verdict is printed as well. best <= 1e-8 is always required.
 */
void fd_ladder(TestReport& rep, CudaContext& ctx, Harness& hs, const std::vector<real>& u,
               const std::vector<real>& v, const std::string& label,
               real roundoff_level = -1.0) {
    hs.freeze(ctx, u);
    const auto Jv = hs.jvp(ctx, v);
    const real ts[3] = {1e-3, 1e-4, 1e-5};
    real err[3], fdn[3];
    for (int k = 0; k < 3; ++k) {
        std::vector<real> up(u.size()), um(u.size());
        for (std::size_t i = 0; i < u.size(); ++i) {
            up[i] = u[i] + ts[k] * v[i];
            um[i] = u[i] - ts[k] * v[i];
        }
        const auto Ep = hs.residual(ctx, up);
        const auto Em = hs.residual(ctx, um);
        std::vector<real> fd(u.size());
        for (std::size_t i = 0; i < u.size(); ++i)
            fd[i] = (Ep[i] - Em[i]) / (2.0 * ts[k]);
        err[k] = rel_diff(Jv, fd);
        fdn[k] = norm2(fd);
        std::printf("[INFO] JVPTEST %s N=%d step=%.0e  ||Jv - FD||/||FD|| = %.3e  (||Jv|| = %.3e)\n",
                    label.c_str(), hs.g.n, ts[k], err[k], norm2(Jv));
    }
    const real r1 = err[0] / err[1];
    const real r2 = err[1] / err[2];
    const real best = std::fmin(err[0], std::fmin(err[1], err[2]));
    char det[200];
    std::snprintf(det, sizeof(det), "errors %.3e %.3e %.3e  ratios %.1f %.1f  best %.3e", err[0],
                  err[1], err[2], r1, r2, best);
    const auto in_band = [](real r) { return r >= 30.0 && r <= 300.0; };
    const bool strict = in_band(r1) && in_band(r2) && best <= 1e-8;
    if (roundoff_level < 0.0) {
        rep.check(strict,
                  "JVP vs central FD, " + label + ": ratios in [30, 300] (~t^2) and best <= 1e-8",
                  det);
        return;
    }
    real floor_t[3];
    for (int k = 0; k < 3; ++k)
        floor_t[k] = roundoff_level / (ts[k] * fdn[k]);
    std::printf("[INFO] JVPTEST %s: measured residual roundoff %.3e -> FD floor estimate %.3e "
                "%.3e %.3e; literal strict gate (all ratios in [30, 300]): %s\n",
                label.c_str(), roundoff_level, floor_t[0], floor_t[1], floor_t[2],
                strict ? "PASS" : "FAIL (roundoff-limited step, see floor)");
    const bool ok1 = in_band(r1) || err[1] <= floor_t[1];
    const bool ok2 = in_band(r2) || err[2] <= floor_t[2];
    // the first ratio must be a genuine t^2 ratio (truncation-dominated), never floor-excused
    rep.check(in_band(r1) && ok1 && ok2 && best <= 1e-8,
              "JVP vs central FD, " + label +
                  ": ratio 1e-3/1e-4 in [30, 300] (~t^2), ratio 1e-4/1e-5 in [30, 300] or the "
                  "1e-5 error at/below the measured FD roundoff floor, best <= 1e-8",
              det);
}

// ------------------------------------------------------------------------------------------------
// 1, 2. FD ladder at the perturbed exact pair; linearity
// ------------------------------------------------------------------------------------------------
void case_fd_and_linearity(TestReport& rep, CudaContext& ctx) {
    Harness hs;
    hs.build(ctx, kN, exact_pair_spec());
    const auto& g = hs.g;
    std::mt19937_64 rng(3301);
    const real rms_ex = rms(hs.u_base);
    const auto noise = gaussian(rng, g.unknown_size());
    std::vector<real> u(hs.u_base);
    for (std::size_t i = 0; i < u.size(); ++i)
        u[i] += 1e-2 * rms_ex * noise[i];
    const real rms_u = rms(u);
    auto v = gaussian(rng, g.unknown_size());
    const real sv = rms_u / rms(v);
    for (auto& x : v)
        x *= sv;
    const auto E0 = hs.residual(ctx, u);
    std::printf("[INFO] base state: exact pair + 1e-2 RMS(u) noise, N = %d, RMS(u) = %.6e, "
                "||E(u)|| = %.3e\n",
                g.n, rms_u, norm2(E0));

    fd_ladder(rep, ctx, hs, u, v, "full direction");
    fd_ladder(rep, ctx, hs, u, restrict_planes(g, v, g.n - 5, g.n), "outlet planes N-5..N");
    fd_ladder(rep, ctx, hs, u, restrict_planes(g, v, 1, 6), "inlet-side planes 1..6");

    // 2. linearity
    auto w = gaussian(rng, g.unknown_size());
    const real a = 0.3, b = -1.7;
    std::vector<real> comb(v.size());
    for (std::size_t i = 0; i < v.size(); ++i)
        comb[i] = a * v[i] + b * w[i];
    hs.freeze(ctx, u);
    const auto Jv = hs.jvp(ctx, v);
    const auto Jw = hs.jvp(ctx, w);
    const auto Jc = hs.jvp(ctx, comb);
    std::vector<real> lin(v.size());
    for (std::size_t i = 0; i < v.size(); ++i)
        lin[i] = a * Jv[i] + b * Jw[i];
    const real lerr = rel_diff(Jc, lin);
    char det[120];
    std::snprintf(det, sizeof(det), "||J(av+bw) - (aJv+bJw)|| / ||aJv+bJw|| = %.3e", lerr);
    rep.check(lerr <= 1e-13, "linearity, a = 0.3, b = -1.7: <= 1e-13 relative", det);
}

// ------------------------------------------------------------------------------------------------
// 3. k = 1, u = 0 symbol
// ------------------------------------------------------------------------------------------------
void case_k1_symbol(TestReport& rep, CudaContext& ctx) {
    Harness hs;
    hs.build(ctx, kN, k1_spec());
    const auto& g = hs.g;
    sl::SlabStencilTable tab;
    tab.build(g);
    const sl::SlabStencilView st = tab.host_view();
    hs.freeze(ctx, hs.u_base); // u = 0
    const auto mode = [](real x1, real x2, real x3) {
        return x1 * x1 * (1.0 - x1) * std::cos(kTwoPi * (2.0 * x2 + 3.0 * x3));
    };
    for (int field = 0; field < 2; ++field) {
        // full arrays of the direction, plane 0 = 0
        std::vector<real> dA = sample_full(g, mode);
        for (std::size_t i = 0; i < g.plane_size(); ++i)
            dA[i] = 0.0;
        const std::vector<real> zero(g.full_size(), 0.0);
        const std::vector<real>& dU1 = field == 0 ? dA : zero;
        const std::vector<real>& dU2 = field == 0 ? zero : dA;
        const std::size_t nf = g.field_size();
        const std::size_t np = g.plane_size();
        std::vector<real> v(2 * nf, 0.0);
        for (std::size_t i = 0; i < nf; ++i)
            v[static_cast<std::size_t>(field) * nf + i] = dA[np + i];
        // host reference: the documented k = 1 linearization with the N0 stencils
        std::vector<real> ref(2 * nf, 0.0);
        for (int j = 1; j <= g.n; ++j)
            for (int m2 = 0; m2 < g.n; ++m2)
                for (int m3 = 0; m3 < g.n; ++m3) {
                    const std::size_t ui = g.unknown_index(j, m2, m3);
                    if (j < g.n) {
                        const std::size_t off = g.plane_index(m2, m3);
                        const real d11_1 = sl::x1_stencil_sum(st.d11[j], dU1.data(), np, off) /
                                           (g.h * g.h);
                        const real d11_2 = sl::x1_stencil_sum(st.d11[j], dU2.data(), np, off) /
                                           (g.h * g.h);
                        const real d22_1 = sl::dpp4_at(dU1.data(), g, j, m2, m3, 2);
                        const real d33_2 = sl::dpp4_at(dU2.data(), g, j, m2, m3, 3);
                        const real d23_1 = sl::d23_at(dU1.data(), g, j, m2, m3);
                        const real d23_2 = sl::d23_at(dU2.data(), g, j, m2, m3);
                        ref[ui] = -((d11_1 + d22_1) + d23_2);
                        ref[nf + ui] = -((d11_2 + d33_2) + d23_1);
                    } else {
                        ref[ui] = (2.0 / g.h) * sl::d1_fd4_at(dU1.data(), st, g, j, m2, m3);
                        ref[nf + ui] = (2.0 / g.h) * sl::d1_fd4_at(dU2.data(), st, g, j, m2, m3);
                    }
                }
        const auto Jv = hs.jvp(ctx, v);
        const real err = rel_diff(Jv, ref);
        // separate outlet-row and coupling-row checks (so a sign error cannot hide in the norm)
        std::vector<real> Jo, Ro, Jc, Rc;
        for (int m2 = 0; m2 < g.n; ++m2)
            for (int m3 = 0; m3 < g.n; ++m3) {
                const std::size_t ui = g.unknown_index(g.n, m2, m3);
                Jo.push_back(Jv[static_cast<std::size_t>(field) * nf + ui]);
                Ro.push_back(ref[static_cast<std::size_t>(field) * nf + ui]);
            }
        for (int j = 1; j < g.n; ++j)
            for (int m2 = 0; m2 < g.n; ++m2)
                for (int m3 = 0; m3 < g.n; ++m3) {
                    const std::size_t ui = g.unknown_index(j, m2, m3);
                    Jc.push_back(Jv[static_cast<std::size_t>(1 - field) * nf + ui]);
                    Rc.push_back(ref[static_cast<std::size_t>(1 - field) * nf + ui]);
                }
        const real eo = rel_diff(Jo, Ro);
        const real ec = rel_diff(Jc, Rc);
        char det[220];
        std::snprintf(det, sizeof(det),
                      "all rows %.3e  outlet rows of field %d %.3e (||.|| = %.3e)  d23 coupling "
                      "rows of field %d %.3e (||.|| = %.3e)",
                      err, field + 1, eo, norm2(Ro), 2 - field, ec, norm2(Rc));
        rep.check(err <= 1e-12 && eo <= 1e-12 && ec <= 1e-12 && norm2(Ro) > 0.0 &&
                      norm2(Rc) > 0.0,
                  std::string("k = 1, u = 0 symbol, direction du") + (field == 0 ? "1" : "2") +
                      " = s(x1) cos 2pi(2x2 + 3x3): J = documented linearization to 1e-12",
                  det);
    }
}

// ------------------------------------------------------------------------------------------------
// 4. exact-pair state, direction (Phi(x3), 0)
// ------------------------------------------------------------------------------------------------
void case_exact_pair_direction(TestReport& rep, CudaContext& ctx) {
    Harness hs;
    hs.build(ctx, kN, exact_pair_spec());
    const auto& g = hs.g;
    const auto E0 = hs.residual(ctx, hs.u_base);
    std::printf("[INFO] exact pair N = %d: ||E(u_ex)|| = %.3e\n", g.n, norm2(E0));
    std::vector<real> v(g.unknown_size(), 0.0);
    for (int j = 1; j <= g.n; ++j)
        for (int m2 = 0; m2 < g.n; ++m2)
            for (int m3 = 0; m3 < g.n; ++m3)
                v[g.unknown_index(j, m2, m3)] = phi_x3(g.coord(m3));
    // E(u_ex) is zero in exact arithmetic: its norm is the measured roundoff of one evaluation.
    fd_ladder(rep, ctx, hs, hs.u_base, v, "exact pair, direction (Phi(x3), 0)", norm2(E0));
}

// ------------------------------------------------------------------------------------------------
// 5. contracts
// ------------------------------------------------------------------------------------------------
void case_contracts(TestReport& rep, CudaContext& ctx) {
    const auto g = sl::InletSlabGrid::make(kN);
    {
        sl::SlabJvpWorkspace ws;
        DeviceBuffer<real> d(g.unknown_size()), o(g.unknown_size());
        bool threw = false;
        try {
            ws.apply(ctx, g, cspan(d), mspan(o));
        } catch (const std::logic_error&) {
            threw = true;
        }
        rep.check(threw, "contracts: apply on an unprepared workspace throws std::logic_error");
        ws.prepare(g);
        threw = false;
        try {
            ws.apply(ctx, g, cspan(d), mspan(o));
        } catch (const std::logic_error&) {
            threw = true;
        }
        rep.check(threw, "contracts: apply before prepare_base throws std::logic_error");
    }

    Harness hs;
    hs.build(ctx, kN, exact_pair_spec());
    std::mt19937_64 rng(3302);
    auto u = hs.u_base;
    const auto nz = gaussian(rng, u.size());
    for (std::size_t i = 0; i < u.size(); ++i)
        u[i] += 1e-2 * nz[i];
    const auto v = gaussian(rng, u.size());
    hs.freeze(ctx, u);
    upload(hs.dvec, v);
    const std::size_t bytes0 = hs.jws.allocated_bytes();
    const auto ptr0 = hs.jws.storage_pointers();
    std::size_t free0 = 0, free1 = 0, total = 0;
    MACROFLOW3D_CUDA_CHECK(cudaMemGetInfo(&free0, &total));
    hs.jws.apply(ctx, hs.g, cspan(hs.dvec), mspan(hs.Jv));
    ctx.synchronize();
    const auto J1 = download(hs.Jv.data(), hs.Jv.size());
    for (int r = 0; r < 3; ++r)
        hs.jws.apply(ctx, hs.g, cspan(hs.dvec), mspan(hs.Jv));
    ctx.synchronize();
    MACROFLOW3D_CUDA_CHECK(cudaMemGetInfo(&free1, &total));
    const auto J2 = download(hs.Jv.data(), hs.Jv.size());
    const auto vback = download(hs.dvec.data(), hs.dvec.size());
    // cudaMemGetInfo is device-wide (other processes change it): informational only.
    std::printf("[INFO] contracts: JVP workspace %zu bytes at N = %d (4 (N+1) N^2 doubles + table); "
                "device free before/after 4 applies: %zu / %zu (delta %.0f, device-wide)\n",
                bytes0, kN, free0, free1, static_cast<double>(free0) - static_cast<double>(free1));
    rep.check(bytes0 == hs.jws.allocated_bytes() && ptr0 == hs.jws.storage_pointers(),
              "contracts: no allocation in apply (workspace bytes and buffer pointers "
              "unchanged)");
    rep.check(vback == v, "contracts: the direction is not mutated by apply (bitwise)");
    rep.check(J1 == J2, "contracts: repeated apply is bitwise reproducible");
    const std::size_t expect = 4 * hs.g.full_size() * sizeof(real);
    rep.check(bytes0 >= expect, "contracts: allocated_bytes() >= 4 full arrays",
              std::to_string(bytes0) + " >= " + std::to_string(expect));
    bool threw = false;
    try {
        hs.jws.apply(ctx, hs.g, cspan(hs.dvec),
                     DeviceSpan<real>(hs.dvec.data(), hs.dvec.size()));
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    rep.check(threw, "contracts: overlapping direction / out throws std::invalid_argument");
    hs.jws.prepare(hs.g);
    threw = false;
    try {
        hs.jws.apply(ctx, hs.g, cspan(hs.dvec), mspan(hs.Jv));
    } catch (const std::logic_error&) {
        threw = true;
    }
    rep.check(threw, "contracts: prepare invalidates the frozen base (apply throws logic_error)");
}

} // namespace

int main() {
    TestReport rep;
    CudaContext ctx(0);
    try {
        std::printf("=== SF-33 N1: (1, 2) FD ladder at the perturbed exact pair; linearity ===\n");
        case_fd_and_linearity(rep, ctx);
        std::printf("=== SF-33 N1: (3) k = 1, u = 0 symbol ===\n");
        case_k1_symbol(rep, ctx);
        std::printf("=== SF-33 N1: (4) exact pair, direction (Phi(x3), 0) ===\n");
        case_exact_pair_direction(rep, ctx);
        std::printf("=== SF-33 N1: (5) contracts ===\n");
        case_contracts(rep, ctx);
    } catch (const std::exception& e) {
        std::printf("[FAIL] unexpected exception: %s\n", e.what());
        rep.overall_pass = false;
    }
    std::printf("\n=== inlet_slab_jvp: %d checks, %s ===\n", rep.checks,
                rep.overall_pass ? "PASS" : "FAIL");
    return rep.overall_pass ? 0 : 1;
}
