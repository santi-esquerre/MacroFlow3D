/**
 * @file slab_contract_tests.cu
 * @brief SF-33 N0: fast contract tests of the inlet-slab foundation (grid/types, 4th-order
 * stencils, residual, metrics, CASE line). Standalone ctest runner in the printed-checks style of
 *        tests/flow/affine_periodic_flow_tests.cu; every grid is <= 32^3.
 *
 * Cases (acceptance items of the SF-33 N0 task):
 *   1  Fornberg weights: exactness on polynomials and the hard-coded weights*12 of the boundary
 * rows; device derivs4 on polynomials in x1. 2  Device derivs4 on sin(2 pi x1) cos(2 pi x2) sin(2
 * pi x3): max errors = prototype's at 16 and 32; observed orders 16 -> 32 (see case_orders for
 * which derivatives reach 3.8 there). 3  Exact discrete pair k = exp(0.7 sin 2 pi x1), u1 =
 * Phi(x3), u2 = 0 (understanding record section 3): r_F, r_out at roundoff for N = 12, 16;
 * crossed-pairing mutant prints r_F ~ |Phi''|. 4  k = 1, u = 0 (affine labels): r_F = r_out = 0. 5
 * Metrics: k = 1 control, den_ref fallback, device percentiles vs host numpy-style, e_v of the
 * exact pair. 6  CASE line byte for byte. 7  No allocation after prepare, deterministic repeated
 * evaluation, device vs host residual. A generic manufactured state at N = 12 is printed with %.17g
 * ([XCHK] lines) and gated against the values the SF-29 prototype (candidate_i.system,
 * metrics.fd_metrics(order=4), metrics.case_line) gives on the same analytic state (embedded
 * constants, computed offline). Option --with-64 (not used by ctest) adds N = 64 to case 2 and
 * applies the prototype selfcheck gate (order 32 -> 64 >= 3.8 for every derivative).
 */

#include "src/core/DeviceBuffer.cuh"
#include "src/core/DeviceSpan.cuh"
#include "src/core/Scalar.hpp"
#include "src/physics/streamfunctions/inlet_slab/InletSlabGrid.cuh"
#include "src/physics/streamfunctions/inlet_slab/SlabMetrics.cuh"
#include "src/physics/streamfunctions/inlet_slab/SlabResidual.cuh"
#include "src/physics/streamfunctions/inlet_slab/SlabStencils4.cuh"
#include "src/runtime/cuda_check.cuh"
#include "src/runtime/CudaContext.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
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
        overall_pass = overall_pass && cond;
    }
};

std::string fmt(const char* f, double v) {
    char b[128];
    std::snprintf(b, sizeof(b), f, v);
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

using F3 = std::function<real(real, real, real)>;

/// Full array (planes 0..N) sampled at x1 = j / N, x2 = m2 / N, x3 = m3 / N.
std::vector<real> sample_full(const sl::InletSlabGrid& g, const F3& f) {
    std::vector<real> a(g.full_size());
    for (int j = 0; j <= g.n; ++j)
        for (int m2 = 0; m2 < g.n; ++m2)
            for (int m3 = 0; m3 < g.n; ++m3)
                a[g.full_index(j, m2, m3)] = f(g.coord(j), g.coord(m2), g.coord(m3));
    return a;
}

std::vector<real> sample_plane(const sl::InletSlabGrid& g,
                               const std::function<real(real, real)>& f) {
    std::vector<real> a(g.plane_size());
    for (int m2 = 0; m2 < g.n; ++m2)
        for (int m3 = 0; m3 < g.n; ++m3)
            a[g.plane_index(m2, m3)] = f(g.coord(m2), g.coord(m3));
    return a;
}

std::vector<real> plane0(const sl::InletSlabGrid& g, const std::vector<real>& full) {
    return std::vector<real>(full.begin(),
                             full.begin() + static_cast<std::ptrdiff_t>(g.plane_size()));
}

real rms_host(const std::vector<real>& a) {
    real s = 0.0;
    for (real v : a)
        s += v * v;
    return std::sqrt(s / static_cast<real>(a.size()));
}

/// A state + inputs on the device and its host copies.
struct Fixture {
    sl::InletSlabGrid g;
    sl::SlabStageInputs in;
    sl::SlabReferenceData ref;
    DeviceBuffer<real> U1, U2, E;
    std::vector<real> hU1, hU2, hq, hgl[3], hv2, hv3, hvD[3], hpor[2];
};

struct FixtureSpec {
    F3 lnk, gl0, gl1, gl2, u1, u2;
    std::function<real(real, real)> v2in, v3in;
    F3 vD0, vD1, vD2; // reference velocity (metrics, v_rms)
    F3 por1, por2;    // oracle FULL labels (optional)
};

void build_fixture(CudaContext& ctx, int N, const FixtureSpec& s, Fixture& fx) {
    fx.g = sl::InletSlabGrid::make(N);
    const auto& g = fx.g;
    fx.in.allocate(g);
    fx.in.field = "test";
    fx.in.eps = 0.0;
    upload(fx.in.lnk, sample_full(g, s.lnk));
    sl::fill_q_from_lnk(ctx, g, fx.in);
    ctx.synchronize();
    fx.hq = download(fx.in.q.data(), g.full_size());
    const F3* gl[3] = {&s.gl0, &s.gl1, &s.gl2};
    for (int k = 0; k < 3; ++k) {
        fx.hgl[k] = sample_full(g, *gl[k]);
        upload(fx.in.grad_lnk[k], fx.hgl[k]);
    }
    fx.hU1 = sample_full(g, s.u1);
    fx.hU2 = sample_full(g, s.u2);
    upload(fx.U1, fx.hU1);
    upload(fx.U2, fx.hU2);
    upload(fx.in.u0[0], plane0(g, fx.hU1));
    upload(fx.in.u0[1], plane0(g, fx.hU2));
    fx.hv2 = sample_plane(g, s.v2in);
    fx.hv3 = sample_plane(g, s.v3in);
    upload(fx.in.vperp_in[0], fx.hv2);
    upload(fx.in.vperp_in[1], fx.hv3);
    const F3* vd[3] = {&s.vD0, &s.vD1, &s.vD2};
    real sv = 0.0;
    for (int k = 0; k < 3; ++k)
        fx.hvD[k] = sample_full(g, *vd[k]);
    for (std::size_t i = 0; i < g.full_size(); ++i)
        sv +=
            fx.hvD[0][i] * fx.hvD[0][i] + fx.hvD[1][i] * fx.hvD[1][i] + fx.hvD[2][i] * fx.hvD[2][i];
    fx.in.v_rms = std::sqrt(sv / static_cast<real>(g.full_size()));
    const bool with_por = static_cast<bool>(s.por1);
    fx.ref.allocate(g, true, with_por);
    for (int k = 0; k < 3; ++k)
        upload(fx.ref.vD[k], fx.hvD[k]);
    if (with_por) {
        fx.hpor[0] = sample_full(g, s.por1);
        fx.hpor[1] = sample_full(g, s.por2);
        upload(fx.ref.psi_or[0], fx.hpor[0]);
        upload(fx.ref.psi_or[1], fx.hpor[1]);
    }
    fx.E.resize(g.unknown_size());
}

/// Host evaluation of the residual with the same inline algebra and the host stencil table.
/// crossed = true builds the crossed-pairing mutant F1 = -q (L1 - S2), F2 = -q (L2 - S1).
void host_residual(const Fixture& fx, const sl::SlabStencilView& st, bool crossed,
                   std::vector<real>& E, sl::SlabResidualNorms& nrm) {
    const auto& g = fx.g;
    const std::size_t nf = g.field_size();
    E.assign(2 * nf, 0.0);
    real s1 = 0, s2 = 0, sq = 0, so = 0;
    for (int j = 1; j <= g.n; ++j)
        for (int m2 = 0; m2 < g.n; ++m2)
            for (int m3 = 0; m3 < g.n; ++m3) {
                const std::size_t fi = g.full_index(j, m2, m3);
                const std::size_t ui = g.unknown_index(j, m2, m3);
                real du1[3], du2[3], H1[6], H2[6], g1[3], g2[3];
                sl::derivs4_at(fx.hU1.data(), st, g, j, m2, m3, du1, H1);
                sl::derivs4_at(fx.hU2.data(), st, g, j, m2, m3, du2, H2);
                sl::slab_label_gradients(du1, du2, g1, g2);
                const real q = fx.hq[fi];
                if (j < g.n) {
                    const real glnk[3] = {fx.hgl[0][fi], fx.hgl[1][fi], fx.hgl[2][fi]};
                    sl::SlabPointState s;
                    sl::slab_equation_point(g1, g2, H1, H2, glnk, q, s);
                    const real F1 = crossed ? -q * (s.L1 - s.S2) : s.F1;
                    const real F2 = crossed ? -q * (s.L2 - s.S1) : s.F2;
                    E[ui] = F1;
                    E[nf + ui] = F2;
                    s1 += F1 * F1;
                    s2 += F2 * F2;
                    sq += q * q;
                } else {
                    const std::size_t pi = g.plane_index(m2, m3);
                    real e1, e2, d2, d3;
                    sl::slab_outlet_point(g1, g2, fx.hv2[pi], fx.hv3[pi], q, g.h, e1, e2, d2, d3);
                    E[ui] = e1;
                    E[nf + ui] = e2;
                    so += d2 * d2 + d3 * d3;
                }
            }
    const real neq = static_cast<real>(g.n - 1) * static_cast<real>(g.plane_size());
    nrm.r_F = std::sqrt((s1 / neq + s2 / neq) / 2.0) / std::sqrt(sq / neq);
    nrm.r_out = std::sqrt(so / static_cast<real>(g.plane_size())) / fx.in.v_rms;
}

// --------------------------------------------------------------------------------------------------------------
// Test kernels
// --------------------------------------------------------------------------------------------------------------

/// derivs4 on every vertex of planes 1..N: out[k * N^3 + idx], k = 0..2 gradient, 3..8 Hessian
/// (packed order).
__global__ void derivs_dump_kernel(sl::InletSlabGrid g, sl::SlabStencilView st, const real* U,
                                   real* out) {
    const std::size_t nf = g.field_size();
    const std::size_t np = g.plane_size();
    for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < nf; idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        const int j = 1 + static_cast<int>(idx / np);
        const std::size_t rem = idx % np;
        const int m2 = static_cast<int>(rem / g.n);
        const int m3 = static_cast<int>(rem % g.n);
        real gr[3], H[6];
        sl::derivs4_at(U, st, g, j, m2, m3, gr, H);
        for (int k = 0; k < 3; ++k)
            out[k * nf + idx] = gr[k];
        for (int k = 0; k < 6; ++k)
            out[(3 + k) * nf + idx] = H[k];
    }
}

std::vector<real> device_derivs(const sl::InletSlabGrid& g, const sl::SlabStencilTable& tab,
                                const std::vector<real>& hU) {
    DeviceBuffer<real> U, out;
    upload(U, hU);
    out.resize(9 * g.field_size());
    derivs_dump_kernel<<<64, 256>>>(g, tab.device_view(), U.data(), out.data());
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
    MACROFLOW3D_CUDA_CHECK(cudaDeviceSynchronize());
    return download(out.data(), out.size());
}

// --------------------------------------------------------------------------------------------------------------
// 0. grid validation
// --------------------------------------------------------------------------------------------------------------
void case_grid(TestReport& rep) {
    for (int bad : {0, 6, 7, 9, 15}) {
        bool threw = false;
        try {
            (void)sl::InletSlabGrid::make(bad);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        rep.check(threw,
                  "grid: InletSlabGrid::make(" + std::to_string(bad) + ") throws invalid_argument");
    }
    const auto g = sl::InletSlabGrid::make(12);
    rep.check(g.full_size() == 13u * 144u && g.field_size() == 1728u && g.unknown_size() == 3456u &&
                  g.full_index(2, 3, 5) == 5u + 12u * (3u + 12u * 2u) &&
                  g.unknown_index(1, 0, 0) == 0u && g.wrap(-2) == 10 && g.wrap(13) == 1 &&
                  g.h == 1.0 / 12.0,
              "grid: sizes, LOCKED index layout, exact wrap, h = 1/N (N = 12)");
}

// --------------------------------------------------------------------------------------------------------------
// 1. Fornberg weights
// --------------------------------------------------------------------------------------------------------------
void print_row(const char* name, int p, const sl::X1Stencil& s) {
    std::printf("  %-4s plane %2d: planes %d..%d weights*12 =", name, p, s.first,
                s.first + s.npts - 1);
    for (int k = 0; k < s.npts; ++k)
        std::printf(" %+.15g", 12.0 * s.w[k]);
    std::printf("\n");
}

bool row_equals(const sl::X1Stencil& s, int first, const std::vector<double>& w12, double tol,
                double& maxdiff) {
    maxdiff = 0.0;
    if (s.first != first || s.npts != static_cast<int>(w12.size()))
        return false;
    for (int k = 0; k < s.npts; ++k)
        maxdiff = std::max(maxdiff, std::abs(12.0 * s.w[k] - w12[k]));
    return maxdiff <= tol;
}

void case_fornberg(TestReport& rep, CudaContext& ctx) {
    (void)ctx;
    const int N = 16;
    const auto t = sl::build_x1_stencils4(N);
    for (int p : {0, 1, 2, N / 2, N - 2, N - 1, N}) {
        print_row("d1", p, t.d1[p]);
        print_row("d11", p, t.d11[p]);
    }
    double md;
    bool ok = row_equals(t.d1[1], 0, {-3, -10, 18, -6, 1}, 1e-12, md);
    rep.check(ok, "fornberg: d1 plane 1 weights*12 = (-3, -10, 18, -6, 1) on planes 0..4",
              fmt("max|diff| = %.2e", md));
    ok = row_equals(t.d11[1], 0, {10, -15, -4, 14, -6, 1}, 1e-12, md);
    rep.check(ok, "fornberg: d11 plane 1 weights*12 = (10, -15, -4, 14, -6, 1) on planes 0..5",
              fmt("max|diff| = %.2e", md));
    ok = row_equals(t.d1[N], N - 4, {3, -16, 36, -48, 25}, 1e-12, md);
    rep.check(ok, "fornberg: d1 plane N weights*12 = (3, -16, 36, -48, 25) on planes N-4..N",
              fmt("max|diff| = %.2e", md));
    ok = row_equals(t.d1[0], 0, {-25, 48, -36, 16, -3}, 1e-12, md);
    rep.check(ok, "fornberg: d1 plane 0 weights*12 = (-25, 48, -36, 16, -3) on planes 0..4",
              fmt("max|diff| = %.2e", md));
    ok = row_equals(t.d1[N - 1], N - 4, {-1, 6, -18, 10, 3}, 1e-12, md);
    rep.check(ok, "fornberg: d1 plane N-1 = mirror of plane 1 (-1, 6, -18, 10, 3) on planes N-4..N",
              fmt("max|diff| = %.2e", md));
    ok = row_equals(t.d11[N - 1], N - 5, {1, -6, 14, -4, -15, 10}, 1e-12, md);
    rep.check(
        ok, "fornberg: d11 plane N-1 = mirror of plane 1 (1, -6, 14, -4, -15, 10) on planes N-5..N",
        fmt("max|diff| = %.2e", md));
    ok = row_equals(t.d1[N / 2], N / 2 - 2, {1, -8, 0, 8, -1}, 0.0, md) &&
         row_equals(t.d11[N / 2], N / 2 - 2, {-1, 16, -30, 16, -1}, 0.0, md);
    rep.check(ok,
              "fornberg: centered rows are the literal (1,-8,0,8,-1)/12 and (-1,16,-30,16,-1)/12");

    // Exactness on monomials x^deg, deg = 0..npts-1, every plane 0..N (prototype selfcheck error
    // measure).
    const double h = 1.0 / N;
    struct Cat {
        const char* name;
        double err;
        int maxdeg;
    };
    Cat cats[6] = {{"d1 centered (planes 2..N-2), degree <= 4", 0, 4},
                   {"d1 skewed (planes 1, N-1), degree <= 4", 0, 4},
                   {"d1 one-sided (planes 0, N), degree <= 4", 0, 4},
                   {"d11 centered (planes 2..N-2), degree <= 4 (symmetric: also 5)", 0, 5},
                   {"d11 skewed 6-point (planes 1, N-1), degree <= 5", 0, 5},
                   {"d11 one-sided 6-point (planes 0, N), degree <= 5", 0, 5}};
    for (int deriv = 1; deriv <= 2; ++deriv) {
        for (int p = 0; p <= N; ++p) {
            const sl::X1Stencil& s = (deriv == 1) ? t.d1[p] : t.d11[p];
            const int cat =
                (deriv - 1) * 3 + ((p == 0 || p == N) ? 2 : ((p == 1 || p == N - 1) ? 1 : 0));
            for (int deg = 0; deg <= s.npts - 1; ++deg) {
                const double xp = p * h;
                double exact = 0.0;
                if (deriv == 1 && deg >= 1)
                    exact = deg * std::pow(xp, deg - 1);
                if (deriv == 2 && deg >= 2)
                    exact = deg * (deg - 1) * std::pow(xp, deg - 2);
                double val = 0.0;
                for (int k = 0; k < s.npts; ++k)
                    val += s.w[k] * std::pow((s.first + k) * h, deg);
                val /= std::pow(h, deriv);
                cats[cat].err =
                    std::max(cats[cat].err, std::abs(val - exact) / std::max(1.0, std::abs(exact)));
            }
        }
    }
    for (const Cat& c : cats) {
        rep.check(c.err <= 1e-9, std::string("fornberg: ") + c.name + " exact (rel err <= 1e-9)",
                  fmt("max rel err = %.2e", c.err));
    }

    // Device derivs4 on x1^deg (deg 0..4), planes 1..N (prototype selfcheck: max abs err <= 1e-8).
    const auto g = sl::InletSlabGrid::make(N);
    sl::SlabStencilTable tab;
    tab.build(g);
    double emax = 0.0;
    for (int deg = 0; deg <= 4; ++deg) {
        const auto U = sample_full(g, [deg](real x1, real, real) { return std::pow(x1, deg); });
        const auto d = device_derivs(g, tab, U);
        double e = 0.0;
        for (int j = 1; j <= N; ++j) {
            const double x = g.coord(j);
            const double e1 = deg >= 1 ? deg * std::pow(x, deg - 1) : 0.0;
            const double e11 = deg >= 2 ? deg * (deg - 1) * std::pow(x, deg - 2) : 0.0;
            for (int m = 0; m < N * N; ++m) {
                const std::size_t i = static_cast<std::size_t>(j - 1) * N * N + m;
                e = std::max(
                    e, std::max(std::abs(d[i] - e1), std::abs(d[3 * g.field_size() + i] - e11)));
                // in-plane and mixed derivatives of a function of x1 only vanish
                for (int k : {1, 2, 4, 5, 6, 7, 8})
                    e = std::max(e, std::abs(d[k * g.field_size() + i]));
            }
        }
        std::printf("  device derivs4 on x1^%d: max abs err (d1, d11; others 0) = %.2e\n", deg, e);
        emax = std::max(emax, e);
    }
    rep.check(emax <= 1e-8,
              "fornberg: device derivs4 exact on x1^0..x1^4, planes 1..N (max abs err <= 1e-8)",
              fmt("max = %.2e", emax));
}

// --------------------------------------------------------------------------------------------------------------
// 2. observed orders of the device derivs4
// --------------------------------------------------------------------------------------------------------------
/**
 * Max errors of the device derivs4 on sin(2 pi x1) cos(2 pi x2) sin(2 pi x3), planes 1..N.
 *
 * Gates (ctest, N <= 32):
 *   (a) every derivative: the device max errors at N = 16 and 32 equal those of the prototype's
 * derivs4 (candidate_i.derivs4 + Stencils4, recorded below with %.10e) to 1e-6 relative -> same
 * discrete operators; (b) observed order 16 -> 32 >= 3.8 for every derivative whose 16 -> 32 order
 * reaches it in the prototype. d1, d12 and d13 do NOT reach 3.8 on 16 -> 32 in the prototype either
 * (3.719 / 3.760 / 3.760, dominated by the one-sided rows of plane N); their 16 -> 32 orders are
 * printed as [INFO], and with --with-64 (not run by ctest) the prototype's own selfcheck gate,
 * order 32 -> 64 >= 3.8 for every derivative, is applied.
 */
void case_orders(TestReport& rep, bool with64) {
    const char* names[9] = {"d1", "d2", "d3", "d11", "d22", "d33", "d12", "d13", "d23"};
    const int map[9] = {
        0,           1, 2, 3 + sl::kH00, 3 + sl::kH11, 3 + sl::kH22, 3 + sl::kH01, 3 + sl::kH02,
        3 + sl::kH12};
    // prototype (SF-29 candidate_i.derivs4) max errors at N = 16, 32 on the same function and
    // planes
    const double proto[2][9] = {
        {2.3141105861e-02, 4.8901665551e-03, 4.8901665551e-03, 5.0347078024e-01, 1.0289134770e-02,
         1.0289134770e-02, 1.7601251512e-01, 1.7601251512e-01, 6.1427731569e-02},
        {1.7579034865e-03, 3.0987377193e-04, 3.0987377192e-04, 1.7588389679e-02, 6.4974351911e-04,
         6.4974351911e-04, 1.2991682960e-02, 1.2991682960e-02, 3.8938926399e-03}};
    const bool below_in_prototype[9] = {true, false, false, false, false, false, true, true, false};
    const int nr = with64 ? 3 : 2;
    double err[3][9];
    const int Ns[3] = {16, 32, 64};
    for (int r = 0; r < nr; ++r) {
        const auto g = sl::InletSlabGrid::make(Ns[r]);
        sl::SlabStencilTable tab;
        tab.build(g);
        const auto U = sample_full(g, [](real x1, real x2, real x3) {
            return std::sin(kTwoPi * x1) * std::cos(kTwoPi * x2) * std::sin(kTwoPi * x3);
        });
        const auto d = device_derivs(g, tab, U);
        for (int q = 0; q < 9; ++q)
            err[r][q] = 0.0;
        const double A = kTwoPi;
        for (int j = 1; j <= g.n; ++j)
            for (int m2 = 0; m2 < g.n; ++m2)
                for (int m3 = 0; m3 < g.n; ++m3) {
                    const double x = g.coord(j), y = g.coord(m2), z = g.coord(m3);
                    const double sx = std::sin(A * x), cx = std::cos(A * x), sy = std::sin(A * y),
                                 cy = std::cos(A * y), sz = std::sin(A * z), cz = std::cos(A * z);
                    const double ex[9] = {
                        A * cx * cy * sz,      -A * sx * sy * sz,     A * sx * cy * cz,
                        -A * A * sx * cy * sz, -A * A * sx * cy * sz, -A * A * sx * cy * sz,
                        -A * A * cx * sy * sz, A * A * cx * cy * cz,  -A * A * sx * sy * cz};
                    const std::size_t i = g.unknown_index(j, m2, m3);
                    for (int q = 0; q < 9; ++q)
                        err[r][q] =
                            std::max(err[r][q], std::abs(d[map[q] * g.field_size() + i] - ex[q]));
                }
    }
    for (int q = 0; q < 9; ++q) {
        const double o = std::log(err[0][q] / err[1][q]) / std::log(2.0);
        const double rel = std::max(std::abs(err[0][q] - proto[0][q]) / proto[0][q],
                                    std::abs(err[1][q] - proto[1][q]) / proto[1][q]);
        char det[200];
        std::snprintf(det, sizeof(det),
                      "max err 16: %.4e  32: %.4e  order %.3f | prototype rel diff %.1e", err[0][q],
                      err[1][q], o, rel);
        rep.check(rel <= 1e-6,
                  std::string("orders: derivs4 ") + names[q] +
                      " max errors at 16, 32 = prototype derivs4 (rel <= 1e-6)",
                  det);
        if (!below_in_prototype[q]) {
            rep.check(o >= 3.8,
                      std::string("orders: derivs4 ") + names[q] +
                          " observed order 16 -> 32 >= 3.8",
                      fmt("order %.3f", o));
        } else {
            std::printf(
                "[INFO] orders: derivs4 %s observed order 16 -> 32 = %.3f < 3.8, as in the "
                "prototype "
                "(one-sided plane-N rows pre-asymptotic at 16); gated on 32 -> 64 with --with-64\n",
                names[q], o);
        }
        if (with64) {
            const double o64 = std::log(err[1][q] / err[2][q]) / std::log(2.0);
            rep.check(o64 >= 3.8,
                      std::string("orders (--with-64): derivs4 ") + names[q] +
                          " observed order 32 -> 64 >= 3.8",
                      fmt("max err 64: %.4e", err[2][q]) + fmt("  order %.3f", o64));
        }
    }
}

// --------------------------------------------------------------------------------------------------------------
// 3. exact discrete pair, 4. k = 1 control
// --------------------------------------------------------------------------------------------------------------
FixtureSpec exact_pair_spec(bool with_por) {
    FixtureSpec s;
    s.lnk = [](real x1, real, real) { return 0.7 * std::sin(kTwoPi * x1); };
    s.gl0 = [](real x1, real, real) { return 0.7 * kTwoPi * std::cos(kTwoPi * x1); };
    s.gl1 = [](real, real, real) { return 0.0; };
    s.gl2 = [](real, real, real) { return 0.0; };
    s.u1 = [](real, real, real x3) {
        return 0.3 * std::sin(kTwoPi * x3) + 0.1 * std::cos(2.0 * kTwoPi * x3);
    };
    s.u2 = [](real, real, real) { return 0.0; };
    s.v2in = [](real, real) { return 0.0; };
    s.v3in = [](real, real) { return 0.0; };
    s.vD0 = [](real, real, real) { return 1.0; };
    s.vD1 = [](real, real, real) { return 0.0; };
    s.vD2 = [](real, real, real) { return 0.0; };
    if (with_por) {
        s.por1 = [](real, real x2, real x3) {
            return x2 + 0.3 * std::sin(kTwoPi * x3) + 0.1 * std::cos(2.0 * kTwoPi * x3);
        };
        s.por2 = [](real, real, real x3) { return x3; };
    }
    return s;
}

FixtureSpec k1_spec() {
    FixtureSpec s;
    auto zero = [](real, real, real) { return 0.0; };
    s.lnk = zero;
    s.gl0 = zero;
    s.gl1 = zero;
    s.gl2 = zero;
    s.u1 = zero;
    s.u2 = zero;
    s.v2in = [](real, real) { return 0.0; };
    s.v3in = [](real, real) { return 0.0; };
    s.vD0 = [](real, real, real) { return 1.0; };
    s.vD1 = zero;
    s.vD2 = zero;
    s.por1 = [](real, real x2, real) { return x2; };
    s.por2 = [](real, real, real x3) { return x3; };
    return s;
}

void case_exact_pair(TestReport& rep, CudaContext& ctx) {
    for (int N : {12, 16}) {
        Fixture fx;
        build_fixture(ctx, N, exact_pair_spec(false), fx);
        sl::SlabResidualWorkspace ws;
        ws.prepare(fx.g);
        sl::SlabResidualNorms nrm;
        sl::evaluate_residual(ctx, fx.g, fx.in, cspan(fx.U1), cspan(fx.U2), mspan(fx.E), ws, &nrm);
        char det[160];
        std::snprintf(det, sizeof(det), "r_F = %.3e  r_out = %.3e", nrm.r_F, nrm.r_out);
        rep.check(nrm.r_F <= 1e-12 && nrm.r_out <= 1e-12,
                  "exact pair k = exp(0.7 sin 2pi x1), u1 = Phi(x3), u2 = 0, N = " +
                      std::to_string(N) + ": r_F <= 1e-12 and r_out <= 1e-12",
                  det);
        // crossed-pairing mutant (host, same inline algebra)
        sl::SlabStencilTable tab;
        tab.build(fx.g);
        std::vector<real> Ex;
        sl::SlabResidualNorms nx;
        host_residual(fx, tab.host_view(), true, Ex, nx);
        std::snprintf(det, sizeof(det), "r_F(crossed) = %.3e (|Phi''| ~ 1e1)", nx.r_F);
        std::printf(
            "[INFO] exact pair N = %d: crossed-pairing mutant r_F = %.3e, same-index r_F = %.3e\n",
            N, nx.r_F, nrm.r_F);
        rep.check(nx.r_F > 1e-6,
                  "exact pair N = " + std::to_string(N) + ": crossed pairing is NOT a solution",
                  det);
    }
}

void case_k1(TestReport& rep, CudaContext& ctx) {
    Fixture fx;
    build_fixture(ctx, 16, k1_spec(), fx);
    sl::SlabResidualWorkspace ws;
    ws.prepare(fx.g);
    sl::SlabResidualNorms nrm;
    sl::evaluate_residual(ctx, fx.g, fx.in, cspan(fx.U1), cspan(fx.U2), mspan(fx.E), ws, &nrm);
    const auto E = download(fx.E.data(), fx.E.size());
    double emax = 0.0;
    for (real v : E)
        emax = std::max(emax, std::abs(v));
    char det[160];
    std::snprintf(det, sizeof(det), "r_F = %.3e  r_out = %.3e  max|E| = %.3e", nrm.r_F, nrm.r_out,
                  emax);
    rep.check(nrm.r_F <= 1e-15 && nrm.r_out <= 1e-15 && emax == 0.0,
              "k = 1, u = 0 (affine labels), N = 16: r_F, r_out <= 1e-15 and E == 0", det);
}

// --------------------------------------------------------------------------------------------------------------
// 5. metrics
// --------------------------------------------------------------------------------------------------------------
/// Independent host implementation of numpy.percentile(method="linear") on an unsorted array.
double numpy_percentile_host(std::vector<double> a, double pct) {
    std::sort(a.begin(), a.end());
    const std::size_t n = a.size();
    const double virt = static_cast<double>(n - 1) * (pct / 100.0);
    double prev = std::floor(virt);
    std::size_t lo = static_cast<std::size_t>(prev), hi = lo + 1;
    if (virt >= static_cast<double>(n - 1)) {
        lo = hi = n - 1;
        prev = -1.0;
    }
    const double gma = virt - prev;
    const double diff = a[hi] - a[lo];
    return gma >= 0.5 ? a[hi] - diff * (1.0 - gma) : a[lo] + diff * gma;
}

void case_metrics(TestReport& rep, CudaContext& ctx) {
    // (i) k = 1 control
    {
        Fixture fx;
        build_fixture(ctx, 16, k1_spec(), fx);
        sl::SlabMetricsWorkspace mws;
        mws.prepare(fx.g);
        const auto m = sl::evaluate_metrics(ctx, fx.g, cspan(fx.U1), cspan(fx.U2), fx.ref, mws);
        char det[256];
        std::snprintf(det, sizeof(det),
                      "e_v=%.3e e_i=(%.3e,%.3e) e_div=%.3e min_c=%.17g p50=%.17g e_psi=(%.3e,%.3e) "
                      "den=(%.3e,%.3e)",
                      m.e_v, m.e_i1, m.e_i2, m.e_div, m.min_c, m.p50, m.e_psi1, m.e_psi2,
                      m.den_used1, m.den_used2);
        // e_div is roundoff, not 0: div_h of c = e1 is d1_fd4(1) = (sum of the d1 weights)/h, and
        // the floating-point sums of the Fornberg one-sided / skewed weights are +-2e-16 (plane
        // 0: 2.2e-16, plane 1: -1.5e-16); the prototype's metrics.fd_metrics gives the identical
        // e_div = 1.197e-15 on this control (offline check).
        rep.check(m.e_v == 0.0 && m.e_i1 == 0.0 && m.e_i2 == 0.0 && m.e_div <= 1e-14 &&
                      m.min_c == 1.0 && m.p50 == 1.0 && m.p0_1 == 1.0 && m.has_psi &&
                      m.e_psi1 == 0.0 && m.e_psi2 == 0.0 && m.e_psi == 0.0 && m.den_used1 == 0.0 &&
                      m.den_used2 == 0.0 && m.v_rms == 1.0 && m.nonfinite == 0,
                  "metrics (i): k = 1, u = 0, affine psi_or, vD = e1: e_v = e_i = e_psi = 0, e_div "
                  "roundoff (<= 1e-14), "
                  "min|c| = p50 = 1, den_ref = 0 branch",
                  det);
    }
    // (ii) den_ref fallback and (iii) percentiles on a non-trivial state
    {
        FixtureSpec s = k1_spec();
        s.u1 = [](real x1, real x2, real x3) {
            return 0.05 * std::sin(kTwoPi * (x1 + x2)) + 0.03 * x1 * x1 * std::cos(kTwoPi * x3);
        };
        s.u2 = [](real x1, real x2, real x3) {
            return 0.04 * std::cos(kTwoPi * (x2 - x3)) * (1.0 + x1) +
                   0.02 * std::sin(kTwoPi * x1) * std::sin(kTwoPi * x2);
        };
        s.vD1 = [](real, real, real x3) { return 0.05 * std::sin(kTwoPi * x3); };
        s.por1 = [](real x1, real x2, real x3) {
            return x2 + 0.2 * std::sin(kTwoPi * x1) * std::cos(kTwoPi * x2) +
                   0.01 * std::cos(kTwoPi * x3);
        };
        s.por2 = [](real x1, real, real x3) {
            return x3 + 1e-9 * std::sin(kTwoPi * x1) * std::sin(kTwoPi * x3);
        };
        Fixture fx;
        build_fixture(ctx, 16, s, fx);
        sl::SlabMetricsWorkspace mws;
        mws.prepare(fx.g);
        const auto m = sl::evaluate_metrics(ctx, fx.g, cspan(fx.U1), cspan(fx.U2), fx.ref, mws);
        // host expectation from the definition
        const auto& g = fx.g;
        double sa[2] = {0, 0}, sd[2] = {0, 0};
        for (int j = 0; j <= g.n; ++j)
            for (int m2 = 0; m2 < g.n; ++m2)
                for (int m3 = 0; m3 < g.n; ++m3) {
                    const std::size_t i = g.full_index(j, m2, m3);
                    const double x[2] = {g.coord(m2), g.coord(m3)};
                    const double p[2] = {x[0] + fx.hU1[i], x[1] + fx.hU2[i]};
                    for (int k = 0; k < 2; ++k) {
                        sa[k] += (p[k] - fx.hpor[k][i]) * (p[k] - fx.hpor[k][i]);
                        sd[k] += (fx.hpor[k][i] - x[k]) * (fx.hpor[k][i] - x[k]);
                    }
                }
        const double n = static_cast<double>(g.full_size());
        const double den1 = std::sqrt(sd[0] / n), den2 = std::sqrt(sd[1] / n),
                     dref = std::max(den1, den2);
        const double e1 = std::sqrt(sa[0] / n) / den1;
        const double e2 = std::sqrt(sa[1] / n) / dref; // den2 <= 1e-6 den_ref -> den_ref
        const double r1 = std::abs(m.e_psi1 - e1) / e1, r2 = std::abs(m.e_psi2 - e2) / e2;
        char det[256];
        std::snprintf(
            det, sizeof(det),
            "den2/den1 = %.2e  e_psi = (%.6e, %.6e) host (%.6e, %.6e) rel diff (%.1e, %.1e)",
            den2 / den1, m.e_psi1, m.e_psi2, e1, e2, r1, r2);
        rep.check(
            den2 <= 1e-6 * den1 && m.den_used2 == m.den_used1 && r1 <= 1e-12 && r2 <= 1e-12 &&
                m.e_psi == std::max(m.e_psi1, m.e_psi2),
            "metrics (ii): den_2 <= 1e-6 den_1 -> label 2 normalized by den_ref (host definition)",
            det);

        // (iii) device percentiles vs a host numpy-style computation on the downloaded |c|
        const auto cn = download(mws.norm_c().data(), g.full_size());
        const auto cs = download(mws.sorted_c().data(), g.full_size());
        std::vector<double> hs = cn;
        std::sort(hs.begin(), hs.end());
        const bool sorted_same = (hs == cs);
        const double pct[4] = {0.1, 1.0, 5.0, 50.0};
        const double dev[4] = {m.p0_1, m.p1, m.p5, m.p50};
        double rmax = 0.0;
        for (int p = 0; p < 4; ++p) {
            const double hv = numpy_percentile_host(cn, pct[p]);
            rmax = std::max(rmax, std::abs(dev[p] - hv) / std::abs(hv));
            std::printf("  percentile %4.1f %%: device %.17g host %.17g\n", pct[p], dev[p], hv);
        }
        std::vector<double> vn(g.full_size());
        for (std::size_t i = 0; i < vn.size(); ++i)
            vn[i] = std::sqrt(fx.hvD[0][i] * fx.hvD[0][i] + fx.hvD[1][i] * fx.hvD[1][i] +
                              fx.hvD[2][i] * fx.hvD[2][i]);
        for (int p = 0; p < 4; ++p) {
            const double hv = numpy_percentile_host(vn, pct[p]);
            rmax = std::max(rmax, std::abs(m.vD_p[p] - hv) / std::abs(hv));
        }
        const double minc = *std::min_element(cn.begin(), cn.end());
        std::snprintf(det, sizeof(det), "max rel diff = %.2e; sorted |c| bitwise = %s; min|c| %s",
                      rmax, sorted_same ? "yes" : "no", m.min_c == minc ? "equal" : "DIFFERENT");
        rep.check(rmax <= 1e-13 && sorted_same && m.min_c == minc,
                  "metrics (iii): device percentiles of |c| and |vD| = host numpy-linear on the "
                  "downloaded arrays",
                  det);
    }
    // (iv) e_v of the exact pair
    {
        Fixture fx;
        build_fixture(ctx, 16, exact_pair_spec(true), fx);
        sl::SlabMetricsWorkspace mws;
        mws.prepare(fx.g);
        const auto m = sl::evaluate_metrics(ctx, fx.g, cspan(fx.U1), cspan(fx.U2), fx.ref, mws);
        char det[256];
        std::snprintf(det, sizeof(det), "e_v = %.3e  e_psi = %.3e  min_c = %.17g", m.e_v, m.e_psi,
                      m.min_c);
        rep.check(
            m.e_v <= 1e-14 && m.e_psi <= 1e-15,
            "metrics (iv): exact pair vs vD = e1: e_v = 0 to roundoff (<= 1e-14), labels = oracle",
            det);
        // perturbed reference vD = e1 + 1e-3 (0, sin 2 pi x2, 0)
        FixtureSpec s = exact_pair_spec(true);
        s.vD1 = [](real, real x2, real) { return 1e-3 * std::sin(kTwoPi * x2); };
        Fixture fp;
        build_fixture(ctx, 16, s, fp);
        const auto mp = sl::evaluate_metrics(ctx, fp.g, cspan(fp.U1), cspan(fp.U2), fp.ref, mws);
        const double rms_sin = rms_host(fp.hvD[1]) / 1e-3;
        const double hand = 1e-3 * rms_sin / fp.in.v_rms;
        const double rel = std::abs(mp.e_v - hand) / hand;
        std::snprintf(det, sizeof(det), "e_v = %.17g  hand = %.17g  rel diff = %.2e", mp.e_v, hand,
                      rel);
        rep.check(rel <= 1e-12,
                  "metrics (iv): exact pair vs vD = e1 + 1e-3 (0, sin 2pi x2, 0): e_v = 1e-3 "
                  "RMS(sin)/RMS(vD)",
                  det);
    }
}

// --------------------------------------------------------------------------------------------------------------
// 6. CASE line
// --------------------------------------------------------------------------------------------------------------
void case_case_line(TestReport& rep) {
    sl::SlabMetrics m;
    m.e_v = 5.506e-03;
    m.e_psi = 7.727e-02;
    m.e_i1 = 1.234e-05;
    m.e_i2 = 2.345e-06;
    m.e_div = 3.456e-04;
    m.min_c = 7.890e-01;
    m.p0_1 = 8.000e-01;
    m.p1 = 8.100e-01;
    m.p5 = 8.500e-01;
    m.p50 = 9.900e-01;
    m.has_psi = true;
    m.e_psi1 = 7.727e-02;
    m.e_psi2 = 5.859e-02;
    sl::CaseLineExtras ex;
    ex.r_F = 8.631e-14;
    ex.its = 10;
    ex.t = 12.3;
    const std::string want = "CASE field=gauss eps=0.25 N=16 cand=i1o4 | r_F=8.631e-14 its=10 | "
                             "e_v=5.506e-03 e_psi=7.727e-02 "
                             "e_i=(1.234e-05,2.345e-06) e_div=3.456e-04 min_c=7.890e-01 "
                             "p0.1=8.000e-01 p1=8.100e-01 p5=8.500e-01 "
                             "p50=9.900e-01 | t=12.3 | e_psi1=7.727e-02 e_psi2=5.859e-02";
    const std::string got = sl::format_case_line("gauss", 0.25, 16, "i1o4", m, ex);
    std::printf("  got:  %s\n", got.c_str());
    rep.check(got == want, "case line: reference string reproduced byte for byte");
    const std::string g1 = sl::format_case_line("gauss", 1.0, 16, "i1o4", m, ex);
    rep.check(g1.find(" eps=1 N=16 ") != std::string::npos, "case line: %g of eps = 1 -> 'eps=1'");
    const std::string g2 = sl::format_case_line("gauss", 0.8125, 16, "i1o4", m, ex);
    rep.check(g2.find(" eps=0.8125 N=16 ") != std::string::npos,
              "case line: %g of eps = 0.8125 -> 'eps=0.8125'");
    sl::SlabMetrics mo = m;
    mo.has_psi = false;
    mo.e_psi = std::nan("");
    const std::string g3 =
        sl::format_case_line("gauss_ch", 0.5, 24, "oracle_fd4", mo, sl::CaseLineExtras{});
    std::printf("  got:  %s\n", g3.c_str());
    const std::string want3 = "CASE field=gauss_ch eps=0.5 N=24 cand=oracle_fd4 | r_F=nan its=nan "
                              "| e_v=5.506e-03 e_psi=nan "
                              "e_i=(1.234e-05,2.345e-06) e_div=3.456e-04 min_c=7.890e-01 "
                              "p0.1=8.000e-01 p1=8.100e-01 p5=8.500e-01 "
                              "p50=9.900e-01 | t=0.0";
    rep.check(g3 == want3,
              "case line: absent r_F / its print 'nan', no e_psi suffix without labels");
}

// --------------------------------------------------------------------------------------------------------------
// 7. allocation stability, determinism, device vs host residual; [XCHK] lines for the prototype
// comparison
// --------------------------------------------------------------------------------------------------------------
FixtureSpec generic_spec() {
    FixtureSpec s;
    s.lnk = [](real x1, real x2, real x3) {
        return 0.4 * std::sin(kTwoPi * x1) * std::cos(kTwoPi * x2) +
               0.25 * std::sin(kTwoPi * x3 + 0.3);
    };
    s.gl0 = [](real x1, real x2, real) {
        return 0.4 * kTwoPi * std::cos(kTwoPi * x1) * std::cos(kTwoPi * x2);
    };
    s.gl1 = [](real x1, real x2, real) {
        return -0.4 * kTwoPi * std::sin(kTwoPi * x1) * std::sin(kTwoPi * x2);
    };
    s.gl2 = [](real, real, real x3) { return 0.25 * kTwoPi * std::cos(kTwoPi * x3 + 0.3); };
    s.u1 = [](real x1, real x2, real x3) {
        return 0.05 * std::sin(kTwoPi * (x1 + x2)) + 0.03 * x1 * x1 * std::cos(kTwoPi * x3);
    };
    s.u2 = [](real x1, real x2, real x3) {
        return 0.04 * std::cos(kTwoPi * (x2 - x3)) * (1.0 + x1) +
               0.02 * std::sin(kTwoPi * x1) * std::sin(kTwoPi * x2);
    };
    s.v2in = [](real, real x3) { return 0.1 * std::sin(kTwoPi * x3); };
    s.v3in = [](real x2, real) { return -0.05 * std::cos(kTwoPi * x2); };
    s.vD0 = [](real, real x2, real) { return 1.0 + 0.1 * std::cos(kTwoPi * x2); };
    s.vD1 = [](real, real, real x3) { return 0.05 * std::sin(kTwoPi * x3); };
    s.vD2 = [](real x1, real, real) { return 0.02 * std::cos(kTwoPi * x1); };
    s.por1 = [](real x1, real x2, real x3) {
        return x2 +
               0.9 * (0.05 * std::sin(kTwoPi * (x1 + x2)) + 0.03 * x1 * x1 * std::cos(kTwoPi * x3));
    };
    s.por2 = [](real x1, real x2, real x3) {
        return x3 + 1.1 * (0.04 * std::cos(kTwoPi * (x2 - x3)) * (1.0 + x1)) +
               0.001 * std::sin(kTwoPi * x1);
    };
    return s;
}

void case_contracts(TestReport& rep, CudaContext& ctx) {
    Fixture fx;
    build_fixture(ctx, 12, generic_spec(), fx);
    const auto& g = fx.g;
    sl::SlabResidualWorkspace ws;
    ws.prepare(g);
    sl::SlabMetricsWorkspace mws;
    mws.prepare(g);
    // assemble / extract round trip (view convention)
    DeviceBuffer<real> uvec(g.unknown_size()), A1(g.full_size()), A2(g.full_size());
    sl::extract_unknowns(ctx, g, cspan(fx.U1), cspan(fx.U2), mspan(uvec));
    sl::assemble_full_planes(ctx, g, cspan(uvec), fx.in, mspan(A1), mspan(A2));
    ctx.synchronize();
    rep.check(
        download(A1.data(), A1.size()) == fx.hU1 && download(A2.data(), A2.size()) == fx.hU2,
        "contracts: extract_unknowns + assemble_full_planes reproduce the full arrays bitwise");

    const void* p_res[2] = {ws.partials_data(), ws.sums_data()};
    const std::size_t b_res = ws.allocated_bytes();
    const auto p_met = mws.storage_pointers();
    const std::size_t b_met = mws.allocated_bytes();
    std::size_t free0 = 0, total = 0, free1 = 0;
    MACROFLOW3D_CUDA_CHECK(cudaMemGetInfo(&free0, &total));

    sl::SlabResidualNorms n1, n2;
    sl::evaluate_residual(ctx, g, fx.in, cspan(fx.U1), cspan(fx.U2), mspan(fx.E), ws, &n1);
    const auto E1 = download(fx.E.data(), fx.E.size());
    const auto m1 = sl::evaluate_metrics(ctx, g, cspan(fx.U1), cspan(fx.U2), fx.ref, mws);
    for (int rep_i = 0; rep_i < 3; ++rep_i) {
        sl::evaluate_residual(ctx, g, fx.in, cspan(fx.U1), cspan(fx.U2), mspan(fx.E), ws, nullptr);
        sl::evaluate_residual(ctx, g, fx.in, cspan(fx.U1), cspan(fx.U2), mspan(fx.E), ws, &n2);
        (void)sl::evaluate_metrics(ctx, g, cspan(fx.U1), cspan(fx.U2), fx.ref, mws);
    }
    const auto m2 = sl::evaluate_metrics(ctx, g, cspan(fx.U1), cspan(fx.U2), fx.ref, mws);
    const auto E2 = download(fx.E.data(), fx.E.size());
    MACROFLOW3D_CUDA_CHECK(cudaMemGetInfo(&free1, &total));
    rep.check(p_res[0] == ws.partials_data() && p_res[1] == ws.sums_data() &&
                  b_res == ws.allocated_bytes() && p_met == mws.storage_pointers() &&
                  b_met == mws.allocated_bytes(),
              "contracts: no (re)allocation in evaluate_residual / evaluate_metrics after prepare "
              "(pointers, bytes)");
    std::printf(
        "[INFO] contracts: device free memory before/after repeated evaluations: %zu / %zu bytes\n",
        free0, free1);
    rep.check(n1.r_F == n2.r_F && n1.r_out == n2.r_out && E1 == E2 && m1.e_v == m2.e_v &&
                  m1.e_div == m2.e_div && m1.e_psi == m2.e_psi && m1.p0_1 == m2.p0_1,
              "contracts: repeated evaluation is bitwise reproducible (E, norms, metrics)");

    // device vs host evaluation of the same inline algebra
    sl::SlabStencilTable tab;
    tab.build(g);
    std::vector<real> Eh;
    sl::SlabResidualNorms nh;
    host_residual(fx, tab.host_view(), false, Eh, nh);
    double emax = 0.0, escale = 0.0;
    for (std::size_t i = 0; i < Eh.size(); ++i) {
        emax = std::max(emax, std::abs(Eh[i] - E1[i]));
        escale = std::max(escale, std::abs(Eh[i]));
    }
    char det[256];
    std::snprintf(det, sizeof(det),
                  "max|E_dev - E_host| / max|E| = %.2e;  r_F dev %.17g host %.17g", emax / escale,
                  n1.r_F, nh.r_F);
    rep.check(emax / escale <= 1e-12 && std::abs(n1.r_F - nh.r_F) <= 1e-12 * nh.r_F &&
                  std::abs(n1.r_out - nh.r_out) <= 1e-12 * nh.r_out,
              "contracts: device residual = host evaluation of the same algebra (rel <= 1e-12), "
              "generic state N = 12",
              det);

    // [XCHK] values for the offline comparison with the SF-29 prototype (candidate_i.system,
    // metrics.fd_metrics)
    double sabs = 0.0, ssq = 0.0;
    for (real v : E1) {
        sabs += std::abs(v);
        ssq += v * v;
    }
    const std::size_t nf = g.field_size();
    std::printf("[XCHK] N=12 r_F=%.17g r_out=%.17g sum|E|=%.17g sumE2=%.17g\n", n1.r_F, n1.r_out,
                sabs, ssq);
    std::printf("[XCHK] E[f0,j1,0,0]=%.17g E[f1,j5,3,7]=%.17g E[f0,j11,2,9]=%.17g "
                "E[f1,j12,4,1]=%.17g E[f0,j12,0,5]=%.17g\n",
                E1[g.unknown_index(1, 0, 0)], E1[nf + g.unknown_index(5, 3, 7)],
                E1[g.unknown_index(11, 2, 9)], E1[nf + g.unknown_index(12, 4, 1)],
                E1[g.unknown_index(12, 0, 5)]);
    std::printf("[XCHK] e_v=%.17g e_i1=%.17g e_i2=%.17g e_div=%.17g min_c=%.17g\n", m1.e_v, m1.e_i1,
                m1.e_i2, m1.e_div, m1.min_c);
    std::printf(
        "[XCHK] p=%.17g %.17g %.17g %.17g vD_p=%.17g %.17g %.17g %.17g vD_min=%.17g v_rms=%.17g\n",
        m1.p0_1, m1.p1, m1.p5, m1.p50, m1.vD_p[0], m1.vD_p[1], m1.vD_p[2], m1.vD_p[3], m1.vD_min,
        m1.v_rms);
    std::printf("[XCHK] e_psi1=%.17g e_psi2=%.17g a_psi=%.17g %.17g den=%.17g %.17g\n", m1.e_psi1,
                m1.e_psi2, m1.a_psi1, m1.a_psi2, m1.den_used1, m1.den_used2);
    sl::CaseLineExtras ex;
    ex.r_F = n1.r_F;
    ex.its = 3;
    ex.t = 1.25;
    std::printf("[XCHK] %s\n", sl::format_case_line("generic", 0.8125, 12, "i1o4", m1, ex).c_str());

    // Gate against the SF-29 prototype on the same state (values of candidate_i.system /
    // metrics.fd_metrics(order=4) computed offline with %.17g on this exact analytic fixture;
    // relative tolerance 1e-11 covers FMA contraction and summation-order differences, observed
    // ~1e-14).
    const double proto_vals[] = {
        3.0390487058986997,   0.26748494064954487, 9065.2345225628596,  40156.516240495941,
        0.25024921941687245,  -2.0651546814683064, 8.8947626075119626,  1.2474823049799435,
        5.472625220916064,    0.44102716697391375, 0.2149279578298825,  0.067909388762718431,
        0.017729451420864017, 0.31066395127126162, 0.33694973303008291, 0.40674251912946341,
        0.53360232125371343,  0.99470316181038165, 0.11111111111111108, 0.22154914387635175};
    const double ours[] = {n1.r_F,
                           n1.r_out,
                           sabs,
                           ssq,
                           E1[g.unknown_index(1, 0, 0)],
                           E1[nf + g.unknown_index(5, 3, 7)],
                           E1[g.unknown_index(11, 2, 9)],
                           E1[nf + g.unknown_index(12, 4, 1)],
                           E1[g.unknown_index(12, 0, 5)],
                           m1.e_v,
                           m1.e_i1,
                           m1.e_i2,
                           m1.e_div,
                           m1.min_c,
                           m1.p0_1,
                           m1.p1,
                           m1.p5,
                           m1.p50,
                           m1.e_psi1,
                           m1.e_psi2};
    double pmax = 0.0;
    for (std::size_t i = 0; i < sizeof(ours) / sizeof(ours[0]); ++i)
        pmax = std::max(pmax, std::abs(ours[i] - proto_vals[i]) / std::abs(proto_vals[i]));
    rep.check(pmax <= 1e-11,
              "contracts: generic state N = 12 reproduces the SF-29 prototype (r_F, r_out, E "
              "samples, sums, metrics)",
              fmt("max rel diff = %.2e", pmax));
    const std::string proto_line = "CASE field=generic eps=0.8125 N=12 cand=i1o4 | r_F=3.039e+00 "
                                   "its=3 | e_v=4.410e-01 e_psi=2.215e-01 "
                                   "e_i=(2.149e-01,6.791e-02) e_div=1.773e-02 min_c=3.107e-01 "
                                   "p0.1=3.369e-01 p1=4.067e-01 p5=5.336e-01 "
                                   "p50=9.947e-01 | t=1.2 | e_psi1=1.111e-01 e_psi2=2.215e-01";
    rep.check(sl::format_case_line("generic", 0.8125, 12, "i1o4", m1, ex) == proto_line,
              "contracts: CASE line of the generic state equals the prototype's metrics.case_line "
              "byte for byte");

    // unprepared workspace is rejected (no hidden allocation)
    sl::SlabResidualWorkspace fresh;
    bool threw = false;
    try {
        sl::evaluate_residual(ctx, g, fx.in, cspan(fx.U1), cspan(fx.U2), mspan(fx.E), fresh,
                              nullptr);
    } catch (const std::logic_error&) {
        threw = true;
    }
    rep.check(threw,
              "contracts: evaluate_residual with an unprepared workspace throws (never allocates)");
}

} // namespace

int main(int argc, char** argv) {
    TestReport rep;
    bool with64 = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--with-64")
            with64 = true;
    }
    CudaContext ctx(0);
    try {
        std::printf("=== SF-33 N0: grid ===\n");
        case_grid(rep);
        std::printf("=== SF-33 N0: (1) Fornberg weights and stencil exactness ===\n");
        case_fornberg(rep, ctx);
        std::printf("=== SF-33 N0: (2) derivs4 observed orders, 16 -> 32%s ===\n",
                    with64 ? " -> 64" : "");
        case_orders(rep, with64);
        std::printf("=== SF-33 N0: (3) exact discrete pair, N = 12, 16 ===\n");
        case_exact_pair(rep, ctx);
        std::printf("=== SF-33 N0: (4) k = 1 affine control ===\n");
        case_k1(rep, ctx);
        std::printf("=== SF-33 N0: (5) metrics ===\n");
        case_metrics(rep, ctx);
        std::printf("=== SF-33 N0: (6) CASE line ===\n");
        case_case_line(rep);
        std::printf("=== SF-33 N0: (7) workspace / determinism / host-device contracts ===\n");
        case_contracts(rep, ctx);
    } catch (const std::exception& e) {
        std::printf("[FAIL] unexpected exception: %s\n", e.what());
        rep.overall_pass = false;
    }
    std::printf("\n=== inlet_slab_contracts: %d checks, %s ===\n", rep.checks,
                rep.overall_pass ? "PASS" : "FAIL");
    return rep.overall_pass ? 0 : 1;
}
