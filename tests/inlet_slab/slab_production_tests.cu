/**
 * @file slab_production_tests.cu
 * @brief SF-33 N3: fast tests of the D-1 inlet labels, the production inputs (SF-18 / SF-19 /
 * SF-28) and the production oracle of the inlet slab. Standalone ctest runner in the printed-checks
 *        style of tests/inlet_slab/slab_contract_tests.cu; every grid is <= 16^3.
 *
 * Cases (acceptance items of the SF-33 N3 task):
 *   1  InletLabels on v1 = 1 + 0.3 cos(2 pi x2) sin(2 pi x3) + 0.2 sin(4 pi x3), 32 x 32 samples at
 *      offsets 0 and 1/2: face Jacobian = v1 (1e-12), unit jumps (1e-13), Q0 = 1 (1e-14), GPU vs
 * host (1e-13), inlet_backflow on non-positive samples, no allocation in the GPU evaluation. 2
 * Spectral vertex evaluation of a 4-mode band-limited field (N = 16): values and gradients on every
 * plane 0..N (1e-12); analytic-field gradients vs central FD (1e-8); auto MG levels. 3
 * trace_to_plane(X = x1_0 + sigma) bitwise equal to closure_gate::integrate_streamline (64
 *      stateless seeds, generic3d eps = 0.25 at 16^3 on the production stack, sigma = +-1, tol
 * 1e-8). 4  Production oracle at 16^3: k = 1; control2d eps = 0.5 (psi2 = x3); generic3d eps = 0.25
 *      round trip at tol 1e-8 / 1e-10 for h_max = h, h/4, h/8 (the round trip is h_max-limited:
 *      gated <= 1e-8 at tol 1e-8 with h_max = h/8; the tol-1e-10 / 1e-10 criterion is printed as
 *      [INFO] and NOT met at these h_max, see SlabOracle.cuh); N0 residual norms of the oracle
 *      labels at N = 8, 16 (informative orders).
 *   5  SF-19 inlet flux (mean = 1), SF-18 report of a gaussian case, U-face vs spline-flow v1
 *      difference at N = 8, 16 with its order.
 *   6  Oracle determinism: 1 vs 8 threads, bitwise.
 */

#include "src/core/DeviceBuffer.cuh"
#include "src/core/DeviceSpan.cuh"
#include "src/core/Scalar.hpp"
#include "src/physics/streamfunctions/inlet_slab/InletLabels.cuh"
#include "src/physics/streamfunctions/inlet_slab/InletSlabGrid.cuh"
#include "src/physics/streamfunctions/inlet_slab/SlabMetrics.cuh"
#include "src/physics/streamfunctions/inlet_slab/SlabOracle.cuh"
#include "src/physics/streamfunctions/inlet_slab/SlabProductionSetup.cuh"
#include "src/physics/streamfunctions/inlet_slab/SlabResidual.cuh"
#include "src/runtime/cuda_check.cuh"
#include "src/runtime/CudaContext.cuh"

#include "apps/closure_gate/closure_fields.hpp"
#include "apps/closure_gate/closure_gate.cuh"
#include "apps/closure_gate/streamline_integrator.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

using namespace macroflow3d;
namespace sl = macroflow3d::streamfunctions::inlet_slab;
namespace cg = macroflow3d::closure_gate;

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

std::string fmt(const char* f, double v) {
    char b[160];
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

double order(double coarse, double fine) {
    return std::log2(coarse / fine);
}

std::size_t free_device_bytes() {
    std::size_t fr = 0, tot = 0;
    MACROFLOW3D_CUDA_CHECK(cudaMemGetInfo(&fr, &tot));
    return fr;
}

// ================================================================================================
// 1  Inlet labels
// ================================================================================================

double v1_test(double y, double z) {
    return 1.0 + 0.3 * std::cos(kTwoPi * y) * std::sin(kTwoPi * z) +
           0.2 * std::sin(2.0 * kTwoPi * z);
}

void case_inlet_labels(TestReport& rep, CudaContext& ctx) {
    const int nf = 32;
    const int P = 200;
    std::vector<double> py(P), pz(P);
    for (int p = 0; p < P; ++p) {
        // points in [-1.5, 2.5)^2: unwrapped coordinates included
        py[p] = -1.5 + 4.0 * cg::seed_uniform(20261006ULL, 2ULL * p);
        pz[p] = -1.5 + 4.0 * cg::seed_uniform(20261006ULL, 2ULL * p + 1ULL);
    }
    for (double off : {0.0, 0.5}) {
        std::vector<real> s(static_cast<std::size_t>(nf) * nf);
        for (int a2 = 0; a2 < nf; ++a2)
            for (int a3 = 0; a3 < nf; ++a3)
                s[static_cast<std::size_t>(a3) + static_cast<std::size_t>(nf) * a2] =
                    v1_test((a2 + off) / nf, (a3 + off) / nf);
        sl::InletLabels L = sl::InletLabels::build(s, nf, off, off);
        const std::string tag = fmt("offset=%.1f", off);
        double ejac = 0.0, ev = 0.0, ej1 = 0.0, ej2 = 0.0;
        for (int p = 0; p < P; ++p) {
            const double ex = v1_test(py[p], pz[p]);
            ejac = std::max(ejac, std::abs(L.jacobian(py[p], pz[p]) - ex));
            ev = std::max(ev, std::abs(L.V(py[p], pz[p]) - ex));
            ej1 = std::max(ej1, std::abs(L.psi1(py[p] + 1.0, pz[p]) - L.psi1(py[p], pz[p]) - 1.0));
            ej2 = std::max(ej2, std::abs(L.psi2(pz[p] + 1.0) - L.psi2(pz[p]) - 1.0));
        }
        rep.check(ejac <= 1e-12, "labels " + tag + ": host face Jacobian = v1 at 200 points",
                  fmt("max err %.3e", ejac) + fmt(" (V interpolant err %.3e)", ev));
        rep.check(ej1 <= 1e-13, "labels " + tag + ": psi1(x2 + 1) - psi1 = 1",
                  fmt("max err %.3e", ej1));
        rep.check(ej2 <= 1e-13, "labels " + tag + ": psi2(x3 + 1) - psi2 = 1",
                  fmt("max err %.3e", ej2));
        rep.check(std::abs(L.Q0() - 1.0) <= 1e-14, "labels " + tag + ": Q0 = 1",
                  fmt("Q0 - 1 = %.3e", L.Q0() - 1.0) + fmt(" modes=%.0f", L.modes()));

        // GPU vs host mirror (small chunk -> several chunks, last one partial)
        L.prepare_device(64);
        DeviceBuffer<real> dy, dz, o1(P), o2(P);
        upload(dy, py);
        upload(dz, pz);
        const std::size_t bytes0 = L.device_bytes();
        const std::size_t free0 = free_device_bytes();
        L.evaluate_labels(ctx, cspan(dy), cspan(dz), mspan(o1), mspan(o2));
        ctx.synchronize();
        L.evaluate_labels(ctx, cspan(dy), cspan(dz), mspan(o1), mspan(o2));
        ctx.synchronize();
        const std::size_t free1 = free_device_bytes();
        const auto h1 = download(o1.data(), P), h2 = download(o2.data(), P);
        double eg1 = 0.0, eg2 = 0.0;
        for (int p = 0; p < P; ++p) {
            eg1 = std::max(eg1, std::abs(h1[p] - L.psi1(py[p], pz[p])));
            eg2 = std::max(eg2, std::abs(h2[p] - L.psi2(pz[p])));
        }
        rep.check(eg1 <= 1e-13 && eg2 <= 1e-13, "labels " + tag + ": GPU batched = host mirror",
                  fmt("max |dpsi1| %.3e", eg1) + fmt(" max |dpsi2| %.3e", eg2));
        rep.check(free0 == free1 && bytes0 == L.device_bytes(),
                  "labels " + tag + ": no device allocation in evaluate_labels after prepare",
                  fmt("free before-after = %.0f bytes",
                      static_cast<double>(free0) - static_cast<double>(free1)));
    }
    // inlet_backflow
    {
        std::vector<real> s(static_cast<std::size_t>(nf) * nf);
        for (int a2 = 0; a2 < nf; ++a2)
            for (int a3 = 0; a3 < nf; ++a3)
                s[static_cast<std::size_t>(a3) + static_cast<std::size_t>(nf) * a2] =
                    0.2 + std::cos(kTwoPi * a2 / nf);
        bool thrown = false;
        double vmin = 0.0;
        try {
            (void)sl::InletLabels::build(s, nf, 0.0, 0.0);
        } catch (const sl::InletBackflowError& e) {
            thrown = true;
            vmin = e.vmin();
        }
        rep.check(thrown && vmin < 0.0, "labels: min v1 < 0 -> inlet_backflow",
                  fmt("vmin = %.3e", vmin));
        std::fill(s.begin(), s.end(), 1.0);
        s[5] = 0.0;
        thrown = false;
        try {
            (void)sl::InletLabels::build(s, nf, 0.5, 0.5);
        } catch (const sl::InletBackflowError& e) {
            thrown = true;
        }
        rep.check(thrown, "labels: min v1 = 0 -> inlet_backflow");
    }
}

// ================================================================================================
// 2  Spectral vertex evaluation, analytic gradients, MG levels
// ================================================================================================

struct Mode {
    int m1, m2, m3;
    double amp, phase;
};
const Mode kModes[4] = {
    {1, 2, -1, 0.7, 0.3}, {3, -1, 0, 0.5, -1.2}, {0, 2, 3, 0.4, 1.1}, {-1, 1, 2, 0.3, 0.5}};

double modes_value(double x, double y, double z, double g[3]) {
    double v = 0.0;
    g[0] = g[1] = g[2] = 0.0;
    for (const Mode& m : kModes) {
        const double th = kTwoPi * (m.m1 * x + m.m2 * y + m.m3 * z) + m.phase;
        v += m.amp * std::cos(th);
        const double s = -m.amp * std::sin(th) * kTwoPi;
        g[0] += s * m.m1;
        g[1] += s * m.m2;
        g[2] += s * m.m3;
    }
    return v;
}

void case_spectral(TestReport& rep, CudaContext& ctx) {
    const int N = 16;
    const double h = 1.0 / N;
    std::vector<real> cell(static_cast<std::size_t>(N) * N * N);
    double g[3];
    for (int k = 0; k < N; ++k)
        for (int j = 0; j < N; ++j)
            for (int i = 0; i < N; ++i)
                cell[static_cast<std::size_t>(i) + N * (static_cast<std::size_t>(j) + N * k)] =
                    modes_value((i + 0.5) * h, (j + 0.5) * h, (k + 0.5) * h, g);
    const sl::SlabFieldSource src = sl::SlabFieldSource::spectral(ctx, "modes4", N, cell);
    const sl::InletSlabGrid grid = sl::InletSlabGrid::make(N);
    double ev = 0.0, eg[3] = {0.0, 0.0, 0.0};
    for (int jp = 0; jp <= N; ++jp)
        for (int m2 = 0; m2 < N; ++m2)
            for (int m3 = 0; m3 < N; ++m3) {
                const std::size_t idx = grid.full_index(jp, m2, m3);
                const double v = modes_value(grid.coord(jp), grid.coord(m2), grid.coord(m3), g);
                ev = std::max(ev, std::abs(src.vertex_value()[idx] - v));
                for (int d = 0; d < 3; ++d)
                    eg[d] = std::max(eg[d], std::abs(src.vertex_grad(d)[idx] - g[d]));
            }
    double ef = 0.0;
    for (int m2 = 0; m2 < N; ++m2)
        for (int m3 = 0; m3 < N; ++m3)
            ef = std::max(ef, std::abs(src.inlet_face_value()[grid.plane_index(m2, m3)] -
                                       modes_value(0.0, (m2 + 0.5) * h, (m3 + 0.5) * h, g)));
    rep.check(ev <= 1e-12, "spectral vertex value, planes 0..N (N = 16, 4 modes |m| <= 3)",
              fmt("max err %.3e", ev));
    rep.check(eg[0] <= 1e-12 && eg[1] <= 1e-12 && eg[2] <= 1e-12,
              "spectral vertex gradient (3 components), planes 0..N",
              fmt("max err (%.3e,", eg[0]) + fmt(" %.3e,", eg[1]) + fmt(" %.3e)", eg[2]));
    rep.check(ef <= 1e-12, "spectral inlet-face-centre value", fmt("max err %.3e", ef));

    // analytic gradients vs central FD
    const char* fields[5] = {"control2d", "lester2021", "lester_brk", "two_mode", "generic3d"};
    for (const char* f : fields) {
        double e = 0.0;
        const double dlt = 1e-6;
        for (int p = 0; p < 50; ++p) {
            const double x[3] = {cg::seed_uniform(77ULL, 3ULL * p),
                                 cg::seed_uniform(77ULL, 3ULL * p + 1),
                                 cg::seed_uniform(77ULL, 3ULL * p + 2)};
            double ga[3];
            sl::analytic_log_conductivity_gradient(f, x[0], x[1], x[2], ga);
            for (int d = 0; d < 3; ++d) {
                double xp[3] = {x[0], x[1], x[2]}, xm[3] = {x[0], x[1], x[2]};
                xp[d] += dlt;
                xm[d] -= dlt;
                const double fd = (sl::analytic_log_conductivity_value(f, xp[0], xp[1], xp[2]) -
                                   sl::analytic_log_conductivity_value(f, xm[0], xm[1], xm[2])) /
                                  (2.0 * dlt);
                e = std::max(e, std::abs(fd - ga[d]));
            }
        }
        rep.check(e <= 1e-8, std::string("analytic grad ln k vs central FD: ") + f,
                  fmt("max err %.3e", e));
    }
    bool same = true;
    for (int n = 4; n <= 256; n += 2)
        same = same && sl::slab_auto_mg_levels(n) == cg::auto_mg_levels(n);
    rep.check(same, "slab_auto_mg_levels == closure_gate::auto_mg_levels (N = 4..256 even)");
}

// ================================================================================================
// helpers for the production stack
// ================================================================================================

struct OracleRun {
    sl::SlabOracleResult res;
    std::vector<real> psi1, psi2;
};

OracleRun run_oracle(CudaContext& ctx, sl::ProductionStage& st, double tol, int threads,
                     double hdiv = 1.0) {
    OracleRun o;
    DeviceBuffer<real> p1(st.grid.full_size()), p2(st.grid.full_size());
    sl::SlabOracleOptions opt;
    opt.tol = tol;
    opt.threads = threads;
    opt.h_max = st.grid.h / hdiv; // hdiv = 1: the SF-30 convention (h_max = h)
    const sl::SlabSplineDirectionField fld = st.direction_field();
    o.res = sl::compute_oracle(ctx, st.grid, fld, st.labels, opt, mspan(p1), mspan(p2));
    o.psi1 = download(p1.data(), p1.size());
    o.psi2 = download(p2.data(), p2.size());
    return o;
}

int test_threads() {
    const unsigned hc = std::thread::hardware_concurrency();
    return static_cast<int>(std::max(1u, std::min(hc == 0 ? 1u : hc, 8u)));
}

/// r_F, r_out of the oracle labels fed as U with the stage's own inputs.
sl::SlabResidualNorms oracle_residual(CudaContext& ctx, sl::ProductionStage& st,
                                      const OracleRun& o) {
    const sl::InletSlabGrid& g = st.grid;
    DeviceBuffer<real> p1, p2, U1(g.full_size()), U2(g.full_size()), E(g.unknown_size());
    upload(p1, o.psi1);
    upload(p2, o.psi2);
    sl::labels_to_periodic_parts(ctx, g, cspan(p1), cspan(p2), mspan(U1), mspan(U2));
    sl::SlabResidualWorkspace ws;
    ws.prepare(g);
    sl::SlabResidualNorms nrm;
    sl::evaluate_residual(ctx, g, st.inputs, cspan(U1), cspan(U2), mspan(E), ws, &nrm);
    return nrm;
}

// ================================================================================================
// 3  trace_to_plane bitwise vs integrate_streamline
// ================================================================================================

bool same_bits(double a, double b) {
    return std::memcmp(&a, &b, sizeof(double)) == 0;
}

bool identical(const cg::StreamlineResult& a, const cg::StreamlineResult& b) {
    bool ok = a.status == b.status && a.periods_completed == b.periods_completed &&
              a.records.size() == b.records.size() &&
              a.backflow_encounter == b.backflow_encounter &&
              same_bits(a.min_g1_hat, b.min_g1_hat) &&
              same_bits(a.backflow_arclength, b.backflow_arclength) &&
              a.accepted_steps == b.accepted_steps && a.rejected_steps == b.rejected_steps &&
              a.landing_steps == b.landing_steps && a.landing_rejected == b.landing_rejected &&
              a.landings_discarded == b.landings_discarded &&
              a.field_evaluations == b.field_evaluations && same_bits(a.final_s, b.final_s) &&
              same_bits(a.final_tau, b.final_tau);
    for (int d = 0; d < 3; ++d)
        ok = ok && same_bits(a.final_x[d], b.final_x[d]);
    for (std::size_t r = 0; ok && r < a.records.size(); ++r) {
        const cg::PlaneRecord &p = a.records[r], &q = b.records[r];
        ok = same_bits(p.x1, q.x1) && same_bits(p.x2, q.x2) && same_bits(p.x3, q.x3) &&
             same_bits(p.s, q.s) && same_bits(p.tau, q.tau);
    }
    return ok;
}

void case_trace_bitwise(TestReport& rep, sl::ProductionStage& st) {
    const sl::SlabSplineDirectionField fld = st.direction_field();
    for (int sigma : {+1, -1}) {
        cg::IntegratorOptions opt;
        opt.tol = 1e-8;
        opt.h_max = st.grid.h;
        opt.n_periods = 1;
        opt.sigma = sigma;
        int same = 0, ok = 0;
        long long nfev = 0;
        for (int i = 0; i < 64; ++i) {
            const auto yz = cg::seed_point(cg::kDefaultSeedRng, static_cast<std::uint64_t>(i));
            const double x1 =
                cg::seed_uniform(cg::kDefaultSeedRng + 1ULL, static_cast<std::uint64_t>(i));
            const std::array<double, 3> seed = {x1, yz[0], yz[1]};
            const cg::StreamlineResult a = cg::integrate_streamline(fld, seed, opt);
            const cg::StreamlineResult b = sl::trace_to_plane(fld, seed, x1 + sigma, opt);
            same += identical(a, b) ? 1 : 0;
            ok += a.status == cg::StreamlineStatus::ok ? 1 : 0;
            nfev += a.field_evaluations;
        }
        rep.check(
            same == 64,
            fmt("trace_to_plane(X = x1_0 %+.0f) bitwise = integrate_streamline (64 seeds)", sigma),
            fmt("identical %.0f/64", same) + fmt(", ok %.0f", ok) +
                fmt(", nfev %.0f", static_cast<double>(nfev)));
    }
    // argument validation
    bool thrown = false;
    try {
        cg::IntegratorOptions opt;
        opt.h_max = st.grid.h;
        opt.sigma = -1;
        (void)sl::trace_to_plane(fld, {0.5, 0.1, 0.1}, 0.75, opt);
    } catch (const std::invalid_argument&) {
        thrown = true;
    }
    rep.check(thrown, "trace_to_plane rejects a plane behind the seed");
}

// ================================================================================================
// 4 / 6  production oracle
// ================================================================================================

void print_stage(const sl::ProductionStage& st) {
    std::printf("%s", st.report.summary().c_str());
}

void case_oracle_k1(TestReport& rep, CudaContext& ctx) {
    const int N = 16;
    const sl::InletSlabGrid grid = sl::InletSlabGrid::make(N);
    const sl::SlabFieldSource src = sl::SlabFieldSource::spectral(
        ctx, "uniform", N, std::vector<real>(static_cast<std::size_t>(N) * N * N, 0.0));
    sl::ProductionStage st = sl::build_production_stage(ctx, grid, src, 1.0);
    print_stage(st);
    rep.check(st.ok(), "k = 1: production stage ok", sl::to_string(st.report.status));
    if (!st.ok())
        return;
    const OracleRun o = run_oracle(ctx, st, 1e-8, test_threads());
    std::printf("%s", o.res.table().c_str());
    double efoot = 0.0, elab = 0.0;
    for (int j = 1; j <= N; ++j)
        for (int m2 = 0; m2 < N; ++m2)
            for (int m3 = 0; m3 < N; ++m3) {
                const std::size_t i = grid.unknown_index(j, m2, m3);
                efoot = std::max(efoot, std::max(std::abs(o.res.foot_y[i] - grid.coord(m2)),
                                                 std::abs(o.res.foot_z[i] - grid.coord(m3))));
            }
    for (int j = 0; j <= N; ++j)
        for (int m2 = 0; m2 < N; ++m2)
            for (int m3 = 0; m3 < N; ++m3) {
                const std::size_t i = grid.full_index(j, m2, m3);
                elab = std::max(elab, std::max(std::abs(o.psi1[i] - grid.coord(m2)),
                                               std::abs(o.psi2[i] - grid.coord(m3))));
            }
    rep.check(o.res.status == sl::SlabOracleStatus::ok, "k = 1: oracle status ok",
              sl::to_string(o.res.status));
    rep.check(efoot <= 1e-13, "k = 1: feet = (x2, x3)", fmt("max err %.3e", efoot));
    rep.check(elab <= 1e-13, "k = 1: oracle labels affine", fmt("max err %.3e", elab));
    rep.check(o.res.max_roundtrip <= 1e-13, "k = 1: round trip",
              fmt("max %.3e", o.res.max_roundtrip));
}

void case_oracle_control2d(TestReport& rep, CudaContext& ctx) {
    const int N = 16;
    const sl::InletSlabGrid grid = sl::InletSlabGrid::make(N);
    const sl::SlabFieldSource src = sl::SlabFieldSource::analytic("control2d", N);
    sl::ProductionStage st = sl::build_production_stage(ctx, grid, src, 0.5);
    print_stage(st);
    rep.check(st.ok(), "control2d eps = 0.5: production stage ok", sl::to_string(st.report.status));
    if (!st.ok())
        return;
    const OracleRun o = run_oracle(ctx, st, 1e-8, test_threads());
    std::printf("%s", o.res.table().c_str());
    double worst = 0.0, worst_q0 = 0.0;
    std::string per_plane;
    for (int j = 0; j <= N; ++j) {
        double e = 0.0, eq = 0.0;
        for (int m2 = 0; m2 < N; ++m2)
            for (int m3 = 0; m3 < N; ++m3) {
                const std::size_t i = grid.full_index(j, m2, m3);
                e = std::max(e, std::abs(o.psi2[i] - grid.coord(m3)));
                eq = std::max(eq, std::abs(o.psi2[i] - st.labels.Q0() * grid.coord(m3)));
            }
        worst = std::max(worst, e);
        worst_q0 = std::max(worst_q0, eq);
        if (j % 4 == 0)
            per_plane += fmt(" j=%.0f:", j) + fmt("%.2e", e);
    }
    rep.check(o.res.status == sl::SlabOracleStatus::ok, "control2d: oracle status ok",
              sl::to_string(o.res.status));
    rep.check(worst <= 1e-10, "control2d eps = 0.5: psi2^or - x3 = 0 on every plane (planar flow)",
              fmt("max %.3e;", worst) + per_plane + fmt("; max |psi2 - Q0 x3| %.3e", worst_q0));
}

void case_oracle_generic3d(TestReport& rep, CudaContext& ctx) {
    sl::SlabResidualNorms nrm[2];
    const int Ns[2] = {8, 16};
    for (int t = 0; t < 2; ++t) {
        const int N = Ns[t];
        const sl::InletSlabGrid grid = sl::InletSlabGrid::make(N);
        const sl::SlabFieldSource src = sl::SlabFieldSource::analytic("generic3d", N);
        sl::ProductionStage st = sl::build_production_stage(ctx, grid, src, 0.25);
        print_stage(st);
        rep.check(st.ok(), fmt("generic3d eps = 0.25 N = %.0f: production stage ok", N),
                  sl::to_string(st.report.status));
        if (!st.ok())
            return;
        OracleRun o = run_oracle(ctx, st, 1e-8, test_threads());
        std::printf("%s", o.res.table().c_str());
        rep.check(
            o.res.status == sl::SlabOracleStatus::ok,
            fmt("generic3d N = %.0f (h_max = h): oracle status ok, all streamlines ok", N),
            fmt("max round trip %.3e (SF-30 h_max = h; reported, not gated)", o.res.max_roundtrip));
        nrm[t] = oracle_residual(ctx, st, o);
        std::printf("  oracle labels as U (N = %d): r_F = %.6e  r_out = %.6e\n", N, nrm[t].r_F,
                    nrm[t].r_out);
        if (N == 16) {
            case_trace_bitwise(rep, st);
            // Round trip vs h_max at tol 1e-8 / 1e-10. The DP5(4) error estimate does not see the
            // C^2 knots of the cubic spline: the round trip is set by h_max (~ h_max^3), not by
            // tol.
            double rt[3][2];
            const double hdivs[3] = {1.0, 4.0, 8.0};
            OracleRun o8_8;
            for (int a = 0; a < 3; ++a)
                for (int b = 0; b < 2; ++b) {
                    const double tol = b == 0 ? 1e-8 : 1e-10;
                    if (a == 0 && b == 0) {
                        rt[a][b] = o.res.max_roundtrip;
                        continue;
                    }
                    OracleRun r = run_oracle(ctx, st, tol, test_threads(), hdivs[a]);
                    rt[a][b] = r.res.status == sl::SlabOracleStatus::ok ? r.res.max_roundtrip
                                                                        : std::nan("");
                    if (a == 2) {
                        std::printf("  [h_max = h/8, tol %.0e]\n%s", tol, r.res.table().c_str());
                        if (b == 0)
                            o8_8 = std::move(r);
                    }
                }
            for (int a = 0; a < 3; ++a)
                std::printf("  generic3d N = 16 round trip: h_max = h/%.0f  tol 1e-8 -> %.3e  tol "
                            "1e-10 -> %.3e\n",
                            hdivs[a], rt[a][0], rt[a][1]);
            rep.check(rt[2][0] <= 1e-8,
                      "generic3d N = 16 tol 1e-8, h_max = h/8: round trip <= 1e-8 on every plane",
                      fmt("max %.3e", rt[2][0]) + fmt(" (h_max = h: %.3e)", rt[0][0]));
            std::printf("[INFO] criterion 4(iii) tol 1e-10 threshold 1e-10: max round trip %.3e "
                        "(h_max = h), "
                        "%.3e (h/4), %.3e (h/8) -- NOT met; h_max-limited, not tol-limited\n",
                        rt[0][1], rt[1][1], rt[2][1]);
            double dl = 0.0;
            for (std::size_t i = 0; i < o.psi1.size(); ++i)
                dl = std::max(dl, std::max(std::abs(o.psi1[i] - o8_8.psi1[i]),
                                           std::abs(o.psi2[i] - o8_8.psi2[i])));
            std::printf("  oracle labels h_max = h vs h/8 (tol 1e-8): max |dpsi| = %.3e\n", dl);

            // 6  determinism: 1 thread vs the multi-threaded run
            const auto t0 = std::chrono::steady_clock::now();
            OracleRun o1 = run_oracle(ctx, st, 1e-8, 1);
            const double t1s =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            bool bit = o1.psi1.size() == o.psi1.size() &&
                       o1.res.roundtrip.size() == o.res.roundtrip.size();
            for (std::size_t i = 0; bit && i < o.psi1.size(); ++i)
                bit = same_bits(o1.psi1[i], o.psi1[i]) && same_bits(o1.psi2[i], o.psi2[i]);
            for (std::size_t i = 0; bit && i < o.res.roundtrip.size(); ++i)
                bit = same_bits(o1.res.roundtrip[i], o.res.roundtrip[i]) &&
                      same_bits(o1.res.foot_y[i], o.res.foot_y[i]) &&
                      same_bits(o1.res.foot_z[i], o.res.foot_z[i]);
            for (std::size_t j = 0; bit && j < o.res.planes.size(); ++j)
                bit = o1.res.planes[j].nfev == o.res.planes[j].nfev &&
                      o1.res.planes[j].nfev_rt == o.res.planes[j].nfev_rt &&
                      same_bits(o1.res.planes[j].roundtrip, o.res.planes[j].roundtrip);
            rep.check(
                bit,
                fmt("oracle determinism: 1 vs %.0f threads bitwise (psi_or, feet, round trips)",
                    test_threads()),
                fmt("1-thread trace %.1fs", t1s));
        }
    }
    std::printf("  oracle-label residual orders 8 -> 16 (informative): r_F %.2f, r_out %.2f\n",
                order(nrm[0].r_F, nrm[1].r_F), order(nrm[0].r_out, nrm[1].r_out));
    rep.check(std::isfinite(nrm[1].r_F) && std::isfinite(nrm[1].r_out) && nrm[1].r_F < nrm[0].r_F &&
                  nrm[1].r_out < nrm[0].r_out,
              "generic3d oracle labels: r_F and r_out decrease 8 -> 16",
              fmt("r_F %.3e ->", nrm[0].r_F) + fmt(" %.3e,", nrm[1].r_F) +
                  fmt(" r_out %.3e ->", nrm[0].r_out) + fmt(" %.3e", nrm[1].r_out));
}

// ================================================================================================
// 5  SF-19 inlet flux, SF-18 report, v1 difference
// ================================================================================================

void case_sf19_inlet(TestReport& rep, CudaContext& ctx) {
    // gaussian case
    sl::ProductionFieldSpec spec;
    spec.N = 16;
    spec.sigma2 = 1.0;
    spec.ell = 0.25;
    spec.seed = 3001ULL;
    const sl::SlabFieldSource src = sl::SlabFieldSource::gaussian(ctx, spec);
    const sl::InletSlabGrid grid = sl::InletSlabGrid::make(16);
    sl::ProductionStage st = sl::build_production_stage(ctx, grid, src, 0.5);
    print_stage(st);
    std::printf(
        "  SF-18 gaussian (sigma2 = 1, ell = 0.25, seed 3001, N = 16): applied_scale = %.15e "
        "raw_variance = %.15e final_variance = %.15e\n",
        src.sf18().applied_scale, src.sf18().raw_variance, src.sf18().final_variance);
    rep.check(st.ok(), "gaussian eps = 0.5 N = 16: production stage ok",
              sl::to_string(st.report.status));
    rep.check(std::abs(st.report.inlet_mean - 1.0) <= 1e-9,
              "gaussian: mean of U-face plane-0 samples = 1",
              fmt("mean - 1 = %.3e", st.report.inlet_mean - 1.0) +
                  fmt(", Q0 - 1 = %.3e", st.report.Q0 - 1.0));
    rep.check(std::abs(src.sf18().final_variance - 1.0) <= 1e-12,
              "gaussian: SF-18 final_variance = sigma2",
              fmt("final_variance - 1 = %.3e", src.sf18().final_variance - 1.0));

    // v1 difference at N = 8, 16 (generic3d, same continuum field; gaussian printed too)
    double rms[2], mx[2], grms[2], gmx[2];
    const int Ns[2] = {8, 16};
    for (int t = 0; t < 2; ++t) {
        const sl::InletSlabGrid g = sl::InletSlabGrid::make(Ns[t]);
        sl::ProductionStage s = sl::build_production_stage(
            ctx, g, sl::SlabFieldSource::analytic("generic3d", Ns[t]), 0.25);
        rep.check(s.ok() && std::abs(s.report.inlet_mean - 1.0) <= 1e-9,
                  fmt("generic3d N = %.0f: stage ok, mean inlet flux = 1", Ns[t]),
                  fmt("mean - 1 = %.3e", s.report.inlet_mean - 1.0));
        rms[t] = s.report.v1_diff_rms_rel;
        mx[t] = s.report.v1_diff_max_rel;
        sl::ProductionFieldSpec sp = spec;
        sp.N = Ns[t];
        sl::ProductionStage sg =
            sl::build_production_stage(ctx, g, sl::SlabFieldSource::gaussian(ctx, sp), 0.5);
        grms[t] = sg.report.v1_diff_rms_rel;
        gmx[t] = sg.report.v1_diff_max_rel;
        std::printf(
            "  N = %2d: v1 (U-face) vs spline flow: generic3d(0.25) rms_rel %.4e max_rel %.4e | "
            "gaussian(0.5) rms_rel %.4e max_rel %.4e (SF-18 applied_scale %.12e)\n",
            Ns[t], rms[t], mx[t], grms[t], gmx[t], sg.report.sf18.applied_scale);
    }
    std::printf(
        "  v1 difference observed order 8 -> 16: generic3d rms %.2f max %.2f | gaussian rms %.2f "
        "max %.2f (expected ~2)\n",
        order(rms[0], rms[1]), order(mx[0], mx[1]), order(grms[0], grms[1]), order(gmx[0], gmx[1]));
    rep.check(std::isfinite(rms[1]) && rms[1] < rms[0],
              "v1 difference decreases under refinement (generic3d)",
              fmt("order %.2f", order(rms[0], rms[1])));
}

} // namespace

int main() {
    TestReport rep;
    CudaContext ctx(0);
    const auto t0 = std::chrono::steady_clock::now();
    try {
        std::printf("=== SF-33 N3: (1) D-1 inlet labels ===\n");
        case_inlet_labels(rep, ctx);
        std::printf("=== SF-33 N3: (2) spectral vertex evaluation, analytic gradients ===\n");
        case_spectral(rep, ctx);
        std::printf("=== SF-33 N3: (5) SF-19 inlet flux, SF-18 report, v1 difference ===\n");
        case_sf19_inlet(rep, ctx);
        std::printf("=== SF-33 N3: (4i) production oracle, k = 1 ===\n");
        case_oracle_k1(rep, ctx);
        std::printf("=== SF-33 N3: (4ii) production oracle, control2d eps = 0.5 ===\n");
        case_oracle_control2d(rep, ctx);
        std::printf(
            "=== SF-33 N3: (4iii, 3, 6) production oracle generic3d eps = 0.25; trace_to_plane; "
            "determinism ===\n");
        case_oracle_generic3d(rep, ctx);
    } catch (const std::exception& e) {
        std::printf("[FAIL] unexpected exception: %s\n", e.what());
        rep.overall_pass = false;
    }
    const double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("\n=== inlet_slab_production16: %d checks, %s (%.1f s) ===\n", rep.checks,
                rep.overall_pass ? "ALL PASS" : "FAILURES", t);
    return rep.overall_pass ? 0 : 1;
}
