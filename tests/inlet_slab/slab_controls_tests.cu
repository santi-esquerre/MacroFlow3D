/**
 * @file slab_controls_tests.cu
 * @brief SF-35 N2: fast tests of the verification-without-tracing controls of the inlet slab.
 *        Standalone ctest runner (inlet_slab_controls) in the TestReport style of
 *        tests/inlet_slab/slab_production_tests.cu. Cases are independent functions registered in
 *        main(); later nodes append further cases (positive controls A1-A4) to this file.
 *
 * Cases:
 *   (i)   evaluate_plane_metrics on manufactured labels (N = 16, k = 1 stage, vD = e1): plane sums
 *         add up to the global sums of evaluate_metrics (1e-12 relative), min_j min_c_j == min_c
 *         exactly, no backflow, no non-finite values.
 *   (i-b) labels with U_i = O(x1^2): plane 0 labels are affine to 4th order -> e_v_0 <= 1e-14.
 *   (ii)  manufactured backflow set in vD1: exact per-plane counts and minima.
 *   (iii) bitwise determinism of repeated evaluations; no workspace reallocation.
 *   (iv)  non-finite propagation: sum_j nonfinite_j == evaluate_metrics.nonfinite.
 *
 * SF-35 N4 positive controls (exactly known labels; the labels are not verified by tracing):
 *   A1  k = 1 at N = 16, 32: feet = (x2, x3), psi1 = x2, psi2 = x3 on planes 0..N (1e-13).
 *   A2  k = exp(0.5 cos 2 pi x2) at N = 16, 32: straight streamlines, labels invariant along x1
 *       (1e-10), psi2(0) = Q0 x3 and psi1(0) = host InletLabels (1e-12); e_v and r_F decrease with
 *       observed order >= 3.5 (16 -> 32); per-plane e_v growth ratio recorded.
 *   A3  control2d eps = 0.5 at N = 16: psi2 = x3 on every plane (1e-10).
 *   A4  synthetic direction field in m2 bands: seed_backflow, cap_exceeded, stagnation,
 *       invalid_conductivity and landing_failed (through discarded landings) with derived
 *       per-plane counts, derived backflow_encounters from a pocket in x3, NaN labels exactly at
 *       the failed seeds.
 */

#include "src/core/DeviceBuffer.cuh"
#include "src/core/DeviceSpan.cuh"
#include "src/core/Scalar.hpp"
#include "src/physics/streamfunctions/inlet_slab/InletSlabGrid.cuh"
#include "src/physics/streamfunctions/inlet_slab/SlabLabelTransport.cuh"
#include "src/physics/streamfunctions/inlet_slab/SlabMetrics.cuh"
#include "src/physics/streamfunctions/inlet_slab/SlabResidual.cuh"
#include "src/physics/streamfunctions/inlet_slab/SlabProductionSetup.cuh"
#include "src/runtime/CudaContext.cuh"
#include "src/runtime/cuda_check.cuh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace macroflow3d;
namespace sl = macroflow3d::streamfunctions::inlet_slab;

namespace {

constexpr double kTwoPi = 6.283185307179586476925286766559;

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
    // Complete the pageable copy device-wide before stream-ordered work on the non-blocking
    // CudaContext stream reads the buffer (see slab_production_tests.cu::upload).
    MACROFLOW3D_CUDA_CHECK(cudaDeviceSynchronize());
}

std::vector<real> download(const DeviceBuffer<real>& b) {
    std::vector<real> h(b.size());
    MACROFLOW3D_CUDA_CHECK(
        cudaMemcpy(h.data(), b.data(), h.size() * sizeof(real), cudaMemcpyDeviceToHost));
    return h;
}

DeviceSpan<const real> cspan(const DeviceBuffer<real>& b) {
    return DeviceSpan<const real>(b.data(), b.size());
}

/// Bitwise equality of two plane-metric vectors (field by field; no struct padding compared).
bool bitwise_equal(const std::vector<sl::SlabPlaneMetrics>& a,
                   const std::vector<sl::SlabPlaneMetrics>& b) {
    if (a.size() != b.size())
        return false;
    auto eq = [](real x, real y) { return std::memcmp(&x, &y, sizeof(real)) == 0; };
    for (std::size_t i = 0; i < a.size(); ++i) {
        const auto &p = a[i], &q = b[i];
        if (p.j != q.j || p.vD1_nonpos != q.vD1_nonpos || p.nonfinite != q.nonfinite ||
            !eq(p.x1, q.x1) || !eq(p.e_v, q.e_v) || !eq(p.e_i1, q.e_i1) || !eq(p.e_i2, q.e_i2) ||
            !eq(p.min_c, q.min_c) || !eq(p.vD1_min, q.vD1_min) || !eq(p.sum_cv2, q.sum_cv2) ||
            !eq(p.sum_vg1_2, q.sum_vg1_2) || !eq(p.sum_g1_2, q.sum_g1_2) ||
            !eq(p.sum_vg2_2, q.sum_vg2_2) || !eq(p.sum_g2_2, q.sum_g2_2))
            return false;
    }
    return true;
}

double rel(double a, double b) {
    const double d = std::max(std::abs(a), std::abs(b));
    return d > 0.0 ? std::abs(a - b) / d : 0.0;
}

/// k = 1 production stage (Y = 0): vD = e1 everywhere (as in slab_production_tests.cu).
struct Fixture {
    int N = 16;
    sl::InletSlabGrid grid = sl::InletSlabGrid::make(16);
    sl::ProductionStage stage;
    sl::SlabMetricsWorkspace ws;
    DeviceBuffer<real> U1, U2;
    bool ok = false;

    Fixture(TestReport& rep, CudaContext& ctx) {
        const sl::SlabFieldSource src = sl::SlabFieldSource::spectral(
            ctx, "uniform", N, std::vector<real>(static_cast<std::size_t>(N) * N * N, 0.0));
        stage = sl::build_production_stage(ctx, grid, src, 1.0);
        ok = stage.ok();
        rep.check(ok, "fixture: k = 1 production stage ok", sl::to_string(stage.report.status));
        if (ok)
            ws.prepare(grid);
    }

    /// U1 = a sin(2 pi x2) cos(2 pi x3) x1^p1, U2 = b cos(2 pi x2) sin(4 pi x3) x1^p2.
    void set_labels(real a, real b, int p1, int p2) {
        std::vector<real> h1(grid.full_size()), h2(grid.full_size());
        for (int j = 0; j <= N; ++j)
            for (int m2 = 0; m2 < N; ++m2)
                for (int m3 = 0; m3 < N; ++m3) {
                    const std::size_t i = grid.full_index(j, m2, m3);
                    const real x1 = grid.coord(j), x2 = grid.coord(m2), x3 = grid.coord(m3);
                    h1[i] = a * std::sin(kTwoPi * x2) * std::cos(kTwoPi * x3) * std::pow(x1, p1);
                    h2[i] = b * std::cos(kTwoPi * x2) * std::sin(2.0 * kTwoPi * x3) *
                            std::pow(x1, p2);
                }
        upload(U1, h1);
        upload(U2, h2);
    }
};

void case_identities(TestReport& rep, CudaContext& ctx, Fixture& f) {
    f.set_labels(0.05, 0.04, 1, 2);
    const sl::SlabMetrics m = sl::evaluate_metrics(ctx, f.grid, cspan(f.U1), cspan(f.U2),
                                                   f.stage.reference, f.ws);
    const auto pl = sl::evaluate_plane_metrics(ctx, f.grid, cspan(f.U1), cspan(f.U2),
                                               f.stage.reference, f.ws);
    rep.check(pl.size() == static_cast<std::size_t>(f.N + 1), "(i) N + 1 planes returned");
    const double np = static_cast<double>(f.grid.plane_size());
    const double nall = static_cast<double>(f.grid.full_size());
    double scv = 0, svg1 = 0, sg1 = 0, svg2 = 0, sg2 = 0, minc = INFINITY, e_max = 0.0;
    long long nonpos = 0, nonfinite = 0;
    bool xok = true;
    for (const auto& p : pl) {
        scv += p.sum_cv2;
        svg1 += p.sum_vg1_2;
        sg1 += p.sum_g1_2;
        svg2 += p.sum_vg2_2;
        sg2 += p.sum_g2_2;
        minc = std::min(minc, p.min_c);
        nonpos += p.vD1_nonpos;
        nonfinite += p.nonfinite;
        e_max = std::max(e_max, p.e_v);
        xok = xok && p.x1 == f.grid.coord(p.j);
        // e_v_j is consistent with its raw plane sum and the global v_rms.
        xok = xok && rel(p.e_v, std::sqrt(p.sum_cv2 / np) / m.v_rms) <= 1e-14;
    }
    rep.check(xok, "(i) x1_j = j / N and e_v_j = sqrt(sum_cv2 / N^2) / v_rms");
    rep.check(e_max > 1e-6, "(i) manufactured labels are non-trivial (max_j e_v_j > 1e-6)",
              fmt("%.3e", e_max));
    const double r_v = rel(scv, nall * m.e_v * m.e_v * m.v_rms * m.v_rms);
    rep.check(r_v <= 1e-12, "(i) sum_j N^2 e_v_j^2 = (N+1) N^2 e_v^2", fmt("rel %.2e", r_v));
    const double r_i1 = rel(svg1 / sg1, m.e_i1 * m.e_i1 * m.v_rms * m.v_rms);
    const double r_i2 = rel(svg2 / sg2, m.e_i2 * m.e_i2 * m.v_rms * m.v_rms);
    rep.check(r_i1 <= 1e-12 && r_i2 <= 1e-12,
              "(i) plane sums of (vD.g_i)^2 and |g_i|^2 add up to the global e_i",
              fmt("rel1 %.2e", r_i1) + " " + fmt("rel2 %.2e", r_i2));
    rep.check(minc == m.min_c, "(i) min_j min_c_j == min_c exactly",
              fmt("%.17g", minc) + " vs " + fmt("%.17g", m.min_c));
    rep.check(nonpos == 0, "(i) vD1_nonpos_j == 0 on every plane (vD = e1)");
    rep.check(nonfinite == 0 && m.nonfinite == 0, "(i) nonfinite_j == 0 on every plane");
    double e1ref = 0.0;
    for (const auto& p : pl)
        e1ref = std::max(e1ref, std::abs(p.vD1_min - 1.0));
    rep.check(e1ref == 0.0, "(i) vD1_min_j == 1 on every plane");
}

void case_plane0_affine(TestReport& rep, CudaContext& ctx, Fixture& f) {
    // x1^2 growth: d_x1 U = 0 on plane 0 and the one-sided 4th-order stencil is exact for x1^2, so
    // plane 0 has g1 = e2, g2 = e3, c = e1 = vD. (For U ~ x1 the plane-0 d_x1 U is nonzero.)
    f.set_labels(0.05, 0.04, 2, 2);
    const auto pl = sl::evaluate_plane_metrics(ctx, f.grid, cspan(f.U1), cspan(f.U2),
                                               f.stage.reference, f.ws);
    rep.check(pl[0].e_v <= 1e-14, "(i-b) plane 0 of O(x1^2) labels: e_v_0 <= 1e-14",
              fmt("%.3e", pl[0].e_v));
    rep.check(pl[0].e_i1 <= 1e-14 && pl[0].e_i2 <= 1e-14, "(i-b) plane 0: e_i_0 <= 1e-14",
              fmt("%.3e", std::max(pl[0].e_i1, pl[0].e_i2)));
    rep.check(pl[f.N].e_v > 1e-8, "(i-b) plane N is not affine (e_v_N > 1e-8)",
              fmt("%.3e", pl[f.N].e_v));
}

void case_backflow(TestReport& rep, CudaContext& ctx, Fixture& f) {
    f.set_labels(0.05, 0.04, 1, 2);
    auto& ref = f.stage.reference;
    const std::vector<real> v0_orig = download(ref.vD[0]);
    std::vector<real> v0 = v0_orig;
    for (int m2 = 0; m2 < 4; ++m2)
        for (int m3 = 0; m3 < f.N; ++m3)
            v0[f.grid.full_index(3, m2, m3)] = -1.0;
    v0[f.grid.full_index(7, 5, 9)] = -1.0;
    upload(ref.vD[0], v0);
    const auto pl = sl::evaluate_plane_metrics(ctx, f.grid, cspan(f.U1), cspan(f.U2), ref, f.ws);
    bool ok = true;
    std::string bad;
    for (const auto& p : pl) {
        const long long want = p.j == 3 ? 64 : (p.j == 7 ? 1 : 0);
        const real wmin = want > 0 ? -1.0 : 1.0;
        if (p.vD1_nonpos != want || p.vD1_min != wmin) {
            ok = false;
            bad += " j=" + std::to_string(p.j);
        }
    }
    rep.check(ok, "(ii) backflow counts 64 / 1 / 0 and vD1_min -1 / -1 / 1 per plane", bad);
    upload(ref.vD[0], v0_orig);
}

void case_determinism(TestReport& rep, CudaContext& ctx, Fixture& f) {
    f.set_labels(0.05, 0.04, 1, 2);
    const auto ptr0 = f.ws.storage_pointers();
    const std::size_t bytes0 = f.ws.allocated_bytes();
    const auto a = sl::evaluate_plane_metrics(ctx, f.grid, cspan(f.U1), cspan(f.U2),
                                              f.stage.reference, f.ws);
    const auto b = sl::evaluate_plane_metrics(ctx, f.grid, cspan(f.U1), cspan(f.U2),
                                              f.stage.reference, f.ws);
    // Interleave a global evaluation: it must not change the per-plane result either.
    (void)sl::evaluate_metrics(ctx, f.grid, cspan(f.U1), cspan(f.U2), f.stage.reference, f.ws);
    const auto c = sl::evaluate_plane_metrics(ctx, f.grid, cspan(f.U1), cspan(f.U2),
                                              f.stage.reference, f.ws);
    rep.check(bitwise_equal(a, b) && bitwise_equal(a, c),
              "(iii) repeated evaluate_plane_metrics are bitwise identical");
    rep.check(ptr0 == f.ws.storage_pointers() && bytes0 == f.ws.allocated_bytes(),
              "(iii) workspace storage pointers and bytes unchanged (no allocation after prepare)");
}

void case_nonfinite(TestReport& rep, CudaContext& ctx, Fixture& f) {
    f.set_labels(0.05, 0.04, 1, 2);
    std::vector<real> h = download(f.U1);
    h[f.grid.full_index(5, 3, 4)] = std::nan("");
    upload(f.U1, h);
    const sl::SlabMetrics m = sl::evaluate_metrics(ctx, f.grid, cspan(f.U1), cspan(f.U2),
                                                   f.stage.reference, f.ws);
    const auto pl = sl::evaluate_plane_metrics(ctx, f.grid, cspan(f.U1), cspan(f.U2),
                                               f.stage.reference, f.ws);
    long long tot = 0;
    int first = -1, last = -1;
    for (const auto& p : pl) {
        tot += p.nonfinite;
        if (p.nonfinite > 0) {
            if (first < 0)
                first = p.j;
            last = p.j;
        }
    }
    rep.check(m.nonfinite > 0 && tot == m.nonfinite,
              "(iv) sum_j nonfinite_j == evaluate_metrics.nonfinite",
              std::to_string(tot) + " vs " + std::to_string(m.nonfinite));
    rep.check(first >= 0 && first <= 5 && last >= 5 && pl[5].nonfinite > 0,
              "(iv) the plane holding the NaN vertex is flagged, others are within the stencil reach",
              "planes " + std::to_string(first) + ".." + std::to_string(last));
    rep.check(std::isnan(pl[5].min_c) && std::isfinite(pl[0].min_c) == (pl[0].nonfinite == 0),
              "(iv) min_c is NaN exactly on planes with non-finite values");
}

// ================================================================================================
// SF-35 N4: positive controls A1-A4 (labels known exactly; nothing here is verified by tracing the
// same object: decision 2026-10-07 item 3 (ii)). The tracer is the instrument under test.
// ================================================================================================

DeviceSpan<real> mspan(DeviceBuffer<real>& b) {
    return DeviceSpan<real>(b.data(), b.size());
}

int test_threads() {
    const unsigned hc = std::thread::hardware_concurrency();
    return static_cast<int>(std::max(1u, std::min(hc == 0 ? 1u : hc, 8u)));
}

struct Traced {
    sl::LabelTransportResult res;
    DeviceBuffer<real> p1, p2; ///< device labels, planes 0..N
    std::vector<real> psi1, psi2;
};

Traced trace_stage(CudaContext& ctx, const sl::ProductionStage& st, int threads) {
    Traced t;
    t.p1.resize(st.grid.full_size());
    t.p2.resize(st.grid.full_size());
    sl::LabelTransportOptions opt;
    opt.threads = threads;
    t.res = sl::construct_labels_by_tracing(ctx, st, opt, mspan(t.p1), mspan(t.p2));
    t.psi1 = download(t.p1);
    t.psi2 = download(t.p2);
    return t;
}

/// Max over plane j of |psi1 - x2|, |psi2 - x3| (affine labels).
double affine_error(const sl::InletSlabGrid& g, const Traced& t, int j) {
    double e = 0.0;
    for (int m2 = 0; m2 < g.n; ++m2)
        for (int m3 = 0; m3 < g.n; ++m3) {
            const std::size_t i = g.full_index(j, m2, m3);
            e = std::max(e, std::max(std::abs(t.psi1[i] - g.coord(m2)),
                                     std::abs(t.psi2[i] - g.coord(m3))));
        }
    return e;
}

// A1 ---------------------------------------------------------------------------------------------
// k = 1 (Y = 0): h_tilde = 0, g = e1, feet = (x2, x3), D-1 of constant samples -> psi1 = x2,
// psi2 = x3 on every plane.
void case_A1_k1(TestReport& rep, CudaContext& ctx) {
    for (const int N : {16, 32}) {
        const sl::InletSlabGrid grid = sl::InletSlabGrid::make(N);
        const sl::SlabFieldSource src = sl::SlabFieldSource::spectral(
            ctx, "uniform", N, std::vector<real>(static_cast<std::size_t>(N) * N * N, 0.0));
        const sl::ProductionStage st = sl::build_production_stage(ctx, grid, src, 0.25);
        const std::string tag = "A1 k = 1 N = " + std::to_string(N) + ": ";
        rep.check(st.ok(), tag + "production stage ok", sl::to_string(st.report.status));
        if (!st.ok())
            continue;
        const Traced t = trace_stage(ctx, st, test_threads());
        double efoot = 0.0, elab = 0.0;
        for (int j = 1; j <= N; ++j)
            for (int m2 = 0; m2 < N; ++m2)
                for (int m3 = 0; m3 < N; ++m3) {
                    const std::size_t i = grid.unknown_index(j, m2, m3);
                    efoot = std::max(efoot, std::max(std::abs(t.res.foot_y[i] - grid.coord(m2)),
                                                     std::abs(t.res.foot_z[i] - grid.coord(m3))));
                }
        for (int j = 0; j <= N; ++j)
            elab = std::max(elab, affine_error(grid, t, j));
        rep.check(t.res.status == sl::LabelTransportStatus::ok && t.res.non_ok == 0,
                  tag + "status ok, no failed streamline", sl::to_string(t.res.status));
        rep.check(efoot <= 1e-13, tag + "feet = (x2, x3)", fmt("max err %.3e", efoot));
        rep.check(elab <= 1e-13, tag + "psi1 = x2, psi2 = x3 on every plane 0..N",
                  fmt("max err %.3e", elab));
        rep.check(std::abs(st.labels.Q0() - 1.0) <= 1e-14, tag + "Q0 = 1",
                  fmt("Q0 - 1 = %.3e", st.labels.Q0() - 1.0));
    }
}

// A2 ---------------------------------------------------------------------------------------------
struct A2Run {
    int N = 0;
    double e_v = 0.0, r_F = 0.0, r_out = 0.0, plane_inv = 0.0, q0_err = 0.0, psi1_err = 0.0;
    double Q0 = 0.0, growth_ratio = 0.0;
    bool ok = false;
};

A2Run run_A2(TestReport& rep, CudaContext& ctx, int N) {
    A2Run R;
    R.N = N;
    const double h = 1.0 / N;
    const sl::InletSlabGrid grid = sl::InletSlabGrid::make(N);
    std::vector<real> cell(static_cast<std::size_t>(N) * N * N);
    for (int k = 0; k < N; ++k)
        for (int j = 0; j < N; ++j)
            for (int i = 0; i < N; ++i)
                cell[static_cast<std::size_t>(i) + N * (static_cast<std::size_t>(j) + N * k)] =
                    std::cos(kTwoPi * (j + 0.5) * h);
    const sl::SlabFieldSource src = sl::SlabFieldSource::spectral(ctx, "cos_x2", N, cell);
    sl::ProductionStage st = sl::build_production_stage(ctx, grid, src, 0.5);
    const std::string tag = "A2 k = exp(0.5 cos 2 pi x2) N = " + std::to_string(N) + ": ";
    rep.check(st.ok(), tag + "production stage ok", sl::to_string(st.report.status));
    if (!st.ok())
        return R;
    const Traced t = trace_stage(ctx, st, test_threads());
    rep.check(t.res.status == sl::LabelTransportStatus::ok && t.res.non_ok == 0,
              tag + "tracing status ok, no failed streamline", sl::to_string(t.res.status));
    R.Q0 = st.labels.Q0();
    // (a) plane invariance of the transported labels on straight streamlines
    for (int j = 1; j <= N; ++j)
        for (int m2 = 0; m2 < N; ++m2)
            for (int m3 = 0; m3 < N; ++m3) {
                const std::size_t i = grid.full_index(j, m2, m3), i0 = grid.full_index(0, m2, m3);
                R.plane_inv = std::max(R.plane_inv, std::max(std::abs(t.psi1[i] - t.psi1[i0]),
                                                             std::abs(t.psi2[i] - t.psi2[i0])));
            }
    // (b) psi2 = Q0 x3 and psi1 against the host label evaluation, plane 0
    for (int m2 = 0; m2 < N; ++m2)
        for (int m3 = 0; m3 < N; ++m3) {
            const std::size_t i0 = grid.full_index(0, m2, m3);
            R.q0_err = std::max(R.q0_err, std::abs(t.psi2[i0] - R.Q0 * grid.coord(m3)));
            R.psi1_err = std::max(
                R.psi1_err, std::abs(t.psi1[i0] - st.labels.psi1(grid.coord(m2), grid.coord(m3))));
        }
    rep.check(R.plane_inv <= 1e-10,
              tag + "labels invariant along x1: |psi_i(j) - psi_i(0)| <= 1e-10",
              fmt("max %.3e", R.plane_inv));
    rep.check(R.q0_err <= 1e-12, tag + "psi2(0) = Q0 x3 (Q0 = mean of the D-1 interpolant)",
              fmt("Q0 = %.15f;", R.Q0) + fmt(" max err %.3e", R.q0_err));
    rep.check(R.psi1_err <= 1e-12,
              tag + "psi1(0) = host InletLabels::psi1 (GPU vs host evaluation)",
              fmt("max err %.3e", R.psi1_err));
    // SF-19 normalizes the MEAN flux to 1 (qbar = e1): G1 = 1 / mean(K) = 1 / I_0(0.5) (flow
    // parallel to the layers, arithmetic mean), G2 = G3 = 0, so the inlet flux is K G1 with mean 1
    // and Q0 = 1 (the orchestrator's I_0(0.5) was the unnormalized mean). psi1 is nonetheless
    // nontrivial: V = exp(0.5 cos 2 pi y) / I_0(0.5) varies with y.
    double i0 = 0.0, term = 1.0;
    for (int k = 0; k < 30; ++k) {
        i0 += term;
        term *= 0.0625 / ((k + 1.0) * (k + 1.0)); // (x/2)^2 / k^2, x = 0.5
    }
    double dpsi1 = 0.0;
    for (int m2 = 0; m2 < N; ++m2)
        for (int m3 = 0; m3 < N; ++m3)
            dpsi1 = std::max(dpsi1, std::abs(t.psi1[grid.full_index(0, m2, m3)] - grid.coord(m2)));
    rep.check(std::abs(R.Q0 - 1.0) <= 1e-8, tag + "Q0 = 1 (SF-19 mean flux 1)",
              fmt("Q0 - 1 = %.3e", R.Q0 - 1.0));
    rep.check(std::abs(st.G[0] * i0 - 1.0) <= 1e-8 && std::abs(st.G[1]) <= 1e-8 &&
                  std::abs(st.G[2]) <= 1e-8,
              tag + "SF-19 G = (1 / I_0(0.5), 0, 0)",
              fmt("G1 I_0 - 1 = %.3e", st.G[0] * i0 - 1.0) + fmt(" (I_0 = %.10f)", i0));
    rep.check(dpsi1 > 1e-2 && dpsi1 < 0.2, tag + "psi1 - x2 nontrivial on the inlet (~ 0.5 / 2 pi)",
              fmt("max |psi1 - x2| = %.3e", dpsi1));
    // (c) metrics of the traced labels
    DeviceBuffer<real> U1(grid.full_size()), U2(grid.full_size()), E(grid.unknown_size());
    sl::labels_to_periodic_parts(ctx, grid, cspan(t.p1), cspan(t.p2), mspan(U1), mspan(U2));
    sl::SlabMetricsWorkspace mws;
    mws.prepare(grid);
    const sl::SlabMetrics m =
        sl::evaluate_metrics(ctx, grid, cspan(U1), cspan(U2), st.reference, mws);
    const auto pl = sl::evaluate_plane_metrics(ctx, grid, cspan(U1), cspan(U2), st.reference, mws);
    sl::SlabResidualWorkspace rws;
    rws.prepare(grid);
    sl::SlabResidualNorms nrm;
    sl::evaluate_residual(ctx, grid, st.inputs, cspan(U1), cspan(U2), mspan(E), rws, &nrm);
    R.e_v = m.e_v;
    R.r_F = nrm.r_F;
    R.r_out = nrm.r_out;
    R.growth_ratio = pl[0].e_v > 0.0 ? pl[N].e_v / pl[0].e_v : std::nan("");
    bool fin = std::isfinite(m.e_v) && std::isfinite(nrm.r_F) && m.nonfinite == 0;
    for (const auto& p : pl)
        fin = fin && std::isfinite(p.e_v);
    rep.check(fin, tag + "e_v, r_F, per-plane e_v finite",
              fmt("e_v %.6e", m.e_v) + fmt(" r_F %.6e", nrm.r_F) + fmt(" r_out %.6e", nrm.r_out));
    std::printf("  A2 N = %d: e_v = %.6e  e_i1 = %.3e  e_i2 = %.3e  r_F = %.6e  r_out = %.6e\n", N,
                m.e_v, m.e_i1, m.e_i2, nrm.r_F, nrm.r_out);
    std::printf("  A2 N = %d: per-plane e_v (j = 0, 1, N/2, N-1, N) = %.3e %.3e %.3e %.3e %.3e;"
                " GROWTH-like e_v_N / e_v_0 = %.4f (recorded, not asserted)\n",
                N, pl[0].e_v, pl[1].e_v, pl[N / 2].e_v, pl[N - 1].e_v, pl[N].e_v, R.growth_ratio);
    R.ok = true;
    return R;
}

void case_A2_cos(TestReport& rep, CudaContext& ctx) {
    const A2Run a = run_A2(rep, ctx, 16);
    const A2Run b = run_A2(rep, ctx, 32);
    if (!a.ok || !b.ok)
        return;
    const double ov = std::log2(a.e_v / b.e_v), of = std::log2(a.r_F / b.r_F);
    std::printf("  A2 observed orders 16 -> 32: e_v %.3f (%.3e -> %.3e), r_F %.3f (%.3e -> %.3e)\n",
                ov, a.e_v, b.e_v, of, a.r_F, b.r_F);
    rep.check(b.e_v < a.e_v && b.r_F < a.r_F, "A2: e_v and r_F decrease 16 -> 32");
    rep.check(ov >= 3.5, "A2: observed order log2(e_v_16 / e_v_32) >= 3.5", fmt("%.3f", ov));
    rep.check(of >= 3.5, "A2: observed order log2(r_F_16 / r_F_32) >= 3.5", fmt("%.3f", of));
}

// A3 ---------------------------------------------------------------------------------------------
void case_A3_control2d(TestReport& rep, CudaContext& ctx) {
    const int N = 16;
    const sl::InletSlabGrid grid = sl::InletSlabGrid::make(N);
    const sl::SlabFieldSource src = sl::SlabFieldSource::analytic("control2d", N);
    const sl::ProductionStage st = sl::build_production_stage(ctx, grid, src, 0.5);
    rep.check(st.ok(), "A3 control2d eps = 0.5: production stage ok",
              sl::to_string(st.report.status));
    if (!st.ok())
        return;
    const Traced t = trace_stage(ctx, st, test_threads());
    double worst = 0.0;
    for (int j = 0; j <= N; ++j)
        for (int m2 = 0; m2 < N; ++m2)
            for (int m3 = 0; m3 < N; ++m3)
                worst = std::max(worst,
                                 std::abs(t.psi2[grid.full_index(j, m2, m3)] - grid.coord(m3)));
    // The round trip of the spline-direction field is h_max-limited (SlabOracle.cuh header), so
    // the stage status may be roundtrip_exceeded at (h/8, 1e-8); here only streamline failures
    // (non_ok) are excluded and the round trip is recorded.
    rep.check(t.res.non_ok == 0 && t.res.status != sl::LabelTransportStatus::trace_fail,
              "A3: no failed streamline", std::string(sl::to_string(t.res.status)) +
                                              fmt(", max round trip %.3e (recorded)", t.res.max_roundtrip));
    rep.check(std::abs(st.labels.Q0() - 1.0) <= 1e-9,
              "A3: Q0 = 1 (SF-19 mean flux 1, V independent of x3)",
              fmt("Q0 - 1 = %.3e", st.labels.Q0() - 1.0));
    rep.check(worst <= 1e-10, "A3 control2d: psi2 = x3 on every plane (planar flow, v3 = 0)",
              fmt("max |psi2 - x3| %.3e", worst));
}

// A4 ---------------------------------------------------------------------------------------------
/// Direction field with regions in m2 bands (x2 is constant along every streamline: g2 = 0), N = 16:
///   m2 {0,1,14,15}: g = e1 (ok, straight);        m2 {2,3}: g1 = -1 (seed_backflow);
///   m2 {4,5}: g = (1e-3, 0, 1) (cap_exceeded at max_arclength 2: needs arclength > 62);
///   m2 {6,7}: g1 = NaN (stagnation at the seed);  m2 {8,9}: k = -1 (invalid_conductivity);
///   m2 {10,11}: g = e1 except g1 = -1 at x1 == 0.0 exactly (the landing's last stage sits exactly
///               on X = 0 -> every landing is discarded: landings_discarded > 0, final
///               landing_failed);
///   m2 {12,13}: "pocket" in x3: g = (1, 0, 1) except g = (-1e-3, 0, 1) for x3 mod 1 in
///               [7.5/16, 11.5/16): seeds in the pocket -> seed_backflow; others descend in x3 at
///               the rate they descend in x1 (backward trace) and cross a pocket (interior
///               g1 <= 0 steps: backflow_encounter, ok streamline) iff the nominal path meets it.
constexpr int kBandSeed = 2, kBandCap = 4, kBandStag = 6, kBandInvk = 8, kBandLand = 10,
              kBandPocket = 12;
constexpr double kPocketLo = 7.5 / 16.0, kPocketHi = 11.5 / 16.0;

struct FailureField {
    int N = 16;
    void operator()(const double x[3], double g[3], double& k) const {
        k = 1.0;
        g[0] = 1.0;
        g[1] = 0.0;
        g[2] = 0.0;
        const int band = static_cast<int>(std::floor(x[1] * N + 0.5)) / 2 * 2; // pairs of m2
        if (band == kBandSeed) {
            g[0] = -1.0;
        } else if (band == kBandCap) {
            g[0] = 1e-3;
            g[2] = 1.0;
        } else if (band == kBandStag) {
            g[0] = std::nan("");
        } else if (band == kBandInvk) {
            k = -1.0;
        } else if (band == kBandLand) {
            if (x[0] == 0.0)
                g[0] = -1.0;
        } else if (band == kBandPocket) {
            const double z = x[2] - std::floor(x[2]);
            g[2] = 1.0;
            g[0] = (z >= kPocketLo && z < kPocketHi) ? -1e-3 : 1.0;
        }
    }
};

/// Nominal pocket traversal of the backward path from (x1 = j/16, x3 = m3/16), in units of h = 1/16
/// (the pocket is [7.5, 11.5) mod 16 units). seed_in: seed in the pocket; crossings: number of
/// pockets crossed; foot: nominal unwrapped foot x3 in units (pocket gain O(1e-3 h) ignored).
struct PocketPath {
    bool seed_in = false;
    int crossings = 0;
    double foot = 0.0;
};

PocketPath pocket_path(int m3, int j) {
    PocketPath p;
    const double z = m3 % 16;
    if (z >= 7.5 && z < 11.5) {
        p.seed_in = true;
        return p;
    }
    double pos = m3, rem = j;
    while (rem > 0.0) {
        // upper pocket edge strictly below pos: 11.5 + 16 n
        const double e = 11.5 + 16.0 * std::floor((pos - 11.5) / 16.0 - 1e-12);
        const double d = pos - e;
        if (d < rem) {
            ++p.crossings;
            rem -= d;
            pos = e - 4.0;
        } else {
            break;
        }
    }
    p.foot = pos - rem;
    return p;
}

void case_A4_failures(TestReport& rep, CudaContext& ctx) {
    const int N = 16;
    const sl::InletSlabGrid grid = sl::InletSlabGrid::make(N);
    const sl::SlabFieldSource src = sl::SlabFieldSource::spectral(
        ctx, "uniform", N, std::vector<real>(static_cast<std::size_t>(N) * N * N, 0.0));
    const sl::ProductionStage st = sl::build_production_stage(ctx, grid, src, 1.0);
    rep.check(st.ok(), "A4: k = 1 stage (labels) ok");
    if (!st.ok())
        return;
    sl::LabelTransportOptions opt;
    opt.max_arclength = 2.0;
    opt.threads = test_threads();
    DeviceBuffer<real> p1(grid.full_size()), p2(grid.full_size());
    const sl::LabelTransportResult res = sl::construct_labels_by_tracing(
        ctx, grid, FailureField{N}, st.labels, opt, mspan(p1), mspan(p2));
    const std::vector<real> psi1 = download(p1), psi2 = download(p2);
    std::printf("%s", res.table().c_str());

    // Derived counts per plane j >= 1 (16 seeds per m2 row, two rows per band).
    long long pocket_ok = 0, pocket_seed = 0;
    for (int m3 = 0; m3 < N; ++m3)
        (pocket_path(m3, 1).seed_in ? pocket_seed : pocket_ok) += 2;
    const long long rows2 = 2LL * N;
    const long long want_ok = 4LL * N + pocket_ok, want_seed = rows2 + pocket_seed;
    const long long want_back_non_ok = want_seed + 3 * rows2 + rows2; // seed + cap, stag, invk, land
    bool hist_ok = true, enc_ok = true, ldisc_ok = true;
    long long enc_total = 0, ldisc_total = 0;
    std::string bad;
    for (const sl::LabelTransportPlane& P : res.planes) {
        if (P.j == 0)
            continue;
        long long want_enc = 0;
        for (int m3 = 0; m3 < N; ++m3) {
            const PocketPath pp = pocket_path(m3, P.j);
            if (!pp.seed_in && pp.crossings > 0)
                want_enc += 2;
        }
        const bool h = P.status_hist[0] == want_ok && P.status_hist[1] == want_seed &&
                       P.status_hist[2] == rows2 && P.status_hist[3] == rows2 &&
                       P.status_hist[4] == 0 && P.status_hist[5] == rows2 &&
                       P.status_hist[6] == rows2 && P.back_non_ok == want_back_non_ok &&
                       P.fwd_non_ok == 0 && P.non_ok == want_back_non_ok && P.flagged;
        hist_ok = hist_ok && h;
        enc_ok = enc_ok && P.backflow_encounters == want_enc;
        ldisc_ok = ldisc_ok && P.landings_discarded >= rows2;
        enc_total += want_enc;
        ldisc_total += P.landings_discarded;
        if (!h)
            bad += " j=" + std::to_string(P.j);
    }
    rep.check(res.status == sl::LabelTransportStatus::trace_fail, "A4: status trace_fail",
              sl::to_string(res.status));
    rep.check(hist_ok,
              "A4: per-plane histogram (ok, seed_backflow, cap, stagnation, underflow, "
              "landing_failed, invalid_k) = derived counts",
              "want (" + std::to_string(want_ok) + "," + std::to_string(want_seed) + "," +
                  std::to_string(rows2) + "," + std::to_string(rows2) + ",0," +
                  std::to_string(rows2) + "," + std::to_string(rows2) + ")" + bad);
    const long long want_non_ok = static_cast<long long>(N) * want_back_non_ok;
    rep.check(res.non_ok == want_non_ok &&
                  res.seed_backflow == static_cast<long long>(N) * want_seed,
              "A4: non_ok and seed_backflow stage totals",
              "non_ok " + std::to_string(res.non_ok) + " (want " + std::to_string(want_non_ok) +
                  "), seed_backflow " + std::to_string(res.seed_backflow));
    rep.check(enc_ok && res.backflow_encounters == enc_total && enc_total > 0,
              "A4: backflow_encounters per plane = 2 x #(pocket-crossing ok seeds) (derived)",
              "total " + std::to_string(res.backflow_encounters) + " (want " +
                  std::to_string(enc_total) + ")");
    rep.check(ldisc_ok && res.landings_discarded >= static_cast<long long>(N) * rows2,
              "A4: landings_discarded >= 1 per landing_failed seed on every plane",
              "total " + std::to_string(res.landings_discarded) + " (min " +
                  std::to_string(static_cast<long long>(N) * rows2) + ")");

    // Labels: NaN exactly at failed seeds, finite elsewhere; ok seeds against the nominal paths.
    bool nan_ok = true, fin_ok = true, pocket_lab_ok = true;
    double worst_pocket = 0.0;
    for (int j = 0; j <= N; ++j)
        for (int m2 = 0; m2 < N; ++m2)
            for (int m3 = 0; m3 < N; ++m3) {
                const std::size_t i = grid.full_index(j, m2, m3);
                const int band = m2 / 2 * 2;
                const PocketPath pp = pocket_path(m3, std::max(j, 1));
                const bool failed = j >= 1 && (band == kBandSeed || band == kBandCap ||
                                               band == kBandStag || band == kBandInvk ||
                                               band == kBandLand ||
                                               (band == kBandPocket && pp.seed_in));
                const bool nan_both = std::isnan(psi1[i]) && std::isnan(psi2[i]);
                if (failed) {
                    nan_ok = nan_ok && nan_both;
                    continue;
                }
                fin_ok = fin_ok && std::isfinite(psi1[i]) && std::isfinite(psi2[i]);
                if (band == kBandPocket && j >= 1) {
                    // psi1 = x2; psi2 = foot x3 (unwrapped): nominal path minus the x1 gain of each
                    // pocket crossing (dx1/dx3 = 1e-3 over the width 0.25 = 2.5e-4, consumed
                    // outside the pocket at unit x3 / x1 rate).
                    const double e = std::abs(psi2[i] - (pp.foot / N - pp.crossings * 2.5e-4));
                    worst_pocket = std::max(worst_pocket, e);
                    pocket_lab_ok = pocket_lab_ok && std::abs(psi1[i] - grid.coord(m2)) <= 1e-13 &&
                                    e <= 2e-5;
                } else {
                    fin_ok = fin_ok && std::abs(psi1[i] - grid.coord(m2)) <= 1e-13 &&
                             std::abs(psi2[i] - grid.coord(m3)) <= 1e-13;
                }
            }
    rep.check(nan_ok, "A4: NaN labels (psi1 and psi2) exactly at the failed seeds");
    rep.check(fin_ok, "A4: finite affine labels at every straight ok seed, plane 0 included");
    rep.check(pocket_lab_ok,
              "A4: pocket ok seeds: psi1 = x2, psi2 = nominal foot x3 (shift 4 h per crossing)",
              fmt("max |psi2 - nominal| = %.3e", worst_pocket) + fmt(" (limit %.3e)", 2e-5));
    long long mn = -1;
    for (const auto& P : res.planes)
        if (P.j > 0 && (mn < 0 || P.landings_discarded < mn))
            mn = P.landings_discarded;
    std::printf("  A4: landings_discarded per plane (min over planes) = %lld; stage total %lld\n", mn,
                ldisc_total);
}

} // namespace

int main() {
    TestReport rep;
    CudaContext ctx(0);
    const auto t0 = std::chrono::steady_clock::now();
    try {
        std::printf("=== SF-35 N2: per-plane Gate 3A metrics ===\n");
        Fixture f(rep, ctx);
        if (f.ok) {
            case_identities(rep, ctx, f);
            case_plane0_affine(rep, ctx, f);
            case_backflow(rep, ctx, f);
            case_determinism(rep, ctx, f);
            case_nonfinite(rep, ctx, f);
        }
        auto since = [&](const std::chrono::steady_clock::time_point& a) {
            return std::chrono::duration<double>(std::chrono::steady_clock::now() - a).count();
        };
        auto tc = std::chrono::steady_clock::now();
        std::printf("=== SF-35 N4: A1 k = 1 (N = 16, 32) ===\n");
        case_A1_k1(rep, ctx);
        std::printf("  [A1 %.1f s]\n", since(tc));
        tc = std::chrono::steady_clock::now();
        std::printf("=== SF-35 N4: A2 k = exp(0.5 cos 2 pi x2) (N = 16, 32) ===\n");
        case_A2_cos(rep, ctx);
        std::printf("  [A2 %.1f s]\n", since(tc));
        tc = std::chrono::steady_clock::now();
        std::printf("=== SF-35 N4: A3 control2d eps = 0.5 (N = 16) ===\n");
        case_A3_control2d(rep, ctx);
        std::printf("  [A3 %.1f s]\n", since(tc));
        tc = std::chrono::steady_clock::now();
        std::printf("=== SF-35 N4: A4 synthetic failure field (N = 16) ===\n");
        case_A4_failures(rep, ctx);
        std::printf("  [A4 %.1f s]\n", since(tc));
    } catch (const std::exception& e) {
        std::printf("[FAIL] unexpected exception: %s\n", e.what());
        rep.overall_pass = false;
    }
    const double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("\n=== inlet_slab_controls: %d checks, %s (%.1f s) ===\n", rep.checks,
                rep.overall_pass ? "ALL PASS" : "FAILURES", t);
    return rep.overall_pass ? 0 : 1;
}
