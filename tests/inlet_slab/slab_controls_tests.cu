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
 */

#include "src/core/DeviceBuffer.cuh"
#include "src/core/DeviceSpan.cuh"
#include "src/core/Scalar.hpp"
#include "src/physics/streamfunctions/inlet_slab/InletSlabGrid.cuh"
#include "src/physics/streamfunctions/inlet_slab/SlabMetrics.cuh"
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
    } catch (const std::exception& e) {
        std::printf("[FAIL] unexpected exception: %s\n", e.what());
        rep.overall_pass = false;
    }
    const double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("\n=== inlet_slab_controls: %d checks, %s (%.1f s) ===\n", rep.checks,
                rep.overall_pass ? "ALL PASS" : "FAILURES", t);
    return rep.overall_pass ? 0 : 1;
}
