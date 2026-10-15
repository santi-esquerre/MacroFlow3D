/**
 * @file slab_trace_tests.cu
 * @brief SF-35 N1: fast tests (N = 16) of the label-transport constructor
 *        `construct_labels_by_tracing` (SlabLabelTransport.cuh), a thin wrapper of the SF-33
 *        production oracle. Standalone ctest runner in the printed-checks style of
 *        tests/inlet_slab/slab_production_tests.cu.
 *
 * Cases:
 *   (i)   k = 1: status ok, h_max_effective = h/8, feet = (x2, x3), affine labels on planes 0..N,
 *         status histogram all ok, no seed backflow / discarded landings.
 *   (ii)  generic3d eps = 0.25: status ok at (h/8, 1e-8) with max round trip <= 1e-8; bitwise
 *         identical outputs for 1 and 4 threads; labels bitwise equal to a direct compute_oracle;
 *         status roundtrip_exceeded for max_roundtrip = 1e-14.
 *   (iii) failure accounting with a synthetic direction field: exact per-plane status histograms,
 *         NaN labels exactly at the failed seeds, status trace_fail.
 *   (iv)  save_labels round trip (.npy bitwise, labels.json keys).
 */

#include "src/core/DeviceBuffer.cuh"
#include "src/core/DeviceSpan.cuh"
#include "src/core/Scalar.hpp"
#include "src/physics/streamfunctions/inlet_slab/InletLabels.cuh"
#include "src/physics/streamfunctions/inlet_slab/InletSlabGrid.cuh"
#include "src/physics/streamfunctions/inlet_slab/NpyIo.hpp"
#include "src/physics/streamfunctions/inlet_slab/SlabLabelTransport.cuh"
#include "src/physics/streamfunctions/inlet_slab/SlabOracle.cuh"
#include "src/physics/streamfunctions/inlet_slab/SlabProductionSetup.cuh"
#include "src/runtime/CudaContext.cuh"
#include "src/runtime/cuda_check.cuh"

#include "src/external/nlohmann/json.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace macroflow3d;
namespace sl = macroflow3d::streamfunctions::inlet_slab;
namespace cg = macroflow3d::closure_gate;

namespace {

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

std::vector<real> download(const real* d, std::size_t n) {
    std::vector<real> h(n);
    MACROFLOW3D_CUDA_CHECK(cudaMemcpy(h.data(), d, n * sizeof(real), cudaMemcpyDeviceToHost));
    return h;
}

bool same_bits(double a, double b) {
    return std::memcmp(&a, &b, sizeof(double)) == 0;
}

bool same_vec(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size())
        return false;
    for (std::size_t i = 0; i < a.size(); ++i)
        if (!same_bits(a[i], b[i]))
            return false;
    return true;
}

struct Run {
    sl::LabelTransportResult res;
    std::vector<real> psi1, psi2;
};

template <class Field>
Run run_tracing(CudaContext& ctx, const sl::InletSlabGrid& grid, const Field& fld,
                const sl::InletLabels& labels, const sl::LabelTransportOptions& opt) {
    DeviceBuffer<real> p1(grid.full_size()), p2(grid.full_size());
    Run r;
    r.res = sl::construct_labels_by_tracing(ctx, grid, fld, labels, opt,
                                            DeviceSpan<real>(p1.data(), p1.size()),
                                            DeviceSpan<real>(p2.data(), p2.size()));
    r.psi1 = download(p1.data(), p1.size());
    r.psi2 = download(p2.data(), p2.size());
    return r;
}

Run run_stage(CudaContext& ctx, const sl::ProductionStage& st,
              const sl::LabelTransportOptions& opt) {
    DeviceBuffer<real> p1(st.grid.full_size()), p2(st.grid.full_size());
    Run r;
    r.res = sl::construct_labels_by_tracing(ctx, st, opt, DeviceSpan<real>(p1.data(), p1.size()),
                                            DeviceSpan<real>(p2.data(), p2.size()));
    r.psi1 = download(p1.data(), p1.size());
    r.psi2 = download(p2.data(), p2.size());
    return r;
}

std::string tables_of(const sl::LabelTransportResult& r) {
    // per-plane table without the timing-bearing TRANSPORT line
    std::string t = r.table();
    const std::size_t p = t.rfind("  TRANSPORT");
    return p == std::string::npos ? t : t.substr(0, p);
}

// (i) --------------------------------------------------------------------------------------------
void case_k1(TestReport& rep, CudaContext& ctx) {
    const int N = 16;
    const sl::InletSlabGrid grid = sl::InletSlabGrid::make(N);
    const sl::SlabFieldSource src = sl::SlabFieldSource::spectral(
        ctx, "uniform", N, std::vector<real>(static_cast<std::size_t>(N) * N * N, 0.0));
    sl::ProductionStage st = sl::build_production_stage(ctx, grid, src, 1.0);
    rep.check(st.ok(), "k = 1: production stage ok", sl::to_string(st.report.status));
    if (!st.ok())
        return;
    sl::LabelTransportOptions opt;
    opt.threads = 2;
    const Run o = run_stage(ctx, st, opt);
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
    bool hist_ok = true;
    long long landings = 0;
    for (const sl::LabelTransportPlane& P : o.res.planes) {
        if (P.j == 0)
            continue;
        hist_ok = hist_ok && P.status_hist[0] == static_cast<long long>(N) * N;
        for (int s = 1; s < 7; ++s)
            hist_ok = hist_ok && P.status_hist[s] == 0;
        landings += P.landings_discarded;
    }
    rep.check(o.res.status == sl::LabelTransportStatus::ok, "k = 1: status ok",
              sl::to_string(o.res.status));
    rep.check(o.res.h_max_effective == grid.h / 8.0, "k = 1: h_max_effective = h/8",
              fmt("%.6e", o.res.h_max_effective));
    rep.check(efoot <= 1e-13, "k = 1: feet = (x2, x3)", fmt("max err %.3e", efoot));
    rep.check(elab <= 1e-13, "k = 1: labels affine on planes 0..N", fmt("max err %.3e", elab));
    rep.check(hist_ok, "k = 1: per-plane status histogram all ok");
    rep.check(o.res.seed_backflow == 0 && o.res.landings_discarded == 0 && landings == 0 &&
                  o.res.non_ok == 0,
              "k = 1: seed_backflow = 0, landings_discarded = 0, non_ok = 0");
}

// (ii) -------------------------------------------------------------------------------------------
void case_generic3d(TestReport& rep, CudaContext& ctx) {
    const int N = 16;
    const sl::InletSlabGrid grid = sl::InletSlabGrid::make(N);
    const sl::SlabFieldSource src = sl::SlabFieldSource::analytic("generic3d", N);
    sl::ProductionStage st = sl::build_production_stage(ctx, grid, src, 0.25);
    rep.check(st.ok(), "generic3d eps = 0.25: production stage ok", sl::to_string(st.report.status));
    if (!st.ok())
        return;
    sl::LabelTransportOptions o1;
    o1.threads = 1;
    sl::LabelTransportOptions o4 = o1;
    o4.threads = 4;
    const Run a = run_stage(ctx, st, o1);
    const Run b = run_stage(ctx, st, o4);
    std::printf("%s", a.res.table().c_str());
    rep.check(a.res.status == sl::LabelTransportStatus::ok && a.res.max_roundtrip <= 1e-8,
              "generic3d (h/8, 1e-8): status ok, max round trip <= 1e-8",
              fmt("max round trip %.3e", a.res.max_roundtrip));
    rep.check(same_vec(a.psi1, b.psi1) && same_vec(a.psi2, b.psi2) &&
                  same_vec(a.res.foot_y, b.res.foot_y) && same_vec(a.res.foot_z, b.res.foot_z) &&
                  same_vec(a.res.roundtrip, b.res.roundtrip) &&
                  tables_of(a.res) == tables_of(b.res),
              "generic3d: threads = 1 vs 4 bitwise identical (psi1, psi2, feet, round trip, "
              "per-plane tables)");

    sl::SlabOracleOptions oo;
    oo.tol = 1e-8;
    oo.h_max = grid.h / 8.0;
    oo.threads = 1;
    oo.max_roundtrip = 1e-8;
    DeviceBuffer<real> p1(grid.full_size()), p2(grid.full_size());
    const sl::SlabSplineDirectionField fld = st.direction_field();
    const sl::SlabOracleResult direct =
        sl::compute_oracle(ctx, grid, fld, st.labels, oo, DeviceSpan<real>(p1.data(), p1.size()),
                           DeviceSpan<real>(p2.data(), p2.size()));
    rep.check(same_vec(download(p1.data(), p1.size()), a.psi1) &&
                  same_vec(download(p2.data(), p2.size()), a.psi2) &&
                  same_vec(direct.foot_y, a.res.foot_y) && same_vec(direct.roundtrip, a.res.roundtrip),
              "generic3d: wrapper labels bitwise equal to a direct compute_oracle call");

    sl::LabelTransportOptions tight = o1;
    tight.max_roundtrip = 1e-14;
    const Run c = run_stage(ctx, st, tight);
    rep.check(c.res.status == sl::LabelTransportStatus::roundtrip_exceeded,
              "generic3d: max_roundtrip = 1e-14 -> roundtrip_exceeded",
              std::string(sl::to_string(c.res.status)) + fmt(", max round trip %.3e", c.res.max_roundtrip));
    bool gpu_throws = false;
    sl::LabelTransportOptions og = o1;
    og.backend = sl::TracerBackend::gpu;
    try {
        (void)run_stage(ctx, st, og);
    } catch (const std::runtime_error&) {
        gpu_throws = true;
    }
    rep.check(gpu_throws, "stage overload: gpu backend throws (tracer not built, N6)");
}

// (iii) ------------------------------------------------------------------------------------------
/// g = (1, 0, 0) except: x2 < 1/4 - 1e-9 (seeds m2 < N/4): g = (-1, 0, 0) (seed backflow);
/// 1/4 - 1e-9 <= x2 < 1/2 - 1e-9 (N/4 <= m2 < N/2): g = (1e-3, 0, 1): x2 constant, the line moves
/// in x3 and needs arclength >= x1 / 1e-3 > 2 to reach x1 = 0 (cap_exceeded at max_arclength 2).
struct SyntheticField {
    void operator()(const double x[3], double g[3], double& k) const {
        k = 1.0;
        if (x[1] < 0.25 - 1e-9) {
            g[0] = -1.0;
            g[1] = 0.0;
            g[2] = 0.0;
        } else if (x[1] < 0.5 - 1e-9) {
            g[0] = 1e-3;
            g[1] = 0.0;
            g[2] = 1.0;
        } else {
            g[0] = 1.0;
            g[1] = 0.0;
            g[2] = 0.0;
        }
    }
};

void case_failure_accounting(TestReport& rep, CudaContext& ctx) {
    const int N = 16;
    const sl::InletSlabGrid grid = sl::InletSlabGrid::make(N);
    const sl::SlabFieldSource src = sl::SlabFieldSource::spectral(
        ctx, "uniform", N, std::vector<real>(static_cast<std::size_t>(N) * N * N, 0.0));
    sl::ProductionStage st = sl::build_production_stage(ctx, grid, src, 1.0);
    rep.check(st.ok(), "synthetic: k = 1 stage (labels) ok");
    if (!st.ok())
        return;
    sl::LabelTransportOptions opt;
    opt.max_arclength = 2.0;
    opt.threads = 3;
    const Run r = run_tracing(ctx, grid, SyntheticField{}, st.labels, opt);
    std::printf("%s", r.res.table().c_str());
    const long long nq = static_cast<long long>(N) * (N / 4); // seeds per region per plane
    const long long nok = static_cast<long long>(N) * N - 2 * nq;
    bool hist_ok = true;
    for (const sl::LabelTransportPlane& P : r.res.planes) {
        if (P.j == 0)
            continue;
        hist_ok = hist_ok && P.status_hist[0] == nok && P.status_hist[1] == nq &&
                  P.status_hist[2] == nq && P.status_hist[3] == 0 && P.status_hist[4] == 0 &&
                  P.status_hist[5] == 0 && P.status_hist[6] == 0 && P.back_non_ok == 2 * nq &&
                  P.fwd_non_ok == 0 && P.non_ok == 2 * nq && P.flagged;
    }
    rep.check(r.res.status == sl::LabelTransportStatus::trace_fail, "synthetic: status trace_fail",
              sl::to_string(r.res.status));
    rep.check(hist_ok, "synthetic: per-plane histogram = (ok, seed_backflow, cap_exceeded) = "
                       "(N^2 - N^2/2, N^2/4, N^2/4), rest 0");
    rep.check(r.res.non_ok == static_cast<long long>(N) * 2 * nq &&
                  r.res.seed_backflow == static_cast<long long>(N) * nq,
              "synthetic: non_ok and seed_backflow totals",
              fmt("non_ok %.0f", static_cast<double>(r.res.non_ok)) +
                  fmt(" seed_backflow %.0f", static_cast<double>(r.res.seed_backflow)));
    bool nan_ok = true, fin_ok = true;
    for (int j = 0; j <= N; ++j)
        for (int m2 = 0; m2 < N; ++m2)
            for (int m3 = 0; m3 < N; ++m3) {
                const std::size_t i = grid.full_index(j, m2, m3);
                const bool failed = j >= 1 && m2 < N / 2;
                const bool nan_both = std::isnan(r.psi1[i]) && std::isnan(r.psi2[i]);
                if (failed)
                    nan_ok = nan_ok && nan_both;
                else
                    fin_ok = fin_ok && std::isfinite(r.psi1[i]) && std::isfinite(r.psi2[i]) &&
                             std::abs(r.psi1[i] - grid.coord(m2)) <= 1e-13 &&
                             std::abs(r.psi2[i] - grid.coord(m3)) <= 1e-13;
            }
    rep.check(nan_ok, "synthetic: NaN labels exactly at the failed seeds (psi1 and psi2)");
    rep.check(fin_ok, "synthetic: finite (affine) labels at every other vertex, plane 0 included");
}

// (iv) -------------------------------------------------------------------------------------------
void case_save(TestReport& rep, CudaContext& ctx) {
    const int N = 16;
    const sl::InletSlabGrid grid = sl::InletSlabGrid::make(N);
    const sl::SlabFieldSource src = sl::SlabFieldSource::analytic("generic3d", N);
    sl::ProductionStage st = sl::build_production_stage(ctx, grid, src, 0.25);
    rep.check(st.ok(), "save_labels: production stage ok");
    if (!st.ok())
        return;
    sl::LabelTransportOptions opt;
    opt.threads = 2;
    const Run r = run_stage(ctx, st, opt);
    nlohmann::json meta = sl::to_json(r.res);
    meta["N"] = N;
    meta["h"] = grid.h;
    const std::filesystem::path dir =
        std::filesystem::temp_directory_path() / "sf35_n1_save_labels_test";
    std::filesystem::remove_all(dir);
    sl::save_labels(dir.string(), grid, r.psi1, r.psi2, meta);
    const sl::NpyArray a1 = sl::read_npy((dir / "psi1.npy").string());
    const sl::NpyArray a2 = sl::read_npy((dir / "psi2.npy").string());
    const std::vector<std::size_t> shape = {static_cast<std::size_t>(N) + 1,
                                            static_cast<std::size_t>(N),
                                            static_cast<std::size_t>(N)};
    rep.check(a1.shape == shape && a2.shape == shape, "save_labels: npy shape (N+1, N, N)",
              a1.shape_string());
    rep.check(same_vec(a1.data, r.psi1) && same_vec(a2.data, r.psi2),
              "save_labels: npy values bitwise equal to the downloaded labels");
    bool keys = false;
    try {
        std::ifstream f((dir / "labels.json").string());
        const nlohmann::json j = nlohmann::json::parse(f);
        keys = j.contains("N") && j.contains("h") && j.contains("h_max") && j.contains("tol") &&
               j.contains("backend") && j.contains("status") && j.contains("shape") &&
               j.contains("layout") && j.contains("format") && j["N"] == N &&
               j["backend"] == "host" && j["status"] == "ok";
    } catch (const std::exception&) {
        keys = false;
    }
    rep.check(keys, "save_labels: labels.json parses with N, h, h_max, tol, backend, status, "
                    "shape, layout, format");
    std::filesystem::remove_all(dir);
}

} // namespace

int main() {
    TestReport rep;
    CudaContext ctx(0);
    const auto t0 = std::chrono::steady_clock::now();
    try {
        std::printf("=== SF-35 N1: (i) k = 1 ===\n");
        case_k1(rep, ctx);
        std::printf("=== SF-35 N1: (ii) generic3d eps = 0.25 ===\n");
        case_generic3d(rep, ctx);
        std::printf("=== SF-35 N1: (iii) failure accounting (synthetic field) ===\n");
        case_failure_accounting(rep, ctx);
        std::printf("=== SF-35 N1: (iv) save_labels ===\n");
        case_save(rep, ctx);
    } catch (const std::exception& e) {
        std::printf("[FAIL] unexpected exception: %s\n", e.what());
        rep.overall_pass = false;
    }
    const double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("\n=== inlet_slab_trace16: %d checks, %s (%.1f s) ===\n", rep.checks,
                rep.overall_pass ? "ALL PASS" : "FAILURES", t);
    return rep.overall_pass ? 0 : 1;
}
