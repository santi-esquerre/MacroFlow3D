/**
 * @file tracker_l6_tests.cu
 * @brief SF-40 L6: bitwise tests of the opt-in tracker speedups (TrackOptions::fused_splines,
 *        TrackOptions::single_launch).
 *
 *  1. FusedSplineLagrangianField::eval / eval_ext vs SplineLagrangianField::eval / eval_ext on a
 *     gaussian 32^3 production stage at >= 1e5 points (random; exactly on knots and cell faces;
 *     wrapped / negative / large coordinates such as x + 3, x - 2, x + 1000): memcmp of g, k,
 *     grad_lnk, H.
 *  2. Tracker outputs of {fused, single, both} vs the default variant (every per-particle array,
 *     PassStats incl. nfev, statuses, moments), with and without covectors, on gaussian eps 1 and
 *     eps 2 (16^3), plus one case with P = 0 (times only) and one with T = 0 (planes only).
 *  3. Timings/JSON contract (RR10): default track.json has the legacy key set; single_launch adds
 *     timings_s.single_kernel and nulls plane_kernel / time_kernel.
 *
 * Instantiates device templates: compiled with --fmad=false (CMakeLists.txt).
 * Prints `CHECK <name> PASS|FAIL`; exit 0 iff all pass.
 */

#include "src/core/DeviceBuffer.cuh"
#include "src/core/Scalar.hpp"
#include "src/external/nlohmann/json.hpp"
#include "src/physics/particles/lagrangian_labels/FusedSplineLagrangianField.cuh"
#include "src/physics/particles/lagrangian_labels/LagrangianTracker.cuh"
#include "src/physics/streamfunctions/inlet_slab/InletSlabGrid.cuh"
#include "src/physics/streamfunctions/inlet_slab/SlabProductionSetup.cuh"
#include "src/runtime/CudaContext.cuh"
#include "src/runtime/cuda_check.cuh"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <unistd.h>
#include <vector>

using namespace macroflow3d;
namespace sl = macroflow3d::streamfunctions::inlet_slab;
namespace ll = macroflow3d::lagrangian_labels;

namespace {

int g_failed = 0;
void check(const std::string& name, bool ok, const std::string& detail = "") {
    std::printf("CHECK %s %s%s%s\n", name.c_str(), ok ? "PASS" : "FAIL", detail.empty() ? "" : "  ",
                detail.c_str());
    std::fflush(stdout);
    if (!ok)
        ++g_failed;
}

// ------------------------------------------------------------------------------------------------
// 1. Field evaluation
// ------------------------------------------------------------------------------------------------
/// out_a / out_b: per point eval (g3, k) then eval_ext (g3, k, gl3, H6) = 4 + 13 doubles.
constexpr int kPerPoint = 4 + 13;

template <class Field>
__global__ void eval_kernel(const Field f, const double* pts, int n, double* out) {
    const int p = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (p >= n)
        return;
    const double x[3] = {pts[3 * p], pts[3 * p + 1], pts[3 * p + 2]};
    double* o = out + static_cast<std::size_t>(p) * kPerPoint;
    double g[3], k;
    f.eval(x, g, k);
    o[0] = g[0];
    o[1] = g[1];
    o[2] = g[2];
    o[3] = k;
    double g2[3], k2, gl[3], H[6];
    f.eval_ext(x, g2, k2, gl, H);
    o[4] = g2[0];
    o[5] = g2[1];
    o[6] = g2[2];
    o[7] = k2;
    o[8] = gl[0];
    o[9] = gl[1];
    o[10] = gl[2];
    for (int i = 0; i < 6; ++i)
        o[11 + i] = H[i];
}

void test_fused_field(CudaContext& ctx) {
    const int N = 32;
    const sl::InletSlabGrid grid = sl::InletSlabGrid::make(N);
    sl::ProductionFieldSpec spec;
    spec.N = N;
    spec.sigma2 = 1.0;
    spec.ell = 0.25;
    spec.seed = 3001ULL;
    const sl::SlabFieldSource src = sl::SlabFieldSource::gaussian(ctx, spec);
    sl::ProductionStageOptions so;
    so.require_inlet_labels = false;
    sl::ProductionStage st = sl::build_production_stage(ctx, grid, src, 1.0, so);
    check("field_stage_ready", st.flow_ready());
    if (!st.flow_ready())
        return;

    const std::size_t nc = st.potential_coefficients.size();
    DeviceBuffer<real> d_pot(nc), d_logk(nc);
    DeviceBuffer<double2> d_fused(nc);
    const std::vector<double2> inter = ll::interleave_coefficients(
        st.potential_coefficients.data(), st.logk_coefficients.data(), nc);
    MACROFLOW3D_CUDA_CHECK(cudaMemcpy(d_pot.data(), st.potential_coefficients.data(),
                                      nc * sizeof(real), cudaMemcpyHostToDevice));
    MACROFLOW3D_CUDA_CHECK(cudaMemcpy(d_logk.data(), st.logk_coefficients.data(),
                                      nc * sizeof(real), cudaMemcpyHostToDevice));
    MACROFLOW3D_CUDA_CHECK(
        cudaMemcpy(d_fused.data(), inter.data(), nc * sizeof(double2), cudaMemcpyHostToDevice));
    MACROFLOW3D_CUDA_CHECK(cudaDeviceSynchronize());

    const sl::SlabSplineDirectionField hf = st.direction_field();
    ll::SplineLagrangianField f;
    f.potential = hf.potential;
    f.potential.coeff = d_pot.data();
    f.log_conductivity = hf.log_conductivity;
    f.log_conductivity.coeff = d_logk.data();
    for (int d = 0; d < 3; ++d)
        f.G[d] = hf.G[d];
    const ll::FusedSplineLagrangianField ff = ll::make_fused_field(f, d_fused.data());

    // points
    std::mt19937_64 rng(20261022ULL);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    const double h = 1.0 / N;
    std::vector<double> pts;
    auto add = [&](double a, double b, double c) {
        pts.push_back(a);
        pts.push_back(b);
        pts.push_back(c);
    };
    for (int i = 0; i < 60000; ++i) // random in [-2, 3)^3
        add(-2.0 + 5.0 * U(rng), -2.0 + 5.0 * U(rng), -2.0 + 5.0 * U(rng));
    const double shifts[] = {0.0, 3.0, -2.0, 1.0, -1.0, 1000.0};
    for (int i = 0; i < 40000; ++i) { // exactly on knots (cell centres) / cell faces, shifted
        const double off = (i % 2 == 0) ? 0.5 : 0.0;
        double c[3];
        for (int d = 0; d < 3; ++d) {
            const int cell = static_cast<int>(rng() % N);
            c[d] = (cell + off) * h + shifts[rng() % 6];
        }
        add(c[0], c[1], c[2]);
    }
    const int n = static_cast<int>(pts.size() / 3);

    DeviceBuffer<double> d_pts(pts.size());
    DeviceBuffer<double> d_a(static_cast<std::size_t>(n) * kPerPoint),
        d_b(static_cast<std::size_t>(n) * kPerPoint);
    MACROFLOW3D_CUDA_CHECK(cudaMemcpy(d_pts.data(), pts.data(), pts.size() * sizeof(double),
                                      cudaMemcpyHostToDevice));
    MACROFLOW3D_CUDA_CHECK(cudaDeviceSynchronize());
    const int blk = 64, grd = (n + blk - 1) / blk;
    eval_kernel<ll::SplineLagrangianField><<<grd, blk>>>(f, d_pts.data(), n, d_a.data());
    eval_kernel<ll::FusedSplineLagrangianField><<<grd, blk>>>(ff, d_pts.data(), n, d_b.data());
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
    MACROFLOW3D_CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<double> ha(d_a.size()), hb(d_b.size());
    MACROFLOW3D_CUDA_CHECK(cudaMemcpy(ha.data(), d_a.data(), ha.size() * sizeof(double),
                                      cudaMemcpyDeviceToHost));
    MACROFLOW3D_CUDA_CHECK(cudaMemcpy(hb.data(), d_b.data(), hb.size() * sizeof(double),
                                      cudaMemcpyDeviceToHost));
    long long bad_eval = 0, bad_ext = 0, nonfinite = 0;
    for (int p = 0; p < n; ++p) {
        const double* a = ha.data() + static_cast<std::size_t>(p) * kPerPoint;
        const double* b = hb.data() + static_cast<std::size_t>(p) * kPerPoint;
        if (std::memcmp(a, b, 4 * sizeof(double)) != 0)
            ++bad_eval;
        if (std::memcmp(a + 4, b + 4, 13 * sizeof(double)) != 0)
            ++bad_ext;
        for (int i = 0; i < kPerPoint; ++i)
            if (!std::isfinite(a[i]))
                ++nonfinite;
    }
    const std::string np = std::to_string(n) + " points";
    check("fused_eval_bitwise", bad_eval == 0, "differing " + std::to_string(bad_eval) + " of " + np);
    check("fused_eval_ext_bitwise", bad_ext == 0,
          "differing " + std::to_string(bad_ext) + " of " + np);
    check("fused_eval_points_finite", nonfinite == 0, "nonfinite " + std::to_string(nonfinite));
}

// ------------------------------------------------------------------------------------------------
// 2. Tracker variants
// ------------------------------------------------------------------------------------------------
bool same_bits(double a, double b) {
    return std::memcmp(&a, &b, sizeof(double)) == 0;
}
bool same_vec(const std::vector<double>& a, const std::vector<double>& b) {
    return a.size() == b.size() &&
           (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(double)) == 0);
}
bool same_stats(const std::vector<ll::PassStats>& a, const std::vector<ll::PassStats>& b) {
    return a.size() == b.size() &&
           (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(ll::PassStats)) == 0);
}

bool same_all(const ll::LagrangianTrackResult& a, const ll::LagrangianTrackResult& b) {
    bool ok = a.Np == b.Np && a.P == b.P && a.T == b.T && same_vec(a.seeds, b.seeds) &&
              same_vec(a.weights, b.weights) && same_vec(a.labels, b.labels) &&
              same_vec(a.planes_x23, b.planes_x23) && same_vec(a.planes_tau, b.planes_tau) &&
              same_vec(a.planes_I, b.planes_I) && same_vec(a.planes_cov, b.planes_cov) &&
              same_vec(a.times_x, b.times_x) && same_vec(a.times_I, b.times_I) &&
              same_vec(a.times_cov, b.times_cov) && a.status_plane == b.status_plane &&
              a.status_time == b.status_time && same_stats(a.stats_plane, b.stats_plane) &&
              same_stats(a.stats_time, b.stats_time) && a.seed_backflow == b.seed_backflow &&
              a.moments_time.size() == b.moments_time.size() &&
              a.moments_planes.size() == b.moments_planes.size();
    for (std::size_t n = 0; ok && n < a.moments_time.size(); ++n) {
        const auto &p = a.moments_time[n], &q = b.moments_time[n];
        ok = same_bits(p.time, q.time) && p.active == q.active;
        for (int k = 0; k < 3; ++k)
            ok = ok && same_bits(p.mean[k], q.mean[k]) && same_bits(p.var[k], q.var[k]);
    }
    for (std::size_t m = 0; ok && m < a.moments_planes.size(); ++m) {
        const auto &p = a.moments_planes[m], &q = b.moments_planes[m];
        ok = same_bits(p.x1, q.x1) && same_bits(p.mean_x2, q.mean_x2) &&
             same_bits(p.var_x2, q.var_x2) && same_bits(p.mean_x3, q.mean_x3) &&
             same_bits(p.var_x3, q.var_x3) && same_bits(p.mean_tau, q.mean_tau) &&
             same_bits(p.var_tau, q.var_tau) && same_bits(p.rms_ev, q.rms_ev) &&
             same_bits(p.max_ev, q.max_ev) && p.contributing == q.contributing &&
             p.non_ok == q.non_ok;
    }
    return ok;
}

ll::LagrangianTrackResult run(CudaContext& ctx, const sl::ProductionStage& st,
                              const ll::InjectionSpec& inj, ll::TrackOptions to, bool fused,
                              bool single) {
    to.fused_splines = fused;
    to.single_launch = single;
    ll::LagrangianTracker tr;
    tr.prepare(ctx, st, inj, to);
    return tr.track(ctx); // copy
}

nlohmann::json read_json(const std::filesystem::path& p) {
    std::ifstream f(p);
    return nlohmann::json::parse(f);
}

void test_tracker_case(CudaContext& ctx, const sl::ProductionStage& st, const sl::InletSlabGrid& grid,
                       const std::string& tag, const std::vector<double>& planes,
                       const std::vector<double>& times, bool cov, bool json_checks) {
    ll::InjectionSpec inj;
    inj.X0 = 0.0;
    inj.M = 8;
    ll::TrackOptions to;
    to.tol = 1e-8;
    to.h_max = grid.h / 8.0;
    to.planes = planes;
    to.times = times;
    to.covectors = cov;

    const ll::LagrangianTrackResult D = run(ctx, st, inj, to, false, false);
    check(tag + "_default_sane",
          D.Np == 64 && (planes.empty() || D.totals_plane.status_hist[0] > 0) &&
              (times.empty() || D.totals_time.status_hist[0] > 0),
          "ok plane " + std::to_string(D.totals_plane.status_hist[0]) + " time " +
              std::to_string(D.totals_time.status_hist[0]) + " of " + std::to_string(D.Np));
    const struct {
        const char* name;
        bool fused, single;
    } variants[] = {{"fused", true, false}, {"single", false, true}, {"both", true, true}};
    for (const auto& v : variants) {
        const ll::LagrangianTrackResult V = run(ctx, st, inj, to, v.fused, v.single);
        check(tag + "_" + v.name + "_bitwise", same_all(D, V));
        if (v.single) {
            check(tag + "_" + v.name + "_timings",
                  V.single_launch && std::isnan(V.seconds_plane_kernel) &&
                      std::isnan(V.seconds_time_kernel) && V.seconds_single_kernel >= 0.0);
        } else {
            check(tag + "_" + v.name + "_timings",
                  !V.single_launch && std::isfinite(V.seconds_plane_kernel) &&
                      std::isfinite(V.seconds_time_kernel) && V.seconds_single_kernel == 0.0);
        }
    }
    if (!json_checks)
        return;
    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() / ("sf40_l6_json_" + std::to_string(::getpid()));
    ll::TrackMeta meta;
    meta.field = "gaussian";
    meta.eps = 1.0;
    meta.N = grid.n;
    meta.h = grid.h;
    ll::write_track_outputs((dir / "default").string(), D, meta);
    const ll::LagrangianTrackResult S = run(ctx, st, inj, to, true, true);
    ll::write_track_outputs((dir / "single").string(), S, meta);
    const nlohmann::json jd = read_json(dir / "default" / "track.json"),
                         js = read_json(dir / "single" / "track.json");
    std::vector<std::string> kd, ks;
    for (auto it = jd["timings_s"].begin(); it != jd["timings_s"].end(); ++it)
        kd.push_back(it.key());
    for (auto it = js["timings_s"].begin(); it != js["timings_s"].end(); ++it)
        ks.push_back(it.key());
    const std::vector<std::string> legacy = {"download", "moments", "plane_kernel", "time_kernel"};
    const std::vector<std::string> with_single = {"download", "moments", "plane_kernel",
                                                  "single_kernel", "time_kernel"};
    check("json_default_timings_keys_unchanged", kd == legacy);
    check("json_single_timings_keys", ks == with_single &&
                                          js["timings_s"]["plane_kernel"].is_null() &&
                                          js["timings_s"]["time_kernel"].is_null() &&
                                          js["timings_s"]["single_kernel"].is_number());
    // all other keys identical
    nlohmann::json a = jd, b = js;
    a.erase("timings_s");
    b.erase("timings_s");
    check("json_other_keys_identical", a == b);
    std::error_code ec;
    fs::remove_all(dir, ec);
}

} // namespace

int main() {
    CudaContext ctx(0);
    test_fused_field(ctx);

    const int N = 16;
    const sl::InletSlabGrid grid = sl::InletSlabGrid::make(N);
    sl::ProductionFieldSpec spec;
    spec.N = N;
    spec.sigma2 = 1.0;
    spec.ell = 0.25;
    spec.seed = 3001ULL;
    const sl::SlabFieldSource src = sl::SlabFieldSource::gaussian(ctx, spec);
    for (const double eps : {1.0, 2.0}) {
        sl::ProductionStageOptions so;
        so.require_inlet_labels = false;
        sl::ProductionStage st = sl::build_production_stage(ctx, grid, src, eps, so);
        const std::string e = eps == 1.0 ? "eps1" : "eps2";
        check(e + "_stage_ready", st.flow_ready());
        if (!st.flow_ready())
            continue;
        const std::vector<double> planes = {0.25, 0.5, 1.0, 2.0}, times = {0.25, 0.7, 1.5};
        test_tracker_case(ctx, st, grid, e + "_nocov", planes, times, false, eps == 1.0);
        test_tracker_case(ctx, st, grid, e + "_cov", planes, times, true, false);
        if (eps == 1.0) {
            test_tracker_case(ctx, st, grid, e + "_cov_P0", {}, times, true, false);
            test_tracker_case(ctx, st, grid, e + "_nocov_T0", planes, {}, false, false);
            test_tracker_case(ctx, st, grid, e + "_cov_T0", planes, {}, true, false);
        }
    }
    std::printf("SUMMARY: %d failed\n", g_failed);
    return g_failed == 0 ? 0 : 1;
}
