/**
 * @file long_box_tests.cu
 * @brief SF-40 L3: LongBoxStage + the tracker's device-coefficient API.
 *
 *  1. box_auto_mg_levels == slab_auto_mg_levels(N) for every even N in [4, 1024]; 3328x384x384 -> 7
 *     levels with coarsest 52x6x6.
 *  2. Cube equivalence (BITWISE): build_production_stage (require_inlet_labels = false, as the
 *     driver) + LagrangianTracker::prepare(ProductionStage) versus build_long_box_stage +
 *     LagrangianTracker::prepare(DeviceSplineFlow) on gaussian cubes (every per-particle array,
 *     PassStats, statuses, weights, labels, moments; G; SF-18 / SF-19 report fields; coefficient
 *     arrays; inlet samples; D-1 labels; faces; the SF-18 / SF-19 / corrector lines of summary()),
 *     with and without covectors and for {default, fused_splines, single_launch, both}; plus the
 *     darcy_failed status path.
 *  3. Memory records (C-MEMPHASE): exact_bytes == cufft_work + sum(items), item names and size
 *     rules, S2 == the S+ closed form computed independently, S2 largest, categories sum, the
 *     predicted per-phase bytes (the --cufft-size model) equal the recorded ones.
 *  4. F1-F4 (RR6 of the pre-registration) on non-cubic boxes: fast subset by default, the stated
 *     sizes with --full (V100 part `controls`).
 *
 * Usage: long_box_tests [--full] [--only levels|cube|mem|F1|F2|F3|F4] [--cufft-size nx,ny,nz]
 * `--cufft-size` only creates cuFFT plans without work-area allocation and prints the predicted
 * exact bytes of F, S1, S2, S3 at that geometry (used by the campaign at 3328,384,384).
 * Prints `CHECK <name> PASS|FAIL` and `ALL_PASS` / `FAILED n`; exit 0 iff all pass.
 * Compiled with --fmad=false (CMakeLists.txt) like the other lagrangian tests.
 */

#include "src/core/DeviceBuffer.cuh"
#include "src/core/Scalar.hpp"
#include "src/numerics/blas/sum.cuh"
#include "src/numerics/interpolation/PeriodicTricubicBSpline.cuh"
#include "src/physics/particles/lagrangian_labels/LagrangianTracker.cuh"
#include "src/physics/particles/lagrangian_labels/LongBoxStage.cuh"
#include "src/physics/streamfunctions/inlet_slab/InletSlabGrid.cuh"
#include "src/physics/streamfunctions/inlet_slab/SlabProductionSetup.cuh"
#include "src/runtime/CudaContext.cuh"
#include "src/runtime/cuda_check.cuh"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <cufft.h>

using namespace macroflow3d;
namespace sl = macroflow3d::streamfunctions::inlet_slab;
namespace ll = macroflow3d::lagrangian_labels;

namespace {

constexpr double kTwoPi = 6.283185307179586476925286766559;

int g_failed = 0;
bool g_full = false;
std::set<std::string> g_only;

bool run_case(const std::string& c) {
    return g_only.empty() || g_only.count(c) > 0;
}

void check(const std::string& name, bool ok, const std::string& detail = "") {
    std::printf("CHECK %s %s%s%s\n", name.c_str(), ok ? "PASS" : "FAIL", detail.empty() ? "" : "  ",
                detail.c_str());
    std::fflush(stdout);
    if (!ok)
        ++g_failed;
}

std::string fmt(const char* f, double v) {
    char b[128];
    std::snprintf(b, sizeof(b), f, v);
    return b;
}

double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

// ------------------------------------------------------------------------------------------------
// Bitwise comparison helpers
// ------------------------------------------------------------------------------------------------
bool same_bits(double a, double b) {
    return std::memcmp(&a, &b, sizeof(double)) == 0;
}
template <class T> bool same_vec(const std::vector<T>& a, const std::vector<T>& b) {
    return a.size() == b.size() &&
           (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0);
}
template <class T, std::size_t K> bool same_arr(const T (&a)[K], const T (&b)[K]) {
    return std::memcmp(a, b, sizeof(a)) == 0;
}

std::vector<real> download(const real* d, std::size_t n) {
    std::vector<real> h(n);
    MACROFLOW3D_CUDA_CHECK(cudaMemcpy(h.data(), d, n * sizeof(real), cudaMemcpyDeviceToHost));
    return h;
}

bool same_pcg(const solvers::ProjectedPCGResult& a, const solvers::ProjectedPCGResult& b) {
    return a.iterations == b.iterations && same_bits(a.raw_rhs_mean, b.raw_rhs_mean) &&
           same_bits(a.raw_rhs_l2_norm, b.raw_rhs_l2_norm) &&
           same_bits(a.raw_rhs_compatibility_defect, b.raw_rhs_compatibility_defect) &&
           same_bits(a.initial_projected_residual, b.initial_projected_residual) &&
           same_bits(a.final_projected_residual, b.final_projected_residual) &&
           same_bits(a.relative_projected_residual, b.relative_projected_residual) &&
           same_bits(a.final_field_mean, b.final_field_mean) && a.converged == b.converged &&
           a.status == b.status;
}

bool same_darcy(const physics::AffinePeriodicFlowReport& a,
                const physics::AffinePeriodicFlowReport& b) {
    bool ok = same_arr(a.K_eff[0], b.K_eff[0]) && same_arr(a.K_eff[1], b.K_eff[1]) &&
              same_arr(a.K_eff[2], b.K_eff[2]) &&
              same_bits(a.symmetry_defect_rel, b.symmetry_defect_rel) &&
              same_arr(a.eigenvalues_symmetric_part, b.eigenvalues_symmetric_part) &&
              same_arr(a.G, b.G) && same_arr(a.achieved_mean_flux, b.achieved_mean_flux) &&
              same_bits(a.div_max_abs, b.div_max_abs) && same_bits(a.div_rms, b.div_rms);
    for (int d = 0; d < 3; ++d)
        ok = ok && same_pcg(a.corrector_results[d], b.corrector_results[d]);
    return ok;
}

bool same_sf18(const physics::PeriodicGaussianFieldReport& a,
               const physics::PeriodicGaussianFieldReport& b) {
    return same_bits(a.raw_mean, b.raw_mean) && same_bits(a.raw_variance, b.raw_variance) &&
           same_bits(a.applied_scale, b.applied_scale) &&
           same_bits(a.final_variance, b.final_variance) &&
           a.active_mode_count == b.active_mode_count && a.spectrum_bytes == b.spectrum_bytes &&
           a.cufft_work_area_bytes == b.cufft_work_area_bytes &&
           a.reduction_scratch_bytes == b.reduction_scratch_bytes &&
           a.total_device_bytes == b.total_device_bytes;
}

bool same_labels(const sl::InletLabels& a, const sl::InletLabels& b) {
    return a.nf() == b.nf() && same_bits(a.Q0(), b.Q0()) && same_bits(a.vmin(), b.vmin()) &&
           same_bits(a.sample_mean(), b.sample_mean()) && same_vec(a.A(), b.A()) &&
           same_vec(a.Qh(), b.Qh()) && same_vec(a.B(), b.B()) &&
           same_vec(a.V_coefficients(), b.V_coefficients()) &&
           same_vec(a.mode_indices(), b.mode_indices());
}

bool same_stats(const std::vector<ll::PassStats>& a, const std::vector<ll::PassStats>& b) {
    return a.size() == b.size() &&
           (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(ll::PassStats)) == 0);
}

bool same_track(const ll::LagrangianTrackResult& a, const ll::LagrangianTrackResult& b) {
    bool ok = a.Np == b.Np && a.P == b.P && a.T == b.T && same_vec(a.seeds, b.seeds) &&
              same_vec(a.weights, b.weights) && same_vec(a.labels, b.labels) &&
              same_vec(a.planes_x23, b.planes_x23) && same_vec(a.planes_tau, b.planes_tau) &&
              same_vec(a.planes_I, b.planes_I) && same_vec(a.planes_cov, b.planes_cov) &&
              same_vec(a.times_x, b.times_x) && same_vec(a.times_I, b.times_I) &&
              same_vec(a.times_cov, b.times_cov) && a.status_plane == b.status_plane &&
              a.status_time == b.status_time && same_stats(a.stats_plane, b.stats_plane) &&
              same_stats(a.stats_time, b.stats_time) && a.seed_backflow == b.seed_backflow &&
              a.single_launch == b.single_launch &&
              a.moments_time.size() == b.moments_time.size() &&
              a.moments_planes.size() == b.moments_planes.size() &&
              a.totals_plane.status_hist == b.totals_plane.status_hist &&
              a.totals_time.status_hist == b.totals_time.status_hist &&
              a.totals_plane.nfev == b.totals_plane.nfev &&
              a.totals_time.nfev == b.totals_time.nfev;
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

/// The SF-18 / SF-19 / corrector lines of ProductionStageReport::summary() (the driver's printed
/// lines; the STAGE, inlet and v1 lines are excluded by the specification).
std::vector<std::string> summary_lines(const sl::ProductionStageReport& r) {
    std::vector<std::string> out;
    std::istringstream is(r.summary());
    std::string line;
    while (std::getline(is, line)) {
        if (line.rfind("  SF-18:", 0) == 0 || line.rfind("  SF-19:", 0) == 0 ||
            line.rfind("  SF-19 corrector", 0) == 0)
            out.push_back(line);
    }
    return out;
}

// ------------------------------------------------------------------------------------------------
// 1. MG levels
// ------------------------------------------------------------------------------------------------
void test_levels() {
    int bad = 0, first_bad = 0;
    for (int N = 4; N <= 1024; N += 2) {
        if (ll::box_auto_mg_levels(N, N, N) != sl::slab_auto_mg_levels(N)) {
            if (bad == 0)
                first_bad = N;
            ++bad;
        }
    }
    check("box_levels_equal_cube_levels", bad == 0,
          bad ? "first mismatch N=" + std::to_string(first_bad) : "N = 4..1024 even");
    check("box_levels_3328x384x384_is_7", ll::box_auto_mg_levels(3328, 384, 384) == 7);
    bool coarsest = true;
    int a = 3328, b = 384, c = 384;
    for (int l = 1; l < 7; ++l) {
        a /= 2;
        b /= 2;
        c /= 2;
    }
    coarsest = a == 52 && b == 6 && c == 6;
    check("box_levels_3328x384x384_coarsest_52x6x6", coarsest);
    // the minimum extent governs: a long box coarsens only as far as its shortest side allows
    check("box_levels_min_extent_governs",
          ll::box_auto_mg_levels(64, 16, 16) == sl::slab_auto_mg_levels(16) &&
              ll::box_auto_mg_levels(128, 32, 32) == sl::slab_auto_mg_levels(32));
}

// ------------------------------------------------------------------------------------------------
// 2. Cube equivalence
// ------------------------------------------------------------------------------------------------
ll::LagrangianTrackResult run_ref(CudaContext& ctx, const sl::ProductionStage& st,
                                  const ll::InjectionSpec& inj, const ll::TrackOptions& to) {
    ll::LagrangianTracker tr;
    tr.prepare(ctx, st, inj, to);
    return tr.track(ctx); // copy
}

ll::LagrangianTrackResult run_lb(CudaContext& ctx, const ll::LongBoxStage& S,
                                 const ll::InjectionSpec& inj, const ll::TrackOptions& to) {
    ll::LagrangianTracker tr;
    tr.prepare(ctx, S.device_flow(), inj, to);
    return tr.track(ctx); // copy
}

void test_cube_equivalence(CudaContext& ctx, int N, double eps, bool all_variants,
                           unsigned long long seed = 3101ULL, bool expect_backflow = false) {
    const std::string tag = std::string(expect_backflow ? "backflow_" : "") + "cube" +
                            std::to_string(N) + "_eps" + fmt("%g", eps) +
                            (seed == 3101ULL ? "" : "_seed" + std::to_string(seed));
    const double t0 = now_s();
    sl::ProductionFieldSpec ps;
    ps.N = N;
    ps.sigma2 = 1.0;
    ps.ell = 0.25;
    ps.seed = seed;
    const sl::SlabFieldSource src = sl::SlabFieldSource::gaussian(ctx, ps);
    const sl::InletSlabGrid grid = sl::InletSlabGrid::make(N);
    sl::ProductionStageOptions so;
    so.require_inlet_labels = false; // as the driver
    so.keep_face_fluxes = true;
    const sl::ProductionStage ref = sl::build_production_stage(ctx, grid, src, eps, so);

    ll::LongBoxSpec ls;
    ls.nx = ls.ny = ls.nz = N;
    ls.sigma2 = ps.sigma2;
    ls.ell = ps.ell;
    ls.seed = seed;
    ls.eps = eps;
    ls.keep_face_fluxes = true;
    const ll::LongBoxStage lb = ll::build_long_box_stage(ctx, ls);
    // the interleaved-coefficient stage (fused_splines) is built only in the full variant set
    ll::LongBoxSpec lsf = ls;
    lsf.fused_coefficients = true;
    std::unique_ptr<ll::LongBoxStage> lbf_p;
    if (all_variants)
        lbf_p = std::make_unique<ll::LongBoxStage>(ll::build_long_box_stage(ctx, lsf));

    check(tag + "_stage_ready",
          ref.flow_ready() && lb.flow_ready() && (!lbf_p || lbf_p->flow_ready()));
    if (!(ref.flow_ready() && lb.flow_ready() && (!lbf_p || lbf_p->flow_ready())))
        return;
    const ll::LongBoxStage& lbf = lbf_p ? *lbf_p : lb; // reduced tier: lb stands in for lbf
    const sl::ProductionStageReport &rr = ref.report, &lr = lb.report;
    check(tag + "_report_status_fields",
          rr.status == lr.status && rr.field == lr.field && same_bits(rr.eps, lr.eps) &&
              rr.N == lr.N && rr.mg_levels == lr.mg_levels && rr.has_sf18 == lr.has_sf18 &&
              rr.darcy_converged == lr.darcy_converged,
          std::string("status ") + sl::to_string(lr.status));
    if (expect_backflow) {
        check(tag + "_inlet_backflow_recorded_not_fatal",
              rr.status == sl::SlabProductionStatus::inlet_backflow &&
                  lr.status == sl::SlabProductionStatus::inlet_backflow && !ref.labels_available &&
                  !lb.labels_available && !lbf.labels_available && std::isnan(lr.Q0) &&
                  lr.inlet_vmin < 0.0 && lb.flow_ready(),
              fmt("inlet_vmin %.3e", lr.inlet_vmin));
    }
    check(tag + "_sf18_report_bitwise", same_sf18(rr.sf18, lr.sf18));
    check(tag + "_sf19_report_bitwise", same_darcy(rr.darcy, lr.darcy));
    check(tag + "_inlet_vmin_mean_Q0_bitwise",
          same_bits(rr.inlet_vmin, lr.inlet_vmin) && same_bits(rr.inlet_mean, lr.inlet_mean) &&
              same_bits(rr.Q0, lr.Q0));
    check(tag + "_v1_diag_fields_remain_nan",
          std::isnan(lr.v1_diff_rms_rel) && std::isnan(lr.v1_diff_max_rel) &&
              std::isnan(lr.v_rms) && std::isnan(lr.spline_potential_gpu_vs_host_rel));
    check(tag + "_summary_sf18_sf19_corrector_lines_identical",
          summary_lines(rr) == summary_lines(lr) && summary_lines(lr).size() == 5,
          std::to_string(summary_lines(lr).size()) + " lines");
    check(tag + "_G_bitwise", same_arr(ref.G, lb.G) && same_arr(ref.G, lbf.G));
    check(tag + "_inlet_samples_bitwise", same_vec(ref.inlet_v1_samples, lb.inlet_v1_samples));
    {
        const bool avail = ref.labels_available == lb.labels_available &&
                           ref.labels_available == lbf.labels_available;
        const bool same = !ref.labels_available ||
                          (same_labels(ref.labels, lb.labels) && same_labels(ref.labels, lbf.labels) &&
                           lb.labels.device_prepared() && lbf.labels.device_prepared());
        check(tag + "_labels_bitwise", avail && same,
              ref.labels_available ? "labels available" : "inlet backflow: labels unavailable");
    }
    check(tag + "_faces_bitwise",
          lb.faces_available() && same_vec(ref.face_flux_u, lb.face_flux_u) &&
              same_vec(ref.face_flux_v, lb.face_flux_v) && same_vec(ref.face_flux_w, lb.face_flux_w));
    const std::size_t n = static_cast<std::size_t>(N) * N * N;
    {
        const std::vector<real> hp = download(lb.coef_pot.data(), n);
        const std::vector<real> hl = download(lb.coef_logk.data(), n);
        check(tag + "_coefficients_bitwise", same_vec(hp, ref.potential_coefficients) &&
                                                 same_vec(hl, ref.logk_coefficients));
        if (lbf_p) {
            std::vector<double2> hi(n);
            MACROFLOW3D_CUDA_CHECK(cudaMemcpy(hi.data(), lbf.coef_interleaved.data(),
                                              n * sizeof(double2), cudaMemcpyDeviceToHost));
            bool ok = lbf.coef_pot.size() == 0 && lbf.coef_logk.size() == 0;
            for (std::size_t i = 0; ok && i < n; ++i)
                ok = same_bits(hi[i].x, ref.potential_coefficients[i]) &&
                     same_bits(hi[i].y, ref.logk_coefficients[i]);
            check(tag + "_interleaved_coefficients_bitwise", ok);
        }
    }

    // ---- tracker: same options through both APIs
    ll::InjectionSpec inj;
    inj.X0 = 0.0;
    inj.M = all_variants ? 8 : 6; // M x M seeds
    ll::TrackOptions to;
    to.tol = 1e-8;
    to.h_max = grid.h / 8.0;
    // planes up to 2 only where cheap (16^3, eps 1); shorter paths elsewhere keep the tier < 60 s
    to.planes = (N == 16 && eps == 1.0 && !expect_backflow) ? std::vector<double>{0.25, 0.5, 1.0, 2.0}
                                                             : std::vector<double>{0.25, 0.5, 1.0};
    to.times = {0.5, 1.0};
    struct V {
        const char* name;
        bool fused, single;
    };
    const std::vector<V> variants = {{"default", false, false},
                                     {"fused", true, false},
                                     {"single", false, true},
                                     {"both", true, true}};
    for (const bool cov : {false, true}) {
        to.covectors = cov;
        for (const V& v : variants) {
            // reduced tier: (no covectors, default) and (covectors, single launch) only
            if (!all_variants && !((!cov && !v.fused && !v.single) || (cov && !v.fused && v.single)))
                continue;
            to.fused_splines = v.fused;
            to.single_launch = v.single;
            const ll::LagrangianTrackResult A = run_ref(ctx, ref, inj, to);
            const ll::LagrangianTrackResult B = run_lb(ctx, v.fused ? lbf : lb, inj, to);
            const std::string nm = tag + (cov ? "_cov_" : "_nocov_") + v.name;
            check(nm + "_tracker_bitwise", same_track(A, B) && B.totals_plane.status_hist[0] > 0,
                  "ok plane " + std::to_string(B.totals_plane.status_hist[0]) + "/" +
                      std::to_string(B.Np) + " time " +
                      std::to_string(B.totals_time.status_hist[0]));
        }
    }
    std::printf("  [%s %.1f s]\n", tag.c_str(), now_s() - t0);
}

void test_api_errors(CudaContext& ctx) {
    ll::LongBoxSpec ls;
    ls.nx = ls.ny = ls.nz = 16;
    ls.sigma2 = 1.0;
    ls.ell = 0.25;
    ls.seed = 3101ULL;
    const ll::LongBoxStage lb = ll::build_long_box_stage(ctx, ls);
    ll::InjectionSpec inj;
    inj.M = 4;
    ll::TrackOptions to;
    to.h_max = lb.grid.dx / 8.0;
    to.planes = {0.5};
    to.fused_splines = true; // the stage holds separate arrays, not the interleaved one
    ll::LagrangianTracker tr;
    bool threw = false;
    try {
        tr.prepare(ctx, lb.device_flow(), inj, to);
    } catch (const std::logic_error&) {
        threw = true;
    }
    check("tracker_api_throws_when_arrays_missing", threw);
    ll::LongBoxSpec bad = ls;
    bad.nz = 12;
    bool inv = false;
    try {
        (void)ll::build_long_box_stage(ctx, bad);
    } catch (const std::invalid_argument&) {
        inv = true;
    }
    check("long_box_rejects_ny_ne_nz", inv);
    bad = ls;
    bad.nx = 15;
    inv = false;
    try {
        (void)ll::build_long_box_stage(ctx, bad);
    } catch (const std::invalid_argument&) {
        inv = true;
    }
    check("long_box_rejects_odd_nx", inv);
    // an invalid MG hierarchy inside S2 is rethrown as runtime_error naming the phase
    bad = ls;
    bad.mg_levels = 9;
    bool named = false;
    try {
        (void)ll::build_long_box_stage(ctx, bad);
    } catch (const std::runtime_error& e) {
        named = std::string(e.what()).find("phase=S2") != std::string::npos;
    }
    check("long_box_phase_named_in_exception", named);
}

void test_darcy_failed(CudaContext& ctx) {
    const int N = 16;
    sl::ProductionFieldSpec ps;
    ps.N = N;
    ps.sigma2 = 1.0;
    ps.ell = 0.25;
    ps.seed = 3101ULL;
    const sl::SlabFieldSource src = sl::SlabFieldSource::gaussian(ctx, ps);
    sl::ProductionStageOptions so;
    so.require_inlet_labels = false;
    so.pcg_max_iter = 1; // cannot converge
    const sl::ProductionStage ref =
        sl::build_production_stage(ctx, sl::InletSlabGrid::make(N), src, 1.0, so);
    ll::LongBoxSpec ls;
    ls.nx = ls.ny = ls.nz = N;
    ls.sigma2 = 1.0;
    ls.ell = 0.25;
    ls.seed = 3101ULL;
    ls.pcg_max_iter = 1;
    const ll::LongBoxStage lb = ll::build_long_box_stage(ctx, ls);
    check("darcy_failed_status_and_report",
          ref.report.status == sl::SlabProductionStatus::darcy_failed &&
              lb.report.status == ref.report.status && !lb.flow_ready() &&
              !lb.report.darcy_converged && lb.coef_pot.size() == 0 && lb.coef_logk.size() == 0 &&
              same_darcy(ref.report.darcy, lb.report.darcy) &&
              summary_lines(ref.report) == summary_lines(lb.report));
}

// ------------------------------------------------------------------------------------------------
// 3. Memory records
// ------------------------------------------------------------------------------------------------
std::size_t size_of_labels_device(int nf, int chunk) {
    const std::size_t M = static_cast<std::size_t>(nf) - 1; // even nf: Nyquist dropped
    const std::size_t cplx = 16;
    return 8 * M + cplx * (M * M + 2 * M) + 2 * cplx * static_cast<std::size_t>(chunk) * M;
}

struct Parsed {
    std::map<std::string, std::string> kv;
    std::vector<std::pair<std::string, std::size_t>> items;
};

Parsed parse_line(const std::string& line) {
    Parsed p;
    std::istringstream is(line);
    std::string tok;
    is >> tok; // tag
    while (is >> tok) {
        const std::size_t eq = tok.find('=');
        if (eq != std::string::npos)
            p.kv[tok.substr(0, eq)] = tok.substr(eq + 1);
    }
    const std::string it = p.kv.count("items") ? p.kv["items"] : "-";
    if (it != "-") {
        std::size_t pos = 0;
        while (pos <= it.size()) {
            std::size_t comma = it.find(',', pos);
            if (comma == std::string::npos)
                comma = it.size();
            const std::string part = it.substr(pos, comma - pos);
            const std::size_t colon = part.rfind(':');
            p.items.emplace_back(part.substr(0, colon),
                                 static_cast<std::size_t>(std::stoull(part.substr(colon + 1))));
            pos = comma + 1;
        }
    }
    return p;
}

struct Predicted {
    std::size_t F = 0, S1 = 0, S2 = 0, S3 = 0;
};

/// Closed-form/model prediction of the exact bytes of F, S1, S2, S3 (no allocation, no device
/// kernels): cuFFT work areas are passed in (plan sizes), everything else is geometry.
Predicted predict(int nx, int ny, int nz, int levels, std::size_t cufft_gen,
                  std::size_t cufft_prefilter, int label_chunk) {
    const std::size_t n = static_cast<std::size_t>(nx) * ny * nz;
    const std::size_t hs = static_cast<std::size_t>(nx / 2 + 1) * ny * nz;
    const std::size_t m = std::max({static_cast<std::size_t>(nx + 1) * ny * nz,
                                    static_cast<std::size_t>(nx) * (ny + 1) * nz,
                                    static_cast<std::size_t>(nx) * ny * (nz + 1)});
    Predicted p;
    const std::size_t blocks = std::min<std::size_t>(256, (n + 255) / 256);
    p.F = 8 * n + 16 * hs + 8 + 8 * blocks + cufft_gen;
    p.S1 = 8 * n + 8 * n + 16 * hs + cufft_prefilter;
    p.S3 = 8 * m + 8 * n + 8 * n + 16 * hs + size_of_labels_device(ny, label_chunk) + cufft_prefilter;
    // S2 = 8 (7 n + 4 sum_{l>=1} n_l) + 32 (m - n) + scratch
    std::size_t sum_coarse = 0, scratch = 8 + 32 + 2048;
    scratch += 2 * (blas::sum_workspace_bytes(n) + 8);
    for (int l = 0; l < levels; ++l) {
        const std::size_t nl = static_cast<std::size_t>(nx >> l) * (ny >> l) * (nz >> l);
        if (l >= 1)
            sum_coarse += nl;
        scratch += blas::sum_workspace_bytes(nl) + 8;
    }
    p.S2 = 8 * (7 * n + 4 * sum_coarse) + 32 * (m - n) + scratch;
    return p;
}

void cufft_work_sizes(int nx, int ny, int nz, std::size_t& z2d, std::size_t& d2z) {
    auto one = [&](cufftType t) {
        cufftHandle h;
        if (cufftCreate(&h) != CUFFT_SUCCESS)
            throw std::runtime_error("cufftCreate failed");
        if (cufftSetAutoAllocation(h, 0) != CUFFT_SUCCESS)
            throw std::runtime_error("cufftSetAutoAllocation failed");
        std::size_t ws = 0;
        const cufftResult r = cufftMakePlan3d(h, nz, ny, nx, t, &ws);
        cufftDestroy(h);
        if (r != CUFFT_SUCCESS)
            throw std::runtime_error("cufftMakePlan3d failed (code " + std::to_string(static_cast<int>(r)) +
                                     ")");
        return ws;
    };
    z2d = one(CUFFT_Z2D);
    d2z = one(CUFFT_D2Z);
}

void test_memory(CudaContext& ctx, const std::string& tag, const ll::LongBoxSpec& ls) {
    const ll::LongBoxStage S = ll::build_long_box_stage(ctx, ls);
    check(tag + "_mem_stage_ready", S.flow_ready());
    if (!S.flow_ready())
        return;
    ll::InjectionSpec inj;
    inj.M = 4;
    ll::TrackOptions to;
    to.h_max = S.grid.dx / 8.0;
    to.planes = {0.25, 0.5};
    to.times = {0.25};
    to.fused_splines = ls.fused_coefficients;
    ll::LagrangianTracker tr;
    tr.prepare(ctx, S.device_flow(), inj, to);
    const std::vector<std::string> lines = ll::mem_lines(S, &tr, 0.0);
    const std::size_t n = S.grid.num_cells();
    const int nx = S.grid.nx, ny = S.grid.ny, nz = S.grid.nz;
    const std::size_t hs = static_cast<std::size_t>(nx / 2 + 1) * ny * nz;
    const std::size_t m = std::max({static_cast<std::size_t>(nx + 1) * ny * nz,
                                    static_cast<std::size_t>(nx) * (ny + 1) * nz,
                                    static_cast<std::size_t>(nx) * ny * (nz + 1)});

    std::map<std::string, Parsed> ph;
    Parsed geom;
    bool geom_seen = false;
    for (const std::string& l : lines) {
        if (l.rfind("MEM_GEOM ", 0) == 0) {
            geom = parse_line(l);
            geom_seen = true;
        } else if (l.rfind("MEM_PHASE ", 0) == 0) {
            Parsed p = parse_line(l);
            ph[p.kv["phase"]] = p;
        }
    }
    check(tag + "_mem_lines_present", geom_seen && ph.size() == 5 && ph.count("F") && ph.count("S1") &&
                                          ph.count("S2") && ph.count("S3") && ph.count("T"),
          std::to_string(lines.size()) + " lines");
    if (!(geom_seen && ph.size() == 5))
        return;
    for (const auto& l : lines)
        std::printf("  %s\n", l.c_str());

    // identity and key syntax
    bool identity = true, keys = true;
    for (auto& kvp : ph) {
        Parsed& p = kvp.second;
        std::size_t sum = std::stoull(p.kv["cufft_work"]);
        for (auto& it : p.items)
            sum += it.second;
        identity = identity && sum == std::stoull(p.kv["exact_bytes"]);
        keys = keys && p.kv.count("device_used") && p.kv.count("seconds") && p.kv.count("items");
    }
    check(tag + "_mem_exact_equals_cufft_plus_items", identity);
    check(tag + "_mem_line_keys", keys);

    // geometry
    {
        const int levels = std::stoi(geom.kv["levels"]);
        bool ok = std::stoull(geom.kv["n"]) == n && std::stoull(geom.kv["face_max"]) == m &&
                  levels == ll::box_auto_mg_levels(nx, ny, nz) &&
                  geom.kv["bhat"] == (ls.bhat_mapped_host ? "mapped" : "device") &&
                  std::stoull(geom.kv["bhat_host_bytes"]) == (ls.bhat_mapped_host ? 8 * n : 0) &&
                  geom.kv["coarsest"] == std::to_string(nx >> (levels - 1)) + "x" +
                                             std::to_string(ny >> (levels - 1)) + "x" +
                                             std::to_string(nz >> (levels - 1));
        check(tag + "_mem_geom", ok);
    }

    auto find = [](const Parsed& p, const std::string& name, std::size_t& out) {
        for (const auto& it : p.items)
            if (it.first == name) {
                out = it.second;
                return true;
            }
        return false;
    };
    // item names and size rules per phase
    {
        bool ok = true;
        std::size_t v = 0;
        const Parsed& F = ph["F"];
        std::set<std::string> names;
        for (auto& it : F.items)
            names.insert(it.first);
        ok = names == std::set<std::string>{"y", "sf18_spectrum", "sf18_mode_counter",
                                            "sf18_reduce_blocks"};
        ok = ok && find(F, "y", v) && v == 8 * n && find(F, "sf18_spectrum", v) && v == 16 * hs &&
             find(F, "sf18_mode_counter", v) && v == 8 && find(F, "sf18_reduce_blocks", v) &&
             v == 8 * std::min<std::size_t>(256, (n + 255) / 256) && F.kv.count("cufft_gen") &&
             F.kv.at("cufft_gen") == F.kv.at("cufft_work");
        check(tag + "_mem_F_items", ok);

        const Parsed& S1 = ph["S1"];
        names.clear();
        for (auto& it : S1.items)
            names.insert(it.first);
        ok = names == std::set<std::string>{"lnk", "coef_logk", "prefilter_spectrum"} &&
             find(S1, "lnk", v) && v == 8 * n && find(S1, "coef_logk", v) && v == 8 * n &&
             find(S1, "prefilter_spectrum", v) && v == 16 * hs &&
             S1.kv.count("cufft_prefilter_logk") &&
             S1.kv.at("cufft_prefilter_logk") == S1.kv.at("cufft_work");
        check(tag + "_mem_S1_items", ok);

        const Parsed& S3 = ph["S3"];
        names.clear();
        for (auto& it : S3.items)
            names.insert(it.first);
        ok = names == std::set<std::string>{"h_tilde", "coef_logk", "labels_device", "coef_pot",
                                            "prefilter_spectrum"} &&
             find(S3, "h_tilde", v) && v == 8 * m && find(S3, "coef_logk", v) && v == 8 * n &&
             find(S3, "coef_pot", v) && v == 8 * n && find(S3, "prefilter_spectrum", v) &&
             v == 16 * hs && find(S3, "labels_device", v) &&
             v == size_of_labels_device(ny, ls.label_chunk) && v == S.labels.device_bytes() &&
             S3.kv.count("cufft_prefilter_pot") &&
             S3.kv.at("cufft_prefilter_pot") == S3.kv.at("cufft_work");
        check(tag + "_mem_S3_items", ok);

        const Parsed& T = ph["T"];
        const std::size_t Np = 16, P = 2, Tn = 1;
        ok = true;
        if (ls.fused_coefficients) {
            ok = find(T, "coef_interleaved", v) && v == 16 * n && !find(T, "coef_pot", v);
        } else {
            ok = find(T, "coef_pot", v) && v == 8 * n && find(T, "coef_logk", v) && v == 8 * n &&
                 !find(T, "coef_interleaved", v);
        }
        ok = ok && find(T, "labels_device", v) && v == S.labels.device_bytes() &&
             find(T, "tracker_seeds", v) && v == 24 * Np && find(T, "tracker_planes", v) &&
             v == 8 * P && find(T, "tracker_times", v) && v == 8 * Tn &&
             find(T, "tracker_stats_plane", v) && v == Np * sizeof(ll::PassStats) &&
             find(T, "tracker_planes_x23", v) && v == 16 * Np * P &&
             find(T, "tracker_times_x", v) && v == 24 * Np * Tn &&
             find(T, "tracker_prepare_transient", v) && v == 32 * Np; // labels used: 4 Np doubles
        for (auto& it : T.items) {
            const bool known = it.first == "coef_pot" || it.first == "coef_logk" ||
                               it.first == "coef_interleaved" || it.first == "labels_device" ||
                               it.first.rfind("tracker_", 0) == 0;
            ok = ok && known;
        }
        check(tag + "_mem_T_items", ok);
    }

    // S2: lean items verbatim, S+ closed form computed independently, categories
    {
        const Parsed& S2 = ph["S2"];
        const int levels = std::stoi(geom.kv["levels"]);
        const Predicted pr = predict(nx, ny, nz, levels, 0, 0, ls.label_chunk);
        const std::size_t exact = std::stoull(S2.kv.at("exact_bytes"));
        std::size_t expect = pr.S2;
        if (!ls.bhat_mapped_host)
            expect += 8 * n;
        check(tag + "_mem_S2_equals_Splus_closed_form", exact == expect && S2.kv.at("cufft_work") == "0",
              "reported " + std::to_string(exact) + " expected " + std::to_string(expect));
        std::size_t cat_sum = 0;
        {
            std::istringstream cs(S2.kv.at("categories"));
            std::string part;
            int count = 0;
            while (std::getline(cs, part, ',')) {
                cat_sum += std::stoull(part.substr(part.rfind(':') + 1));
                ++count;
            }
            check(tag + "_mem_S2_categories", count == 8 && cat_sum == exact,
                  std::to_string(count) + " categories, sum " + std::to_string(cat_sum));
        }
        bool names_ok = true;
        for (auto& it : S2.items) {
            const std::string& nm = it.first;
            names_ok = names_ok && (nm.rfind("mg_l", 0) == 0 || nm == "pcg_r" || nm == "pcg_p" ||
                                    nm == "corrector" || nm == "bhat_device" ||
                                    nm.rfind("scratch_", 0) == 0);
        }
        check(tag + "_mem_S2_item_names", names_ok);
        // The label tables of S3 (2 chunk x (ny - 1) complex doubles) are a fixed cost independent
        // of n: at test sizes they exceed S2, at production sizes S2 dominates (checked on the
        // model below). Here: S2 is the largest phase once the label tables are set aside.
        std::size_t largest_other = 0;
        for (const char* k : {"F", "S1", "S3"}) {
            std::size_t b = std::stoull(ph[k].kv.at("exact_bytes"));
            std::size_t lab = 0;
            if (find(ph[k], "labels_device", lab))
                b -= lab;
            largest_other = std::max(largest_other, b);
        }
        check(tag + "_mem_S2_is_largest_phase_excluding_label_tables", exact >= largest_other,
              "S2 " + std::to_string(exact) + " max other " + std::to_string(largest_other));
    }

    // the prediction model of --cufft-size (plan sizes via cufftMakePlan3d, no allocation)
    {
        std::size_t z2d = 0, d2z = 0;
        cufft_work_sizes(nx, ny, nz, z2d, d2z);
        const int levels = std::stoi(geom.kv["levels"]);
        const Predicted pr = predict(nx, ny, nz, levels, z2d, d2z + z2d, ls.label_chunk);
        check(tag + "_mem_prediction_equals_recorded",
              pr.F == std::stoull(ph["F"].kv.at("exact_bytes")) &&
                  pr.S1 == std::stoull(ph["S1"].kv.at("exact_bytes")) &&
                  pr.S3 == std::stoull(ph["S3"].kv.at("exact_bytes")) &&
                  pr.S2 + (ls.bhat_mapped_host ? 0 : 8 * n) == std::stoull(ph["S2"].kv.at("exact_bytes")),
              "F " + std::to_string(pr.F) + "/" + ph["F"].kv.at("exact_bytes") + " S1 " +
                  std::to_string(pr.S1) + "/" + ph["S1"].kv.at("exact_bytes") + " S3 " +
                  std::to_string(pr.S3) + "/" + ph["S3"].kv.at("exact_bytes"));
    }
}

/// S2 is the largest phase at the production geometries (model: cuFFT plan sizes, no allocation).
void test_s2_largest_at_production_sizes() {
    for (const std::array<int, 3>& g : {std::array<int, 3>{512, 512, 512}, std::array<int, 3>{3328, 384, 384}}) {
        std::size_t z2d = 0, d2z = 0;
        const std::string tag = "s2_largest_predicted_" + std::to_string(g[0]) + "x" +
                                std::to_string(g[1]) + "x" + std::to_string(g[2]);
        try {
            cufft_work_sizes(g[0], g[1], g[2], z2d, d2z);
        } catch (const std::exception& e) {
            check(tag, false, e.what());
            continue;
        }
        const Predicted p = predict(g[0], g[1], g[2], ll::box_auto_mg_levels(g[0], g[1], g[2]), z2d,
                                    d2z + z2d, 16384);
        const std::size_t other = std::max({p.F, p.S1, p.S3});
        const double n = static_cast<double>(g[0]) * g[1] * g[2];
        check(tag, p.S2 >= other,
              "S2 " + std::to_string(p.S2) + " (" + fmt("%.2f", static_cast<double>(p.S2) / n) +
                  " B/cell) max other " + std::to_string(other));
    }
}

void test_memory_cells_source(CudaContext& ctx) {
    ll::LongBoxSpec ls;
    ls.nx = 64;
    ls.ny = ls.nz = 16;
    ls.source = ll::LongBoxSourceKind::cells;
    ls.cells.assign(static_cast<std::size_t>(64) * 16 * 16, 0.0);
    const ll::LongBoxStage S = ll::build_long_box_stage(ctx, ls);
    const std::vector<std::string> lines = ll::mem_lines(S);
    bool f_empty = false;
    for (const auto& l : lines)
        if (l.rfind("MEM_PHASE phase=F ", 0) == 0) {
            const Parsed p = parse_line(l);
            f_empty = p.kv.at("exact_bytes") == "0" && p.items.empty() && p.kv.at("items") == "-";
        }
    check("mem_cells_source_F_phase_empty", f_empty && S.report.field == "cells" && !S.report.has_sf18);
}

// ------------------------------------------------------------------------------------------------
// 4. F1-F4 helpers
// ------------------------------------------------------------------------------------------------
using VFun = std::function<void(const double x[3], double v[3])>;

struct HostFlow {
    std::vector<real> pot, logk;
    sl::SlabSplineDirectionField fld;
    VFun vf;
};

/// Host direction field over the stage's device coefficient arrays (non-fused stage).
std::unique_ptr<HostFlow> host_flow(const ll::LongBoxStage& S) {
    auto H = std::make_unique<HostFlow>();
    const std::size_t n = S.grid.num_cells();
    H->pot = download(S.coef_pot.data(), n);
    H->logk = download(S.coef_logk.data(), n);
    H->fld.potential = interpolation::make_host_view(S.grid, H->pot.data());
    H->fld.log_conductivity = interpolation::make_host_view(S.grid, H->logk.data());
    for (int d = 0; d < 3; ++d)
        H->fld.G[d] = S.G[d];
    const sl::SlabSplineDirectionField f = H->fld;
    H->vf = [f](const double x[3], double v[3]) {
        double g[3], k;
        f(x, g, k);
        v[0] = k * g[0];
        v[1] = k * g[1];
        v[2] = k * g[2];
    };
    return H;
}

void initial_covectors(const VFun& vf, const double x[3], double a[3], double b[3]) {
    double v[3];
    vf(x, v);
    a[0] = -v[1];
    a[1] = v[0];
    a[2] = 0.0;
    b[0] = -v[2] / v[0];
    b[1] = 0.0;
    b[2] = 1.0;
}

template <class F> void for_each_record(const ll::LagrangianTrackResult& R, F f) {
    const std::size_t Np = static_cast<std::size_t>(R.Np), P = R.planes.size(), T = R.times.size();
    for (std::size_t p = 0; p < Np; ++p) {
        if (R.status_plane[p] == 0) {
            for (std::size_t m = 0; m < P; ++m) {
                const double x[3] = {R.planes[m], R.planes_x23[(p * P + m) * 2],
                                     R.planes_x23[(p * P + m) * 2 + 1]};
                f(0, m, p, x, R.planes_I[p * P + m], R.planes_tau[p * P + m],
                  R.covectors ? &R.planes_cov[(p * P + m) * 6] : nullptr);
            }
        }
        if (R.status_time[p] == 0) {
            for (std::size_t n = 0; n < T; ++n) {
                f(1, n, p, &R.times_x[(p * T + n) * 3], R.times_I[p * T + n], R.times[n],
                  R.covectors ? &R.times_cov[(p * T + n) * 6] : nullptr);
            }
        }
    }
}

// Gauss-Kronrod G7K15 (QUADPACK), copied from lagrangian_controls_tests.cu (SF-36 A2 helper)
template <class F> double gk15(const F& f, double a, double b, double& err) {
    static const double xgk[8] = {0.991455371120812639206854697526329, 0.949107912342758524526189684047851,
                                  0.864864423359769072789712788640926, 0.741531185599394439863864773280788,
                                  0.586087235467691130294144838258730, 0.405845151377397166906606412076961,
                                  0.207784955007898467600689403773245, 0.0};
    static const double wgk[8] = {0.022935322010529224963732008058970, 0.063092092629978553290700663189204,
                                  0.104790010322250183839876322541518, 0.140653259715525918745189590510238,
                                  0.169004726639267902826583426598550, 0.190350578064785409913256402421014,
                                  0.204432940075298892414161999234649, 0.209482141084727828012999174891714};
    static const double wg[4] = {0.129484966168869693270611432679082, 0.279705391489276667901467771423780,
                                 0.381830050505118944950369775488975, 0.417959183673469387755102040816327};
    const double c = 0.5 * (a + b), hl = 0.5 * (b - a);
    const double fc = f(c);
    double rk = wgk[7] * fc, rg = wg[3] * fc;
    for (int j = 0; j < 7; ++j) {
        const double dx = hl * xgk[j];
        const double s = f(c - dx) + f(c + dx);
        rk += wgk[j] * s;
        if (j % 2 == 1)
            rg += wg[j / 2] * s;
    }
    err = std::abs((rk - rg) * hl);
    return rk * hl;
}

template <class F> double adaptive_gk(const F& f, double a, double b, double atol, int depth) {
    double err;
    const double r = gk15(f, a, b, err);
    if (err <= atol || depth >= 40)
        return r;
    const double m = 0.5 * (a + b);
    return adaptive_gk(f, a, m, 0.5 * atol, depth + 1) + adaptive_gk(f, m, b, 0.5 * atol, depth + 1);
}

/// Integral over [a, b] split at the half-cell lattice (spline piece boundaries lie on it).
template <class F> double quad_split(const F& f, double a, double b, double h) {
    double total = 0.0;
    double x = a;
    const double step = 0.5 * h;
    long long j = static_cast<long long>(std::floor(a / step)) + 1;
    while (x < b) {
        const double nxt = std::min(b, static_cast<double>(j) * step);
        if (nxt > x)
            total += adaptive_gk(f, x, nxt, 1e-15, 0);
        x = nxt;
        ++j;
    }
    return total;
}

ll::TrackOptions control_options(double h) {
    ll::TrackOptions to;
    to.tol = 1e-8;
    to.h_max = h / 8.0;
    to.planes = {0.25, 0.5, 1.0, 2.0, 3.0};
    to.times = {0.5, 1.0, 2.0};
    to.covectors = true;
    return to;
}

// ---- F1: k = 1 on a box
void case_F1(CudaContext& ctx, int nx, int ny, int M) {
    const std::string tag = "F1_" + std::to_string(nx) + "x" + std::to_string(ny) + "x" +
                            std::to_string(ny) + "_M" + std::to_string(M);
    const double t0 = now_s();
    ll::LongBoxSpec ls;
    ls.nx = nx;
    ls.ny = ls.nz = ny;
    ls.source = ll::LongBoxSourceKind::cells;
    ls.cells.assign(static_cast<std::size_t>(nx) * ny * ny, 0.0);
    ls.field_name = "uniform";
    const ll::LongBoxStage S = ll::build_long_box_stage(ctx, ls);
    check(tag + "_stage_ready", S.flow_ready(), sl::to_string(S.report.status));
    if (!S.flow_ready())
        return;
    const auto& dr = S.report.darcy;
    double dK = 0.0, dF = 0.0;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j)
            dK = std::max(dK, std::abs(dr.K_eff[i][j] - (i == j ? 1.0 : 0.0)));
        dF = std::max(dF, std::abs(dr.achieved_mean_flux[i] - (i == 0 ? 1.0 : 0.0)));
    }
    check(tag + "_Keff_equals_I", dK <= 1e-12, fmt("max %.3e", dK));
    check(tag + "_mean_flux_equals_e1", dF <= 1e-12, fmt("max %.3e", dF));

    const double h = 1.0 / ny;
    ll::InjectionSpec inj;
    inj.X0 = 0.0;
    inj.M = M;
    const ll::TrackOptions to = control_options(h);
    const ll::LagrangianTrackResult R = run_lb(ctx, S, inj, to);
    const auto H = host_flow(S);
    double dx = 0.0, dtau = 0.0, dcov = 0.0, mI = 0.0, dxt = 0.0;
    long long nrec = 0;
    for_each_record(R, [&](int kind, std::size_t idx, std::size_t p, const double* x, double I,
                           double tau, const double* cov) {
        const double* sd = &R.seeds[3 * p];
        dx = std::max({dx, std::abs(x[1] - sd[1]), std::abs(x[2] - sd[2])});
        double a0[3], b0[3];
        initial_covectors(H->vf, sd, a0, b0);
        for (int i = 0; i < 3; ++i) {
            dcov = std::max(dcov, std::abs(cov[i] - a0[i]));
            dcov = std::max(dcov, std::abs(cov[3 + i] - b0[i]));
        }
        mI = std::max(mI, std::abs(I));
        if (kind == 0) {
            const double X = R.planes[idx];
            dtau = std::max(dtau, std::abs(tau - (X - inj.X0)) / (X - inj.X0));
        } else {
            dxt = std::max(dxt, std::abs(x[0] - (inj.X0 + R.times[idx])));
        }
        ++nrec;
    });
    check(tag + "_all_particles_ok", R.totals_plane.status_hist[0] == R.Np &&
                                         R.totals_time.status_hist[0] == R.Np);
    check(tag + "_transverse_displacement_le_1e-12", dx <= 1e-12,
          fmt("max %.3e", dx) + " over " + std::to_string(nrec) + " records");
    check(tag + "_tau_equals_X_rel_le_1e-12", dtau <= 1e-12, fmt("max rel %.3e", dtau));
    check(tag + "_covectors_equal_initial_le_1e-12", dcov <= 1e-12, fmt("max %.3e", dcov));
    check(tag + "_abs_I_le_1e-14", mI <= 1e-14, fmt("max %.3e", mI));
    std::printf("  %s INFO max |x1(t) - (X0 + t)| at snapshots %.3e  [%.1f s]\n", tag.c_str(), dxt,
                now_s() - t0);
}

// ---- F2: layered k = exp(0.5 cos 2 pi x1) on a box
void case_F2(CudaContext& ctx, int nx, int ny, int M) {
    const std::string tag = "F2_" + std::to_string(nx) + "x" + std::to_string(ny) + "x" +
                            std::to_string(ny) + "_M" + std::to_string(M);
    const double t0 = now_s();
    const double h = 1.0 / ny;
    ll::LongBoxSpec ls;
    ls.nx = nx;
    ls.ny = ls.nz = ny;
    ls.source = ll::LongBoxSourceKind::cells;
    ls.field_name = "layered";
    ls.eps = 0.5;
    ls.cells.resize(static_cast<std::size_t>(nx) * ny * ny);
    for (int k = 0; k < ny; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i)
                ls.cells[static_cast<std::size_t>(i) +
                         static_cast<std::size_t>(nx) * (j + static_cast<std::size_t>(ny) * k)] =
                    std::cos(kTwoPi * (static_cast<double>(i) + 0.5) * h);
    const ll::LongBoxStage S = ll::build_long_box_stage(ctx, ls);
    check(tag + "_stage_ready", S.flow_ready(), sl::to_string(S.report.status));
    if (!S.flow_ready())
        return;
    ll::InjectionSpec inj;
    inj.X0 = 0.0;
    inj.M = M;
    const ll::TrackOptions to = control_options(h);
    const ll::LagrangianTrackResult R = run_lb(ctx, S, inj, to);
    const auto H = host_flow(S);
    const sl::SlabSplineDirectionField& fld = H->fld;
    auto v1_at = [&](double x1, double x2, double x3) {
        double v[3];
        const double x[3] = {x1, x2, x3};
        H->vf(x, v);
        return v[0];
    };
    auto dI_at = [&](double x1, double x2, double x3) {
        real val, gx, gy, gz, Hs[6];
        interpolation::evaluate_point_hessian(fld.potential, x1, x2, x3, val, gx, gy, gz, Hs);
        real y, yx, yy, yz;
        interpolation::evaluate_point(fld.log_conductivity, x1, x2, x3, y, yx, yy, yz);
        const double g1 = fld.G[0] + gx, g2 = fld.G[1] + gy, g3 = fld.G[2] + gz;
        return (yx * g1 + yy * g2 + yz * g3 + Hs[0] + Hs[1] + Hs[2]) / g1;
    };
    double dx = 0.0, dcov = 0.0, rel_tau = 0.0, abs_I = 0.0, max_Iref = 0.0;
    long long nrec = 0;
    for_each_record(R, [&](int kind, std::size_t, std::size_t p, const double* x, double I,
                           double tau, const double* cov) {
        const double* sd = &R.seeds[3 * p];
        dx = std::max({dx, std::abs(x[1] - sd[1]), std::abs(x[2] - sd[2])});
        double a0[3], b0[3];
        initial_covectors(H->vf, sd, a0, b0);
        for (int i = 0; i < 3; ++i) {
            dcov = std::max(dcov, std::abs(cov[i] - a0[i]));
            dcov = std::max(dcov, std::abs(cov[3 + i] - b0[i]));
        }
        if (kind == 0) {
            const double tau_ref =
                quad_split([&](double s) { return 1.0 / v1_at(s, sd[1], sd[2]); }, sd[0], x[0], h);
            rel_tau = std::max(rel_tau, std::abs(tau - tau_ref) / std::abs(tau_ref));
        }
        const double I_ref =
            quad_split([&](double s) { return dI_at(s, sd[1], sd[2]); }, sd[0], x[0], h);
        abs_I = std::max(abs_I, std::abs(I - I_ref));
        max_Iref = std::max(max_Iref, std::abs(I_ref));
        ++nrec;
    });
    check(tag + "_all_particles_ok", R.totals_plane.status_hist[0] == R.Np &&
                                         R.totals_time.status_hist[0] == R.Np);
    check(tag + "_transverse_positions_constant_le_1e-12", dx <= 1e-12, fmt("max %.3e", dx));
    check(tag + "_covectors_constant_le_1e-12", dcov <= 1e-12, fmt("max %.3e", dcov));
    check(tag + "_tau_vs_quadrature_rel_le_1e-9", rel_tau <= 1e-9, fmt("max rel %.3e", rel_tau));
    std::printf("[RECORD] %s I vs independent quadrature (no threshold): max abs err %.3e, max |I_ref| "
                "%.3e over %lld records  [%.1f s]\n",
                tag.c_str(), abs_I, max_Iref, nrec, now_s() - t0);
}

// ---- F3: transverse periodicity of v_s
void case_F3(CudaContext& ctx, int nx, int ny, int npts) {
    const std::string tag = "F3_" + std::to_string(nx) + "x" + std::to_string(ny) + "x" +
                            std::to_string(ny) + "_" + std::to_string(npts) + "pts";
    const double t0 = now_s();
    ll::LongBoxSpec ls;
    ls.nx = nx;
    ls.ny = ls.nz = ny;
    ls.sigma2 = 1.0;
    ls.ell = 0.25;
    ls.seed = 3101ULL;
    ls.eps = 1.0;
    const ll::LongBoxStage S = ll::build_long_box_stage(ctx, ls);
    check(tag + "_stage_ready", S.flow_ready(), sl::to_string(S.report.status));
    if (!S.flow_ready())
        return;
    const auto H = host_flow(S);
    const double h = 1.0 / ny, Lx = nx * h;
    std::mt19937_64 rng(20261022ULL);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    const double edge[] = {0.0, 0.25 * h, 0.5 * h, h, 1.0 - h, 1.0 - 0.5 * h, 1.0 - 1e-9, 1.0};
    std::vector<std::array<double, 3>> pts;
    for (int i = 0; i < npts; ++i) {
        std::array<double, 3> p{Lx * U(rng), U(rng), U(rng)};
        if (i % 4 == 1) // near the x2 / x3 periodic faces
            p[1] = edge[rng() % 8];
        if (i % 4 == 2)
            p[2] = edge[rng() % 8];
        if (i % 4 == 3) { // near the x1 faces and edges of the box
            p[0] = (rng() % 2 == 0) ? edge[rng() % 8] * 0.25 : Lx - edge[rng() % 8] * 0.25;
            p[1] = edge[rng() % 8];
            p[2] = edge[rng() % 8];
        }
        pts.push_back(p);
    }
    double worst_abs = 0.0, worst_rel = 0.0;
    long long bad = 0;
    for (const auto& p : pts) {
        double v0[3];
        H->vf(p.data(), v0);
        const double vn = std::sqrt(v0[0] * v0[0] + v0[1] * v0[1] + v0[2] * v0[2]);
        for (int k = 1; k <= 2; ++k) {
            double q[3] = {p[0], p[1], p[2]};
            q[k] += 1.0;
            double v1[3];
            H->vf(q, v1);
            for (int c = 0; c < 3; ++c) {
                const double d = std::abs(v1[c] - v0[c]);
                worst_abs = std::max(worst_abs, d);
                worst_rel = std::max(worst_rel, d / vn);
                if (!(d <= 1e-14 && d <= 1e-14 * vn))
                    ++bad;
            }
        }
    }
    check(tag + "_shift_plus_1_in_x2_x3_le_1e-14", bad == 0,
          fmt("max abs %.3e", worst_abs) + fmt(" max rel %.3e", worst_rel) + " violating " +
              std::to_string(bad));
    std::printf("  %s [%.1f s]\n", tag.c_str(), now_s() - t0);
}

// ---- F4: tiling
void case_F4(CudaContext& ctx, int Nc, int M) {
    const std::string tag = "F4_" + std::to_string(Nc) + "cube_to_" + std::to_string(2 * Nc) + "x" +
                            std::to_string(Nc) + "x" + std::to_string(Nc) + "_M" + std::to_string(M);
    const double t0 = now_s();
    ll::LongBoxSpec cs;
    cs.nx = cs.ny = cs.nz = Nc;
    cs.sigma2 = 1.0;
    cs.ell = 0.25;
    cs.seed = 3101ULL;
    cs.eps = 1.0;
    cs.pcg_rtol = 1e-12;
    cs.keep_host_field = true;
    const ll::LongBoxStage cube = ll::build_long_box_stage(ctx, cs);

    ll::LongBoxSpec bs;
    bs.nx = 2 * Nc;
    bs.ny = bs.nz = Nc;
    bs.source = ll::LongBoxSourceKind::cells;
    bs.eps = 1.0;
    bs.pcg_rtol = 1e-12;
    bs.cells.resize(static_cast<std::size_t>(bs.nx) * Nc * Nc);
    for (int k = 0; k < Nc; ++k)
        for (int j = 0; j < Nc; ++j)
            for (int i = 0; i < bs.nx; ++i)
                bs.cells[static_cast<std::size_t>(i) +
                         static_cast<std::size_t>(bs.nx) * (j + static_cast<std::size_t>(Nc) * k)] =
                    cube.host_field[static_cast<std::size_t>(i % Nc) +
                                    static_cast<std::size_t>(Nc) * (j + static_cast<std::size_t>(Nc) * k)];
    const ll::LongBoxStage box = ll::build_long_box_stage(ctx, bs);
    check(tag + "_stages_ready", cube.flow_ready() && box.flow_ready());
    if (!(cube.flow_ready() && box.flow_ready()))
        return;

    // G and K_eff: normwise relative error max|a - b| / max|b|; entrywise recorded
    auto normwise = [](const double* a, const double* b, int cnt, double& entry_max) {
        double d = 0.0, mb = 0.0;
        entry_max = 0.0;
        for (int i = 0; i < cnt; ++i) {
            d = std::max(d, std::abs(a[i] - b[i]));
            mb = std::max(mb, std::abs(b[i]));
            if (b[i] != 0.0)
                entry_max = std::max(entry_max, std::abs(a[i] - b[i]) / std::abs(b[i]));
        }
        return d / mb;
    };
    double eG = 0.0, eK = 0.0;
    const double nG = normwise(box.G, cube.G, 3, eG);
    const double nK = normwise(&box.report.darcy.K_eff[0][0], &cube.report.darcy.K_eff[0][0], 9, eK);
    check(tag + "_G_normwise_rel_le_1e-8", nG <= 1e-8, fmt("%.3e", nG) + fmt(" (entrywise max %.3e)", eG));
    check(tag + "_Keff_normwise_rel_le_1e-8", nK <= 1e-8,
          fmt("%.3e", nK) + fmt(" (entrywise max %.3e)", eK));

    ll::InjectionSpec inj;
    inj.X0 = 0.0;
    inj.M = M;
    ll::TrackOptions to;
    to.tol = 1e-8;
    to.h_max = (1.0 / Nc) / 8.0;
    to.planes = {0.25, 0.5, 1.0, 2.0, 4.0};
    const ll::LagrangianTrackResult A = run_lb(ctx, box, inj, to);
    const ll::LagrangianTrackResult B = run_lb(ctx, cube, inj, to);
    double worst = 0.0;
    bool finite = A.moments_planes.size() == B.moments_planes.size() && !A.moments_planes.empty();
    for (std::size_t m = 0; finite && m < A.moments_planes.size(); ++m) {
        const auto &a = A.moments_planes[m], &b = B.moments_planes[m];
        const double sa = 0.5 * (a.var_x2 + a.var_x3), sb = 0.5 * (b.var_x2 + b.var_x3);
        const double rel[3] = {std::abs(sa - sb) / std::abs(sb),
                               std::abs(a.mean_tau - b.mean_tau) / std::abs(b.mean_tau),
                               std::abs(a.var_tau - b.var_tau) / std::abs(b.var_tau)};
        for (double r : rel) {
            finite = finite && std::isfinite(r);
            worst = std::max(worst, r);
        }
        std::printf("  %s plane %.3g: sigma_T^2 %.12e / %.12e  mean_tau %.12e / %.12e  var_tau %.12e / "
                    "%.12e  (box / cube)\n",
                    tag.c_str(), a.x1, sa, sb, a.mean_tau, b.mean_tau, a.var_tau, b.var_tau);
    }
    check(tag + "_plane_stats_rel_le_1e-8", finite && worst <= 1e-8, fmt("max rel %.3e", worst));
    std::printf("  %s [%.1f s]\n", tag.c_str(), now_s() - t0);
}

// ------------------------------------------------------------------------------------------------
// --cufft-size
// ------------------------------------------------------------------------------------------------
int cufft_size_mode(const std::string& arg) {
    int nx = 0, ny = 0, nz = 0;
    if (std::sscanf(arg.c_str(), "%d,%d,%d", &nx, &ny, &nz) != 3 || nx < 4 || ny < 4 || nz < 4) {
        std::fprintf(stderr, "--cufft-size expects nx,ny,nz\n");
        return 2;
    }
    CudaContext ctx(0); // device initialisation only; nothing is allocated
    (void)ctx;
    std::size_t z2d = 0, d2z = 0;
    bool plans_ok = true;
    try {
        cufft_work_sizes(nx, ny, nz, z2d, d2z);
    } catch (const std::exception& e) {
        std::printf("CUFFT_ERROR %s\n", e.what());
        plans_ok = false;
    }
    check("cufft_plans_created", plans_ok);
    if (!plans_ok)
        return 1;
    const int levels = ll::box_auto_mg_levels(nx, ny, nz);
    const Predicted p = predict(nx, ny, nz, levels, z2d, d2z + z2d, 16384);
    const double n = static_cast<double>(nx) * ny * nz;
    std::printf("CUFFT_SIZE nx=%d ny=%d nz=%d gen_z2d=%zu prefilter_d2z=%zu prefilter_z2d=%zu "
                "levels=%d\n",
                nx, ny, nz, z2d, d2z, z2d, levels);
    const struct {
        const char* ph;
        std::size_t b;
    } rows[] = {{"F", p.F}, {"S1", p.S1}, {"S2", p.S2}, {"S3", p.S3}};
    for (const auto& r : rows)
        std::printf("PREDICT phase=%s exact_bytes=%zu bytes_per_cell=%.4f MiB=%.1f\n", r.ph, r.b,
                    static_cast<double>(r.b) / n, static_cast<double>(r.b) / (1024.0 * 1024.0));
    const std::size_t other = std::max({p.F, p.S1, p.S3});
    check("s2_largest_predicted", p.S2 >= other,
          "S2 " + std::to_string(p.S2) + " max other " + std::to_string(other));
    return g_failed == 0 ? 0 : 1;
}

} // namespace

int main(int argc, char** argv) {
    std::string cufft_arg;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--full")
            g_full = true;
        else if (a == "--only" && i + 1 < argc)
            g_only.insert(argv[++i]);
        else if (a == "--cufft-size" && i + 1 < argc)
            cufft_arg = argv[++i];
        else {
            std::fprintf(stderr,
                         "usage: %s [--full] [--only levels|cube|api|mem|F1|F2|F3|F4] [--cufft-size nx,ny,nz]\n",
                         argv[0]);
            return 2;
        }
    }
    if (!cufft_arg.empty()) {
        const int rc = cufft_size_mode(cufft_arg);
        std::printf(g_failed == 0 && rc == 0 ? "ALL_PASS\n" : "FAILED %d\n", g_failed);
        return rc == 0 && g_failed == 0 ? 0 : 1;
    }
    const double t_all = now_s();
    CudaContext ctx(0);
    if (run_case("levels"))
        test_levels();
    if (run_case("cube")) {
        // default tier: 16^3 and 32^3, eps 1 and 2; the variant set is reduced at 32^3
        test_cube_equivalence(ctx, 16, 1.0, true);
        test_cube_equivalence(ctx, 16, 2.0, true);
        test_cube_equivalence(ctx, 32, 1.0, g_full);
        test_cube_equivalence(ctx, 32, 2.0, g_full);
        // inlet backflow (min v1 <= 0) is recorded, not fatal: labels unavailable, flow still built
        test_cube_equivalence(ctx, 16, 3.0, false, 3108ULL, true);
    }
    if (run_case("api")) {
        test_api_errors(ctx);
        test_darcy_failed(ctx);
    }
    if (run_case("mem")) {
        ll::LongBoxSpec a;
        a.nx = a.ny = a.nz = 32;
        a.sigma2 = 1.0;
        a.ell = 0.25;
        a.seed = 3101ULL;
        test_memory(ctx, "cube32", a);
        ll::LongBoxSpec b = a;
        b.nx = 64;
        b.ny = b.nz = 16;
        b.ell = 0.25;
        b.bhat_mapped_host = false; // level S
        b.fused_coefficients = true;
        test_memory(ctx, "box64x16x16_levelS_fused", b);
        test_memory_cells_source(ctx);
        test_s2_largest_at_production_sizes();
    }
    if (run_case("F1")) {
        if (g_full) {
            case_F1(ctx, 64, 16, 16);
            case_F1(ctx, 128, 16, 16);
        } else {
            case_F1(ctx, 64, 16, 4);
        }
    }
    if (run_case("F2")) {
        if (g_full) {
            case_F2(ctx, 64, 32, 16);
            case_F2(ctx, 128, 32, 16);
        } else {
            case_F2(ctx, 64, 32, 4);
        }
    }
    if (run_case("F3")) {
        if (g_full)
            case_F3(ctx, 128, 32, 4096);
        else
            case_F3(ctx, 64, 32, 512);
    }
    if (run_case("F4")) {
        if (g_full)
            case_F4(ctx, 64, 64);
        else
            case_F4(ctx, 32, 16);
    }
    std::printf("  total %.1f s (%s tier)\n", now_s() - t_all, g_full ? "full" : "fast");
    if (g_failed == 0) {
        std::printf("ALL_PASS\n");
        return 0;
    }
    std::printf("FAILED %d\n", g_failed);
    return 1;
}
