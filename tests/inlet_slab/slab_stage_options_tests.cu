/**
 * @file slab_stage_options_tests.cu
 * @brief SF-36 N2: ProductionStageOptions::require_inlet_labels, ProductionStage::labels_available
 *        and flow_ready() of build_production_stage (fast tier, N = 16).
 *
 * (a) Field WITH inlet backflow. SlabFieldSource::gaussian, N = 16, sigma2 = 1, ell = 0.15,
 *     seed 2, eps = 4. Found by a scan at N = 16 over ell in {0.25, 0.15}, eps in {2, 3, 4},
 *     seeds 1..6 (default options, status + inlet_vmin printed): backflow occurs only for the
 *     short correlation length ell = 0.15 at eps >= 2 (e.g. eps 2 seed 5: vmin -1.2e-3; eps 3
 *     seeds 5, 6: -2.9e-2, -3.5e-2); eps 4 seed 2 has the most negative inlet minimum
 *     (-0.82), hence the largest margin against roundoff. Darcy converges there.
 *     Default options -> early return (status inlet_backflow, no spline coefficients, not
 *     flow_ready); require_inlet_labels = false -> same status, labels_available == false,
 *     flow_ready(), finite coefficients and vD, finite host direction field.
 * (b) Field WITHOUT backflow (gaussian seed 3001, ell 0.25, eps 0.5): both option values give
 *     bitwise-identical coefficients, vD, inlet u0, and labels_available == true.
 *
 * SF-37 N2 (contract C-STAGE), ProductionStageOptions::keep_face_fluxes, N = 16:
 * (c) default options: faces_available() false, face vectors empty.
 * (d) option off vs on (gaussian seed 3001 ell 0.25 eps 1; analytic generic3d eps 0.25): every
 *     other output (coefficients, vD, inputs arrays, inlet samples, G, labels, report) bitwise
 *     identical.
 * (e) option on: CompactMAC sizes, duplicate planes == plane 0 bitwise, U plane 0 ==
 *     inlet_v1_samples bitwise, unique-face means (1, 0, 0) within 1e-12.
 * (f) Darcy failure forced by pcg_max_iter = 1: status darcy_failed, faces not kept.
 */

#include "src/core/DeviceBuffer.cuh"
#include "src/physics/streamfunctions/inlet_slab/SlabProductionSetup.cuh"
#include "src/runtime/cuda_check.cuh"
#include "src/runtime/CudaContext.cuh"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

using namespace macroflow3d;
namespace sl = macroflow3d::streamfunctions::inlet_slab;

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

std::vector<real> download(const DeviceBuffer<real>& b) {
    std::vector<real> h(b.size());
    MACROFLOW3D_CUDA_CHECK(cudaDeviceSynchronize());
    MACROFLOW3D_CUDA_CHECK(
        cudaMemcpy(h.data(), b.data(), h.size() * sizeof(real), cudaMemcpyDeviceToHost));
    return h;
}

bool all_finite(const std::vector<real>& v) {
    if (v.empty())
        return false;
    for (real x : v)
        if (!std::isfinite(x))
            return false;
    return true;
}

bool bitwise_equal(const std::vector<real>& a, const std::vector<real>& b) {
    return a.size() == b.size() &&
           (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(real)) == 0);
}

constexpr int kN = 16;

sl::ProductionStage build(CudaContext& ctx, const sl::SlabFieldSource& src, real eps,
                          bool require_labels) {
    const sl::InletSlabGrid grid = sl::InletSlabGrid::make(kN);
    sl::ProductionStageOptions opt;
    opt.require_inlet_labels = require_labels;
    return sl::build_production_stage(ctx, grid, src, eps, opt);
}

void case_backflow(TestReport& rep, CudaContext& ctx) {
    sl::ProductionFieldSpec spec;
    spec.N = kN;
    spec.sigma2 = 1.0;
    spec.ell = 0.15;
    spec.seed = 2ULL;
    const real eps = 4.0;
    const sl::SlabFieldSource src = sl::SlabFieldSource::gaussian(ctx, spec);

    const sl::ProductionStage def = build(ctx, src, eps, true);
    char b[160];
    std::snprintf(b, sizeof(b), "status=%s inlet_vmin=%.6e", sl::to_string(def.report.status),
                  def.report.inlet_vmin);
    rep.check(def.report.darcy_converged, "backflow field: Darcy converged");
    rep.check(def.report.status == sl::SlabProductionStatus::inlet_backflow &&
                  def.report.inlet_vmin <= 0.0,
              "default options: status inlet_backflow, inlet_vmin <= 0", b);
    rep.check(!def.ok() && !def.labels_available && !def.flow_ready(),
              "default options: early return (not ok, no labels, not flow_ready)");
    rep.check(def.potential_coefficients.empty() && def.logk_coefficients.empty(),
              "default options: no spline coefficients built");

    const sl::ProductionStage nl = build(ctx, src, eps, false);
    std::snprintf(b, sizeof(b), "status=%s inlet_vmin=%.6e", sl::to_string(nl.report.status),
                  nl.report.inlet_vmin);
    rep.check(nl.report.status == sl::SlabProductionStatus::inlet_backflow &&
                  nl.report.inlet_vmin == def.report.inlet_vmin,
              "require_inlet_labels=false: status inlet_backflow, same inlet_vmin", b);
    rep.check(!nl.ok(), "require_inlet_labels=false: ok() stays false");
    rep.check(!nl.labels_available, "require_inlet_labels=false: labels_available == false");
    rep.check(nl.flow_ready(), "require_inlet_labels=false: flow_ready() == true");
    rep.check(all_finite(nl.potential_coefficients) && all_finite(nl.logk_coefficients),
              "require_inlet_labels=false: spline coefficients non-empty and finite");
    bool vd_ok = true;
    for (int d = 0; d < 3; ++d)
        vd_ok = vd_ok && all_finite(download(nl.reference.vD[d]));
    rep.check(vd_ok, "require_inlet_labels=false: vD finite at every vertex");
    rep.check(std::isfinite(nl.report.v_rms) && std::isfinite(nl.report.v1_diff_rms_rel) &&
                  std::isfinite(nl.report.spline_potential_gpu_vs_host_rel),
              "require_inlet_labels=false: v_rms and the (vi) diagnostic finite");
    // the Darcy part is independent of the option
    rep.check(bitwise_equal(nl.inlet_v1_samples, def.inlet_v1_samples) &&
                  std::memcmp(nl.G, def.G, sizeof(nl.G)) == 0,
              "inlet samples and G identical to the default-options stage");

    const sl::SlabSplineDirectionField f = nl.direction_field();
    bool fin = true;
    const real pts[3][3] = {{0.0, 0.3, 0.7}, {0.5, 0.11, 0.93}, {1.0, 0.5, 0.5}};
    for (const auto& p : pts) {
        double x[3] = {p[0], p[1], p[2]}, g[3], k;
        f(x, g, k);
        fin = fin && std::isfinite(g[0]) && std::isfinite(g[1]) && std::isfinite(g[2]) &&
              std::isfinite(k) && k > 0.0;
    }
    rep.check(fin, "require_inlet_labels=false: host direction field evaluates finitely");
}

void case_no_backflow(TestReport& rep, CudaContext& ctx) {
    sl::ProductionFieldSpec spec;
    spec.N = kN;
    spec.sigma2 = 1.0;
    spec.ell = 0.25;
    spec.seed = 3001ULL;
    const sl::SlabFieldSource src = sl::SlabFieldSource::gaussian(ctx, spec);
    const sl::ProductionStage a = build(ctx, src, 0.5, true);
    const sl::ProductionStage c = build(ctx, src, 0.5, false);
    rep.check(a.ok() && c.ok() && a.flow_ready() && c.flow_ready(),
              "gaussian 3001 eps 0.5: ok() and flow_ready() for both options");
    rep.check(a.labels_available && c.labels_available,
              "gaussian 3001 eps 0.5: labels_available == true for both options");
    rep.check(bitwise_equal(a.potential_coefficients, c.potential_coefficients),
              "potential coefficients bitwise identical");
    rep.check(bitwise_equal(a.logk_coefficients, c.logk_coefficients),
              "logk coefficients bitwise identical");
    bool vd = true, u0 = true;
    for (int d = 0; d < 3; ++d)
        vd = vd && bitwise_equal(download(a.reference.vD[d]), download(c.reference.vD[d]));
    for (int d = 0; d < 2; ++d)
        u0 = u0 && bitwise_equal(download(a.inputs.u0[d]), download(c.inputs.u0[d]));
    rep.check(vd, "vD bitwise identical");
    rep.check(u0, "inlet periodic parts u0 bitwise identical");
    rep.check(a.report.inlet_vmin == c.report.inlet_vmin && a.report.Q0 == c.report.Q0 &&
                  a.report.v_rms == c.report.v_rms,
              "inlet_vmin, Q0, v_rms identical");
}

// ---- SF-37 N2 (C-STAGE) -----------------------------------------------------------------------

sl::ProductionStage build_faces(CudaContext& ctx, const sl::SlabFieldSource& src, real eps,
                                bool keep, int pcg_max_iter = -1) {
    const sl::InletSlabGrid grid = sl::InletSlabGrid::make(kN);
    sl::ProductionStageOptions opt;
    opt.keep_face_fluxes = keep;
    opt.pcg_max_iter = pcg_max_iter;
    return sl::build_production_stage(ctx, grid, src, eps, opt);
}

bool same_bits(real a, real b) { return std::memcmp(&a, &b, sizeof(real)) == 0; }

bool report_identical(const sl::ProductionStageReport& a, const sl::ProductionStageReport& b) {
    bool ok = a.status == b.status && a.field == b.field && a.N == b.N &&
              a.mg_levels == b.mg_levels && a.darcy_converged == b.darcy_converged &&
              a.has_sf18 == b.has_sf18 && same_bits(a.eps, b.eps) &&
              same_bits(a.inlet_vmin, b.inlet_vmin) && same_bits(a.inlet_mean, b.inlet_mean) &&
              same_bits(a.Q0, b.Q0) && same_bits(a.v1_diff_rms_rel, b.v1_diff_rms_rel) &&
              same_bits(a.v1_diff_max_rel, b.v1_diff_max_rel) && same_bits(a.v_rms, b.v_rms) &&
              same_bits(a.spline_potential_gpu_vs_host_rel, b.spline_potential_gpu_vs_host_rel) &&
              same_bits(a.darcy.symmetry_defect_rel, b.darcy.symmetry_defect_rel) &&
              same_bits(a.darcy.div_max_abs, b.darcy.div_max_abs) &&
              same_bits(a.darcy.div_rms, b.darcy.div_rms);
    for (int d = 0; d < 3; ++d) {
        const auto& ca = a.darcy.corrector_results[d];
        const auto& cb = b.darcy.corrector_results[d];
        ok = ok && same_bits(a.darcy.G[d], b.darcy.G[d]) &&
             same_bits(a.darcy.achieved_mean_flux[d], b.darcy.achieved_mean_flux[d]) &&
             same_bits(a.darcy.eigenvalues_symmetric_part[d],
                       b.darcy.eigenvalues_symmetric_part[d]) &&
             ca.iterations == cb.iterations && ca.converged == cb.converged &&
             same_bits(ca.relative_projected_residual, cb.relative_projected_residual) &&
             same_bits(ca.final_projected_residual, cb.final_projected_residual);
        for (int i = 0; i < 3; ++i)
            ok = ok && same_bits(a.darcy.K_eff[i][d], b.darcy.K_eff[i][d]);
    }
    return ok && a.summary() == b.summary();
}

bool stage_outputs_identical(const sl::ProductionStage& a, const sl::ProductionStage& b) {
    bool ok = bitwise_equal(a.potential_coefficients, b.potential_coefficients) &&
              bitwise_equal(a.logk_coefficients, b.logk_coefficients) &&
              bitwise_equal(a.inlet_v1_samples, b.inlet_v1_samples) &&
              a.labels_available == b.labels_available &&
              std::memcmp(a.G, b.G, sizeof(a.G)) == 0 && same_bits(a.inputs.v_rms, b.inputs.v_rms) &&
              same_bits(a.inputs.eps, b.inputs.eps) && a.inputs.field == b.inputs.field &&
              a.inputs.N == b.inputs.N;
    for (int d = 0; d < 3; ++d)
        ok = ok && bitwise_equal(download(a.reference.vD[d]), download(b.reference.vD[d])) &&
             bitwise_equal(download(a.inputs.grad_lnk[d]), download(b.inputs.grad_lnk[d]));
    for (int d = 0; d < 2; ++d)
        ok = ok && bitwise_equal(download(a.inputs.u0[d]), download(b.inputs.u0[d])) &&
             bitwise_equal(download(a.inputs.vperp_in[d]), download(b.inputs.vperp_in[d]));
    ok = ok && bitwise_equal(download(a.inputs.q), download(b.inputs.q)) &&
         bitwise_equal(download(a.inputs.lnk), download(b.inputs.lnk));
    return ok && report_identical(a.report, b.report);
}

void check_faces(TestReport& rep, const sl::ProductionStage& on, const std::string& tag) {
    const std::size_t N = kN;
    rep.check(on.faces_available(), tag + ": faces_available() true");
    rep.check(on.face_flux_u.size() == (N + 1) * N * N && on.face_flux_v.size() == N * (N + 1) * N &&
                  on.face_flux_w.size() == N * N * (N + 1),
              tag + ": CompactMAC sizes (N+1)N^2, N(N+1)N, N^2(N+1)");
    bool dup_u = true, dup_v = true, dup_w = true;
    double su = 0.0, sv = 0.0, sw = 0.0;
    // SF-19 indexing: U i + j (N+1) + k (N+1) N; V i + j N + k N (N+1); W i + j N + k N N
    for (std::size_t k = 0; k < N; ++k)
        for (std::size_t j = 0; j < N; ++j)
            dup_u = dup_u && same_bits(on.face_flux_u[0 + j * (N + 1) + k * (N + 1) * N],
                                       on.face_flux_u[N + j * (N + 1) + k * (N + 1) * N]);
    for (std::size_t k = 0; k < N; ++k)
        for (std::size_t i = 0; i < N; ++i)
            dup_v = dup_v && same_bits(on.face_flux_v[i + 0 * N + k * N * (N + 1)],
                                       on.face_flux_v[i + N * N + k * N * (N + 1)]);
    for (std::size_t j = 0; j < N; ++j)
        for (std::size_t i = 0; i < N; ++i)
            dup_w = dup_w && same_bits(on.face_flux_w[i + j * N + 0 * N * N],
                                       on.face_flux_w[i + j * N + N * N * N]);
    for (std::size_t k = 0; k < N; ++k)
        for (std::size_t j = 0; j < N; ++j)
            for (std::size_t i = 0; i < N; ++i) {
                su += on.face_flux_u[i + j * (N + 1) + k * (N + 1) * N];
                sv += on.face_flux_v[i + j * N + k * N * (N + 1)];
                sw += on.face_flux_w[i + j * N + k * N * N];
            }
    const double nn = static_cast<double>(N * N * N);
    rep.check(dup_u && dup_v && dup_w,
              tag + ": duplicate planes (U i=N, V j=N, W k=N) == plane 0 bitwise");
    // inlet sample index m3 + N m2 with m2 = j, m3 = k
    bool inlet = on.inlet_v1_samples.size() == N * N;
    for (std::size_t k = 0; inlet && k < N; ++k)
        for (std::size_t j = 0; j < N; ++j)
            inlet = inlet && same_bits(on.face_flux_u[0 + j * (N + 1) + k * (N + 1) * N],
                                       on.inlet_v1_samples[k + N * j]);
    rep.check(inlet, tag + ": U plane 0 == inlet_v1_samples bitwise");
    char b[160];
    std::snprintf(b, sizeof(b), "mean U-1=%.3e mean V=%.3e mean W=%.3e", su / nn - 1.0, sv / nn,
                  sw / nn);
    rep.check(std::abs(su / nn - 1.0) < 1e-12 && std::abs(sv / nn) < 1e-12 &&
                  std::abs(sw / nn) < 1e-12,
              tag + ": unique-face means = (1, 0, 0) within 1e-12 (qbar = e1)", b);
}

void case_faces(TestReport& rep, CudaContext& ctx) {
    {
        const sl::SlabFieldSource src = sl::SlabFieldSource::analytic("generic3d", kN);
        const sl::ProductionStage off = build_faces(ctx, src, 0.25, false);
        rep.check(off.ok() && !off.faces_available() && off.face_flux_u.empty() &&
                      off.face_flux_v.empty() && off.face_flux_w.empty(),
                  "default options (keep_face_fluxes=false): faces_available() false, vectors empty");
        const sl::ProductionStage on = build_faces(ctx, src, 0.25, true);
        rep.check(on.ok(), "generic3d eps 0.25, keep_face_fluxes: stage ok");
        rep.check(stage_outputs_identical(off, on),
                  "generic3d eps 0.25: all other outputs bitwise identical, option off vs on");
        check_faces(rep, on, "generic3d eps 0.25");
    }
    {
        sl::ProductionFieldSpec spec;
        spec.N = kN;
        spec.sigma2 = 1.0;
        spec.ell = 0.25;
        spec.seed = 3001ULL;
        const sl::SlabFieldSource src = sl::SlabFieldSource::gaussian(ctx, spec);
        const sl::ProductionStage off = build_faces(ctx, src, 1.0, false);
        const sl::ProductionStage on = build_faces(ctx, src, 1.0, true);
        rep.check(off.ok() && on.ok() && !off.faces_available(),
                  "gaussian 3001 eps 1: ok, option off keeps no faces");
        rep.check(stage_outputs_identical(off, on),
                  "gaussian 3001 eps 1: all other outputs bitwise identical, option off vs on");
        check_faces(rep, on, "gaussian 3001 eps 1");
    }
    {
        // forced Darcy failure: one PCG iteration cannot reach rtol 1e-10
        const sl::SlabFieldSource src = sl::SlabFieldSource::analytic("generic3d", kN);
        const sl::ProductionStage bad = build_faces(ctx, src, 0.25, true, 1);
        rep.check(bad.report.status == sl::SlabProductionStatus::darcy_failed &&
                      !bad.report.darcy_converged,
                  "pcg_max_iter=1: status darcy_failed");
        rep.check(!bad.faces_available() && bad.face_flux_u.empty() && bad.face_flux_v.empty() &&
                      bad.face_flux_w.empty(),
                  "darcy_failed with keep_face_fluxes: faces not kept (host vectors empty)");
    }
}

} // namespace

int main() {
    TestReport rep;
    CudaContext ctx(0);
    const auto t0 = std::chrono::steady_clock::now();
    try {
        std::printf("=== SF-36 N2: (a) field with inlet backflow ===\n");
        case_backflow(rep, ctx);
        std::printf("=== SF-36 N2: (b) field without backflow ===\n");
        case_no_backflow(rep, ctx);
        std::printf("=== SF-37 N2: keep_face_fluxes (C-STAGE) ===\n");
        case_faces(rep, ctx);
    } catch (const std::exception& e) {
        std::printf("[FAIL] unexpected exception: %s\n", e.what());
        rep.overall_pass = false;
    }
    const double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("\n=== inlet_slab_stage_options: %d checks, %s (%.1f s) ===\n", rep.checks,
                rep.overall_pass ? "ALL PASS" : "FAILURES", t);
    return rep.overall_pass ? 0 : 1;
}
