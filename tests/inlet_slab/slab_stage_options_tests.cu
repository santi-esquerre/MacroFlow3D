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
    } catch (const std::exception& e) {
        std::printf("[FAIL] unexpected exception: %s\n", e.what());
        rep.overall_pass = false;
    }
    const double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("\n=== inlet_slab_stage_options: %d checks, %s (%.1f s) ===\n", rep.checks,
                rep.overall_pass ? "ALL PASS" : "FAILURES", t);
    return rep.overall_pass ? 0 : 1;
}
