#include "src/core/Grid3D.hpp"
#include "src/core/Scalar.hpp"
#include "src/physics/common/fields.cuh"
#include "src/physics/particles/pspta/invariants/GaugeFixer.cuh"
#include "src/physics/particles/pspta/invariants/PsptaInvariantField.cuh"
#include "src/physics/particles/pspta/invariants/RefinementAC.cuh"
#include "src/runtime/CudaContext.cuh"

#include <cassert>
#include <cmath>
#include <vector>

using namespace macroflow3d;
using namespace macroflow3d::physics;
using namespace macroflow3d::physics::particles::pspta;

namespace {

void fill_uniform_x_velocity(const Grid3D& grid, VelocityField& vel) {
    std::vector<real> hU(vel.size_U(), real(1));
    std::vector<real> hV(vel.size_V(), real(0));
    std::vector<real> hW(vel.size_W(), real(0));
    cudaMemcpy(vel.U.data(), hU.data(), hU.size() * sizeof(real), cudaMemcpyHostToDevice);
    cudaMemcpy(vel.V.data(), hV.data(), hV.size() * sizeof(real), cudaMemcpyHostToDevice);
    cudaMemcpy(vel.W.data(), hW.data(), hW.size() * sizeof(real), cudaMemcpyHostToDevice);
}

void fill_perturbed_uniform_gauge(const Grid3D& grid, PsptaInvariantField& inv) {
    const int nx = grid.nx;
    const int ny = grid.ny;
    const int nz = grid.nz;
    const size_t n = static_cast<size_t>(nx) * ny * nz;

    std::vector<float> psi1(n, 0.0f);
    std::vector<float> psi2(n, 0.0f);

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                const size_t idx = static_cast<size_t>(i) +
                                   static_cast<size_t>(nx) *
                                       (static_cast<size_t>(j) + static_cast<size_t>(ny) * k);
                const double x = (static_cast<double>(i) + 0.5) * static_cast<double>(grid.dx);
                const double y = (static_cast<double>(j) + 0.5) * static_cast<double>(grid.dy);
                const double z = (static_cast<double>(k) + 0.5) * static_cast<double>(grid.dz);
                psi1[idx] = static_cast<float>(y + 0.10 * std::sin(2.0 * M_PI * x));
                psi2[idx] = static_cast<float>(z);
            }
        }
    }

    cudaMemcpy(inv.psi1_ptr(), psi1.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(inv.psi2_ptr(), psi2.data(), n * sizeof(float), cudaMemcpyHostToDevice);
}

} // namespace

int main() {
    CudaContext ctx(0);

    constexpr int N = 12;
    constexpr real dx = real(1.0 / N);
    Grid3D grid(N, N, N, dx, dx, dx);
    VelocityField vel(grid);
    fill_uniform_x_velocity(grid, vel);

    PsptaInvariantField inv;
    inv.resize(grid);
    fill_perturbed_uniform_gauge(grid, inv);

    const InvariantQualityReport initial_quality = inv.compute_quality(vel, ctx.cuda_stream());
    assert(initial_quality.valid);

    RefinementACConfig cfg;
    cfg.enabled = true;
    cfg.max_iterations = 2;
    cfg.omega = 1.0;
    cfg.max_backtracks = 8;
    cfg.stop_rel_quality = 1.0e-4;
    cfg.stop_abs_quality = 0.0;

    RefinementAC refinement(grid, &vel, cfg);
    GaugeFixerConfig gf_cfg;
    gf_cfg.method = GaugeMethod::None;
    refinement.set_gauge_fixer(std::make_unique<GaugeFixer>(gf_cfg));

    const RefinementACReport report = refinement.refine(inv, ctx);

    assert(report.enabled);
    assert(report.stop_reason != "not_implemented");
    assert(report.initial_quality.valid);
    assert(report.final_quality.valid);
    assert(report.iterations_done > 0 || report.converged);
    assert(report.final_quality.cross_product.rel_rms_mismatch <
           report.initial_quality.cross_product.rel_rms_mismatch);

    const double initial_invariance =
        report.initial_quality.invariance.rms_r1 + report.initial_quality.invariance.rms_r2;
    const double final_invariance =
        report.final_quality.invariance.rms_r1 + report.final_quality.invariance.rms_r2;
    assert(final_invariance <= initial_invariance * 1.25);

    return 0;
}
