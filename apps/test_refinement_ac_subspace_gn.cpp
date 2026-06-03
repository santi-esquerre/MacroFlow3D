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

std::vector<std::vector<float>> make_uniform_subspace_modes(const Grid3D& grid) {
    const int nx = grid.nx;
    const int ny = grid.ny;
    const int nz = grid.nz;
    const size_t n = static_cast<size_t>(nx) * ny * nz;
    std::vector<std::vector<float>> modes(4, std::vector<float>(n, 0.0f));

    double mean_y2 = 0.0;
    double mean_z2 = 0.0;
    for (int j = 0; j < ny; ++j) {
        const double y = (static_cast<double>(j) + 0.5) * static_cast<double>(grid.dy);
        mean_y2 += y * y;
    }
    mean_y2 /= static_cast<double>(ny);
    for (int k = 0; k < nz; ++k) {
        const double z = (static_cast<double>(k) + 0.5) * static_cast<double>(grid.dz);
        mean_z2 += z * z;
    }
    mean_z2 /= static_cast<double>(nz);

    for (int k = 0; k < nz; ++k) {
        const double z = (static_cast<double>(k) + 0.5) * static_cast<double>(grid.dz);
        for (int j = 0; j < ny; ++j) {
            const double y = (static_cast<double>(j) + 0.5) * static_cast<double>(grid.dy);
            for (int i = 0; i < nx; ++i) {
                const size_t idx = static_cast<size_t>(i) +
                                   static_cast<size_t>(nx) *
                                       (static_cast<size_t>(j) + static_cast<size_t>(ny) * k);
                modes[0][idx] = static_cast<float>(y);
                modes[1][idx] = static_cast<float>(z);
                modes[2][idx] = static_cast<float>(y * y - mean_y2);
                modes[3][idx] = static_cast<float>(z * z - mean_z2);
            }
        }
    }

    return modes;
}

void fill_bad_scaled_gauge(const Grid3D& grid, PsptaInvariantField& inv) {
    const auto modes = make_uniform_subspace_modes(grid);
    const size_t n = modes[0].size();
    std::vector<float> psi1(n, 0.0f);
    std::vector<float> psi2(n, 0.0f);
    for (size_t idx = 0; idx < n; ++idx) {
        psi1[idx] = 1.20f * modes[0][idx] + 0.05f * modes[2][idx];
        psi2[idx] = 0.70f * modes[1][idx] - 0.03f * modes[3][idx];
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
    fill_bad_scaled_gauge(grid, inv);

    const InvariantQualityReport initial_quality = inv.compute_quality(vel, ctx.cuda_stream());
    assert(initial_quality.valid);
    assert(initial_quality.cross_product.rel_rms_mismatch > 1.0e-2);

    RefinementACConfig cfg;
    cfg.enabled = true;
    cfg.strategy = RefinementACStrategy::SubspaceQuadraticGaussNewton;
    cfg.max_iterations = 5;
    cfg.max_backtracks = 6;
    cfg.stop_rel_quality = 1.0e-4;
    cfg.stop_abs_quality = 0.0;

    RefinementAC refinement(grid, &vel, cfg);
    GaugeFixerConfig gf_cfg;
    gf_cfg.method = GaugeMethod::None;
    refinement.set_gauge_fixer(std::make_unique<GaugeFixer>(gf_cfg));
    refinement.set_subspace_basis_host(make_uniform_subspace_modes(grid));

    const RefinementACReport report = refinement.refine(inv, ctx);

    assert(report.enabled);
    assert(report.stop_reason != "invalid_subspace");
    assert(report.stop_reason != "not_implemented");
    assert(report.initial_quality.valid);
    assert(report.final_quality.valid);
    assert(report.iterations_done > 0 || report.converged);
    assert(report.final_quality.cross_product.rel_rms_mismatch <
           report.initial_quality.cross_product.rel_rms_mismatch * 0.6);

    const double initial_invariance =
        report.initial_quality.invariance.rms_r1 + report.initial_quality.invariance.rms_r2;
    const double final_invariance =
        report.final_quality.invariance.rms_r1 + report.final_quality.invariance.rms_r2;
    assert(final_invariance <= initial_invariance * 1.10 + 1.0e-10);

    return 0;
}
