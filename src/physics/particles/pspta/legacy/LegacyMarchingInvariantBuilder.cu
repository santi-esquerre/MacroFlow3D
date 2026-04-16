/**
 * @file LegacyMarchingInvariantBuilder.cu
 * @brief Implementation of legacy marching wrapper.
 */

#include "../../../../runtime/cuda_check.cuh"
#include "LegacyMarchingInvariantBuilder.cuh"
#include <chrono>

namespace macroflow3d {
namespace physics {
namespace particles {
namespace pspta {

LegacyMarchingInvariantBuilder::LegacyMarchingInvariantBuilder(const LegacyMarchingConfig& cfg)
    : config_(cfg) {}

LegacyMarchingReport LegacyMarchingInvariantBuilder::build(PsptaInvariantField& out,
                                                           const VelocityField& vel,
                                                           const Grid3D& grid,
                                                           cudaStream_t stream) {
    auto start = std::chrono::high_resolution_clock::now();

    LegacyMarchingReport report;

    // Resize output field
    out.resize(grid);

    // Use legacy PsptaPsiField to compute psi1, psi2
    legacy_.resize(grid);
    report.precompute = legacy_.precompute_levelA(vel, grid, stream, config_.eps_vx);

    // Optional refinement
    if (config_.enable_refine) {
        report.refine = legacy_.refine_psi(vel, grid, stream, config_.refine_cfg);
    }

    // Copy from legacy field to new interface
    // psi1, psi2 are the same layout (float32 cell-centered), so direct copy
    const size_t n = out.num_cells();
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(out.psi1_ptr(), legacy_.psi1_ptr(), n * sizeof(float),
                                           cudaMemcpyDeviceToDevice, stream));
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(out.psi2_ptr(), legacy_.psi2_ptr(), n * sizeof(float),
                                           cudaMemcpyDeviceToDevice, stream));

    MACROFLOW3D_CUDA_CHECK(cudaStreamSynchronize(stream));

    // Set construction metadata
    InvariantConstructionInfo info;
    info.method = InvariantConstructionMethod::LegacyMarching;
    if (config_.enable_refine) {
        info.refinement_iterations = report.refine.iters_done;
        info.refinement_omega = config_.refine_cfg.omega;
        info.refinement_stop_reason = report.refine.stop_reason;
    }
    out.set_construction_info(info);

    // Compute quality metrics
    auto quality = out.compute_quality(vel, stream);
    out.set_quality(quality);

    auto end = std::chrono::high_resolution_clock::now();
    report.total_time_ms = std::chrono::duration<double, std::milli>(end - start).count();

    return report;
}

// ============================================================================
// InvariantFieldAdapter implementation
// ============================================================================

InvariantFieldAdapter::InvariantFieldAdapter(PsptaInvariantField* inv) : inv_(inv) {}

} // namespace pspta
} // namespace particles
} // namespace physics
} // namespace macroflow3d
