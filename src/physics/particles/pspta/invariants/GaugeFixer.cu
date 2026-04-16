/**
 * @file GaugeFixer.cu
 * @brief Implementation of gauge fixing for invariant fields.
 */

#include "../../../../runtime/cuda_check.cuh"
#include "GaugeFixer.cuh"

namespace macroflow3d {
namespace physics {
namespace particles {
namespace pspta {

// ============================================================================
// Kernel: Apply inlet gauge
// ============================================================================

/**
 * @brief Set psi1 = y, psi2 = z at inlet plane (i=0).
 */
__global__ void kernel_apply_inlet_gauge(float* __restrict__ psi1, float* __restrict__ psi2, int nx,
                                         int ny, int nz, double dy, double dz) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= ny || k >= nz)
        return;

    const int idx = 0 + nx * (j + ny * k); // i = 0

    psi1[idx] = static_cast<float>((j + 0.5) * dy);
    psi2[idx] = static_cast<float>((k + 0.5) * dz);
}

// ============================================================================
// GaugeFixer implementation
// ============================================================================

GaugeFixer::GaugeFixer(const GaugeFixerConfig& config) : config_(config) {}

void GaugeFixer::apply(PsptaInvariantField& inv, const VelocityField& vel, cudaStream_t stream) {
    switch (config_.method) {
    case GaugeMethod::None:
        break;
    case GaugeMethod::InletPlane:
        apply_inlet_gauge(inv, stream);
        break;
    case GaugeMethod::MeanZero:
        // TODO: Implement mean subtraction
        break;
    case GaugeMethod::ScaledPeriodic:
        // TODO: Implement periodic scaling
        break;
    }
}

void GaugeFixer::apply_inlet_gauge(PsptaInvariantField& inv, cudaStream_t stream) {
    if (!inv.is_valid())
        return;

    const int ny = inv.ny();
    const int nz = inv.nz();

    const dim3 block(16, 16);
    const dim3 grid((ny + 15) / 16, (nz + 15) / 16);

    kernel_apply_inlet_gauge<<<grid, block, 0, stream>>>(inv.psi1_ptr(), inv.psi2_ptr(), inv.nx(),
                                                         inv.ny(), inv.nz(), inv.dy(), inv.dz());
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
}

} // namespace pspta
} // namespace particles
} // namespace physics
} // namespace macroflow3d
