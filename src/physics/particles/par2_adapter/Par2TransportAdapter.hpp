#pragma once

/**
 * @file Par2TransportAdapter.hpp
 * @brief PIMPL wrapper around par2::TransportEngine (no Par2 types in header)
 * @ingroup physics_particles
 *
 * Owns the engine for ONE realization.  Construct, bind, step, destroy.
 * Hot path (`step(dt)`) goes through a single pointer indirection — negligible
 * compared to the GPU kernel launch it triggers.
 */

#include "par2_views.hpp"
#include "../../../core/Grid3D.hpp"
#include "../../../core/BCSpec.hpp"
#include "../../common/fields.cuh"
#include <cuda_runtime.h>
#include <memory>

namespace macroflow3d {
namespace physics {
namespace particles {

class Par2TransportAdapter {
public:
    Par2TransportAdapter(const Grid3D& grid,
                         const BCSpec& bc,
                         const TransportAdapterConfig& cfg,
                         cudaStream_t stream);
    ~Par2TransportAdapter();

    // Movable, non-copyable
    Par2TransportAdapter(Par2TransportAdapter&&) noexcept;
    Par2TransportAdapter& operator=(Par2TransportAdapter&&) noexcept;
    Par2TransportAdapter(const Par2TransportAdapter&) = delete;
    Par2TransportAdapter& operator=(const Par2TransportAdapter&) = delete;

    /// Bind velocity field (zero-copy; field must outlive the adapter)
    void bind_velocity(const PaddedVelocityField& vel);

    /// Bind particle arrays (non-owning; buffers must outlive the adapter)
    void bind_particles(ParticlesSoA<real>& p);

    /// Inject particles in a box region
    void inject_box(real x0, real y0, real z0,
                    real x1, real y1, real z1,
                    int first, int count);

    void ensure_tracking();
    void prepare();

    /// Advance one time step (async, allocation-free)
    void step(real dt);

    void synchronize();

    /// Read-only view of current particle state
    ConstParticlesSoA<real> particles() const;

    /// Compute unwrapped positions into caller-provided buffers
    void compute_unwrapped(UnwrappedSoA<real>& uw, cudaStream_t stream);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace particles
} // namespace physics
} // namespace macroflow3d
