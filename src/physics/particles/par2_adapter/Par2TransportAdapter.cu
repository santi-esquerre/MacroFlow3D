/**
 * @file Par2TransportAdapter.cu
 * @brief PIMPL implementation — wraps par2::TransportEngine
 */

#include "Par2TransportAdapter.hpp"
#include "par2_mapping.cuh"

namespace macroflow3d {
namespace physics {
namespace particles {

using namespace detail;

// ============================================================================
// PIMPL body
// ============================================================================

struct Par2TransportAdapter::Impl {
    par2::TransportEngine<real> engine;
    par2::GridDesc<real>        grid;

    Impl(const par2::GridDesc<real>& g,
         const par2::TransportParams<real>& tp,
         const par2::BoundaryConfig<real>& bc,
         const par2::EngineConfig& ecfg)
        : engine(g, tp, bc, ecfg), grid(g) {}
};

// ============================================================================
// Public API
// ============================================================================

Par2TransportAdapter::Par2TransportAdapter(
    const Grid3D& grid,
    const BCSpec& bc,
    const TransportAdapterConfig& cfg,
    cudaStream_t stream)
{
    auto p2_grid = make_par2_grid(grid);
    auto p2_bc   = make_par2_bc(bc);

    par2::TransportParams<real> tp;
    tp.molecular_diffusion = cfg.molecular_diffusion;
    tp.alpha_l             = cfg.alpha_l;
    tp.alpha_t             = cfg.alpha_t;

    par2::EngineConfig ecfg;
    ecfg.interpolation_mode = cfg.linear_interpolation
                                  ? par2::InterpolationMode::Linear
                                  : par2::InterpolationMode::Trilinear;
    ecfg.drift_mode = cfg.has_dispersion()
                          ? par2::DriftCorrectionMode::TrilinearOnFly
                          : par2::DriftCorrectionMode::None;
    ecfg.rng_seed = cfg.rng_seed;

    impl_ = std::make_unique<Impl>(p2_grid, tp, p2_bc, ecfg);
    impl_->engine.set_stream(stream);
}

Par2TransportAdapter::~Par2TransportAdapter() = default;
Par2TransportAdapter::Par2TransportAdapter(Par2TransportAdapter&&) noexcept = default;
Par2TransportAdapter& Par2TransportAdapter::operator=(Par2TransportAdapter&&) noexcept = default;

void Par2TransportAdapter::bind_velocity(const PaddedVelocityField& vel) {
    auto vv = make_velocity_view(vel);
    impl_->engine.bind_velocity(vv);
}

void Par2TransportAdapter::bind_particles(ParticlesSoA<real>& p) {
    auto pv = to_par2(p);
    impl_->engine.bind_particles(pv);
}

void Par2TransportAdapter::inject_box(real x0, real y0, real z0,
                                      real x1, real y1, real z1,
                                      int first, int count) {
    impl_->engine.inject_box(x0, y0, z0, x1, y1, z1, first, count);
}

void Par2TransportAdapter::ensure_tracking() {
    impl_->engine.ensure_tracking_arrays();
}

void Par2TransportAdapter::prepare() {
    impl_->engine.prepare();
}

void Par2TransportAdapter::step(real dt) {
    impl_->engine.step(dt);
}

void Par2TransportAdapter::synchronize() {
    impl_->engine.synchronize();
}

ConstParticlesSoA<real> Par2TransportAdapter::particles() const {
    return from_par2(impl_->engine.particles());
}

void Par2TransportAdapter::compute_unwrapped(UnwrappedSoA<real>& uw,
                                             cudaStream_t stream) {
    auto p2_uw = to_par2(uw);
    impl_->engine.compute_unwrapped_positions(p2_uw, stream);
}

} // namespace particles
} // namespace physics
} // namespace macroflow3d
