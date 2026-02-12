/**
 * @file Par2StatsAdapter.cu
 * @brief PIMPL implementation — wraps par2::StatsComputer
 */

#include "Par2StatsAdapter.hpp"
#include "par2_mapping.cuh"
#include <cstdio>

namespace macroflow3d {
namespace physics {
namespace particles {

using namespace detail;

// ============================================================================
// PIMPL body
// ============================================================================

struct Par2StatsAdapter::Impl {
    par2::StatsComputer<real> stats;
    par2::StatsConfig         scfg;
    par2::GridDesc<real>      grid;
    bool has_periodic;
    bool use_biased;
    std::vector<StatsSample<real>> samples;

    Impl(int max_particles, const par2::GridDesc<real>& g,
         bool periodic, bool biased)
        : stats(max_particles), grid(g)
        , has_periodic(periodic), use_biased(biased)
    {
        scfg.use_unwrapped      = periodic;
        scfg.filter_active_only = true;
    }
};

// ============================================================================
// Public API
// ============================================================================

Par2StatsAdapter::Par2StatsAdapter(int max_particles, const Grid3D& grid,
                                   bool has_periodic_bc, bool use_biased_var)
    : impl_(std::make_unique<Impl>(
          max_particles, make_par2_grid(grid), has_periodic_bc, use_biased_var))
{}

Par2StatsAdapter::~Par2StatsAdapter() = default;

void Par2StatsAdapter::sample_async(const ConstParticlesSoA<real>& particles,
                                    cudaStream_t stream) {
    auto cpv = to_par2_const(particles);
    impl_->stats.compute_async(cpv, impl_->grid, impl_->scfg, stream);
}

void Par2StatsAdapter::store_sample(int step, real dt) {
    auto res = impl_->stats.fetch_result();
    if (!res.computed) return;

    StatsSample<real> s;
    s.time   = step * dt;
    s.active = impl_->scfg.filter_active_only ? res.counts.active : res.counts.total;

    for (int k = 0; k < 3; ++k) {
        s.mean[k] = res.moments.mean[k];
        real v = res.moments.var[k];
        if (impl_->use_biased && s.active > 1) {
            v = v * real(s.active - 1) / real(s.active);
        }
        s.var[k] = v;
    }
    impl_->samples.push_back(s);
}

void Par2StatsAdapter::reset() {
    impl_->samples.clear();
}

const std::vector<StatsSample<real>>& Par2StatsAdapter::samples() const {
    return impl_->samples;
}

int Par2StatsAdapter::num_samples() const {
    return static_cast<int>(impl_->samples.size());
}

bool Par2StatsAdapter::write_csv(const std::string& filename) const {
    FILE* f = std::fopen(filename.c_str(), "w");
    if (!f) return false;
    std::fprintf(f, "t,mean_x,mean_y,mean_z,var_x,var_y,var_z,active\n");
    for (const auto& s : impl_->samples) {
        std::fprintf(f, "%.15e,%.15e,%.15e,%.15e,%.15e,%.15e,%.15e,%d\n",
                     (double)s.time,
                     (double)s.mean[0], (double)s.mean[1], (double)s.mean[2],
                     (double)s.var[0],  (double)s.var[1],  (double)s.var[2],
                     s.active);
    }
    std::fclose(f);
    return true;
}

} // namespace particles
} // namespace physics
} // namespace macroflow3d
