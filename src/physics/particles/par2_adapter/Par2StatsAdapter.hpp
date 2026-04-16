#pragma once

/**
 * @file Par2StatsAdapter.hpp
 * @brief PIMPL wrapper around par2::StatsComputer (no Par2 types in header)
 *
 * Owns one StatsComputer with persistent pinned buffers.
 * Reset between realizations — no realloc.
 */

#include "../../../core/Grid3D.hpp"
#include "../stats_types.hpp"
#include "par2_views.hpp"
#include <cuda_runtime.h>
#include <memory>
#include <string>
#include <vector>

namespace macroflow3d {
namespace physics {
namespace particles {

class Par2StatsAdapter {
  public:
    Par2StatsAdapter(int max_particles, const Grid3D& grid, bool has_periodic_bc,
                     bool use_biased_var);
    ~Par2StatsAdapter();

    // Non-copyable, non-movable (owns pinned memory)
    Par2StatsAdapter(const Par2StatsAdapter&) = delete;
    Par2StatsAdapter& operator=(const Par2StatsAdapter&) = delete;

    /// Launch async stats computation (does NOT sync the stream)
    void sample_async(const ConstParticlesSoA<real>& particles, cudaStream_t stream);

    /// After stream sync: read result and append to time-series
    void store_sample(int step, real dt);

    /// Clear time-series for next realization (no buffer dealloc)
    void reset();

    const std::vector<StatsSample<real>>& samples() const;
    int num_samples() const;

    bool write_csv(const std::string& filename) const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace particles
} // namespace physics
} // namespace macroflow3d
