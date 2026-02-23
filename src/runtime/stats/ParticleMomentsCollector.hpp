#pragma once

/**
 * @file ParticleMomentsCollector.hpp
 * @brief Runtime stats collection: compute mean/var from particle views.
 * @ingroup runtime_stats
 *
 * This collector works on the device-side views exposed by the transport
 * adapter (ConstParticlesSoA / UnwrappedSoA). It delegates the actual GPU
 * reduction to Par2StatsAdapter, but this header has NO par2 dependency.
 *
 * HPC contract:
 *   - NO allocations in the hot loop.
 *   - Workspace is persistent — allocated once in construct/prepare.
 *   - sample_async launches GPU work without host sync.
 *   - store_sample reads back results after the caller's sync point.
 */

#include "../../io/writers/CsvTimeSeriesWriter.hpp"
#include "../../physics/particles/par2_adapter/par2_views.hpp"
#include "../../physics/particles/par2_adapter/Par2StatsAdapter.hpp"
#include "../../core/Scalar.hpp"
#include "../../core/Grid3D.hpp"

#include <cuda_runtime.h>
#include <vector>

namespace macroflow3d {
namespace runtime {

using namespace macroflow3d::physics::particles;

/**
 * @brief Collects particle moments (mean, variance) per step.
 *
 * Wraps Par2StatsAdapter for GPU reduction, but exposes results as
 * io::TimeSeriesPoint — decoupled from par2 types.
 */
class ParticleMomentsCollector {
public:
    ParticleMomentsCollector(int max_particles, const Grid3D& grid,
                             bool has_periodic_bc, bool use_biased_var)
        : adapter_(max_particles, grid, has_periodic_bc, use_biased_var)
    {}

    /// Launch async GPU reduction (does NOT synchronize)
    void sample_async(const ConstParticlesSoA<real>& particles,
                      cudaStream_t stream) {
        adapter_.sample_async(particles, stream);
    }

    /// After stream sync: fetch result and convert to TimeSeriesPoint
    bool store_sample(int step, real dt, io::TimeSeriesPoint<real>& out) {
        // Adapter stores internally — we retrieve and convert
        int prev_count = adapter_.num_samples();
        adapter_.store_sample(step, dt);

        if (adapter_.num_samples() > prev_count) {
            const auto& s = adapter_.samples().back();
            out.time   = s.time;
            out.active = s.active;
            for (int k = 0; k < 3; ++k) {
                out.mean[k] = s.mean[k];
                out.var[k]  = s.var[k];
            }
            return true;
        }
        return false;
    }

    /// Clear for next realization (no realloc)
    void reset() { adapter_.reset(); }

    int num_samples() const { return adapter_.num_samples(); }

private:
    Par2StatsAdapter adapter_;
};

} // namespace runtime
} // namespace macroflow3d
