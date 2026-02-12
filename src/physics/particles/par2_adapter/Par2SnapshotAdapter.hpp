#pragma once

/**
 * @file Par2SnapshotAdapter.hpp
 * @brief PIMPL wrapper around par2::io::CsvSnapshotWriter (no Par2 in header)
 *
 * Owns one writer with persistent pinned buffers.
 * Reuse across realizations — no realloc.
 */

#include "par2_views.hpp"
#include "../../../core/Scalar.hpp"
#include <cuda_runtime.h>
#include <memory>
#include <string>

namespace macroflow3d {
namespace physics {
namespace particles {

class Par2SnapshotAdapter {
public:
    Par2SnapshotAdapter(int max_particles, const SnapshotWriterConfig& cfg);
    ~Par2SnapshotAdapter();

    // Non-copyable (pinned memory ownership), movable
    Par2SnapshotAdapter(Par2SnapshotAdapter&&) noexcept;
    Par2SnapshotAdapter& operator=(Par2SnapshotAdapter&&) noexcept;
    Par2SnapshotAdapter(const Par2SnapshotAdapter&) = delete;
    Par2SnapshotAdapter& operator=(const Par2SnapshotAdapter&) = delete;

    bool write_snapshot(const ConstParticlesSoA<real>& particles,
                        const char* filename,
                        real time,
                        cudaStream_t stream,
                        const UnwrappedSoA<real>* unwrapped = nullptr);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace particles
} // namespace physics
} // namespace macroflow3d
