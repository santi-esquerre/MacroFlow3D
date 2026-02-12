/**
 * @file Par2SnapshotAdapter.cu
 * @brief PIMPL implementation — wraps par2::io::CsvSnapshotWriter
 */

#include "Par2SnapshotAdapter.hpp"
#include "par2_mapping.cuh"

namespace macroflow3d {
namespace physics {
namespace particles {

using namespace detail;

// ============================================================================
// PIMPL body
// ============================================================================

struct Par2SnapshotAdapter::Impl {
    par2::io::CsvSnapshotWriter<real> writer;

    Impl(int max_particles, const par2::io::CsvSnapshotConfig& cfg)
        : writer(max_particles, cfg) {}
};

// ============================================================================
// Public API
// ============================================================================

Par2SnapshotAdapter::Par2SnapshotAdapter(int max_particles,
                                         const SnapshotWriterConfig& cfg)
    : impl_(std::make_unique<Impl>(max_particles, to_par2(cfg)))
{}

Par2SnapshotAdapter::~Par2SnapshotAdapter() = default;
Par2SnapshotAdapter::Par2SnapshotAdapter(Par2SnapshotAdapter&&) noexcept = default;
Par2SnapshotAdapter& Par2SnapshotAdapter::operator=(Par2SnapshotAdapter&&) noexcept = default;

bool Par2SnapshotAdapter::write_snapshot(
    const ConstParticlesSoA<real>& particles,
    const char* filename,
    real time,
    cudaStream_t stream,
    const UnwrappedSoA<real>* unwrapped)
{
    auto cpv = to_par2_const(particles);

    if (unwrapped && unwrapped->valid()) {
        // Need a mutable copy for to_par2 — but the view is only read
        UnwrappedSoA<real> uw_copy = *unwrapped;
        auto p2_uw = to_par2(uw_copy);
        return impl_->writer.write_snapshot(cpv, filename, time, stream, &p2_uw);
    }
    return impl_->writer.write_snapshot(cpv, filename, time, stream, nullptr);
}

} // namespace particles
} // namespace physics
} // namespace macroflow3d
