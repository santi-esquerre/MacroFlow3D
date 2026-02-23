#pragma once

/**
 * @file par2_views.hpp
 * @brief Par2-agnostic view types for particle data (device pointers)
 * @ingroup physics_particles
 *
 * These POD structs mirror par2::ParticlesView / ConstParticlesView /
 * UnwrappedPositionsView without introducing a dependency on Par2_Core.
 * Inside the adapter .cu files, trivial field-by-field conversion is used.
 */

#include "../../../core/Scalar.hpp"
#include <cstdint>

namespace macroflow3d {
namespace physics {
namespace particles {

/// Mutable SoA view of particle arrays (device pointers, non-owning)
template <typename T>
struct ParticlesSoA {
    T* x = nullptr;
    T* y = nullptr;
    T* z = nullptr;
    int n = 0;
    uint8_t* status = nullptr;
    int32_t* wrapX = nullptr;
    int32_t* wrapY = nullptr;
    int32_t* wrapZ = nullptr;
};

/// Const SoA view of particle arrays (device pointers, non-owning)
template <typename T>
struct ConstParticlesSoA {
    const T* x = nullptr;
    const T* y = nullptr;
    const T* z = nullptr;
    int n = 0;
    const uint8_t* status = nullptr;
    const int32_t* wrapX = nullptr;
    const int32_t* wrapY = nullptr;
    const int32_t* wrapZ = nullptr;
};

/// Device-side unwrapped-position buffers (non-owning)
template <typename T>
struct UnwrappedSoA {
    T* x_u = nullptr;
    T* y_u = nullptr;
    T* z_u = nullptr;
    int capacity = 0;
    bool valid() const { return x_u && y_u && z_u && capacity > 0; }
};

/// Configuration for snapshot writer (par2-agnostic)
struct SnapshotWriterConfig {
    bool legacy_format       = true;
    bool include_time        = false;
    bool include_status      = false;
    bool include_wrap_counts = false;
    bool include_unwrapped   = false;
    int  stride              = 1;
    int  max_particles       = -1;   // -1 = no limit
    int  precision           = 15;
};

/// Configuration for transport engine (par2-agnostic)
struct TransportAdapterConfig {
    real     molecular_diffusion = 0.0;
    real     alpha_l             = 0.0;
    real     alpha_t             = 0.0;
    bool     linear_interpolation = true;
    uint64_t rng_seed            = 0;

    /// True if any form of dispersion is active
    bool has_dispersion() const {
        return molecular_diffusion > 0 || alpha_l > 0 || alpha_t > 0;
    }
};

} // namespace particles
} // namespace physics
} // namespace macroflow3d
