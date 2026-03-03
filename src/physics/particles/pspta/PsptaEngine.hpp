#pragma once

/**
 * @file PsptaEngine.hpp
 * @brief PSPTA transport engine — Tier-2 pseudo-symplectic particle tracking.
 *
 * Advances particles by enforcing two streamline invariants ψ1, ψ2 per step:
 *   - x advanced with RK2 (midpoint) in time using CompactMAC vx.
 *   - (y,z) projected via 2×2 Newton after each x sub-step to keep
 *     ψ1(x,y,z) = ψ1_const[p]  and  ψ2(x,y,z) = ψ2_const[p].
 *
 * The invariants ψ1, ψ2 are precomputed by PsptaPsiField::precompute_levelA()
 * BEFORE constructing this engine.  The engine stores (psi1_const, psi2_const)
 * per particle during prepare() as trilinear samples of the ψ fields at the
 * initial injection positions.
 *
 * @par Interface contract (mirrors Par2TransportAdapter)
 * @code
 *   PsptaEngine eng(grid, stream, inject_seed);
 *   eng.bind_velocity(&vel);
 *   eng.bind_psifield(&psi);
 *   eng.bind_particles(pv);
 *   eng.inject_box(x0, 0, 0, x0, Ly, Lz, 0, NP);
 *   eng.ensure_tracking();
 *   eng.prepare();
 *   for (int s = 1; s <= n_steps; ++s) {
 *       eng.step(dt);            // async, allocation-free
 *       ...
 *       eng.synchronize();
 *   }
 *   eng.compute_unwrapped(uw, stream);
 * @endcode
 *
 * @par Hot-loop guarantee
 * step() launches kernels only.  No host allocations occur inside step().
 * All per-particle buffers are preallocated in prepare().
 *
 * @ingroup physics_particles_pspta
 */

#include "../par2_adapter/par2_views.hpp"  // ParticlesSoA / ConstParticlesSoA / UnwrappedSoA
#include "PsptaPsiField.cuh"          // PsptaPsiField, PsptaPrecomputeReport
#include "../../../core/DeviceBuffer.cuh"
#include "../../../core/Grid3D.hpp"
#include "../../common/fields.cuh"    // VelocityField (CompactMAC)
#include <cuda_runtime.h>
#include <cstdint>

namespace macroflow3d {
namespace physics {
namespace particles {
namespace pspta {

// ============================================================================
// Newton solver parameters (compile-time tuning)
// ============================================================================

/// Maximum Newton iterations per call to NewtonSolveYZ.
inline constexpr int    PSPTA_N_NEWTON      = 4;

/// Newton convergence: tolerance = PSPTA_TOL_FACTOR * fmin(dy, dz).
/// 1e-4 is appropriate for float32 ψ storage (float resolution ~ 1e-4 * cell).
inline constexpr double PSPTA_TOL_FACTOR    = 1e-4;

/// Minimum |det(J)| before declaring Newton ill-conditioned.
inline constexpr double PSPTA_DET_MIN       = 1e-14;

/// Trust-region clamp: maximum Newton step as a fraction of cell spacing.
inline constexpr double PSPTA_TRUST_FACTOR  = 0.5;

/// Newton step scaling (damping).  1.0 = full Newton step.
/// Trust-region already bounds steps, so 1.0 is recommended.
inline constexpr double PSPTA_DAMPING       = 1.0;

/// Number of histogram bins for Newton fail-count distribution.
inline constexpr int PSPTA_FAIL_HIST_BINS = 7;

/// Particle status set when a particle exits the domain in OPEN_X mode.
/// Any non-zero status is treated as inactive; 2 is used here to distinguish
/// a deliberate domain exit from other deactivation reasons.
inline constexpr uint8_t PSPTA_STATUS_EXITED = 2;

// ============================================================================
// PsptaEngine
// ============================================================================

/**
 * @brief Tier-2 PSPTA particle transport engine.
 *
 * Non-PIMPL concrete class; movable but not copyable.
 */
class PsptaEngine {
public:
    /**
     * @brief Construct engine with fixed grid geometry.
     *
     * @param grid         Cell-centered 3D grid (matches velocity field dims).
     * @param stream       CUDA stream for all async launches.
     * @param inject_seed  Base seed for deterministic injection hashing.
     */
    PsptaEngine(const Grid3D& grid,
                cudaStream_t  stream,
                uint64_t      inject_seed = 0ULL);

    ~PsptaEngine() = default;

    // Non-copyable, movable
    PsptaEngine(PsptaEngine&&) = default;
    PsptaEngine& operator=(PsptaEngine&&) = default;
    PsptaEngine(const PsptaEngine&) = delete;
    PsptaEngine& operator=(const PsptaEngine&) = delete;

    // ── Binding (zero-copy; objects must outlive engine) ──────────────────────

    /// Bind CompactMAC velocity field (owned externally).
    void bind_velocity(const VelocityField* vel);

    /// Bind ψ fields (must already have precompute_levelA called).
    void bind_psifield(const PsptaPsiField* psi);

    /// Bind particle arrays (device SoA, non-owning view).
    void bind_particles(ParticlesSoA<real>& p);

    // ── Lifecycle ─────────────────────────────────────────────────────────────

    /**
     * @brief Inject particles uniformly inside an axis-aligned box [p0,p1].
     *
     * For plane injection (e.g. x0==x1), all particles land on that x plane
     * with (y,z) uniformly sampled in [y0,y1] × [z0,z1].
     * Sets status[i] = 0 (active) and zeros wrap counters for injected range.
     *
     * @param x0,y0,z0  Min corner of injection box.
     * @param x1,y1,z1  Max corner of injection box.
     * @param first      First particle index to initialize.
     * @param count      Number of particles to initialize.
     */
    void inject_box(real x0, real y0, real z0,
                    real x1, real y1, real z1,
                    int first, int count);

    /**
     * @brief No-op — wrap arrays are owned by the caller's ParticlesSoA.
     * Provided for interface compatibility with Par2TransportAdapter callers.
     */
    void ensure_tracking();

    /**
     * @brief Allocate/resize per-particle invariant buffers and initialize them.
     *
     * Must be called ONCE after inject_box and before the hot loop.
     * Allowed to allocate device memory (not in hot path).
     *
     * Computes for each active particle p:
     *   psi1_const[p] = ψ1(x[p], y[p], z[p])  (trilinear with periodic lifting)
     *   psi2_const[p] = ψ2(x[p], y[p], z[p])
     *   y_guess[p]    = y[p]
     *   z_guess[p]    = z[p]
     */
    void prepare();

    /**
     * @brief Advance all active particles by one time step (async, hot path).
     *
     * Algorithm per active particle:
     *   1. Project (y,z) at current x via 2×2 Newton.
     *   2. Sample vx at (x, y, z).  Compute x_mid = x + 0.5*dt*vx.
     *   3. Project (y,z) at x_mid.
     *   4. Sample vx at (x_mid, y_mid, z_mid).  Compute x_new = x + dt*vx_mid.
     *   5. Project (y,z) at x_new.
     *   6. Wrap x_new + update wrapX; detect y,z wrap crossings + update wrapY/Z.
     *   7. Commit positions; update y_guess/z_guess.
     *
     * NO allocations.  Fully async on the bound stream.
     *
     * @param dt  Physical time step [T].
     */
    void step(real dt);

    // ── Output ────────────────────────────────────────────────────────────────

    /// Synchronize bound stream.
    void synchronize();

    /// Read-only SoA view of current particle state (device pointers).
    ConstParticlesSoA<real> particles() const;

    /**
     * @brief Compute unwrapped positions into caller-provided device buffers.
     *
     * Unwrapped positions are:
     *   x_u[p] = x[p] + wrapX[p] * Lx
     *   y_u[p] = y[p] + wrapY[p] * Ly
     *   z_u[p] = z[p] + wrapZ[p] * Lz
     *
     * @param uw      Non-owning view of pre-allocated device buffers.
     * @param stream  Stream for kernel launch (may differ from engine stream).
     */
    void compute_unwrapped(UnwrappedSoA<real>& uw, cudaStream_t stream);

    // ── Transport diagnostics ─────────────────────────────────────────────────

    /**
     * @brief Particle status + Newton failure counters (synchronizes stream).
     *
     * Counts are taken over ALL particles in the bound ParticlesSoA.
     * Launches a kernel + device-to-host copy; do NOT call inside the hot loop.
     */
    struct TransportStats {
        int       n_active  = 0;   ///< status == 0 (still being tracked)
        int       n_exited  = 0;   ///< status == PSPTA_STATUS_EXITED
        int       n_other   = 0;   ///< other non-zero status
        long long total_fail = 0;  ///< sum of per-particle Newton fail counts
        uint32_t  n_nonzero_fail = 0;   ///< particles with fail_count > 0
        uint32_t  max_fail_count = 0;   ///< maximum fail_count over all particles
        /// Histogram bins: [0],[1],[2],[3-4],[5-8],[9-16],[>=17]
        uint32_t  hist[PSPTA_FAIL_HIST_BINS] = {};
    };

    /// Compute and return transport stats (synchronizes stream internally).
    TransportStats compute_transport_stats();

    // ── Stream management ─────────────────────────────────────────────────────
    void set_stream(cudaStream_t stream) { stream_ = stream; }

    /// Override injection seed — call BEFORE inject_box() when reusing the
    /// engine across multiple realizations to ensure distinct injection patterns.
    void set_inject_seed(uint64_t s) { inject_seed_ = s; }

    /// Set x-boundary mode.  false (default) = OPEN_X: particle exits when
    /// x leaves [0, Lx).  true = periodic x (use for doubly-periodic toy flows).
    void set_x_periodic(bool v) { x_periodic_ = v; }

private:

    // ── Grid constants (host copy for kernel launches) ─────────────────────────
    Grid3D  grid_;
    double  Lx_, Ly_, Lz_;

    // ── Bound external pointers ────────────────────────────────────────────────
    const VelocityField* vel_  = nullptr;
    const PsptaPsiField* psi_  = nullptr;
    ParticlesSoA<real>   parts_;        ///< Non-owning view

    // ── CUDA execution context ────────────────────────────────────────────────
    cudaStream_t stream_  = nullptr;
    uint64_t     inject_seed_;
    bool         x_periodic_ = false;  ///< false = OPEN_X (particle exits at x=0 or x=Lx)

    // ── Per-particle invariant buffers (owned, preallocated in prepare()) ──────
    int capacity_ = 0;                  ///< allocated size
    DeviceBuffer<float>    d_psi1_const_;  ///< ψ1 invariant per particle
    DeviceBuffer<float>    d_psi2_const_;  ///< ψ2 invariant per particle
    DeviceBuffer<float>    d_y_guess_;     ///< last successful y from Newton
    DeviceBuffer<float>    d_z_guess_;     ///< last successful z from Newton
    DeviceBuffer<uint32_t>            d_fail_count_;  ///< Newton failure counter per particle
    DeviceBuffer<unsigned long long>  d_stats_buf_;      ///< 4× ull scratch for compute_transport_stats()
    DeviceBuffer<uint32_t>            d_fail_detail_buf_; ///< 9× uint32: [n_nonzero, max_fail, hist[7]]
};

} // namespace pspta
} // namespace particles
} // namespace physics
} // namespace macroflow3d
