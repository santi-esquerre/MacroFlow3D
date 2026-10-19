#pragma once

/**
 * @file FaceFluxRepack.cuh
 * @brief SF-37 N3: repack of the SF-19 CompactMAC face fluxes into the periodic face layout read by the
 *        SF-32 Pollock tracker (contract C-REPACK).
 *
 * SF-19 (`solve_affine_periodic_flow`) writes CompactMAC arrays on an (nx, ny, nz) periodic grid
 * (x fastest), each with its periodic duplicate plane (equal to plane 0):
 *
 *   U (nx+1, ny, nz): idx = i + j (nx+1) + k (nx+1) ny     (x-face at x = i dx, lower face of cell i)
 *   V (nx, ny+1, nz): idx = i + j nx + k nx (ny+1)
 *   W (nx, ny, nz+1): idx = i + j nx + k nx ny
 *
 * The SF-32 Pollock core (`physics::particles::streamline_tracker::PeriodicFaceFluxView`) reads three arrays of exactly
 * nx ny nz entries, idx = i + nx (j + ny k), u[idx] = x-face flux at x = i dx (lower face of cell i), etc.
 * The values are face fluxes per unit area (velocity, porosity 1). The repack is therefore a pure
 * copy of the unique planes (i < nx, j < ny, k < nz): bitwise, no arithmetic, the duplicate planes are
 * dropped. Non-cubic nx, ny, nz are supported (SF-38 reuses this API read-only; keep it stable).
 *
 * Device memory of PeriodicFaceFluxes::upload_and_repack: 3 nx ny nz doubles (the persistent periodic
 * arrays) + one staging array of max((nx+1) ny nz, nx (ny+1) nz, nx ny (nz+1)) doubles that is freed
 * before the call returns. Peak = 8 (3 nx ny nz + stage) bytes ~ 32 nx ny nz bytes; persistent
 * 24 nx ny nz bytes. Components are staged one at a time.
 *
 * Streams/synchronization (explicit, not a hot path): every legacy-stream host-to-device cudaMemcpy is
 * followed by cudaDeviceSynchronize before ctx-stream work (SF-33 C2: the ctx stream is non-blocking
 * and not ordered after the legacy default stream); the ctx stream is synchronized before the staging
 * buffer is reused and before upload_and_repack returns.
 */

#include "../../../core/DeviceBuffer.cuh"
#include "../../../core/DeviceSpan.cuh"
#include "../../../core/Scalar.hpp"
#include "../../../runtime/CudaContext.cuh"
#include "../streamline_tracker/StokesFaceVelocity.cuh"

#include <cstddef>
#include <vector>

namespace macroflow3d {
namespace lagrangian_labels {

/// CompactMAC array sizes of an (nx, ny, nz) grid: U (nx+1) ny nz, V nx (ny+1) nz, W nx ny (nz+1).
std::size_t compact_mac_u_size(int nx, int ny, int nz);
std::size_t compact_mac_v_size(int nx, int ny, int nz);
std::size_t compact_mac_w_size(int nx, int ny, int nz);

/**
 * Device repack on ctx.cuda_stream() (asynchronous with respect to the host):
 *   u_p[i + nx (j + ny k)] = U[i + (nx+1)(j + ny k)]
 *   v_p[i + nx (j + ny k)] = V[i + nx j + nx (ny+1) k]
 *   w_p[i + nx (j + ny k)] = W[i + nx j + nx ny k],   i < nx, j < ny, k < nz.
 * Throws std::invalid_argument if nx, ny, nz < 1 or any span size differs from the exact expected size.
 */
void repack_compact_mac_to_periodic_faces(CudaContext& ctx, int nx, int ny, int nz,
                                          DeviceSpan<const real> U, DeviceSpan<const real> V,
                                          DeviceSpan<const real> W, DeviceSpan<real> u_p,
                                          DeviceSpan<real> v_p, DeviceSpan<real> w_p);

/// Move-only owner of the three periodic face arrays (nx ny nz doubles each) and their geometry.
class PeriodicFaceFluxes {
  public:
    PeriodicFaceFluxes() = default;
    PeriodicFaceFluxes(PeriodicFaceFluxes&&) noexcept = default;
    PeriodicFaceFluxes& operator=(PeriodicFaceFluxes&&) noexcept = default;
    PeriodicFaceFluxes(const PeriodicFaceFluxes&) = delete;
    PeriodicFaceFluxes& operator=(const PeriodicFaceFluxes&) = delete;

    /**
     * Uploads host CompactMAC arrays (sizes as compact_mac_*_size) and repacks them, one component at a
     * time through one device staging buffer. Blocking: all work has completed on return. h is the
     * uniform spacing (dx = dy = dz = h; periods n_a h). Throws std::invalid_argument on a wrong size,
     * non-positive dimension or non-finite / non-positive h.
     */
    static PeriodicFaceFluxes upload_and_repack(CudaContext& ctx, const std::vector<real>& U,
                                                const std::vector<real>& V,
                                                const std::vector<real>& W, int nx, int ny, int nz,
                                                real h);

    /// Device view for the SF-32 core: dx = dy = dz = h, Lx = nx h, Ly = ny h, Lz = nz h.
    /// Throws std::logic_error if empty. Valid while this object lives.
    physics::particles::streamline_tracker::PeriodicFaceFluxView view() const;

    bool empty() const { return nx_ == 0; }
    int nx() const { return nx_; }
    int ny() const { return ny_; }
    int nz() const { return nz_; }
    real h() const { return h_; }
    /// Persistent device bytes (3 nx ny nz doubles).
    std::size_t device_bytes() const { return u_.size() * 3 * sizeof(real); }

  private:
    DeviceBuffer<real> u_, v_, w_;
    int nx_ = 0, ny_ = 0, nz_ = 0;
    real h_ = 0;
};

} // namespace lagrangian_labels
} // namespace macroflow3d
