#pragma once

/**
 * @file DirichletMedium.cuh
 * @brief SF-38 N2 (contract C-DIRICHLET): medium (b) of the A/B evaluation. The SAME conductivity as the
 *        affine-periodic medium (a) (K = exp(eps Y), formed on the host exactly as LongBoxStage S1), solved
 *        with `physics::solve_head` (harmonic face K, PCG_MG, pin off) with Dirichlet in x
 *        (west head = G1 L1, east head = 0, so the mean head gradient is -G1) and periodic y/z, velocities
 *        by `physics::compute_velocity_from_head` (CompactMAC; Dirichlet faces one-sided with distance h/2
 *        and K_cell).
 *
 * Box [0, L1] x [0,1]^2, h = 1/ny, ny == nz, L1 = nx h; cells x fastest (i + nx (j + ny k)).
 *
 * Phases (each recorded in `phases`, exact device bytes of every allocation + cudaMemGetInfo):
 *   K        host K = exp(eps Y), upload; allocate the head                      (K, head)
 *   solve    FlowWorkspace (rhs, MG hierarchy 4 arrays/level, PCG r z p Ap), init_head_guess, PCG_MG
 *   release  the FlowWorkspace is destroyed (device memory freed) BEFORE any face array is allocated
 *   velocity CompactMAC U V W allocated, compute_velocity_from_head; Q_in, Q_out, balance, max_div
 *            are measured; then K and head are freed (dead)
 *   faces    optional host download of U V W; optional device repack to the periodic Pollock layout
 *            (outlet plane U[nx] dropped; owned buffers); then the CompactMAC device arrays are freed.
 * Peak: solve phase ~ 92.6 B/cell (K, head, rhs, MG, PCG); faces phase 48 B/cell (CompactMAC 24 +
 * periodic 24).
 *
 * Reductions of the report (Q_in, Q_out, qbar) are host sums in index order over the gathered
 * x-face planes; max_div is a maximum (order independent).
 */

#include "../../../core/DeviceBuffer.cuh"
#include "../../../core/Grid3D.hpp"
#include "../../../core/Scalar.hpp"
#include "../../../runtime/CudaContext.cuh"
#include "FaceFluxRepack.cuh"

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace macroflow3d {
namespace lagrangian_labels {

struct DirichletMediumSpec {
    int nx = 0, ny = 0, nz = 0; ///< ny == nz; nx, ny, nz divisible by 2^(mg_levels-1), coarsest >= 2
    real eps = 1.0;             ///< K = exp(eps * Y)
    real G1 = 1.0;              ///< mean gradient of (a): west head = G1 L1, east head = 0 (G1 > 0)
    int mg_levels = 0;          ///< explicit, >= 1 (no default: solve_head's default 4 is not production)
    real cg_rtol = 1e-10;       ///< PCG relative tolerance on its own residual ||r|| / ||r_0||
    int cg_max_iter = 1000;
    bool keep_host_faces = false;     ///< download CompactMAC U V W to the host (E1)
    bool build_periodic_faces = false; ///< device periodic repack (Pollock (b))
    /// SF-41 (owner decision E9), opt-in: download the solved head to `host_head` right before it is freed
    /// (the bitwise reference of the lean stage). Off: no download, no allocation, outputs unchanged.
    bool keep_host_head = false;
};

struct DirichletPhaseRecord {
    std::string phase; ///< "K" | "solve" | "release" | "velocity" | "faces"
    /// Exact bytes of every device allocation alive at the phase's measurement point.
    std::vector<std::pair<std::string, std::size_t>> items;
    std::size_t device_used_start = 0; ///< cudaMemGetInfo total - free at the phase start
    std::size_t device_used = 0;       ///< same, at the phase's measurement point (peak of the phase)
    double seconds = 0.0;
    std::size_t exact_bytes() const {
        std::size_t b = 0;
        for (const auto& it : items)
            b += it.second;
        return b;
    }
};

struct DirichletMediumReport {
    bool converged = false;
    int iterations = 0;
    real initial_residual = 0.0;
    real final_residual = 0.0;
    real rel = 0.0;      ///< final / initial (0 if initial == 0)
    real qbar = 0.0;     ///< mean of U over x-face plane 0
    real Q_in = 0.0;     ///< h^2 sum U[0, j, k]
    real Q_out = 0.0;    ///< h^2 sum U[nx, j, k]
    real balance = 0.0;  ///< |Q_in - Q_out| / |Q_in|
    real max_div = 0.0;  ///< max_c |U(i+1)-U(i) + V(j+1)-V(j) + W(k+1)-W(k)| / qbar
    real head_drop = 0.0; ///< G1 L1
    std::size_t workspace_bytes = 0; ///< FlowWorkspace bytes (released before the faces)
};

class DirichletMedium {
  public:
    DirichletMediumReport report;
    Grid3D grid;
    real h = 0.0, L1 = 0.0;
    std::vector<DirichletPhaseRecord> phases;
    /// Host CompactMAC faces (U (nx+1) ny nz, V nx (ny+1) nz, W nx ny (nz+1)) iff keep_host_faces.
    std::vector<real> face_u, face_v, face_w;
    /// Solved head (nx*ny*nz, x fastest) iff keep_host_head.
    std::vector<real> host_head;

    bool periodic_available() const { return pu_.size() > 0; }
    /// Device view for the SF-32 Pollock core (outlet plane dropped); valid while this object lives.
    /// Throws std::logic_error if build_periodic_faces was not requested.
    physics::particles::streamline_tracker::PeriodicFaceFluxView periodic_view() const;
    /// Persistent device bytes (periodic arrays only).
    std::size_t device_bytes() const { return (pu_.size() + pv_.size() + pw_.size()) * sizeof(real); }
    /// One `MEM_PHASE_B phase=<p> exact_bytes=<B> device_used_start=<U0> device_used=<U> seconds=<t>
    /// items=name:bytes,...` line per phase.
    std::vector<std::string> mem_lines() const;

  private:
    friend DirichletMedium build_dirichlet_medium(CudaContext&, const DirichletMediumSpec&,
                                                  const std::vector<real>&);
    DeviceBuffer<real> pu_, pv_, pw_;
};

/// sizeof(DirichletMedium) as compiled in DirichletMedium.cu. Tests compare it with their own sizeof to detect a
/// stale library object (header changed, object not rebuilt: e.g. a sync that preserves older mtimes).
std::size_t dirichlet_medium_object_bytes();

/// Builds medium (b) from the host unit-amplitude `Y` (nx*ny*nz, x fastest). Throws
/// std::invalid_argument for bad input; solver non-convergence is reported (report.converged), not thrown.
DirichletMedium build_dirichlet_medium(CudaContext& ctx, const DirichletMediumSpec& spec,
                                       const std::vector<real>& Y);

} // namespace lagrangian_labels
} // namespace macroflow3d
