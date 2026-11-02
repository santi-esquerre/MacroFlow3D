#pragma once

/**
 * @file PlaneCrossingRecorder.cuh
 * @brief SF-42 N1: first-crossing statistics of control planes x1 = X_k for the Par2 path.
 * @ingroup runtime_stats
 *
 * Opt-in (`analysis.plane_crossings.enabled`); nothing here runs, allocates or writes unless enabled.
 *
 * Definition (decision 2026-10-29, G4). The Par2 trajectory is piecewise linear between steps
 * (explicit Euler, fixed dt). For each particle and plane X_k, with t_n = n dt (t = 0 at injection)
 * and the UNWRAPPED positions (x_n, y_n, z_n) = (x + wrapX Lx, y + wrapY Ly, z + wrapZ Lz) of Par2:
 *     first step n with  x_n < X_k <= x_{n+1}:
 *     f = (X_k - x_n) / (x_{n+1} - x_n),  t_c = t_n + f dt,
 *     x2_c = y_n + f (y_{n+1} - y_n),  x3_c = z_n + f (z_{n+1} - z_n).
 * One step may cross several planes. A particle that never reaches X_k is not counted in that plane.
 * The device arithmetic uses explicit round-to-nearest intrinsics (no FMA contraction), so a naive host
 * evaluation of the formulas above on the same positions is bitwise equal.
 *
 * Per plane and realization the moments are taken over the particles that crossed (uniform weights,
 * biased: divide by n_crossed, two-pass centred) for t, the transverse position (x2, x3) and the
 * transverse displacement from the injection point (dx = crossing - injection).
 *
 * Closed outlet (Dirichlet x faces map to Par2 `Closed`). Par2_Core zeroes the x displacement of a move that
 * would leave the domain, so a particle closer to Lx than one step never crosses a plane in (x_n, Lx):
 * planes closer to Lx than the largest step |dx_step| (advective U dt plus the diffusive/dispersive jump,
 * several sigma) are not reliably crossed. The validator only requires the last plane < Lx; leave a
 * clearance of at least several steps (e.g. >= 10 U_max dt) and treat n_crossed < n_injected on planes
 * near the outlet as this limitation, not as a particle property.
 *
 * HPC contract: device buffers are allocated once in the constructor (no allocation, no host sync
 * in update()); records are per particle and plane (n_particles x P x (3 doubles + 1 byte)), which
 * is the dominant memory cost of the option (25 B per particle and plane).
 */

#include "../../core/DeviceBuffer.cuh"
#include "../../core/Scalar.hpp"
#include "../../physics/particles/par2_adapter/par2_views.hpp"

#include <cuda_runtime.h>
#include <string>
#include <vector>

namespace macroflow3d {
namespace runtime {

/// One output row of `crossings/realization_%04d_planes.csv`.
struct PlaneCrossingRow {
    double x1 = 0.0;
    long long n_injected = 0;
    long long n_crossed = 0;
    double mean_t = 0.0, var_t = 0.0;
    double mean_x2 = 0.0, mean_x3 = 0.0, var_x2 = 0.0, var_x3 = 0.0, cov_x2x3 = 0.0;
    double mean_dx2 = 0.0, mean_dx3 = 0.0, var_dx2 = 0.0, var_dx3 = 0.0, cov_dx2dx3 = 0.0;
};

class PlaneCrossingRecorder {
  public:
    /// @param planes strictly increasing plane positions (config length units).
    PlaneCrossingRecorder(int n_particles, const std::vector<double>& planes, double Lx, double Ly,
                          double Lz);

    /// After injection and before step 1: captures the injection positions, clears the records.
    void begin(const physics::particles::ConstParticlesSoA<real>& p, cudaStream_t stream);

    /// After step `step` (1-based) of size dt: detects first crossings (async, no host sync).
    void update(const physics::particles::ConstParticlesSoA<real>& p, int step, double dt,
                cudaStream_t stream);

    /// End of the realization: syncs, downloads the records and reduces them (one row per plane).
    std::vector<PlaneCrossingRow> finish(cudaStream_t stream);

    /// Injection positions (x2, x3), valid after finish() (downloaded there); for the optional dump.
    const std::vector<double>& injection_x2() const { return h_inj_y_; }
    const std::vector<double>& injection_x3() const { return h_inj_z_; }

    /// Bytes of device memory owned (report/diagnostics).
    size_t device_bytes() const;

  private:
    int n_;
    int P_;
    double Lx_, Ly_, Lz_;
    std::vector<double> h_planes_;
    DeviceBuffer<double> d_planes_;
    DeviceBuffer<double> prev_x_, prev_y_, prev_z_; // unwrapped position after the previous step
    DeviceBuffer<double> inj_y_, inj_z_;            // unwrapped injection position
    DeviceBuffer<double> rec_t_, rec_y_, rec_z_;    // [k * n + i]
    DeviceBuffer<unsigned char> rec_flag_;          // [k * n + i]
    std::vector<double> h_inj_y_, h_inj_z_;
};

/// `# format_version=1 sf42_plane_crossings ...` header + one `%.17g` row per plane.
bool write_plane_crossings_csv(const std::string& path, const std::vector<PlaneCrossingRow>& rows);

/// Injection positions (`x1,x2,x3`, `%.17g`), x1 constant; opt-in test/verification dump.
bool write_injection_csv(const std::string& path, double x1, const std::vector<double>& x2,
                         const std::vector<double>& x3);

} // namespace runtime
} // namespace macroflow3d
