#pragma once

/**
 * @file SlabTraceGpu.cuh
 * @brief SF-35 N6: GPU backward/forward streamline tracer reproducing the host SF-30/SF-33 tracer
 *        (`trace_to_plane`, SlabOracle.cuh) seed by seed; one thread per seed.
 *
 * Authority: SF-35 specification, implementation item 5 and acceptance threshold F (device feet and
 * labels within 1e-12 of the host tracer; otherwise the GPU node is dropped).
 *
 * `trace_dev` (SlabTraceGpu.cu) holds DEVICE COPIES of the frozen SF-30 routines (apps/closure_gate/
 * streamline_integrator.hpp: ArclengthRhs, LandingRhs, dp5_step, step_factor, land_on_plane) and of
 * SF-33 `trace_to_plane`, with the host expression order, constants, branch structure, counters
 * and statuses (closure_gate::StreamlineStatus values). The translation unit SlabTraceGpu.cu is
 * compiled with --fmad=false (CMakeLists.txt) so that device mult/add are IEEE like the host
 * (x86-64 SSE2, no FMA); sqrt/floor are exact on both sides; `exp` only enters dtau/ds (excluded
 * from the error control and the labels); pow(err, -0.2) only enters the step factor. Expected
 * differences are therefore decisions within ~1e-16 of an accept/step threshold.
 *
 * Seeds: planes j = 1..N, seed = (coord(j), coord(m2), coord(m3)), flat index (j-1) N^2 + N m2 +
 * m3 (the compute_oracle enumeration). Per seed: backward trace to x1 = 0 (sigma = -1); if ok the
 * forward re-trace from (0, fy, fz) to seed[0]; the round trip is the max-norm of the difference
 * to the seed, exactly as compute_oracle. Outputs are per-seed only (no atomics); the per-plane
 * reductions run on the host in index order (compute_oracle_gpu), copied from compute_oracle.
 *
 * Allocation contract: SlabGpuTracer::prepare is the ONLY allocating call (device coefficient
 * arrays and per-seed output buffers); trace() performs one kernel launch, no device allocation,
 * and resizes the caller's host vectors (host allocation only on the first call / size change).
 * Every legacy-stream cudaMemcpy host-to-device is followed by cudaDeviceSynchronize before
 * ctx.cuda_stream() work (SF-33 C2).
 */

#include "../../../core/DeviceBuffer.cuh"
#include "../../../core/Scalar.hpp"
#include "../../../numerics/interpolation/PeriodicTricubicBSpline.cuh"
#include "../../../runtime/CudaContext.cuh"
#include "InletSlabGrid.cuh"
#include "SlabOracle.cuh"
#include "SlabProductionSetup.cuh"

#include "apps/closure_gate/streamline_integrator.hpp"

#include <cstdint>
#include <vector>

namespace macroflow3d {
namespace streamfunctions {
namespace inlet_slab {

/// Device direction field: same expression order as SlabSplineDirectionField (g = G + grad s_h,
/// k = exp(s_Y)); views point to DEVICE coefficient arrays.
struct DeviceSplineDirectionField {
    interpolation::PeriodicTricubicBSplineView potential{};
    interpolation::PeriodicTricubicBSplineView log_conductivity{};
    double G[3] = {0.0, 0.0, 0.0};

    __device__ void operator()(const double x[3], double g[3], double& k) const {
        real v = 0.0, gx = 0.0, gy = 0.0, gz = 0.0;
        interpolation::evaluate_point(potential, x[0], x[1], x[2], v, gx, gy, gz);
        g[0] = G[0] + gx;
        g[1] = G[1] + gy;
        g[2] = G[2] + gz;
        real y = 0.0, yx = 0.0, yy = 0.0, yz = 0.0;
        interpolation::evaluate_point(log_conductivity, x[0], x[1], x[2], y, yx, yy, yz);
        k = exp(y);
    }
};

/// Per-seed outputs (host vectors of size N^3, flat index (j-1) N^2 + N m2 + m3).
struct SlabGpuTraceOutputs {
    std::vector<double> foot_y, foot_z; ///< NaN unless backward and forward traces are ok
    std::vector<double> roundtrip;      ///< NaN unless ok
    std::vector<std::uint8_t> st_b, st_f, bfe;
    std::vector<long long> nfev_b, nfev_f, ldisc_b;
    std::vector<double> ming_b;
    double seconds_kernel = 0.0; ///< launch to synchronization
};

class SlabGpuTracer {
  public:
    static constexpr int kBlock = 64; ///< threads per block (one seed per thread)

    /// Uploads both coefficient arrays and allocates the per-seed output buffers (N^3). Only
    /// allocating call. Requires stage.ok().
    void prepare(const CudaContext& ctx, const ProductionStage& stage);

    /// Traces all N^3 seeds. `base`: tol, h_max, min_step, max_arclength_per_period (sigma and
    /// n_periods are set internally; validated as the host tracer).
    void trace(CudaContext& ctx, const InletSlabGrid& grid,
               const closure_gate::IntegratorOptions& base, SlabGpuTraceOutputs& out);

    bool prepared() const { return prepared_; }
    /// Device storage pointer i (0..10) for the no-allocation test: stable between prepare and
    /// trace.
    const void* storage_ptr(int i) const;

  private:
    bool prepared_ = false;
    int n_ = 0;
    DeviceBuffer<real> coef_pot_, coef_logk_;
    DeviceSplineDirectionField field_{};
    DeviceBuffer<double> d_foot_y_, d_foot_z_, d_rt_, d_ming_;
    DeviceBuffer<long long> d_nfev_b_, d_nfev_f_, d_ldisc_;
    DeviceBuffer<std::uint8_t> d_st_b_, d_st_f_, d_bfe_;
};

/**
 * GPU counterpart of compute_oracle for a ProductionStage: runs the GPU tracer, then the per-plane
 * reductions EXACTLY as compute_oracle (same loop, index order) and the labels by
 * detail::oracle_evaluate_labels. `opt.threads` is ignored. `tracer` must be prepared with
 * `stage`. seconds_trace = kernel + device-to-host copies.
 */
SlabOracleResult compute_oracle_gpu(CudaContext& ctx, const InletSlabGrid& grid,
                                    const ProductionStage& stage, SlabGpuTracer& tracer,
                                    const SlabOracleOptions& opt, DeviceSpan<real> psi1,
                                    DeviceSpan<real> psi2);

} // namespace inlet_slab
} // namespace streamfunctions
} // namespace macroflow3d
