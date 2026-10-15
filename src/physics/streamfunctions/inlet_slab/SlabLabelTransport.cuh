#pragma once

/**
 * @file SlabLabelTransport.cuh
 * @brief SF-35 N1: production constructor of the inlet-anchored Darcy labels on the x1-slab by
 *        backward streamline tracing (label transport), with failure accounting, per-plane
 *        statistics and export.
 *
 * Authority: docs/decisions/2026-10-07-label-transport-constructor.md (Decision items 1-3) and the
 * SF-35 specification. The labels are DEFINED by transport (v . grad psi_i = 0 with the D-1 inlet
 * data on x1 = 0): every vertex (j/N, m2/N, m3/N), j = 1..N, is traced backward along g = G +
 * grad s_h with the SF-30 DP5(4) integrator and Henon landing on x1 = 0 and (psi1, psi2) is the
 * D-1 inlet label at the foot (InletLabels::evaluate_labels, GPU); plane 0 = inlet labels at the
 * vertices. The constructor is a thin, Newton-free wrapper of the SF-33 production oracle
 * (`compute_oracle`, SlabOracle.cuh): it does not change the computation, it only adds the option
 * mapping (h_max = 0 -> grid.h / 8), the status classification and the export. Nothing is clamped
 * or regularized; interior backflow and stagnation are followed and counted; non-ok streamlines
 * (backward or forward) get NaN labels. The forward round trip per plane is the self-consistency
 * check (required <= max_roundtrip, default 1e-8).
 *
 * Determinism: outputs are a pure function of (field, labels, grid, options) and bitwise
 * independent of `threads` (per-plane reductions sequential in index order).
 *
 * Allocation contract: the constructor allocates only its result storage once per call (as
 * compute_oracle: per-seed host vectors, the device foot arrays 2 N^3 doubles, and the one-element
 * `records` vector inside trace_to_plane); the GPU label kernel allocates nothing.
 *
 * Backends: `host` (compute_oracle) is implemented here; `gpu` (SF-35 N6, SlabTraceGpu.cuh; reports threads = 0) is only
 * reachable through the ProductionStage overload.
 */

#include "../../../core/DeviceSpan.cuh"
#include "../../../core/Scalar.hpp"
#include "../../../runtime/CudaContext.cuh"
#include "InletLabels.cuh"
#include "InletSlabGrid.cuh"
#include "SlabOracle.cuh"
#include "SlabProductionSetup.cuh"

#include "src/external/nlohmann/json.hpp"

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace macroflow3d {
namespace streamfunctions {
namespace inlet_slab {

enum class TracerBackend { host = 0, gpu = 1 };
const char* to_string(TracerBackend b);

enum class LabelTransportStatus { ok = 0, trace_fail = 1, roundtrip_exceeded = 2 };
const char* to_string(LabelTransportStatus s);

struct LabelTransportOptions {
    double tol = 1e-8;
    double h_max = 0.0; ///< 0 -> grid.h / 8 (recorded in the result as h_max_effective)
    double min_step = 1e-13;
    double max_arclength = 100.0;
    int threads = 1;
    double max_roundtrip = 1e-8;
    TracerBackend backend = TracerBackend::host;
};

struct LabelTransportPlane {
    int j = 0;
    double x1 = 0.0;
    long long nfev = 0, nfev_rt = 0;
    double roundtrip = 0.0;
    long long back_non_ok = 0, fwd_non_ok = 0, non_ok = 0;
    long long backflow_encounters = 0;
    long long landings_discarded = 0;
    double min_g1_hat = 0.0;
    /// Backward-trace status counts: ok, seed_backflow, cap_exceeded, stagnation, step_underflow,
    /// landing_failed, invalid_conductivity.
    long long status_hist[7] = {0, 0, 0, 0, 0, 0, 0};
    bool flagged = false;
};

struct LabelTransportResult {
    LabelTransportStatus status = LabelTransportStatus::ok;
    TracerBackend backend = TracerBackend::host;
    double tol = 0.0, h_max_effective = 0.0;
    int threads = 1;
    std::vector<LabelTransportPlane> planes; ///< j = 0..N
    long long non_ok = 0, seed_backflow = 0, backflow_encounters = 0, landings_discarded = 0;
    double max_roundtrip = 0.0, min_g1_hat = 0.0;
    double seconds_trace = 0.0, seconds_labels = 0.0;
    std::vector<double> foot_y, foot_z, roundtrip; ///< N^3, planes 1..N, NaN if non-ok
    /// Per-plane lines (PLANE) + one stage line (TRANSPORT).
    std::string table() const;
};

/// Metadata/summary of a result as JSON (status, backend, tol, h_max, threads, histograms, round
/// trips per plane, counters, timings).
nlohmann::json to_json(const LabelTransportResult& r);

/// Shared conversion SlabOracleResult -> LabelTransportResult (host and GPU backends).
inline LabelTransportResult make_label_transport_result(SlabOracleResult&& o,
                                                        const LabelTransportOptions& opt,
                                                        TracerBackend backend,
                                                        double h_max_effective, int threads) {
    LabelTransportResult R;
    R.backend = backend;
    R.tol = o.tol;
    R.h_max_effective = h_max_effective;
    R.threads = threads;
    R.non_ok = o.non_ok;
    R.seed_backflow = o.seed_backflow;
    R.landings_discarded = o.landings_discarded;
    R.min_g1_hat = o.min_g1_hat;
    R.max_roundtrip = o.max_roundtrip;
    R.seconds_trace = o.seconds_trace;
    R.seconds_labels = o.seconds_labels;
    R.planes.reserve(o.planes.size());
    bool rt_exceeded = false;
    for (const SlabOraclePlane& P : o.planes) {
        LabelTransportPlane Q;
        Q.j = P.j;
        Q.x1 = P.x1;
        Q.nfev = P.nfev;
        Q.nfev_rt = P.nfev_rt;
        Q.roundtrip = P.roundtrip;
        Q.back_non_ok = P.back_non_ok;
        Q.fwd_non_ok = P.fwd_non_ok;
        Q.non_ok = P.back_non_ok + P.fwd_non_ok;
        Q.backflow_encounters = P.backflow_encounters;
        Q.landings_discarded = P.landings_discarded;
        Q.min_g1_hat = P.min_g1_hat;
        for (int s = 0; s < 7; ++s)
            Q.status_hist[s] = P.status_hist[s];
        Q.flagged = P.flagged;
        R.backflow_encounters += P.backflow_encounters;
        rt_exceeded = rt_exceeded || P.roundtrip > opt.max_roundtrip;
        R.planes.push_back(Q);
    }
    R.foot_y = std::move(o.foot_y);
    R.foot_z = std::move(o.foot_z);
    R.roundtrip = std::move(o.roundtrip);
    R.status = R.non_ok > 0 ? LabelTransportStatus::trace_fail
                            : (rt_exceeded ? LabelTransportStatus::roundtrip_exceeded
                                           : LabelTransportStatus::ok);
    return R;
}

/**
 * Template entry point (host tracer). `labels` must be device-prepared; psi1/psi2 receive the FULL
 * labels on planes 0..N ((N+1) N^2 each, device). gpu backend: std::invalid_argument.
 */
template <class Field>
LabelTransportResult construct_labels_by_tracing(CudaContext& ctx, const InletSlabGrid& grid,
                                                 const Field& field, const InletLabels& labels,
                                                 const LabelTransportOptions& opt,
                                                 DeviceSpan<real> psi1, DeviceSpan<real> psi2) {
    if (opt.backend == TracerBackend::gpu) {
        throw std::invalid_argument(
            "construct_labels_by_tracing: gpu backend requires the ProductionStage overload");
    }
    if (!labels.device_prepared()) {
        throw std::logic_error("construct_labels_by_tracing: inlet labels not device-prepared");
    }
    require_valid_grid(grid, "construct_labels_by_tracing");
    SlabOracleOptions oo;
    oo.tol = opt.tol;
    oo.h_max = opt.h_max > 0.0 ? opt.h_max : grid.h / 8.0;
    oo.min_step = opt.min_step;
    oo.max_arclength = opt.max_arclength;
    oo.threads = opt.threads;
    oo.max_roundtrip = opt.max_roundtrip;
    SlabOracleResult o = compute_oracle(ctx, grid, field, labels, oo, psi1, psi2);
    return make_label_transport_result(std::move(o), opt, opt.backend, oo.h_max, opt.threads);
}

/// ProductionStage overload (requires stage.ok()). gpu backend: SlabGpuTracer (SlabTraceGpu.cuh), threads reported as 0.
LabelTransportResult construct_labels_by_tracing(CudaContext& ctx, const ProductionStage& stage,
                                                 const LabelTransportOptions& opt,
                                                 DeviceSpan<real> psi1, DeviceSpan<real> psi2);

/// Writes psi1.npy, psi2.npy (shape (N+1, N, N), float64, slab layout) and labels.json (metadata
/// plus shape / layout / format) into `dir` (created if needed).
void save_labels(const std::string& dir, const InletSlabGrid& grid,
                 const std::vector<real>& psi1_host, const std::vector<real>& psi2_host,
                 const nlohmann::json& metadata);

} // namespace inlet_slab
} // namespace streamfunctions
} // namespace macroflow3d
