#pragma once

/**
 * @file LongBoxStage.cuh
 * @brief SF-40 L3: lean parametric Grid3D stage (nx, ny, nz) of the Lagrangian-label path
 *        (SF-18 field -> SF-28 ln k spline -> SF-19 flow -> D-1 labels -> SF-28 potential spline),
 *        built phase by phase at memory level S+ (contracts C-LEAN-PLACEMENT, C-SYNC, C-MEMPHASE of
 *        docs/experiments/artifacts/2026-10-22-sf40-lean-lagrangian-stage/preregistration.md).
 *
 * Bitwise rule: on a cube N^3 with the same field this stage and
 * `LagrangianTracker::prepare(ctx, DeviceSplineFlow, ...)` reproduce `build_production_stage` +
 * `LagrangianTracker::prepare(ctx, ProductionStage, ...)` bit for bit: it re-arranges lifetimes and
 * buffer placement, never an arithmetic operation (the SF-19 part is
 * `physics::solve_affine_periodic_flow_lean`, bitwise by its own ctest).
 *
 * Geometry. Box [0, nx h] x [0, 1]^2 (ny == nz), h = 1/ny, cells x1 fastest (i + nx (j + ny k)).
 * The restriction ny == nz exists because InletLabels takes one transverse N.
 *
 * Phases (each frees what it no longer uses; every legacy-stream copy follows C-SYNC; no managed
 * memory; an exception inside a phase is rethrown as std::runtime_error whose message contains
 * `phase=<F|S1|S2|S3>`):
 *   F  (gaussian only) SF-18 call identical to SlabFieldSource::gaussian, Y downloaded, device freed.
 *   S1 host lnK = eps Y, K = exp(lnK); prefilter of lnK, coefficients downloaded to the host.
 *   S2 solve_affine_periodic_flow_lean (S+); G, inlet samples, D-1 labels.
 *   S3 coef_logk uploaded, labels device tables, prefilter of h_tilde -> coef_pot (device).
 *   T  (reported by `mem_lines` after LagrangianTracker::prepare) coefficient arrays + labels + tracker
 *      buffers.
 * Per-phase exact device bytes are recorded (LongBoxPhaseRecord) and formatted by `mem_lines`.
 */

#include "../../../core/DeviceBuffer.cuh"
#include "../../../core/Grid3D.hpp"
#include "../../../core/Scalar.hpp"
#include "../../../runtime/CudaContext.cuh"
#include "LagrangianTracker.cuh"

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace macroflow3d {
namespace lagrangian_labels {

/// Deepest MG hierarchy obtained by applying the `slab_auto_mg_levels` loop condition
/// (`n % 2 == 0 && n/2 >= 4 && (n/2) % 2 == 0`) to all three extents simultaneously. Equals
/// `streamfunctions::inlet_slab::slab_auto_mg_levels(N)` on cubes.
int box_auto_mg_levels(int nx, int ny, int nz);

enum class LongBoxSourceKind { gaussian, cells };

struct LongBoxSpec {
    int nx = 0, ny = 0, nz = 0; ///< even, >= 4, ny == nz
    LongBoxSourceKind source = LongBoxSourceKind::gaussian;
    // gaussian (SF-18, box units)
    real sigma2 = 1.0;
    real ell = 0.0;
    unsigned long long seed = 0ULL;
    bool normalize_variance = true;
    // cells: host unit-amplitude Y, x1 fastest, nx*ny*nz values
    std::vector<real> cells;
    std::string field_name; ///< empty: "gaussian" / "cells" by source
    real eps = 1.0;
    real pcg_rtol = 1e-10;
    int pcg_max_iter = -1; ///< -1: library default
    int mg_levels = 0;     ///< 0: box_auto_mg_levels
    bool bhat_mapped_host = true; ///< true: memory level S+ (pinned mapped b_hat); false: level S
    bool keep_face_fluxes = false; ///< host CompactMAC faces (Pollock), kept iff all correctors converged
    int label_chunk = 16384;       ///< ZGEMM leading dimension: do not change
    bool fused_coefficients = false; ///< one interleaved double2 {pot, logk} device array
    bool keep_host_field = false;    ///< keep the unit-amplitude Y on the host (tests: tiling control)
};

/// One MEM_PHASE record (C-MEMPHASE). exact_bytes = cufft_work + sum(items).
struct LongBoxPhaseRecord {
    std::string phase;                 ///< "F" | "S1" | "S2" | "S3" | "T"
    std::size_t cufft_work = 0;
    std::vector<std::pair<std::string, std::size_t>> cufft_named; ///< e.g. {"cufft_gen", bytes}
    std::vector<std::pair<std::string, std::size_t>> items;
    std::size_t device_used = 0;       ///< cudaMemGetInfo total - free at the phase peak
    double seconds = 0.0;
    std::string categories;            ///< S2: "correctors:..,rhs_scratch:..,..." (eight totals)
    bool present = false;
    std::size_t exact_bytes() const {
        std::size_t b = cufft_work;
        for (const auto& it : items)
            b += it.second;
        return b;
    }
};

/// MEM_GEOM record (C-MEMPHASE).
struct LongBoxGeometry {
    long long pid = 0;
    int nx = 0, ny = 0, nz = 0;
    std::size_t n = 0;
    int levels = 0;
    std::vector<std::size_t> level_cells;
    int coarsest[3] = {0, 0, 0};
    std::size_t face_max = 0;
    bool bhat_mapped = true;
    std::size_t bhat_host_bytes = 0;
    std::vector<std::size_t> cub_sum_bytes;
};

class LongBoxStage {
  public:
    streamfunctions::inlet_slab::ProductionStageReport report;
    Grid3D grid;
    real G[3] = {0.0, 0.0, 0.0};
    streamfunctions::inlet_slab::InletLabels labels; ///< device-prepared iff labels_available
    bool labels_available = false;
    std::vector<real> inlet_v1_samples; ///< U-face plane 0, layout m3 + ny m2 (as ProductionStage)
    /// Device coefficient arrays (SF-28, layout i + nx (j + ny k)): coef_pot/coef_logk, or
    /// coef_interleaved (double2 {pot, logk}) when LongBoxSpec::fused_coefficients.
    DeviceBuffer<real> coef_pot, coef_logk;
    DeviceBuffer<double2> coef_interleaved;
    /// SF-19 CompactMAC faces (same convention as ProductionStage::face_flux_*), iff requested and
    /// all correctors converged.
    std::vector<real> face_flux_u, face_flux_v, face_flux_w;
    /// Unit-amplitude Y (LongBoxSpec::keep_host_field only; the gaussian source or a copy of `cells`).
    std::vector<real> host_field;

    LongBoxGeometry geometry;
    LongBoxPhaseRecord phase_F, phase_S1, phase_S2, phase_S3;
    std::size_t device_used_start = 0; ///< cudaMemGetInfo total - free when the build began
    bool fused_coefficients = false;

    bool ok() const { return report.status == streamfunctions::inlet_slab::SlabProductionStatus::ok; }
    /// Darcy converged and the coefficient arrays are present.
    bool flow_ready() const {
        return report.darcy_converged &&
               (fused_coefficients ? coef_interleaved.size() > 0
                                   : (coef_pot.size() > 0 && coef_logk.size() > 0));
    }
    bool faces_available() const {
        return !face_flux_u.empty() && !face_flux_v.empty() && !face_flux_w.empty();
    }

    /// Non-owning view for LagrangianTracker::prepare (valid while this stage lives). Labels are
    /// included iff labels_available.
    DeviceSplineFlow device_flow() const;

};

/// Builds one stage (see the file header). Never throws for darcy_failed / inlet_backflow (status);
/// throws std::invalid_argument for invalid arguments and std::runtime_error naming `phase=<p>` for
/// every failure inside a phase.
LongBoxStage build_long_box_stage(CudaContext& ctx, const LongBoxSpec& spec);

/// `MEM_GEOM ...` followed by one `MEM_PHASE phase=<p> exact_bytes=<B> cufft_work=<W> [cufft_*=<w>]
/// device_used=<U> seconds=<t> [categories=...] items=name:bytes,...` line per recorded phase
/// (space-separated key=value; items comma-separated name:bytes; `items=-` when empty). The T line
/// (coefficient arrays + labels + `tracker_<name>` for every `tracker->device_buffer_bytes()` entry)
/// is produced only if `tracker` is a prepared tracker; call it after LagrangianTracker::prepare
/// (device_used is sampled at call time; `tracker_prepare_seconds` is printed as T's seconds).
std::vector<std::string> mem_lines(const LongBoxStage& stage, const LagrangianTracker* tracker = nullptr,
                                   double tracker_prepare_seconds = 0.0);

} // namespace lagrangian_labels
} // namespace macroflow3d
