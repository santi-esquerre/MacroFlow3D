#pragma once

/**
 * @file InletSlabGrid.cuh
 * @brief SF-33 N0: grid, layout and public types of the inlet-label equation (14) solver on the
 *        x1-non-periodic slab.
 *
 * Authoritative numerical contract: `docs/decisions/2026-10-06-eq14-inlet-label-formulation.md`
 * (items 1-4, 7), the SF-33 increment specification and the SF-29 prototype that DEFINES the
 * discrete problem
 * (`docs/experiments/artifacts/2026-10-02-sf29-inlet-labels/scripts/candidate_i.py --order 4`,
 * variant `i1`, and `metrics.py`). This header fixes conventions that every later SF-33 node builds
 * on.
 *
 * Grid (vertex grid of the unit slab):
 *   x1 = j h, j = 0..N (N + 1 planes; plane 0 = inlet, plane N = outlet),
 *   x2 = m2 h, x3 = m3 h, m = 0..N-1, periodic,  h = 1/N,  N even and N >= 8.
 * The affine coordinates used for the labels are x2 = m2 / N and x3 = m3 / N computed as a single
 * division
 * (`coord`), exactly as `np.arange(N) / float(N)` of the prototype.
 *
 * Layout (LOCKED):
 *   full arrays (labels, periodic parts, k, q, ln k, grad ln k, vD):  (N+1) N^2 entries,
 *       idx = m3 + N * (m2 + N * j)          (m3 fastest, then m2, then plane j = 0..N);
 *   unknown / residual vectors: field-major [u1 | u2], each N^3 entries for the planes j = 1..N,
 *       idx = m3 + N * (m2 + N * (j - 1))    (field f adds f * N^3);
 *   inlet / plane arrays (u0, vperp_in): N^2 entries, idx = m3 + N * m2.
 * This is the C order of the prototype's (N+1, N, N) / (N, N) arrays: `.npy` exports load without
 * transposition. A full array's planes 1..N are therefore one contiguous block equal to the
 * corresponding unknown block.
 *
 * Labels: psi1 = x2 + u1, psi2 = x3 + u2 with u_i periodic in (x2, x3). Only u_i is differenced;
 * the affine gradients e2, e3 are added exactly. Double precision throughout. No regularization of
 * |c|^2 anywhere.
 */

#include "../../../core/DeviceBuffer.cuh"
#include "../../../core/Scalar.hpp"

#include <cmath>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>

namespace macroflow3d {
namespace streamfunctions {
namespace inlet_slab {

// ================================================================================================
// Grid
// ================================================================================================

struct InletSlabGrid {
    int n =
        0; ///< N: number of cells per direction (planes 0..N in x1; N periodic vertices in x2, x3)
    real h = 0.0; ///< 1/N (computed as 1.0 / N, as the prototype's `self.h = 1.0 / N`)

    /// Validated construction: throws std::invalid_argument unless N is even and N >= 8 (the
    /// 4th-order boundary stencils need 6 planes; even N for the FFTs of the inlet / production
    /// path).
    static InletSlabGrid make(int n_cells) {
        if (n_cells < 8 || (n_cells % 2) != 0) {
            throw std::invalid_argument("InletSlabGrid: N must be even and >= 8 (got " +
                                        std::to_string(n_cells) + ")");
        }
        InletSlabGrid g;
        g.n = n_cells;
        g.h = 1.0 / static_cast<real>(n_cells);
        return g;
    }

    bool valid() const { return n >= 8 && (n % 2) == 0 && h == 1.0 / static_cast<real>(n); }

    __host__ __device__ std::size_t plane_size() const {
        return static_cast<std::size_t>(n) * static_cast<std::size_t>(n);
    }
    /// (N+1) N^2: full arrays on planes 0..N.
    __host__ __device__ std::size_t full_size() const {
        return static_cast<std::size_t>(n + 1) * plane_size();
    }
    /// N^3: one field of the unknown / residual vector (planes 1..N).
    __host__ __device__ std::size_t field_size() const {
        return static_cast<std::size_t>(n) * plane_size();
    }
    /// 2 N^3: the unknown / residual vector [u1 | u2].
    __host__ __device__ std::size_t unknown_size() const { return 2 * field_size(); }

    /// Full-array index, plane j = 0..N.
    __host__ __device__ std::size_t full_index(int j, int m2, int m3) const {
        return static_cast<std::size_t>(m3) +
               static_cast<std::size_t>(n) *
                   (static_cast<std::size_t>(m2) +
                    static_cast<std::size_t>(n) * static_cast<std::size_t>(j));
    }
    /// Index inside one field of the unknown vector, plane j = 1..N.
    __host__ __device__ std::size_t unknown_index(int j, int m2, int m3) const {
        return full_index(j - 1, m2, m3);
    }
    /// In-plane index (u0, vperp_in).
    __host__ __device__ std::size_t plane_index(int m2, int m3) const {
        return static_cast<std::size_t>(m3) +
               static_cast<std::size_t>(n) * static_cast<std::size_t>(m2);
    }
    /// Exact periodic wrap of an in-plane index shifted by at most +-N (integer arithmetic only).
    __host__ __device__ int wrap(int m) const { return m < 0 ? m + n : (m >= n ? m - n : m); }
    /// Affine coordinate m / N (one rounding), identical to `np.arange(N) / float(N)`.
    __host__ __device__ real coord(int m) const {
        return static_cast<real>(m) / static_cast<real>(n);
    }
};

inline void require_valid_grid(const InletSlabGrid& g, const char* who) {
    if (!g.valid()) {
        throw std::invalid_argument(
            std::string(who) + ": invalid InletSlabGrid (use InletSlabGrid::make; N even, N >= 8)");
    }
}

// ================================================================================================
// Stage inputs and reference data
// ================================================================================================

/**
 * Inputs of one continuation stage (the prototype's `light_case` + `Ctx`).
 *
 * Ownership: this struct OWNS its device storage (DeviceBuffer, move-only). `allocate(grid)` sizes
 * every buffer exactly (grow-only DeviceBuffer::resize: no reallocation when it already fits); the
 * producer (prototype loader, production path) fills them; the residual / metrics evaluators only
 * READ them and never allocate.
 *
 *   q, lnk, grad_lnk[3]: full arrays, planes 0..N ((N+1) N^2). q = 1/k at the vertices (the
 * prototype's `1.0 / np.exp(lnk)`; `fill_q_from_lnk` in SlabResidual.cuh computes exactly that).
 *                        grad_lnk is the analytic / spectral gradient of ln k at the vertices
 * (never differenced here). The residual reads planes 1..N only. u0[2]:               inlet
 * periodic parts u_i on plane 0 (N^2 each): psi0_i minus the affine coordinate. vperp_in[2]: (v2,
 * v3) of the Darcy velocity on the inlet face at the inlet vertices (N^2 each); the outlet-row data
 * of the oblique condition D-2. v_rms:               RMS|vD| over all (N+1) N^2 vertices (host
 * scalar; normalizes r_out). Must be > 0, finite. field, eps, N:       metadata (host), used for
 * CASE lines and logs only.
 */
struct SlabStageInputs {
    DeviceBuffer<real> q;
    DeviceBuffer<real> lnk;
    DeviceBuffer<real> grad_lnk[3];
    DeviceBuffer<real> u0[2];
    DeviceBuffer<real> vperp_in[2];
    real v_rms = 0.0;
    std::string field;
    real eps = 0.0;
    int N = 0;

    void allocate(const InletSlabGrid& g) {
        require_valid_grid(g, "SlabStageInputs::allocate");
        q.resize(g.full_size());
        lnk.resize(g.full_size());
        for (auto& b : grad_lnk)
            b.resize(g.full_size());
        for (auto& b : u0)
            b.resize(g.plane_size());
        for (auto& b : vperp_in)
            b.resize(g.plane_size());
        N = g.n;
    }

    /// Host-only size / scalar validation (no device access, no synchronization).
    void check(const InletSlabGrid& g) const {
        require_valid_grid(g, "SlabStageInputs::check");
        bool ok = q.size() == g.full_size() && grad_lnk[0].size() == g.full_size() &&
                  grad_lnk[1].size() == g.full_size() && grad_lnk[2].size() == g.full_size() &&
                  u0[0].size() == g.plane_size() && u0[1].size() == g.plane_size() &&
                  vperp_in[0].size() == g.plane_size() && vperp_in[1].size() == g.plane_size();
        if (!ok)
            throw std::invalid_argument("SlabStageInputs: buffer sizes do not match the grid");
        if (!(std::isfinite(v_rms) && v_rms > 0.0)) {
            throw std::invalid_argument("SlabStageInputs: v_rms must be finite and > 0");
        }
    }
};

/**
 * Reference data for the metrics (the prototype's `case["vD"]`, `case["psi_or"]`). Owns its device
 * storage (same allocation contract as SlabStageInputs). vD[3]:     Darcy velocity at every vertex,
 * planes 0..N. psi_or[2]: oracle labels, FULL labels (affine part included), planes 0..N.
 */
struct SlabReferenceData {
    DeviceBuffer<real> vD[3];
    DeviceBuffer<real> psi_or[2];
    bool has_vD = false;
    bool has_psi_or = false;

    void allocate(const InletSlabGrid& g, bool with_vD, bool with_psi_or) {
        require_valid_grid(g, "SlabReferenceData::allocate");
        if (with_vD)
            for (auto& b : vD)
                b.resize(g.full_size());
        if (with_psi_or)
            for (auto& b : psi_or)
                b.resize(g.full_size());
        has_vD = with_vD;
        has_psi_or = with_psi_or;
    }
};

// ================================================================================================
// Norms and metrics
// ================================================================================================

/// r_F = sqrt((RMS F1^2 + RMS F2^2)/2) / q_rms over the equation rows (planes 1..N-1, q_rms over
/// the same planes); r_out = RMS|c_perp(x1 = 1) - vperp_in| / v_rms over the outlet plane N. NaN
/// until evaluated.
struct SlabResidualNorms {
    real r_F = std::nan("");
    real r_out = std::nan("");
    real merit() const { return std::sqrt(r_F * r_F + r_out * r_out); }
};

/**
 * The metrics dict of `metrics.fd_metrics(order=4)` (C-i4 e_psi normalization) as a struct. Unset
 * values are NaN. Percentile arrays are {0.1 %, 1 %, 5 %, 50 %} with numpy's default (linear)
 * definition.
 */
struct SlabMetrics {
    real e_v = std::nan("");
    real e_i1 = std::nan("");
    real e_i2 = std::nan("");
    real e_div = std::nan("");
    real min_c = std::nan("");
    real p0_1 = std::nan("");
    real p1 = std::nan("");
    real p5 = std::nan("");
    real p50 = std::nan("");
    real vD_min = std::nan("");
    real vD_p[4] = {std::nan(""), std::nan(""), std::nan(""), std::nan("")};
    real v_rms = std::nan("");
    // label errors (present iff has_psi)
    bool has_psi = false;
    real e_psi = std::nan("");
    real e_psi1 = std::nan("");
    real e_psi2 = std::nan("");
    real a_psi1 = std::nan("");
    real a_psi2 = std::nan("");
    real den_used1 = std::nan("");
    real den_used2 = std::nan("");
    /// Number of vertices where |c| or |vD| is not finite (percentiles are NaN when > 0; nothing is
    /// clamped).
    long long nonfinite = 0;
};

/// Optional fields of the CASE line (Python `None` -> "nan").
struct CaseLineExtras {
    std::optional<real> r_F;
    std::optional<int> its;
    real t = 0.0;
};

/**
 * The SF-29 one-line parseable CASE format (`metrics.case_line`), byte for byte:
 *   CASE field=<f> eps=<%g> N=<%d> cand=<name> | r_F=<..> its=<..> | e_v=<..> e_psi=<..>
 * e_i=(<..>,<..>) e_div=<..> min_c=<..> p0.1=<..> p1=<..> p5=<..> p50=<..> | t=<%.1f>[ |
 * e_psi1=<..> e_psi2=<..>] Floats print as %.3e, ints as %d, missing / NaN as `nan`; the
 * e_psi1/e_psi2 suffix appears iff m.has_psi. Defined in SlabMetrics.cu.
 */
std::string format_case_line(const std::string& field, real eps, int N, const std::string& cand,
                             const SlabMetrics& m, const CaseLineExtras& extras = CaseLineExtras{});

// ================================================================================================
// Deterministic block reductions (shared by the residual and metrics kernels)
// ================================================================================================

namespace detail {

constexpr int kSlabBlock =
    256; ///< every slab reduction kernel is launched with exactly this block size
constexpr int kSlabMaxBlocks =
    1024; ///< fixed upper bound on the grid of grid-stride reduction kernels

/// Grid size of a grid-stride reduction kernel over `count` items: a pure function of `count`, so
/// the per-thread accumulation order, the partial sums and the final tree are reproducible run to
/// run.
inline int slab_reduce_blocks(std::size_t count) {
    std::size_t b = (count + kSlabBlock - 1) / kSlabBlock;
    if (b < 1)
        b = 1;
    if (b > static_cast<std::size_t>(kSlabMaxBlocks))
        b = kSlabMaxBlocks;
    return static_cast<int>(b);
}

/// Block tree reduction of K per-thread values; thread 0 stores the block sums to partials[k *
/// nblocks + block]. Requires blockDim.x == kSlabBlock. Fixed order (deterministic).
template <int K>
__device__ inline void block_reduce_store(const real (&v)[K], real* partials, int nblocks) {
    __shared__ real sh[K][kSlabBlock];
    const int t = threadIdx.x;
    for (int k = 0; k < K; ++k)
        sh[k][t] = v[k];
    __syncthreads();
    for (int s = kSlabBlock / 2; s > 0; s >>= 1) {
        if (t < s) {
            for (int k = 0; k < K; ++k)
                sh[k][t] += sh[k][t + s];
        }
        __syncthreads();
    }
    if (t == 0) {
        for (int k = 0; k < K; ++k)
            partials[static_cast<std::size_t>(k) * nblocks + blockIdx.x] = sh[k][0];
    }
}

/// One block of kSlabBlock threads: out[k] = sum of partials[k * nblocks + 0..nblocks-1] (fixed
/// order).
template <int K>
__global__ void finalize_partials_kernel(const real* partials, int nblocks, real* out) {
    real v[K];
    for (int k = 0; k < K; ++k) {
        real acc = 0.0;
        for (int i = threadIdx.x; i < nblocks; i += kSlabBlock)
            acc += partials[static_cast<std::size_t>(k) * nblocks + i];
        v[k] = acc;
    }
    __shared__ real sh[K][kSlabBlock];
    const int t = threadIdx.x;
    for (int k = 0; k < K; ++k)
        sh[k][t] = v[k];
    __syncthreads();
    for (int s = kSlabBlock / 2; s > 0; s >>= 1) {
        if (t < s) {
            for (int k = 0; k < K; ++k)
                sh[k][t] += sh[k][t + s];
        }
        __syncthreads();
    }
    if (t == 0) {
        for (int k = 0; k < K; ++k)
            out[k] = sh[k][0];
    }
}

} // namespace detail

} // namespace inlet_slab
} // namespace streamfunctions
} // namespace macroflow3d
