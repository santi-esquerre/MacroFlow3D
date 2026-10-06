#pragma once

/**
 * @file SlabCoarseCorrection.cuh
 * @brief SF-33 N7c (PROBE): Galerkin coarse-space correction of the inlet-slab preconditioner on
 *        the weakly determined x1-independent ("xi1 = 0") subspace, combined with P-A.
 *
 * Motivation (orchestrator diagnosis, SF-33 audits N6/N7a/N7b; SF-29 UNDERSTAND 2.1)
 * ---------------------------------------------------------------------------------
 * For x1-independent perturbations the principal symbol of the linearization has an identically
 * zero determinant: that family is controlled only by lower-order terms and the boundary rows
 * (discrete singular values ~h^2). P-A (SlabModePreconditioner) inverts it exactly for the
 * PLANE-AVERAGED coefficients; the local transverse coefficient variation is amplified by P-A's
 * ~h^-2 response on that family. Hypothesis H of N7c: correcting that subspace with the TRUE local
 * coefficients (Galerkin projection of the actual operator) restores GMRES convergence.
 *
 * Coarse space V (K columns, never stored densely)
 * -----------------------------------------------
 * For every in-plane column (m2, m3) of the N0 layout, every field f in {0, 1} and every profile
 * p < P (P = `profiles`, 1 or 2):
 *     v_{(m2,m3),f,p}[f', j, m2', m3'] = delta_{f f'} delta_{m2 m2'} delta_{m3 m3'} w_p(j),
 *     w_0(j) = 1 (x1-constant),  w_1(j) = j h = j / N (x1-linear),   j = 1..N (the unknown planes)
 * (the inlet plane j = 0 is not an unknown: the direction vanishes there). K = 2 P N^2.
 * Coarse index (LOCKED for the probe): c = ((m2 N + m3) * 2 + f) * P + p.
 *   restrict:  (V^T r)_c   = sum_j w_p(j) r[f, j, m2, m3]    (one kernel, one thread per (m2,m3,f),
 *                                                             fixed summation order j = 1..N)
 *   prolong:   out += V y,  out[f, j, m2, m3] += sum_p w_p(j) y_c   (one kernel)
 *
 * Galerkin matrix and its factorization
 * -------------------------------------
 * E = V^T A V (K x K), A the operator handed to `build` (in the Newton solver: the shifted JVP
 * J + mu D with the same mu as P-A). Assembled column by column: K applications of A to the dense
 * basis vectors (a fill kernel writes v_c), each followed by the restriction into column c of a
 * device K x K buffer; ONE device-to-host copy of E at the end. Host dense LU with partial pivoting
 * (row-major, right-looking, unblocked: K^3 / 3 flops; LAPACK dgetrf semantics: piv[k] = the row
 * swapped with row k at step k). A zero or non-finite pivot is COUNTED and reported (nothing is
 * perturbed; the correction is then unusable and `build` reports `zero_pivots > 0`).
 * Logged quality numbers: ||E||_1, the Hager-Higham estimate of ||E^-1||_1 (LAPACK dlacon-style,
 * <= 5 iterations, plus Higham's alternating test vector), rcond = 1 / (||E||_1 est ||E^-1||_1),
 * min / max |U_kk|. Column scaling of V does not change the correction V (V^T A V)^-1 V^T.
 *
 * Preconditioner application (right preconditioning, SlabGmres Operator interface)
 * -------------------------------------------------------------------------------
 *   off:  out = P_A^-1 in
 *   add:  out = P_A^-1 in + V E^-1 V^T in                         (no extra A application)
 *   mult: z = P_A^-1 in;  out = z + V E^-1 V^T (in - A z)         (one extra A application)
 * E^-1 (V^T w) is a HOST solve of size K: one device-to-host copy of K doubles (pinned), one
 * cudaStreamSynchronize, the LU solve on the host (2 K^2 flops), one host-to-device copy of K
 * doubles (pinned, async). Documented host round trip per application (probe; a production version
 * would keep the coarse solve on the device).
 *
 * Memory (device): E column buffer K^2 doubles (24^3, P = 2: K = 2304, 42 MB; NOT suitable for
 * 64^3 / 128^3 — the productization step replaces it by colored assembly + banded LU), 2 work
 * vectors of 2 N^3 doubles, 1 dense basis vector, 2 coarse vectors of K. Host: E and its LU (2 K^2
 * doubles), pivots, pinned staging of 2 K doubles.
 *
 * Allocation / synchronization: prepare() is the only allocating call. build(): K x (fill + A +
 * restrict) enqueued, one D2H copy of E + one cudaStreamSynchronize, host LU; no allocation.
 * apply(): see above (one sync per application in add / mult; none in off).
 */

#include "../../../core/DeviceBuffer.cuh"
#include "../../../core/DeviceSpan.cuh"
#include "../../../core/Scalar.hpp"
#include "../../../runtime/CudaContext.cuh"
#include "InletSlabGrid.cuh"
#include "SlabSolverTypes.cuh"

#include <cstddef>
#include <functional>
#include <vector>

namespace macroflow3d {
namespace streamfunctions {
namespace inlet_slab {

struct SlabCoarseBuildReport {
    int K = 0;
    int profiles = 0;
    double t_assembly = 0.0;  ///< K operator applications + restrictions + the D2H copy of E
    double t_lu = 0.0;        ///< host LU factorization
    double t_cond = 0.0;      ///< host condition estimate
    int zero_pivots = 0;      ///< exactly zero or non-finite pivots (never hidden)
    real norm1 = 0.0;         ///< ||E||_1
    real inv_norm1_est = 0.0; ///< Hager-Higham estimate of ||E^-1||_1
    real rcond_est = 0.0;     ///< 1 / (||E||_1 * est ||E^-1||_1)
    real min_abs_u = 0.0;     ///< min |U_kk|
    real max_abs_u = 0.0;     ///< max |U_kk|
};

class SlabCoarseCorrection {
  public:
    using Operator = std::function<void(DeviceSpan<const real> in, DeviceSpan<real> out)>;

    SlabCoarseCorrection() = default;
    ~SlabCoarseCorrection();
    SlabCoarseCorrection(const SlabCoarseCorrection&) = delete;
    SlabCoarseCorrection& operator=(const SlabCoarseCorrection&) = delete;

    /// The only allocating call (grow-only device buffers, host matrices, pinned staging).
    /// profiles in {1, 2}.
    void prepare(CudaContext& ctx, const InletSlabGrid& g, int profiles);
    bool prepared_for(const InletSlabGrid& g) const { return n_ == g.n && n_ > 0; }
    int K() const { return K_; }
    int profiles() const { return P_; }

    /// Assemble E = V^T A V (K applications of A) and LU-factor it on the host. See the header.
    SlabCoarseBuildReport build(CudaContext& ctx, const InletSlabGrid& g, const Operator& A);
    bool built() const { return built_; }

    /// rc = V^T r (K entries, device). Enqueue only.
    void restrict_to_coarse(CudaContext& ctx, const InletSlabGrid& g, DeviceSpan<const real> r,
                            DeviceSpan<real> rc) const;
    /// out += V y (y: K entries, device). Enqueue only.
    void prolong_add(CudaContext& ctx, const InletSlabGrid& g, DeviceSpan<const real> y,
                     DeviceSpan<real> out) const;
    /// out = v_c (dense basis vector c, 2 N^3). Enqueue only.
    void basis_vector(CudaContext& ctx, const InletSlabGrid& g, int c, DeviceSpan<real> out) const;

    /// out += V E^-1 V^T r. One host round trip (documented).
    void add_correction(CudaContext& ctx, const InletSlabGrid& g, DeviceSpan<const real> r,
                        DeviceSpan<real> out);

    /// Preconditioner application (see the header). PA = the P-A application (out = P_A^-1 in).
    /// in and out must not overlap. mode off: out = PA(in) (no correction, no sync).
    void apply(CudaContext& ctx, const InletSlabGrid& g, SlabCoarseMode mode, const Operator& A,
               const Operator& PA, DeviceSpan<const real> in, DeviceSpan<real> out);

    /// Application statistics since the last build() (wall time including the host round trip).
    int applications() const { return n_apply_; }
    double apply_seconds() const { return t_apply_; }
    double host_solve_seconds() const { return t_host_solve_; }
    void reset_apply_stats() {
        n_apply_ = 0;
        t_apply_ = 0.0;
        t_host_solve_ = 0.0;
    }

    /// Host copy of E (row-major, E[r * K + c]) as assembled by the last build().
    const std::vector<real>& galerkin_matrix() const { return E_; }

    std::size_t allocated_bytes() const;
    std::vector<const void*> buffer_pointers() const;

    // ---- host dense LU (exposed for the tests) --------------------------------------------------
    /// In-place LU with partial pivoting of the row-major n x n matrix A (PA = LU, unit L below
    /// the diagonal, U on and above). piv[k] = row swapped with row k at step k. Returns the number
    /// of exactly zero / non-finite pivots (elimination skips such a column; never perturbed).
    static int lu_factor(int n, real* A, int* piv);
    /// Solves A x = b in place (b <- x) with the factors of lu_factor.
    static void lu_solve(int n, const real* LU, const int* piv, real* b);
    /// Solves A^T x = b in place with the factors of lu_factor.
    static void lu_solve_transpose(int n, const real* LU, const int* piv, real* b);
    /// Hager-Higham estimate of ||A^-1||_1 from the factors (work: 2 n scratch).
    static real inv_norm1_estimate(int n, const real* LU, const int* piv, real* work);

  private:
    int n_ = 0;
    int P_ = 0;
    int K_ = 0;
    bool built_ = false;
    DeviceBuffer<real> Ecols_;   ///< K * K, column c at c * K (V^T A v_c)
    DeviceBuffer<real> basis_;   ///< 2 N^3 dense basis vector
    DeviceBuffer<real> Abasis_;  ///< 2 N^3
    DeviceBuffer<real> t1_, t2_; ///< 2 N^3 work vectors of apply(mult)
    DeviceBuffer<real> rc_, yc_; ///< K each
    std::vector<real> E_;        ///< host E, row-major
    std::vector<real> LU_;       ///< host LU factors, row-major
    std::vector<int> piv_;
    std::vector<real> work_; ///< 2 K
    real* h_rc_ = nullptr;   ///< pinned K
    real* h_yc_ = nullptr;   ///< pinned K
    std::size_t pinned_K_ = 0;
    int n_apply_ = 0;
    double t_apply_ = 0.0;
    double t_host_solve_ = 0.0;
};

} // namespace inlet_slab
} // namespace streamfunctions
} // namespace macroflow3d
