#pragma once

/**
 * @file SlabModePreconditioner.cuh
 * @brief SF-33 N2: preconditioner P-A of the inlet-slab Newton-Krylov solver: per transverse
 *        Fourier mode, the exact inverse of the PLANE-AVERAGED, ROW-SCALED frozen linearization.
 *
 * Operator (variant implemented: the FULL plane-averaged linearization, not the reduced lin0)
 * ------------------------------------------------------------------------------------------
 * The N1 JVP at a vertex is linear in the 18 stencil derivatives of the direction,
 *   d = (dg1[0..2], dg2[0..2], dH1[0..5], dH2[0..5])     (Hessian packing kH00..kH12),
 *   (J du)_row = q_row * sum_k C_row,k(x) d_k(du),
 * with pointwise coefficients C that depend on the frozen base state (g_i, H_i, B, c, S_i,
 * grad ln k). Here C_row,k = the derivative of the ROW-SCALED residual E / q: for the equation rows
 * (planes 1..N-1) the N1 `slab_equation_point_jvp` evaluated with q = 1 on the 18 unit vectors
 * (exact by linearity: it contains grad ln k and every S-term of the base state); for the outlet
 * rows (plane N) `slab_outlet_point_jvp` with q = 1 on the 6 gradient unit vectors (the H
 * coefficients are 0). The coefficients are AVERAGED over each plane (x2, x3) (deterministic block
 * reduction, one block per plane), giving a(j, i, k), j = 1..N, i = row field, k = 0..17. With
 * constant-in-plane coefficients the operator is diagonal in the transverse Fourier modes, exactly,
 * because every in-plane stencil is translation invariant:
 *   d2 -> i s2, d3 -> i s3, d22 -> a2, d33 -> a3, d23 = d3 d2 -> -s2 s3,
 *   d1 -> D1 (x1 stencil rows of N0, plane 0 dropped: the direction vanishes on the inlet),
 *   d11 -> D11, d12 -> D1 * (i s2), d13 -> D1 * (i s3),
 *   a2 = (-2 cos 2T + 32 cos T - 30) / (12 h^2),  s2 = (8 sin T - sin 2T) / (6 h),  T = 2 pi m2 / N
 * (same for m3), i.e. the 4th-order symbols of the prototype's `ModePrec(order=4)`. The outlet rows
 * use only the one-sided plane-N d1 row (the plane-N d11 row of N0 is never referenced).
 * Applying the preconditioner: z = M^-1 r with M = diag(q) P, i.e.
 *   r / q (pointwise row scaling, the prototype's `rowscale`)  ->  cuFFT batched 2-D R2C over the
 *   2 N planes (each an N x N contiguous block of the N0 layout)  ->  per mode (m2, m3h),
 *   N x (N/2 + 1) modes, solve the 2N x 2N complex banded system P_mode z_mode = r_mode  ->
 *   C2R  ->  scale by 1/N^2 (cuFFT is unnormalized).
 * At k = 1, u = 0 every coefficient is constant, the row-scaled J IS P and M^-1 J x = x to roundoff
 * (the prototype's `--k1check`, `lin0`): this is acceptance item 1 of the node. For u != 0 or
 * k = k(x1, x2, x3), P-A is an approximation (its quality is an N6 gate item, not asserted here).
 *
 * Pseudo-time shift (SF-33 N7b): factor(..., mu) assembles P-A for the SHIFTED operator
 * J + mu D, D = diag(q_v / h^2) on the equation rows (planes 1..N-1), 0 on the outlet rows. In the
 * row-scaled frame of P (E / q) the shift is D / q = 1 / h^2 on every equation row, a constant
 * whose plane average is itself; P gains + mu / h^2 on the diagonal of the equation rows of every
 * mode and M = diag(q) P carries exactly the pointwise shift mu q_v / h^2. Hence M is exact for the
 * shifted operator wherever it is exact for J (k = 1, u = 0; x1-only states): ||M^-1 (J + mu D) x -
 * x|| /
 * ||x|| = roundoff for every mu (test inlet_slab_newton case 1s). (This is the "mu qbar_j / h^2"
 * of the N7b specification written in P's row-scaled frame: q / q = 1 has plane average 1; adding
 * qbar_j / h^2 to the row-scaled diagonal instead would give diag(q) qbar_j / h^2, i.e. a q^2
 * weighting, which is not the averaged shifted operator.) mu = 0 skips the addition: bitwise the
 * N7a band.
 *
 * Banded LU (custom batched kernel, one thread per mode)
 * -----------------------------------------------------
 * Unknown ordering inside a mode: row = 2 (j - 1) + field (fields interleaved per plane). The x1
 * stencils reach at most 4 planes away (skewed d11 on planes 1 and N-1, one-sided d1 on plane N),
 * so kl = ku = 2*4 + 1 = 9. LU with partial pivoting (LAPACK zgbtf2 / zgbtrs algorithm, pivot =
 * max |re| + |im|), band storage ldab = 2 kl + ku + 1 = 28 rows; storage is mode-fastest
 * (element e of mode md at e * n_modes + md) so the one-thread-per-mode kernels coalesce.
 * A zero or non-finite pivot is COUNTED (factor() returns singular_modes; nothing is perturbed).
 * Factored once per Newton step (factor()).
 *
 * Memory (bytes), N^3 grid, M = N (N/2 + 1) modes, n = 2N:
 *   band   28 * n * M * 16          (954 MB at 128^3, 15.4 MB at 32^3)
 *   pivots n * M * 4
 *   spec   n * M * 16                (complex spectra of the 2N planes)
 *   scratch 2 N^3 * 8, coefficients 36 N * 8, cuFFT work areas (reported by cuFFT).
 * `allocated_bytes()` returns the exact device footprint including the cuFFT work areas.
 *
 * Synchronization / allocation
 * ----------------------------
 *   prepare(grid): the only allocating call (grow-only buffers, cuFFT plans created here; the
 *     plans are re-created only when N changes). Builds the stencil table (synchronous upload).
 *   factor(...): 2 kernels + 1 device-to-host copy of the singular-mode counter and EXACTLY ONE
 *     cudaStreamSynchronize (documented; once per Newton step). References `inputs` (q) for apply:
 *     the inputs object must outlive and stay unchanged until the next factor().
 *   apply(...): 3 kernels + 2 cuFFT executions on ctx.cuda_stream(); no allocation, no host sync.
 */

#include "../../../core/DeviceBuffer.cuh"
#include "../../../core/DeviceSpan.cuh"
#include "../../../core/Scalar.hpp"
#include "../../../runtime/CudaContext.cuh"
#include "InletSlabGrid.cuh"
#include "SlabSolverTypes.cuh"
#include "SlabStencils4.cuh"

#include <cufft.h>

#include <cstddef>
#include <vector>

namespace macroflow3d {
namespace streamfunctions {
namespace inlet_slab {

class SlabModePreconditioner {
  public:
    static constexpr int kCoefPerRow = 18;   ///< (dg1, dg2, dH1, dH2)
    static constexpr int kCoefPerPlane = 36; ///< 2 row fields x 18
    static constexpr int kKl = 9;
    static constexpr int kKu = 9;
    static constexpr int kLdab = 2 * kKl + kKu + 1;

    SlabModePreconditioner() = default;
    ~SlabModePreconditioner();
    SlabModePreconditioner(const SlabModePreconditioner&) = delete;
    SlabModePreconditioner& operator=(const SlabModePreconditioner&) = delete;

    void prepare(CudaContext& ctx, const InletSlabGrid& g);
    bool prepared_for(const InletSlabGrid& g) const { return n_ == g.n && n_ > 0; }

    /**
     * Plane-averaged frozen linearization at the base state (U1, U2: full periodic-part arrays,
     * planes 0..N, as evaluate_residual takes them; inputs: q, grad ln k), assembled per mode and
     * LU-factored. One documented host sync (singular-mode count). mu >= 0: pseudo-time shift
     * (SF-33 N7b; see the header), 0 = unshifted.
     */
    SlabPrecFactorReport factor(CudaContext& ctx, const InletSlabGrid& grid,
                                const SlabStageInputs& inputs, DeviceSpan<const real> U1,
                                DeviceSpan<const real> U2, real mu = 0.0);
    bool factored() const { return inputs_ != nullptr; }

    /// out = M^-1 in (2 N^3 each, field-major). Enqueue only. in and out must not overlap.
    void apply(CudaContext& ctx, const InletSlabGrid& grid, DeviceSpan<const real> in,
               DeviceSpan<real> out);

    /// Plane-averaged coefficients a(j, i, k) at index ((j - 1) * 2 + i) * 18 + k (host copy,
    /// sync).
    std::vector<real> download_coefficients(CudaContext& ctx) const;

    std::size_t allocated_bytes() const;
    std::vector<const void*> buffer_pointers() const;
    static std::size_t band_bytes(int N) {
        const std::size_t M = static_cast<std::size_t>(N) * static_cast<std::size_t>(N / 2 + 1);
        return static_cast<std::size_t>(kLdab) * 2 * static_cast<std::size_t>(N) * M * 16;
    }
    int n_modes() const { return n_modes_; }

  private:
    void destroy_plans();
    int n_ = 0;
    int n_modes_ = 0;
    SlabStencilTable table_;
    DeviceBuffer<real> coef_;               ///< N * 36
    DeviceBuffer<cufftDoubleComplex> band_; ///< kLdab * 2N * n_modes
    DeviceBuffer<int> ipiv_;                ///< 2N * n_modes
    DeviceBuffer<cufftDoubleComplex> spec_; ///< 2N * n_modes
    DeviceBuffer<real> scratch_;            ///< 2 N^3
    DeviceBuffer<int> singular_;            ///< 1 counter
    DeviceBuffer<char> fft_work_;           ///< shared cuFFT work area
    cufftHandle plan_r2c_ = 0, plan_c2r_ = 0;
    bool plans_ = false;
    std::size_t fft_work_bytes_ = 0;
    const SlabStageInputs* inputs_ = nullptr;
    int host_singular_ = 0;
};

} // namespace inlet_slab
} // namespace streamfunctions
} // namespace macroflow3d
