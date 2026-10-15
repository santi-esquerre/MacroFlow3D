#pragma once

/**
 * @file SlabGmres.cuh
 * @brief SF-33 N2: restarted right-preconditioned GMRES on raw device vectors of the inlet slab
 *        (2 N^3 doubles, field-major [u1 | u2], InletSlabGrid.cuh layout), plus the small
 *        deterministic vector kernels it and the Newton driver use.
 *
 * Algorithm (the SF-29 prototype's `gmres_right`, with MGS instead of CGS2)
 * ------------------------------------------------------------------------
 * Solves A x = b with right preconditioning, A M^-1 y = b, x = M^-1 y, from x = 0:
 *   cycle: v_0 = r / ||r||, g = (||r||, 0, ...);
 *     inner k = 0..m-1:  w = A M^-1 v_k;  modified Gram-Schmidt against v_0..v_k with the
 *       coefficients kept on the device (dot -> device slot -> axpy reads the slot); ONE extra MGS
 *       pass when ||w_after|| < (1/sqrt 2) ||w_before|| (counted); h_{k+1,k} = ||w||; Givens
 *       rotations on the host; the cycle ends when |g_{k+1}| / ||b|| < inner_tol_factor * tol (0.1
 *       tol, the prototype), on a happy breakdown (h_{k+1,k} = 0), on a singular rotated diagonal
 *       (status breakdown), or at the inner cap.
 *     end of cycle: y = R^-1 g (host back substitution), x += M^-1 (V y) (deferred preconditioner
 *       application, one M^-1 per cycle), then the TRUE residual r = b - A x and ||r|| / ||b|| are
 *       recomputed (never the recurrence alone); the recurrence estimate |g_k| / ||b|| and the true
 *       value are both recorded per cycle.
 *   stop: converged       true relative residual <= tol;
 *         nonfinite       any non-finite norm / Hessenberg entry / true residual;
 *         breakdown       singular rotated Hessenberg diagonal (x updated with the nonsingular
 * part); stagnation      true residual at the end of a cycle > 0.9 x the one two cycles earlier;
 *         max_iterations  total inner iterations reached max_iterations (default 6000).
 *   b = 0 returns x = 0, converged, 0 iterations.
 *
 * Memory: basis V of (m + 1) vectors of 2 N^3 doubles = 2 (m+1) N^3 * 8 bytes (m = 50:
 * 1.71 GB at 128^3, 26.7 MB at 32^3) + 3 work vectors (w, z, r) 3 * 2 N^3 * 8 bytes + O(m) scalars
 * and the reduction partials. `basis_bytes(N, m)` returns the basis term; `allocated_bytes()`
 * the exact device footprint.
 *
 * Host synchronizations (documented, explicit; all on ctx.cuda_stream()):
 *   - ||b|| at entry: 1;
 *   - per inner iteration: 1 (H column + ||w|| before / after the MGS pass copied to the host for
 *     the Givens recurrence and the re-orthogonalization test), +1 when a second MGS pass runs;
 *   - per cycle: 1 (the true residual norm).
 * Device scalars: the MGS coefficients never round-trip through the host before their axpy.
 * No DEVICE allocation in solve(): every device buffer and the host Hessenberg / Givens scalars
 * are sized in prepare(). The per-cycle vectors of the returned report (host memory only) are
 * reserved once at the start of solve() (bounded by max_iterations), never inside the cycles.
 *
 * Reductions: grid-stride kernels with a grid size that depends only on the vector length
 * (detail::slab_reduce_blocks), fixed block tree and fixed-order final pass: bitwise reproducible.
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

// ------------------------------------------------------------------------------------------------
// Deterministic vector kernels (enqueue only unless the name says _host)
// ------------------------------------------------------------------------------------------------

/// y += alpha x
void slab_axpy(CudaContext& ctx, real alpha, DeviceSpan<const real> x, DeviceSpan<real> y);
/// y += sign * alpha_dev[0] * x  (alpha read on the device: no host round trip)
void slab_axpy_dev(CudaContext& ctx, const real* alpha_dev, real sign, DeviceSpan<const real> x,
                   DeviceSpan<real> y);
/// y = alpha x
void slab_scale_copy(CudaContext& ctx, real alpha, DeviceSpan<const real> x, DeviceSpan<real> y);
/// out = a x + b y
void slab_axpby(CudaContext& ctx, real a, DeviceSpan<const real> x, real b,
                DeviceSpan<const real> y, DeviceSpan<real> out);
/// y = value
void slab_fill(CudaContext& ctx, real value, DeviceSpan<real> y);

/**
 * Deterministic reductions over vectors of one fixed length (set by prepare). The *_host variants
 * perform exactly one cudaStreamSynchronize; dot_device only enqueues (result in a device slot).
 */
class SlabReduction {
  public:
    void prepare(std::size_t n, int slots);
    std::size_t length() const { return n_; }
    /// slots[slot] = a . b (enqueue only)
    void dot_device(CudaContext& ctx, DeviceSpan<const real> a, DeviceSpan<const real> b, int slot);
    /// Copies slots[first .. first + count) to host (one sync).
    void slots_to_host(CudaContext& ctx, int first, int count, real* host);
    real* slot_ptr(int slot) { return slots_.data() + slot; }
    real dot_host(CudaContext& ctx, DeviceSpan<const real> a, DeviceSpan<const real> b);
    real nrm2_host(CudaContext& ctx, DeviceSpan<const real> a);
    /// max |a_i| (NaN if any entry is NaN); one sync.
    real maxabs_host(CudaContext& ctx, DeviceSpan<const real> a);
    std::size_t allocated_bytes() const;
    std::vector<const void*> buffer_pointers() const;

  private:
    std::size_t n_ = 0;
    int nblocks_ = 0;
    int nslots_ = 0;
    DeviceBuffer<real> partials_;
    DeviceBuffer<real> slots_;
};

// ------------------------------------------------------------------------------------------------
// GMRES
// ------------------------------------------------------------------------------------------------

class SlabGmres {
  public:
    /// out = Op(in); in and out never alias; implementations must be stream-ordered on ctx.
    using Operator = std::function<void(DeviceSpan<const real> in, DeviceSpan<real> out)>;

    /// Allocates the basis (restart + 1 vectors), w, z, r and the reduction workspace for vectors
    /// of grid.unknown_size(), and the host Hessenberg / Givens scalars; max_iterations bounds the
    /// per-cycle report reservation made at the start of solve(). The only device-allocating call
    /// (grow-only).
    void prepare(const InletSlabGrid& grid, int restart, int max_iterations = 6000);
    bool prepared_for(const InletSlabGrid& grid, int restart) const {
        return n_ == grid.unknown_size() && n_ > 0 && restart <= m_;
    }

    /// Solves A x = b (x overwritten, start x = 0). See the header comment for the algorithm.
    SlabGmresReport solve(CudaContext& ctx, const Operator& A, const Operator& Minv,
                          DeviceSpan<const real> b, DeviceSpan<real> x, const SlabGmresConfig& cfg);

    static std::size_t basis_bytes(int N, int restart) {
        const std::size_t n3 = static_cast<std::size_t>(N) * N * N;
        return 2 * static_cast<std::size_t>(restart + 1) * n3 * sizeof(real);
    }
    std::size_t allocated_bytes() const;
    std::vector<const void*> buffer_pointers() const;
    SlabReduction& reduction() { return red_; }

  private:
    DeviceSpan<real> vec(int i) {
        return DeviceSpan<real>(V_.data() + static_cast<std::size_t>(i) * n_, n_);
    }
    std::size_t n_ = 0;
    int m_ = 0;
    int max_its_reserved_ = 0;
    DeviceBuffer<real> V_; ///< (m + 1) * n
    DeviceBuffer<real> w_, z_, r_;
    SlabReduction red_; ///< slots: 0..m (MGS coefficients), m+1 (||w||^2 before), m+2 (after)
    // host scalars (sized in prepare)
    std::vector<real> H_; ///< (m + 1) x m, column-major: H_[i + (m + 1) k]
    std::vector<real> cs_, sn_, g_, y_, col_;
};

} // namespace inlet_slab
} // namespace streamfunctions
} // namespace macroflow3d
