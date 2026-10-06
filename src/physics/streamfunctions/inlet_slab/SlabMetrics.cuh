#pragma once

/**
 * @file SlabMetrics.cuh
 * @brief SF-33 N0: Gate 3A label metrics of the inlet slab, the prototype's
 * `metrics.fd_metrics(order=4)` with the C-i4 per-label e_psi normalization, and the CASE line
 * formatter.
 *
 * Definitions (all RMS over the (N+1) N^2 vertices of planes 0..N; vector RMS over the Euclidean
 * norms): g_i      = 4th-order gradients of the periodic parts on ALL planes (metrics.d1_fd4:
 * one-sided 5-point on planes 0 / N, skewed on 1 / N-1, centered elsewhere; dp_fd4 in plane) +
 * exact affine e2 / e3; c        = g1 x g2; v_rms    = RMS|vD|;  e_v = RMS|c - vD| / v_rms; e_i =
 * RMS(vD . g_i) / (v_rms RMS|g_i|); e_div    = RMS(div_h c) / v_rms,  div_h c = d1_fd4(c1) +
 * dp_fd4(c2, x2) + dp_fd4(c3, x3); min_c, p0.1, p1, p5, p50: min and percentiles of |c|; vD_min,
 * vD_p: the same of |vD|; percentiles use numpy's default definition: virtual index (n - 1) (p /
 * 100), lower / upper order statistics a, b, gamma = index - floor(index), value = a + (b - a)
 * gamma if gamma < 0.5 else b - (b - a)(1 - gamma); labels   psi1 = x2 + U1, psi2 = x3 + U2 (x = m
 * / N); a_psi_i  = RMS(psi_i - psi_i^or); den_i = RMS(psi_i^or - affine_i); den_ref = max(den_1,
 * den_2); den(i) = den_i if den_i > 1e-6 den_ref else den_ref; e_psi_i = a_psi_i / den(i) if den(i)
 * > 0 else a_psi_i; e_psi = max(e_psi1, e_psi2). Nothing is regularized or clamped. If |c| or |vD|
 * is non-finite anywhere, `nonfinite` counts the vertices and the percentile / min entries are NaN
 * (the sums propagate the non-finite values as they are).
 *
 * Synchronization / allocation contract:
 *   - SlabMetricsWorkspace::prepare(grid) is the only allocating call (grow-only: c field, |c| /
 * |vD| arrays and their sorted copies, reduction partials, CUB radix-sort temporary storage queried
 * here, stencil table).
 *   - evaluate_metrics never allocates. It enqueues: the pointwise kernel (gradients, c, norms, 11
 * sums), the div kernel, two CUB DeviceRadixSort::SortKeys (preallocated temp storage), the
 * fixed-order reductions, a gather of the order statistics; then ONE cudaMemcpyAsync of 30 doubles
 * to the host and EXACTLY ONE cudaStreamSynchronize at the end. Final scalar arithmetic is on the
 * host.
 *   - Reductions are deterministic (fixed launch geometry and fixed tree; see InletSlabGrid.cuh
 * detail::).
 */

#include "../../../core/DeviceBuffer.cuh"
#include "../../../core/DeviceSpan.cuh"
#include "../../../core/Scalar.hpp"
#include "../../../runtime/CudaContext.cuh"
#include "InletSlabGrid.cuh"
#include "SlabStencils4.cuh"

#include <cstddef>
#include <string>
#include <vector>

namespace macroflow3d {
namespace streamfunctions {
namespace inlet_slab {

/// The e_psi normalization threshold of the C-i4 metrics (metrics.E_PSI_DEN_REL).
constexpr real kEPsiDenRel = 1e-6;

/// numpy default (`method="linear"`) percentile of an ascending array (host; exact numpy operation
/// order).
real percentile_linear_sorted(const real* sorted, std::size_t n, real pct);

class SlabMetricsWorkspace {
  public:
    void prepare(const InletSlabGrid& g);
    bool prepared_for(const InletSlabGrid& g) const {
        return n_ == g.n && n_ > 0 && table_.built_for(g);
    }
    std::size_t allocated_bytes() const;

    /// Sorted |c| and |vD| of the last evaluate_metrics (device, full_size each) - for tests /
    /// diagnostics.
    const DeviceBuffer<real>& sorted_c() const { return cn_sorted_; }
    const DeviceBuffer<real>& sorted_vD() const { return vn_sorted_; }
    /// Unsorted |c| at the vertices (full-array layout) of the last evaluation.
    const DeviceBuffer<real>& norm_c() const { return cn_; }
    /// Data pointers of every owned device buffer, for allocation-stability tests.
    std::vector<const void*> storage_pointers() const;

  private:
    friend SlabMetrics evaluate_metrics(CudaContext&, const InletSlabGrid&, DeviceSpan<const real>,
                                        DeviceSpan<const real>, const SlabReferenceData&,
                                        SlabMetricsWorkspace&);
    int n_ = 0;
    int nblocks_ = 0;
    SlabStencilTable table_;
    DeviceBuffer<real> c_[3];
    DeviceBuffer<real> cn_, vn_, cn_sorted_, vn_sorted_;
    DeviceBuffer<real> partials_; ///< (11 + 1) * nblocks
    DeviceBuffer<real> out_;      ///< 32 doubles: 11 sums, div sum, 18 gathered order statistics
    DeviceBuffer<unsigned char> sort_temp_;
    std::size_t sort_temp_bytes_ = 0;
    real host_out_[32] = {};
};

/**
 * Metrics of the labels psi1 = x2 + U1, psi2 = x3 + U2 against the reference.
 *   U1, U2: periodic parts on planes 0..N (full arrays, read-only).
 *   ref:    vD is required (has_vD); psi_or (full labels) optional -> has_psi / e_psi*.
 */
SlabMetrics evaluate_metrics(CudaContext& ctx, const InletSlabGrid& grid, DeviceSpan<const real> U1,
                             DeviceSpan<const real> U2, const SlabReferenceData& ref,
                             SlabMetricsWorkspace& ws);

/// u = psi - affine on planes 0..N (psi_or -> periodic parts, e.g. for the oracle ceiling line).
/// Enqueue only.
void labels_to_periodic_parts(CudaContext& ctx, const InletSlabGrid& grid,
                              DeviceSpan<const real> psi1, DeviceSpan<const real> psi2,
                              DeviceSpan<real> U1, DeviceSpan<real> U2);

} // namespace inlet_slab
} // namespace streamfunctions
} // namespace macroflow3d
