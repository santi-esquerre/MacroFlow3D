#pragma once

/**
 * @file InletLabels.cuh
 * @brief SF-33 N3: inlet labels of the x1-non-periodic slab (deviation D-1, normalized triangular
 *        construction), built spectrally from face samples of v1(0, x2, x3); host mirror and GPU
 *        batched evaluation.
 *
 * Authoritative definition: `docs/experiments/artifacts/2026-10-02-sf29-inlet-labels/scripts/
 * inlet.py` (class InletLabels) and the artifact README section "Inlet labels (deviation D-1)";
 * decision record `docs/decisions/2026-10-06-eq14-inlet-label-formulation.md` item 2.
 *
 * Construction. The face samples s(a2, a3), a = 0..nf-1, sit at
 *   x2 = (a2 + o2) / nf,   x3 = (a3 + o3) / nf          (offset (o2, o3) in cell units)
 * (o = 0: the prototype's samples at m h; o = 1/2: the SF-19 U-face centres (m + 1/2) h). Storage
 * of the samples: s[a3 + nf * a2] (x3 fastest, the slab in-plane layout). The 2-D trigonometric
 * interpolant of the samples AT THEIR TRUE POSITIONS is
 *   V(x2, x3) = sum_{m2, m3} V(m2, m3) e2 e3,  e_j = exp(i 2 pi m_j x_j),
 *   V(m2, m3) = DFT(s)(m2, m3) / nf^2 * exp(-i 2 pi (m2 o2 + m3 o3) / nf),
 * with signed modes |m| <= nf/2 - 1: the Nyquist lines (|m| = nf/2, even nf) are DROPPED, as the
 * prototype does. The setup DFT is a host separable direct DFT with an exact twiddle table
 * (index (m a) mod nf), O(nf^3) operations, run once per build (nf <= 256: ~1.7e7 complex
 * products). Then, exactly for V (inlet.py): Qh(m3)      = V(0, m3) Q(x3) = sum Qh e3,  Q0 = Re
 * V(0, 0) A(m2, m3)   = V(m2, m3) / (i 2 pi m2), m2 != 0;  A(0, .) = 0 B(m3)       = Qh(m3) / (i 2
 * pi m3),    m3 != 0;  B(0) = 0 num(x2, x3) = Re sum_{m2, m3} A(m2, m3) (e2 - 1) e3 psi1^0      =
 * x2 + num / Q(x3),   psi2^0 = Q0 x3 + Re sum_{m3} B(m3) (e3 - 1) so that d2 psi1^0 d3 psi2^0 - d3
 * psi1^0 d2 psi2^0 = V on the face (host `jacobian`).
 *
 * Unwrapped points: the periodic sums are evaluated at the canonical reductions
 * x_r = x - floor(x) (identical on host and device), the affine parts x2, Q0 x3 at the unwrapped
 * coordinate; the labels are therefore defined at arbitrary real face points.
 *
 * Positivity: the build requires min of the samples > 0 (finite); otherwise it throws
 * InletBackflowError (the `inlet_backflow` status; never silently continued).
 *
 * GPU batched evaluation (evaluate_labels). Points are processed in chunks of at most P_c points
 * (prepare_device(P_c)); per chunk:
 *   1. kernel: E2m1(p, m2) = exp(i 2 pi m2 y_r(p)) - 1   (P_c x M column-major table, M = active
 *      modes = nf - 1 for even nf);
 *   2. cuBLAS ZGEMM on ctx's handle (bound to ctx.cuda_stream()): T = E2m1 * A  (P_c x M) * (M x
 * M);
 *   3. kernel: per point, one pass over m3 with e3 = exp(i 2 pi m3 z_r) (one sincos per mode):
 *      Q = Re sum Qh e3, num = Re sum T(p, m3) e3, s2 = Re sum B (e3 - 1);
 *      psi1 = y + num / Q, psi2 = Q0 z + s2.
 * Cost model per point: 2 M sincos + 8 M^2 real flops (the GEMM, dominant) + O(M) flops. At
 * nf = 256 (M = 255) and 2.1e6 points (128^3 + one plane): 1.1e12 flops in ZGEMM (~0.2-0.4 s on a
 * V100 at 3-6 TFLOP/s FP64) + 1.1e9 sincos (~0.05 s); at nf = 128 a quarter of that. Device memory:
 * 2 P_c M complex doubles (E2m1, T) + the tables (M^2 + 3 M complex doubles + M doubles); the
 * default P_c = 16384 gives 134 MB at M = 255.
 * Contract: prepare_device is the only allocating call; evaluate_labels allocates nothing, performs
 * no host synchronization (stream-ordered on ctx.cuda_stream(), the caller synchronizes) and throws
 * std::logic_error if the device tables were not prepared. ZGEMM is deterministic for a fixed
 * device / build / chunk geometry; the summation order of the GPU path differs from the host mirror
 * (the two agree to roundoff, gated at 1e-13 in tests/inlet_slab/slab_production_tests.cu).
 */

#include "../../../core/DeviceBuffer.cuh"
#include "../../../core/DeviceSpan.cuh"
#include "../../../core/Scalar.hpp"
#include "../../../runtime/CudaContext.cuh"

#include <complex>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuComplex.h>

namespace macroflow3d {
namespace streamfunctions {
namespace inlet_slab {

/// `inlet_backflow`: the inlet-face v1 samples are not strictly positive (or not finite). The D-1
/// construction is undefined; the caller must stop (backflow is outside the formulation).
class InletBackflowError : public std::runtime_error {
  public:
    InletBackflowError(const std::string& what, real vmin)
        : std::runtime_error(what), vmin_(vmin) {}
    real vmin() const noexcept { return vmin_; }

  private:
    real vmin_;
};

class InletLabels {
  public:
    using cplx = std::complex<real>;

    InletLabels() = default;
    InletLabels(const InletLabels&) = delete;
    InletLabels& operator=(const InletLabels&) = delete;
    InletLabels(InletLabels&&) noexcept = default;
    InletLabels& operator=(InletLabels&&) noexcept = default;

    /**
     * Host build from nf x nf samples s[a3 + nf * a2] at ((a2 + o2)/nf, (a3 + o3)/nf).
     * Throws std::invalid_argument (nf < 4, size mismatch, non-finite offsets) and
     * InletBackflowError (min sample <= 0 or a non-finite sample).
     */
    static InletLabels build(const std::vector<real>& samples, int nf, real o2, real o3);

    // ---- metadata
    int nf() const { return nf_; }
    int modes() const { return static_cast<int>(m_.size()); } ///< M: active modes per axis
    const std::vector<int>& mode_indices() const {
        return m_;
    } ///< signed m (fftfreq order, no Nyquist)
    real Q0() const { return Q0_; }
    real vmin() const { return vmin_; }
    real sample_mean() const { return sample_mean_; }
    const std::vector<cplx>& Qh() const { return Qh_; }
    const std::vector<cplx>& A() const { return A_; } ///< A[m2i + M * m3i] (column-major)
    const std::vector<cplx>& B() const { return B_; }
    const std::vector<cplx>& V_coefficients() const { return V_; } ///< V[m2i + M * m3i]

    // ---- host evaluation (mirror of inlet.py)
    /// Q(x3) and its derivatives d^deriv Q / dx3^deriv.
    real Q(real z, int deriv = 0) const;
    /// num(x2, x3) (periodic part of int_0^{x2} V ds) and its derivatives (d2 <= 1, d3 <= 1).
    real num(real y, real z, int d2 = 0, int d3 = 0) const;
    real psi1(real y, real z) const;
    real psi2(real z) const;
    /// 2-D trigonometric interpolant V of the samples.
    real V(real y, real z) const;
    /// d2 psi1 d3 psi2 - d3 psi1 d2 psi2 from the analytic derivatives of the representation.
    real jacobian(real y, real z) const;

    // ---- GPU
    /// Uploads the tables and allocates the chunk workspace (P_c points). The only allocating call.
    void prepare_device(int max_points_per_chunk = 16384);
    bool device_prepared() const { return chunk_ > 0; }
    int chunk_points() const { return chunk_; }
    std::size_t device_bytes() const;
    /// Data pointers of every owned device buffer, for allocation-stability tests (SF-33 C1).
    std::vector<const void*> device_storage_pointers() const {
        return {d_kk_.data(), d_A_.data(), d_Qh_.data(), d_B_.data(), d_E2m1_.data(), d_T_.data()};
    }
    /// psi1, psi2 at the points (py[p], pz[p]); all four spans of equal length. Enqueue only.
    void evaluate_labels(CudaContext& ctx, DeviceSpan<const real> py, DeviceSpan<const real> pz,
                         DeviceSpan<real> psi1, DeviceSpan<real> psi2) const;

  private:
    int nf_ = 0;
    std::vector<int> m_;
    std::vector<real> kk_; ///< 2 pi m
    std::vector<cplx> V_, A_, Qh_, B_;
    real Q0_ = 0.0;
    real vmin_ = 0.0;
    real sample_mean_ = 0.0;

    int chunk_ = 0;
    DeviceBuffer<real> d_kk_;
    DeviceBuffer<cuDoubleComplex> d_A_, d_Qh_, d_B_;
    // chunk scratch: written by the (logically const) evaluation
    mutable DeviceBuffer<cuDoubleComplex> d_E2m1_, d_T_;
};

} // namespace inlet_slab
} // namespace streamfunctions
} // namespace macroflow3d
