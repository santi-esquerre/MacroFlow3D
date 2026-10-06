/**
 * @file InletLabels.cu
 * @brief SF-33 N3: D-1 inlet labels (host build + mirror, GPU batched evaluation). See
 *        InletLabels.cuh for the definitions, the cost model and the allocation contract.
 */

#include "InletLabels.cuh"

#include "../../../runtime/cuda_check.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>

#include <cublas_v2.h>

namespace macroflow3d {
namespace streamfunctions {
namespace inlet_slab {

namespace {

constexpr real kPi = 3.141592653589793238462643383279502884;
constexpr real kTwoPi = 2.0 * kPi;
constexpr int kBlock = 256;

/// Canonical reduction to [0, 1) (identical on host and device).
__host__ __device__ inline real reduce_unit(real x) {
    real r = x - floor(x);
    if (r >= 1.0)
        r -= 1.0;
    if (r < 0.0)
        r += 1.0;
    return r;
}

__global__ void e2m1_kernel(const real* __restrict__ py, std::size_t offset, int count, int pc,
                            int M, const real* __restrict__ kk, cuDoubleComplex* __restrict__ E) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    const int a = blockIdx.y;
    if (p >= count || a >= M)
        return;
    const real yr = reduce_unit(py[offset + static_cast<std::size_t>(p)]);
    real s, c;
    sincos(kk[a] * yr, &s, &c);
    E[static_cast<std::size_t>(p) + static_cast<std::size_t>(pc) * a] =
        make_cuDoubleComplex(c - 1.0, s);
}

__global__ void finish_kernel(const real* __restrict__ py, const real* __restrict__ pz,
                              std::size_t offset, int count, int pc, int M,
                              const real* __restrict__ kk, const cuDoubleComplex* __restrict__ Qh,
                              const cuDoubleComplex* __restrict__ B,
                              const cuDoubleComplex* __restrict__ T, real Q0,
                              real* __restrict__ psi1, real* __restrict__ psi2) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= count)
        return;
    const std::size_t g = offset + static_cast<std::size_t>(p);
    const real y = py[g];
    const real z = pz[g];
    const real zr = reduce_unit(z);
    real q = 0.0, nm = 0.0, s2 = 0.0;
    for (int b = 0; b < M; ++b) {
        real s, c;
        sincos(kk[b] * zr, &s, &c);
        const cuDoubleComplex t = T[static_cast<std::size_t>(p) + static_cast<std::size_t>(pc) * b];
        q += Qh[b].x * c - Qh[b].y * s;
        nm += t.x * c - t.y * s;
        s2 += B[b].x * (c - 1.0) - B[b].y * s;
    }
    psi1[g] = y + nm / q;
    psi2[g] = Q0 * z + s2;
}

template <class T> void upload(DeviceBuffer<T>& d, const T* h, std::size_t n) {
    d.resize(n);
    if (n > 0) {
        MACROFLOW3D_CUDA_CHECK(cudaMemcpy(d.data(), h, n * sizeof(T), cudaMemcpyHostToDevice));
    }
}

} // namespace

InletLabels InletLabels::build(const std::vector<real>& samples, int nf, real o2, real o3) {
    if (nf < 4) {
        throw std::invalid_argument("InletLabels::build: nf must be >= 4");
    }
    const std::size_t n2 = static_cast<std::size_t>(nf) * static_cast<std::size_t>(nf);
    if (samples.size() != n2) {
        throw std::invalid_argument("InletLabels::build: samples.size() != nf * nf");
    }
    if (!std::isfinite(o2) || !std::isfinite(o3)) {
        throw std::invalid_argument("InletLabels::build: offsets must be finite");
    }
    InletLabels L;
    L.nf_ = nf;
    // positivity (inlet_backflow): no clamping, no continuation
    real vmin = std::numeric_limits<real>::infinity();
    bool finite = true;
    real sum = 0.0;
    for (real v : samples) {
        finite = finite && std::isfinite(v);
        vmin = std::min(vmin, v);
        sum += v;
    }
    L.vmin_ = finite ? vmin : std::numeric_limits<real>::quiet_NaN();
    L.sample_mean_ = sum / static_cast<real>(n2);
    if (!finite || !(vmin > 0.0)) {
        char buf[160];
        std::snprintf(buf, sizeof(buf),
                      "inlet_backflow: inlet-face v1 samples not strictly positive (min %.6e%s)",
                      vmin, finite ? "" : ", non-finite sample present");
        throw InletBackflowError(buf, L.vmin_);
    }

    // active modes, fftfreq order, Nyquist (|m| = nf/2, even nf) dropped
    const int half = (nf % 2 == 0) ? nf / 2 - 1 : (nf - 1) / 2;
    for (int m = 0; m <= half; ++m)
        L.m_.push_back(m);
    for (int m = -half; m < 0; ++m)
        L.m_.push_back(m);
    const int M = static_cast<int>(L.m_.size());
    L.kk_.resize(M);
    for (int a = 0; a < M; ++a)
        L.kk_[a] = kTwoPi * static_cast<real>(L.m_[a]);

    // exact twiddle table W^j = exp(-i 2 pi j / nf)
    std::vector<cplx> W(nf);
    for (int j = 0; j < nf; ++j) {
        const real th = kTwoPi * static_cast<real>(j) / static_cast<real>(nf);
        W[j] = cplx(std::cos(th), -std::sin(th));
    }
    auto widx = [nf](int m, int a) {
        long long r = (static_cast<long long>(m) * a) % nf;
        if (r < 0)
            r += nf;
        return static_cast<int>(r);
    };
    // G(a2, m3) = sum_a3 s(a2, a3) W^{m3 a3}
    std::vector<cplx> G(static_cast<std::size_t>(nf) * M);
    for (int a2 = 0; a2 < nf; ++a2) {
        for (int b = 0; b < M; ++b) {
            cplx acc(0.0, 0.0);
            for (int a3 = 0; a3 < nf; ++a3)
                acc += samples[static_cast<std::size_t>(a3) + static_cast<std::size_t>(nf) * a2] *
                       W[widx(L.m_[b], a3)];
            G[static_cast<std::size_t>(b) + static_cast<std::size_t>(M) * a2] = acc;
        }
    }
    // V(m2, m3) = sum_a2 G(a2, m3) W^{m2 a2} / nf^2 * offset phase
    L.V_.assign(static_cast<std::size_t>(M) * M, cplx(0.0, 0.0));
    const real inv = 1.0 / static_cast<real>(n2);
    for (int a = 0; a < M; ++a) {
        for (int b = 0; b < M; ++b) {
            cplx acc(0.0, 0.0);
            for (int a2 = 0; a2 < nf; ++a2)
                acc += G[static_cast<std::size_t>(b) + static_cast<std::size_t>(M) * a2] *
                       W[widx(L.m_[a], a2)];
            const real ph = -kTwoPi *
                            (static_cast<real>(L.m_[a]) * o2 + static_cast<real>(L.m_[b]) * o3) /
                            static_cast<real>(nf);
            L.V_[static_cast<std::size_t>(a) + static_cast<std::size_t>(M) * b] =
                acc * inv * cplx(std::cos(ph), std::sin(ph));
        }
    }
    // tables of inlet.py
    L.Q0_ = L.V_[0].real();
    L.Qh_.resize(M);
    for (int b = 0; b < M; ++b)
        L.Qh_[b] = L.V_[static_cast<std::size_t>(M) * b];
    L.A_.assign(static_cast<std::size_t>(M) * M, cplx(0.0, 0.0));
    for (int a = 0; a < M; ++a) {
        if (L.m_[a] == 0)
            continue;
        const cplx den(0.0, L.kk_[a]);
        for (int b = 0; b < M; ++b) {
            const std::size_t i = static_cast<std::size_t>(a) + static_cast<std::size_t>(M) * b;
            L.A_[i] = L.V_[i] / den;
        }
    }
    L.B_.assign(M, cplx(0.0, 0.0));
    for (int b = 0; b < M; ++b) {
        if (L.m_[b] != 0)
            L.B_[b] = L.Qh_[b] / cplx(0.0, L.kk_[b]);
    }
    return L;
}

// ------------------------------------------------------------------------------------------------
// Host mirror
// ------------------------------------------------------------------------------------------------

real InletLabels::Q(real z, int deriv) const {
    const real zr = reduce_unit(z);
    const int M = modes();
    real acc = 0.0;
    for (int b = 0; b < M; ++b) {
        cplx c = Qh_[b];
        for (int d = 0; d < deriv; ++d)
            c *= cplx(0.0, kk_[b]);
        acc += (c * std::exp(cplx(0.0, kk_[b] * zr))).real();
    }
    return acc;
}

real InletLabels::num(real y, real z, int d2, int d3) const {
    const real yr = reduce_unit(y);
    const real zr = reduce_unit(z);
    const int M = modes();
    std::vector<cplx> e2(M);
    for (int a = 0; a < M; ++a) {
        const cplx e(std::cos(kk_[a] * yr), std::sin(kk_[a] * yr));
        e2[a] = d2 == 0 ? e - 1.0 : e * cplx(0.0, kk_[a]);
    }
    real acc = 0.0;
    for (int b = 0; b < M; ++b) {
        cplx t(0.0, 0.0);
        for (int a = 0; a < M; ++a)
            t += e2[a] * A_[static_cast<std::size_t>(a) + static_cast<std::size_t>(M) * b];
        cplx e3(std::cos(kk_[b] * zr), std::sin(kk_[b] * zr));
        if (d3)
            e3 *= cplx(0.0, kk_[b]);
        acc += (t * e3).real();
    }
    return acc;
}

real InletLabels::psi1(real y, real z) const {
    return y + num(y, z) / Q(z);
}

real InletLabels::psi2(real z) const {
    const real zr = reduce_unit(z);
    const int M = modes();
    real acc = 0.0;
    for (int b = 0; b < M; ++b) {
        const real c = std::cos(kk_[b] * zr), s = std::sin(kk_[b] * zr);
        acc += B_[b].real() * (c - 1.0) - B_[b].imag() * s;
    }
    return Q0_ * z + acc;
}

real InletLabels::V(real y, real z) const {
    const real yr = reduce_unit(y);
    const real zr = reduce_unit(z);
    const int M = modes();
    real acc = 0.0;
    for (int b = 0; b < M; ++b) {
        cplx t(0.0, 0.0);
        for (int a = 0; a < M; ++a)
            t += cplx(std::cos(kk_[a] * yr), std::sin(kk_[a] * yr)) *
                 V_[static_cast<std::size_t>(a) + static_cast<std::size_t>(M) * b];
        acc += (t * cplx(std::cos(kk_[b] * zr), std::sin(kk_[b] * zr))).real();
    }
    return acc;
}

real InletLabels::jacobian(real y, real z) const {
    const real Qz = Q(z), dQ = Q(z, 1);
    const real nm = num(y, z), n2 = num(y, z, 1, 0), n3 = num(y, z, 0, 1);
    const real d2p1 = 1.0 + n2 / Qz;
    const real d3p1 = (n3 * Qz - nm * dQ) / (Qz * Qz);
    const real d2p2 = 0.0;
    const real d3p2 = Qz;
    return d2p1 * d3p2 - d3p1 * d2p2;
}

// ------------------------------------------------------------------------------------------------
// GPU
// ------------------------------------------------------------------------------------------------

void InletLabels::prepare_device(int max_points_per_chunk) {
    if (nf_ < 4) {
        throw std::logic_error("InletLabels::prepare_device: labels not built");
    }
    if (max_points_per_chunk < 1) {
        throw std::invalid_argument("InletLabels::prepare_device: chunk must be >= 1");
    }
    const int M = modes();
    upload(d_kk_, kk_.data(), kk_.size());
    std::vector<cuDoubleComplex> tmp(static_cast<std::size_t>(M) * M);
    for (std::size_t i = 0; i < tmp.size(); ++i)
        tmp[i] = make_cuDoubleComplex(A_[i].real(), A_[i].imag());
    upload(d_A_, tmp.data(), tmp.size());
    tmp.resize(M);
    for (int b = 0; b < M; ++b)
        tmp[b] = make_cuDoubleComplex(Qh_[b].real(), Qh_[b].imag());
    upload(d_Qh_, tmp.data(), static_cast<std::size_t>(M));
    for (int b = 0; b < M; ++b)
        tmp[b] = make_cuDoubleComplex(B_[b].real(), B_[b].imag());
    upload(d_B_, tmp.data(), static_cast<std::size_t>(M));
    const std::size_t tab = static_cast<std::size_t>(max_points_per_chunk) * M;
    d_E2m1_.resize(tab);
    d_T_.resize(tab);
    chunk_ = max_points_per_chunk;
}

std::size_t InletLabels::device_bytes() const {
    return d_kk_.capacity() * sizeof(real) + (d_A_.capacity() + d_Qh_.capacity() + d_B_.capacity() +
                                              d_E2m1_.capacity() + d_T_.capacity()) *
                                                 sizeof(cuDoubleComplex);
}

void InletLabels::evaluate_labels(CudaContext& ctx, DeviceSpan<const real> py,
                                  DeviceSpan<const real> pz, DeviceSpan<real> psi1,
                                  DeviceSpan<real> psi2) const {
    if (!device_prepared()) {
        throw std::logic_error("InletLabels::evaluate_labels: prepare_device() not called");
    }
    const std::size_t n = py.size();
    if (pz.size() != n || psi1.size() != n || psi2.size() != n) {
        throw std::invalid_argument("InletLabels::evaluate_labels: span sizes differ");
    }
    if (n == 0)
        return;
    const int M = modes();
    const cuDoubleComplex one = make_cuDoubleComplex(1.0, 0.0);
    const cuDoubleComplex zero = make_cuDoubleComplex(0.0, 0.0);
    // alpha / beta are host scalars: force (and afterwards restore) the host pointer mode
    cublasPointerMode_t old_mode;
    MACROFLOW3D_CUBLAS_CHECK(cublasGetPointerMode(ctx.cublas_handle(), &old_mode));
    MACROFLOW3D_CUBLAS_CHECK(cublasSetPointerMode(ctx.cublas_handle(), CUBLAS_POINTER_MODE_HOST));
    for (std::size_t off = 0; off < n; off += static_cast<std::size_t>(chunk_)) {
        const int count = static_cast<int>(std::min<std::size_t>(chunk_, n - off));
        const dim3 g1((count + kBlock - 1) / kBlock, M);
        e2m1_kernel<<<g1, kBlock, 0, ctx.cuda_stream()>>>(py.data(), off, count, chunk_, M,
                                                          d_kk_.data(), d_E2m1_.data());
        MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
        // T (count x M) = E2m1 (count x M) * A (M x M), column-major, ld = chunk_
        MACROFLOW3D_CUBLAS_CHECK(cublasZgemm(ctx.cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N, count,
                                             M, M, &one, d_E2m1_.data(), chunk_, d_A_.data(), M,
                                             &zero, d_T_.data(), chunk_));
        finish_kernel<<<(count + kBlock - 1) / kBlock, kBlock, 0, ctx.cuda_stream()>>>(
            py.data(), pz.data(), off, count, chunk_, M, d_kk_.data(), d_Qh_.data(), d_B_.data(),
            d_T_.data(), Q0_, psi1.data(), psi2.data());
        MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
    }
    MACROFLOW3D_CUBLAS_CHECK(cublasSetPointerMode(ctx.cublas_handle(), old_mode));
}

} // namespace inlet_slab
} // namespace streamfunctions
} // namespace macroflow3d
