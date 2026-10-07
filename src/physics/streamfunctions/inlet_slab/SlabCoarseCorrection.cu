/**
 * @file SlabCoarseCorrection.cu
 * @brief SF-33 N7c (probe): Galerkin coarse-space correction on the x1-constant / x1-linear
 *        column subspace (see SlabCoarseCorrection.cuh for the definition and the contracts).
 */

#include "SlabCoarseCorrection.cuh"

#include "../../../runtime/cuda_check.cuh"
#include "SlabGmres.cuh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

namespace macroflow3d {
namespace streamfunctions {
namespace inlet_slab {

namespace {

double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

/// Profile weight w_p(j), j = 1..N: p = 0 -> 1; p = 1 -> j / N (= j h, one rounding).
__host__ __device__ inline real profile_weight(int p, int j, int n) {
    return p == 0 ? 1.0 : static_cast<real>(j) / static_cast<real>(n);
}

constexpr int kThreads = 256;

inline int blocks_for(std::size_t n) {
    std::size_t b = (n + kThreads - 1) / kThreads;
    if (b < 1)
        b = 1;
    if (b > 65535)
        b = 65535;
    return static_cast<int>(b);
}

/// rc[(col * 2 + f) * P + p] = sum_{j=1..N} w_p(j) r[f nf + (j-1) np + col]; one thread per
/// (col, f), fixed summation order.
__global__ void restrict_kernel(int n, int P, const real* __restrict__ r, real* __restrict__ rc) {
    const std::size_t np = static_cast<std::size_t>(n) * n;
    const std::size_t nf = np * n;
    for (std::size_t t = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         t < 2 * np; t += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        const std::size_t col = t / 2;
        const int f = static_cast<int>(t % 2);
        real acc0 = 0.0, acc1 = 0.0;
        const real* rf = r + static_cast<std::size_t>(f) * nf + col;
        for (int j = 1; j <= n; ++j) {
            const real v = rf[static_cast<std::size_t>(j - 1) * np];
            acc0 += v;
            acc1 += profile_weight(1, j, n) * v;
        }
        rc[(col * 2 + f) * P + 0] = acc0;
        if (P > 1)
            rc[(col * 2 + f) * P + 1] = acc1;
    }
}

/// out[i] += sum_p w_p(j) y[(col * 2 + f) * P + p].
__global__ void prolong_add_kernel(int n, int P, const real* __restrict__ y,
                                   real* __restrict__ out) {
    const std::size_t np = static_cast<std::size_t>(n) * n;
    const std::size_t nf = np * n;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < 2 * nf; i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        const int f = i < nf ? 0 : 1;
        const std::size_t u = i - static_cast<std::size_t>(f) * nf;
        const int j = 1 + static_cast<int>(u / np);
        const std::size_t col = u % np;
        const real* yc = y + (col * 2 + f) * P;
        real v = yc[0];
        if (P > 1)
            v += profile_weight(1, j, n) * yc[1];
        out[i] += v;
    }
}

/// out (already zero) gets w_p(j) on planes j = 1..N of column col, field f.
__global__ void basis_kernel(int n, int col, int f, int p, real* __restrict__ out) {
    const int j = 1 + static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (j > n)
        return;
    const std::size_t np = static_cast<std::size_t>(n) * n;
    const std::size_t nf = np * n;
    out[static_cast<std::size_t>(f) * nf + static_cast<std::size_t>(j - 1) * np + col] =
        profile_weight(p, j, n);
}

/// Colored vector: w_q(j) on planes 1..N of every column with (m2 mod p, m3 mod p) = (a, b),
/// field f; zero elsewhere (both fields written).
__global__ void colored_vector_kernel(int n, int p, int a, int b, int f, int q,
                                      real* __restrict__ out) {
    const std::size_t np = static_cast<std::size_t>(n) * n;
    const std::size_t nf = np * n;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < 2 * nf; i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        const int fi = i < nf ? 0 : 1;
        const std::size_t u = i - static_cast<std::size_t>(fi) * nf;
        const int j = 1 + static_cast<int>(u / np);
        const std::size_t col = u % np;
        const int m2 = static_cast<int>(col / static_cast<std::size_t>(n));
        const int m3 = static_cast<int>(col % static_cast<std::size_t>(n));
        out[i] = (fi == f && m2 % p == a && m3 % p == b) ? profile_weight(q, j, n) : 0.0;
    }
}

/// For every column (m2, m3) of color (a, b): Esp[c * nb + o] = rc[c'] with c = the coarse index
/// of (column, f, q) and c' = the coarse index of (column + (d2, d3) periodic, f', q'),
/// o = ((d2 + R) (2R + 1) + (d3 + R)) 2P + f' P + q'.
__global__ void scatter_color_kernel(int n, int P, int p, int a, int b, int f, int q, int R,
                                     const real* __restrict__ rc, real* __restrict__ Esp) {
    const int per = n / p;
    const int ncols = per * per;
    const int w = 2 * R + 1;
    const int nb = w * w * 2 * P;
    for (int t = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x; t < ncols * nb;
         t += gridDim.x * blockDim.x) {
        const int k = t / nb;
        const int o = t % nb;
        const int m2 = a + p * (k / per);
        const int m3 = b + p * (k % per);
        const int c = ((m2 * n + m3) * 2 + f) * P + q;
        const int fq = o % (2 * P);
        const int d = o / (2 * P);
        const int d2 = d / w - R;
        const int d3 = d % w - R;
        int n2 = m2 + d2, n3 = m3 + d3;
        n2 = n2 < 0 ? n2 + n : (n2 >= n ? n2 - n : n2);
        n3 = n3 < 0 ? n3 + n : (n3 >= n ? n3 - n : n3);
        const int cp = (n2 * n + n3) * 2 * P + fq;
        Esp[static_cast<std::size_t>(c) * nb + o] = rc[cp];
    }
}

void require_size(std::size_t got, std::size_t want, const char* what) {
    if (got != want) {
        throw std::invalid_argument(std::string("SlabCoarseCorrection: ") + what + " has size " +
                                    std::to_string(got) + ", expected " + std::to_string(want));
    }
}

} // namespace

SlabCoarseCorrection::~SlabCoarseCorrection() {
    if (h_rc_ != nullptr)
        cudaFreeHost(h_rc_);
    if (h_yc_ != nullptr)
        cudaFreeHost(h_yc_);
}

int SlabCoarseCorrection::colored_period(int N) {
    const int pmin = 2 * kCoarseStencilRadius + 1;
    for (int p = pmin; p <= N; ++p)
        if (N % p == 0)
            return p;
    throw std::invalid_argument("SlabCoarseCorrection: no color period >= 5 divides N");
}

void SlabCoarseCorrection::prepare(CudaContext& ctx, const InletSlabGrid& g, int profiles,
                                   SlabCoarseAssembly assembly, SlabCoarseFactor factor) {
    require_valid_grid(g, "SlabCoarseCorrection::prepare");
    if (profiles != 1 && profiles != 2)
        throw std::invalid_argument("SlabCoarseCorrection::prepare: profiles must be 1 or 2");
    if (factor == SlabCoarseFactor::banded && assembly != SlabCoarseAssembly::colored)
        throw std::invalid_argument("SlabCoarseCorrection::prepare: the banded factorization "
                                    "requires the colored (sparse) assembly");
    n_ = g.n;
    P_ = profiles;
    K_ = 2 * P_ * static_cast<int>(g.plane_size());
    assembly_ = assembly;
    factor_ = factor;
    const std::size_t K = static_cast<std::size_t>(K_);
    const int R = kCoarseStencilRadius;
    p_ = 0;
    nb_ = 0;
    kl_ = ku_ = ldab_ = 0;
    if (assembly == SlabCoarseAssembly::direct) {
        Ecols_.resize(K * K);
    } else {
        p_ = colored_period(n_);
        nb_ = (2 * R + 1) * (2 * R + 1) * 2 * P_;
        Esp_.resize(K * static_cast<std::size_t>(nb_));
        Esp_h_.assign(K * static_cast<std::size_t>(nb_), 0.0);
    }
    if (factor == SlabCoarseFactor::dense) {
        E_.assign(K * K, 0.0);
        LU_.assign(K * K, 0.0);
    } else {
        // folded ordering of m2: 0, N-1, 1, N-2, ... ; m3, (f, q) inner
        perm_.assign(K, 0);
        const int N = n_;
        const int bw = 2 * P_;
        for (int m2 = 0; m2 < N; ++m2) {
            const int pos = m2 < N / 2 ? 2 * m2 : 2 * (N - 1 - m2) + 1;
            for (int m3 = 0; m3 < N; ++m3)
                for (int fq = 0; fq < bw; ++fq)
                    perm_[static_cast<std::size_t>((m2 * N + m3) * bw + fq)] =
                        (pos * N + m3) * bw + fq;
        }
        // half bandwidths from the sparsity pattern (radius R, periodic)
        int kl = 0, ku = 0;
        for (int m2 = 0; m2 < N; ++m2)
            for (int m3 = 0; m3 < N; ++m3)
                for (int d2 = -R; d2 <= R; ++d2)
                    for (int d3 = -R; d3 <= R; ++d3) {
                        const int n2 = (m2 + d2 + N) % N, n3 = (m3 + d3 + N) % N;
                        for (int a = 0; a < bw; ++a)
                            for (int b = 0; b < bw; ++b) {
                                const int col =
                                    perm_[static_cast<std::size_t>((m2 * N + m3) * bw + a)];
                                const int row =
                                    perm_[static_cast<std::size_t>((n2 * N + n3) * bw + b)];
                                kl = std::max(kl, row - col);
                                ku = std::max(ku, col - row);
                            }
                    }
        kl_ = kl;
        ku_ = ku;
        ldab_ = 2 * kl_ + ku_ + 1;
        band_.assign(static_cast<std::size_t>(ldab_) * K, 0.0);
        perm_work_.assign(K, 0.0);
    }
    basis_.resize(g.unknown_size());
    Abasis_.resize(g.unknown_size());
    t1_.resize(g.unknown_size());
    t2_.resize(g.unknown_size());
    rc_.resize(K);
    yc_.resize(K);
    piv_.assign(K, 0);
    work_.assign(2 * K, 0.0);
    if (pinned_K_ < K) {
        if (h_rc_ != nullptr)
            MACROFLOW3D_CUDA_CHECK(cudaFreeHost(h_rc_));
        if (h_yc_ != nullptr)
            MACROFLOW3D_CUDA_CHECK(cudaFreeHost(h_yc_));
        h_rc_ = h_yc_ = nullptr;
        MACROFLOW3D_CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&h_rc_), K * sizeof(real)));
        MACROFLOW3D_CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&h_yc_), K * sizeof(real)));
        pinned_K_ = K;
    }
    built_ = false;
    reset_apply_stats();
    ctx.synchronize();
}

std::size_t SlabCoarseCorrection::allocated_bytes() const {
    return (Ecols_.capacity() + Esp_.capacity() + basis_.capacity() + Abasis_.capacity() +
            t1_.capacity() + t2_.capacity() + rc_.capacity() + yc_.capacity()) *
           sizeof(real);
}

std::vector<const void*> SlabCoarseCorrection::buffer_pointers() const {
    return {Ecols_.data(), Esp_.data(), basis_.data(), Abasis_.data(),
            t1_.data(),    t2_.data(),  rc_.data(),    yc_.data()};
}

void SlabCoarseCorrection::restrict_to_coarse(CudaContext& ctx, const InletSlabGrid& g,
                                              DeviceSpan<const real> r, DeviceSpan<real> rc) const {
    require_size(r.size(), g.unknown_size(), "restrict input");
    require_size(rc.size(), static_cast<std::size_t>(K_), "restrict output");
    restrict_kernel<<<blocks_for(2 * g.plane_size()), kThreads, 0, ctx.cuda_stream()>>>(
        g.n, P_, r.data(), rc.data());
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
}

void SlabCoarseCorrection::prolong_add(CudaContext& ctx, const InletSlabGrid& g,
                                       DeviceSpan<const real> y, DeviceSpan<real> out) const {
    require_size(y.size(), static_cast<std::size_t>(K_), "prolong input");
    require_size(out.size(), g.unknown_size(), "prolong output");
    prolong_add_kernel<<<blocks_for(g.unknown_size()), kThreads, 0, ctx.cuda_stream()>>>(
        g.n, P_, y.data(), out.data());
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
}

void SlabCoarseCorrection::basis_vector(CudaContext& ctx, const InletSlabGrid& g, int c,
                                        DeviceSpan<real> out) const {
    require_size(out.size(), g.unknown_size(), "basis output");
    if (c < 0 || c >= K_)
        throw std::invalid_argument("SlabCoarseCorrection::basis_vector: index out of range");
    const int col = c / (2 * P_);
    const int f = (c / P_) % 2;
    const int p = c % P_;
    slab_fill(ctx, 0.0, out);
    basis_kernel<<<(g.n + 127) / 128, 128, 0, ctx.cuda_stream()>>>(g.n, col, f, p, out.data());
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
}

std::vector<real> SlabCoarseCorrection::galerkin_dense_copy() const {
    if (factor_ == SlabCoarseFactor::dense)
        return E_;
    return galerkin_dense_copy_from_sparse();
}

SlabCoarseBuildReport SlabCoarseCorrection::build(CudaContext& ctx, const InletSlabGrid& g,
                                                  const Operator& A) {
    if (!prepared_for(g))
        throw std::logic_error("SlabCoarseCorrection::build: not prepared for this grid");
    SlabCoarseBuildReport rep;
    rep.K = K_;
    rep.profiles = P_;
    rep.assembly = assembly_;
    rep.factor = factor_;
    rep.color_period = p_;
    const std::size_t K = static_cast<std::size_t>(K_);
    const auto t0 = std::chrono::steady_clock::now();
    const DeviceSpan<real> vb(basis_.data(), basis_.size());
    const DeviceSpan<real> av(Abasis_.data(), Abasis_.size());
    const DeviceSpan<const real> cvb(vb.data(), vb.size()), cav(av.data(), av.size());
    if (assembly_ == SlabCoarseAssembly::direct) {
        for (int c = 0; c < K_; ++c) {
            basis_vector(ctx, g, c, vb);
            A(cvb, av);
            restrict_to_coarse(
                ctx, g, cav, DeviceSpan<real>(Ecols_.data() + static_cast<std::size_t>(c) * K, K));
        }
        rep.applications = K_;
        // column-major device buffer -> row-major host E (LU_ used as the staging copy)
        MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(LU_.data(), Ecols_.data(), K * K * sizeof(real),
                                               cudaMemcpyDeviceToHost, ctx.cuda_stream()));
        MACROFLOW3D_CUDA_CHECK(cudaStreamSynchronize(ctx.cuda_stream())); // documented
        for (std::size_t c = 0; c < K; ++c)
            for (std::size_t r = 0; r < K; ++r)
                E_[r * K + c] = LU_[c * K + r];
    } else {
        const int R = kCoarseStencilRadius;
        const int per = n_ / p_;
        const int nthreads = per * per * nb_;
        int apps = 0;
        for (int a = 0; a < p_; ++a)
            for (int b = 0; b < p_; ++b)
                for (int f = 0; f < 2; ++f)
                    for (int q = 0; q < P_; ++q) {
                        colored_vector_kernel<<<blocks_for(g.unknown_size()), kThreads, 0,
                                                ctx.cuda_stream()>>>(n_, p_, a, b, f, q, vb.data());
                        MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
                        A(cvb, av);
                        ++apps;
                        restrict_to_coarse(ctx, g, cav, DeviceSpan<real>(rc_.data(), K));
                        scatter_color_kernel<<<blocks_for(static_cast<std::size_t>(nthreads)),
                                               kThreads, 0, ctx.cuda_stream()>>>(
                            n_, P_, p_, a, b, f, q, R, rc_.data(), Esp_.data());
                        MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
                    }
        rep.applications = apps;
        MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(Esp_h_.data(), Esp_.data(),
                                               Esp_h_.size() * sizeof(real), cudaMemcpyDeviceToHost,
                                               ctx.cuda_stream()));
        MACROFLOW3D_CUDA_CHECK(cudaStreamSynchronize(ctx.cuda_stream())); // documented
        if (factor_ == SlabCoarseFactor::dense)
            E_ = galerkin_dense_copy_from_sparse();
    }
    rep.t_assembly = seconds_since(t0);

    // ||E||_1 (column sums)
    real n1 = 0.0;
    if (factor_ == SlabCoarseFactor::dense) {
        for (std::size_t c = 0; c < K; ++c) {
            real s = 0.0;
            for (std::size_t r = 0; r < K; ++r)
                s += std::fabs(E_[r * K + c]);
            n1 = std::fmax(n1, s);
        }
    } else {
        for (std::size_t c = 0; c < K; ++c) {
            real s = 0.0;
            for (int o = 0; o < nb_; ++o)
                s += std::fabs(Esp_h_[c * nb_ + o]);
            n1 = std::fmax(n1, s);
        }
    }
    rep.norm1 = n1;

    const auto t1 = std::chrono::steady_clock::now();
    real umin = INFINITY, umax = 0.0;
    if (factor_ == SlabCoarseFactor::dense) {
        LU_ = E_;
        rep.zero_pivots = lu_factor(K_, LU_.data(), piv_.data());
        for (std::size_t k = 0; k < K; ++k) {
            const real u = std::fabs(LU_[k * K + k]);
            umin = std::fmin(umin, u);
            umax = std::fmax(umax, u);
        }
    } else {
        std::fill(band_.begin(), band_.end(), 0.0);
        const int R = kCoarseStencilRadius, w = 2 * R + 1, bw = 2 * P_, N = n_;
        const int kv = kl_ + ku_;
        for (std::size_t c = 0; c < K; ++c) {
            const int col = static_cast<int>(c) / bw;
            const int m2 = col / N, m3 = col % N;
            const int jj = perm_[c];
            for (int o = 0; o < nb_; ++o) {
                const int fq = o % bw, d = o / bw;
                const int n2 = (m2 + d / w - R + N) % N, n3 = (m3 + d % w - R + N) % N;
                const int ii = perm_[static_cast<std::size_t>((n2 * N + n3) * bw + fq)];
                band_[static_cast<std::size_t>(jj) * ldab_ + kv + ii - jj] = Esp_h_[c * nb_ + o];
            }
        }
        rep.zero_pivots = band_lu_factor(K_, kl_, ku_, band_.data(), piv_.data());
        for (std::size_t j = 0; j < K; ++j) {
            const real u = std::fabs(band_[j * ldab_ + kv]);
            umin = std::fmin(umin, u);
            umax = std::fmax(umax, u);
        }
        rep.kl = kl_;
        rep.ku = ku_;
    }
    rep.t_lu = seconds_since(t1);
    rep.min_abs_u = umin;
    rep.max_abs_u = umax;

    const auto t2 = std::chrono::steady_clock::now();
    if (rep.zero_pivots == 0) {
        rep.inv_norm1_est = inv_norm1_estimate_fn(
            K_, [this](real* b) { coarse_solve_host(b); },
            [this](real* b) { coarse_solve_transpose_host(b); }, work_.data());
        rep.rcond_est = 1.0 / (n1 * rep.inv_norm1_est);
    } else {
        rep.inv_norm1_est = INFINITY;
        rep.rcond_est = 0.0;
    }
    rep.t_cond = seconds_since(t2);
    built_ = rep.zero_pivots == 0;
    reset_apply_stats();
    return rep;
}

std::vector<real> SlabCoarseCorrection::galerkin_dense_copy_from_sparse() const {
    const std::size_t K = static_cast<std::size_t>(K_);
    std::vector<real> E(K * K, 0.0);
    const int R = kCoarseStencilRadius, w = 2 * R + 1, bw = 2 * P_, N = n_;
    for (std::size_t c = 0; c < K; ++c) {
        const int col = static_cast<int>(c) / bw;
        const int m2 = col / N, m3 = col % N;
        for (int o = 0; o < nb_; ++o) {
            const int fq = o % bw, d = o / bw;
            const int n2 = (m2 + d / w - R + N) % N, n3 = (m3 + d % w - R + N) % N;
            const std::size_t cp = static_cast<std::size_t>((n2 * N + n3) * bw + fq);
            E[cp * K + c] = Esp_h_[c * nb_ + o];
        }
    }
    return E;
}

void SlabCoarseCorrection::coarse_solve_host(real* b) {
    if (factor_ == SlabCoarseFactor::dense) {
        lu_solve(K_, LU_.data(), piv_.data(), b);
        return;
    }
    const std::size_t K = static_cast<std::size_t>(K_);
    for (std::size_t c = 0; c < K; ++c)
        perm_work_[static_cast<std::size_t>(perm_[c])] = b[c];
    band_lu_solve(K_, kl_, ku_, band_.data(), piv_.data(), perm_work_.data());
    for (std::size_t c = 0; c < K; ++c)
        b[c] = perm_work_[static_cast<std::size_t>(perm_[c])];
}

void SlabCoarseCorrection::coarse_solve_transpose_host(real* b) {
    if (factor_ == SlabCoarseFactor::dense) {
        lu_solve_transpose(K_, LU_.data(), piv_.data(), b);
        return;
    }
    const std::size_t K = static_cast<std::size_t>(K_);
    for (std::size_t c = 0; c < K; ++c)
        perm_work_[static_cast<std::size_t>(perm_[c])] = b[c];
    band_lu_solve_transpose(K_, kl_, ku_, band_.data(), piv_.data(), perm_work_.data());
    for (std::size_t c = 0; c < K; ++c)
        b[c] = perm_work_[static_cast<std::size_t>(perm_[c])];
}

void SlabCoarseCorrection::add_correction(CudaContext& ctx, const InletSlabGrid& g,
                                          DeviceSpan<const real> r, DeviceSpan<real> out) {
    if (!built_)
        throw std::logic_error("SlabCoarseCorrection::add_correction: not built (or singular E)");
    const std::size_t K = static_cast<std::size_t>(K_);
    restrict_to_coarse(ctx, g, r, DeviceSpan<real>(rc_.data(), K));
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(h_rc_, rc_.data(), K * sizeof(real),
                                           cudaMemcpyDeviceToHost, ctx.cuda_stream()));
    MACROFLOW3D_CUDA_CHECK(cudaStreamSynchronize(ctx.cuda_stream())); // documented round trip
    const auto t0 = std::chrono::steady_clock::now();
    std::memcpy(h_yc_, h_rc_, K * sizeof(real));
    coarse_solve_host(h_yc_);
    t_host_solve_ += seconds_since(t0);
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(yc_.data(), h_yc_, K * sizeof(real),
                                           cudaMemcpyHostToDevice, ctx.cuda_stream()));
    prolong_add(ctx, g, DeviceSpan<const real>(yc_.data(), K), out);
}

void SlabCoarseCorrection::apply(CudaContext& ctx, const InletSlabGrid& g, SlabCoarseMode mode,
                                 const Operator& A, const Operator& PA, DeviceSpan<const real> in,
                                 DeviceSpan<real> out) {
    if (mode == SlabCoarseMode::off) {
        PA(in, out);
        return;
    }
    const auto t0 = std::chrono::steady_clock::now();
    PA(in, out);
    if (mode == SlabCoarseMode::add) {
        add_correction(ctx, g, in, out);
    } else {
        const DeviceSpan<real> t1(t1_.data(), t1_.size()), t2(t2_.data(), t2_.size());
        A(DeviceSpan<const real>(out.data(), out.size()), t1);
        slab_axpby(ctx, 1.0, in, -1.0, DeviceSpan<const real>(t1.data(), t1.size()), t2);
        add_correction(ctx, g, DeviceSpan<const real>(t2.data(), t2.size()), out);
    }
    ++n_apply_;
    t_apply_ += seconds_since(t0);
}

// ------------------------------------------------------------------------------------------------
// Host dense LU
// ------------------------------------------------------------------------------------------------

int SlabCoarseCorrection::lu_factor(int n, real* A, int* piv) {
    int zero = 0;
    const std::size_t N = static_cast<std::size_t>(n);
    for (std::size_t k = 0; k < N; ++k) {
        std::size_t p = k;
        real best = std::fabs(A[k * N + k]);
        bool nonfinite = !std::isfinite(A[k * N + k]);
        for (std::size_t i = k + 1; i < N; ++i) {
            const real v = std::fabs(A[i * N + k]);
            if (!std::isfinite(v))
                nonfinite = true;
            if (v > best) {
                best = v;
                p = i;
            }
        }
        piv[k] = static_cast<int>(p);
        if (p != k) {
            real* rk = A + k * N;
            real* rp = A + p * N;
            for (std::size_t j = 0; j < N; ++j) {
                const real t = rk[j];
                rk[j] = rp[j];
                rp[j] = t;
            }
        }
        const real akk = A[k * N + k];
        if (akk == 0.0 || nonfinite || !std::isfinite(akk)) {
            ++zero; // counted, never perturbed; the column is skipped
            continue;
        }
        const real* rk = A + k * N;
        for (std::size_t i = k + 1; i < N; ++i) {
            real* ri = A + i * N;
            const real l = ri[k] / akk;
            ri[k] = l;
            if (l == 0.0)
                continue;
            for (std::size_t j = k + 1; j < N; ++j)
                ri[j] -= l * rk[j];
        }
    }
    return zero;
}

void SlabCoarseCorrection::lu_solve(int n, const real* LU, const int* piv, real* b) {
    const std::size_t N = static_cast<std::size_t>(n);
    for (std::size_t k = 0; k < N; ++k) {
        const std::size_t p = static_cast<std::size_t>(piv[k]);
        if (p != k) {
            const real t = b[k];
            b[k] = b[p];
            b[p] = t;
        }
    }
    for (std::size_t i = 1; i < N; ++i) { // L y = P b (unit lower)
        const real* ri = LU + i * N;
        real s = b[i];
        for (std::size_t j = 0; j < i; ++j)
            s -= ri[j] * b[j];
        b[i] = s;
    }
    for (std::size_t ii = N; ii-- > 0;) { // U x = y
        const real* ri = LU + ii * N;
        real s = b[ii];
        for (std::size_t j = ii + 1; j < N; ++j)
            s -= ri[j] * b[j];
        b[ii] = s / ri[ii];
    }
}

void SlabCoarseCorrection::lu_solve_transpose(int n, const real* LU, const int* piv, real* b) {
    const std::size_t N = static_cast<std::size_t>(n);
    // U^T w = b (forward; column access of the row-major U, axpy form)
    for (std::size_t i = 0; i < N; ++i) {
        b[i] /= LU[i * N + i];
        const real bi = b[i];
        const real* ri = LU + i * N;
        for (std::size_t j = i + 1; j < N; ++j)
            b[j] -= ri[j] * bi;
    }
    // L^T y = w (backward, unit diagonal)
    for (std::size_t ii = N; ii-- > 0;) {
        const real yi = b[ii];
        const real* ri = LU + ii * N;
        for (std::size_t j = 0; j < ii; ++j)
            b[j] -= ri[j] * yi;
    }
    // x = P^T y
    for (std::size_t kk = N; kk-- > 0;) {
        const std::size_t p = static_cast<std::size_t>(piv[kk]);
        if (p != kk) {
            const real t = b[kk];
            b[kk] = b[p];
            b[p] = t;
        }
    }
}

// ------------------------------------------------------------------------------------------------
// Host banded LU (LAPACK dgbtf2 / dgbtrs, unblocked, partial pivoting)
// ------------------------------------------------------------------------------------------------

int SlabCoarseCorrection::band_lu_factor(int n, int kl, int ku, real* AB, int* piv) {
    const int kv = kl + ku;
    const std::size_t ld = static_cast<std::size_t>(2 * kl + ku + 1);
    auto at = [&](int r, int c) -> real& { return AB[static_cast<std::size_t>(c) * ld + r]; };
    int zero = 0;
    int ju = 0;
    for (int j = 0; j < n; ++j) {
        const int km = std::min(kl, n - 1 - j);
        int jp = 0;
        real best = std::fabs(at(kv, j));
        bool nonfinite = !std::isfinite(at(kv, j));
        for (int i = 1; i <= km; ++i) {
            const real v = std::fabs(at(kv + i, j));
            if (!std::isfinite(v))
                nonfinite = true;
            if (v > best) {
                best = v;
                jp = i;
            }
        }
        piv[j] = j + jp;
        if (at(kv + jp, j) == 0.0 || nonfinite || !std::isfinite(at(kv + jp, j))) {
            ++zero; // counted, never perturbed
            continue;
        }
        ju = std::max(ju, std::min(j + ku + jp, n - 1));
        if (jp != 0) {
            for (int c = j; c <= ju; ++c) {
                real& x = at(kv + j + jp - c, c);
                real& y = at(kv + j - c, c);
                const real t = x;
                x = y;
                y = t;
            }
        }
        if (km > 0) {
            const real inv = 1.0 / at(kv, j);
            for (int i = 1; i <= km; ++i)
                at(kv + i, j) *= inv;
            for (int c = j + 1; c <= ju; ++c) {
                const real t = at(kv + j - c, c);
                if (t == 0.0)
                    continue;
                real* colc = AB + static_cast<std::size_t>(c) * ld + (kv + j - c);
                const real* colj = AB + static_cast<std::size_t>(j) * ld + kv;
                for (int i = 1; i <= km; ++i)
                    colc[i] -= colj[i] * t;
            }
        }
    }
    return zero;
}

void SlabCoarseCorrection::band_lu_solve(int n, int kl, int ku, const real* AB, const int* piv,
                                         real* b) {
    const int kv = kl + ku;
    const std::size_t ld = static_cast<std::size_t>(2 * kl + ku + 1);
    for (int j = 0; j < n - 1; ++j) {
        const int km = std::min(kl, n - 1 - j);
        const int l = piv[j];
        if (l != j) {
            const real t = b[l];
            b[l] = b[j];
            b[j] = t;
        }
        const real bj = b[j];
        const real* colj = AB + static_cast<std::size_t>(j) * ld + kv;
        for (int i = 1; i <= km; ++i)
            b[j + i] -= colj[i] * bj;
    }
    for (int j = n - 1; j >= 0; --j) {
        const real* colj = AB + static_cast<std::size_t>(j) * ld;
        b[j] /= colj[kv];
        const real bj = b[j];
        const int i0 = std::max(0, j - kv);
        for (int i = i0; i < j; ++i)
            b[i] -= colj[kv + i - j] * bj;
    }
}

void SlabCoarseCorrection::band_lu_solve_transpose(int n, int kl, int ku, const real* AB,
                                                   const int* piv, real* b) {
    const int kv = kl + ku;
    const std::size_t ld = static_cast<std::size_t>(2 * kl + ku + 1);
    for (int j = 0; j < n; ++j) { // U^T y = b
        const real* colj = AB + static_cast<std::size_t>(j) * ld;
        real s = b[j];
        const int i0 = std::max(0, j - kv);
        for (int i = i0; i < j; ++i)
            s -= colj[kv + i - j] * b[i];
        b[j] = s / colj[kv];
    }
    for (int j = n - 2; j >= 0; --j) { // L^T with the interchanges in reverse order
        const int km = std::min(kl, n - 1 - j);
        const real* colj = AB + static_cast<std::size_t>(j) * ld + kv;
        real s = b[j];
        for (int i = 1; i <= km; ++i)
            s -= colj[i] * b[j + i];
        b[j] = s;
        const int l = piv[j];
        if (l != j) {
            const real t = b[l];
            b[l] = b[j];
            b[j] = t;
        }
    }
}

real SlabCoarseCorrection::inv_norm1_estimate_fn(int n, const std::function<void(real*)>& solve,
                                                 const std::function<void(real*)>& solve_t,
                                                 real* work) {
    const std::size_t N = static_cast<std::size_t>(n);
    real* x = work;
    real* z = work + N;
    for (std::size_t i = 0; i < N; ++i)
        x[i] = 1.0 / static_cast<real>(n);
    real est = 0.0;
    std::size_t jlast = N;
    for (int iter = 0; iter < 5; ++iter) {
        solve(x); // x <- A^-1 x
        real s = 0.0;
        for (std::size_t i = 0; i < N; ++i)
            s += std::fabs(x[i]);
        est = std::fmax(est, s);
        for (std::size_t i = 0; i < N; ++i)
            z[i] = x[i] >= 0.0 ? 1.0 : -1.0;
        solve_t(z); // z <- A^-T sign(y)
        std::size_t jmax = 0;
        real zmax = -1.0;
        for (std::size_t i = 0; i < N; ++i) {
            if (std::fabs(z[i]) > zmax) {
                zmax = std::fabs(z[i]);
                jmax = i;
            }
        }
        if (iter > 0 && (jmax == jlast))
            break;
        jlast = jmax;
        for (std::size_t i = 0; i < N; ++i)
            x[i] = 0.0;
        x[jmax] = 1.0;
    }
    // Higham's alternating-sign test vector
    for (std::size_t i = 0; i < N; ++i)
        x[i] = ((i % 2) ? -1.0 : 1.0) *
               (1.0 + static_cast<real>(i) / static_cast<real>(n > 1 ? n - 1 : 1));
    solve(x);
    real s = 0.0;
    for (std::size_t i = 0; i < N; ++i)
        s += std::fabs(x[i]);
    return std::fmax(est, 2.0 * s / (3.0 * static_cast<real>(n)));
}

real SlabCoarseCorrection::inv_norm1_estimate(int n, const real* LU, const int* piv, real* work) {
    return inv_norm1_estimate_fn(
        n, [&](real* b) { lu_solve(n, LU, piv, b); },
        [&](real* b) { lu_solve_transpose(n, LU, piv, b); }, work);
}

} // namespace inlet_slab
} // namespace streamfunctions
} // namespace macroflow3d
