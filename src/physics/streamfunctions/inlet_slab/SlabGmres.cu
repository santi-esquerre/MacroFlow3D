/**
 * @file SlabGmres.cu
 * @brief SF-33 N2: restarted right-preconditioned GMRES and deterministic vector kernels of the
 *        inlet slab (see SlabGmres.cuh for the algorithm, memory and synchronization contract).
 */

#include "SlabGmres.cuh"

#include "../../../runtime/cuda_check.cuh"

#include <chrono>
#include <cmath>
#include <stdexcept>
#include <string>

namespace macroflow3d {
namespace streamfunctions {
namespace inlet_slab {

namespace {

__global__ void axpy_kernel(std::size_t n, real alpha, const real* __restrict__ x,
                            real* __restrict__ y) {
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<std::size_t>(gridDim.x) * blockDim.x)
        y[i] += alpha * x[i];
}

__global__ void axpy_dev_kernel(std::size_t n, const real* __restrict__ alpha, real sign,
                                const real* __restrict__ x, real* __restrict__ y) {
    const real a = sign * alpha[0];
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<std::size_t>(gridDim.x) * blockDim.x)
        y[i] += a * x[i];
}

__global__ void scale_copy_kernel(std::size_t n, real alpha, const real* __restrict__ x,
                                  real* __restrict__ y) {
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<std::size_t>(gridDim.x) * blockDim.x)
        y[i] = alpha * x[i];
}

__global__ void axpby_kernel(std::size_t n, real a, const real* __restrict__ x, real b,
                             const real* __restrict__ y, real* __restrict__ out) {
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<std::size_t>(gridDim.x) * blockDim.x)
        out[i] = a * x[i] + b * y[i];
}

__global__ void fill_kernel(std::size_t n, real v, real* __restrict__ y) {
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<std::size_t>(gridDim.x) * blockDim.x)
        y[i] = v;
}

__global__ void dot_partials_kernel(std::size_t n, const real* __restrict__ a,
                                    const real* __restrict__ b, real* partials, int nblocks) {
    real acc[1] = {0.0};
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<std::size_t>(gridDim.x) * blockDim.x)
        acc[0] += a[i] * b[i];
    detail::block_reduce_store<1>(acc, partials, nblocks);
}

/// NaN-propagating max of |a_i|.
__device__ inline real nan_max(real a, real b) {
    if (a != a)
        return a;
    if (b != b)
        return b;
    return a > b ? a : b;
}

__global__ void maxabs_partials_kernel(std::size_t n, const real* __restrict__ a, real* partials) {
    __shared__ real sh[detail::kSlabBlock];
    real m = 0.0;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<std::size_t>(gridDim.x) * blockDim.x)
        m = nan_max(m, fabs(a[i]));
    const int t = threadIdx.x;
    sh[t] = m;
    __syncthreads();
    for (int s = detail::kSlabBlock / 2; s > 0; s >>= 1) {
        if (t < s)
            sh[t] = nan_max(sh[t], sh[t + s]);
        __syncthreads();
    }
    if (t == 0)
        partials[blockIdx.x] = sh[0];
}

__global__ void maxabs_finalize_kernel(const real* partials, int nblocks, real* out) {
    __shared__ real sh[detail::kSlabBlock];
    real m = 0.0;
    for (int i = threadIdx.x; i < nblocks; i += detail::kSlabBlock)
        m = nan_max(m, partials[i]);
    const int t = threadIdx.x;
    sh[t] = m;
    __syncthreads();
    for (int s = detail::kSlabBlock / 2; s > 0; s >>= 1) {
        if (t < s)
            sh[t] = nan_max(sh[t], sh[t + s]);
        __syncthreads();
    }
    if (t == 0)
        out[0] = sh[0];
}

int grid_for(std::size_t n) {
    return detail::slab_reduce_blocks(n);
}

void require_len(std::size_t a, std::size_t b, const char* who) {
    if (a != b)
        throw std::invalid_argument(std::string(who) + ": vector length mismatch (" +
                                    std::to_string(a) + " vs " + std::to_string(b) + ")");
}

bool finite(real v) {
    return std::isfinite(v);
}

} // namespace

// ------------------------------------------------------------------------------------------------
// vector kernels
// ------------------------------------------------------------------------------------------------

void slab_axpy(CudaContext& ctx, real alpha, DeviceSpan<const real> x, DeviceSpan<real> y) {
    require_len(x.size(), y.size(), "slab_axpy");
    axpy_kernel<<<grid_for(y.size()), detail::kSlabBlock, 0, ctx.cuda_stream()>>>(
        y.size(), alpha, x.data(), y.data());
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
}

void slab_axpy_dev(CudaContext& ctx, const real* alpha_dev, real sign, DeviceSpan<const real> x,
                   DeviceSpan<real> y) {
    require_len(x.size(), y.size(), "slab_axpy_dev");
    axpy_dev_kernel<<<grid_for(y.size()), detail::kSlabBlock, 0, ctx.cuda_stream()>>>(
        y.size(), alpha_dev, sign, x.data(), y.data());
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
}

void slab_scale_copy(CudaContext& ctx, real alpha, DeviceSpan<const real> x, DeviceSpan<real> y) {
    require_len(x.size(), y.size(), "slab_scale_copy");
    scale_copy_kernel<<<grid_for(y.size()), detail::kSlabBlock, 0, ctx.cuda_stream()>>>(
        y.size(), alpha, x.data(), y.data());
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
}

void slab_axpby(CudaContext& ctx, real a, DeviceSpan<const real> x, real b,
                DeviceSpan<const real> y, DeviceSpan<real> out) {
    require_len(x.size(), out.size(), "slab_axpby");
    require_len(y.size(), out.size(), "slab_axpby");
    axpby_kernel<<<grid_for(out.size()), detail::kSlabBlock, 0, ctx.cuda_stream()>>>(
        out.size(), a, x.data(), b, y.data(), out.data());
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
}

void slab_fill(CudaContext& ctx, real value, DeviceSpan<real> y) {
    fill_kernel<<<grid_for(y.size()), detail::kSlabBlock, 0, ctx.cuda_stream()>>>(y.size(), value,
                                                                                  y.data());
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
}

// ------------------------------------------------------------------------------------------------
// SlabReduction
// ------------------------------------------------------------------------------------------------

void SlabReduction::prepare(std::size_t n, int slots) {
    if (n == 0 || slots < 1)
        throw std::invalid_argument("SlabReduction::prepare: empty length or no slots");
    n_ = n;
    nblocks_ = detail::slab_reduce_blocks(n);
    nslots_ = slots;
    partials_.resize(static_cast<std::size_t>(nblocks_));
    slots_.resize(static_cast<std::size_t>(slots));
}

void SlabReduction::dot_device(CudaContext& ctx, DeviceSpan<const real> a, DeviceSpan<const real> b,
                               int slot) {
    require_len(a.size(), n_, "SlabReduction::dot_device");
    require_len(b.size(), n_, "SlabReduction::dot_device");
    if (slot < 0 || slot >= nslots_)
        throw std::out_of_range("SlabReduction::dot_device: slot out of range");
    const cudaStream_t s = ctx.cuda_stream();
    dot_partials_kernel<<<nblocks_, detail::kSlabBlock, 0, s>>>(n_, a.data(), b.data(),
                                                                partials_.data(), nblocks_);
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
    detail::finalize_partials_kernel<1>
        <<<1, detail::kSlabBlock, 0, s>>>(partials_.data(), nblocks_, slots_.data() + slot);
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
}

void SlabReduction::slots_to_host(CudaContext& ctx, int first, int count, real* host) {
    if (first < 0 || count < 0 || first + count > nslots_)
        throw std::out_of_range("SlabReduction::slots_to_host: range out of bounds");
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(host, slots_.data() + first, count * sizeof(real),
                                           cudaMemcpyDeviceToHost, ctx.cuda_stream()));
    MACROFLOW3D_CUDA_CHECK(cudaStreamSynchronize(ctx.cuda_stream()));
}

real SlabReduction::dot_host(CudaContext& ctx, DeviceSpan<const real> a, DeviceSpan<const real> b) {
    dot_device(ctx, a, b, 0);
    real v = 0.0;
    slots_to_host(ctx, 0, 1, &v);
    return v;
}

real SlabReduction::nrm2_host(CudaContext& ctx, DeviceSpan<const real> a) {
    const real d = dot_host(ctx, a, a);
    return std::sqrt(d);
}

real SlabReduction::maxabs_host(CudaContext& ctx, DeviceSpan<const real> a) {
    require_len(a.size(), n_, "SlabReduction::maxabs_host");
    const cudaStream_t s = ctx.cuda_stream();
    maxabs_partials_kernel<<<nblocks_, detail::kSlabBlock, 0, s>>>(n_, a.data(), partials_.data());
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
    maxabs_finalize_kernel<<<1, detail::kSlabBlock, 0, s>>>(partials_.data(), nblocks_,
                                                            slots_.data());
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
    real v = 0.0;
    slots_to_host(ctx, 0, 1, &v);
    return v;
}

std::size_t SlabReduction::allocated_bytes() const {
    return (partials_.capacity() + slots_.capacity()) * sizeof(real);
}

std::vector<const void*> SlabReduction::buffer_pointers() const {
    return {partials_.data(), slots_.data()};
}

// ------------------------------------------------------------------------------------------------
// SlabGmres
// ------------------------------------------------------------------------------------------------

void SlabGmres::prepare(const InletSlabGrid& grid, int restart, int max_iterations) {
    require_valid_grid(grid, "SlabGmres::prepare");
    if (restart < 1)
        throw std::invalid_argument("SlabGmres::prepare: restart must be >= 1");
    if (max_iterations < 1)
        throw std::invalid_argument("SlabGmres::prepare: max_iterations must be >= 1");
    n_ = grid.unknown_size();
    m_ = restart;
    V_.resize(static_cast<std::size_t>(m_ + 1) * n_);
    w_.resize(n_);
    z_.resize(n_);
    r_.resize(n_);
    red_.prepare(n_, m_ + 3);
    H_.assign(static_cast<std::size_t>(m_ + 1) * m_, 0.0);
    cs_.assign(m_, 0.0);
    sn_.assign(m_, 0.0);
    g_.assign(m_ + 1, 0.0);
    y_.assign(m_, 0.0);
    col_.assign(m_ + 3, 0.0);
    max_its_reserved_ = max_iterations;
}

std::size_t SlabGmres::allocated_bytes() const {
    return (V_.capacity() + w_.capacity() + z_.capacity() + r_.capacity()) * sizeof(real) +
           red_.allocated_bytes();
}

std::vector<const void*> SlabGmres::buffer_pointers() const {
    std::vector<const void*> p = {V_.data(), w_.data(), z_.data(), r_.data()};
    const auto q = red_.buffer_pointers();
    p.insert(p.end(), q.begin(), q.end());
    return p;
}

SlabGmresReport SlabGmres::solve(CudaContext& ctx, const Operator& A, const Operator& Minv,
                                 DeviceSpan<const real> b, DeviceSpan<real> x,
                                 const SlabGmresConfig& cfg) {
    const auto t0 = std::chrono::steady_clock::now();
    if (n_ == 0)
        throw std::logic_error("SlabGmres::solve: not prepared");
    require_len(b.size(), n_, "SlabGmres::solve (b)");
    require_len(x.size(), n_, "SlabGmres::solve (x)");
    if (cfg.restart < 1 || cfg.restart > m_)
        throw std::invalid_argument(
            "SlabGmres::solve: cfg.restart must be in [1, prepared restart]");
    if (cfg.max_iterations < 1)
        throw std::invalid_argument("SlabGmres::solve: cfg.max_iterations must be >= 1");
    const int m = cfg.restart;
    const int ld = m_ + 1; // leading dimension of H_
    auto Hm = [&](int i, int k) -> real& {
        return H_[static_cast<std::size_t>(i) + static_cast<std::size_t>(ld) * k];
    };

    SlabGmresReport rep;
    rep.cycle_true.clear();
    rep.cycle_recurrence.clear();
    const std::size_t reserve = static_cast<std::size_t>(
        cfg.max_iterations < max_its_reserved_ ? cfg.max_iterations : max_its_reserved_);
    rep.cycle_true.reserve(reserve);
    rep.cycle_recurrence.reserve(reserve);
    auto finish = [&](SlabLinearStatus st) {
        rep.status = st;
        rep.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        return rep;
    };

    slab_fill(ctx, 0.0, x);
    const real nb = red_.nrm2_host(ctx, b);
    rep.b_norm = nb;
    if (!finite(nb)) {
        rep.rel_residual = nb;
        return finish(SlabLinearStatus::nonfinite);
    }
    if (nb == 0.0) {
        rep.rel_residual = 0.0;
        rep.rel_recurrence = 0.0;
        return finish(SlabLinearStatus::converged);
    }
    const DeviceSpan<real> w(w_.data(), n_), z(z_.data(), n_), r(r_.data(), n_);
    // r = b (x = 0)
    slab_scale_copy(ctx, 1.0, b, r);
    real beta = nb;

    while (true) {
        slab_scale_copy(ctx, 1.0 / beta, r, vec(0));
        for (int i = 0; i <= m; ++i)
            g_[i] = 0.0;
        g_[0] = beta;
        int k_used = 0;
        bool singular = false;
        bool nonfinite = false;
        for (int k = 0; k < m; ++k) {
            Minv(vec(k), z);
            A(z, w);
            // modified Gram-Schmidt, coefficients in device slots 0..k
            red_.dot_device(ctx, w, w, m_ + 1);
            for (int i = 0; i <= k; ++i) {
                red_.dot_device(ctx, vec(i), w, i);
                slab_axpy_dev(ctx, red_.slot_ptr(i), -1.0, vec(i), w);
            }
            red_.dot_device(ctx, w, w, m_ + 2);
            red_.slots_to_host(ctx, 0, m_ + 3, col_.data()); // documented sync (Givens/MGS scalars)
            for (int i = 0; i <= k; ++i)
                Hm(i, k) = col_[i];
            const real nbefore = std::sqrt(col_[m_ + 1]);
            real nafter = std::sqrt(col_[m_ + 2]);
            if (finite(nbefore) && finite(nafter) && nafter < cfg.reorth_threshold * nbefore) {
                ++rep.reorthogonalizations;
                for (int i = 0; i <= k; ++i) {
                    red_.dot_device(ctx, vec(i), w, i);
                    slab_axpy_dev(ctx, red_.slot_ptr(i), -1.0, vec(i), w);
                }
                red_.dot_device(ctx, w, w, m_ + 2);
                red_.slots_to_host(ctx, 0, m_ + 3, col_.data()); // second documented sync
                for (int i = 0; i <= k; ++i)
                    Hm(i, k) += col_[i];
                nafter = std::sqrt(col_[m_ + 2]);
            }
            bool col_ok = finite(nbefore) && finite(nafter);
            for (int i = 0; i <= k && col_ok; ++i)
                col_ok = finite(Hm(i, k));
            if (!col_ok) {
                nonfinite = true;
                break;
            }
            Hm(k + 1, k) = nafter;
            const bool happy = (nafter == 0.0);
            if (!happy)
                slab_scale_copy(ctx, 1.0 / nafter, w, vec(k + 1));
            // previous rotations
            for (int i = 0; i < k; ++i) {
                const real t = cs_[i] * Hm(i, k) + sn_[i] * Hm(i + 1, k);
                Hm(i + 1, k) = -sn_[i] * Hm(i, k) + cs_[i] * Hm(i + 1, k);
                Hm(i, k) = t;
            }
            const real den = std::hypot(Hm(k, k), Hm(k + 1, k));
            ++rep.iterations;
            if (den == 0.0) {
                singular = true; // column k is zero after rotation: Hessenberg singular
                break;
            }
            cs_[k] = Hm(k, k) / den;
            sn_[k] = Hm(k + 1, k) / den;
            Hm(k, k) = den;
            Hm(k + 1, k) = 0.0;
            g_[k + 1] = -sn_[k] * g_[k];
            g_[k] = cs_[k] * g_[k];
            k_used = k + 1;
            if (std::fabs(g_[k + 1]) / nb < cfg.inner_tol_factor * cfg.tol || happy ||
                rep.iterations >= cfg.max_iterations)
                break;
        }
        if (nonfinite) {
            rep.rel_residual = std::nan("");
            rep.cycle_true.push_back(rep.rel_residual);
            rep.cycle_recurrence.push_back(rep.rel_residual);
            ++rep.cycles;
            return finish(SlabLinearStatus::nonfinite);
        }
        // y = R^-1 g (host back substitution), t = V y accumulated in w, x += M^-1 t
        for (int i = k_used - 1; i >= 0; --i) {
            real s = g_[i];
            for (int j = i + 1; j < k_used; ++j)
                s -= Hm(i, j) * y_[j];
            y_[i] = s / Hm(i, i);
        }
        if (k_used > 0) {
            slab_scale_copy(ctx, y_[0], vec(0), w);
            for (int i = 1; i < k_used; ++i)
                slab_axpy(ctx, y_[i], vec(i), w);
            Minv(w, z);
            slab_axpy(ctx, 1.0, z, x);
        }
        // TRUE residual r = b - A x
        A(x, w);
        slab_axpby(ctx, 1.0, b, -1.0, w, r);
        beta = red_.nrm2_host(ctx, r); // documented sync (true residual per cycle)
        ++rep.cycles;
        const real rel = beta / nb;
        const real rec = std::fabs(g_[k_used]) / nb;
        rep.rel_residual = rel;
        rep.rel_recurrence = rec;
        rep.cycle_true.push_back(rel);
        rep.cycle_recurrence.push_back(rec);
        if (!finite(rel))
            return finish(SlabLinearStatus::nonfinite);
        if (rel <= cfg.tol)
            return finish(SlabLinearStatus::converged);
        if (singular)
            return finish(SlabLinearStatus::breakdown);
        const std::size_t nc = rep.cycle_true.size();
        if (nc >= 3 && rep.cycle_true[nc - 1] > cfg.stagnation_factor * rep.cycle_true[nc - 3])
            return finish(SlabLinearStatus::stagnation);
        if (rep.iterations >= cfg.max_iterations)
            return finish(SlabLinearStatus::max_iterations);
    }
}

} // namespace inlet_slab
} // namespace streamfunctions
} // namespace macroflow3d
