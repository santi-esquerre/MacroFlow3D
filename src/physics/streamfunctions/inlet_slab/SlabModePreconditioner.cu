/**
 * @file SlabModePreconditioner.cu
 * @brief SF-33 N2: plane-averaged per-transverse-mode preconditioner P-A (see
 *        SlabModePreconditioner.cuh for the operator, the banded LU and the contracts).
 */

#include "SlabModePreconditioner.cuh"

#include "../../../runtime/cuda_check.cuh"
#include "SlabJacobianVectorProduct.cuh"
#include "SlabResidual.cuh"

#include <chrono>
#include <cmath>
#include <stdexcept>
#include <string>

namespace macroflow3d {
namespace streamfunctions {
namespace inlet_slab {

namespace {

inline void cufft_check(cufftResult r, const char* what) {
    if (r != CUFFT_SUCCESS) {
        throw std::runtime_error(std::string("SlabModePreconditioner: cuFFT error in ") + what +
                                 " (code " + std::to_string(static_cast<int>(r)) + ")");
    }
}

constexpr int kKl = SlabModePreconditioner::kKl;
constexpr int kKu = SlabModePreconditioner::kKu;
constexpr int kKv = kKl + kKu;
constexpr int kLdab = SlabModePreconditioner::kLdab;
constexpr int kNC = SlabModePreconditioner::kCoefPerPlane;
constexpr int kNR = SlabModePreconditioner::kCoefPerRow;

// --- complex helpers (double2 = cufftDoubleComplex) ----------------------------------------------
__device__ inline cufftDoubleComplex cmk(real re, real im) {
    cufftDoubleComplex z;
    z.x = re;
    z.y = im;
    return z;
}
__device__ inline cufftDoubleComplex cmul(cufftDoubleComplex a, cufftDoubleComplex b) {
    return cmk(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}
__device__ inline cufftDoubleComplex csub(cufftDoubleComplex a, cufftDoubleComplex b) {
    return cmk(a.x - b.x, a.y - b.y);
}
__device__ inline cufftDoubleComplex cdiv(cufftDoubleComplex a, cufftDoubleComplex b) {
    // Smith's algorithm (no overflow for moderate ratios)
    if (fabs(b.x) >= fabs(b.y)) {
        const real r = b.y / b.x;
        const real d = b.x + b.y * r;
        return cmk((a.x + a.y * r) / d, (a.y - a.x * r) / d);
    }
    const real r = b.x / b.y;
    const real d = b.x * r + b.y;
    return cmk((a.x * r + a.y) / d, (a.y * r - a.x) / d);
}
__device__ inline real cabs1(cufftDoubleComplex a) {
    return fabs(a.x) + fabs(a.y);
}

// --- plane-averaged coefficients ---------------------------------------------------------------
/// One block (kSlabBlock threads) per plane j = 1..N; deterministic fixed-order tree reduction.
__global__ void plane_coefficients_kernel(InletSlabGrid g, SlabStencilView st,
                                          const real* __restrict__ U1, const real* __restrict__ U2,
                                          const real* __restrict__ q, const real* __restrict__ gl0,
                                          const real* __restrict__ gl1,
                                          const real* __restrict__ gl2, real* __restrict__ coef) {
    const int j = 1 + static_cast<int>(blockIdx.x);
    const std::size_t np = g.plane_size();
    real acc[kNC];
    for (int k = 0; k < kNC; ++k)
        acc[k] = 0.0;
    for (std::size_t p = threadIdx.x; p < np; p += blockDim.x) {
        const int m2 = static_cast<int>(p / static_cast<std::size_t>(g.n));
        const int m3 = static_cast<int>(p % static_cast<std::size_t>(g.n));
        real du1[3], du2[3], g1[3], g2[3];
        if (j < g.n) {
            const std::size_t fi = g.full_index(j, m2, m3);
            real H1[6], H2[6];
            derivs4_at(U1, st, g, j, m2, m3, du1, H1);
            derivs4_at(U2, st, g, j, m2, m3, du2, H2);
            slab_label_gradients(du1, du2, g1, g2);
            const real glnk[3] = {gl0[fi], gl1[fi], gl2[fi]};
            SlabPointState s;
            slab_equation_point(g1, g2, H1, H2, glnk, q[fi], s);
#pragma unroll 1
            for (int k = 0; k < kNR; ++k) {
                real d[kNR];
                for (int t = 0; t < kNR; ++t)
                    d[t] = 0.0;
                d[k] = 1.0;
                real dF1, dF2;
                // q = 1: coefficients of the ROW-SCALED residual E / q
                slab_equation_point_jvp(g1, g2, H1, H2, glnk, 1.0, s, d, d + 3, d + 6, d + 12, dF1,
                                        dF2);
                acc[k] += dF1;
                acc[kNR + k] += dF2;
            }
        } else {
            grad4_at(U1, st, g, j, m2, m3, du1);
            grad4_at(U2, st, g, j, m2, m3, du2);
            slab_label_gradients(du1, du2, g1, g2);
#pragma unroll 1
            for (int k = 0; k < 6; ++k) {
                real d[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
                d[k] = 1.0;
                real dE1, dE2;
                slab_outlet_point_jvp(g1, g2, d, d + 3, 1.0, g.h, dE1, dE2);
                acc[k] += dE1;
                acc[kNR + k] += dE2;
            }
        }
    }
    __shared__ real sh[detail::kSlabBlock];
    const int t = threadIdx.x;
    for (int k = 0; k < kNC; ++k) {
        sh[t] = acc[k];
        __syncthreads();
        for (int s = detail::kSlabBlock / 2; s > 0; s >>= 1) {
            if (t < s)
                sh[t] += sh[t + s];
            __syncthreads();
        }
        if (t == 0)
            coef[static_cast<std::size_t>(j - 1) * kNC + k] = sh[0] / static_cast<real>(np);
        __syncthreads();
    }
}

// --- per-mode assembly + banded LU (one thread per mode) -----------------------------------------
struct BandView {
    cufftDoubleComplex* band;
    int n;  ///< 2N
    int nm; ///< number of modes
    int md;
    __device__ cufftDoubleComplex& at(int brow, int col) const {
        return band[(static_cast<std::size_t>(brow) * n + col) * nm + md];
    }
    /// element (i, c) of the matrix
    __device__ cufftDoubleComplex& el(int i, int c) const { return at(kKv + i - c, c); }
};

__device__ inline void band_add(const BandView& B, int i, int c, cufftDoubleComplex v,
                                int* violations) {
    const int off = i - c;
    if (off > kKl || -off > kKu) {
        atomicAdd(violations, 1);
        return;
    }
    cufftDoubleComplex& e = B.el(i, c);
    e.x += v.x;
    e.y += v.y;
}

__global__ void assemble_factor_kernel(int N, real h, real shift, int nmh, int nm,
                                       SlabStencilView st, const real* __restrict__ coef,
                                       cufftDoubleComplex* band, int* ipiv, int* counters) {
    const int md = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (md >= nm)
        return;
    const int n = 2 * N;
    BandView B{band, n, nm, md};
    for (int e = 0; e < kLdab * n; ++e)
        band[static_cast<std::size_t>(e) * nm + md] = cmk(0.0, 0.0);

    const int m2 = md / nmh;
    const int m3 = md % nmh;
    const real two_pi = 6.283185307179586476925286766559;
    const real T2 = two_pi * static_cast<real>(m2) / static_cast<real>(N);
    const real T3 = two_pi * static_cast<real>(m3) / static_cast<real>(N);
    const real a2 = (-2.0 * cos(2.0 * T2) + 32.0 * cos(T2) - 30.0) / (12.0 * h * h);
    const real a3 = (-2.0 * cos(2.0 * T3) + 32.0 * cos(T3) - 30.0) / (12.0 * h * h);
    const real s2 = (8.0 * sin(T2) - sin(2.0 * T2)) / (6.0 * h);
    const real s3 = (8.0 * sin(T3) - sin(2.0 * T3)) / (6.0 * h);

    for (int j = 1; j <= N; ++j) {
        for (int i = 0; i < 2; ++i) {
            const int row = 2 * (j - 1) + i;
            const real* C = coef + (static_cast<std::size_t>(j - 1) * 2 + i) * kNR;
            for (int f = 0; f < 2; ++f) {
                const real* Cg = C + 3 * f;
                const real* CH = C + 6 + 6 * f;
                // in-plane (same-plane) part
                band_add(B, row, 2 * (j - 1) + f,
                         cmk(CH[kH11] * a2 + CH[kH22] * a3 - CH[kH12] * s2 * s3,
                             Cg[1] * s2 + Cg[2] * s3),
                         counters + 1);
                // x1 first-derivative part: d1 and the mixed d12, d13 (= D1 applied to i s_j)
                const real c1r = Cg[0];
                const real c1i = CH[kH01] * s2 + CH[kH02] * s3;
                const X1Stencil& s1 = st.d1[j];
                for (int t = 0; t < s1.npts; ++t) {
                    const int p = s1.first + t;
                    if (p < 1)
                        continue; // inlet plane: the direction vanishes there
                    const real w = s1.w[t] / h;
                    band_add(B, row, 2 * (p - 1) + f, cmk(c1r * w, c1i * w), counters + 1);
                }
                if (j < N) {
                    const X1Stencil& s11 = st.d11[j];
                    for (int t = 0; t < s11.npts; ++t) {
                        const int p = s11.first + t;
                        if (p < 1)
                            continue;
                        band_add(B, row, 2 * (p - 1) + f, cmk(CH[kH00] * s11.w[t] / (h * h), 0.0),
                                 counters + 1);
                    }
                }
            }
        }
    }

    // SF-33 N7b pseudo-time shift of the ROW-SCALED operator: + mu / h^2 on the diagonal of the
    // equation rows (planes 1..N-1); outlet rows (plane N) unshifted. Skipped when mu = 0 (bitwise
    // N7a band).
    if (shift != 0.0) {
        for (int r = 0; r < 2 * (N - 1); ++r) {
            cufftDoubleComplex& e = B.el(r, r);
            e.x += shift;
        }
    }

    // LU with partial pivoting (zgbtf2): element (i, c) at band row kKv + i - c.
    int ju = 0;
    bool singular = false;
    for (int j = 0; j < n; ++j) {
        const int km = (kKl < n - 1 - j) ? kKl : n - 1 - j;
        int jp = 0;
        real best = cabs1(B.at(kKv, j));
        for (int t = 1; t <= km; ++t) {
            const real v = cabs1(B.at(kKv + t, j));
            if (v > best) {
                best = v;
                jp = t;
            }
        }
        ipiv[static_cast<std::size_t>(j) * nm + md] = j + jp;
        if (!(best > 0.0) || !isfinite(best)) {
            singular = true; // counted, never perturbed
            continue;
        }
        const int jmax = (j + kKu + jp < n - 1) ? j + kKu + jp : n - 1;
        if (jmax > ju)
            ju = jmax;
        if (jp != 0) {
            for (int c = j; c <= ju; ++c) {
                cufftDoubleComplex& a = B.at(kKv + j - c, c);
                cufftDoubleComplex& b = B.at(kKv + j + jp - c, c);
                const cufftDoubleComplex tmp = a;
                a = b;
                b = tmp;
            }
        }
        if (km > 0) {
            const cufftDoubleComplex piv = B.at(kKv, j);
            for (int t = 1; t <= km; ++t)
                B.at(kKv + t, j) = cdiv(B.at(kKv + t, j), piv);
            for (int c = j + 1; c <= ju; ++c) {
                const cufftDoubleComplex ajc = B.at(kKv + j - c, c);
                if (ajc.x == 0.0 && ajc.y == 0.0)
                    continue;
                for (int t = 1; t <= km; ++t) {
                    cufftDoubleComplex& e = B.at(kKv + j + t - c, c);
                    e = csub(e, cmul(B.at(kKv + t, j), ajc));
                }
            }
        }
    }
    if (singular)
        atomicAdd(counters, 1);
}

// --- apply kernels -------------------------------------------------------------------------------
__global__ void row_scale_kernel(InletSlabGrid g, const real* __restrict__ in,
                                 const real* __restrict__ q, real* __restrict__ out) {
    const std::size_t nf = g.field_size();
    const std::size_t np = g.plane_size();
    const std::size_t n = 2 * nf;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        const std::size_t u = i < nf ? i : i - nf;
        out[i] = in[i] / q[np + u];
    }
}

__global__ void mode_solve_kernel(int N, int nm, const cufftDoubleComplex* __restrict__ band,
                                  const int* __restrict__ ipiv, cufftDoubleComplex* spec) {
    const int md = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (md >= nm)
        return;
    const int n = 2 * N;
    auto A = [&](int brow, int col) -> cufftDoubleComplex {
        return band[(static_cast<std::size_t>(brow) * n + col) * nm + md];
    };
    auto addr = [&](int r) -> std::size_t {
        return (static_cast<std::size_t>(r & 1) * N + static_cast<std::size_t>(r >> 1)) * nm + md;
    };
    // forward: row interchanges and unit-lower L
    for (int j = 0; j < n - 1; ++j) {
        const int lm = (kKl < n - 1 - j) ? kKl : n - 1 - j;
        const int l = ipiv[static_cast<std::size_t>(j) * nm + md];
        if (l != j) {
            const cufftDoubleComplex t = spec[addr(l)];
            spec[addr(l)] = spec[addr(j)];
            spec[addr(j)] = t;
        }
        const cufftDoubleComplex bj = spec[addr(j)];
        for (int t = 1; t <= lm; ++t)
            spec[addr(j + t)] = csub(spec[addr(j + t)], cmul(A(kKv + t, j), bj));
    }
    // backward: upper U with bandwidth kKv
    for (int j = n - 1; j >= 0; --j) {
        const cufftDoubleComplex bj = cdiv(spec[addr(j)], A(kKv, j));
        spec[addr(j)] = bj;
        const int i0 = (j - kKv > 0) ? j - kKv : 0;
        for (int i = i0; i < j; ++i)
            spec[addr(i)] = csub(spec[addr(i)], cmul(A(kKv + i - j, j), bj));
    }
}

__global__ void scale_kernel(std::size_t n, real a, real* __restrict__ y) {
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<std::size_t>(gridDim.x) * blockDim.x)
        y[i] *= a;
}

void require_size(std::size_t got, std::size_t want, const char* what) {
    if (got != want) {
        throw std::invalid_argument(std::string("SlabModePreconditioner: ") + what + " has size " +
                                    std::to_string(got) + ", expected " + std::to_string(want));
    }
}

bool overlaps(const void* a, std::size_t na, const void* b, std::size_t nb) {
    const char* pa = static_cast<const char*>(a);
    const char* pb = static_cast<const char*>(b);
    return pa < pb + nb && pb < pa + na;
}

} // namespace

SlabModePreconditioner::~SlabModePreconditioner() {
    destroy_plans();
}

void SlabModePreconditioner::destroy_plans() {
    if (plans_) {
        cufftDestroy(plan_r2c_);
        cufftDestroy(plan_c2r_);
        plans_ = false;
    }
}

void SlabModePreconditioner::prepare(CudaContext& ctx, const InletSlabGrid& g) {
    require_valid_grid(g, "SlabModePreconditioner::prepare");
    if (!table_.built_for(g))
        table_.build(g);
    const int N = g.n;
    const int nm = N * (N / 2 + 1);
    const std::size_t n = 2 * static_cast<std::size_t>(N);
    coef_.resize(static_cast<std::size_t>(N) * kNC);
    band_.resize(static_cast<std::size_t>(kLdab) * n * nm);
    ipiv_.resize(n * nm);
    spec_.resize(n * nm);
    scratch_.resize(g.unknown_size());
    singular_.resize(2);
    if (!plans_ || n_ != N) {
        destroy_plans();
        int dims[2] = {N, N};
        std::size_t ws1 = 0, ws2 = 0;
        cufft_check(cufftCreate(&plan_r2c_), "cufftCreate");
        cufft_check(cufftCreate(&plan_c2r_), "cufftCreate");
        plans_ = true;
        cufft_check(cufftSetAutoAllocation(plan_r2c_, 0), "cufftSetAutoAllocation");
        cufft_check(cufftSetAutoAllocation(plan_c2r_, 0), "cufftSetAutoAllocation");
        cufft_check(cufftMakePlanMany(plan_r2c_, 2, dims, nullptr, 1, N * N, nullptr, 1,
                                      N * (N / 2 + 1), CUFFT_D2Z, 2 * N, &ws1),
                    "cufftMakePlanMany(D2Z)");
        cufft_check(cufftMakePlanMany(plan_c2r_, 2, dims, nullptr, 1, N * (N / 2 + 1), nullptr, 1,
                                      N * N, CUFFT_Z2D, 2 * N, &ws2),
                    "cufftMakePlanMany(Z2D)");
        fft_work_bytes_ = ws1 > ws2 ? ws1 : ws2;
        fft_work_.resize(fft_work_bytes_ > 0 ? fft_work_bytes_ : 1);
        cufft_check(cufftSetWorkArea(plan_r2c_, fft_work_.data()), "cufftSetWorkArea");
        cufft_check(cufftSetWorkArea(plan_c2r_, fft_work_.data()), "cufftSetWorkArea");
    }
    cufft_check(cufftSetStream(plan_r2c_, ctx.cuda_stream()), "cufftSetStream");
    cufft_check(cufftSetStream(plan_c2r_, ctx.cuda_stream()), "cufftSetStream");
    n_ = N;
    n_modes_ = nm;
    inputs_ = nullptr;
}

SlabPrecFactorReport SlabModePreconditioner::factor(CudaContext& ctx, const InletSlabGrid& grid,
                                                    const SlabStageInputs& inputs,
                                                    DeviceSpan<const real> U1,
                                                    DeviceSpan<const real> U2, real mu) {
    const auto t0 = std::chrono::steady_clock::now();
    if (!(mu >= 0.0) || !std::isfinite(mu))
        throw std::invalid_argument("SlabModePreconditioner::factor: mu must be finite and >= 0");
    require_valid_grid(grid, "SlabModePreconditioner::factor");
    if (!prepared_for(grid))
        throw std::logic_error("SlabModePreconditioner::factor: not prepared for this grid");
    inputs.check(grid);
    require_size(U1.size(), grid.full_size(), "U1");
    require_size(U2.size(), grid.full_size(), "U2");
    const cudaStream_t s = ctx.cuda_stream();
    MACROFLOW3D_CUDA_CHECK(cudaMemsetAsync(singular_.data(), 0, 2 * sizeof(int), s));
    plane_coefficients_kernel<<<grid.n, detail::kSlabBlock, 0, s>>>(
        grid, table_.device_view(), U1.data(), U2.data(), inputs.q.data(),
        inputs.grad_lnk[0].data(), inputs.grad_lnk[1].data(), inputs.grad_lnk[2].data(),
        coef_.data());
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
    const int threads = 64;
    const int blocks = (n_modes_ + threads - 1) / threads;
    const real shift = mu / (grid.h * grid.h);
    assemble_factor_kernel<<<blocks, threads, 0, s>>>(grid.n, grid.h, shift, grid.n / 2 + 1,
                                                      n_modes_, table_.device_view(), coef_.data(),
                                                      band_.data(), ipiv_.data(), singular_.data());
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
    int host[2] = {0, 0};
    MACROFLOW3D_CUDA_CHECK(
        cudaMemcpyAsync(host, singular_.data(), 2 * sizeof(int), cudaMemcpyDeviceToHost, s));
    // The single documented host synchronization of factor().
    MACROFLOW3D_CUDA_CHECK(cudaStreamSynchronize(s));
    if (host[1] != 0) {
        throw std::logic_error("SlabModePreconditioner::factor: stencil entry outside the band "
                               "(kl = ku = 9); implementation defect");
    }
    host_singular_ = host[0];
    inputs_ = &inputs;
    SlabPrecFactorReport rep;
    rep.singular_modes = host[0];
    rep.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return rep;
}

void SlabModePreconditioner::apply(CudaContext& ctx, const InletSlabGrid& grid,
                                   DeviceSpan<const real> in, DeviceSpan<real> out) {
    if (!prepared_for(grid))
        throw std::logic_error("SlabModePreconditioner::apply: not prepared for this grid");
    if (inputs_ == nullptr)
        throw std::logic_error("SlabModePreconditioner::apply: not factored (call factor first)");
    require_size(in.size(), grid.unknown_size(), "in");
    require_size(out.size(), grid.unknown_size(), "out");
    if (overlaps(in.data(), in.size() * sizeof(real), out.data(), out.size() * sizeof(real)))
        throw std::invalid_argument("SlabModePreconditioner::apply: in and out overlap");
    const cudaStream_t s = ctx.cuda_stream();
    const std::size_t n = grid.unknown_size();
    const int blocks = detail::slab_reduce_blocks(n);
    row_scale_kernel<<<blocks, detail::kSlabBlock, 0, s>>>(grid, in.data(), inputs_->q.data(),
                                                           scratch_.data());
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
    cufft_check(cufftExecD2Z(plan_r2c_, scratch_.data(), spec_.data()), "cufftExecD2Z");
    const int threads = 64;
    mode_solve_kernel<<<(n_modes_ + threads - 1) / threads, threads, 0, s>>>(
        grid.n, n_modes_, band_.data(), ipiv_.data(), spec_.data());
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
    cufft_check(cufftExecZ2D(plan_c2r_, spec_.data(), out.data()), "cufftExecZ2D");
    const real inv = 1.0 / (static_cast<real>(grid.n) * static_cast<real>(grid.n));
    scale_kernel<<<blocks, detail::kSlabBlock, 0, s>>>(n, inv, out.data());
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
}

std::vector<real> SlabModePreconditioner::download_coefficients(CudaContext& ctx) const {
    std::vector<real> h(static_cast<std::size_t>(n_) * kNC);
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(h.data(), coef_.data(), h.size() * sizeof(real),
                                           cudaMemcpyDeviceToHost, ctx.cuda_stream()));
    MACROFLOW3D_CUDA_CHECK(cudaStreamSynchronize(ctx.cuda_stream()));
    return h;
}

std::size_t SlabModePreconditioner::allocated_bytes() const {
    return table_.allocated_bytes() + coef_.capacity() * sizeof(real) +
           band_.capacity() * sizeof(cufftDoubleComplex) + ipiv_.capacity() * sizeof(int) +
           spec_.capacity() * sizeof(cufftDoubleComplex) + scratch_.capacity() * sizeof(real) +
           singular_.capacity() * sizeof(int) + fft_work_.capacity();
}

std::vector<const void*> SlabModePreconditioner::buffer_pointers() const {
    return {coef_.data(),    band_.data(),     ipiv_.data(),    spec_.data(),
            scratch_.data(), singular_.data(), fft_work_.data()};
}

} // namespace inlet_slab
} // namespace streamfunctions
} // namespace macroflow3d
