#include "cg_kernels.cuh"
#include "../../runtime/cuda_check.cuh"

namespace rwpt {
namespace blas {

__global__ void compute_alpha_kernel(const real* d_rr, const real* d_pAp, real* d_alpha) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *d_alpha = (*d_rr) / (*d_pAp);
    }
}

__global__ void compute_beta_kernel(const real* d_rr_new, const real* d_rr, real* d_beta) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *d_beta = (*d_rr_new) / (*d_rr);
    }
}

__global__ void update_x_and_r_kernel(const real* d_alpha, const real* p, real* x,
                                       const real* Ap, real* r, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    
    real alpha = *d_alpha;
    
    for (size_t i = idx; i < n; i += stride) {
        x[i] = x[i] + alpha * p[i];
        r[i] = r[i] - alpha * Ap[i];
    }
}

__global__ void update_p_kernel(const real* d_beta, const real* r, real* p, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    
    real beta = *d_beta;
    
    for (size_t i = idx; i < n; i += stride) {
        p[i] = r[i] + beta * p[i];
    }
}

void compute_alpha(CudaContext& ctx,
                   DeviceSpan<const real> d_rr,
                   DeviceSpan<const real> d_pAp,
                   DeviceSpan<real> d_alpha) {
    compute_alpha_kernel<<<1, 1, 0, ctx.cuda_stream()>>>(
        d_rr.data(), d_pAp.data(), d_alpha.data()
    );
    RWPT_CUDA_CHECK(cudaGetLastError());
}

void compute_beta(CudaContext& ctx,
                  DeviceSpan<const real> d_rr_new,
                  DeviceSpan<const real> d_rr,
                  DeviceSpan<real> d_beta) {
    compute_beta_kernel<<<1, 1, 0, ctx.cuda_stream()>>>(
        d_rr_new.data(), d_rr.data(), d_beta.data()
    );
    RWPT_CUDA_CHECK(cudaGetLastError());
}

void update_x_and_r(CudaContext& ctx,
                    DeviceSpan<const real> d_alpha,
                    DeviceSpan<const real> p,
                    DeviceSpan<real> x,
                    DeviceSpan<const real> Ap,
                    DeviceSpan<real> r) {
    size_t n = x.size();
    const int block_size = 256;
    const int max_blocks = 65535;
    const int grid_size = std::min((int)((n + block_size - 1) / block_size), max_blocks);
    
    update_x_and_r_kernel<<<grid_size, block_size, 0, ctx.cuda_stream()>>>(
        d_alpha.data(), p.data(), x.data(), Ap.data(), r.data(), n
    );
    RWPT_CUDA_CHECK(cudaGetLastError());
}

void update_p(CudaContext& ctx,
              DeviceSpan<const real> d_beta,
              DeviceSpan<const real> r,
              DeviceSpan<real> p) {
    size_t n = p.size();
    const int block_size = 256;
    const int max_blocks = 65535;
    const int grid_size = std::min((int)((n + block_size - 1) / block_size), max_blocks);
    
    update_p_kernel<<<grid_size, block_size, 0, ctx.cuda_stream()>>>(
        d_beta.data(), r.data(), p.data(), n
    );
    RWPT_CUDA_CHECK(cudaGetLastError());
}

__global__ void check_pAp_valid_kernel(const real* d_pAp, int* d_is_valid) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        real pAp = *d_pAp;
        // Check for zero or NaN
        // Note: For NEGATIVE definite operators, pAp < 0 is expected and valid!
        // CG works with negative-definite operators as long as pAp != 0
        const real eps = 1e-30;
        if (!isfinite(pAp) || fabs(pAp) < eps) {
            *d_is_valid = 0;  // Bad
        } else {
            *d_is_valid = 1;  // OK (positive OR negative)
        }
    }
}

void check_pAp_valid(CudaContext& ctx,
                     DeviceSpan<const real> d_pAp,
                     DeviceSpan<int> d_is_valid) {
    check_pAp_valid_kernel<<<1, 1, 0, ctx.cuda_stream()>>>(
        d_pAp.data(), d_is_valid.data()
    );
    RWPT_CUDA_CHECK(cudaGetLastError());
}

} // namespace blas
} // namespace rwpt
