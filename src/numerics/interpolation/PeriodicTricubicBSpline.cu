/**
 * @file PeriodicTricubicBSpline.cu
 * @brief Periodic tricubic B-spline interpolation (SF-28) - implementation.
 *
 * See PeriodicTricubicBSpline.cuh for the full specification (conventions,
 * prefilter symbol, reduction-first evaluation, weights, memory accounting,
 * plan lifetime, size rules).
 */

#include "../../runtime/cuda_check.cuh"
#include "PeriodicTricubicBSpline.cuh"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace macroflow3d {
namespace interpolation {

namespace {

// ============================================================================
// cuFFT error checking (local to this file; mirrors the SF-18 pattern in
// physics/stochastic/PeriodicGaussianField.cu)
// ============================================================================

inline void cufft_check_impl(cufftResult status, const char* file, int line) {
    if (status != CUFFT_SUCCESS) {
        std::string msg = std::string("cuFFT error at ") + file + ":" + std::to_string(line) +
                          " - code " + std::to_string(static_cast<int>(status));
        throw std::runtime_error(msg);
    }
}

#define MF3D_CUFFT_CHECK(expr)                                                                     \
    ::macroflow3d::interpolation::cufft_check_impl((expr), __FILE__, __LINE__)

static constexpr double PI_D = 3.141592653589793238462643383279502884;

// Destroys a cuFFT plan on scope exit if it was not released explicitly
// (exception path only; the normal path destroys under MF3D_CUFFT_CHECK).
struct CufftPlanGuard {
    cufftHandle handle = 0;
    bool owned = false;
    ~CufftPlanGuard() {
        if (owned) {
            cufftDestroy(handle);
        }
    }
    cufftHandle release() {
        owned = false;
        return handle;
    }
};

// ============================================================================
// Validation (distinct messages; see header section 1)
// ============================================================================

void validate_grid(const Grid3D& grid, const char* who) {
    if (grid.nx < 4 || grid.ny < 4 || grid.nz < 4) {
        throw std::invalid_argument(
            std::string(who) +
            ": grid extents (nx,ny,nz) must each be >= 4 (the 4-point cubic B-spline stencil "
            "must not wrap onto itself)");
    }
    if (!std::isfinite(grid.dx) || grid.dx <= 0.0 || !std::isfinite(grid.dy) || grid.dy <= 0.0 ||
        !std::isfinite(grid.dz) || grid.dz <= 0.0) {
        throw std::invalid_argument(std::string(who) +
                                    ": grid spacings (dx,dy,dz) must be finite and > 0");
    }
}

PeriodicTricubicBSplineView view_from_grid(const Grid3D& grid, const real* coeff) {
    PeriodicTricubicBSplineView v;
    v.coeff = coeff;
    v.nx = grid.nx;
    v.ny = grid.ny;
    v.nz = grid.nz;
    v.hx = grid.dx;
    v.hy = grid.dy;
    v.hz = grid.dz;
    v.Lx = grid.Lx();
    v.Ly = grid.Ly();
    v.Lz = grid.Lz();
    return v;
}

// ============================================================================
// Kernels
// ============================================================================

__device__ __forceinline__ real bspline_symbol(int m, int n) {
    return static_cast<real>(2.0) / static_cast<real>(3.0) +
           cos(static_cast<real>(2.0 * PI_D) * static_cast<real>(m) / static_cast<real>(n)) /
               static_cast<real>(3.0);
}

// Multiplies each half-spectrum entry by 1 / (Bhat_x Bhat_y Bhat_z N1 N2 N3).
// No phase twist (samples and coefficients share the index lattice).
__global__ void kernel_divide_by_symbol(cufftDoubleComplex* __restrict__ spec, const int nx,
                                        const int ny, const int nz, const int hx,
                                        const real inv_total) {
    const int gx = threadIdx.x + blockIdx.x * blockDim.x;
    const int gy = threadIdx.y + blockIdx.y * blockDim.y;
    const int gz = threadIdx.z + blockIdx.z * blockDim.z;
    if (gx >= hx || gy >= ny || gz >= nz)
        return;

    const size_t idx = static_cast<size_t>(gx) +
                       static_cast<size_t>(hx) *
                           (static_cast<size_t>(gy) + static_cast<size_t>(ny) * static_cast<size_t>(gz));

    const int mx = gx; // stored non-negative x-frequency
    const int my_s = (gy < ny / 2) ? gy : gy - ny;
    const int mz_s = (gz < nz / 2) ? gz : gz - nz;

    const real b = bspline_symbol(mx, nx) * bspline_symbol(my_s, ny) * bspline_symbol(mz_s, nz);
    const real scale = inv_total / b;

    cufftDoubleComplex val = spec[idx];
    val.x *= scale;
    val.y *= scale;
    spec[idx] = val;
}

__global__ void kernel_evaluate(const PeriodicTricubicBSplineView view, const size_t n,
                                const real* __restrict__ px, const real* __restrict__ py,
                                const real* __restrict__ pz, real* __restrict__ value,
                                real* __restrict__ gx, real* __restrict__ gy,
                                real* __restrict__ gz) {
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t p = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; p < n;
         p += stride) {
        real f, fx, fy, fz;
        evaluate_point(view, px[p], py[p], pz[p], f, fx, fy, fz);
        value[p] = f;
        gx[p] = fx;
        gy[p] = fy;
        gz[p] = fz;
    }
}

// ============================================================================
// CPU mirror: exact cyclic tridiagonal solve of c_{i-1} + 4 c_i + c_{i+1} = r_i
// (Thomas algorithm + Sherman-Morrison for the two corner entries; Numerical
// Recipes "cyclic" with alpha = beta = 1, gamma = -b_0 = -4).
// ============================================================================

class CyclicBSplineSolver {
  public:
    explicit CyclicBSplineSolver(int n)
        : n_(n), cp_(static_cast<size_t>(n)), inv_den_(static_cast<size_t>(n)),
          z_(static_cast<size_t>(n)), tmp_(static_cast<size_t>(n)) {
        // Modified diagonal: bb_0 = 4 - gamma, bb_{n-1} = 4 - alpha*beta/gamma, else 4.
        for (int i = 0; i < n_; ++i) {
            real bb = 4.0;
            if (i == 0)
                bb = 4.0 - kGamma;
            if (i == n_ - 1)
                bb = 4.0 - 1.0 / kGamma;
            const real den = (i == 0) ? bb : bb - cp_[static_cast<size_t>(i - 1)];
            inv_den_[static_cast<size_t>(i)] = 1.0 / den;
            cp_[static_cast<size_t>(i)] = 1.0 / den; // super-diagonal c = 1
        }
        // z solves the modified tridiagonal system with u = [gamma, 0, ..., 0, alpha].
        std::fill(tmp_.begin(), tmp_.end(), 0.0);
        tmp_[0] = kGamma;
        tmp_[static_cast<size_t>(n_ - 1)] = 1.0;
        thomas(tmp_.data(), z_.data());
        z_factor_den_ = 1.0 + z_[0] + z_[static_cast<size_t>(n_ - 1)] / kGamma;
    }

    // In-place deconvolution of one line base[i*stride], i = 0..n-1: on entry
    // the samples f_i, on exit the coefficients c_i solving
    // (1/6) c_{i-1} + (2/3) c_i + (1/6) c_{i+1} = f_i (periodic).
    void solve_line(real* base, size_t stride) {
        for (int i = 0; i < n_; ++i) {
            tmp_[static_cast<size_t>(i)] = 6.0 * base[static_cast<size_t>(i) * stride];
        }
        std::vector<real>& x = line_;
        x.resize(static_cast<size_t>(n_));
        thomas(tmp_.data(), x.data());
        const real fact = (x[0] + x[static_cast<size_t>(n_ - 1)] / kGamma) / z_factor_den_;
        for (int i = 0; i < n_; ++i) {
            base[static_cast<size_t>(i) * stride] =
                x[static_cast<size_t>(i)] - fact * z_[static_cast<size_t>(i)];
        }
    }

  private:
    static constexpr real kGamma = -4.0;

    // Tridiagonal solve with sub/super-diagonal 1 and the modified diagonal
    // factored in the constructor. r and x may not alias.
    void thomas(const real* r, real* x) const {
        x[0] = r[0] * inv_den_[0];
        for (int i = 1; i < n_; ++i) {
            x[i] = (r[i] - x[i - 1]) * inv_den_[static_cast<size_t>(i)];
        }
        for (int i = n_ - 2; i >= 0; --i) {
            x[i] -= cp_[static_cast<size_t>(i)] * x[i + 1];
        }
    }

    int n_;
    std::vector<real> cp_;
    std::vector<real> inv_den_;
    std::vector<real> z_;
    std::vector<real> tmp_;
    std::vector<real> line_;
    real z_factor_den_ = 1.0;
};

} // namespace

// ============================================================================
// Host API
// ============================================================================

PeriodicTricubicBSplineView PeriodicTricubicBSplineWorkspace::view() const {
    if (grid.nx <= 0 || grid.ny <= 0 || grid.nz <= 0 || coefficients.data() == nullptr ||
        coefficients.size() != grid.num_cells()) {
        throw std::logic_error(
            "PeriodicTricubicBSplineWorkspace::view: no prefilter has completed on this workspace");
    }
    return view_from_grid(grid, coefficients.data());
}

PeriodicTricubicBSplineReport
prefilter_periodic_tricubic_bspline(const CudaContext& ctx, const Grid3D& grid,
                                    DeviceSpan<const real> samples,
                                    PeriodicTricubicBSplineWorkspace& workspace) {
    validate_grid(grid, "prefilter_periodic_tricubic_bspline");
    const size_t n = grid.num_cells();
    if (samples.size() != n) {
        throw std::invalid_argument(
            "prefilter_periodic_tricubic_bspline: samples size must equal grid.num_cells()");
    }
    if (samples.data() == nullptr) {
        throw std::invalid_argument(
            "prefilter_periodic_tricubic_bspline: samples pointer must be non-null");
    }

    const int nx = grid.nx, ny = grid.ny, nz = grid.nz;
    const int hx = nx / 2 + 1;
    const size_t spectrum_count =
        static_cast<size_t>(hx) * static_cast<size_t>(ny) * static_cast<size_t>(nz);

    // Invalidate the recorded grid until this call completes.
    workspace.grid = Grid3D();
    workspace.coefficients.resize(n);
    workspace.spectrum.resize(spectrum_count);

    // Plans created per call (D-6); cuFFT row-major (last argument fastest)
    // matches the x-fastest layout when called as (nz, ny, nx).
    size_t work_d2z = 0, work_z2d = 0;
    CufftPlanGuard plan_d2z, plan_z2d;
    MF3D_CUFFT_CHECK(cufftPlan3d(&plan_d2z.handle, nz, ny, nx, CUFFT_D2Z));
    plan_d2z.owned = true;
    MF3D_CUFFT_CHECK(cufftSetStream(plan_d2z.handle, ctx.cuda_stream()));
    MF3D_CUFFT_CHECK(cufftGetSize(plan_d2z.handle, &work_d2z));

    MF3D_CUFFT_CHECK(cufftPlan3d(&plan_z2d.handle, nz, ny, nx, CUFFT_Z2D));
    plan_z2d.owned = true;
    MF3D_CUFFT_CHECK(cufftSetStream(plan_z2d.handle, ctx.cuda_stream()));
    MF3D_CUFFT_CHECK(cufftGetSize(plan_z2d.handle, &work_z2d));

    // --- Stage 1: forward D2Z (out-of-place; the real input is not modified) ---
    MF3D_CUFFT_CHECK(cufftExecD2Z(plan_d2z.handle,
                                  const_cast<cufftDoubleReal*>(samples.data()),
                                  workspace.spectrum.data()));

    // --- Stage 2: divide by the B-spline symbol and by N1 N2 N3 ---
    {
        const real inv_total = 1.0 / static_cast<real>(n);
        dim3 block(8, 8, 8);
        dim3 grid_dim(static_cast<unsigned int>((hx + block.x - 1) / block.x),
                      static_cast<unsigned int>((ny + block.y - 1) / block.y),
                      static_cast<unsigned int>((nz + block.z - 1) / block.z));
        kernel_divide_by_symbol<<<grid_dim, block, 0, ctx.cuda_stream()>>>(
            workspace.spectrum.data(), nx, ny, nz, hx, inv_total);
        MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
    }

    // --- Stage 3: inverse Z2D into the coefficients (overwrites the spectrum) ---
    MF3D_CUFFT_CHECK(
        cufftExecZ2D(plan_z2d.handle, workspace.spectrum.data(), workspace.coefficients.data()));

    ctx.synchronize();
    MF3D_CUFFT_CHECK(cufftDestroy(plan_d2z.release()));
    MF3D_CUFFT_CHECK(cufftDestroy(plan_z2d.release()));

    workspace.grid = grid;

    PeriodicTricubicBSplineReport report;
    report.coefficient_bytes = workspace.coefficients.capacity() * sizeof(real);
    report.spectrum_bytes = workspace.spectrum.capacity() * sizeof(cufftDoubleComplex);
    report.cufft_work_area_bytes = work_d2z + work_z2d;
    report.total_device_bytes =
        report.coefficient_bytes + report.spectrum_bytes + report.cufft_work_area_bytes;
    return report;
}

void evaluate_periodic_tricubic_bspline(const CudaContext& ctx,
                                        const PeriodicTricubicBSplineView& view,
                                        DeviceSpan<const real> px, DeviceSpan<const real> py,
                                        DeviceSpan<const real> pz, DeviceSpan<real> value,
                                        DeviceSpan<real> gx, DeviceSpan<real> gy,
                                        DeviceSpan<real> gz) {
    if (view.coeff == nullptr || view.nx < 4 || view.ny < 4 || view.nz < 4) {
        throw std::invalid_argument(
            "evaluate_periodic_tricubic_bspline: view has null coefficients or extents < 4");
    }
    const size_t n = px.size();
    if (py.size() != n || pz.size() != n) {
        throw std::invalid_argument(
            "evaluate_periodic_tricubic_bspline: point spans px, py, pz must have equal size");
    }
    if (value.size() != n || gx.size() != n || gy.size() != n || gz.size() != n) {
        throw std::invalid_argument(
            "evaluate_periodic_tricubic_bspline: output spans value, gx, gy, gz must have the "
            "size of the point spans");
    }
    if (n == 0) {
        return;
    }

    const int block = 256;
    const size_t blocks_needed = (n + static_cast<size_t>(block) - 1) / static_cast<size_t>(block);
    const unsigned int blocks =
        static_cast<unsigned int>(std::min<size_t>(blocks_needed, static_cast<size_t>(1) << 20));
    kernel_evaluate<<<blocks, block, 0, ctx.cuda_stream()>>>(view, n, px.data(), py.data(),
                                                             pz.data(), value.data(), gx.data(),
                                                             gy.data(), gz.data());
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
}

void prefilter_periodic_tricubic_bspline_host(const Grid3D& grid, const real* samples,
                                              real* coefficients) {
    validate_grid(grid, "prefilter_periodic_tricubic_bspline_host");
    if (samples == nullptr || coefficients == nullptr) {
        throw std::invalid_argument(
            "prefilter_periodic_tricubic_bspline_host: samples and coefficients pointers must be "
            "non-null");
    }
    const int nx = grid.nx, ny = grid.ny, nz = grid.nz;
    const size_t n = grid.num_cells();
    if (samples != coefficients) {
        std::memmove(coefficients, samples, n * sizeof(real));
    }

    const size_t sx = 1;
    const size_t sy = static_cast<size_t>(nx);
    const size_t sz = static_cast<size_t>(nx) * static_cast<size_t>(ny);

    // Axis x: lines along i for every (j, k).
    {
        CyclicBSplineSolver solver(nx);
        for (int k = 0; k < nz; ++k)
            for (int j = 0; j < ny; ++j)
                solver.solve_line(coefficients + grid.idx(0, j, k), sx);
    }
    // Axis y: lines along j for every (i, k).
    {
        CyclicBSplineSolver solver(ny);
        for (int k = 0; k < nz; ++k)
            for (int i = 0; i < nx; ++i)
                solver.solve_line(coefficients + grid.idx(i, 0, k), sy);
    }
    // Axis z: lines along k for every (i, j).
    {
        CyclicBSplineSolver solver(nz);
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i)
                solver.solve_line(coefficients + grid.idx(i, j, 0), sz);
    }
}

PeriodicTricubicBSplineView make_host_view(const Grid3D& grid, const real* host_coefficients) {
    validate_grid(grid, "make_host_view");
    if (host_coefficients == nullptr) {
        throw std::invalid_argument("make_host_view: host_coefficients pointer must be non-null");
    }
    return view_from_grid(grid, host_coefficients);
}

} // namespace interpolation
} // namespace macroflow3d
