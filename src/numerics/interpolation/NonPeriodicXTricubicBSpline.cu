/**
 * @file NonPeriodicXTricubicBSpline.cu
 * @brief Non-periodic-in-x1 tricubic B-spline (SF-41, C-SPLINE) - implementation.
 *
 * See NonPeriodicXTricubicBSpline.cuh for the specification (conventions,
 * not-a-knot closure, evaluation domain, prefilter order, memory accounting).
 */

#include "../../runtime/cuda_check.cuh"
#include "NonPeriodicXTricubicBSpline.cuh"

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <vector>

namespace macroflow3d {
namespace interpolation {

namespace {

inline void cufft_check_impl(cufftResult status, const char* file, int line) {
    if (status != CUFFT_SUCCESS) {
        std::string msg = std::string("cuFFT error at ") + file + ":" + std::to_string(line) +
                          " - code " + std::to_string(static_cast<int>(status));
        throw std::runtime_error(msg);
    }
}

#define MF3D_NP_CUFFT_CHECK(expr)                                                                  \
    ::macroflow3d::interpolation::cufft_check_impl((expr), __FILE__, __LINE__)

static constexpr double PI_D = 3.141592653589793238462643383279502884;

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

void validate_grid(const Grid3D& grid, const char* who) {
    if (grid.nx < 8 || grid.ny < 4 || grid.nz < 4) {
        throw std::invalid_argument(
            std::string(who) +
            ": grid extents must satisfy nx >= 8 (reduced not-a-knot x system) and ny, nz >= 4 "
            "(periodic 4-point stencil must not wrap onto itself)");
    }
    if (!std::isfinite(grid.dx) || grid.dx <= 0.0 || !std::isfinite(grid.dy) || grid.dy <= 0.0 ||
        !std::isfinite(grid.dz) || grid.dz <= 0.0) {
        throw std::invalid_argument(std::string(who) +
                                    ": grid spacings (dx,dy,dz) must be finite and > 0");
    }
}

NonPeriodicXTricubicBSplineView view_from_grid(const Grid3D& grid, const real* coeff) {
    NonPeriodicXTricubicBSplineView v;
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
// Not-a-knot x line solve ("reduced" variant; header section 3). Same inline
// code on host and device.
// ============================================================================

// Reciprocal Thomas pivots of the [1 4 1] system, m entries: d_0 = 4,
// d_i = 4 - 1/d_{i-1}, inv[i] = 1/d_i.
__host__ __device__ inline void build_pivot_table(real* inv, int m) {
    real d = 4.0;
    inv[0] = 1.0 / d;
    for (int i = 1; i < m; ++i) {
        d = 4.0 - inv[i - 1];
        inv[i] = 1.0 / d;
    }
}

// f[i], i = 0..n-1 (stride 1) -> c[a + 2], a = -2..n+1. inv: n-4 pivots.
__host__ __device__ inline void nak_x_line(const real* __restrict__ f, real* __restrict__ c,
                                           const real* __restrict__ inv, int n) {
    const int m = n - 4; // unknowns c_2..c_{n-3}, stored at c[4 + i]
    const real c1 = (8.0 * f[1] - f[0] - f[2]) / 6.0;
    const real cn2 = (8.0 * f[n - 2] - f[n - 1] - f[n - 3]) / 6.0;

    for (int i = 0; i < m; ++i) { // forward sweep (m >= 4 since nx >= 8)
        real r = 6.0 * f[2 + i];
        if (i == 0)
            r -= c1;
        if (i == m - 1)
            r -= cn2;
        c[4 + i] = (i == 0) ? r * inv[0] : (r - c[4 + i - 1]) * inv[i];
    }
    for (int i = m - 2; i >= 0; --i)
        c[4 + i] = c[4 + i] - inv[i] * c[4 + i + 1];

    c[3] = c1;                             // c_1
    c[n] = cn2;                            // c_{n-2}
    c[2] = 6.0 * f[1] - 4.0 * c1 - c[4];   // c_0
    c[1] = 6.0 * f[0] - 4.0 * c[2] - c1;   // c_{-1}
    c[n + 1] = 6.0 * f[n - 2] - 4.0 * cn2 - c[n - 1]; // c_{n-1}
    c[n + 2] = 6.0 * f[n - 1] - 4.0 * c[n + 1] - cn2; // c_n
    c[0] = 4.0 * c[1] - 6.0 * c[2] + 4.0 * c1 - c[4];                   // c_{-2}
    c[n + 3] = 4.0 * c[n + 2] - 6.0 * c[n + 1] + 4.0 * cn2 - c[n - 1]; // c_{n+1}
}

// ============================================================================
// Kernels
// ============================================================================

__global__ void kernel_pivot_table(real* inv, int m) {
    if (blockIdx.x == 0 && threadIdx.x == 0)
        build_pivot_table(inv, m);
}

__global__ void kernel_x_solve(const real* __restrict__ samples, real* __restrict__ coeff,
                               const real* __restrict__ inv, const int nx, const size_t nlines) {
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t line = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; line < nlines;
         line += stride) {
        nak_x_line(samples + line * static_cast<size_t>(nx),
                   coeff + line * (static_cast<size_t>(nx) + 4), inv, nx);
    }
}

__device__ __forceinline__ real bspline_symbol(int m, int n) {
    return static_cast<real>(2.0) / static_cast<real>(3.0) +
           cos(static_cast<real>(2.0 * PI_D) * static_cast<real>(m) / static_cast<real>(n)) /
               static_cast<real>(3.0);
}

// Spectrum layout b + nxe*(gy + hy*gz): b = x plane (batch), gy half-spectrum
// y frequency, gz z frequency (wrapped to signed; the symbol is even).
__global__ void kernel_divide_by_symbol(cufftDoubleComplex* __restrict__ spec, const size_t count,
                                        const int nxe, const int ny, const int nz, const int hy,
                                        const real inv_total) {
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < count;
         idx += stride) {
        const size_t rest = idx / static_cast<size_t>(nxe);
        const int gy = static_cast<int>(rest % static_cast<size_t>(hy));
        const int gz = static_cast<int>(rest / static_cast<size_t>(hy));
        const int mz_s = (gz < nz / 2) ? gz : gz - nz;
        const real b = bspline_symbol(gy, ny) * bspline_symbol(mz_s, nz);
        const real scale = inv_total / b;
        cufftDoubleComplex val = spec[idx];
        val.x *= scale;
        val.y *= scale;
        spec[idx] = val;
    }
}

__global__ void kernel_evaluate(const NonPeriodicXTricubicBSplineView view, const size_t n,
                                const real* __restrict__ px, const real* __restrict__ py,
                                const real* __restrict__ pz, real* __restrict__ value,
                                real* __restrict__ gx, real* __restrict__ gy,
                                real* __restrict__ gz, int* __restrict__ inside) {
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t p = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; p < n;
         p += stride) {
        real f, fx, fy, fz;
        const bool in = evaluate_point(view, px[p], py[p], pz[p], f, fx, fy, fz);
        value[p] = f;
        gx[p] = fx;
        gy[p] = fy;
        gz[p] = fz;
        inside[p] = in ? 1 : 0;
    }
}

__global__ void kernel_evaluate_hessian(const NonPeriodicXTricubicBSplineView view, const size_t n,
                                        const real* __restrict__ px, const real* __restrict__ py,
                                        const real* __restrict__ pz, real* __restrict__ value,
                                        real* __restrict__ gx, real* __restrict__ gy,
                                        real* __restrict__ gz, real* __restrict__ hessian,
                                        int* __restrict__ inside) {
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t p = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; p < n;
         p += stride) {
        real f, fx, fy, fz, H[6];
        const bool in = evaluate_point_hessian(view, px[p], py[p], pz[p], f, fx, fy, fz, H);
        value[p] = f;
        gx[p] = fx;
        gy[p] = fy;
        gz[p] = fz;
        for (int c = 0; c < 6; ++c)
            hessian[6 * p + c] = H[c];
        inside[p] = in ? 1 : 0;
    }
}

// ============================================================================
// CPU mirror: cyclic tridiagonal solve (copy of the SF-28 algorithm; SF-28
// files stay untouched): c_{i-1} + 4 c_i + c_{i+1} = r_i, Thomas +
// Sherman-Morrison (alpha = beta = 1, gamma = -4).
// ============================================================================

class CyclicBSplineSolver {
  public:
    explicit CyclicBSplineSolver(int n)
        : n_(n), cp_(static_cast<size_t>(n)), inv_den_(static_cast<size_t>(n)),
          z_(static_cast<size_t>(n)), tmp_(static_cast<size_t>(n)) {
        for (int i = 0; i < n_; ++i) {
            real bb = 4.0;
            if (i == 0)
                bb = 4.0 - kGamma;
            if (i == n_ - 1)
                bb = 4.0 - 1.0 / kGamma;
            const real den = (i == 0) ? bb : bb - cp_[static_cast<size_t>(i - 1)];
            inv_den_[static_cast<size_t>(i)] = 1.0 / den;
            cp_[static_cast<size_t>(i)] = 1.0 / den;
        }
        std::fill(tmp_.begin(), tmp_.end(), 0.0);
        tmp_[0] = kGamma;
        tmp_[static_cast<size_t>(n_ - 1)] = 1.0;
        thomas(tmp_.data(), z_.data());
        z_factor_den_ = 1.0 + z_[0] + z_[static_cast<size_t>(n_ - 1)] / kGamma;
    }

    // In-place deconvolution of one line base[i*stride] (samples -> coefficients of
    // (1/6) c_{i-1} + (2/3) c_i + (1/6) c_{i+1} = f_i, periodic).
    void solve_line(real* base, size_t stride) {
        for (int i = 0; i < n_; ++i)
            tmp_[static_cast<size_t>(i)] = 6.0 * base[static_cast<size_t>(i) * stride];
        std::vector<real>& x = line_;
        x.resize(static_cast<size_t>(n_));
        thomas(tmp_.data(), x.data());
        const real fact = (x[0] + x[static_cast<size_t>(n_ - 1)] / kGamma) / z_factor_den_;
        for (int i = 0; i < n_; ++i)
            base[static_cast<size_t>(i) * stride] =
                x[static_cast<size_t>(i)] - fact * z_[static_cast<size_t>(i)];
    }

  private:
    static constexpr real kGamma = -4.0;

    void thomas(const real* r, real* x) const {
        x[0] = r[0] * inv_den_[0];
        for (int i = 1; i < n_; ++i)
            x[i] = (r[i] - x[i - 1]) * inv_den_[static_cast<size_t>(i)];
        for (int i = n_ - 2; i >= 0; --i)
            x[i] -= cp_[static_cast<size_t>(i)] * x[i + 1];
    }

    int n_;
    std::vector<real> cp_;
    std::vector<real> inv_den_;
    std::vector<real> z_;
    std::vector<real> tmp_;
    std::vector<real> line_;
    real z_factor_den_ = 1.0;
};

void check_spans_equal(size_t n, std::initializer_list<size_t> sizes, const char* who) {
    for (size_t sz : sizes)
        if (sz != n)
            throw std::invalid_argument(std::string(who) +
                                        ": all point/output spans must have the same length");
}

unsigned int grid_blocks(size_t n, int block) {
    const size_t needed = (n + static_cast<size_t>(block) - 1) / static_cast<size_t>(block);
    return static_cast<unsigned int>(std::min<size_t>(needed, static_cast<size_t>(1) << 20));
}

} // namespace

// ============================================================================
// Host API
// ============================================================================

NonPeriodicXTricubicBSplineView NonPeriodicXTricubicBSplineWorkspace::view() const {
    if (grid.nx <= 0 || grid.ny <= 0 || grid.nz <= 0 || coefficients.data() == nullptr ||
        coefficients.size() != (static_cast<size_t>(grid.nx) + 4) * static_cast<size_t>(grid.ny) *
                                   static_cast<size_t>(grid.nz)) {
        throw std::logic_error("NonPeriodicXTricubicBSplineWorkspace::view: no prefilter has "
                               "completed on this workspace");
    }
    return view_from_grid(grid, coefficients.data());
}

NonPeriodicXTricubicBSplineView make_nonperiodic_x_view(const Grid3D& grid,
                                                        const real* coefficients) {
    validate_grid(grid, "make_nonperiodic_x_view");
    if (coefficients == nullptr) {
        throw std::invalid_argument("make_nonperiodic_x_view: coefficients pointer must be non-null");
    }
    return view_from_grid(grid, coefficients);
}

NonPeriodicXTricubicBSplineReport
prefilter_nonperiodic_x_tricubic_bspline(const CudaContext& ctx, const Grid3D& grid,
                                         DeviceSpan<const real> samples,
                                         NonPeriodicXTricubicBSplineWorkspace& workspace) {
    validate_grid(grid, "prefilter_nonperiodic_x_tricubic_bspline");
    const size_t n = grid.num_cells();
    if (samples.size() != n) {
        throw std::invalid_argument(
            "prefilter_nonperiodic_x_tricubic_bspline: samples size must equal grid.num_cells()");
    }
    if (samples.data() == nullptr) {
        throw std::invalid_argument(
            "prefilter_nonperiodic_x_tricubic_bspline: samples pointer must be non-null");
    }

    const int nx = grid.nx, ny = grid.ny, nz = grid.nz;
    const int nxe = nx + 4;
    const int hy = ny / 2 + 1;
    const size_t nlines = static_cast<size_t>(ny) * static_cast<size_t>(nz);
    const size_t coeff_count = static_cast<size_t>(nxe) * nlines;
    const size_t spectrum_count =
        static_cast<size_t>(nxe) * static_cast<size_t>(hy) * static_cast<size_t>(nz);

    workspace.grid = Grid3D(); // invalid until this call completes
    workspace.coefficients.resize(coeff_count);
    workspace.spectrum.resize(spectrum_count);
    workspace.line_table.resize(static_cast<size_t>(nx - 4));

    // Plans per call (SF-28 D-6): batched 2-D transform over (y,z) planes; cuFFT
    // row-major n = {nz, ny} (y fastest); the batch index is the x index, which
    // is the fastest memory index: element (kz,ky) of batch b is at
    // b + nxe*(ky + ny*kz) in real space, b + nxe*(ky + hy*kz) in the spectrum.
    int dims[2] = {nz, ny};
    int real_embed[2] = {nz, ny};
    int cplx_embed[2] = {nz, hy};
    size_t work_d2z = 0, work_z2d = 0;
    CufftPlanGuard plan_d2z, plan_z2d;
    MF3D_NP_CUFFT_CHECK(cufftPlanMany(&plan_d2z.handle, 2, dims, real_embed, nxe, 1, cplx_embed, nxe,
                                      1, CUFFT_D2Z, nxe));
    plan_d2z.owned = true;
    MF3D_NP_CUFFT_CHECK(cufftSetStream(plan_d2z.handle, ctx.cuda_stream()));
    MF3D_NP_CUFFT_CHECK(cufftGetSize(plan_d2z.handle, &work_d2z));
    MF3D_NP_CUFFT_CHECK(cufftPlanMany(&plan_z2d.handle, 2, dims, cplx_embed, nxe, 1, real_embed, nxe,
                                      1, CUFFT_Z2D, nxe));
    plan_z2d.owned = true;
    MF3D_NP_CUFFT_CHECK(cufftSetStream(plan_z2d.handle, ctx.cuda_stream()));
    MF3D_NP_CUFFT_CHECK(cufftGetSize(plan_z2d.handle, &work_z2d));

    // --- Stage 1: not-a-knot x line solve, samples -> extended coefficient lines ---
    kernel_pivot_table<<<1, 1, 0, ctx.cuda_stream()>>>(workspace.line_table.data(), nx - 4);
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
    {
        const int block = 128;
        kernel_x_solve<<<grid_blocks(nlines, block), block, 0, ctx.cuda_stream()>>>(
            samples.data(), workspace.coefficients.data(), workspace.line_table.data(), nx, nlines);
        MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
    }

    // --- Stage 2: y/z deconvolution of all nx+4 planes (FFT, symbol division) ---
    MF3D_NP_CUFFT_CHECK(
        cufftExecD2Z(plan_d2z.handle, workspace.coefficients.data(), workspace.spectrum.data()));
    {
        const int block = 256;
        const real inv_total = 1.0 / (static_cast<real>(ny) * static_cast<real>(nz));
        kernel_divide_by_symbol<<<grid_blocks(spectrum_count, block), block, 0, ctx.cuda_stream()>>>(
            workspace.spectrum.data(), spectrum_count, nxe, ny, nz, hy, inv_total);
        MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
    }
    MF3D_NP_CUFFT_CHECK(
        cufftExecZ2D(plan_z2d.handle, workspace.spectrum.data(), workspace.coefficients.data()));

    ctx.synchronize();
    MF3D_NP_CUFFT_CHECK(cufftDestroy(plan_d2z.release()));
    MF3D_NP_CUFFT_CHECK(cufftDestroy(plan_z2d.release()));

    workspace.grid = grid;

    NonPeriodicXTricubicBSplineReport report;
    report.coefficient_bytes = workspace.coefficients.capacity() * sizeof(real);
    report.spectrum_bytes = workspace.spectrum.capacity() * sizeof(cufftDoubleComplex);
    report.line_table_bytes = workspace.line_table.capacity() * sizeof(real);
    report.cufft_work_area_bytes = work_d2z + work_z2d;
    report.total_device_bytes = report.coefficient_bytes + report.spectrum_bytes +
                                report.line_table_bytes + report.cufft_work_area_bytes;
    return report;
}

void prefilter_nonperiodic_x_tricubic_bspline_host(const Grid3D& grid, const real* samples,
                                                   real* coefficients) {
    validate_grid(grid, "prefilter_nonperiodic_x_tricubic_bspline_host");
    if (samples == nullptr || coefficients == nullptr) {
        throw std::invalid_argument("prefilter_nonperiodic_x_tricubic_bspline_host: samples and "
                                    "coefficients pointers must be non-null");
    }
    const int nx = grid.nx, ny = grid.ny, nz = grid.nz;
    const size_t nxe = static_cast<size_t>(nx) + 4;
    const size_t nlines = static_cast<size_t>(ny) * static_cast<size_t>(nz);

    // Stage 1: x solve (same inline code as the device kernel).
    std::vector<real> inv(static_cast<size_t>(nx - 4));
    build_pivot_table(inv.data(), nx - 4);
    for (size_t line = 0; line < nlines; ++line)
        nak_x_line(samples + line * static_cast<size_t>(nx), coefficients + line * nxe, inv.data(),
                   nx);

    // Stage 2: cyclic tridiagonal line solves in y then z on all nx+4 planes.
    const size_t sy = nxe;
    const size_t sz = nxe * static_cast<size_t>(ny);
    {
        CyclicBSplineSolver solver(ny);
        for (int k = 0; k < nz; ++k)
            for (size_t a = 0; a < nxe; ++a)
                solver.solve_line(coefficients + a + sz * static_cast<size_t>(k), sy);
    }
    {
        CyclicBSplineSolver solver(nz);
        for (int j = 0; j < ny; ++j)
            for (size_t a = 0; a < nxe; ++a)
                solver.solve_line(coefficients + a + sy * static_cast<size_t>(j), sz);
    }
}

void evaluate_nonperiodic_x_tricubic_bspline(const CudaContext& ctx,
                                             const NonPeriodicXTricubicBSplineView& view,
                                             DeviceSpan<const real> px, DeviceSpan<const real> py,
                                             DeviceSpan<const real> pz, DeviceSpan<real> value,
                                             DeviceSpan<real> gx, DeviceSpan<real> gy,
                                             DeviceSpan<real> gz, DeviceSpan<int> inside) {
    if (view.coeff == nullptr || view.nx < 8 || view.ny < 4 || view.nz < 4) {
        throw std::invalid_argument(
            "evaluate_nonperiodic_x_tricubic_bspline: view has null coefficients or small extents");
    }
    const size_t n = px.size();
    check_spans_equal(n, {py.size(), pz.size(), value.size(), gx.size(), gy.size(), gz.size(),
                          inside.size()},
                      "evaluate_nonperiodic_x_tricubic_bspline");
    if (n == 0)
        return;
    const int block = 256;
    kernel_evaluate<<<grid_blocks(n, block), block, 0, ctx.cuda_stream()>>>(
        view, n, px.data(), py.data(), pz.data(), value.data(), gx.data(), gy.data(), gz.data(),
        inside.data());
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
}

void evaluate_nonperiodic_x_tricubic_bspline_hessian(
    const CudaContext& ctx, const NonPeriodicXTricubicBSplineView& view, DeviceSpan<const real> px,
    DeviceSpan<const real> py, DeviceSpan<const real> pz, DeviceSpan<real> value,
    DeviceSpan<real> gx, DeviceSpan<real> gy, DeviceSpan<real> gz, DeviceSpan<real> hessian,
    DeviceSpan<int> inside) {
    if (view.coeff == nullptr || view.nx < 8 || view.ny < 4 || view.nz < 4) {
        throw std::invalid_argument("evaluate_nonperiodic_x_tricubic_bspline_hessian: view has "
                                    "null coefficients or small extents");
    }
    const size_t n = px.size();
    check_spans_equal(n, {py.size(), pz.size(), value.size(), gx.size(), gy.size(), gz.size(),
                          inside.size()},
                      "evaluate_nonperiodic_x_tricubic_bspline_hessian");
    if (hessian.size() != 6 * n) {
        throw std::invalid_argument(
            "evaluate_nonperiodic_x_tricubic_bspline_hessian: hessian span must have 6 entries "
            "per point");
    }
    if (n == 0)
        return;
    const int block = 128;
    kernel_evaluate_hessian<<<grid_blocks(n, block), block, 0, ctx.cuda_stream()>>>(
        view, n, px.data(), py.data(), pz.data(), value.data(), gx.data(), gy.data(), gz.data(),
        hessian.data(), inside.data());
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
}

} // namespace interpolation
} // namespace macroflow3d
