/**
 * @file velocity_from_head.cu
 * @brief Implementation of velocity computation from head (Darcy's law)
 * 
 * Structure:
 *   1. Device helpers (inline, no overhead)
 *   2. Interior kernels (bulk, no branches, max performance)
 *   3. Boundary kernels (6 faces + edges/vertices if needed)
 *   4. Host orchestration (launch all kernels)
 * 
 * Reference: legacy/compute_velocity_from_head_for_par2.cu
 */

#include "velocity_from_head.cuh"
#include "../../core/BCSpecDevice.cuh"
#include "../../core/DeviceBuffer.cuh"
#include "../../runtime/cuda_check.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <vector>
#include <algorithm>

namespace macroflow3d {
namespace physics {

// ============================================================================
// Device helpers (inline, zero-overhead)
// ============================================================================

/**
 * @brief Harmonic mean of two positive values
 * K_eff = 2*Ka*Kb / (Ka + Kb) = 2 / (1/Ka + 1/Kb)
 */
__device__ __forceinline__
real harmonic_mean(real Ka, real Kb) {
    return 2.0 * Ka * Kb / (Ka + Kb);
}

/**
 * @brief Cell-centered linear index: idx = i + j*nx + k*nx*ny
 */
__device__ __forceinline__
int cell_idx(int i, int j, int k, int nx, int ny) {
    return i + j * nx + k * nx * ny;
}

/**
 * @brief U-face index (staggered in x): idx = i + j*(nx+1) + k*(nx+1)*ny
 * Face i is between cell i-1 and cell i (for i=1..nx-1)
 * Face 0 is at x=0 boundary, face nx is at x=Lx boundary
 */
__device__ __forceinline__
int U_idx(int i, int j, int k, int nx, int ny) {
    return i + j * (nx + 1) + k * (nx + 1) * ny;
}

/**
 * @brief V-face index (staggered in y): idx = i + j*nx + k*nx*(ny+1)
 */
__device__ __forceinline__
int V_idx(int i, int j, int k, int nx, int ny) {
    return i + j * nx + k * nx * (ny + 1);
}

/**
 * @brief W-face index (staggered in z): idx = i + j*nx + k*nx*ny
 */
__device__ __forceinline__
int W_idx(int i, int j, int k, int nx, int ny) {
    return i + j * nx + k * nx * ny;
}

/**
 * @brief Padded merge_id: iz*(ny+1)*(nx+1) + iy*(nx+1) + ix
 * All components share this index space of size (nx+1)*(ny+1)*(nz+1).
 */
__device__ __forceinline__
int padded_idx(int ix, int iy, int iz, int nx, int ny) {
    return iz * (ny + 1) * (nx + 1) + iy * (nx + 1) + ix;
}

// BC type constants (matching legacy)
constexpr uint8_t BC_NEUMANN = 0;
constexpr uint8_t BC_PERIODIC = 1;
constexpr uint8_t BC_DIRICHLET = 2;

// Convert BCType enum to legacy-compatible int (host+device)
__host__ __device__ __forceinline__
uint8_t bc_to_int(BCType t) {
    switch(t) {
        case BCType::Neumann:   return BC_NEUMANN;
        case BCType::Periodic:  return BC_PERIODIC;
        case BCType::Dirichlet: return BC_DIRICHLET;
        default:                return BC_NEUMANN;
    }
}

// ============================================================================
// Interior kernels - pure bulk computation, NO boundary logic
// ============================================================================

/**
 * @brief Compute U-velocity at interior faces (i = 1 to nx-1)
 * 
 * Face i is between cell (i-1,j,k) and cell (i,j,k)
 * U[i,j,k] = -K_harmonic * (H[i,j,k] - H[i-1,j,k]) / dx
 */
__global__ void kernel_U_interior(
    real* __restrict__ U,
    const real* __restrict__ H,
    const real* __restrict__ K,
    int nx, int ny, int nz,
    real dx)
{
    // Thread maps to face (i,j,k) where i = 1 + threadIdx.x + blockIdx.x*blockDim.x
    int face_i = 1 + threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.z + blockIdx.z * blockDim.z;
    
    // Interior faces: i = 1 to nx-1 (faces between adjacent cells)
    if (face_i > nx - 1 || j >= ny || k >= nz) return;
    
    // Cells on either side of face
    int c_left  = cell_idx(face_i - 1, j, k, nx, ny);  // cell i-1
    int c_right = cell_idx(face_i,     j, k, nx, ny);  // cell i
    
    real H_left  = H[c_left];
    real H_right = H[c_right];
    real K_left  = K[c_left];
    real K_right = K[c_right];
    
    // Darcy: u = -K_eff * (H_right - H_left) / dx
    real K_eff = harmonic_mean(K_left, K_right);
    real u = -K_eff * (H_right - H_left) / dx;
    
    U[U_idx(face_i, j, k, nx, ny)] = u;
}

/**
 * @brief Compute V-velocity at interior faces (j = 1 to ny-1)
 */
__global__ void kernel_V_interior(
    real* __restrict__ V,
    const real* __restrict__ H,
    const real* __restrict__ K,
    int nx, int ny, int nz,
    real dy)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int face_j = 1 + threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.z + blockIdx.z * blockDim.z;
    
    if (i >= nx || face_j > ny - 1 || k >= nz) return;
    
    int c_south = cell_idx(i, face_j - 1, k, nx, ny);
    int c_north = cell_idx(i, face_j,     k, nx, ny);
    
    real H_south = H[c_south];
    real H_north = H[c_north];
    real K_south = K[c_south];
    real K_north = K[c_north];
    
    real K_eff = harmonic_mean(K_south, K_north);
    real v = -K_eff * (H_north - H_south) / dy;
    
    V[V_idx(i, face_j, k, nx, ny)] = v;
}

/**
 * @brief Compute W-velocity at interior faces (k = 1 to nz-1)
 */
__global__ void kernel_W_interior(
    real* __restrict__ W,
    const real* __restrict__ H,
    const real* __restrict__ K,
    int nx, int ny, int nz,
    real dz)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int face_k = 1 + threadIdx.z + blockIdx.z * blockDim.z;
    
    if (i >= nx || j >= ny || face_k > nz - 1) return;
    
    int c_bottom = cell_idx(i, j, face_k - 1, nx, ny);
    int c_top    = cell_idx(i, j, face_k,     nx, ny);
    
    real H_bottom = H[c_bottom];
    real H_top    = H[c_top];
    real K_bottom = K[c_bottom];
    real K_top    = K[c_top];
    
    real K_eff = harmonic_mean(K_bottom, K_top);
    real w = -K_eff * (H_top - H_bottom) / dz;
    
    W[W_idx(i, j, face_k, nx, ny)] = w;
}

// ============================================================================
// Boundary kernels - one kernel per face
// ============================================================================

/**
 * @brief U-velocity at WEST face (i=0): face between domain boundary and cell(0,j,k)
 * 
 * Neumann: U[0,j,k] = 0
 * Dirichlet: U[0,j,k] = -K_cell * (H_cell - H_bc) / (dx/2)
 * Periodic: U[0,j,k] = -K_eff * (H[0,j,k] - H[nx-1,j,k]) / dx  (wrap)
 */
__global__ void kernel_U_west(
    real* __restrict__ U,
    const real* __restrict__ H,
    const real* __restrict__ K,
    int nx, int ny, int nz,
    real dx,
    uint8_t bc_type,
    real H_bc)
{
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    int k = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (j >= ny || k >= nz) return;
    
    int c_inside = cell_idx(0, j, k, nx, ny);  // first cell
    real H_inside = H[c_inside];
    real K_inside = K[c_inside];
    
    real u = 0.0;  // default: Neumann
    
    if (bc_type == BC_DIRICHLET) {
        // One-sided: distance is dx/2 from cell center to face
        // Gradient: (H_inside - H_bc) / (dx/2) pointing into domain
        // Darcy: u = -K * dH/dx
        u = -K_inside * (H_inside - H_bc) / (dx * 0.5);
    }
    else if (bc_type == BC_PERIODIC) {
        // Wrap: neighbor is cell (nx-1, j, k)
        int c_wrap = cell_idx(nx - 1, j, k, nx, ny);
        real H_wrap = H[c_wrap];
        real K_wrap = K[c_wrap];
        real K_eff = harmonic_mean(K_wrap, K_inside);
        // Gradient from wrap to inside (conceptually wrap is "to the left")
        u = -K_eff * (H_inside - H_wrap) / dx;
    }
    // else: Neumann -> u = 0
    
    U[U_idx(0, j, k, nx, ny)] = u;
}

/**
 * @brief U-velocity at EAST face (i=nx): face between cell(nx-1,j,k) and domain boundary
 */
__global__ void kernel_U_east(
    real* __restrict__ U,
    const real* __restrict__ H,
    const real* __restrict__ K,
    int nx, int ny, int nz,
    real dx,
    uint8_t bc_type,
    real H_bc)
{
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    int k = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (j >= ny || k >= nz) return;
    
    int c_inside = cell_idx(nx - 1, j, k, nx, ny);  // last cell
    real H_inside = H[c_inside];
    real K_inside = K[c_inside];
    
    real u = 0.0;
    
    if (bc_type == BC_DIRICHLET) {
        // Gradient: (H_bc - H_inside) / (dx/2)
        u = -K_inside * (H_bc - H_inside) / (dx * 0.5);
    }
    else if (bc_type == BC_PERIODIC) {
        // Wrap: neighbor is cell (0, j, k)
        int c_wrap = cell_idx(0, j, k, nx, ny);
        real H_wrap = H[c_wrap];
        real K_wrap = K[c_wrap];
        real K_eff = harmonic_mean(K_inside, K_wrap);
        u = -K_eff * (H_wrap - H_inside) / dx;
    }
    
    U[U_idx(nx, j, k, nx, ny)] = u;
}

/**
 * @brief V-velocity at SOUTH face (j=0)
 */
__global__ void kernel_V_south(
    real* __restrict__ V,
    const real* __restrict__ H,
    const real* __restrict__ K,
    int nx, int ny, int nz,
    real dy,
    uint8_t bc_type,
    real H_bc)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int k = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (i >= nx || k >= nz) return;
    
    int c_inside = cell_idx(i, 0, k, nx, ny);
    real H_inside = H[c_inside];
    real K_inside = K[c_inside];
    
    real v = 0.0;
    
    if (bc_type == BC_DIRICHLET) {
        v = -K_inside * (H_inside - H_bc) / (dy * 0.5);
    }
    else if (bc_type == BC_PERIODIC) {
        int c_wrap = cell_idx(i, ny - 1, k, nx, ny);
        real H_wrap = H[c_wrap];
        real K_wrap = K[c_wrap];
        real K_eff = harmonic_mean(K_wrap, K_inside);
        v = -K_eff * (H_inside - H_wrap) / dy;
    }
    
    V[V_idx(i, 0, k, nx, ny)] = v;
}

/**
 * @brief V-velocity at NORTH face (j=ny)
 */
__global__ void kernel_V_north(
    real* __restrict__ V,
    const real* __restrict__ H,
    const real* __restrict__ K,
    int nx, int ny, int nz,
    real dy,
    uint8_t bc_type,
    real H_bc)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int k = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (i >= nx || k >= nz) return;
    
    int c_inside = cell_idx(i, ny - 1, k, nx, ny);
    real H_inside = H[c_inside];
    real K_inside = K[c_inside];
    
    real v = 0.0;
    
    if (bc_type == BC_DIRICHLET) {
        v = -K_inside * (H_bc - H_inside) / (dy * 0.5);
    }
    else if (bc_type == BC_PERIODIC) {
        int c_wrap = cell_idx(i, 0, k, nx, ny);
        real H_wrap = H[c_wrap];
        real K_wrap = K[c_wrap];
        real K_eff = harmonic_mean(K_inside, K_wrap);
        v = -K_eff * (H_wrap - H_inside) / dy;
    }
    
    V[V_idx(i, ny, k, nx, ny)] = v;
}

/**
 * @brief W-velocity at BOTTOM face (k=0)
 */
__global__ void kernel_W_bottom(
    real* __restrict__ W,
    const real* __restrict__ H,
    const real* __restrict__ K,
    int nx, int ny, int nz,
    real dz,
    uint8_t bc_type,
    real H_bc)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (i >= nx || j >= ny) return;
    
    int c_inside = cell_idx(i, j, 0, nx, ny);
    real H_inside = H[c_inside];
    real K_inside = K[c_inside];
    
    real w = 0.0;
    
    if (bc_type == BC_DIRICHLET) {
        w = -K_inside * (H_inside - H_bc) / (dz * 0.5);
    }
    else if (bc_type == BC_PERIODIC) {
        int c_wrap = cell_idx(i, j, nz - 1, nx, ny);
        real H_wrap = H[c_wrap];
        real K_wrap = K[c_wrap];
        real K_eff = harmonic_mean(K_wrap, K_inside);
        w = -K_eff * (H_inside - H_wrap) / dz;
    }
    
    W[W_idx(i, j, 0, nx, ny)] = w;
}

/**
 * @brief W-velocity at TOP face (k=nz)
 */
__global__ void kernel_W_top(
    real* __restrict__ W,
    const real* __restrict__ H,
    const real* __restrict__ K,
    int nx, int ny, int nz,
    real dz,
    uint8_t bc_type,
    real H_bc)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (i >= nx || j >= ny) return;
    
    int c_inside = cell_idx(i, j, nz - 1, nx, ny);
    real H_inside = H[c_inside];
    real K_inside = K[c_inside];
    
    real w = 0.0;
    
    if (bc_type == BC_DIRICHLET) {
        w = -K_inside * (H_bc - H_inside) / (dz * 0.5);
    }
    else if (bc_type == BC_PERIODIC) {
        int c_wrap = cell_idx(i, j, 0, nx, ny);
        real H_wrap = H[c_wrap];
        real K_wrap = K[c_wrap];
        real K_eff = harmonic_mean(K_inside, K_wrap);
        w = -K_eff * (H_wrap - H_inside) / dz;
    }
    
    W[W_idx(i, j, nz, nx, ny)] = w;
}

// ============================================================================
// Padded facefield kernels — same physics, padded_idx layout
// ============================================================================

// --- Interior ---

__global__ void kernel_U_interior_padded(
    real* __restrict__ U,
    const real* __restrict__ H,
    const real* __restrict__ K,
    int nx, int ny, int nz,
    real dx)
{
    int face_i = 1 + threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.z + blockIdx.z * blockDim.z;
    if (face_i > nx - 1 || j >= ny || k >= nz) return;

    int c_left  = cell_idx(face_i - 1, j, k, nx, ny);
    int c_right = cell_idx(face_i,     j, k, nx, ny);
    real K_eff = harmonic_mean(K[c_left], K[c_right]);
    real u = -K_eff * (H[c_right] - H[c_left]) / dx;
    U[padded_idx(face_i, j, k, nx, ny)] = u;
}

__global__ void kernel_V_interior_padded(
    real* __restrict__ V,
    const real* __restrict__ H,
    const real* __restrict__ K,
    int nx, int ny, int nz,
    real dy)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int face_j = 1 + threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.z + blockIdx.z * blockDim.z;
    if (i >= nx || face_j > ny - 1 || k >= nz) return;

    int c_south = cell_idx(i, face_j - 1, k, nx, ny);
    int c_north = cell_idx(i, face_j,     k, nx, ny);
    real K_eff = harmonic_mean(K[c_south], K[c_north]);
    real v = -K_eff * (H[c_north] - H[c_south]) / dy;
    V[padded_idx(i, face_j, k, nx, ny)] = v;
}

__global__ void kernel_W_interior_padded(
    real* __restrict__ W,
    const real* __restrict__ H,
    const real* __restrict__ K,
    int nx, int ny, int nz,
    real dz)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int face_k = 1 + threadIdx.z + blockIdx.z * blockDim.z;
    if (i >= nx || j >= ny || face_k > nz - 1) return;

    int c_bottom = cell_idx(i, j, face_k - 1, nx, ny);
    int c_top    = cell_idx(i, j, face_k,     nx, ny);
    real K_eff = harmonic_mean(K[c_bottom], K[c_top]);
    real w = -K_eff * (H[c_top] - H[c_bottom]) / dz;
    W[padded_idx(i, j, face_k, nx, ny)] = w;
}

// --- Boundary (padded) ---

__global__ void kernel_U_west_padded(
    real* __restrict__ U,
    const real* __restrict__ H,
    const real* __restrict__ K,
    int nx, int ny, int nz,
    real dx, uint8_t bc_type, real H_bc)
{
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    int k = threadIdx.y + blockIdx.y * blockDim.y;
    if (j >= ny || k >= nz) return;

    int c_in = cell_idx(0, j, k, nx, ny);
    real u = 0.0;
    if (bc_type == BC_DIRICHLET)
        u = -K[c_in] * (H[c_in] - H_bc) / (dx * 0.5);
    else if (bc_type == BC_PERIODIC) {
        int c_w = cell_idx(nx - 1, j, k, nx, ny);
        u = -harmonic_mean(K[c_w], K[c_in]) * (H[c_in] - H[c_w]) / dx;
    }
    U[padded_idx(0, j, k, nx, ny)] = u;
}

__global__ void kernel_U_east_padded(
    real* __restrict__ U,
    const real* __restrict__ H,
    const real* __restrict__ K,
    int nx, int ny, int nz,
    real dx, uint8_t bc_type, real H_bc)
{
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    int k = threadIdx.y + blockIdx.y * blockDim.y;
    if (j >= ny || k >= nz) return;

    int c_in = cell_idx(nx - 1, j, k, nx, ny);
    real u = 0.0;
    if (bc_type == BC_DIRICHLET)
        u = -K[c_in] * (H_bc - H[c_in]) / (dx * 0.5);
    else if (bc_type == BC_PERIODIC) {
        int c_w = cell_idx(0, j, k, nx, ny);
        u = -harmonic_mean(K[c_in], K[c_w]) * (H[c_w] - H[c_in]) / dx;
    }
    U[padded_idx(nx, j, k, nx, ny)] = u;
}

__global__ void kernel_V_south_padded(
    real* __restrict__ V,
    const real* __restrict__ H,
    const real* __restrict__ K,
    int nx, int ny, int nz,
    real dy, uint8_t bc_type, real H_bc)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int k = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= nx || k >= nz) return;

    int c_in = cell_idx(i, 0, k, nx, ny);
    real v = 0.0;
    if (bc_type == BC_DIRICHLET)
        v = -K[c_in] * (H[c_in] - H_bc) / (dy * 0.5);
    else if (bc_type == BC_PERIODIC) {
        int c_w = cell_idx(i, ny - 1, k, nx, ny);
        v = -harmonic_mean(K[c_w], K[c_in]) * (H[c_in] - H[c_w]) / dy;
    }
    V[padded_idx(i, 0, k, nx, ny)] = v;
}

__global__ void kernel_V_north_padded(
    real* __restrict__ V,
    const real* __restrict__ H,
    const real* __restrict__ K,
    int nx, int ny, int nz,
    real dy, uint8_t bc_type, real H_bc)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int k = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= nx || k >= nz) return;

    int c_in = cell_idx(i, ny - 1, k, nx, ny);
    real v = 0.0;
    if (bc_type == BC_DIRICHLET)
        v = -K[c_in] * (H_bc - H[c_in]) / (dy * 0.5);
    else if (bc_type == BC_PERIODIC) {
        int c_w = cell_idx(i, 0, k, nx, ny);
        v = -harmonic_mean(K[c_in], K[c_w]) * (H[c_w] - H[c_in]) / dy;
    }
    V[padded_idx(i, ny, k, nx, ny)] = v;
}

__global__ void kernel_W_bottom_padded(
    real* __restrict__ W,
    const real* __restrict__ H,
    const real* __restrict__ K,
    int nx, int ny, int nz,
    real dz, uint8_t bc_type, real H_bc)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= nx || j >= ny) return;

    int c_in = cell_idx(i, j, 0, nx, ny);
    real w = 0.0;
    if (bc_type == BC_DIRICHLET)
        w = -K[c_in] * (H[c_in] - H_bc) / (dz * 0.5);
    else if (bc_type == BC_PERIODIC) {
        int c_w = cell_idx(i, j, nz - 1, nx, ny);
        w = -harmonic_mean(K[c_w], K[c_in]) * (H[c_in] - H[c_w]) / dz;
    }
    W[padded_idx(i, j, 0, nx, ny)] = w;
}

__global__ void kernel_W_top_padded(
    real* __restrict__ W,
    const real* __restrict__ H,
    const real* __restrict__ K,
    int nx, int ny, int nz,
    real dz, uint8_t bc_type, real H_bc)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= nx || j >= ny) return;

    int c_in = cell_idx(i, j, nz - 1, nx, ny);
    real w = 0.0;
    if (bc_type == BC_DIRICHLET)
        w = -K[c_in] * (H_bc - H[c_in]) / (dz * 0.5);
    else if (bc_type == BC_PERIODIC) {
        int c_w = cell_idx(i, j, 0, nx, ny);
        w = -harmonic_mean(K[c_in], K[c_w]) * (H[c_w] - H[c_in]) / dz;
    }
    W[padded_idx(i, j, nz, nx, ny)] = w;
}

// ============================================================================
// Reduction kernels for checksums
// ============================================================================

__global__ void kernel_sum_sq(const real* data, real* partial, int n)
{
    extern __shared__ real sdata[];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    real sum = 0.0;
    while (i < n) {
        real v = data[i];
        sum += v * v;
        i += blockDim.x * gridDim.x;
    }
    sdata[tid] = sum;
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        partial[blockIdx.x] = sdata[0];
    }
}

__global__ void kernel_check_nans(const real* data, int* has_nan, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    while (i < n) {
        if (isnan(data[i])) {
            atomicExch(has_nan, 1);
            return;
        }
        i += blockDim.x * gridDim.x;
    }
}

// ============================================================================
// Host orchestration
// ============================================================================

void compute_velocity_from_head(
    VelocityField& vel,
    const HeadField& head,
    const KField& K,
    const Grid3D& grid,
    const BCSpec& bc,
    CudaContext& ctx)
{
    int nx = grid.nx;
    int ny = grid.ny;
    int nz = grid.nz;
    real dx = grid.dx;
    real dy = grid.dy;
    real dz = grid.dz;
    
    // Get device pointers
    real* U = vel.U_ptr();
    real* V = vel.V_ptr();
    real* W = vel.W_ptr();
    const real* H = head.device_ptr();
    const real* K_ptr = K.device_ptr();
    
    // Convert BC types
    uint8_t bc_xmin = bc_to_int(bc.xmin.type);
    uint8_t bc_xmax = bc_to_int(bc.xmax.type);
    uint8_t bc_ymin = bc_to_int(bc.ymin.type);
    uint8_t bc_ymax = bc_to_int(bc.ymax.type);
    uint8_t bc_zmin = bc_to_int(bc.zmin.type);
    uint8_t bc_zmax = bc_to_int(bc.zmax.type);
    
    // ========================================================================
    // Launch interior kernels
    // ========================================================================
    
    // Block size for 3D kernels
    dim3 block(8, 8, 8);
    
    // U interior: faces i = 1 to nx-1
    {
        int num_faces_x = nx - 1;  // interior faces in x
        dim3 grid_U(
            (num_faces_x + block.x - 1) / block.x,
            (ny + block.y - 1) / block.y,
            (nz + block.z - 1) / block.z
        );
        kernel_U_interior<<<grid_U, block>>>(U, H, K_ptr, nx, ny, nz, dx);
        MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
    }
    
    // V interior: faces j = 1 to ny-1
    {
        int num_faces_y = ny - 1;
        dim3 grid_V(
            (nx + block.x - 1) / block.x,
            (num_faces_y + block.y - 1) / block.y,
            (nz + block.z - 1) / block.z
        );
        kernel_V_interior<<<grid_V, block>>>(V, H, K_ptr, nx, ny, nz, dy);
        MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
    }
    
    // W interior: faces k = 1 to nz-1
    {
        int num_faces_z = nz - 1;
        dim3 grid_W(
            (nx + block.x - 1) / block.x,
            (ny + block.y - 1) / block.y,
            (num_faces_z + block.z - 1) / block.z
        );
        kernel_W_interior<<<grid_W, block>>>(W, H, K_ptr, nx, ny, nz, dz);
        MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
    }
    
    // ========================================================================
    // Launch boundary kernels (6 faces)
    // ========================================================================
    
    dim3 block2D(16, 16);
    
    // U boundaries (west/east faces)
    {
        dim3 grid_yz((ny + block2D.x - 1) / block2D.x, (nz + block2D.y - 1) / block2D.y);
        kernel_U_west<<<grid_yz, block2D>>>(U, H, K_ptr, nx, ny, nz, dx, bc_xmin, bc.xmin.value);
        kernel_U_east<<<grid_yz, block2D>>>(U, H, K_ptr, nx, ny, nz, dx, bc_xmax, bc.xmax.value);
        MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
    }
    
    // V boundaries (south/north faces)
    {
        dim3 grid_xz((nx + block2D.x - 1) / block2D.x, (nz + block2D.y - 1) / block2D.y);
        kernel_V_south<<<grid_xz, block2D>>>(V, H, K_ptr, nx, ny, nz, dy, bc_ymin, bc.ymin.value);
        kernel_V_north<<<grid_xz, block2D>>>(V, H, K_ptr, nx, ny, nz, dy, bc_ymax, bc.ymax.value);
        MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
    }
    
    // W boundaries (bottom/top faces)
    {
        dim3 grid_xy((nx + block2D.x - 1) / block2D.x, (ny + block2D.y - 1) / block2D.y);
        kernel_W_bottom<<<grid_xy, block2D>>>(W, H, K_ptr, nx, ny, nz, dz, bc_zmin, bc.zmin.value);
        kernel_W_top<<<grid_xy, block2D>>>(W, H, K_ptr, nx, ny, nz, dz, bc_zmax, bc.zmax.value);
        MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
    }
    
    // Synchronize
    ctx.synchronize();
}

// ============================================================================
// Padded facefield host orchestration
// ============================================================================

void compute_velocity_from_head(
    PaddedVelocityField& vel,
    const HeadField& head,
    const KField& K,
    const Grid3D& grid,
    const BCSpec& bc,
    CudaContext& ctx)
{
    int nx = grid.nx;
    int ny = grid.ny;
    int nz = grid.nz;
    real dx = grid.dx;
    real dy = grid.dy;
    real dz = grid.dz;

    real* U = vel.U_ptr();
    real* V = vel.V_ptr();
    real* W = vel.W_ptr();
    const real* H = head.device_ptr();
    const real* K_ptr = K.device_ptr();

    uint8_t bc_xmin = bc_to_int(bc.xmin.type);
    uint8_t bc_xmax = bc_to_int(bc.xmax.type);
    uint8_t bc_ymin = bc_to_int(bc.ymin.type);
    uint8_t bc_ymax = bc_to_int(bc.ymax.type);
    uint8_t bc_zmin = bc_to_int(bc.zmin.type);
    uint8_t bc_zmax = bc_to_int(bc.zmax.type);

    // ========================================================================
    // Interior kernels (padded layout)
    // ========================================================================
    dim3 block(8, 8, 8);

    // U interior: faces i = 1..nx-1, j = 0..ny-1, k = 0..nz-1
    {
        int nf = nx - 1;
        dim3 g((nf + block.x - 1) / block.x,
               (ny + block.y - 1) / block.y,
               (nz + block.z - 1) / block.z);
        kernel_U_interior_padded<<<g, block>>>(U, H, K_ptr, nx, ny, nz, dx);
        MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
    }
    // V interior: i = 0..nx-1, faces j = 1..ny-1, k = 0..nz-1
    {
        int nf = ny - 1;
        dim3 g((nx + block.x - 1) / block.x,
               (nf + block.y - 1) / block.y,
               (nz + block.z - 1) / block.z);
        kernel_V_interior_padded<<<g, block>>>(V, H, K_ptr, nx, ny, nz, dy);
        MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
    }
    // W interior: i = 0..nx-1, j = 0..ny-1, faces k = 1..nz-1
    {
        int nf = nz - 1;
        dim3 g((nx + block.x - 1) / block.x,
               (ny + block.y - 1) / block.y,
               (nf + block.z - 1) / block.z);
        kernel_W_interior_padded<<<g, block>>>(W, H, K_ptr, nx, ny, nz, dz);
        MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
    }

    // ========================================================================
    // Boundary kernels (padded layout)
    // ========================================================================
    dim3 block2D(16, 16);

    // U boundaries
    {
        dim3 g((ny + block2D.x - 1) / block2D.x,
               (nz + block2D.y - 1) / block2D.y);
        kernel_U_west_padded<<<g, block2D>>>(U, H, K_ptr, nx, ny, nz, dx, bc_xmin, bc.xmin.value);
        kernel_U_east_padded<<<g, block2D>>>(U, H, K_ptr, nx, ny, nz, dx, bc_xmax, bc.xmax.value);
        MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
    }
    // V boundaries
    {
        dim3 g((nx + block2D.x - 1) / block2D.x,
               (nz + block2D.y - 1) / block2D.y);
        kernel_V_south_padded<<<g, block2D>>>(V, H, K_ptr, nx, ny, nz, dy, bc_ymin, bc.ymin.value);
        kernel_V_north_padded<<<g, block2D>>>(V, H, K_ptr, nx, ny, nz, dy, bc_ymax, bc.ymax.value);
        MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
    }
    // W boundaries
    {
        dim3 g((nx + block2D.x - 1) / block2D.x,
               (ny + block2D.y - 1) / block2D.y);
        kernel_W_bottom_padded<<<g, block2D>>>(W, H, K_ptr, nx, ny, nz, dz, bc_zmin, bc.zmin.value);
        kernel_W_top_padded<<<g, block2D>>>(W, H, K_ptr, nx, ny, nz, dz, bc_zmax, bc.zmax.value);
        MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
    }

    ctx.synchronize();
}

// ============================================================================
// Utility implementations
// ============================================================================

real compute_norm2(DeviceSpan<const real> data, CudaContext& ctx)
{
    int n = static_cast<int>(data.size());
    if (n == 0) return 0.0;
    
    const int block_size = 256;
    int num_blocks = std::min((n + block_size - 1) / block_size, 1024);
    
    // Allocate partial sums on device
    DeviceBuffer<real> partial(num_blocks);
    
    kernel_sum_sq<<<num_blocks, block_size, block_size * sizeof(real)>>>(
        data.data(), partial.data(), n);
    
    // Copy back and reduce on host
    std::vector<real> h_partial(num_blocks);
    cudaMemcpy(h_partial.data(), partial.data(), num_blocks * sizeof(real), cudaMemcpyDeviceToHost);
    ctx.synchronize();
    
    real sum = 0.0;
    for (int i = 0; i < num_blocks; ++i) {
        sum += h_partial[i];
    }
    
    return std::sqrt(sum);
}

bool check_no_nans(DeviceSpan<const real> data, CudaContext& ctx)
{
    int n = static_cast<int>(data.size());
    if (n == 0) return true;
    
    DeviceBuffer<int> has_nan(1);
    int zero = 0;
    cudaMemcpy(has_nan.data(), &zero, sizeof(int), cudaMemcpyHostToDevice);
    
    const int block_size = 256;
    int num_blocks = std::min((n + block_size - 1) / block_size, 1024);
    
    kernel_check_nans<<<num_blocks, block_size>>>(data.data(), has_nan.data(), n);
    
    int h_has_nan;
    cudaMemcpy(&h_has_nan, has_nan.data(), sizeof(int), cudaMemcpyDeviceToHost);
    ctx.synchronize();
    
    return h_has_nan == 0;
}

// Kernel for sum reduction
__global__ void kernel_sum(const real* data, real* partial, int n)
{
    extern __shared__ real sdata[];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    real sum = 0.0;
    while (i < n) {
        sum += data[i];
        i += blockDim.x * gridDim.x;
    }
    sdata[tid] = sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        partial[blockIdx.x] = sdata[0];
    }
}

// Kernel for computing 1/K (for harmonic mean calculation)
__global__ void kernel_sum_inv(const real* data, real* partial, int n)
{
    extern __shared__ real sdata[];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    real sum = 0.0;
    while (i < n) {
        sum += 1.0 / data[i];
        i += blockDim.x * gridDim.x;
    }
    sdata[tid] = sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        partial[blockIdx.x] = sdata[0];
    }
}

real compute_sum(DeviceSpan<const real> data, CudaContext& ctx)
{
    int n = static_cast<int>(data.size());
    if (n == 0) return 0.0;
    
    const int block_size = 256;
    int num_blocks = std::min((n + block_size - 1) / block_size, 1024);
    
    DeviceBuffer<real> partial(num_blocks);
    
    kernel_sum<<<num_blocks, block_size, block_size * sizeof(real)>>>(
        data.data(), partial.data(), n);
    
    std::vector<real> h_partial(num_blocks);
    cudaMemcpy(h_partial.data(), partial.data(), num_blocks * sizeof(real), cudaMemcpyDeviceToHost);
    ctx.synchronize();
    
    real sum = 0.0;
    for (int i = 0; i < num_blocks; ++i) {
        sum += h_partial[i];
    }
    
    return sum;
}

real compute_mean(DeviceSpan<const real> data, CudaContext& ctx)
{
    if (data.size() == 0) return 0.0;
    return compute_sum(data, ctx) / static_cast<real>(data.size());
}

// Compute harmonic mean of K field: n / sum(1/K)
real compute_harmonic_mean_K(DeviceSpan<const real> K, CudaContext& ctx)
{
    int n = static_cast<int>(K.size());
    if (n == 0) return 0.0;
    
    const int block_size = 256;
    int num_blocks = std::min((n + block_size - 1) / block_size, 1024);
    
    DeviceBuffer<real> partial(num_blocks);
    
    kernel_sum_inv<<<num_blocks, block_size, block_size * sizeof(real)>>>(
        K.data(), partial.data(), n);
    
    std::vector<real> h_partial(num_blocks);
    cudaMemcpy(h_partial.data(), partial.data(), num_blocks * sizeof(real), cudaMemcpyDeviceToHost);
    ctx.synchronize();
    
    real sum_inv = 0.0;
    for (int i = 0; i < num_blocks; ++i) {
        sum_inv += h_partial[i];
    }
    
    return static_cast<real>(n) / sum_inv;
}

void print_velocity_checksums(const VelocityField& vel, CudaContext& ctx)
{
    std::cout << "  Velocity checksums:\n";
    
    // U
    real norm_U = compute_norm2(vel.U_span(), ctx);
    bool ok_U = check_no_nans(vel.U_span(), ctx);
    std::cout << "    U: norm2 = " << norm_U << (ok_U ? " (no NaNs)" : " [HAS NaNs!]") << "\n";
    
    // V
    real norm_V = compute_norm2(vel.V_span(), ctx);
    bool ok_V = check_no_nans(vel.V_span(), ctx);
    std::cout << "    V: norm2 = " << norm_V << (ok_V ? " (no NaNs)" : " [HAS NaNs!]") << "\n";
    
    // W
    real norm_W = compute_norm2(vel.W_span(), ctx);
    bool ok_W = check_no_nans(vel.W_span(), ctx);
    std::cout << "    W: norm2 = " << norm_W << (ok_W ? " (no NaNs)" : " [HAS NaNs!]") << "\n";
    
    bool all_ok = ok_U && ok_V && ok_W;
    std::cout << "    Status: " << (all_ok ? "OK" : "ERROR - NaNs detected!") << "\n";
}

void verify_mean_velocity_darcy(
    const VelocityField& vel,
    const KField& K,
    const Grid3D& grid,
    const BCSpec& bc,
    CudaContext& ctx)
{
    std::cout << "\n  Darcy velocity verification:\n";
    
    // Get domain length: Lx = nx * dx
    real Lx = grid.nx * grid.dx;
    
    // Get BC values (assuming Dirichlet west-east)
    real H_west = bc.xmin.value;
    real H_east = bc.xmax.value;
    real dH = H_west - H_east;  // Head drop
    
    std::cout << "    H_west = " << H_west << ", H_east = " << H_east << "\n";
    std::cout << "    dH = " << dH << ", Lx = " << Lx << "\n";
    
    // Compute K statistics
    DeviceSpan<const real> K_span(K.device_ptr(), K.size());
    real K_harmonic = compute_harmonic_mean_K(K_span, ctx);
    real K_arithmetic = compute_mean(K_span, ctx);
    
    std::cout << "    K_harmonic = " << K_harmonic << "\n";
    std::cout << "    K_arithmetic = " << K_arithmetic << "\n";
    
    // Theoretical velocity using harmonic mean (exact for 1D steady flow)
    // u = -K_eff * dH/dx = K_eff * (H_west - H_east) / Lx
    real u_theory_harmonic = K_harmonic * dH / Lx;
    real u_theory_arithmetic = K_arithmetic * dH / Lx;
    
    std::cout << "    u_theory (K_harmonic) = " << u_theory_harmonic << "\n";
    std::cout << "    u_theory (K_arithmetic) = " << u_theory_arithmetic << "\n";
    
    // Compute mean U from the field
    real U_mean = compute_mean(vel.U_span(), ctx);
    std::cout << "    U_mean (computed) = " << U_mean << "\n";
    
    // Relative errors
    real rel_err_harmonic = std::abs(U_mean - u_theory_harmonic) / std::abs(u_theory_harmonic) * 100.0;
    real rel_err_arithmetic = std::abs(U_mean - u_theory_arithmetic) / std::abs(u_theory_arithmetic) * 100.0;
    
    std::cout << "    Relative error vs K_harmonic:   " << rel_err_harmonic << " %\n";
    std::cout << "    Relative error vs K_arithmetic: " << rel_err_arithmetic << " %\n";
    
    // For 1D steady Darcy flow with heterogeneous K, the effective K is:
    // - Harmonic mean for flow perpendicular to layers (series)
    // - Arithmetic mean for flow parallel to layers (parallel)
    // In 3D random field, it's somewhere between, often close to geometric mean
    std::cout << "\n    Note: For 3D random K, effective K is between harmonic and arithmetic.\n";
    std::cout << "    The computed mean velocity should be consistent with the solved head field.\n";
}

} // namespace physics
} // namespace macroflow3d
