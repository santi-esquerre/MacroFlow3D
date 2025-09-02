#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <curand_kernel.h>
#include <device_launch_parameters.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <omp.h>
#include <sstream>
#include <stdexcept>
#include <unistd.h>
#include <vector>

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>

#include "./par2/Geometry/CartesianGrid.cuh"
#include "./par2/Geometry/CellField.cuh"
#include "./par2/Geometry/CornerField.cuh"
#include "./par2/Geometry/FaceField.cuh"
#include "./par2/Geometry/Interpolation.cuh"
#include "./par2/Geometry/Point.cuh"
#include "./par2/Geometry/Vector.cuh"
#include "./par2/Particles/MoveParticle.cuh"
#include "./par2/Particles/PParticles.cuh"

#include "./header/MG_struct.h"
#include "./header/homogenization_permeability.h"
#include "./header/linear_operator.h"
#include "./header/routines_CCMG.h"
#include "./header/routines_solver_type.h"

#include <fstream>
#include <nlohmann/json.hpp>
#include <sstream>

#pragma region preamble

using namespace std;

#define neumann 0
#define periodic 1
#define dirichlet 2

#define wall 0   // equiv to neumann for ADV & DIFF
#define inlet 2  // equiv. to dirichlet for ADV & DIFF
#define outlet 3 // equiv. neumann for DIFF, and phib <- phiC for ADV

#define BC_type BC_WEST, BC_EAST, BC_SOUTH, BC_NORTH, BC_BOTTOM, BC_TOP
#define BC_value C_west, C_east, C_south, C_north, C_bottom, C_top
#define moments                                                                \
  dev_momento1x, dev_momento2x, dev_momento1y, dev_momento2y, dev_momento1z,   \
      dev_momento2z
#define common_arg Uf, Vf, Wf, Nx, Ny, Nz, A, V_dt, nuA_h

#define CUDA_CALL(x)                                                           \
  do {                                                                         \
    if ((x) != cudaSuccess) {                                                  \
      printf("Error at %s:%d\n", __FILE__, __LINE__);                          \
      return EXIT_FAILURE;                                                     \
    }                                                                          \
  } while (0)

inline void check_cuda_err(const cudaError_t &cerr) {
  if (cudaSuccess != cerr)
    std::runtime_error(cudaGetErrorString(cerr));
}

class Crono {
private:
  cudaEvent_t srt;
  cudaEvent_t end;

public:
  Crono() {
    check_cuda_err(cudaEventCreate(&this->srt));
    check_cuda_err(cudaEventCreate(&this->end));
  }

  void start() { check_cuda_err(cudaEventRecord(this->srt, 0)); }

  void finish(float &itime) {
    check_cuda_err(cudaEventRecord(this->end, 0));
    check_cuda_err(cudaEventSynchronize(this->end));
    // time in milisecond [ms]
    check_cuda_err(cudaEventElapsedTime(&itime, this->srt, this->end));
  }

  ~Crono() {
    check_cuda_err(cudaEventDestroy(this->srt));
    check_cuda_err(cudaEventDestroy(this->end));
  }
};

double norm_infty(double *X, int N, cublasHandle_t handle) {
  int *index;
  index = new int[1];
  cublasIdamax(handle, N, X, 1, index);
  (*index) = (*index) - 1;
  double *max;
  max = new double[1];
  cudaMemcpy(max, X + (*index), sizeof(double), cudaMemcpyDeviceToHost);
  return abs(*max);
}

void step_FE_TVD_with_moments(
    double *phi_out, double *phi_in, double *U, double *V, double *W, int Nx,
    int Ny, int Nz, double A, double V_dt, double nuA_h, int BC_WEST,
    int BC_EAST, int BC_SOUTH, int BC_NORTH, int BC_BOTTOM, int BC_TOP,
    double phi_WEST, double phi_EAST, double phi_SOUTH, double phi_NORTH,
    double phi_BOTTOM, double phi_TOP, dim3 grid, dim3 block, double h,
    float *momento1x, float *momento2x, float *momento1y, float *momento2y,
    float *momento1z, float *momento2z, float *C_float);

void step_FE_TVD(double *phi_out, double *phi_in, double *U, double *V,
                 double *W, int Nx, int Ny, int Nz, double A, double V_dt,
                 double nuA_h, int BC_WEST, int BC_EAST, int BC_SOUTH,
                 int BC_NORTH, int BC_BOTTOM, int BC_TOP, double phi_WEST,
                 double phi_EAST, double phi_SOUTH, double phi_NORTH,
                 double phi_BOTTOM, double phi_TOP, dim3 grid, dim3 block);

void compute_velocity_from_head(double *U, double *V, double *W, double *H,
                                double *K, int Nx, int Ny, int Nz, double h,
                                int BC_WEST, int BC_EAST, int BC_SOUTH,
                                int BC_NORTH, int BC_BOTTOM, int BC_TOP,
                                double H_WEST, double H_EAST, double H_SOUTH,
                                double H_NORTH, double H_BOTTOM, double H_TOP,
                                dim3 grid, dim3 block);

__global__ void setup_uniform_distrib(curandState *state, const int i_max);
__global__ void conductivity_kernel_3D(double *V1, double *V2, double *V3,
                                       double *a, double *b, const int i_max,
                                       double *K, const double lambda,
                                       const double h, const int Nx,
                                       const int Ny, const int Nz,
                                       const double sigma_f);
__global__ void random_kernel_3D(curandState *state, double *V1, double *V2,
                                 double *V3, double *a, double *b,
                                 const double lambda, const int i_max);
__global__ void random_kernel_3D_gauss(curandState *state, double *V1,
                                       double *V2, double *V3, double *a,
                                       double *b, const double lambda,
                                       const int i_max, const int k_m);
__global__ void
conductivity_kernel_3D_logK(double *V1, double *V2, double *V3, double *a,
                            double *b, const int i_max, double *K,
                            const double lambda, const double h, const int Nx,
                            const int Ny, const int Nz, const double sigma_f) {
  const int ix = threadIdx.x + blockIdx.x * blockDim.x;
  const int iy = threadIdx.y + blockIdx.y * blockDim.y;
  const int iz = threadIdx.z + blockIdx.z * blockDim.z;
  if (ix >= Nx || iy >= Ny || iz >= Nz)
    return;
  int in_idx = ix + iy * Nx + iz * Nx * Ny;
  double fx = 0.0, tmp;
  for (int i = 0; i < i_max; i++) {
    tmp = h * ((ix + 0.5) * V1[i] + (iy + 0.5) * V2[i] + (iz + 0.5) * V3[i]);
    fx += a[i] * sin(tmp) + b[i] * cos(tmp);
  }
  K[in_idx] = (sigma_f / pow((double)i_max, 0.5) * fx);
}
__global__ void compute_expK(double *K, const int Nx, const int Ny,
                             const int Nz) {
  const int ix = threadIdx.x + blockIdx.x * blockDim.x;
  const int iy = threadIdx.y + blockIdx.y * blockDim.y;
  const int iz = threadIdx.z + blockIdx.z * blockDim.z;
  if (ix >= Nx || iy >= Ny || iz >= Nz)
    return;
  int in_idx = ix + iy * Nx + iz * Nx * Ny;
  K[in_idx] = exp(K[in_idx]);
}

void print_vectorial_value_per_point3D(double *u, double *v, double *w,
                                       std::string nombre, int n, int Nx,
                                       int Ny, int Nz, double hx, double hy,
                                       double hz) {
  double *u_h, *v_h, *w_h;
  u_h = new double[(Nx + 1) * Ny * Nz];
  v_h = new double[Nx * (Ny + 1) * Nz];
  w_h = new double[Nx * Ny * (Nz + 1)];
  cudaMemcpy(u_h, u, (Nx + 1) * Ny * Nz * sizeof(double),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(v_h, v, Nx * (Ny + 1) * Nz * sizeof(double),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(w_h, w, Nx * Ny * (Nz + 1) * sizeof(double),
             cudaMemcpyDeviceToHost);

  stringstream filename;
  filename << nombre << "_POINT_DATA_" << n << ".vtk";
  fstream file_salida(filename.str().c_str(),
                      std::fstream::trunc | std::fstream::out);
  file_salida << fixed;
  file_salida << "# vtk DataFile Version 2.0" << endl;
  file_salida << "Comment goes here" << endl;
  file_salida << "ASCII" << endl << endl;
  file_salida << "DATASET STRUCTURED_POINTS" << endl;
  file_salida << "DIMENSIONS " << Nx << " " << Ny << " " << Nz << endl << endl;
  file_salida << "ORIGIN " << hx / 2.0 << " " << hy / 2.0 << " " << hz / 2.0
              << endl;
  file_salida << "SPACING " << hx << " " << hy << " " << hz << endl << endl;

  file_salida << "POINT_DATA " << Nx * Ny * Nz << endl;

  file_salida << "VECTORS velocidad float" << endl;
  // file_salida<<"LOOKUP_TABLE default"<<endl<<endl;;
  for (int k = 0; k < Nz; ++k) {
    for (int j = 0; j < Ny; ++j) {
      for (int i = 0; i < Nx; ++i) {
        file_salida << u_h[i + 1 + j * (Nx + 1) + k * (Nx + 1) * Ny] << " "
                    << v_h[i + (j + 1) * Nx + k * Nx * (Ny + 1)] << " "
                    << w_h[i + j * Nx + (k + 1) * Nx * Ny] << " ";
      }
      file_salida << endl;
    }
    file_salida << endl;
  }
  file_salida.close();
  delete[] u_h;
  delete[] v_h;
  delete[] w_h;
}

void print_value_per_point3D(double *vector, const char *name, int n, int Nx,
                             int Ny, int Nz, double h) {
  double *vector_h = new double[Nx * Ny * Nz];
  cudaMemcpy(vector_h, vector, Nx * Ny * Nz * sizeof(double),
             cudaMemcpyDeviceToHost);
  char filename[256];
  snprintf(filename, sizeof(filename), "%s_%d.vtk", name, n);
  FILE *file_salida = fopen(filename, "w");
  if (!file_salida) {
    fprintf(stderr, "Error: no se pudo abrir el archivo %s para escritura.\n",
            filename);
    delete[] vector_h;
    return;
  }
  fprintf(file_salida, "# vtk DataFile Version 2.0\n");
  fprintf(file_salida, "Comment goes here\n");
  fprintf(file_salida, "ASCII\n\n");
  fprintf(file_salida, "DATASET STRUCTURED_POINTS\n");
  fprintf(file_salida, "DIMENSIONS %d %d %d\n\n", Nx, Ny, Nz);
  fprintf(file_salida, "ORIGIN %.6f %.6f %.6f\n", h / 2.0, h / 2.0, h / 2.0);
  fprintf(file_salida, "SPACING %.6f %.6f %.6f\n\n", h, h, h);
  fprintf(file_salida, "POINT_DATA %d\n", Nx * Ny * Nz);
  fprintf(file_salida, "SCALARS %s float\n", name);
  fprintf(file_salida, "LOOKUP_TABLE default\n\n");
  for (int k = 0; k < Nz; ++k) {
    for (int j = 0; j < Ny; ++j) {
      for (int i = 0; i < Nx; ++i) {
        fprintf(file_salida, "%.6f ", vector_h[i + j * Nx + k * Nx * Ny]);
      }
      fprintf(file_salida, "\n");
    }
    fprintf(file_salida, "\n");
  }
  fclose(file_salida);
  delete[] vector_h;
}

void stencil_head(double *H_output, const double *H_input, const double *K,
                  int Nx, int Ny, int Nz, double A, double h, int BCbottom,
                  int BCtop, int BCsouth, int BCnorth, int BCwest, int BCeast,
                  bool pin1stCell, dim3 grid, dim3 block);

class laplacianHead : public Matrix_t {
private:
  double *K;
  int Nx, Ny, Nz;
  double h, A;
  int BCbottom, BCtop, BCsouth, BCnorth, BCwest, BCeast;
  dim3 grid, block;
  bool pin1stCell;

public:
  // constructor
  laplacianHead(double *K, int Nx, int Ny, int Nz, double A, double h,
                int BCbottom, int BCtop, int BCsouth, int BCnorth, int BCwest,
                int BCeast, bool pin1stCell, dim3 grid, dim3 block)
      : K(K), Nx(Nx), Ny(Ny), Nz(Nz), A(A), h(h), BCbottom(BCbottom),
        BCtop(BCtop), BCsouth(BCsouth), BCnorth(BCnorth), BCwest(BCwest),
        BCeast(BCeast), pin1stCell(pin1stCell), grid(grid), block(block){};

  void stencil(double *output, double *input) {
    stencil_head(output, input, K, Nx, Ny, Nz, A, h, BCbottom, BCtop, BCsouth,
                 BCnorth, BCwest, BCeast, pin1stCell, grid, block);
  }
};

class laplacianHeadCoarse : public Matrix_t {
private:
  double *K;
  int Nx, Ny, Nz;
  double h, A;
  int BCbottom, BCtop, BCsouth, BCnorth, BCwest, BCeast;
  dim3 grid, block;
  bool pin1stCell;

public:
  // constructor
  laplacianHeadCoarse(double *K, int Nx, int Ny, int Nz, double A, double h,
                      int BCbottom, int BCtop, int BCsouth, int BCnorth,
                      int BCwest, int BCeast, bool pin1stCell, dim3 grid,
                      dim3 block)
      : K(K), Nx(Nx), Ny(Ny), Nz(Nz), A(A), h(h), BCbottom(BCbottom),
        BCtop(BCtop), BCsouth(BCsouth), BCnorth(BCnorth), BCwest(BCwest),
        BCeast(BCeast), pin1stCell(pin1stCell), grid(grid), block(block){};

  void stencil(double *output, double *input) {
    stencil_head(output, input, K, Nx, Ny, Nz, A, h, BCbottom, BCtop, BCsouth,
                 BCnorth, BCwest, BCeast, pin1stCell, grid, block);
  }
};

class MGprecond2 : public Matrix_t {
private:
  int Nx, Ny, Nz;
  int BCbottom, BCtop, BCsouth, BCnorth, BCwest, BCeast;
  bool pin1stCell;
  dim3 *grid, *block;
  MG_levels MG;
  cublasHandle_t handle;
  int ratioX, ratioY, ratioZ;
  double Ly;
  double **e, **r, **rr, **K;
  Matrix_t &M;
  Matrix_t &precond;
  blas_t &BLAS;
  double *aux, *aux2;

public:
  MGprecond2(int Nx, int Ny, int Nz, int BCbottom, int BCtop, int BCsouth,
             int BCnorth, int BCwest, int BCeast, bool pin1stCell, dim3 *grid,
             dim3 *block, double Ly, int ratioX, int ratioY, int ratioZ,
             cublasHandle_t handle, double **e, double **r, double **rr,
             double **K, MG_levels MG, Matrix_t &M, Matrix_t &precond,
             blas_t &BLAS, double *aux, double *aux2)
      : Nx(Nx), Ny(Ny), Nz(Nz), BCbottom(BCbottom), BCtop(BCtop),
        BCsouth(BCsouth), BCnorth(BCnorth), BCwest(BCwest), BCeast(BCeast),
        pin1stCell(pin1stCell), grid(grid), block(block), Ly(Ly),
        ratioX(ratioX), ratioY(ratioY), ratioZ(ratioZ), handle(handle), e(e),
        r(r), rr(rr), K(K), MG(MG), M(M), precond(precond), BLAS(BLAS),
        aux(aux), aux2(aux2){};
  void stencil(double *output, double *input) {
    cudaMemset(output, 0, sizeof(double) * Nx * Ny * Nz);
    Precond_CCMG_Vcycle2(output, input, e, r, rr, K, grid, block, BCbottom,
                         BCtop, BCsouth, BCnorth, BCwest, BCeast, pin1stCell,
                         Nx, Ny, Nz, MG, handle, ratioX, ratioY, ratioZ, Ly, M,
                         precond, BLAS, aux, aux2);
  }
};

void RHS_head(double *RHS, const double *K, int Nx, int Ny, int Nz, double A,
              double h, int BCbottom, int BCtop, int BCsouth, int BCnorth,
              int BCwest, int BCeast, double Hbottom, double Htop,
              double Hsouth, double Hnorth, double Hwest, double Heast,
              dim3 grid, dim3 block);

#define SETUP_BLOCK_GRID3D(n)                                                  \
  dim3 block##n(n, n, n);                                                      \
  dim3 grid##n(Nx / block##n.x, Ny / block##n.y, Nz / block##n.z);

#define SETUP_GRID_BLOCK_MG(MG, nx_n, ny_n, nz_n)                              \
  dim3 *_grid = new dim3[MG.L];                                                \
  dim3 *_block = new dim3[MG.L];                                               \
  int cpd; /* cells per direction */                                           \
  for (int i = 0; i < MG.L; ++i) {                                             \
    cpd = pow(2, MG.L - 1 - i);                                                \
    _block[MG.L - 1 - i].x = (cpd * nx_n < 32) ? cpd * nx_n : 32;              \
    _block[MG.L - 1 - i].y = (cpd * ny_n < 32) ? cpd * ny_n : 32;              \
    _block[MG.L - 1 - i].z = (cpd * nz_n < 32) ? cpd * nz_n : 32;              \
                                                                               \
    int NumBlocksx = (cpd * nx_n) / _block[MG.L - 1 - i].x;                    \
    int NumBlocksy = (cpd * ny_n) / _block[MG.L - 1 - i].y;                    \
    int NumBlocksz = (cpd * nz_n) / _block[MG.L - 1 - i].z;                    \
    if ((cpd * nx_n) % _block[MG.L - 1 - i].x)                                 \
      NumBlocksx++;                                                            \
    if ((cpd * ny_n) % _block[MG.L - 1 - i].y)                                 \
      NumBlocksy++;                                                            \
    if ((cpd * nz_n) % _block[MG.L - 1 - i].z)                                 \
      NumBlocksz++;                                                            \
    _grid[MG.L - 1 - i].x = NumBlocksx;                                        \
    _grid[MG.L - 1 - i].y = NumBlocksy;                                        \
    _grid[MG.L - 1 - i].z = NumBlocksz;                                        \
  }

#define ALLOCATE_MG_STRUCTURE_MEMORY(MG, nx_n, ny_n, nz_n)                     \
  /* Declare pointers to arrays for MG levels */                               \
  double **_r, **_e, **_rr;                                                    \
  double **_K;                                                                 \
  /* Allocate memory for the pointers in the host */                           \
  _r = (double **)malloc((MG.L - 1) * sizeof(double *));                       \
  _e = (double **)malloc((MG.L - 1) * sizeof(double *));                       \
  _rr = (double **)malloc(MG.L * sizeof(double *));                            \
  _K = (double **)malloc(MG.L * sizeof(double *));                             \
                                                                               \
  /* Allocate memory for each level on the GPU */                              \
  for (int i = 0; i < MG.L - 1; ++i) {                                         \
    int N = pow(2, i);                                                         \
    cudaMalloc(&_e[i], sizeof(double) * N * N * N * nx_n * ny_n * nz_n);       \
    cudaMalloc(&_r[i], sizeof(double) * N * N * N * nx_n * ny_n * nz_n);       \
  }                                                                            \
                                                                               \
  /* Allocate memory for rr at all levels on the GPU */                        \
  for (int i = 0; i < MG.L; ++i) {                                             \
    int N = pow(2, i);                                                         \
    cudaMalloc(&_rr[i], sizeof(double) * N * N * N * nx_n * ny_n * nz_n);      \
    cudaMalloc(&_K[i], sizeof(double) * N * N * N * nx_n * ny_n * nz_n);       \
  }

#define INIT_CUBLAS_ENVIRONMENT                                                \
  cublasHandle_t handle;                                                       \
  cublasCreate(&handle);                                                       \
  cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);                    \
  cublasHandle_t handle_host;                                                  \
  cublasCreate(&handle_host);                                                  \
  cublasSetPointerMode(handle_host, CUBLAS_POINTER_MODE_HOST);

#define CUDA_ALLOCATE_VECTOR(type, size, name)                                 \
  type *name;                                                                  \
  cudaMalloc((void **)&name, sizeof(type) * size);

/*======================================================================
  TO_DEVICE_VECTOR(type, ptr, size, name)

  - type : tipo de dato (float, double, …)
  - ptr  : puntero GPU devuelto por cudaMalloc (ej. Uf)
  - size : número de elementos
  - name : identificador para el nuevo thrust::device_vector

  Efectos:
    • crea  thrust::device_vector<type>  llamado ‹name› copiando [ptr, ptr+size)
    • libera el bloque original (cudaFree)
    • hace  ptr = name.data()  para mantener compatibilidad

  Ejemplo de uso:
      CUDA_ALLOCATE_VECTOR(double, n, Uf);
      ...  // llenamos Uf
      TO_DEVICE_VECTOR(double, Uf, n, datax);

      // ahora:
      //   datax  es un device_vector<double>
      //   Uf     apunta al almacenamiento interno de datax
 ======================================================================*/
#define TO_DEVICE_VECTOR(type, ptr, size, name)                                \
  thrust::device_vector<type> name(thrust::device_pointer_cast(ptr),           \
                                   thrust::device_pointer_cast(ptr) +          \
                                       static_cast<std::size_t>(size));        \
  cudaFree(ptr);                                                               \
  ptr = thrust::raw_pointer_cast(name.data())

struct square {
  __host__ __device__ double operator()(double a) const { return a * a; }
};
struct abs_val {
  __host__ __device__ double operator()(double x) const {
    return x < 0.0 ? -x : x;
  }
};

// ---- Conversion kernels: compact → cúbico ----
__global__ void convertU(const double *__restrict__ Uc,
                         double *__restrict__ Ucube,
                         par2::grid::Grid<double> g) {
  int ix = threadIdx.x + blockIdx.x * blockDim.x;
  int iy = threadIdx.y + blockIdx.y * blockDim.y;
  int iz = threadIdx.z + blockIdx.z * blockDim.z;
  if (ix > g.nx + 1 || iy >= g.ny || iz >= g.nz)
    return;
  int id_comp = ix + iy * (g.nx + 1) + iz * (g.nx + 1) * g.ny;
  int id_cube = par2::facefield::mergeId(g, ix + 1, iy, iz);
  Ucube[id_cube] = Uc[id_comp];
}

__global__ void convertV(const double *__restrict__ Vc,
                         double *__restrict__ Vcube,
                         par2::grid::Grid<double> g) {
  int ix = threadIdx.x + blockIdx.x * blockDim.x;
  int iy = threadIdx.y + blockIdx.y * blockDim.y;
  int iz = threadIdx.z + blockIdx.z * blockDim.z;
  if (ix >= g.nx || iy > g.ny + 1 || iz >= g.nz)
    return;
  int id_comp = ix + iy * g.nx + iz * g.nx * (g.ny + 1);
  int id_cube = par2::facefield::mergeId(g, ix, iy + 1, iz);
  Vcube[id_cube] = Vc[id_comp];
}

__global__ void convertW(const double *__restrict__ Wc,
                         double *__restrict__ Wcube,
                         par2::grid::Grid<double> g) {
  int ix = threadIdx.x + blockIdx.x * blockDim.x;
  int iy = threadIdx.y + blockIdx.y * blockDim.y;
  int iz = threadIdx.z + blockIdx.z * blockDim.z;
  if (ix >= g.nx || iy >= g.ny || iz > g.nz + 1)
    return;
  int id_comp = ix + iy * g.nx + iz * g.nx * g.ny;
  int id_cube = par2::facefield::mergeId(g, ix, iy, iz + 1);
  Wcube[id_cube] = Wc[id_comp];
}

// Scale velocity vectors by dividing by (h*h)
struct TimesH2 {
  double factor;
  __host__ __device__ TimesH2(double h) : factor(h * h) {}
  __host__ __device__ double operator()(const double &x) const {
    return x * factor;
  }
};

#define SETUP_BLOCK_GRID_RANDOM_KERNEL(blockname, gridname, Nx, Ny, Nz)        \
  dim3 blockname(32, 16, 2);                                                   \
  int NumBlocksx = (Nx + blockname.x - 1) / blockname.x;                       \
  int NumBlocksy = (Ny + blockname.y - 1) / blockname.y;                       \
  int NumBlocksz = (Nz + blockname.z - 1) / blockname.z;                       \
  dim3 gridname(NumBlocksx, NumBlocksy, NumBlocksz);

// Función auxiliar para liberar un solo puntero
inline void cudaFreeHelper(void *ptr) {
  if (ptr)
    cudaFree(ptr);
}

// Función principal para liberar múltiples punteros
template <typename... Args> void cudaFreeMultiple(Args *...ptrs) {
  int dummy[] = {(cudaFreeHelper(ptrs), 0)...};
  (void)dummy; // Evitar advertencias de variable no utilizada
}

// #define COMPUTE_KII(KiiX, KiiY, KiiZ, C) \
// cublasSdot(handle, Nx*Ny*Nz, dev_momento1x, 1, dev_onesN , 1, M1X); \
// cublasSdot(handle, Nx*Ny*Nz, dev_momento2x, 1, dev_onesN , 1, M2X); \
// cublasSdot(handle, Nx*Ny*Nz, dev_momento1y, 1, dev_onesN , 1, M1Y); \
// cublasSdot(handle, Nx*Ny*Nz, dev_momento2y, 1, dev_onesN , 1, M2Y); \
// cublasSdot(handle, Nx*Ny*Nz, dev_momento1z, 1, dev_onesN , 1, M1Z); \
// cublasSdot(handle, Nx*Ny*Nz, dev_momento2z, 1, dev_onesN , 1, M2Z); \
// cublasSdot(handle, Nx*Ny*Nz, C, 1, dev_onesN , 1, integralC); \
// cudaDeviceSynchronize(); \
// cudaMemcpy(host_M1X, M1X, sizeof(float), cudaMemcpyDeviceToHost); \
// cudaMemcpy(host_M2X, M2X, sizeof(float), cudaMemcpyDeviceToHost); \
// cudaMemcpy(host_M1Y, M1Y, sizeof(float), cudaMemcpyDeviceToHost); \
// cudaMemcpy(host_M2Y, M2Y, sizeof(float), cudaMemcpyDeviceToHost); \
// cudaMemcpy(host_M1Z, M1Z, sizeof(float), cudaMemcpyDeviceToHost); \
// cudaMemcpy(host_M2Z, M2Z, sizeof(float), cudaMemcpyDeviceToHost); \
// cudaMemcpy(host_integralC, integralC, sizeof(float), cudaMemcpyDeviceToHost); \
// tot_mass = ((*host_integralC)*h*h*h); \
// KiiX = (*host_M2X)/tot_mass - powf(*host_M1X/tot_mass,2.0f); \
// KiiY = (*host_M2Y)/tot_mass - powf(*host_M1Y/tot_mass,2.0f); \
// KiiZ = (*host_M2Z)/tot_mass - powf(*host_M1Z/tot_mass,2.0f);

#define COMPUTE_KII(KiiX, KiiY, KiiZ, C)                                       \
  cublasDdot(handle, Nx *Ny *Nz, C, 1, coord_X, 1, M1X);                       \
  cublasDdot(handle, Nx *Ny *Nz, C, 1, coord_XX, 1, M2X);                      \
  cublasDdot(handle, Nx *Ny *Nz, C, 1, coord_Y, 1, M1Y);                       \
  cublasDdot(handle, Nx *Ny *Nz, C, 1, coord_YY, 1, M2Y);                      \
  cublasDdot(handle, Nx *Ny *Nz, C, 1, coord_Z, 1, M1Z);                       \
  cublasDdot(handle, Nx *Ny *Nz, C, 1, coord_ZZ, 1, M2Z);                      \
  cublasDdot(handle, Nx *Ny *Nz, C, 1, dev_onesN, 1, integralC);               \
  cudaDeviceSynchronize();                                                     \
  cudaMemcpy(host_M1X, M1X, sizeof(double), cudaMemcpyDeviceToHost);           \
  cudaMemcpy(host_M2X, M2X, sizeof(double), cudaMemcpyDeviceToHost);           \
  cudaMemcpy(host_M1Y, M1Y, sizeof(double), cudaMemcpyDeviceToHost);           \
  cudaMemcpy(host_M2Y, M2Y, sizeof(double), cudaMemcpyDeviceToHost);           \
  cudaMemcpy(host_M1Z, M1Z, sizeof(double), cudaMemcpyDeviceToHost);           \
  cudaMemcpy(host_M2Z, M2Z, sizeof(double), cudaMemcpyDeviceToHost);           \
  cudaMemcpy(host_integralC, integralC, sizeof(double),                        \
             cudaMemcpyDeviceToHost);                                          \
  cudaDeviceSynchronize();                                                     \
  tot_mass = (*host_integralC);                                                \
  KiiX = (*host_M2X) / tot_mass - powf(*host_M1X / tot_mass, 2.0f);            \
  KiiY = (*host_M2Y) / tot_mass - powf(*host_M1Y / tot_mass, 2.0f);            \
  KiiZ = (*host_M2Z) / tot_mass - powf(*host_M1Z / tot_mass, 2.0f);

__global__ void copy_double2float(const double *C_in, double *C, const int Nx,
                                  const int Ny, const int Nz) {
  const int ix = threadIdx.x + blockIdx.x * blockDim.x;
  const int iy = threadIdx.y + blockIdx.y * blockDim.y;
  const int iz = threadIdx.z + blockIdx.z * blockDim.z;
  if (ix >= Nx || iy >= Ny || iz >= Nz)
    return;
  int in_idx = ix + iy * Nx + iz * Nx * Ny;
  C[in_idx] = static_cast<double>(C_in[in_idx]);
}
#define COPY_TO_C_FLOAT(C_out)                                                 \
  copy_double2float<<<grid2, block2>>>(C_out, C_float, Nx, Ny, Nz);            \
  cudaDeviceSynchronize();

void write_result_mfile(const char *nombre_archivo, int tt, double t,
                        float dKiiX_dt, float dKiiY_dt, float dKiiZ_dt) {
  FILE *archivo = fopen(nombre_archivo, "a");
  if (archivo == NULL) {
    perror("Error al abrir el archivo");
    return;
  }
  // Guardar los valores de dKiiX_dt, dKiiY_dt y dKiiZ_dt con el tiempo 'tt'
  fprintf(archivo, "%d,%.6e,%.6e,%.6e,%.6e\n", tt, t, dKiiX_dt, dKiiY_dt,
          dKiiZ_dt);
  fclose(archivo);
}

// Function to load parameters from JSON file using nlohmann/json library
bool loadParameters(const std::string &filename, double &sigma2, int &Nx,
                    int &Ny, int &Nz, double &h, double &t_max,
                    double &diffusion, double &alphaL, double &alphaT,
                    int &nParticles, int &nRealizations) {
  try {
    // Open the file
    std::ifstream file(filename);
    if (!file.is_open()) {
      std::cerr << "Error: Could not open file " << filename << std::endl;
      return false;
    }

    // Parse JSON
    nlohmann::json json;
    file >> json;

    // Extract values
    sigma2 = json["sigma2"];
    Nx = json["Nx"];
    Ny = json["Ny"];
    Nz = json["Nz"];
    h = json["delta_x"];
    t_max = json["t_max"];
    diffusion = json["diffusion"];
    alphaL = json["alphaL"];
    alphaT = json["alphaT"];
    nParticles = json["nParticles"];
    nRealizations = json["nRealizations"];

    return true;
  } catch (const std::exception &e) {
    std::cerr << "Error parsing JSON: " << e.what() << std::endl;
    return false;
  }
}

struct squared_magnitude_functor {
  __host__ __device__ double
  operator()(const thrust::tuple<double, double, double> &t) const {
    double u = thrust::get<0>(t);
    double v = thrust::get<1>(t);
    double w = thrust::get<2>(t);
    return u * u + v * v + w * w;
  }
};

__global__ void shiftParticles(double *posY, double *posZ, const double *posY0,
                               const double *posZ0, int *nY, int *nZ, double Ly,
                               double Lz, int nParticles) {

  // Compute global thread index
  int in_idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Only operate if inside bounds
  if (in_idx < nParticles) {
    posY[in_idx] = posY0[in_idx] + (double)nY[in_idx] * Ly;
    posZ[in_idx] = posZ0[in_idx] + (double)nZ[in_idx] * Lz;
  }
}

#pragma endregion

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// %%%%%%%%%%%%%%%%%%%%%%%%%% MAIN FUNCTION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// %%%%%%%%%%%%%%%%%%%%  ENTRY POINT TO PROGRAM %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
int main(int argc, char *argv[]) {

  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <device_id> <sigma2_value>"
              << std::endl;
    return -1;
  }

  std::string sigma2_str = argv[2];
  std::string filename = "./input/parameters_sigma" + sigma2_str + ".json";
  double sigma2, h, t_max;
  int n = 32, Nx, Ny, Nz;

  double diffusion, alphaL, alphaT;
  int nParticles, nRealizations;

  if (!loadParameters(filename, sigma2, Nx, Ny, Nz, h, t_max, diffusion, alphaL,
                      alphaT, nParticles, nRealizations))
    return -1;

  double sigma_f = sqrt(sigma2);

  int device = atoi(argv[1]);
  cudaSetDevice(device);
  cudaDeviceReset();

  int nx_n = Nx / n; // nx_n = Nx/N ratio
  int ny_n = Ny / n; // ny_n = Ny/N ratio
  int nz_n = Nz / n; // nz_n = Nz/N ratio

  double A = h * h;
  double lambda = 10.0 * h; // correlation length
  double Lx = Nx * h;
  double Ly = Ny * h;
  double Lz = Nz * h;
  size_t N = (size_t)(Nx + 1) * (Ny + 1) * (Nz + 1);

  double vm = 100 / Lx;
  t_max /= vm / lambda; // Adjust t_max based on vm and lambda
  double dt;
  double max_magnitude;
  int levels = log(n) / log(2.0) + 1.0;

  // Print simulation parameters
  std::cout << "======================================================="
            << std::endl;
  std::cout << "                 SIMULATION PARAMETERS                 "
            << std::endl;
  std::cout << "======================================================="
            << std::endl;
  std::cout << "Domain Information:" << std::endl;
  std::cout << "  - Grid Size: " << Nx << " x " << Ny << " x " << Nz
            << std::endl;
  std::cout << "  - Grid Spacing (h): " << h << std::endl;
  std::cout << "  - Domain Dimensions: " << Lx << " x " << Ly << " x " << Lz
            << std::endl;
  std::cout << std::endl;

  std::cout << "Flow Parameters:" << std::endl;
  std::cout << "  - Correlation Length (lambda): " << lambda << std::endl;
  std::cout << "  - Log-K Variance (sigma²): " << sigma2 << std::endl;
  std::cout << "  - Standard Deviation (sigma): " << sigma_f << std::endl;
  std::cout << std::endl;

  std::cout << "Transport Parameters:" << std::endl;
  std::cout << "  - Maximum Simulation Time: " << t_max << std::endl;
  std::cout << "  - Molecular Diffusion: " << diffusion << std::endl;
  std::cout << "  - Longitudinal Dispersivity (αL): " << alphaL << std::endl;
  std::cout << "  - Transverse Dispersivity (αT): " << alphaT << std::endl;
  std::cout << "  - Number of Particles: " << nParticles << std::endl;
  std::cout << std::endl;

  std::cout << "Simulation Configuration:" << std::endl;
  std::cout << "  - Number of Realizations: " << nRealizations << std::endl;
  // std::cout << "  - Estimated Time Step (dt): " << dt << std::endl;
  std::cout << "  - Estimated Mean Velocity: " << vm << std::endl;
  std::cout << "  - Multigrid Levels: " << levels << std::endl;
  std::cout << "======================================================="
            << std::endl;
  std::cout << std::endl;

  //%%%%% INIT MULTIGRID CONFIGURATION %%%%%%%%%%%%%%%%%%%
  std::cout << "Setting multigrid configurations..." << std::endl;
  SETUP_BLOCK_GRID3D(
      16) // block32 & grid32: configuration for general kernel ejecution
  // MG solver config
  // set multigrid parameters:
  int npre = 4, npos = 4;
  struct MG_levels MG = {levels, npre, npos};

  // one configuration per grid level
  // dim3 _grid[MG.L-1],_block[MG.L-1]
  SETUP_GRID_BLOCK_MG(MG, nx_n, ny_n,
                      nz_n) // _grid/_block configuration for CUDA-kernels

  // declare pointer to array of pointer for MG-levels
  // double *_r[MG.L-1], *_e[MG.L-1], *_rr[MG.L], *_K[MG.L]
  // ALLOCATE_MG_STRUCTURE_MEMORY(MG, nx_n, ny_n, nz_n);

  // Initialize cuBLAS environment
  INIT_CUBLAS_ENVIRONMENT

  int i_max = 10000;
  float time = 0, aux = 0;
  Crono cron;
  // initialization random secuences, see cuRAND web page
  curandState *devStates;
  CUDA_CALL(cudaMalloc((void **)&devStates, i_max * sizeof(curandState)));
  cudaDeviceSynchronize();
  dim3 block1(1024, 1);
  dim3 grid1((i_max + block1.x - 1) / block1.x, 1);
  setup_uniform_distrib<<<grid1, block1>>>(devStates, i_max);
  // declare & allocatate vector in GPU
  // K: conductivity field with
  // CUDA_ALLOCATE_VECTOR(double, Nx*Ny*Nz, K);
  // K vector is allocated later in _K[MG.L-1]
  // V1, V2, V3, a, b: auxiliary vectors for random sequences
  CUDA_ALLOCATE_VECTOR(double, i_max, V1);
  CUDA_ALLOCATE_VECTOR(double, i_max, V2);
  CUDA_ALLOCATE_VECTOR(double, i_max, V3);
  CUDA_ALLOCATE_VECTOR(double, i_max, a);
  CUDA_ALLOCATE_VECTOR(double, i_max, b);
  // std::vector<double> host_ones_double(Nx * Ny * Nz, 1.0);
  // CUDA_ALLOCATE_VECTOR(double, Nx *Ny *Nz, dev_ones_double);
  // cudaMemcpy(dev_ones_double, host_ones_double.data(), sizeof(double) * Nx *
  // Ny * Nz, cudaMemcpyHostToDevice); cudaFree(dev_ones_double);
  double *K_eq;
  cudaMalloc(&K_eq, sizeof(double));
  double *host_K_eq;
  host_K_eq = new double[1];
  SETUP_BLOCK_GRID_RANDOM_KERNEL(block2, grid2, Nx, Ny, Nz)

  // %%%%%%%%%%%%%%%%%%%% FLOW EQUATION SETUP
  // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% boundary conditions
  std::cout << "Setting up flow equation..." << std::endl;
  int BCbottom = periodic;
  int BCtop = periodic;
  int BCsouth = periodic;
  int BCnorth = periodic;
  int BCwest = dirichlet;
  int BCeast = dirichlet;
  bool pin1stCell = false;

  double Hbottom = 0.0 / 0.0; // Assign NaN to unused values
  double Htop = 0.0 / 0.0;    // Assign NaN to unused values
  double Hsouth = 0.0 / 0.0;  // Assign NaN to unused values
  double Hnorth = 0.0 / 0.0;  // Assign NaN to unused values
  double Heast = 0.0;
  double Hwest = 100.0;

  // set the blas operations using in solvers in finest level
  blas BLAS(Nx, Ny, Nz, grid16, block16, handle);
  // set linear operator for preconditioner y CG method
  // ----------coarse solver for CCMG------------------------
  int Nx_coarse = nx_n;
  int Ny_coarse = ny_n;
  int Nz_coarse = nz_n;
  double h_coarse = Ly / Ny_coarse;
  double A_coarse = h_coarse * h_coarse;
  CUDA_ALLOCATE_VECTOR(double, nx_n *ny_n *nz_n, y_coarse);
  CUDA_ALLOCATE_VECTOR(double, nx_n *ny_n *nz_n, r_coarse);
  blas BLAS_coarse(Nx_coarse, Ny_coarse, Nz_coarse, _grid[1], _block[1],
                   handle);
  IdentityPrecond Icoarse(Nx_coarse, Ny_coarse, Nz_coarse);

  IdentityPrecond I(Nx, Ny, Nz);

  char outdir[50];
  sprintf(outdir, "./output/out_%.2f", sigma2);
  std::string outdir_str(outdir);
  char command[100];
  sprintf(command, "rm -rf %s && mkdir -p %s", outdir, outdir);
  system(command);

  // double vm = 0.0;
  // declare pointer to array of pointer for MG-levels
  // double *_r[MG.L-1], *_e[MG.L-1], *_rr[MG.L], *_K[MG.L]
  /* Declare pointers to arrays for MG levels */
  double **_r, **_e, **_rr;
  double **_K;
  /* Allocate memory for the pointers in the host */
  _r = (double **)malloc((MG.L - 1) * sizeof(double *));
  _e = (double **)malloc((MG.L - 1) * sizeof(double *));
  _rr = (double **)malloc(MG.L * sizeof(double *));
  _K = (double **)malloc(MG.L * sizeof(double *));

  /* Allocate memory for each level on the GPU */

  int threadsPerBlock = 256;
  int blocksPerGrid = (nParticles + threadsPerBlock - 1) / threadsPerBlock;
  double *posY, *posZ;
  int *nY, *nZ;

  cudaMalloc((void **)&posY, nParticles * sizeof(double));
  cudaMalloc((void **)&posZ, nParticles * sizeof(double));
  cudaMalloc((void **)&nY, nParticles * sizeof(int));
  cudaMalloc((void **)&nZ, nParticles * sizeof(int));
  // Inicializar a cero
  cudaMemset(posY, 0, nParticles * sizeof(double));
  cudaMemset(posZ, 0, nParticles * sizeof(double));
  cudaMemset(nY, 0, nParticles * sizeof(int));
  cudaMemset(nZ, 0, nParticles * sizeof(int));

  thrust::device_ptr<double> posY_ptr(posY);
  thrust::device_ptr<double> posZ_ptr(posZ);
  thrust::device_ptr<int> nY_ptr(nY);
  thrust::device_ptr<int> nZ_ptr(nZ);

  // construimos device_vector temporal **apuntando a la memoria ya existente**
  thrust::device_vector<double> posY_dev(posY_ptr, posY_ptr + nParticles);
  thrust::device_vector<double> posZ_dev(posZ_ptr, posZ_ptr + nParticles);
  thrust::device_vector<int> nY_dev(nY_ptr, nY_ptr + nParticles);
  thrust::device_vector<int> nZ_dev(nZ_ptr, nZ_ptr + nParticles);

  for (int k = 0; k < nRealizations; k++) {

    cout << "Realization " << k + 1 << " of " << nRealizations << endl;

    for (int i = 0; i < MG.L - 1; ++i) {
      int N = pow(2, i);
      cudaMalloc(&_e[i], sizeof(double) * N * N * N * nx_n * ny_n * nz_n);
      cudaMalloc(&_r[i], sizeof(double) * N * N * N * nx_n * ny_n * nz_n);
    }
    /* Allocate memory for rr at all levels on the GPU */
    for (int i = 0; i < MG.L; ++i) {
      int N = pow(2, i);
      cudaMalloc(&_rr[i], sizeof(double) * N * N * N * nx_n * ny_n * nz_n);
      cudaMalloc(&_K[i], sizeof(double) * N * N * N * nx_n * ny_n * nz_n);
    }

    // declaration variables for flow eq., and PCG solver
    CUDA_ALLOCATE_VECTOR(double, Nx *Ny *Nz, RHS_flow);
    CUDA_ALLOCATE_VECTOR(double, Nx *Ny *Nz, Head);
    CUDA_ALLOCATE_VECTOR(double, Nx *Ny *Nz, r);
    CUDA_ALLOCATE_VECTOR(double, Nx *Ny *Nz, z);

    random_kernel_3D_gauss<<<grid1, block1>>>(devStates, V1, V2, V3, a, b,
                                              lambda, i_max, 100);
    cudaDeviceSynchronize();
    conductivity_kernel_3D_logK<<<grid2, block2>>>(
        V1, V2, V3, a, b, i_max, _K[MG.L - 1], lambda, h, Nx, Ny, Nz, sigma_f);
    cudaDeviceSynchronize();

    // cublasDdot(handle, Nx*Ny*Nz, _K[MG.L-1], 1, dev_ones_double , 1, K_eq);
    // cudaDeviceSynchronize();
    // cudaMemcpy(host_K_eq, K_eq, sizeof(double), cudaMemcpyDeviceToHost);
    // vm += *host_K_eq / (Nx * Ny * Nz)*(Heast-Hwest)/Lx;
    compute_expK<<<grid2, block2>>>(_K[MG.L - 1], Nx, Ny, Nz);
    cudaDeviceSynchronize();

    // compute K for all grid levels
    for (int i = MG.L - 1; i > 1; --i)
      HomogenizationPermeability(_K[i - 1], _K[i], pow(2, i - 1) * nx_n,
                                 pow(2, i - 1) * ny_n, pow(2, i - 1) * nz_n,
                                 _grid[i - 1], _block[i - 1]);

    // set linear operator for preconditioner y CG method
    // ----------coarse solver for CCMG------------------------
    laplacianHead AH(_K[MG.L - 1], Nx, Ny, Nz, A, h, BCbottom, BCtop, BCsouth,
                     BCnorth, BCwest, BCeast, pin1stCell, grid16, block16);
    laplacianHeadCoarse Ap_coarse(_K[1], Nx_coarse, Ny_coarse, Nz_coarse,
                                  A_coarse, h_coarse, BCbottom, BCtop, BCsouth,
                                  BCnorth, BCwest, BCeast, pin1stCell, _grid[1],
                                  _block[1]);
    // ----------preconditioner for CG using CCMG--------------
    MGprecond2 PCCMG_CG(Nx, Ny, Nz, BCbottom, BCtop, BCsouth, BCnorth, BCwest,
                        BCeast, pin1stCell, _grid, _block, Ly, nx_n, ny_n, nz_n,
                        handle, _e, _r, _rr, _K, MG, Ap_coarse, Icoarse,
                        BLAS_coarse, r_coarse, y_coarse);

    //%%%%%%%%%%%%%%%%%%%% ASSEMBLY AND SOLUTION OF THE FLOW EQUATION
    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    std::cout << "Assembling and solving flow equation..." << std::endl;
    int iterHead = 0;
    // int print_monitor = 1; //print only last residual
    int print_monitor = 2; // print residual evolution
    // int print_monitor = 0; //hide residual evolution
    cudaMemset(Head, 0, sizeof(double) * Nx * Ny * Nz);
    cudaMemset(RHS_flow, 0, sizeof(double) * Nx * Ny * Nz);

    RHS_head(RHS_flow, _K[MG.L - 1], Nx, Ny, Nz, A, h, BCbottom, BCtop, BCsouth,
             BCnorth, BCwest, BCeast, Hbottom, Htop, Hsouth, Hnorth, Hwest,
             Heast, grid16, block16);
    iterHead = solver_CG(AH, PCCMG_CG, BLAS, Head, z, r, RHS_flow,
                         _rr[MG.L - 1], 0.0, 1e-6, 200, print_monitor);

    cudaFree(RHS_flow);
    cudaFree(r);
    cudaFree(z);
    for (int i = 0; i < MG.L; ++i)
      cudaFree(_rr[i]);
    for (int i = 0; i < MG.L - 1; ++i) {
      cudaFree(_e[i]);
      cudaFree(_r[i]);
    }

    // Reservar memoria para los vectores de velocidad en layout CÚBICO
    thrust::device_vector<double> U_cube((Nx + 1) * (Ny + 1) * (Nz + 1));
    thrust::device_vector<double> V_cube((Nx + 1) * (Ny + 1) * (Nz + 1));
    thrust::device_vector<double> W_cube((Nx + 1) * (Ny + 1) * (Nz + 1));

    std::cout << "Computing velocity field form hydraulic head..." << std::endl;
    compute_velocity_from_head(thrust::raw_pointer_cast(U_cube.data()),
                               thrust::raw_pointer_cast(V_cube.data()),
                               thrust::raw_pointer_cast(W_cube.data()), Head,
                               _K[MG.L - 1], Nx, Ny, Nz, h, BCwest, BCeast,
                               BCsouth, BCnorth, BCbottom, BCtop, Hwest, Heast,
                               Hsouth, Hnorth, Hbottom, Htop, grid16, block16);
    {
      cudaError_t cerr = cudaDeviceSynchronize();
      if (cerr != cudaSuccess) {
        fprintf(stderr, "Error in compute_velocity_from_head: %s\n",
                cudaGetErrorString(cerr));
        return -1;
      }
    }

    for (int i = 0; i < MG.L; ++i)
      cudaFree(_K[i]);
    cudaFree(Head);

    //%%%%%%%%%%%%%%%%%%%% TRANSPORT SETUP %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    std::cout << "Setting up transport..." << std::endl;

    if (k == 0) {
      auto absval = [] __device__(double v) { return fabs(v); };

      double umax = thrust::transform_reduce(
          U_cube.begin(), U_cube.end(), absval, 0.0, thrust::maximum<double>());
      double vmax = thrust::transform_reduce(
          V_cube.begin(), V_cube.end(), absval, 0.0, thrust::maximum<double>());
      double wmax = thrust::transform_reduce(
          W_cube.begin(), W_cube.end(), absval, 0.0, thrust::maximum<double>());

      // cota conservadora de la magnitud máxima
      max_magnitude = sqrt(umax * umax + vmax * vmax + wmax * wmax);
      dt = (h / (2 * max_magnitude));

      std::cout << "  - ||v_max|| : " << max_magnitude << std::endl;
      std::cout << "  - Time Step (dt): " << dt << std::endl;
    }

    // SET INJECTION BOX
    const double p1x = 10.0 * h, p1y = Ly * 0.1, p1z = Lz * 0.1;
    const double p2x = 10.0 * h, p2y = Ly - p1y, p2z = Lz - p1z;

    bool useTrilinearCorrection = true;
    long int seed = 123456789 * (k + 1);
    int steps = t_max / dt; // number of steps to simulate

    // 2) Construir grid PAR2 y reservar “cúbicos” inicializados a 0
    auto grid = par2::grid::build<double>(Nx, Ny, Nz, h, h, h);

    // 4) Crear PParticles con punteros al layout “cúbico”
    par2::PParticles<double> particles(
        grid, std::move(U_cube), std::move(V_cube), std::move(W_cube),
        std::move(nY_dev), std::move(nZ_dev), diffusion, alphaL, alphaT,
        nParticles, seed, useTrilinearCorrection);

    particles.initializeBox(p1x, p1y, p1z, p2x, p2y, p2z, true);

    // // Pointers to unwrapped buffers provided by PParticles (if available)
    // const double *yRaw =
    //     particles.yUnwrapPtr() ? particles.yUnwrapPtr() : particles.yPtr();
    // const double *zRaw = (particles.zUnwrapPtr() && Nz != 1)
    //                          ? particles.zUnwrapPtr()
    //                          : particles.zPtr();
    // thrust::device_ptr<const double> yBeg(yRaw);
    // thrust::device_ptr<const double> zBeg(zRaw);
    thrust::device_ptr<const double> xBeg(particles.xPtr());
    thrust::device_ptr<const double> yBeg(posY_ptr);
    thrust::device_ptr<const double> zBeg(posZ_ptr);
    const double *posY0 = particles.yPtr(), *posZ0 = particles.zPtr();

    std::string outPath = outdir_str + "/macrodispersion_var_v9_" +
                          std::to_string(sigma2).substr(0, 4) + "_" +
                          std::to_string(k) + ".csv";

    /* -------- CSV macrodispersion -------- */
    std::ofstream csv(outPath);
    csv << "t,Dx,Dy,Dz\n";
    double prevVarX = 0.0, prevVarY = 0.0, prevVarZ = 0.0;
    const int reg = std::max(1, steps / 300);
    /* ===================  Bucle principal  =================== */
    std::cout << "Starting transport simulation..." << std::endl;
    for (int i = 0; i < steps; i++) {
      particles.move(dt);

      if (i % static_cast<int>(round(steps * 0.2)) == 0 && i != 0) {
        std::cout << "  - Step: " << i << " / " << steps << " ("
                  << static_cast<int>(i * 100.0 / steps) << "%)" << std::endl;
      }

      if ((i + 1) % reg == 0 || i == 1) {
        shiftParticles<<<blocksPerGrid, threadsPerBlock>>>(
            posY, posZ, posY0, posZ0, nY, nZ, Ly, Lz, nParticles);
        cudaDeviceSynchronize();
        // X
        double sumX = thrust::reduce(xBeg, xBeg + nParticles, 0.0,
                                     thrust::plus<double>());
        double sumX2 = thrust::transform_reduce(
            xBeg, xBeg + nParticles, square(), 0.0, thrust::plus<double>());

        // Y/Z unwrapped if available
        double sumY = thrust::reduce(yBeg, yBeg + nParticles, 0.0,
                                     thrust::plus<double>());
        double sumY2 = thrust::transform_reduce(
            yBeg, yBeg + nParticles, square(), 0.0, thrust::plus<double>());

        double sumZ = thrust::reduce(zBeg, zBeg + nParticles, 0.0,
                                     thrust::plus<double>());
        double sumZ2 = thrust::transform_reduce(
            zBeg, zBeg + nParticles, square(), 0.0, thrust::plus<double>());

        double meanX = sumX / nParticles, meanY = sumY / nParticles,
               meanZ = sumZ / nParticles;
        double varX = sumX2 / nParticles - meanX * meanX;
        double varY = sumY2 / nParticles - meanY * meanY;
        double varZ = sumZ2 / nParticles - meanZ * meanZ;

        prevVarX = varX;
        prevVarY = varY;
        prevVarZ = varZ;
      }
      if (i % reg == 0 || i == 2) {
        shiftParticles<<<blocksPerGrid, threadsPerBlock>>>(
            posY, posZ, posY0, posZ0, nY, nZ, Ly, Lz, nParticles);
        cudaDeviceSynchronize();
        double sumX = thrust::reduce(xBeg, xBeg + nParticles, 0.0,
                                     thrust::plus<double>());
        double sumX2 = thrust::transform_reduce(
            xBeg, xBeg + nParticles, square(), 0.0, thrust::plus<double>());

        double sumY = thrust::reduce(yBeg, yBeg + nParticles, 0.0,
                                     thrust::plus<double>());
        double sumY2 = thrust::transform_reduce(
            yBeg, yBeg + nParticles, square(), 0.0, thrust::plus<double>());

        double sumZ = thrust::reduce(zBeg, zBeg + nParticles, 0.0,
                                     thrust::plus<double>());
        double sumZ2 = thrust::transform_reduce(
            zBeg, zBeg + nParticles, square(), 0.0, thrust::plus<double>());

        double meanX = sumX / nParticles, meanY = sumY / nParticles,
               meanZ = sumZ / nParticles;
        double varX = sumX2 / nParticles - meanX * meanX;
        double varY = sumY2 / nParticles - meanY * meanY;
        double varZ = sumZ2 / nParticles - meanZ * meanZ;

        double Dx = (varX - prevVarX) / dt;
        double Dy = (varY - prevVarY) / dt;
        double Dz = (varZ - prevVarZ) / dt;
        csv << (i * dt) << ',' << Dx << ',' << Dy << ',' << Dz << '\n';
      }

      if (i % reg == 0 && k == 0) {
        particles.exportCSV(outdir_str + "/transport_var" +
                            std::to_string(sigma2).substr(0, 4) + "_step_" +
                            std::to_string(i) + ".csv");
      }
    }
  }

  return 0;
}