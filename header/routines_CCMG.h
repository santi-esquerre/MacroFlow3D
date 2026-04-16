// AUXILIAR ROUTINES FOR CELL CENTERED MULTIGRID METHOD

#include "cublas_v2.h"
void prolongation(double *phiFine, const double *phiCoarse, int Nx, int Ny, int Nz,
	dim3 grid, dim3 block);

void restriction(double *phiCoarse, const double *phiFine, int Nx, int Ny, int Nz, dim3 grid, dim3 block);

void HomogenizationPermeability(double *phiCoarse, const double *phiFine, int Nx, int Ny, int Nz, dim3 grid, dim3 block);

void smooth_GSRB(double *xk, const double *rhs, double *rk_1, const double *K,
	double dxdx, int Nx, int Ny, int Nz,
	int dirichletBottom, int dirichletTop,
	int dirichletSouth, int dirichletNorth,
	int dirichletWest, int dirichletEast, bool pin1stCell,
	int itMAX, bool update_res, dim3 grid, dim3 block);

void update_res(double *xk, const double *rhs, double *rk_1, const double *K,
	double dxdx, int Nx, int Ny, int Nz,
	int dirichletBottom, int dirichletTop,
	int dirichletSouth, int dirichletNorth,
	int dirichletWest, int dirichletEast, bool pin1stCell,
	dim3 grid, dim3 block);

void SolveCoarseSystemGSRB(double *xnew, const double *rhs, double *r, const double *K, double dxdx, int Nx, int Ny, int Nz, const int itMAX,
	dim3 grid, dim3 block,
	cublasHandle_t handle,
	int dirichletBottom, int dirichletTop,
	int dirichletSouth, int dirichletNorth,
	int dirichletWest, int dirichletEast, bool pin1stCell);
