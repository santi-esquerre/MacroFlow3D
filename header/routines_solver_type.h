// WARNING:
// All routines consider variable conductivities K at all grid levels (**K[] array).
// If conductivities are constant, the K array should be removed.

// V-cycle using GSRB as the coarsest solver
void V_cycle(double **e_pre, double **r, double **rr, double **K,
	dim3 *grid, dim3 *block,
	int dirichletBottom, int dirichletTop,
	int dirichletSouth, int dirichletNorth,
	int dirichletWest, int dirichletEast, bool pin1stCell,
	MG_levels MG, int l, cublasHandle_t handle, int ratioX, int ratioY, int ratioZ, double Ly);

// V-cycle using CG as the coarsest solver
void V_cycle2(double **e_pre, double **r, double **rr, double **K,
	dim3 *grid, dim3 *block,
	int dirichletBottom, int dirichletTop,
	int dirichletSouth, int dirichletNorth,
	int dirichletWest, int dirichletEast, bool pin1stCell,
	MG_levels MG, int l, cublasHandle_t handle, int ratioX, int ratioY, int ratioZ, double Ly,
	Matrix_t &M, Matrix_t &precond, blas_t &BLAS, double *aux, double *aux2);

// V-cycle (GSRB coarsest solver) using as preconditioner
void Precond_CCMG_Vcycle(double *e0fine, const double *rfine,
	double **e, double **r, double **rr, double **K,
	dim3 *grid, dim3 *block,
	int dirichletBottom, int dirichletTop,
	int dirichletSouth, int dirichletNorth,
	int dirichletWest, int dirichletEast, bool pin1stCell,
	int Nx, int Ny, int Nz, MG_levels MG, cublasHandle_t handle, int ratioX, int ratioY, int ratioZ, double Ly);

// V-cycle (CG coarsest solver) using as preconditioner
void Precond_CCMG_Vcycle2(double *e0fine, const double *rfine,
	double **e, double **r, double **rr, double **K,
	dim3 *grid, dim3 *block,
	int dirichletBottom, int dirichletTop,
	int dirichletSouth, int dirichletNorth,
	int dirichletWest, int dirichletEast, bool pin1stCell,
	int Nx, int Ny, int Nz, MG_levels MG, cublasHandle_t handle, int ratioX, int ratioY, int ratioZ, double Ly,
	Matrix_t &M, Matrix_t &precond, blas_t &BLAS, double *aux, double *aux2);

int solver_CG(Matrix_t &M, Matrix_t &precond, blas_t &BLAS, double *x, double *P, double *r, double *z, double *y, double tol_abs, double tol_rel, int iter_max, int print_monitor);
