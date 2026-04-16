//AUXILIAR ROUTINES FOR CG
void alpha(const double *x, double *y,
const double *rz,  const double *yP, bool plus_minus,
const int Nx, const int Ny, const int Nz,
dim3 grid, dim3 block);

void beta(const double *x, double *y,
const double *rz, const double *rz_old,
const int Nx, const int Ny, const int Nz,
dim3 grid, dim3 block);


//COMMON ROUTINES FOR SOLVER
void AXPBY(const double *x, double *y, double *output,
const double a, const double b,
const int Nx, const int Ny, const int Nz,
dim3 grid, dim3 block);
