//AUXILIAR ROUTINES FOR BiCGStab
void alphaU(const double *Ap, const double *r,  double *s, const double *rr_, const double *Apr_, const int Nx, const int Ny, const int Nz,
	dim3 grid, dim3 block);

void betaU(const double *r, double *p, const double *Ap, const double *rr_, const double *Apr_, const double *Ass, const double *AsAs,
	const int Nx, const int Ny, const int Nz,
	dim3 grid, dim3 block);

void omegaX(const double *p, double *x, const double *s, const double *rr_, const double *Apr_, const double *Ass, const double *AsAs,
	const int Nx, const int Ny, const int Nz,
	dim3 grid, dim3 block);

void omegaR(const double *s, double *r, const double *As, const double *Ass, const double *AsAs,
	const int Nx, const int Ny, const int Nz,
	dim3 grid, dim3 block);
