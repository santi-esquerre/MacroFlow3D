void HomogenizationPermeability(double *phiCoarse, const double *phiFine, int Nx, int Ny, int Nz, dim3 grid, dim3 block);

void restriction(double *phiCoarse, const double *phiFine, int Nx, int Ny, int Nz, dim3 grid, dim3 block);

void prolongation(double *phiFine, const double *phiCoarse, int Nx, int Ny, int Nz,
	dim3 grid, dim3 block);
