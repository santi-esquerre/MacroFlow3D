#include <cuda_runtime_api.h>
#define gridXY dim3(grid.x,grid.y)
#define blockXY dim3(block.x,block.y)
#define gridXZ dim3(grid.x,grid.z)
#define blockXZ dim3(block.x,block.z)
#define gridYZ dim3(grid.y,grid.z)
#define blockYZ dim3(block.y,block.z)

//#######################################################################
// 	routine omega_r (uses in BiCGStab)
//	r_{j+1} = s_j - omega*A*M*s
//	omega = (AMs, s) / (AMs, AMs)
//#######################################################################
//-----------------------------------------------------------------------
__global__ void omega_r_int_bottom_top(const double *s, double *r, const double *As, const double *Ass, const double *AsAs,
const int Nx, const int Ny, const int Nz){
	int ix = threadIdx.x + blockIdx.x*blockDim.x;
	int iy = threadIdx.y + blockIdx.y*blockDim.y;
	if (ix >= Nx-2 || iy >= Ny-2) return;
	int stride = Nx*Ny;
	int in_idx = (ix+1) + (iy+1)*Nx;
	for(int iz = 0; iz<Nz; ++iz){
		r[in_idx] = s[in_idx] - Ass[0]/AsAs[0]*As[in_idx];
		in_idx+=stride;
	}
}

__global__ void omega_r_south_north(const double *s, double *r, const double *As, const double *Ass, const double *AsAs,
const int Nx, const int Ny, const int Nz){
	int ix = threadIdx.x + blockIdx.x*blockDim.x;
	int iz = threadIdx.y + blockIdx.y*blockDim.y;
	if (ix >= Nx-2 || iz >= Nz-2) return;
	int stride = Nx*Ny;
	int in_idx;
	in_idx = (ix + 1) + (iz + 1)*stride;
	r[in_idx] = s[in_idx] - Ass[0]/AsAs[0]*As[in_idx];

	in_idx = (ix + 1) + (Ny - 1)*Nx + (iz + 1)*stride;
	r[in_idx] = s[in_idx] - Ass[0]/AsAs[0]*As[in_idx];
}

__global__ void omega_r_este_oeste(const double *s, double *r, const double *As, const double *Ass, const double *AsAs,
const int Nx, const int Ny, const int Nz){
	int iy = threadIdx.x + blockIdx.x*blockDim.x;
	int iz  = threadIdx.y + blockIdx.y*blockDim.y;
	if (iy >= Ny-2 || iz >= Nz-2) return;
	int stride = Nx*Ny;
	int in_idx;
	in_idx = (Nx - 1) + (iy + 1)*Nx + (iz + 1)*stride;
	r[in_idx] = s[in_idx] - Ass[0]/AsAs[0]*As[in_idx];

	in_idx = (iy + 1)*Nx + (iz + 1)*stride;
	r[in_idx] = s[in_idx] - Ass[0]/AsAs[0]*As[in_idx];
}

__global__ void omega_r_edge_X_South_Bottom(const double *s, double *r, const double *As, const double *Ass, const double *AsAs,
const int Nx, const int Ny, const int Nz){
	int ix = threadIdx.x + blockIdx.x*blockDim.x;
	if (ix >= Nx-2) return;
	int stride = Nx*Ny;
	int iy = 0;
	int iz = 0;
	int in_idx = (ix+1) + iy*Nx + iz*stride;
	r[in_idx] = s[in_idx] - Ass[0]/AsAs[0]*As[in_idx];
}

__global__ void omega_r_edge_X_South_Top(const double *s, double *r, const double *As, const double *Ass, const double *AsAs,
const int Nx, const int Ny, const int Nz){
	int ix = threadIdx.x + blockIdx.x*blockDim.x;
	if (ix >= Nx-2) return;
	int stride = Nx*Ny;
	int iy = 0;
	int iz = Nz-1;
	int in_idx = (ix+1) + iy*Nx + iz*stride;
	r[in_idx] = s[in_idx] - Ass[0]/AsAs[0]*As[in_idx];
}

__global__ void omega_r_edge_X_North_Bottom(const double *s, double *r, const double *As, const double *Ass, const double *AsAs,
const int Nx, const int Ny, const int Nz){
	int ix = threadIdx.x + blockIdx.x*blockDim.x;
	if (ix >= Nx-2) return;
	int stride = Nx*Ny;
	int iy = Ny-1;
	int iz = 0;
	int in_idx = (ix+1) + iy*Nx + iz*stride;
	r[in_idx] = s[in_idx] - Ass[0]/AsAs[0]*As[in_idx];
}

__global__ void omega_r_edge_X_North_Top(const double *s, double *r, const double *As, const double *Ass, const double *AsAs,
const int Nx, const int Ny, const int Nz){
	int ix = threadIdx.x + blockIdx.x*blockDim.x;
	if (ix >= Nx-2) return;
	int stride = Nx*Ny;
	int iy = Ny-1;
	int iz = Nz-1;
	int in_idx = (ix+1) + iy*Nx + iz*stride;
	r[in_idx] = s[in_idx] - Ass[0]/AsAs[0]*As[in_idx];
}

__global__ void omega_r_edge_Z_South_West(const double *s, double *r, const double *As, const double *Ass, const double *AsAs,
const int Nx, const int Ny, const int Nz){
	int iz = threadIdx.x + blockIdx.x*blockDim.x;
	if (iz >= Nz-2) return;
	int stride = Nx*Ny;
	int ix = 0;
	int iy = 0;
	int in_idx = ix + iy*Nx + (iz+1)*stride;
	r[in_idx] = s[in_idx] - Ass[0]/AsAs[0]*As[in_idx];
}

__global__ void omega_r_edge_Z_South_East(const double *s, double *r, const double *As, const double *Ass, const double *AsAs,
const int Nx, const int Ny, const int Nz){
	int iz = threadIdx.x + blockIdx.x*blockDim.x;
	if (iz >= Nz-2) return;
	int stride = Nx*Ny;
	int ix = Nx-1;
	int iy = 0;
	int in_idx = ix + iy*Nx + (iz+1)*stride;
	r[in_idx] = s[in_idx] - Ass[0]/AsAs[0]*As[in_idx];
}
__global__ void omega_r_edge_Z_North_West(const double *s, double *r, const double *As, const double *Ass, const double *AsAs,
const int Nx, const int Ny, const int Nz){
	int iz = threadIdx.x + blockIdx.x*blockDim.x;
	if (iz >= Nz-2) return;
	int stride = Nx*Ny;
	int ix = 0;
	int iy = Ny-1;
	int in_idx = ix + iy*Nx + (iz+1)*stride;
	r[in_idx] = s[in_idx] - Ass[0]/AsAs[0]*As[in_idx];
}

__global__ void omega_r_edge_Z_North_East(const double *s, double *r, const double *As, const double *Ass, const double *AsAs,
const int Nx, const int Ny, const int Nz){
	int iz = threadIdx.x + blockIdx.x*blockDim.x;
	if (iz >= Nz-2) return;
	int stride = Nx*Ny;
	int ix = Nx-1;
	int iy = Ny-1;
	int in_idx = ix + iy*Nx + (iz+1)*stride;
	r[in_idx] = s[in_idx] - Ass[0]/AsAs[0]*As[in_idx];
}

__global__ void omega_r_edge_Y_West_Bottom(const double *s, double *r, const double *As, const double *Ass, const double *AsAs,
const int Nx, const int Ny, const int Nz){
	int iy = threadIdx.x + blockIdx.x*blockDim.x;
	if (iy >= Ny-2) return;
	int stride = Nx*Ny;
	int ix = 0;
	int iz = 0;
	int in_idx = ix + (iy+1)*Nx + iz*stride;
	r[in_idx] = s[in_idx] - Ass[0]/AsAs[0]*As[in_idx];
}
__global__ void omega_r_edge_Y_West_Top(const double *s, double *r, const double *As, const double *Ass, const double *AsAs,
const int Nx, const int Ny, const int Nz){
	int iy = threadIdx.x + blockIdx.x*blockDim.x;
	if (iy >= Ny-2) return;
	int stride = Nx*Ny;
	int ix = 0;
	int iz = Nz-1;
	int in_idx = ix + (iy+1)*Nx + iz*stride;
	r[in_idx] = s[in_idx] - Ass[0]/AsAs[0]*As[in_idx];
}
__global__ void omega_r_edge_Y_East_Bottom(const double *s, double *r, const double *As, const double *Ass, const double *AsAs,
const int Nx, const int Ny, const int Nz){
	int iy = threadIdx.x + blockIdx.x*blockDim.x;
	if (iy >= Ny-2) return;
	int stride = Nx*Ny;
	int ix = Nx-1;
	int iz = 0;
	int in_idx = ix + (iy+1)*Nx + iz*stride;
	r[in_idx] = s[in_idx] - Ass[0]/AsAs[0]*As[in_idx];
}
__global__ void omega_r_edge_Y_East_Top(const double *s, double *r, const double *As, const double *Ass, const double *AsAs,
const int Nx, const int Ny, const int Nz){
	int iy = threadIdx.x + blockIdx.x*blockDim.x;
	if (iy >= Ny-2) return;
	int stride = Nx*Ny;
	int ix = Nx-1;
	int iz = Nz-1;
	int in_idx = ix + (iy+1)*Nx + iz*stride;
	r[in_idx] = s[in_idx] - Ass[0]/AsAs[0]*As[in_idx];
}

__global__ void omega_r_vertex_SWB(const double *s, double *r, const double *As, const double *Ass, const double *AsAs,
const int Nx, const int Ny, const int Nz){
	int ix = 0;
	int iy = 0;
	int iz = 0;
	int in_idx = ix + iy*Nx + iz*Nx*Ny;
	r[in_idx] = s[in_idx] - Ass[0]/AsAs[0]*As[in_idx];
}
__global__ void omega_r_vertex_SWT(const double *s, double *r, const double *As, const double *Ass, const double *AsAs,
const int Nx, const int Ny, const int Nz){
	int ix = 0;
	int iy = 0;
	int iz = Nz-1;
	int in_idx = ix + iy*Nx + iz*Nx*Ny;
	r[in_idx] = s[in_idx] - Ass[0]/AsAs[0]*As[in_idx];
}
__global__ void omega_r_vertex_SEB(const double *s, double *r, const double *As, const double *Ass, const double *AsAs,
const int Nx, const int Ny, const int Nz){
	int ix = Nx-1;
	int iy = 0;
	int iz = 0;
	int in_idx = ix + iy*Nx + iz*Nx*Ny;
	r[in_idx] = s[in_idx] - Ass[0]/AsAs[0]*As[in_idx];
}
__global__ void omega_r_vertex_SET(const double *s, double *r, const double *As, const double *Ass, const double *AsAs,
const int Nx, const int Ny, const int Nz){
	int ix = Nx-1;
	int iy = 0;
	int iz = Nz-1;
	int in_idx = ix + iy*Nx + iz*Nx*Ny;
	r[in_idx] = s[in_idx] - Ass[0]/AsAs[0]*As[in_idx];
}




__global__ void omega_r_vertex_NWB(const double *s, double *r, const double *As, const double *Ass, const double *AsAs,
const int Nx, const int Ny, const int Nz){
	int ix = 0;
	int iy = Ny-1;
	int iz = 0;
	int in_idx = ix + iy*Nx + iz*Nx*Ny;
	r[in_idx] = s[in_idx] - Ass[0]/AsAs[0]*As[in_idx];
}
__global__ void omega_r_vertex_NWT(const double *s, double *r, const double *As, const double *Ass, const double *AsAs,
const int Nx, const int Ny, const int Nz){
	int ix = 0;
	int iy = Ny-1;
	int iz = Nz-1;
	int in_idx = ix + iy*Nx + iz*Nx*Ny;
	r[in_idx] = s[in_idx] - Ass[0]/AsAs[0]*As[in_idx];
}
__global__ void omega_r_vertex_NEB(const double *s, double *r, const double *As, const double *Ass, const double *AsAs,
const int Nx, const int Ny, const int Nz){
	int ix = Nx-1;
	int iy = Ny-1;
	int iz = 0;
	int in_idx = ix + iy*Nx + iz*Nx*Ny;
	r[in_idx] = s[in_idx] - Ass[0]/AsAs[0]*As[in_idx];
}
__global__ void omega_r_vertex_NET(const double *s, double *r, const double *As, const double *Ass, const double *AsAs,
const int Nx, const int Ny, const int Nz){
	int ix = Nx-1;
	int iy = Ny-1;
	int iz = Nz-1;
	int in_idx = ix + iy*Nx + iz*Nx*Ny;
	r[in_idx] = s[in_idx] - Ass[0]/AsAs[0]*As[in_idx];
}


void omegaR(const double *s, double *r, const double *As, const double *Ass, const double *AsAs,
	const int Nx, const int Ny, const int Nz,
	dim3 grid, dim3 block){
	omega_r_int_bottom_top<<<gridXY,blockXY>>>(s,r,As,Ass,AsAs,Nx,Ny,Nz);
	omega_r_south_north<<<gridXZ,blockXZ>>>(s,r,As,Ass,AsAs,Nx,Ny,Nz);
	omega_r_este_oeste<<<gridYZ,blockYZ>>>(s,r,As,Ass,AsAs,Nx,Ny,Nz);

	omega_r_edge_X_South_Bottom<<<grid.x,block.x>>>(s,r,As,Ass,AsAs,Nx,Ny,Nz);
	omega_r_edge_X_South_Top<<<grid.x,block.x>>>(s,r,As,Ass,AsAs,Nx,Ny,Nz);
	omega_r_edge_X_North_Bottom<<<grid.x,block.x>>>(s,r,As,Ass,AsAs,Nx,Ny,Nz);
	omega_r_edge_X_North_Top<<<grid.x,block.x>>>(s,r,As,Ass,AsAs,Nx,Ny,Nz);

	omega_r_edge_Z_South_West<<<grid.z,block.z>>>(s,r,As,Ass,AsAs,Nx,Ny,Nz);
	omega_r_edge_Z_South_East<<<grid.z,block.z>>>(s,r,As,Ass,AsAs,Nx,Ny,Nz);
	omega_r_edge_Z_North_West<<<grid.z,block.z>>>(s,r,As,Ass,AsAs,Nx,Ny,Nz);
	omega_r_edge_Z_North_East<<<grid.z,block.z>>>(s,r,As,Ass,AsAs,Nx,Ny,Nz);

	omega_r_edge_Y_West_Bottom<<<grid.y,block.y>>>(s,r,As,Ass,AsAs,Nx,Ny,Nz);
	omega_r_edge_Y_West_Top<<<grid.y,block.y>>>(s,r,As,Ass,AsAs,Nx,Ny,Nz);
	omega_r_edge_Y_East_Bottom<<<grid.y,block.y>>>(s,r,As,Ass,AsAs,Nx,Ny,Nz);
	omega_r_edge_Y_East_Top<<<grid.y,block.y>>>(s,r,As,Ass,AsAs,Nx,Ny,Nz);

	omega_r_vertex_SWB<<<1,1>>>(s,r,As,Ass,AsAs,Nx,Ny,Nz);
	omega_r_vertex_SWT<<<1,1>>>(s,r,As,Ass,AsAs,Nx,Ny,Nz);
	omega_r_vertex_SEB<<<1,1>>>(s,r,As,Ass,AsAs,Nx,Ny,Nz);
	omega_r_vertex_SET<<<1,1>>>(s,r,As,Ass,AsAs,Nx,Ny,Nz);
	omega_r_vertex_NWB<<<1,1>>>(s,r,As,Ass,AsAs,Nx,Ny,Nz);
	omega_r_vertex_NWT<<<1,1>>>(s,r,As,Ass,AsAs,Nx,Ny,Nz);
	omega_r_vertex_NEB<<<1,1>>>(s,r,As,Ass,AsAs,Nx,Ny,Nz);
	omega_r_vertex_NET<<<1,1>>>(s,r,As,Ass,AsAs,Nx,Ny,Nz);
	cudaDeviceSynchronize();
}
