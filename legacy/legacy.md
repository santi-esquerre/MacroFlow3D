# alpha_3D.cu

```cu
#include <cuda_runtime_api.h>
#define gridXY dim3(grid.x,grid.y)
#define blockXY dim3(block.x,block.y)
#define gridXZ dim3(grid.x,grid.z)
#define blockXZ dim3(block.x,block.z)
#define gridYZ dim3(grid.y,grid.z)
#define blockYZ dim3(block.y,block.z)

//#######################################################################
// 	routine alpha (uses in ConjGrad)
//	plus_minus == true ==> x = x + alpha * y
//	plus_minus == false ==> x = x - alpha * y
//	alpha = <r,z>/<y,P>
//#######################################################################
//
//
//-----------------------------------------------------------------------
__global__ void alpha_int_bottom_top(const double *y, double *x,
const double *rz, const double *yP, bool plus_minus,
const int Nx, const int Ny, const int Nz){
	int ix = threadIdx.x + blockIdx.x*blockDim.x;
	int iy = threadIdx.y + blockIdx.y*blockDim.y;
	if (ix >= Nx-2 || iy >= Ny-2) return;
	int stride = Nx*Ny;
	int in_idx = (ix+1) + (iy+1)*Nx;
	for(int iz = 0; iz<Nz; ++iz){
		if(plus_minus) x[in_idx] += rz[0]/yP[0]*y[in_idx];
		else x[in_idx] -= rz[0]/yP[0]*y[in_idx];
		in_idx+=stride;
	}
}

__global__ void alpha_south_north(const double *y, double *x,
const double *rz, const double *yP, bool plus_minus,
const int Nx, const int Ny, const int Nz){
	int ix = threadIdx.x + blockIdx.x*blockDim.x;
	int iz = threadIdx.y + blockIdx.y*blockDim.y;
	if (ix >= Nx-2 || iz >= Nz-2) return;
	int stride = Nx*Ny;
	int in_idx;
	in_idx = (ix + 1) + (iz + 1)*stride;
	if(plus_minus) x[in_idx] += rz[0]/yP[0]*y[in_idx];
	else x[in_idx] -= rz[0]/yP[0]*y[in_idx];

	in_idx = (ix + 1) + (Ny - 1)*Nx + (iz + 1)*stride;
	if(plus_minus) x[in_idx] += rz[0]/yP[0]*y[in_idx];
	else x[in_idx] -= rz[0]/yP[0]*y[in_idx];
}

__global__ void alpha_este_oeste(const double *y, double *x,
const double *rz, const double *yP, bool plus_minus,
const int Nx, const int Ny, const int Nz){
	int iy = threadIdx.x + blockIdx.x*blockDim.x;
	int iz  = threadIdx.y + blockIdx.y*blockDim.y;
	if (iy >= Ny-2 || iz >= Nz-2) return;
	int stride = Nx*Ny;
	int in_idx;
	in_idx = (Nx - 1) + (iy + 1)*Nx + (iz + 1)*stride;
	if(plus_minus) x[in_idx] += rz[0]/yP[0]*y[in_idx];
	else x[in_idx] -= rz[0]/yP[0]*y[in_idx];

	in_idx = (iy + 1)*Nx + (iz + 1)*stride;
	if(plus_minus) x[in_idx] += rz[0]/yP[0]*y[in_idx];
	else x[in_idx] -= rz[0]/yP[0]*y[in_idx];
}

__global__ void alpha_edge_X_South_Bottom(const double *y, double *x,
const double *rz, const double *yP, bool plus_minus,
const int Nx, const int Ny, const int Nz){
	int ix = threadIdx.x + blockIdx.x*blockDim.x;
	if (ix >= Nx-2) return;
	int stride = Nx*Ny;
	int iy = 0;
	int iz = 0;
	int in_idx = (ix+1) + iy*Nx + iz*stride;
	if(plus_minus) x[in_idx] += rz[0]/yP[0]*y[in_idx];
	else x[in_idx] -= rz[0]/yP[0]*y[in_idx];
}

__global__ void alpha_edge_X_South_Top(const double *y, double *x,
const double *rz, const double *yP, bool plus_minus,
const int Nx, const int Ny, const int Nz){
	int ix = threadIdx.x + blockIdx.x*blockDim.x;
	if (ix >= Nx-2) return;
	int stride = Nx*Ny;
	int iy = 0;
	int iz = Nz-1;
	int in_idx = (ix+1) + iy*Nx + iz*stride;
	if(plus_minus) x[in_idx] += rz[0]/yP[0]*y[in_idx];
	else x[in_idx] -= rz[0]/yP[0]*y[in_idx];
}

__global__ void alpha_edge_X_North_Bottom(const double *y, double *x,
const double *rz, const double *yP, bool plus_minus,
const int Nx, const int Ny, const int Nz){
	int ix = threadIdx.x + blockIdx.x*blockDim.x;
	if (ix >= Nx-2) return;
	int stride = Nx*Ny;
	int iy = Ny-1;
	int iz = 0;
	int in_idx = (ix+1) + iy*Nx + iz*stride;
	if(plus_minus) x[in_idx] += rz[0]/yP[0]*y[in_idx];
	else x[in_idx] -= rz[0]/yP[0]*y[in_idx];
}

__global__ void alpha_edge_X_North_Top(const double *y, double *x,
const double *rz, const double *yP, bool plus_minus,
const int Nx, const int Ny, const int Nz){
	int ix = threadIdx.x + blockIdx.x*blockDim.x;
	if (ix >= Nx-2) return;
	int stride = Nx*Ny;
	int iy = Ny-1;
	int iz = Nz-1;
	int in_idx = (ix+1) + iy*Nx + iz*stride;
	if(plus_minus) x[in_idx] += rz[0]/yP[0]*y[in_idx];
	else x[in_idx] -= rz[0]/yP[0]*y[in_idx];
}

__global__ void alpha_edge_Z_South_West(const double *y, double *x,
const double *rz, const double *yP, bool plus_minus,
const int Nx, const int Ny, const int Nz){
	int iz = threadIdx.x + blockIdx.x*blockDim.x;
	if (iz >= Nz-2) return;
	int stride = Nx*Ny;
	int ix = 0;
	int iy = 0;
	int in_idx = ix + iy*Nx + (iz+1)*stride;
	if(plus_minus) x[in_idx] += rz[0]/yP[0]*y[in_idx];
	else x[in_idx] -= rz[0]/yP[0]*y[in_idx];
}

__global__ void alpha_edge_Z_South_East(const double *y, double *x,
const double *rz, const double *yP, bool plus_minus,
const int Nx, const int Ny, const int Nz){
	int iz = threadIdx.x + blockIdx.x*blockDim.x;
	if (iz >= Nz-2) return;
	int stride = Nx*Ny;
	int ix = Nx-1;
	int iy = 0;
	int in_idx = ix + iy*Nx + (iz+1)*stride;
	if(plus_minus) x[in_idx] += rz[0]/yP[0]*y[in_idx];
	else x[in_idx] -= rz[0]/yP[0]*y[in_idx];
}
__global__ void alpha_edge_Z_North_West(const double *y, double *x,
const double *rz, const double *yP, bool plus_minus,
const int Nx, const int Ny, const int Nz){
	int iz = threadIdx.x + blockIdx.x*blockDim.x;
	if (iz >= Nz-2) return;
	int stride = Nx*Ny;
	int ix = 0;
	int iy = Ny-1;
	int in_idx = ix + iy*Nx + (iz+1)*stride;
	if(plus_minus) x[in_idx] += rz[0]/yP[0]*y[in_idx];
	else x[in_idx] -= rz[0]/yP[0]*y[in_idx];
}

__global__ void alpha_edge_Z_North_East(const double *y, double *x,
const double *rz, const double *yP, bool plus_minus,
const int Nx, const int Ny, const int Nz){
	int iz = threadIdx.x + blockIdx.x*blockDim.x;
	if (iz >= Nz-2) return;
	int stride = Nx*Ny;
	int ix = Nx-1;
	int iy = Ny-1;
	int in_idx = ix + iy*Nx + (iz+1)*stride;
	if(plus_minus) x[in_idx] += rz[0]/yP[0]*y[in_idx];
	else x[in_idx] -= rz[0]/yP[0]*y[in_idx];
}

__global__ void alpha_edge_Y_West_Bottom(const double *y, double *x,
const double *rz, const double *yP, bool plus_minus,
const int Nx, const int Ny, const int Nz){
	int iy = threadIdx.x + blockIdx.x*blockDim.x;
	if (iy >= Ny-2) return;
	int stride = Nx*Ny;
	int ix = 0;
	int iz = 0;
	int in_idx = ix + (iy+1)*Nx + iz*stride;
	if(plus_minus) x[in_idx] += rz[0]/yP[0]*y[in_idx];
	else x[in_idx] -= rz[0]/yP[0]*y[in_idx];
}
__global__ void alpha_edge_Y_West_Top(const double *y, double *x,
const double *rz, const double *yP, bool plus_minus,
const int Nx, const int Ny, const int Nz){
	int iy = threadIdx.x + blockIdx.x*blockDim.x;
	if (iy >= Ny-2) return;
	int stride = Nx*Ny;
	int ix = 0;
	int iz = Nz-1;
	int in_idx = ix + (iy+1)*Nx + iz*stride;
	if(plus_minus) x[in_idx] += rz[0]/yP[0]*y[in_idx];
	else x[in_idx] -= rz[0]/yP[0]*y[in_idx];
}
__global__ void alpha_edge_Y_East_Bottom(const double *y, double *x,
const double *rz, const double *yP, bool plus_minus,
const int Nx, const int Ny, const int Nz){
	int iy = threadIdx.x + blockIdx.x*blockDim.x;
	if (iy >= Ny-2) return;
	int stride = Nx*Ny;
	int ix = Nx-1;
	int iz = 0;
	int in_idx = ix + (iy+1)*Nx + iz*stride;
	if(plus_minus) x[in_idx] += rz[0]/yP[0]*y[in_idx];
	else x[in_idx] -= rz[0]/yP[0]*y[in_idx];
}
__global__ void alpha_edge_Y_East_Top(const double *y, double *x,
const double *rz, const double *yP, bool plus_minus,
const int Nx, const int Ny, const int Nz){
	int iy = threadIdx.x + blockIdx.x*blockDim.x;
	if (iy >= Ny-2) return;
	int stride = Nx*Ny;
	int ix = Nx-1;
	int iz = Nz-1;
	int in_idx = ix + (iy+1)*Nx + iz*stride;
	if(plus_minus) x[in_idx] += rz[0]/yP[0]*y[in_idx];
	else x[in_idx] -= rz[0]/yP[0]*y[in_idx];
}

__global__ void alpha_vertex_SWB(const double *y, double *x,
const double *rz, const double *yP, bool plus_minus,
const int Nx, const int Ny, const int Nz){
	int ix = 0;
	int iy = 0;
	int iz = 0;
	int in_idx = ix + iy*Nx + iz*Nx*Ny;
	if(plus_minus) x[in_idx] += rz[0]/yP[0]*y[in_idx];
	else x[in_idx] -= rz[0]/yP[0]*y[in_idx];
}
__global__ void alpha_vertex_SWT(const double *y, double *x,
const double *rz, const double *yP, bool plus_minus,
const int Nx, const int Ny, const int Nz){
	int ix = 0;
	int iy = 0;
	int iz = Nz-1;
	int in_idx = ix + iy*Nx + iz*Nx*Ny;
	if(plus_minus) x[in_idx] += rz[0]/yP[0]*y[in_idx];
	else x[in_idx] -= rz[0]/yP[0]*y[in_idx];
}
__global__ void alpha_vertex_SEB(const double *y, double *x,
const double *rz, const double *yP, bool plus_minus,
const int Nx, const int Ny, const int Nz){
	int ix = Nx-1;
	int iy = 0;
	int iz = 0;
	int in_idx = ix + iy*Nx + iz*Nx*Ny;
	if(plus_minus) x[in_idx] += rz[0]/yP[0]*y[in_idx];
	else x[in_idx] -= rz[0]/yP[0]*y[in_idx];
}
__global__ void alpha_vertex_SET(const double *y, double *x,
const double *rz, const double *yP, bool plus_minus,
const int Nx, const int Ny, const int Nz){
	int ix = Nx-1;
	int iy = 0;
	int iz = Nz-1;
	int in_idx = ix + iy*Nx + iz*Nx*Ny;
	if(plus_minus) x[in_idx] += rz[0]/yP[0]*y[in_idx];
	else x[in_idx] -= rz[0]/yP[0]*y[in_idx];
}




__global__ void alpha_vertex_NWB(const double *y, double *x,
const double *rz, const double *yP, bool plus_minus,
const int Nx, const int Ny, const int Nz){
	int ix = 0;
	int iy = Ny-1;
	int iz = 0;
	int in_idx = ix + iy*Nx + iz*Nx*Ny;
	if(plus_minus) x[in_idx] += rz[0]/yP[0]*y[in_idx];
	else x[in_idx] -= rz[0]/yP[0]*y[in_idx];
}
__global__ void alpha_vertex_NWT(const double *y, double *x,
const double *rz, const double *yP, bool plus_minus,
const int Nx, const int Ny, const int Nz){
	int ix = 0;
	int iy = Ny-1;
	int iz = Nz-1;
	int in_idx = ix + iy*Nx + iz*Nx*Ny;
	if(plus_minus) x[in_idx] += rz[0]/yP[0]*y[in_idx];
	else x[in_idx] -= rz[0]/yP[0]*y[in_idx];
}
__global__ void alpha_vertex_NEB(const double *y, double *x,
const double *rz, const double *yP, bool plus_minus,
const int Nx, const int Ny, const int Nz){
	int ix = Nx-1;
	int iy = Ny-1;
	int iz = 0;
	int in_idx = ix + iy*Nx + iz*Nx*Ny;
	if(plus_minus) x[in_idx] += rz[0]/yP[0]*y[in_idx];
	else x[in_idx] -= rz[0]/yP[0]*y[in_idx];
}
__global__ void alpha_vertex_NET(const double *y, double *x,
const double *rz, const double *yP, bool plus_minus,
const int Nx, const int Ny, const int Nz){
	int ix = Nx-1;
	int iy = Ny-1;
	int iz = Nz-1;
	int in_idx = ix + iy*Nx + iz*Nx*Ny;
	if(plus_minus) x[in_idx] += rz[0]/yP[0]*y[in_idx];
	else x[in_idx] -= rz[0]/yP[0]*y[in_idx];
}


void alpha(const double *x, double *y,
const double *rz,  const double *yP, bool plus_minus,
const int Nx, const int Ny, const int Nz,
dim3 grid, dim3 block){
	alpha_int_bottom_top<<<gridXY,blockXY>>>(x,y,rz,yP,plus_minus,Nx,Ny,Nz);
	alpha_south_north<<<gridXZ,blockXZ>>>(x,y,rz,yP,plus_minus,Nx,Ny,Nz);
	alpha_este_oeste<<<gridYZ,blockYZ>>>(x,y,rz,yP,plus_minus,Nx,Ny,Nz);

	alpha_edge_X_South_Bottom<<<grid.x,block.x>>>(x,y,rz,yP,plus_minus,Nx,Ny,Nz);
	alpha_edge_X_South_Top<<<grid.x,block.x>>>(x,y,rz,yP,plus_minus,Nx,Ny,Nz);
	alpha_edge_X_North_Bottom<<<grid.x,block.x>>>(x,y,rz,yP,plus_minus,Nx,Ny,Nz);
	alpha_edge_X_North_Top<<<grid.x,block.x>>>(x,y,rz,yP,plus_minus,Nx,Ny,Nz);

	alpha_edge_Z_South_West<<<grid.z,block.z>>>(x,y,rz,yP,plus_minus,Nx,Ny,Nz);
	alpha_edge_Z_South_East<<<grid.z,block.z>>>(x,y,rz,yP,plus_minus,Nx,Ny,Nz);
	alpha_edge_Z_North_West<<<grid.z,block.z>>>(x,y,rz,yP,plus_minus,Nx,Ny,Nz);
	alpha_edge_Z_North_East<<<grid.z,block.z>>>(x,y,rz,yP,plus_minus,Nx,Ny,Nz);

	alpha_edge_Y_West_Bottom<<<grid.y,block.y>>>(x,y,rz,yP,plus_minus,Nx,Ny,Nz);
	alpha_edge_Y_West_Top<<<grid.y,block.y>>>(x,y,rz,yP,plus_minus,Nx,Ny,Nz);
	alpha_edge_Y_East_Bottom<<<grid.y,block.y>>>(x,y,rz,yP,plus_minus,Nx,Ny,Nz);
	alpha_edge_Y_East_Top<<<grid.y,block.y>>>(x,y,rz,yP,plus_minus,Nx,Ny,Nz);


	alpha_vertex_SWB<<<1,1>>>(x,y,rz,yP,plus_minus,Nx,Ny,Nz);
	alpha_vertex_SWT<<<1,1>>>(x,y,rz,yP,plus_minus,Nx,Ny,Nz);
	alpha_vertex_SEB<<<1,1>>>(x,y,rz,yP,plus_minus,Nx,Ny,Nz);
	alpha_vertex_SET<<<1,1>>>(x,y,rz,yP,plus_minus,Nx,Ny,Nz);
	alpha_vertex_NWB<<<1,1>>>(x,y,rz,yP,plus_minus,Nx,Ny,Nz);
	alpha_vertex_NWT<<<1,1>>>(x,y,rz,yP,plus_minus,Nx,Ny,Nz);
	alpha_vertex_NEB<<<1,1>>>(x,y,rz,yP,plus_minus,Nx,Ny,Nz);
	alpha_vertex_NET<<<1,1>>>(x,y,rz,yP,plus_minus,Nx,Ny,Nz);
	cudaDeviceSynchronize();
}



```

# alphaU_3D.cu

```cu

#include <cuda_runtime_api.h>

//#######################################################################
// 	routine alphaU (uses in BiCGStab)
//	s_j = r_j - alpha * AMp
//	alpha = (r_j, r_star) / (A*M*p, r_star)
//#######################################################################
#define gridXY dim3(grid.x,grid.y)
#define blockXY dim3(block.x,block.y)
#define gridXZ dim3(grid.x,grid.z)
#define blockXZ dim3(block.x,block.z)
#define gridYZ dim3(grid.y,grid.z)
#define blockYZ dim3(block.y,block.z)

__global__ void alphaU_int_bottom_top(const double *Ap, const double *r,  double *s, const double *rr_, const double *Apr_, const int Nx, const int Ny, const int Nz){
	int ix = threadIdx.x + blockIdx.x*blockDim.x;
	int iy = threadIdx.y + blockIdx.y*blockDim.y;
	if (ix >= Nx-2 || iy >= Ny-2) return;
	int stride = Nx*Ny;
	int in_idx = (ix+1) + (iy+1)*Nx;
	for(int iz = 0; iz<Nz; ++iz){
		s[in_idx] = r[in_idx] - rr_[0]/Apr_[0]*Ap[in_idx];
		in_idx+=stride;
	}
}

__global__ void alphaU_south_north(const double *Ap, const double *r,  double *s, const double *rr_, const double *Apr_, const int Nx, const int Ny, const int Nz){
	int ix = threadIdx.x + blockIdx.x*blockDim.x;
	int iz = threadIdx.y + blockIdx.y*blockDim.y;
	if (ix >= Nx-2 || iz >= Nz-2) return;
	int stride = Nx*Ny;
	int in_idx;
	in_idx = (ix + 1) + (iz + 1)*stride;
	s[in_idx] = r[in_idx] - rr_[0]/Apr_[0]*Ap[in_idx];

	in_idx = (ix + 1) + (Ny - 1)*Nx + (iz + 1)*stride;
	s[in_idx] = r[in_idx] - rr_[0]/Apr_[0]*Ap[in_idx];
}

__global__ void alphaU_este_oeste(const double *Ap, const double *r,  double *s, const double *rr_, const double *Apr_, const int Nx, const int Ny, const int Nz){
	int iy = threadIdx.x + blockIdx.x*blockDim.x;
	int iz  = threadIdx.y + blockIdx.y*blockDim.y;
	if (iy >= Ny-2 || iz >= Nz-2) return;
	int stride = Nx*Ny;
	int in_idx;
	in_idx = (Nx - 1) + (iy + 1)*Nx + (iz + 1)*stride;
	s[in_idx] = r[in_idx] - rr_[0]/Apr_[0]*Ap[in_idx];

	in_idx = (iy + 1)*Nx + (iz + 1)*stride;
	s[in_idx] = r[in_idx] - rr_[0]/Apr_[0]*Ap[in_idx];
}

__global__ void alphaU_edge_X_South_Bottom(const double *Ap, const double *r,  double *s, const double *rr_, const double *Apr_, const int Nx, const int Ny, const int Nz){
	int ix = threadIdx.x + blockIdx.x*blockDim.x;
	if (ix >= Nx-2) return;
	int stride = Nx*Ny;
	int iy = 0;
	int iz = 0;
	int in_idx = (ix+1) + iy*Nx + iz*stride;
	s[in_idx] = r[in_idx] - rr_[0]/Apr_[0]*Ap[in_idx];
}

__global__ void alphaU_edge_X_South_Top(const double *Ap, const double *r,  double *s, const double *rr_, const double *Apr_, const int Nx, const int Ny, const int Nz){
	int ix = threadIdx.x + blockIdx.x*blockDim.x;
	if (ix >= Nx-2) return;
	int stride = Nx*Ny;
	int iy = 0;
	int iz = Nz-1;
	int in_idx = (ix+1) + iy*Nx + iz*stride;
	s[in_idx] = r[in_idx] - rr_[0]/Apr_[0]*Ap[in_idx];
}

__global__ void alphaU_edge_X_North_Bottom(const double *Ap, const double *r,  double *s, const double *rr_, const double *Apr_, const int Nx, const int Ny, const int Nz){
	int ix = threadIdx.x + blockIdx.x*blockDim.x;
	if (ix >= Nx-2) return;
	int stride = Nx*Ny;
	int iy = Ny-1;
	int iz = 0;
	int in_idx = (ix+1) + iy*Nx + iz*stride;
	s[in_idx] = r[in_idx] - rr_[0]/Apr_[0]*Ap[in_idx];
}

__global__ void alphaU_edge_X_North_Top(const double *Ap, const double *r,  double *s, const double *rr_, const double *Apr_, const int Nx, const int Ny, const int Nz){
	int ix = threadIdx.x + blockIdx.x*blockDim.x;
	if (ix >= Nx-2) return;
	int stride = Nx*Ny;
	int iy = Ny-1;
	int iz = Nz-1;
	int in_idx = (ix+1) + iy*Nx + iz*stride;
	s[in_idx] = r[in_idx] - rr_[0]/Apr_[0]*Ap[in_idx];
}

__global__ void alphaU_edge_Z_South_West(const double *Ap, const double *r,  double *s, const double *rr_, const double *Apr_, const int Nx, const int Ny, const int Nz){
	int iz = threadIdx.x + blockIdx.x*blockDim.x;
	if (iz >= Nz-2) return;
	int stride = Nx*Ny;
	int ix = 0;
	int iy = 0;
	int in_idx = ix + iy*Nx + (iz+1)*stride;
	s[in_idx] = r[in_idx] - rr_[0]/Apr_[0]*Ap[in_idx];
}

__global__ void alphaU_edge_Z_South_East(const double *Ap, const double *r,  double *s, const double *rr_, const double *Apr_, const int Nx, const int Ny, const int Nz){
	int iz = threadIdx.x + blockIdx.x*blockDim.x;
	if (iz >= Nz-2) return;
	int stride = Nx*Ny;
	int ix = Nx-1;
	int iy = 0;
	int in_idx = ix + iy*Nx + (iz+1)*stride;
	s[in_idx] = r[in_idx] - rr_[0]/Apr_[0]*Ap[in_idx];
}
__global__ void alphaU_edge_Z_North_West(const double *Ap, const double *r,  double *s, const double *rr_, const double *Apr_, const int Nx, const int Ny, const int Nz){
	int iz = threadIdx.x + blockIdx.x*blockDim.x;
	if (iz >= Nz-2) return;
	int stride = Nx*Ny;
	int ix = 0;
	int iy = Ny-1;
	int in_idx = ix + iy*Nx + (iz+1)*stride;
	s[in_idx] = r[in_idx] - rr_[0]/Apr_[0]*Ap[in_idx];
}

__global__ void alphaU_edge_Z_North_East(const double *Ap, const double *r,  double *s, const double *rr_, const double *Apr_, const int Nx, const int Ny, const int Nz){
	int iz = threadIdx.x + blockIdx.x*blockDim.x;
	if (iz >= Nz-2) return;
	int stride = Nx*Ny;
	int ix = Nx-1;
	int iy = Ny-1;
	int in_idx = ix + iy*Nx + (iz+1)*stride;
	s[in_idx] = r[in_idx] - rr_[0]/Apr_[0]*Ap[in_idx];
}

__global__ void alphaU_edge_Y_West_Bottom(const double *Ap, const double *r,  double *s, const double *rr_, const double *Apr_, const int Nx, const int Ny, const int Nz){
	int iy = threadIdx.x + blockIdx.x*blockDim.x;
	if (iy >= Ny-2) return;
	int stride = Nx*Ny;
	int ix = 0;
	int iz = 0;
	int in_idx = ix + (iy+1)*Nx + iz*stride;
	s[in_idx] = r[in_idx] - rr_[0]/Apr_[0]*Ap[in_idx];
}
__global__ void alphaU_edge_Y_West_Top(const double *Ap, const double *r,  double *s, const double *rr_, const double *Apr_, const int Nx, const int Ny, const int Nz){
	int iy = threadIdx.x + blockIdx.x*blockDim.x;
	if (iy >= Ny-2) return;
	int stride = Nx*Ny;
	int ix = 0;
	int iz = Nz-1;
	int in_idx = ix + (iy+1)*Nx + iz*stride;
	s[in_idx] = r[in_idx] - rr_[0]/Apr_[0]*Ap[in_idx];
}
__global__ void alphaU_edge_Y_East_Bottom(const double *Ap, const double *r,  double *s, const double *rr_, const double *Apr_, const int Nx, const int Ny, const int Nz){
	int iy = threadIdx.x + blockIdx.x*blockDim.x;
	if (iy >= Ny-2) return;
	int stride = Nx*Ny;
	int ix = Nx-1;
	int iz = 0;
	int in_idx = ix + (iy+1)*Nx + iz*stride;
	s[in_idx] = r[in_idx] - rr_[0]/Apr_[0]*Ap[in_idx];
}
__global__ void alphaU_edge_Y_East_Top(const double *Ap, const double *r,  double *s, const double *rr_, const double *Apr_, const int Nx, const int Ny, const int Nz){
	int iy = threadIdx.x + blockIdx.x*blockDim.x;
	if (iy >= Ny-2) return;
	int stride = Nx*Ny;
	int ix = Nx-1;
	int iz = Nz-1;
	int in_idx = ix + (iy+1)*Nx + iz*stride;
	s[in_idx] = r[in_idx] - rr_[0]/Apr_[0]*Ap[in_idx];
}

__global__ void alphaU_vertex_SWB(const double *Ap, const double *r,  double *s, const double *rr_, const double *Apr_, const int Nx, const int Ny, const int Nz){
	int ix = 0;
	int iy = 0;
	int iz = 0;
	int in_idx = ix + iy*Nx + iz*Nx*Ny;
	s[in_idx] = r[in_idx] - rr_[0]/Apr_[0]*Ap[in_idx];
}
__global__ void alphaU_vertex_SWT(const double *Ap, const double *r,  double *s, const double *rr_, const double *Apr_, const int Nx, const int Ny, const int Nz){
	int ix = 0;
	int iy = 0;
	int iz = Nz-1;
	int in_idx = ix + iy*Nx + iz*Nx*Ny;
	s[in_idx] = r[in_idx] - rr_[0]/Apr_[0]*Ap[in_idx];
}
__global__ void alphaU_vertex_SEB(const double *Ap, const double *r,  double *s, const double *rr_, const double *Apr_, const int Nx, const int Ny, const int Nz){
	int ix = Nx-1;
	int iy = 0;
	int iz = 0;
	int in_idx = ix + iy*Nx + iz*Nx*Ny;
	s[in_idx] = r[in_idx] - rr_[0]/Apr_[0]*Ap[in_idx];
}
__global__ void alphaU_vertex_SET(const double *Ap, const double *r,  double *s, const double *rr_, const double *Apr_, const int Nx, const int Ny, const int Nz){
	int ix = Nx-1;
	int iy = 0;
	int iz = Nz-1;
	int in_idx = ix + iy*Nx + iz*Nx*Ny;
	s[in_idx] = r[in_idx] - rr_[0]/Apr_[0]*Ap[in_idx];
}


__global__ void alphaU_vertex_NWB(const double *Ap, const double *r,  double *s, const double *rr_, const double *Apr_, const int Nx, const int Ny, const int Nz){
	int ix = 0;
	int iy = Ny-1;
	int iz = 0;
	int in_idx = ix + iy*Nx + iz*Nx*Ny;
	s[in_idx] = r[in_idx] - rr_[0]/Apr_[0]*Ap[in_idx];
}
__global__ void alphaU_vertex_NWT(const double *Ap, const double *r,  double *s, const double *rr_, const double *Apr_, const int Nx, const int Ny, const int Nz){
	int ix = 0;
	int iy = Ny-1;
	int iz = Nz-1;
	int in_idx = ix + iy*Nx + iz*Nx*Ny;
	s[in_idx] = r[in_idx] - rr_[0]/Apr_[0]*Ap[in_idx];
}
__global__ void alphaU_vertex_NEB(const double *Ap, const double *r,  double *s, const double *rr_, const double *Apr_, const int Nx, const int Ny, const int Nz){
	int ix = Nx-1;
	int iy = Ny-1;
	int iz = 0;
	int in_idx = ix + iy*Nx + iz*Nx*Ny;
	s[in_idx] = r[in_idx] - rr_[0]/Apr_[0]*Ap[in_idx];
}
__global__ void alphaU_vertex_NET(const double *Ap, const double *r,  double *s, const double *rr_, const double *Apr_, const int Nx, const int Ny, const int Nz){
	int ix = Nx-1;
	int iy = Ny-1;
	int iz = Nz-1;
	int in_idx = ix + iy*Nx + iz*Nx*Ny;
	s[in_idx] = r[in_idx] - rr_[0]/Apr_[0]*Ap[in_idx];
}


void alphaU(const double *Ap, const double *r,  double *s, const double *rr_, const double *Apr_, const int Nx, const int Ny, const int Nz,
	dim3 grid, dim3 block){
	alphaU_int_bottom_top<<<gridXY,blockXY>>>(Ap,r,s,rr_,Apr_,Nx,Ny,Nz);
	alphaU_south_north<<<gridXZ,blockXZ>>>(Ap,r,s,rr_,Apr_,Nx,Ny,Nz);
	alphaU_este_oeste<<<gridYZ,blockYZ>>>(Ap,r,s,rr_,Apr_,Nx,Ny,Nz);

	alphaU_edge_X_South_Bottom<<<grid.x,block.x>>>(Ap,r,s,rr_,Apr_,Nx,Ny,Nz);
	alphaU_edge_X_South_Top<<<grid.x,block.x>>>(Ap,r,s,rr_,Apr_,Nx,Ny,Nz);
	alphaU_edge_X_North_Bottom<<<grid.x,block.x>>>(Ap,r,s,rr_,Apr_,Nx,Ny,Nz);
	alphaU_edge_X_North_Top<<<grid.x,block.x>>>(Ap,r,s,rr_,Apr_,Nx,Ny,Nz);

	alphaU_edge_Z_South_West<<<grid.z,block.z>>>(Ap,r,s,rr_,Apr_,Nx,Ny,Nz);
	alphaU_edge_Z_South_East<<<grid.z,block.z>>>(Ap,r,s,rr_,Apr_,Nx,Ny,Nz);
	alphaU_edge_Z_North_West<<<grid.z,block.z>>>(Ap,r,s,rr_,Apr_,Nx,Ny,Nz);
	alphaU_edge_Z_North_East<<<grid.z,block.z>>>(Ap,r,s,rr_,Apr_,Nx,Ny,Nz);

	alphaU_edge_Y_West_Bottom<<<grid.y,block.y>>>(Ap,r,s,rr_,Apr_,Nx,Ny,Nz);
	alphaU_edge_Y_West_Top<<<grid.y,block.y>>>(Ap,r,s,rr_,Apr_,Nx,Ny,Nz);
	alphaU_edge_Y_East_Bottom<<<grid.y,block.y>>>(Ap,r,s,rr_,Apr_,Nx,Ny,Nz);
	alphaU_edge_Y_East_Top<<<grid.y,block.y>>>(Ap,r,s,rr_,Apr_,Nx,Ny,Nz);

	alphaU_vertex_SWB<<<1,1>>>(Ap,r,s,rr_,Apr_,Nx,Ny,Nz);
	alphaU_vertex_SWT<<<1,1>>>(Ap,r,s,rr_,Apr_,Nx,Ny,Nz);
	alphaU_vertex_SEB<<<1,1>>>(Ap,r,s,rr_,Apr_,Nx,Ny,Nz);
	alphaU_vertex_SET<<<1,1>>>(Ap,r,s,rr_,Apr_,Nx,Ny,Nz);
	alphaU_vertex_NWB<<<1,1>>>(Ap,r,s,rr_,Apr_,Nx,Ny,Nz);
	alphaU_vertex_NWT<<<1,1>>>(Ap,r,s,rr_,Apr_,Nx,Ny,Nz);
	alphaU_vertex_NEB<<<1,1>>>(Ap,r,s,rr_,Apr_,Nx,Ny,Nz);
	alphaU_vertex_NET<<<1,1>>>(Ap,r,s,rr_,Apr_,Nx,Ny,Nz);
	cudaDeviceSynchronize();
}

```

# AXPBY_3D.cu

```cu
#include <cuda_runtime_api.h>
#define gridXY dim3(grid.x,grid.y)
#define blockXY dim3(block.x,block.y)
#define gridXZ dim3(grid.x,grid.z)
#define blockXZ dim3(block.x,block.z)
#define gridYZ dim3(grid.y,grid.z)
#define blockYZ dim3(block.y,block.z)

//#######################################################################
// 	routine AXPBY (uses in ConjGrad)
//	p <- z + AXPBY*p
//	AXPBY = rho / rho_old
//	rho = <r,z>
//#######################################################################
//
//
//-----------------------------------------------------------------------
__global__ void AXPBY_int_bottom_top(const double *x, double *y, double *output,
const double a, const double b,
const int Nx, const int Ny, const int Nz){
	int ix = threadIdx.x + blockIdx.x*blockDim.x;
	int iy = threadIdx.y + blockIdx.y*blockDim.y;
	if (ix >= Nx-2 || iy >= Ny-2) return;
	int stride = Nx*Ny;
	int in_idx = (ix+1) + (iy+1)*Nx;
	for(int iz = 0; iz<Nz; ++iz){
		output[in_idx] = a*x[in_idx]+b*y[in_idx];
		in_idx+=stride;
	}
}

__global__ void AXPBY_south_north(const double *x, double *y, double *output,
const double a, const double b,
const int Nx, const int Ny, const int Nz){
	int ix = threadIdx.x + blockIdx.x*blockDim.x;
	int iz = threadIdx.y + blockIdx.y*blockDim.y;
	if (ix >= Nx-2 || iz >= Nz-2) return;
	int stride = Nx*Ny;
	int in_idx;
	in_idx = (ix + 1) + (iz + 1)*stride;
	output[in_idx] = a*x[in_idx]+b*y[in_idx];

	in_idx = (ix + 1) + (Ny - 1)*Nx + (iz + 1)*stride;
	output[in_idx] = a*x[in_idx]+b*y[in_idx];
}

__global__ void AXPBY_este_oeste(const double *x, double *y, double *output,
const double a, const double b,
const int Nx, const int Ny, const int Nz){
	int iy = threadIdx.x + blockIdx.x*blockDim.x;
	int iz  = threadIdx.y + blockIdx.y*blockDim.y;
	if (iy >= Ny-2 || iz >= Nz-2) return;
	int stride = Nx*Ny;
	int in_idx;
	in_idx = (Nx - 1) + (iy + 1)*Nx + (iz + 1)*stride;
	output[in_idx] = a*x[in_idx]+b*y[in_idx];

	in_idx = (iy + 1)*Nx + (iz + 1)*stride;
	output[in_idx] = a*x[in_idx]+b*y[in_idx];
}

__global__ void AXPBY_edge_X_South_Bottom(const double *x, double *y, double *output,
const double a, const double b,
const int Nx, const int Ny, const int Nz){
	int ix = threadIdx.x + blockIdx.x*blockDim.x;
	if (ix >= Nx-2) return;
	int stride = Nx*Ny;
	int iy = 0;
	int iz = 0;
	int in_idx = (ix+1) + iy*Nx + iz*stride;
	output[in_idx] = a*x[in_idx]+b*y[in_idx];
}

__global__ void AXPBY_edge_X_South_Top(const double *x, double *y, double *output,
const double a, const double b,
const int Nx, const int Ny, const int Nz){
	int ix = threadIdx.x + blockIdx.x*blockDim.x;
	if (ix >= Nx-2) return;
	int stride = Nx*Ny;
	int iy = 0;
	int iz = Nz-1;
	int in_idx = (ix+1) + iy*Nx + iz*stride;
	output[in_idx] = a*x[in_idx]+b*y[in_idx];
}

__global__ void AXPBY_edge_X_North_Bottom(const double *x, double *y, double *output,
const double a, const double b,
const int Nx, const int Ny, const int Nz){
	int ix = threadIdx.x + blockIdx.x*blockDim.x;
	if (ix >= Nx-2) return;
	int stride = Nx*Ny;
	int iy = Ny-1;
	int iz = 0;
	int in_idx = (ix+1) + iy*Nx + iz*stride;
	output[in_idx] = a*x[in_idx]+b*y[in_idx];
}

__global__ void AXPBY_edge_X_North_Top(const double *x, double *y, double *output,
const double a, const double b,
const int Nx, const int Ny, const int Nz){
	int ix = threadIdx.x + blockIdx.x*blockDim.x;
	if (ix >= Nx-2) return;
	int stride = Nx*Ny;
	int iy = Ny-1;
	int iz = Nz-1;
	int in_idx = (ix+1) + iy*Nx + iz*stride;
	output[in_idx] = a*x[in_idx]+b*y[in_idx];
}

__global__ void AXPBY_edge_Z_South_West(const double *x, double *y, double *output,
const double a, const double b,
const int Nx, const int Ny, const int Nz){
	int iz = threadIdx.x + blockIdx.x*blockDim.x;
	if (iz >= Nz-2) return;
	int stride = Nx*Ny;
	int ix = 0;
	int iy = 0;
	int in_idx = ix + iy*Nx + (iz+1)*stride;
	output[in_idx] = a*x[in_idx]+b*y[in_idx];
}

__global__ void AXPBY_edge_Z_South_East(const double *x, double *y, double *output,
const double a, const double b,
const int Nx, const int Ny, const int Nz){
	int iz = threadIdx.x + blockIdx.x*blockDim.x;
	if (iz >= Nz-2) return;
	int stride = Nx*Ny;
	int ix = Nx-1;
	int iy = 0;
	int in_idx = ix + iy*Nx + (iz+1)*stride;
	output[in_idx] = a*x[in_idx]+b*y[in_idx];
}
__global__ void AXPBY_edge_Z_North_West(const double *x, double *y, double *output,
const double a, const double b,
const int Nx, const int Ny, const int Nz){
	int iz = threadIdx.x + blockIdx.x*blockDim.x;
	if (iz >= Nz-2) return;
	int stride = Nx*Ny;
	int ix = 0;
	int iy = Ny-1;
	int in_idx = ix + iy*Nx + (iz+1)*stride;
	output[in_idx] = a*x[in_idx]+b*y[in_idx];
}

__global__ void AXPBY_edge_Z_North_East(const double *x, double *y, double *output,
const double a, const double b,
const int Nx, const int Ny, const int Nz){
	int iz = threadIdx.x + blockIdx.x*blockDim.x;
	if (iz >= Nz-2) return;
	int stride = Nx*Ny;
	int ix = Nx-1;
	int iy = Ny-1;
	int in_idx = ix + iy*Nx + (iz+1)*stride;
	output[in_idx] = a*x[in_idx]+b*y[in_idx];
}

__global__ void AXPBY_edge_Y_West_Bottom(const double *x, double *y, double *output,
const double a, const double b,
const int Nx, const int Ny, const int Nz){
	int iy = threadIdx.x + blockIdx.x*blockDim.x;
	if (iy >= Ny-2) return;
	int stride = Nx*Ny;
	int ix = 0;
	int iz = 0;
	int in_idx = ix + (iy+1)*Nx + iz*stride;
	output[in_idx] = a*x[in_idx]+b*y[in_idx];
}
__global__ void AXPBY_edge_Y_West_Top(const double *x, double *y, double *output,
const double a, const double b,
const int Nx, const int Ny, const int Nz){
	int iy = threadIdx.x + blockIdx.x*blockDim.x;
	if (iy >= Ny-2) return;
	int stride = Nx*Ny;
	int ix = 0;
	int iz = Nz-1;
	int in_idx = ix + (iy+1)*Nx + iz*stride;
	output[in_idx] = a*x[in_idx]+b*y[in_idx];
}
__global__ void AXPBY_edge_Y_East_Bottom(const double *x, double *y, double *output,
const double a, const double b,
const int Nx, const int Ny, const int Nz){
	int iy = threadIdx.x + blockIdx.x*blockDim.x;
	if (iy >= Ny-2) return;
	int stride = Nx*Ny;
	int ix = Nx-1;
	int iz = 0;
	int in_idx = ix + (iy+1)*Nx + iz*stride;
	output[in_idx] = a*x[in_idx]+b*y[in_idx];
}
__global__ void AXPBY_edge_Y_East_Top(const double *x, double *y, double *output,
const double a, const double b,
const int Nx, const int Ny, const int Nz){
	int iy = threadIdx.x + blockIdx.x*blockDim.x;
	if (iy >= Ny-2) return;
	int stride = Nx*Ny;
	int ix = Nx-1;
	int iz = Nz-1;
	int in_idx = ix + (iy+1)*Nx + iz*stride;
	output[in_idx] = a*x[in_idx]+b*y[in_idx];
}

__global__ void AXPBY_vertex_SWB(const double *x, double *y, double *output,
const double a, const double b,
const int Nx, const int Ny, const int Nz){
	int ix = 0;
	int iy = 0;
	int iz = 0;
	int in_idx = ix + iy*Nx + iz*Nx*Ny;
	output[in_idx] = a*x[in_idx]+b*y[in_idx];
}
__global__ void AXPBY_vertex_SWT(const double *x, double *y, double *output,
const double a, const double b,
const int Nx, const int Ny, const int Nz){
	int ix = 0;
	int iy = 0;
	int iz = Nz-1;
	int in_idx = ix + iy*Nx + iz*Nx*Ny;
	output[in_idx] = a*x[in_idx]+b*y[in_idx];
}
__global__ void AXPBY_vertex_SEB(const double *x, double *y, double *output,
const double a, const double b,
const int Nx, const int Ny, const int Nz){
	int ix = Nx-1;
	int iy = 0;
	int iz = 0;
	int in_idx = ix + iy*Nx + iz*Nx*Ny;
	output[in_idx] = a*x[in_idx]+b*y[in_idx];
}
__global__ void AXPBY_vertex_SET(const double *x, double *y, double *output,
const double a, const double b,
const int Nx, const int Ny, const int Nz){
	int ix = Nx-1;
	int iy = 0;
	int iz = Nz-1;
	int in_idx = ix + iy*Nx + iz*Nx*Ny;
	output[in_idx] = a*x[in_idx]+b*y[in_idx];
}




__global__ void AXPBY_vertex_NWB(const double *x, double *y, double *output,
const double a, const double b,
const int Nx, const int Ny, const int Nz){
	int ix = 0;
	int iy = Ny-1;
	int iz = 0;
	int in_idx = ix + iy*Nx + iz*Nx*Ny;
	output[in_idx] = a*x[in_idx]+b*y[in_idx];
}
__global__ void AXPBY_vertex_NWT(const double *x, double *y, double *output,
const double a, const double b,
const int Nx, const int Ny, const int Nz){
	int ix = 0;
	int iy = Ny-1;
	int iz = Nz-1;
	int in_idx = ix + iy*Nx + iz*Nx*Ny;
	output[in_idx] = a*x[in_idx]+b*y[in_idx];
}
__global__ void AXPBY_vertex_NEB(const double *x, double *y, double *output,
const double a, const double b,
const int Nx, const int Ny, const int Nz){
	int ix = Nx-1;
	int iy = Ny-1;
	int iz = 0;
	int in_idx = ix + iy*Nx + iz*Nx*Ny;
	output[in_idx] = a*x[in_idx]+b*y[in_idx];
}
__global__ void AXPBY_vertex_NET(const double *x, double *y, double *output,
const double a, const double b,
const int Nx, const int Ny, const int Nz){
	int ix = Nx-1;
	int iy = Ny-1;
	int iz = Nz-1;
	int in_idx = ix + iy*Nx + iz*Nx*Ny;
	output[in_idx] = a*x[in_idx]+b*y[in_idx];
}


void AXPBY(const double *x, double *y, double *output,
const double a, const double b,
const int Nx, const int Ny, const int Nz,
dim3 grid, dim3 block){
	AXPBY_int_bottom_top<<<gridXY,blockXY>>>(x,y,output,a,b,Nx,Ny,Nz);
	AXPBY_south_north<<<gridXZ,blockXZ>>>(x,y,output,a,b,Nx,Ny,Nz);
	AXPBY_este_oeste<<<gridYZ,blockYZ>>>(x,y,output,a,b,Nx,Ny,Nz);

	AXPBY_edge_X_South_Bottom<<<grid.x,block.x>>>(x,y,output,a,b,Nx,Ny,Nz);
	AXPBY_edge_X_South_Top<<<grid.x,block.x>>>(x,y,output,a,b,Nx,Ny,Nz);
	AXPBY_edge_X_North_Bottom<<<grid.x,block.x>>>(x,y,output,a,b,Nx,Ny,Nz);
	AXPBY_edge_X_North_Top<<<grid.x,block.x>>>(x,y,output,a,b,Nx,Ny,Nz);

	AXPBY_edge_Z_South_West<<<grid.z,block.z>>>(x,y,output,a,b,Nx,Ny,Nz);
	AXPBY_edge_Z_South_East<<<grid.z,block.z>>>(x,y,output,a,b,Nx,Ny,Nz);
	AXPBY_edge_Z_North_West<<<grid.z,block.z>>>(x,y,output,a,b,Nx,Ny,Nz);
	AXPBY_edge_Z_North_East<<<grid.z,block.z>>>(x,y,output,a,b,Nx,Ny,Nz);

	AXPBY_edge_Y_West_Bottom<<<grid.y,block.y>>>(x,y,output,a,b,Nx,Ny,Nz);
	AXPBY_edge_Y_West_Top<<<grid.y,block.y>>>(x,y,output,a,b,Nx,Ny,Nz);
	AXPBY_edge_Y_East_Bottom<<<grid.y,block.y>>>(x,y,output,a,b,Nx,Ny,Nz);
	AXPBY_edge_Y_East_Top<<<grid.y,block.y>>>(x,y,output,a,b,Nx,Ny,Nz);


	AXPBY_vertex_SWB<<<1,1>>>(x,y,output,a,b,Nx,Ny,Nz);
	AXPBY_vertex_SWT<<<1,1>>>(x,y,output,a,b,Nx,Ny,Nz);
	AXPBY_vertex_SEB<<<1,1>>>(x,y,output,a,b,Nx,Ny,Nz);
	AXPBY_vertex_SET<<<1,1>>>(x,y,output,a,b,Nx,Ny,Nz);
	AXPBY_vertex_NWB<<<1,1>>>(x,y,output,a,b,Nx,Ny,Nz);
	AXPBY_vertex_NWT<<<1,1>>>(x,y,output,a,b,Nx,Ny,Nz);
	AXPBY_vertex_NEB<<<1,1>>>(x,y,output,a,b,Nx,Ny,Nz);
	AXPBY_vertex_NET<<<1,1>>>(x,y,output,a,b,Nx,Ny,Nz);
	cudaDeviceSynchronize();
}

```

# beta_3D.cu

```cu
#include <cuda_runtime_api.h>
#define gridXY dim3(grid.x,grid.y)
#define blockXY dim3(block.x,block.y)
#define gridXZ dim3(grid.x,grid.z)
#define blockXZ dim3(block.x,block.z)
#define gridYZ dim3(grid.y,grid.z)
#define blockYZ dim3(block.y,block.z)

//#######################################################################
// 	routine beta (uses in ConjGrad)
//	p <- z + beta*p
//	beta = rho / rho_old
//	rho = <r,z>
//#######################################################################
//
//
//-----------------------------------------------------------------------
__global__ void beta_int_bottom_top(const double *x, double *y,
const double *rz, const double *rz_old,
const int Nx, const int Ny, const int Nz){
	int ix = threadIdx.x + blockIdx.x*blockDim.x;
	int iy = threadIdx.y + blockIdx.y*blockDim.y;
	if (ix >= Nx-2 || iy >= Ny-2) return;
	int stride = Nx*Ny;
	int in_idx = (ix+1) + (iy+1)*Nx;
	for(int iz = 0; iz<Nz; ++iz){
		y[in_idx] = x[in_idx]+rz[0]/rz_old[0]*y[in_idx];
		in_idx+=stride;
	}
}

__global__ void beta_south_north(const double *x, double *y,
const double *rz, const double *rz_old,
const int Nx, const int Ny, const int Nz){
	int ix = threadIdx.x + blockIdx.x*blockDim.x;
	int iz = threadIdx.y + blockIdx.y*blockDim.y;
	if (ix >= Nx-2 || iz >= Nz-2) return;
	int stride = Nx*Ny;
	int in_idx;
	in_idx = (ix + 1) + (iz + 1)*stride;
	y[in_idx] = x[in_idx]+rz[0]/rz_old[0]*y[in_idx];

	in_idx = (ix + 1) + (Ny - 1)*Nx + (iz + 1)*stride;
	y[in_idx] = x[in_idx]+rz[0]/rz_old[0]*y[in_idx];
}

__global__ void beta_este_oeste(const double *x, double *y,
const double *rz, const double *rz_old,
const int Nx, const int Ny, const int Nz){
	int iy = threadIdx.x + blockIdx.x*blockDim.x;
	int iz  = threadIdx.y + blockIdx.y*blockDim.y;
	if (iy >= Ny-2 || iz >= Nz-2) return;
	int stride = Nx*Ny;
	int in_idx;
	in_idx = (Nx - 1) + (iy + 1)*Nx + (iz + 1)*stride;
	y[in_idx] = x[in_idx]+rz[0]/rz_old[0]*y[in_idx];

	in_idx = (iy + 1)*Nx + (iz + 1)*stride;
	y[in_idx] = x[in_idx]+rz[0]/rz_old[0]*y[in_idx];
}

__global__ void beta_edge_X_South_Bottom(const double *x, double *y,
const double *rz, const double *rz_old,
const int Nx, const int Ny, const int Nz){
	int ix = threadIdx.x + blockIdx.x*blockDim.x;
	if (ix >= Nx-2) return;
	int stride = Nx*Ny;
	int iy = 0;
	int iz = 0;
	int in_idx = (ix+1) + iy*Nx + iz*stride;
	y[in_idx] = x[in_idx]+rz[0]/rz_old[0]*y[in_idx];
}

__global__ void beta_edge_X_South_Top(const double *x, double *y,
const double *rz, const double *rz_old,
const int Nx, const int Ny, const int Nz){
	int ix = threadIdx.x + blockIdx.x*blockDim.x;
	if (ix >= Nx-2) return;
	int stride = Nx*Ny;
	int iy = 0;
	int iz = Nz-1;
	int in_idx = (ix+1) + iy*Nx + iz*stride;
	y[in_idx] = x[in_idx]+rz[0]/rz_old[0]*y[in_idx];
}

__global__ void beta_edge_X_North_Bottom(const double *x, double *y,
const double *rz, const double *rz_old,
const int Nx, const int Ny, const int Nz){
	int ix = threadIdx.x + blockIdx.x*blockDim.x;
	if (ix >= Nx-2) return;
	int stride = Nx*Ny;
	int iy = Ny-1;
	int iz = 0;
	int in_idx = (ix+1) + iy*Nx + iz*stride;
	y[in_idx] = x[in_idx]+rz[0]/rz_old[0]*y[in_idx];
}

__global__ void beta_edge_X_North_Top(const double *x, double *y,
const double *rz, const double *rz_old,
const int Nx, const int Ny, const int Nz){
	int ix = threadIdx.x + blockIdx.x*blockDim.x;
	if (ix >= Nx-2) return;
	int stride = Nx*Ny;
	int iy = Ny-1;
	int iz = Nz-1;
	int in_idx = (ix+1) + iy*Nx + iz*stride;
	y[in_idx] = x[in_idx]+rz[0]/rz_old[0]*y[in_idx];
}

__global__ void beta_edge_Z_South_West(const double *x, double *y,
const double *rz, const double *rz_old,
const int Nx, const int Ny, const int Nz){
	int iz = threadIdx.x + blockIdx.x*blockDim.x;
	if (iz >= Nz-2) return;
	int stride = Nx*Ny;
	int ix = 0;
	int iy = 0;
	int in_idx = ix + iy*Nx + (iz+1)*stride;
	y[in_idx] = x[in_idx]+rz[0]/rz_old[0]*y[in_idx];
}

__global__ void beta_edge_Z_South_East(const double *x, double *y,
const double *rz, const double *rz_old,
const int Nx, const int Ny, const int Nz){
	int iz = threadIdx.x + blockIdx.x*blockDim.x;
	if (iz >= Nz-2) return;
	int stride = Nx*Ny;
	int ix = Nx-1;
	int iy = 0;
	int in_idx = ix + iy*Nx + (iz+1)*stride;
	y[in_idx] = x[in_idx]+rz[0]/rz_old[0]*y[in_idx];
}
__global__ void beta_edge_Z_North_West(const double *x, double *y,
const double *rz, const double *rz_old,
const int Nx, const int Ny, const int Nz){
	int iz = threadIdx.x + blockIdx.x*blockDim.x;
	if (iz >= Nz-2) return;
	int stride = Nx*Ny;
	int ix = 0;
	int iy = Ny-1;
	int in_idx = ix + iy*Nx + (iz+1)*stride;
	y[in_idx] = x[in_idx]+rz[0]/rz_old[0]*y[in_idx];
}

__global__ void beta_edge_Z_North_East(const double *x, double *y,
const double *rz, const double *rz_old,
const int Nx, const int Ny, const int Nz){
	int iz = threadIdx.x + blockIdx.x*blockDim.x;
	if (iz >= Nz-2) return;
	int stride = Nx*Ny;
	int ix = Nx-1;
	int iy = Ny-1;
	int in_idx = ix + iy*Nx + (iz+1)*stride;
	y[in_idx] = x[in_idx]+rz[0]/rz_old[0]*y[in_idx];
}

__global__ void beta_edge_Y_West_Bottom(const double *x, double *y,
const double *rz, const double *rz_old,
const int Nx, const int Ny, const int Nz){
	int iy = threadIdx.x + blockIdx.x*blockDim.x;
	if (iy >= Ny-2) return;
	int stride = Nx*Ny;
	int ix = 0;
	int iz = 0;
	int in_idx = ix + (iy+1)*Nx + iz*stride;
	y[in_idx] = x[in_idx]+rz[0]/rz_old[0]*y[in_idx];
}
__global__ void beta_edge_Y_West_Top(const double *x, double *y,
const double *rz, const double *rz_old,
const int Nx, const int Ny, const int Nz){
	int iy = threadIdx.x + blockIdx.x*blockDim.x;
	if (iy >= Ny-2) return;
	int stride = Nx*Ny;
	int ix = 0;
	int iz = Nz-1;
	int in_idx = ix + (iy+1)*Nx + iz*stride;
	y[in_idx] = x[in_idx]+rz[0]/rz_old[0]*y[in_idx];
}
__global__ void beta_edge_Y_East_Bottom(const double *x, double *y,
const double *rz, const double *rz_old,
const int Nx, const int Ny, const int Nz){
	int iy = threadIdx.x + blockIdx.x*blockDim.x;
	if (iy >= Ny-2) return;
	int stride = Nx*Ny;
	int ix = Nx-1;
	int iz = 0;
	int in_idx = ix + (iy+1)*Nx + iz*stride;
	y[in_idx] = x[in_idx]+rz[0]/rz_old[0]*y[in_idx];
}
__global__ void beta_edge_Y_East_Top(const double *x, double *y,
const double *rz, const double *rz_old,
const int Nx, const int Ny, const int Nz){
	int iy = threadIdx.x + blockIdx.x*blockDim.x;
	if (iy >= Ny-2) return;
	int stride = Nx*Ny;
	int ix = Nx-1;
	int iz = Nz-1;
	int in_idx = ix + (iy+1)*Nx + iz*stride;
	y[in_idx] = x[in_idx]+rz[0]/rz_old[0]*y[in_idx];
}

__global__ void beta_vertex_SWB(const double *x, double *y,
const double *rz, const double *rz_old,
const int Nx, const int Ny, const int Nz){
	int ix = 0;
	int iy = 0;
	int iz = 0;
	int in_idx = ix + iy*Nx + iz*Nx*Ny;
	y[in_idx] = x[in_idx]+rz[0]/rz_old[0]*y[in_idx];
}
__global__ void beta_vertex_SWT(const double *x, double *y,
const double *rz, const double *rz_old,
const int Nx, const int Ny, const int Nz){
	int ix = 0;
	int iy = 0;
	int iz = Nz-1;
	int in_idx = ix + iy*Nx + iz*Nx*Ny;
	y[in_idx] = x[in_idx]+rz[0]/rz_old[0]*y[in_idx];
}
__global__ void beta_vertex_SEB(const double *x, double *y,
const double *rz, const double *rz_old,
const int Nx, const int Ny, const int Nz){
	int ix = Nx-1;
	int iy = 0;
	int iz = 0;
	int in_idx = ix + iy*Nx + iz*Nx*Ny;
	y[in_idx] = x[in_idx]+rz[0]/rz_old[0]*y[in_idx];
}
__global__ void beta_vertex_SET(const double *x, double *y,
const double *rz, const double *rz_old,
const int Nx, const int Ny, const int Nz){
	int ix = Nx-1;
	int iy = 0;
	int iz = Nz-1;
	int in_idx = ix + iy*Nx + iz*Nx*Ny;
	y[in_idx] = x[in_idx]+rz[0]/rz_old[0]*y[in_idx];
}




__global__ void beta_vertex_NWB(const double *x, double *y,
const double *rz, const double *rz_old,
const int Nx, const int Ny, const int Nz){
	int ix = 0;
	int iy = Ny-1;
	int iz = 0;
	int in_idx = ix + iy*Nx + iz*Nx*Ny;
	y[in_idx] = x[in_idx]+rz[0]/rz_old[0]*y[in_idx];
}
__global__ void beta_vertex_NWT(const double *x, double *y,
const double *rz, const double *rz_old,
const int Nx, const int Ny, const int Nz){
	int ix = 0;
	int iy = Ny-1;
	int iz = Nz-1;
	int in_idx = ix + iy*Nx + iz*Nx*Ny;
	y[in_idx] = x[in_idx]+rz[0]/rz_old[0]*y[in_idx];
}
__global__ void beta_vertex_NEB(const double *x, double *y,
const double *rz, const double *rz_old,
const int Nx, const int Ny, const int Nz){
	int ix = Nx-1;
	int iy = Ny-1;
	int iz = 0;
	int in_idx = ix + iy*Nx + iz*Nx*Ny;
	y[in_idx] = x[in_idx]+rz[0]/rz_old[0]*y[in_idx];
}
__global__ void beta_vertex_NET(const double *x, double *y,
const double *rz, const double *rz_old,
const int Nx, const int Ny, const int Nz){
	int ix = Nx-1;
	int iy = Ny-1;
	int iz = Nz-1;
	int in_idx = ix + iy*Nx + iz*Nx*Ny;
	y[in_idx] = x[in_idx]+rz[0]/rz_old[0]*y[in_idx];
}


void beta(const double *x, double *y,
const double *rz, const double *rz_old,
const int Nx, const int Ny, const int Nz,
dim3 grid, dim3 block){
	beta_int_bottom_top<<<gridXY,blockXY>>>(x,y,rz,rz_old,Nx,Ny,Nz);
	beta_south_north<<<gridXZ,blockXZ>>>(x,y,rz,rz_old,Nx,Ny,Nz);
	beta_este_oeste<<<gridYZ,blockYZ>>>(x,y,rz,rz_old,Nx,Ny,Nz);

	beta_edge_X_South_Bottom<<<grid.x,block.x>>>(x,y,rz,rz_old,Nx,Ny,Nz);
	beta_edge_X_South_Top<<<grid.x,block.x>>>(x,y,rz,rz_old,Nx,Ny,Nz);
	beta_edge_X_North_Bottom<<<grid.x,block.x>>>(x,y,rz,rz_old,Nx,Ny,Nz);
	beta_edge_X_North_Top<<<grid.x,block.x>>>(x,y,rz,rz_old,Nx,Ny,Nz);

	beta_edge_Z_South_West<<<grid.z,block.z>>>(x,y,rz,rz_old,Nx,Ny,Nz);
	beta_edge_Z_South_East<<<grid.z,block.z>>>(x,y,rz,rz_old,Nx,Ny,Nz);
	beta_edge_Z_North_West<<<grid.z,block.z>>>(x,y,rz,rz_old,Nx,Ny,Nz);
	beta_edge_Z_North_East<<<grid.z,block.z>>>(x,y,rz,rz_old,Nx,Ny,Nz);

	beta_edge_Y_West_Bottom<<<grid.y,block.y>>>(x,y,rz,rz_old,Nx,Ny,Nz);
	beta_edge_Y_West_Top<<<grid.y,block.y>>>(x,y,rz,rz_old,Nx,Ny,Nz);
	beta_edge_Y_East_Bottom<<<grid.y,block.y>>>(x,y,rz,rz_old,Nx,Ny,Nz);
	beta_edge_Y_East_Top<<<grid.y,block.y>>>(x,y,rz,rz_old,Nx,Ny,Nz);


	beta_vertex_SWB<<<1,1>>>(x,y,rz,rz_old,Nx,Ny,Nz);
	beta_vertex_SWT<<<1,1>>>(x,y,rz,rz_old,Nx,Ny,Nz);
	beta_vertex_SEB<<<1,1>>>(x,y,rz,rz_old,Nx,Ny,Nz);
	beta_vertex_SET<<<1,1>>>(x,y,rz,rz_old,Nx,Ny,Nz);
	beta_vertex_NWB<<<1,1>>>(x,y,rz,rz_old,Nx,Ny,Nz);
	beta_vertex_NWT<<<1,1>>>(x,y,rz,rz_old,Nx,Ny,Nz);
	beta_vertex_NEB<<<1,1>>>(x,y,rz,rz_old,Nx,Ny,Nz);
	beta_vertex_NET<<<1,1>>>(x,y,rz,rz_old,Nx,Ny,Nz);
	cudaDeviceSynchronize();
}

```

# betaU_3D.cu

```cu
#include <cuda_runtime_api.h>
#define gridXY dim3(grid.x,grid.y)
#define blockXY dim3(block.x,block.y)
#define gridXZ dim3(grid.x,grid.z)
#define blockXZ dim3(block.x,block.z)
#define gridYZ dim3(grid.y,grid.z)
#define blockYZ dim3(block.y,block.z)

//#######################################################################
// 	routine betaU (uses in BiCGStab)
//	p_{j+1} = r_{j+1} + beta*(p_j - omega*A*M*p)
//	beta_j = (r_{j+1}, r_star) / (r_j, r_star) * (alpha/omega)
//	omega = (AMs, s) / (AMs, AMs)
//  alpha = (r_j, r_star) / (A*M*p, r_star)
//#######################################################################
//-----------------------------------------------------------------------
__global__ void betaU_int_bottom_top(const double *r, double *p, const double *Ap, const double *rr_, const double *Apr_, const double *Ass, const double *AsAs,
const int Nx, const int Ny, const int Nz){
	int ix = threadIdx.x + blockIdx.x*blockDim.x;
	int iy = threadIdx.y + blockIdx.y*blockDim.y;
	if (ix >= Nx-2 || iy >= Ny-2) return;
	int stride = Nx*Ny;
	int in_idx = (ix+1) + (iy+1)*Nx;
	for(int iz = 0; iz<Nz; ++iz){
		p[in_idx] = r[in_idx] + (rr_[0]/Apr_[0])*(AsAs[0]/Ass[0])*(p[in_idx]-Ass[0]/AsAs[0]*Ap[in_idx]);
		in_idx+=stride;
	}
}

__global__ void betaU_south_north(const double *r, double *p, const double *Ap, const double *rr_, const double *Apr_, const double *Ass, const double *AsAs,
const int Nx, const int Ny, const int Nz){
	int ix = threadIdx.x + blockIdx.x*blockDim.x;
	int iz = threadIdx.y + blockIdx.y*blockDim.y;
	if (ix >= Nx-2 || iz >= Nz-2) return;
	int stride = Nx*Ny;
	int in_idx;
	in_idx = (ix + 1) + (iz + 1)*stride;
	p[in_idx] = r[in_idx] + (rr_[0]/Apr_[0])*(AsAs[0]/Ass[0])*(p[in_idx]-Ass[0]/AsAs[0]*Ap[in_idx]);

	in_idx = (ix + 1) + (Ny - 1)*Nx + (iz + 1)*stride;
	p[in_idx] = r[in_idx] + (rr_[0]/Apr_[0])*(AsAs[0]/Ass[0])*(p[in_idx]-Ass[0]/AsAs[0]*Ap[in_idx]);
}

__global__ void betaU_este_oeste(const double *r, double *p, const double *Ap, const double *rr_, const double *Apr_, const double *Ass, const double *AsAs,
const int Nx, const int Ny, const int Nz){
	int iy = threadIdx.x + blockIdx.x*blockDim.x;
	int iz  = threadIdx.y + blockIdx.y*blockDim.y;
	if (iy >= Ny-2 || iz >= Nz-2) return;
	int stride = Nx*Ny;
	int in_idx;
	in_idx = (Nx - 1) + (iy + 1)*Nx + (iz + 1)*stride;
	p[in_idx] = r[in_idx] + (rr_[0]/Apr_[0])*(AsAs[0]/Ass[0])*(p[in_idx]-Ass[0]/AsAs[0]*Ap[in_idx]);

	in_idx = (iy + 1)*Nx + (iz + 1)*stride;
	p[in_idx] = r[in_idx] + (rr_[0]/Apr_[0])*(AsAs[0]/Ass[0])*(p[in_idx]-Ass[0]/AsAs[0]*Ap[in_idx]);
}

__global__ void betaU_edge_X_South_Bottom(const double *r, double *p, const double *Ap, const double *rr_, const double *Apr_, const double *Ass, const double *AsAs,
const int Nx, const int Ny, const int Nz){
	int ix = threadIdx.x + blockIdx.x*blockDim.x;
	if (ix >= Nx-2) return;
	int stride = Nx*Ny;
	int iy = 0;
	int iz = 0;
	int in_idx = (ix+1) + iy*Nx + iz*stride;
	p[in_idx] = r[in_idx] + (rr_[0]/Apr_[0])*(AsAs[0]/Ass[0])*(p[in_idx]-Ass[0]/AsAs[0]*Ap[in_idx]);
}

__global__ void betaU_edge_X_South_Top(const double *r, double *p, const double *Ap, const double *rr_, const double *Apr_, const double *Ass, const double *AsAs,
const int Nx, const int Ny, const int Nz){
	int ix = threadIdx.x + blockIdx.x*blockDim.x;
	if (ix >= Nx-2) return;
	int stride = Nx*Ny;
	int iy = 0;
	int iz = Nz-1;
	int in_idx = (ix+1) + iy*Nx + iz*stride;
	p[in_idx] = r[in_idx] + (rr_[0]/Apr_[0])*(AsAs[0]/Ass[0])*(p[in_idx]-Ass[0]/AsAs[0]*Ap[in_idx]);
}

__global__ void betaU_edge_X_North_Bottom(const double *r, double *p, const double *Ap, const double *rr_, const double *Apr_, const double *Ass, const double *AsAs,
const int Nx, const int Ny, const int Nz){
	int ix = threadIdx.x + blockIdx.x*blockDim.x;
	if (ix >= Nx-2) return;
	int stride = Nx*Ny;
	int iy = Ny-1;
	int iz = 0;
	int in_idx = (ix+1) + iy*Nx + iz*stride;
	p[in_idx] = r[in_idx] + (rr_[0]/Apr_[0])*(AsAs[0]/Ass[0])*(p[in_idx]-Ass[0]/AsAs[0]*Ap[in_idx]);
}

__global__ void betaU_edge_X_North_Top(const double *r, double *p, const double *Ap, const double *rr_, const double *Apr_, const double *Ass, const double *AsAs,
const int Nx, const int Ny, const int Nz){
	int ix = threadIdx.x + blockIdx.x*blockDim.x;
	if (ix >= Nx-2) return;
	int stride = Nx*Ny;
	int iy = Ny-1;
	int iz = Nz-1;
	int in_idx = (ix+1) + iy*Nx + iz*stride;
	p[in_idx] = r[in_idx] + (rr_[0]/Apr_[0])*(AsAs[0]/Ass[0])*(p[in_idx]-Ass[0]/AsAs[0]*Ap[in_idx]);
}

__global__ void betaU_edge_Z_South_West(const double *r, double *p, const double *Ap, const double *rr_, const double *Apr_, const double *Ass, const double *AsAs,
const int Nx, const int Ny, const int Nz){
	int iz = threadIdx.x + blockIdx.x*blockDim.x;
	if (iz >= Nz-2) return;
	int stride = Nx*Ny;
	int ix = 0;
	int iy = 0;
	int in_idx = ix + iy*Nx + (iz+1)*stride;
	p[in_idx] = r[in_idx] + (rr_[0]/Apr_[0])*(AsAs[0]/Ass[0])*(p[in_idx]-Ass[0]/AsAs[0]*Ap[in_idx]);
}

__global__ void betaU_edge_Z_South_East(const double *r, double *p, const double *Ap, const double *rr_, const double *Apr_, const double *Ass, const double *AsAs,
const int Nx, const int Ny, const int Nz){
	int iz = threadIdx.x + blockIdx.x*blockDim.x;
	if (iz >= Nz-2) return;
	int stride = Nx*Ny;
	int ix = Nx-1;
	int iy = 0;
	int in_idx = ix + iy*Nx + (iz+1)*stride;
	p[in_idx] = r[in_idx] + (rr_[0]/Apr_[0])*(AsAs[0]/Ass[0])*(p[in_idx]-Ass[0]/AsAs[0]*Ap[in_idx]);
}
__global__ void betaU_edge_Z_North_West(const double *r, double *p, const double *Ap, const double *rr_, const double *Apr_, const double *Ass, const double *AsAs,
const int Nx, const int Ny, const int Nz){
	int iz = threadIdx.x + blockIdx.x*blockDim.x;
	if (iz >= Nz-2) return;
	int stride = Nx*Ny;
	int ix = 0;
	int iy = Ny-1;
	int in_idx = ix + iy*Nx + (iz+1)*stride;
	p[in_idx] = r[in_idx] + (rr_[0]/Apr_[0])*(AsAs[0]/Ass[0])*(p[in_idx]-Ass[0]/AsAs[0]*Ap[in_idx]);
}

__global__ void betaU_edge_Z_North_East(const double *r, double *p, const double *Ap, const double *rr_, const double *Apr_, const double *Ass, const double *AsAs,
const int Nx, const int Ny, const int Nz){
	int iz = threadIdx.x + blockIdx.x*blockDim.x;
	if (iz >= Nz-2) return;
	int stride = Nx*Ny;
	int ix = Nx-1;
	int iy = Ny-1;
	int in_idx = ix + iy*Nx + (iz+1)*stride;
	p[in_idx] = r[in_idx] + (rr_[0]/Apr_[0])*(AsAs[0]/Ass[0])*(p[in_idx]-Ass[0]/AsAs[0]*Ap[in_idx]);
}

__global__ void betaU_edge_Y_West_Bottom(const double *r, double *p, const double *Ap, const double *rr_, const double *Apr_, const double *Ass, const double *AsAs,
const int Nx, const int Ny, const int Nz){
	int iy = threadIdx.x + blockIdx.x*blockDim.x;
	if (iy >= Ny-2) return;
	int stride = Nx*Ny;
	int ix = 0;
	int iz = 0;
	int in_idx = ix + (iy+1)*Nx + iz*stride;
	p[in_idx] = r[in_idx] + (rr_[0]/Apr_[0])*(AsAs[0]/Ass[0])*(p[in_idx]-Ass[0]/AsAs[0]*Ap[in_idx]);
}
__global__ void betaU_edge_Y_West_Top(const double *r, double *p, const double *Ap, const double *rr_, const double *Apr_, const double *Ass, const double *AsAs,
const int Nx, const int Ny, const int Nz){
	int iy = threadIdx.x + blockIdx.x*blockDim.x;
	if (iy >= Ny-2) return;
	int stride = Nx*Ny;
	int ix = 0;
	int iz = Nz-1;
	int in_idx = ix + (iy+1)*Nx + iz*stride;
	p[in_idx] = r[in_idx] + (rr_[0]/Apr_[0])*(AsAs[0]/Ass[0])*(p[in_idx]-Ass[0]/AsAs[0]*Ap[in_idx]);
}
__global__ void betaU_edge_Y_East_Bottom(const double *r, double *p, const double *Ap, const double *rr_, const double *Apr_, const double *Ass, const double *AsAs,
const int Nx, const int Ny, const int Nz){
	int iy = threadIdx.x + blockIdx.x*blockDim.x;
	if (iy >= Ny-2) return;
	int stride = Nx*Ny;
	int ix = Nx-1;
	int iz = 0;
	int in_idx = ix + (iy+1)*Nx + iz*stride;
	p[in_idx] = r[in_idx] + (rr_[0]/Apr_[0])*(AsAs[0]/Ass[0])*(p[in_idx]-Ass[0]/AsAs[0]*Ap[in_idx]);
}
__global__ void betaU_edge_Y_East_Top(const double *r, double *p, const double *Ap, const double *rr_, const double *Apr_, const double *Ass, const double *AsAs,
const int Nx, const int Ny, const int Nz){
	int iy = threadIdx.x + blockIdx.x*blockDim.x;
	if (iy >= Ny-2) return;
	int stride = Nx*Ny;
	int ix = Nx-1;
	int iz = Nz-1;
	int in_idx = ix + (iy+1)*Nx + iz*stride;
	p[in_idx] = r[in_idx] + (rr_[0]/Apr_[0])*(AsAs[0]/Ass[0])*(p[in_idx]-Ass[0]/AsAs[0]*Ap[in_idx]);
}

__global__ void betaU_vertex_SWB(const double *r, double *p, const double *Ap, const double *rr_, const double *Apr_, const double *Ass, const double *AsAs,
const int Nx, const int Ny, const int Nz){
	int ix = 0;
	int iy = 0;
	int iz = 0;
	int in_idx = ix + iy*Nx + iz*Nx*Ny;
	p[in_idx] = r[in_idx] + (rr_[0]/Apr_[0])*(AsAs[0]/Ass[0])*(p[in_idx]-Ass[0]/AsAs[0]*Ap[in_idx]);
}
__global__ void betaU_vertex_SWT(const double *r, double *p, const double *Ap, const double *rr_, const double *Apr_, const double *Ass, const double *AsAs,
const int Nx, const int Ny, const int Nz){
	int ix = 0;
	int iy = 0;
	int iz = Nz-1;
	int in_idx = ix + iy*Nx + iz*Nx*Ny;
	p[in_idx] = r[in_idx] + (rr_[0]/Apr_[0])*(AsAs[0]/Ass[0])*(p[in_idx]-Ass[0]/AsAs[0]*Ap[in_idx]);
}
__global__ void betaU_vertex_SEB(const double *r, double *p, const double *Ap, const double *rr_, const double *Apr_, const double *Ass, const double *AsAs,
const int Nx, const int Ny, const int Nz){
	int ix = Nx-1;
	int iy = 0;
	int iz = 0;
	int in_idx = ix + iy*Nx + iz*Nx*Ny;
	p[in_idx] = r[in_idx] + (rr_[0]/Apr_[0])*(AsAs[0]/Ass[0])*(p[in_idx]-Ass[0]/AsAs[0]*Ap[in_idx]);
}
__global__ void betaU_vertex_SET(const double *r, double *p, const double *Ap, const double *rr_, const double *Apr_, const double *Ass, const double *AsAs,
const int Nx, const int Ny, const int Nz){
	int ix = Nx-1;
	int iy = 0;
	int iz = Nz-1;
	int in_idx = ix + iy*Nx + iz*Nx*Ny;
	p[in_idx] = r[in_idx] + (rr_[0]/Apr_[0])*(AsAs[0]/Ass[0])*(p[in_idx]-Ass[0]/AsAs[0]*Ap[in_idx]);
}




__global__ void betaU_vertex_NWB(const double *r, double *p, const double *Ap, const double *rr_, const double *Apr_, const double *Ass, const double *AsAs,
const int Nx, const int Ny, const int Nz){
	int ix = 0;
	int iy = Ny-1;
	int iz = 0;
	int in_idx = ix + iy*Nx + iz*Nx*Ny;
	p[in_idx] = r[in_idx] + (rr_[0]/Apr_[0])*(AsAs[0]/Ass[0])*(p[in_idx]-Ass[0]/AsAs[0]*Ap[in_idx]);
}
__global__ void betaU_vertex_NWT(const double *r, double *p, const double *Ap, const double *rr_, const double *Apr_, const double *Ass, const double *AsAs,
const int Nx, const int Ny, const int Nz){
	int ix = 0;
	int iy = Ny-1;
	int iz = Nz-1;
	int in_idx = ix + iy*Nx + iz*Nx*Ny;
	p[in_idx] = r[in_idx] + (rr_[0]/Apr_[0])*(AsAs[0]/Ass[0])*(p[in_idx]-Ass[0]/AsAs[0]*Ap[in_idx]);
}
__global__ void betaU_vertex_NEB(const double *r, double *p, const double *Ap, const double *rr_, const double *Apr_, const double *Ass, const double *AsAs,
const int Nx, const int Ny, const int Nz){
	int ix = Nx-1;
	int iy = Ny-1;
	int iz = 0;
	int in_idx = ix + iy*Nx + iz*Nx*Ny;
	p[in_idx] = r[in_idx] + (rr_[0]/Apr_[0])*(AsAs[0]/Ass[0])*(p[in_idx]-Ass[0]/AsAs[0]*Ap[in_idx]);
}
__global__ void betaU_vertex_NET(const double *r, double *p, const double *Ap, const double *rr_, const double *Apr_, const double *Ass, const double *AsAs,
const int Nx, const int Ny, const int Nz){
	int ix = Nx-1;
	int iy = Ny-1;
	int iz = Nz-1;
	int in_idx = ix + iy*Nx + iz*Nx*Ny;
	p[in_idx] = r[in_idx] + (rr_[0]/Apr_[0])*(AsAs[0]/Ass[0])*(p[in_idx]-Ass[0]/AsAs[0]*Ap[in_idx]);
}


void betaU(const double *r, double *p, const double *Ap, const double *rr_, const double *Apr_, const double *Ass, const double *AsAs,
	const int Nx, const int Ny, const int Nz,
	dim3 grid, dim3 block){

	betaU_int_bottom_top<<<gridXY,blockXY>>>(r,p,Ap,rr_,Apr_,Ass,AsAs,Nx,Ny,Nz);
	betaU_south_north<<<gridXZ,blockXZ>>>(r,p,Ap,rr_,Apr_,Ass,AsAs,Nx,Ny,Nz);
	betaU_este_oeste<<<gridYZ,blockYZ>>>(r,p,Ap,rr_,Apr_,Ass,AsAs,Nx,Ny,Nz);

	betaU_edge_X_South_Bottom<<<grid.x,block.x>>>(r,p,Ap,rr_,Apr_,Ass,AsAs,Nx,Ny,Nz);
	betaU_edge_X_South_Top<<<grid.x,block.x>>>(r,p,Ap,rr_,Apr_,Ass,AsAs,Nx,Ny,Nz);
	betaU_edge_X_North_Bottom<<<grid.x,block.x>>>(r,p,Ap,rr_,Apr_,Ass,AsAs,Nx,Ny,Nz);
	betaU_edge_X_North_Top<<<grid.x,block.x>>>(r,p,Ap,rr_,Apr_,Ass,AsAs,Nx,Ny,Nz);

	betaU_edge_Z_South_West<<<grid.z,block.z>>>(r,p,Ap,rr_,Apr_,Ass,AsAs,Nx,Ny,Nz);
	betaU_edge_Z_South_East<<<grid.z,block.z>>>(r,p,Ap,rr_,Apr_,Ass,AsAs,Nx,Ny,Nz);
	betaU_edge_Z_North_West<<<grid.z,block.z>>>(r,p,Ap,rr_,Apr_,Ass,AsAs,Nx,Ny,Nz);
	betaU_edge_Z_North_East<<<grid.z,block.z>>>(r,p,Ap,rr_,Apr_,Ass,AsAs,Nx,Ny,Nz);

	betaU_edge_Y_West_Bottom<<<grid.y,block.y>>>(r,p,Ap,rr_,Apr_,Ass,AsAs,Nx,Ny,Nz);
	betaU_edge_Y_West_Top<<<grid.y,block.y>>>(r,p,Ap,rr_,Apr_,Ass,AsAs,Nx,Ny,Nz);
	betaU_edge_Y_East_Bottom<<<grid.y,block.y>>>(r,p,Ap,rr_,Apr_,Ass,AsAs,Nx,Ny,Nz);
	betaU_edge_Y_East_Top<<<grid.y,block.y>>>(r,p,Ap,rr_,Apr_,Ass,AsAs,Nx,Ny,Nz);

	betaU_vertex_SWB<<<1,1>>>(r,p,Ap,rr_,Apr_,Ass,AsAs,Nx,Ny,Nz);
	betaU_vertex_SWT<<<1,1>>>(r,p,Ap,rr_,Apr_,Ass,AsAs,Nx,Ny,Nz);
	betaU_vertex_SEB<<<1,1>>>(r,p,Ap,rr_,Apr_,Ass,AsAs,Nx,Ny,Nz);
	betaU_vertex_SET<<<1,1>>>(r,p,Ap,rr_,Apr_,Ass,AsAs,Nx,Ny,Nz);
	betaU_vertex_NWB<<<1,1>>>(r,p,Ap,rr_,Apr_,Ass,AsAs,Nx,Ny,Nz);
	betaU_vertex_NWT<<<1,1>>>(r,p,Ap,rr_,Apr_,Ass,AsAs,Nx,Ny,Nz);
	betaU_vertex_NEB<<<1,1>>>(r,p,Ap,rr_,Apr_,Ass,AsAs,Nx,Ny,Nz);
	betaU_vertex_NET<<<1,1>>>(r,p,Ap,rr_,Apr_,Ass,AsAs,Nx,Ny,Nz);
	cudaDeviceSynchronize();
}
```

# BiCGStab.cu

```cu
#include <iostream>
#include "header/linear_operator.h"
// for residual evolution monitoring
void resMonitor(int levelPrint, int cccc, double *rr_0_h, double &res_relative_old, double *rr_new_h, int i){
	if(levelPrint==2 && cccc==1){
		std::cout<< "<b,b>: "<<*rr_new_h<<std::endl;
		std::cout<<"Iters       ||r||_2     conv.rate  ||r||_2/||b||_2"<<std::endl;
	    std::cout<<"-----    ------------   ---------  ------------"<<std::endl;
	}
	if(levelPrint==2 && cccc==2){
		std::cout.width(5); std::cout << std::right<<i<<"    ";
		std::cout<<std::right<<std::scientific<<pow(*rr_new_h,0.5)<<"    "<<std::right<<std::fixed<<(pow(*rr_new_h,0.5)/pow(*rr_0_h,0.5))/res_relative_old<<"    ";
		res_relative_old = pow(*rr_new_h,0.5)/pow(*rr_0_h,0.5);
		std::cout<<std::right<<std::scientific<<res_relative_old<<std::endl;
	}
	if(levelPrint==1 && cccc==0){
	std::cout.width(5); std::cout << std::right<<i<<"    ";
	std::cout<<std::right<<std::scientific<<pow(*rr_new_h,0.5)<<std::endl;
	}
}

// WARNING!
// Whenever possible the following is used:
// b (RHS) vector is used as the auxiliary vector P
int solver_BiCGStab(Matrix_t &M, Matrix_t &precond, blas_t &BLAS,
	double *x, double *s, double *r, double *r_,
	double *Mp, double *Ms,double *AMs, double *AMp, double *RHS,
	double tol_abs, double tol_rel, int iter_max, int print_monitor){

	double *rr_, *AMpr_, *AMss, *AMsAMs, *rr_new;
	double *rr__new;
	cudaMalloc(&rr_new    , sizeof(double));
	cudaMalloc(&rr_    , sizeof(double));
	cudaMalloc(&rr__new    , sizeof(double));
	cudaMalloc(&AMpr_		 , sizeof(double));
	cudaMalloc(&AMss    	 , sizeof(double));
	cudaMalloc(&AMsAMs		 , sizeof(double));
	double *rr_0_h, *rr_new_h, *ss;
	rr_0_h = new double[1]; rr_new_h = new double[1]; ss = new double[1];
	M.stencil(r,x); // y <- Ax
	BLAS.AXPBY3D(RHS,r,r,1.0,-1.0); // r <- b - A*x
	BLAS.copyVector_d2d(RHS,r); //p <- r
	// BLAS.copyVector_d2d(P,r); //p <- r
	BLAS.copyVector_d2d(r_,r); //r_ = hat(r) // r_star <- r
	BLAS.Ddot(r_,r,rr_);
	BLAS.copyScalar_d2h(rr_0_h,rr_);
	*rr_new_h = *rr_0_h;
	int i = 0;
	double res_relative_old = 1;// auxiliar for printing
	resMonitor(print_monitor,1,rr_0_h,res_relative_old,rr_new_h,i); // if print_monitor==2 (print table)
	while( ( pow(*rr_new_h,0.5) > tol_abs + pow(*rr_0_h,0.5) * tol_rel ) && i<iter_max){
		precond.stencil(Mp,RHS); // Mp = M*p
		// precond.stencil(Mp,P); // Mp = M*p
		M.stencil(AMp,Mp); // AMp = A*Mp
		// alpha = (r_j, r_star) / (A*M*p, r_star)
		BLAS.Ddot(AMp,r_,AMpr_); //AMpr_ = (A*M*p,r_star)
		// s_j = r_j - alpha * AMp /////rr_ = (r_j, r_star)
		BLAS.alphaU3D(AMp,r,s,rr_,AMpr_); //Ap = A*M*p
		BLAS.Ddot(s,s,AMsAMs); //AMsAMs = ss = <s,s>;
		BLAS.copyScalar_d2h(ss,AMsAMs);
		if( ( pow(*ss,0.5) < tol_abs + pow(*rr_0_h,0.5) * tol_rel ) || (i>=iter_max)   ){
			BLAS.alpha3D(Mp,x,rr_,AMpr_,true); // x += alpha*M*p_j
			break;
		}
		precond.stencil(Ms,s); // Ms = M*s_j
		M.stencil(AMs,Ms); // AMs = A*Ms
		// omega = (AMs, s) / (AMs, AMs)
		BLAS.Ddot(AMs,s,AMss); //Ass, //(AMs, s)
		BLAS.Ddot(AMs,AMs,AMsAMs); //AsAs, //(AMs, AMs)
		BLAS.omegaX3D(Mp,x,Ms,rr_,AMpr_,AMss,AMsAMs); //x_{j+1} = x_j + alpha*M*p_j + omega*M*s_j
		BLAS.omegaR3D(s,r,AMs,AMss,AMsAMs); //r_{j+1} = s_j - omega*A*M*s
		// beta_j = (r_{j+1}, r_star) / (r_j, r_star) * (alpha/omega)
		BLAS.Ddot(r_,r,rr__new); //rr__new = (r_{j+1}, r_star)
		// p_{j+1} = r_{j+1} + beta*(p_j - omega*A*M*p)
		BLAS.betaU3D(r,RHS,AMp,rr__new,AMpr_,AMss,AMsAMs);
		// BLAS.betaU3D(r,P,AMp,rr__new,AMpr_,AMss,AMsAMs);
		//r_r_star_old = r_r_star_new;
		BLAS.copyScalar_d2d(rr_,rr__new);
		BLAS.Ddot(r,r,rr_new);
		BLAS.copyScalar_d2h(rr_new_h, rr_new);
		i+=1;
		// if print_monitor==2 (print table)
		resMonitor(print_monitor,2,rr_0_h,res_relative_old,rr_new_h,i);
	}
	// if print_monitor==1 (print finalAbsRes, iterNumber)
	resMonitor(print_monitor,0,rr_0_h,res_relative_old,rr_new_h,i);

	cudaFree(rr_new);
    cudaFree(rr_);
	cudaFree(rr__new);
    cudaFree(AMpr_);
    cudaFree(AMss);
	cudaFree(AMsAMs);

	delete [] rr_0_h;
	delete [] rr_new_h;
	delete [] ss;
	return i;
}
```

# CCMG_V_cycle.cu

```cu
#include "header/routines_CCMG.h"
#include "header/MG_struct.h"
#include "cublas_v2.h"
#include "header/linear_operator.h"

#define boundaryCond dirichletBottom,dirichletTop,dirichletSouth,dirichletNorth,dirichletWest,dirichletEast,pin1stCell

void V_cycle(double **e_pre, double **r, double **rr, double **K,
	dim3 *grid, dim3 *block,
	int dirichletBottom, int dirichletTop,
	int dirichletSouth, int dirichletNorth,
	int dirichletWest, int dirichletEast, bool pin1stCell,
	MG_levels MG, int l, cublasHandle_t handle, int ratioX, int ratioY, int ratioZ, double Ly){
	int Nx = pow(2,l)*ratioX;
	int Ny = pow(2,l)*ratioY;
	int Nz = pow(2,l)*ratioZ;
	double h = Ly/(double)Ny;// carasteristic length is Ly
	double dxdx = h*h;
	double h_H = h*2.0;
	cudaMemset(e_pre[l],0,sizeof(double)*Nx*Ny*Nz);
	//el_pre <- smooth(0,rl); //rrl <- rl-A*el_pre;
	smooth_GSRB(e_pre[l],r[l],rr[l],K[l],dxdx,Nx,Ny,Nz,boundaryCond,MG.npre,true,grid[l],block[l]);
	//rrl1 <- restrict(rrl)
	restriction(r[l-1],rr[l],Nx/2,Ny/2,Nz/2,grid[l-1],block[l-1]);
	//solve coarse system
	if (l==2) SolveCoarseSystemGSRB(e_pre[l-1],r[l-1],rr[l-1],K[l-1],h_H*h_H,Nx/2,Ny/2,Nz/2,10000,grid[l-1],block[l-1],handle,boundaryCond);
	//e[l-1] <- MGCYCLE(rl1)
	else V_cycle(e_pre,r,rr,K,grid,block,boundaryCond,MG,l-1,handle,ratioX,ratioY,ratioZ,Ly);
	//el_cgc = prolong(el1); //el_sum = el_pre+el_cgc
	prolongation(e_pre[l],e_pre[l-1],Nx,Ny,Nz,grid[l],block[l]);
	//el <- smooth(el_sum,rl)
	smooth_GSRB(e_pre[l],r[l],rr[l],K[l],dxdx,Nx,Ny,Nz,boundaryCond,MG.npos,false,grid[l],block[l]);
}

int solver_CG(Matrix_t &M, Matrix_t &precond, blas_t &BLAS, double *x, double *P, double *r, double *z, double *y, double tol_abs, double tol_rel, int iter_max, int print_monitor);

void V_cycle2(double **e_pre, double **r, double **rr, double **K,
	dim3 *grid, dim3 *block,
	int dirichletBottom, int dirichletTop,
	int dirichletSouth, int dirichletNorth,
	int dirichletWest, int dirichletEast, bool pin1stCell,
	MG_levels MG, int l, cublasHandle_t handle, int ratioX, int ratioY, int ratioZ, double Ly,
	Matrix_t &M, Matrix_t &precond, blas_t &BLAS, double *aux, double *aux2
	){
	int Nx = pow(2,l)*ratioX;
	int Ny = pow(2,l)*ratioY;
	int Nz = pow(2,l)*ratioZ;
	double h = Ly/(double)Ny;
	double dxdx = h*h;
	// double h_H = h*2.0;
	cudaMemset(e_pre[l],0,sizeof(double)*Nx*Ny*Nz);
	//el_pre <- smooth(0,rl); //rrl <- rl-A*el_pre;
	smooth_GSRB(e_pre[l],r[l],rr[l],K[l],dxdx,Nx,Ny,Nz,boundaryCond,MG.npre,true,grid[l],block[l]);
	//rrl1 <- restrict(rrl)
	restriction(r[l-1],rr[l],Nx/2,Ny/2,Nz/2,grid[l-1],block[l-1]);
	//solve coarse system
	if (l==1)solver_CG(M,precond,BLAS,e_pre[l-1],rr[l-1],aux,r[l-1],aux2,1e-16,0,1000,0);
	//e[l-1] <- MGCYCLE(rl1)
	else V_cycle2(e_pre,r,rr,K,grid,block,boundaryCond,MG,l-1,handle,ratioX,ratioY,ratioZ,Ly,M,precond,BLAS,aux,aux2);
	//el_cgc = prolong(el1); //el_sum = el_pre+el_cgc
	prolongation(e_pre[l],e_pre[l-1],Nx,Ny,Nz,grid[l],block[l]);
	//el <- smooth(el_sum,rl)
	smooth_GSRB(e_pre[l],r[l],rr[l],K[l],dxdx,Nx,Ny,Nz,boundaryCond,MG.npos,false,grid[l],block[l]);
}

void Precond_CCMG_Vcycle(double *e0fine, const double *rfine,
	double **e, double **r, double **rr, double **K,
	dim3 *grid, dim3 *block,
	int dirichletBottom, int dirichletTop,
	int dirichletSouth, int dirichletNorth,
	int dirichletWest, int dirichletEast, bool pin1stCell,
	int Nx, int Ny, int Nz, MG_levels MG, cublasHandle_t handle, int ratioX, int ratioY, int ratioZ, double Ly){
	int l = MG.L-1;
	double h = Ly/(double)Ny;
	double dxdx = h*h;
	smooth_GSRB(e0fine,rfine,rr[l],K[l],dxdx,Nx,Ny,Nz,boundaryCond,MG.npre,true,grid[l],block[l]);
	restriction(r[l-1],rr[l],Nx/2,Ny/2,Nz/2,grid[l-1],block[l-1]);
	V_cycle(e,r,rr,K,grid,block,boundaryCond,MG,l-1,handle,ratioX,ratioY,ratioZ,Ly);
	prolongation(e0fine,e[l-1],Nx,Ny,Nz,grid[l],block[l]);
	smooth_GSRB(e0fine,rfine,rr[l],K[l],dxdx,Nx,Ny,Nz,boundaryCond,MG.npos,false,grid[l],block[l]);
}

void Precond_CCMG_Vcycle2(double *e0fine, const double *rfine,
	double **e, double **r, double **rr, double **K,
	dim3 *grid, dim3 *block,
	int dirichletBottom, int dirichletTop,
	int dirichletSouth, int dirichletNorth,
	int dirichletWest, int dirichletEast, bool pin1stCell,
	int Nx, int Ny, int Nz, MG_levels MG, cublasHandle_t handle, int ratioX, int ratioY, int ratioZ, double Ly,
	Matrix_t &M, Matrix_t &precond, blas_t &BLAS, double *aux, double *aux2
	){
	int l = MG.L-1;
	double h = Ly/(double)Ny;
	double dxdx = h*h;
	smooth_GSRB(e0fine,rfine,rr[l],K[l],dxdx,Nx,Ny,Nz,boundaryCond,MG.npre,true,grid[l],block[l]);
	restriction(r[l-1],rr[l],Nx/2,Ny/2,Nz/2,grid[l-1],block[l-1]);
	V_cycle2(e,r,rr,K,grid,block,boundaryCond,MG,l-1,handle,ratioX,ratioY,ratioZ,Ly,M,precond,BLAS,aux,aux2);
	prolongation(e0fine,e[l-1],Nx,Ny,Nz,grid[l],block[l]);
	smooth_GSRB(e0fine,rfine,rr[l],K[l],dxdx,Nx,Ny,Nz,boundaryCond,MG.npos,false,grid[l],block[l]);
}

// // grid defined from  z-direction
// void V_cycle3(double **e_pre, double **r, double **rr,
// 	dim3 *gridXY, dim3 *blockXY, dim3 *gridXZ, dim3 *blockXZ, dim3 *gridYZ, dim3 *blockYZ,
// 	bool dirichletBottom, bool dirichletTop,
// 	bool dirichletSouth, bool dirichletNorth,
// 	bool dirichletWest, bool dirichletEast, bool pin1stCell,
// 	MG_levels MG, int level, cublasHandle_t handle, int ratioZX, int ratioZY, double Lz,
// 	Matrix_t &M, Matrix_t &precond, blas_t &BLAS, double *aux, double *aux2
// 	){
// 	int Nz = pow(2,level);
// 	int Nx = Nz*ratioZX;
// 	int Ny = Nz*ratioZY;

// 	double h = Lz/(double)Nz;
// 	double dxdx = h*h;
// 	// double h_H = h*2.0;
// 	cudaMemset(e_pre[level],0,sizeof(double)*Nx*Ny*Nz);
// 	//el_pre <- smooth(0,rl); //rrl <- rl-A*el_pre;
// 	smooth_GSRB(e_pre[level],r[level],rr[level],dxdx,Nx,Ny,Nz,boundaryCond,MG.npre,true,GRID3D_L);
// 	//rrl1 <- restrict(rrl)
// 	restriction(r[level-1],rr[level],Nx/2,Ny/2,Nz/2,gridXY[level-1],blockXY[level-1]);
// 	//solve coarse system
// 	if (level==2)solver_CG(M,precond,BLAS,e_pre[level-1],rr[level-1],aux,r[level-1],aux2,1e-16,0,10000,0);
// 	//e[level-1] <- MGCYCLE(rl1)
// 	else V_cycle3(e_pre,r,rr,gridXY,blockXY,gridXZ,blockXZ,gridYZ,blockYZ,boundaryCond,MG,level-1,handle,ratioZX,ratioZY,Lz,M,precond,BLAS,aux,aux2);
// 	//el_cgc = prolong(el1); //el_sum = el_pre+el_cgc
// 	prolongation(e_pre[level],e_pre[level-1],Nx,Ny,Nz,GRID3D_L);
// 	//el <- smooth(el_sum,rl)
// 	smooth_GSRB(e_pre[level],r[level],rr[level],dxdx,Nx,Ny,Nz,boundaryCond,MG.npos,false,GRID3D_L);
// }

// void Precond_CCMG_Vcycle3(double *e0fine, const double *rfine,
// 	double **e, double **r, double **rr,
// 	dim3 *gridXY, dim3 *blockXY, dim3 *gridXZ, dim3 *blockXZ, dim3 *gridYZ, dim3 *blockYZ,
// 	bool dirichletBottom, bool dirichletTop,
// 	bool dirichletSouth, bool dirichletNorth,
// 	bool dirichletWest, bool dirichletEast, bool pin1stCell,
// 	int Nx, int Ny, int Nz, MG_levels MG, cublasHandle_t handle, int ratioZX, int ratioZY, double Lz,
// 	Matrix_t &M, Matrix_t &precond, blas_t &BLAS, double *aux, double *aux2
// 	){
// 	int level = MG.L-1;
// 	double h = Lz/(double)Nz;
// 	double dxdx = h*h;
// 	smooth_GSRB(e0fine,rfine,rr[level],dxdx,Nx,Ny,Nz,boundaryCond,MG.npre,true,GRID3D_L);
// 	restriction(r[level-1],rr[level],Nx/2,Ny/2,Nz/2,gridXY[level-1],blockXY[level-1]);
// 	V_cycle3(e,r,rr,gridXY,blockXY,gridXZ,blockXZ,gridYZ,blockYZ,boundaryCond,MG,level-1,handle,ratioZX,ratioZY,Lz,M,precond,BLAS,aux,aux2);
// 	prolongation(e0fine,e[level-1],Nx,Ny,Nz,GRID3D_L);
// 	smooth_GSRB(e0fine,rfine,rr[level],dxdx,Nx,Ny,Nz,boundaryCond,MG.npos,false,GRID3D_L);
// }

// int solver_BiCGStab(Matrix_t &M, Matrix_t &precond, blas_t &BLAS,
// 	double *x, double *s, double *r, double *r_,
// 	double *Mp, double *Ms,double *AMs, double *AMp, double *RHS,
// 	double tol_abs, double tol_rel, int iter_max, int print_monitor);

// // grid defined from  z-direction
// void V_cycle4(double **e_pre, double **r, double **rr,
// 	dim3 *gridXY, dim3 *blockXY, dim3 *gridXZ, dim3 *blockXZ, dim3 *gridYZ, dim3 *blockYZ,
// 	bool dirichletBottom, bool dirichletTop,
// 	bool dirichletSouth, bool dirichletNorth,
// 	bool dirichletWest, bool dirichletEast, bool pin1stCell,
// 	MG_levels MG, int level, cublasHandle_t handle, int ratioZX, int ratioZY, double Lz,
// 	Matrix_t &M, Matrix_t &precond, blas_t &BLAS, double *aux, double *aux2,
// 	double *Mp, double *Ms, double *AMs, double *AMp
// 	){
// 	int Nz = pow(2,level);
// 	int Nx = Nz*ratioZX;
// 	int Ny = Nz*ratioZY;

// 	double h = Lz/(double)Nz;
// 	double dxdx = h*h;
// 	// double h_H = h*2.0;
// 	cudaMemset(e_pre[level],0,sizeof(double)*Nx*Ny*Nz);
// 	//el_pre <- smooth(0,rl); //rrl <- rl-A*el_pre;
// 	smooth_GSRB(e_pre[level],r[level],rr[level],dxdx,Nx,Ny,Nz,boundaryCond,MG.npre,true,GRID3D_L);
// 	//rrl1 <- restrict(rrl)
// 	restriction(r[level-1],rr[level],Nx/2,Ny/2,Nz/2,gridXY[level-1],blockXY[level-1]);
// 	//solve coarse system
// 	if (level==2) solver_BiCGStab(M,precond,BLAS,e_pre[level-1],rr[level-1],aux,aux2,Mp,Ms,AMs,AMp,r[level-1],1e-16,0,10000,0);
// 	//e[level-1] <- MGCYCLE(rl1)
// 	else V_cycle4(e_pre,r,rr,gridXY,blockXY,gridXZ,blockXZ,gridYZ,blockYZ,boundaryCond,MG,level-1,handle,ratioZX,ratioZY,Lz,M,precond,BLAS,aux,aux2,Mp,Ms,AMs,AMp);
// 	//el_cgc = prolong(el1); //el_sum = el_pre+el_cgc
// 	prolongation(e_pre[level],e_pre[level-1],Nx,Ny,Nz,GRID3D_L);
// 	//el <- smooth(el_sum,rl)
// 	smooth_GSRB(e_pre[level],r[level],rr[level],dxdx,Nx,Ny,Nz,boundaryCond,MG.npos,false,GRID3D_L);
// }

// void Precond_CCMG_Vcycle4(double *e0fine, const double *rfine,
// 	double **e, double **r, double **rr,
// 	dim3 *gridXY, dim3 *blockXY, dim3 *gridXZ, dim3 *blockXZ, dim3 *gridYZ, dim3 *blockYZ,
// 	bool dirichletBottom, bool dirichletTop,
// 	bool dirichletSouth, bool dirichletNorth,
// 	bool dirichletWest, bool dirichletEast, bool pin1stCell,
// 	int Nx, int Ny, int Nz, MG_levels MG, cublasHandle_t handle, int ratioZX, int ratioZY, double Lz,
// 	Matrix_t &M, Matrix_t &precond, blas_t &BLAS, double *aux, double *aux2,double *Mp, double *Ms, double *AMs, double *AMp
// 	){
// 	int level = MG.L-1;
// 	double h = Lz/(double)Nz;
// 	double dxdx = h*h;
// 	smooth_GSRB(e0fine,rfine,rr[level],dxdx,Nx,Ny,Nz,boundaryCond,MG.npre,true,GRID3D_L);
// 	restriction(r[level-1],rr[level],Nx/2,Ny/2,Nz/2,gridXY[level-1],blockXY[level-1]);
// 	V_cycle4(e,r,rr,gridXY,blockXY,gridXZ,blockXZ,gridYZ,blockYZ,boundaryCond,MG,level-1,handle,ratioZX,ratioZY,Lz,M,precond,BLAS,aux,aux2,Mp,Ms,AMs,AMp);
// 	prolongation(e0fine,e[level-1],Nx,Ny,Nz,GRID3D_L);
// 	smooth_GSRB(e0fine,rfine,rr[level],dxdx,Nx,Ny,Nz,boundaryCond,MG.npos,false,GRID3D_L);
// }
#undef boundaryCond
```

# CG.cu

```cu
#include <iostream>
#include "header/linear_operator.h"

// for residual evolution monitoring
void resMonitor(int levelPrint, int cccc, double *rr_0_h, double &res_relative_old, double *rr_new_h, int i){
	if(levelPrint==2 && cccc==1){
		std::cout<< "<b,b>: "<<*rr_new_h<<std::endl;
		std::cout<<"Iters       ||r||_2     conv.rate  ||r||_2/||b||_2"<<std::endl;
	    std::cout<<"-----    ------------   ---------  ------------"<<std::endl;
	}
	if(levelPrint==2 && cccc==2){
		std::cout.width(5); std::cout << std::right<<i<<"    ";
		std::cout<<std::right<<std::scientific<<pow(*rr_new_h,0.5)<<"    "<<std::right<<std::fixed<<(pow(*rr_new_h,0.5)/pow(*rr_0_h,0.5))/res_relative_old<<"    ";
		res_relative_old = pow(*rr_new_h,0.5)/pow(*rr_0_h,0.5);
		std::cout<<std::right<<std::scientific<<res_relative_old<<std::endl;
	}
	if(levelPrint==1 && cccc==0){
	std::cout.width(5); std::cout << std::right<<i<<"    ";
	std::cout<<std::right<<std::scientific<<pow(*rr_new_h,0.5)<<std::endl;
	}
}

// WARNING!
// Whenever possible the following is used:
// b (RHS) vector is used as the auxiliary vector P
// z is used auxiliary vector y
// otherwise you must modify to add the vectors as input and
// replace z by y, P by RHS within algorithm
// P is RHS at the beginning
int solver_CG(Matrix_t &M, Matrix_t &precond, blas_t &BLAS, double *x, double *P, double *r, double *z, double *y, double tol_abs, double tol_rel, int iter_max, int print_monitor){
	double *rr_new, *rho, *rho_old, *pAp;
	double *rr_0_h, *rr_new_h;
	cudaMalloc(&rr_new 	, sizeof(double));
	cudaMalloc(&rho    	, sizeof(double));
	cudaMalloc(&rho_old	, sizeof(double));
	cudaMalloc(&pAp		, sizeof(double));
	rr_0_h = new double[1]; rr_new_h = new double[1];
	M.stencil(y,x);
	BLAS.AXPBY3D(z,y,r,1.0,-1.0);
	BLAS.Ddot(r,r,rr_new);
	BLAS.copyScalar_d2h(rr_0_h, rr_new);
	*rr_new_h = *rr_0_h;
	int i = 0;
	double res_relative_old = 1;// auxiliar for printing
	resMonitor(print_monitor,1,rr_0_h,res_relative_old,rr_new_h,i); // if print_monitor==2 (print table)
	while( ( pow(*rr_new_h,0.5) > tol_abs + pow(*rr_0_h,0.5) * tol_rel ) && i<iter_max){
		precond.stencil(z,r);
		BLAS.copyScalar_d2d(rho_old, rho);
		BLAS.Ddot(r,z,rho);
		if (i==0) BLAS.copyVector_d2d(P,z);
		else BLAS.beta3D(z,P,rho,rho_old);
		M.stencil(y,P);
		BLAS.Ddot(y,P,pAp);
		BLAS.alpha3D(P,x,rho,pAp,true);
		BLAS.alpha3D(y,r,rho,pAp,false);
		BLAS.Ddot(r,r,rr_new);
		BLAS.copyScalar_d2h(rr_new_h,rr_new);
		i+=1;
		// if print_monitor==2 (print table)
		resMonitor(print_monitor,2,rr_0_h,res_relative_old,rr_new_h,i);
	}
	// if print_monitor==1 (print finalAbsRes, iterNumber)
	resMonitor(print_monitor,0,rr_0_h,res_relative_old,rr_new_h,i);
    cudaFree(rr_new);
    cudaFree(rho);
    cudaFree(rho_old);
	cudaFree(pAp);
	delete [] rr_new_h;
	delete [] rr_0_h;
	return i;
}
```

# compute_velocity_from_head_for_par2.cu

```cu
#include <cuda_runtime_api.h>
#include "header/macros_index_kernel.h"
#include "header/macros_index_mf_par2.h"
#define neumann 0
#define periodic 1
#define dirichlet 2

__global__ void velocity_int(double *U, double *V, double *W,
 double *H,  double *K, int Nx, int Ny, int Nz, double h){
	int ix = threadIdx.x + blockIdx.x*blockDim.x;
	int iy = threadIdx.y + blockIdx.y*blockDim.y;
	if (ix >= Nx-2 || iy >= Ny-2) return;
	int in_idx = (ix + 1) + (iy + 1)*Nx;
	int idx_U, idx_V, idx_W;
	double H_current = H[in_idx];
	double K_current = K[in_idx];
	in_idx += stride;
	double H_top = H[in_idx];
	double K_top = K[in_idx];
	for(int iz=1; iz<Nz-1; ++iz){
		H_current = H_top;
		K_current = K_top;
		// idx_U = (ix+1+1) + (iy+1)*(Nx+1) + (iz)*(Nx+1)*Ny;
		// idx_V = (ix+1)   + (iy+1+1)*(Nx) + (iz)*Nx*(Ny+1);
		// idx_W = (ix+1)   + (iy+1)*(Nx)   + (iz+1)*stride;
		idx_U = (ix+1+1) + (iy+1)*(Nx+1) + (iz)*(Nx+1)*(Ny+1);
		idx_V = (ix+1)   + (iy+1+1)*(Nx+1) + (iz)*(Nx+1)*(Ny+1);
		idx_W = (ix+1)   + (iy+1)*(Nx+1)   + (iz+1)*(Nx+1)*(Ny+1);

		U[idx_U] = -2.0/(1.0/K[in_idx+1]+ 1.0/K_current) * (H[in_idx+1]-H_current)/h;
		V[idx_V] = -2.0/(1.0/K[in_idx+Nx]+ 1.0/K_current) * (H[in_idx+Nx]-H_current)/h;
		in_idx += stride;
		H_top = H[in_idx];
		K_top = K[in_idx];
		W[idx_W] = -2.0/(1.0/K_top+ 1.0/K_current) * (H_top-H_current)/h;
	}
}

__global__ void velocity_side_BOTTOM(double *U, double *V, double *W,
double *H, double *K, int Nx, int Ny, int Nz, double h, int BC_BOTTOM,
double H_BOTTOM){
	COMPUTE_INDEX_BOTTOM
	COMPUTE_INDEX_NORMAL_VELOCITY_BOTTOM
	double K_current = K[in_idx], H_current = H[in_idx];
	U[idx_U] = -2.0/(1.0/K[in_idx+1]+ 1.0/K_current) * (H[in_idx+1]-H_current)/h;
	V[idx_V] = -2.0/(1.0/K[in_idx+Nx]+ 1.0/K_current) * (H[in_idx+Nx]-H_current)/h;
	W[idx_W] = -2.0/(1.0/K[in_idx+stride]+ 1.0/K_current) * (H[in_idx+stride]-H_current)/h;

	if (BC_BOTTOM == dirichlet) W[idx_W-(Nx+1)*(Ny+1)] = -K_current * (H_current-H_BOTTOM)/(h/2.0);
	if (BC_BOTTOM == periodic) W[idx_W-(Nx+1)*(Ny+1)] = -2.0/(1.0/K[in_idx+PERIODIC_BOTTOM]+ 1.0/K_current) * (H_current-H[in_idx+PERIODIC_BOTTOM])/h;
}

__global__ void velocity_side_TOP(double *U, double *V, double *W,
double *H, double *K, int Nx, int Ny, int Nz, double h, int BC_TOP,
double H_TOP){
	COMPUTE_INDEX_TOP
	COMPUTE_INDEX_NORMAL_VELOCITY_TOP
	double K_current = K[in_idx], H_current = H[in_idx];
	U[idx_U] = -2.0/(1.0/K[in_idx+1]+ 1.0/K_current) * (H[in_idx+1]-H_current)/h;
	V[idx_V] = -2.0/(1.0/K[in_idx+Nx]+ 1.0/K_current) * (H[in_idx+Nx]-H_current)/h;
	double result = 0; // default no flux (neumann BC)
	if (BC_TOP == dirichlet) result = -K_current * (H_TOP-H_current)/(h/2.0);
	if (BC_TOP == periodic) result = -2.0/(1.0/K[in_idx+PERIODIC_TOP]+ 1.0/K_current) * (H[in_idx+PERIODIC_TOP]-H_current)/h;
	W[idx_W] = result;
}

__global__ void velocity_side_SOUTH(double *U, double *V, double *W,
double *H, double *K, int Nx, int Ny, int Nz, double h, int BC_SOUTH,
double H_SOUTH){
	COMPUTE_INDEX_SOUTH
	COMPUTE_INDEX_NORMAL_VELOCITY_SOUTH
	double K_current = K[in_idx], H_current = H[in_idx];

	U[idx_U] = -2.0/(1.0/K[in_idx+1]+ 1.0/K_current) * (H[in_idx+1]-H_current)/h;
	V[idx_V] = -2.0/(1.0/K[in_idx+Nx]+ 1.0/K_current) * (H[in_idx+Nx]-H_current)/h;
	W[idx_W] = -2.0/(1.0/K[in_idx+stride]+ 1.0/K_current) * (H[in_idx+stride]-H_current)/h;

	if (BC_SOUTH == dirichlet) V[idx_V-(Nx+1)] = -K_current * (H_current-H_SOUTH)/(h/2.0);
	if (BC_SOUTH == periodic) V[idx_V-(Nx+1)] = -2.0/(1.0/K[in_idx+PERIODIC_SOUTH]+ 1.0/K_current) * (H_current-H[in_idx+PERIODIC_SOUTH])/h;
}

__global__ void velocity_side_WEST(double *U, double *V, double *W,
double *H, double *K, int Nx, int Ny, int Nz, double h, int BC_WEST,
double H_WEST){
	COMPUTE_INDEX_WEST
	COMPUTE_INDEX_NORMAL_VELOCITY_WEST
	double K_current = K[in_idx], H_current = H[in_idx];

	U[idx_U] = -2.0/(1.0/K[in_idx+1]+ 1.0/K_current) * (H[in_idx+1]-H_current)/h;
	V[idx_V] = -2.0/(1.0/K[in_idx+Nx]+ 1.0/K_current) * (H[in_idx+Nx]-H_current)/h;
	W[idx_W] = -2.0/(1.0/K[in_idx+stride]+ 1.0/K_current) * (H[in_idx+stride]-H_current)/h;

	if (BC_WEST == dirichlet) U[idx_U-1] = -K_current * (H_current-H_WEST)/(h/2.0);
	if (BC_WEST == periodic) U[idx_U-1] = -2.0/(1.0/K[in_idx+PERIODIC_WEST]+ 1.0/K_current) * (H_current-H[in_idx+PERIODIC_WEST])/h;
}

__global__ void velocity_side_NORTH(double *U, double *V, double *W,
double *H, double *K, int Nx, int Ny, int Nz, double h, int BC_NORTH,
double H_NORTH){
	COMPUTE_INDEX_NORTH
	COMPUTE_INDEX_NORMAL_VELOCITY_NORTH
	double K_current = K[in_idx], H_current = H[in_idx];
	U[idx_U] = -2.0/(1.0/K[in_idx+1]+ 1.0/K_current) * (H[in_idx+1]-H_current)/h;
	W[idx_W] = -2.0/(1.0/K[in_idx+stride]+ 1.0/K_current) * (H[in_idx+stride]-H_current)/h;
	double result = 0; // default no flux (neumann BC)
	if (BC_NORTH == dirichlet) result = -K_current * (H_NORTH-H_current)/(h/2.0);
	if (BC_NORTH == periodic) result = -2.0/(1.0/K[in_idx+PERIODIC_NORTH]+ 1.0/K_current) * (H[in_idx+PERIODIC_NORTH]-H_current)/h;
	V[idx_V] = result;
}

__global__ void velocity_side_EAST(double *U, double *V, double *W,
double *H, double *K, int Nx, int Ny, int Nz, double h, int BC_EAST,
double H_EAST){
	COMPUTE_INDEX_EAST
	COMPUTE_INDEX_NORMAL_VELOCITY_EAST
	double K_current = K[in_idx], H_current = H[in_idx];
	V[idx_V] = -2.0/(1.0/K[in_idx+Nx]+ 1.0/K_current) * (H[in_idx+Nx]-H_current)/h;
	W[idx_W] = -2.0/(1.0/K[in_idx+stride]+ 1.0/K_current) * (H[in_idx+stride]-H_current)/h;
	double result = 0; // default no flux (neumann BC)
	if (BC_EAST == dirichlet) result = -K_current * (H_EAST-H_current)/(h/2.0);
	if (BC_EAST == periodic) result = -2.0/(1.0/K[in_idx+PERIODIC_EAST]+ 1.0/K_current) * (H[in_idx+PERIODIC_EAST]-H_current)/h;
	U[idx_U] = result;
}

__global__ void velocity_edge_WEST_BOTTOM(double *U, double *V, double *W,
double *H, double *K, int Nx, int Ny, int Nz, double h, int BC_WEST, int BC_BOTTOM,
double H_WEST, double H_BOTTOM){
COMPUTE_INDEX_WEST_BOTTOM
COMPUTE_INDEX_NORMAL_VELOCITY_WEST_BOTTOM
double K_current = K[in_idx], H_current = H[in_idx];
U[idx_U] = -2.0/(1.0/K[in_idx+1]+ 1.0/K_current) * (H[in_idx+1]-H_current)/h;
V[idx_V] = -2.0/(1.0/K[in_idx+Nx]+ 1.0/K_current) * (H[in_idx+Nx]-H_current)/h;
W[idx_W] = -2.0/(1.0/K[in_idx+stride]+ 1.0/K_current) * (H[in_idx+stride]-H_current)/h;

if (BC_WEST == dirichlet) U[idx_U-1] = -K_current * (H_current-H_WEST)/(h/2.0);
if (BC_WEST == periodic) U[idx_U-1] = -2.0/(1.0/K[in_idx+PERIODIC_WEST]+ 1.0/K_current) * (H_current-H[in_idx+PERIODIC_WEST])/h;
if (BC_BOTTOM == dirichlet) W[idx_W-(Nx+1)*(Ny+1)] = -K_current * (H_current-H_BOTTOM)/(h/2.0);
if (BC_BOTTOM == periodic) W[idx_W-(Nx+1)*(Ny+1)] = -2.0/(1.0/K[in_idx+PERIODIC_BOTTOM]+ 1.0/K_current) * (H_current-H[in_idx+PERIODIC_BOTTOM])/h;
}

__global__ void velocity_edge_WEST_TOP(double *U, double *V, double *W,
double *H, double *K, int Nx, int Ny, int Nz, double h, int BC_WEST, int BC_TOP,
double H_WEST, double H_TOP){
COMPUTE_INDEX_WEST_TOP
COMPUTE_INDEX_NORMAL_VELOCITY_WEST_TOP
double K_current = K[in_idx], H_current = H[in_idx];
U[idx_U] = -2.0/(1.0/K[in_idx+1]+ 1.0/K_current) * (H[in_idx+1]-H_current)/h;
V[idx_V] = -2.0/(1.0/K[in_idx+Nx]+ 1.0/K_current) * (H[in_idx+Nx]-H_current)/h;

if (BC_WEST == dirichlet) U[idx_U-1] = -K_current * (H_current-H_WEST)/(h/2.0);
if (BC_WEST == periodic) U[idx_U-1] = -2.0/(1.0/K[in_idx+PERIODIC_WEST]+ 1.0/K_current) * (H_current-H[in_idx+PERIODIC_WEST])/h;
double result = 0; // default no flux (neumann BC)
if (BC_TOP == dirichlet) result = -K_current * (H_TOP-H_current)/(h/2.0);
if (BC_TOP == periodic) result = -2.0/(1.0/K[in_idx+PERIODIC_TOP]+ 1.0/K_current) * (H[in_idx+PERIODIC_TOP]-H_current)/h;
W[idx_W] = result;
}

__global__ void velocity_edge_EAST_BOTTOM(double *U, double *V, double *W,
double *H, double *K, int Nx, int Ny, int Nz, double h, int BC_EAST, int BC_BOTTOM,
double H_EAST, double H_BOTTOM){
COMPUTE_INDEX_EAST_BOTTOM
COMPUTE_INDEX_NORMAL_VELOCITY_EAST_BOTTOM
double K_current = K[in_idx], H_current = H[in_idx];
V[idx_V] = -2.0/(1.0/K[in_idx+Nx]+ 1.0/K_current) * (H[in_idx+Nx]-H_current)/h;
W[idx_W] = -2.0/(1.0/K[in_idx+stride]+ 1.0/K_current) * (H[in_idx+stride]-H_current)/h;

double result = 0; // default no flux (neumann BC)
if (BC_EAST == dirichlet) result = -K_current * (H_EAST-H_current)/(h/2.0);
if (BC_EAST == periodic) result = -2.0/(1.0/K[in_idx+PERIODIC_EAST]+ 1.0/K_current) * (H[in_idx+PERIODIC_EAST]-H_current)/h;
U[idx_U] = result;
if (BC_BOTTOM == dirichlet) W[idx_W-(Nx+1)*(Ny+1)] = -K_current * (H_current-H_BOTTOM)/(h/2.0);
if (BC_BOTTOM == periodic) W[idx_W-(Nx+1)*(Ny+1)] = -2.0/(1.0/K[in_idx+PERIODIC_BOTTOM]+ 1.0/K_current) * (H_current-H[in_idx+PERIODIC_BOTTOM])/h;
}

__global__ void velocity_edge_EAST_TOP(double *U, double *V, double *W,
double *H, double *K, int Nx, int Ny, int Nz, double h, int BC_EAST, int BC_TOP,
double H_EAST, double H_TOP){
COMPUTE_INDEX_EAST_TOP
COMPUTE_INDEX_NORMAL_VELOCITY_EAST_TOP
double K_current = K[in_idx], H_current = H[in_idx];
V[idx_V] = -2.0/(1.0/K[in_idx+Nx]+ 1.0/K_current) * (H[in_idx+Nx]-H_current)/h;

double result = 0; // default no flux (neumann BC)
if (BC_EAST == dirichlet) result = -K_current * (H_EAST-H_current)/(h/2.0);
if (BC_EAST == periodic) result = -2.0/(1.0/K[in_idx+PERIODIC_EAST]+ 1.0/K_current) * (H[in_idx+PERIODIC_EAST]-H_current)/h;
U[idx_U] = result;
result = 0;
if (BC_TOP == dirichlet) result = -K_current * (H_TOP-H_current)/(h/2.0);
if (BC_TOP == periodic) result = -2.0/(1.0/K[in_idx+PERIODIC_TOP]+ 1.0/K_current) * (H[in_idx+PERIODIC_TOP]-H_current)/h;
W[idx_W] = result;
}

__global__ void velocity_edge_WEST_SOUTH(double *U, double *V, double *W,
double *H, double *K, int Nx, int Ny, int Nz, double h, int BC_WEST, int BC_SOUTH,
double H_WEST, double H_SOUTH){
COMPUTE_INDEX_WEST_SOUTH
COMPUTE_INDEX_NORMAL_VELOCITY_WEST_SOUTH
double K_current = K[in_idx], H_current = H[in_idx];
U[idx_U] = -2.0/(1.0/K[in_idx+1]+ 1.0/K_current) * (H[in_idx+1]-H_current)/h;
V[idx_V] = -2.0/(1.0/K[in_idx+Nx]+ 1.0/K_current) * (H[in_idx+Nx]-H_current)/h;
W[idx_W] = -2.0/(1.0/K[in_idx+stride]+ 1.0/K_current) * (H[in_idx+stride]-H_current)/h;

if (BC_WEST == dirichlet) U[idx_U-1] = -K_current * (H_current-H_WEST)/(h/2.0);
if (BC_WEST == periodic) U[idx_U-1] = -2.0/(1.0/K[in_idx+PERIODIC_WEST]+ 1.0/K_current) * (H_current-H[in_idx+PERIODIC_WEST])/h;
if (BC_SOUTH == dirichlet) V[idx_V-(Nx+1)] = -K_current * (H_current-H_SOUTH)/(h/2.0);
if (BC_SOUTH == periodic) V[idx_V-(Nx+1)] = -2.0/(1.0/K[in_idx+PERIODIC_SOUTH]+ 1.0/K_current) * (H_current-H[in_idx+PERIODIC_SOUTH])/h;
}

__global__ void velocity_edge_WEST_NORTH(double *U, double *V, double *W,
double *H, double *K, int Nx, int Ny, int Nz, double h, int BC_WEST, int BC_NORTH,
double H_WEST, double H_NORTH){
COMPUTE_INDEX_WEST_NORTH
COMPUTE_INDEX_NORMAL_VELOCITY_WEST_NORTH
double K_current = K[in_idx], H_current = H[in_idx];
U[idx_U] = -2.0/(1.0/K[in_idx+1]+ 1.0/K_current) * (H[in_idx+1]-H_current)/h;
W[idx_W] = -2.0/(1.0/K[in_idx+stride]+ 1.0/K_current) * (H[in_idx+stride]-H_current)/h;

if (BC_WEST == dirichlet) U[idx_U-1] = -K_current * (H_current-H_WEST)/(h/2.0);
if (BC_WEST == periodic) U[idx_U-1] = -2.0/(1.0/K[in_idx+PERIODIC_WEST]+ 1.0/K_current) * (H_current-H[in_idx+PERIODIC_WEST])/h;

double result = 0; // default no flux (neumann BC)
if (BC_NORTH == dirichlet) result = -K_current * (H_NORTH-H_current)/(h/2.0);
if (BC_NORTH == periodic) result = -2.0/(1.0/K[in_idx+PERIODIC_NORTH]+ 1.0/K_current) * (H[in_idx+PERIODIC_NORTH]-H_current)/h;
V[idx_V] = result;
}

__global__ void velocity_edge_EAST_SOUTH(double *U, double *V, double *W,
double *H, double *K, int Nx, int Ny, int Nz, double h, int BC_EAST, int BC_SOUTH,
double H_EAST, double H_SOUTH){
COMPUTE_INDEX_EAST_SOUTH
COMPUTE_INDEX_NORMAL_VELOCITY_EAST_SOUTH
double K_current = K[in_idx], H_current = H[in_idx];
V[idx_V] = -2.0/(1.0/K[in_idx+Nx]+ 1.0/K_current) * (H[in_idx+Nx]-H_current)/h;
W[idx_W] = -2.0/(1.0/K[in_idx+stride]+ 1.0/K_current) * (H[in_idx+stride]-H_current)/h;

double result = 0; // default no flux (neumann BC)
if (BC_EAST == dirichlet) result = -K_current * (H_EAST-H_current)/(h/2.0);
if (BC_EAST == periodic) result = -2.0/(1.0/K[in_idx+PERIODIC_EAST]+ 1.0/K_current) * (H[in_idx+PERIODIC_EAST]-H_current)/h;
U[idx_U] = result;

if (BC_SOUTH == dirichlet) V[idx_V-(Nx+1)] = -K_current * (H_current-H_SOUTH)/(h/2.0);
if (BC_SOUTH == periodic) V[idx_V-(Nx+1)] = -2.0/(1.0/K[in_idx+PERIODIC_SOUTH]+ 1.0/K_current) * (H_current-H[in_idx+PERIODIC_SOUTH])/h;
}

__global__ void velocity_edge_EAST_NORTH(double *U, double *V, double *W,
double *H, double *K, int Nx, int Ny, int Nz, double h, int BC_EAST, int BC_NORTH,
double H_EAST, double H_NORTH){
COMPUTE_INDEX_EAST_NORTH
COMPUTE_INDEX_NORMAL_VELOCITY_EAST_NORTH
double K_current = K[in_idx], H_current = H[in_idx];
W[idx_W] = -2.0/(1.0/K[in_idx+stride]+ 1.0/K_current) * (H[in_idx+stride]-H_current)/h;

double result = 0; // default no flux (neumann BC)
if (BC_EAST == dirichlet) result = -K_current * (H_EAST-H_current)/(h/2.0);
if (BC_EAST == periodic) result = -2.0/(1.0/K[in_idx+PERIODIC_EAST]+ 1.0/K_current) * (H[in_idx+PERIODIC_EAST]-H_current)/h;
U[idx_U] = result;

result = 0; // default no flux (neumann BC)
if (BC_NORTH == dirichlet) result = -K_current * (H_NORTH-H_current)/(h/2.0);
if (BC_NORTH == periodic) result = -2.0/(1.0/K[in_idx+PERIODIC_NORTH]+ 1.0/K_current) * (H[in_idx+PERIODIC_NORTH]-H_current)/h;
V[idx_V] = result;
}

__global__ void velocity_edge_SOUTH_BOTTOM(double *U, double *V, double *W,
double *H, double *K, int Nx, int Ny, int Nz, double h, int BC_SOUTH, int BC_BOTTOM,
double H_SOUTH, double H_BOTTOM){
COMPUTE_INDEX_SOUTH_BOTTOM
COMPUTE_INDEX_NORMAL_VELOCITY_SOUTH_BOTTOM
double K_current = K[in_idx], H_current = H[in_idx];
U[idx_U] = -2.0/(1.0/K[in_idx+1]+ 1.0/K_current) * (H[in_idx+1]-H_current)/h;
V[idx_V] = -2.0/(1.0/K[in_idx+Nx]+ 1.0/K_current) * (H[in_idx+Nx]-H_current)/h;
W[idx_W] = -2.0/(1.0/K[in_idx+stride]+ 1.0/K_current) * (H[in_idx+stride]-H_current)/h;

if (BC_SOUTH == dirichlet) V[idx_V-(Nx+1)] = -K_current * (H_current-H_SOUTH)/(h/2.0);
if (BC_SOUTH == periodic) V[idx_V-(Nx+1)] = -2.0/(1.0/K[in_idx+PERIODIC_SOUTH]+ 1.0/K_current) * (H_current-H[in_idx+PERIODIC_SOUTH])/h;
if (BC_BOTTOM == dirichlet) W[idx_W-(Nx+1)*(Ny+1)] = -K_current * (H_current-H_BOTTOM)/(h/2.0);
if (BC_BOTTOM == periodic) W[idx_W-(Nx+1)*(Ny+1)] = -2.0/(1.0/K[in_idx+PERIODIC_BOTTOM]+ 1.0/K_current) * (H_current-H[in_idx+PERIODIC_BOTTOM])/h;
}

__global__ void velocity_edge_SOUTH_TOP(double *U, double *V, double *W,
double *H, double *K, int Nx, int Ny, int Nz, double h, int BC_SOUTH, int BC_TOP,
double H_SOUTH, double H_TOP){
COMPUTE_INDEX_SOUTH_TOP
COMPUTE_INDEX_NORMAL_VELOCITY_SOUTH_TOP
double K_current = K[in_idx], H_current = H[in_idx];
U[idx_U] = -2.0/(1.0/K[in_idx+1]+ 1.0/K_current) * (H[in_idx+1]-H_current)/h;
V[idx_V] = -2.0/(1.0/K[in_idx+Nx]+ 1.0/K_current) * (H[in_idx+Nx]-H_current)/h;

if (BC_SOUTH == dirichlet) V[idx_V-(Nx+1)] = -K_current * (H_current-H_SOUTH)/(h/2.0);
if (BC_SOUTH == periodic) V[idx_V-(Nx+1)] = -2.0/(1.0/K[in_idx+PERIODIC_SOUTH]+ 1.0/K_current) * (H_current-H[in_idx+PERIODIC_SOUTH])/h;
double result = 0; // default no flux (neumann BC)
if (BC_TOP == dirichlet) result = -K_current * (H_TOP-H_current)/(h/2.0);
if (BC_TOP == periodic) result = -2.0/(1.0/K[in_idx+PERIODIC_TOP]+ 1.0/K_current) * (H[in_idx+PERIODIC_TOP]-H_current)/h;
W[idx_W] = result;
}

__global__ void velocity_edge_NORTH_BOTTOM(double *U, double *V, double *W,
double *H, double *K, int Nx, int Ny, int Nz, double h, int BC_NORTH, int BC_BOTTOM,
double H_NORTH, double H_BOTTOM){
COMPUTE_INDEX_NORTH_BOTTOM
COMPUTE_INDEX_NORMAL_VELOCITY_NORTH_BOTTOM
double K_current = K[in_idx], H_current = H[in_idx];
U[idx_U] = -2.0/(1.0/K[in_idx+1]+ 1.0/K_current) * (H[in_idx+1]-H_current)/h;
W[idx_W] = -2.0/(1.0/K[in_idx+stride]+ 1.0/K_current) * (H[in_idx+stride]-H_current)/h;

double result = 0; // default no flux (neumann BC)
if (BC_NORTH == dirichlet) result = -K_current * (H_NORTH-H_current)/(h/2.0);
if (BC_NORTH == periodic) result = -2.0/(1.0/K[in_idx+PERIODIC_NORTH]+ 1.0/K_current) * (H[in_idx+PERIODIC_NORTH]-H_current)/h;
V[idx_V] = result;
if (BC_BOTTOM == dirichlet) W[idx_W-(Nx+1)*(Ny+1)] = -K_current * (H_current-H_BOTTOM)/(h/2.0);
if (BC_BOTTOM == periodic) W[idx_W-(Nx+1)*(Ny+1)] = -2.0/(1.0/K[in_idx+PERIODIC_BOTTOM]+ 1.0/K_current) * (H_current-H[in_idx+PERIODIC_BOTTOM])/h;
}

__global__ void velocity_edge_NORTH_TOP(double *U, double *V, double *W,
double *H, double *K, int Nx, int Ny, int Nz, double h, int BC_NORTH, int BC_TOP,
double H_NORTH, double H_TOP){
COMPUTE_INDEX_NORTH_TOP
COMPUTE_INDEX_NORMAL_VELOCITY_NORTH_TOP
double K_current = K[in_idx], H_current = H[in_idx];
U[idx_U] = -2.0/(1.0/K[in_idx+1]+ 1.0/K_current) * (H[in_idx+1]-H_current)/h;

double result = 0; // default no flux (neumann BC)
if (BC_NORTH == dirichlet) result = -K_current * (H_NORTH-H_current)/(h/2.0);
if (BC_NORTH == periodic) result = -2.0/(1.0/K[in_idx+PERIODIC_NORTH]+ 1.0/K_current) * (H[in_idx+PERIODIC_NORTH]-H_current)/h;
V[idx_V] = result;
result = 0; // default no flux (neumann BC)
if (BC_TOP == dirichlet) result = -K_current * (H_TOP-H_current)/(h/2.0);
if (BC_TOP == periodic) result = -2.0/(1.0/K[in_idx+PERIODIC_TOP]+ 1.0/K_current) * (H[in_idx+PERIODIC_TOP]-H_current)/h;
W[idx_W] = result;
}

__global__ void velocity_vertex_WEST_SOUTH_BOTTOM(double *U, double *V, double *W,
double *H, double *K, int Nx, int Ny, int Nz, double h, int BC_WEST, int BC_SOUTH, int BC_BOTTOM, double H_WEST, double H_SOUTH, double H_BOTTOM){
COMPUTE_INDEX_WEST_SOUTH_BOTTOM
COMPUTE_INDEX_NORMAL_VELOCITY_WEST_SOUTH_BOTTOM
double K_current = K[in_idx], H_current = H[in_idx];
U[idx_U] = -2.0/(1.0/K[in_idx+1]+ 1.0/K_current) * (H[in_idx+1]-H_current)/h;
V[idx_V] = -2.0/(1.0/K[in_idx+Nx]+ 1.0/K_current) * (H[in_idx+Nx]-H_current)/h;
W[idx_W] = -2.0/(1.0/K[in_idx+stride]+ 1.0/K_current) * (H[in_idx+stride]-H_current)/h;

if (BC_WEST == dirichlet) U[idx_U-1] = -K_current * (H_current-H_WEST)/(h/2.0);
if (BC_WEST == periodic) U[idx_U-1] = -2.0/(1.0/K[in_idx+PERIODIC_WEST]+ 1.0/K_current) * (H_current-H[in_idx+PERIODIC_WEST])/h;
if (BC_SOUTH == dirichlet) V[idx_V-(Nx+1)] = -K_current * (H_current-H_SOUTH)/(h/2.0);
if (BC_SOUTH == periodic) V[idx_V-(Nx+1)] = -2.0/(1.0/K[in_idx+PERIODIC_SOUTH]+ 1.0/K_current) * (H_current-H[in_idx+PERIODIC_SOUTH])/h;
if (BC_BOTTOM == dirichlet) W[idx_W-(Nx+1)*(Ny+1)] = -K_current * (H_current-H_BOTTOM)/(h/2.0);
if (BC_BOTTOM == periodic) W[idx_W-(Nx+1)*(Ny+1)] = -2.0/(1.0/K[in_idx+PERIODIC_BOTTOM]+ 1.0/K_current) * (H_current-H[in_idx+PERIODIC_BOTTOM])/h;
}

__global__ void velocity_vertex_WEST_SOUTH_TOP(double *U, double *V, double *W,
double *H, double *K, int Nx, int Ny, int Nz, double h, int BC_WEST, int BC_SOUTH, int BC_TOP, double H_WEST, double H_SOUTH, double H_TOP){
COMPUTE_INDEX_WEST_SOUTH_TOP
COMPUTE_INDEX_NORMAL_VELOCITY_WEST_SOUTH_TOP
double K_current = K[in_idx], H_current = H[in_idx];
U[idx_U] = -2.0/(1.0/K[in_idx+1]+ 1.0/K_current) * (H[in_idx+1]-H_current)/h;
V[idx_V] = -2.0/(1.0/K[in_idx+Nx]+ 1.0/K_current) * (H[in_idx+Nx]-H_current)/h;

if (BC_WEST == dirichlet) U[idx_U-1] = -K_current * (H_current-H_WEST)/(h/2.0);
if (BC_WEST == periodic) U[idx_U-1] = -2.0/(1.0/K[in_idx+PERIODIC_WEST]+ 1.0/K_current) * (H_current-H[in_idx+PERIODIC_WEST])/h;
if (BC_SOUTH == dirichlet) V[idx_V-(Nx+1)] = -K_current * (H_current-H_SOUTH)/(h/2.0);
if (BC_SOUTH == periodic) V[idx_V-(Nx+1)] = -2.0/(1.0/K[in_idx+PERIODIC_SOUTH]+ 1.0/K_current) * (H_current-H[in_idx+PERIODIC_SOUTH])/h;
double result = 0; // default no flux (neumann BC)
if (BC_TOP == dirichlet) result = -K_current * (H_TOP-H_current)/(h/2.0);
if (BC_TOP == periodic) result = -2.0/(1.0/K[in_idx+PERIODIC_TOP]+ 1.0/K_current) * (H[in_idx+PERIODIC_TOP]-H_current)/h;
W[idx_W] = result;
}

__global__ void velocity_vertex_WEST_NORTH_BOTTOM(double *U, double *V, double *W,
double *H, double *K, int Nx, int Ny, int Nz, double h, int BC_WEST, int BC_NORTH, int BC_BOTTOM, double H_WEST, double H_NORTH, double H_BOTTOM){
COMPUTE_INDEX_WEST_NORTH_BOTTOM
COMPUTE_INDEX_NORMAL_VELOCITY_WEST_NORTH_BOTTOM
double K_current = K[in_idx], H_current = H[in_idx];
U[idx_U] = -2.0/(1.0/K[in_idx+1]+ 1.0/K_current) * (H[in_idx+1]-H_current)/h;
W[idx_W] = -2.0/(1.0/K[in_idx+stride]+ 1.0/K_current) * (H[in_idx+stride]-H_current)/h;

if (BC_WEST == dirichlet) U[idx_U-1] = -K_current * (H_current-H_WEST)/(h/2.0);
if (BC_WEST == periodic) U[idx_U-1] = -2.0/(1.0/K[in_idx+PERIODIC_WEST]+ 1.0/K_current) * (H_current-H[in_idx+PERIODIC_WEST])/h;
double result = 0; // default no flux (neumann BC)
if (BC_NORTH == dirichlet) result = -K_current * (H_NORTH-H_current)/(h/2.0);
if (BC_NORTH == periodic) result = -2.0/(1.0/K[in_idx+PERIODIC_NORTH]+ 1.0/K_current) * (H[in_idx+PERIODIC_NORTH]-H_current)/h;
V[idx_V] = result;
if (BC_BOTTOM == dirichlet) W[idx_W-(Nx+1)*(Ny+1)] = -K_current * (H_current-H_BOTTOM)/(h/2.0);
if (BC_BOTTOM == periodic) W[idx_W-(Nx+1)*(Ny+1)] = -2.0/(1.0/K[in_idx+PERIODIC_BOTTOM]+ 1.0/K_current) * (H_current-H[in_idx+PERIODIC_BOTTOM])/h;
}

__global__ void velocity_vertex_WEST_NORTH_TOP(double *U, double *V, double *W,
double *H, double *K, int Nx, int Ny, int Nz, double h, int BC_WEST, int BC_NORTH, int BC_TOP, double H_WEST, double H_NORTH, double H_TOP){
COMPUTE_INDEX_WEST_NORTH_TOP
COMPUTE_INDEX_NORMAL_VELOCITY_WEST_NORTH_TOP
double K_current = K[in_idx], H_current = H[in_idx];
U[idx_U] = -2.0/(1.0/K[in_idx+1]+ 1.0/K_current) * (H[in_idx+1]-H_current)/h;

if (BC_WEST == dirichlet) U[idx_U-1] = -K_current * (H_current-H_WEST)/(h/2.0);
if (BC_WEST == periodic) U[idx_U-1] = -2.0/(1.0/K[in_idx+PERIODIC_WEST]+ 1.0/K_current) * (H_current-H[in_idx+PERIODIC_WEST])/h;
double result = 0; // default no flux (neumann BC)
if (BC_NORTH == dirichlet) result = -K_current * (H_NORTH-H_current)/(h/2.0);
if (BC_NORTH == periodic) result = -2.0/(1.0/K[in_idx+PERIODIC_NORTH]+ 1.0/K_current) * (H[in_idx+PERIODIC_NORTH]-H_current)/h;
V[idx_V] = result;
result = 0; // default no flux (neumann BC)
if (BC_TOP == dirichlet) result = -K_current * (H_TOP-H_current)/(h/2.0);
if (BC_TOP == periodic) result = -2.0/(1.0/K[in_idx+PERIODIC_TOP]+ 1.0/K_current) * (H[in_idx+PERIODIC_TOP]-H_current)/h;
W[idx_W] = result;
}

__global__ void velocity_vertex_EAST_SOUTH_BOTTOM(double *U, double *V, double *W,
double *H, double *K, int Nx, int Ny, int Nz, double h, int BC_EAST, int BC_SOUTH, int BC_BOTTOM, double H_EAST, double H_SOUTH, double H_BOTTOM){
COMPUTE_INDEX_EAST_SOUTH_BOTTOM
COMPUTE_INDEX_NORMAL_VELOCITY_EAST_SOUTH_BOTTOM
double K_current = K[in_idx], H_current = H[in_idx];
V[idx_V] = -2.0/(1.0/K[in_idx+Nx]+ 1.0/K_current) * (H[in_idx+Nx]-H_current)/h;
W[idx_W] = -2.0/(1.0/K[in_idx+stride]+ 1.0/K_current) * (H[in_idx+stride]-H_current)/h;

double result = 0; // default no flux (neumann BC)
if (BC_EAST == dirichlet) result = -K_current * (H_EAST-H_current)/(h/2.0);
if (BC_EAST == periodic) result = -2.0/(1.0/K[in_idx+PERIODIC_EAST]+ 1.0/K_current) * (H[in_idx+PERIODIC_EAST]-H_current)/h;
U[idx_U] = result;
if (BC_SOUTH == dirichlet) V[idx_V-(Nx+1)] = -K_current * (H_current-H_SOUTH)/(h/2.0);
if (BC_SOUTH == periodic) V[idx_V-(Nx+1)] = -2.0/(1.0/K[in_idx+PERIODIC_SOUTH]+ 1.0/K_current) * (H_current-H[in_idx+PERIODIC_SOUTH])/h;
if (BC_BOTTOM == dirichlet) W[idx_W-(Nx+1)*(Ny+1)] = -K_current * (H_current-H_BOTTOM)/(h/2.0);
if (BC_BOTTOM == periodic) W[idx_W-(Nx+1)*(Ny+1)] = -2.0/(1.0/K[in_idx+PERIODIC_BOTTOM]+ 1.0/K_current) * (H_current-H[in_idx+PERIODIC_BOTTOM])/h;
}

__global__ void velocity_vertex_EAST_SOUTH_TOP(double *U, double *V, double *W,
double *H, double *K, int Nx, int Ny, int Nz, double h, int BC_EAST, int BC_SOUTH, int BC_TOP, double H_EAST, double H_SOUTH, double H_TOP){
COMPUTE_INDEX_EAST_SOUTH_TOP
COMPUTE_INDEX_NORMAL_VELOCITY_EAST_SOUTH_TOP
double K_current = K[in_idx], H_current = H[in_idx];
V[idx_V] = -2.0/(1.0/K[in_idx+Nx]+ 1.0/K_current) * (H[in_idx+Nx]-H_current)/h;

double result = 0; // default no flux (neumann BC)
if (BC_EAST == dirichlet) result = -K_current * (H_EAST-H_current)/(h/2.0);
if (BC_EAST == periodic) result = -2.0/(1.0/K[in_idx+PERIODIC_EAST]+ 1.0/K_current) * (H[in_idx+PERIODIC_EAST]-H_current)/h;
U[idx_U] = result;
if (BC_SOUTH == dirichlet) V[idx_V-(Nx+1)] = -K_current * (H_current-H_SOUTH)/(h/2.0);
if (BC_SOUTH == periodic) V[idx_V-(Nx+1)] = -2.0/(1.0/K[in_idx+PERIODIC_SOUTH]+ 1.0/K_current) * (H_current-H[in_idx+PERIODIC_SOUTH])/h;
result = 0; // default no flux (neumann BC)
if (BC_TOP == dirichlet) result = -K_current * (H_TOP-H_current)/(h/2.0);
if (BC_TOP == periodic) result = -2.0/(1.0/K[in_idx+PERIODIC_TOP]+ 1.0/K_current) * (H[in_idx+PERIODIC_TOP]-H_current)/h;
W[idx_W] = result;
}

__global__ void velocity_vertex_EAST_NORTH_BOTTOM(double *U, double *V, double *W,
double *H, double *K, int Nx, int Ny, int Nz, double h, int BC_EAST, int BC_NORTH, int BC_BOTTOM, double H_EAST, double H_NORTH, double H_BOTTOM){
COMPUTE_INDEX_EAST_NORTH_BOTTOM
COMPUTE_INDEX_NORMAL_VELOCITY_EAST_NORTH_BOTTOM
double K_current = K[in_idx], H_current = H[in_idx];
W[idx_W] = -2.0/(1.0/K[in_idx+stride]+ 1.0/K_current) * (H[in_idx+stride]-H_current)/h;

double result = 0; // default no flux (neumann BC)
if (BC_EAST == dirichlet) result = -K_current * (H_EAST-H_current)/(h/2.0);
if (BC_EAST == periodic) result = -2.0/(1.0/K[in_idx+PERIODIC_EAST]+ 1.0/K_current) * (H[in_idx+PERIODIC_EAST]-H_current)/h;
U[idx_U] = result;
result = 0; // default no flux (neumann BC)
if (BC_NORTH == dirichlet) result = -K_current * (H_NORTH-H_current)/(h/2.0);
if (BC_NORTH == periodic) result = -2.0/(1.0/K[in_idx+PERIODIC_NORTH]+ 1.0/K_current) * (H[in_idx+PERIODIC_NORTH]-H_current)/h;
V[idx_V] = result;
if (BC_BOTTOM == dirichlet) W[idx_W-(Nx+1)*(Ny+1)] = -K_current * (H_current-H_BOTTOM)/(h/2.0);
if (BC_BOTTOM == periodic) W[idx_W-(Nx+1)*(Ny+1)] = -2.0/(1.0/K[in_idx+PERIODIC_BOTTOM]+ 1.0/K_current) * (H_current-H[in_idx+PERIODIC_BOTTOM])/h;
}

__global__ void velocity_vertex_EAST_NORTH_TOP(double *U, double *V, double *W,
double *H, double *K, int Nx, int Ny, int Nz, double h, int BC_EAST, int BC_NORTH, int BC_TOP, double H_EAST, double H_NORTH, double H_TOP){
COMPUTE_INDEX_EAST_NORTH_TOP
COMPUTE_INDEX_NORMAL_VELOCITY_EAST_NORTH_TOP
double K_current = K[in_idx], H_current = H[in_idx];

double result = 0; // default no flux (neumann BC)
if (BC_EAST == dirichlet) result = -K_current * (H_EAST-H_current)/(h/2.0);
if (BC_EAST == periodic) result = -2.0/(1.0/K[in_idx+PERIODIC_EAST]+ 1.0/K_current) * (H[in_idx+PERIODIC_EAST]-H_current)/h;
U[idx_U] = result;
result = 0; // default no flux (neumann BC)
if (BC_NORTH == dirichlet) result = -K_current * (H_NORTH-H_current)/(h/2.0);
if (BC_NORTH == periodic) result = -2.0/(1.0/K[in_idx+PERIODIC_NORTH]+ 1.0/K_current) * (H[in_idx+PERIODIC_NORTH]-H_current)/h;
V[idx_V] = result;
result = 0; // default no flux (neumann BC)
if (BC_TOP == dirichlet) result = -K_current * (H_TOP-H_current)/(h/2.0);
if (BC_TOP == periodic) result = -2.0/(1.0/K[in_idx+PERIODIC_TOP]+ 1.0/K_current) * (H[in_idx+PERIODIC_TOP]-H_current)/h;
W[idx_W] = result;
}

#define LAUNCH_KERNEL_SIDE(FACE) \
    velocity_side_##FACE<<<GRIDBLOCK_##FACE>>>(U,V,W,H,K,Nx,Ny,Nz,h,BC_##FACE,H_##FACE)
#define LAUNCH_KERNEL_EDGE(FACE1,FACE2) \
    velocity_edge_##FACE1##_##FACE2<<<GRIDBLOCK_##FACE1##_##FACE2>>>(U,V,W,H,K,Nx,Ny,Nz,h, BC_##FACE1,BC_##FACE2, H_##FACE1,H_##FACE2)
#define LAUNCH_KERNEL_VERTEX(FACE1,FACE2,FACE3) \
    velocity_vertex_##FACE1##_##FACE2##_##FACE3<<<1,1>>>(U,V,W,H,K,Nx,Ny,Nz,h,BC_##FACE1,BC_##FACE2,BC_##FACE3, H_##FACE1,H_##FACE2,H_##FACE3)
#define LAUNCH_KERNEL_INT \
    velocity_int<<<GRIDBLOCK_BOTTOM>>>(U,V,W,H,K,Nx,Ny,Nz,h)

void compute_velocity_from_head(double *U, double *V, double *W, double *H, double *K, int Nx, int Ny, int Nz, double h, int BC_WEST, int BC_EAST, int BC_SOUTH, int BC_NORTH, int BC_BOTTOM, int BC_TOP, double H_WEST, double H_EAST, double H_SOUTH, double H_NORTH, double H_BOTTOM, double H_TOP, dim3 grid, dim3 block){
LAUNCH_KERNEL_INT;
LAUNCH_KERNEL_SIDE(BOTTOM);
LAUNCH_KERNEL_SIDE(TOP);
LAUNCH_KERNEL_SIDE(NORTH);
LAUNCH_KERNEL_SIDE(SOUTH);
LAUNCH_KERNEL_SIDE(WEST);
LAUNCH_KERNEL_SIDE(EAST);

LAUNCH_KERNEL_EDGE(SOUTH,BOTTOM);
LAUNCH_KERNEL_EDGE(SOUTH,TOP);
LAUNCH_KERNEL_EDGE(NORTH,BOTTOM);
LAUNCH_KERNEL_EDGE(NORTH,TOP);

LAUNCH_KERNEL_EDGE(WEST,SOUTH);
LAUNCH_KERNEL_EDGE(WEST,NORTH);
LAUNCH_KERNEL_EDGE(EAST,SOUTH);
LAUNCH_KERNEL_EDGE(EAST,NORTH);

LAUNCH_KERNEL_EDGE(WEST,BOTTOM);
LAUNCH_KERNEL_EDGE(WEST,TOP);
LAUNCH_KERNEL_EDGE(EAST,BOTTOM);
LAUNCH_KERNEL_EDGE(EAST,TOP);

LAUNCH_KERNEL_VERTEX(WEST,SOUTH,BOTTOM);
LAUNCH_KERNEL_VERTEX(WEST,SOUTH,TOP);
LAUNCH_KERNEL_VERTEX(WEST,NORTH,BOTTOM);
LAUNCH_KERNEL_VERTEX(WEST,NORTH,TOP);
LAUNCH_KERNEL_VERTEX(EAST,SOUTH,BOTTOM);
LAUNCH_KERNEL_VERTEX(EAST,SOUTH,TOP);
LAUNCH_KERNEL_VERTEX(EAST,NORTH,BOTTOM);
LAUNCH_KERNEL_VERTEX(EAST,NORTH,TOP);
cudaDeviceSynchronize();
}
```

# compute_velocity_from_head_v1.cu

```cu
#include <cuda_runtime_api.h>
#include "header/macros_index_kernel.h"
#include "header/macros_index_for_mf.h"
#define neumann 0
#define periodic 1
#define dirichlet 2

__global__ void velocity_int(double *U, double *V, double *W,
 double *H,  double *K, int Nx, int Ny, int Nz, double h){
	int ix = threadIdx.x + blockIdx.x*blockDim.x;
	int iy = threadIdx.y + blockIdx.y*blockDim.y;
	if (ix >= Nx-2 || iy >= Ny-2) return;
	int in_idx = (ix + 1) + (iy + 1)*Nx;
	int idx_U, idx_V, idx_W;
	double H_current = H[in_idx];
	double K_current = K[in_idx];
	in_idx += stride;
	double H_top = H[in_idx];
	double K_top = K[in_idx];
	for(int iz=1; iz<Nz-1; ++iz){
		H_current = H_top;
		K_current = K_top;
		idx_U = (ix+1+1) + (iy+1)*(Nx+1) + (iz)*(Nx+1)*Ny;
		idx_V = (ix+1)   + (iy+1+1)*(Nx) + (iz)*Nx*(Ny+1);
		idx_W = (ix+1)   + (iy+1)*(Nx)   + (iz+1)*stride;

		U[idx_U] = -2.0/(1.0/K[in_idx+1]+ 1.0/K_current) * (H[in_idx+1]-H_current)/h;
		V[idx_V] = -2.0/(1.0/K[in_idx+Nx]+ 1.0/K_current) * (H[in_idx+Nx]-H_current)/h;
		in_idx += stride;
		H_top = H[in_idx];
		K_top = K[in_idx];
		W[idx_W] = -2.0/(1.0/K_top+ 1.0/K_current) * (H_top-H_current)/h;
	}
}

__global__ void velocity_side_BOTTOM(double *U, double *V, double *W,
double *H, double *K, int Nx, int Ny, int Nz, double h, int BC_BOTTOM,
double H_BOTTOM){
	COMPUTE_INDEX_BOTTOM
	COMPUTE_INDEX_NORMAL_VELOCITY_BOTTOM
	double K_current = K[in_idx], H_current = H[in_idx];
	U[idx_U] = -2.0/(1.0/K[in_idx+1]+ 1.0/K_current) * (H[in_idx+1]-H_current)/h;
	V[idx_V] = -2.0/(1.0/K[in_idx+Nx]+ 1.0/K_current) * (H[in_idx+Nx]-H_current)/h;
	W[idx_W] = -2.0/(1.0/K[in_idx+stride]+ 1.0/K_current) * (H[in_idx+stride]-H_current)/h;

	if (BC_BOTTOM == dirichlet) W[idx_W-stride] = -K_current * (H_current-H_BOTTOM)/(h/2.0);
	if (BC_BOTTOM == periodic) W[idx_W-stride] = -2.0/(1.0/K[in_idx+PERIODIC_BOTTOM]+ 1.0/K_current) * (H_current-H[in_idx+PERIODIC_BOTTOM])/h;
}

__global__ void velocity_side_TOP(double *U, double *V, double *W,
double *H, double *K, int Nx, int Ny, int Nz, double h, int BC_TOP,
double H_TOP){
	COMPUTE_INDEX_TOP
	COMPUTE_INDEX_NORMAL_VELOCITY_TOP
	double K_current = K[in_idx], H_current = H[in_idx];
	U[idx_U] = -2.0/(1.0/K[in_idx+1]+ 1.0/K_current) * (H[in_idx+1]-H_current)/h;
	V[idx_V] = -2.0/(1.0/K[in_idx+Nx]+ 1.0/K_current) * (H[in_idx+Nx]-H_current)/h;
	double result = 0; // default no flux (neumann BC)
	if (BC_TOP == dirichlet) result = -K_current * (H_TOP-H_current)/(h/2.0);
	if (BC_TOP == periodic) result = -2.0/(1.0/K[in_idx+PERIODIC_TOP]+ 1.0/K_current) * (H[in_idx+PERIODIC_TOP]-H_current)/h;
	W[idx_W] = result;
}

__global__ void velocity_side_SOUTH(double *U, double *V, double *W,
double *H, double *K, int Nx, int Ny, int Nz, double h, int BC_SOUTH,
double H_SOUTH){
	COMPUTE_INDEX_SOUTH
	COMPUTE_INDEX_NORMAL_VELOCITY_SOUTH
	double K_current = K[in_idx], H_current = H[in_idx];

	U[idx_U] = -2.0/(1.0/K[in_idx+1]+ 1.0/K_current) * (H[in_idx+1]-H_current)/h;
	V[idx_V] = -2.0/(1.0/K[in_idx+Nx]+ 1.0/K_current) * (H[in_idx+Nx]-H_current)/h;
	W[idx_W] = -2.0/(1.0/K[in_idx+stride]+ 1.0/K_current) * (H[in_idx+stride]-H_current)/h;

	if (BC_SOUTH == dirichlet) V[idx_V-Nx] = -K_current * (H_current-H_SOUTH)/(h/2.0);
	if (BC_SOUTH == periodic) V[idx_V-Nx] = -2.0/(1.0/K[in_idx+PERIODIC_SOUTH]+ 1.0/K_current) * (H_current-H[in_idx+PERIODIC_SOUTH])/h;
}

__global__ void velocity_side_WEST(double *U, double *V, double *W,
double *H, double *K, int Nx, int Ny, int Nz, double h, int BC_WEST,
double H_WEST){
	COMPUTE_INDEX_WEST
	COMPUTE_INDEX_NORMAL_VELOCITY_WEST
	double K_current = K[in_idx], H_current = H[in_idx];

	U[idx_U] = -2.0/(1.0/K[in_idx+1]+ 1.0/K_current) * (H[in_idx+1]-H_current)/h;
	V[idx_V] = -2.0/(1.0/K[in_idx+Nx]+ 1.0/K_current) * (H[in_idx+Nx]-H_current)/h;
	W[idx_W] = -2.0/(1.0/K[in_idx+stride]+ 1.0/K_current) * (H[in_idx+stride]-H_current)/h;

	if (BC_WEST == dirichlet) U[idx_U-1] = -K_current * (H_current-H_WEST)/(h/2.0);
	if (BC_WEST == periodic) U[idx_U-1] = -2.0/(1.0/K[in_idx+PERIODIC_WEST]+ 1.0/K_current) * (H_current-H[in_idx+PERIODIC_WEST])/h;
}

__global__ void velocity_side_NORTH(double *U, double *V, double *W,
double *H, double *K, int Nx, int Ny, int Nz, double h, int BC_NORTH,
double H_NORTH){
	COMPUTE_INDEX_NORTH
	COMPUTE_INDEX_NORMAL_VELOCITY_NORTH
	double K_current = K[in_idx], H_current = H[in_idx];
	U[idx_U] = -2.0/(1.0/K[in_idx+1]+ 1.0/K_current) * (H[in_idx+1]-H_current)/h;
	W[idx_W] = -2.0/(1.0/K[in_idx+stride]+ 1.0/K_current) * (H[in_idx+stride]-H_current)/h;
	double result = 0; // default no flux (neumann BC)
	if (BC_NORTH == dirichlet) result = -K_current * (H_NORTH-H_current)/(h/2.0);
	if (BC_NORTH == periodic) result = -2.0/(1.0/K[in_idx+PERIODIC_NORTH]+ 1.0/K_current) * (H[in_idx+PERIODIC_NORTH]-H_current)/h;
	V[idx_V] = result;
}

__global__ void velocity_side_EAST(double *U, double *V, double *W,
double *H, double *K, int Nx, int Ny, int Nz, double h, int BC_EAST,
double H_EAST){
	COMPUTE_INDEX_EAST
	COMPUTE_INDEX_NORMAL_VELOCITY_EAST
	double K_current = K[in_idx], H_current = H[in_idx];
	V[idx_V] = -2.0/(1.0/K[in_idx+Nx]+ 1.0/K_current) * (H[in_idx+Nx]-H_current)/h;
	W[idx_W] = -2.0/(1.0/K[in_idx+stride]+ 1.0/K_current) * (H[in_idx+stride]-H_current)/h;
	double result = 0; // default no flux (neumann BC)
	if (BC_EAST == dirichlet) result = -K_current * (H_EAST-H_current)/(h/2.0);
	if (BC_EAST == periodic) result = -2.0/(1.0/K[in_idx+PERIODIC_EAST]+ 1.0/K_current) * (H[in_idx+PERIODIC_EAST]-H_current)/h;
	U[idx_U] = result;
}

__global__ void velocity_edge_WEST_BOTTOM(double *U, double *V, double *W,
double *H, double *K, int Nx, int Ny, int Nz, double h, int BC_WEST, int BC_BOTTOM,
double H_WEST, double H_BOTTOM){
COMPUTE_INDEX_WEST_BOTTOM
COMPUTE_INDEX_NORMAL_VELOCITY_WEST_BOTTOM
double K_current = K[in_idx], H_current = H[in_idx];
U[idx_U] = -2.0/(1.0/K[in_idx+1]+ 1.0/K_current) * (H[in_idx+1]-H_current)/h;
V[idx_V] = -2.0/(1.0/K[in_idx+Nx]+ 1.0/K_current) * (H[in_idx+Nx]-H_current)/h;
W[idx_W] = -2.0/(1.0/K[in_idx+stride]+ 1.0/K_current) * (H[in_idx+stride]-H_current)/h;

if (BC_WEST == dirichlet) U[idx_U-1] = -K_current * (H_current-H_WEST)/(h/2.0);
if (BC_WEST == periodic) U[idx_U-1] = -2.0/(1.0/K[in_idx+PERIODIC_WEST]+ 1.0/K_current) * (H_current-H[in_idx+PERIODIC_WEST])/h;
if (BC_BOTTOM == dirichlet) W[idx_W-stride] = -K_current * (H_current-H_BOTTOM)/(h/2.0);
if (BC_BOTTOM == periodic) W[idx_W-stride] = -2.0/(1.0/K[in_idx+PERIODIC_BOTTOM]+ 1.0/K_current) * (H_current-H[in_idx+PERIODIC_BOTTOM])/h;
}

__global__ void velocity_edge_WEST_TOP(double *U, double *V, double *W,
double *H, double *K, int Nx, int Ny, int Nz, double h, int BC_WEST, int BC_TOP,
double H_WEST, double H_TOP){
COMPUTE_INDEX_WEST_TOP
COMPUTE_INDEX_NORMAL_VELOCITY_WEST_TOP
double K_current = K[in_idx], H_current = H[in_idx];
U[idx_U] = -2.0/(1.0/K[in_idx+1]+ 1.0/K_current) * (H[in_idx+1]-H_current)/h;
V[idx_V] = -2.0/(1.0/K[in_idx+Nx]+ 1.0/K_current) * (H[in_idx+Nx]-H_current)/h;

if (BC_WEST == dirichlet) U[idx_U-1] = -K_current * (H_current-H_WEST)/(h/2.0);
if (BC_WEST == periodic) U[idx_U-1] = -2.0/(1.0/K[in_idx+PERIODIC_WEST]+ 1.0/K_current) * (H_current-H[in_idx+PERIODIC_WEST])/h;
double result = 0; // default no flux (neumann BC)
if (BC_TOP == dirichlet) result = -K_current * (H_TOP-H_current)/(h/2.0);
if (BC_TOP == periodic) result = -2.0/(1.0/K[in_idx+PERIODIC_TOP]+ 1.0/K_current) * (H[in_idx+PERIODIC_TOP]-H_current)/h;
W[idx_W] = result;
}

__global__ void velocity_edge_EAST_BOTTOM(double *U, double *V, double *W,
double *H, double *K, int Nx, int Ny, int Nz, double h, int BC_EAST, int BC_BOTTOM,
double H_EAST, double H_BOTTOM){
COMPUTE_INDEX_EAST_BOTTOM
COMPUTE_INDEX_NORMAL_VELOCITY_EAST_BOTTOM
double K_current = K[in_idx], H_current = H[in_idx];
V[idx_V] = -2.0/(1.0/K[in_idx+Nx]+ 1.0/K_current) * (H[in_idx+Nx]-H_current)/h;
W[idx_W] = -2.0/(1.0/K[in_idx+stride]+ 1.0/K_current) * (H[in_idx+stride]-H_current)/h;

double result = 0; // default no flux (neumann BC)
if (BC_EAST == dirichlet) result = -K_current * (H_EAST-H_current)/(h/2.0);
if (BC_EAST == periodic) result = -2.0/(1.0/K[in_idx+PERIODIC_EAST]+ 1.0/K_current) * (H[in_idx+PERIODIC_EAST]-H_current)/h;
U[idx_U] = result;
if (BC_BOTTOM == dirichlet) W[idx_W-stride] = -K_current * (H_current-H_BOTTOM)/(h/2.0);
if (BC_BOTTOM == periodic) W[idx_W-stride] = -2.0/(1.0/K[in_idx+PERIODIC_BOTTOM]+ 1.0/K_current) * (H_current-H[in_idx+PERIODIC_BOTTOM])/h;
}

__global__ void velocity_edge_EAST_TOP(double *U, double *V, double *W,
double *H, double *K, int Nx, int Ny, int Nz, double h, int BC_EAST, int BC_TOP,
double H_EAST, double H_TOP){
COMPUTE_INDEX_EAST_TOP
COMPUTE_INDEX_NORMAL_VELOCITY_EAST_TOP
double K_current = K[in_idx], H_current = H[in_idx];
V[idx_V] = -2.0/(1.0/K[in_idx+Nx]+ 1.0/K_current) * (H[in_idx+Nx]-H_current)/h;

double result = 0; // default no flux (neumann BC)
if (BC_EAST == dirichlet) result = -K_current * (H_EAST-H_current)/(h/2.0);
if (BC_EAST == periodic) result = -2.0/(1.0/K[in_idx+PERIODIC_EAST]+ 1.0/K_current) * (H[in_idx+PERIODIC_EAST]-H_current)/h;
U[idx_U] = result;
result = 0;
if (BC_TOP == dirichlet) result = -K_current * (H_TOP-H_current)/(h/2.0);
if (BC_TOP == periodic) result = -2.0/(1.0/K[in_idx+PERIODIC_TOP]+ 1.0/K_current) * (H[in_idx+PERIODIC_TOP]-H_current)/h;
W[idx_W] = result;
}

__global__ void velocity_edge_WEST_SOUTH(double *U, double *V, double *W,
double *H, double *K, int Nx, int Ny, int Nz, double h, int BC_WEST, int BC_SOUTH,
double H_WEST, double H_SOUTH){
COMPUTE_INDEX_WEST_SOUTH
COMPUTE_INDEX_NORMAL_VELOCITY_WEST_SOUTH
double K_current = K[in_idx], H_current = H[in_idx];
U[idx_U] = -2.0/(1.0/K[in_idx+1]+ 1.0/K_current) * (H[in_idx+1]-H_current)/h;
V[idx_V] = -2.0/(1.0/K[in_idx+Nx]+ 1.0/K_current) * (H[in_idx+Nx]-H_current)/h;
W[idx_W] = -2.0/(1.0/K[in_idx+stride]+ 1.0/K_current) * (H[in_idx+stride]-H_current)/h;

if (BC_WEST == dirichlet) U[idx_U-1] = -K_current * (H_current-H_WEST)/(h/2.0);
if (BC_WEST == periodic) U[idx_U-1] = -2.0/(1.0/K[in_idx+PERIODIC_WEST]+ 1.0/K_current) * (H_current-H[in_idx+PERIODIC_WEST])/h;
if (BC_SOUTH == dirichlet) V[idx_V-Nx] = -K_current * (H_current-H_SOUTH)/(h/2.0);
if (BC_SOUTH == periodic) V[idx_V-Nx] = -2.0/(1.0/K[in_idx+PERIODIC_SOUTH]+ 1.0/K_current) * (H_current-H[in_idx+PERIODIC_SOUTH])/h;
}

__global__ void velocity_edge_WEST_NORTH(double *U, double *V, double *W,
double *H, double *K, int Nx, int Ny, int Nz, double h, int BC_WEST, int BC_NORTH,
double H_WEST, double H_NORTH){
COMPUTE_INDEX_WEST_NORTH
COMPUTE_INDEX_NORMAL_VELOCITY_WEST_NORTH
double K_current = K[in_idx], H_current = H[in_idx];
U[idx_U] = -2.0/(1.0/K[in_idx+1]+ 1.0/K_current) * (H[in_idx+1]-H_current)/h;
W[idx_W] = -2.0/(1.0/K[in_idx+stride]+ 1.0/K_current) * (H[in_idx+stride]-H_current)/h;

if (BC_WEST == dirichlet) U[idx_U-1] = -K_current * (H_current-H_WEST)/(h/2.0);
if (BC_WEST == periodic) U[idx_U-1] = -2.0/(1.0/K[in_idx+PERIODIC_WEST]+ 1.0/K_current) * (H_current-H[in_idx+PERIODIC_WEST])/h;

double result = 0; // default no flux (neumann BC)
if (BC_NORTH == dirichlet) result = -K_current * (H_NORTH-H_current)/(h/2.0);
if (BC_NORTH == periodic) result = -2.0/(1.0/K[in_idx+PERIODIC_NORTH]+ 1.0/K_current) * (H[in_idx+PERIODIC_NORTH]-H_current)/h;
V[idx_V] = result;
}

__global__ void velocity_edge_EAST_SOUTH(double *U, double *V, double *W,
double *H, double *K, int Nx, int Ny, int Nz, double h, int BC_EAST, int BC_SOUTH,
double H_EAST, double H_SOUTH){
COMPUTE_INDEX_EAST_SOUTH
COMPUTE_INDEX_NORMAL_VELOCITY_EAST_SOUTH
double K_current = K[in_idx], H_current = H[in_idx];
V[idx_V] = -2.0/(1.0/K[in_idx+Nx]+ 1.0/K_current) * (H[in_idx+Nx]-H_current)/h;
W[idx_W] = -2.0/(1.0/K[in_idx+stride]+ 1.0/K_current) * (H[in_idx+stride]-H_current)/h;

double result = 0; // default no flux (neumann BC)
if (BC_EAST == dirichlet) result = -K_current * (H_EAST-H_current)/(h/2.0);
if (BC_EAST == periodic) result = -2.0/(1.0/K[in_idx+PERIODIC_EAST]+ 1.0/K_current) * (H[in_idx+PERIODIC_EAST]-H_current)/h;
U[idx_U] = result;

if (BC_SOUTH == dirichlet) V[idx_V-Nx] = -K_current * (H_current-H_SOUTH)/(h/2.0);
if (BC_SOUTH == periodic) V[idx_V-Nx] = -2.0/(1.0/K[in_idx+PERIODIC_SOUTH]+ 1.0/K_current) * (H_current-H[in_idx+PERIODIC_SOUTH])/h;
}

__global__ void velocity_edge_EAST_NORTH(double *U, double *V, double *W,
double *H, double *K, int Nx, int Ny, int Nz, double h, int BC_EAST, int BC_NORTH,
double H_EAST, double H_NORTH){
COMPUTE_INDEX_EAST_NORTH
COMPUTE_INDEX_NORMAL_VELOCITY_EAST_NORTH
double K_current = K[in_idx], H_current = H[in_idx];
W[idx_W] = -2.0/(1.0/K[in_idx+stride]+ 1.0/K_current) * (H[in_idx+stride]-H_current)/h;

double result = 0; // default no flux (neumann BC)
if (BC_EAST == dirichlet) result = -K_current * (H_EAST-H_current)/(h/2.0);
if (BC_EAST == periodic) result = -2.0/(1.0/K[in_idx+PERIODIC_EAST]+ 1.0/K_current) * (H[in_idx+PERIODIC_EAST]-H_current)/h;
U[idx_U] = result;

result = 0; // default no flux (neumann BC)
if (BC_NORTH == dirichlet) result = -K_current * (H_NORTH-H_current)/(h/2.0);
if (BC_NORTH == periodic) result = -2.0/(1.0/K[in_idx+PERIODIC_NORTH]+ 1.0/K_current) * (H[in_idx+PERIODIC_NORTH]-H_current)/h;
V[idx_V] = result;
}

__global__ void velocity_edge_SOUTH_BOTTOM(double *U, double *V, double *W,
double *H, double *K, int Nx, int Ny, int Nz, double h, int BC_SOUTH, int BC_BOTTOM,
double H_SOUTH, double H_BOTTOM){
COMPUTE_INDEX_SOUTH_BOTTOM
COMPUTE_INDEX_NORMAL_VELOCITY_SOUTH_BOTTOM
double K_current = K[in_idx], H_current = H[in_idx];
U[idx_U] = -2.0/(1.0/K[in_idx+1]+ 1.0/K_current) * (H[in_idx+1]-H_current)/h;
V[idx_V] = -2.0/(1.0/K[in_idx+Nx]+ 1.0/K_current) * (H[in_idx+Nx]-H_current)/h;
W[idx_W] = -2.0/(1.0/K[in_idx+stride]+ 1.0/K_current) * (H[in_idx+stride]-H_current)/h;

if (BC_SOUTH == dirichlet) V[idx_V-Nx] = -K_current * (H_current-H_SOUTH)/(h/2.0);
if (BC_SOUTH == periodic) V[idx_V-Nx] = -2.0/(1.0/K[in_idx+PERIODIC_SOUTH]+ 1.0/K_current) * (H_current-H[in_idx+PERIODIC_SOUTH])/h;
if (BC_BOTTOM == dirichlet) W[idx_W-stride] = -K_current * (H_current-H_BOTTOM)/(h/2.0);
if (BC_BOTTOM == periodic) W[idx_W-stride] = -2.0/(1.0/K[in_idx+PERIODIC_BOTTOM]+ 1.0/K_current) * (H_current-H[in_idx+PERIODIC_BOTTOM])/h;
}

__global__ void velocity_edge_SOUTH_TOP(double *U, double *V, double *W,
double *H, double *K, int Nx, int Ny, int Nz, double h, int BC_SOUTH, int BC_TOP,
double H_SOUTH, double H_TOP){
COMPUTE_INDEX_SOUTH_TOP
COMPUTE_INDEX_NORMAL_VELOCITY_SOUTH_TOP
double K_current = K[in_idx], H_current = H[in_idx];
U[idx_U] = -2.0/(1.0/K[in_idx+1]+ 1.0/K_current) * (H[in_idx+1]-H_current)/h;
V[idx_V] = -2.0/(1.0/K[in_idx+Nx]+ 1.0/K_current) * (H[in_idx+Nx]-H_current)/h;

if (BC_SOUTH == dirichlet) V[idx_V-Nx] = -K_current * (H_current-H_SOUTH)/(h/2.0);
if (BC_SOUTH == periodic) V[idx_V-Nx] = -2.0/(1.0/K[in_idx+PERIODIC_SOUTH]+ 1.0/K_current) * (H_current-H[in_idx+PERIODIC_SOUTH])/h;
double result = 0; // default no flux (neumann BC)
if (BC_TOP == dirichlet) result = -K_current * (H_TOP-H_current)/(h/2.0);
if (BC_TOP == periodic) result = -2.0/(1.0/K[in_idx+PERIODIC_TOP]+ 1.0/K_current) * (H[in_idx+PERIODIC_TOP]-H_current)/h;
W[idx_W] = result;
}

__global__ void velocity_edge_NORTH_BOTTOM(double *U, double *V, double *W,
double *H, double *K, int Nx, int Ny, int Nz, double h, int BC_NORTH, int BC_BOTTOM,
double H_NORTH, double H_BOTTOM){
COMPUTE_INDEX_NORTH_BOTTOM
COMPUTE_INDEX_NORMAL_VELOCITY_NORTH_BOTTOM
double K_current = K[in_idx], H_current = H[in_idx];
U[idx_U] = -2.0/(1.0/K[in_idx+1]+ 1.0/K_current) * (H[in_idx+1]-H_current)/h;
W[idx_W] = -2.0/(1.0/K[in_idx+stride]+ 1.0/K_current) * (H[in_idx+stride]-H_current)/h;

double result = 0; // default no flux (neumann BC)
if (BC_NORTH == dirichlet) result = -K_current * (H_NORTH-H_current)/(h/2.0);
if (BC_NORTH == periodic) result = -2.0/(1.0/K[in_idx+PERIODIC_NORTH]+ 1.0/K_current) * (H[in_idx+PERIODIC_NORTH]-H_current)/h;
V[idx_V] = result;
if (BC_BOTTOM == dirichlet) W[idx_W-stride] = -K_current * (H_current-H_BOTTOM)/(h/2.0);
if (BC_BOTTOM == periodic) W[idx_W-stride] = -2.0/(1.0/K[in_idx+PERIODIC_BOTTOM]+ 1.0/K_current) * (H_current-H[in_idx+PERIODIC_BOTTOM])/h;
}

__global__ void velocity_edge_NORTH_TOP(double *U, double *V, double *W,
double *H, double *K, int Nx, int Ny, int Nz, double h, int BC_NORTH, int BC_TOP,
double H_NORTH, double H_TOP){
COMPUTE_INDEX_NORTH_TOP
COMPUTE_INDEX_NORMAL_VELOCITY_NORTH_TOP
double K_current = K[in_idx], H_current = H[in_idx];
U[idx_U] = -2.0/(1.0/K[in_idx+1]+ 1.0/K_current) * (H[in_idx+1]-H_current)/h;

double result = 0; // default no flux (neumann BC)
if (BC_NORTH == dirichlet) result = -K_current * (H_NORTH-H_current)/(h/2.0);
if (BC_NORTH == periodic) result = -2.0/(1.0/K[in_idx+PERIODIC_NORTH]+ 1.0/K_current) * (H[in_idx+PERIODIC_NORTH]-H_current)/h;
V[idx_V] = result;
result = 0; // default no flux (neumann BC)
if (BC_TOP == dirichlet) result = -K_current * (H_TOP-H_current)/(h/2.0);
if (BC_TOP == periodic) result = -2.0/(1.0/K[in_idx+PERIODIC_TOP]+ 1.0/K_current) * (H[in_idx+PERIODIC_TOP]-H_current)/h;
W[idx_W] = result;
}

__global__ void velocity_vertex_WEST_SOUTH_BOTTOM(double *U, double *V, double *W,
double *H, double *K, int Nx, int Ny, int Nz, double h, int BC_WEST, int BC_SOUTH, int BC_BOTTOM, double H_WEST, double H_SOUTH, double H_BOTTOM){
COMPUTE_INDEX_WEST_SOUTH_BOTTOM
COMPUTE_INDEX_NORMAL_VELOCITY_WEST_SOUTH_BOTTOM
double K_current = K[in_idx], H_current = H[in_idx];
U[idx_U] = -2.0/(1.0/K[in_idx+1]+ 1.0/K_current) * (H[in_idx+1]-H_current)/h;
V[idx_V] = -2.0/(1.0/K[in_idx+Nx]+ 1.0/K_current) * (H[in_idx+Nx]-H_current)/h;
W[idx_W] = -2.0/(1.0/K[in_idx+stride]+ 1.0/K_current) * (H[in_idx+stride]-H_current)/h;

if (BC_WEST == dirichlet) U[idx_U-1] = -K_current * (H_current-H_WEST)/(h/2.0);
if (BC_WEST == periodic) U[idx_U-1] = -2.0/(1.0/K[in_idx+PERIODIC_WEST]+ 1.0/K_current) * (H_current-H[in_idx+PERIODIC_WEST])/h;
if (BC_SOUTH == dirichlet) V[idx_V-Nx] = -K_current * (H_current-H_SOUTH)/(h/2.0);
if (BC_SOUTH == periodic) V[idx_V-Nx] = -2.0/(1.0/K[in_idx+PERIODIC_SOUTH]+ 1.0/K_current) * (H_current-H[in_idx+PERIODIC_SOUTH])/h;
if (BC_BOTTOM == dirichlet) W[idx_W-stride] = -K_current * (H_current-H_BOTTOM)/(h/2.0);
if (BC_BOTTOM == periodic) W[idx_W-stride] = -2.0/(1.0/K[in_idx+PERIODIC_BOTTOM]+ 1.0/K_current) * (H_current-H[in_idx+PERIODIC_BOTTOM])/h;
}

__global__ void velocity_vertex_WEST_SOUTH_TOP(double *U, double *V, double *W,
double *H, double *K, int Nx, int Ny, int Nz, double h, int BC_WEST, int BC_SOUTH, int BC_TOP, double H_WEST, double H_SOUTH, double H_TOP){
COMPUTE_INDEX_WEST_SOUTH_TOP
COMPUTE_INDEX_NORMAL_VELOCITY_WEST_SOUTH_TOP
double K_current = K[in_idx], H_current = H[in_idx];
U[idx_U] = -2.0/(1.0/K[in_idx+1]+ 1.0/K_current) * (H[in_idx+1]-H_current)/h;
V[idx_V] = -2.0/(1.0/K[in_idx+Nx]+ 1.0/K_current) * (H[in_idx+Nx]-H_current)/h;

if (BC_WEST == dirichlet) U[idx_U-1] = -K_current * (H_current-H_WEST)/(h/2.0);
if (BC_WEST == periodic) U[idx_U-1] = -2.0/(1.0/K[in_idx+PERIODIC_WEST]+ 1.0/K_current) * (H_current-H[in_idx+PERIODIC_WEST])/h;
if (BC_SOUTH == dirichlet) V[idx_V-Nx] = -K_current * (H_current-H_SOUTH)/(h/2.0);
if (BC_SOUTH == periodic) V[idx_V-Nx] = -2.0/(1.0/K[in_idx+PERIODIC_SOUTH]+ 1.0/K_current) * (H_current-H[in_idx+PERIODIC_SOUTH])/h;
double result = 0; // default no flux (neumann BC)
if (BC_TOP == dirichlet) result = -K_current * (H_TOP-H_current)/(h/2.0);
if (BC_TOP == periodic) result = -2.0/(1.0/K[in_idx+PERIODIC_TOP]+ 1.0/K_current) * (H[in_idx+PERIODIC_TOP]-H_current)/h;
W[idx_W] = result;
}

__global__ void velocity_vertex_WEST_NORTH_BOTTOM(double *U, double *V, double *W,
double *H, double *K, int Nx, int Ny, int Nz, double h, int BC_WEST, int BC_NORTH, int BC_BOTTOM, double H_WEST, double H_NORTH, double H_BOTTOM){
COMPUTE_INDEX_WEST_NORTH_BOTTOM
COMPUTE_INDEX_NORMAL_VELOCITY_WEST_NORTH_BOTTOM
double K_current = K[in_idx], H_current = H[in_idx];
U[idx_U] = -2.0/(1.0/K[in_idx+1]+ 1.0/K_current) * (H[in_idx+1]-H_current)/h;
W[idx_W] = -2.0/(1.0/K[in_idx+stride]+ 1.0/K_current) * (H[in_idx+stride]-H_current)/h;

if (BC_WEST == dirichlet) U[idx_U-1] = -K_current * (H_current-H_WEST)/(h/2.0);
if (BC_WEST == periodic) U[idx_U-1] = -2.0/(1.0/K[in_idx+PERIODIC_WEST]+ 1.0/K_current) * (H_current-H[in_idx+PERIODIC_WEST])/h;
double result = 0; // default no flux (neumann BC)
if (BC_NORTH == dirichlet) result = -K_current * (H_NORTH-H_current)/(h/2.0);
if (BC_NORTH == periodic) result = -2.0/(1.0/K[in_idx+PERIODIC_NORTH]+ 1.0/K_current) * (H[in_idx+PERIODIC_NORTH]-H_current)/h;
V[idx_V] = result;
if (BC_BOTTOM == dirichlet) W[idx_W-stride] = -K_current * (H_current-H_BOTTOM)/(h/2.0);
if (BC_BOTTOM == periodic) W[idx_W-stride] = -2.0/(1.0/K[in_idx+PERIODIC_BOTTOM]+ 1.0/K_current) * (H_current-H[in_idx+PERIODIC_BOTTOM])/h;
}

__global__ void velocity_vertex_WEST_NORTH_TOP(double *U, double *V, double *W,
double *H, double *K, int Nx, int Ny, int Nz, double h, int BC_WEST, int BC_NORTH, int BC_TOP, double H_WEST, double H_NORTH, double H_TOP){
COMPUTE_INDEX_WEST_NORTH_TOP
COMPUTE_INDEX_NORMAL_VELOCITY_WEST_NORTH_TOP
double K_current = K[in_idx], H_current = H[in_idx];
U[idx_U] = -2.0/(1.0/K[in_idx+1]+ 1.0/K_current) * (H[in_idx+1]-H_current)/h;

if (BC_WEST == dirichlet) U[idx_U-1] = -K_current * (H_current-H_WEST)/(h/2.0);
if (BC_WEST == periodic) U[idx_U-1] = -2.0/(1.0/K[in_idx+PERIODIC_WEST]+ 1.0/K_current) * (H_current-H[in_idx+PERIODIC_WEST])/h;
double result = 0; // default no flux (neumann BC)
if (BC_NORTH == dirichlet) result = -K_current * (H_NORTH-H_current)/(h/2.0);
if (BC_NORTH == periodic) result = -2.0/(1.0/K[in_idx+PERIODIC_NORTH]+ 1.0/K_current) * (H[in_idx+PERIODIC_NORTH]-H_current)/h;
V[idx_V] = result;
result = 0; // default no flux (neumann BC)
if (BC_TOP == dirichlet) result = -K_current * (H_TOP-H_current)/(h/2.0);
if (BC_TOP == periodic) result = -2.0/(1.0/K[in_idx+PERIODIC_TOP]+ 1.0/K_current) * (H[in_idx+PERIODIC_TOP]-H_current)/h;
W[idx_W] = result;
}

__global__ void velocity_vertex_EAST_SOUTH_BOTTOM(double *U, double *V, double *W,
double *H, double *K, int Nx, int Ny, int Nz, double h, int BC_EAST, int BC_SOUTH, int BC_BOTTOM, double H_EAST, double H_SOUTH, double H_BOTTOM){
COMPUTE_INDEX_EAST_SOUTH_BOTTOM
COMPUTE_INDEX_NORMAL_VELOCITY_EAST_SOUTH_BOTTOM
double K_current = K[in_idx], H_current = H[in_idx];
V[idx_V] = -2.0/(1.0/K[in_idx+Nx]+ 1.0/K_current) * (H[in_idx+Nx]-H_current)/h;
W[idx_W] = -2.0/(1.0/K[in_idx+stride]+ 1.0/K_current) * (H[in_idx+stride]-H_current)/h;

double result = 0; // default no flux (neumann BC)
if (BC_EAST == dirichlet) result = -K_current * (H_EAST-H_current)/(h/2.0);
if (BC_EAST == periodic) result = -2.0/(1.0/K[in_idx+PERIODIC_EAST]+ 1.0/K_current) * (H[in_idx+PERIODIC_EAST]-H_current)/h;
U[idx_U] = result;
if (BC_SOUTH == dirichlet) V[idx_V-Nx] = -K_current * (H_current-H_SOUTH)/(h/2.0);
if (BC_SOUTH == periodic) V[idx_V-Nx] = -2.0/(1.0/K[in_idx+PERIODIC_SOUTH]+ 1.0/K_current) * (H_current-H[in_idx+PERIODIC_SOUTH])/h;
if (BC_BOTTOM == dirichlet) W[idx_W-stride] = -K_current * (H_current-H_BOTTOM)/(h/2.0);
if (BC_BOTTOM == periodic) W[idx_W-stride] = -2.0/(1.0/K[in_idx+PERIODIC_BOTTOM]+ 1.0/K_current) * (H_current-H[in_idx+PERIODIC_BOTTOM])/h;
}

__global__ void velocity_vertex_EAST_SOUTH_TOP(double *U, double *V, double *W,
double *H, double *K, int Nx, int Ny, int Nz, double h, int BC_EAST, int BC_SOUTH, int BC_TOP, double H_EAST, double H_SOUTH, double H_TOP){
COMPUTE_INDEX_EAST_SOUTH_TOP
COMPUTE_INDEX_NORMAL_VELOCITY_EAST_SOUTH_TOP
double K_current = K[in_idx], H_current = H[in_idx];
V[idx_V] = -2.0/(1.0/K[in_idx+Nx]+ 1.0/K_current) * (H[in_idx+Nx]-H_current)/h;

double result = 0; // default no flux (neumann BC)
if (BC_EAST == dirichlet) result = -K_current * (H_EAST-H_current)/(h/2.0);
if (BC_EAST == periodic) result = -2.0/(1.0/K[in_idx+PERIODIC_EAST]+ 1.0/K_current) * (H[in_idx+PERIODIC_EAST]-H_current)/h;
U[idx_U] = result;
if (BC_SOUTH == dirichlet) V[idx_V-Nx] = -K_current * (H_current-H_SOUTH)/(h/2.0);
if (BC_SOUTH == periodic) V[idx_V-Nx] = -2.0/(1.0/K[in_idx+PERIODIC_SOUTH]+ 1.0/K_current) * (H_current-H[in_idx+PERIODIC_SOUTH])/h;
result = 0; // default no flux (neumann BC)
if (BC_TOP == dirichlet) result = -K_current * (H_TOP-H_current)/(h/2.0);
if (BC_TOP == periodic) result = -2.0/(1.0/K[in_idx+PERIODIC_TOP]+ 1.0/K_current) * (H[in_idx+PERIODIC_TOP]-H_current)/h;
W[idx_W] = result;
}

__global__ void velocity_vertex_EAST_NORTH_BOTTOM(double *U, double *V, double *W,
double *H, double *K, int Nx, int Ny, int Nz, double h, int BC_EAST, int BC_NORTH, int BC_BOTTOM, double H_EAST, double H_NORTH, double H_BOTTOM){
COMPUTE_INDEX_EAST_NORTH_BOTTOM
COMPUTE_INDEX_NORMAL_VELOCITY_EAST_NORTH_BOTTOM
double K_current = K[in_idx], H_current = H[in_idx];
W[idx_W] = -2.0/(1.0/K[in_idx+stride]+ 1.0/K_current) * (H[in_idx+stride]-H_current)/h;

double result = 0; // default no flux (neumann BC)
if (BC_EAST == dirichlet) result = -K_current * (H_EAST-H_current)/(h/2.0);
if (BC_EAST == periodic) result = -2.0/(1.0/K[in_idx+PERIODIC_EAST]+ 1.0/K_current) * (H[in_idx+PERIODIC_EAST]-H_current)/h;
U[idx_U] = result;
result = 0; // default no flux (neumann BC)
if (BC_NORTH == dirichlet) result = -K_current * (H_NORTH-H_current)/(h/2.0);
if (BC_NORTH == periodic) result = -2.0/(1.0/K[in_idx+PERIODIC_NORTH]+ 1.0/K_current) * (H[in_idx+PERIODIC_NORTH]-H_current)/h;
V[idx_V] = result;
if (BC_BOTTOM == dirichlet) W[idx_W-stride] = -K_current * (H_current-H_BOTTOM)/(h/2.0);
if (BC_BOTTOM == periodic) W[idx_W-stride] = -2.0/(1.0/K[in_idx+PERIODIC_BOTTOM]+ 1.0/K_current) * (H_current-H[in_idx+PERIODIC_BOTTOM])/h;
}

__global__ void velocity_vertex_EAST_NORTH_TOP(double *U, double *V, double *W,
double *H, double *K, int Nx, int Ny, int Nz, double h, int BC_EAST, int BC_NORTH, int BC_TOP, double H_EAST, double H_NORTH, double H_TOP){
COMPUTE_INDEX_EAST_NORTH_TOP
COMPUTE_INDEX_NORMAL_VELOCITY_EAST_NORTH_TOP
double K_current = K[in_idx], H_current = H[in_idx];

double result = 0; // default no flux (neumann BC)
if (BC_EAST == dirichlet) result = -K_current * (H_EAST-H_current)/(h/2.0);
if (BC_EAST == periodic) result = -2.0/(1.0/K[in_idx+PERIODIC_EAST]+ 1.0/K_current) * (H[in_idx+PERIODIC_EAST]-H_current)/h;
U[idx_U] = result;
result = 0; // default no flux (neumann BC)
if (BC_NORTH == dirichlet) result = -K_current * (H_NORTH-H_current)/(h/2.0);
if (BC_NORTH == periodic) result = -2.0/(1.0/K[in_idx+PERIODIC_NORTH]+ 1.0/K_current) * (H[in_idx+PERIODIC_NORTH]-H_current)/h;
V[idx_V] = result;
result = 0; // default no flux (neumann BC)
if (BC_TOP == dirichlet) result = -K_current * (H_TOP-H_current)/(h/2.0);
if (BC_TOP == periodic) result = -2.0/(1.0/K[in_idx+PERIODIC_TOP]+ 1.0/K_current) * (H[in_idx+PERIODIC_TOP]-H_current)/h;
W[idx_W] = result;
}

#define LAUNCH_KERNEL_SIDE(FACE) \
    velocity_side_##FACE<<<GRIDBLOCK_##FACE>>>(U,V,W,H,K,Nx,Ny,Nz,h,BC_##FACE,H_##FACE)
#define LAUNCH_KERNEL_EDGE(FACE1,FACE2) \
    velocity_edge_##FACE1##_##FACE2<<<GRIDBLOCK_##FACE1##_##FACE2>>>(U,V,W,H,K,Nx,Ny,Nz,h, BC_##FACE1,BC_##FACE2, H_##FACE1,H_##FACE2)
#define LAUNCH_KERNEL_VERTEX(FACE1,FACE2,FACE3) \
    velocity_vertex_##FACE1##_##FACE2##_##FACE3<<<1,1>>>(U,V,W,H,K,Nx,Ny,Nz,h,BC_##FACE1,BC_##FACE2,BC_##FACE3, H_##FACE1,H_##FACE2,H_##FACE3)
#define LAUNCH_KERNEL_INT \
    velocity_int<<<GRIDBLOCK_BOTTOM>>>(U,V,W,H,K,Nx,Ny,Nz,h)

void compute_velocity_from_head(double *U, double *V, double *W, double *H, double *K, int Nx, int Ny, int Nz, double h, int BC_WEST, int BC_EAST, int BC_SOUTH, int BC_NORTH, int BC_BOTTOM, int BC_TOP, double H_WEST, double H_EAST, double H_SOUTH, double H_NORTH, double H_BOTTOM, double H_TOP, dim3 grid, dim3 block){
LAUNCH_KERNEL_INT;
LAUNCH_KERNEL_SIDE(BOTTOM);
LAUNCH_KERNEL_SIDE(TOP);
LAUNCH_KERNEL_SIDE(NORTH);
LAUNCH_KERNEL_SIDE(SOUTH);
LAUNCH_KERNEL_SIDE(WEST);
LAUNCH_KERNEL_SIDE(EAST);

LAUNCH_KERNEL_EDGE(SOUTH,BOTTOM);
LAUNCH_KERNEL_EDGE(SOUTH,TOP);
LAUNCH_KERNEL_EDGE(NORTH,BOTTOM);
LAUNCH_KERNEL_EDGE(NORTH,TOP);

LAUNCH_KERNEL_EDGE(WEST,SOUTH);
LAUNCH_KERNEL_EDGE(WEST,NORTH);
LAUNCH_KERNEL_EDGE(EAST,SOUTH);
LAUNCH_KERNEL_EDGE(EAST,NORTH);

LAUNCH_KERNEL_EDGE(WEST,BOTTOM);
LAUNCH_KERNEL_EDGE(WEST,TOP);
LAUNCH_KERNEL_EDGE(EAST,BOTTOM);
LAUNCH_KERNEL_EDGE(EAST,TOP);

LAUNCH_KERNEL_VERTEX(WEST,SOUTH,BOTTOM);
LAUNCH_KERNEL_VERTEX(WEST,SOUTH,TOP);
LAUNCH_KERNEL_VERTEX(WEST,NORTH,BOTTOM);
LAUNCH_KERNEL_VERTEX(WEST,NORTH,TOP);
LAUNCH_KERNEL_VERTEX(EAST,SOUTH,BOTTOM);
LAUNCH_KERNEL_VERTEX(EAST,SOUTH,TOP);
LAUNCH_KERNEL_VERTEX(EAST,NORTH,BOTTOM);
LAUNCH_KERNEL_VERTEX(EAST,NORTH,TOP);
cudaDeviceSynchronize();
}
```

# GSRB_Smooth_up_residual_3D_bien.cu

```cu
/**
* @file GSRB_Smooth_up_residual_3D.cu
* @brief Gauss Seidel Red Black method implementation (matrix-free style) for solving
* the flow equation with CCMG method
*
* @author Lucas Bessone (contact: lcbessone@gmail.com)
*
* @copyright This file is part of the EU-PAR software.
*            Copyright (C) 2025 Lucas Bessone
*
* @license This program is free software: you can redistribute it and/or modify
*          it under the terms of the GNU General Public License as published by
*          the Free Software Foundation, either version 3 of the License, or
*          (at your option) any later version.
*
*          This program is distributed in the hope that it will be useful,
*          but WITHOUT ANY WARRANTY; without even the implied warranty of
*          MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*          GNU General Public License for more details.
*
*          You should have received a copy of the GNU General Public License
*          along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/
#include <cuda_runtime_api.h>
#include "cublas_v2.h"
#include <iostream>

#include "header/macros_index_kernel.h"// macros para manejo de indices y aplicación de condiciones de borde

#define neumann 0
#define periodic 1
#define dirichlet 2

#define gridXY dim3(grid.x,grid.y)
#define blockXY dim3(block.x,block.y)
#define gridXZ dim3(grid.x,grid.z)
#define blockXZ dim3(block.x,block.z)
#define gridYZ dim3(grid.y,grid.z)
#define blockYZ dim3(block.y,block.z)

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// %%%%%%% Declare auxiliar function for update residual %%%%%%%
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
void update_res(double *rk_1, double *xk, const double *rhs, const double *K,
	double dxdx, int Nx, int Ny, int Nz,
	int BCbottom, int BCtop,
	int BCsouth, int BCnorth,
	int BCwest, int BCeast, bool pin1stCell,
	dim3 grid, dim3 block);

//#######################################################################
// 	routine smooth_GSRB (system A*phi = r)
//	perform one iteration Symetric Gauss-Seidel
//  Red-Black ordering (two half step)
//  phiC <- (rC - sum(AF*phiF) )/AC , F~{E, W, N, S, T, B}
//	The matrix A must be SPD, else, change the stencil
//  for other problems (eg. variable permeability) the stencil must be modified
//#######################################################################
// kernel for interior cells
__global__ void GSRB_int(double *h_in, const double *rhs, const double *K, double dxdx, int Nx, int Ny, int Nz, bool isred){
	int ix = threadIdx.x + blockIdx.x*blockDim.x;
	int iy = threadIdx.y + blockIdx.y*blockDim.y;
	if (ix >= Nx-2 || iy >= Ny-2) return;
	double result;
	int out_idx = (ix+1) + (iy+1)*Nx;
	double aC;
	double KC;
	double KN;
	int offsets[] = {1, Nx, -1, -Nx, stride, -stride};
	for(int iz=1; iz<Nz-1; ++iz){
		out_idx += stride;
		KC = K[out_idx];
		if ((isred && (ix + 1 + iy + 1 + iz) % 2 == 0) || (!isred && (ix + 1 + iy + 1 + iz) % 2 != 0)) {
		    result = 0.0;
		    KN = 0.0;
		    aC = 0.0;
		    for (int i = 0; i < 6; i++) {
		        KN = 2.0 / (1.0 / KC + 1.0 / K[out_idx + offsets[i]]);
		        result += h_in[out_idx + offsets[i]] * KN;
		        aC += KN;
		    }
		    h_in[out_idx] = -(rhs[out_idx] - result / dxdx) / (aC / dxdx);
		}
	}
}

// kernel for boundary face cells
#define KERNEL_SIDE(FACE) GSRB_side_##FACE(double *h_in, const double *rhs, const double *K, double dxdx, int Nx, int Ny, int Nz, bool isred, int BC_##FACE){ \
COMPUTE_INDEX_##FACE \
OFFSETS_##FACE; \
double result = 0.0, KC = K[in_idx], KN = 0.0, aC = 0.0; \
if ((isred && (ix + iy + iz + 1 + 1) % 2 == 0) || (!isred && (ix + iy + iz + 1 + 1) % 2 != 0)){ \
	for (int i = 0; i < 5; i++) { \
	    KN = 2.0 / (1.0/KC  +  1.0/K[in_idx + offsets[i]]); \
	    result += h_in[in_idx + offsets[i]] * KN; \
	    aC += KN; \
	} \
	if (BC_##FACE == periodic) { \
	    KN = 2.0 / (1.0/KC  +  1.0/K[in_idx + PERIODIC_##FACE]); \
	    result += h_in[in_idx + PERIODIC_##FACE] * KN; \
	    aC += KN; \
	} \
	if (BC_##FACE == dirichlet) aC += 2.0*KC; \
	h_in[in_idx] = -(rhs[in_idx] - result/dxdx) / (aC/dxdx); \
} \
}

// kernel for boundary edge cells
#define KERNEL_EDGE(FACE1,FACE2) GSRB_edge_##FACE1##_##FACE2(double *h_in, const double *rhs, const double *K, double dxdx, int Nx, int Ny, int Nz, bool isred, int BC_##FACE1, int BC_##FACE2){ \
COMPUTE_INDEX_##FACE1##_##FACE2 \
OFFSETS_##FACE1##_##FACE2; \
double result = 0.0, KC = K[in_idx], KN = 0.0, aC = 0.0; \
if ((isred && (ix + iy + iz + 1) % 2 == 0) || (!isred && (ix + iy + iz + 1) % 2 != 0)){ \
	for (int i = 0; i < 4; i++) { \
	    KN = 2.0 / (1.0/KC  +  1.0/K[in_idx + offsets[i]]); \
	    result += h_in[in_idx + offsets[i]] * KN; \
	    aC += KN; \
	} \
	if (BC_##FACE1 == periodic) { \
	    KN = 2.0 / (1.0/KC  +  1.0/K[in_idx + PERIODIC_##FACE1]); \
	    result += h_in[in_idx + PERIODIC_##FACE1] * KN; \
	    aC += KN; \
	} \
	if (BC_##FACE2 == periodic) { \
	    KN = 2.0 / (1.0/KC  +  1.0/K[in_idx + PERIODIC_##FACE2]); \
	    result += h_in[in_idx + PERIODIC_##FACE2] * KN; \
	    aC += KN; \
	} \
	if (BC_##FACE1 == dirichlet) aC += 2.0 * KC; \
	if (BC_##FACE2 == dirichlet) aC += 2.0 * KC; \
	h_in[in_idx] = -(rhs[in_idx]  - result/dxdx) / (aC/dxdx); \
} \
}

// kernel for boundary vertex cells
#define KERNEL_VERTEX(FACE1,FACE2,FACE3) GSRB_vertex_##FACE1##_##FACE2##_##FACE3(double *h_in, const double *rhs, const double *K, double dxdx, int Nx, int Ny, int Nz, bool isred, int BC_##FACE1, int BC_##FACE2, int BC_##FACE3){ \
COMPUTE_INDEX_##FACE1##_##FACE2##_##FACE3  \
OFFSETS_##FACE1##_##FACE2##_##FACE3; \
double result = 0.0, KC = K[in_idx], KN = 0.0, aC = 0.0; \
if ((isred && (ix+iy+iz)%2 == 0) || (!isred && (ix+iy+iz)%2 != 0)){ \
	for (int i = 0; i < 3; i++) { \
	    KN = 2.0 / (1.0/KC  +  1.0/K[in_idx + offsets[i]]); \
	    result += h_in[in_idx + offsets[i]] * KN; \
	    aC += KN; \
	} \
	if (BC_##FACE1 == periodic) { \
	    KN = 2.0 / (1.0/KC  +  1.0/K[in_idx + PERIODIC_##FACE1]); \
	    result += h_in[in_idx + PERIODIC_##FACE1] * KN; \
	    aC += KN; \
	} \
	if (BC_##FACE2 == periodic) { \
	    KN = 2.0 / (1.0/KC  +  1.0/K[in_idx + PERIODIC_##FACE2]); \
	    result += h_in[in_idx + PERIODIC_##FACE2] * KN; \
	    aC += KN; \
	} \
	if (BC_##FACE3 == periodic) { \
	    KN = 2.0 / (1.0/KC  +  1.0/K[in_idx + PERIODIC_##FACE3]); \
	    result += h_in[in_idx + PERIODIC_##FACE3] * KN; \
	    aC += KN; \
	} \
	if (BC_##FACE1 == dirichlet) aC += 2.0 * KC; \
	if (BC_##FACE2 == dirichlet) aC += 2.0 * KC; \
	if (BC_##FACE3 == dirichlet) aC += 2.0 * KC; \
	h_in[in_idx] = -(rhs[in_idx]  - result/dxdx) / (aC/dxdx); \
} \
}

// kernel declaration for all domain except interior
__global__ void KERNEL_SIDE(BOTTOM)
__global__ void KERNEL_SIDE(TOP)
__global__ void KERNEL_SIDE(NORTH)
__global__ void KERNEL_SIDE(SOUTH)
__global__ void KERNEL_SIDE(WEST)
__global__ void KERNEL_SIDE(EAST)
// edge y
__global__ void KERNEL_EDGE(WEST,BOTTOM)
__global__ void KERNEL_EDGE(WEST,TOP)
__global__ void KERNEL_EDGE(EAST,BOTTOM)
__global__ void KERNEL_EDGE(EAST,TOP)
// edge z
__global__ void KERNEL_EDGE(WEST,SOUTH)
__global__ void KERNEL_EDGE(WEST,NORTH)
__global__ void KERNEL_EDGE(EAST,SOUTH)
__global__ void KERNEL_EDGE(EAST,NORTH)
// edge x
__global__ void KERNEL_EDGE(SOUTH,BOTTOM)
__global__ void KERNEL_EDGE(SOUTH,TOP)
__global__ void KERNEL_EDGE(NORTH,BOTTOM)
__global__ void KERNEL_EDGE(NORTH,TOP)

__global__ void KERNEL_VERTEX(WEST,SOUTH,BOTTOM)
__global__ void KERNEL_VERTEX(WEST,SOUTH,TOP)
__global__ void KERNEL_VERTEX(WEST,NORTH,BOTTOM)
__global__ void KERNEL_VERTEX(WEST,NORTH,TOP)
__global__ void KERNEL_VERTEX(EAST,SOUTH,BOTTOM)
__global__ void KERNEL_VERTEX(EAST,SOUTH,TOP)
__global__ void KERNEL_VERTEX(EAST,NORTH,BOTTOM)
__global__ void KERNEL_VERTEX(EAST,NORTH,TOP)

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// %%% ITERACION GAUSS SEIDEL RED BLACK (two half steps) %%%%%%%
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
void smooth_GSRB(double *xk, const double *rhs, double *rk_1, const double *K,
	double dxdx, int Nx, int Ny, int Nz, int BCbottom, int BCtop, int BCsouth, int BCnorth, int BCwest, int BCeast, bool pin1stCell,
	int itMAX, bool updateRes, dim3 grid, dim3 block){
	for(int i = 0; i<itMAX; i++){
		GSRB_int<<<gridXY,blockXY>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,true);
		GSRB_side_BOTTOM<<<gridXY,blockXY>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,true,BCbottom);
		GSRB_side_TOP	<<<gridXY,blockXY>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,true,BCtop);
		GSRB_side_SOUTH<<<gridXZ,blockXZ>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,true,BCsouth);
		GSRB_side_NORTH<<<gridXZ,blockXZ>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,true,BCnorth);
		GSRB_side_WEST<<<gridYZ,blockYZ>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,true,BCwest);
		GSRB_side_EAST<<<gridYZ,blockYZ>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,true,BCeast);
		GSRB_edge_SOUTH_BOTTOM<<<grid.x,block.x>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,true,BCsouth,BCbottom);
		GSRB_edge_SOUTH_TOP	<<<grid.x,block.x>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,true,BCsouth,BCtop);
		GSRB_edge_NORTH_BOTTOM<<<grid.x,block.x>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,true,BCnorth,BCbottom);
		GSRB_edge_NORTH_TOP	<<<grid.x,block.x>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,true,BCnorth,BCtop);
		GSRB_edge_WEST_SOUTH<<<grid.z,block.z>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,true,BCwest,BCsouth);
		GSRB_edge_EAST_SOUTH<<<grid.z,block.z>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,true,BCeast,BCsouth);
		GSRB_edge_WEST_NORTH<<<grid.z,block.z>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,true,BCwest,BCnorth);
		GSRB_edge_EAST_NORTH<<<grid.z,block.z>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,true,BCeast,BCnorth);
		GSRB_edge_WEST_BOTTOM	<<<grid.y,block.y>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,true,BCwest,BCbottom);
		GSRB_edge_WEST_TOP	<<<grid.y,block.y>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,true,BCwest,BCtop);
		GSRB_edge_EAST_BOTTOM	<<<grid.y,block.y>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,true,BCeast,BCbottom);
		GSRB_edge_EAST_TOP	<<<grid.y,block.y>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,true,BCeast,BCtop);
		GSRB_vertex_WEST_SOUTH_BOTTOM<<<1,1>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,true,BCwest,BCsouth,BCbottom);//,pin1stCell);
		GSRB_vertex_WEST_SOUTH_TOP<<<1,1>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,true,BCwest,BCsouth,BCtop);
		GSRB_vertex_EAST_SOUTH_BOTTOM<<<1,1>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,true,BCeast,BCsouth,BCbottom);
		GSRB_vertex_EAST_SOUTH_TOP<<<1,1>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,true,BCeast,BCsouth,BCtop);
		GSRB_vertex_EAST_NORTH_BOTTOM<<<1,1>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,true,BCeast,BCnorth,BCbottom);
		GSRB_vertex_EAST_NORTH_TOP<<<1,1>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,true,BCeast,BCnorth,BCtop);
		GSRB_vertex_WEST_NORTH_BOTTOM<<<1,1>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,true,BCwest,BCnorth,BCbottom);
		GSRB_vertex_WEST_NORTH_TOP<<<1,1>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,true,BCwest,BCnorth,BCtop);
		cudaDeviceSynchronize();

		GSRB_int		<<<gridXY,blockXY>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,false);
		GSRB_side_BOTTOM<<<gridXY,blockXY>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,false,BCbottom);
		GSRB_side_TOP	<<<gridXY,blockXY>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,false,BCtop);
		GSRB_side_SOUTH<<<gridXZ,blockXZ>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,false,BCsouth);
		GSRB_side_NORTH<<<gridXZ,blockXZ>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,false,BCnorth);
		GSRB_side_WEST<<<gridYZ,blockYZ>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,false,BCwest);
		GSRB_side_EAST<<<gridYZ,blockYZ>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,false,BCeast);
		GSRB_edge_SOUTH_BOTTOM<<<grid.x,block.x>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,false,BCsouth,BCbottom);
		GSRB_edge_SOUTH_TOP	<<<grid.x,block.x>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,false,BCsouth,BCtop);
		GSRB_edge_NORTH_BOTTOM<<<grid.x,block.x>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,false,BCnorth,BCbottom);
		GSRB_edge_NORTH_TOP	<<<grid.x,block.x>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,false,BCnorth,BCtop);
		GSRB_edge_WEST_SOUTH	<<<grid.z,block.z>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,false,BCwest,BCsouth);
		GSRB_edge_EAST_SOUTH	<<<grid.z,block.z>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,false,BCeast,BCsouth);
		GSRB_edge_WEST_NORTH	<<<grid.z,block.z>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,false,BCwest,BCnorth);
		GSRB_edge_EAST_NORTH	<<<grid.z,block.z>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,false,BCeast,BCnorth);
		GSRB_edge_WEST_BOTTOM	<<<grid.y,block.y>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,false,BCwest,BCbottom);
		GSRB_edge_WEST_TOP	<<<grid.y,block.y>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,false,BCwest,BCtop);
		GSRB_edge_EAST_BOTTOM	<<<grid.y,block.y>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,false,BCeast,BCbottom);
		GSRB_edge_EAST_TOP	<<<grid.y,block.y>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,false,BCeast,BCtop);
		GSRB_vertex_WEST_SOUTH_BOTTOM<<<1,1>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,false,BCwest,BCsouth,BCbottom);//,pin1stCell);
		GSRB_vertex_WEST_SOUTH_TOP<<<1,1>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,false,BCwest,BCsouth,BCtop);
		GSRB_vertex_EAST_SOUTH_BOTTOM<<<1,1>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,false,BCeast,BCsouth,BCbottom);
		GSRB_vertex_EAST_SOUTH_TOP<<<1,1>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,false,BCeast,BCsouth,BCtop);
		GSRB_vertex_EAST_NORTH_BOTTOM<<<1,1>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,false,BCeast,BCnorth,BCbottom);
		GSRB_vertex_EAST_NORTH_TOP<<<1,1>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,false,BCeast,BCnorth,BCtop);
		GSRB_vertex_WEST_NORTH_BOTTOM<<<1,1>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,false,BCwest,BCnorth,BCbottom);
		GSRB_vertex_WEST_NORTH_TOP<<<1,1>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,false,BCwest,BCnorth,BCtop);
		cudaDeviceSynchronize();
	}
	if(updateRes) update_res(rk_1,xk,rhs,K,dxdx,Nx,Ny,Nz,
		BCbottom,BCtop,BCsouth,BCnorth,BCwest,BCeast,pin1stCell,
		grid,block);
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// %%%%%%%%   SOLVE COARSEST SYSTEM WITH GS-RB   %%%%%%%%%%%%%%%
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
void SolveCoarseSystemGSRB(double *xnew, const double *rhs, double *r, const double *K, double dxdx, int Nx, int Ny, int Nz, const int itMAX,
	dim3 grid, dim3 block,
	cublasHandle_t handle,
	int BCbottom, int BCtop,
	int BCsouth, int BCnorth,
	int BCwest, int BCeast, bool pin1stCell){
	double *rr_new; cudaMalloc(&rr_new,sizeof(double));
	double *rr_new_h; rr_new_h = new double[1];
	int iter = 0;
	*rr_new_h = 1.0;
	while( (pow((*rr_new_h),0.5) > 1e-16 ) && iter<itMAX){
		smooth_GSRB(xnew,rhs,r,K,dxdx,Nx,Ny,Nz,
			BCbottom,BCtop,BCsouth,
			BCnorth,BCwest,BCeast,
			pin1stCell,4,false,
			grid,block);
		iter+=4;
		update_res(r,xnew,rhs,K,dxdx,Nx,Ny,Nz,
		BCbottom,BCtop,BCsouth,BCnorth,BCwest,BCeast,pin1stCell,
		grid,block);
		cublasDdot(handle,Nx*Ny*Nz,r,1,r,1, rr_new); cudaDeviceSynchronize();
		cudaMemcpy(rr_new_h, rr_new, sizeof(double), cudaMemcpyDeviceToHost);
	}
	cudaFree(rr_new);
	delete [] rr_new_h;
}

```

# GSRB_Smooth_up_residual_3D_macro.cu

```cu
/**
* @file GSRB_Smooth_up_residual_3D.cu
* @brief Gauss Seidel Red Black method implementation (matrix-free style) for solving
* the flow equation with CCMG method
*
* @author Lucas Bessone (contact: lcbessone@gmail.com)
*
* @copyright This file is part of the EU-PAR software.
*            Copyright (C) 2025 Lucas Bessone
*
* @license This program is free software: you can redistribute it and/or modify
*          it under the terms of the GNU General Public License as published by
*          the Free Software Foundation, either version 3 of the License, or
*          (at your option) any later version.
*
*          This program is distributed in the hope that it will be useful,
*          but WITHOUT ANY WARRANTY; without even the implied warranty of
*          MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*          GNU General Public License for more details.
*
*          You should have received a copy of the GNU General Public License
*          along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/
#include <cuda_runtime_api.h>
#include "cublas_v2.h"
#include <iostream>

#define neumann 0
#define periodic 1
#define dirichlet 2

#define gridXY dim3(grid.x,grid.y)
#define blockXY dim3(block.x,block.y)
#define gridXZ dim3(grid.x,grid.z)
#define blockXZ dim3(block.x,block.z)
#define gridYZ dim3(grid.y,grid.z)
#define blockYZ dim3(block.y,block.z)

#define stride (Nx*Ny)
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// %%%%%%% Declare auxiliar function for update residual %%%%%%%
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
void update_res(double *rk_1, double *xk, const double *rhs, const double *K,
	double dxdx, int Nx, int Ny, int Nz,
	int BCbottom, int BCtop,
	int BCsouth, int BCnorth,
	int BCwest, int BCeast, bool pin1stCell,
	dim3 grid, dim3 block);

//#######################################################################
// 	routine smooth_GSRB (system A*phi = r)
//	perform one iteration Symetric Gauss-Seidel
//  Red-Black ordering (two half step)
//  phiC <- (rC - sum(AF*phiF) )/AC , F~{E, W, N, S, T, B}
//	The matrix A must be SPD, else, change the stencil
//  for other problems (eg. variable permeability) the stencil must be modified
//#######################################################################
__global__ void GSRB_int(double *h_in, const double *rhs, const double *K, double dxdx, int Nx, int Ny, int Nz, bool isred){
	int ix = threadIdx.x + blockIdx.x*blockDim.x;
	int iy = threadIdx.y + blockIdx.y*blockDim.y;
	if (ix >= Nx-2 || iy >= Ny-2) return;
	double result;
	int out_idx = (ix+1) + (iy+1)*Nx;
	double aC;
	double KC;
	double KN;
	int offsets[] = {1, Nx, -1, -Nx, stride, -stride};
	for(int iz=1; iz<Nz-1; ++iz){
		out_idx += stride;
		KC = K[out_idx];
		if ((isred && (ix + 1 + iy + 1 + iz) % 2 == 0) || (!isred && (ix + 1 + iy + 1 + iz) % 2 != 0)) {
		    result = 0.0;
		    KN = 0.0;
		    aC = 0.0;
		    for (int i = 0; i < 6; i++) {
		        KN = 2.0 / (1.0 / KC + 1.0 / K[out_idx + offsets[i]]);
		        result += h_in[out_idx + offsets[i]] * KN;
		        aC += KN;
		    }
		    h_in[out_idx] = -(rhs[out_idx] - result / dxdx) / (aC / dxdx);
		}
	}
}



#define KERNEL_SIDE(FACE,BC_FACE) GSRB_side_##FACE(double *h_in, const double *rhs, const double *K, double dxdx, int Nx, int Ny, int Nz, bool isred, int BC_FACE){ \
COMPUTE_INDEX_##FACE \
int offsets[5] = OFFSETS5(FACE,Nx,Ny) \
double result = 0.0, KC = K[in_idx], KN = 0.0, aC = 0.0; \
if ((isred && (ix + 1 + iy + 1 + iz) % 2 == 0) || (!isred && (ix + 1 + iy + 1 + iz) % 2 != 0)){ \
	for (int i = 0; i < 5; i++) { \
	    KN = 2.0 / (1.0/KC  +  1.0/K[in_idx + offsets[i]]); \
	    result += h_in[in_idx + offsets[i]] * KN; \
	    aC += KN; \
	} \
	if (BC_FACE == periodic) { \
	    KN = 2.0 / (1.0/KC  +  1.0/K[in_idx + PERIODIC_##FACE]); \
	    result += h_in[in_idx + PERIODIC_##FACE] * KN; \
	    aC += KN; \
	} \
	if (BC_FACE == dirichlet) aC += 2.0 * KC; \
	h_in[in_idx] = -(rhs[in_idx] - result/dxdx) / (aC/dxdx); \
} \
}

__global__ void KERNEL_SIDE(BOTTOM,BC_BOTTOM)
__global__ void KERNEL_SIDE(TOP,BC_TOP)
__global__ void KERNEL_SIDE(NORTH,BC_NORTH)
__global__ void KERNEL_SIDE(SOUTH,BC_SOUTH)
__global__ void KERNEL_SIDE(WEST,BC_WEST)
__global__ void KERNEL_SIDE(EAST,BC_EAST)

// GSRB_side_BOTTOM<<<,>>>(h_in,rhs,K,dxdx,Nx,Ny,Nz,isred,BC_BOTTOM);


#define KERNEL_EDGE(FACE1,FACE2) GSRB_edge_##FACE1##_##FACE2(double *h_in, const double *rhs, const double *K, double dxdx, int Nx, int Ny, int Nz, bool isred, int BC_##FACE1, int BC_##FACE2){ \
COMPUTE_INDEX_##FACE1##_##FACE2 \
int offsets[4] = OFFSETS4(FACE1,FACE2,Nx,Ny) \
double result = 0.0, KC = K[in_idx], KN = 0.0, aC = 0.0; \
if ((isred && (ix + 1 + iy + 1 + iz) % 2 == 0) || (!isred && (ix + 1 + iy + 1 + iz) % 2 != 0)){ \
	for (int i = 0; i < 4; i++) { \
	    KN = 2.0 / (1.0/KC  +  1.0/K[in_idx + offsets[i]]); \
	    result += h_in[in_idx + offsets[i]] * KN; \
	    aC += KN; \
	} \
	if (BC_##FACE1 == periodic) { \
	    KN = 2.0 / (1.0/KC  +  1.0/K[in_idx + PERIODIC_##FACE1]); \
	    result += h_in[in_idx + PERIODIC_##FACE1] * KN; \
	    aC += KN; \
	} \
	if (BC_##FACE2 == periodic) { \
	    KN = 2.0 / (1.0/KC  +  1.0/K[in_idx + PERIODIC_##FACE2]); \
	    result += h_in[in_idx + PERIODIC_##FACE2] * KN; \
	    aC += KN; \
	} \
	if (BC_##FACE1 == dirichlet) aC += 2.0 * KC; \
	if (BC_##FACE2 == dirichlet) aC += 2.0 * KC; \
	h_in[in_idx] = -(rhs[in_idx]  - result/dxdx) / (aC/dxdx); \
} \
}
// edge y
__global__ void KERNEL_EDGE(WEST,BOTTOM)
__global__ void KERNEL_EDGE(WEST,TOP)
__global__ void KERNEL_EDGE(EAST,BOTTOM)
__global__ void KERNEL_EDGE(EAST,TOP)
// edge z
__global__ void KERNEL_EDGE(WEST,SOUTH)
__global__ void KERNEL_EDGE(WEST,NORTH)
__global__ void KERNEL_EDGE(EAST,SOUTH)
__global__ void KERNEL_EDGE(EAST,NORTH)
// edge x
__global__ void KERNEL_EDGE(SOUTH,BOTTOM)
__global__ void KERNEL_EDGE(SOUTH,TOP)
__global__ void KERNEL_EDGE(NORTH,BOTTOM)
__global__ void KERNEL_EDGE(NORTH,TOP)


#define KERNEL_VERTEX(FACE1,FACE2,FACE3) GSRB_vertex_##FACE1##_##FACE2##_##FACE3(double *h_in, const double *rhs, const double *K, double dxdx, int Nx, int Ny, int Nz, bool isred, int BC_##FACE1, int BC_##FACE2, int BC_##FACE3){ \
COMPUTE_INDEX_##FACE1##_##FACE2##_##FACE3;  \
int offsets[3] = OFFSETS3(FACE1,FACE2,FACE3,Nx,Ny) \
double result = 0.0, KC = K[in_idx], KN = 0.0, aC = 0.0; \
if ((isred && (ix+iy+iz)%2 == 0) || (!isred && (ix+iy+iz)%2 != 0)){ \
	for (int i = 0; i < 3; i++) { \
	    KN = 2.0 / (1.0/KC  +  1.0/K[in_idx + offsets[i]]); \
	    result += h_in[in_idx + offsets[i]] * KN; \
	    aC += KN; \
	} \
	if (BC_##FACE1 == periodic) { \
	    KN = 2.0 / (1.0/KC  +  1.0/K[in_idx + PERIODIC_##FACE1]); \
	    result += h_in[in_idx + PERIODIC_##FACE1] * KN; \
	    aC += KN; \
	} \
	if (BC_##FACE2 == periodic) { \
	    KN = 2.0 / (1.0/KC  +  1.0/K[in_idx + PERIODIC_##FACE2]); \
	    result += h_in[in_idx + PERIODIC_##FACE2] * KN; \
	    aC += KN; \
	} \
	if (BC_##FACE3 == periodic) { \
	    KN = 2.0 / (1.0/KC  +  1.0/K[in_idx + PERIODIC_##FACE3]); \
	    result += h_in[in_idx + PERIODIC_##FACE3] * KN; \
	    aC += KN; \
	} \
	if (BC_##FACE1 == dirichlet) aC += 2.0 * KC; \
	if (BC_##FACE2 == dirichlet) aC += 2.0 * KC; \
	if (BC_##FACE3 == dirichlet) aC += 2.0 * KC; \
	h_in[in_idx] = -(rhs[in_idx]  - result/dxdx) / (aC/dxdx); \
} \
}

__global__ void KERNEL_VERTEX(WEST,SOUTH,BOTTOM)
__global__ void KERNEL_VERTEX(WEST,SOUTH,TOP)
__global__ void KERNEL_VERTEX(WEST,NORTH,BOTTOM)
__global__ void KERNEL_VERTEX(WEST,NORTH,TOP)
__global__ void KERNEL_VERTEX(EAST,SOUTH,BOTTOM)
__global__ void KERNEL_VERTEX(EAST,SOUTH,TOP)
__global__ void KERNEL_VERTEX(EAST,NORTH,BOTTOM)
__global__ void KERNEL_VERTEX(EAST,NORTH,TOP)

// GSRB_side_BOTTOM<<<,>>>(h_in,rhs,K,dxdx,Nx,Ny,Nz,isred,BC_BOTTOM);
// GSRB_edge_BOTTOM<<<,>>>(h_in,rhs,K,dxdx,Nx,Ny,Nz,isred,BC_BOTTOM);
// GSRB_vertex_BOTTOM<<<,>>>(h_in,rhs,K,dxdx,Nx,Ny,Nz,isred,BC_BOTTOM);

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// %%% ITERACION GAUSS SEIDEL RED BLACK (two half steps) %%%%%%%
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
void smooth_GSRB(double *xk, const double *rhs, double *rk_1, const double *K,
	double dxdx, int Nx, int Ny, int Nz, int BC_BOTTOM, int BC_TOP, int BC_SOUTH, int BC_NORTH, int BC_WEST, int BC_EAST, bool pin1stCell,
	int itMAX, bool updateRes, dim3 grid, dim3 block){
	for(int i = 0; i<itMAX; i++){
		GSRB_int<<<gridXY,blockXY>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,true);
		GSRB_side_BOTTOM<<<gridXY,blockXY>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,true,BC_BOTTOM);
		GSRB_side_TOP	<<<gridXY,blockXY>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,true,BC_TOP);
		GSRB_side_SOUTH<<<gridXZ,blockXZ>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,true,BC_SOUTH);
		GSRB_side_NORTH<<<gridXZ,blockXZ>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,true,BC_NORTH);
		GSRB_side_WEST<<<gridYZ,blockYZ>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,true,BC_WEST);
		GSRB_side_EAST<<<gridYZ,blockYZ>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,true,BC_EAST);
		//edge x
		GSRB_edge_SOUTH_BOTTOM<<<grid.x,block.x>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,true,BC_SOUTH,BC_BOTTOM);
		GSRB_edge_SOUTH_TOP	<<<grid.x,block.x>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,true,BC_SOUTH,BC_TOP);
		GSRB_edge_NORTH_BOTTOM<<<grid.x,block.x>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,true,BC_NORTH,BC_BOTTOM);
		GSRB_edge_NORTH_TOP	<<<grid.x,block.x>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,true,BC_NORTH,BC_TOP);
		//edge z
		GSRB_edge_WEST_SOUTH<<<grid.z,block.z>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,true,BC_WEST,BC_SOUTH);
		GSRB_edge_EAST_SOUTH<<<grid.z,block.z>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,true,BC_EAST,BC_SOUTH);
		GSRB_edge_WEST_NORTH<<<grid.z,block.z>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,true,BC_WEST,BC_NORTH);
		GSRB_edge_EAST_NORTH<<<grid.z,block.z>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,true,BC_EAST,BC_NORTH);
		//edge y
		GSRB_edge_WEST_BOTTOM	<<<grid.y,block.y>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,true,BC_WEST,BC_BOTTOM);
		GSRB_edge_WEST_TOP	<<<grid.y,block.y>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,true,BC_WEST,BC_TOP);
		GSRB_edge_EAST_BOTTOM	<<<grid.y,block.y>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,true,BC_EAST,BC_BOTTOM);
		GSRB_edge_EAST_TOP	<<<grid.y,block.y>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,true,BC_EAST,BC_TOP);

		GSRB_vertex_WEST_SOUTH_BOTTOM<<<1,1>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,true,BC_WEST,BC_SOUTH,BC_BOTTOM);//,pin1stCell);
		GSRB_vertex_WEST_SOUTH_TOP<<<1,1>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,true,BC_WEST,BC_SOUTH,BC_TOP);
		GSRB_vertex_EAST_SOUTH_BOTTOM<<<1,1>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,true,BC_EAST,BC_SOUTH,BC_BOTTOM);
		GSRB_vertex_EAST_SOUTH_TOP<<<1,1>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,true,BC_EAST,BC_SOUTH,BC_TOP);
		GSRB_vertex_EAST_NORTH_BOTTOM<<<1,1>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,true,BC_EAST,BC_NORTH,BC_BOTTOM);
		GSRB_vertex_EAST_NORTH_TOP<<<1,1>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,true,BC_EAST,BC_NORTH,BC_TOP);
		GSRB_vertex_WEST_NORTH_BOTTOM<<<1,1>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,true,BC_WEST,BC_NORTH,BC_BOTTOM);
		GSRB_vertex_WEST_NORTH_TOP<<<1,1>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,true,BC_WEST,BC_NORTH,BC_TOP);
		cudaDeviceSynchronize();

		GSRB_int<<<gridXY,blockXY>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,false);
		GSRB_side_BOTTOM<<<gridXY,blockXY>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,false,BC_BOTTOM);
		GSRB_side_TOP	<<<gridXY,blockXY>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,false,BC_TOP);
		GSRB_side_SOUTH<<<gridXZ,blockXZ>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,false,BC_SOUTH);
		GSRB_side_NORTH<<<gridXZ,blockXZ>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,false,BC_NORTH);
		GSRB_side_WEST<<<gridYZ,blockYZ>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,false,BC_WEST);
		GSRB_side_EAST<<<gridYZ,blockYZ>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,false,BC_EAST);
		//edge x
		GSRB_edge_SOUTH_BOTTOM<<<grid.x,block.x>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,false,BC_SOUTH,BC_BOTTOM);
		GSRB_edge_SOUTH_TOP	<<<grid.x,block.x>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,false,BC_SOUTH,BC_TOP);
		GSRB_edge_NORTH_BOTTOM<<<grid.x,block.x>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,false,BC_NORTH,BC_BOTTOM);
		GSRB_edge_NORTH_TOP	<<<grid.x,block.x>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,false,BC_NORTH,BC_TOP);
		//edge z
		GSRB_edge_WEST_SOUTH<<<grid.z,block.z>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,false,BC_WEST,BC_SOUTH);
		GSRB_edge_EAST_SOUTH<<<grid.z,block.z>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,false,BC_EAST,BC_SOUTH);
		GSRB_edge_WEST_NORTH<<<grid.z,block.z>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,false,BC_WEST,BC_NORTH);
		GSRB_edge_EAST_NORTH<<<grid.z,block.z>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,false,BC_EAST,BC_NORTH);
		//edge y
		GSRB_edge_WEST_BOTTOM	<<<grid.y,block.y>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,false,BC_WEST,BC_BOTTOM);
		GSRB_edge_WEST_TOP	<<<grid.y,block.y>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,false,BC_WEST,BC_TOP);
		GSRB_edge_EAST_BOTTOM	<<<grid.y,block.y>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,false,BC_EAST,BC_BOTTOM);
		GSRB_edge_EAST_TOP	<<<grid.y,block.y>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,false,BC_EAST,BC_TOP);

		GSRB_vertex_WEST_SOUTH_BOTTOM<<<1,1>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,false,BC_WEST,BC_SOUTH,BC_BOTTOM);//,pin1stCell);
		GSRB_vertex_WEST_SOUTH_TOP<<<1,1>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,false,BC_WEST,BC_SOUTH,BC_TOP);
		GSRB_vertex_EAST_SOUTH_BOTTOM<<<1,1>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,false,BC_EAST,BC_SOUTH,BC_BOTTOM);
		GSRB_vertex_EAST_SOUTH_TOP<<<1,1>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,false,BC_EAST,BC_SOUTH,BC_TOP);
		GSRB_vertex_EAST_NORTH_BOTTOM<<<1,1>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,false,BC_EAST,BC_NORTH,BC_BOTTOM);
		GSRB_vertex_EAST_NORTH_TOP<<<1,1>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,false,BC_EAST,BC_NORTH,BC_TOP);
		GSRB_vertex_WEST_NORTH_BOTTOM<<<1,1>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,false,BC_WEST,BC_NORTH,BC_BOTTOM);
		GSRB_vertex_WEST_NORTH_TOP<<<1,1>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,false,BC_WEST,BC_NORTH,BC_TOP);
		cudaDeviceSynchronize();
	}
	if(updateRes) update_res(rk_1,xk,rhs,K,dxdx,Nx,Ny,Nz,
		BC_BOTTOM,BC_TOP,BC_SOUTH,BC_NORTH,BC_WEST,BC_EAST,pin1stCell,
		grid,block);
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// %%%%%%%%   SOLVE COARSEST SYSTEM WITH GS-RB   %%%%%%%%%%%%%%%
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
void SolveCoarseSystemGSRB(double *xnew, const double *rhs, double *r, const double *K, double dxdx, int Nx, int Ny, int Nz, const int itMAX,
	dim3 grid, dim3 block,
	cublasHandle_t handle,
	int BC_BOTTOM, int BC_TOP, int BC_SOUTH, int BC_NORTH, int BC_WEST, int BC_EAST, bool pin1stCell){
	double *rr_new; cudaMalloc(&rr_new,sizeof(double));
	double *rr_new_h; rr_new_h = new double[1];
	int iter = 0;
	*rr_new_h = 1.0;
	while( (pow((*rr_new_h),0.5) > 1e-16 ) && iter<itMAX){
		smooth_GSRB(xnew,rhs,r,K,dxdx,Nx,Ny,Nz,
			BC_BOTTOM,BC_TOP,BC_SOUTH,BC_NORTH,BC_WEST,BC_EAST,
			pin1stCell,4,false,
			grid,block);
		iter+=4;
		update_res(r,xnew,rhs,K,dxdx,Nx,Ny,Nz,
		BC_BOTTOM,BC_TOP,BC_SOUTH,BC_NORTH,BC_WEST,BC_EAST,pin1stCell,
		grid,block);
		cublasDdot(handle,Nx*Ny*Nz,r,1,r,1, rr_new); cudaDeviceSynchronize();
		cudaMemcpy(rr_new_h, rr_new, sizeof(double), cudaMemcpyDeviceToHost);
	}
	cudaFree(rr_new);
	delete [] rr_new_h;
}

```

# GSRB_Smooth_up_residual_3D_orig.cu

```cu
/**
* @file GSRB_Smooth_up_residual_3D.cu
* @brief Gauss Seidel Red Black method implementation (matrix-free style) for solving
* the flow equation with CCMG method
*
* @author Lucas Bessone (contact: lcbessone@gmail.com)
*
* @copyright This file is part of the EU-PAR software.
*            Copyright (C) 2025 Lucas Bessone
*
* @license This program is free software: you can redistribute it and/or modify
*          it under the terms of the GNU General Public License as published by
*          the Free Software Foundation, either version 3 of the License, or
*          (at your option) any later version.
*
*          This program is distributed in the hope that it will be useful,
*          but WITHOUT ANY WARRANTY; without even the implied warranty of
*          MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*          GNU General Public License for more details.
*
*          You should have received a copy of the GNU General Public License
*          along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/
#include <cuda_runtime_api.h>
#include "cublas_v2.h"
#include <iostream>


#define neumann 0
#define periodic 1
#define dirichlet 2

#define gridXY dim3(grid.x,grid.y)
#define blockXY dim3(block.x,block.y)
#define gridXZ dim3(grid.x,grid.z)
#define blockXZ dim3(block.x,block.z)
#define gridYZ dim3(grid.y,grid.z)
#define blockYZ dim3(block.y,block.z)
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// %%%%%%% Declare auxiliar function for update residual %%%%%%%
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
void update_res(double *rk_1, double *xk, const double *rhs, const double *K,
	double dxdx, int Nx, int Ny, int Nz,
	int BCbottom, int BCtop,
	int BCsouth, int BCnorth,
	int BCwest, int BCeast, bool pin1stCell,
	dim3 grid, dim3 block);

//#######################################################################
// 	routine smooth_GSRB (system A*phi = r)
//	perform one iteration Symetric Gauss-Seidel
//  Red-Black ordering (two half step)
//  phiC <- (rC - sum(AF*phiF) )/AC , F~{E, W, N, S, T, B}
//	The matrix A must be SPD, else, change the stencil
//  for other problems (eg. variable permeability) the stencil must be modified
//#######################################################################
__global__ void GSRB_int(double *h_in, const double *rhs, const double *K, double dxdx, int Nx, int Ny, int Nz, bool isred){
	int ix = threadIdx.x + blockIdx.x*blockDim.x;
	int iy = threadIdx.y + blockIdx.y*blockDim.y;
	if (ix >= Nx-2 || iy >= Ny-2) return;
	int stride = Nx*Ny;
	double result;
	int out_idx = (ix+1) + (iy+1)*Nx;
	double aC;
	double KC;
	double KN;
	for(int iz=1; iz<Nz-1; ++iz){
		out_idx += stride;
		KC = K[out_idx];
		if(isred){
			if ((ix+1+iy+1+iz)%2 == 0){
					result=0.0; KN = 0.0; aC = 0.0;
					KN = 2.0 / (1.0/KC  +  1.0/K[out_idx+1]);
					result += h_in[out_idx+1 ]*KN;
					aC += KN;

					KN = 2.0 / (1.0/KC  +  1.0/K[out_idx+Nx]);
					result += h_in[out_idx+Nx]*KN;
					aC += KN;

					KN = 2.0 / (1.0/KC  +  1.0/K[out_idx-1]);
					result += h_in[out_idx-1 ]*KN;
					aC += KN;

					KN = 2.0 / (1.0/KC  +  1.0/K[out_idx-Nx]);
					result += h_in[out_idx-Nx]*KN;
					aC += KN;

					KN = 2.0 / (1.0/KC  +  1.0/K[out_idx+stride]);
					result += h_in[out_idx+stride]*KN;
					aC += KN;

					KN = 2.0 / (1.0/KC  +  1.0/K[out_idx-stride]);
					result += h_in[out_idx-stride]*KN;
					aC += KN;
					h_in[out_idx] = -(rhs[out_idx] - result/dxdx)/(aC/dxdx);
				}
		}
		else {
			if((ix+1+iy+1+iz)%2 != 0){
					result=0.0; KN = 0.0; aC = 0.0;
					KN = 2.0 / (1.0/KC  +  1.0/K[out_idx+1]);
					result += h_in[out_idx+1 ]*KN;
					aC += KN;

					KN = 2.0 / (1.0/KC  +  1.0/K[out_idx+Nx]);
					result += h_in[out_idx+Nx]*KN;
					aC += KN;

					KN = 2.0 / (1.0/KC  +  1.0/K[out_idx-1]);
					result += h_in[out_idx-1 ]*KN;
					aC += KN;

					KN = 2.0 / (1.0/KC  +  1.0/K[out_idx-Nx]);
					result += h_in[out_idx-Nx]*KN;
					aC += KN;

					KN = 2.0 / (1.0/KC  +  1.0/K[out_idx+stride]);
					result += h_in[out_idx+stride]*KN;
					aC += KN;

					KN = 2.0 / (1.0/KC  +  1.0/K[out_idx-stride]);
					result += h_in[out_idx-stride]*KN;
					aC += KN;
					h_in[out_idx] = -(rhs[out_idx] - result/dxdx)/(aC/dxdx);
			}
		}
	}
}

__global__ void GSRB_side_bottom(double *h_in, const double *rhs, const double *K, double dxdx, int Nx, int Ny, int Nz, bool isred, int BCtype){
	int ix = threadIdx.x + blockIdx.x*blockDim.x;
	int iy = threadIdx.y + blockIdx.y*blockDim.y;
	if (ix >= Nx-2 || iy >= Ny-2) return;
	int stride = Nx*Ny;
	double result;
	int in_idx;
	int iz=0;
	in_idx = (ix+1) + (iy+1)*Nx + iz*stride;
	double aC;
	double KC = K[in_idx], KN;
	if(isred) {if ((ix+1+iy+1+iz)%2 == 0){
				result=0.0; KN = 0.0; aC = 0.0;
				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+1]);
				result += h_in[in_idx+1 ]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+Nx]);
				result += h_in[in_idx+Nx]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-1]);
				result += h_in[in_idx-1 ]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-Nx]);
				result += h_in[in_idx-Nx]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+stride]);
				result += h_in[in_idx+stride]*KN;
				aC += KN;

				if(BCtype==periodic) {
					KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+(Nz-1)*stride]);
					result += h_in[in_idx+(Nz-1)*stride]*KN;
					aC += KN;
				}
				if(BCtype==dirichlet) aC+=2.0*KC;
				h_in[in_idx] = -(rhs[in_idx] - result/dxdx)/(aC/dxdx);
			}}
	else {if((ix+1+iy+1+iz)%2 != 0){
				result=0.0; KN = 0.0; aC = 0.0;
				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+1]);
				result += h_in[in_idx+1 ]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+Nx]);
				result += h_in[in_idx+Nx]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-1]);
				result += h_in[in_idx-1 ]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-Nx]);
				result += h_in[in_idx-Nx]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+stride]);
				result += h_in[in_idx+stride]*KN;
				aC += KN;

				if(BCtype==periodic) {
					KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+(Nz-1)*stride]);
					result += h_in[in_idx+(Nz-1)*stride]*KN;
					aC += KN;
				}
				if(BCtype==dirichlet) aC+=2.0*KC;
				h_in[in_idx] = -(rhs[in_idx] - result/dxdx)/(aC/dxdx);
	}}
}

__global__ void GSRB_side_top(double *h_in, const double *rhs, const double *K, double dxdx, int Nx, int Ny, int Nz, bool isred, int BCtype){
	int ix = threadIdx.x + blockIdx.x*blockDim.x;
	int iy = threadIdx.y + blockIdx.y*blockDim.y;
	if (ix >= Nx-2 || iy >= Ny-2) return;
	int stride = Nx*Ny;
	double result;
	int in_idx;
	int iz=Nz-1;
	in_idx = (ix+1) + (iy+1)*Nx + iz*stride;
	double aC;
	double KC = K[in_idx], KN;
	if(isred) {if ((ix+1+iy+1+iz)%2 == 0){
			result=0.0; KN = 0.0; aC = 0.0;
			KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+1]);
			result += h_in[in_idx+1 ]*KN;
			aC += KN;

			KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+Nx]);
			result += h_in[in_idx+Nx]*KN;
			aC += KN;

			KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-1]);
			result += h_in[in_idx-1 ]*KN;
			aC += KN;

			KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-Nx]);
			result += h_in[in_idx-Nx]*KN;
			aC += KN;

			KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-stride]);
			result += h_in[in_idx-stride]*KN;
			aC += KN;

			if(BCtype==periodic) {
				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-(Nz-1)*stride]);
				result += h_in[in_idx-(Nz-1)*stride]*KN;
				aC += KN;
			}
			if(BCtype==dirichlet) aC+=2.0*KC;
			h_in[in_idx] = -(rhs[in_idx] - result/dxdx)/(aC/dxdx);
			}}
	else {if((ix+1+iy+1+iz)%2 != 0){
			result=0.0; KN = 0.0; aC = 0.0;
			KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+1]);
			result += h_in[in_idx+1 ]*KN;
			aC += KN;

			KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+Nx]);
			result += h_in[in_idx+Nx]*KN;
			aC += KN;

			KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-1]);
			result += h_in[in_idx-1 ]*KN;
			aC += KN;

			KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-Nx]);
			result += h_in[in_idx-Nx]*KN;
			aC += KN;

			KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-stride]);
			result += h_in[in_idx-stride]*KN;
			aC += KN;

			if(BCtype==periodic) {
				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-(Nz-1)*stride]);
				result += h_in[in_idx-(Nz-1)*stride]*KN;
				aC += KN;
			}
			if(BCtype==dirichlet) aC+=2.0*KC;
			h_in[in_idx] = -(rhs[in_idx] - result/dxdx)/(aC/dxdx);
	}}
}

__global__ void GSRB_side_south(double *h_in, const double *rhs, const double *K, double dxdx, int Nx, int Ny, int Nz, bool isred, int BCtype){
	int ix = threadIdx.x + blockIdx.x*blockDim.x;
	int iz = threadIdx.y + blockIdx.y*blockDim.y;
	if (ix >= Nx-2 || iz >= Nz-2) return;
	int stride = Nx*Ny;
	double result;
	int in_idx;
	int iy = 0;
	in_idx = (ix + 1) + iy*Nx + (iz + 1)*stride;
	double aC;
	double KC = K[in_idx], KN;
	if(isred) {if ((ix+1+iy+iz+1)%2 == 0){
				result=0.0; KN = 0.0; aC = 0.0;
				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+1]);
				result += h_in[in_idx+1 ]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+stride]);
				result += h_in[in_idx+stride]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-1]);
				result += h_in[in_idx-1 ]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-stride]);
				result += h_in[in_idx-stride]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+Nx]);
				result += h_in[in_idx+Nx]*KN;
				aC += KN;

				if(BCtype==periodic) {
					KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+(Ny-1)*Nx]);
					result += h_in[in_idx+(Ny-1)*Nx]*KN;
					aC += KN;
				}
				if(BCtype==dirichlet) aC+=2.0*KC;
				h_in[in_idx] = -(rhs[in_idx] - result/dxdx)/(aC/dxdx);
			}}
	else {if((ix+1+iy+iz+1)%2 != 0){
				result=0.0; KN = 0.0; aC = 0.0;
				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+1]);
				result += h_in[in_idx+1 ]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+stride]);
				result += h_in[in_idx+stride]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-1]);
				result += h_in[in_idx-1 ]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-stride]);
				result += h_in[in_idx-stride]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+Nx]);
				result += h_in[in_idx+Nx]*KN;
				aC += KN;

				if(BCtype==periodic) {
					KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+(Ny-1)*Nx]);
					result += h_in[in_idx+(Ny-1)*Nx]*KN;
					aC += KN;
				}
				if(BCtype==dirichlet) aC+=2.0*KC;
				h_in[in_idx] = -(rhs[in_idx] - result/dxdx)/(aC/dxdx);
	}}
}

__global__ void GSRB_side_north(double *h_in, const double *rhs, const double *K, double dxdx, int Nx, int Ny, int Nz, bool isred, int BCtype){
	int ix = threadIdx.x + blockIdx.x*blockDim.x;
	int iz = threadIdx.y + blockIdx.y*blockDim.y;
	if (ix >= Nx-2 || iz >= Nz-2) return;
	int stride = Nx*Ny;
	double result;
	int in_idx;
	int iy = Ny-1;
	in_idx = (ix + 1) + iy*Nx + (iz + 1)*stride;
	double aC;
	double KC = K[in_idx],KN;
	if(isred) {if ((ix+1+iy+iz+1)%2 == 0){
				result=0.0; KN = 0.0; aC = 0.0;
				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+1]);
				result += h_in[in_idx+1 ]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+stride]);
				result += h_in[in_idx+stride]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-1]);
				result += h_in[in_idx-1 ]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-stride]);
				result += h_in[in_idx-stride]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-Nx]);
				result += h_in[in_idx-Nx]*KN;
				aC += KN;

				if(BCtype==periodic) {
					KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-(Ny-1)*Nx]);
					result += h_in[in_idx-(Ny-1)*Nx]*KN;
					aC += KN;
				}
				if(BCtype==dirichlet) aC+=2.0*KC;
				h_in[in_idx] = -(rhs[in_idx] - result/dxdx)/(aC/dxdx);
			}}
	else {if((ix+1+iy+iz+1)%2 != 0){
				result=0.0; KN = 0.0; aC = 0.0;
				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+1]);
				result += h_in[in_idx+1 ]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+stride]);
				result += h_in[in_idx+stride]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-1]);
				result += h_in[in_idx-1 ]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-stride]);
				result += h_in[in_idx-stride]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-Nx]);
				result += h_in[in_idx-Nx]*KN;
				aC += KN;

				if(BCtype==periodic) {
					KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-(Ny-1)*Nx]);
					result += h_in[in_idx-(Ny-1)*Nx]*KN;
					aC += KN;
				}
				if(BCtype==dirichlet) aC+=2.0*KC;
				h_in[in_idx] = -(rhs[in_idx] - result/dxdx)/(aC/dxdx);
	}}
}

__global__ void GSRB_side_west(double *h_in, const double *rhs, const double *K, double dxdx, int Nx, int Ny, int Nz, bool isred, int BCtype){
	int iy = threadIdx.x + blockIdx.x*blockDim.x;
	int iz  = threadIdx.y + blockIdx.y*blockDim.y;
	if (iy >= Ny-2 || iz >= Nz-2) return;
	int stride = Nx*Ny;
	double result;
	int in_idx;
	int ix = 0;
	in_idx = ix + (iy + 1)*Nx + (iz + 1)*stride;
	double aC;
	double KC = K[in_idx],KN;
	if(isred) {if ((ix+iy+1+iz+1)%2 == 0){
				result=0.0; KN = 0.0; aC = 0.0;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+Nx]);
				result += h_in[in_idx+Nx]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+stride]);
				result += h_in[in_idx+stride]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-Nx]);
				result += h_in[in_idx-Nx ]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-stride]);
				result += h_in[in_idx-stride]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+1]);
				result += h_in[in_idx+1 ]*KN;
				aC += KN;

				if(BCtype==periodic) {
					KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+(Nx-1)]);
					result += h_in[in_idx+(Nx-1)]*KN;
					aC += KN;
				}
				if(BCtype==dirichlet) aC+=2.0*KC;
				h_in[in_idx] = -(rhs[in_idx] - result/dxdx)/(aC/dxdx);
			}}
	else {if((ix+iy+1+iz+1)%2 != 0){
				result=0.0; KN = 0.0; aC = 0.0;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+Nx]);
				result += h_in[in_idx+Nx]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+stride]);
				result += h_in[in_idx+stride]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-Nx]);
				result += h_in[in_idx-Nx ]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-stride]);
				result += h_in[in_idx-stride]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+1]);
				result += h_in[in_idx+1 ]*KN;
				aC += KN;

				if(BCtype==periodic) {
					KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+(Nx-1)]);
					result += h_in[in_idx+(Nx-1)]*KN;
					aC += KN;
				}
				if(BCtype==dirichlet) aC+=2.0*KC;
				h_in[in_idx] = -(rhs[in_idx] - result/dxdx)/(aC/dxdx);
	}}
}

__global__ void GSRB_side_east(double *h_in, const double *rhs, const double *K, double dxdx, int Nx, int Ny, int Nz, bool isred, int BCtype){
	int iy = threadIdx.x + blockIdx.x*blockDim.x;
	int iz  = threadIdx.y + blockIdx.y*blockDim.y;
	if (iy >= Ny-2 || iz >= Nz-2) return;
	int stride = Nx*Ny;
	double result;
	int in_idx;
	int ix = Nx-1;
	in_idx = ix + (iy + 1)*Nx + (iz + 1)*stride;
	double aC;
	double KC = K[in_idx],KN;
	if(isred) {if ((ix+iy+1+iz+1)%2 == 0){
				result=0.0; KN = 0.0; aC = 0.0;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+Nx]);
				result += h_in[in_idx+Nx]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+stride]);
				result += h_in[in_idx+stride]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-Nx]);
				result += h_in[in_idx-Nx ]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-stride]);
				result += h_in[in_idx-stride]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-1]);
				result += h_in[in_idx-1 ]*KN;
				aC += KN;

				if(BCtype==periodic) {
					KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-(Nx-1)]);
					result += h_in[in_idx-(Nx-1)]*KN;
					aC += KN;
				}
				if(BCtype==dirichlet) aC+=2.0*KC;
				h_in[in_idx] = -(rhs[in_idx] - result/dxdx)/(aC/dxdx);
			}}
	else {if((ix+iy+1+iz+1)%2 != 0){
				result=0.0; KN = 0.0; aC = 0.0;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+Nx]);
				result += h_in[in_idx+Nx]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+stride]);
				result += h_in[in_idx+stride]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-Nx]);
				result += h_in[in_idx-Nx ]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-stride]);
				result += h_in[in_idx-stride]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-1]);
				result += h_in[in_idx-1 ]*KN;
				aC += KN;

				if(BCtype==periodic) {
					KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-(Nx-1)]);
					result += h_in[in_idx-(Nx-1)]*KN;
					aC += KN;
				}
				if(BCtype==dirichlet) aC+=2.0*KC;
				h_in[in_idx] = -(rhs[in_idx] - result/dxdx)/(aC/dxdx);
	}}
}

__global__ void GSRB_edge_X_South_Bottom(double *h_in, const double *rhs, const double *K, double dxdx, int Nx, int Ny, int Nz, bool isred, int BCsouth, int BCbottom){
	int ix = threadIdx.x + blockIdx.x*blockDim.x;
	if (ix >= Nx-2) return;
	int stride = Nx*Ny;
	int iy = 0;
	int iz = 0;
	int in_idx;
	in_idx = (ix+1) + iy*Nx + iz*stride;
	double result;
	double aC;
	double KC = K[in_idx],KN;
	if(isred) {if ((ix+1+iy+iz)%2 == 0){
				result=0.0; KN = 0.0; aC = 0.0;
				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+1]);
				result += h_in[in_idx+1 ]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-1]);
				result += h_in[in_idx-1 ]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+Nx]);
				result += h_in[in_idx+Nx]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+stride]);
				result += h_in[in_idx+stride]*KN;
				aC += KN;

				if(BCsouth==periodic) {
					KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+(Ny-1)*Nx]);
					result += h_in[in_idx+(Ny-1)*Nx]*KN;
					aC += KN;
				}
				if(BCbottom==periodic) {
					KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+(Nz-1)*stride]);
					result += h_in[in_idx+(Nz-1)*stride]*KN;
					aC += KN;
				}
				if(BCsouth==dirichlet) aC+=2.0*KC;
				if(BCbottom==dirichlet) aC+=2.0*KC;
				h_in[in_idx] = -(rhs[in_idx] - result/dxdx)/(aC/dxdx);
			}}
	else {if((ix+1+iy+iz)%2 != 0){
				result=0.0; KN = 0.0; aC = 0.0;
				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+1]);
				result += h_in[in_idx+1 ]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-1]);
				result += h_in[in_idx-1 ]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+Nx]);
				result += h_in[in_idx+Nx]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+stride]);
				result += h_in[in_idx+stride]*KN;
				aC += KN;

				if(BCsouth==periodic) {
					KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+(Ny-1)*Nx]);
					result += h_in[in_idx+(Ny-1)*Nx]*KN;
					aC += KN;
				}
				if(BCbottom==periodic) {
					KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+(Nz-1)*stride]);
					result += h_in[in_idx+(Nz-1)*stride]*KN;
					aC += KN;
				}
				if(BCsouth==dirichlet) aC+=2.0*KC;
				if(BCbottom==dirichlet) aC+=2.0*KC;
				h_in[in_idx] = -(rhs[in_idx] - result/dxdx)/(aC/dxdx);
			}}
}
__global__ void GSRB_edge_X_South_Top(double *h_in, const double *rhs, const double *K, double dxdx, int Nx, int Ny, int Nz, bool isred, int BCsouth, int BCtop){
	int ix = threadIdx.x + blockIdx.x*blockDim.x;
	if (ix >= Nx-2) return;
	int stride = Nx*Ny;
	int iy = 0;
	int iz = Nz-1;
	int in_idx;
	in_idx = (ix+1) + iy*Nx + iz*stride;
	double result;
	double aC;
	double KC = K[in_idx],KN;
	if(isred) {if ((ix+1+iy+iz)%2 == 0){
				result=0.0; KN = 0.0; aC = 0.0;
				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+1]);
				result += h_in[in_idx+1 ]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-1]);
				result += h_in[in_idx-1 ]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+Nx]);
				result += h_in[in_idx+Nx]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-stride]);
				result += h_in[in_idx-stride]*KN;
				aC += KN;

				if(BCsouth==periodic) {
					KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+(Ny-1)*Nx]);
					result += h_in[in_idx+(Ny-1)*Nx]*KN;
					aC += KN;
				}
				if(BCtop==periodic) {
					KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-(Nz-1)*stride]);
					result += h_in[in_idx-(Nz-1)*stride]*KN;
					aC += KN;
				}
				if(BCsouth==dirichlet) aC+=2.0*KC;
				if(BCtop==dirichlet) aC+=2.0*KC;
				h_in[in_idx] = -(rhs[in_idx] - result/dxdx)/(aC/dxdx);
			}}
	else {if((ix+1+iy+iz)%2 != 0){
				result=0.0; KN = 0.0; aC = 0.0;
				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+1]);
				result += h_in[in_idx+1 ]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-1]);
				result += h_in[in_idx-1 ]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+Nx]);
				result += h_in[in_idx+Nx]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-stride]);
				result += h_in[in_idx-stride]*KN;
				aC += KN;

				if(BCsouth==periodic) {
					KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+(Ny-1)*Nx]);
					result += h_in[in_idx+(Ny-1)*Nx]*KN;
					aC += KN;
				}
				if(BCtop==periodic) {
					KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-(Nz-1)*stride]);
					result += h_in[in_idx-(Nz-1)*stride]*KN;
					aC += KN;
				}
				if(BCsouth==dirichlet) aC+=2.0*KC;
				if(BCtop==dirichlet) aC+=2.0*KC;
				h_in[in_idx] = -(rhs[in_idx] - result/dxdx)/(aC/dxdx);
			}}
}

__global__ void GSRB_edge_X_North_Bottom(double *h_in, const double *rhs, const double *K, double dxdx, int Nx, int Ny, int Nz, bool isred, int BCnorth, int BCbottom){
	int ix = threadIdx.x + blockIdx.x*blockDim.x;
	if (ix >= Nx-2) return;
	int stride = Nx*Ny;
	int iy = Ny-1;
	int iz = 0;
	int in_idx;
	in_idx = (ix+1) + iy*Nx + iz*stride;
	double result;
	double aC;
	double KC = K[in_idx],KN;
	if(isred) {if ((ix+1+iy+iz)%2 == 0){
				result=0.0; KN = 0.0; aC = 0.0;
				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+1]);
				result += h_in[in_idx+1 ]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-1]);
				result += h_in[in_idx-1 ]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-Nx]);
				result += h_in[in_idx-Nx]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+stride]);
				result += h_in[in_idx+stride]*KN;
				aC += KN;

				if(BCnorth==periodic) {
					KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-(Ny-1)*Nx]);
					result += h_in[in_idx-(Ny-1)*Nx]*KN;
					aC += KN;
				}
				if(BCbottom==periodic) {
					KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+(Nz-1)*stride]);
					result += h_in[in_idx+(Nz-1)*stride]*KN;
					aC += KN;
				}
				if(BCnorth==dirichlet) aC+=2.0*KC;
				if(BCbottom==dirichlet) aC+=2.0*KC;
				h_in[in_idx] = -(rhs[in_idx] - result/dxdx)/(aC/dxdx);
			}}
	else {if((ix+1+iy+iz)%2 != 0){
				result=0.0; KN = 0.0; aC = 0.0;
				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+1]);
				result += h_in[in_idx+1 ]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-1]);
				result += h_in[in_idx-1 ]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-Nx]);
				result += h_in[in_idx-Nx]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+stride]);
				result += h_in[in_idx+stride]*KN;
				aC += KN;

				if(BCnorth==periodic) {
					KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-(Ny-1)*Nx]);
					result += h_in[in_idx-(Ny-1)*Nx]*KN;
					aC += KN;
				}
				if(BCbottom==periodic) {
					KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+(Nz-1)*stride]);
					result += h_in[in_idx+(Nz-1)*stride]*KN;
					aC += KN;
				}
				if(BCnorth==dirichlet) aC+=2.0*KC;
				if(BCbottom==dirichlet) aC+=2.0*KC;
				h_in[in_idx] = -(rhs[in_idx] - result/dxdx)/(aC/dxdx);
			}}
}
__global__ void GSRB_edge_X_North_Top(double *h_in, const double *rhs, const double *K, double dxdx, int Nx, int Ny, int Nz, bool isred, int BCnorth, int BCtop){
	int ix = threadIdx.x + blockIdx.x*blockDim.x;
	if (ix >= Nx-2) return;
	int stride = Nx*Ny;
	int iy = Ny-1;
	int iz = Nz-1;
	int in_idx;
	in_idx = (ix+1) + iy*Nx + iz*stride;
	double result;
	double aC;
	double KC = K[in_idx],KN;
	if(isred) {if ((ix+1+iy+iz)%2 == 0){
				result=0.0; KN = 0.0; aC = 0.0;
				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+1]);
				result += h_in[in_idx+1 ]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-1]);
				result += h_in[in_idx-1 ]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-Nx]);
				result += h_in[in_idx-Nx]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-stride]);
				result += h_in[in_idx-stride]*KN;
				aC += KN;

				if(BCnorth==periodic) {
					KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-(Ny-1)*Nx]);
					result += h_in[in_idx-(Ny-1)*Nx]*KN;
					aC += KN;
				}
				if(BCtop==periodic) {
					KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-(Nz-1)*stride]);
					result += h_in[in_idx-(Nz-1)*stride]*KN;
					aC += KN;
				}
				if(BCnorth==dirichlet) aC+=2.0*KC;
				if(BCtop==dirichlet) aC+=2.0*KC;
				h_in[in_idx] = -(rhs[in_idx] - result/dxdx)/(aC/dxdx);
			}}
	else {if((ix+1+iy+iz)%2 != 0){
				result=0.0; KN = 0.0; aC = 0.0;
				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+1]);
				result += h_in[in_idx+1 ]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-1]);
				result += h_in[in_idx-1 ]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-Nx]);
				result += h_in[in_idx-Nx]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-stride]);
				result += h_in[in_idx-stride]*KN;
				aC += KN;

				if(BCnorth==periodic) {
					KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-(Ny-1)*Nx]);
					result += h_in[in_idx-(Ny-1)*Nx]*KN;
					aC += KN;
				}
				if(BCtop==periodic) {
					KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-(Nz-1)*stride]);
					result += h_in[in_idx-(Nz-1)*stride]*KN;
					aC += KN;
				}
				if(BCnorth==dirichlet) aC+=2.0*KC;
				if(BCtop==dirichlet) aC+=2.0*KC;
				h_in[in_idx] = -(rhs[in_idx] - result/dxdx)/(aC/dxdx);
			}}
}

__global__ void GSRB_edge_Z_South_West(double *h_in, const double *rhs, const double *K, double dxdx, int Nx, int Ny, int Nz, bool isred, int BCsouth, int BCwest){
	int iz = threadIdx.x + blockIdx.x*blockDim.x;
	if (iz >= Nz-2) return;
	int stride = Nx*Ny;
	int ix = 0;
	int iy = 0;
	int in_idx;
	in_idx = ix + iy*Nx + (iz+1)*stride;
	double result;
	double aC;
	double KC = K[in_idx],KN;
	if(isred) {if ((ix+iy+iz+1)%2 == 0){
				result=0.0; KN = 0.0; aC = 0.0;
				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+stride]);
				result += h_in[in_idx+stride]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-stride]);
				result += h_in[in_idx-stride ]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+Nx]);
				result += h_in[in_idx+Nx]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+1]);
				result += h_in[in_idx+1 ]*KN;
				aC += KN;

				if(BCsouth==periodic) {
					KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+(Ny-1)*Nx]);
					result += h_in[in_idx+(Ny-1)*Nx]*KN;
					aC += KN;
				}
				if(BCwest==periodic) {
					KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+(Nx-1)]);
					result += h_in[in_idx+(Nx-1)]*KN;
					aC += KN;
				}
				if(BCsouth==dirichlet) aC+=2.0*KC;
				if(BCwest==dirichlet) aC+=2.0*KC;
				h_in[in_idx] = -(rhs[in_idx] - result/dxdx)/(aC/dxdx);
			}}
	else {if((ix+iy+iz+1)%2 != 0){
				result=0.0; KN = 0.0; aC = 0.0;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+stride]);
				result += h_in[in_idx+stride]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-stride]);
				result += h_in[in_idx-stride ]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+Nx]);
				result += h_in[in_idx+Nx]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+1]);
				result += h_in[in_idx+1 ]*KN;
				aC += KN;

				if(BCsouth==periodic) {
					KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+(Ny-1)*Nx]);
					result += h_in[in_idx+(Ny-1)*Nx]*KN;
					aC += KN;
				}
				if(BCwest==periodic) {
					KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+(Nx-1)]);
					result += h_in[in_idx+(Nx-1)]*KN;
					aC += KN;
				}
				if(BCsouth==dirichlet) aC+=2.0*KC;
				if(BCwest==dirichlet) aC+=2.0*KC;
				h_in[in_idx] = -(rhs[in_idx] - result/dxdx)/(aC/dxdx);
			}}
}

__global__ void GSRB_edge_Z_South_East(double *h_in, const double *rhs, const double *K, double dxdx, int Nx, int Ny, int Nz, bool isred, int BCsouth, int BCeast){
	int iz = threadIdx.x + blockIdx.x*blockDim.x;
	if (iz >= Nz-2) return;
	int stride = Nx*Ny;
	int ix = Nx-1;
	int iy = 0;
	int in_idx;
	in_idx = ix + iy*Nx + (iz+1)*stride;
	double result;
	double aC;
	double KC = K[in_idx],KN;
	if(isred) {if ((ix+iy+iz+1)%2 == 0){
				result=0.0; KN = 0.0; aC = 0.0;
				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+stride]);
				result += h_in[in_idx+stride]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-stride]);
				result += h_in[in_idx-stride ]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+Nx]);
				result += h_in[in_idx+Nx]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-1]);
				result += h_in[in_idx-1 ]*KN;
				aC += KN;

				if(BCsouth==periodic) {
					KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+(Ny-1)*Nx]);
					result += h_in[in_idx+(Ny-1)*Nx]*KN;
					aC += KN;
				}
				if(BCeast==periodic) {
					KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-(Nx-1)]);
					result += h_in[in_idx-(Nx-1)]*KN;
					aC += KN;
				}
				if(BCsouth==dirichlet) aC+=2.0*KC;
				if(BCeast==dirichlet) aC+=2.0*KC;
				h_in[in_idx] = -(rhs[in_idx] - result/dxdx)/(aC/dxdx);
			}}
	else {if((ix+iy+iz+1)%2 != 0){
				result=0.0; KN = 0.0; aC = 0.0;
				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+stride]);
				result += h_in[in_idx+stride]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-stride]);
				result += h_in[in_idx-stride ]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+Nx]);
				result += h_in[in_idx+Nx]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-1]);
				result += h_in[in_idx-1 ]*KN;
				aC += KN;

				if(BCsouth==periodic) {
					KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+(Ny-1)*Nx]);
					result += h_in[in_idx+(Ny-1)*Nx]*KN;
					aC += KN;
				}
				if(BCeast==periodic) {
					KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-(Nx-1)]);
					result += h_in[in_idx-(Nx-1)]*KN;
					aC += KN;
				}
				if(BCsouth==dirichlet) aC+=2.0*KC;
				if(BCeast==dirichlet) aC+=2.0*KC;
				h_in[in_idx] = -(rhs[in_idx] - result/dxdx)/(aC/dxdx);
			}}
}

__global__ void GSRB_edge_Z_North_West(double *h_in, const double *rhs, const double *K, double dxdx, int Nx, int Ny, int Nz, bool isred, int BCnorth, int BCwest){
	int iz = threadIdx.x + blockIdx.x*blockDim.x;
	if (iz >= Nz-2) return;
	int stride = Nx*Ny;
	int ix = 0;
	int iy = Ny-1;
	int in_idx;
	in_idx = ix + iy*Nx + (iz+1)*stride;
	double result;
	double aC;
	double KC = K[in_idx],KN;
	if(isred) {if ((ix+iy+iz+1)%2 == 0){
				result=0.0; KN = 0.0; aC = 0.0;
				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+stride]);
				result += h_in[in_idx+stride]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-stride]);
				result += h_in[in_idx-stride ]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-Nx]);
				result += h_in[in_idx-Nx]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+1]);
				result += h_in[in_idx+1 ]*KN;
				aC += KN;

				if(BCnorth==periodic) {
					KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-(Ny-1)*Nx]);
					result += h_in[in_idx-(Ny-1)*Nx]*KN;
					aC += KN;
				}
				if(BCwest==periodic) {
					KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+(Nx-1)]);
					result += h_in[in_idx+(Nx-1)]*KN;
					aC += KN;
				}
				if(BCnorth==dirichlet) aC+=2.0*KC;
				if(BCwest==dirichlet) aC+=2.0*KC;
				h_in[in_idx] = -(rhs[in_idx] - result/dxdx)/(aC/dxdx);
			}}
	else {if((ix+iy+iz+1)%2 != 0){
				result=0.0; KN = 0.0; aC = 0.0;
				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+stride]);
				result += h_in[in_idx+stride]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-stride]);
				result += h_in[in_idx-stride ]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-Nx]);
				result += h_in[in_idx-Nx]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+1]);
				result += h_in[in_idx+1 ]*KN;
				aC += KN;

				if(BCnorth==periodic) {
					KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-(Ny-1)*Nx]);
					result += h_in[in_idx-(Ny-1)*Nx]*KN;
					aC += KN;
				}
				if(BCwest==periodic) {
					KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+(Nx-1)]);
					result += h_in[in_idx+(Nx-1)]*KN;
					aC += KN;
				}
				if(BCnorth==dirichlet) aC+=2.0*KC;
				if(BCwest==dirichlet) aC+=2.0*KC;
				h_in[in_idx] = -(rhs[in_idx] - result/dxdx)/(aC/dxdx);
			}}
}

__global__ void GSRB_edge_Z_North_East(double *h_in, const double *rhs, const double *K, double dxdx, int Nx, int Ny, int Nz, bool isred, int BCnorth, int BCeast){
	int iz = threadIdx.x + blockIdx.x*blockDim.x;
	if (iz >= Nz-2) return;
	int stride = Nx*Ny;
	int ix = Nx-1;
	int iy = Ny-1;
	int in_idx;
	in_idx = ix + iy*Nx + (iz+1)*stride;
	double result;
	double aC;
	double KC = K[in_idx],KN;
	if(isred) {if ((ix+iy+iz+1)%2 == 0){
				result=0.0; KN = 0.0; aC = 0.0;
				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+stride]);
				result += h_in[in_idx+stride]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-stride]);
				result += h_in[in_idx-stride ]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-Nx]);
				result += h_in[in_idx-Nx]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-1]);
				result += h_in[in_idx-1 ]*KN;
				aC += KN;

				if(BCnorth==periodic) {
					KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-(Ny-1)*Nx]);
					result += h_in[in_idx-(Ny-1)*Nx]*KN;
					aC += KN;
				}
				if(BCeast==periodic) {
					KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-(Nx-1)]);
					result += h_in[in_idx-(Nx-1)]*KN;
					aC += KN;
				}
				if(BCnorth==dirichlet) aC+=2.0*KC;
				if(BCeast==dirichlet) aC+=2.0*KC;
				h_in[in_idx] = -(rhs[in_idx] - result/dxdx)/(aC/dxdx);
			}}
	else {if((ix+iy+iz+1)%2 != 0){
				result=0.0; KN = 0.0; aC = 0.0;
				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+stride]);
				result += h_in[in_idx+stride]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-stride]);
				result += h_in[in_idx-stride ]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-Nx]);
				result += h_in[in_idx-Nx]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-1]);
				result += h_in[in_idx-1 ]*KN;
				aC += KN;

				if(BCnorth==periodic) {
					KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-(Ny-1)*Nx]);
					result += h_in[in_idx-(Ny-1)*Nx]*KN;
					aC += KN;
				}
				if(BCeast==periodic) {
					KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-(Nx-1)]);
					result += h_in[in_idx-(Nx-1)]*KN;
					aC += KN;
				}
				if(BCnorth==dirichlet) aC+=2.0*KC;
				if(BCeast==dirichlet) aC+=2.0*KC;
				h_in[in_idx] = -(rhs[in_idx] - result/dxdx)/(aC/dxdx);
			}}
}

__global__ void GSRB_edge_Y_West_Bottom(double *h_in, const double *rhs, const double *K, double dxdx, int Nx, int Ny, int Nz, bool isred, int BCbottom, int BCwest){
	int iy = threadIdx.x + blockIdx.x*blockDim.x;
	if (iy >= Ny-2) return;
	int stride = Nx*Ny;
	int ix = 0;
	int iz = 0;
	int in_idx;
	in_idx = ix + (iy+1)*Nx + iz*stride;
	double result;
	double aC;
	double KC = K[in_idx],KN;
	if(isred) {if ((ix+iy+1+iz)%2 == 0){
				result=0.0; KN = 0.0; aC = 0.0;
				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+Nx]);
				result += h_in[in_idx+Nx]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-Nx]);
				result += h_in[in_idx-Nx ]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+stride]);
				result += h_in[in_idx+stride]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+1]);
				result += h_in[in_idx+1 ]*KN;
				aC += KN;

				if(BCbottom==periodic) {
					KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+(Nz-1)*stride]);
					result += h_in[in_idx+(Nz-1)*stride]*KN;
					aC += KN;
				}
				if(BCwest==periodic) {
					KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+(Nx-1)]);
					result += h_in[in_idx+(Nx-1)]*KN;
					aC += KN;
				}
				if(BCbottom==dirichlet) aC+=2.0*KC;
				if(BCwest==dirichlet) aC+=2.0*KC;
				h_in[in_idx] = -(rhs[in_idx] - result/dxdx)/(aC/dxdx);
			}}
	else {if((ix+iy+1+iz)%2 != 0){
				result=0.0; KN = 0.0; aC = 0.0;
				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+Nx]);
				result += h_in[in_idx+Nx]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-Nx]);
				result += h_in[in_idx-Nx ]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+stride]);
				result += h_in[in_idx+stride]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+1]);
				result += h_in[in_idx+1 ]*KN;
				aC += KN;

				if(BCbottom==periodic) {
					KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+(Nz-1)*stride]);
					result += h_in[in_idx+(Nz-1)*stride]*KN;
					aC += KN;
				}
				if(BCwest==periodic) {
					KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+(Nx-1)]);
					result += h_in[in_idx+(Nx-1)]*KN;
					aC += KN;
				}
				if(BCbottom==dirichlet) aC+=2.0*KC;
				if(BCwest==dirichlet) aC+=2.0*KC;
				h_in[in_idx] = -(rhs[in_idx] - result/dxdx)/(aC/dxdx);
			}}
}
__global__ void GSRB_edge_Y_West_Top(double *h_in, const double *rhs, const double *K, double dxdx, int Nx, int Ny, int Nz, bool isred, int BCtop, int BCwest){
	int iy = threadIdx.x + blockIdx.x*blockDim.x;
	if (iy >= Ny-2) return;
	int stride = Nx*Ny;
	int ix = 0;
	int iz = Nz-1;
	int in_idx;
	in_idx = ix + (iy+1)*Nx + iz*stride;
	double result;
	double aC;
	double KC = K[in_idx],KN;
	if(isred) {if ((ix+iy+1+iz)%2 == 0){
				result=0.0; KN = 0.0; aC = 0.0;
				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+Nx]);
				result += h_in[in_idx+Nx]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-Nx]);
				result += h_in[in_idx-Nx ]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-stride]);
				result += h_in[in_idx-stride]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+1]);
				result += h_in[in_idx+1 ]*KN;
				aC += KN;

				if(BCtop==periodic) {
					KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-(Nz-1)*stride]);
					result += h_in[in_idx-(Nz-1)*stride]*KN;
					aC += KN;
				}
				if(BCwest==periodic) {
					KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+(Nx-1)]);
					result += h_in[in_idx+(Nx-1)]*KN;
					aC += KN;
				}
				if(BCtop==dirichlet) aC+=2.0*KC;
				if(BCwest==dirichlet) aC+=2.0*KC;
				h_in[in_idx] = -(rhs[in_idx] - result/dxdx)/(aC/dxdx);
			}}
	else {if((ix+iy+1+iz)%2 != 0){
				result=0.0; KN = 0.0; aC = 0.0;
				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+Nx]);
				result += h_in[in_idx+Nx]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-Nx]);
				result += h_in[in_idx-Nx ]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-stride]);
				result += h_in[in_idx-stride]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+1]);
				result += h_in[in_idx+1 ]*KN;
				aC += KN;

				if(BCtop==periodic) {
					KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-(Nz-1)*stride]);
					result += h_in[in_idx-(Nz-1)*stride]*KN;
					aC += KN;
				}
				if(BCwest==periodic) {
					KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+(Nx-1)]);
					result += h_in[in_idx+(Nx-1)]*KN;
					aC += KN;
				}
				if(BCtop==dirichlet) aC+=2.0*KC;
				if(BCwest==dirichlet) aC+=2.0*KC;
				h_in[in_idx] = -(rhs[in_idx] - result/dxdx)/(aC/dxdx);
			}}
}

__global__ void GSRB_edge_Y_East_Bottom(double *h_in, const double *rhs, const double *K, double dxdx, int Nx, int Ny, int Nz, bool isred, int BCbottom, int BCeast){
	int iy = threadIdx.x + blockIdx.x*blockDim.x;
	if (iy >= Ny-2) return;
	int stride = Nx*Ny;
	int ix = Nx-1;
	int iz = 0;
	int in_idx;
	in_idx = ix + (iy+1)*Nx + iz*stride;
	double result;
	double aC;
	double KC = K[in_idx],KN;
	if(isred) {if ((ix+iy+1+iz)%2 == 0){
				result=0.0; KN = 0.0; aC = 0.0;
				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+Nx]);
				result += h_in[in_idx+Nx]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-Nx]);
				result += h_in[in_idx-Nx ]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+stride]);
				result += h_in[in_idx+stride]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-1]);
				result += h_in[in_idx-1 ]*KN;
				aC += KN;

				if(BCbottom==periodic) {
					KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+(Nz-1)*stride]);
					result += h_in[in_idx+(Nz-1)*stride]*KN;
					aC += KN;
				}
				if(BCeast==periodic) {
					KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-(Nx-1)]);
					result += h_in[in_idx-(Nx-1)]*KN;
					aC += KN;
				}
				if(BCbottom==dirichlet) aC+=2.0*KC;
				if(BCeast==dirichlet) aC+=2.0*KC;
				h_in[in_idx] = -(rhs[in_idx] - result/dxdx)/(aC/dxdx);
			}}
	else {if((ix+iy+1+iz)%2 != 0){
				result=0.0; KN = 0.0; aC = 0.0;
				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+Nx]);
				result += h_in[in_idx+Nx]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-Nx]);
				result += h_in[in_idx-Nx ]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+stride]);
				result += h_in[in_idx+stride]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-1]);
				result += h_in[in_idx-1 ]*KN;
				aC += KN;

				if(BCbottom==periodic) {
					KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+(Nz-1)*stride]);
					result += h_in[in_idx+(Nz-1)*stride]*KN;
					aC += KN;
				}
				if(BCeast==periodic) {
					KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-(Nx-1)]);
					result += h_in[in_idx-(Nx-1)]*KN;
					aC += KN;
				}
				if(BCbottom==dirichlet) aC+=2.0*KC;
				if(BCeast==dirichlet) aC+=2.0*KC;
				h_in[in_idx] = -(rhs[in_idx] - result/dxdx)/(aC/dxdx);
			}}
}
__global__ void GSRB_edge_Y_East_Top(double *h_in, const double *rhs, const double *K, double dxdx, int Nx, int Ny, int Nz, bool isred, int BCtop, int BCeast){
	int iy = threadIdx.x + blockIdx.x*blockDim.x;
	if (iy >= Ny-2) return;
	int stride = Nx*Ny;
	int ix = Nx-1;
	int iz = Nz-1;
	int in_idx;
	in_idx = ix + (iy+1)*Nx + iz*stride;
	double result;
	double aC;
	double KC = K[in_idx],KN;
	if(isred) {if ((ix+iy+1+iz)%2 == 0){
				result=0.0; KN = 0.0; aC = 0.0;
				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+Nx]);
				result += h_in[in_idx+Nx]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-Nx]);
				result += h_in[in_idx-Nx ]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-stride]);
				result += h_in[in_idx-stride]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-1]);
				result += h_in[in_idx-1 ]*KN;
				aC += KN;

				if(BCtop==periodic) {
					KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-(Nz-1)*stride]);
					result += h_in[in_idx-(Nz-1)*stride]*KN;
					aC += KN;
				}
				if(BCeast==periodic) {
					KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-(Nx-1)]);
					result += h_in[in_idx-(Nx-1)]*KN;
					aC += KN;
				}
				if(BCtop==dirichlet) aC+=2.0*KC;
				if(BCeast==dirichlet) aC+=2.0*KC;
				h_in[in_idx] = -(rhs[in_idx] - result/dxdx)/(aC/dxdx);
			}}
	else {if((ix+iy+1+iz)%2 != 0){
				result=0.0; KN = 0.0; aC = 0.0;
				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+Nx]);
				result += h_in[in_idx+Nx]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-Nx]);
				result += h_in[in_idx-Nx ]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-stride]);
				result += h_in[in_idx-stride]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-1]);
				result += h_in[in_idx-1 ]*KN;
				aC += KN;

				if(BCtop==periodic) {
					KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-(Nz-1)*stride]);
					result += h_in[in_idx-(Nz-1)*stride]*KN;
					aC += KN;
				}
				if(BCeast==periodic) {
					KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-(Nx-1)]);
					result += h_in[in_idx-(Nx-1)]*KN;
					aC += KN;
				}
				if(BCtop==dirichlet) aC+=2.0*KC;
				if(BCeast==dirichlet) aC+=2.0*KC;
				h_in[in_idx] = -(rhs[in_idx] - result/dxdx)/(aC/dxdx);
			}}
}

__global__ void GSRB_vertex_SWB(double *h_in, const double *rhs, const double *K, double dxdx, int Nx, int Ny, int Nz, bool isred, int BCsouth, int BCwest, int BCbottom, bool pin1stCell){
	int stride = Nx*Ny;
	int ix = 0;
	int iy = 0;
	int iz = 0;
	int in_idx;
	in_idx = ix + iy*Nx + iz*stride;
	double result;
	double aC;
	double KC = K[in_idx],KN;
	if(isred) {if ((ix+iy+iz)%2 == 0){
				result=0.0; KN = 0.0; aC = 0.0;
				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+Nx]);
				result += h_in[in_idx+Nx]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+stride]);
				result += h_in[in_idx+stride]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+1]);
				result += h_in[in_idx+1 ]*KN;
				aC += KN;
				if(BCsouth==periodic) {
					KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+(Ny-1)*Nx]);
					result += h_in[in_idx+(Ny-1)*Nx]*KN;
					aC += KN;
				}
				if(BCbottom==periodic) {
					KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+(Nz-1)*stride]);
					result += h_in[in_idx+(Nz-1)*stride]*KN;
					aC += KN;
				}
				if(BCwest==periodic) {
					KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+(Nx-1)]);
					result += h_in[in_idx+(Nx-1)]*KN;
					aC += KN;
				}
				if(BCsouth==dirichlet) aC+=2.0*KC;
				if(BCbottom==dirichlet) aC+=2.0*KC;
				if(BCwest==dirichlet) aC+=2.0*KC;
				if(pin1stCell) aC*=2.0; //if pin first cell for solvabolity (if all BCtype are homog-Neumann)
				h_in[in_idx] = -(rhs[in_idx] - result/dxdx)/(aC/dxdx);
			}}
	else {if((ix+iy+iz)%2 != 0){
				result=0.0; KN = 0.0; aC = 0.0;
				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+Nx]);
				result += h_in[in_idx+Nx]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+stride]);
				result += h_in[in_idx+stride]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+1]);
				result += h_in[in_idx+1 ]*KN;
				aC += KN;
				if(BCsouth==periodic) {
					KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+(Ny-1)*Nx]);
					result += h_in[in_idx+(Ny-1)*Nx]*KN;
					aC += KN;
				}
				if(BCbottom==periodic) {
					KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+(Nz-1)*stride]);
					result += h_in[in_idx+(Nz-1)*stride]*KN;
					aC += KN;
				}
				if(BCwest==periodic) {
					KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+(Nx-1)]);
					result += h_in[in_idx+(Nx-1)]*KN;
					aC += KN;
				}
				if(BCsouth==dirichlet) aC+=2.0*KC;
				if(BCbottom==dirichlet) aC+=2.0*KC;
				if(BCwest==dirichlet) aC+=2.0*KC;
				if(pin1stCell) aC*=2.0; //if pin first cell for solvabolity (if all BCtype are homog-Neumann)
				h_in[in_idx] = -(rhs[in_idx] - result/dxdx)/(aC/dxdx);
			}}
}
__global__ void GSRB_vertex_SWT(double *h_in, const double *rhs, const double *K, double dxdx, int Nx, int Ny, int Nz, bool isred, int BCsouth, int BCwest, int BCtop){
	int stride = Nx*Ny;
	int ix = 0;
	int iy = 0;
	int iz = Nz-1;
	int in_idx;
	in_idx = ix + iy*Nx + iz*stride;
	double result;
	double aC;
	double KC = K[in_idx],KN;
	if(isred) {if ((ix+iy+iz)%2 == 0){
				result=0.0; KN = 0.0; aC = 0.0;
				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+Nx]);
				result += h_in[in_idx+Nx]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-stride]);
				result += h_in[in_idx-stride]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+1]);
				result += h_in[in_idx+1 ]*KN;
				aC += KN;
				if(BCsouth==periodic) {
					KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+(Ny-1)*Nx]);
					result += h_in[in_idx+(Ny-1)*Nx]*KN;
					aC += KN;
				}
				if(BCtop==periodic) {
					KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-(Nz-1)*stride]);
					result += h_in[in_idx-(Nz-1)*stride]*KN;
					aC += KN;
				}
				if(BCwest==periodic) {
					KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+(Nx-1)]);
					result += h_in[in_idx+(Nx-1)]*KN;
					aC += KN;
				}
				if(BCsouth==dirichlet) aC+=2.0*KC;
				if(BCtop==dirichlet) aC+=2.0*KC;
				if(BCwest==dirichlet) aC+=2.0*KC;
				h_in[in_idx] = -(rhs[in_idx] - result/dxdx)/(aC/dxdx);
			}}
	else {if((ix+iy+iz)%2 != 0){
				result=0.0; KN = 0.0; aC = 0.0;
				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+Nx]);
				result += h_in[in_idx+Nx]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-stride]);
				result += h_in[in_idx-stride]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+1]);
				result += h_in[in_idx+1 ]*KN;
				aC += KN;
				if(BCsouth==periodic) {
					KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+(Ny-1)*Nx]);
					result += h_in[in_idx+(Ny-1)*Nx]*KN;
					aC += KN;
				}
				if(BCtop==periodic) {
					KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-(Nz-1)*stride]);
					result += h_in[in_idx-(Nz-1)*stride]*KN;
					aC += KN;
				}
				if(BCwest==periodic) {
					KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+(Nx-1)]);
					result += h_in[in_idx+(Nx-1)]*KN;
					aC += KN;
				}
				if(BCsouth==dirichlet) aC+=2.0*KC;
				if(BCtop==dirichlet) aC+=2.0*KC;
				if(BCwest==dirichlet) aC+=2.0*KC;
				h_in[in_idx] = -(rhs[in_idx] - result/dxdx)/(aC/dxdx);
			}}
}

__global__ void GSRB_vertex_SEB(double *h_in, const double *rhs, const double *K, double dxdx, int Nx, int Ny, int Nz, bool isred, int BCsouth, int BCeast, int BCbottom){
	int stride = Nx*Ny;
	int ix = Nx-1;
	int iy = 0;
	int iz = 0;
	int in_idx;
	in_idx = ix + iy*Nx + iz*stride;
	double result;
	double aC;
	double KC = K[in_idx],KN;
	if(isred) {if ((ix+iy+iz)%2 == 0){
				result=0.0; KN = 0.0; aC = 0.0;
				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+Nx]);
				result += h_in[in_idx+Nx]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+stride]);
				result += h_in[in_idx+stride]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-1]);
				result += h_in[in_idx-1 ]*KN;
				aC += KN;
				if(BCsouth==periodic) {
					KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+(Ny-1)*Nx]);
					result += h_in[in_idx+(Ny-1)*Nx]*KN;
					aC += KN;
				}
				if(BCbottom==periodic) {
					KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+(Nz-1)*stride]);
					result += h_in[in_idx+(Nz-1)*stride]*KN;
					aC += KN;
				}
				if(BCeast==periodic) {
					KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-(Nx-1)]);
					result += h_in[in_idx-(Nx-1)]*KN;
					aC += KN;
				}
				if(BCsouth==dirichlet) aC+=2.0*KC;
				if(BCbottom==dirichlet) aC+=2.0*KC;
				if(BCeast==dirichlet) aC+=2.0*KC;
				h_in[in_idx] = -(rhs[in_idx] - result/dxdx)/(aC/dxdx);
			}}
	else {if((ix+iy+iz)%2 != 0){
				result=0.0; KN = 0.0; aC = 0.0;
				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+Nx]);
				result += h_in[in_idx+Nx]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+stride]);
				result += h_in[in_idx+stride]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-1]);
				result += h_in[in_idx-1 ]*KN;
				aC += KN;
				if(BCsouth==periodic) {
					KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+(Ny-1)*Nx]);
					result += h_in[in_idx+(Ny-1)*Nx]*KN;
					aC += KN;
				}
				if(BCbottom==periodic) {
					KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+(Nz-1)*stride]);
					result += h_in[in_idx+(Nz-1)*stride]*KN;
					aC += KN;
				}
				if(BCeast==periodic) {
					KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-(Nx-1)]);
					result += h_in[in_idx-(Nx-1)]*KN;
					aC += KN;
				}
				if(BCsouth==dirichlet) aC+=2.0*KC;
				if(BCbottom==dirichlet) aC+=2.0*KC;
				if(BCeast==dirichlet) aC+=2.0*KC;
				h_in[in_idx] = -(rhs[in_idx] - result/dxdx)/(aC/dxdx);
			}}
}
__global__ void GSRB_vertex_SET(double *h_in, const double *rhs, const double *K, double dxdx, int Nx, int Ny, int Nz, bool isred, int BCsouth, int BCeast, int BCtop){
	int stride = Nx*Ny;
	int ix = Nx-1;
	int iy = 0;
	int iz = Nz-1;
	int in_idx;
	in_idx = ix + iy*Nx + iz*stride;
	double result;
	double aC;
	double KC = K[in_idx],KN;
	if(isred) {if ((ix+iy+iz)%2 == 0){
				result=0.0; KN = 0.0; aC = 0.0;
				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+Nx]);
				result += h_in[in_idx+Nx]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-stride]);
				result += h_in[in_idx-stride]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-1]);
				result += h_in[in_idx-1 ]*KN;
				aC += KN;
				if(BCsouth==periodic) {
					KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+(Ny-1)*Nx]);
					result += h_in[in_idx+(Ny-1)*Nx]*KN;
					aC += KN;
				}
				if(BCtop==periodic) {
					KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-(Nz-1)*stride]);
					result += h_in[in_idx-(Nz-1)*stride]*KN;
					aC += KN;
				}
				if(BCeast==periodic) {
					KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-(Nx-1)]);
					result += h_in[in_idx-(Nx-1)]*KN;
					aC += KN;
				}
				if(BCsouth==dirichlet) aC+=2.0*KC;
				if(BCtop==dirichlet) aC+=2.0*KC;
				if(BCeast==dirichlet) aC+=2.0*KC;
				h_in[in_idx] = -(rhs[in_idx] - result/dxdx)/(aC/dxdx);
			}}
	else {if((ix+iy+iz)%2 != 0){
				result=0.0; KN = 0.0; aC = 0.0;
				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+Nx]);
				result += h_in[in_idx+Nx]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-stride]);
				result += h_in[in_idx-stride]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-1]);
				result += h_in[in_idx-1 ]*KN;
				aC += KN;
				if(BCsouth==periodic) {
					KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+(Ny-1)*Nx]);
					result += h_in[in_idx+(Ny-1)*Nx]*KN;
					aC += KN;
				}
				if(BCtop==periodic) {
					KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-(Nz-1)*stride]);
					result += h_in[in_idx-(Nz-1)*stride]*KN;
					aC += KN;
				}
				if(BCeast==periodic) {
					KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-(Nx-1)]);
					result += h_in[in_idx-(Nx-1)]*KN;
					aC += KN;
				}
				if(BCsouth==dirichlet) aC+=2.0*KC;
				if(BCtop==dirichlet) aC+=2.0*KC;
				if(BCeast==dirichlet) aC+=2.0*KC;
				h_in[in_idx] = -(rhs[in_idx] - result/dxdx)/(aC/dxdx);
			}}
}

__global__ void GSRB_vertex_NEB(double *h_in, const double *rhs, const double *K, double dxdx, int Nx, int Ny, int Nz, bool isred, int BCnorth, int BCeast, int BCbottom){
	int stride = Nx*Ny;
	int ix = Nx-1;
	int iy = Ny-1;
	int iz = 0;
	int in_idx;
	in_idx = ix + iy*Nx + iz*stride;
	double result;
	double aC;
	double KC = K[in_idx],KN;
	if(isred) {if ((ix+iy+iz)%2 == 0){
				result=0.0; KN = 0.0; aC = 0.0;
				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-Nx]);
				result += h_in[in_idx-Nx]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+stride]);
				result += h_in[in_idx+stride]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-1]);
				result += h_in[in_idx-1 ]*KN;
				aC += KN;
				if(BCnorth==periodic) {
					KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-(Ny-1)*Nx]);
					result += h_in[in_idx-(Ny-1)*Nx]*KN;
					aC += KN;
				}
				if(BCbottom==periodic) {
					KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+(Nz-1)*stride]);
					result += h_in[in_idx+(Nz-1)*stride]*KN;
					aC += KN;
				}
				if(BCeast==periodic) {
					KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-(Nx-1)]);
					result += h_in[in_idx-(Nx-1)]*KN;
					aC += KN;
				}
				if(BCnorth==dirichlet) aC+=2.0*KC;
				if(BCbottom==dirichlet) aC+=2.0*KC;
				if(BCeast==dirichlet) aC+=2.0*KC;
				h_in[in_idx] = -(rhs[in_idx] - result/dxdx)/(aC/dxdx);
			}}
	else {if((ix+iy+iz)%2 != 0){
				result=0.0; KN = 0.0; aC = 0.0;
				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-Nx]);
				result += h_in[in_idx-Nx]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+stride]);
				result += h_in[in_idx+stride]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-1]);
				result += h_in[in_idx-1 ]*KN;
				aC += KN;
				if(BCnorth==periodic) {
					KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-(Ny-1)*Nx]);
					result += h_in[in_idx-(Ny-1)*Nx]*KN;
					aC += KN;
				}
				if(BCbottom==periodic) {
					KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+(Nz-1)*stride]);
					result += h_in[in_idx+(Nz-1)*stride]*KN;
					aC += KN;
				}
				if(BCeast==periodic) {
					KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-(Nx-1)]);
					result += h_in[in_idx-(Nx-1)]*KN;
					aC += KN;
				}
				if(BCnorth==dirichlet) aC+=2.0*KC;
				if(BCbottom==dirichlet) aC+=2.0*KC;
				if(BCeast==dirichlet) aC+=2.0*KC;
				h_in[in_idx] = -(rhs[in_idx] - result/dxdx)/(aC/dxdx);
			}}
}
__global__ void GSRB_vertex_NET(double *h_in, const double *rhs, const double *K, double dxdx, int Nx, int Ny, int Nz, bool isred, int BCnorth, int BCeast, int BCtop){
	int stride = Nx*Ny;
	int ix = Nx-1;
	int iy = Ny-1;
	int iz = Nz-1;
	int in_idx;
	in_idx = ix + iy*Nx + iz*stride;
	double result;
	double aC;
	double KC = K[in_idx],KN;
	if(isred) {if ((ix+iy+iz)%2 == 0){
				result=0.0; KN = 0.0; aC = 0.0;
				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-Nx]);
				result += h_in[in_idx-Nx]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-stride]);
				result += h_in[in_idx-stride]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-1]);
				result += h_in[in_idx-1 ]*KN;
				aC += KN;
				if(BCnorth==periodic) {
					KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-(Ny-1)*Nx]);
					result += h_in[in_idx-(Ny-1)*Nx]*KN;
					aC += KN;
				}
				if(BCtop==periodic) {
					KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-(Nz-1)*stride]);
					result += h_in[in_idx-(Nz-1)*stride]*KN;
					aC += KN;
				}
				if(BCeast==periodic) {
					KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-(Nx-1)]);
					result += h_in[in_idx-(Nx-1)]*KN;
					aC += KN;
				}
				if(BCnorth==dirichlet) aC+=2.0*KC;
				if(BCtop==dirichlet) aC+=2.0*KC;
				if(BCeast==dirichlet) aC+=2.0*KC;
				h_in[in_idx] = -(rhs[in_idx] - result/dxdx)/(aC/dxdx);
			}}
	else {if((ix+iy+iz)%2 != 0){
				result=0.0; KN = 0.0; aC = 0.0;
				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-Nx]);
				result += h_in[in_idx-Nx]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-stride]);
				result += h_in[in_idx-stride]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-1]);
				result += h_in[in_idx-1 ]*KN;
				aC += KN;
				if(BCnorth==periodic) {
					KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-(Ny-1)*Nx]);
					result += h_in[in_idx-(Ny-1)*Nx]*KN;
					aC += KN;
				}
				if(BCtop==periodic) {
					KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-(Nz-1)*stride]);
					result += h_in[in_idx-(Nz-1)*stride]*KN;
					aC += KN;
				}
				if(BCeast==periodic) {
					KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-(Nx-1)]);
					result += h_in[in_idx-(Nx-1)]*KN;
					aC += KN;
				}
				if(BCnorth==dirichlet) aC+=2.0*KC;
				if(BCtop==dirichlet) aC+=2.0*KC;
				if(BCeast==dirichlet) aC+=2.0*KC;
				h_in[in_idx] = -(rhs[in_idx] - result/dxdx)/(aC/dxdx);
			}}
}
__global__ void GSRB_vertex_NWB(double *h_in, const double *rhs, const double *K, double dxdx, int Nx, int Ny, int Nz, bool isred, int BCnorth, int BCwest, int BCbottom){
	int stride = Nx*Ny;
	int ix = 0;
	int iy = Ny-1;
	int iz = 0;
	int in_idx;
	in_idx = ix + iy*Nx + iz*stride;
	double result;
	double aC;
	double KC = K[in_idx],KN;
	if(isred) {if ((ix+iy+iz)%2 == 0){
				result=0.0; KN = 0.0; aC = 0.0;
				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-Nx]);
				result += h_in[in_idx-Nx]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+stride]);
				result += h_in[in_idx+stride]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+1]);
				result += h_in[in_idx+1 ]*KN;
				aC += KN;
				if(BCnorth==periodic) {
					KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-(Ny-1)*Nx]);
					result += h_in[in_idx-(Ny-1)*Nx]*KN;
					aC += KN;
				}
				if(BCbottom==periodic) {
					KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+(Nz-1)*stride]);
					result += h_in[in_idx+(Nz-1)*stride]*KN;
					aC += KN;
				}
				if(BCwest==periodic) {
					KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+(Nx-1)]);
					result += h_in[in_idx+(Nx-1)]*KN;
					aC += KN;
				}
				if(BCnorth==dirichlet) aC+=2.0*KC;
				if(BCbottom==dirichlet) aC+=2.0*KC;
				if(BCwest==dirichlet) aC+=2.0*KC;
				h_in[in_idx] = -(rhs[in_idx] - result/dxdx)/(aC/dxdx);
			}}
	else {if((ix+iy+iz)%2 != 0){
				result=0.0; KN = 0.0; aC = 0.0;
				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-Nx]);
				result += h_in[in_idx-Nx]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+stride]);
				result += h_in[in_idx+stride]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+1]);
				result += h_in[in_idx+1 ]*KN;
				aC += KN;
				if(BCnorth==periodic) {
					KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-(Ny-1)*Nx]);
					result += h_in[in_idx-(Ny-1)*Nx]*KN;
					aC += KN;
				}
				if(BCbottom==periodic) {
					KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+(Nz-1)*stride]);
					result += h_in[in_idx+(Nz-1)*stride]*KN;
					aC += KN;
				}
				if(BCwest==periodic) {
					KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+(Nx-1)]);
					result += h_in[in_idx+(Nx-1)]*KN;
					aC += KN;
				}
				if(BCnorth==dirichlet) aC+=2.0*KC;
				if(BCbottom==dirichlet) aC+=2.0*KC;
				if(BCwest==dirichlet) aC+=2.0*KC;
				h_in[in_idx] = -(rhs[in_idx] - result/dxdx)/(aC/dxdx);
			}}
}
__global__ void GSRB_vertex_NWT(double *h_in, const double *rhs, const double *K, double dxdx, int Nx, int Ny, int Nz, bool isred, int BCnorth, int BCwest, int BCtop){
	int stride = Nx*Ny;
	int ix = 0;
	int iy = Ny-1;
	int iz = Nz-1;
	int in_idx;
	in_idx = ix + iy*Nx + iz*stride;
	double result;
	double aC;
	double KC = K[in_idx],KN;
	if(isred) {if ((ix+iy+iz)%2 == 0){
				result=0.0; KN = 0.0; aC = 0.0;
				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-Nx]);
				result += h_in[in_idx-Nx]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-stride]);
				result += h_in[in_idx-stride]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+1]);
				result += h_in[in_idx+1 ]*KN;
				aC += KN;
				if(BCnorth==periodic) {
					KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-(Ny-1)*Nx]);
					result += h_in[in_idx-(Ny-1)*Nx]*KN;
					aC += KN;
				}
				if(BCtop==periodic) {
					KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-(Nz-1)*stride]);
					result += h_in[in_idx-(Nz-1)*stride]*KN;
					aC += KN;
				}
				if(BCwest==periodic) {
					KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+(Nx-1)]);
					result += h_in[in_idx+(Nx-1)]*KN;
					aC += KN;
				}
				if(BCnorth==dirichlet) aC+=2.0*KC;
				if(BCtop==dirichlet) aC+=2.0*KC;
				if(BCwest==dirichlet) aC+=2.0*KC;
				h_in[in_idx] = -(rhs[in_idx] - result/dxdx)/(aC/dxdx);
			}}
	else {if((ix+iy+iz)%2 != 0){
				result=0.0; KN = 0.0; aC = 0.0;
				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-Nx]);
				result += h_in[in_idx-Nx]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-stride]);
				result += h_in[in_idx-stride]*KN;
				aC += KN;

				KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+1]);
				result += h_in[in_idx+1 ]*KN;
				aC += KN;
				if(BCnorth==periodic) {
					KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-(Ny-1)*Nx]);
					result += h_in[in_idx-(Ny-1)*Nx]*KN;
					aC += KN;
				}
				if(BCtop==periodic) {
					KN = 2.0 / (1.0/KC  +  1.0/K[in_idx-(Nz-1)*stride]);
					result += h_in[in_idx-(Nz-1)*stride]*KN;
					aC += KN;
				}
				if(BCwest==periodic) {
					KN = 2.0 / (1.0/KC  +  1.0/K[in_idx+(Nx-1)]);
					result += h_in[in_idx+(Nx-1)]*KN;
					aC += KN;
				}
				if(BCnorth==dirichlet) aC+=2.0*KC;
				if(BCtop==dirichlet) aC+=2.0*KC;
				if(BCwest==dirichlet) aC+=2.0*KC;
				h_in[in_idx] = -(rhs[in_idx] - result/dxdx)/(aC/dxdx);
			}}
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// %%% ITERACION GAUSS SEIDEL RED BLACK (two half steps) %%%%%%%
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
void smooth_GSRB(double *xk, const double *rhs, double *rk_1, const double *K,
	double dxdx, int Nx, int Ny, int Nz, int BCbottom, int BCtop, int BCsouth, int BCnorth, int BCwest, int BCeast, bool pin1stCell,
	int itMAX, bool updateRes, dim3 grid, dim3 block){
	for(int i = 0; i<itMAX; i++){
		GSRB_int<<<gridXY,blockXY>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,true);
		GSRB_side_bottom<<<gridXY,blockXY>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,true,BCbottom);
		GSRB_side_top	<<<gridXY,blockXY>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,true,BCtop);
		GSRB_side_south<<<gridXZ,blockXZ>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,true,BCsouth);
		GSRB_side_north<<<gridXZ,blockXZ>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,true,BCnorth);
		GSRB_side_west<<<gridYZ,blockYZ>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,true,BCwest);
		GSRB_side_east<<<gridYZ,blockYZ>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,true,BCeast);
		GSRB_edge_X_South_Bottom<<<grid.x,block.x>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,true,BCsouth,BCbottom);
		GSRB_edge_X_South_Top	<<<grid.x,block.x>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,true,BCsouth,BCtop);
		GSRB_edge_X_North_Bottom<<<grid.x,block.x>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,true,BCnorth,BCbottom);
		GSRB_edge_X_North_Top	<<<grid.x,block.x>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,true,BCnorth,BCtop);
		GSRB_edge_Z_South_West<<<grid.z,block.z>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,true,BCsouth,BCwest);
		GSRB_edge_Z_South_East<<<grid.z,block.z>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,true,BCsouth,BCeast);
		GSRB_edge_Z_North_West<<<grid.z,block.z>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,true,BCnorth,BCwest);
		GSRB_edge_Z_North_East<<<grid.z,block.z>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,true,BCnorth,BCeast);
		GSRB_edge_Y_West_Bottom	<<<grid.y,block.y>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,true,BCbottom,BCwest);
		GSRB_edge_Y_West_Top	<<<grid.y,block.y>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,true,BCtop,BCwest);
		GSRB_edge_Y_East_Bottom	<<<grid.y,block.y>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,true,BCbottom,BCeast);
		GSRB_edge_Y_East_Top	<<<grid.y,block.y>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,true,BCtop,BCeast);
		GSRB_vertex_SWB<<<1,1>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,true,BCsouth,BCwest,BCbottom,pin1stCell);
		GSRB_vertex_SWT<<<1,1>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,true,BCsouth,BCwest,BCtop);
		GSRB_vertex_SEB<<<1,1>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,true,BCsouth,BCeast,BCbottom);
		GSRB_vertex_SET<<<1,1>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,true,BCsouth,BCeast,BCtop);
		GSRB_vertex_NEB<<<1,1>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,true,BCnorth,BCeast,BCbottom);
		GSRB_vertex_NET<<<1,1>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,true,BCnorth,BCeast,BCtop);
		GSRB_vertex_NWB<<<1,1>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,true,BCnorth,BCwest,BCbottom);
		GSRB_vertex_NWT<<<1,1>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,true,BCnorth,BCwest,BCtop);
		cudaDeviceSynchronize();

		GSRB_int		<<<gridXY,blockXY>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,false);
		GSRB_side_bottom<<<gridXY,blockXY>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,false,BCbottom);
		GSRB_side_top	<<<gridXY,blockXY>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,false,BCtop);
		GSRB_side_south<<<gridXZ,blockXZ>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,false,BCsouth);
		GSRB_side_north<<<gridXZ,blockXZ>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,false,BCnorth);
		GSRB_side_west<<<gridYZ,blockYZ>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,false,BCwest);
		GSRB_side_east<<<gridYZ,blockYZ>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,false,BCeast);
		GSRB_edge_X_South_Bottom<<<grid.x,block.x>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,false,BCsouth,BCbottom);
		GSRB_edge_X_South_Top	<<<grid.x,block.x>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,false,BCsouth,BCtop);
		GSRB_edge_X_North_Bottom<<<grid.x,block.x>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,false,BCnorth,BCbottom);
		GSRB_edge_X_North_Top	<<<grid.x,block.x>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,false,BCnorth,BCtop);
		GSRB_edge_Z_South_West	<<<grid.z,block.z>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,false,BCsouth,BCwest);
		GSRB_edge_Z_South_East	<<<grid.z,block.z>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,false,BCsouth,BCeast);
		GSRB_edge_Z_North_West	<<<grid.z,block.z>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,false,BCnorth,BCwest);
		GSRB_edge_Z_North_East	<<<grid.z,block.z>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,false,BCnorth,BCeast);
		GSRB_edge_Y_West_Bottom	<<<grid.y,block.y>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,false,BCbottom,BCwest);
		GSRB_edge_Y_West_Top	<<<grid.y,block.y>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,false,BCtop,BCwest);
		GSRB_edge_Y_East_Bottom	<<<grid.y,block.y>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,false,BCbottom,BCeast);
		GSRB_edge_Y_East_Top	<<<grid.y,block.y>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,false,BCtop,BCeast);
		GSRB_vertex_SWB<<<1,1>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,false,BCsouth,BCwest,BCbottom,pin1stCell);
		GSRB_vertex_SWT<<<1,1>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,false,BCsouth,BCwest,BCtop);
		GSRB_vertex_SEB<<<1,1>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,false,BCsouth,BCeast,BCbottom);
		GSRB_vertex_SET<<<1,1>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,false,BCsouth,BCeast,BCtop);
		GSRB_vertex_NEB<<<1,1>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,false,BCnorth,BCeast,BCbottom);
		GSRB_vertex_NET<<<1,1>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,false,BCnorth,BCeast,BCtop);
		GSRB_vertex_NWB<<<1,1>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,false,BCnorth,BCwest,BCbottom);
		GSRB_vertex_NWT<<<1,1>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,false,BCnorth,BCwest,BCtop);
		cudaDeviceSynchronize();
	}
	if(updateRes) update_res(rk_1,xk,rhs,K,dxdx,Nx,Ny,Nz,
		BCbottom,BCtop,BCsouth,BCnorth,BCwest,BCeast,pin1stCell,
		grid,block);
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// %%%%%%%%   SOLVE COARSEST SYSTEM WITH GS-RB   %%%%%%%%%%%%%%%
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
void SolveCoarseSystemGSRB(double *xnew, const double *rhs, double *r, const double *K, double dxdx, int Nx, int Ny, int Nz, const int itMAX,
	dim3 grid, dim3 block,
	cublasHandle_t handle,
	int BCbottom, int BCtop,
	int BCsouth, int BCnorth,
	int BCwest, int BCeast, bool pin1stCell){
	double *rr_new; cudaMalloc(&rr_new,sizeof(double));
	double *rr_new_h; rr_new_h = new double[1];
	int iter = 0;
	*rr_new_h = 1.0;
	while( (pow((*rr_new_h),0.5) > 1e-16 ) && iter<itMAX){
		smooth_GSRB(xnew,rhs,r,K,dxdx,Nx,Ny,Nz,
			BCbottom,BCtop,BCsouth,
			BCnorth,BCwest,BCeast,
			pin1stCell,4,false,
			grid,block);
		iter+=4;
		update_res(r,xnew,rhs,K,dxdx,Nx,Ny,Nz,
		BCbottom,BCtop,BCsouth,BCnorth,BCwest,BCeast,pin1stCell,
		grid,block);
		cublasDdot(handle,Nx*Ny*Nz,r,1,r,1, rr_new); cudaDeviceSynchronize();
		cudaMemcpy(rr_new_h, rr_new, sizeof(double), cudaMemcpyDeviceToHost);
	}
	cudaFree(rr_new);
	delete [] rr_new_h;
}

```

# GSRB_Smooth_up_residual_3D.cu

```cu
/**
* @file GSRB_Smooth_up_residual_3D.cu
* @brief Gauss Seidel Red Black method implementation (matrix-free style) for solving
* the flow equation with CCMG method
*
* @author Lucas Bessone (contact: lcbessone@gmail.com)
*
* @copyright This file is part of the EU-PAR software.
*            Copyright (C) 2025 Lucas Bessone
*
* @license This program is free software: you can redistribute it and/or modify
*          it under the terms of the GNU General Public License as published by
*          the Free Software Foundation, either version 3 of the License, or
*          (at your option) any later version.
*
*          This program is distributed in the hope that it will be useful,
*          but WITHOUT ANY WARRANTY; without even the implied warranty of
*          MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*          GNU General Public License for more details.
*
*          You should have received a copy of the GNU General Public License
*          along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/
#include <cuda_runtime_api.h>
#include "cublas_v2.h"
#include <iostream>

#include "header/macros_index_kernel.h"// macros para manejo de indices y aplicación de condiciones de borde

#define neumann 0
#define periodic 1
#define dirichlet 2

#define gridXY dim3(grid.x,grid.y)
#define blockXY dim3(block.x,block.y)
#define gridXZ dim3(grid.x,grid.z)
#define blockXZ dim3(block.x,block.z)
#define gridYZ dim3(grid.y,grid.z)
#define blockYZ dim3(block.y,block.z)

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// %%%%%%% Declare auxiliar function for update residual %%%%%%%
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
void update_res(double *rk_1, double *xk, const double *rhs, const double *K,
	double dxdx, int Nx, int Ny, int Nz,
	int BCbottom, int BCtop,
	int BCsouth, int BCnorth,
	int BCwest, int BCeast, bool pin1stCell,
	dim3 grid, dim3 block);

//#######################################################################
// 	routine smooth_GSRB (system A*phi = r)
//	perform one iteration Symetric Gauss-Seidel
//  Red-Black ordering (two half step)
//  phiC <- (rC - sum(AF*phiF) )/AC , F~{E, W, N, S, T, B}
//	The matrix A must be SPD, else, change the stencil
//  for other problems (eg. variable permeability) the stencil must be modified
//#######################################################################
// kernel for interior cells
__global__ void GSRB_int(double *h_in, const double *rhs, const double *K, double dxdx, int Nx, int Ny, int Nz, bool isred){
	int ix = threadIdx.x + blockIdx.x*blockDim.x;
	int iy = threadIdx.y + blockIdx.y*blockDim.y;
	if (ix >= Nx-2 || iy >= Ny-2) return;
	double result;
	int out_idx = (ix+1) + (iy+1)*Nx;
	double aC;
	double KC;
	double KN;
	int offsets[] = {1, Nx, -1, -Nx, stride, -stride};
	for(int iz=1; iz<Nz-1; ++iz){
		out_idx += stride;
		KC = K[out_idx];
		if ((isred && (ix + 1 + iy + 1 + iz) % 2 == 0) || (!isred && (ix + 1 + iy + 1 + iz) % 2 != 0)) {
		    result = 0.0;
		    KN = 0.0;
		    aC = 0.0;
		    for (int i = 0; i < 6; i++) {
		        KN = 2.0 / (1.0 / KC + 1.0 / K[out_idx + offsets[i]]);
		        result += h_in[out_idx + offsets[i]] * KN;
		        aC += KN;
		    }
		    h_in[out_idx] = -(rhs[out_idx] - result / dxdx) / (aC / dxdx);
		}
	}
}

// kernel for boundary face cells
#define KERNEL_SIDE(FACE) GSRB_side_##FACE(double *h_in, const double *rhs, const double *K, double dxdx, int Nx, int Ny, int Nz, bool isred, int BC_##FACE){ \
COMPUTE_INDEX_##FACE \
OFFSETS_##FACE; \
double result = 0.0, KC = K[in_idx], KN = 0.0, aC = 0.0; \
if ((isred && (ix + iy + iz + 1 + 1) % 2 == 0) || (!isred && (ix + iy + iz + 1 + 1) % 2 != 0)){ \
	for (int i = 0; i < 5; i++) { \
	    KN = 2.0 / (1.0/KC  +  1.0/K[in_idx + offsets[i]]); \
	    result += h_in[in_idx + offsets[i]] * KN; \
	    aC += KN; \
	} \
	if (BC_##FACE == periodic) { \
	    KN = 2.0 / (1.0/KC  +  1.0/K[in_idx + PERIODIC_##FACE]); \
	    result += h_in[in_idx + PERIODIC_##FACE] * KN; \
	    aC += KN; \
	} \
	if (BC_##FACE == dirichlet) aC += 2.0*KC; \
	h_in[in_idx] = -(rhs[in_idx] - result/dxdx) / (aC/dxdx); \
} \
}

// kernel for boundary edge cells
#define KERNEL_EDGE(FACE1,FACE2) GSRB_edge_##FACE1##_##FACE2(double *h_in, const double *rhs, const double *K, double dxdx, int Nx, int Ny, int Nz, bool isred, int BC_##FACE1, int BC_##FACE2){ \
COMPUTE_INDEX_##FACE1##_##FACE2 \
OFFSETS_##FACE1##_##FACE2; \
double result = 0.0, KC = K[in_idx], KN = 0.0, aC = 0.0; \
if ((isred && (ix + iy + iz + 1) % 2 == 0) || (!isred && (ix + iy + iz + 1) % 2 != 0)){ \
	for (int i = 0; i < 4; i++) { \
	    KN = 2.0 / (1.0/KC  +  1.0/K[in_idx + offsets[i]]); \
	    result += h_in[in_idx + offsets[i]] * KN; \
	    aC += KN; \
	} \
	if (BC_##FACE1 == periodic) { \
	    KN = 2.0 / (1.0/KC  +  1.0/K[in_idx + PERIODIC_##FACE1]); \
	    result += h_in[in_idx + PERIODIC_##FACE1] * KN; \
	    aC += KN; \
	} \
	if (BC_##FACE2 == periodic) { \
	    KN = 2.0 / (1.0/KC  +  1.0/K[in_idx + PERIODIC_##FACE2]); \
	    result += h_in[in_idx + PERIODIC_##FACE2] * KN; \
	    aC += KN; \
	} \
	if (BC_##FACE1 == dirichlet) aC += 2.0 * KC; \
	if (BC_##FACE2 == dirichlet) aC += 2.0 * KC; \
	h_in[in_idx] = -(rhs[in_idx]  - result/dxdx) / (aC/dxdx); \
} \
}

// kernel for boundary vertex cells
#define KERNEL_VERTEX(FACE1,FACE2,FACE3) GSRB_vertex_##FACE1##_##FACE2##_##FACE3(double *h_in, const double *rhs, const double *K, double dxdx, int Nx, int Ny, int Nz, bool isred, int BC_##FACE1, int BC_##FACE2, int BC_##FACE3){ \
COMPUTE_INDEX_##FACE1##_##FACE2##_##FACE3  \
OFFSETS_##FACE1##_##FACE2##_##FACE3; \
double result = 0.0, KC = K[in_idx], KN = 0.0, aC = 0.0; \
if ((isred && (ix+iy+iz)%2 == 0) || (!isred && (ix+iy+iz)%2 != 0)){ \
	for (int i = 0; i < 3; i++) { \
	    KN = 2.0 / (1.0/KC  +  1.0/K[in_idx + offsets[i]]); \
	    result += h_in[in_idx + offsets[i]] * KN; \
	    aC += KN; \
	} \
	if (BC_##FACE1 == periodic) { \
	    KN = 2.0 / (1.0/KC  +  1.0/K[in_idx + PERIODIC_##FACE1]); \
	    result += h_in[in_idx + PERIODIC_##FACE1] * KN; \
	    aC += KN; \
	} \
	if (BC_##FACE2 == periodic) { \
	    KN = 2.0 / (1.0/KC  +  1.0/K[in_idx + PERIODIC_##FACE2]); \
	    result += h_in[in_idx + PERIODIC_##FACE2] * KN; \
	    aC += KN; \
	} \
	if (BC_##FACE3 == periodic) { \
	    KN = 2.0 / (1.0/KC  +  1.0/K[in_idx + PERIODIC_##FACE3]); \
	    result += h_in[in_idx + PERIODIC_##FACE3] * KN; \
	    aC += KN; \
	} \
	if (BC_##FACE1 == dirichlet) aC += 2.0 * KC; \
	if (BC_##FACE2 == dirichlet) aC += 2.0 * KC; \
	if (BC_##FACE3 == dirichlet) aC += 2.0 * KC; \
	h_in[in_idx] = -(rhs[in_idx]  - result/dxdx) / (aC/dxdx); \
} \
}

// kernel declaration for all domain except interior
__global__ void KERNEL_SIDE(BOTTOM)
__global__ void KERNEL_SIDE(TOP)
__global__ void KERNEL_SIDE(NORTH)
__global__ void KERNEL_SIDE(SOUTH)
__global__ void KERNEL_SIDE(WEST)
__global__ void KERNEL_SIDE(EAST)
// edge y
__global__ void KERNEL_EDGE(WEST,BOTTOM)
__global__ void KERNEL_EDGE(WEST,TOP)
__global__ void KERNEL_EDGE(EAST,BOTTOM)
__global__ void KERNEL_EDGE(EAST,TOP)
// edge z
__global__ void KERNEL_EDGE(WEST,SOUTH)
__global__ void KERNEL_EDGE(WEST,NORTH)
__global__ void KERNEL_EDGE(EAST,SOUTH)
__global__ void KERNEL_EDGE(EAST,NORTH)
// edge x
__global__ void KERNEL_EDGE(SOUTH,BOTTOM)
__global__ void KERNEL_EDGE(SOUTH,TOP)
__global__ void KERNEL_EDGE(NORTH,BOTTOM)
__global__ void KERNEL_EDGE(NORTH,TOP)

__global__ void KERNEL_VERTEX(WEST,SOUTH,BOTTOM)
__global__ void KERNEL_VERTEX(WEST,SOUTH,TOP)
__global__ void KERNEL_VERTEX(WEST,NORTH,BOTTOM)
__global__ void KERNEL_VERTEX(WEST,NORTH,TOP)
__global__ void KERNEL_VERTEX(EAST,SOUTH,BOTTOM)
__global__ void KERNEL_VERTEX(EAST,SOUTH,TOP)
__global__ void KERNEL_VERTEX(EAST,NORTH,BOTTOM)
__global__ void KERNEL_VERTEX(EAST,NORTH,TOP)

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// %%% ITERACION GAUSS SEIDEL RED BLACK (two half steps) %%%%%%%
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
void smooth_GSRB(double *xk, const double *rhs, double *rk_1, const double *K,
	double dxdx, int Nx, int Ny, int Nz, int BCbottom, int BCtop, int BCsouth, int BCnorth, int BCwest, int BCeast, bool pin1stCell,
	int itMAX, bool updateRes, dim3 grid, dim3 block){
	for(int i = 0; i<itMAX; i++){
		GSRB_int<<<gridXY,blockXY>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,true);
		GSRB_side_BOTTOM<<<gridXY,blockXY>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,true,BCbottom);
		GSRB_side_TOP	<<<gridXY,blockXY>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,true,BCtop);
		GSRB_side_SOUTH<<<gridXZ,blockXZ>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,true,BCsouth);
		GSRB_side_NORTH<<<gridXZ,blockXZ>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,true,BCnorth);
		GSRB_side_WEST<<<gridYZ,blockYZ>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,true,BCwest);
		GSRB_side_EAST<<<gridYZ,blockYZ>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,true,BCeast);
		GSRB_edge_SOUTH_BOTTOM<<<grid.x,block.x>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,true,BCsouth,BCbottom);
		GSRB_edge_SOUTH_TOP	<<<grid.x,block.x>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,true,BCsouth,BCtop);
		GSRB_edge_NORTH_BOTTOM<<<grid.x,block.x>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,true,BCnorth,BCbottom);
		GSRB_edge_NORTH_TOP	<<<grid.x,block.x>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,true,BCnorth,BCtop);
		GSRB_edge_WEST_SOUTH<<<grid.z,block.z>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,true,BCwest,BCsouth);
		GSRB_edge_EAST_SOUTH<<<grid.z,block.z>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,true,BCeast,BCsouth);
		GSRB_edge_WEST_NORTH<<<grid.z,block.z>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,true,BCwest,BCnorth);
		GSRB_edge_EAST_NORTH<<<grid.z,block.z>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,true,BCeast,BCnorth);
		GSRB_edge_WEST_BOTTOM	<<<grid.y,block.y>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,true,BCwest,BCbottom);
		GSRB_edge_WEST_TOP	<<<grid.y,block.y>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,true,BCwest,BCtop);
		GSRB_edge_EAST_BOTTOM	<<<grid.y,block.y>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,true,BCeast,BCbottom);
		GSRB_edge_EAST_TOP	<<<grid.y,block.y>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,true,BCeast,BCtop);
		GSRB_vertex_WEST_SOUTH_BOTTOM<<<1,1>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,true,BCwest,BCsouth,BCbottom);//,pin1stCell);
		GSRB_vertex_WEST_SOUTH_TOP<<<1,1>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,true,BCwest,BCsouth,BCtop);
		GSRB_vertex_EAST_SOUTH_BOTTOM<<<1,1>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,true,BCeast,BCsouth,BCbottom);
		GSRB_vertex_EAST_SOUTH_TOP<<<1,1>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,true,BCeast,BCsouth,BCtop);
		GSRB_vertex_EAST_NORTH_BOTTOM<<<1,1>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,true,BCeast,BCnorth,BCbottom);
		GSRB_vertex_EAST_NORTH_TOP<<<1,1>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,true,BCeast,BCnorth,BCtop);
		GSRB_vertex_WEST_NORTH_BOTTOM<<<1,1>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,true,BCwest,BCnorth,BCbottom);
		GSRB_vertex_WEST_NORTH_TOP<<<1,1>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,true,BCwest,BCnorth,BCtop);
		cudaDeviceSynchronize();

		GSRB_int		<<<gridXY,blockXY>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,false);
		GSRB_side_BOTTOM<<<gridXY,blockXY>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,false,BCbottom);
		GSRB_side_TOP	<<<gridXY,blockXY>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,false,BCtop);
		GSRB_side_SOUTH<<<gridXZ,blockXZ>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,false,BCsouth);
		GSRB_side_NORTH<<<gridXZ,blockXZ>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,false,BCnorth);
		GSRB_side_WEST<<<gridYZ,blockYZ>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,false,BCwest);
		GSRB_side_EAST<<<gridYZ,blockYZ>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,false,BCeast);
		GSRB_edge_SOUTH_BOTTOM<<<grid.x,block.x>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,false,BCsouth,BCbottom);
		GSRB_edge_SOUTH_TOP	<<<grid.x,block.x>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,false,BCsouth,BCtop);
		GSRB_edge_NORTH_BOTTOM<<<grid.x,block.x>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,false,BCnorth,BCbottom);
		GSRB_edge_NORTH_TOP	<<<grid.x,block.x>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,false,BCnorth,BCtop);
		GSRB_edge_WEST_SOUTH	<<<grid.z,block.z>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,false,BCwest,BCsouth);
		GSRB_edge_EAST_SOUTH	<<<grid.z,block.z>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,false,BCeast,BCsouth);
		GSRB_edge_WEST_NORTH	<<<grid.z,block.z>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,false,BCwest,BCnorth);
		GSRB_edge_EAST_NORTH	<<<grid.z,block.z>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,false,BCeast,BCnorth);
		GSRB_edge_WEST_BOTTOM	<<<grid.y,block.y>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,false,BCwest,BCbottom);
		GSRB_edge_WEST_TOP	<<<grid.y,block.y>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,false,BCwest,BCtop);
		GSRB_edge_EAST_BOTTOM	<<<grid.y,block.y>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,false,BCeast,BCbottom);
		GSRB_edge_EAST_TOP	<<<grid.y,block.y>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,false,BCeast,BCtop);
		GSRB_vertex_WEST_SOUTH_BOTTOM<<<1,1>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,false,BCwest,BCsouth,BCbottom);//,pin1stCell);
		GSRB_vertex_WEST_SOUTH_TOP<<<1,1>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,false,BCwest,BCsouth,BCtop);
		GSRB_vertex_EAST_SOUTH_BOTTOM<<<1,1>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,false,BCeast,BCsouth,BCbottom);
		GSRB_vertex_EAST_SOUTH_TOP<<<1,1>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,false,BCeast,BCsouth,BCtop);
		GSRB_vertex_EAST_NORTH_BOTTOM<<<1,1>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,false,BCeast,BCnorth,BCbottom);
		GSRB_vertex_EAST_NORTH_TOP<<<1,1>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,false,BCeast,BCnorth,BCtop);
		GSRB_vertex_WEST_NORTH_BOTTOM<<<1,1>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,false,BCwest,BCnorth,BCbottom);
		GSRB_vertex_WEST_NORTH_TOP<<<1,1>>>(xk,rhs,K,dxdx,Nx,Ny,Nz,false,BCwest,BCnorth,BCtop);
		cudaDeviceSynchronize();
	}
	if(updateRes) update_res(rk_1,xk,rhs,K,dxdx,Nx,Ny,Nz,
		BCbottom,BCtop,BCsouth,BCnorth,BCwest,BCeast,pin1stCell,
		grid,block);
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// %%%%%%%%   SOLVE COARSEST SYSTEM WITH GS-RB   %%%%%%%%%%%%%%%
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
void SolveCoarseSystemGSRB(double *xnew, const double *rhs, double *r, const double *K, double dxdx, int Nx, int Ny, int Nz, const int itMAX,
	dim3 grid, dim3 block,
	cublasHandle_t handle,
	int BCbottom, int BCtop,
	int BCsouth, int BCnorth,
	int BCwest, int BCeast, bool pin1stCell){
	double *rr_new; cudaMalloc(&rr_new,sizeof(double));
	double *rr_new_h; rr_new_h = new double[1];
	int iter = 0;
	*rr_new_h = 1.0;
	while( (pow((*rr_new_h),0.5) > 1e-16 ) && iter<itMAX){
		smooth_GSRB(xnew,rhs,r,K,dxdx,Nx,Ny,Nz,
			BCbottom,BCtop,BCsouth,
			BCnorth,BCwest,BCeast,
			pin1stCell,4,false,
			grid,block);
		iter+=4;
		update_res(r,xnew,rhs,K,dxdx,Nx,Ny,Nz,
		BCbottom,BCtop,BCsouth,BCnorth,BCwest,BCeast,pin1stCell,
		grid,block);
		cublasDdot(handle,Nx*Ny*Nz,r,1,r,1, rr_new); cudaDeviceSynchronize();
		cudaMemcpy(rr_new_h, rr_new, sizeof(double), cudaMemcpyDeviceToHost);
	}
	cudaFree(rr_new);
	delete [] rr_new_h;
}

```

# header\homogenization_permeability.h

```h
void HomogenizationPermeability(double *phiCoarse, const double *phiFine, int Nx, int Ny, int Nz, dim3 grid, dim3 block);

void restriction(double *phiCoarse, const double *phiFine, int Nx, int Ny, int Nz, dim3 grid, dim3 block);

void prolongation(double *phiFine, const double *phiCoarse, int Nx, int Ny, int Nz,
	dim3 grid, dim3 block);



```

# header\linear_operator.h

```h

class Matrix_t{
public:
	// generic stencil operation output <- A*input
	virtual void stencil(double *output, double *input)=0;
};

class IdentityPrecond : public  Matrix_t {
private:
	int Nx, Ny, Nz;
public:
	IdentityPrecond(int Nx, int Ny, int Nz):
		Nx(Nx), Ny(Ny), Nz(Nz) {};
	void stencil(double *output, double *input){
		cudaMemcpy(output,input,sizeof(double)*Nx*Ny*Nz,cudaMemcpyDeviceToDevice);
		cudaDeviceSynchronize();
	}
};

#define gridXY dim3(grid.x,grid.y)
#define blockXY dim3(block.x,block.y)
#define gridXZ dim3(grid.x,grid.z)
#define blockXZ dim3(block.x,block.z)
#define gridYZ dim3(grid.y,grid.z)
#define blockYZ dim3(block.y,block.z)

class blas_t{
public:
	// common routines for CG (& BiCGStab) method
	virtual void AXPBY3D(const double *x, double *y, double *output, const double &a, const double &b)=0;
	virtual void alpha3D(const double *y, double *x, const double *rz, const double *yP, bool plus_minus)=0;
	virtual void beta3D(const double *x, double *y, const double *rz, const double *rz_old)=0;

	// common routines for BiCGStab method
	virtual void alphaU3D(const double *Ap, const double *r,  double *s, const double *rr_, const double *Apr_)=0;
	virtual void betaU3D(const double *r, double *p, const double *Ap, const double *rr_, const double *Apr_, const double *Ass, const double *AsAs)=0;
	virtual void omegaX3D(const double *p, double *x, const double *s, const double *rr_, const double *Apr_, const double *Ass, const double *AsAs)=0;
	virtual void omegaR3D(const double *s, double *r, const double *As, const double *Ass, const double *AsAs)=0;

	//
	virtual void Ddot(double *x, double *y, double *result)=0;
	virtual void copyVector_d2d(double *dst, double *src)=0;
	virtual void copyScalar_d2d(double *dst, double *src)=0;
	virtual void copyVector_d2h(double *dst, double *src)=0;
	virtual void copyScalar_d2h(double *dst, double *src)=0;
	virtual void copyVector_h2d(double *dst, double *src)=0;
	virtual void copyScalar_h2d(double *dst, double *src)=0;
};

#include "routines_CG.h"
#include "routines_BiCGStab.h"
#include "cublas_v2.h"
class blas: public blas_t {
private:
	int Nx, Ny, Nz;
	dim3 grid, block;
	cublasHandle_t handle;
public:
	blas(int Nx, int Ny, int Nz, dim3 grid, dim3 block, cublasHandle_t &handle):
	Nx(Nx), Ny(Ny), Nz(Nz), grid(grid), block(block), handle(handle)  {};
	void AXPBY3D(const double *x, double *y, double *output, const double &a, const double &b){
		AXPBY(x,y,output,a,b,Nx,Ny,Nz,grid,block);
	}
	void alpha3D(const double *y, double *x, const double *rz, const double *yP, bool plus_minus){
		alpha(y,x,rz,yP,plus_minus,Nx,Ny,Nz,grid,block);
	}
	void beta3D(const double *x, double *y, const double *rz, const double *rz_old){
		beta(x,y,rz,rz_old,Nx,Ny,Nz,grid,block);
	}

	void alphaU3D(const double *Ap, const double *r,  double *s, const double *rr_, const double *Apr_){
		alphaU(Ap,r,s,rr_,Apr_,Nx,Ny,Nz,grid,block);
	}
	void betaU3D(const double *r, double *p, const double *Ap, const double *rr_, const double *Apr_, const double *Ass, const double *AsAs){
		betaU(r,p,Ap,rr_,Apr_,Ass,AsAs,Nx,Ny,Nz,grid,block);
	}
	void omegaX3D(const double *p, double *x, const double *s, const double *rr_, const double *Apr_, const double *Ass, const double *AsAs){
		omegaX(p,x,s,rr_,Apr_,Ass,AsAs,Nx,Ny,Nz,grid,block);
	}
	void omegaR3D(const double *s, double *r, const double *As, const double *Ass, const double *AsAs){
		omegaR(s,r,As,Ass,AsAs,Nx,Ny,Nz,grid,block);
	}

	void Ddot(double *x, double *y, double *result){
		cublasDdot(handle,Nx*Ny*Nz,x,1,y,1,result);
		cudaDeviceSynchronize();
	}
	void copyVector_d2d(double *dst, double *src){
		cudaMemcpy(dst,src,sizeof(double)*Nx*Ny*Nz,cudaMemcpyDeviceToDevice);
		cudaDeviceSynchronize();
	}
	void copyScalar_d2d(double *dst, double *src){
		cudaMemcpy(dst,src,sizeof(double),cudaMemcpyDeviceToDevice);
		cudaDeviceSynchronize();
	}
	void copyVector_d2h(double *dst, double *src){
		cudaMemcpy(dst,src,sizeof(double)*Nx*Ny*Nz,cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
	}
	void copyScalar_d2h(double *dst, double *src){
		cudaMemcpy(dst,src,sizeof(double),cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
	}
	void copyVector_h2d(double *dst, double *src){
		cudaMemcpy(dst,src,sizeof(double)*Nx*Ny*Nz,cudaMemcpyHostToDevice);
		cudaDeviceSynchronize();
	}
	void copyScalar_h2d(double *dst, double *src){
		cudaMemcpy(dst,src,sizeof(double),cudaMemcpyHostToDevice);
		cudaDeviceSynchronize();
	}
};
```

# header\macros_index_for_mf.h

```h
#define COMPUTE_INDEX_NORMAL_VELOCITY_BOTTOM \
int idx_U = (ix+1+1) + (iy+1)*(Nx+1) + (iz)*(Nx+1)*Ny; \
int idx_V = (ix+1)   + (iy+1+1)*(Nx) + (iz)*Nx*(Ny+1); \
int idx_W = (ix+1)   + (iy+1)*(Nx)   + (iz+1)*stride;
#define COMPUTE_INDEX_NORMAL_VELOCITY_TOP COMPUTE_INDEX_NORMAL_VELOCITY_BOTTOM

#define COMPUTE_INDEX_NORMAL_VELOCITY_SOUTH \
int idx_U = (ix+1+1) + (iy)*(Nx+1) + (iz+1)*(Nx+1)*Ny; \
int idx_V = (ix+1)   + (iy+1)*(Nx) + (iz+1)*Nx*(Ny+1); \
int idx_W = (ix+1)   + (iy)*(Nx)   + (iz+1+1)*stride;
#define COMPUTE_INDEX_NORMAL_VELOCITY_NORTH COMPUTE_INDEX_NORMAL_VELOCITY_SOUTH

#define COMPUTE_INDEX_NORMAL_VELOCITY_WEST \
int idx_U = (ix+1) + (iy+1)*(Nx+1) + (iz+1)*(Nx+1)*Ny; \
int idx_V = (ix)   + (iy+1+1)*(Nx) + (iz+1)*Nx*(Ny+1); \
int idx_W = (ix)   + (iy+1)*(Nx)   + (iz+1+1)*stride;
#define COMPUTE_INDEX_NORMAL_VELOCITY_EAST COMPUTE_INDEX_NORMAL_VELOCITY_WEST

#define COMPUTE_INDEX_NORMAL_VELOCITY_SOUTH_BOTTOM \
int idx_U = (ix+1+1) + (iy)*(Nx+1) + (iz)*(Nx+1)*Ny; \
int idx_V = (ix+1)   + (iy+1)*(Nx) + (iz)*Nx*(Ny+1); \
int idx_W = (ix+1)   + (iy)*(Nx)   + (iz+1)*stride;
#define COMPUTE_INDEX_NORMAL_VELOCITY_SOUTH_TOP COMPUTE_INDEX_NORMAL_VELOCITY_SOUTH_BOTTOM
#define COMPUTE_INDEX_NORMAL_VELOCITY_NORTH_BOTTOM COMPUTE_INDEX_NORMAL_VELOCITY_SOUTH_BOTTOM
#define COMPUTE_INDEX_NORMAL_VELOCITY_NORTH_TOP COMPUTE_INDEX_NORMAL_VELOCITY_SOUTH_BOTTOM

#define COMPUTE_INDEX_NORMAL_VELOCITY_WEST_BOTTOM \
int idx_U = (ix+1) + (iy+1)*(Nx+1) + (iz)*(Nx+1)*Ny; \
int idx_V = (ix)   + (iy+1+1)*(Nx) + (iz)*Nx*(Ny+1); \
int idx_W = (ix)   + (iy+1)*(Nx)   + (iz+1)*stride;
#define COMPUTE_INDEX_NORMAL_VELOCITY_WEST_TOP COMPUTE_INDEX_NORMAL_VELOCITY_WEST_BOTTOM
#define COMPUTE_INDEX_NORMAL_VELOCITY_EAST_BOTTOM COMPUTE_INDEX_NORMAL_VELOCITY_WEST_BOTTOM
#define COMPUTE_INDEX_NORMAL_VELOCITY_EAST_TOP COMPUTE_INDEX_NORMAL_VELOCITY_WEST_BOTTOM

#define COMPUTE_INDEX_NORMAL_VELOCITY_WEST_SOUTH \
int idx_U = (ix+1) + (iy)*(Nx+1) + (iz+1)*(Nx+1)*Ny; \
int idx_V = (ix)   + (iy+1)*(Nx) + (iz+1)*Nx*(Ny+1); \
int idx_W = (ix)   + (iy)*(Nx)   + (iz+1+1)*stride;
#define COMPUTE_INDEX_NORMAL_VELOCITY_EAST_SOUTH COMPUTE_INDEX_NORMAL_VELOCITY_WEST_SOUTH
#define COMPUTE_INDEX_NORMAL_VELOCITY_WEST_NORTH COMPUTE_INDEX_NORMAL_VELOCITY_WEST_SOUTH
#define COMPUTE_INDEX_NORMAL_VELOCITY_EAST_NORTH COMPUTE_INDEX_NORMAL_VELOCITY_WEST_SOUTH

#define COMPUTE_INDEX_NORMAL_VELOCITY_WEST_SOUTH_BOTTOM \
int idx_U = (ix+1) + (iy)*(Nx+1) + (iz)*(Nx+1)*Ny; \
int idx_V = (ix)   + (iy+1)*(Nx) + (iz)*Nx*(Ny+1); \
int idx_W = (ix)   + (iy)*(Nx)   + (iz+1)*stride;
#define COMPUTE_INDEX_NORMAL_VELOCITY_WEST_SOUTH_TOP COMPUTE_INDEX_NORMAL_VELOCITY_WEST_SOUTH_BOTTOM
#define COMPUTE_INDEX_NORMAL_VELOCITY_WEST_NORTH_BOTTOM COMPUTE_INDEX_NORMAL_VELOCITY_WEST_SOUTH_BOTTOM
#define COMPUTE_INDEX_NORMAL_VELOCITY_WEST_NORTH_TOP COMPUTE_INDEX_NORMAL_VELOCITY_WEST_SOUTH_BOTTOM
#define COMPUTE_INDEX_NORMAL_VELOCITY_EAST_SOUTH_BOTTOM COMPUTE_INDEX_NORMAL_VELOCITY_WEST_SOUTH_BOTTOM
#define COMPUTE_INDEX_NORMAL_VELOCITY_EAST_SOUTH_TOP COMPUTE_INDEX_NORMAL_VELOCITY_WEST_SOUTH_BOTTOM
#define COMPUTE_INDEX_NORMAL_VELOCITY_EAST_NORTH_BOTTOM COMPUTE_INDEX_NORMAL_VELOCITY_WEST_SOUTH_BOTTOM
#define COMPUTE_INDEX_NORMAL_VELOCITY_EAST_NORTH_TOP COMPUTE_INDEX_NORMAL_VELOCITY_WEST_SOUTH_BOTTOM

// The order must match with the OFFSETS array as they are iterated in the same loop (see macros_indez_kernel.h)
// #define OFFSETS_BOTTOM int offsets[5] = { -1, 1, -Nx, Nx,  stride }
// #define OFFSETS_TOP int    offsets[5] = { -1, 1, -Nx, Nx, -stride }
// #define OFFSETS_EAST int offsets[5] = { -1, -Nx,  Nx, -stride, stride }
// #define OFFSETS_WEST int offsets[5] = {  1, -Nx,  Nx, -stride, stride }
// #define OFFSETS_NORTH int offsets[5] = { -1,  1, -Nx, -stride,  stride }
// #define OFFSETS_SOUTH int offsets[5] = { -1,  1,  Nx, -stride,  stride }
#define LOAD_FACE_FLUX_BOTTOM double m_f[6] = {-Up[idx_U-1]*A, Up[idx_U]*A, -Vp[idx_V-Nx]*A, Vp[idx_V]*A, Wp[idx_W]*A, -Wp[idx_W-stride]*A}
#define LOAD_FACE_FLUX_TOP double m_f[6] = {-Up[idx_U-1]*A, Up[idx_U]*A, -Vp[idx_V-Nx]*A, Vp[idx_V]*A, -Wp[idx_W-stride]*A, Wp[idx_W]*A}
#define LOAD_FACE_FLUX_EAST double m_f[6] = {-Up[idx_U-1]*A, -Vp[idx_V-Nx]*A, Vp[idx_V]*A, -Wp[idx_W-stride]*A, Wp[idx_W]*A, Up[idx_U]*A}
#define LOAD_FACE_FLUX_WEST double m_f[6] = { Up[idx_U]*A, -Vp[idx_V-Nx]*A, Vp[idx_V]*A, -Wp[idx_W-stride]*A, Wp[idx_W]*A, -Up[idx_U-1]*A}
#define LOAD_FACE_FLUX_NORTH double m_f[6] = {-Up[idx_U-1]*A, Up[idx_U]*A, -Vp[idx_V-Nx]*A, -Wp[idx_W-stride]*A, Wp[idx_W]*A, Vp[idx_V]*A}
#define LOAD_FACE_FLUX_SOUTH double m_f[6] = {-Up[idx_U-1]*A, Up[idx_U]*A, Vp[idx_V]*A, -Wp[idx_W-stride]*A, Wp[idx_W]*A, -Vp[idx_V-Nx]*A}

// edge x
#define LOAD_FACE_FLUX_SOUTH_BOTTOM double m_f[6] = {-Up[idx_U-1]*A, Up[idx_U]*A, Vp[idx_V]*A, Wp[idx_W]*A, -Vp[idx_V-Nx]*A, -Wp[idx_W-stride]*A}
#define LOAD_FACE_FLUX_SOUTH_TOP    double m_f[6] = {-Up[idx_U-1]*A, Up[idx_U]*A, Vp[idx_V]*A, -Wp[idx_W-stride]*A, -Vp[idx_V-Nx]*A, Wp[idx_W]*A}
#define LOAD_FACE_FLUX_NORTH_BOTTOM double m_f[6] = {-Up[idx_U-1]*A, Up[idx_U]*A, -Vp[idx_V-Nx]*A, Wp[idx_W]*A, Vp[idx_V]*A, -Wp[idx_W-stride]*A}
#define LOAD_FACE_FLUX_NORTH_TOP    double m_f[6] = {-Up[idx_U-1]*A, Up[idx_U]*A, -Vp[idx_V-Nx]*A, -Wp[idx_W-stride]*A, Vp[idx_V]*A, Wp[idx_W]*A}
// edge y
#define LOAD_FACE_FLUX_WEST_BOTTOM double m_f[6] = { Up[idx_U]*A, -Vp[idx_V-Nx]*A, Vp[idx_V]*A, Wp[idx_W]*A, -Up[idx_U-1]*A, -Wp[idx_W-stride]*A}
#define LOAD_FACE_FLUX_WEST_TOP    double m_f[6] = { Up[idx_U]*A, -Vp[idx_V-Nx]*A, Vp[idx_V]*A, -Wp[idx_W-stride]*A, -Up[idx_U-1]*A, Wp[idx_W]*A}
#define LOAD_FACE_FLUX_EAST_BOTTOM double m_f[6] = {-Up[idx_U-1]*A, -Vp[idx_V-Nx]*A, Vp[idx_V]*A, Wp[idx_W]*A, Up[idx_U]*A, -Wp[idx_W-stride]*A}
#define LOAD_FACE_FLUX_EAST_TOP    double m_f[6] = {-Up[idx_U-1]*A, -Vp[idx_V-Nx]*A, Vp[idx_V]*A, -Wp[idx_W-stride]*A, Up[idx_U]*A, Wp[idx_W]*A}

// edge z
#define LOAD_FACE_FLUX_WEST_SOUTH double m_f[6] = {Up[idx_U]*A, Vp[idx_V]*A, -Wp[idx_W-stride]*A, Wp[idx_W]*A, -Up[idx_U-1]*A, -Vp[idx_V-Nx]*A}
#define LOAD_FACE_FLUX_WEST_NORTH double m_f[6] = {Up[idx_U]*A, -Vp[idx_V-Nx]*A, -Wp[idx_W-stride]*A, Wp[idx_W]*A, -Up[idx_U-1]*A, Vp[idx_V]*A}
#define LOAD_FACE_FLUX_EAST_SOUTH double m_f[6] = {-Up[idx_U-1]*A, Vp[idx_V]*A, -Wp[idx_W-stride]*A, Wp[idx_W]*A, Up[idx_U]*A, -Vp[idx_V-Nx]*A}
#define LOAD_FACE_FLUX_EAST_NORTH double m_f[6] = {-Up[idx_U-1]*A, -Vp[idx_V-Nx]*A, -Wp[idx_W-stride]*A, Wp[idx_W]*A, Up[idx_U]*A, Vp[idx_V]*A}

#define LOAD_FACE_FLUX_WEST_SOUTH_BOTTOM double m_f[6] = { Up[idx_U]*A, Vp[idx_V]*A, Wp[idx_W]*A, -Up[idx_U-1]*A, -Vp[idx_V-Nx]*A, -Wp[idx_W-stride]*A}
#define LOAD_FACE_FLUX_WEST_SOUTH_TOP    double m_f[6] = { Up[idx_U]*A, Vp[idx_V]*A, -Wp[idx_W-stride]*A, -Up[idx_U-1]*A, -Vp[idx_V-Nx]*A, Wp[idx_W]*A}
#define LOAD_FACE_FLUX_WEST_NORTH_BOTTOM double m_f[6] = { Up[idx_U]*A, -Vp[idx_V-Nx]*A, Wp[idx_W]*A, -Up[idx_U-1]*A, Vp[idx_V]*A, -Wp[idx_W-stride]*A}
#define LOAD_FACE_FLUX_WEST_NORTH_TOP    double m_f[6] = { Up[idx_U]*A, -Vp[idx_V-Nx]*A, -Wp[idx_W-stride]*A, -Up[idx_U-1]*A, Vp[idx_V]*A, Wp[idx_W]*A}
#define LOAD_FACE_FLUX_EAST_SOUTH_BOTTOM double m_f[6] = {-Up[idx_U-1]*A, Vp[idx_V]*A, Wp[idx_W]*A, Up[idx_U]*A, -Vp[idx_V-Nx]*A, -Wp[idx_W-stride]*A}
#define LOAD_FACE_FLUX_EAST_SOUTH_TOP    double m_f[6] = {-Up[idx_U-1]*A, Vp[idx_V]*A, -Wp[idx_W-stride]*A, Up[idx_U]*A, -Vp[idx_V-Nx]*A, Wp[idx_W]*A}
#define LOAD_FACE_FLUX_EAST_NORTH_BOTTOM double m_f[6] = {-Up[idx_U-1]*A, -Vp[idx_V-Nx]*A, Wp[idx_W]*A, Up[idx_U]*A, Vp[idx_V]*A, -Wp[idx_W-stride]*A}
#define LOAD_FACE_FLUX_EAST_NORTH_TOP    double m_f[6] = {-Up[idx_U-1]*A, -Vp[idx_V-Nx]*A, -Wp[idx_W-stride]*A, Up[idx_U]*A, Vp[idx_V]*A, Wp[idx_W]*A}

```

# header\macros_index_kernel.h

```h
#define stride (Nx*Ny)
// #define OFFSETS5(FACE, Nx, Ny) OFFSETS_##FACE(Nx, Ny)
#define OFFSETS_BOTTOM int offsets[5] = { -1, 1, -Nx, Nx,  stride }
#define OFFSETS_TOP int    offsets[5] = { -1, 1, -Nx, Nx, -stride }
#define OFFSETS_EAST int offsets[5] = { -1, -Nx,  Nx, -stride, stride }
#define OFFSETS_WEST int offsets[5] = {  1, -Nx,  Nx, -stride, stride }
#define OFFSETS_NORTH int offsets[5] = { -1,  1, -Nx, -stride,  stride }
#define OFFSETS_SOUTH int offsets[5] = { -1,  1,  Nx, -stride,  stride }

// #define OFFSETS4(FACE1,FACE2,Nx,Ny) OFFSETS_##FACE1##_##FACE2(Nx, Ny)se
// edge x
#define OFFSETS_SOUTH_BOTTOM int offsets[4] = { -1,  1,  Nx,  stride }
#define OFFSETS_SOUTH_TOP    int offsets[4] = { -1,  1,  Nx, -stride }
#define OFFSETS_NORTH_BOTTOM int offsets[4] = { -1,  1, -Nx,  stride }
#define OFFSETS_NORTH_TOP    int offsets[4] = { -1,  1, -Nx, -stride }

// edge y
#define OFFSETS_WEST_BOTTOM int offsets[4] = { 1, -Nx, Nx,  stride }
#define OFFSETS_WEST_TOP    int offsets[4] = { 1, -Nx, Nx, -stride }
#define OFFSETS_EAST_BOTTOM int offsets[4] = {-1, -Nx, Nx,  stride }
#define OFFSETS_EAST_TOP    int offsets[4] = {-1, -Nx, Nx, -stride }

// edge z
#define OFFSETS_WEST_SOUTH int offsets[4] = {  1,  Nx, -stride, stride }
#define OFFSETS_WEST_NORTH int offsets[4] = {  1, -Nx, -stride, stride }
#define OFFSETS_EAST_SOUTH int offsets[4] = { -1,  Nx, -stride, stride }
#define OFFSETS_EAST_NORTH int offsets[4] = { -1, -Nx, -stride, stride }

// #define OFFSETS3(FACE1,FACE2,FACE3,Nx,Ny) OFFSETS_##FACE1##_##FACE2##_##FACE3(Nx, Ny)
// vertex
#define OFFSETS_WEST_SOUTH_BOTTOM int offsets[3] = { 1,  Nx,  stride }
#define OFFSETS_WEST_SOUTH_TOP    int offsets[3] = { 1,  Nx, -stride }
#define OFFSETS_WEST_NORTH_BOTTOM int offsets[3] = { 1, -Nx,  stride }
#define OFFSETS_WEST_NORTH_TOP    int offsets[3] = { 1, -Nx, -stride }
#define OFFSETS_EAST_SOUTH_BOTTOM int offsets[3] = {-1,  Nx,  stride }
#define OFFSETS_EAST_SOUTH_TOP    int offsets[3] = {-1,  Nx, -stride }
#define OFFSETS_EAST_NORTH_BOTTOM int offsets[3] = {-1, -Nx,  stride }
#define OFFSETS_EAST_NORTH_TOP    int offsets[3] = {-1, -Nx, -stride }

#define PERIODIC_BOTTOM ((Nz-1)*stride)
#define PERIODIC_TOP (-(Nz-1)*stride)
#define PERIODIC_WEST (Nx-1)
#define PERIODIC_EAST (-(Nx-1))
#define PERIODIC_SOUTH ((Ny-1)*Nx)
#define PERIODIC_NORTH (-(Ny-1)*Nx)

#define COMPUTE_INDEX_BOTTOM \
int ix = threadIdx.x + blockIdx.x*blockDim.x; \
int iy = threadIdx.y + blockIdx.y*blockDim.y; \
if (ix >= Nx-2 || iy >= Ny-2) return; \
int iz=0; \
int in_idx = (ix+1) + (iy+1)*Nx + (iz)*stride;

#define COMPUTE_INDEX_TOP \
int ix = threadIdx.x + blockIdx.x*blockDim.x; \
int iy = threadIdx.y + blockIdx.y*blockDim.y; \
if (ix >= Nx-2 || iy >= Ny-2) return; \
int iz=Nz-1; \
int in_idx = (ix+1) + (iy+1)*Nx + (iz)*stride;

#define COMPUTE_INDEX_SOUTH \
int ix = threadIdx.x + blockIdx.x*blockDim.x; \
int iz = threadIdx.y + blockIdx.y*blockDim.y; \
if (ix >= Nx-2 || iz >= Nz-2) return; \
int iy=0; \
int in_idx = (ix+1) + (iy)*Nx + (iz+1)*stride;

#define COMPUTE_INDEX_NORTH \
int ix = threadIdx.x + blockIdx.x*blockDim.x; \
int iz = threadIdx.y + blockIdx.y*blockDim.y; \
if (ix >= Nx-2 || iz >= Nz-2) return; \
int iy=Ny-1; \
int in_idx = (ix+1) + (iy)*Nx + (iz+1)*stride;

#define COMPUTE_INDEX_WEST \
int iy = threadIdx.x + blockIdx.x*blockDim.x; \
int iz = threadIdx.y + blockIdx.y*blockDim.y; \
if (iy >= Ny-2 || iz >= Nz-2) return; \
int ix=0; \
int in_idx = (ix) + (iy+1)*Nx + (iz+1)*stride;

#define COMPUTE_INDEX_EAST \
int iy = threadIdx.x + blockIdx.x*blockDim.x; \
int iz = threadIdx.y + blockIdx.y*blockDim.y; \
if (iy >= Ny-2 || iz >= Nz-2) return; \
int ix=Nx-1; \
int in_idx = (ix) + (iy+1)*Nx + (iz+1)*stride;

#define COMPUTE_INDEX_SOUTH_BOTTOM \
int ix = threadIdx.x + blockIdx.x*blockDim.x; \
if (ix >= Nx-2) return; \
int iy = 0; \
int iz = 0; \
int in_idx = (ix+1) + iy*Nx + iz*stride;

#define COMPUTE_INDEX_SOUTH_TOP \
int ix = threadIdx.x + blockIdx.x*blockDim.x; \
if (ix >= Nx-2) return; \
int iy = 0; \
int iz = Nz-1; \
int in_idx = (ix+1) + iy*Nx + iz*stride;

#define COMPUTE_INDEX_NORTH_BOTTOM \
int ix = threadIdx.x + blockIdx.x*blockDim.x; \
if (ix >= Nx-2) return; \
int iy = Ny-1; \
int iz = 0; \
int in_idx = (ix+1) + iy*Nx + iz*stride;

#define COMPUTE_INDEX_NORTH_TOP \
int ix = threadIdx.x + blockIdx.x*blockDim.x; \
if (ix >= Nx-2) return; \
int iy = Ny-1; \
int iz = Nz-1; \
int in_idx = (ix+1) + iy*Nx + iz*stride;

#define COMPUTE_INDEX_WEST_BOTTOM \
int iy = threadIdx.x + blockIdx.x*blockDim.x; \
if (iy >= Ny-2) return; \
int ix = 0; \
int iz = 0; \
int in_idx = ix + (iy+1)*Nx + iz*stride;

#define COMPUTE_INDEX_WEST_TOP \
int iy = threadIdx.x + blockIdx.x*blockDim.x; \
if (iy >= Ny-2) return; \
int ix = 0; \
int iz = Nz-1; \
int in_idx = ix + (iy+1)*Nx + iz*stride;

#define COMPUTE_INDEX_EAST_BOTTOM \
int iy = threadIdx.x + blockIdx.x*blockDim.x; \
if (iy >= Ny-2) return; \
int ix = Nx-1; \
int iz = 0; \
int in_idx = ix + (iy+1)*Nx + iz*stride;

#define COMPUTE_INDEX_EAST_TOP \
int iy = threadIdx.x + blockIdx.x*blockDim.x; \
if (iy >= Ny-2) return; \
int ix = Nx-1; \
int iz = Nz-1; \
int in_idx = ix + (iy+1)*Nx + iz*stride;

#define COMPUTE_INDEX_WEST_SOUTH \
int iz = threadIdx.x + blockIdx.x*blockDim.x; \
if (iz >= Nz-2) return; \
int ix = 0; \
int iy = 0; \
int in_idx = ix + iy*Nx + (iz+1)*stride;

#define COMPUTE_INDEX_EAST_SOUTH \
int iz = threadIdx.x + blockIdx.x*blockDim.x; \
if (iz >= Nz-2) return; \
int ix = Nx-1; \
int iy = 0; \
int in_idx = ix + iy*Nx + (iz+1)*stride;

#define COMPUTE_INDEX_WEST_NORTH \
int iz = threadIdx.x + blockIdx.x*blockDim.x; \
if (iz >= Nz-2) return; \
int ix = 0; \
int iy = Ny-1; \
int in_idx = ix + iy*Nx + (iz+1)*stride;

#define COMPUTE_INDEX_EAST_NORTH \
int iz = threadIdx.x + blockIdx.x*blockDim.x; \
if (iz >= Nz-2) return; \
int ix = Nx-1; \
int iy = Ny-1; \
int in_idx = ix + iy*Nx + (iz+1)*stride;

#define COMPUTE_INDEX_WEST_SOUTH_BOTTOM \
int ix = 0; \
int iy = 0; \
int iz = 0; \
int in_idx = ix + iy*Nx + iz*stride;

#define COMPUTE_INDEX_WEST_SOUTH_TOP \
int ix = 0; \
int iy = 0; \
int iz = Nz-1; \
int in_idx = ix + iy*Nx + iz*stride;

#define COMPUTE_INDEX_WEST_NORTH_BOTTOM \
int ix = 0; \
int iy = Ny-1; \
int iz = 0; \
int in_idx = ix + iy*Nx + iz*stride;

#define COMPUTE_INDEX_WEST_NORTH_TOP \
int ix = 0; \
int iy = Ny-1; \
int iz = Nz-1; \
int in_idx = ix + iy*Nx + iz*stride;

#define COMPUTE_INDEX_EAST_SOUTH_BOTTOM \
int ix = Nx-1; \
int iy = 0; \
int iz = 0; \
int in_idx = ix + iy*Nx + iz*stride;

#define COMPUTE_INDEX_EAST_SOUTH_TOP \
int ix = Nx-1; \
int iy = 0; \
int iz = Nz-1; \
int in_idx = ix + iy*Nx + iz*stride;

#define COMPUTE_INDEX_EAST_NORTH_BOTTOM \
int ix = Nx-1; \
int iy = Ny-1; \
int iz = 0; \
int in_idx = ix + iy*Nx + iz*stride;

#define COMPUTE_INDEX_EAST_NORTH_TOP \
int ix = Nx-1; \
int iy = Ny-1; \
int iz = Nz-1; \
int in_idx = ix + iy*Nx + iz*stride;

#define GRIDBLOCK_BOTTOM dim3(grid.x,grid.y),dim3(block.x,block.y)
#define GRIDBLOCK_TOP GRIDBLOCK_BOTTOM
#define GRIDBLOCK_SOUTH dim3(grid.x,grid.z),dim3(block.x,block.z)
#define GRIDBLOCK_NORTH GRIDBLOCK_SOUTH
#define GRIDBLOCK_WEST dim3(grid.y,grid.z),dim3(block.y,block.z)
#define GRIDBLOCK_EAST GRIDBLOCK_WEST

#define GRIDBLOCK_SOUTH_BOTTOM grid.x, block.x
#define GRIDBLOCK_NORTH_BOTTOM GRIDBLOCK_SOUTH_BOTTOM
#define GRIDBLOCK_SOUTH_TOP GRIDBLOCK_SOUTH_BOTTOM
#define GRIDBLOCK_NORTH_TOP GRIDBLOCK_SOUTH_BOTTOM
#define GRIDBLOCK_WEST_SOUTH grid.z, block.z
#define GRIDBLOCK_WEST_NORTH GRIDBLOCK_WEST_SOUTH
#define GRIDBLOCK_EAST_SOUTH GRIDBLOCK_WEST_SOUTH
#define GRIDBLOCK_EAST_NORTH GRIDBLOCK_WEST_SOUTH
#define GRIDBLOCK_WEST_BOTTOM grid.y, block.y
#define GRIDBLOCK_WEST_TOP GRIDBLOCK_WEST_SOUTH
#define GRIDBLOCK_EAST_BOTTOM GRIDBLOCK_WEST_SOUTH
#define GRIDBLOCK_EAST_TOP GRIDBLOCK_WEST_SOUTH
```

# header\macros_index_mf_par2.h

```h
#define COMPUTE_INDEX_NORMAL_VELOCITY_BOTTOM \
int idx_U = (ix+1+1) + (iy+1)*(Nx+1) + (iz)*(Nx+1)*(Ny+1); \
int idx_V = (ix+1)   + (iy+1+1)*(Nx+1) + (iz)*(Nx+1)*(Ny+1); \
int idx_W = (ix+1)   + (iy+1)*(Nx+1)   + (iz+1)*(Nx+1)*(Ny+1);
#define COMPUTE_INDEX_NORMAL_VELOCITY_TOP COMPUTE_INDEX_NORMAL_VELOCITY_BOTTOM

#define COMPUTE_INDEX_NORMAL_VELOCITY_SOUTH \
int idx_U = (ix+1+1) + (iy)*(Nx+1) + (iz+1)*(Nx+1)*(Ny+1); \
int idx_V = (ix+1)   + (iy+1)*(Nx+1) + (iz+1)*(Nx+1)*(Ny+1); \
int idx_W = (ix+1)   + (iy)*(Nx+1)  + (iz+1+1)*(Nx+1)*(Ny+1);
#define COMPUTE_INDEX_NORMAL_VELOCITY_NORTH COMPUTE_INDEX_NORMAL_VELOCITY_SOUTH

#define COMPUTE_INDEX_NORMAL_VELOCITY_WEST \
int idx_U = (ix+1) + (iy+1)*(Nx+1) + (iz+1)*(Nx+1)*(Ny+1); \
int idx_V = (ix)   + (iy+1+1)*(Nx+1) + (iz+1)*(Nx+1)*(Ny+1); \
int idx_W = (ix)   + (iy+1)*(Nx+1)   + (iz+1+1)*(Nx+1)*(Ny+1);
#define COMPUTE_INDEX_NORMAL_VELOCITY_EAST COMPUTE_INDEX_NORMAL_VELOCITY_WEST

#define COMPUTE_INDEX_NORMAL_VELOCITY_SOUTH_BOTTOM \
int idx_U = (ix+1+1) + (iy)*(Nx+1) + (iz)*(Nx+1)*(Ny+1); \
int idx_V = (ix+1)   + (iy+1)*(Nx+1) + (iz)*(Nx+1)*(Ny+1); \
int idx_W = (ix+1)   + (iy)*(Nx+1)   + (iz+1)*(Nx+1)*(Ny+1);
#define COMPUTE_INDEX_NORMAL_VELOCITY_SOUTH_TOP COMPUTE_INDEX_NORMAL_VELOCITY_SOUTH_BOTTOM
#define COMPUTE_INDEX_NORMAL_VELOCITY_NORTH_BOTTOM COMPUTE_INDEX_NORMAL_VELOCITY_SOUTH_BOTTOM
#define COMPUTE_INDEX_NORMAL_VELOCITY_NORTH_TOP COMPUTE_INDEX_NORMAL_VELOCITY_SOUTH_BOTTOM

#define COMPUTE_INDEX_NORMAL_VELOCITY_WEST_BOTTOM \
int idx_U = (ix+1) + (iy+1)*(Nx+1) + (iz)*(Nx+1)*(Ny+1); \
int idx_V = (ix)   + (iy+1+1)*(Nx+1) + (iz)*(Nx+1)*(Ny+1); \
int idx_W = (ix)   + (iy+1)*(Nx+1)   + (iz+1)*(Nx+1)*(Ny+1);
#define COMPUTE_INDEX_NORMAL_VELOCITY_WEST_TOP COMPUTE_INDEX_NORMAL_VELOCITY_WEST_BOTTOM
#define COMPUTE_INDEX_NORMAL_VELOCITY_EAST_BOTTOM COMPUTE_INDEX_NORMAL_VELOCITY_WEST_BOTTOM
#define COMPUTE_INDEX_NORMAL_VELOCITY_EAST_TOP COMPUTE_INDEX_NORMAL_VELOCITY_WEST_BOTTOM

#define COMPUTE_INDEX_NORMAL_VELOCITY_WEST_SOUTH \
int idx_U = (ix+1) + (iy)*(Nx+1) + (iz+1)*(Nx+1)*(Ny+1); \
int idx_V = (ix)   + (iy+1)*(Nx+1) + (iz+1)*(Nx+1)*(Ny+1); \
int idx_W = (ix)   + (iy)*(Nx+1)   + (iz+1+1)*(Nx+1)*(Ny+1);
#define COMPUTE_INDEX_NORMAL_VELOCITY_EAST_SOUTH COMPUTE_INDEX_NORMAL_VELOCITY_WEST_SOUTH
#define COMPUTE_INDEX_NORMAL_VELOCITY_WEST_NORTH COMPUTE_INDEX_NORMAL_VELOCITY_WEST_SOUTH
#define COMPUTE_INDEX_NORMAL_VELOCITY_EAST_NORTH COMPUTE_INDEX_NORMAL_VELOCITY_WEST_SOUTH

#define COMPUTE_INDEX_NORMAL_VELOCITY_WEST_SOUTH_BOTTOM \
int idx_U = (ix+1) + (iy)*(Nx+1) + (iz)*(Nx+1)*(Ny+1); \
int idx_V = (ix)   + (iy+1)*(Nx+1) + (iz)*(Nx+1)*(Ny+1); \
int idx_W = (ix)   + (iy)*(Nx+1)   + (iz+1)*(Nx+1)*(Ny+1);
#define COMPUTE_INDEX_NORMAL_VELOCITY_WEST_SOUTH_TOP COMPUTE_INDEX_NORMAL_VELOCITY_WEST_SOUTH_BOTTOM
#define COMPUTE_INDEX_NORMAL_VELOCITY_WEST_NORTH_BOTTOM COMPUTE_INDEX_NORMAL_VELOCITY_WEST_SOUTH_BOTTOM
#define COMPUTE_INDEX_NORMAL_VELOCITY_WEST_NORTH_TOP COMPUTE_INDEX_NORMAL_VELOCITY_WEST_SOUTH_BOTTOM
#define COMPUTE_INDEX_NORMAL_VELOCITY_EAST_SOUTH_BOTTOM COMPUTE_INDEX_NORMAL_VELOCITY_WEST_SOUTH_BOTTOM
#define COMPUTE_INDEX_NORMAL_VELOCITY_EAST_SOUTH_TOP COMPUTE_INDEX_NORMAL_VELOCITY_WEST_SOUTH_BOTTOM
#define COMPUTE_INDEX_NORMAL_VELOCITY_EAST_NORTH_BOTTOM COMPUTE_INDEX_NORMAL_VELOCITY_WEST_SOUTH_BOTTOM
#define COMPUTE_INDEX_NORMAL_VELOCITY_EAST_NORTH_TOP COMPUTE_INDEX_NORMAL_VELOCITY_WEST_SOUTH_BOTTOM

// The order must match with the OFFSETS array as they are iterated in the same loop (see macros_indez_kernel.h)
// #define OFFSETS_BOTTOM int offsets[5] = { -1, 1, -Nx, Nx,  stride }
// #define OFFSETS_TOP int    offsets[5] = { -1, 1, -Nx, Nx, -stride }
// #define OFFSETS_EAST int offsets[5] = { -1, -Nx,  Nx, -stride, stride }
// #define OFFSETS_WEST int offsets[5] = {  1, -Nx,  Nx, -stride, stride }
// #define OFFSETS_NORTH int offsets[5] = { -1,  1, -Nx, -stride,  stride }
// #define OFFSETS_SOUTH int offsets[5] = { -1,  1,  Nx, -stride,  stride }
#define LOAD_FACE_FLUX_BOTTOM double m_f[6] = {-Up[idx_U-1]*A, Up[idx_U]*A, -Vp[idx_V-Nx]*A, Vp[idx_V]*A, Wp[idx_W]*A, -Wp[idx_W-stride]*A}
#define LOAD_FACE_FLUX_TOP double m_f[6] = {-Up[idx_U-1]*A, Up[idx_U]*A, -Vp[idx_V-Nx]*A, Vp[idx_V]*A, -Wp[idx_W-stride]*A, Wp[idx_W]*A}
#define LOAD_FACE_FLUX_EAST double m_f[6] = {-Up[idx_U-1]*A, -Vp[idx_V-Nx]*A, Vp[idx_V]*A, -Wp[idx_W-stride]*A, Wp[idx_W]*A, Up[idx_U]*A}
#define LOAD_FACE_FLUX_WEST double m_f[6] = { Up[idx_U]*A, -Vp[idx_V-Nx]*A, Vp[idx_V]*A, -Wp[idx_W-stride]*A, Wp[idx_W]*A, -Up[idx_U-1]*A}
#define LOAD_FACE_FLUX_NORTH double m_f[6] = {-Up[idx_U-1]*A, Up[idx_U]*A, -Vp[idx_V-Nx]*A, -Wp[idx_W-stride]*A, Wp[idx_W]*A, Vp[idx_V]*A}
#define LOAD_FACE_FLUX_SOUTH double m_f[6] = {-Up[idx_U-1]*A, Up[idx_U]*A, Vp[idx_V]*A, -Wp[idx_W-stride]*A, Wp[idx_W]*A, -Vp[idx_V-Nx]*A}

// edge x
#define LOAD_FACE_FLUX_SOUTH_BOTTOM double m_f[6] = {-Up[idx_U-1]*A, Up[idx_U]*A, Vp[idx_V]*A, Wp[idx_W]*A, -Vp[idx_V-Nx]*A, -Wp[idx_W-stride]*A}
#define LOAD_FACE_FLUX_SOUTH_TOP    double m_f[6] = {-Up[idx_U-1]*A, Up[idx_U]*A, Vp[idx_V]*A, -Wp[idx_W-stride]*A, -Vp[idx_V-Nx]*A, Wp[idx_W]*A}
#define LOAD_FACE_FLUX_NORTH_BOTTOM double m_f[6] = {-Up[idx_U-1]*A, Up[idx_U]*A, -Vp[idx_V-Nx]*A, Wp[idx_W]*A, Vp[idx_V]*A, -Wp[idx_W-stride]*A}
#define LOAD_FACE_FLUX_NORTH_TOP    double m_f[6] = {-Up[idx_U-1]*A, Up[idx_U]*A, -Vp[idx_V-Nx]*A, -Wp[idx_W-stride]*A, Vp[idx_V]*A, Wp[idx_W]*A}
// edge y
#define LOAD_FACE_FLUX_WEST_BOTTOM double m_f[6] = { Up[idx_U]*A, -Vp[idx_V-Nx]*A, Vp[idx_V]*A, Wp[idx_W]*A, -Up[idx_U-1]*A, -Wp[idx_W-stride]*A}
#define LOAD_FACE_FLUX_WEST_TOP    double m_f[6] = { Up[idx_U]*A, -Vp[idx_V-Nx]*A, Vp[idx_V]*A, -Wp[idx_W-stride]*A, -Up[idx_U-1]*A, Wp[idx_W]*A}
#define LOAD_FACE_FLUX_EAST_BOTTOM double m_f[6] = {-Up[idx_U-1]*A, -Vp[idx_V-Nx]*A, Vp[idx_V]*A, Wp[idx_W]*A, Up[idx_U]*A, -Wp[idx_W-stride]*A}
#define LOAD_FACE_FLUX_EAST_TOP    double m_f[6] = {-Up[idx_U-1]*A, -Vp[idx_V-Nx]*A, Vp[idx_V]*A, -Wp[idx_W-stride]*A, Up[idx_U]*A, Wp[idx_W]*A}

// edge z
#define LOAD_FACE_FLUX_WEST_SOUTH double m_f[6] = {Up[idx_U]*A, Vp[idx_V]*A, -Wp[idx_W-stride]*A, Wp[idx_W]*A, -Up[idx_U-1]*A, -Vp[idx_V-Nx]*A}
#define LOAD_FACE_FLUX_WEST_NORTH double m_f[6] = {Up[idx_U]*A, -Vp[idx_V-Nx]*A, -Wp[idx_W-stride]*A, Wp[idx_W]*A, -Up[idx_U-1]*A, Vp[idx_V]*A}
#define LOAD_FACE_FLUX_EAST_SOUTH double m_f[6] = {-Up[idx_U-1]*A, Vp[idx_V]*A, -Wp[idx_W-stride]*A, Wp[idx_W]*A, Up[idx_U]*A, -Vp[idx_V-Nx]*A}
#define LOAD_FACE_FLUX_EAST_NORTH double m_f[6] = {-Up[idx_U-1]*A, -Vp[idx_V-Nx]*A, -Wp[idx_W-stride]*A, Wp[idx_W]*A, Up[idx_U]*A, Vp[idx_V]*A}

#define LOAD_FACE_FLUX_WEST_SOUTH_BOTTOM double m_f[6] = { Up[idx_U]*A, Vp[idx_V]*A, Wp[idx_W]*A, -Up[idx_U-1]*A, -Vp[idx_V-Nx]*A, -Wp[idx_W-stride]*A}
#define LOAD_FACE_FLUX_WEST_SOUTH_TOP    double m_f[6] = { Up[idx_U]*A, Vp[idx_V]*A, -Wp[idx_W-stride]*A, -Up[idx_U-1]*A, -Vp[idx_V-Nx]*A, Wp[idx_W]*A}
#define LOAD_FACE_FLUX_WEST_NORTH_BOTTOM double m_f[6] = { Up[idx_U]*A, -Vp[idx_V-Nx]*A, Wp[idx_W]*A, -Up[idx_U-1]*A, Vp[idx_V]*A, -Wp[idx_W-stride]*A}
#define LOAD_FACE_FLUX_WEST_NORTH_TOP    double m_f[6] = { Up[idx_U]*A, -Vp[idx_V-Nx]*A, -Wp[idx_W-stride]*A, -Up[idx_U-1]*A, Vp[idx_V]*A, Wp[idx_W]*A}
#define LOAD_FACE_FLUX_EAST_SOUTH_BOTTOM double m_f[6] = {-Up[idx_U-1]*A, Vp[idx_V]*A, Wp[idx_W]*A, Up[idx_U]*A, -Vp[idx_V-Nx]*A, -Wp[idx_W-stride]*A}
#define LOAD_FACE_FLUX_EAST_SOUTH_TOP    double m_f[6] = {-Up[idx_U-1]*A, Vp[idx_V]*A, -Wp[idx_W-stride]*A, Up[idx_U]*A, -Vp[idx_V-Nx]*A, Wp[idx_W]*A}
#define LOAD_FACE_FLUX_EAST_NORTH_BOTTOM double m_f[6] = {-Up[idx_U-1]*A, -Vp[idx_V-Nx]*A, Wp[idx_W]*A, Up[idx_U]*A, Vp[idx_V]*A, -Wp[idx_W-stride]*A}
#define LOAD_FACE_FLUX_EAST_NORTH_TOP    double m_f[6] = {-Up[idx_U-1]*A, -Vp[idx_V-Nx]*A, -Wp[idx_W-stride]*A, Up[idx_U]*A, Vp[idx_V]*A, Wp[idx_W]*A}

```

# header\macros_momentos.h

```h
#define COMPUTE_MOMENTO_BOTTOM \
momento1x[in_idx] = (float)((ix+1.5)*h)  *  (float)phiC  *(float)(h*h*h); \
momento1y[in_idx] = (float)((iy+1.5)*h ) *  (float)phiC  *(float)(h*h*h); \
momento1z[in_idx] = (float)((iz+0.5)*h)  *  (float)phiC  *(float)(h*h*h); \
momento2x[in_idx] = (float)powf((ix+1.5f)*(float)h,2.0f)*(float)phiC*(float)(h*h*h); \
momento2y[in_idx] = (float)powf((iy+1.5f)*(float)h,2.0f)*(float)phiC*(float)(h*h*h); \
momento2z[in_idx] = (float)powf((iz+0.5f)*(float)h,2.0f) *(float)phiC*(float)(h*h*h); \
C_float[in_idx] = (float)phiC;

#define COMPUTE_MOMENTO_TOP \
momento1x[in_idx] = (float)((ix+1.5)*h)  *  (float)phiC  *(float)(h*h*h); \
momento1y[in_idx] = (float)((iy+1.5)*h ) *  (float)phiC  *(float)(h*h*h); \
momento1z[in_idx] = (float)((iz+0.5)*h)  *  (float)phiC  *(float)(h*h*h); \
momento2x[in_idx] = (float)powf((ix+1.5f)*(float)h,2.0f)*(float)phiC*(float)(h*h*h); \
momento2y[in_idx] = (float)powf((iy+1.5f)*(float)h,2.0f)*(float)phiC*(float)(h*h*h); \
momento2z[in_idx] = (float)powf((iz+0.5f)*(float)h,2.0f) *(float)phiC*(float)(h*h*h); \
C_float[in_idx] = (float)phiC;

#define COMPUTE_MOMENTO_SOUTH \
momento1x[in_idx] = (float)((ix+1.5)*h)  *  (float)phiC  *(float)(h*h*h); \
momento1y[in_idx] = (float)((iy+0.5)*h ) *  (float)phiC  *(float)(h*h*h); \
momento1z[in_idx] = (float)((iz+1.5)*h)  *  (float)phiC  *(float)(h*h*h); \
momento2x[in_idx] = (float)powf((ix+1.5f)*(float)h,2.0f)*(float)phiC*(float)(h*h*h); \
momento2y[in_idx] = (float)powf((iy+0.5f)*(float)h,2.0f)*(float)phiC*(float)(h*h*h); \
momento2z[in_idx] = (float)powf((iz+1.5f)*(float)h,2.0f) *(float)phiC*(float)(h*h*h); \
C_float[in_idx] = (float)phiC;

#define COMPUTE_MOMENTO_NORTH \
momento1x[in_idx] = (float)((ix+1.5)*h)  *  (float)phiC  *(float)(h*h*h); \
momento1y[in_idx] = (float)((iy+0.5)*h ) *  (float)phiC  *(float)(h*h*h); \
momento1z[in_idx] = (float)((iz+1.5)*h)  *  (float)phiC  *(float)(h*h*h); \
momento2x[in_idx] = (float)powf((ix+1.5f)*(float)h,2.0f)*(float)phiC*(float)(h*h*h); \
momento2y[in_idx] = (float)powf((iy+0.5f)*(float)h,2.0f)*(float)phiC*(float)(h*h*h); \
momento2z[in_idx] = (float)powf((iz+1.5f)*(float)h,2.0f) *(float)phiC*(float)(h*h*h); \
C_float[in_idx] = (float)phiC;

#define COMPUTE_MOMENTO_WEST \
momento1x[in_idx] = (float)((ix+0.5)*h)  *  (float)phiC  *(float)(h*h*h); \
momento1y[in_idx] = (float)((iy+1.5)*h ) *  (float)phiC  *(float)(h*h*h); \
momento1z[in_idx] = (float)((iz+1.5)*h)  *  (float)phiC  *(float)(h*h*h); \
momento2x[in_idx] = (float)powf((ix+0.5f)*(float)h,2.0f)*(float)phiC*(float)(h*h*h); \
momento2y[in_idx] = (float)powf((iy+1.5f)*(float)h,2.0f)*(float)phiC*(float)(h*h*h); \
momento2z[in_idx] = (float)powf((iz+1.5f)*(float)h,2.0f) *(float)phiC*(float)(h*h*h); \
C_float[in_idx] = (float)phiC;

#define COMPUTE_MOMENTO_EAST \
momento1x[in_idx] = (float)((ix+0.5)*h)  *  (float)phiC  *(float)(h*h*h); \
momento1y[in_idx] = (float)((iy+1.5)*h ) *  (float)phiC  *(float)(h*h*h); \
momento1z[in_idx] = (float)((iz+1.5)*h)  *  (float)phiC  *(float)(h*h*h); \
momento2x[in_idx] = (float)powf((ix+0.5f)*(float)h,2.0f)*(float)phiC*(float)(h*h*h); \
momento2y[in_idx] = (float)powf((iy+1.5f)*(float)h,2.0f)*(float)phiC*(float)(h*h*h); \
momento2z[in_idx] = (float)powf((iz+1.5f)*(float)h,2.0f) *(float)phiC*(float)(h*h*h); \
C_float[in_idx] = (float)phiC;

#define COMPUTE_MOMENTO_SOUTH_BOTTOM \
momento1x[in_idx] = (float)((ix+1.5)*h)  *  (float)phiC  *(float)(h*h*h); \
momento1y[in_idx] = (float)((iy+0.5)*h ) *  (float)phiC  *(float)(h*h*h); \
momento1z[in_idx] = (float)((iz+0.5)*h)  *  (float)phiC  *(float)(h*h*h); \
momento2x[in_idx] = (float)powf((ix+1.5f)*(float)h,2.0f)*(float)phiC*(float)(h*h*h); \
momento2y[in_idx] = (float)powf((iy+0.5f)*(float)h,2.0f)*(float)phiC*(float)(h*h*h); \
momento2z[in_idx] = (float)powf((iz+0.5f)*(float)h,2.0f) *(float)phiC*(float)(h*h*h); \
C_float[in_idx] = (float)phiC;

#define COMPUTE_MOMENTO_SOUTH_TOP \
momento1x[in_idx] = (float)((ix+1.5)*h)  *  (float)phiC  *(float)(h*h*h); \
momento1y[in_idx] = (float)((iy+0.5)*h ) *  (float)phiC  *(float)(h*h*h); \
momento1z[in_idx] = (float)((iz+0.5)*h)  *  (float)phiC  *(float)(h*h*h); \
momento2x[in_idx] = (float)powf((ix+1.5f)*(float)h,2.0f)*(float)phiC*(float)(h*h*h); \
momento2y[in_idx] = (float)powf((iy+0.5f)*(float)h,2.0f)*(float)phiC*(float)(h*h*h); \
momento2z[in_idx] = (float)powf((iz+0.5f)*(float)h,2.0f) *(float)phiC*(float)(h*h*h); \
C_float[in_idx] = (float)phiC;

#define COMPUTE_MOMENTO_NORTH_BOTTOM \
momento1x[in_idx] = (float)((ix+1.5)*h)  *  (float)phiC  *(float)(h*h*h); \
momento1y[in_idx] = (float)((iy+0.5)*h ) *  (float)phiC  *(float)(h*h*h); \
momento1z[in_idx] = (float)((iz+0.5)*h)  *  (float)phiC  *(float)(h*h*h); \
momento2x[in_idx] = (float)powf((ix+1.5f)*(float)h,2.0f)*(float)phiC*(float)(h*h*h); \
momento2y[in_idx] = (float)powf((iy+0.5f)*(float)h,2.0f)*(float)phiC*(float)(h*h*h); \
momento2z[in_idx] = (float)powf((iz+0.5f)*(float)h,2.0f) *(float)phiC*(float)(h*h*h); \
C_float[in_idx] = (float)phiC;

#define COMPUTE_MOMENTO_NORTH_TOP \
momento1x[in_idx] = (float)((ix+1.5)*h)  *  (float)phiC  *(float)(h*h*h); \
momento1y[in_idx] = (float)((iy+0.5)*h ) *  (float)phiC  *(float)(h*h*h); \
momento1z[in_idx] = (float)((iz+0.5)*h)  *  (float)phiC  *(float)(h*h*h); \
momento2x[in_idx] = (float)powf((ix+1.5f)*(float)h,2.0f)*(float)phiC*(float)(h*h*h); \
momento2y[in_idx] = (float)powf((iy+0.5f)*(float)h,2.0f)*(float)phiC*(float)(h*h*h); \
momento2z[in_idx] = (float)powf((iz+0.5f)*(float)h,2.0f) *(float)phiC*(float)(h*h*h); \
C_float[in_idx] = (float)phiC;

#define COMPUTE_MOMENTO_WEST_BOTTOM \
momento1x[in_idx] = (float)((ix+0.5)*h)  *  (float)phiC  *(float)(h*h*h); \
momento1y[in_idx] = (float)((iy+1.5)*h ) *  (float)phiC  *(float)(h*h*h); \
momento1z[in_idx] = (float)((iz+0.5)*h)  *  (float)phiC  *(float)(h*h*h); \
momento2x[in_idx] = (float)powf((ix+0.5f)*(float)h,2.0f)*(float)phiC*(float)(h*h*h); \
momento2y[in_idx] = (float)powf((iy+1.5f)*(float)h,2.0f)*(float)phiC*(float)(h*h*h); \
momento2z[in_idx] = (float)powf((iz+0.5f)*(float)h,2.0f) *(float)phiC*(float)(h*h*h); \
C_float[in_idx] = (float)phiC;

#define COMPUTE_MOMENTO_WEST_TOP \
momento1x[in_idx] = (float)((ix+0.5)*h)  *  (float)phiC  *(float)(h*h*h); \
momento1y[in_idx] = (float)((iy+1.5)*h ) *  (float)phiC  *(float)(h*h*h); \
momento1z[in_idx] = (float)((iz+0.5)*h)  *  (float)phiC  *(float)(h*h*h); \
momento2x[in_idx] = (float)powf((ix+0.5f)*(float)h,2.0f)*(float)phiC*(float)(h*h*h); \
momento2y[in_idx] = (float)powf((iy+1.5f)*(float)h,2.0f)*(float)phiC*(float)(h*h*h); \
momento2z[in_idx] = (float)powf((iz+0.5f)*(float)h,2.0f) *(float)phiC*(float)(h*h*h); \
C_float[in_idx] = (float)phiC;

#define COMPUTE_MOMENTO_EAST_BOTTOM \
momento1x[in_idx] = (float)((ix+0.5)*h)  *  (float)phiC  *(float)(h*h*h); \
momento1y[in_idx] = (float)((iy+1.5)*h ) *  (float)phiC  *(float)(h*h*h); \
momento1z[in_idx] = (float)((iz+0.5)*h)  *  (float)phiC  *(float)(h*h*h); \
momento2x[in_idx] = (float)powf((ix+0.5f)*(float)h,2.0f)*(float)phiC*(float)(h*h*h); \
momento2y[in_idx] = (float)powf((iy+1.5f)*(float)h,2.0f)*(float)phiC*(float)(h*h*h); \
momento2z[in_idx] = (float)powf((iz+0.5f)*(float)h,2.0f) *(float)phiC*(float)(h*h*h); \
C_float[in_idx] = (float)phiC;

#define COMPUTE_MOMENTO_EAST_TOP \
momento1x[in_idx] = (float)((ix+0.5)*h)  *  (float)phiC  *(float)(h*h*h); \
momento1y[in_idx] = (float)((iy+1.5)*h ) *  (float)phiC  *(float)(h*h*h); \
momento1z[in_idx] = (float)((iz+0.5)*h)  *  (float)phiC  *(float)(h*h*h); \
momento2x[in_idx] = (float)powf((ix+0.5f)*(float)h,2.0f)*(float)phiC*(float)(h*h*h); \
momento2y[in_idx] = (float)powf((iy+1.5f)*(float)h,2.0f)*(float)phiC*(float)(h*h*h); \
momento2z[in_idx] = (float)powf((iz+0.5f)*(float)h,2.0f) *(float)phiC*(float)(h*h*h); \
C_float[in_idx] = (float)phiC;

#define COMPUTE_MOMENTO_WEST_SOUTH \
momento1x[in_idx] = (float)((ix+0.5)*h)  *  (float)phiC  *(float)(h*h*h); \
momento1y[in_idx] = (float)((iy+0.5)*h ) *  (float)phiC  *(float)(h*h*h); \
momento1z[in_idx] = (float)((iz+1.5)*h)  *  (float)phiC  *(float)(h*h*h); \
momento2x[in_idx] = (float)powf((ix+0.5f)*(float)h,2.0f)*(float)phiC*(float)(h*h*h); \
momento2y[in_idx] = (float)powf((iy+0.5f)*(float)h,2.0f)*(float)phiC*(float)(h*h*h); \
momento2z[in_idx] = (float)powf((iz+1.5f)*(float)h,2.0f) *(float)phiC*(float)(h*h*h); \
C_float[in_idx] = (float)phiC;

#define COMPUTE_MOMENTO_EAST_SOUTH \
momento1x[in_idx] = (float)((ix+0.5)*h)  *  (float)phiC  *(float)(h*h*h); \
momento1y[in_idx] = (float)((iy+0.5)*h ) *  (float)phiC  *(float)(h*h*h); \
momento1z[in_idx] = (float)((iz+1.5)*h)  *  (float)phiC  *(float)(h*h*h); \
momento2x[in_idx] = (float)powf((ix+0.5f)*(float)h,2.0f)*(float)phiC*(float)(h*h*h); \
momento2y[in_idx] = (float)powf((iy+0.5f)*(float)h,2.0f)*(float)phiC*(float)(h*h*h); \
momento2z[in_idx] = (float)powf((iz+1.5f)*(float)h,2.0f) *(float)phiC*(float)(h*h*h); \
C_float[in_idx] = (float)phiC;

#define COMPUTE_MOMENTO_WEST_NORTH \
momento1x[in_idx] = (float)((ix+0.5)*h)  *  (float)phiC  *(float)(h*h*h); \
momento1y[in_idx] = (float)((iy+0.5)*h ) *  (float)phiC  *(float)(h*h*h); \
momento1z[in_idx] = (float)((iz+1.5)*h)  *  (float)phiC  *(float)(h*h*h); \
momento2x[in_idx] = (float)powf((ix+0.5f)*(float)h,2.0f)*(float)phiC*(float)(h*h*h); \
momento2y[in_idx] = (float)powf((iy+0.5f)*(float)h,2.0f)*(float)phiC*(float)(h*h*h); \
momento2z[in_idx] = (float)powf((iz+1.5f)*(float)h,2.0f) *(float)phiC*(float)(h*h*h); \
C_float[in_idx] = (float)phiC;

#define COMPUTE_MOMENTO_EAST_NORTH \
momento1x[in_idx] = (float)((ix+0.5)*h)  *  (float)phiC  *(float)(h*h*h); \
momento1y[in_idx] = (float)((iy+0.5)*h ) *  (float)phiC  *(float)(h*h*h); \
momento1z[in_idx] = (float)((iz+1.5)*h)  *  (float)phiC  *(float)(h*h*h); \
momento2x[in_idx] = (float)powf((ix+0.5f)*(float)h,2.0f)*(float)phiC*(float)(h*h*h); \
momento2y[in_idx] = (float)powf((iy+0.5f)*(float)h,2.0f)*(float)phiC*(float)(h*h*h); \
momento2z[in_idx] = (float)powf((iz+1.5f)*(float)h,2.0f) *(float)phiC*(float)(h*h*h); \
C_float[in_idx] = (float)phiC;

#define COMPUTE_MOMENTO_WEST_SOUTH_BOTTOM \
momento1x[in_idx] = (float)((ix+0.5)*h)  *  (float)phiC  *(float)(h*h*h); \
momento1y[in_idx] = (float)((iy+0.5)*h ) *  (float)phiC  *(float)(h*h*h); \
momento1z[in_idx] = (float)((iz+0.5)*h)  *  (float)phiC  *(float)(h*h*h); \
momento2x[in_idx] = (float)powf((ix+0.5f)*(float)h,2.0f)*(float)phiC*(float)(h*h*h); \
momento2y[in_idx] = (float)powf((iy+0.5f)*(float)h,2.0f)*(float)phiC*(float)(h*h*h); \
momento2z[in_idx] = (float)powf((iz+0.5f)*(float)h,2.0f) *(float)phiC*(float)(h*h*h); \
C_float[in_idx] = (float)phiC;

#define COMPUTE_MOMENTO_WEST_SOUTH_TOP COMPUTE_MOMENTO_WEST_SOUTH_BOTTOM
#define COMPUTE_MOMENTO_WEST_NORTH_BOTTOM COMPUTE_MOMENTO_WEST_SOUTH_BOTTOM
#define COMPUTE_MOMENTO_WEST_NORTH_TOP COMPUTE_MOMENTO_WEST_SOUTH_BOTTOM
#define COMPUTE_MOMENTO_EAST_SOUTH_BOTTOM COMPUTE_MOMENTO_WEST_SOUTH_BOTTOM
#define COMPUTE_MOMENTO_EAST_SOUTH_TOP COMPUTE_MOMENTO_WEST_SOUTH_BOTTOM
#define COMPUTE_MOMENTO_EAST_NORTH_BOTTOM COMPUTE_MOMENTO_WEST_SOUTH_BOTTOM
#define COMPUTE_MOMENTO_EAST_NORTH_TOP COMPUTE_MOMENTO_WEST_SOUTH_BOTTOM
```

# header\MG_struct.h

```h
#ifndef MG_STRUCT_H
#define MG_STRUCT_H
struct MG_levels{
	int L; // number of level
	int npre; //number of pre-smooth
	int npos; //number of post-smooth
};
#endif // MG_STRUCT_H ///:~
```

# header\routines_BiCGStab.h

```h
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

```

# header\routines_CCMG.h

```h
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
```

# header\routines_CG.h

```h
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

```

# header\routines_solver_type.h

```h
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
```

# main_transport_JSON_input.cu

```cu
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

// Libera _r, _e, _rr, _K creados por ALLOCATE_MG_STRUCTURE_MEMORY
#define FREE_MG_STRUCTURE_MEMORY(MG)                                           \
  do {                                                                         \
    /* Free device arrays for levels 0..L-2 */                                 \
    if (_e) {                                                                  \
      for (int i = 0; i < (MG).L - 1; ++i) {                                   \
        if (_e[i]) {                                                           \
          cudaFree(_e[i]);                                                     \
          _e[i] = nullptr;                                                     \
        }                                                                      \
      }                                                                        \
      free(_e);                                                                \
      _e = nullptr;                                                            \
    }                                                                          \
    if (_r) {                                                                  \
      for (int i = 0; i < (MG).L - 1; ++i) {                                   \
        if (_r[i]) {                                                           \
          cudaFree(_r[i]);                                                     \
          _r[i] = nullptr;                                                     \
        }                                                                      \
      }                                                                        \
      free(_r);                                                                \
      _r = nullptr;                                                            \
    }                                                                          \
    /* Free device arrays for all levels 0..L-1 */                             \
    if (_rr) {                                                                 \
      for (int i = 0; i < (MG).L; ++i) {                                       \
        if (_rr[i]) {                                                          \
          cudaFree(_rr[i]);                                                    \
          _rr[i] = nullptr;                                                    \
        }                                                                      \
      }                                                                        \
      free(_rr);                                                               \
      _rr = nullptr;                                                           \
    }                                                                          \
    if (_K) {                                                                  \
      for (int i = 0; i < (MG).L; ++i) {                                       \
        if (_K[i]) {                                                           \
          cudaFree(_K[i]);                                                     \
          _K[i] = nullptr;                                                     \
        }                                                                      \
      }                                                                        \
      free(_K);                                                                \
      _K = nullptr;                                                            \
    }                                                                          \
  } while (0)

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
                      alphaT, nParticles, nRealizations)) {
    std::cerr << "Error loading parameters from " << filename << std::endl;
    return -1;
  }

  double sigma_f = sqrt(sigma2);

  int device = atoi(argv[1]);
  cudaSetDevice(device);
  cudaDeviceReset();

  int nx_n = Nx / n;
  int ny_n = Ny / n;
  int nz_n = Nz / n;

  double A = h * h;
  double lambda = 10.0 * h;
  double Lx = Nx * h, Ly = Ny * h, Lz = Nz * h;

  double vm = 100 / Lx;
  t_max /= vm / lambda; // ajustar t_max con vm y lambda

  double dt, max_magnitude;
  int levels = int(std::log(double(n)) / std::log(2.0) + 1.0);

  // --------- Print parámetros -----------
  std::cout << "=======================================================\n"
            << "                 SIMULATION PARAMETERS                 \n"
            << "=======================================================\n"
            << "Domain Information:\n"
            << "  - Grid Size: " << Nx << " x " << Ny << " x " << Nz << "\n"
            << "  - Grid Spacing (h): " << h << "\n"
            << "  - Domain Dimensions: " << Lx << " x " << Ly << " x " << Lz
            << "\n\n"
            << "Flow Parameters:\n"
            << "  - Correlation Length (lambda): " << lambda << "\n"
            << "  - Log-K Variance (sigma²): " << sigma2 << "\n"
            << "  - Standard Deviation (sigma): " << sigma_f << "\n\n"
            << "Transport Parameters:\n"
            << "  - Maximum Simulation Time: " << t_max << "\n"
            << "  - Molecular Diffusion: " << diffusion << "\n"
            << "  - Longitudinal Dispersivity (αL): " << alphaL << "\n"
            << "  - Transverse Dispersivity (αT): " << alphaT << "\n"
            << "  - Number of Particles: " << nParticles << "\n\n"
            << "Simulation Configuration:\n"
            << "  - Number of Realizations: " << nRealizations << "\n"
            << "  - Estimated Mean Velocity: " << vm << "\n"
            << "  - Multigrid Levels: " << levels << "\n"
            << "=======================================================\n\n";

  //%%%%% INIT MULTIGRID CONFIGURATION %%%%%%%%%%%%%%%%%%%
  std::cout << "Setting multigrid configurations..." << std::endl;
  SETUP_BLOCK_GRID3D(16)
  int npre = 4, npos = 4;
  MG_levels MG{levels, npre, npos};
  SETUP_GRID_BLOCK_MG(MG, nx_n, ny_n, nz_n)

  // BLAS/cuBLAS
  INIT_CUBLAS_ENVIRONMENT
  blas BLAS(Nx, Ny, Nz, grid16, block16, handle);

  // RNG y buffers auxiliares (una vez)
  int i_max = 10000;
  curandState *devStates;
  CUDA_CALL(cudaMalloc((void **)&devStates, i_max * sizeof(curandState)));
  dim3 block1(1024, 1);
  dim3 grid1((i_max + block1.x - 1) / block1.x, 1);
  setup_uniform_distrib<<<grid1, block1>>>(devStates, i_max);
  cudaDeviceSynchronize();

  CUDA_ALLOCATE_VECTOR(double, i_max, V1);
  CUDA_ALLOCATE_VECTOR(double, i_max, V2);
  CUDA_ALLOCATE_VECTOR(double, i_max, V3);
  CUDA_ALLOCATE_VECTOR(double, i_max, a);
  CUDA_ALLOCATE_VECTOR(double, i_max, b);
  double *K_eq;
  cudaMalloc(&K_eq, sizeof(double));
  double *host_K_eq = new double[1];
  SETUP_BLOCK_GRID_RANDOM_KERNEL(block2, grid2, Nx, Ny, Nz)

  // BCs para flujo
  std::cout << "Setting up flow equation..." << std::endl;
  int BCbottom = periodic, BCtop = periodic;
  int BCsouth = periodic, BCnorth = periodic;
  int BCwest = dirichlet, BCeast = dirichlet;
  bool pin1stCell = false;

  double Hbottom = NAN, Htop = NAN, Hsouth = NAN, Hnorth = NAN;
  double Heast = 0.0, Hwest = 100.0;

  // coarse
  int Nx_coarse = nx_n, Ny_coarse = ny_n, Nz_coarse = nz_n;
  double h_coarse = Ly / Ny_coarse, A_coarse = h_coarse * h_coarse;
  CUDA_ALLOCATE_VECTOR(double, nx_n *ny_n *nz_n, y_coarse);
  CUDA_ALLOCATE_VECTOR(double, nx_n *ny_n *nz_n, r_coarse);
  blas BLAS_coarse(Nx_coarse, Ny_coarse, Nz_coarse, _grid[1], _block[1],
                   handle);
  IdentityPrecond Icoarse(Nx_coarse, Ny_coarse, Nz_coarse);
  IdentityPrecond I(Nx, Ny, Nz);

  // salida
  char outdir[50];
  sprintf(outdir, "./output/out_%.2f", sigma2);
  std::string outdir_str(outdir);
  char command[100];
  sprintf(command, "rm -rf %s && mkdir -p %s", outdir, outdir);
  system(command);

  for (int k = 0; k < nRealizations; k++) {
    std::cout << "Realization " << (k + 1) << " of " << nRealizations
              << std::endl;

    // ===== Estructuras MG por realización =====
    ALLOCATE_MG_STRUCTURE_MEMORY(MG, nx_n, ny_n, nz_n);

    // Limpieza de workspaces MG (evita basura/NaN)
    for (int i = 0; i < MG.L - 1; ++i) {
      size_t Ni = (size_t)std::pow(2, i) * nx_n;
      size_t Mi = (size_t)std::pow(2, i) * ny_n;
      size_t Ki = (size_t)std::pow(2, i) * nz_n;
      size_t bytes = Ni * Mi * Ki * sizeof(double);
      cudaMemset(_e[i], 0, bytes);
      cudaMemset(_r[i], 0, bytes);
    }
    for (int i = 0; i < MG.L; ++i) {
      size_t Ni = (size_t)std::pow(2, i) * nx_n;
      size_t Mi = (size_t)std::pow(2, i) * ny_n;
      size_t Ki = (size_t)std::pow(2, i) * nz_n;
      cudaMemset(_rr[i], 0, Ni * Mi * Ki * sizeof(double));
    }

    // ===== Buffers flujo =====
    CUDA_ALLOCATE_VECTOR(double, Nx *Ny *Nz, RHS_flow);
    CUDA_ALLOCATE_VECTOR(double, Nx *Ny *Nz, Head);
    CUDA_ALLOCATE_VECTOR(double, Nx *Ny *Nz, r);
    CUDA_ALLOCATE_VECTOR(double, Nx *Ny *Nz, z);
    cudaMemset(Head, 0, sizeof(double) * Nx * Ny * Nz);
    cudaMemset(RHS_flow, 0, sizeof(double) * Nx * Ny * Nz);
    cudaMemset(r, 0, sizeof(double) * Nx * Ny * Nz);
    cudaMemset(z, 0, sizeof(double) * Nx * Ny * Nz);

    // ===== Campo K (logK -> expK) =====
    random_kernel_3D_gauss<<<grid1, block1>>>(devStates, V1, V2, V3, a, b,
                                              lambda, i_max, 100);
    cudaDeviceSynchronize();
    conductivity_kernel_3D_logK<<<grid2, block2>>>(
        V1, V2, V3, a, b, i_max, _K[MG.L - 1], lambda, h, Nx, Ny, Nz, sigma_f);
    cudaDeviceSynchronize();
    compute_expK<<<grid2, block2>>>(_K[MG.L - 1], Nx, Ny, Nz);
    cudaDeviceSynchronize();

    // Homogenización niveles
    for (int i = MG.L - 1; i > 1; --i)
      HomogenizationPermeability(
          _K[i - 1], _K[i], (int)std::pow(2, i - 1) * nx_n,
          (int)std::pow(2, i - 1) * ny_n, (int)std::pow(2, i - 1) * nz_n,
          _grid[i - 1], _block[i - 1]);

    // Operadores/precondicionador
    laplacianHead AH(_K[MG.L - 1], Nx, Ny, Nz, A, h, BCbottom, BCtop, BCsouth,
                     BCnorth, BCwest, BCeast, pin1stCell, grid16, block16);

    laplacianHeadCoarse Ap_coarse(_K[1], Nx_coarse, Ny_coarse, Nz_coarse,
                                  A_coarse, h_coarse, BCbottom, BCtop, BCsouth,
                                  BCnorth, BCwest, BCeast, pin1stCell, _grid[1],
                                  _block[1]);

    MGprecond2 PCCMG_CG(Nx, Ny, Nz, BCbottom, BCtop, BCsouth, BCnorth, BCwest,
                        BCeast, pin1stCell, _grid, _block, Ly, nx_n, ny_n, nz_n,
                        handle, _e, _r, _rr, _K, MG, Ap_coarse, Icoarse,
                        BLAS_coarse, r_coarse, y_coarse);

    // ===== Montaje y solución de flujo =====
    std::cout << "Assembling and solving flow equation..." << std::endl;
    RHS_head(RHS_flow, _K[MG.L - 1], Nx, Ny, Nz, A, h, BCbottom, BCtop, BCsouth,
             BCnorth, BCwest, BCeast, Hbottom, Htop, Hsouth, Hnorth, Hwest,
             Heast, grid16, block16);

    // Semilla residual: r := b
    cudaMemcpy(r, RHS_flow, sizeof(double) * Nx * Ny * Nz,
               cudaMemcpyDeviceToDevice);

    int print_monitor = 2;
    int iterHead = solver_CG(AH, PCCMG_CG, BLAS, Head, z, r, RHS_flow,
                             _rr[MG.L - 1], 0.0, 1e-6, 200, print_monitor);

    // Libera buffers de flujo que no se usan más
    cudaFree(RHS_flow);
    cudaFree(r);
    cudaFree(z);

    if (_rr) {
      for (int i = 0; i < MG.L; ++i)
        if (_rr[i])
          cudaFree(_rr[i]);
      free(_rr);
    }
    if (_e) {
      for (int i = 0; i < MG.L - 1; ++i)
        if (_e[i])
          cudaFree(_e[i]);
      free(_e);
    }
    if (_r) {
      for (int i = 0; i < MG.L - 1; ++i)
        if (_r[i])
          cudaFree(_r[i]);
      free(_r);
    }

    // ===== Velocidad (layout cúbico) =====
    thrust::device_vector<double> U_cube((Nx + 1) * (Ny + 1) * (Nz + 1));
    thrust::device_vector<double> V_cube((Nx + 1) * (Ny + 1) * (Nz + 1));
    thrust::device_vector<double> W_cube((Nx + 1) * (Ny + 1) * (Nz + 1));

    std::cout << "Computing velocity field from hydraulic head..." << std::endl;
    compute_velocity_from_head(thrust::raw_pointer_cast(U_cube.data()),
                               thrust::raw_pointer_cast(V_cube.data()),
                               thrust::raw_pointer_cast(W_cube.data()), Head,
                               _K[MG.L - 1], Nx, Ny, Nz, h, BCwest, BCeast,
                               BCsouth, BCnorth, BCbottom, BCtop, Hwest, Heast,
                               Hsouth, Hnorth, Hbottom, Htop, grid16, block16);

    CUDA_CALL(cudaDeviceSynchronize());

    // _K y Head ya no se usan tras construir velocidades
    if (_K) {
      for (int i = 0; i < MG.L; ++i)
        if (_K[i])
          cudaFree(_K[i]);
      free(_K);
    }
    cudaFree(Head);

    // ===== Transporte =====
    std::cout << "Setting up transport..." << std::endl;

    if (k == 0) {
      double max_magnitude_squared = thrust::transform_reduce(
          thrust::make_zip_iterator(thrust::make_tuple(
              U_cube.begin(), V_cube.begin(), W_cube.begin())),
          thrust::make_zip_iterator(
              thrust::make_tuple(U_cube.end(), V_cube.end(), W_cube.end())),
          squared_magnitude_functor(), 0.0, thrust::maximum<double>());

      max_magnitude = std::sqrt(max_magnitude_squared);
      dt = h / (2 * max_magnitude); // CFL simple
      std::cout << "  - ||v_max|| : " << max_magnitude << std::endl;
      std::cout << "  - Time Step (dt): " << dt << std::endl;
    }

    // Caja de inyección
    const double p1x = 10.0 * h, p1y = Ly * 0.1, p1z = Lz * 0.1;
    const double p2x = 10.0 * h, p2y = Ly - p1y, p2z = Lz - p1z;

    bool useTrilinearCorrection = true;
    long int seed = 123456789L * (k + 1);
    int steps = int(t_max / dt);

    auto grid = par2::grid::build<double>(Nx, Ny, Nz, h, h, h);

    // *** NUEVO: sin nY/nZ ni posY/posZ (periodic adentro de
    // PParticles/moveParticle) ***
    par2::PParticles<double> particles(
        grid, std::move(U_cube), std::move(V_cube), std::move(W_cube),
        diffusion, alphaL, alphaT, nParticles, seed, useTrilinearCorrection);

    // La lib ya maneja periódicos; inicializamos normalmente
    particles.initializeBox(p1x, p1y, p1z, p2x, p2y, p2z, /*shuffle=*/true);

    // punteros para estadística (asumiendo que yPtr()/zPtr() ya están
    // “unwrapped” por adentro)
    thrust::device_ptr<const double> xBeg(particles.xPtr());
    thrust::device_ptr<const double> yBeg(
        particles
            .yUnwrapPtr()); // si tu clase expone yPtrUnwrapped(), usalo acá
    thrust::device_ptr<const double> zBeg(particles.zUnwrapPtr());

    std::string outPath = outdir_str + "/macrodispersion_var_v9_" +
                          std::to_string(sigma2).substr(0, 4) + "_" +
                          std::to_string(k) + ".csv";

    std::ofstream csv(outPath);
    csv << "t,Dx,Dy,Dz\n";
    double prevVarX = 0.0, prevVarY = 0.0, prevVarZ = 0.0;
    const int reg = std::max(1, steps / 300);

    std::cout << "Starting transport simulation..." << std::endl;
    for (int i = 0; i < steps; i++) {
      particles.move(dt); // *** aquí adentro ya se aplican BC periódicas ***

      if (i % int(std::round(steps * 0.2)) == 0 && i != 0) {
        std::cout << "  - Step: " << i << " / " << steps << " ("
                  << int(i * 100.0 / steps) << "%)" << std::endl;
      }

      // --- primer pase: guardar var previas
      if ((i + 1) % reg == 0 || i == 1) {
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

        prevVarX = varX;
        prevVarY = varY;
        prevVarZ = varZ;
      }

      // --- segundo pase: derivada temporal de varianzas
      if (i % reg == 0 || i == 2) {
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

  // ===== Liberaciones finales (una vez) =====
  cudaFree(K_eq);
  delete[] host_K_eq;

  // cublasDestroy(handle); // si tu macro INIT_CUBLAS_ENVIRONMENT no lo hace
  cudaFree(V1);
  cudaFree(V2);
  cudaFree(V3);
  cudaFree(a);
  cudaFree(b);
  cudaFree(devStates);
  cudaFree(y_coarse);
  cudaFree(r_coarse);

  return 0;
}

```

# Makefile

```
# Nombre del ejecutable final
TARGET = run_flow2.out

# Librería estática
LIB = lib_flow.a

# Archivos fuente para la librería
LIB_SRC = alpha_3D.cu AXPBY_3D.cu beta_3D.cu CCMG_V_cycle.cu CG.cu \
          GSRB_Smooth_up_residual_3D.cu random_field_generation.cu \
          RHS_head_3D.cu stencil_head_3D.cu transf_operator_3D.cu \
          up_residual_3D.cu alphaU_3D.cu betaU_3D.cu omega_r_3D.cu omega_x_3D.cu

# Archivos objeto de la librería
LIB_OBJ = $(LIB_SRC:.cu=.o)

# Archivos fuente principales
MAIN_SRC = main_transport_JSON_input.cu compute_velocity_from_head_for_par2.cu

# Archivos objeto principales
MAIN_OBJ = $(MAIN_SRC:.cu=.o)

# Compilador
NVCC = nvcc

# Flags de compilación
NVCC_FLAGS = -std=c++11 -Iinclude --extended-lambda

# Librerías
LIBS = -lcublas -L. ./$(LIB)

# Habilitar compilación paralela con todos los núcleos disponibles
MAKEFLAGS += -j$(shell nproc)

# -------------------
# Reglas principales
# -------------------

all: $(TARGET)

# 1. Compilar ejecutable final
$(TARGET): $(MAIN_OBJ) $(LIB)
	$(NVCC) $(NVCC_FLAGS) $^ -o $@ $(LIBS)

# 2. Generar la librería estática
$(LIB): $(LIB_OBJ)
	ar -rcs $@ $^

# 3. Compilar objetos de la librería
%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) -Xcompiler -static -c $< -o $@

# Limpiar archivos generados
clean:
	rm -f $(LIB_OBJ) $(MAIN_OBJ) $(TARGET) $(LIB)

# Recompilar desde cero
rebuild: clean all

```

# omega_r_3D.cu

```cu
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
```

# omega_x_3D.cu

```cu
#include <cuda_runtime_api.h>
#define gridXY dim3(grid.x,grid.y)
#define blockXY dim3(block.x,block.y)
#define gridXZ dim3(grid.x,grid.z)
#define blockXZ dim3(block.x,block.z)
#define gridYZ dim3(grid.y,grid.z)
#define blockYZ dim3(block.y,block.z)

//#######################################################################
// 	routine omega_x (uses in BiCGStab)
//  x_{j+1} = x_j + alpha*M*p_j + omega*M*s_j
//	omega = (AMs, s) / (AMs, AMs)
//  alpha = (r_j, r_star) / (A*M*p, r_star)
//#######################################################################
//-----------------------------------------------------------------------

__global__ void omega_x_int_bottom_top(const double *p, double *x, const double *s, const double *rr_, const double *Apr_, const double *Ass, const double *AsAs,
const int Nx, const int Ny, const int Nz){
	int ix = threadIdx.x + blockIdx.x*blockDim.x;
	int iy = threadIdx.y + blockIdx.y*blockDim.y;
	if (ix >= Nx-2 || iy >= Ny-2) return;
	int stride = Nx*Ny;
	int in_idx = (ix+1) + (iy+1)*Nx;
	for(int iz = 0; iz<Nz; ++iz){
		x[in_idx] +=  rr_[0]/Apr_[0]*p[in_idx] + Ass[0]/AsAs[0]*s[in_idx];
		in_idx+=stride;
	}
}

__global__ void omega_x_south_north(const double *p, double *x, const double *s, const double *rr_, const double *Apr_, const double *Ass, const double *AsAs,
const int Nx, const int Ny, const int Nz){
	int ix = threadIdx.x + blockIdx.x*blockDim.x;
	int iz = threadIdx.y + blockIdx.y*blockDim.y;
	if (ix >= Nx-2 || iz >= Nz-2) return;
	int stride = Nx*Ny;
	int in_idx;
	in_idx = (ix + 1) + (iz + 1)*stride;
	x[in_idx] +=  rr_[0]/Apr_[0]*p[in_idx] + Ass[0]/AsAs[0]*s[in_idx];

	in_idx = (ix + 1) + (Ny - 1)*Nx + (iz + 1)*stride;
	x[in_idx] +=  rr_[0]/Apr_[0]*p[in_idx] + Ass[0]/AsAs[0]*s[in_idx];
}

__global__ void omega_x_este_oeste(const double *p, double *x, const double *s, const double *rr_, const double *Apr_, const double *Ass, const double *AsAs,
const int Nx, const int Ny, const int Nz){
	int iy = threadIdx.x + blockIdx.x*blockDim.x;
	int iz  = threadIdx.y + blockIdx.y*blockDim.y;
	if (iy >= Ny-2 || iz >= Nz-2) return;
	int stride = Nx*Ny;
	int in_idx;
	in_idx = (Nx - 1) + (iy + 1)*Nx + (iz + 1)*stride;
	x[in_idx] +=  rr_[0]/Apr_[0]*p[in_idx] + Ass[0]/AsAs[0]*s[in_idx];

	in_idx = (iy + 1)*Nx + (iz + 1)*stride;
	x[in_idx] +=  rr_[0]/Apr_[0]*p[in_idx] + Ass[0]/AsAs[0]*s[in_idx];
}

__global__ void omega_x_edge_X_South_Bottom(const double *p, double *x, const double *s, const double *rr_, const double *Apr_, const double *Ass, const double *AsAs,
const int Nx, const int Ny, const int Nz){
	int ix = threadIdx.x + blockIdx.x*blockDim.x;
	if (ix >= Nx-2) return;
	int stride = Nx*Ny;
	int iy = 0;
	int iz = 0;
	int in_idx = (ix+1) + iy*Nx + iz*stride;
	x[in_idx] +=  rr_[0]/Apr_[0]*p[in_idx] + Ass[0]/AsAs[0]*s[in_idx];
}

__global__ void omega_x_edge_X_South_Top(const double *p, double *x, const double *s, const double *rr_, const double *Apr_, const double *Ass, const double *AsAs,
const int Nx, const int Ny, const int Nz){
	int ix = threadIdx.x + blockIdx.x*blockDim.x;
	if (ix >= Nx-2) return;
	int stride = Nx*Ny;
	int iy = 0;
	int iz = Nz-1;
	int in_idx = (ix+1) + iy*Nx + iz*stride;
	x[in_idx] +=  rr_[0]/Apr_[0]*p[in_idx] + Ass[0]/AsAs[0]*s[in_idx];
}

__global__ void omega_x_edge_X_North_Bottom(const double *p, double *x, const double *s, const double *rr_, const double *Apr_, const double *Ass, const double *AsAs,
const int Nx, const int Ny, const int Nz){
	int ix = threadIdx.x + blockIdx.x*blockDim.x;
	if (ix >= Nx-2) return;
	int stride = Nx*Ny;
	int iy = Ny-1;
	int iz = 0;
	int in_idx = (ix+1) + iy*Nx + iz*stride;
	x[in_idx] +=  rr_[0]/Apr_[0]*p[in_idx] + Ass[0]/AsAs[0]*s[in_idx];
}

__global__ void omega_x_edge_X_North_Top(const double *p, double *x, const double *s, const double *rr_, const double *Apr_, const double *Ass, const double *AsAs,
const int Nx, const int Ny, const int Nz){
	int ix = threadIdx.x + blockIdx.x*blockDim.x;
	if (ix >= Nx-2) return;
	int stride = Nx*Ny;
	int iy = Ny-1;
	int iz = Nz-1;
	int in_idx = (ix+1) + iy*Nx + iz*stride;
	x[in_idx] +=  rr_[0]/Apr_[0]*p[in_idx] + Ass[0]/AsAs[0]*s[in_idx];
}

__global__ void omega_x_edge_Z_South_West(const double *p, double *x, const double *s, const double *rr_, const double *Apr_, const double *Ass, const double *AsAs,
const int Nx, const int Ny, const int Nz){
	int iz = threadIdx.x + blockIdx.x*blockDim.x;
	if (iz >= Nz-2) return;
	int stride = Nx*Ny;
	int ix = 0;
	int iy = 0;
	int in_idx = ix + iy*Nx + (iz+1)*stride;
	x[in_idx] +=  rr_[0]/Apr_[0]*p[in_idx] + Ass[0]/AsAs[0]*s[in_idx];
}

__global__ void omega_x_edge_Z_South_East(const double *p, double *x, const double *s, const double *rr_, const double *Apr_, const double *Ass, const double *AsAs,
const int Nx, const int Ny, const int Nz){
	int iz = threadIdx.x + blockIdx.x*blockDim.x;
	if (iz >= Nz-2) return;
	int stride = Nx*Ny;
	int ix = Nx-1;
	int iy = 0;
	int in_idx = ix + iy*Nx + (iz+1)*stride;
	x[in_idx] +=  rr_[0]/Apr_[0]*p[in_idx] + Ass[0]/AsAs[0]*s[in_idx];
}
__global__ void omega_x_edge_Z_North_West(const double *p, double *x, const double *s, const double *rr_, const double *Apr_, const double *Ass, const double *AsAs,
const int Nx, const int Ny, const int Nz){
	int iz = threadIdx.x + blockIdx.x*blockDim.x;
	if (iz >= Nz-2) return;
	int stride = Nx*Ny;
	int ix = 0;
	int iy = Ny-1;
	int in_idx = ix + iy*Nx + (iz+1)*stride;
	x[in_idx] +=  rr_[0]/Apr_[0]*p[in_idx] + Ass[0]/AsAs[0]*s[in_idx];
}

__global__ void omega_x_edge_Z_North_East(const double *p, double *x, const double *s, const double *rr_, const double *Apr_, const double *Ass, const double *AsAs,
const int Nx, const int Ny, const int Nz){
	int iz = threadIdx.x + blockIdx.x*blockDim.x;
	if (iz >= Nz-2) return;
	int stride = Nx*Ny;
	int ix = Nx-1;
	int iy = Ny-1;
	int in_idx = ix + iy*Nx + (iz+1)*stride;
	x[in_idx] +=  rr_[0]/Apr_[0]*p[in_idx] + Ass[0]/AsAs[0]*s[in_idx];
}

__global__ void omega_x_edge_Y_West_Bottom(const double *p, double *x, const double *s, const double *rr_, const double *Apr_, const double *Ass, const double *AsAs,
const int Nx, const int Ny, const int Nz){
	int iy = threadIdx.x + blockIdx.x*blockDim.x;
	if (iy >= Ny-2) return;
	int stride = Nx*Ny;
	int ix = 0;
	int iz = 0;
	int in_idx = ix + (iy+1)*Nx + iz*stride;
	x[in_idx] +=  rr_[0]/Apr_[0]*p[in_idx] + Ass[0]/AsAs[0]*s[in_idx];
}
__global__ void omega_x_edge_Y_West_Top(const double *p, double *x, const double *s, const double *rr_, const double *Apr_, const double *Ass, const double *AsAs,
const int Nx, const int Ny, const int Nz){
	int iy = threadIdx.x + blockIdx.x*blockDim.x;
	if (iy >= Ny-2) return;
	int stride = Nx*Ny;
	int ix = 0;
	int iz = Nz-1;
	int in_idx = ix + (iy+1)*Nx + iz*stride;
	x[in_idx] +=  rr_[0]/Apr_[0]*p[in_idx] + Ass[0]/AsAs[0]*s[in_idx];
}
__global__ void omega_x_edge_Y_East_Bottom(const double *p, double *x, const double *s, const double *rr_, const double *Apr_, const double *Ass, const double *AsAs,
const int Nx, const int Ny, const int Nz){
	int iy = threadIdx.x + blockIdx.x*blockDim.x;
	if (iy >= Ny-2) return;
	int stride = Nx*Ny;
	int ix = Nx-1;
	int iz = 0;
	int in_idx = ix + (iy+1)*Nx + iz*stride;
	x[in_idx] +=  rr_[0]/Apr_[0]*p[in_idx] + Ass[0]/AsAs[0]*s[in_idx];
}
__global__ void omega_x_edge_Y_East_Top(const double *p, double *x, const double *s, const double *rr_, const double *Apr_, const double *Ass, const double *AsAs,
const int Nx, const int Ny, const int Nz){
	int iy = threadIdx.x + blockIdx.x*blockDim.x;
	if (iy >= Ny-2) return;
	int stride = Nx*Ny;
	int ix = Nx-1;
	int iz = Nz-1;
	int in_idx = ix + (iy+1)*Nx + iz*stride;
	x[in_idx] +=  rr_[0]/Apr_[0]*p[in_idx] + Ass[0]/AsAs[0]*s[in_idx];
}

__global__ void omega_x_vertex_SWB(const double *p, double *x, const double *s, const double *rr_, const double *Apr_, const double *Ass, const double *AsAs,
const int Nx, const int Ny, const int Nz){
	int ix = 0;
	int iy = 0;
	int iz = 0;
	int in_idx = ix + iy*Nx + iz*Nx*Ny;
	x[in_idx] +=  rr_[0]/Apr_[0]*p[in_idx] + Ass[0]/AsAs[0]*s[in_idx];
}
__global__ void omega_x_vertex_SWT(const double *p, double *x, const double *s, const double *rr_, const double *Apr_, const double *Ass, const double *AsAs,
const int Nx, const int Ny, const int Nz){
	int ix = 0;
	int iy = 0;
	int iz = Nz-1;
	int in_idx = ix + iy*Nx + iz*Nx*Ny;
	x[in_idx] +=  rr_[0]/Apr_[0]*p[in_idx] + Ass[0]/AsAs[0]*s[in_idx];
}
__global__ void omega_x_vertex_SEB(const double *p, double *x, const double *s, const double *rr_, const double *Apr_, const double *Ass, const double *AsAs,
const int Nx, const int Ny, const int Nz){
	int ix = Nx-1;
	int iy = 0;
	int iz = 0;
	int in_idx = ix + iy*Nx + iz*Nx*Ny;
	x[in_idx] +=  rr_[0]/Apr_[0]*p[in_idx] + Ass[0]/AsAs[0]*s[in_idx];
}
__global__ void omega_x_vertex_SET(const double *p, double *x, const double *s, const double *rr_, const double *Apr_, const double *Ass, const double *AsAs,
const int Nx, const int Ny, const int Nz){
	int ix = Nx-1;
	int iy = 0;
	int iz = Nz-1;
	int in_idx = ix + iy*Nx + iz*Nx*Ny;
	x[in_idx] +=  rr_[0]/Apr_[0]*p[in_idx] + Ass[0]/AsAs[0]*s[in_idx];
}




__global__ void omega_x_vertex_NWB(const double *p, double *x, const double *s, const double *rr_, const double *Apr_, const double *Ass, const double *AsAs,
const int Nx, const int Ny, const int Nz){
	int ix = 0;
	int iy = Ny-1;
	int iz = 0;
	int in_idx = ix + iy*Nx + iz*Nx*Ny;
	x[in_idx] +=  rr_[0]/Apr_[0]*p[in_idx] + Ass[0]/AsAs[0]*s[in_idx];
}
__global__ void omega_x_vertex_NWT(const double *p, double *x, const double *s, const double *rr_, const double *Apr_, const double *Ass, const double *AsAs,
const int Nx, const int Ny, const int Nz){
	int ix = 0;
	int iy = Ny-1;
	int iz = Nz-1;
	int in_idx = ix + iy*Nx + iz*Nx*Ny;
	x[in_idx] +=  rr_[0]/Apr_[0]*p[in_idx] + Ass[0]/AsAs[0]*s[in_idx];
}
__global__ void omega_x_vertex_NEB(const double *p, double *x, const double *s, const double *rr_, const double *Apr_, const double *Ass, const double *AsAs,
const int Nx, const int Ny, const int Nz){
	int ix = Nx-1;
	int iy = Ny-1;
	int iz = 0;
	int in_idx = ix + iy*Nx + iz*Nx*Ny;
	x[in_idx] +=  rr_[0]/Apr_[0]*p[in_idx] + Ass[0]/AsAs[0]*s[in_idx];
}
__global__ void omega_x_vertex_NET(const double *p, double *x, const double *s, const double *rr_, const double *Apr_, const double *Ass, const double *AsAs,
const int Nx, const int Ny, const int Nz){
	int ix = Nx-1;
	int iy = Ny-1;
	int iz = Nz-1;
	int in_idx = ix + iy*Nx + iz*Nx*Ny;
	x[in_idx] +=  rr_[0]/Apr_[0]*p[in_idx] + Ass[0]/AsAs[0]*s[in_idx];
}


void omegaX(const double *p, double *x, const double *s, const double *rr_, const double *Apr_, const double *Ass, const double *AsAs,
	const int Nx, const int Ny, const int Nz,
	dim3 grid, dim3 block){
	omega_x_int_bottom_top<<<gridXY,blockXY>>>(p,x,s,rr_,Apr_,Ass,AsAs,Nx,Ny,Nz);
	omega_x_south_north<<<gridXZ,blockXZ>>>(p,x,s,rr_,Apr_,Ass,AsAs,Nx,Ny,Nz);
	omega_x_este_oeste<<<gridYZ,blockYZ>>>(p,x,s,rr_,Apr_,Ass,AsAs,Nx,Ny,Nz);

	omega_x_edge_X_South_Bottom<<<grid.x,block.x>>>(p,x,s,rr_,Apr_,Ass,AsAs,Nx,Ny,Nz);
	omega_x_edge_X_South_Top<<<grid.x,block.x>>>(p,x,s,rr_,Apr_,Ass,AsAs,Nx,Ny,Nz);
	omega_x_edge_X_North_Bottom<<<grid.x,block.x>>>(p,x,s,rr_,Apr_,Ass,AsAs,Nx,Ny,Nz);
	omega_x_edge_X_North_Top<<<grid.x,block.x>>>(p,x,s,rr_,Apr_,Ass,AsAs,Nx,Ny,Nz);

	omega_x_edge_Z_South_West<<<grid.z,block.z>>>(p,x,s,rr_,Apr_,Ass,AsAs,Nx,Ny,Nz);
	omega_x_edge_Z_South_East<<<grid.z,block.z>>>(p,x,s,rr_,Apr_,Ass,AsAs,Nx,Ny,Nz);
	omega_x_edge_Z_North_West<<<grid.z,block.z>>>(p,x,s,rr_,Apr_,Ass,AsAs,Nx,Ny,Nz);
	omega_x_edge_Z_North_East<<<grid.z,block.z>>>(p,x,s,rr_,Apr_,Ass,AsAs,Nx,Ny,Nz);

	omega_x_edge_Y_West_Bottom<<<grid.y,block.y>>>(p,x,s,rr_,Apr_,Ass,AsAs,Nx,Ny,Nz);
	omega_x_edge_Y_West_Top<<<grid.y,block.y>>>(p,x,s,rr_,Apr_,Ass,AsAs,Nx,Ny,Nz);
	omega_x_edge_Y_East_Bottom<<<grid.y,block.y>>>(p,x,s,rr_,Apr_,Ass,AsAs,Nx,Ny,Nz);
	omega_x_edge_Y_East_Top<<<grid.y,block.y>>>(p,x,s,rr_,Apr_,Ass,AsAs,Nx,Ny,Nz);

	omega_x_vertex_SWB<<<1,1>>>(p,x,s,rr_,Apr_,Ass,AsAs,Nx,Ny,Nz);
	omega_x_vertex_SWT<<<1,1>>>(p,x,s,rr_,Apr_,Ass,AsAs,Nx,Ny,Nz);
	omega_x_vertex_SEB<<<1,1>>>(p,x,s,rr_,Apr_,Ass,AsAs,Nx,Ny,Nz);
	omega_x_vertex_SET<<<1,1>>>(p,x,s,rr_,Apr_,Ass,AsAs,Nx,Ny,Nz);
	omega_x_vertex_NWB<<<1,1>>>(p,x,s,rr_,Apr_,Ass,AsAs,Nx,Ny,Nz);
	omega_x_vertex_NWT<<<1,1>>>(p,x,s,rr_,Apr_,Ass,AsAs,Nx,Ny,Nz);
	omega_x_vertex_NEB<<<1,1>>>(p,x,s,rr_,Apr_,Ass,AsAs,Nx,Ny,Nz);
	omega_x_vertex_NET<<<1,1>>>(p,x,s,rr_,Apr_,Ass,AsAs,Nx,Ny,Nz);
	cudaDeviceSynchronize();
}
```

# par2\Geometry\CartesianGrid.cuh

```cuh
/**
 * @file CellVariable.cuh
 * @brief Header file for grid.
 *        Data stuctures and functions of a Cartesian grid
 *
 * @author Calogero B. Rizzo
 *
 * @copyright This file is part of the PAR2 software.
 *            Copyright (C) 2018 Calogero B. Rizzo
 *
 * @license This program is free software: you can redistribute it and/or modify
 *          it under the terms of the GNU General Public License as published by
 *          the Free Software Foundation, either version 3 of the License, or
 *          (at your option) any later version.
 *
 *          This program is distributed in the hope that it will be useful,
 *          but WITHOUT ANY WARRANTY; without even the implied warranty of
 *          MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *          GNU General Public License for more details.
 *
 *          You should have received a copy of the GNU General Public License
 *          along with this program.  If not, see
 * <http://www.gnu.org/licenses/>.
 */

#ifndef PAR2_CARTESIANGRID_CUH
#define PAR2_CARTESIANGRID_CUH

#include "Point.cuh"
#include <stddef.h>

namespace par2 {

namespace grid {
/**
 * @struct Grid
 * @brief Struct containing the Cartesian grid parameters.
 * @tparam T Float number precision
 */
template <typename T> struct Grid {
  int nx, ny, nz;
  T dx, dy, dz;
  T px, py, pz;
  T xmax, ymax, zmax;
  T xmin = px;
  T ymin = py;
  T zmin = pz;


  Grid(){};

  __host__ __device__ Grid(const Grid &g)
      : nx(g.nx), ny(g.ny), nz(g.nz), dx(g.dx), dy(g.dy), dz(g.dz), px(g.px),
        py(g.py), pz(g.pz){};
};

/**
 * @brief Build the grid.
 * @param _nx Number of cells along x
 * @param _ny Number of cells along y
 * @param _nz Number of cells along z
 * @param _dx Size of cell along x
 * @param _dy Size of cell along y
 * @param _dz Size of cell along z
 * @param _px Origin along x
 * @param _py Origin along y
 * @param _pz Origin along z
 * @tparam T Float number precision
 */
template <typename T>
Grid<T> build(int _nx, int _ny, int _nz, T _dx, T _dy, T _dz, T _px = 0,
              T _py = 0, T _pz = 0) {
  Grid<T> g;
  g.nx = _nx;
  g.ny = _ny;
  g.nz = _nz;
  g.dx = _dx;
  g.dy = _dy;
  g.dz = _dz;
  g.px = _px;
  g.py = _py;
  g.pz = _pz;
  g.xmax = g.px + g.nx * g.dx;
  g.ymax = g.py + g.ny * g.dy;
  g.zmax = g.pz + g.nz * g.dz;
  g.xmin = _px;
  g.ymin = _py;
  g.zmin = _pz;
  return g;
}

/**
 * @brief Check if x is inside the grid.
 * @param g Grid parameters
 * @param x Coordinate
 * @tparam T Float number precision
 * @return True if x is inside the grid
 */
template <typename T>
__host__ __device__ bool validX(const Grid<T> &g, const T x) {
  return g.px < x && x < g.px + g.dx * g.nx;
}

/**
 * @brief Check if y is inside the grid.
 * @param g Grid parameters
 * @param y Coordinate
 * @tparam T Float number precision
 * @return True if y is inside the grid
 */
template <typename T>
__host__ __device__ bool validY(const Grid<T> &g, const T y) {
  return g.py < y && y < g.py + g.dy * g.ny;
}

/**
 * @brief Check if z is inside the grid.
 * @param g Grid parameters
 * @param z Coordinate
 * @tparam T Float number precision
 * @return True if z is inside the grid
 */
template <typename T>
__host__ __device__ bool validZ(const Grid<T> &g, const T z) {
  return g.pz < z && z < g.pz + g.dz * g.nz;
}

/**
 * @brief Get the total number of cells.
 * @param g Grid parameters
 * @tparam T Float number precision
 * @return Number of cells
 */
template <typename T> __host__ __device__ int numberOfCells(const Grid<T> &g) {
  return g.nx * g.ny * g.nz;
}

/**
 * @brief Get the total number of faces.
 * @param g Grid parameters
 * @tparam T Float number precision
 * @return Number of faces
 */
template <typename T> __host__ __device__ int numberOfFaces(const Grid<T> &g) {
  return g.nx * g.ny * g.nz + g.nx * g.ny + g.ny * g.nz + g.nx * g.nz;
}

/**
 * @brief Get the ID of the cell containing at given position
 * @param g Grid parameters
 * @param px Position x-coordinate
 * @param py Position y-coordinate
 * @param pz Position z-coordinate
 * @param idx ID along x
 * @param idy ID along y
 * @param idz ID along z
 * @tparam T Float number precision
 */
template <typename T>
__host__ __device__ void idPoint(const Grid<T> &g, T px, T py, T pz, int *idx,
                                 int *idy, int *idz) {
  *idx = floor((px - g.px) / g.dx);
  *idy = floor((py - g.py) / g.dy);
  *idz = floor((pz - g.pz) / g.dz);
}

/**
 * @brief Check if a given ID is valid (3 Ids)
 * @param g Grid parameters
 * @param idx ID along x
 * @param idy ID along y
 * @param idz ID along z
 * @tparam T Float number precision
 * @return True if the ID is valid
 */
template <typename T>
__host__ __device__ bool validId(const Grid<T> &g, int idx, int idy, int idz) {
  return idx >= 0 && idx < g.nx && idy >= 0 && idy < g.ny && idz >= 0 &&
         idz < g.nz;
}

/**
 * @brief Check if a given ID is valid (unique Id)
 * @param g Grid parameters
 * @param id Unique ID
 * @tparam T Float number precision
 * @return True if the ID is valid
 */
template <typename T>
__host__ __device__ bool validId(const Grid<T> &g, int id) {
  return id >= 0 && id < numberOfCells(g);
}

/**
 * @brief Given the unique ID of a cell, find the corresponding
 *        IDs along each principal direction.
 * @param g Grid parameters
 * @param id Unique ID
 * @param idx ID along x
 * @param idy ID along y
 * @param idz ID along z
 * @tparam T Float number precision
 */
template <typename T>
__host__ __device__ void splitId(const Grid<T> &g, int id, int *idx, int *idy,
                                 int *idz) {
  *idz = id / (g.nx * g.ny);
  id = id % (g.nx * g.ny);
  *idy = id / g.nx;
  id = id % g.nx;
  *idx = id;
}

/**
 * @brief Given the IDs along each principal direction,
 *        find the corresponding unique ID of a cell without
 *        performing validation.
 * @param g Grid parameters
 * @param idx ID along x
 * @param idy ID along y
 * @param idz ID along z
 * @tparam T Float number precision
 * @return Unique ID
 */
template <typename T>
__host__ __device__ int mergeId(const Grid<T> &g, int idx, int idy, int idz) {
  return idz * g.ny * g.nx + idy * g.nx + idx;
}

/**
 * @brief Given the IDs along each principal direction,
 *        find the corresponding unique ID of a cell
 *        performing validation. If the IDs are not valid,
 *        return the total number of cells.
 * @param g Grid parameters
 * @param idx ID along x
 * @param idy ID along y
 * @param idz ID along z
 * @tparam T Float number precision
 * @return Unique ID
 */
template <typename T>
__host__ __device__ int uniqueId(const Grid<T> &g, int idx, int idy, int idz) {
  return validId(g, idx, idy, idz) ? mergeId(g, idx, idy, idz)
                                   : numberOfCells(g);
}

/**
 * @brief Compute the volume of a single cell.
 * @param g Grid parameters
 * @tparam T Float number precision
 * @return Volume of a cell
 */
template <typename T> __host__ __device__ T volumeCell(const Grid<T> &g) {
  return g.dx * g.dy * g.dz;
}

/**
 * @brief Compute the volume of the whole grid.
 * @param g Grid parameters
 * @tparam T Float number precision
 * @return Volume of the grid
 */
template <typename T> __host__ __device__ T volumeGrid(const Grid<T> &g) {
  return volumeCell(g) * numberOfCells(g);
}

/**
 * @brief Compute the center point of a cell.
 * @param g Grid parameters
 * @param idx ID along x
 * @param idy ID along y
 * @param idz ID along z
 * @param px Center x-coordinate
 * @param py Center y-coordinate
 * @param pz Center z-coordinate
 * @tparam T Float number precision
 * @return Volume of the grid
 */
template <typename T>
__host__ __device__ void centerOfCell(const Grid<T> &g, int idx, int idy,
                                      int idz, T *px, T *py, T *pz) {
  *px = g.px + idx * g.dx + 0.5 * g.dx;
  *py = g.py + idy * g.dy + 0.5 * g.dy;
  *pz = g.pz + idz * g.dz + 0.5 * g.dz;
}

/**
 * @brief Direction needed to find the neighbors cells.
 */
enum Direction {
  XP = 0, // XP: cell increasing x
  XM,     // XM: cell decreasing x
  YP,
  YM,
  ZP,
  ZM
};

/**
 * @brief Find the ID of a neighbor cell.
 * @param g Grid parameters
 * @param idx ID along x
 * @param idy ID along y
 * @param idz ID along z
 * @tparam T Float number precision
 * @tparam dir Direction of the neighbor
 * @return Volume of the grid
 */
template <typename T, int dir>
__host__ __device__ int idNeighbor(const Grid<T> &g, int idx, int idy,
                                   int idz) {
  switch (dir) {
  case XP:
    return (idx != g.nx - 1) ? mergeId(g, idx + 1, idy, idz) : numberOfCells(g);
  case XM:
    return (idx != 0) ? mergeId(g, idx - 1, idy, idz) : numberOfCells(g);
  case YP:
    return (idy != g.ny - 1) ? mergeId(g, idx, idy + 1, idz) : numberOfCells(g);
  case YM:
    return (idy != 0) ? mergeId(g, idx, idy - 1, idz) : numberOfCells(g);
  case ZP:
    return (idz != g.nz - 1) ? mergeId(g, idx, idy, idz + 1) : numberOfCells(g);
  case ZM:
    return (idz != 0) ? mergeId(g, idx, idy, idz - 1) : numberOfCells(g);
  }
  return numberOfCells(g);
}

/**
 * @brief Find the cell containing a given point and check if it's valid
 * @param g Grid parameters
 * @param px Position x-coordinate
 * @param py Position y-coordinate
 * @param pz Position z-coordinate
 * @param idx Output: ID along x
 * @param idy Output: ID along y
 * @param idz Output: ID along z
 * @tparam T Float number precision
 * @return True if the point is inside a valid cell
 */
template <typename T>
__host__ __device__ bool findCell(const Grid<T> &g, T px, T py, T pz, int *idx,
                                  int *idy, int *idz) {
  // Calcular los índices de la celda
  idPoint(g, px, py, pz, idx, idy, idz);

  // Verificar si los índices son válidos
  return validId(g, *idx, *idy, *idz);
}
}; // namespace grid

} // namespace par2

#endif // PAR2_CARTESIANGRID_CUH

```

# par2\Geometry\CellField.cuh

```cuh
/**
 * @file CellField.cuh
 * @brief Header file for cellfield.
 *        A cellfield is a field that is defined at the center of
 *        every cells of the grid.
 *
 * @author Calogero B. Rizzo
 *
 * @copyright This file is part of the PAR2 software.
 *            Copyright (C) 2018 Calogero B. Rizzo
 *
 * @license This program is free software: you can redistribute it and/or modify
 *          it under the terms of the GNU General Public License as published by
 *          the Free Software Foundation, either version 3 of the License, or
 *          (at your option) any later version.
 *
 *          This program is distributed in the hope that it will be useful,
 *          but WITHOUT ANY WARRANTY; without even the implied warranty of
 *          MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *          GNU General Public License for more details.
 *
 *          You should have received a copy of the GNU General Public License
 *          along with this program.  If not, see
 * <http://www.gnu.org/licenses/>.
 */

#ifndef PAR2_CELLFIELD_CUH
#define PAR2_CELLFIELD_CUH

#include <algorithm>
#include <fstream>
#include <iostream>
#include <thrust/device_vector.h>
// #include <thrust/raw_pointer_cast.h>

#include "CartesianGrid.cuh"
#include "FaceField.cuh"
#include "Vector.cuh"

namespace par2 {

namespace cellfield {
/**
 * @brief Build the vector containing a cellfield.
 * @param g Grid where the field is defined
 * @param data Vector that contains the cellfield values
 * @param v Init value for data
 * @tparam T Float number precision
 * @tparam Vector Container for data vectors
 */
template <typename T, class Vector>
void build(const grid::Grid<T> &g, Vector &data, T v = 0) {
  int size = g.nx * g.ny * g.nz;
  data.resize(size, v);
}

/**
 * @brief Compute drift correction in each cell.
 * @param g Grid where the field is defined
 * @param datax Vector that contains the values on the interfaces
 *        orthogonal to the x-axis
 * @param datay Vector that contains the values on the interfaces
 *        orthogonal to the y-axis
 * @param dataz Vector that contains the values on the interfaces
 *        orthogonal to the z-axis
 * @param driftx Vector that contains the x-values of the correction
 * @param drifty Vector that contains the y-values of the correction
 * @param driftz Vector that contains the z-values of the correction
 * @param Dm Effective molecular diffusion
 * @param alphaL Longitudinal dispersivity
 * @param alphaT Transverse dispersivity
 * @tparam T Float number precision
 * @tparam Vector Container for data vectors
 */
template <typename T, class Vector>
void computeDriftCorrection(const grid::Grid<T> &g, const Vector &datax,
                            const Vector &datay, const Vector &dataz,
                            Vector &driftx, Vector &drifty, Vector &driftz,
                            T Dm, T alphaL, T alphaT) {
  Vector D11, D22, D33;
  Vector D12, D13, D23;
  build(g, D11);
  build(g, D22);
  build(g, D33);
  build(g, D12);
  build(g, D13);
  build(g, D23);

  // Compute tensor D in all the cells
  for (auto idz = 0; idz < g.nz; idz++) {
    for (auto idy = 0; idy < g.ny; idy++) {
      for (auto idx = 0; idx < g.nx; idx++) {
        T cx, cy, cz;
        grid::centerOfCell<T>(g, idx, idy, idz, &cx, &cy, &cz);

        T vx, vy, vz;
        par2::facefield::in<T>(datax.data(), datay.data(), dataz.data(), g, idx,
                               idy, idz, true, cx, cy, cz, &vx, &vy, &vz);

        int id = grid::mergeId(g, idx, idy, idz);
        T vnorm = par2::vector::norm(vx, vy, vz);

        D11[id] = (alphaT * vnorm + Dm) + (alphaL - alphaT) * vx * vx / vnorm;
        D22[id] = (alphaT * vnorm + Dm) + (alphaL - alphaT) * vy * vy / vnorm;
        D33[id] = (alphaT * vnorm + Dm) + (alphaL - alphaT) * vz * vz / vnorm;
        D12[id] = (alphaL - alphaT) * vx * vy / vnorm;
        D13[id] = (alphaL - alphaT) * vx * vz / vnorm;
        D23[id] = (alphaL - alphaT) * vy * vz / vnorm;
      }
    }
  }

  // Compute drift correction term
  for (auto idz = 0; idz < g.nz; idz++) {
    for (auto idy = 0; idy < g.ny; idy++) {
      for (auto idx = 0; idx < g.nx; idx++) {
        int id1, id2;

        // Derivatives in x-direction
        T dD11x, dD12x, dD13x;

        T ddx;
        int idx1, idx2;
        if (idx == 0) {
          ddx = g.dx;
          idx1 = idx;
          idx2 = idx + 1;
        } else if (idx == g.nx - 1) {
          ddx = g.dx;
          idx1 = idx - 1;
          idx2 = idx;
        } else {
          ddx = 2 * g.dx;
          idx1 = idx - 1;
          idx2 = idx + 1;
        }

        id1 = grid::mergeId(g, idx1, idy, idz);
        id2 = grid::mergeId(g, idx2, idy, idz);

        dD11x = (D11[id2] - D11[id1]) / ddx;
        dD12x = (D12[id2] - D12[id1]) / ddx;
        dD13x = (D13[id2] - D13[id1]) / ddx;

        // Derivatives in y-direction
        T dD12y, dD22y, dD23y;

        T ddy;
        int idy1, idy2;
        if (idy == 0) {
          ddy = g.dy;
          idy1 = idy;
          idy2 = idy + 1;
        } else if (idy == g.ny - 1) {
          ddy = g.dy;
          idy1 = idy - 1;
          idy2 = idy;
        } else {
          ddy = 2 * g.dy;
          idy1 = idy - 1;
          idy2 = idy + 1;
        }

        id1 = grid::mergeId(g, idx, idy1, idz);
        id2 = grid::mergeId(g, idx, idy2, idz);

        dD12y = (D12[id2] - D12[id1]) / ddy;
        dD22y = (D22[id2] - D22[id1]) / ddy;
        dD23y = (D23[id2] - D23[id1]) / ddy;

        // Derivatives in y-direction
        T dD13z, dD23z, dD33z;

        // TODO fix 2D case
        T ddz;
        int idz1, idz2;
        if (idz == 0) {
          ddz = g.dz;
          idz1 = idz;
          idz2 = idz + 1;
        } else if (idz == g.nz - 1) {
          ddz = g.dz;
          idz1 = idz - 1;
          idz2 = idz;
        } else {
          ddz = 2 * g.dz;
          idz1 = idz - 1;
          idz2 = idz + 1;
        }

        id1 = grid::mergeId(g, idx, idy, idz1);
        id2 = grid::mergeId(g, idx, idy, idz2);

        dD13z = (D13[id2] - D13[id1]) / ddz;
        dD23z = (D23[id2] - D23[id1]) / ddz;
        dD33z = (D33[id2] - D33[id1]) / ddz;

        // Compute drift coefficient in the cell
        int id = grid::mergeId(g, idx, idy, idz);

        driftx[id] = dD11x + dD12y + dD13z;
        drifty[id] = dD12x + dD22y + dD23z;
        driftz[id] = dD13x + dD23y + dD33z;
      }
    }
  }
}

// Kernel para calcular el tensor de dispersión
template <typename T>
__global__ void
computeDispersionTensorKernel(const grid::Grid<T> g, const T *dataxPtr,
                              const T *datayPtr, const T *datazPtr, T *D11Ptr,
                              T *D22Ptr, T *D33Ptr, T *D12Ptr, T *D13Ptr,
                              T *D23Ptr, T Dm, T alphaL, T alphaT) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;
  int idz = blockIdx.z * blockDim.z + threadIdx.z;

  if (idx < g.nx && idy < g.ny && idz < g.nz) {
    T cx, cy, cz;
    grid::centerOfCell<T>(g, idx, idy, idz, &cx, &cy, &cz);

    T vx, vy, vz;
    par2::facefield::in<T>(dataxPtr, datayPtr, datazPtr, g, idx, idy, idz, true,
                           cx, cy, cz, &vx, &vy, &vz);

    int id = grid::mergeId(g, idx, idy, idz);
    T vnorm = par2::vector::norm(vx, vy, vz);

    // Evitar división por cero si la velocidad es muy pequeña
    if (vnorm < 1e-10) {
      vnorm = 1e-10;
    }

    D11Ptr[id] = (alphaT * vnorm + Dm) + (alphaL - alphaT) * vx * vx / vnorm;
    D22Ptr[id] = (alphaT * vnorm + Dm) + (alphaL - alphaT) * vy * vy / vnorm;
    D33Ptr[id] = (alphaT * vnorm + Dm) + (alphaL - alphaT) * vz * vz / vnorm;
    D12Ptr[id] = (alphaL - alphaT) * vx * vy / vnorm;
    D13Ptr[id] = (alphaL - alphaT) * vx * vz / vnorm;
    D23Ptr[id] = (alphaL - alphaT) * vy * vz / vnorm;
  }
}

// Kernel para calcular la corrección de deriva
template <typename T>
__global__ void
computeDriftCorrectionKernel(const grid::Grid<T> g, const T *D11Ptr,
                             const T *D22Ptr, const T *D33Ptr, const T *D12Ptr,
                             const T *D13Ptr, const T *D23Ptr, T *driftxPtr,
                             T *driftyPtr, T *driftzPtr) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;
  int idz = blockIdx.z * blockDim.z + threadIdx.z;

  if (idx < g.nx && idy < g.ny && idz < g.nz) {
    int id1, id2;

    // Derivadas en dirección x
    T dD11x, dD12x, dD13x;
    T ddx;
    int idx1, idx2;

    if (idx == 0) {
      ddx = g.dx;
      idx1 = idx;
      idx2 = idx + 1;
    } else if (idx == g.nx - 1) {
      ddx = g.dx;
      idx1 = idx - 1;
      idx2 = idx;
    } else {
      ddx = 2 * g.dx;
      idx1 = idx - 1;
      idx2 = idx + 1;
    }

    id1 = grid::mergeId(g, idx1, idy, idz);
    id2 = grid::mergeId(g, idx2, idy, idz);

    dD11x = (D11Ptr[id2] - D11Ptr[id1]) / ddx;
    dD12x = (D12Ptr[id2] - D12Ptr[id1]) / ddx;
    dD13x = (D13Ptr[id2] - D13Ptr[id1]) / ddx;

    // Derivadas en dirección y
    T dD12y, dD22y, dD23y;
    T ddy;
    int idy1, idy2;

    if (idy == 0) {
      ddy = g.dy;
      idy1 = idy;
      idy2 = idy + 1;
    } else if (idy == g.ny - 1) {
      ddy = g.dy;
      idy1 = idy - 1;
      idy2 = idy;
    } else {
      ddy = 2 * g.dy;
      idy1 = idy - 1;
      idy2 = idy + 1;
    }

    id1 = grid::mergeId(g, idx, idy1, idz);
    id2 = grid::mergeId(g, idx, idy2, idz);

    dD12y = (D12Ptr[id2] - D12Ptr[id1]) / ddy;
    dD22y = (D22Ptr[id2] - D22Ptr[id1]) / ddy;
    dD23y = (D23Ptr[id2] - D23Ptr[id1]) / ddy;

    // Derivadas en dirección z
    T dD13z, dD23z, dD33z;
    T ddz;
    int idz1, idz2;

    if (idz == 0) {
      ddz = g.dz;
      idz1 = idz;
      idz2 = idz + 1;
    } else if (idz == g.nz - 1) {
      ddz = g.dz;
      idz1 = idz - 1;
      idz2 = idz;
    } else {
      ddz = 2 * g.dz;
      idz1 = idz - 1;
      idz2 = idz + 1;
    }

    id1 = grid::mergeId(g, idx, idy, idz1);
    id2 = grid::mergeId(g, idx, idy, idz2);

    dD13z = (D13Ptr[id2] - D13Ptr[id1]) / ddz;
    dD23z = (D23Ptr[id2] - D23Ptr[id1]) / ddz;
    dD33z = (D33Ptr[id2] - D33Ptr[id1]) / ddz;

    // Calcular coeficiente de deriva
    int id = grid::mergeId(g, idx, idy, idz);

    driftxPtr[id] = dD11x + dD12y + dD13z;
    driftyPtr[id] = dD12x + dD22y + dD23z;
    driftzPtr[id] = dD13x + dD23y + dD33z;
  }
}

/**
 * @brief Compute drift correction in each cell (GPU version).
 * @param g Grid where the field is defined
 * @param datax Vector that contains the values on the interfaces
 *        orthogonal to the x-axis
 * @param datay Vector that contains the values on the interfaces
 *        orthogonal to the y-axis
 * @param dataz Vector that contains the values on the interfaces
 *        orthogonal to the z-axis
 * @param driftx Vector that contains the x-values of the correction
 * @param drifty Vector that contains the y-values of the correction
 * @param driftz Vector that contains the z-values of the correction
 * @param Dm Effective molecular diffusion
 * @param alphaL Longitudinal dispersivity
 * @param alphaT Transverse dispersivity
 * @tparam T Float number precision
 */
template <typename T>
void computeDriftCorrectionGPU(const grid::Grid<T> &g,
                               const thrust::device_vector<T> &datax,
                               const thrust::device_vector<T> &datay,
                               const thrust::device_vector<T> &dataz,
                               thrust::device_vector<T> &driftx,
                               thrust::device_vector<T> &drifty,
                               thrust::device_vector<T> &driftz, T Dm, T alphaL,
                               T alphaT) {
  // Crear vectores para el tensor de dispersión en device
  thrust::device_vector<T> D11, D22, D33;
  thrust::device_vector<T> D12, D13, D23;

  build(g, D11);
  build(g, D22);
  build(g, D33);
  build(g, D12);
  build(g, D13);
  build(g, D23);

  // Obtener punteros raw para usar en los kernels
  const T *dataxPtr = thrust::raw_pointer_cast(datax.data());
  const T *datayPtr = thrust::raw_pointer_cast(datay.data());
  const T *datazPtr = thrust::raw_pointer_cast(dataz.data());

  T *D11Ptr = thrust::raw_pointer_cast(D11.data());
  T *D22Ptr = thrust::raw_pointer_cast(D22.data());
  T *D33Ptr = thrust::raw_pointer_cast(D33.data());
  T *D12Ptr = thrust::raw_pointer_cast(D12.data());
  T *D13Ptr = thrust::raw_pointer_cast(D13.data());
  T *D23Ptr = thrust::raw_pointer_cast(D23.data());

  T *driftxPtr = thrust::raw_pointer_cast(driftx.data());
  T *driftyPtr = thrust::raw_pointer_cast(drifty.data());
  T *driftzPtr = thrust::raw_pointer_cast(driftz.data());

  // Configurar la dimensión de los bloques y la grid
  dim3 blockSize(8, 8, 8);
  dim3 gridSize((g.nx + blockSize.x - 1) / blockSize.x,
                (g.ny + blockSize.y - 1) / blockSize.y,
                (g.nz + blockSize.z - 1) / blockSize.z);

  // Lanzar kernel para calcular tensor de dispersión
  computeDispersionTensorKernel<<<gridSize, blockSize>>>(
      g, dataxPtr, datayPtr, datazPtr, D11Ptr, D22Ptr, D33Ptr, D12Ptr, D13Ptr,
      D23Ptr, Dm, alphaL, alphaT);

  // Comprobar errores
  cudaDeviceSynchronize();
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error in computeDispersionTensorKernel: %s\n",
           cudaGetErrorString(error));
  }

  // Lanzar kernel para calcular corrección de deriva
  computeDriftCorrectionKernel<<<gridSize, blockSize>>>(
      g, D11Ptr, D22Ptr, D33Ptr, D12Ptr, D13Ptr, D23Ptr, driftxPtr, driftyPtr,
      driftzPtr);

  // Comprobar errores
  cudaDeviceSynchronize();
  error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error in computeDriftCorrectionKernel: %s\n",
           cudaGetErrorString(error));
  }
}

}; // namespace cellfield
}; // namespace par2

#endif // PAR2_CELLFIELD_CUH

```

# par2\Geometry\CornerField.cuh

```cuh
/**
 * @file CornerField.cuh
 * @brief Header file for cornerfield.
 *        A cornerfield is a field that is defined at the each corner
 *        of every cells of the grid.
 *
 * @author Calogero B. Rizzo
 *
 * @copyright This file is part of the PAR2 software.
 *            Copyright (C) 2018 Calogero B. Rizzo
 *
 * @license This program is free software: you can redistribute it and/or modify
 *          it under the terms of the GNU General Public License as published by
 *          the Free Software Foundation, either version 3 of the License, or
 *          (at your option) any later version.
 *
 *          This program is distributed in the hope that it will be useful,
 *          but WITHOUT ANY WARRANTY; without even the implied warranty of
 *          MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *          GNU General Public License for more details.
 *
 *          You should have received a copy of the GNU General Public License
 *          along with this program.  If not, see
 * <http://www.gnu.org/licenses/>.
 */

#ifndef PAR2_CORNERFIELD_CUH
#define PAR2_CORNERFIELD_CUH

#include <fstream>
#include <iostream>
#include <thrust/device_vector.h>

#include "CartesianGrid.cuh"
#include "FaceField.cuh"
#include "Vector.cuh"

namespace par2 {

namespace cornerfield {
/**
 * @brief Build the vector containing a cornerfield.
 * @param g Grid where the field is defined
 * @param data Vector that contains the cornerfield values
 * @param v Init value for data
 * @tparam T Float number precision
 * @tparam Vector Container for data vectors
 */
template <typename T, class Vector>
void build(const grid::Grid<T> &g, Vector &data, T v = 0) {
  int size = (g.nx + 1) * (g.ny + 1) * (g.nz + 1);
  data.resize(size, v);
}

/**
 * @brief Get the position on the data vector for the corner
 *        defined by the IDs in each direction.
 * @param g Grid where the field is defined
 * @param idx ID along x
 * @param idy ID along y
 * @param idz ID along z
 * @tparam T Float number precision
 * @return The position on the data vector
 */
template <typename T>
__host__ __device__ int mergeId(const grid::Grid<T> &g, int idx, int idy,
                                int idz) {
  return idz * (g.ny + 1) * (g.nx + 1) + idy * (g.nx + 1) + idx;
}

/**
 * @brief Identification for each corner. Each cell has
 *        8 corners.
 */
enum Direction {
  C000 = 0, // CXYZ
  C100,
  C010,
  C110,
  C001,
  C101,
  C011,
  C111
};

/**
 * @brief Get the value of the cornerfield at given corner.
 *        The location is found from the IDs of the cell and the
 *        direction of the corner.
 * @param data Vector that contains the cornerfield values
 * @param g Grid where the field is defined
 * @param idx ID along x
 * @param idy ID along y
 * @param idz ID along z
 * @tparam T Float number precision
 * @tparam dir Direction of the interface
 * @return The value of the cornerfield at the corner
 */
template <typename T, int dir>
__host__ __device__ T get(const T *data, const grid::Grid<T> &g, int idx,
                          int idy, int idz) {
  int id;
  switch (dir) {
  case Direction::C000:
    id = cornerfield::mergeId(g, idx, idy, idz);
    return data[id];
  case Direction::C100:
    id = cornerfield::mergeId(g, idx + 1, idy, idz);
    return data[id];
  case Direction::C010:
    id = cornerfield::mergeId(g, idx, idy + 1, idz);
    return data[id];
  case Direction::C110:
    id = cornerfield::mergeId(g, idx + 1, idy + 1, idz);
    return data[id];
  case Direction::C001:
    id = cornerfield::mergeId(g, idx, idy, idz + 1);
    return data[id];
  case Direction::C101:
    id = cornerfield::mergeId(g, idx + 1, idy, idz + 1);
    return data[id];
  case Direction::C011:
    id = cornerfield::mergeId(g, idx, idy + 1, idz + 1);
    return data[id];
  case Direction::C111:
    id = cornerfield::mergeId(g, idx + 1, idy + 1, idz + 1);
    return data[id];
  }
  return 0;
}

/**
 * @brief Get the value of the field at given point using trilinear
 *        interpolation.
 * @param cdatax Vector that contains the x-values of the field
 * @param cdatay Vector that contains the y-values of the field
 * @param cdataz Vector that contains the z-values of the field
 * @param g Grid where the field is defined
 * @param idx ID along x
 * @param idy ID along y
 * @param idz ID along z
 * @param idValid True if the position is inside the grid
 * @param px Position x-coordinate
 * @param py Position y-coordinate
 * @param pz Position z-coordinate
 * @param vx Result x-coordinate
 * @param vy Result y-coordinate
 * @param vz Result z-coordinate
 * @tparam T Float number precision
 */
template <typename T>
__host__ __device__ void in(const T *cdatax, const T *cdatay, const T *cdataz,
                            const grid::Grid<T> &g, int idx, int idy, int idz,
                            bool idValid, T px, T py, T pz, T *vx, T *vy,
                            T *vz) {
  using Direction = cornerfield::Direction;

  T cx, cy, cz;
  grid::centerOfCell<T>(g, idx, idy, idz, &cx, &cy, &cz);

  T x = (px - cx) / g.dx + 0.5;
  T y = (py - cy) / g.dy + 0.5;
  T z = (pz - cz) / g.dz + 0.5;

  *vx = idValid
            ? interpolation::trilinear(
                  x, y, z, get<T, Direction::C000>(cdatax, g, idx, idy, idz),
                  get<T, Direction::C100>(cdatax, g, idx, idy, idz),
                  get<T, Direction::C010>(cdatax, g, idx, idy, idz),
                  get<T, Direction::C110>(cdatax, g, idx, idy, idz),
                  get<T, Direction::C001>(cdatax, g, idx, idy, idz),
                  get<T, Direction::C101>(cdatax, g, idx, idy, idz),
                  get<T, Direction::C011>(cdatax, g, idx, idy, idz),
                  get<T, Direction::C111>(cdatax, g, idx, idy, idz))
            : 0;

  *vy = idValid
            ? interpolation::trilinear(
                  x, y, z, get<T, Direction::C000>(cdatay, g, idx, idy, idz),
                  get<T, Direction::C100>(cdatay, g, idx, idy, idz),
                  get<T, Direction::C010>(cdatay, g, idx, idy, idz),
                  get<T, Direction::C110>(cdatay, g, idx, idy, idz),
                  get<T, Direction::C001>(cdatay, g, idx, idy, idz),
                  get<T, Direction::C101>(cdatay, g, idx, idy, idz),
                  get<T, Direction::C011>(cdatay, g, idx, idy, idz),
                  get<T, Direction::C111>(cdatay, g, idx, idy, idz))
            : 0;

  *vz = idValid
            ? interpolation::trilinear(
                  x, y, z, get<T, Direction::C000>(cdataz, g, idx, idy, idz),
                  get<T, Direction::C100>(cdataz, g, idx, idy, idz),
                  get<T, Direction::C010>(cdataz, g, idx, idy, idz),
                  get<T, Direction::C110>(cdataz, g, idx, idy, idz),
                  get<T, Direction::C001>(cdataz, g, idx, idy, idz),
                  get<T, Direction::C101>(cdataz, g, idx, idy, idz),
                  get<T, Direction::C011>(cdataz, g, idx, idy, idz),
                  get<T, Direction::C111>(cdataz, g, idx, idy, idz))
            : 0;
}

/**
 * @brief Compute velocity correction for the particle tracking.
 * @param cdatax Vector that contains the x-values of the field
 * @param cdatay Vector that contains the y-values of the field
 * @param cdataz Vector that contains the z-values of the field
 * @param g Grid where the field is defined
 * @param idx ID along x
 * @param idy ID along y
 * @param idz ID along z
 * @param idValid True if the position is inside the grid
 * @param px Position x-coordinate
 * @param py Position y-coordinate
 * @param pz Position z-coordinate
 * @param Dm Effective molecular diffusion
 * @param alphaL Longitudinal dispersivity
 * @param alphaT Transverse dispersivity
 * @param vx Result x-coordinate
 * @param vy Result y-coordinate
 * @param vz Result z-coordinate
 * @tparam T Float number precision
 */
template <typename T>
__host__ __device__ void
velocityCorrection(const T *cdatax, const T *cdatay, const T *cdataz,
                   const grid::Grid<T> &g, int idx, int idy, int idz,
                   bool idValid, T px, T py, T pz, T Dm, T alphaL, T alphaT,
                   T *vx, T *vy, T *vz) {
  using Direction = cornerfield::Direction;

  T cx, cy, cz;
  grid::centerOfCell<T>(g, idx, idy, idz, &cx, &cy, &cz);

  T x = (px - cx) / g.dx + 0.5;
  T y = (py - cy) / g.dy + 0.5;
  T z = (pz - cz) / g.dz + 0.5;

  T vx000 = idValid ? get<T, Direction::C000>(cdatax, g, idx, idy, idz) : 1;
  T vx100 = idValid ? get<T, Direction::C100>(cdatax, g, idx, idy, idz) : 1;
  T vx010 = idValid ? get<T, Direction::C010>(cdatax, g, idx, idy, idz) : 1;
  T vx110 = idValid ? get<T, Direction::C110>(cdatax, g, idx, idy, idz) : 1;
  T vx001 = idValid ? get<T, Direction::C001>(cdatax, g, idx, idy, idz) : 1;
  T vx101 = idValid ? get<T, Direction::C101>(cdatax, g, idx, idy, idz) : 1;
  T vx011 = idValid ? get<T, Direction::C011>(cdatax, g, idx, idy, idz) : 1;
  T vx111 = idValid ? get<T, Direction::C111>(cdatax, g, idx, idy, idz) : 1;

  T vy000 = idValid ? get<T, Direction::C000>(cdatay, g, idx, idy, idz) : 1;
  T vy100 = idValid ? get<T, Direction::C100>(cdatay, g, idx, idy, idz) : 1;
  T vy010 = idValid ? get<T, Direction::C010>(cdatay, g, idx, idy, idz) : 1;
  T vy110 = idValid ? get<T, Direction::C110>(cdatay, g, idx, idy, idz) : 1;
  T vy001 = idValid ? get<T, Direction::C001>(cdatay, g, idx, idy, idz) : 1;
  T vy101 = idValid ? get<T, Direction::C101>(cdatay, g, idx, idy, idz) : 1;
  T vy011 = idValid ? get<T, Direction::C011>(cdatay, g, idx, idy, idz) : 1;
  T vy111 = idValid ? get<T, Direction::C111>(cdatay, g, idx, idy, idz) : 1;

  T vz000 = idValid ? get<T, Direction::C000>(cdataz, g, idx, idy, idz) : 1;
  T vz100 = idValid ? get<T, Direction::C100>(cdataz, g, idx, idy, idz) : 1;
  T vz010 = idValid ? get<T, Direction::C010>(cdataz, g, idx, idy, idz) : 1;
  T vz110 = idValid ? get<T, Direction::C110>(cdataz, g, idx, idy, idz) : 1;
  T vz001 = idValid ? get<T, Direction::C001>(cdataz, g, idx, idy, idz) : 1;
  T vz101 = idValid ? get<T, Direction::C101>(cdataz, g, idx, idy, idz) : 1;
  T vz011 = idValid ? get<T, Direction::C011>(cdataz, g, idx, idy, idz) : 1;
  T vz111 = idValid ? get<T, Direction::C111>(cdataz, g, idx, idy, idz) : 1;

  // If velocity is zero in one corner, add eps to vx component to avoid nan
  const T toll = 0.01 * Dm / alphaL;

  vx000 = (vx000 < toll && vy000 < toll && vz000 < toll) ? toll : vx000;
  vx100 = (vx100 < toll && vy100 < toll && vz100 < toll) ? toll : vx100;
  vx010 = (vx010 < toll && vy010 < toll && vz010 < toll) ? toll : vx010;
  vx110 = (vx110 < toll && vy110 < toll && vz110 < toll) ? toll : vx110;
  vx001 = (vx001 < toll && vy001 < toll && vz001 < toll) ? toll : vx001;
  vx101 = (vx101 < toll && vy101 < toll && vz101 < toll) ? toll : vx101;
  vx011 = (vx011 < toll && vy011 < toll && vz011 < toll) ? toll : vx011;
  vx111 = (vx111 < toll && vy111 < toll && vz111 < toll) ? toll : vx111;

  T vnorm000 = par2::vector::norm(vx000, vy000, vz000);
  T vnorm100 = par2::vector::norm(vx100, vy100, vz100);
  T vnorm010 = par2::vector::norm(vx010, vy010, vz010);
  T vnorm110 = par2::vector::norm(vx110, vy110, vz110);
  T vnorm001 = par2::vector::norm(vx001, vy001, vz001);
  T vnorm101 = par2::vector::norm(vx101, vy101, vz101);
  T vnorm011 = par2::vector::norm(vx011, vy011, vz011);
  T vnorm111 = par2::vector::norm(vx111, vy111, vz111);

  T dDxxx = interpolation::trilinearDevX(
      x, y, z, g.dx,
      (alphaT * vnorm000 + Dm) + (alphaL - alphaT) * vx000 * vx000 / vnorm000,
      (alphaT * vnorm100 + Dm) + (alphaL - alphaT) * vx100 * vx100 / vnorm100,
      (alphaT * vnorm010 + Dm) + (alphaL - alphaT) * vx010 * vx010 / vnorm010,
      (alphaT * vnorm110 + Dm) + (alphaL - alphaT) * vx110 * vx110 / vnorm110,
      (alphaT * vnorm001 + Dm) + (alphaL - alphaT) * vx001 * vx001 / vnorm001,
      (alphaT * vnorm101 + Dm) + (alphaL - alphaT) * vx101 * vx101 / vnorm101,
      (alphaT * vnorm011 + Dm) + (alphaL - alphaT) * vx011 * vx011 / vnorm011,
      (alphaT * vnorm111 + Dm) + (alphaL - alphaT) * vx111 * vx111 / vnorm111);

  T dDyyy = interpolation::trilinearDevY(
      x, y, z, g.dy,
      (alphaT * vnorm000 + Dm) + (alphaL - alphaT) * vy000 * vy000 / vnorm000,
      (alphaT * vnorm100 + Dm) + (alphaL - alphaT) * vy100 * vy100 / vnorm100,
      (alphaT * vnorm010 + Dm) + (alphaL - alphaT) * vy010 * vy010 / vnorm010,
      (alphaT * vnorm110 + Dm) + (alphaL - alphaT) * vy110 * vy110 / vnorm110,
      (alphaT * vnorm001 + Dm) + (alphaL - alphaT) * vy001 * vy001 / vnorm001,
      (alphaT * vnorm101 + Dm) + (alphaL - alphaT) * vy101 * vy101 / vnorm101,
      (alphaT * vnorm011 + Dm) + (alphaL - alphaT) * vy011 * vy011 / vnorm011,
      (alphaT * vnorm111 + Dm) + (alphaL - alphaT) * vy111 * vy111 / vnorm111);

  T dDzzz = interpolation::trilinearDevZ(
      x, y, z, g.dz,
      (alphaT * vnorm000 + Dm) + (alphaL - alphaT) * vz000 * vz000 / vnorm000,
      (alphaT * vnorm100 + Dm) + (alphaL - alphaT) * vz100 * vz100 / vnorm100,
      (alphaT * vnorm010 + Dm) + (alphaL - alphaT) * vz010 * vz010 / vnorm010,
      (alphaT * vnorm110 + Dm) + (alphaL - alphaT) * vz110 * vz110 / vnorm110,
      (alphaT * vnorm001 + Dm) + (alphaL - alphaT) * vz001 * vz001 / vnorm001,
      (alphaT * vnorm101 + Dm) + (alphaL - alphaT) * vz101 * vz101 / vnorm101,
      (alphaT * vnorm011 + Dm) + (alphaL - alphaT) * vz011 * vz011 / vnorm011,
      (alphaT * vnorm111 + Dm) + (alphaL - alphaT) * vz111 * vz111 / vnorm111);

  T dDxyx = interpolation::trilinearDevX(
      x, y, z, g.dx, (alphaL - alphaT) * vx000 * vy000 / vnorm000,
      (alphaL - alphaT) * vx100 * vy100 / vnorm100,
      (alphaL - alphaT) * vx010 * vy010 / vnorm010,
      (alphaL - alphaT) * vx110 * vy110 / vnorm110,
      (alphaL - alphaT) * vx001 * vy001 / vnorm001,
      (alphaL - alphaT) * vx101 * vy101 / vnorm101,
      (alphaL - alphaT) * vx011 * vy011 / vnorm011,
      (alphaL - alphaT) * vx111 * vy111 / vnorm111);

  T dDxyy = interpolation::trilinearDevY(
      x, y, z, g.dy, (alphaL - alphaT) * vx000 * vy000 / vnorm000,
      (alphaL - alphaT) * vx100 * vy100 / vnorm100,
      (alphaL - alphaT) * vx010 * vy010 / vnorm010,
      (alphaL - alphaT) * vx110 * vy110 / vnorm110,
      (alphaL - alphaT) * vx001 * vy001 / vnorm001,
      (alphaL - alphaT) * vx101 * vy101 / vnorm101,
      (alphaL - alphaT) * vx011 * vy011 / vnorm011,
      (alphaL - alphaT) * vx111 * vy111 / vnorm111);

  T dDxzx = interpolation::trilinearDevX(
      x, y, z, g.dx, (alphaL - alphaT) * vx000 * vz000 / vnorm000,
      (alphaL - alphaT) * vx100 * vz100 / vnorm100,
      (alphaL - alphaT) * vx010 * vz010 / vnorm010,
      (alphaL - alphaT) * vx110 * vz110 / vnorm110,
      (alphaL - alphaT) * vx001 * vz001 / vnorm001,
      (alphaL - alphaT) * vx101 * vz101 / vnorm101,
      (alphaL - alphaT) * vx011 * vz011 / vnorm011,
      (alphaL - alphaT) * vx111 * vz111 / vnorm111);

  T dDxzz = interpolation::trilinearDevZ(
      x, y, z, g.dz, (alphaL - alphaT) * vx000 * vz000 / vnorm000,
      (alphaL - alphaT) * vx100 * vz100 / vnorm100,
      (alphaL - alphaT) * vx010 * vz010 / vnorm010,
      (alphaL - alphaT) * vx110 * vz110 / vnorm110,
      (alphaL - alphaT) * vx001 * vz001 / vnorm001,
      (alphaL - alphaT) * vx101 * vz101 / vnorm101,
      (alphaL - alphaT) * vx011 * vz011 / vnorm011,
      (alphaL - alphaT) * vx111 * vz111 / vnorm111);

  T dDyzy = interpolation::trilinearDevY(
      x, y, z, g.dy, (alphaL - alphaT) * vy000 * vz000 / vnorm000,
      (alphaL - alphaT) * vy100 * vz100 / vnorm100,
      (alphaL - alphaT) * vy010 * vz010 / vnorm010,
      (alphaL - alphaT) * vy110 * vz110 / vnorm110,
      (alphaL - alphaT) * vy001 * vz001 / vnorm001,
      (alphaL - alphaT) * vy101 * vz101 / vnorm101,
      (alphaL - alphaT) * vy011 * vz011 / vnorm011,
      (alphaL - alphaT) * vy111 * vz111 / vnorm111);

  T dDyzz = interpolation::trilinearDevZ(
      x, y, z, g.dz, (alphaL - alphaT) * vy000 * vz000 / vnorm000,
      (alphaL - alphaT) * vy100 * vz100 / vnorm100,
      (alphaL - alphaT) * vy010 * vz010 / vnorm010,
      (alphaL - alphaT) * vy110 * vz110 / vnorm110,
      (alphaL - alphaT) * vy001 * vz001 / vnorm001,
      (alphaL - alphaT) * vy101 * vz101 / vnorm101,
      (alphaL - alphaT) * vy011 * vz011 / vnorm011,
      (alphaL - alphaT) * vy111 * vz111 / vnorm111);

  *vx = dDxxx + dDxyy + dDxzz;
  *vy = dDxyx + dDyyy + dDyzz;
  *vz = dDxzx + dDyzy + dDzzz;
}

/**
 * @brief Compute displacement matrix for the particle tracking.
 * @param cdatax Vector that contains the x-values of the field
 * @param cdatay Vector that contains the y-values of the field
 * @param cdataz Vector that contains the z-values of the field
 * @param g Grid where the field is defined
 * @param idx ID along x
 * @param idy ID along y
 * @param idz ID along z
 * @param idValid True if the position is inside the grid
 * @param px Position x-coordinate
 * @param py Position y-coordinate
 * @param pz Position z-coordinate
 * @param Dm Effective molecular diffusion
 * @param alphaL Longitudinal dispersivity
 * @param alphaT Transverse dispersivity
 * @param dt Time step
 * @param B00 Matrix component 00
 * @param B00 Matrix component 11
 * @param B00 Matrix component 22
 * @param B00 Matrix component 01
 * @param B00 Matrix component 02
 * @param B00 Matrix component 12
 * @tparam T Float number precision
 */
template <typename T>
__host__ __device__ void
displacementMatrix(const T *cdatax, const T *cdatay, const T *cdataz,
                   const grid::Grid<T> &g, int idx, int idy, int idz,
                   bool idValid, T px, T py, T pz, T Dm, T alphaL, T alphaT,
                   T dt, T *B00, T *B11, T *B22, T *B01, T *B02, T *B12) {
  T vx, vy, vz;
  cornerfield::in<T>(cdatax, cdatay, cdataz, g, idx, idy, idz, idValid, px, py,
                     pz, &vx, &vy, &vz);

  const T toll = 0.01 * Dm / alphaL;

  vx = (vx < toll) ? toll : vx;

  T vnorm2 = vector::norm2<T>(vx, vy, vz);
  T vnorm = sqrt(vnorm2);

  // Compute dispersion Matrix
  T alpha = alphaT * vnorm + Dm;
  T beta = (alphaL - alphaT) / vnorm; // Check threshold needed

  // Eigenvectors
  // vx0 == vx, etc.

  T vx1 = -vy;
  T vy1 = vx;
  T vz1 = 0.0;

  T vx2 = -vz * vx;
  T vy2 = -vz * vy;
  T vz2 = vx * vx + vy * vy;

  // vnorm2_0 == vnorm2
  T vnorm2_1 = vector::norm2<T>(vx1, vy1, vz1);
  T vnorm2_2 = vector::norm2<T>(vx2, vy2, vz2);

  T gamma0 = sqrt(alpha + beta * vnorm2) / vnorm2;
  T gamma1 = sqrt(alpha) / vnorm2_1;
  T gamma2 = sqrt(alpha) / vnorm2_2;

  // Pre-coefficient
  T coeff = sqrt(2.0 * dt);

  // Dispersion Matrix
  *B00 = coeff * (gamma0 * vx * vx + gamma1 * vx1 * vx1 + gamma2 * vx2 * vx2);

  *B11 = coeff * (gamma0 * vy * vy + gamma1 * vy1 * vy1 + gamma2 * vy2 * vy2);

  *B22 = coeff * (gamma0 * vz * vz + gamma1 * vz1 * vz1 + gamma2 * vz2 * vz2);

  *B01 = coeff * (gamma0 * vx * vy + gamma1 * vx1 * vy1 + gamma2 * vx2 * vy2);

  // B10 == B01

  *B02 = coeff * (gamma0 * vx * vz + gamma1 * vx1 * vz1 + gamma2 * vx2 * vz2);

  // B20 == B02

  *B12 = coeff * (gamma0 * vy * vz + gamma1 * vy1 * vz1 + gamma2 * vy2 * vz2);

  // B21 == B12
}

/**
 * @brief Compute cornerfield from facefield (CPU version).
 * @param g Grid where the field is defined
 * @param datax Vector that contains the values on the interfaces
 *        orthogonal to the x-axis
 * @param datay Vector that contains the values on the interfaces
 *        orthogonal to the y-axis
 * @param dataz Vector that contains the values on the interfaces
 *        orthogonal to the z-axis
 * @param cdatax Vector that contains the x-values of the cornerfield
 * @param cdatay Vector that contains the y-values of the cornerfield
 * @param cdataz Vector that contains the z-values of the cornerfield
 * @tparam T Float number precision
 * @tparam Vector Container for data vectors
 */
template <typename T, class Vector>
void computeCornerVelocities(const grid::Grid<T> &g, const Vector &datax,
                             const Vector &datay, const Vector &dataz,
                             Vector &cdatax, Vector &cdatay, Vector &cdataz) {
  using Direction = grid::Direction;

  auto dataxPtr = datax.data();
  auto datayPtr = datay.data();
  auto datazPtr = dataz.data();

  for (auto idz = 0; idz < g.nz + 1; idz++) {
    for (auto idy = 0; idy < g.ny + 1; idy++) {
      for (auto idx = 0; idx < g.nx + 1; idx++) {
        int id = cornerfield::mergeId(g, idx, idy, idz);

        T vx, vy, vz;
        int fx, fy, fz;
        int tx, ty, tz;

        // Velocity x-direction
        vx = 0;
        fx = 0;
        tx = 0;

        if (idx == g.nx) {
          tx = 1;
        }

        for (auto ty = 0; ty <= 1; ty++) {
          for (auto tz = 0; tz <= 1; tz++) {
            if (grid::validId(g, idx - tx, idy - ty, idz - tz)) {
              if (tx == 0) {
                vx += facefield::get<T, Direction::XM>(dataxPtr, datayPtr,
                                                       datazPtr, g, idx - tx,
                                                       idy - ty, idz - tz);
              } else {
                vx += facefield::get<T, Direction::XP>(dataxPtr, datayPtr,
                                                       datazPtr, g, idx - tx,
                                                       idy - ty, idz - tz);
              }
              fx += 1;
            }
          }
        }

        cdatax[id] = vx / fx;

        // Velocity y-direction
        vy = 0;
        fy = 0;
        ty = 0;

        if (idy == g.ny) {
          ty = 1;
        }

        for (auto tx = 0; tx <= 1; tx++) {
          for (auto tz = 0; tz <= 1; tz++) {
            if (grid::validId(g, idx - tx, idy - ty, idz - tz)) {
              if (ty == 0) {
                vy += facefield::get<T, Direction::YM>(dataxPtr, datayPtr,
                                                       datazPtr, g, idx - tx,
                                                       idy - ty, idz - tz);
              } else {
                vy += facefield::get<T, Direction::YP>(dataxPtr, datayPtr,
                                                       datazPtr, g, idx - tx,
                                                       idy - ty, idz - tz);
              }
              fy += 1;
            }
          }
        }
        cdatay[id] = vy / fy;

        // Velocity z-direction
        vz = 0;
        fz = 0;
        tz = 0;

        if (idz == g.nz) {
          tz = 1;
        }

        for (auto tx = 0; tx <= 1; tx++) {
          for (auto ty = 0; ty <= 1; ty++) {
            if (grid::validId(g, idx - tx, idy - ty, idz - tz)) {
              if (tz == 0) {
                vz += facefield::get<T, Direction::ZM>(dataxPtr, datayPtr,
                                                       datazPtr, g, idx - tx,
                                                       idy - ty, idz - tz);
              } else {
                vz += facefield::get<T, Direction::ZP>(dataxPtr, datayPtr,
                                                       datazPtr, g, idx - tx,
                                                       idy - ty, idz - tz);
              }
              fz += 1;
            }
          }
        }
        cdataz[id] = vz / fz;
      }
    }
  }
}

/**
 * @brief CUDA kernel para calcular velocidades en esquinas
 * @param g Grid donde el campo está definido
 * @param dataxPtr Puntero a los datos en x
 * @param datayPtr Puntero a los datos en y
 * @param datazPtr Puntero a los datos en z
 * @param cdataxPtr Puntero donde guardar resultados en x
 * @param cdatayPtr Puntero donde guardar resultados en y
 * @param cdatazPtr Puntero donde guardar resultados en z
 */
template <typename T>
__global__ void
computeCornerVelocitiesKernel(const grid::Grid<T> g, const T *dataxPtr,
                              const T *datayPtr, const T *datazPtr,
                              T *cdataxPtr, T *cdatayPtr, T *cdatazPtr) {
  using Direction = grid::Direction;

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;
  int idz = blockIdx.z * blockDim.z + threadIdx.z;

  if (idx < g.nx + 1 && idy < g.ny + 1 && idz < g.nz + 1) {
    int id = cornerfield::mergeId(g, idx, idy, idz);

    T vx, vy, vz;
    int fx, fy, fz;
    int tx, ty, tz;

    // Velocity x-direction
    vx = 0;
    fx = 0;
    tx = 0;

    if (idx == g.nx) {
      tx = 1;
    }

    for (auto ty = 0; ty <= 1; ty++) {
      for (auto tz = 0; tz <= 1; tz++) {
        if (grid::validId(g, idx - tx, idy - ty, idz - tz)) {
          if (tx == 0) {
            vx += facefield::get<T, Direction::XM>(
                dataxPtr, datayPtr, datazPtr, g, idx - tx, idy - ty, idz - tz);
          } else {
            vx += facefield::get<T, Direction::XP>(
                dataxPtr, datayPtr, datazPtr, g, idx - tx, idy - ty, idz - tz);
          }
          fx += 1;
        }
      }
    }

    cdataxPtr[id] = vx / fx;

    // Velocity y-direction
    vy = 0;
    fy = 0;
    ty = 0;

    if (idy == g.ny) {
      ty = 1;
    }

    for (auto tx = 0; tx <= 1; tx++) {
      for (auto tz = 0; tz <= 1; tz++) {
        if (grid::validId(g, idx - tx, idy - ty, idz - tz)) {
          if (ty == 0) {
            vy += facefield::get<T, Direction::YM>(
                dataxPtr, datayPtr, datazPtr, g, idx - tx, idy - ty, idz - tz);
          } else {
            vy += facefield::get<T, Direction::YP>(
                dataxPtr, datayPtr, datazPtr, g, idx - tx, idy - ty, idz - tz);
          }
          fy += 1;
        }
      }
    }
    cdatayPtr[id] = vy / fy;

    // Velocity z-direction
    vz = 0;
    fz = 0;
    tz = 0;

    if (idz == g.nz) {
      tz = 1;
    }

    for (auto tx = 0; tx <= 1; tx++) {
      for (auto ty = 0; ty <= 1; ty++) {
        if (grid::validId(g, idx - tx, idy - ty, idz - tz)) {
          if (tz == 0) {
            vz += facefield::get<T, Direction::ZM>(
                dataxPtr, datayPtr, datazPtr, g, idx - tx, idy - ty, idz - tz);
          } else {
            vz += facefield::get<T, Direction::ZP>(
                dataxPtr, datayPtr, datazPtr, g, idx - tx, idy - ty, idz - tz);
          }
          fz += 1;
        }
      }
    }
    cdatazPtr[id] = vz / fz;
  }
}

/**
 * @brief Compute cornerfield from facefield (GPU version).
 * @param g Grid where the field is defined
 * @param datax Vector that contains the values on the interfaces
 *        orthogonal to the x-axis
 * @param datay Vector that contains the values on the interfaces
 *        orthogonal to the y-axis
 * @param dataz Vector that contains the values on the interfaces
 *        orthogonal to the z-axis
 * @param cdatax Vector that contains the x-values of the cornerfield
 * @param cdatay Vector that contains the y-values of the cornerfield
 * @param cdataz Vector that contains the z-values of the cornerfield
 * @tparam T Float number precision
 */
template <typename T>
void computeCornerVelocitiesGPU(const grid::Grid<T> &g,
                                const thrust::device_vector<T> &datax,
                                const thrust::device_vector<T> &datay,
                                const thrust::device_vector<T> &dataz,
                                thrust::device_vector<T> &cdatax,
                                thrust::device_vector<T> &cdatay,
                                thrust::device_vector<T> &cdataz) {
  using Direction = grid::Direction;

  // Obtener punteros raw para usar en el kernel
  const T *dataxPtr = thrust::raw_pointer_cast(datax.data());
  const T *datayPtr = thrust::raw_pointer_cast(datay.data());
  const T *datazPtr = thrust::raw_pointer_cast(dataz.data());
  T *cdataxPtr = thrust::raw_pointer_cast(cdatax.data());
  T *cdatayPtr = thrust::raw_pointer_cast(cdatay.data());
  T *cdatazPtr = thrust::raw_pointer_cast(cdataz.data());

  // Calcular dimensiones de la grid y lanzar el kernel
  dim3 block(8, 8, 8);
  dim3 grid((g.nx + 1 + block.x - 1) / block.x,
            (g.ny + 1 + block.y - 1) / block.y,
            (g.nz + 1 + block.z - 1) / block.z);

  // Lanzar el kernel
  computeCornerVelocitiesKernel<T><<<grid, block>>>(
      g, dataxPtr, datayPtr, datazPtr, cdataxPtr, cdatayPtr, cdatazPtr);

  cudaDeviceSynchronize();
}
} // namespace cornerfield

} // namespace par2

#endif // PAR2_CORNERFIELD_CUH

```

# par2\Geometry\FaceField.cuh

```cuh
/**
* @file FaceField.cuh
* @brief Header file for facefield.
*        A facefield is a variable that is defined at the center of
*        of each cell interface of the grid.
*
* @author Calogero B. Rizzo
*
* @copyright This file is part of the PAR2 software.
*            Copyright (C) 2018 Calogero B. Rizzo
*
* @license This program is free software: you can redistribute it and/or modify
*          it under the terms of the GNU General Public License as published by
*          the Free Software Foundation, either version 3 of the License, or
*          (at your option) any later version.
*
*          This program is distributed in the hope that it will be useful,
*          but WITHOUT ANY WARRANTY; without even the implied warranty of
*          MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*          GNU General Public License for more details.
*
*          You should have received a copy of the GNU General Public License
*          along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef PAR2_FACEFIELD_CUH
#define PAR2_FACEFIELD_CUH

#include <fstream>
#include <iostream>
#include <algorithm>

#include "CartesianGrid.cuh"
#include "Interpolation.cuh"

namespace par2
{

    namespace facefield
    {
        /**
        * @brief Build the vectors containing a facefield.
        * @param g Grid where the field is defined
        * @param datax Vector that contains the values on the interfaces
        *        orthogonal to the x-axis
        * @param datay Vector that contains the values on the interfaces
        *        orthogonal to the y-axis
        * @param dataz Vector that contains the values on the interfaces
        *        orthogonal to the z-axis
        * @param vx Init value for datax
        * @param vy Init value for datay
        * @param vz Init value for dataz
        * @tparam T Float number precision
        * @tparam Vector Container for data vectors
        */
        template<typename T, class Vector>
        void build(const grid::Grid<T>& g, Vector& datax, Vector& datay, Vector& dataz,
                               T vx = 0, T vy = 0, T vz = 0)
        {
            int size = (g.nx+1)*(g.ny+1)*(g.nz+1);
            datax.resize(size, vx);
            datay.resize(size, vy);
            dataz.resize(size, vz);
        }

        /**
        * @brief Get the position on the data vectors for the interface
        *        defined by the IDs in each direction.
        * @param g Grid where the field is defined
        * @param idx ID along x
        * @param idy ID along y
        * @param idz ID along z
        * @tparam T Float number precision
        * @return The position on the data vectors
        */
        template<typename T>
        __host__ __device__
        int mergeId(const grid::Grid<T>& g, int idx, int idy, int idz)
        {
            return idz*(g.ny+1)*(g.nx+1) + idy*(g.nx+1) + idx;
        }

        /**
        * @brief Get the value of the facefield at given interface.
        *        The interface is found from the IDs of the cell and the
        *        direction of the interface.
        * @param datax Vector that contains the values on the interfaces
        *        orthogonal to the x-axis
        * @param datay Vector that contains the values on the interfaces
        *        orthogonal to the y-axis
        * @param dataz Vector that contains the values on the interfaces
        *        orthogonal to the z-axis
        * @param g Grid where the field is defined
        * @param idx ID along x
        * @param idy ID along y
        * @param idz ID along z
        * @tparam T Float number precision
        * @tparam dir Direction of the interface
        * @return The value of the facefield at the interface
        */
        template<typename T, int dir>
        __host__ __device__
        T get(const T* datax, const T* datay, const T* dataz, const grid::Grid<T>& g,
              int idx, int idy, int idz)
        {
            int id;
            switch (dir)
            {
                case grid::XP:
                    id = facefield::mergeId(g, idx+1, idy, idz);
                    //id = (idz)*(g.nx+1)*(g.ny+1) + (idy)*(g.nx+1) + (idx+1);
                    return datax[id];
                case grid::XM:
                    id = facefield::mergeId(g, idx  , idy, idz);
                    //id = (idz)*(g.nx+1)*(g.ny+1) + (idy)*(g.nx+1) + (idx  );
                    return datax[id];
                case grid::YP:
                    id = facefield::mergeId(g, idx, idy+1, idz);
                    //id = (idz)*(g.nx+1)*(g.ny+1) + (idy+1)*(g.nx+1) + (idx);
                    return datay[id];
                case grid::YM:
                    id = facefield::mergeId(g, idx, idy  , idz);
                    //id = (idz)*(g.nx+1)*(g.ny+1) + (idy  )*(g.nx+1) + (idx);
                    return datay[id];
                case grid::ZP:
                    id = facefield::mergeId(g, idx, idy, idz+1);
                    //id = (idz+1)*(g.nx+1)*(g.ny+1) + (idy)*(g.nx+1) + (idx);
                    return dataz[id];
                case grid::ZM:
                    id = facefield::mergeId(g, idx, idy, idz  );
                    //id = (idz  )*(g.nx+1)*(g.ny+1) + (idy)*(g.nx+1) + (idx);
                    return dataz[id];
            }
            return 0;
        }

        /**
        * @brief Get the value of the field at given point using linear
        *        interpolation.
        * @param datax Vector that contains the values on the interfaces
        *        orthogonal to the x-axis
        * @param datay Vector that contains the values on the interfaces
        *        orthogonal to the y-axis
        * @param dataz Vector that contains the values on the interfaces
        *        orthogonal to the z-axis
        * @param g Grid where the field is defined
        * @param idx ID along x
        * @param idy ID along y
        * @param idz ID along z
        * @param idValid True if the position is inside the grid
        * @param px Position x-coordinate
        * @param py Position y-coordinate
        * @param pz Position z-coordinate
        * @param vx Result x-coordinate
        * @param vy Result y-coordinate
        * @param vz Result z-coordinate
        * @tparam T Float number precision
        */
        template<typename T>
        __host__ __device__
        void in(const T* datax, const T* datay, const T* dataz, const grid::Grid<T>& g,
                int idx, int idy, int idz, bool idValid, T px, T py, T pz, T* vx, T* vy, T* vz)
        {
            using Direction = grid::Direction;

            T cx, cy, cz;
            grid::centerOfCell<T>(g, idx, idy, idz, &cx, &cy, &cz);

            T Dx = px - cx;
            T Dy = py - cy;
            T Dz = pz - cz;

            T vxp = idValid ? get<T, Direction::XP>(datax, datay, dataz, g, idx, idy, idz) : 1;
            T vxm = idValid ? get<T, Direction::XM>(datax, datay, dataz, g, idx, idy, idz) : 1;
            T vyp = idValid ? get<T, Direction::YP>(datax, datay, dataz, g, idx, idy, idz) : 1;
            T vym = idValid ? get<T, Direction::YM>(datax, datay, dataz, g, idx, idy, idz) : 1;
            T vzp = idValid ? get<T, Direction::ZP>(datax, datay, dataz, g, idx, idy, idz) : 1;
            T vzm = idValid ? get<T, Direction::ZM>(datax, datay, dataz, g, idx, idy, idz) : 1;

            *vx = interpolation::linear<T>(Dx/g.dx + 0.5, vxm, vxp);
            *vy = interpolation::linear<T>(Dy/g.dy + 0.5, vym, vyp);
            *vz = interpolation::linear<T>(Dz/g.dz + 0.5, vzm, vzp);
        }

        /**
        * @brief Import velocity field from modflow.
        * @param g Grid where the field is defined
        * @param datax Vector that contains the values on the interfaces
        *        orthogonal to the x-axis
        * @param datay Vector that contains the values on the interfaces
        *        orthogonal to the y-axis
        * @param dataz Vector that contains the values on the interfaces
        *        orthogonal to the z-axis
        * @param fileName Path to the modflow file
        * @param rho Porosity
        * @tparam T Float number precision
        * @tparam Vector Container for data vectors
        */
        template<typename T, class Vector>
        void importFromModflow(const grid::Grid<T>& g,
                               Vector& datax, Vector& datay, Vector& dataz,
                               const std::string fileName,
                               const T rho)
        {
            std::ifstream inStream;
            inStream.open(fileName, std::ifstream::in);
            if (inStream.is_open())
            {
                std::string line;

                bool is2D = g.nz == 1;
                T area, val;
                int id;

                std::string prefix;

                prefix = " 'QXX";

                while (line.compare(0, prefix.size(), prefix) != 0)
                {
                    std::getline(inStream, line);
                }

                area = g.dy*g.dz;

                for (auto z = 0; z < g.nz; z++)
                {
                    for (auto y = 0; y < g.ny; y++)
                    {
                        for (auto x = 0; x < g.nx; x++)
                        {
                            id = z*(g.nx+1)*(g.ny+1) + y*(g.nx+1) + (x+1);
                            inStream >> val;
                            datax[id] = val/area/rho;   // velocity x
                        }
                    }
                }

                prefix = " 'QYY";

                while (line.compare(0, prefix.size(), prefix) != 0)
                {
                    std::getline(inStream, line);
                }

                area = g.dx*g.dz;

                for (auto z = 0; z < g.nz; z++)
                {
                    for (auto y = 0; y < g.ny; y++)
                    {
                        for (auto x = 0; x < g.nx; x++)
                        {
                            id = z*(g.nx+1)*(g.ny+1) + (y+1)*(g.nx+1) + x;
                            inStream >> val;
                            datay[id] = val/area/rho;   // velocity y
                        }
                    }
                }

                if (!is2D)
                {
                    prefix = " 'QZZ";

                    while (line.compare(0, prefix.size(), prefix) != 0)
                    {
                        std::getline(inStream, line);
                    }

                    area = g.dx*g.dy;

                    for (auto z = 0; z < g.nz; z++)
                    {
                        for (auto y = 0; y < g.ny; y++)
                        {
                            for (auto x = 0; x < g.nx; x++)
                            {
                                id = (z+1)*(g.nx+1)*(g.ny+1) + y*(g.nx+1) + x;
                                inStream >> val;
                                dataz[id] = val/area/rho;   // velocity z
                            }
                        }
                    }
                }
            }
            else
            {
                throw std::runtime_error(std::string("Could not open file ") + fileName);
            }
            inStream.close();
        }

        /**
        * @brief Export velocity field to VTK (unusued).
        * @param g Grid where the field is defined
        * @param datax Vector that contains the values on the interfaces
        *        orthogonal to the x-axis
        * @param datay Vector that contains the values on the interfaces
        *        orthogonal to the y-axis
        * @param dataz Vector that contains the values on the interfaces
        *        orthogonal to the z-axis
        * @param fileName Path to the vtk file
        * @tparam T Float number precision
        * @tparam Vector Container for data vectors
        */
        template<typename T, class Vector>
        void exportVTK(const grid::Grid<T>& g, Vector& datax, Vector& datay, Vector& dataz, const std::string fileName)
        {
            std::ofstream outStream;
            outStream.open(fileName);
            if (outStream.is_open())
            {
                outStream << "# vtk DataFile Version 2.0" << std::endl;
                outStream << "Velocity Field" << std::endl;
                outStream << "ASCII" << std::endl;
                outStream << "DATASET STRUCTURED_POINTS" << std::endl;
                outStream << "DIMENSIONS " << g.nx << " " << g.ny << " " << g.nz << std::endl;
                outStream << "ORIGIN " << g.dx << " " << g.dy << " " << g.dz << std::endl;
                outStream << "SPACING " << g.dx << " " << g.dy << " " << g.dz << std::endl;
                outStream << "POINT_DATA " << g.nx * g.ny * g.nz << std::endl;

                outStream << std::endl;
                outStream << "SCALARS velocityX double" << std::endl;
                outStream << "LOOKUP_TABLE default" << std::endl;
                for (auto idz = 0; idz < g.nz; idz++)
                {
                    for (auto idy = 0; idy < g.ny; idy++)
                    {
                        for (auto idx = 0; idx < g.nx; idx++)
                        {
                            T px, py, pz;
                            grid::centerOfCell(g, idx, idy, idz, &px, &py, &pz);

                            // Velocity term (linear interpolation)
                            T vlx, vly, vlz;
                            facefield::in<T>
                                        (datax.data(), datay.data(), dataz.data(),
                                         g, idx, idy, idz,
                                         true,
                                         px, py, pz,
                                         &vlx, &vly, &vlz);

                            outStream << vlx << std::endl;
                        }
                    }
                }

                outStream << std::endl;
                outStream << "SCALARS velocityY double" << std::endl;
                outStream << "LOOKUP_TABLE default" << std::endl;
                for (auto idz = 0; idz < g.nz; idz++)
                {
                    for (auto idy = 0; idy < g.ny; idy++)
                    {
                        for (auto idx = 0; idx < g.nx; idx++)
                        {
                            T px, py, pz;
                            grid::centerOfCell(g, idx, idy, idz, &px, &py, &pz);

                            // Velocity term (linear interpolation)
                            T vlx, vly, vlz;
                            facefield::in<T>
                                        (datax.data(), datay.data(), dataz.data(),
                                         g, idx, idy, idz,
                                         true,
                                         px, py, pz,
                                         &vlx, &vly, &vlz);

                            outStream << vly << std::endl;
                        }
                    }
                }

                outStream << std::endl;
                outStream << "SCALARS velocityZ double" << std::endl;
                outStream << "LOOKUP_TABLE default" << std::endl;
                for (auto idz = 0; idz < g.nz; idz++)
                {
                    for (auto idy = 0; idy < g.ny; idy++)
                    {
                        for (auto idx = 0; idx < g.nx; idx++)
                        {
                            T px, py, pz;
                            grid::centerOfCell(g, idx, idy, idz, &px, &py, &pz);

                            // Velocity term (linear interpolation)
                            T vlx, vly, vlz;
                            facefield::in<T>
                                        (datax.data(), datay.data(), dataz.data(),
                                         g, idx, idy, idz,
                                         true,
                                         px, py, pz,
                                         &vlx, &vly, &vlz);

                            outStream << vlz << std::endl;
                        }
                    }
                }
            }
            else
            {
                throw std::runtime_error(std::string("Could not open file ") + fileName);
            }
            outStream.close();
        }

    };
};


#endif //PAR2_FACEFIELD_CUH

```

# par2\Geometry\Interpolation.cuh

```cuh
/**
* @file Point.cuh
* @brief Header file for interpolation.
*
* @author Calogero B. Rizzo
*
* @copyright This file is part of the PAR2 software.
*            Copyright (C) 2018 Calogero B. Rizzo
*
* @license This program is free software: you can redistribute it and/or modify
*          it under the terms of the GNU General Public License as published by
*          the Free Software Foundation, either version 3 of the License, or
*          (at your option) any later version.
*
*          This program is distributed in the hope that it will be useful,
*          but WITHOUT ANY WARRANTY; without even the implied warranty of
*          MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*          GNU General Public License for more details.
*
*          You should have received a copy of the GNU General Public License
*          along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef PAR2_INTERPOLATION_CUH
#define PAR2_INTERPOLATION_CUH

namespace par2
{

    namespace interpolation
    {
        /**
        * @brief Normalized linear interpolation.
        * @param x Value between 0 and 1
        * @param v0 Value in 0
        * @param v1 Value in 1
        * @return Linear interpolation
        */
        template<typename T>
        __host__ __device__
        T linear(T x, T v0, T v1)
        {
            return v0*(1-x) + v1*x;
        }

        /**
        * @brief Normalized trilinear interpolation.
        * @param x Value between 0 and 1
        * @param y Value between 0 and 1
        * @param z Value between 0 and 1
        * @param v000 Value in 000
        * @param v100 Value in 100
        * @param v010 Value in 010
        * @param v110 Value in 110
        * @param v001 Value in 001
        * @param v101 Value in 101
        * @param v011 Value in 011
        * @param v111 Value in 111
        * @return Trilinear interpolation
        */
        template<typename T>
        __host__ __device__
        T trilinear(T x, T y, T z,
                    T v000, T v100, T v010, T v110,
                    T v001, T v101, T v011, T v111)
        {
            T v00 = linear(x, v000, v100);
            T v01 = linear(x, v001, v101);
            T v10 = linear(x, v010, v110);
            T v11 = linear(x, v011, v111);

            T v0 = linear(y, v00, v10);
            T v1 = linear(y, v01, v11);

            return linear(z, v0, v1);
        }

        /**
        * @brief Normalized trilinear interpolation of x-derivative.
        * @param x Value between 0 and 1
        * @param y Value between 0 and 1
        * @param z Value between 0 and 1
        * @param dx Block size in x-direction
        * @param v000 Value in 000
        * @param v100 Value in 100
        * @param v010 Value in 010
        * @param v110 Value in 110
        * @param v001 Value in 001
        * @param v101 Value in 101
        * @param v011 Value in 011
        * @param v111 Value in 111
        * @return Trilinear interpolation of x-derivative
        */
        template<typename T>
        __host__ __device__
        T trilinearDevX(T x, T y, T z, T dx,
                        T v000, T v100, T v010, T v110,
                        T v001, T v101, T v011, T v111)
        {
            // vYZ
            T v00 = (v100 - v000)/dx;
            T v01 = (v101 - v001)/dx;
            T v10 = (v110 - v010)/dx;
            T v11 = (v111 - v011)/dx;

            // vZ
            T v0 = linear(y, v00, v10);
            T v1 = linear(y, v01, v11);

            return linear(z, v0, v1);
        }

        /**
        * @brief Normalized trilinear interpolation of y-derivative.
        * @param x Value between 0 and 1
        * @param y Value between 0 and 1
        * @param z Value between 0 and 1
        * @param dy Block size in y-direction
        * @param v000 Value in 000
        * @param v100 Value in 100
        * @param v010 Value in 010
        * @param v110 Value in 110
        * @param v001 Value in 001
        * @param v101 Value in 101
        * @param v011 Value in 011
        * @param v111 Value in 111
        * @return Trilinear interpolation of y-derivative
        */
        template<typename T>
        __host__ __device__
        T trilinearDevY(T x, T y, T z, T dy,
                        T v000, T v100, T v010, T v110,
                        T v001, T v101, T v011, T v111)
        {
            // vXZ
            T v00 = (v010 - v000)/dy;
            T v01 = (v011 - v001)/dy;
            T v10 = (v110 - v100)/dy;
            T v11 = (v111 - v101)/dy;

            // vZ
            T v0 = linear(x, v00, v10);
            T v1 = linear(x, v01, v11);

            return linear(z, v0, v1);
        }

        /**
        * @brief Normalized trilinear interpolation of z-derivative.
        * @param x Value between 0 and 1
        * @param y Value between 0 and 1
        * @param z Value between 0 and 1
        * @param dz Block size in z-direction
        * @param v000 Value in 000
        * @param v100 Value in 100
        * @param v010 Value in 010
        * @param v110 Value in 110
        * @param v001 Value in 001
        * @param v101 Value in 101
        * @param v011 Value in 011
        * @param v111 Value in 111
        * @return Trilinear interpolation of z-derivative
        */
        template<typename T>
        __host__ __device__
        T trilinearDevZ(T x, T y, T z, T dz,
                        T v000, T v100, T v010, T v110,
                        T v001, T v101, T v011, T v111)
        {
            // vXY
            T v00 = (v001 - v000)/dz;
            T v01 = (v011 - v010)/dz;
            T v10 = (v101 - v100)/dz;
            T v11 = (v111 - v110)/dz;

            // vY
            T v0 = linear(x, v00, v10);
            T v1 = linear(x, v01, v11);

            return linear(y, v0, v1);
        }

    }
}


#endif //PAR2_INTERPOLATION_CUH

```

# par2\Geometry\Point.cuh

```cuh
/**
* @file Point.cuh
* @brief Header file for point.
*
* @author Calogero B. Rizzo
*
* @copyright This file is part of the PAR2 software.
*            Copyright (C) 2018 Calogero B. Rizzo
*
* @license This program is free software: you can redistribute it and/or modify
*          it under the terms of the GNU General Public License as published by
*          the Free Software Foundation, either version 3 of the License, or
*          (at your option) any later version.
*
*          This program is distributed in the hope that it will be useful,
*          but WITHOUT ANY WARRANTY; without even the implied warranty of
*          MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*          GNU General Public License for more details.
*
*          You should have received a copy of the GNU General Public License
*          along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef PAR2_POINT_CUH
#define PAR2_POINT_CUH

namespace par2
{

    namespace point
    {
        /**
        * @brief Compute the distance between two points.
        * @param px0 x-component of p0
        * @param py0 y-component of p0
        * @param pz0 z-component of p0
        * @param px1 x-component of p1
        * @param py1 y-component of p1
        * @param pz1 z-component of p1
        * @return Distance between p0 and p1
        */
        template<typename T>
        __host__ __device__
        T distance(T px0, T py0, T pz0, T px1, T py1, T pz1)
        {
            return sqrt((px0 - px1) * (px0 - px1) + (py0 - py1) * (py0 - py1) + (pz0 - pz1) * (pz0 - pz1));
        };

        /**
        * @brief Compute the square of the distance between two points.
        * @param px0 x-component of p0
        * @param py0 y-component of p0
        * @param pz0 z-component of p0
        * @param px1 x-component of p1
        * @param py1 y-component of p1
        * @param pz1 z-component of p1
        * @return Square of the distance between p0 and p1
        */
        template<typename T>
        __host__ __device__
        T distance2(T px0, T py0, T pz0, T px1, T py1, T pz1)
        {
            return (px0 - px1) * (px0 - px1) + (py0 - py1) * (py0 - py1) + (pz0 - pz1) * (pz0 - pz1);
        };

        /**
        * @brief Summation of the coordinates of two points.
        * @param px0 x-component of p0
        * @param py0 y-component of p0
        * @param pz0 z-component of p0
        * @param px1 x-component of p1
        * @param py1 y-component of p1
        * @param pz1 z-component of p1
        * @param rx x-component of p0 + p1
        * @param ry y-component of p0 + p1
        * @param rz z-component of p0 + p1
        */
        template<typename T>
        __host__ __device__
        void plus(T px0, T py0, T pz0, T px1, T py1, T pz1, T* rx, T* ry, T* rz)
        {
            *rx = px0 + px1;
            *ry = py0 + py1;
            *rz = pz0 + pz1;
        };

        /**
        * @brief Subtraction the coordinates of two points.
        * @param px0 x-component of p0
        * @param py0 y-component of p0
        * @param pz0 z-component of p0
        * @param px1 x-component of p1
        * @param py1 y-component of p1
        * @param pz1 z-component of p1
        * @param rx x-component of p0 - p1
        * @param ry y-component of p0 - p1
        * @param rz z-component of p0 - p1
        */
        template<typename T>
        __host__ __device__
        void minus(T px0, T py0, T pz0, T px1, T py1, T pz1, T* rx, T* ry, T* rz)
        {
            *rx = px0 - px1;
            *ry = py0 - py1;
            *rz = pz0 - pz1;
        };

        /**
        * @brief Multiplication between a point and a scalar.
        * @param px0 x-component of p0
        * @param py0 y-component of p0
        * @param pz0 z-component of p0
        * @param val scalar
        * @param rx x-component of val*p0
        * @param ry y-component of val*p0
        * @param rz z-component of val*p0
        */
        template<typename T>
        __host__ __device__
        void multiply(T px0, T py0, T pz0, T val, T* rx, T* ry, T* rz)
        {
            *rx = px0*val;
            *ry = py0*val;
            *rz = pz0*val;
        };

        /**
        * @brief Division between a point and a scalar.
        * @param px0 x-component of p0
        * @param py0 y-component of p0
        * @param pz0 z-component of p0
        * @param val scalar
        * @param rx x-component of p0/val
        * @param ry y-component of p0/val
        * @param rz z-component of p0/val
        */
        template<typename T>
        __host__ __device__
        void divide(T px0, T py0, T pz0, T val, T* rx, T* ry, T* rz)
        {
            *rx = px0/val;
            *ry = py0/val;
            *rz = pz0/val;
        };

    };

}

#endif //PAR2_POINT_CUH

```

# par2\Geometry\Vector.cuh

```cuh
/**
* @file Vector.cuh
* @brief Header file for vector.
*
* @author Calogero B. Rizzo
*
* @copyright This file is part of the PAR2 software.
*            Copyright (C) 2018 Calogero B. Rizzo
*
* @license This program is free software: you can redistribute it and/or modify
*          it under the terms of the GNU General Public License as published by
*          the Free Software Foundation, either version 3 of the License, or
*          (at your option) any later version.
*
*          This program is distributed in the hope that it will be useful,
*          but WITHOUT ANY WARRANTY; without even the implied warranty of
*          MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*          GNU General Public License for more details.
*
*          You should have received a copy of the GNU General Public License
*          along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef PAR2_VECTOR_CUH
#define PAR2_VECTOR_CUH

namespace par2
{

    namespace vector
    {
        /**
        * @brief Compute the square of the Euclidean norm.
        * @param vx x-component
        * @param vy y-component
        * @param vz z-component
        * @return Square of the Euclidean norm
        */
        template<typename T>
        __host__ __device__
        T norm2(T vx, T vy, T vz)
        {
            return vx*vx + vy*vy + vz*vz;
        }

        /**
        * @brief Compute the Euclidean norm.
        * @param vx x-component
        * @param vy y-component
        * @param vz z-component
        * @return Euclidean norm
        */
        template<typename T>
        __host__ __device__
        T norm(T vx, T vy, T vz)
        {
            return sqrt(norm2(vx, vy, vz));
        }

        /**
        * @brief Dot product between two vectors.
        * @param vx0 x-component of v0
        * @param vy0 y-component of v0
        * @param vz0 z-component of v0
        * @param vx1 x-component of v1
        * @param vy1 y-component of v1
        * @param vz1 z-component of v1
        * @return v0 dot v1
        */
        template<typename T>
        __host__ __device__
        T dot(T vx0, T vy0, T vz0, T vx1, T vy1, T vz1)
        {
            return vx0 * vx1 + vy0 * vy1 + vz0 * vz1;
        }

    }

}

#endif //PAR2_VECTOR_CUH

```

# par2\Particles\MoveParticle.cuh

```cuh
/**
 * @file PParticles.cu
 * @brief MoveParticle functor.
 *
 * @author Calogero B. Rizzo
 *
 * @copyright This file is part of the PAR2 software.
 *            Copyright (C) 2018 Calogero B. Rizzo
 *
 * @license This program is free software: you can redistribute it and/or modify
 *          it under the terms of the GNU General Public License as published by
 *          the Free Software Foundation, either version 3 of the License, or
 *          (at your option) any later version.
 *
 *          This program is distributed in the hope that it will be useful,
 *          but WITHOUT ANY WARRANTY; without even the implied warranty of
 *          MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *          GNU General Public License for more details.
 *
 *          You should have received a copy of the GNU General Public License
 *          along with this program.  If not, see
 * <http://www.gnu.org/licenses/>.
 */

#ifndef PAR2_MOVEPARTICLE_CUH
#define PAR2_MOVEPARTICLE_CUH

#include "../Geometry/CartesianGrid.cuh"
#include "../Geometry/CornerField.cuh"
#include "../Geometry/FaceField.cuh"
#include "../Geometry/Vector.cuh"
#include <curand_kernel.h>
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/normal_distribution.h>
#include <thrust/tuple.h>

namespace par2 {
/**
 * @struct MoveParticle
 * @brief Thrust functor for one step of the particle tracking method.
 * @tparam T Float number precision
 */
template <typename T> struct MoveParticle {
  // Raw pointer to velocity vectors (facefield)
  T *datax;
  T *datay;
  T *dataz;
  // Raw pointer to velocity vectors (cornerfield or cellfield)
  T *cdatax;
  T *cdatay;
  T *cdataz;
  // Grid and physical variables
  grid::Grid<T> grid;
  T dt;
  T molecularDiffusion;
  T alphaL;
  T alphaT;
  // Raw pointer to curand state vector
  curandState_t *states;
  // If true, use trilinear interpolation
  bool useTrilinearCorrection;

  // Contadores de imágenes (unwrapping) por partícula
  int *imgY = nullptr;
  int *imgZ = nullptr;

  // (opcional) buffers para guardar coordenadas unwrapped en tiempo real
  T *yUnwrap = nullptr;
  T *zUnwrap = nullptr;

  // Longitudes periódicas
  T Ly = 0;
  T Lz = 0;

  // Base index for current kernel chunk to compute global particle id
  unsigned int baseIndex = 0;

  /**
   * @brief Constructor.
   * @param _grid Grid where the velocity is defined
   */
  MoveParticle(const grid::Grid<T> &_grid) : grid(_grid), dt(0.0){};

  /**
   * @brief Initialize the functor.
   * @param _datax Velocity vector (x-direction)
   * @param _datay Velocity vector (y-direction)
   * @param _dataz Velocity vector (z-direction)
   * @param _molecularDiffusion Effective molecular diffusion
   * @param _alphaL Longitudinal dispersivity
   * @param _alphaT Transverse dispersivity
   * @param _states Curand states
   * @param _useTrilinearCorrection True if trilinear correction is used
   */
  void initialize(thrust::device_vector<T> &_datax,
                  thrust::device_vector<T> &_datay,
                  thrust::device_vector<T> &_dataz,
                  thrust::device_vector<T> &_cdatax,
                  thrust::device_vector<T> &_cdatay,
                  thrust::device_vector<T> &_cdataz, T _molecularDiffusion,
                  T _alphaL, T _alphaT,
                  thrust::device_vector<curandState_t> &_states,
                  bool _useTrilinearCorrection) {
    datax = thrust::raw_pointer_cast(_datax.data());
    datay = thrust::raw_pointer_cast(_datay.data());
    dataz = thrust::raw_pointer_cast(_dataz.data());

    cdatax = thrust::raw_pointer_cast(_cdatax.data());
    cdatay = thrust::raw_pointer_cast(_cdatay.data());
    cdataz = thrust::raw_pointer_cast(_cdataz.data());

    molecularDiffusion = _molecularDiffusion;
    alphaL = _alphaL;
    alphaT = _alphaT;
    states = thrust::raw_pointer_cast(_states.data());

    useTrilinearCorrection = _useTrilinearCorrection;
  }

  __host__ __device__ inline void setUnwrapBuffers(int *_imgY, int *_imgZ,
                                                   T *_yUnwrap, T *_zUnwrap,
                                                   T _Ly, T _Lz) {
    imgY = _imgY;
    imgZ = _imgZ;
    yUnwrap = _yUnwrap;
    zUnwrap = _zUnwrap;
    Ly = _Ly;
    Lz = _Lz;
  }

  /**
   * @brief Set time step.
   * @param _dt Time step
   */
  void setTimeStep(T _dt) { dt = _dt; }

  __host__ __device__ inline void setBaseIndex(unsigned int base) {
    baseIndex = base;
  }

  using Position = thrust::tuple<T, T, T, unsigned int>;

  /**
   * @brief Execute one step of the particle tracking method
   *        on one particle.
   * @param p Initial position of the particle
   * @return Final position of the particle
   */
  __device__ Position operator()(Position p) const {
    int idx, idy, idz;
    grid::idPoint(grid, thrust::get<0>(p), thrust::get<1>(p), thrust::get<2>(p),
                  &idx, &idy, &idz);

    int id = grid::mergeId(grid, idx, idy, idz);

    bool idValid = grid::validId<T>(grid, idx, idy, idz);

    // Velocity term (linear interpolation)
    T vlx, vly, vlz;
    facefield::in<T>(datax, datay, dataz, grid, idx, idy, idz, idValid,
                     thrust::get<0>(p), thrust::get<1>(p), thrust::get<2>(p),
                     &vlx, &vly, &vlz);

    T vcx, vcy, vcz;
    if (useTrilinearCorrection) {
      // Velocity correction div(D) (trilinear interpolation)
      cornerfield::velocityCorrection<T>(
          cdatax, cdatay, cdataz, grid, idx, idy, idz, idValid,
          thrust::get<0>(p), thrust::get<1>(p), thrust::get<2>(p),
          molecularDiffusion, alphaL, alphaT, &vcx, &vcy, &vcz);
    } else {
      // Velocity correction div(D) (block-centered finite difference)
      vcx = idValid ? cdatax[id] : 0;
      vcy = idValid ? cdatay[id] : 0;
      vcz = idValid ? cdataz[id] : 0;
    }

    // Displacement Matrix (trilinear interpolation)
    T B00, B11, B22, B01, B02, B12;
    cornerfield::displacementMatrix<T>(
        cdatax, cdatay, cdataz, grid, idx, idy, idz, idValid, thrust::get<0>(p),
        thrust::get<1>(p), thrust::get<2>(p), molecularDiffusion, alphaL,
        alphaT, dt, &B00, &B11, &B22, &B01, &B02, &B12);

    // Random Displacement
    T xi0 = curand_normal_double(&states[thrust::get<3>(p)]);
    T xi1 = curand_normal_double(&states[thrust::get<3>(p)]);
    T xi2 = curand_normal_double(&states[thrust::get<3>(p)]);

    // Update positions
    T dpx, dpy, dpz;

    dpx = (idValid ? ((vlx + vcx) * dt + (B00 * xi0 + B01 * xi1 + B02 * xi2))
                   : 0);
    dpy = (idValid ? ((vly + vcy) * dt + (B01 * xi0 + B11 * xi1 + B12 * xi2))
                   : 0);
    if (grid.nz != 1)
      dpz = (idValid ? ((vlz + vcz) * dt + (B02 * xi0 + B12 * xi1 + B22 * xi2))
                     : 0);

    // Closed condition: particles cannot exit the domain
    // thrust::get<0>(p) +=
    //     (grid::validX(grid, thrust::get<0>(p) + dpx) ? dpx : 0);
    // thrust::get<1>(p) += (grid::validY(grid, thrust::get<1>(p) + dpy) ?
    // dpy :
    // 0); if (grid.nz != 1)
    //     thrust::get<2>(p) += (grid::validZ(grid, thrust::get<2>(p) + dpz)
    // ?
    //     dpz : 0);

    // X: no periódico (conservar la lógica actual)
    const unsigned int pid = baseIndex + thrust::get<3>(p);
    T x_old = thrust::get<0>(p);
    T x_try = x_old + dpx;
    T x_new = grid::validX(grid, x_try) ? x_try : x_old;
    thrust::get<0>(p) = x_new;

    // Y: periódico + contadores
    T y_old = thrust::get<1>(p);
    T y_new = y_old + dpy;
    int wrapsY = 0;
    while (y_new < grid.py) {
      y_new += Ly;
      wrapsY--;
    }
    while (y_new >= grid.py + Ly) {
      y_new -= Ly;
      wrapsY++;
    }
    thrust::get<1>(p) = y_new;
    if (wrapsY != 0 && imgY)
      imgY[pid] += wrapsY;

    // Z: periódico + contadores (solo si nz>1)
    T z_new_local = thrust::get<2>(p);
    if (grid.nz != 1) {
      T z_try = z_new_local + dpz;
      int wrapsZ = 0;
      while (z_try < grid.pz) {
        z_try += Lz;
        wrapsZ--;
      }
      while (z_try >= grid.pz + Lz) {
        z_try -= Lz;
        wrapsZ++;
      }
      z_new_local = z_try;
      thrust::get<2>(p) = z_new_local;
      if (wrapsZ != 0 && imgZ)
        imgZ[pid] += wrapsZ;

      if (zUnwrap)
        zUnwrap[pid] = z_new_local + static_cast<T>(imgZ ? imgZ[pid] : 0) * Lz;
    }

    // Buffers unwrapped (Y)
    if (yUnwrap)
      yUnwrap[pid] = y_new + static_cast<T>(imgY ? imgY[pid] : 0) * Ly;

    return p;
  }
};
} // namespace par2

#endif // PAR2_MOVEPARTICLE_CUH

```

# par2\Particles\PParticles.cu

```cu
/**
 * @file PParticles.cu
 * @brief Implementation file for PParticles class.
 *
 * @author Calogero B. Rizzo
 *
 * @copyright This file is part of the PAR2 software.
 *            Copyright (C) 2018 Calogero B. Rizzo
 *
 * @license This program is free software: you can redistribute it and/or modify
 *          it under the terms of the GNU General Public License as published by
 *          the Free Software Foundation, either version 3 of the License, or
 *          (at your option) any later version.
 *
 *          This program is distributed in the hope that it will be useful,
 *          but WITHOUT ANY WARRANTY; without even the implied warranty of
 *          MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *          GNU General Public License for more details.
 *
 *          You should have received a copy of the GNU General Public License
 *          along with this program.  If not, see
 * <http://www.gnu.org/licenses/>.
 */

#include "../Geometry/CornerField.cuh"
#include "../Geometry/FaceField.cuh"

#include <algorithm>
#include <fstream>
#include <thrust/count.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/sort.h>

namespace par2 {
template <typename T> struct InitCURAND {
  unsigned long long seed;
  curandState_t *states;
  InitCURAND(unsigned long long _seed,
             thrust::device_vector<curandState_t> &_states) {
    seed = _seed;
    states = thrust::raw_pointer_cast(_states.data());
  }

  __device__ void operator()(unsigned int i) {
    curand_init(seed, i, 0, &states[i]);
  }
};

template <typename T> struct InitVolume {
  curandState_t *states;
  T p1x, p1y, p1z;
  T p2x, p2y, p2z;
  InitVolume(thrust::device_vector<curandState_t> &_states, T _p1x, T _p1y,
             T _p1z, T _p2x, T _p2y, T _p2z) {
    states = thrust::raw_pointer_cast(_states.data());
    p1x = _p1x;
    p1y = _p1y;
    p1z = _p1z;
    p2x = _p2x;
    p2y = _p2y;
    p2z = _p2z;
  }

  using Position = thrust::tuple<T, T, T>;

  __device__ Position operator()(unsigned int i) const {
    Position p;

    thrust::get<0>(p) = p1x + (p2x - p1x) * curand_uniform(&states[i]);
    thrust::get<1>(p) = p1y + (p2y - p1y) * curand_uniform(&states[i]);
    thrust::get<2>(p) = p1z + (p2z - p1z) * curand_uniform(&states[i]);

    return p;
  }
};

// Estructura para almacenar información de cada celda y su probabilidad
template <typename T> struct CellProbability {
  int ix, iy, iz; // Índices de celda
  T probability;  // Probabilidad normalizada
};

// Función para generar la distribución de probabilidad basada en velocidades
template <typename T>
void generateVelocityDistribution(const thrust::device_vector<T> &datax,
                                  const thrust::device_vector<T> &datay,
                                  const thrust::device_vector<T> &dataz,
                                  const grid::Grid<T> &grid, T p1x, T p1y,
                                  T p1z, T p2x, T p2y, T p2z,
                                  std::vector<CellProbability<T>> &cellProbs,
                                  thrust::device_vector<T> &cdf) {
  // 1. Determinar los índices de celda que abarcan la caja
  int minCellX, minCellY, minCellZ;
  int maxCellX, maxCellY, maxCellZ;

  // Encontrar las celdas que contienen los puntos extremos
  bool valid1 =
      grid::findCell(grid, p1x, p1y, p1z, &minCellX, &minCellY, &minCellZ);
  bool valid2 =
      grid::findCell(grid, p2x, p2y, p2z, &maxCellX, &maxCellY, &maxCellZ);

  // Si algún punto está fuera de la malla, usar límites seguros
  if (!valid1 || !valid2) {
    minCellX = 0;
    minCellY = 0;
    minCellZ = 0;
    maxCellX = grid.nx - 1;
    maxCellY = grid.ny - 1;
    maxCellZ = grid.nz - 1;
  }

  // Asegurar que maxCell >= minCell (por si los puntos están en orden inverso)
  if (maxCellX < minCellX)
    std::swap(maxCellX, minCellX);
  if (maxCellY < minCellY)
    std::swap(maxCellY, minCellY);
  if (maxCellZ < minCellZ)
    std::swap(maxCellZ, minCellZ);

  // Ajustar límites para que estén dentro del dominio
  minCellX = std::max(0, minCellX);
  minCellY = std::max(0, minCellY);
  minCellZ = std::max(0, minCellZ);

  maxCellX = std::min(grid.nx - 1, maxCellX);
  maxCellY = std::min(grid.ny - 1, maxCellY);
  maxCellZ = std::min(grid.nz - 1, maxCellZ);

  // 2. Calcular velocidades para cada celda y su probabilidad
  cellProbs.clear();

  T sumVelocity = 0;

  // Punteros raw para acceso más eficiente
  const T *dataxPtr = thrust::raw_pointer_cast(datax.data());
  const T *datayPtr = thrust::raw_pointer_cast(datay.data());
  const T *datazPtr = thrust::raw_pointer_cast(dataz.data());

  for (int iz = minCellZ; iz <= maxCellZ; iz++) {
    for (int iy = minCellY; iy <= maxCellY; iy++) {
      for (int ix = minCellX; ix <= maxCellX; ix++) {
        // Obtener centro de celda
        T cx, cy, cz;
        grid::centerOfCell<T>(grid, ix, iy, iz, &cx, &cy, &cz);

        // Verificar si el centro está dentro de la caja de inyección
        if (cx >= p1x && cx <= p2x && cy >= p1y && cy <= p2y && cz >= p1z &&
            cz <= p2z) {
          // Calcular velocidad en el centro
          T vx, vy, vz;
          par2::facefield::in<T>(dataxPtr, datayPtr, datazPtr, grid, ix, iy, iz,
                                 true, cx, cy, cz, &vx, &vy, &vz);

          // Calcular magnitud de velocidad
          // T velocity = sqrt(vx * vx + vy * vy + vz * vz);
          T velocity2 = vx * vx + vy * vy + vz * vz;

          // Asignar probabilidad mínima para evitar celdas con probabilidad
          // cero
          if (velocity2 < 1e-10)
            velocity2 = 1e-10;

          // Añadir a la lista
          CellProbability<T> cellProb;
          cellProb.ix = ix;
          cellProb.iy = iy;
          cellProb.iz = iz;
          cellProb.probability = velocity2;
          cellProbs.push_back(cellProb);

          sumVelocity += velocity2;
        }
      }
    }
  }

  // 3. Normalizar probabilidades y construir CDF
  T cumulativeProb = 0;
  cdf.resize(cellProbs.size() + 1);
  cdf[0] = 0;

  for (size_t i = 0; i < cellProbs.size(); i++) {
    cellProbs[i].probability /= sumVelocity;
    cumulativeProb += cellProbs[i].probability;
    cdf[i + 1] = cumulativeProb;
  }
}

// Functor para generar partículas según la distribución de velocidades

template <typename T> struct InitVolumeByVelocityDistribution {
  curandState_t *states;
  grid::Grid<T> grid;
  const T *cdfPtr;
  int numCells;
  int *d_cellX_ptr;
  int *d_cellY_ptr;
  int *d_cellZ_ptr;

  InitVolumeByVelocityDistribution(
      thrust::device_vector<curandState_t> &_states,
      const thrust::device_vector<T> &_cdf, const grid::Grid<T> &_grid,
      int *_d_cellX_ptr, int *_d_cellY_ptr, int *_d_cellZ_ptr, int _numCells)
      : grid(_grid), numCells(_numCells) {
    states = thrust::raw_pointer_cast(_states.data());
    cdfPtr = thrust::raw_pointer_cast(_cdf.data());
    d_cellX_ptr = _d_cellX_ptr;
    d_cellY_ptr = _d_cellY_ptr;
    d_cellZ_ptr = _d_cellZ_ptr;
  }

  using Position = thrust::tuple<T, T, T>;

  __device__ Position operator()(unsigned int i) const {
    Position p;

    if (numCells <= 0) {
      // Si no hay celdas válidas, devolver posición inválida
      thrust::get<0>(p) = T(0);
      thrust::get<1>(p) = T(0);
      thrust::get<2>(p) = T(0);
      return p;
    }

    // 1. Generar un número aleatorio para seleccionar la celda
    T r = curand_uniform(&states[i]);

    // 2. Buscar en la CDF (búsqueda binaria)
    int low = 0;
    int high = numCells;

    while (low < high) {
      int mid = (low + high) / 2;
      if (cdfPtr[mid] < r)
        low = mid + 1;
      else
        high = mid;
    }

    // Obtener celda seleccionada (ajustar por posible error numérico)
    int selectedCell = min(low - 1, numCells - 1);
    if (selectedCell < 0)
      selectedCell = 0;

    // 3. Obtener índices de la celda seleccionada
    int ix = d_cellX_ptr[selectedCell];
    int iy = d_cellY_ptr[selectedCell];
    int iz = d_cellZ_ptr[selectedCell];

    // 4. Generar una posición aleatoria dentro de la celda
    T cx, cy, cz;
    grid::centerOfCell<T>(grid, ix, iy, iz, &cx, &cy, &cz);

    // Generar offset aleatorio dentro de la celda (entre -0.5 y 0.5 del tamaño
    // de celda)
    T offsetX = (curand_uniform(&states[i]) - 0.5) * grid.dx;
    T offsetY = (curand_uniform(&states[i]) - 0.5) * grid.dy;
    T offsetZ = (curand_uniform(&states[i]) - 0.5) * grid.dz;

    // Posición final
    thrust::get<0>(p) = cx + offsetX;
    thrust::get<1>(p) = cy + offsetY;
    thrust::get<2>(p) = cz + offsetZ;

    return p;
  }
};

template <typename T>
PParticles<T>::PParticles(const grid::Grid<T> &_grid,
                          thrust::device_vector<T> &&_datax,
                          thrust::device_vector<T> &&_datay,
                          thrust::device_vector<T> &&_dataz,
                          T _molecularDiffusion, T _alphaL, T _alphaT,
                          unsigned int _nParticles, long int _seed,
                          bool _useTrilinearCorrection)
    : nParticles(_nParticles), molecularDiffusion(_molecularDiffusion),
      alphaL(_alphaL), alphaT(_alphaT), grid(_grid), moveParticle(_grid),
      useTrilinearCorrection(_useTrilinearCorrection) {
  cx.resize(nParticles);
  cy.resize(nParticles);
  cz.resize(nParticles);

  datax = std::move(_datax);
  datay = std::move(_datay);
  dataz = std::move(_dataz);

  if (useTrilinearCorrection) {
    // Inicializar vectores de salida directamente en GPU
    par2::cornerfield::build(grid, cdatax);
    par2::cornerfield::build(grid, cdatay);
    par2::cornerfield::build(grid, cdataz);

    // Usar versión GPU que opera completamente en device
    par2::cornerfield::computeCornerVelocitiesGPU(grid, datax, datay, dataz,
                                                  cdatax, cdatay, cdataz);
  } else {
    // Inicializar directamente en device para evitar copia adicional
    par2::cellfield::build(grid, cdatax);
    par2::cellfield::build(grid, cdatay);
    par2::cellfield::build(grid, cdataz);

    // Usar versión GPU de la corrección de deriva
    par2::cellfield::computeDriftCorrectionGPU(
        grid, datax, datay, dataz, cdatax, cdatay, cdataz, molecularDiffusion,
        alphaL, alphaT);
  }

  states.resize(maxParticles);
  thrust::counting_iterator<unsigned int> count(0);
  thrust::for_each(count, count + maxParticles, InitCURAND<T>(_seed, states));

  moveParticle.initialize(datax, datay, dataz, cdatax, cdatay, cdataz,
                          molecularDiffusion, alphaL, alphaT, states,
                          useTrilinearCorrection);

  // Setup unwrapping buffers (allocated after knowing nParticles and grid)
  imgY_.assign(nParticles, 0);
  if (grid.nz > 1)
    imgZ_.assign(nParticles, 0);
  yU_.assign(nParticles, 0);
  if (grid.nz > 1)
    zU_.assign(nParticles, 0);

  Ly_ = grid.dy * grid.ny;
  Lz_ = grid.dz * grid.nz;

  moveParticle.setUnwrapBuffers(
      imgY_.empty() ? nullptr : thrust::raw_pointer_cast(imgY_.data()),
      imgZ_.empty() ? nullptr : thrust::raw_pointer_cast(imgZ_.data()),
      yU_.empty() ? nullptr : thrust::raw_pointer_cast(yU_.data()),
      zU_.empty() ? nullptr : thrust::raw_pointer_cast(zU_.data()), Ly_, Lz_);

  cudaDeviceSynchronize();
}

template <typename T> unsigned int PParticles<T>::size() const {
  return nParticles;
}

template <typename T>
void PParticles<T>::initializeBox(T p1x, T p1y, T p1z, T p2x, T p2y, T p2z,
                                  bool velocityBased) {
  thrust::counting_iterator<unsigned int> count(0);
  auto pBeg = thrust::make_zip_iterator(
      thrust::make_tuple(cx.begin(), cy.begin(), cz.begin()));

  if (velocityBased) {
    // Variables para la distribución
    std::vector<CellProbability<T>> cellProbs;
    thrust::device_vector<T> cdf;

    // Generar la distribución de probabilidad basada en velocidad
    generateVelocityDistribution(datax, datay, dataz, grid, p1x, p1y, p1z, p2x,
                                 p2y, p2z, cellProbs, cdf);

    // Si no hay celdas válidas, usar distribución uniforme
    if (cellProbs.empty()) {
      auto functor = InitVolume<T>(states, p1x, p1y, p1z, p2x, p2y, p2z);

      for (auto i = 0; i * maxParticles < nParticles; i++) {
        unsigned int kernelSize = maxParticles;
        if (kernelSize > nParticles - i * maxParticles) {
          kernelSize = nParticles - i * maxParticles;
        }
        thrust::transform(count, count + kernelSize, pBeg + i * maxParticles,
                          functor);
      }
    } else {
      // Copiar índices de celda a device y obtener punteros raw
      thrust::device_vector<int> d_cellX(cellProbs.size());
      thrust::device_vector<int> d_cellY(cellProbs.size());
      thrust::device_vector<int> d_cellZ(cellProbs.size());
      for (size_t i = 0; i < cellProbs.size(); i++) {
        d_cellX[i] = cellProbs[i].ix;
        d_cellY[i] = cellProbs[i].iy;
        d_cellZ[i] = cellProbs[i].iz;
      }
      int *d_cellX_ptr = thrust::raw_pointer_cast(d_cellX.data());
      int *d_cellY_ptr = thrust::raw_pointer_cast(d_cellY.data());
      int *d_cellZ_ptr = thrust::raw_pointer_cast(d_cellZ.data());
      auto functor = InitVolumeByVelocityDistribution<T>(
          states, cdf, grid, d_cellX_ptr, d_cellY_ptr, d_cellZ_ptr,
          cellProbs.size());
      for (auto i = 0; i * maxParticles < nParticles; i++) {
        unsigned int kernelSize = maxParticles;
        if (kernelSize > nParticles - i * maxParticles) {
          kernelSize = nParticles - i * maxParticles;
        }
        thrust::transform(count, count + kernelSize, pBeg + i * maxParticles,
                          functor);
      }
    }
  } else {
    // Comportamiento original con distribución uniforme
    auto functor = InitVolume<T>(states, p1x, p1y, p1z, p2x, p2y, p2z);

    for (auto i = 0; i * maxParticles < nParticles; i++) {
      unsigned int kernelSize = maxParticles;
      if (kernelSize > nParticles - i * maxParticles) {
        kernelSize = nParticles - i * maxParticles;
      }
      thrust::transform(count, count + kernelSize, pBeg + i * maxParticles,
                        functor);
    }
  }
}

template <typename T> void PParticles<T>::move(T dt) {
  thrust::counting_iterator<unsigned int> count(0);
  moveParticle.setTimeStep(dt);

  for (auto i = 0; i * maxParticles < nParticles; i++) {
    unsigned int kernelSize = maxParticles;
    if (kernelSize > nParticles - i * maxParticles) {
      kernelSize = nParticles - i * maxParticles;
    }

    moveParticle.setBaseIndex(i * maxParticles);

    // ensure id is the 4th element for the functor (local id per chunk)
    auto pBeg = thrust::make_zip_iterator(thrust::make_tuple(
        cx.begin() + i * maxParticles, cy.begin() + i * maxParticles,
        cz.begin() + i * maxParticles, count));
    // auto pEnd = thrust::make_zip_iterator(
    //     thrust::make_tuple(cx.end(),   cy.end()  , cz.end()  ,
    //     count+kernelSize));

    thrust::transform(pBeg, pBeg + kernelSize, pBeg, moveParticle);
  }
  cudaDeviceSynchronize();
}

template <typename T>
void PParticles<T>::exportCSV(const std::string &fileName) const {
  // Copy to host memory
  thrust::host_vector<T> hx = cx;
  thrust::host_vector<T> hy = cy;
  thrust::host_vector<T> hz = cz;

  std::ofstream outStream;
  outStream.open(fileName);
  if (outStream.is_open()) {
    outStream << "id,x coord,y coord,z coord" << std::endl;
    for (unsigned int i = 0; i < nParticles; i++) {
      outStream << i << "," << hx[i] << "," << hy[i] << "," << hz[i]
                << std::endl;
    }
  } else {
    throw std::runtime_error(std::string("Could not open file ") + fileName);
  }
  outStream.close();
}

template <typename T> struct isInside {
  T plane;

  T p1x, p1y, p1z;
  T p2x, p2y, p2z;
  isInside(T _p1x, T _p1y, T _p1z, T _p2x, T _p2y, T _p2z) {
    p1x = _p1x;
    p1y = _p1y;
    p1z = _p1z;
    p2x = _p2x;
    p2y = _p2y;
    p2z = _p2z;
  }

  using Position = thrust::tuple<T, T, T>;

  __device__ bool operator()(Position p) const {
    return (p1x <= thrust::get<0>(p) && thrust::get<0>(p) <= p2x) &&
           (p1y <= thrust::get<1>(p) && thrust::get<1>(p) <= p2y) &&
           (p1z <= thrust::get<2>(p) && thrust::get<2>(p) <= p2z);
  }
};

template <typename T>
T PParticles<T>::concentrationBox(T p1x, T p1y, T p1z, T p2x, T p2y,
                                  T p2z) const {
  auto pBeg = thrust::make_zip_iterator(
      thrust::make_tuple(cx.begin(), cy.begin(), cz.begin()));
  auto pEnd = thrust::make_zip_iterator(
      thrust::make_tuple(cx.end(), cy.end(), cz.end()));

  return thrust::count_if(pBeg, pEnd,
                          isInside<T>(p1x, p1y, p1z, p2x, p2y, p2z)) /
         T(nParticles);
}

template <typename T> struct isAfter {
  T plane;

  isAfter(T _plane) : plane(_plane){};

  __device__ bool operator()(T x) { return x > plane; }
};

template <typename T> T PParticles<T>::concentrationAfterX(T xplane) const {
  return thrust::count_if(cx.begin(), cx.end(), isAfter<T>(xplane)) /
         T(nParticles);
}

template <typename T> thrust::device_vector<T> PParticles<T>::getX() const {
  return cx;
}

template <typename T> thrust::device_vector<T> PParticles<T>::getY() const {
  return cy;
}

template <typename T> thrust::device_vector<T> PParticles<T>::getZ() const {
  return cz;
}

} // namespace par2

```

# par2\Particles\PParticles.cuh

```cuh
/**
 * @file PParticles.cuh
 * @brief Header file for PParticles class.
 *
 * @author Calogero B. Rizzo
 *
 * @copyright This file is part of the PAR2 software.
 *            Copyright (C) 2018 Calogero B. Rizzo
 *
 * @license This program is free software: you can redistribute it and/or modify
 *          it under the terms of the GNU General Public License as published by
 *          the Free Software Foundation, either version 3 of the License, or
 *          (at your option) any later version.
 *
 *          This program is distributed in the hope that it will be useful,
 *          but WITHOUT ANY WARRANTY; without even the implied warranty of
 *          MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *          GNU General Public License for more details.
 *
 *          You should have received a copy of the GNU General Public License
 *          along with this program.  If not, see
 * <http://www.gnu.org/licenses/>.
 */

#ifndef PAR2_PPARTICLES_CUH
#define PAR2_PPARTICLES_CUH

#include "../Geometry/CartesianGrid.cuh"
#include "../Geometry/FaceField.cuh"
#include "MoveParticle.cuh"
#include <ctime>
#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace par2 {
/**
 * @class PParticles
 * @brief Class managing a cloud of particles. It provides functions
 *        to initialize and move the particles on device.
 * @tparam T Float number precision
 */
template <typename T> class PParticles {
public:
  /**
   * @brief Constructor.
   * @param _grid Grid where the velocity is defined
   * @param _datax Velocity vector (x-direction)
   * @param _datay Velocity vector (y-direction)
   * @param _dataz Velocity vector (z-direction)
   * @param _molecularDiffusion Effective molecular diffusion
   * @param _alphaL Longitudinal dispersivity
   * @param _alphaT Transverse dispersivity
   * @param _nParticles Total number of particles
   * @param _seed Seed for pseudo-random number generator
   * @param _useTrilinearCorrection True if trilinear correction is used
   */
  PParticles(const grid::Grid<T> &_grid, thrust::device_vector<T> &&_datax,
             thrust::device_vector<T> &&_datay,
             thrust::device_vector<T> &&_dataz, T _molecularDiffusion,
             T _alphaL, T _alphaT, unsigned int _nParticles,
             long int _seed = time(NULL), bool _useTrilinearCorrection = true);

  /**
   * @brief Number of particles in the system.
   * @return Number of particles
   */
  unsigned int size() const;

  /**
   * @brief Initialize particles inside a box defined by
   *        two points p1 and p2. The particles are uniformly
   *        distributed inside the box.
   * @param p1x x-component of p1
   * @param p1y y-component of p1
   * @param p1z z-component of p1
   * @param p2x x-component of p2
   * @param p2y y-component of p2
   * @param p2z z-component of p2
   */
  void initializeBox(T p1x, T p1y, T p1z, T p2x, T p2y, T p2z,
                     bool velocityBased = false);

  /**
   * @brief Execute one step of the particle tracking method using
   *        a time step dt.
   * @param dt Time step
   */
  void move(T dt);

  /**
   * @brief Export all the particles in a csv file.
   *        WARNING: this function is computationally expensive
   *        and should be avoided if computation time is critical.
   * @param fileName Path to the csv file
   */
  void exportCSV(const std::string &fileName) const;

  /**
   * @brief Compute the percentage of particles inside a box
   *        defined by two points p1 and p2.
   * @param p1x x-component of p1
   * @param p1y y-component of p1
   * @param p1z z-component of p1
   * @param p2x x-component of p2
   * @param p2y y-component of p2
   * @param p2z z-component of p2
   * @return Percentage of particles inside the box.
   */
  T concentrationBox(T p1x, T p1y, T p1z, T p2x, T p2y, T p2z) const;

  /**
   * @brief Compute the percentage of particles that crossed
   *        a plane orthogonal to the x-axis (i.e., px > xplane).
   * @param xplane Location of the plane
   * @return Percentage of particles after the plane.
   */
  T concentrationAfterX(T xplane) const;

  /**
   * @brief Getters for particle positions.
   * @return Particle positions (x,y,z) as thrust::device_vector
   */
  thrust::device_vector<T> getX() const;
  thrust::device_vector<T> getY() const;
  thrust::device_vector<T> getZ() const;

  const T *xPtr() const { return thrust::raw_pointer_cast(cx.data()); }
  const T *zPtr() const { return thrust::raw_pointer_cast(cz.data()); }
  const T *yPtr() const { return thrust::raw_pointer_cast(cy.data()); }

  // Unwrapping buffers (optional) getters
  const int *imgYPtr() const {
    return imgY_.empty() ? nullptr : thrust::raw_pointer_cast(imgY_.data());
  }
  const int *imgZPtr() const {
    return imgZ_.empty() ? nullptr : thrust::raw_pointer_cast(imgZ_.data());
  }
  const T *yUnwrapPtr() const {
    return yU_.empty() ? nullptr : thrust::raw_pointer_cast(yU_.data());
  }
  const T *zUnwrapPtr() const {
    return zU_.empty() ? nullptr : thrust::raw_pointer_cast(zU_.data());
  }

private:
  // Thrust functor for one step of particle tracking
  MoveParticle<T> moveParticle;
  // Grid and physical variables
  grid::Grid<T> grid;
  T molecularDiffusion;
  T alphaL;
  T alphaT;
  unsigned int nParticles;
  // Velocity field stored on device (facefield)
  thrust::device_vector<T> datax;
  thrust::device_vector<T> datay;
  thrust::device_vector<T> dataz;
  // Velocity field stored on device (cornerfield or cellfield)
  thrust::device_vector<T> cdatax;
  thrust::device_vector<T> cdatay;
  thrust::device_vector<T> cdataz;
  // Particle positions stored on device
  thrust::device_vector<T> cx;
  thrust::device_vector<T> cy;
  thrust::device_vector<T> cz;
  // Vector of curand states stored on device
  thrust::device_vector<curandState_t> states;
  // If true, use trilinear interpolation
  bool useTrilinearCorrection;
  // Max number of particles simulated in each kernel
  const int maxParticles = 65536;

  // Unwrapping state/buffers
  thrust::device_vector<int> imgY_, imgZ_;
  thrust::device_vector<T> yU_, zU_;
  T Ly_ = 0, Lz_ = 0;
};

} // namespace par2

#include "PParticles.cu"

#endif // PAR2_PPARTICLES_CUH

```

# plot_script\plot.py

```py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Promedia 10 realizaciones de macrodispersión y grafica Dx, Dy, Dz vs tiempo.
Cada macrodispersion_k.csv debe tener cabecera: t,Dx,Dy,Dz
Uso: python plot.py <varianza>
Valores válidos de varianza: 0.25, 1, 2.25, 4, 6.25
"""

import pathlib
import sys
import argparse

import pandas as pd
import math
import matplotlib.pyplot as plt


def get_lx_from_variance(variance):
    """Obtiene el valor de Lx según la varianza"""
    variance_to_lx = {0.25: 1024, 1.0: 2048, 2.25: 4096, 4.0: 8192, 6.25: 8192}
    return variance_to_lx.get(variance)


def get_experimental_data(variance):
    """Obtiene los datos experimentales según la varianza"""

    if variance == 0.25:
        bDL = [
            (1.0000e-01, 1.7175e-02),
            (1.2491e-01, 2.1198e-02),
            (1.5733e-01, 2.6158e-02),
            (1.9976e-01, 3.3266e-02),
            (2.5159e-01, 4.1058e-02),
            (3.1427e-01, 5.0664e-02),
            (3.9582e-01, 6.2532e-02),
            (4.9854e-01, 7.4886e-02),
            (6.2791e-01, 9.1033e-02),
            (7.9068e-01, 1.1066e-01),
            (1.0042e00, 1.2862e-01),
            (1.2543e00, 1.4949e-01),
            (1.5798e00, 1.7112e-01),
            (1.9893e00, 1.9011e-01),
            (2.5055e00, 2.0493e-01),
            (3.1557e00, 2.2090e-01),
            (3.9747e00, 2.2423e-01),
            (5.0061e00, 2.3812e-01),
            (6.3038e00, 2.3812e-01),
            (7.8741e00, 2.4177e-01),
            (1.0000e01, 2.4177e-01),
            (1.2595e01, 2.4541e-01),
            (1.5864e01, 2.4177e-01),
            (1.9976e01, 2.4177e-01),
            (2.4952e01, 2.4177e-01),
            (3.1688e01, 2.4541e-01),
            (3.9582e01, 2.4541e-01),
            (4.9854e01, 2.4912e-01),
        ]
        bDT = []  # No hay datos bDT para varianza 0.25 en el script original

    elif variance == 1.0:
        bDL = [
            (1.0083e-01, 9.0323e-02),
            (1.2491e-01, 1.1324e-01),
            (1.5992e-01, 1.3775e-01),
            (1.9976e-01, 1.7014e-01),
            (2.4952e-01, 2.0697e-01),
            (3.1952e-01, 2.4797e-01),
            (3.9912e-01, 2.9717e-01),
            (5.0269e-01, 3.6149e-01),
            (6.3299e-01, 4.2668e-01),
            (8.0390e-01, 4.9614e-01),
            (1.0042, 5.7677e-01),
            (1.2750, 6.5073e-01),
            (1.5926, 7.3418e-01),
            (1.9893, 8.2832e-01),
            (2.5264, 9.2045e-01),
            (3.1820, 1.0076),
            (4.0077, 1.0703),
            (5.0466, 1.1366),
            (6.3562, 1.1893),
            (8.0057, 1.2257),
            (10.167, 1.2257),
            (12.70, 1.2442),
            (15.864, 1.2633),
            (20.142, 1.2633),
            (25.369, 1.2633),
            (31.952, 1.2823),
            (40.244, 1.2633),
        ]
        bDT = [
            (1.0000e-01, 1.2682e-02),
            (1.1719e-01, 1.3970e-02),
            (1.4138e-01, 1.5733e-02),
            (1.6688e-01, 1.7073e-02),
            (1.9843e-01, 1.9226e-02),
            (2.3588e-01, 2.1493e-02),
            (2.8048e-01, 2.3845e-02),
            (3.3350e-01, 2.5876e-02),
            (3.9655e-01, 2.6853e-02),
            (4.7490e-01, 2.8927e-02),
            (5.6053e-01, 3.2337e-02),
            (6.6650e-01, 3.7239e-02),
            (7.9250e-01, 4.0411e-02),
            (9.4232e-01, 3.8362e-02),
            (1.1202, 3.6149e-02),
            (1.3320, 3.5876e-02),
            (1.5951, 3.6417e-02),
            (1.8832, 3.6149e-02),
            (2.2387, 2.9580e-02),
            (2.6810, 2.3496e-02),
            (3.1652, 1.6819e-02),
            (3.7905, 1.8252e-02),
            (4.4740, 1.1863e-02),
            (5.2820, 1.2311e-02),
            (6.3256, 1.6448e-02),
            (7.5214, 1.4713e-02),
            (8.8777, 1.0691e-02),
            (10.632, 1.0000e-02),
            (12.642, 1.2221e-02),
            (15.139, 7.8850e-03),
            (17.873, 9.9266e-03),
            (21.404, 8.4295e-03),
            (25.264, 9.2151e-03),
            (30.040, 8.8145e-03),
            (35.975, 9.3541e-03),
        ]

    elif variance == 2.25:
        bDL = [
            (0.1250, 0.3615),
            (0.1574, 0.4465),
            (0.1984, 0.5270),
            (0.2479, 0.6314),
            (0.3149, 0.7452),
            (0.3714, 0.8408),
            (0.3936, 0.8798),
            (0.4958, 1.0541),
            (0.6299, 1.2073),
            (0.7936, 1.4038),
            (0.9917, 1.6077),
            (1.2601, 1.8412),
            (1.5874, 2.0156),
            (2.0003, 2.3083),
            (2.4992, 2.5270),
            (3.1492, 2.8080),
            (3.9683, 2.9826),
            (4.9992, 3.2166),
            (6.3518, 3.4682),
            (8.0020, 3.6283),
            (9.9174, 3.8539),
            (12.7028, 4.0327),
            (15.8745, 4.0935),
            (20.0032, 4.2189),
            (25.2000, 4.3481),
            (32.0184, 4.3481),
            (40.0129, 4.4137),
            (50.8276, 4.4813),
            (63.5185, 4.4813),
        ]
        bDT = [
            (1.0072e-01, 3.5539e-02),
            (1.1719e-01, 3.7993e-02),
            (1.4031e-01, 4.3421e-02),
            (1.6444e-01, 4.5394e-02),
            (1.9688e-01, 4.9625e-02),
            (2.3405e-01, 5.3839e-02),
            (2.7823e-01, 6.1518e-02),
            (3.3083e-01, 6.4804e-02),
            (3.9328e-01, 6.4328e-02),
            (4.6752e-01, 6.5283e-02),
            (5.5578e-01, 7.5162e-02),
            (6.6069e-01, 8.2167e-02),
            (7.8542e-01, 9.0469e-02),
            (9.4059e-01, 9.9632e-02),
            (1.1102e00, 1.0188e-01),
            (1.3104e00, 1.0340e-01),
            (1.8518e00, 1.0889e-01),
            (2.2014e00, 1.0809e-01),
            (2.6170e00, 1.0188e-01),
            (3.1565e00, 9.6716e-02),
            (3.6991e00, 9.0469e-02),
            (4.4289e00, 7.7428e-02),
            (5.2650e00, 7.4611e-02),
            (6.2144e00, 7.0307e-02),
            (7.3875e00, 6.6757e-02),
            (8.8471e00, 6.5283e-02),
            (1.0517e01, 6.3372e-02),
            (1.2503e01, 6.2907e-02),
            (1.4757e01, 5.5873e-02),
            (1.7673e01, 5.0734e-02),
            (2.1009e01, 4.4730e-02),
            (2.4975e01, 4.6068e-02),
            (2.9902e01, 4.7457e-02),
            (3.4794e01, 4.4730e-02),
            (4.1957e01, 4.5394e-02),
            (4.9888e01, 4.7109e-02),
            (5.8452e01, 4.3742e-02),
            (7.0000e01, 4.6763e-02),
            (8.3811e01, 4.4066e-02),
        ]

    elif variance == 4.0:
        bDL = [
            (0.1008, 0.6411),
            (0.1249, 0.7798),
            (0.1573, 0.9343),
            (0.1982, 1.1031),
            (0.2516, 1.2633),
            (0.3143, 1.4468),
            (0.3958, 1.6823),
            (0.5027, 1.9266),
            (0.6330, 2.2398),
            (0.7907, 2.5657),
            (0.9959, 2.9383),
            (1.2647, 3.3151),
            (1.5798, 3.7966),
            (2.0059, 4.2835),
            (2.5055, 4.6881),
            (3.1820, 5.1322),
            (3.9747, 5.7903),
            (5.0061, 6.3387),
            (6.3038, 6.9375),
            (7.9396, 7.4817),
            (10.0000, 8.1903),
            (12.5951, 8.6976),
            (15.8635, 9.3799),
            (19.9756, 9.9632),
            (25.1594, 10.5803),
            (31.6884, 11.0713),
            (39.9117, 11.7598),
            (50.2690, 11.9371),
            (63.2995, 12.3027),
            (79.7260, 12.3027),
            (100.4153, 12.6794),
            (127.4970, 12.6794),
            (159.2575, 13.0677),
            (202.2553, 13.4679),
            (254.7417, 13.2648),
        ]
        bDT = [
            (9.9289e-02, 1.4266e-01),
            (1.1633e-01, 1.4481e-01),
            (1.3932e-01, 1.7426e-01),
            (1.6444e-01, 2.0059e-01),
            (1.9829e-01, 2.0356e-01),
            (2.3578e-01, 2.1918e-01),
            (2.7823e-01, 2.2746e-01),
            (3.9039e-01, 2.0811e-01),
            (4.7087e-01, 1.9907e-01),
            (5.5578e-01, 2.0207e-01),
            (7.9122e-01, 2.1918e-01),
            (1.1262e00, 2.1125e-01),
            (1.3391e00, 2.4496e-01),
            (1.5802e00, 2.4860e-01),
            (1.8789e00, 2.2085e-01),
            (2.2336e00, 2.3605e-01),
            (2.6363e00, 1.9041e-01),
            (3.1340e00, 1.8625e-01),
            (3.7256e00, 1.8218e-01),
            (4.4617e00, 1.6062e-01),
            (5.3040e00, 1.5477e-01),
            (6.2604e00, 1.3250e-01),
            (7.4422e00, 1.3954e-01),
            (1.0517e01, 1.2034e-01),
            (1.2503e01, 1.0378e-01),
            (1.4973e01, 8.8166e-02),
            (2.1009e01, 1.0849e-01),
            (2.9902e01, 1.0532e-01),
            (3.5294e01, 1.0301e-01),
            (4.1362e01, 1.0148e-01),
            (4.9888e01, 1.0378e-01),
            (5.9306e01, 1.0301e-01),
            (7.0000e01, 1.0378e-01),
            (8.3811e01, 1.0532e-01),
            (9.9632e01, 1.0849e-01),
            (1.1847e02, 1.0454e-01),
            (1.4083e02, 1.0688e-01),
            (1.6742e02, 1.1174e-01),
            (1.9902e02, 1.0770e-01),
            (2.3659e02, 1.0610e-01),
        ]

    elif variance == 6.25:
        # Para varianza 6.25, asumimos que no hay datos experimentales específicos
        bDL = []
        bDT = []
    else:
        bDL = []
        bDT = []

    return bDL, bDT


# --------------------------------------------------------------------
# Procesamiento de argumentos de línea de comandos
# --------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Plotea macrodispersión promedio para una varianza específica"
    )
    parser.add_argument(
        "variance", type=float, help="Valor de la varianza (0.25, 1, 2.25, 4, 6.25)"
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=100,
        help="Número de realizaciones a promediar (default: 100)",
    )

    args = parser.parse_args()
    variance = args.variance
    n_runs = args.n_runs

    # Validar varianza
    valid_variances = [0.25, 1.0, 2.25, 4.0, 6.25]
    if variance not in valid_variances:
        sys.exit(
            f"ERROR: Varianza {variance} no válida. Valores válidos: {valid_variances}"
        )

    # Configuración basada en la varianza
    lx = get_lx_from_variance(variance)
    if lx is None:
        sys.exit(f"ERROR: No se pudo determinar Lx para varianza {variance}")

    vm = 100 / (lx * 5)
    lamb = 50.0

    # Patrones de archivos
    var_str = f"{variance:.2f}".replace(".", "")
    if len(var_str) == 2:
        var_str = var_str + "0"  # Para casos como 1.00 -> 100

    file_pattern = (
        f"output/out_{variance:.2f}/macrodispersion_var_v9_{variance:.2f}_{{:d}}.csv"
    )
    outfig = f"output/out_{variance:.2f}/macrodispersion_avg.png"

    print(f"Configuración:")
    print(f"  Varianza: {variance}")
    print(f"  Lx: {lx}")
    print(f"  vm: {vm:.6f}")
    print(f"  Patrón de archivos: {file_pattern}")
    print(f"  Número de realizaciones: {n_runs}")
    print(f"  Archivo de salida: {outfig}")
    print()

    # --------------------------------------------------------------------
    # Carga de todas las realizaciones
    # --------------------------------------------------------------------

    dfs = []
    for k in range(n_runs):
        fname = file_pattern.format(k)
        path = pathlib.Path(fname)
        if not path.is_file():
            print(f"ADVERTENCIA: no se encontró «{fname}», saltando...")
            continue

        df = pd.read_csv(path)
        if not {"t", "Dx", "Dy", "Dz"} <= set(df.columns):
            print(
                f"ADVERTENCIA: «{fname}» no contiene las columnas esperadas, saltando..."
            )
            continue
        dfs.append(df)

    if not dfs:
        sys.exit("ERROR: No se pudo cargar ningún archivo de datos")

    print(f"Se cargaron {len(dfs)} realizaciones exitosamente")

    # --------------------------------------------------------------------
    # Cálculo del promedio ⟨Dx⟩, ⟨Dy⟩, ⟨Dz⟩ por instante
    # --------------------------------------------------------------------
    all_data = pd.concat(dfs, axis=0, ignore_index=True)
    mean_data = (
        all_data.groupby("t", as_index=False)[["Dx", "Dy", "Dz"]]
        .mean()
        .sort_values("t")
    )
    mean_data[["Dx", "Dy", "Dz"]] /= 2 * lamb * vm * (2 / math.sqrt(math.pi))
    mean_data["t"] /= lamb / vm

    # Obtener datos experimentales
    bDL, bDT = get_experimental_data(variance)

    # --------------------------------------------------------------------
    # Gráfica: curvas promedio
    # --------------------------------------------------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(mean_data["t"], mean_data["Dx"], label=r"$\langle D_x\rangle$")
    plt.plot(mean_data["t"], mean_data["Dy"], label=r"$\langle D_y\rangle$")
    plt.plot(mean_data["t"], mean_data["Dz"], label=r"$\langle D_z\rangle$")

    # --------------------------------------------------------------------
    # Añadimos los puntos experimentales
    # --------------------------------------------------------------------
    if bDL:
        bx, by = zip(*bDL)
        plt.scatter(bx, by, marker="o", color="black", zorder=4, label="bDL")

    if bDT:
        bx, by = zip(*bDT)
        plt.scatter(
            bx,
            by,
            marker="^",
            facecolors="none",
            edgecolors="black",
            zorder=4,
            label="bDT",
        )

    # --------------------------------------------------------------------
    # Configuración de la gráfica
    # --------------------------------------------------------------------
    plt.xlabel("Tiempo $t$")
    plt.ylabel(r"Macrodispersión $\langle D_\alpha(t)\rangle$")
    plt.title(
        f"Promedio de la macrodispersión (σ²={variance}, {len(dfs)} realizaciones)"
    )
    plt.legend()
    plt.xscale("log")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(outfig, dpi=300)
    plt.show()

    print(f"Figura guardada en «{outfig}»")


if __name__ == "__main__":
    main()

```

# random_field_generation.cu

```cu
#include <curand_kernel.h>
#define PI  3.141592653589793238462643383279502884 /* pi */

__global__ void setup_uniform_distrib(curandState *state, const int i_max){
	const int ix = threadIdx.x + blockIdx.x*blockDim.x;
	if (ix >= i_max) return;
	curand_init(ix, ix, 0, &state[ix]);
}


// K: conductivity field with
// K = I*exp(f(x)) (eq. 2)
// with f(x) = (2/N)^(1/2) * sigma_f^2 * sum_i_to_N cos(k1_i*x + k2_i*y + theta_i) (eq. 1)

// if exponential covariance get k1, k2 & theta with
__global__ void random_kernel_exp(curandState *state, double *k1, double *k2, double *vartheta, const double l, const int i_max){
	const int ix = threadIdx.x + blockIdx.x*blockDim.x;
	if (ix >= i_max) return;
	curandState localState = state[ix];
	double u = curand_uniform(&localState);
	double theta = curand_uniform(&localState)*2.0*PI;
	u = pow( 1.0-u , 2.0);
	u = pow ( (1.0-u)/u , 0.5);
	vartheta[ix] = curand_uniform(&localState)*2.0*PI;
	k1[ix] = u*cos(theta)/l;
	k2[ix] = u*sin(theta)/l;
	state[ix] = localState;
}

// if gaussian covariance uses k1, k2 & theta with
__global__ void random_kernel_gauss(curandState *state, double *k1, double *k2, double *vartheta, const double l, const int i_max){
	const int ix = threadIdx.x + blockIdx.x*blockDim.x;
	if (ix >= i_max) return;
	curandState localState = state[ix];
	vartheta[ix] = curand_uniform(&localState)*2.0*PI;
	k1[ix] = curand_normal_double(&localState)/l;
	k2[ix] = curand_normal_double(&localState)/l;
	state[ix] = localState;
}

// kernel for generation of field log(K)
// l: correlation length
// h: mesh size (dx=dy=h)
// sigma_f: variance
// i_max: N in eq (1)
__global__ void conductivity_kernel(double *k1, double *k2, double *vartheta, const int i_max, double *logK, const double l, const double h, const int Nx, const int Ny, const double sigma_f){
	const int ix = threadIdx.x + blockIdx.x*blockDim.x;
	const int iy = threadIdx.y + blockIdx.y*blockDim.y;
	if (ix >= Nx || iy >= Ny) return;
	int in_idx = ix + iy*Nx;
	double fx=0.0;
	for(int i = 0; i < i_max; i++)	fx += cos(h*((ix+0.5)*k1[i]+(iy+0.5)*k2[i])+vartheta[i]);
	fx = pow(2.0/(double)i_max,0.5)*sigma_f*fx;
	logK[in_idx] = fx;
}

// kernel for compute of final field K (eq. 2) after computing geometric mean
__global__ void exp_kernel(double *logK, const int Nx, const int Ny){
	const int ix = threadIdx.x + blockIdx.x*blockDim.x;
	const int iy = threadIdx.y + blockIdx.y*blockDim.y;
	if (ix >= Nx || iy >= Ny) return;
	int in_idx = ix + iy*Nx;
	logK[in_idx] = exp(logK[in_idx]);
}

//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%% RANDOM FIELD GENERATION %%%%%%%%%%%%%%%%%
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// This code for Exponential Covariance
// is based on the implementation provided in:
//
// Ludovic Räss, Dmitriy Kolyukhin, Alexander Minakov,
// "Efficient parallel random field generator for large 3-D geophysical problems,"
// Computers & Geosciences, Volume 131, Pages 158-169, 2019.
// DOI: https://doi.org/10.1016/j.cageo.2019.06.007
// URL: http://www.sciencedirect.com/science/article/pii/S0098300418309944
//
// Only minor modifications were made to adapt the original implementation to this project.
__global__ void random_kernel_3D(curandState *state, double *V1, double *V2, double *V3, double *a, double *b, const double lambda, const int i_max){
	const int ix = threadIdx.x + blockIdx.x*blockDim.x;
	if (ix >= i_max) return;
	curandState localState = state[ix];
	double fi = 2.0*PI*curand_uniform(&localState);
	double theta = acos(1.0-2.0*curand_uniform(&localState));
	double k, d;
	int flag = 1;
	while(flag==1){
		k = tan(PI*0.5*curand_uniform(&localState));
		d = (k*k)/(1.0 + (k*k));
		if(curand_uniform(&localState) < d) flag = 0;
	}
	V1[ix] = k*sin(fi)*sin(theta) / lambda;
	V2[ix] = k*cos(fi)*sin(theta) / lambda;
	V3[ix] = k*cos(theta) / lambda;
	a[ix] = pow(-2.0*log(curand_uniform(&localState)),0.5)*cos(2.0*PI*curand_uniform(&localState));
	b[ix] = pow(-2.0*log(curand_uniform(&localState)),0.5)*cos(2.0*PI*curand_uniform(&localState));
	state[ix] = localState;
}

__global__ void random_kernel_3D_gauss(curandState *state, double *V1, double *V2, double *V3, double *a, double *b, const double lambda, const int i_max, const int k_m){
	const int ix = threadIdx.x + blockIdx.x*blockDim.x;
	if (ix >= i_max) return;
	curandState localState = state[ix];
	double fi = 2.0*PI*curand_uniform(&localState);
	double theta = acos(1.0-2.0*curand_uniform(&localState));
	double k, d;
	int flag = 1;
	while(flag==1){
		k = k_m*curand_uniform(&localState);
		d = k*k*exp(-0.5*k*k);
		if(curand_uniform(&localState)*2.0*exp(-1.0) < d) flag = 0;
	}
	k = k/( 2.0*lambda/pow(PI,0.5) )*pow(2.0,0.5);
	V1[ix] = k*sin(fi)*sin(theta);
	V2[ix] = k*cos(fi)*sin(theta);
	V3[ix] = k*cos(theta);
	a[ix] = pow(-2.0*log(curand_uniform(&localState)),0.5)*cos(2.0*PI*curand_uniform(&localState));
	b[ix] = pow(-2.0*log(curand_uniform(&localState)),0.5)*cos(2.0*PI*curand_uniform(&localState));
	state[ix] = localState;
}

__global__ void conductivity_kernel_3D(double *V1, double *V2, double *V3, double *a, double *b, const int i_max, double *K, const double lambda, const double h, const int Nx, const int Ny, const int Nz,const double sigma_f){
	const int ix = threadIdx.x + blockIdx.x*blockDim.x;
	const int iy = threadIdx.y + blockIdx.y*blockDim.y;
	const int iz = threadIdx.z + blockIdx.z*blockDim.z;
	if (ix >= Nx || iy >= Ny || iz>=Nz) return;
	int in_idx = ix + iy*Nx + iz*Nx*Ny;
	double fx=0.0,tmp;
	for(int i = 0; i < i_max; i++)	{
		tmp = h*((ix+0.5)*V1[i]+(iy+0.5)*V2[i]+(iz+0.5)*V3[i]);
		fx +=a[i]*sin(tmp)+b[i]*cos(tmp);
	}
	K[in_idx] = exp(sigma_f/pow((double)i_max,(double)0.5)*fx);
}


```

# RHS_head_3D.cu

```cu
/**
* @file RHS_head_3D.cu
* @brief Compute RHS for the flow equation (no-flow, periodic, or Dirichlet BCs are admitted).
*
* @author Lucas Bessone (contact: lcbessone@gmail.com)
*
* @copyright This file is part of the EU-PAR software.
*            Copyright (C) 2025 Lucas Bessone
*
* @license This program is free software: you can redistribute it and/or modify
*          it under the terms of the GNU General Public License as published by
*          the Free Software Foundation, either version 3 of the License, or
*          (at your option) any later version.
*
*          This program is distributed in the hope that it will be useful,
*          but WITHOUT ANY WARRANTY; without even the implied warranty of
*          MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*          GNU General Public License for more details.
*
*          You should have received a copy of the GNU General Public License
*          along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#define neumann 0
#define periodic 1
#define dirichlet 2
#define gridXY dim3(grid.x,grid.y)
#define blockXY dim3(block.x,block.y)
#define gridXZ dim3(grid.x,grid.z)
#define blockXZ dim3(block.x,block.z)
#define gridYZ dim3(grid.y,grid.z)
#define blockYZ dim3(block.y,block.z)


__global__ void RHS_head_int(double *RHS, const double *K, int Nx, int Ny, int Nz){
	const int ix = threadIdx.x + blockIdx.x*blockDim.x;
	const int iy = threadIdx.y + blockIdx.y*blockDim.y;
	if (ix >= Nx-2 || iy >= Ny-2) return;
	int in_idx = (ix + 1) + (iy + 1)*Nx;
	int stride = Nx*Ny;
	for(int i=1; i<Nz-1; ++i){
		in_idx += stride;
		RHS[in_idx] = 0.0;
	}
}

__global__ void RHS_head_side_bottom(double *RHS, const double *K, int Nx, int Ny, int Nz, double A, double h, int BCtype, double Hb){
	int ix = threadIdx.x + blockIdx.x*blockDim.x;
	int iy = threadIdx.y + blockIdx.y*blockDim.y;
	if (ix >= Nx-2 || iy >= Ny-2) return;
	int stride = Nx*Ny;
	int in_idx;
	int iz=0;
	in_idx = (ix+1) + (iy+1)*Nx + iz*stride;
	double KC = K[in_idx];
	if(BCtype==dirichlet) RHS[in_idx] -= 2.0*Hb*KC/h*A/h/h/h; //dirichlet contribution
}

__global__ void RHS_head_side_top(double *RHS, const double *K, int Nx, int Ny, int Nz, double A, double h, int BCtype, double Hb){
    int ix = threadIdx.x + blockIdx.x*blockDim.x;
    int iy = threadIdx.y + blockIdx.y*blockDim.y;
    if (ix >= Nx-2 || iy >= Ny-2) return;
    int stride = Nx*Ny;
    int iz=Nz-1;
    int in_idx = (ix+1) + (iy+1)*Nx + iz*stride;
    double KC = K[in_idx];
    if(BCtype==dirichlet) RHS[in_idx] -= 2.0*Hb*KC/h*A/h/h/h; //dirichlet contribution
}

__global__ void RHS_head_side_south(double *RHS, const double *K, int Nx, int Ny, int Nz, double A, double h, int BCtype, double Hb){
	int ix = threadIdx.x + blockIdx.x*blockDim.x;
	int iz = threadIdx.y + blockIdx.y*blockDim.y;
	if (ix >= Nx-2 || iz >= Nz-2) return;
	int stride = Nx*Ny;
	int in_idx;
	int iy = 0;
	in_idx = (ix + 1) + iy*Nx + (iz + 1)*stride;
    double KC = K[in_idx];
    if(BCtype==dirichlet) RHS[in_idx] -= 2.0*Hb*KC/h*A/h/h/h; //dirichlet contribution
}

__global__ void RHS_head_side_north(double *RHS, const double *K, int Nx, int Ny, int Nz, double A, double h, int BCtype, double Hb){
	int ix = threadIdx.x + blockIdx.x*blockDim.x;
	int iz = threadIdx.y + blockIdx.y*blockDim.y;
	if (ix >= Nx-2 || iz >= Nz-2) return;
	int stride = Nx*Ny;
	int in_idx;
	int iy = Ny-1;
	in_idx = (ix + 1) + iy*Nx + (iz + 1)*stride;
    double KC = K[in_idx];
    if(BCtype==dirichlet) RHS[in_idx] -= 2.0*Hb*KC/h*A/h/h/h; //dirichlet contribution
}

__global__ void RHS_head_side_west(double *RHS, const double *K, int Nx, int Ny, int Nz, double A, double h, int BCtype, double Hb){
	int iy = threadIdx.x + blockIdx.x*blockDim.x;
	int iz  = threadIdx.y + blockIdx.y*blockDim.y;
	if (iy >= Ny-2 || iz >= Nz-2) return;
	int stride = Nx*Ny;
	int in_idx;
	int ix = 0;
	in_idx = ix + (iy + 1)*Nx + (iz + 1)*stride;
    double KC = K[in_idx];
    if(BCtype==dirichlet) RHS[in_idx] -= 2.0*Hb*KC/h*A/h/h/h; //dirichlet contribution
}

__global__ void RHS_head_side_east(double *RHS, const double *K, int Nx, int Ny, int Nz, double A, double h, int BCtype, double Hb){
	int iy = threadIdx.x + blockIdx.x*blockDim.x;
	int iz  = threadIdx.y + blockIdx.y*blockDim.y;
	if (iy >= Ny-2 || iz >= Nz-2) return;
	int stride = Nx*Ny;
	int in_idx;
	int ix = Nx-1;
	in_idx = ix + (iy + 1)*Nx + (iz + 1)*stride;
    double KC = K[in_idx];
    if(BCtype==dirichlet) RHS[in_idx] -= 2.0*Hb*KC/h*A/h/h/h; //dirichlet contribution
}

__global__ void RHS_head_edge_X_South_Bottom(double *RHS, const double *K, int Nx, int Ny, int Nz, double A, double h, int BCsouth, int BCbottom, double Hsouth, double Hbottom){
	int ix = threadIdx.x + blockIdx.x*blockDim.x;
	if (ix >= Nx-2) return;
	int stride = Nx*Ny;
	int iy = 0;
	int iz = 0;
	int in_idx;
	in_idx = (ix+1) + iy*Nx + iz*stride;
    double KC = K[in_idx];
    if(BCsouth==dirichlet) RHS[in_idx] -= 2.0*Hsouth*KC/h*A/h/h/h; //dirichlet contribution
    if(BCbottom==dirichlet) RHS[in_idx] -= 2.0*Hbottom*KC/h*A/h/h/h; //dirichlet contribution
   }


__global__ void RHS_head_edge_X_South_Top(double *RHS, const double *K, int Nx, int Ny, int Nz, double A, double h, int BCsouth, int BCtop, double Hsouth, double Htop){
	int ix = threadIdx.x + blockIdx.x*blockDim.x;
	if (ix >= Nx-2) return;
	int stride = Nx*Ny;
	int iy = 0;
	int iz = Nz-1;
	int in_idx = (ix+1) + iy*Nx + iz*stride;
    double KC = K[in_idx];
    if(BCsouth==dirichlet) RHS[in_idx] -= 2.0*Hsouth*KC/h*A/h/h/h;
    if(BCtop==dirichlet) RHS[in_idx] -= 2.0*Htop*KC/h*A/h/h/h;
}

__global__ void RHS_head_edge_X_North_Bottom(double *RHS, const double *K, int Nx, int Ny, int Nz, double A, double h, int BCnorth, int BCbottom, double Hnorth, double Hbottom){
    int ix = threadIdx.x + blockIdx.x*blockDim.x;
    if (ix >= Nx-2) return;
    int stride = Nx*Ny;
    int iy = Ny-1;
    int iz = 0;
    int in_idx = (ix+1) + iy*Nx + iz*stride;
    double KC = K[in_idx];
    if(BCnorth==dirichlet) RHS[in_idx] -= 2.0*Hnorth*KC/h*A/h/h/h;
    if(BCbottom==dirichlet) RHS[in_idx] -= 2.0*Hbottom*KC/h*A/h/h/h;
}

__global__ void RHS_head_edge_X_North_Top(double *RHS, const double *K, int Nx, int Ny, int Nz, double A, double h, int BCnorth, int BCtop, double Hnorth, double Htop){
    int ix = threadIdx.x + blockIdx.x*blockDim.x;
    if (ix >= Nx-2) return;
    int stride = Nx*Ny;
    int iy = Ny-1;
    int iz = Nz-1;
    int in_idx = (ix+1) + iy*Nx + iz*stride;
    double KC = K[in_idx];
    if(BCnorth==dirichlet) RHS[in_idx] -= 2.0*Hnorth*KC/h*A/h/h/h;
    if(BCtop==dirichlet) RHS[in_idx] -= 2.0*Htop*KC/h*A/h/h/h;
}

__global__ void RHS_head_edge_Z_South_West(double *RHS, const double *K, int Nx, int Ny, int Nz, double A, double h, int BCsouth, int BCwest, double Hsouth, double Hwest){
	int iz = threadIdx.x + blockIdx.x*blockDim.x;
	if (iz >= Nz-2) return;
	int stride = Nx*Ny;
	int ix = 0;
	int iy = 0;
	int in_idx;
	in_idx = ix + iy*Nx + (iz+1)*stride;
    double KC = K[in_idx];
    if(BCsouth==dirichlet) RHS[in_idx] -= 2.0*Hsouth*KC/h*A/h/h/h;
    if(BCwest==dirichlet) RHS[in_idx] -= 2.0*Hwest*KC/h*A/h/h/h;
}

__global__ void RHS_head_edge_Z_South_East(double *RHS, const double *K, int Nx, int Ny, int Nz, double A, double h, int BCsouth, int BCeast, double Hsouth, double Heast){
	int iz = threadIdx.x + blockIdx.x*blockDim.x;
	if (iz >= Nz-2) return;
	int stride = Nx*Ny;
	int ix = Nx-1;
	int iy = 0;
	int in_idx;
	in_idx = ix + iy*Nx + (iz+1)*stride;
    double KC = K[in_idx];
    if(BCsouth==dirichlet) RHS[in_idx] -= 2.0*Hsouth*KC/h*A/h/h/h;
    if(BCeast==dirichlet) RHS[in_idx] -= 2.0*Heast*KC/h*A/h/h/h;
}

__global__ void RHS_head_edge_Z_North_West(double *RHS, const double *K, int Nx, int Ny, int Nz, double A, double h, int BCnorth, int BCwest, double Hnorth, double Hwest){
	int iz = threadIdx.x + blockIdx.x*blockDim.x;
	if (iz >= Nz-2) return;
	int stride = Nx*Ny;
	int ix = 0;
	int iy = Ny-1;
	int in_idx;
	in_idx = ix + iy*Nx + (iz+1)*stride;
    double KC = K[in_idx];
    if(BCnorth==dirichlet) RHS[in_idx] -= 2.0*Hnorth*KC/h*A/h/h/h;
    if(BCwest==dirichlet) RHS[in_idx] -= 2.0*Hwest*KC/h*A/h/h/h;
}

__global__ void RHS_head_edge_Z_North_East(double *RHS, const double *K, int Nx, int Ny, int Nz, double A, double h, int BCnorth, int BCeast, double Hnorth, double Heast){
	int iz = threadIdx.x + blockIdx.x*blockDim.x;
	if (iz >= Nz-2) return;
	int stride = Nx*Ny;
	int ix = Nx-1;
	int iy = Ny-1;
	int in_idx;
	in_idx = ix + iy*Nx + (iz+1)*stride;
    double KC = K[in_idx];
    if(BCnorth==dirichlet) RHS[in_idx] -= 2.0*Hnorth*KC/h*A/h/h/h;
    if(BCeast==dirichlet) RHS[in_idx] -= 2.0*Heast*KC/h*A/h/h/h;
}

__global__ void RHS_head_edge_Y_Bottom_West(double *RHS, const double *K, int Nx, int Ny, int Nz, double A, double h, int BCbottom, int BCwest, double Hbottom, double Hwest){
	int iy = threadIdx.x + blockIdx.x*blockDim.x;
	if (iy >= Ny-2) return;
	int stride = Nx*Ny;
	int ix = 0;
	int iz = 0;
	int in_idx;
	in_idx = ix + (iy+1)*Nx + iz*stride;
    double KC = K[in_idx];
    if(BCwest==dirichlet) RHS[in_idx] -= 2.0*Hwest*KC/h*A/h/h/h;
    if(BCbottom==dirichlet) RHS[in_idx] -= 2.0*Hbottom*KC/h*A/h/h/h;
}

__global__ void RHS_head_edge_Y_Top_West(double *RHS, const double *K, int Nx, int Ny, int Nz, double A, double h, int BCtop, int BCwest, double Htop, double Hwest){
	int iy = threadIdx.x + blockIdx.x*blockDim.x;
	if (iy >= Ny-2) return;
	int stride = Nx*Ny;
	int ix = 0;
	int iz = Nz-1;
	int in_idx;
	in_idx = ix + (iy+1)*Nx + iz*stride;
    double KC = K[in_idx];
    if(BCwest==dirichlet) RHS[in_idx] -= 2.0*Hwest*KC/h*A/h/h/h;
    if(BCtop==dirichlet) RHS[in_idx] -= 2.0*Htop*KC/h*A/h/h/h;
}

__global__ void RHS_head_edge_Y_Bottom_East(double *RHS, const double *K, int Nx, int Ny, int Nz, double A, double h, int BCbottom, int BCeast, double Hbottom, double Heast){
	int iy = threadIdx.x + blockIdx.x*blockDim.x;
	if (iy >= Ny-2) return;
	int stride = Nx*Ny;
	int ix = Nx-1;
	int iz = 0;
	int in_idx;
	in_idx = ix + (iy+1)*Nx + iz*stride;
    double KC = K[in_idx];
    if(BCeast==dirichlet) RHS[in_idx] -= 2.0*Heast*KC/h*A/h/h/h;
    if(BCbottom==dirichlet) RHS[in_idx] -= 2.0*Hbottom*KC/h*A/h/h/h;
}

__global__ void RHS_head_edge_Y_Top_East(double *RHS, const double *K, int Nx, int Ny, int Nz, double A, double h, int BCtop, int BCeast, double Htop, double Heast){
	int iy = threadIdx.x + blockIdx.x*blockDim.x;
	if (iy >= Ny-2) return;
	int stride = Nx*Ny;
	int ix = Nx-1;
	int iz = Nz-1;
	int in_idx;
	in_idx = ix + (iy+1)*Nx + iz*stride;
    double KC = K[in_idx];
    if(BCeast==dirichlet) RHS[in_idx] -= 2.0*Heast*KC/h*A/h/h/h;
    if(BCtop==dirichlet) RHS[in_idx] -= 2.0*Htop*KC/h*A/h/h/h;
}

__global__ void RHS_head_vertex_SWB(double *RHS, const double *K, int Nx, int Ny, int Nz, double A, double h,
                                  int BCsouth, int BCwest, int BCbottom,
                                  double Hsouth, double Hwest, double Hbottom) {
    int ix = 0;
    int iy = 0;
    int iz = 0;
    int stride = Nx*Ny;
    int in_idx = ix + iy*Nx + iz*stride;
    double KC = K[in_idx];
    if(BCsouth==dirichlet) RHS[in_idx] -= 2.0*Hsouth*KC/h*A/h/h/h;
    if(BCwest==dirichlet) RHS[in_idx] -= 2.0*Hwest*KC/h*A/h/h/h;
    if(BCbottom==dirichlet) RHS[in_idx] -= 2.0*Hbottom*KC/h*A/h/h/h;
}

__global__ void RHS_head_vertex_SWT(double *RHS, const double *K, int Nx, int Ny, int Nz, double A, double h,
                                  int BCsouth, int BCwest, int BCtop,
                                  double Hsouth, double Hwest, double Htop) {
	int stride = Nx*Ny;
	int ix = 0;
	int iy = 0;
	int iz = Nz-1;
	int in_idx;
	in_idx = ix + iy*Nx + iz*stride;
    double KC = K[in_idx];
    if(BCsouth==dirichlet) RHS[in_idx] -= 2.0*Hsouth*KC/h*A/h/h/h;
    if(BCwest==dirichlet) RHS[in_idx] -= 2.0*Hwest*KC/h*A/h/h/h;
    if(BCtop==dirichlet) RHS[in_idx] -= 2.0*Htop*KC/h*A/h/h/h;
}

__global__ void RHS_head_vertex_SEB(double *RHS, const double *K, int Nx, int Ny, int Nz, double A, double h,
                                  int BCsouth, int BCeast, int BCbottom,
                                  double Hsouth, double Heast, double Hbottom) {
	int stride = Nx*Ny;
	int ix = Nx-1;
	int iy = 0;
	int iz = 0;
	int in_idx;
	in_idx = ix + iy*Nx + iz*stride;
    double KC = K[in_idx];
    if(BCsouth==dirichlet) RHS[in_idx] -= 2.0*Hsouth*KC/h*A/h/h/h;
    if(BCeast==dirichlet) RHS[in_idx] -= 2.0*Heast*KC/h*A/h/h/h;
    if(BCbottom==dirichlet) RHS[in_idx] -= 2.0*Hbottom*KC/h*A/h/h/h;
}

__global__ void RHS_head_vertex_SET(double *RHS, const double *K, int Nx, int Ny, int Nz, double A, double h,
                                  int BCsouth, int BCeast, int BCtop,
                                  double Hsouth, double Heast, double Htop) {
	int stride = Nx*Ny;
	int ix = Nx-1;
	int iy = 0;
	int iz = Nz-1;
	int in_idx;
	in_idx = ix + iy*Nx + iz*stride;
    double KC = K[in_idx];
    if(BCsouth==dirichlet) RHS[in_idx] -= 2.0*Hsouth*KC/h*A/h/h/h;
    if(BCeast==dirichlet) RHS[in_idx] -= 2.0*Heast*KC/h*A/h/h/h;
    if(BCtop==dirichlet) RHS[in_idx] -= 2.0*Htop*KC/h*A/h/h/h;
}

__global__ void RHS_head_vertex_NWB(double *RHS, const double *K, int Nx, int Ny, int Nz, double A, double h,
                                  int BCnorth, int BCwest, int BCbottom,
                                  double Hnorth, double Hwest, double Hbottom) {
	int stride = Nx*Ny;
	int ix = 0;
	int iy = Ny-1;
	int iz = 0;
	int in_idx;
	in_idx = ix + iy*Nx + iz*stride;
    double KC = K[in_idx];
    if(BCnorth==dirichlet) RHS[in_idx] -= 2.0*Hnorth*KC/h*A/h/h/h;
    if(BCwest==dirichlet) RHS[in_idx] -= 2.0*Hwest*KC/h*A/h/h/h;
    if(BCbottom==dirichlet) RHS[in_idx] -= 2.0*Hbottom*KC/h*A/h/h/h;
}

__global__ void RHS_head_vertex_NWT(double *RHS, const double *K, int Nx, int Ny, int Nz, double A, double h,
                                  int BCnorth, int BCwest, int BCtop,
                                  double Hnorth, double Hwest, double Htop) {
	int stride = Nx*Ny;
	int ix = 0;
	int iy = Ny-1;
	int iz = Nz-1;
	int in_idx;
	in_idx = ix + iy*Nx + iz*stride;

    double KC = K[in_idx];
    if(BCnorth==dirichlet) RHS[in_idx] -= 2.0*Hnorth*KC/h*A/h/h/h;
    if(BCwest==dirichlet) RHS[in_idx] -= 2.0*Hwest*KC/h*A/h/h/h;
    if(BCtop==dirichlet) RHS[in_idx] -= 2.0*Htop*KC/h*A/h/h/h;
}

__global__ void RHS_head_vertex_NEB(double *RHS, const double *K, int Nx, int Ny, int Nz, double A, double h,
                                  int BCnorth, int BCeast, int BCbottom,
                                  double Hnorth, double Heast, double Hbottom) {
	int stride = Nx*Ny;
	int ix = Nx-1;
	int iy = Ny-1;
	int iz = 0;
	int in_idx;
	in_idx = ix + iy*Nx + iz*stride;
    double KC = K[in_idx];
    if(BCnorth==dirichlet) RHS[in_idx] -= 2.0*Hnorth*KC/h*A/h/h/h;
    if(BCeast==dirichlet) RHS[in_idx] -= 2.0*Heast*KC/h*A/h/h/h;
    if(BCbottom==dirichlet) RHS[in_idx] -= 2.0*Hbottom*KC/h*A/h/h/h;
}

__global__ void RHS_head_vertex_NET(double *RHS, const double *K, int Nx, int Ny, int Nz, double A, double h,
                                  int BCnorth, int BCeast, int BCtop,
                                  double Hnorth, double Heast, double Htop) {
	int stride = Nx*Ny;
	int ix = Nx-1;
	int iy = Ny-1;
	int iz = Nz-1;
	int in_idx;
	in_idx = ix + iy*Nx + iz*stride;
    double KC = K[in_idx];
    if(BCnorth==dirichlet) RHS[in_idx] -= 2.0*Hnorth*KC/h*A/h/h/h;
    if(BCeast==dirichlet) RHS[in_idx] -= 2.0*Heast*KC/h*A/h/h/h;
    if(BCtop==dirichlet) RHS[in_idx] -= 2.0*Htop*KC/h*A/h/h/h;
}

void RHS_head(double *RHS, const double *K,
	int Nx, int Ny, int Nz, double A, double h,
	int BCbottom, int BCtop, int BCsouth, int BCnorth, int BCwest, int BCeast,
	double Hbottom, double Htop, double Hsouth, double Hnorth, double Hwest, double Heast,
	dim3 grid, dim3 block){
	RHS_head_int		<<<gridXY,blockXY>>>(RHS,K,Nx,Ny,Nz);
	RHS_head_side_bottom<<<gridXY,blockXY>>>(RHS,K,Nx,Ny,Nz,A,h,BCbottom,Hbottom);
	RHS_head_side_top	<<<gridXY,blockXY>>>(RHS,K,Nx,Ny,Nz,A,h,BCtop,Htop);
	RHS_head_side_south<<<gridXZ,blockXZ>>>(RHS,K,Nx,Ny,Nz,A,h,BCsouth,Hsouth);
	RHS_head_side_north<<<gridXZ,blockXZ>>>(RHS,K,Nx,Ny,Nz,A,h,BCnorth,Hnorth);
	RHS_head_side_west<<<gridYZ,blockYZ>>>(RHS,K,Nx,Ny,Nz,A,h,BCwest,Hwest);
	RHS_head_side_east<<<gridYZ,blockYZ>>>(RHS,K,Nx,Ny,Nz,A,h,BCeast,Heast);
	RHS_head_edge_X_South_Bottom<<<grid.x,block.x>>>(RHS,K,Nx,Ny,Nz,A,h,BCsouth,BCbottom,Hsouth,Hbottom);
	RHS_head_edge_X_South_Top	<<<grid.x,block.x>>>(RHS,K,Nx,Ny,Nz,A,h,BCsouth,BCtop,Hsouth,Htop);
	RHS_head_edge_X_North_Bottom<<<grid.x,block.x>>>(RHS,K,Nx,Ny,Nz,A,h,BCnorth,BCbottom,Hnorth,Hbottom);
	RHS_head_edge_X_North_Top	<<<grid.x,block.x>>>(RHS,K,Nx,Ny,Nz,A,h,BCnorth,BCtop,Hnorth,Htop);
	RHS_head_edge_Z_South_West<<<grid.z,block.z>>>(RHS,K,Nx,Ny,Nz,A,h,BCsouth,BCwest,Hsouth,Hwest);
	RHS_head_edge_Z_South_East<<<grid.z,block.z>>>(RHS,K,Nx,Ny,Nz,A,h,BCsouth,BCeast,Hsouth,Heast);
	RHS_head_edge_Z_North_West<<<grid.z,block.z>>>(RHS,K,Nx,Ny,Nz,A,h,BCnorth,BCwest,Hnorth,Hwest);
	RHS_head_edge_Z_North_East<<<grid.z,block.z>>>(RHS,K,Nx,Ny,Nz,A,h,BCnorth,BCeast,Hnorth,Heast);
	RHS_head_edge_Y_Bottom_West	<<<grid.y,block.y>>>(RHS,K,Nx,Ny,Nz,A,h,BCbottom,BCwest,Hbottom,Hwest);
	RHS_head_edge_Y_Top_West	<<<grid.y,block.y>>>(RHS,K,Nx,Ny,Nz,A,h,BCtop,BCwest,Htop,Hwest);
	RHS_head_edge_Y_Bottom_East	<<<grid.y,block.y>>>(RHS,K,Nx,Ny,Nz,A,h,BCbottom,BCeast,Hbottom,Heast);
	RHS_head_edge_Y_Top_East	<<<grid.y,block.y>>>(RHS,K,Nx,Ny,Nz,A,h,BCtop,BCeast,Htop,Heast);
	RHS_head_vertex_SWB<<<1,1>>>(RHS,K,Nx,Ny,Nz,A,h,BCsouth,BCwest,BCbottom,Hsouth,Hwest,Hbottom);
	RHS_head_vertex_SWT<<<1,1>>>(RHS,K,Nx,Ny,Nz,A,h,BCsouth,BCwest,BCtop,Hsouth,Hwest,Htop);
	RHS_head_vertex_SEB<<<1,1>>>(RHS,K,Nx,Ny,Nz,A,h,BCsouth,BCeast,BCbottom,Hsouth,Heast,Hbottom);
	RHS_head_vertex_SET<<<1,1>>>(RHS,K,Nx,Ny,Nz,A,h,BCsouth,BCeast,BCtop,Hsouth,Heast,Htop);
	RHS_head_vertex_NWB<<<1,1>>>(RHS,K,Nx,Ny,Nz,A,h,BCnorth,BCwest,BCbottom,Hnorth,Hwest,Hbottom);
	RHS_head_vertex_NWT<<<1,1>>>(RHS,K,Nx,Ny,Nz,A,h,BCnorth,BCwest,BCtop,Hnorth,Hwest,Htop);
	RHS_head_vertex_NEB<<<1,1>>>(RHS,K,Nx,Ny,Nz,A,h,BCnorth,BCeast,BCbottom,Hnorth,Heast,Hbottom);
	RHS_head_vertex_NET<<<1,1>>>(RHS,K,Nx,Ny,Nz,A,h,BCnorth,BCeast,BCtop,Hnorth,Heast,Htop);

	cudaDeviceSynchronize();
}
```

# run_test.sh

```sh
#!/bin/bash
set -euo pipefail

rm -rf ./output/* 2>/dev/null || true
mkdir -p ./output/log

# --- Cola GPU 1 ---
commands_gpu1=(
  "./run_flow2.out 1 1 > ./output/log/var_1.log 2>&1"
  "./run_flow2.out 1 4 > ./output/log/var_4.log 2>&1"
)

# --- Cola GPU 0 ---
commands_gpu0=(
  "./run_flow2.out 0 0.25 > ./output/log/var_0.25.log 2>&1"
  "./run_flow2.out 0 2.25 > ./output/log/var_2.25.log 2>&1"
  "./run_flow2.out 0 6.25 > ./output/log/var_6.25.log 2>&1"
)

ts() { date '+%Y-%m-%d %H:%M:%S'; }

run_queue_gpu1() {
  for cmd in "${commands_gpu1[@]}"; do
    echo "[$(ts)] [GPU1] Ejecutando: $cmd"
    eval "$cmd"
    echo "[$(ts)] [GPU1] Finalizado: $cmd"
  done
  echo "[$(ts)] [GPU1] Cola completada."
}

run_queue_gpu0() {
  for cmd in "${commands_gpu0[@]}"; do
    echo "[$(ts)] [GPU0] Ejecutando: $cmd"
    eval "$cmd"
    echo "[$(ts)] [GPU0] Finalizado: $cmd"
  done
  echo "[$(ts)] [GPU0] Cola completada."
}

# Ejecutar ambas colas en paralelo
run_queue_gpu1 &
pid1=$!
run_queue_gpu0 &
pid0=$!

wait $pid1
wait $pid0

echo "[$(ts)] Todas las colas finalizaron."

```

# stencil_head_3D.cu

```cu
/**
* @file stencil_head_3D.cu
* @brief Stencil matrix-vector product (matrix-free) implementation for solving the flow equation.
* (no-flow, periodic, or Dirichlet BCs are admitted).
*
* @author Lucas Bessone (contact: lcbessone@gmail.com)
*
* @copyright This file is part of the EU-PAR software.
*            Copyright (C) 2025 Lucas Bessone
*
* @license This program is free software: you can redistribute it and/or modify
*          it under the terms of the GNU General Public License as published by
*          the Free Software Foundation, either version 3 of the License, or
*          (at your option) any later version.
*
*          This program is distributed in the hope that it will be useful,
*          but WITHOUT ANY WARRANTY; without even the implied warranty of
*          MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*          GNU General Public License for more details.
*
*          You should have received a copy of the GNU General Public License
*          along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/
#include <cuda_runtime_api.h>

#define neumann 0
#define periodic 1
#define dirichlet 2
#define gridXY dim3(grid.x,grid.y)
#define blockXY dim3(block.x,block.y)
#define gridXZ dim3(grid.x,grid.z)
#define blockXZ dim3(block.x,block.z)
#define gridYZ dim3(grid.y,grid.z)
#define blockYZ dim3(block.y,block.z)

//#######################################################################
// 	routine stencil_head
//  linear operator (stencil op) - SpMV matrix-free version
//	the coefficients of A are obtained from the conductivity vector K
//	K face value are interpolated with Harmonic mean
//  BCtype: Boundary Condition Type
//#######################################################################
// interior
__global__ void stencil_head_int(double *H_output, const double *H_input, const double *K, int Nx, int Ny, int Nz, double A, double h){
	// __shared__ double s_H[1+BLOCK_Nx+1][1+BLOCK_Ny+1]; //load in shared mem
	// __shared__ double s_K[1+BLOCK_Nx+1][1+BLOCK_Ny+1]; //load in shared mem
	const int ix = threadIdx.x + blockIdx.x*blockDim.x;
	const int iy = threadIdx.y + blockIdx.y*blockDim.y;
	if (ix >= Nx-2 || iy >= Ny-2) return;
	int in_idx = (ix + 1) + (iy + 1)*Nx;
	int stride = Nx*Ny;

	int out_idx;
	double H_top, H_bottom, H_current;
	double K_top, K_bottom, K_current;

	// const int tx = threadIdx.x + 1;
	// const int ty = threadIdx.y + 1;

	H_current = H_input[in_idx];
	K_current = K[in_idx];
	out_idx = in_idx;
	in_idx += stride;

	H_top = H_input[in_idx];
	K_top = K[in_idx];
	in_idx += stride;
	double value;
	for(int i=1; i<Nz-1; ++i){
		// if( (tx < Nx-1) && (ty < Ny-1) ){
			H_bottom = H_current;
			H_current = H_top;
			H_top = H_input[in_idx];

			K_bottom = K_current;
			K_current = K_top;
			K_top = K[in_idx];

			in_idx += stride;
			out_idx += stride;
			// if(tx==1){
			// 	s_H[0][ty]    = H_input[out_idx-1];
			// 	s_K[0][ty]    = K[out_idx-1];
			// }
			// if(ix==Nx-3 || tx==BLOCK_Nx){
			// 	s_H[tx+1][ty] = H_input[out_idx+1];
			// 	s_K[tx+1][ty] = K[out_idx+1];
			// }

			// if(ty==1){
			// 	s_H[tx][0]    = H_input[out_idx-Nx];
			// 	s_K[tx][0]    = K[out_idx-Nx];
			// }
			// if(iy==Ny-3 || ty==BLOCK_Ny){
			// 	s_H[tx][ty+1] = H_input[out_idx+Nx];
			// 	s_K[tx][ty+1] = K[out_idx+Nx];
			// }

			// s_H[tx][ty] = H_current;
			// s_K[tx][ty] = K_current;
			// __syncthreads();
			// value  = (H_current-s_H[tx+1][ty]) /  (1.0/K_current  +  1.0/s_K[tx+1][ty]);
			// value += (H_current-s_H[tx][ty+1]) /  (1.0/K_current  +  1.0/s_K[tx][ty+1]);
			// value += (H_current-s_H[tx-1][ty]) /  (1.0/K_current  +  1.0/s_K[tx-1][ty]);
			// value += (H_current-s_H[tx][ty-1]) /  (1.0/K_current  +  1.0/s_K[tx][ty-1]);
			// value += (H_current-H_top)         /  (1.0/K_current  +  1.0/K_top        );
			// value += (H_current-H_bottom)      /  (1.0/K_current  +  1.0/K_bottom     );
			value  = (H_current-H_input[out_idx+1]) /  (1.0/K_current  +  1.0/K[out_idx+1]);
			value += (H_current-H_input[out_idx+Nx])/  (1.0/K_current  +  1.0/K[out_idx+Nx]);
			value += (H_current-H_input[out_idx-1]) /  (1.0/K_current  +  1.0/K[out_idx-1]);
			value += (H_current-H_input[out_idx-Nx])/  (1.0/K_current  +  1.0/K[out_idx-Nx]);
			value += (H_current-H_top)         /  (1.0/K_current  +  1.0/K_top        );
			value += (H_current-H_bottom)      /  (1.0/K_current  +  1.0/K_bottom     );

			H_output[out_idx] = -2.0*value/h*A/h/h/h;
		// 	__syncthreads();
		// }
	}
}

// total 6-faces
// xy-face (BCtype: Homogeneous-Neumann (0), Periodic (1), Homogeneous-Dirichlet (2))
__global__ void stencil_head_side_bottom(double *H_output, const double *H_input, const double *K, int Nx, int Ny, int Nz, double A, double h, int BCtype){
	int ix = threadIdx.x + blockIdx.x*blockDim.x;
	int iy = threadIdx.y + blockIdx.y*blockDim.y;
	if (ix >= Nx-2 || iy >= Ny-2) return;
	int stride = Nx*Ny;
	int in_idx;
	int iz=0;
	in_idx = (ix+1) + (iy+1)*Nx + iz*stride;

	double HC = H_input[in_idx];
	double KC = K[in_idx];
    // *************
	//default BCtype Homogeneous - Neumann or dirichlet
	double H_bottom = HC; //factor (HC-H_bottom) is zero 0
	double K_bottom = KC; //(1.0/KC+1.0/K_bottom) <- \neq0
	double result = 0.0;
	if(BCtype>2) return;
	if(BCtype==periodic) {
		H_bottom = H_input[in_idx+(Nz-1)*stride];
		K_bottom = K[in_idx+(Nz-1)*stride];
	}
    // *************
	result  = (HC-H_input[in_idx+1])      /  (1.0/KC  +  1.0/K[in_idx+1]);
	result += (HC-H_input[in_idx+Nx])     /  (1.0/KC  +  1.0/K[in_idx+Nx]);
	result += (HC-H_input[in_idx-1])      /  (1.0/KC  +  1.0/K[in_idx-1]);
	result += (HC-H_input[in_idx-Nx])     /  (1.0/KC  +  1.0/K[in_idx-Nx]);
	result += (HC-H_input[in_idx+stride]) /  (1.0/KC  +  1.0/K[in_idx+stride]);
	result += (HC-H_bottom)      /  (1.0/KC  +  1.0/K_bottom     );
	// *************
	if(BCtype==dirichlet) result += HC*KC; //dirichlet contribution
	// *************
	H_output[in_idx] = -2.0*result/h*A/h/h/h;
}
// xy-face (BCtype: Homogeneous-Neumann (0), Periodic (1), Homogeneous-Dirichlet (2))
__global__ void stencil_head_side_top(double *H_output, const double *H_input, const double *K, int Nx, int Ny, int Nz, double A, double h, int BCtype){
	int ix = threadIdx.x + blockIdx.x*blockDim.x;
	int iy = threadIdx.y + blockIdx.y*blockDim.y;
	if (ix >= Nx-2 || iy >= Ny-2) return;
	int stride = Nx*Ny;
	int in_idx;
	int iz=Nz-1;
	in_idx = (ix+1) + (iy+1)*Nx + iz*stride;

	double HC = H_input[in_idx];
	double KC = K[in_idx];
	// *************
	//default BCtype Homogeneous - Neumann or dirichlet
	double H_top = HC; //factor (HC-H_top) is zero 0
	double K_top = KC; //(1.0/KC+1.0/K_top) <- \neq0
	double result=0.0;
	if(BCtype>2) return;
	if(BCtype==periodic) {
		H_top = H_input[in_idx-(Nz-1)*stride];
		K_top = K[in_idx-(Nz-1)*stride];
	}
	// *************
	result  = (HC-H_input[in_idx+1])      /  (1.0/KC  +  1.0/K[in_idx+1]);
	result += (HC-H_input[in_idx+Nx])     /  (1.0/KC  +  1.0/K[in_idx+Nx]);
	result += (HC-H_input[in_idx-1])      /  (1.0/KC  +  1.0/K[in_idx-1]);
	result += (HC-H_input[in_idx-Nx])     /  (1.0/KC  +  1.0/K[in_idx-Nx]);
	result += (HC-H_input[in_idx-stride]) /  (1.0/KC  +  1.0/K[in_idx-stride]);
	result += (HC-H_top)      /  (1.0/KC  +  1.0/K_top     );
	// *************
	if(BCtype==dirichlet) result += HC*KC; //dirichlet contribution
	// *************
	H_output[in_idx] = -2.0*result/h*A/h/h/h;
}

// xz-face (BCtype: Homogeneous-Neumann (0), Periodic (1), Homogeneous-Dirichlet (2))
__global__ void stencil_head_side_south(double *H_output, const double *H_input, const double *K, int Nx, int Ny, int Nz, double A, double h, int BCtype){
	int ix = threadIdx.x + blockIdx.x*blockDim.x;
	int iz = threadIdx.y + blockIdx.y*blockDim.y;
	if (ix >= Nx-2 || iz >= Nz-2) return;
	int stride = Nx*Ny;
	int in_idx;
	int iy = 0;
	in_idx = (ix + 1) + iy*Nx + (iz + 1)*stride;

	double HC = H_input[in_idx];
	double KC = K[in_idx];
	// *************
	//default BCtype Homogeneous - Neumann or dirichlet
	double H_south = HC; //factor (HC-H_south) is zero 0
	double K_south = KC; //(1.0/KC+1.0/K_south) <- \neq0
	double result=0.0;
	if(BCtype>2) return;
	if(BCtype==periodic) {
		H_south = H_input[in_idx+(Ny-1)*Nx];
		K_south = K[in_idx+(Ny-1)*Nx];
	}
	// *************
	result  = (HC-H_input[in_idx+1])      /  (1.0/KC  +  1.0/K[in_idx+1]);
	result += (HC-H_input[in_idx+Nx])     /  (1.0/KC  +  1.0/K[in_idx+Nx]);
	result += (HC-H_input[in_idx-1])      /  (1.0/KC  +  1.0/K[in_idx-1]);
	result += (HC-H_input[in_idx-stride]) /  (1.0/KC  +  1.0/K[in_idx-stride]);
	result += (HC-H_input[in_idx+stride]) /  (1.0/KC  +  1.0/K[in_idx+stride]);
	result += (HC-H_south)      /  (1.0/KC  +  1.0/K_south     );
	// *************
	if(BCtype==dirichlet) result += HC*KC; //dirichlet contribution
	// *************
	H_output[in_idx] = -2.0*result/h*A/h/h/h;
}

// xz-face (BCtype: Homogeneous-Neumann (0), Periodic (1), Homogeneous-Dirichlet (2))
__global__ void stencil_head_side_north(double *H_output, const double *H_input, const double *K, int Nx, int Ny, int Nz, double A, double h, int BCtype){
	int ix = threadIdx.x + blockIdx.x*blockDim.x;
	int iz = threadIdx.y + blockIdx.y*blockDim.y;
	if (ix >= Nx-2 || iz >= Nz-2) return;
	int stride = Nx*Ny;
	int in_idx;
	int iy = Ny-1;
	in_idx = (ix + 1) + iy*Nx + (iz + 1)*stride;

	double HC = H_input[in_idx];
	double KC = K[in_idx];
	// *************
	//default BCtype Homogeneous - Neumann or dirichlet
	double H_north = HC; //factor (HC-H_north) is zero 0
	double K_north = KC; //(1.0/KC+1.0/K_north) <- \neq0
	double result=0.0;
	if(BCtype>2) return;
	if(BCtype==periodic) {
		H_north = H_input[in_idx-(Ny-1)*Nx];
		K_north = K[in_idx-(Ny-1)*Nx];
	}
	// *************
	result  = (HC-H_input[in_idx+1])      /  (1.0/KC  +  1.0/K[in_idx+1]);
	result += (HC-H_input[in_idx-Nx])     /  (1.0/KC  +  1.0/K[in_idx-Nx]);
	result += (HC-H_input[in_idx-1])      /  (1.0/KC  +  1.0/K[in_idx-1]);
	result += (HC-H_input[in_idx-stride]) /  (1.0/KC  +  1.0/K[in_idx-stride]);
	result += (HC-H_input[in_idx+stride]) /  (1.0/KC  +  1.0/K[in_idx+stride]);
	result += (HC-H_north)      /  (1.0/KC  +  1.0/K_north     );
	// *************
	if(BCtype==dirichlet) result += HC*KC; //dirichlet contribution
	// *************
	H_output[in_idx] = -2.0*result/h*A/h/h/h;
}

// yz-face (BCtype: Homogeneous-Neumann (0), Periodic (1), Homogeneous-Dirichlet (2))
__global__ void stencil_head_side_west(double *H_output, const double *H_input, const double *K, int Nx, int Ny, int Nz, double A, double h, int BCtype){
	int iy = threadIdx.x + blockIdx.x*blockDim.x;
	int iz  = threadIdx.y + blockIdx.y*blockDim.y;
	if (iy >= Ny-2 || iz >= Nz-2) return;
	int stride = Nx*Ny;
	int in_idx;
	int ix = 0;
	in_idx = ix + (iy + 1)*Nx + (iz + 1)*stride;

	double HC = H_input[in_idx];
	double KC = K[in_idx];
	// *************
	//default BCtype Homogeneous - Neumann or dirichlet
	double H_west=HC; //factor (HC-H_west) is zero 0
	double K_west=KC; //(1.0/KC+1.0/K_west) <- \neq0
	double result=0.0;
	if(BCtype>2) return;
	if(BCtype==periodic) {
		H_west = H_input[in_idx+(Nx-1)];
		K_west = K[in_idx+(Nx-1)];
	}
	// *************
	result  = (HC-H_input[in_idx+1])      /  (1.0/KC  +  1.0/K[in_idx+1]);
	result += (HC-H_input[in_idx-Nx])     /  (1.0/KC  +  1.0/K[in_idx-Nx]);
	result += (HC-H_input[in_idx+Nx])      /  (1.0/KC  +  1.0/K[in_idx+Nx]);
	result += (HC-H_input[in_idx-stride]) /  (1.0/KC  +  1.0/K[in_idx-stride]);
	result += (HC-H_input[in_idx+stride]) /  (1.0/KC  +  1.0/K[in_idx+stride]);
	result += (HC-H_west)      /  (1.0/KC  +  1.0/K_west     );
	// *************
	if(BCtype==dirichlet) result += HC*KC; //dirichlet contribution
	// *************
	H_output[in_idx] = -2.0*result/h*A/h/h/h;
}

// yz-face (BCtype: Homogeneous-Neumann (0), Periodic (1), Homogeneous-Dirichlet (2))
__global__ void stencil_head_side_east(double *H_output, const double *H_input, const double *K, int Nx, int Ny, int Nz, double A, double h, int BCtype){
	int iy = threadIdx.x + blockIdx.x*blockDim.x;
	int iz  = threadIdx.y + blockIdx.y*blockDim.y;
	if (iy >= Ny-2 || iz >= Nz-2) return;
	int stride = Nx*Ny;
	int in_idx;
	int ix = Nx-1;
	in_idx = ix + (iy + 1)*Nx + (iz + 1)*stride;

	double HC = H_input[in_idx];
	double KC = K[in_idx];
	// *************
	//default BCtype Homogeneous - Neumann or dirichlet
	double H_east=HC; //factor (HC-H_west) is zero 0
	double K_east=KC; //(1.0/KC+1.0/K_west) <- \neq0
	double result=0.0;
	if(BCtype>2) return;
	if(BCtype==periodic) {
		H_east = H_input[in_idx-(Nx-1)];
		K_east = K[in_idx-(Nx-1)];
	}
	// *************
	result  = (HC-H_input[in_idx-1])      /  (1.0/KC  +  1.0/K[in_idx-1]);
	result += (HC-H_input[in_idx-Nx])     /  (1.0/KC  +  1.0/K[in_idx-Nx]);
	result += (HC-H_input[in_idx+Nx])     /  (1.0/KC  +  1.0/K[in_idx+Nx]);
	result += (HC-H_input[in_idx-stride]) /  (1.0/KC  +  1.0/K[in_idx-stride]);
	result += (HC-H_input[in_idx+stride]) /  (1.0/KC  +  1.0/K[in_idx+stride]);
	result += (HC-H_east)      /  (1.0/KC  +  1.0/K_east     );
	// *************
	if(BCtype==dirichlet) result += HC*KC; //dirichlet contribution
	// *************
	H_output[in_idx] = -2.0*result/h*A/h/h/h;
}

// total 4-X-vertex
//(BCtype: Homogeneous-Neumann (0), Periodic (1), Homogeneous-Dirichlet (2))
__global__ void stencil_head_edge_X_South_Bottom(double *H_output, const double *H_input, const double *K, int Nx, int Ny, int Nz, double A, double h, int BCsouth, int BCbottom){
	int ix = threadIdx.x + blockIdx.x*blockDim.x;
	if (ix >= Nx-2) return;
	int stride = Nx*Ny;
	int iy = 0;
	int iz = 0;
	int in_idx;
	in_idx = (ix+1) + iy*Nx + iz*stride;

	double HC = H_input[in_idx];
	double KC = K[in_idx];
	// *************
	//default BCtype Homogeneous - Neumann or dirichlet
	double H_south=HC, H_bottom=HC; //factor (HC-H_boundary) is zero 0
	double K_south=KC, K_bottom=KC; //factor (1.0/KC+1.0/K_boundary) <- \neq0
	double result=0.0;
	if(BCsouth>2 || BCbottom>2) return;
	if(BCsouth==periodic) {
		H_south = H_input[in_idx+(Ny-1)*Nx];
		K_south = K[in_idx+(Ny-1)*Nx];
	}
	if(BCbottom==periodic) {
		H_bottom = H_input[in_idx+(Nz-1)*stride];
		K_bottom = K[in_idx+(Nz-1)*stride];
	}
	// *************
	result  = (HC-H_input[in_idx-1])      /  (1.0/KC  +  1.0/K[in_idx-1]);
	result += (HC-H_input[in_idx+1])      /  (1.0/KC  +  1.0/K[in_idx+1]);
	result += (HC-H_input[in_idx+Nx])     /  (1.0/KC  +  1.0/K[in_idx+Nx]);
	result += (HC-H_input[in_idx+stride]) /  (1.0/KC  +  1.0/K[in_idx+stride]);
	result += (HC-H_south)      /  (1.0/KC  +  1.0/K_south     );
	result += (HC-H_bottom)     /  (1.0/KC  +  1.0/K_bottom     );
	// *************
	if(BCsouth==dirichlet) result += HC*KC; //dirichlet contribution
	if(BCbottom==dirichlet) result += HC*KC; //dirichlet contribution
	// *************
	H_output[in_idx] = -2.0*result/h*A/h/h/h;
}

__global__ void stencil_head_edge_X_South_Top(double *H_output, const double *H_input, const double *K, int Nx, int Ny, int Nz, double A, double h, int BCsouth, int BCtop){
	int ix = threadIdx.x + blockIdx.x*blockDim.x;
	if (ix >= Nx-2) return;
	int stride = Nx*Ny;
	int iy = 0;
	int iz = Nz-1;
	int in_idx;
	in_idx = (ix+1) + iy*Nx + iz*stride;

	double HC = H_input[in_idx];
	double KC = K[in_idx];
	// *************
	//default BCtype Homogeneous - Neumann or dirichlet
	double H_south=HC, H_top=HC; //factor (HC-H_boundary) is zero 0
	double K_south=KC, K_top=KC; //factor (1.0/KC+1.0/K_boundary) <- \neq0
	double result=0.0;
	if(BCsouth>2 || BCtop>2) return;
	if(BCsouth==periodic) {
		H_south = H_input[in_idx+(Ny-1)*Nx];
		K_south = K[in_idx+(Ny-1)*Nx];
	}
	if(BCtop==periodic) {
		H_top = H_input[in_idx-(Nz-1)*stride];
		K_top = K[in_idx-(Nz-1)*stride];
	}
	// *************
	result  = (HC-H_input[in_idx-1])      /  (1.0/KC  +  1.0/K[in_idx-1]);
	result += (HC-H_input[in_idx+1])      /  (1.0/KC  +  1.0/K[in_idx+1]);
	result += (HC-H_input[in_idx+Nx])     /  (1.0/KC  +  1.0/K[in_idx+Nx]);
	result += (HC-H_input[in_idx-stride]) /  (1.0/KC  +  1.0/K[in_idx-stride]);
	result += (HC-H_south)      /  (1.0/KC  +  1.0/K_south     );
	result += (HC-H_top)     /  (1.0/KC  +  1.0/K_top     );
	// *************
	if(BCsouth==dirichlet) result += HC*KC; //dirichlet contribution
	if(BCtop==dirichlet) result += HC*KC; //dirichlet contribution
	// *************
	H_output[in_idx] = -2.0*result/h*A/h/h/h;
}

__global__ void stencil_head_edge_X_North_Bottom(double *H_output, const double *H_input, const double *K, int Nx, int Ny, int Nz, double A, double h, int BCnorth, int BCbottom){
	int ix = threadIdx.x + blockIdx.x*blockDim.x;
	if (ix >= Nx-2) return;
	int stride = Nx*Ny;
	int iy = Ny-1;
	int iz = 0;
	int in_idx;
	in_idx = (ix+1) + iy*Nx + iz*stride;

	double HC = H_input[in_idx];
	double KC = K[in_idx];
	// *************
	//default BCtype Homogeneous - Neumann or dirichlet
	double H_north=HC,H_bottom=HC; //factor (HC-H_boundary) is zero 0
	double K_north=KC,K_bottom=KC; //factor (1.0/KC+1.0/K_boundary) <- \neq0
	double result=0.0;
	if(BCnorth>2 || BCbottom>2) return;
	if(BCnorth==periodic) {
		H_north = H_input[in_idx-(Ny-1)*Nx];
		K_north = K[in_idx-(Ny-1)*Nx];
	}
	if(BCbottom==periodic) {
		H_bottom = H_input[in_idx+(Nz-1)*stride];
		K_bottom = K[in_idx+(Nz-1)*stride];
	}
	// *************
	result  = (HC-H_input[in_idx-1])      /  (1.0/KC  +  1.0/K[in_idx-1]);
	result += (HC-H_input[in_idx+1])      /  (1.0/KC  +  1.0/K[in_idx+1]);
	result += (HC-H_input[in_idx-Nx])     /  (1.0/KC  +  1.0/K[in_idx-Nx]);
	result += (HC-H_input[in_idx+stride]) /  (1.0/KC  +  1.0/K[in_idx+stride]);
	result += (HC-H_north)      /  (1.0/KC  +  1.0/K_north     );
	result += (HC-H_bottom)     /  (1.0/KC  +  1.0/K_bottom     );
	// *************
	if(BCnorth==dirichlet) result += HC*KC; //dirichlet contribution
	if(BCbottom==dirichlet) result += HC*KC; //dirichlet contribution
	// *************
	H_output[in_idx] = -2.0*result/h*A/h/h/h;
}

__global__ void stencil_head_edge_X_North_Top(double *H_output, const double *H_input, const double *K, int Nx, int Ny, int Nz, double A, double h, int BCnorth, int BCtop){
	int ix = threadIdx.x + blockIdx.x*blockDim.x;
	if (ix >= Nx-2) return;
	int stride = Nx*Ny;
	int iy = Ny-1;
	int iz = Nz-1;
	int in_idx;
	in_idx = (ix+1) + iy*Nx + iz*stride;

	double HC = H_input[in_idx];
	double KC = K[in_idx];
	// *************
	//default BCtype Homogeneous - Neumann or dirichlet
	double H_north=HC,H_top=HC; //factor (HC-H_boundary) is zero 0
	double K_north=KC,K_top=KC; //factor (1.0/KC+1.0/K_boundary) <- \neq0
	double result=0.0;
	if(BCnorth>2 || BCtop>2) return;
	if(BCnorth==periodic) {
		H_north = H_input[in_idx-(Ny-1)*Nx];
		K_north = K[in_idx-(Ny-1)*Nx];
	}
	if(BCtop==periodic) {
		H_top = H_input[in_idx-(Nz-1)*stride];
		K_top = K[in_idx-(Nz-1)*stride];
	}
	// *************
	result  = (HC-H_input[in_idx-1])      /  (1.0/KC  +  1.0/K[in_idx-1]);
	result += (HC-H_input[in_idx+1])      /  (1.0/KC  +  1.0/K[in_idx+1]);
	result += (HC-H_input[in_idx-Nx])     /  (1.0/KC  +  1.0/K[in_idx-Nx]);
	result += (HC-H_input[in_idx-stride]) /  (1.0/KC  +  1.0/K[in_idx-stride]);
	result += (HC-H_north)      /  (1.0/KC  +  1.0/K_north     );
	result += (HC-H_top)     /  (1.0/KC  +  1.0/K_top     );
	// *************
	if(BCnorth==dirichlet) result += HC*KC; //dirichlet contribution
	if(BCtop==dirichlet) result += HC*KC; //dirichlet contribution
	// *************
	H_output[in_idx] = -2.0*result/h*A/h/h/h;
}

// total 4-Z-vertex
//(BCtype: Homogeneous-Neumann (0), Periodic (1), Homogeneous-Dirichlet (2))
__global__ void stencil_head_edge_Z_South_West(double *H_output, const double *H_input, const double *K, int Nx, int Ny, int Nz, double A, double h, int BCsouth, int BCwest){
	int iz = threadIdx.x + blockIdx.x*blockDim.x;
	if (iz >= Nz-2) return;
	int stride = Nx*Ny;
	int ix = 0;
	int iy = 0;
	int in_idx;
	in_idx = ix + iy*Nx + (iz+1)*stride;

	double HC = H_input[in_idx];
	double KC = K[in_idx];
	// *************
	//default BCtype Homogeneous - Neumann or dirichlet
	double H_south=HC,H_west=HC; //factor (HC-H_boundary) is zero 0
	double K_south=KC,K_west=KC; //factor (1.0/KC+1.0/K_boundary) <- \neq0
	double result=0.0;
	if(BCsouth>2 || BCwest>2) return;
	if(BCsouth==periodic) {
		H_south = H_input[in_idx+(Ny-1)*Nx];
		K_south = K[in_idx+(Ny-1)*Nx];
	}
	if(BCwest==periodic) {
		H_west = H_input[in_idx+(Nx-1)];
		K_west = K[in_idx+(Nx-1)];
	}
	// *************
	result  = (HC-H_input[in_idx+1])      /  (1.0/KC  +  1.0/K[in_idx+1]);
	result += (HC-H_input[in_idx+Nx])     /  (1.0/KC  +  1.0/K[in_idx+Nx]);
	result += (HC-H_input[in_idx+stride]) /  (1.0/KC  +  1.0/K[in_idx+stride]);
	result += (HC-H_input[in_idx-stride]) /  (1.0/KC  +  1.0/K[in_idx-stride]);
	result += (HC-H_south)      /  (1.0/KC  +  1.0/K_south     );
	result += (HC-H_west)     /  (1.0/KC  +  1.0/K_west     );
	// *************
	if(BCsouth==dirichlet) result += HC*KC; //dirichlet contribution
	if(BCwest==dirichlet) result += HC*KC; //dirichlet contribution
	// *************
	H_output[in_idx] = -2.0*result/h*A/h/h/h;
}

__global__ void stencil_head_edge_Z_South_East(double *H_output, const double *H_input, const double *K, int Nx, int Ny, int Nz, double A, double h, int BCsouth, int BCeast){
	int iz = threadIdx.x + blockIdx.x*blockDim.x;
	if (iz >= Nz-2) return;
	int stride = Nx*Ny;
	int ix = Nx-1;
	int iy = 0;
	int in_idx;
	in_idx = ix + iy*Nx + (iz+1)*stride;

	double HC = H_input[in_idx];
	double KC = K[in_idx];
	// *************
	//default BCtype Homogeneous - Neumann or dirichlet
	double H_south=HC,H_east=HC; //factor (HC-H_boundary) is zero 0
	double K_south=KC,K_east=KC; //factor (1.0/KC+1.0/K_boundary) <- \neq0
	double result=0.0;
	if(BCsouth>2 || BCeast>2) return;
	if(BCsouth==periodic) {
		H_south = H_input[in_idx+(Ny-1)*Nx];
		K_south = K[in_idx+(Ny-1)*Nx];
	}
	if(BCeast==periodic) {
		H_east = H_input[in_idx-(Nx-1)];
		K_east = K[in_idx-(Nx-1)];
	}
	// *************
	result  = (HC-H_input[in_idx-1])      /  (1.0/KC  +  1.0/K[in_idx-1]);
	result += (HC-H_input[in_idx+Nx])     /  (1.0/KC  +  1.0/K[in_idx+Nx]);
	result += (HC-H_input[in_idx+stride]) /  (1.0/KC  +  1.0/K[in_idx+stride]);
	result += (HC-H_input[in_idx-stride]) /  (1.0/KC  +  1.0/K[in_idx-stride]);
	result += (HC-H_south)    /  (1.0/KC  +  1.0/K_south     );
	result += (HC-H_east)     /  (1.0/KC  +  1.0/K_east     );
	// *************
	if(BCsouth==dirichlet) result += HC*KC; //dirichlet contribution
	if(BCeast==dirichlet) result += HC*KC; //dirichlet contribution
	// *************
	H_output[in_idx] = -2.0*result/h*A/h/h/h;
}

__global__ void stencil_head_edge_Z_North_West(double *H_output, const double *H_input, const double *K, int Nx, int Ny, int Nz, double A, double h, int BCnorth, int BCwest){
	int iz = threadIdx.x + blockIdx.x*blockDim.x;
	if (iz >= Nz-2) return;
	int stride = Nx*Ny;
	int ix = 0;
	int iy = Ny-1;
	int in_idx;
	in_idx = ix + iy*Nx + (iz+1)*stride;

	double HC = H_input[in_idx];
	double KC = K[in_idx];
	// *************
	//default BCtype Homogeneous - Neumann or dirichlet
	double H_north=HC,H_west=HC; //factor (HC-H_boundary) is zero 0
	double K_north=KC,K_west=KC; //factor (1.0/KC+1.0/K_boundary) <- \neq0
	double result=0.0;
	if(BCnorth>2 || BCwest>2) return;
	if(BCnorth==periodic) {
		H_north = H_input[in_idx-(Ny-1)*Nx];
		K_north = K[in_idx-(Ny-1)*Nx];
	}
	if(BCwest==periodic) {
		H_west = H_input[in_idx+(Nx-1)];
		K_west = K[in_idx+(Nx-1)];
	}
	// *************
	result  = (HC-H_input[in_idx+1])      /  (1.0/KC  +  1.0/K[in_idx+1]);
	result += (HC-H_input[in_idx-Nx])     /  (1.0/KC  +  1.0/K[in_idx-Nx]);
	result += (HC-H_input[in_idx+stride]) /  (1.0/KC  +  1.0/K[in_idx+stride]);
	result += (HC-H_input[in_idx-stride]) /  (1.0/KC  +  1.0/K[in_idx-stride]);
	result += (HC-H_north)    /  (1.0/KC  +  1.0/K_north     );
	result += (HC-H_west)     /  (1.0/KC  +  1.0/K_west     );
	// *************
	if(BCnorth==dirichlet) result += HC*KC; //dirichlet contribution
	if(BCwest==dirichlet) result += HC*KC; //dirichlet contribution
	// *************
	H_output[in_idx] = -2.0*result/h*A/h/h/h;
}

__global__ void stencil_head_edge_Z_North_East(double *H_output, const double *H_input, const double *K, int Nx, int Ny, int Nz, double A, double h, int BCnorth, int BCeast){
	int iz = threadIdx.x + blockIdx.x*blockDim.x;
	if (iz >= Nz-2) return;
	int stride = Nx*Ny;
	int ix = Nx-1;
	int iy = Ny-1;
	int in_idx;
	in_idx = ix + iy*Nx + (iz+1)*stride;

	double HC = H_input[in_idx];
	double KC = K[in_idx];
	// *************
	//default BCtype Homogeneous - Neumann or dirichlet
	double H_north=HC,H_east=HC; //factor (HC-H_boundary) is zero 0
	double K_north=KC,K_east=KC; //factor (1.0/KC+1.0/K_boundary) <- \neq0
	double result=0.0;
	if(BCnorth>2 || BCeast>2) return;
	if(BCnorth==periodic) {
		H_north = H_input[in_idx-(Ny-1)*Nx];
		K_north = K[in_idx-(Ny-1)*Nx];
	}
	if(BCeast==periodic) {
		H_east = H_input[in_idx-(Nx-1)];
		K_east = K[in_idx-(Nx-1)];
	}
	// *************
	result  = (HC-H_input[in_idx-1])      /  (1.0/KC  +  1.0/K[in_idx-1]);
	result += (HC-H_input[in_idx-Nx])     /  (1.0/KC  +  1.0/K[in_idx-Nx]);
	result += (HC-H_input[in_idx+stride]) /  (1.0/KC  +  1.0/K[in_idx+stride]);
	result += (HC-H_input[in_idx-stride]) /  (1.0/KC  +  1.0/K[in_idx-stride]);
	result += (HC-H_north)    /  (1.0/KC  +  1.0/K_north     );
	result += (HC-H_east)     /  (1.0/KC  +  1.0/K_east     );
	// *************
	if(BCnorth==dirichlet) result += HC*KC; //dirichlet contribution
	if(BCeast==dirichlet) result += HC*KC; //dirichlet contribution
	// *************
	H_output[in_idx] = -2.0*result/h*A/h/h/h;
}


// total 4-Y-vertex
//(BCtype: Homogeneous-Neumann (0), Periodic (1), Homogeneous-Dirichlet (2))
__global__ void stencil_head_edge_Y_Bottom_West(double *H_output, const double *H_input, const double *K, int Nx, int Ny, int Nz, double A, double h, int BCbottom, int BCwest){
	int iy = threadIdx.x + blockIdx.x*blockDim.x;
	if (iy >= Ny-2) return;
	int stride = Nx*Ny;
	int ix = 0;
	int iz = 0;
	int in_idx;
	in_idx = ix + (iy+1)*Nx + iz*stride;

	double HC = H_input[in_idx];
	double KC = K[in_idx];
	// *************
	//default BCtype Homogeneous - Neumann or dirichlet
	double H_bottom=HC,H_west=HC; //factor (HC-H_boundary) is zero 0
	double K_bottom=KC,K_west=KC; //factor (1.0/KC+1.0/K_boundary) <- \neq0
	double result=0.0;
	if(BCbottom>2 || BCwest>2) return;
	if(BCbottom==periodic) {
		H_bottom = H_input[in_idx+(Nz-1)*stride];
		K_bottom = K[in_idx+(Nz-1)*stride];
	}
	if(BCwest==periodic) {
		H_west = H_input[in_idx+(Nx-1)];
		K_west = K[in_idx+(Nx-1)];
	}
	// *************
	result  = (HC-H_input[in_idx+1])      /  (1.0/KC  +  1.0/K[in_idx+1]);
	result += (HC-H_input[in_idx+Nx])     /  (1.0/KC  +  1.0/K[in_idx+Nx]);
	result += (HC-H_input[in_idx-Nx])     /  (1.0/KC  +  1.0/K[in_idx-Nx]);
	result += (HC-H_input[in_idx+stride]) /  (1.0/KC  +  1.0/K[in_idx+stride]);
	result += (HC-H_bottom)   /  (1.0/KC  +  1.0/K_bottom     );
	result += (HC-H_west)     /  (1.0/KC  +  1.0/K_west     );
	// *************
	if(BCbottom==dirichlet) result += HC*KC; //dirichlet contribution
	if(BCwest==dirichlet) result += HC*KC; //dirichlet contribution
	// *************
	H_output[in_idx] = -2.0*result/h*A/h/h/h;
}

__global__ void stencil_head_edge_Y_Top_West(double *H_output, const double *H_input, const double *K, int Nx, int Ny, int Nz, double A, double h, int BCtop, int BCwest){
	int iy = threadIdx.x + blockIdx.x*blockDim.x;
	if (iy >= Ny-2) return;
	int stride = Nx*Ny;
	int ix = 0;
	int iz = Nz-1;
	int in_idx;
	in_idx = ix + (iy+1)*Nx + iz*stride;

	double HC = H_input[in_idx];
	double KC = K[in_idx];
	// *************
	//default BCtype Homogeneous - Neumann or dirichlet
	double H_top=HC,H_west=HC; //factor (HC-H_boundary) is zero 0
	double K_top=KC,K_west=KC; //factor (1.0/KC+1.0/K_boundary) <- \neq0
	double result=0.0;
	if(BCtop>2 || BCwest>2) return;
	if(BCtop==periodic) {
		H_top = H_input[in_idx-(Nz-1)*stride];
		K_top = K[in_idx-(Nz-1)*stride];
	}
	if(BCwest==periodic) {
		H_west = H_input[in_idx+(Nx-1)];
		K_west = K[in_idx+(Nx-1)];
	}
	// *************
	result  = (HC-H_input[in_idx+1])      /  (1.0/KC  +  1.0/K[in_idx+1]);
	result += (HC-H_input[in_idx+Nx])     /  (1.0/KC  +  1.0/K[in_idx+Nx]);
	result += (HC-H_input[in_idx-Nx])     /  (1.0/KC  +  1.0/K[in_idx-Nx]);
	result += (HC-H_input[in_idx-stride]) /  (1.0/KC  +  1.0/K[in_idx-stride]);
	result += (HC-H_top)   /  (1.0/KC  +  1.0/K_top     );
	result += (HC-H_west)  /  (1.0/KC  +  1.0/K_west     );
	// *************
	if(BCtop==dirichlet) result += HC*KC; //dirichlet contribution
	if(BCwest==dirichlet) result += HC*KC; //dirichlet contribution
	// *************
	H_output[in_idx] = -2.0*result/h*A/h/h/h;
}

__global__ void stencil_head_edge_Y_Bottom_East(double *H_output, const double *H_input, const double *K, int Nx, int Ny, int Nz, double A, double h, int BCbottom, int BCeast){
	int iy = threadIdx.x + blockIdx.x*blockDim.x;
	if (iy >= Ny-2) return;
	int stride = Nx*Ny;
	int ix = Nx-1;
	int iz = 0;
	int in_idx;
	in_idx = ix + (iy+1)*Nx + iz*stride;

	double HC = H_input[in_idx];
	double KC = K[in_idx];
	// *************
	//default BCtype Homogeneous - Neumann or dirichlet
	double H_bottom=HC,H_east=HC; //factor (HC-H_boundary) is zero 0
	double K_bottom=KC,K_east=KC; //factor (1.0/KC+1.0/K_boundary) <- \neq0
	double result=0.0;
	if(BCbottom>2 || BCeast>2) return;
	if(BCbottom==periodic) {
		H_bottom = H_input[in_idx+(Nz-1)*stride];
		K_bottom = K[in_idx+(Nz-1)*stride];
	}
	if(BCeast==periodic) {
		H_east = H_input[in_idx-(Nx-1)];
		K_east = K[in_idx-(Nx-1)];
	}
	// *************
	result  = (HC-H_input[in_idx-1])      /  (1.0/KC  +  1.0/K[in_idx-1]);
	result += (HC-H_input[in_idx+Nx])     /  (1.0/KC  +  1.0/K[in_idx+Nx]);
	result += (HC-H_input[in_idx-Nx])     /  (1.0/KC  +  1.0/K[in_idx-Nx]);
	result += (HC-H_input[in_idx+stride]) /  (1.0/KC  +  1.0/K[in_idx+stride]);
	result += (HC-H_bottom)   /  (1.0/KC  +  1.0/K_bottom     );
	result += (HC-H_east)     /  (1.0/KC  +  1.0/K_east     );
	// *************
	if(BCbottom==dirichlet) result += HC*KC; //dirichlet contribution
	if(BCeast==dirichlet) result += HC*KC; //dirichlet contribution
	// *************
	H_output[in_idx] = -2.0*result/h*A/h/h/h;
}

__global__ void stencil_head_edge_Y_Top_East(double *H_output, const double *H_input, const double *K, int Nx, int Ny, int Nz, double A, double h, int BCtop, int BCeast){
	int iy = threadIdx.x + blockIdx.x*blockDim.x;
	if (iy >= Ny-2) return;
	int stride = Nx*Ny;
	int ix = Nx-1;
	int iz = Nz-1;
	int in_idx;
	in_idx = ix + (iy+1)*Nx + iz*stride;

	double HC = H_input[in_idx];
	double KC = K[in_idx];
	// *************
	//default BCtype Homogeneous - Neumann or dirichlet
	double H_top=HC,H_east=HC; //factor (HC-H_boundary) is zero 0
	double K_top=KC,K_east=KC; //factor (1.0/KC+1.0/K_boundary) <- \neq0
	double result=0.0;
	if(BCtop>2 || BCeast>2) return;
	if(BCtop==periodic) {
		H_top = H_input[in_idx-(Nz-1)*stride];
		K_top = K[in_idx-(Nz-1)*stride];
	}
	if(BCeast==periodic) {
		H_east = H_input[in_idx-(Nx-1)];
		K_east = K[in_idx-(Nx-1)];
	}
	// *************
	result  = (HC-H_input[in_idx-1])      /  (1.0/KC  +  1.0/K[in_idx-1]);
	result += (HC-H_input[in_idx+Nx])     /  (1.0/KC  +  1.0/K[in_idx+Nx]);
	result += (HC-H_input[in_idx-Nx])     /  (1.0/KC  +  1.0/K[in_idx-Nx]);
	result += (HC-H_input[in_idx-stride]) /  (1.0/KC  +  1.0/K[in_idx-stride]);
	result += (HC-H_top)   /  (1.0/KC  +  1.0/K_top     );
	result += (HC-H_east)  /  (1.0/KC  +  1.0/K_east     );
	// *************
	if(BCtop==dirichlet) result += HC*KC; //dirichlet contribution
	if(BCeast==dirichlet) result += HC*KC; //dirichlet contribution
	// *************
	H_output[in_idx] = -2.0*result/h*A/h/h/h;
}

// total 8-vertex
//(BCtype: Homogeneous-Neumann (0), Periodic (1), Homogeneous-Dirichlet (2))
__global__ void stencil_head_vertex_SWB(double *H_output, const double *H_input, const double *K, int Nx, int Ny, int Nz, double A, double h, int BCsouth, int BCwest, int BCbottom, bool pin1stCell){
	int stride = Nx*Ny;
	int ix = 0;
	int iy = 0;
	int iz = 0;
	int in_idx;
	in_idx = ix + iy*Nx + iz*stride;

	double HC = H_input[in_idx];
	double KC = K[in_idx];
	double KN;
	// *************
	//default BCtype Homogeneous - Neumann or dirichlet
	double H_south=HC,H_west=HC,H_bottom=HC; //factor (HC-H_boundary) is zero 0
	double K_south=KC,K_west=KC,K_bottom=KC; //factor (1.0/KC+1.0/K_boundary) <- \neq0
	double result=0.0;
	double aC = 0.0;
	if(BCsouth>2 || BCwest>2 || BCbottom>2) return;
	if(BCsouth==periodic) {
		H_south = H_input[in_idx+(Ny-1)*Nx];
		K_south = K[in_idx+(Ny-1)*Nx];
	}
	if(BCwest==periodic) {
		H_west = H_input[in_idx+(Nx-1)];
		K_west = K[in_idx+(Nx-1)];
	}
	if(BCbottom==periodic) {
		H_bottom = H_input[in_idx+(Nz-1)*stride];
		K_bottom = K[in_idx+(Nz-1)*stride];
	}
	// *************
	KN = 1.0/(1.0/KC  +  1.0/K[in_idx+1]);
	result  = (HC-H_input[in_idx+1])*KN;
	aC += KN;
	KN = 1.0/(1.0/KC  +  1.0/K[in_idx+Nx]);
	result += (HC-H_input[in_idx+Nx])*KN;
	aC += KN;
	KN = 1.0/(1.0/KC  +  1.0/K[in_idx+stride]);
	result += (HC-H_input[in_idx+stride])*KN;
	aC += KN;

	result += (HC-H_south)   /  (1.0/KC  +  1.0/K_south     );
	result += (HC-H_west)    /  (1.0/KC  +  1.0/K_west     );
	result += (HC-H_bottom)  /  (1.0/KC  +  1.0/K_bottom     );
	// *************
	if(BCsouth==dirichlet) result += HC*KC; //dirichlet contribution
	if(BCwest==dirichlet) result += HC*KC; //dirichlet contribution
	if(BCbottom==dirichlet) result += HC*KC; //dirichlet contribution
	// *************
	if(pin1stCell) H_output[in_idx] = -2.0*(result+aC*HC)/h*A/h/h/h;
	else H_output[in_idx] = -2.0*result/h*A/h/h/h;
}

__global__ void stencil_head_vertex_SWT(double *H_output, const double *H_input, const double *K, int Nx, int Ny, int Nz, double A, double h, int BCsouth, int BCwest, int BCtop){
	int stride = Nx*Ny;
	int ix = 0;
	int iy = 0;
	int iz = Nz-1;
	int in_idx;
	in_idx = ix + iy*Nx + iz*stride;

	double HC = H_input[in_idx];
	double KC = K[in_idx];
	// *************
	//default BCtype Homogeneous - Neumann or dirichlet
	double H_south=HC,H_west=HC,H_top=HC; //factor (HC-H_boundary) is zero 0
	double K_south=KC,K_west=KC,K_top=KC; //factor (1.0/KC+1.0/K_boundary) <- \neq0
	double result=0.0;
	if(BCsouth>2 || BCwest>2 || BCtop>2) return;

	if(BCsouth==periodic) {
		H_south = H_input[in_idx+(Ny-1)*Nx];
		K_south = K[in_idx+(Ny-1)*Nx];
	}
	if(BCwest==periodic) {
		H_west = H_input[in_idx+(Nx-1)];
		K_west = K[in_idx+(Nx-1)];
	}
	if(BCtop==periodic) {
		H_top = H_input[in_idx-(Nz-1)*stride];
		K_top = K[in_idx-(Nz-1)*stride];
	}
	// *************
	result  = (HC-H_input[in_idx+1])      /  (1.0/KC  +  1.0/K[in_idx+1]);
	result += (HC-H_input[in_idx+Nx])     /  (1.0/KC  +  1.0/K[in_idx+Nx]);
	result += (HC-H_input[in_idx-stride]) /  (1.0/KC  +  1.0/K[in_idx-stride]);
	result += (HC-H_south)   /  (1.0/KC  +  1.0/K_south     );
	result += (HC-H_west)    /  (1.0/KC  +  1.0/K_west     );
	result += (HC-H_top)     /  (1.0/KC  +  1.0/K_top     );
	// *************
	if(BCsouth==dirichlet) result += HC*KC; //dirichlet contribution
	if(BCwest==dirichlet) result += HC*KC; //dirichlet contribution
	if(BCtop==dirichlet) result += HC*KC; //dirichlet contribution
	// *************
	H_output[in_idx] = -2.0*result/h*A/h/h/h;
}

__global__ void stencil_head_vertex_SEB(double *H_output, const double *H_input, const double *K, int Nx, int Ny, int Nz, double A, double h, int BCsouth, int BCeast, int BCbottom){
	int stride = Nx*Ny;
	int ix = Nx-1;
	int iy = 0;
	int iz = 0;
	int in_idx;
	in_idx = ix + iy*Nx + iz*stride;

	double HC = H_input[in_idx];
	double KC = K[in_idx];
	// *************
	//default BCtype Homogeneous - Neumann or dirichlet
	double H_south=HC,H_east=HC,H_bottom=HC; //factor (HC-H_boundary) is zero 0
	double K_south=KC,K_east=KC,K_bottom=KC; //factor (1.0/KC+1.0/K_boundary) <- \neq0
	double result=0.0;

	if(BCsouth>2 || BCeast>2 || BCbottom>2) return;
	if(BCsouth==periodic) {
		H_south = H_input[in_idx+(Ny-1)*Nx];
		K_south = K[in_idx+(Ny-1)*Nx];
	}
	if(BCeast==periodic) {
		H_east = H_input[in_idx-(Nx-1)];
		K_east = K[in_idx-(Nx-1)];
	}
	if(BCbottom==periodic) {
		H_bottom = H_input[in_idx+(Nz-1)*stride];
		K_bottom = K[in_idx+(Nz-1)*stride];
	}
	// *************
	result  = (HC-H_input[in_idx-1])      /  (1.0/KC  +  1.0/K[in_idx-1]);
	result += (HC-H_input[in_idx+Nx])     /  (1.0/KC  +  1.0/K[in_idx+Nx]);
	result += (HC-H_input[in_idx+stride]) /  (1.0/KC  +  1.0/K[in_idx+stride]);
	result += (HC-H_south)   /  (1.0/KC  +  1.0/K_south     );
	result += (HC-H_east)    /  (1.0/KC  +  1.0/K_east     );
	result += (HC-H_bottom)  /  (1.0/KC  +  1.0/K_bottom     );
	// *************
	if(BCsouth==dirichlet) result += HC*KC; //dirichlet contribution
	if(BCeast==dirichlet) result += HC*KC; //dirichlet contribution
	if(BCbottom==dirichlet) result += HC*KC; //dirichlet contribution
	// *************
	H_output[in_idx] = -2.0*result/h*A/h/h/h;
}

__global__ void stencil_head_vertex_SET(double *H_output, const double *H_input, const double *K, int Nx, int Ny, int Nz, double A, double h, int BCsouth, int BCeast, int BCtop){
	int stride = Nx*Ny;
	int ix = Nx-1;
	int iy = 0;
	int iz = Nz-1;
	int in_idx;
	in_idx = ix + iy*Nx + iz*stride;

	double HC = H_input[in_idx];
	double KC = K[in_idx];
	// *************
	//default BCtype Homogeneous - Neumann or dirichlet
	double H_south=HC,H_east=HC,H_top=HC; //factor (HC-H_boundary) is zero 0
	double K_south=KC,K_east=KC,K_top=KC; //factor (1.0/KC+1.0/K_boundary) <- \neq0
	double result=0.0;

	if(BCsouth>2 || BCeast>2 || BCtop>2) return;
	if(BCsouth==periodic) {
		H_south = H_input[in_idx+(Ny-1)*Nx];
		K_south = K[in_idx+(Ny-1)*Nx];
	}
	if(BCeast==periodic) {
		H_east = H_input[in_idx-(Nx-1)];
		K_east = K[in_idx-(Nx-1)];
	}
	if(BCtop==periodic) {
		H_top = H_input[in_idx-(Nz-1)*stride];
		K_top = K[in_idx-(Nz-1)*stride];
	}
	// *************
	result  = (HC-H_input[in_idx-1])      /  (1.0/KC  +  1.0/K[in_idx-1]);
	result += (HC-H_input[in_idx+Nx])     /  (1.0/KC  +  1.0/K[in_idx+Nx]);
	result += (HC-H_input[in_idx-stride]) /  (1.0/KC  +  1.0/K[in_idx-stride]);
	result += (HC-H_south)/  (1.0/KC  +  1.0/K_south     );
	result += (HC-H_east) /  (1.0/KC  +  1.0/K_east     );
	result += (HC-H_top)  /  (1.0/KC  +  1.0/K_top     );
	// *************
	if(BCsouth==dirichlet) result += HC*KC; //dirichlet contribution
	if(BCeast==dirichlet) result += HC*KC; //dirichlet contribution
	if(BCtop==dirichlet) result += HC*KC; //dirichlet contribution
	// *************
	H_output[in_idx] = -2.0*result/h*A/h/h/h;
}

__global__ void stencil_head_vertex_NWB(double *H_output, const double *H_input, const double *K, int Nx, int Ny, int Nz, double A, double h, int BCnorth, int BCwest, int BCbottom){
	int stride = Nx*Ny;
	int ix = 0;
	int iy = Ny-1;
	int iz = 0;
	int in_idx;
	in_idx = ix + iy*Nx + iz*stride;

	double HC = H_input[in_idx];
	double KC = K[in_idx];
	// *************
	//default BCtype Homogeneous - Neumann or dirichlet
	double H_north=HC,H_west=HC,H_bottom=HC; //factor (HC-H_boundary) is zero 0
	double K_north=KC,K_west=KC,K_bottom=KC; //factor (1.0/KC+1.0/K_boundary) <- \neq0
	double result=0.0;

	if(BCnorth>2 || BCwest>2 || BCbottom>2) return;
	if(BCnorth==periodic) {
		H_north = H_input[in_idx-(Ny-1)*Nx];
		K_north = K[in_idx-(Ny-1)*Nx];
	}
	if(BCwest==periodic) {
		H_west = H_input[in_idx+(Nx-1)];
		K_west = K[in_idx+(Nx-1)];
	}
	if(BCbottom==periodic) {
		H_bottom = H_input[in_idx+(Nz-1)*stride];
		K_bottom = K[in_idx+(Nz-1)*stride];
	}
	// *************
	result  = (HC-H_input[in_idx+1])      /  (1.0/KC  +  1.0/K[in_idx+1]);
	result += (HC-H_input[in_idx-Nx])     /  (1.0/KC  +  1.0/K[in_idx-Nx]);
	result += (HC-H_input[in_idx+stride]) /  (1.0/KC  +  1.0/K[in_idx+stride]);
	result += (HC-H_north)   /  (1.0/KC  +  1.0/K_north     );
	result += (HC-H_west)    /  (1.0/KC  +  1.0/K_west     );
	result += (HC-H_bottom)  /  (1.0/KC  +  1.0/K_bottom     );
	// *************
	if(BCnorth==dirichlet) result += HC*KC; //dirichlet contribution
	if(BCwest==dirichlet) result += HC*KC; //dirichlet contribution
	if(BCbottom==dirichlet) result += HC*KC; //dirichlet contribution
	// *************
	H_output[in_idx] = -2.0*result/h*A/h/h/h;
}

__global__ void stencil_head_vertex_NWT(double *H_output, const double *H_input, const double *K, int Nx, int Ny, int Nz, double A, double h, int BCnorth, int BCwest, int BCtop){
	int stride = Nx*Ny;
	int ix = 0;
	int iy = Ny-1;
	int iz = Nz-1;
	int in_idx;
	in_idx = ix + iy*Nx + iz*stride;

	double HC = H_input[in_idx];
	double KC = K[in_idx];
	// *************
	//default BCtype Homogeneous - Neumann or dirichlet
	double H_north=HC,H_west=HC,H_top=HC; //factor (HC-H_boundary) is zero 0
	double K_north=KC,K_west=KC,K_top=KC; //factor (1.0/KC+1.0/K_boundary) <- \neq0
	double result=0.0;

	if(BCnorth>2 || BCwest>2 || BCtop>2) return;
	if(BCnorth==periodic) {
		H_north = H_input[in_idx-(Ny-1)*Nx];
		K_north = K[in_idx-(Ny-1)*Nx];
	}
	if(BCwest==periodic) {
		H_west = H_input[in_idx+(Nx-1)];
		K_west = K[in_idx+(Nx-1)];
	}
	if(BCtop==periodic) {
		H_top = H_input[in_idx-(Nz-1)*stride];
		K_top = K[in_idx-(Nz-1)*stride];
	}
	// *************
	result  = (HC-H_input[in_idx+1])      /  (1.0/KC  +  1.0/K[in_idx+1]);
	result += (HC-H_input[in_idx-Nx])     /  (1.0/KC  +  1.0/K[in_idx-Nx]);
	result += (HC-H_input[in_idx-stride]) /  (1.0/KC  +  1.0/K[in_idx-stride]);
	result += (HC-H_north)   /  (1.0/KC  +  1.0/K_north   );
	result += (HC-H_west)    /  (1.0/KC  +  1.0/K_west    );
	result += (HC-H_top)     /  (1.0/KC  +  1.0/K_top     );
	// *************
	if(BCnorth==dirichlet) result += HC*KC; //dirichlet contribution
	if(BCwest==dirichlet) result += HC*KC; //dirichlet contribution
	if(BCtop==dirichlet) result += HC*KC; //dirichlet contribution
	// *************
	H_output[in_idx] = -2.0*result/h*A/h/h/h;
}

__global__ void stencil_head_vertex_NEB(double *H_output, const double *H_input, const double *K, int Nx, int Ny, int Nz, double A, double h, int BCnorth, int BCeast, int BCbottom){
	int stride = Nx*Ny;
	int ix = Nx-1;
	int iy = Ny-1;
	int iz = 0;
	int in_idx;
	in_idx = ix + iy*Nx + iz*stride;

	double HC = H_input[in_idx];
	double KC = K[in_idx];
	// *************
	//default BCtype Homogeneous - Neumann or dirichlet
	double H_north=HC,H_east=HC,H_bottom=HC; //factor (HC-H_boundary) is zero 0
	double K_north=KC,K_east=KC,K_bottom=KC; //factor (1.0/KC+1.0/K_boundary) <- \neq0
	double result=0.0;

	if(BCnorth>2 || BCeast>2 || BCbottom>2) return;
	if(BCnorth==periodic) {
		H_north = H_input[in_idx-(Ny-1)*Nx];
		K_north = K[in_idx-(Ny-1)*Nx];
	}
	if(BCeast==periodic) {
		H_east = H_input[in_idx-(Nx-1)];
		K_east = K[in_idx-(Nx-1)];
	}
	if(BCbottom==periodic) {
		H_bottom = H_input[in_idx+(Nz-1)*stride];
		K_bottom = K[in_idx+(Nz-1)*stride];
	}
	// *************
	result  = (HC-H_input[in_idx-1])      /  (1.0/KC  +  1.0/K[in_idx-1]);
	result += (HC-H_input[in_idx-Nx])     /  (1.0/KC  +  1.0/K[in_idx-Nx]);
	result += (HC-H_input[in_idx+stride]) /  (1.0/KC  +  1.0/K[in_idx+stride]);
	result += (HC-H_north)   /  (1.0/KC  +  1.0/K_north     );
	result += (HC-H_east)    /  (1.0/KC  +  1.0/K_east     );
	result += (HC-H_bottom)  /  (1.0/KC  +  1.0/K_bottom     );
	// *************
	if(BCnorth==dirichlet) result += HC*KC; //dirichlet contribution
	if(BCeast==dirichlet) result += HC*KC; //dirichlet contribution
	if(BCbottom==dirichlet) result += HC*KC; //dirichlet contribution
	// *************
	H_output[in_idx] = -2.0*result/h*A/h/h/h;
}

__global__ void stencil_head_vertex_NET(double *H_output, const double *H_input, const double *K, int Nx, int Ny, int Nz, double A, double h, int BCnorth, int BCeast, int BCtop){
	int stride = Nx*Ny;
	int ix = Nx-1;
	int iy = Ny-1;
	int iz = Nz-1;
	int in_idx;
	in_idx = ix + iy*Nx + iz*stride;

	double HC = H_input[in_idx];
	double KC = K[in_idx];
	// *************
	//default BCtype Homogeneous - Neumann or dirichlet
	double H_north=HC,H_east=HC,H_top=HC; //factor (HC-H_boundary) is zero 0
	double K_north=KC,K_east=KC,K_top=KC; //factor (1.0/KC+1.0/K_boundary) <- \neq0
	double result=0.0;

	if(BCnorth>2 || BCeast>2 || BCtop>2) return;
	if(BCnorth==periodic) {
		H_north = H_input[in_idx-(Ny-1)*Nx];
		K_north = K[in_idx-(Ny-1)*Nx];
	}
	if(BCeast==periodic) {
		H_east = H_input[in_idx-(Nx-1)];
		K_east = K[in_idx-(Nx-1)];
	}
	if(BCtop==periodic) {
		H_top = H_input[in_idx-(Nz-1)*stride];
		K_top = K[in_idx-(Nz-1)*stride];
	}
	// *************
	result  = (HC-H_input[in_idx-1])      /  (1.0/KC  +  1.0/K[in_idx-1]);
	result += (HC-H_input[in_idx-Nx])     /  (1.0/KC  +  1.0/K[in_idx-Nx]);
	result += (HC-H_input[in_idx-stride]) /  (1.0/KC  +  1.0/K[in_idx-stride]);
	result += (HC-H_north)/  (1.0/KC  +  1.0/K_north   );
	result += (HC-H_east) /  (1.0/KC  +  1.0/K_east    );
	result += (HC-H_top)  /  (1.0/KC  +  1.0/K_top     );
	// *************
	if(BCnorth==dirichlet) result += HC*KC; //dirichlet contribution
	if(BCeast==dirichlet) result += HC*KC; //dirichlet contribution
	if(BCtop==dirichlet) result += HC*KC; //dirichlet contribution
	// *************
	H_output[in_idx] = -2.0*result/h*A/h/h/h;
}


void stencil_head(double *H_output, const double *H_input, const double *K,
	int Nx, int Ny, int Nz, double A, double h,
	int BCbottom, int BCtop, int BCsouth, int BCnorth, int BCwest, int BCeast, bool pin1stCell,
	dim3 grid, dim3 block){

	stencil_head_int		<<<gridXY,blockXY>>>(H_output,H_input,K,Nx,Ny,Nz,A,h);

	stencil_head_side_bottom<<<gridXY,blockXY>>>(H_output,H_input,K,Nx,Ny,Nz,A,h,BCbottom);
	stencil_head_side_top	<<<gridXY,blockXY>>>(H_output,H_input,K,Nx,Ny,Nz,A,h,BCtop);

	stencil_head_side_south<<<gridXZ,blockXZ>>>(H_output,H_input,K,Nx,Ny,Nz,A,h,BCsouth);
	stencil_head_side_north<<<gridXZ,blockXZ>>>(H_output,H_input,K,Nx,Ny,Nz,A,h,BCnorth);

	stencil_head_side_west<<<gridYZ,blockYZ>>>(H_output,H_input,K,Nx,Ny,Nz,A,h,BCwest);
	stencil_head_side_east<<<gridYZ,blockYZ>>>(H_output,H_input,K,Nx,Ny,Nz,A,h,BCeast);

	stencil_head_edge_X_South_Bottom<<<grid.x,block.x>>>(H_output,H_input,K,Nx,Ny,Nz,A,h,BCsouth,BCbottom);
	stencil_head_edge_X_South_Top	<<<grid.x,block.x>>>(H_output,H_input,K,Nx,Ny,Nz,A,h,BCsouth,BCtop);
	stencil_head_edge_X_North_Bottom<<<grid.x,block.x>>>(H_output,H_input,K,Nx,Ny,Nz,A,h,BCnorth,BCbottom);
	stencil_head_edge_X_North_Top	<<<grid.x,block.x>>>(H_output,H_input,K,Nx,Ny,Nz,A,h,BCnorth,BCtop);

	stencil_head_edge_Z_South_West<<<grid.z,block.z>>>(H_output,H_input,K,Nx,Ny,Nz,A,h,BCsouth,BCwest);
	stencil_head_edge_Z_South_East<<<grid.z,block.z>>>(H_output,H_input,K,Nx,Ny,Nz,A,h,BCsouth,BCeast);
	stencil_head_edge_Z_North_West<<<grid.z,block.z>>>(H_output,H_input,K,Nx,Ny,Nz,A,h,BCnorth,BCwest);
	stencil_head_edge_Z_North_East<<<grid.z,block.z>>>(H_output,H_input,K,Nx,Ny,Nz,A,h,BCnorth,BCeast);

	stencil_head_edge_Y_Bottom_West	<<<grid.y,block.y>>>(H_output,H_input,K,Nx,Ny,Nz,A,h,BCbottom,BCwest);
	stencil_head_edge_Y_Top_West	<<<grid.y,block.y>>>(H_output,H_input,K,Nx,Ny,Nz,A,h,BCtop,BCwest);
	stencil_head_edge_Y_Bottom_East	<<<grid.y,block.y>>>(H_output,H_input,K,Nx,Ny,Nz,A,h,BCbottom,BCeast);
	stencil_head_edge_Y_Top_East	<<<grid.y,block.y>>>(H_output,H_input,K,Nx,Ny,Nz,A,h,BCtop,BCeast);

	stencil_head_vertex_SWB<<<1,1>>>(H_output,H_input,K,Nx,Ny,Nz,A,h,BCsouth,BCwest,BCbottom,pin1stCell);
	stencil_head_vertex_SWT<<<1,1>>>(H_output,H_input,K,Nx,Ny,Nz,A,h,BCsouth,BCwest,BCtop);
	stencil_head_vertex_SEB<<<1,1>>>(H_output,H_input,K,Nx,Ny,Nz,A,h,BCsouth,BCeast,BCbottom);
	stencil_head_vertex_SET<<<1,1>>>(H_output,H_input,K,Nx,Ny,Nz,A,h,BCsouth,BCeast,BCtop);

	stencil_head_vertex_NEB<<<1,1>>>(H_output,H_input,K,Nx,Ny,Nz,A,h,BCnorth,BCeast,BCbottom);
	stencil_head_vertex_NET<<<1,1>>>(H_output,H_input,K,Nx,Ny,Nz,A,h,BCnorth,BCeast,BCtop);
	stencil_head_vertex_NWB<<<1,1>>>(H_output,H_input,K,Nx,Ny,Nz,A,h,BCnorth,BCwest,BCbottom);
	stencil_head_vertex_NWT<<<1,1>>>(H_output,H_input,K,Nx,Ny,Nz,A,h,BCnorth,BCwest,BCtop);
	cudaDeviceSynchronize();
}

// #undef BLOCK_Nx
// #undef BLOCK_Ny
```

# step_ADV_FowEuler_TVD_3D_2nd_order_r1.cu

```cu
/**
* @file *.cu
* @brief *****************************
*
* @author Lucas Bessone (contact: lcbessone@gmail.com)
*
* @copyright This file is part of the EU-PAR software.
*            Copyright (C) 2025 Lucas Bessone
*
* @license This program is free software: you can redistribute it and/or modify
*          it under the terms of the GNU General Public License as published by
*          the Free Software Foundation, either version 3 of the License, or
*          (at your option) any later version.
*
*          This program is distributed in the hope that it will be useful,
*          but WITHOUT ANY WARRANTY; without even the implied warranty of
*          MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*          GNU General Public License for more details.
*
*          You should have received a copy of the GNU General Public License
*          along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/
#include <cuda_runtime_api.h>
#include "../lib_flow/header/macros_index_kernel.h"
#include "../lib_flow/header/macros_index_for_mf.h"


// #define BLOCK_Nx 16
// #define BLOCK_Ny 16

#define neumann 0
#define periodic 1
#define dirichlet 2

#define wall 0 //equiv to neumann for ADV & DIFF
#define inlet 2 //equiv. to dirichlet for ADV & DIFF
#define outlet 3 //equiv. neumann for DIFF, and phib <- phiC for ADV

__device__ inline double mimin (const double & a, const double & b){
	return ((a <= b) ? a : b);
}
__device__ inline double mimax (const double & a, const double & b){
	return ((a >= b) ? a : b);
}

// minmod
__device__ inline double psi(const double &c1, const double &c2, const double &c3){
	double rf = (c1-c2)/(c3-c1+1e-16);
	return mimax( 0, mimin(1.0,rf) );
}

// upwind
// __device__ inline double psi(const double &c1, const double &c2, const double &c3){
// 	return 0.0;
// }

// // CD
// __device__ inline double psi(const double &c1, const double &c2, const double &c3){
// 	// double rf = (c1-c2)/(c3-c1+1e-16);
// 	return 1.0;
// }


// // QUICK
// __device__ inline double psi(const double &c1, const double &c2, const double &c3){
// 	double rf = (c1-c2)/(c3-c1+1e-16);
// 	return (3.0+rf) / 4.0;
// }

// // SECOND ORDER UPWIND
// __device__ inline double psi(const double &c1, const double &c2, const double &c3){
// 	double rf = (c1-c2)/(c3-c1+1e-16);
// 	return rf;
// }

// #define psiEplus1 psi(phi_current,s_phi[tx-1][ty],s_phi[tx+1][ty])
// #define psiEminus1 psi(s_phi[tx+1][ty],phi_G,phi_current)
// #define psiWplus1 psi(phi_current,s_phi[tx+1][ty],s_phi[tx-1][ty])
// #define psiWminus1 psi(s_phi[tx-1][ty],phi_G,phi_current)

// #define psiNplus1 psi(phi_current,s_phi[tx][ty-1],s_phi[tx][ty+1])
// #define psiNminus1 psi(s_phi[tx][ty+1],phi_G,phi_current)
// #define psiSplus1 psi(phi_current,s_phi[tx][ty+1],s_phi[tx][ty-1])
// #define psiSminus1 psi(s_phi[tx][ty-1],phi_G,phi_current)

// #define psiTplus1 psi(phi_current,phi_bottom,phi_top)
// #define psiTminus1 psi(phi_top,phi_G,phi_current)
// #define psiBplus1 psi(phi_current,phi_top,phi_bottom)
// #define psiBminus1 psi(phi_bottom,phi_G,phi_current)

//#######################################################################
// 	routine step Foward Euler (pure advection eq.)
//#######################################################################
//-----------------------------------------------------------------------
__global__ void step_FE_TVD_all_domain(double *phi_out, double *phi_in,
	double *Up, double *Vp, double *Wp,
	int Nx, int Ny, int Nz, double A, double V_dt,
	int BC_WEST, int BC_EAST, int BC_SOUTH, int BC_NORTH, int BC_BOTTOM, int BC_TOP,
	double phi_WEST, double phi_EAST, double phi_SOUTH, double phi_NORTH, double phi_BOTTOM, double phi_TOP){

	int ix = threadIdx.x + blockIdx.x*blockDim.x;
	int iy = threadIdx.y + blockIdx.y*blockDim.y;
	if (ix >= Nx || iy >= Ny) return;
	int in_idx = (ix) + (iy)*Nx;
	double me, mw, mn, ms, mt, mb;
	double phiU, phiUU, phiD, phiDD, result;

	// int out_idx;
	double phiT, phiB, phiC;
	phiC = phi_in[in_idx];
	mb = -Wp[in_idx]*A;
	// out_idx = in_idx;
	// in_idx += stride;

	phiT = phi_in[in_idx+stride];
	phiB = 0.0;
	// in_idx += stride;
for(int iz=0; iz<Nz; ++iz){
	if(iz>0){
		in_idx += stride;
		phiB = phiC;
		phiC = phiT;
		phiT = phi_in[in_idx+stride];
		mb = -mt;
	}

	mt = Wp[in_idx+stride]*A;// Adding stride because Wp is of size Nx*Ny*(Nz+1)
	me = Up[(ix+1)+(iy)*(Nx+1)+(iz)*(Nx+1)*Ny]*A; // Adding offsert because Up is of size (Nx+1)*Ny*Nz
	mw = -Up[(ix)+(iy)*(Nx+1)+(iz)*(Nx+1)*Ny]*A;
	mn = Vp[(ix)+(iy+1)*(Nx)+(iz)*Nx*(Ny+1)]*A; // Adding stride because Vp is of size Nx*(Ny+1)*Nz
	ms = -Vp[(ix)+(iy)*(Nx)+(iz)*Nx*(Ny+1)]*A;
	result = 0.0;

	// =============================================================
	phiU  = (ix<=0)    ? 0 : phi_in[in_idx-1];
	phiUU = (ix<=1)    ? 0 : phi_in[in_idx-2];
	phiD  = (ix>=Nx-1) ? 0 : phi_in[in_idx+1];
	phiDD = (ix>=Nx-2) ? 0 : phi_in[in_idx+2];

	if ((BC_WEST==outlet) || (BC_WEST==wall) && (ix==1)) phiUU = phiU;
	if ((BC_WEST==inlet) && (ix==1)) phiUU = 2.0*phi_WEST-phiU;
	if ((BC_WEST==periodic) && (ix<=1)) {
		phiUU = phi_in[in_idx-1+PERIODIC_WEST];
		if (ix==0) phiU = phi_in[in_idx+PERIODIC_WEST];
	}

	if ((BC_EAST==outlet) || (BC_EAST==wall) && (ix==Nx-2)) phiDD = phiD;
	if ((BC_EAST==inlet) && (ix==Nx-2)) phiDD = 2.0*phi_EAST-phiD;
	if ((BC_EAST==periodic) && (ix>=Nx-2)) {
		phiDD = phi_in[in_idx+1+PERIODIC_EAST];
		if (ix==Nx-1) phiD = phi_in[in_idx+PERIODIC_EAST];
	}

	if (ix>0) result+= (phiU+0.5*psi(phiU,phiUU,phiC)*(phiC-phiU))*mimax(-mw,0.0)
		              -(phiC+0.5*psi(phiC,phiD,phiU)*(phiU-phiC))*mimax(mw,0.0);
	if (ix==0){
		if (BC_WEST == inlet) result += -mw*phi_WEST;
		if (BC_WEST == outlet) result += -mw*phiC;
		if (BC_WEST == periodic) result += (phiU+0.5*psi(phiU,phiUU,phiC)*(phiC-phiU))*mimax(-mw,0.0)
			                              -(phiC+0.5*psi(phiC,phiD,phiU) *(phiU-phiC))*mimax( mw,0.0);
	}
	if (ix<Nx-1) result+= (phiD+0.5*psi(phiD,phiDD,phiC)*(phiC-phiD))*mimax(-me,0.0)
	                     -(phiC+0.5*psi(phiC,phiU,phiD)*(phiD-phiC) )*mimax(me,0.0);
	if (ix==Nx-1){
		if (BC_EAST == inlet) result += -me*phi_EAST;
		if (BC_EAST == outlet) result += -me*phiC;
		if (BC_EAST == periodic) result += (phiD+0.5*psi(phiD,phiDD,phiC)*(phiC-phiD))*mimax(-me,0.0)
	                                      -(phiC+0.5*psi(phiC,phiU,phiD)*(phiD-phiC) )*mimax(me,0.0);
	}
	// =============================================================
	phiU  = (iy<=0)    ? 0 : phi_in[in_idx-Nx];
	phiUU = (iy<=1)    ? 0 : phi_in[in_idx-2*Nx];
	phiD  = (iy>=Ny-1) ? 0 : phi_in[in_idx+Nx];
	phiDD = (iy>=Ny-2) ? 0 : phi_in[in_idx+2*Nx];

	if ((BC_SOUTH==outlet) || (BC_SOUTH==wall) && (iy==1)) phiUU = phiU;
	if ((BC_SOUTH==inlet) && (iy==1)) phiUU = 2.0*phi_SOUTH-phiU;
	if ((BC_SOUTH==periodic) && (iy<=1)) {
		phiUU = phi_in[in_idx-Nx+PERIODIC_SOUTH];
		if (iy==0) phiU = phi_in[in_idx+PERIODIC_SOUTH];
	}

	if ((BC_NORTH==outlet) || (BC_NORTH==wall) && (iy==Ny-2)) phiDD = phiD;
	if ((BC_NORTH==inlet) && (iy==Ny-2)) phiDD = 2.0*phi_NORTH-phiD;
	if ((BC_NORTH==periodic) && (iy>=Ny-2)) {
		phiDD = phi_in[in_idx+Nx+PERIODIC_NORTH];
		if (iy==Ny-1) phiD = phi_in[in_idx+PERIODIC_NORTH];
	}

	if (iy>0) result+= (phiU+0.5*psi(phiU,phiUU,phiC)*(phiC-phiU))*mimax(-ms,0.0)
		              -(phiC+0.5*psi(phiC,phiD,phiU)*(phiU-phiC))*mimax(ms,0.0);
	if (iy==0){
		if (BC_SOUTH == inlet) result += -ms*phi_SOUTH;
		if (BC_SOUTH == outlet) result += -ms*phiC;
		if (BC_SOUTH == periodic) result += (phiU+0.5*psi(phiU,phiUU,phiC)*(phiC-phiU))*mimax(-ms,0.0)
			                               -(phiC+0.5*psi(phiC,phiD,phiU) *(phiU-phiC))*mimax( ms,0.0);
	}
	if (iy<Ny-1) result+= (phiD+0.5*psi(phiD,phiDD,phiC)*(phiC-phiD))*mimax(-mn,0.0)
	                     -(phiC+0.5*psi(phiC,phiU,phiD)*(phiD-phiC) )*mimax(mn,0.0);
	if (iy==Ny-1){
		if (BC_NORTH == inlet) result += -mn*phi_NORTH;
		if (BC_NORTH == outlet) result += -mn*phiC;
		if (BC_NORTH == periodic) result += (phiD+0.5*psi(phiD,phiDD,phiC)*(phiC-phiD))*mimax(-mn,0.0)
	                                       -(phiC+0.5*psi(phiC,phiU,phiD) *(phiD-phiC))*mimax( mn,0.0);
	}
	// =============================================================
	phiU  = (iz<=0)    ? 0 : phiB;
	phiUU = (iz<=1)    ? 0 : phi_in[in_idx-2*stride];
	phiD  = (iz>=Nz-1) ? 0 : phiT;
	phiDD = (iz>=Nz-2) ? 0 : phi_in[in_idx+2*stride];

	if ((BC_BOTTOM==outlet) || (BC_BOTTOM==wall) && (iz==1)) phiUU = phiU;
	if ((BC_BOTTOM==inlet) && (iz==1)) phiUU = 2.0*phi_BOTTOM-phiU;
	if ((BC_BOTTOM==periodic) && (iz<=1)) {
		phiUU = phi_in[in_idx-stride+PERIODIC_BOTTOM];
		if (iz==0) phiU = phi_in[in_idx+PERIODIC_BOTTOM];
	}

	if ((BC_TOP==outlet) || (BC_TOP==wall) && (iz==Nz-2)) phiDD = phiD;
	if ((BC_TOP==inlet) && (iz==Nz-2)) phiDD = 2.0*phi_TOP-phiD;
	if ((BC_TOP==periodic) && (iz>=Nz-2)) {
		phiDD = phi_in[in_idx+stride+PERIODIC_TOP];
		if (iz==Nz-1) phiD = phi_in[in_idx+PERIODIC_TOP];
	}

	if (iz>0) result+= (phiU+0.5*psi(phiU,phiUU,phiC)*(phiC-phiU))*mimax(-mb,0.0)
		              -(phiC+0.5*psi(phiC,phiD,phiU) *(phiU-phiC))*mimax( mb,0.0);
	if (iz==0){
		if (BC_BOTTOM == inlet) result += -mb*phi_BOTTOM;
		if (BC_BOTTOM == outlet) result += -mb*phiC;
		if (BC_BOTTOM == periodic) result += (phiU+0.5*psi(phiU,phiUU,phiC)*(phiC-phiU))*mimax(-mb,0.0)
			                                -(phiC+0.5*psi(phiC,phiD,phiU) *(phiU-phiC))*mimax( mb,0.0);
	}
	if (iz<Nz-1) result+= (phiD+0.5*psi(phiD,phiDD,phiC)*(phiC-phiD))*mimax(-mt,0.0)
	                     -(phiC+0.5*psi(phiC,phiU,phiD) *(phiD-phiC))*mimax( mt,0.0);
	if (iz==Nz-1){
		if (BC_TOP == inlet) result += -mt*phi_TOP;
		if (BC_TOP == outlet) result += -mt*phiC;
		if (BC_TOP == periodic) result += (phiD+0.5*psi(phiD,phiDD,phiC)*(phiC-phiD))*mimax(-mt,0.0)
	                                     -(phiC+0.5*psi(phiC,phiU,phiD)* (phiD-phiC))*mimax( mt,0.0);
	}
	// =============================================================
	phi_out[in_idx] = 1.0/V_dt*(  V_dt*phiC + result);
	}
}

void step_FE_TVD(double *phi_out, double *phi_in, double *U, double *V, double *W, int Nx, int Ny, int Nz, double A, double V_dt, int BC_WEST, int BC_EAST, int BC_SOUTH, int BC_NORTH, int BC_BOTTOM, int BC_TOP, double phi_WEST, double phi_EAST, double phi_SOUTH, double phi_NORTH, double phi_BOTTOM, double phi_TOP, dim3 grid, dim3 block){
step_FE_TVD_all_domain<<<GRIDBLOCK_BOTTOM>>>(phi_out,phi_in,U,V,W,Nx,Ny,Nz,A,V_dt,BC_WEST,BC_EAST,BC_SOUTH,BC_NORTH,BC_BOTTOM,BC_TOP,phi_WEST,phi_EAST,phi_SOUTH,phi_NORTH,phi_BOTTOM,phi_TOP);
cudaDeviceSynchronize();
}

// #undef BLOCK_Nx
// #undef BLOCK_Ny
```

# step_ADV_FowEuler_TVD_3D_momentos.cu

```cu
/**
* @file *.cu
* @brief *****************************
*
* @author Lucas Bessone (contact: lcbessone@gmail.com)
*
* @copyright This file is part of the EU-PAR software.
*            Copyright (C) 2025 Lucas Bessone
*
* @license This program is free software: you can redistribute it and/or modify
*          it under the terms of the GNU General Public License as published by
*          the Free Software Foundation, either version 3 of the License, or
*          (at your option) any later version.
*
*          This program is distributed in the hope that it will be useful,
*          but WITHOUT ANY WARRANTY; without even the implied warranty of
*          MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*          GNU General Public License for more details.
*
*          You should have received a copy of the GNU General Public License
*          along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/
#include <cuda_runtime_api.h>
#include "header/macros_index_kernel.h"
#include "header/macros_index_for_mf.h"
#include "header/macros_momentos.h"

#define BLOCK_Nx 16
#define BLOCK_Ny 16

#define neumann 0
#define periodic 1
#define dirichlet 2

#define wall 0 //equiv to neumann for ADV & DIFF
#define inlet 2 //equiv. to dirichlet for ADV & DIFF
#define outlet 3 //equiv. neumann for DIFF, and phib <- phiC for ADV

__device__ inline double mimin (const double & a, const double & b){
	return ((a <= b) ? a : b);
}
__device__ inline double mimax (const double & a, const double & b){
	return ((a >= b) ? a : b);
}

// minmod
__device__ inline double psi(const double &c1, const double &c2, const double &c3){
	double rf = (c1-c2)/(c3-c1+1e-16);
	return mimax( 0, mimin(1.0,rf) );
}

// upwind
// __device__ inline double psi(const double &c1, const double &c2, const double &c3){
// 	return 0.0;
// }

// // CD
// __device__ inline double psi(const double &c1, const double &c2, const double &c3){
// 	// double rf = (c1-c2)/(c3-c1+1e-16);
// 	return 1.0;
// }


// // QUICK
// __device__ inline double psi(const double &c1, const double &c2, const double &c3){
// 	double rf = (c1-c2)/(c3-c1+1e-16);
// 	return (3.0+rf) / 4.0;
// }

// // SECOND ORDER UPWIND
// __device__ inline double psi(const double &c1, const double &c2, const double &c3){
// 	double rf = (c1-c2)/(c3-c1+1e-16);
// 	return rf;
// }

#define psiEplus1 psi(phi_current,s_phi[tx-1][ty],s_phi[tx+1][ty])
#define psiEminus1 psi(s_phi[tx+1][ty],phi_in[out_idx+2],phi_current)
#define psiWplus1 psi(phi_current,s_phi[tx+1][ty],s_phi[tx-1][ty])
#define psiWminus1 psi(s_phi[tx-1][ty],phi_in[out_idx-2],phi_current)

#define psiNplus1 psi(phi_current,s_phi[tx][ty-1],s_phi[tx][ty+1])
#define psiNminus1 psi(s_phi[tx][ty+1],phi_in[out_idx+2*Nx],phi_current)
#define psiSplus1 psi(phi_current,s_phi[tx][ty+1],s_phi[tx][ty-1])
#define psiSminus1 psi(s_phi[tx][ty-1],phi_in[out_idx-2*Nx],phi_current)

#define psiTplus1 psi(phi_current,phi_bottom,phi_top)
#define psiTminus1 psi(phi_top,phi_in[out_idx+2*stride],phi_current)
#define psiBplus1 psi(phi_current,phi_top,phi_bottom)
#define psiBminus1 psi(phi_bottom,phi_in[out_idx-2*stride],phi_current)

//#######################################################################
// 	routine step Foward Euler (pure advection eq.)
//#######################################################################
//-----------------------------------------------------------------------
__global__ void step_FE_TVD_int(
double *phi_out, double *phi_in, double *Up,
 double *Vp,
 double *Wp,
int Nx, int Ny, int Nz, double A, double V_dt, double nuA_h, double h, float *momento1x, float *momento2x, float *momento1y, float*momento2y, float *momento1z, float *momento2z, float *C_float){
	__shared__ double s_phi[1+BLOCK_Nx+1][1+BLOCK_Ny+1];
	int ix = threadIdx.x + blockIdx.x*blockDim.x;
	int iy = threadIdx.y + blockIdx.y*blockDim.y;
	if (ix >= Nx-2 || iy >= Ny-2) return;
	int in_idx = (ix + 1) + (iy + 1)*Nx;
	double me, mw, mn, ms, mt, mb;
	double mc;
	double west, east, south, north, bottom, top;

	int out_idx;
	double phi_top, phi_bottom, phi_current;

	const int tx = threadIdx.x + 1;
	const int ty = threadIdx.y + 1;

//-------------------------------------
// Rhs = temporal-0.5*3Np+0.5Dp-gradP
//-------------------------------------
	phi_current = phi_in[in_idx];

	mt = Wp[in_idx+stride]*A;// Adding stride because Wp is of size Nx*Ny*(Nz+1)

	out_idx = in_idx;
	in_idx += stride;

	phi_top = phi_in[in_idx];
	in_idx += stride;
	for(int i=1; i<Nz-1; ++i){
		if( (tx < Nx-1) && (ty < Ny-1) ){
			phi_bottom = phi_current;
			phi_current = phi_top;
			phi_top = phi_in[in_idx];

			in_idx += stride;
			out_idx += stride;

			mb = -mt;
			mt = Wp[out_idx+stride]*A;// Adding stride because Wp is of size Nx*Ny*(Nz+1)

			if(tx==1) s_phi[0][ty] = phi_in[out_idx-1];
			if(ix==Nx-3 || tx==BLOCK_Nx) s_phi[tx+1][ty] = phi_in[out_idx+1];
			if(ty==1) s_phi[tx][0] = phi_in[out_idx-Nx];
			if(iy==Ny-3 || ty==BLOCK_Ny) s_phi[tx][ty+1] = phi_in[out_idx+Nx];

			s_phi[tx][ty] = phi_current;

			me = Up[(ix+1+1)+(iy+1)*(Nx+1)+(i)*(Nx+1)*Ny]*A; // Adding offsert because Up is of size (Nx+1)*Ny*Nz
			mw = -Up[(ix+1)+(iy+1)*(Nx+1)+(i)*(Nx+1)*Ny]*A;
			mn = Vp[(ix+1)+(iy+1+1)*(Nx)+(i)*Nx*(Ny+1)]*A; // Adding stride because Vp is of size Nx*(Ny+1)*Nz
			ms = -Vp[(ix+1)+(iy+1)*(Nx)+(i)*Nx*(Ny+1)]*A;
			__syncthreads();

			mc = mimax(me,0.0)+mimax(mw,0.0)
				+mimax(mn,0.0)+mimax(ms,0.0)
				+mimax(mt,0.0)+mimax(mb,0.0);
// =============================================================
if(ix>0 && ix<Nx-3 && iy>0 && iy<Ny-3 && i>1 && i<Nz-2){
	west = (ix==1) ? - ( -mimax(-mw, 0.0) )*s_phi[tx-1][ty] - mimax(mw,0.0)*phi_current
				   : - ( phi_current+0.5*psiWplus1*(s_phi[tx-1][ty]-phi_current) )*mimax(mw,0.0) + (s_phi[tx-1][ty]+0.5*psiWminus1*(phi_current-s_phi[tx-1][ty]))*mimax(-mw,0.0);

	east = (ix==Nx-4) ? - ( -mimax(-me, 0.0) )*s_phi[tx+1][ty] - mimax(me,0.0)*phi_current
				      : - ( phi_current+0.5*psiEplus1*(s_phi[tx+1][ty]-phi_current) )*mimax(me,0.0) + (s_phi[tx+1][ty]+0.5*psiEminus1*(phi_current-s_phi[tx+1][ty]))*mimax(-me,0.0);

	south = (iy==1) ? - ( -mimax(-ms, 0.0) )*s_phi[tx][ty-1] - mimax(ms,0.0)*phi_current
				    : - ( phi_current+0.5*psiSplus1*(s_phi[tx][ty-1]-phi_current) )*mimax(ms,0.0) + (s_phi[tx][ty-1]+0.5*psiSminus1*(phi_current-s_phi[tx][ty-1]))*mimax(-ms,0.0);

	north = (iy==Ny-4) ? - ( -mimax(-mn, 0.0) )*s_phi[tx][ty+1] - mimax(mn,0.0)*phi_current
					   : - ( phi_current+0.5*psiNplus1*(s_phi[tx][ty+1]-phi_current) )*mimax(mn,0.0) + (s_phi[tx][ty+1]+0.5*psiNminus1*(phi_current-s_phi[tx][ty+1]))*mimax(-mn,0.0);

	bottom = (i==2) ? - ( -mimax(-mb, 0.0) )*phi_bottom - mimax(mb,0.0)*phi_current
				    : - ( phi_current+0.5*psiBplus1*(phi_bottom-phi_current) )*mimax(mb,0.0) + (phi_bottom+0.5*psiBminus1*(phi_current-phi_bottom))*mimax(-mb,0.0);

	top = (i==Nz-3) ? - ( -mimax(-mt, 0.0) )*phi_top - mimax(mt,0.0)*phi_current
					: - ( phi_current+0.5*psiTplus1*(phi_top-phi_current) )*mimax(mt,0.0) + (phi_top+0.5*psiTminus1*(phi_current-phi_top))*mimax(-mt,0.0);
	phi_out[out_idx] =
	1.0/V_dt*(
		   (V_dt-nuA_h*6.0)*phi_current
	+ nuA_h* ( s_phi[tx+1][ty]+s_phi[tx-1][ty]
		      +s_phi[tx][ty+1]+s_phi[tx][ty-1]
		      +phi_top+phi_bottom   )
	+ east + west + north + south + top + bottom    //TVD - adv contrib
	         );
} else { // upwind scheme
	phi_out[out_idx] =
	1.0/V_dt*(
		(V_dt-nuA_h*6.0 - mc)*phi_current
	+ (nuA_h + mimax(-me, 0.0)) *s_phi[tx+1][ty]
	+ (nuA_h + mimax(-mw, 0.0)) *s_phi[tx-1][ty]
	+ (nuA_h + mimax(-mn, 0.0)) *s_phi[tx][ty+1]
	+ (nuA_h + mimax(-ms, 0.0)) *s_phi[tx][ty-1]
	+ (nuA_h + mimax(-mt, 0.0)) *phi_top
	+ (nuA_h + mimax(-mb, 0.0)) *phi_bottom
	        );
		}
// =============================================================
					}
   	momento1x[out_idx] = (float)((ix+1.5)*h)          *  (float)phi_current  *(float)(h*h*h);
	momento1y[out_idx] = (float)((iy+1.5)*h )         *  (float)phi_current  *(float)(h*h*h);
	momento1z[out_idx] = (float)((i+0.5)*h)           *  (float)phi_current  *(float)(h*h*h);
	momento2x[out_idx] = (float)powf((ix+1.5f)*(float)h,2.0f)*  (float)phi_current  *(float)(h*h*h);
	momento2y[out_idx] = (float)powf((iy+1.5f)*(float)h,2.0f)*  (float)phi_current  *(float)(h*h*h);
	momento2z[out_idx] = (float)powf((i+0.5f)*(float)h,2.0f) *  (float)phi_current  *(float)(h*h*h);
	C_float[out_idx] = (float)phi_current;
__syncthreads();
	}
}

#define KERNEL_SIDE(FACE) step_FE_TVD_side_##FACE(double *phi_out, double *phi_in, \
double *Up, double *Vp, double *Wp, int Nx, int Ny, int Nz, double A, double V_dt, double nuA_h, \
int BC_##FACE, double phi_##FACE, double h, float *momento1x, float *momento2x, float *momento1y, float*momento2y, float *momento1z, float *momento2z, float *C_float){ \
COMPUTE_INDEX_##FACE \
OFFSETS_##FACE; \
COMPUTE_INDEX_NORMAL_VELOCITY_##FACE \
LOAD_FACE_FLUX_##FACE; \
double mc = 0.0, result = 0.0, aC = 5.0, phiC = phi_in[in_idx]; \
for (int i = 0; i < 5; i++) { \
	mc += mimax(m_f[i],0.0); \
    result += ( mimax(-m_f[i], 0.0) + nuA_h )*phi_in[in_idx+offsets[i]]; \
} \
if (BC_##FACE == inlet){ \
	result += ( -m_f[5] + 2.0*nuA_h )*phi_##FACE; \
	aC += 2.0; \
} \
if (BC_##FACE == outlet) result += ( -m_f[5] )*phiC; \
if (BC_##FACE == periodic) { \
	result += ( mimax(-m_f[5], 0.0) + nuA_h )*phi_in[in_idx+PERIODIC_##FACE]; \
	mc += mimax(m_f[5],0.0); \
} \
phi_out[in_idx] = 1.0/V_dt * (  (V_dt - nuA_h*aC - mc)*phiC + result  ); \
COMPUTE_MOMENTO_##FACE \
}

#define KERNEL_EDGE(FACE1,FACE2) step_FE_TVD_edge_##FACE1##_##FACE2(double *phi_out, double *phi_in, \
double *Up, double *Vp, double *Wp, int Nx, int Ny, int Nz, double A, double V_dt, double nuA_h, \
int BC_##FACE1, int BC_##FACE2, double phi_##FACE1, double phi_##FACE2, double h, float *momento1x, float *momento2x, float *momento1y, float*momento2y, float *momento1z, float *momento2z, float *C_float){ \
COMPUTE_INDEX_##FACE1##_##FACE2 \
OFFSETS_##FACE1##_##FACE2; \
COMPUTE_INDEX_NORMAL_VELOCITY_##FACE1##_##FACE2; \
LOAD_FACE_FLUX_##FACE1##_##FACE2; \
double mc = 0.0, result = 0.0, aC = 5.0, phiC = phi_in[in_idx]; \
for (int i = 0; i < 4; i++) { \
	mc += mimax(m_f[i],0.0); \
    result += ( mimax(-m_f[i], 0.0) + nuA_h )*phi_in[in_idx+offsets[i]]; \
} \
if (BC_##FACE1 == inlet){ \
	result += ( -m_f[4] + 2.0*nuA_h )*phi_##FACE1; \
	aC += 2.0; \
} \
if (BC_##FACE1 == outlet) result += ( -m_f[4] )*phiC; \
if (BC_##FACE1 == periodic) { \
	result += ( mimax(-m_f[4], 0.0) + nuA_h )*phi_in[in_idx+PERIODIC_##FACE1]; \
	mc += mimax(m_f[4],0.0); \
} \
if (BC_##FACE2 == inlet){ \
	result += ( -m_f[5] + 2.0*nuA_h )*phi_##FACE2; \
	aC += 2.0; \
} \
if (BC_##FACE2 == outlet) result += ( -m_f[5] )*phiC; \
if (BC_##FACE2 == periodic) { \
	result += ( mimax(-m_f[5], 0.0) + nuA_h )*phi_in[in_idx+PERIODIC_##FACE2]; \
	mc += mimax(m_f[5],0.0); \
} \
phi_out[in_idx] = 1.0/V_dt * (  (V_dt - nuA_h*aC - mc)*phiC + result  ); \
COMPUTE_MOMENTO_##FACE1##_##FACE2 \
}

#define KERNEL_VERTEX(FACE1,FACE2,FACE3) step_FE_TVD_vertex_##FACE1##_##FACE2##_##FACE3(double *phi_out, double *phi_in, \
double *Up, double *Vp, double *Wp, int Nx, int Ny, int Nz, double A, double V_dt, double nuA_h, \
int BC_##FACE1, int BC_##FACE2, int BC_##FACE3, double phi_##FACE1, double phi_##FACE2, double phi_##FACE3, double h, float *momento1x, float *momento2x, float *momento1y, float*momento2y, float *momento1z, float *momento2z, float *C_float){ \
COMPUTE_INDEX_##FACE1##_##FACE2##_##FACE3 \
OFFSETS_##FACE1##_##FACE2##_##FACE3; \
COMPUTE_INDEX_NORMAL_VELOCITY_##FACE1##_##FACE2##_##FACE3; \
LOAD_FACE_FLUX_##FACE1##_##FACE2##_##FACE3; \
double mc = 0.0, result = 0.0, aC = 5.0, phiC = phi_in[in_idx]; \
for (int i = 0; i < 3; i++) { \
	mc += mimax(m_f[i],0.0); \
    result += ( mimax(-m_f[i], 0.0) + nuA_h )*phi_in[in_idx+offsets[i]]; \
} \
if (BC_##FACE1 == inlet){ \
	result += ( -m_f[3] + 2.0*nuA_h )*phi_##FACE1; \
	aC += 2.0; \
} \
if (BC_##FACE1 == outlet) result += ( -m_f[3] )*phiC; \
if (BC_##FACE1 == periodic) { \
	result += ( mimax(-m_f[3], 0.0) + nuA_h )*phi_in[in_idx+PERIODIC_##FACE1]; \
	mc += mimax(m_f[3],0.0); \
} \
if (BC_##FACE2 == inlet){ \
	result += ( -m_f[4] + 2.0*nuA_h )*phi_##FACE2; \
	aC += 2.0; \
} \
if (BC_##FACE2 == outlet) result += ( -m_f[4] )*phiC; \
if (BC_##FACE2 == periodic) { \
	result += ( mimax(-m_f[4], 0.0) + nuA_h )*phi_in[in_idx+PERIODIC_##FACE2]; \
	mc += mimax(m_f[4],0.0); \
} \
if (BC_##FACE3 == inlet){ \
	result += ( -m_f[5] + 2.0*nuA_h )*phi_##FACE3; \
	aC += 2.0; \
} \
if (BC_##FACE3 == outlet) result += ( -m_f[5] )*phiC; \
if (BC_##FACE3 == periodic) { \
	result += ( mimax(-m_f[5], 0.0) + nuA_h )*phi_in[in_idx+PERIODIC_##FACE3]; \
	mc += mimax(m_f[5],0.0); \
} \
phi_out[in_idx] = 1.0/V_dt * (  (V_dt - nuA_h*aC - mc)*phiC + result  ); \
COMPUTE_MOMENTO_##FACE1##_##FACE2##_##FACE3 \
}


// kernel declaration for all domain except interior
__global__ void KERNEL_SIDE(BOTTOM)
__global__ void KERNEL_SIDE(TOP)
__global__ void KERNEL_SIDE(NORTH)
__global__ void KERNEL_SIDE(SOUTH)
__global__ void KERNEL_SIDE(WEST)
__global__ void KERNEL_SIDE(EAST)
// edge y
__global__ void KERNEL_EDGE(WEST,BOTTOM)
__global__ void KERNEL_EDGE(WEST,TOP)
__global__ void KERNEL_EDGE(EAST,BOTTOM)
__global__ void KERNEL_EDGE(EAST,TOP)
// edge z
__global__ void KERNEL_EDGE(WEST,SOUTH)
__global__ void KERNEL_EDGE(WEST,NORTH)
__global__ void KERNEL_EDGE(EAST,SOUTH)
__global__ void KERNEL_EDGE(EAST,NORTH)
// edge x
__global__ void KERNEL_EDGE(SOUTH,BOTTOM)
__global__ void KERNEL_EDGE(SOUTH,TOP)
__global__ void KERNEL_EDGE(NORTH,BOTTOM)
__global__ void KERNEL_EDGE(NORTH,TOP)

__global__ void KERNEL_VERTEX(WEST,SOUTH,BOTTOM)
__global__ void KERNEL_VERTEX(WEST,SOUTH,TOP)
__global__ void KERNEL_VERTEX(WEST,NORTH,BOTTOM)
__global__ void KERNEL_VERTEX(WEST,NORTH,TOP)
__global__ void KERNEL_VERTEX(EAST,SOUTH,BOTTOM)
__global__ void KERNEL_VERTEX(EAST,SOUTH,TOP)
__global__ void KERNEL_VERTEX(EAST,NORTH,BOTTOM)
__global__ void KERNEL_VERTEX(EAST,NORTH,TOP)

#define LAUNCH_KERNEL_SIDE(FACE) \
    step_FE_TVD_side_##FACE<<<GRIDBLOCK_##FACE>>>(phi_out, phi_in, U, V, W, Nx, Ny, Nz, A, V_dt, nuA_h, BC_##FACE, phi_##FACE,h,momento1x,momento2x,momento1y,momento2y,momento1z,momento2z,C_float)
#define LAUNCH_KERNEL_EDGE(FACE1,FACE2) \
    step_FE_TVD_edge_##FACE1##_##FACE2<<<GRIDBLOCK_##FACE1##_##FACE2>>>(phi_out, phi_in, U, V, W, Nx, Ny, Nz, A, V_dt, nuA_h, BC_##FACE1,BC_##FACE2, phi_##FACE1,phi_##FACE2,h,momento1x,momento2x,momento1y,momento2y,momento1z,momento2z,C_float)
#define LAUNCH_KERNEL_VERTEX(FACE1,FACE2,FACE3) \
    step_FE_TVD_vertex_##FACE1##_##FACE2##_##FACE3<<<1,1>>>(phi_out, phi_in, U, V, W, Nx, Ny, Nz, A, V_dt, nuA_h, BC_##FACE1,BC_##FACE2,BC_##FACE3, phi_##FACE1,phi_##FACE2,phi_##FACE3,h,momento1x,momento2x,momento1y,momento2y,momento1z,momento2z,C_float)
#define LAUNCH_KERNEL_INT \
    step_FE_TVD_int<<<GRIDBLOCK_BOTTOM>>>(phi_out, phi_in, U, V, W, Nx, Ny, Nz, A, V_dt, nuA_h,h,momento1x,momento2x,momento1y,momento2y,momento1z,momento2z,C_float)

void step_FE_TVD_with_moments(double *phi_out, double *phi_in, double *U, double *V, double *W, int Nx, int Ny, int Nz, double A, double V_dt, double nuA_h, int BC_WEST, int BC_EAST, int BC_SOUTH, int BC_NORTH, int BC_BOTTOM, int BC_TOP, double phi_WEST, double phi_EAST, double phi_SOUTH, double phi_NORTH, double phi_BOTTOM, double phi_TOP, dim3 grid, dim3 block, double h,float *momento1x,float *momento2x,float *momento1y,float *momento2y,float *momento1z,float *momento2z, float *C_float){
LAUNCH_KERNEL_INT;
LAUNCH_KERNEL_SIDE(BOTTOM);
LAUNCH_KERNEL_SIDE(TOP);
LAUNCH_KERNEL_SIDE(NORTH);
LAUNCH_KERNEL_SIDE(SOUTH);
LAUNCH_KERNEL_SIDE(WEST);
LAUNCH_KERNEL_SIDE(EAST);

LAUNCH_KERNEL_EDGE(SOUTH,BOTTOM);
LAUNCH_KERNEL_EDGE(SOUTH,TOP);
LAUNCH_KERNEL_EDGE(NORTH,BOTTOM);
LAUNCH_KERNEL_EDGE(NORTH,TOP);

LAUNCH_KERNEL_EDGE(WEST,SOUTH);
LAUNCH_KERNEL_EDGE(WEST,NORTH);
LAUNCH_KERNEL_EDGE(EAST,SOUTH);
LAUNCH_KERNEL_EDGE(EAST,NORTH);

LAUNCH_KERNEL_EDGE(WEST,BOTTOM);
LAUNCH_KERNEL_EDGE(WEST,TOP);
LAUNCH_KERNEL_EDGE(EAST,BOTTOM);
LAUNCH_KERNEL_EDGE(EAST,TOP);

LAUNCH_KERNEL_VERTEX(WEST,SOUTH,BOTTOM);
LAUNCH_KERNEL_VERTEX(WEST,SOUTH,TOP);
LAUNCH_KERNEL_VERTEX(WEST,NORTH,BOTTOM);
LAUNCH_KERNEL_VERTEX(WEST,NORTH,TOP);
LAUNCH_KERNEL_VERTEX(EAST,SOUTH,BOTTOM);
LAUNCH_KERNEL_VERTEX(EAST,SOUTH,TOP);
LAUNCH_KERNEL_VERTEX(EAST,NORTH,BOTTOM);
LAUNCH_KERNEL_VERTEX(EAST,NORTH,TOP);
cudaDeviceSynchronize();
}

#undef BLOCK_Nx
#undef BLOCK_Ny
```

# step_ADV_FowEuler_TVD_3D.cu

```cu
/**
* @file *.cu
* @brief *****************************
*
* @author Lucas Bessone (contact: lcbessone@gmail.com)
*
* @copyright This file is part of the EU-PAR software.
*            Copyright (C) 2025 Lucas Bessone
*
* @license This program is free software: you can redistribute it and/or modify
*          it under the terms of the GNU General Public License as published by
*          the Free Software Foundation, either version 3 of the License, or
*          (at your option) any later version.
*
*          This program is distributed in the hope that it will be useful,
*          but WITHOUT ANY WARRANTY; without even the implied warranty of
*          MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*          GNU General Public License for more details.
*
*          You should have received a copy of the GNU General Public License
*          along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/
#include <cuda_runtime_api.h>
#include "header/macros_index_kernel.h"
#include "header/macros_index_for_mf.h"

#define BLOCK_Nx 16
#define BLOCK_Ny 16

#define neumann 0
#define periodic 1
#define dirichlet 2

#define wall 0 //equiv to neumann for ADV & DIFF
#define inlet 2 //equiv. to dirichlet for ADV & DIFF
#define outlet 3 //equiv. neumann for DIFF, and phib <- phiC for ADV

__device__ inline double mimin (const double & a, const double & b){
	return ((a <= b) ? a : b);
}
__device__ inline double mimax (const double & a, const double & b){
	return ((a >= b) ? a : b);
}

// minmod
__device__ inline double psi(const double &c1, const double &c2, const double &c3){
	double rf = (c1-c2)/(c3-c1+1e-16);
	return mimax( 0, mimin(1.0,rf) );
}

// upwind
// __device__ inline double psi(const double &c1, const double &c2, const double &c3){
// 	return 0.0;
// }

// // CD
// __device__ inline double psi(const double &c1, const double &c2, const double &c3){
// 	// double rf = (c1-c2)/(c3-c1+1e-16);
// 	return 1.0;
// }


// // QUICK
// __device__ inline double psi(const double &c1, const double &c2, const double &c3){
// 	double rf = (c1-c2)/(c3-c1+1e-16);
// 	return (3.0+rf) / 4.0;
// }

// // SECOND ORDER UPWIND
// __device__ inline double psi(const double &c1, const double &c2, const double &c3){
// 	double rf = (c1-c2)/(c3-c1+1e-16);
// 	return rf;
// }

#define psiEplus1 psi(phi_current,s_phi[tx-1][ty],s_phi[tx+1][ty])
#define psiEminus1 psi(s_phi[tx+1][ty],phi_in[out_idx+2],phi_current)
#define psiWplus1 psi(phi_current,s_phi[tx+1][ty],s_phi[tx-1][ty])
#define psiWminus1 psi(s_phi[tx-1][ty],phi_in[out_idx-2],phi_current)

#define psiNplus1 psi(phi_current,s_phi[tx][ty-1],s_phi[tx][ty+1])
#define psiNminus1 psi(s_phi[tx][ty+1],phi_in[out_idx+2*Nx],phi_current)
#define psiSplus1 psi(phi_current,s_phi[tx][ty+1],s_phi[tx][ty-1])
#define psiSminus1 psi(s_phi[tx][ty-1],phi_in[out_idx-2*Nx],phi_current)

#define psiTplus1 psi(phi_current,phi_bottom,phi_top)
#define psiTminus1 psi(phi_top,phi_in[out_idx+2*stride],phi_current)
#define psiBplus1 psi(phi_current,phi_top,phi_bottom)
#define psiBminus1 psi(phi_bottom,phi_in[out_idx-2*stride],phi_current)

//#######################################################################
// 	routine step Foward Euler (pure advection eq.)
//#######################################################################
//-----------------------------------------------------------------------
__global__ void step_FE_TVD_int(
double *phi_out, double *phi_in, double *Up,
 double *Vp,
 double *Wp,
int Nx, int Ny, int Nz, double A, double V_dt, double nuA_h){
	__shared__ double s_phi[1+BLOCK_Nx+1][1+BLOCK_Ny+1];
	int ix = threadIdx.x + blockIdx.x*blockDim.x;
	int iy = threadIdx.y + blockIdx.y*blockDim.y;
	if (ix >= Nx-2 || iy >= Ny-2) return;
	int in_idx = (ix + 1) + (iy + 1)*Nx;
	double me, mw, mn, ms, mt, mb;
	double mc;
	double west, east, south, north, bottom, top;

	int out_idx;
	double phi_top, phi_bottom, phi_current;

	const int tx = threadIdx.x + 1;
	const int ty = threadIdx.y + 1;

//-------------------------------------
// Rhs = temporal-0.5*3Np+0.5Dp-gradP
//-------------------------------------
	phi_current = phi_in[in_idx];

	mt = Wp[in_idx+stride]*A;// Adding stride because Wp is of size Nx*Ny*(Nz+1)

	out_idx = in_idx;
	in_idx += stride;

	phi_top = phi_in[in_idx];
	in_idx += stride;
	for(int i=1; i<Nz-1; ++i){
		if( (tx < Nx-1) && (ty < Ny-1) ){
			phi_bottom = phi_current;
			phi_current = phi_top;
			phi_top = phi_in[in_idx];

			in_idx += stride;
			out_idx += stride;

			mb = -mt;
			mt = Wp[out_idx+stride]*A;// Adding stride because Wp is of size Nx*Ny*(Nz+1)

			if(tx==1) s_phi[0][ty] = phi_in[out_idx-1];
			if(ix==Nx-3 || tx==BLOCK_Nx) s_phi[tx+1][ty] = phi_in[out_idx+1];
			if(ty==1) s_phi[tx][0] = phi_in[out_idx-Nx];
			if(iy==Ny-3 || ty==BLOCK_Ny) s_phi[tx][ty+1] = phi_in[out_idx+Nx];

			s_phi[tx][ty] = phi_current;

			me = Up[(ix+1+1)+(iy+1)*(Nx+1)+(i)*(Nx+1)*Ny]*A; // Adding offsert because Up is of size (Nx+1)*Ny*Nz
			mw = -Up[(ix+1)+(iy+1)*(Nx+1)+(i)*(Nx+1)*Ny]*A;
			mn = Vp[(ix+1)+(iy+1+1)*(Nx)+(i)*Nx*(Ny+1)]*A; // Adding stride because Vp is of size Nx*(Ny+1)*Nz
			ms = -Vp[(ix+1)+(iy+1)*(Nx)+(i)*Nx*(Ny+1)]*A;
			__syncthreads();

			mc = mimax(me,0.0)+mimax(mw,0.0)
				+mimax(mn,0.0)+mimax(ms,0.0)
				+mimax(mt,0.0)+mimax(mb,0.0);
// =============================================================
if(ix>0 && ix<Nx-3 && iy>0 && iy<Ny-3 && i>1 && i<Nz-2){
	west = (ix==1) ? - ( -mimax(-mw, 0.0) )*s_phi[tx-1][ty] - mimax(mw,0.0)*phi_current
				   : - ( phi_current+0.5*psiWplus1*(s_phi[tx-1][ty]-phi_current) )*mimax(mw,0.0) + (s_phi[tx-1][ty]+0.5*psiWminus1*(phi_current-s_phi[tx-1][ty]))*mimax(-mw,0.0);

	east = (ix==Nx-4) ? - ( -mimax(-me, 0.0) )*s_phi[tx+1][ty] - mimax(me,0.0)*phi_current
				      : - ( phi_current+0.5*psiEplus1*(s_phi[tx+1][ty]-phi_current) )*mimax(me,0.0) + (s_phi[tx+1][ty]+0.5*psiEminus1*(phi_current-s_phi[tx+1][ty]))*mimax(-me,0.0);

	south = (iy==1) ? - ( -mimax(-ms, 0.0) )*s_phi[tx][ty-1] - mimax(ms,0.0)*phi_current
				    : - ( phi_current+0.5*psiSplus1*(s_phi[tx][ty-1]-phi_current) )*mimax(ms,0.0) + (s_phi[tx][ty-1]+0.5*psiSminus1*(phi_current-s_phi[tx][ty-1]))*mimax(-ms,0.0);

	north = (iy==Ny-4) ? - ( -mimax(-mn, 0.0) )*s_phi[tx][ty+1] - mimax(mn,0.0)*phi_current
					   : - ( phi_current+0.5*psiNplus1*(s_phi[tx][ty+1]-phi_current) )*mimax(mn,0.0) + (s_phi[tx][ty+1]+0.5*psiNminus1*(phi_current-s_phi[tx][ty+1]))*mimax(-mn,0.0);

	bottom = (i==2) ? - ( -mimax(-mb, 0.0) )*phi_bottom - mimax(mb,0.0)*phi_current
				    : - ( phi_current+0.5*psiBplus1*(phi_bottom-phi_current) )*mimax(mb,0.0) + (phi_bottom+0.5*psiBminus1*(phi_current-phi_bottom))*mimax(-mb,0.0);

	top = (i==Nz-3) ? - ( -mimax(-mt, 0.0) )*phi_top - mimax(mt,0.0)*phi_current
					: - ( phi_current+0.5*psiTplus1*(phi_top-phi_current) )*mimax(mt,0.0) + (phi_top+0.5*psiTminus1*(phi_current-phi_top))*mimax(-mt,0.0);
	phi_out[out_idx] =
	1.0/V_dt*(
		   (V_dt-nuA_h*6.0)*phi_current
	+ nuA_h* ( s_phi[tx+1][ty]+s_phi[tx-1][ty]
		      +s_phi[tx][ty+1]+s_phi[tx][ty-1]
		      +phi_top+phi_bottom   )
	+ east + west + north + south + top + bottom    //TVD - adv contrib
	         );
} else { // upwind scheme
	phi_out[out_idx] =
	1.0/V_dt*(
		(V_dt-nuA_h*6.0 - mc)*phi_current
	+ (nuA_h + mimax(-me, 0.0)) *s_phi[tx+1][ty]
	+ (nuA_h + mimax(-mw, 0.0)) *s_phi[tx-1][ty]
	+ (nuA_h + mimax(-mn, 0.0)) *s_phi[tx][ty+1]
	+ (nuA_h + mimax(-ms, 0.0)) *s_phi[tx][ty-1]
	+ (nuA_h + mimax(-mt, 0.0)) *phi_top
	+ (nuA_h + mimax(-mb, 0.0)) *phi_bottom
	        );
		}
// =============================================================
					}
__syncthreads();
	}
}

#define KERNEL_SIDE(FACE) step_FE_TVD_side_##FACE(double *phi_out, double *phi_in, \
double *Up, double *Vp, double *Wp, int Nx, int Ny, int Nz, double A, double V_dt, double nuA_h, \
int BC_##FACE, double phi_##FACE){ \
COMPUTE_INDEX_##FACE \
OFFSETS_##FACE; \
COMPUTE_INDEX_NORMAL_VELOCITY_##FACE \
LOAD_FACE_FLUX_##FACE; \
double mc = 0.0, result = 0.0, aC = 5.0, phiC = phi_in[in_idx]; \
for (int i = 0; i < 5; i++) { \
	mc += mimax(m_f[i],0.0); \
    result += ( mimax(-m_f[i], 0.0) + nuA_h )*phi_in[in_idx+offsets[i]]; \
} \
if (BC_##FACE == inlet){ \
	result += ( -m_f[5] + 2.0*nuA_h )*phi_##FACE; \
	aC += 2.0; \
} \
if (BC_##FACE == outlet) result += ( -m_f[5] )*phiC; \
if (BC_##FACE == periodic) { \
	result += ( mimax(-m_f[5], 0.0) + nuA_h )*phi_in[in_idx+PERIODIC_##FACE]; \
	mc += mimax(m_f[5],0.0); \
} \
phi_out[in_idx] = 1.0/V_dt * (  (V_dt - nuA_h*aC - mc)*phiC + result  ); \
}

#define KERNEL_EDGE(FACE1,FACE2) step_FE_TVD_edge_##FACE1##_##FACE2(double *phi_out, double *phi_in, \
double *Up, double *Vp, double *Wp, int Nx, int Ny, int Nz, double A, double V_dt, double nuA_h, \
int BC_##FACE1, int BC_##FACE2, double phi_##FACE1, double phi_##FACE2){ \
COMPUTE_INDEX_##FACE1##_##FACE2 \
OFFSETS_##FACE1##_##FACE2; \
COMPUTE_INDEX_NORMAL_VELOCITY_##FACE1##_##FACE2; \
LOAD_FACE_FLUX_##FACE1##_##FACE2; \
double mc = 0.0, result = 0.0, aC = 5.0, phiC = phi_in[in_idx]; \
for (int i = 0; i < 4; i++) { \
	mc += mimax(m_f[i],0.0); \
    result += ( mimax(-m_f[i], 0.0) + nuA_h )*phi_in[in_idx+offsets[i]]; \
} \
if (BC_##FACE1 == inlet){ \
	result += ( -m_f[4] + 2.0*nuA_h )*phi_##FACE1; \
	aC += 2.0; \
} \
if (BC_##FACE1 == outlet) result += ( -m_f[4] )*phiC; \
if (BC_##FACE1 == periodic) { \
	result += ( mimax(-m_f[4], 0.0) + nuA_h )*phi_in[in_idx+PERIODIC_##FACE1]; \
	mc += mimax(m_f[4],0.0); \
} \
if (BC_##FACE2 == inlet){ \
	result += ( -m_f[5] + 2.0*nuA_h )*phi_##FACE2; \
	aC += 2.0; \
} \
if (BC_##FACE2 == outlet) result += ( -m_f[5] )*phiC; \
if (BC_##FACE2 == periodic) { \
	result += ( mimax(-m_f[5], 0.0) + nuA_h )*phi_in[in_idx+PERIODIC_##FACE2]; \
	mc += mimax(m_f[5],0.0); \
} \
phi_out[in_idx] = 1.0/V_dt * (  (V_dt - nuA_h*aC - mc)*phiC + result  ); \
}

#define KERNEL_VERTEX(FACE1,FACE2,FACE3) step_FE_TVD_vertex_##FACE1##_##FACE2##_##FACE3(double *phi_out, double *phi_in, \
double *Up, double *Vp, double *Wp, int Nx, int Ny, int Nz, double A, double V_dt, double nuA_h, \
int BC_##FACE1, int BC_##FACE2, int BC_##FACE3, double phi_##FACE1, double phi_##FACE2, double phi_##FACE3){ \
COMPUTE_INDEX_##FACE1##_##FACE2##_##FACE3 \
OFFSETS_##FACE1##_##FACE2##_##FACE3; \
COMPUTE_INDEX_NORMAL_VELOCITY_##FACE1##_##FACE2##_##FACE3; \
LOAD_FACE_FLUX_##FACE1##_##FACE2##_##FACE3; \
double mc = 0.0, result = 0.0, aC = 5.0, phiC = phi_in[in_idx]; \
for (int i = 0; i < 3; i++) { \
	mc += mimax(m_f[i],0.0); \
    result += ( mimax(-m_f[i], 0.0) + nuA_h )*phi_in[in_idx+offsets[i]]; \
} \
if (BC_##FACE1 == inlet){ \
	result += ( -m_f[3] + 2.0*nuA_h )*phi_##FACE1; \
	aC += 2.0; \
} \
if (BC_##FACE1 == outlet) result += ( -m_f[3] )*phiC; \
if (BC_##FACE1 == periodic) { \
	result += ( mimax(-m_f[3], 0.0) + nuA_h )*phi_in[in_idx+PERIODIC_##FACE1]; \
	mc += mimax(m_f[3],0.0); \
} \
if (BC_##FACE2 == inlet){ \
	result += ( -m_f[4] + 2.0*nuA_h )*phi_##FACE2; \
	aC += 2.0; \
} \
if (BC_##FACE2 == outlet) result += ( -m_f[4] )*phiC; \
if (BC_##FACE2 == periodic) { \
	result += ( mimax(-m_f[4], 0.0) + nuA_h )*phi_in[in_idx+PERIODIC_##FACE2]; \
	mc += mimax(m_f[4],0.0); \
} \
if (BC_##FACE3 == inlet){ \
	result += ( -m_f[5] + 2.0*nuA_h )*phi_##FACE3; \
	aC += 2.0; \
} \
if (BC_##FACE3 == outlet) result += ( -m_f[5] )*phiC; \
if (BC_##FACE3 == periodic) { \
	result += ( mimax(-m_f[5], 0.0) + nuA_h )*phi_in[in_idx+PERIODIC_##FACE3]; \
	mc += mimax(m_f[5],0.0); \
} \
phi_out[in_idx] = 1.0/V_dt * (  (V_dt - nuA_h*aC - mc)*phiC + result  ); \
}


// kernel declaration for all domain except interior
__global__ void KERNEL_SIDE(BOTTOM)
__global__ void KERNEL_SIDE(TOP)
__global__ void KERNEL_SIDE(NORTH)
__global__ void KERNEL_SIDE(SOUTH)
__global__ void KERNEL_SIDE(WEST)
__global__ void KERNEL_SIDE(EAST)
// edge y
__global__ void KERNEL_EDGE(WEST,BOTTOM)
__global__ void KERNEL_EDGE(WEST,TOP)
__global__ void KERNEL_EDGE(EAST,BOTTOM)
__global__ void KERNEL_EDGE(EAST,TOP)
// edge z
__global__ void KERNEL_EDGE(WEST,SOUTH)
__global__ void KERNEL_EDGE(WEST,NORTH)
__global__ void KERNEL_EDGE(EAST,SOUTH)
__global__ void KERNEL_EDGE(EAST,NORTH)
// edge x
__global__ void KERNEL_EDGE(SOUTH,BOTTOM)
__global__ void KERNEL_EDGE(SOUTH,TOP)
__global__ void KERNEL_EDGE(NORTH,BOTTOM)
__global__ void KERNEL_EDGE(NORTH,TOP)

__global__ void KERNEL_VERTEX(WEST,SOUTH,BOTTOM)
__global__ void KERNEL_VERTEX(WEST,SOUTH,TOP)
__global__ void KERNEL_VERTEX(WEST,NORTH,BOTTOM)
__global__ void KERNEL_VERTEX(WEST,NORTH,TOP)
__global__ void KERNEL_VERTEX(EAST,SOUTH,BOTTOM)
__global__ void KERNEL_VERTEX(EAST,SOUTH,TOP)
__global__ void KERNEL_VERTEX(EAST,NORTH,BOTTOM)
__global__ void KERNEL_VERTEX(EAST,NORTH,TOP)

#define LAUNCH_KERNEL_SIDE(FACE) \
    step_FE_TVD_side_##FACE<<<GRIDBLOCK_##FACE>>>(phi_out, phi_in, U, V, W, Nx, Ny, Nz, A, V_dt, nuA_h, BC_##FACE, phi_##FACE)
#define LAUNCH_KERNEL_EDGE(FACE1,FACE2) \
    step_FE_TVD_edge_##FACE1##_##FACE2<<<GRIDBLOCK_##FACE1##_##FACE2>>>(phi_out, phi_in, U, V, W, Nx, Ny, Nz, A, V_dt, nuA_h, BC_##FACE1,BC_##FACE2, phi_##FACE1,phi_##FACE2)
#define LAUNCH_KERNEL_VERTEX(FACE1,FACE2,FACE3) \
    step_FE_TVD_vertex_##FACE1##_##FACE2##_##FACE3<<<1,1>>>(phi_out, phi_in, U, V, W, Nx, Ny, Nz, A, V_dt, nuA_h, BC_##FACE1,BC_##FACE2,BC_##FACE3, phi_##FACE1,phi_##FACE2,phi_##FACE3)
#define LAUNCH_KERNEL_INT \
    step_FE_TVD_int<<<GRIDBLOCK_BOTTOM>>>(phi_out, phi_in, U, V, W, Nx, Ny, Nz, A, V_dt, nuA_h)

void step_FE_TVD(double *phi_out, double *phi_in, double *U, double *V, double *W, int Nx, int Ny, int Nz, double A, double V_dt, double nuA_h, int BC_WEST, int BC_EAST, int BC_SOUTH, int BC_NORTH, int BC_BOTTOM, int BC_TOP, double phi_WEST, double phi_EAST, double phi_SOUTH, double phi_NORTH, double phi_BOTTOM, double phi_TOP, dim3 grid, dim3 block){
LAUNCH_KERNEL_INT;
LAUNCH_KERNEL_SIDE(BOTTOM);
LAUNCH_KERNEL_SIDE(TOP);
LAUNCH_KERNEL_SIDE(NORTH);
LAUNCH_KERNEL_SIDE(SOUTH);
LAUNCH_KERNEL_SIDE(WEST);
LAUNCH_KERNEL_SIDE(EAST);

LAUNCH_KERNEL_EDGE(SOUTH,BOTTOM);
LAUNCH_KERNEL_EDGE(SOUTH,TOP);
LAUNCH_KERNEL_EDGE(NORTH,BOTTOM);
LAUNCH_KERNEL_EDGE(NORTH,TOP);

LAUNCH_KERNEL_EDGE(WEST,SOUTH);
LAUNCH_KERNEL_EDGE(WEST,NORTH);
LAUNCH_KERNEL_EDGE(EAST,SOUTH);
LAUNCH_KERNEL_EDGE(EAST,NORTH);

LAUNCH_KERNEL_EDGE(WEST,BOTTOM);
LAUNCH_KERNEL_EDGE(WEST,TOP);
LAUNCH_KERNEL_EDGE(EAST,BOTTOM);
LAUNCH_KERNEL_EDGE(EAST,TOP);

LAUNCH_KERNEL_VERTEX(WEST,SOUTH,BOTTOM);
LAUNCH_KERNEL_VERTEX(WEST,SOUTH,TOP);
LAUNCH_KERNEL_VERTEX(WEST,NORTH,BOTTOM);
LAUNCH_KERNEL_VERTEX(WEST,NORTH,TOP);
LAUNCH_KERNEL_VERTEX(EAST,SOUTH,BOTTOM);
LAUNCH_KERNEL_VERTEX(EAST,SOUTH,TOP);
LAUNCH_KERNEL_VERTEX(EAST,NORTH,BOTTOM);
LAUNCH_KERNEL_VERTEX(EAST,NORTH,TOP);
cudaDeviceSynchronize();
}

#undef BLOCK_Nx
#undef BLOCK_Ny
```

# transf_operator_3D.cu

```cu
/**
* @file transf_operator_3D.cu
* @brief Constant piecewise transfer operator and homogenization technique
* for the conductivity coefficient of the coarse grid operator A2h,
* used in the cell-centered multigrid method.
*
* @author Lucas Bessone (contact: lcbessone@gmail.com)
*
* @copyright This file is part of the EU-PAR software.
*            Copyright (C) 2025 Lucas Bessone
*
* @license This program is free software: you can redistribute it and/or modify
*          it under the terms of the GNU General Public License as published by
*          the Free Software Foundation, either version 3 of the License, or
*          (at your option) any later version.
*
*          This program is distributed in the hope that it will be useful,
*          but WITHOUT ANY WARRANTY; without even the implied warranty of
*          MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*          GNU General Public License for more details.
*
*          You should have received a copy of the GNU General Public License
*          along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/
#include <cuda_runtime_api.h>
#define gridXY dim3(grid.x,grid.y)
#define blockXY dim3(block.x,block.y)
#define gridXZ dim3(grid.x,grid.z)
#define blockXZ dim3(block.x,block.z)
#define gridYZ dim3(grid.y,grid.z)
#define blockYZ dim3(block.y,block.z)

//#######################################################################
//  Compact homogenization of K-tensor via geometric mean of eight cells
//  Only one kernel is needed
//#######################################################################
__global__ void CompactHomogenizationKtensor(double *phiCoarse, const double *phiFine, int NX, int NY, int NZ){
	int IX = threadIdx.x + blockIdx.x*blockDim.x;
	int IY = threadIdx.y + blockIdx.y*blockDim.y;
	if (IX >= NX || IY >= NY) return;
	int STRIDE = NX*NY;
	int IN_IDX = IX + IY*NX;

	int ix = 2*IX, iy = 2*IY, iz;
	int Nx = 2*NX, Ny = 2*NY;
	int in_idx = ix + iy*Nx;
	int stride = Nx*Ny;
	double result;
	for(int IZ=0; IZ<NZ; ++IZ){
		iz = 2*IZ;
		IN_IDX = IX + IY*NX + IZ*STRIDE;
		in_idx = ix + iy*Nx + iz*stride;
		result = 0.0;
		result += ( log(phiFine[in_idx]) + log(phiFine[in_idx+1])
			      + log(phiFine[in_idx+Nx]) + log(phiFine[in_idx+1+Nx]) );

		result += ( log(phiFine[in_idx+stride]) + log(phiFine[in_idx+1+stride])
			      + log(phiFine[in_idx+Nx+stride]) + log(phiFine[in_idx+1+Nx+stride]) );

		phiCoarse[IN_IDX] = exp(result/8.0);
	}
}

void HomogenizationPermeability(double *phiCoarse, const double *phiFine, int Nx, int Ny, int Nz, dim3 grid, dim3 block){
	CompactHomogenizationKtensor<<<gridXY,blockXY>>>(phiCoarse,phiFine,Nx,Ny,Nz);
	cudaDeviceSynchronize();
}



//#######################################################################
//  Routine for restriction (R operator), used in MG-Cycle (CCMG)
//  Linear restriction: transfers a piecewise constant function
//  from a fine grid to a coarse grid
//#######################################################################
__global__ void restriction_linear3D(double *phiCoarse, const double *phiFine, int NX, int NY, int NZ){
	int IX = threadIdx.x + blockIdx.x*blockDim.x;
	int IY = threadIdx.y + blockIdx.y*blockDim.y;
	if (IX >= NX || IY >= NY) return;
	int STRIDE = NX*NY;
	int IN_IDX = IX + IY*NX;

	int ix = 2*IX, iy = 2*IY, iz;
	int Nx = 2*NX, Ny = 2*NY;
	int in_idx = ix + iy*Nx;
	int stride = Nx*Ny;
	double result;
	for(int IZ=0; IZ<NZ; ++IZ){
		iz = 2*IZ;
		IN_IDX = IX + IY*NX + IZ*STRIDE;
		in_idx = ix + iy*Nx + iz*stride;
		result = 0.0;
		result += phiFine[in_idx] + phiFine[in_idx+1]
			    + phiFine[in_idx+Nx] + phiFine[in_idx+1+Nx];

		result += phiFine[in_idx+stride] + phiFine[in_idx+1+stride]
			    + phiFine[in_idx+Nx+stride] + phiFine[in_idx+1+Nx+stride];

		phiCoarse[IN_IDX] = result/8.0;
	}
}

void restriction(double *phiCoarse, const double *phiFine, int Nx, int Ny, int Nz, dim3 grid, dim3 block){
	restriction_linear3D<<<gridXY,blockXY>>>(phiCoarse,phiFine,Nx,Ny,Nz);
	cudaDeviceSynchronize();
}

//#######################################################################
//  Routine for prolongation (P operator), used in MG-Cycle (CCMG)
//  Linear prolongation: transfers a piecewise constant function
//  from a coarse grid to a fine grid
//#######################################################################
__global__ void prolongation_interior_linear3D(double *phiFine, const double *phiCoarse, int Nx, int Ny, int Nz){
	int ix = threadIdx.x + blockIdx.x*blockDim.x;
	int iy = threadIdx.y + blockIdx.y*blockDim.y;
	if (ix >= Nx-2 || iy >= Ny-2) return;
	int IX = ix/2, IY = iy/2;
	int NX = Nx/2, NY = Ny/2;
	int in_idx = (ix+1) + (iy+1)*Nx;
	int IN_IDX = (IX) + (IY)*NX;
	int fx = ix%2; // flag x
	int fy = iy%2; // flag y
	int fz; // flag z
	int IZ;
	for(int iz=0; iz<Nz-2; iz++){
		fz = iz%2;
		IZ = iz/2;
		in_idx = (ix+1) + (iy+1)*Nx + (iz+1)*Nx*Ny;
		IN_IDX = (IX) + (IY)*NX + (IZ)*NX*NY;

		phiFine[in_idx] += phiCoarse[IN_IDX+fx+fy*NX+fz*NX*NY];
	}
}

__global__ void prolongation_side_bottom(double *phiFine, const double *phiCoarse, int Nx, int Ny, int Nz){
	int ix = threadIdx.x + blockIdx.x*blockDim.x;
	int iy = threadIdx.y + blockIdx.y*blockDim.y;
	if (ix >= Nx-2 || iy >= Ny-2) return;
	int iz = 0;
	int IX = ix/2, IY = iy/2, IZ = iz/2;
	int NX = Nx/2, NY = Ny/2;
	int in_idx = (ix+1) + (iy+1)*Nx + (iz)*Nx*Ny;
	int IN_IDX = (IX) + (IY)*NX + (IZ)*NX*NY;
	int fx = ix%2; // flag x
	int fy = iy%2; // flag y
	phiFine[in_idx] += phiCoarse[IN_IDX+fx+fy*NX];
}

__global__ void prolongation_side_top(double *phiFine, const double *phiCoarse, int Nx, int Ny, int Nz){
	int ix = threadIdx.x + blockIdx.x*blockDim.x;
	int iy = threadIdx.y + blockIdx.y*blockDim.y;
	if (ix >= Nx-2 || iy >= Ny-2) return;
	int iz = Nz-1;
	int IX = ix/2, IY = iy/2, IZ = iz/2;
	int NX = Nx/2, NY = Ny/2;
	int in_idx = (ix+1) + (iy+1)*Nx + (iz)*Nx*Ny;
	int IN_IDX = (IX) + (IY)*NX + (IZ)*NX*NY;
	int fx = ix%2; // flag x
	int fy = iy%2; // flag y
	phiFine[in_idx] += phiCoarse[IN_IDX+fx+fy*NX];
}

__global__ void prolongation_side_south(double *phiFine, const double *phiCoarse, int Nx, int Ny, int Nz){
	int ix = threadIdx.x + blockIdx.x*blockDim.x;
	int iz = threadIdx.y + blockIdx.y*blockDim.y;
	if (ix >= Nx-2 || iz >= Nz-2) return;
	int iy = 0;
	int IX = ix/2, IY = iy/2, IZ = iz/2;
	int NX = Nx/2, NY = Ny/2;
	int in_idx = (ix+1) + (iy)*Nx + (iz+1)*Nx*Ny;
	int IN_IDX = (IX) + (IY)*NX + (IZ)*NX*NY;
	int fx = ix%2; // flag x
	int fz = iz%2; // flag z
	phiFine[in_idx] += phiCoarse[IN_IDX+fx+fz*NX*NY];
}

__global__ void prolongation_side_north(double *phiFine, const double *phiCoarse, int Nx, int Ny, int Nz){
	int ix = threadIdx.x + blockIdx.x*blockDim.x;
	int iz = threadIdx.y + blockIdx.y*blockDim.y;
	if (ix >= Nx-2 || iz >= Nz-2) return;
	int iy = Ny-1;
	int IX = ix/2, IY = iy/2, IZ = iz/2;
	int NX = Nx/2, NY = Ny/2;
	int in_idx = (ix+1) + (iy)*Nx + (iz+1)*Nx*Ny;
	int IN_IDX = (IX) + (IY)*NX + (IZ)*NX*NY;
	int fx = ix%2; // flag x
	int fz = iz%2; // flag z
	phiFine[in_idx] += phiCoarse[IN_IDX+fx+fz*NX*NY];
}

__global__ void prolongation_side_west(double *phiFine, const double *phiCoarse, int Nx, int Ny, int Nz){
	int iy = threadIdx.x + blockIdx.x*blockDim.x;
	int iz = threadIdx.y + blockIdx.y*blockDim.y;
	if (iy >= Ny-2 || iz >= Nz-2) return;
	int ix = 0;
	int IX = ix/2, IY = iy/2, IZ = iz/2;
	int NX = Nx/2, NY = Ny/2;
	int in_idx = (ix) + (iy+1)*Nx + (iz+1)*Nx*Ny;
	int IN_IDX = (IX) + (IY)*NX + (IZ)*NX*NY;
	int fy = iy%2; // flag y
	int fz = iz%2; // flag z
	phiFine[in_idx] += phiCoarse[IN_IDX+fy*NX+fz*NX*NY];
}

__global__ void prolongation_side_east(double *phiFine, const double *phiCoarse, int Nx, int Ny, int Nz){
	int iy = threadIdx.x + blockIdx.x*blockDim.x;
	int iz = threadIdx.y + blockIdx.y*blockDim.y;
	if (iy >= Ny-2 || iz >= Nz-2) return;
	int ix = Nx-1;
	int IX = ix/2, IY = iy/2, IZ = iz/2;
	int NX = Nx/2, NY = Ny/2;
	int in_idx = (ix) + (iy+1)*Nx + (iz+1)*Nx*Ny;
	int IN_IDX = (IX) + (IY)*NX + (IZ)*NX*NY;
	int fy = iy%2; // flag y
	int fz = iz%2; // flag z
	phiFine[in_idx] += phiCoarse[IN_IDX+fy*NX+fz*NX*NY];
}

__global__ void prolongation_edge_X_South_Bottom(double *phiFine, const double *phiCoarse, int Nx, int Ny, int Nz){
	int ix = threadIdx.x + blockIdx.x*blockDim.x;
	if (ix >= Nx-2) return;
	int iy = 0;
	int iz = 0;
	int IX = ix/2, IY = iy/2, IZ = iz/2;
	int NX = Nx/2, NY = Ny/2;
	int in_idx = (ix+1) + (iy)*Nx + (iz)*Nx*Ny;
	int IN_IDX = (IX) + (IY)*NX + (IZ)*NX*NY;
	int fx = ix%2; // flag x
	phiFine[in_idx] += phiCoarse[IN_IDX+fx];
}

__global__ void prolongation_edge_X_South_Top(double *phiFine, const double *phiCoarse, int Nx, int Ny, int Nz){
	int ix = threadIdx.x + blockIdx.x*blockDim.x;
	if (ix >= Nx-2) return;
	int iy = 0;
	int iz = Nz-1;
	int IX = ix/2, IY = iy/2, IZ = iz/2;
	int NX = Nx/2, NY = Ny/2;
	int in_idx = (ix+1) + (iy)*Nx + (iz)*Nx*Ny;
	int IN_IDX = (IX) + (IY)*NX + (IZ)*NX*NY;
	int fx = ix%2; // flag x
	phiFine[in_idx] += phiCoarse[IN_IDX+fx];
}

__global__ void prolongation_edge_X_North_Bottom(double *phiFine, const double *phiCoarse, int Nx, int Ny, int Nz){
	int ix = threadIdx.x + blockIdx.x*blockDim.x;
	if (ix >= Nx-2) return;
	int iy = Ny-1;
	int iz = 0;
	int IX = ix/2, IY = iy/2, IZ = iz/2;
	int NX = Nx/2, NY = Ny/2;
	int in_idx = (ix+1) + (iy)*Nx + (iz)*Nx*Ny;
	int IN_IDX = (IX) + (IY)*NX + (IZ)*NX*NY;
	int fx = ix%2; // flag x
	phiFine[in_idx] += phiCoarse[IN_IDX+fx];
}

__global__ void prolongation_edge_X_North_Top(double *phiFine, const double *phiCoarse, int Nx, int Ny, int Nz){
	int ix = threadIdx.x + blockIdx.x*blockDim.x;
	if (ix >= Nx-2) return;
	int iy = Ny-1;
	int iz = Nz-1;
	int IX = ix/2, IY = iy/2, IZ = iz/2;
	int NX = Nx/2, NY = Ny/2;
	int in_idx = (ix+1) + (iy)*Nx + (iz)*Nx*Ny;
	int IN_IDX = (IX) + (IY)*NX + (IZ)*NX*NY;
	int fx = ix%2; // flag x
	phiFine[in_idx] += phiCoarse[IN_IDX+fx];
}

__global__ void prolongation_edge_Z_South_West(double *phiFine, const double *phiCoarse, int Nx, int Ny, int Nz){
	int iz = threadIdx.x + blockIdx.x*blockDim.x;
	if (iz >= Nz-2) return;
	int ix = 0;
	int iy = 0;
	int IX = ix/2, IY = iy/2, IZ = iz/2;
	int NX = Nx/2, NY = Ny/2;
	int in_idx = (ix) + (iy)*Nx + (iz+1)*Nx*Ny;
	int IN_IDX = (IX) + (IY)*NX + (IZ)*NX*NY;
	int fz = iz%2; // flag z
	phiFine[in_idx] += phiCoarse[IN_IDX+fz*NX*NY];
}

__global__ void prolongation_edge_Z_South_East(double *phiFine, const double *phiCoarse, int Nx, int Ny, int Nz){
	int iz = threadIdx.x + blockIdx.x*blockDim.x;
	if (iz >= Nz-2) return;
	int ix = Nx-1;
	int iy = 0;
	int IX = ix/2, IY = iy/2, IZ = iz/2;
	int NX = Nx/2, NY = Ny/2;
	int in_idx = (ix) + (iy)*Nx + (iz+1)*Nx*Ny;
	int IN_IDX = (IX) + (IY)*NX + (IZ)*NX*NY;
	int fz = iz%2; // flag z
	phiFine[in_idx] += phiCoarse[IN_IDX+fz*NX*NY];
}

__global__ void prolongation_edge_Z_North_West(double *phiFine, const double *phiCoarse, int Nx, int Ny, int Nz){
	int iz = threadIdx.x + blockIdx.x*blockDim.x;
	if (iz >= Nz-2) return;
	int ix = 0;
	int iy = Ny-1;
	int IX = ix/2, IY = iy/2, IZ = iz/2;
	int NX = Nx/2, NY = Ny/2;
	int in_idx = (ix) + (iy)*Nx + (iz+1)*Nx*Ny;
	int IN_IDX = (IX) + (IY)*NX + (IZ)*NX*NY;
	int fz = iz%2; // flag z
	phiFine[in_idx] += phiCoarse[IN_IDX+fz*NX*NY];
}

__global__ void prolongation_edge_Z_North_East(double *phiFine, const double *phiCoarse, int Nx, int Ny, int Nz){
	int iz = threadIdx.x + blockIdx.x*blockDim.x;
	if (iz >= Nz-2) return;
	int ix = Nx-1;
	int iy = Ny-1;
	int IX = ix/2, IY = iy/2, IZ = iz/2;
	int NX = Nx/2, NY = Ny/2;
	int in_idx = (ix) + (iy)*Nx + (iz+1)*Nx*Ny;
	int IN_IDX = (IX) + (IY)*NX + (IZ)*NX*NY;
	int fz = iz%2; // flag z
	phiFine[in_idx] += phiCoarse[IN_IDX+fz*NX*NY];
}

__global__ void prolongation_edge_Y_West_Bottom(double *phiFine, const double *phiCoarse, int Nx, int Ny, int Nz){
	int iy = threadIdx.x + blockIdx.x*blockDim.x;
	if (iy >= Ny-2) return;
	int ix = 0;
	int iz = 0;
	int IX = ix/2, IY = iy/2, IZ = iz/2;
	int NX = Nx/2, NY = Ny/2;
	int in_idx = (ix) + (iy+1)*Nx + (iz)*Nx*Ny;
	int IN_IDX = (IX) + (IY)*NX + (IZ)*NX*NY;
	int fy = iy%2; // flag y
	phiFine[in_idx] += phiCoarse[IN_IDX+fy*NX];
}

__global__ void prolongation_edge_Y_West_Top(double *phiFine, const double *phiCoarse, int Nx, int Ny, int Nz){
	int iy = threadIdx.x + blockIdx.x*blockDim.x;
	if (iy >= Ny-2) return;
	int ix = 0;
	int iz = Nz-1;
	int IX = ix/2, IY = iy/2, IZ = iz/2;
	int NX = Nx/2, NY = Ny/2;
	int in_idx = (ix) + (iy+1)*Nx + (iz)*Nx*Ny;
	int IN_IDX = (IX) + (IY)*NX + (IZ)*NX*NY;
	int fy = iy%2; // flag y
	phiFine[in_idx] += phiCoarse[IN_IDX+fy*NX];
}

__global__ void prolongation_edge_Y_East_Bottom(double *phiFine, const double *phiCoarse, int Nx, int Ny, int Nz){
	int iy = threadIdx.x + blockIdx.x*blockDim.x;
	if (iy >= Ny-2) return;
	int ix = Nx-1;
	int iz = 0;
	int IX = ix/2, IY = iy/2, IZ = iz/2;
	int NX = Nx/2, NY = Ny/2;
	int in_idx = (ix) + (iy+1)*Nx + (iz)*Nx*Ny;
	int IN_IDX = (IX) + (IY)*NX + (IZ)*NX*NY;
	int fy = iy%2; // flag y
	phiFine[in_idx] += phiCoarse[IN_IDX+fy*NX];
}

__global__ void prolongation_edge_Y_East_Top(double *phiFine, const double *phiCoarse, int Nx, int Ny, int Nz){
	int iy = threadIdx.x + blockIdx.x*blockDim.x;
	if (iy >= Ny-2) return;
	int ix = Nx-1;
	int iz = Nz-1;
	int IX = ix/2, IY = iy/2, IZ = iz/2;
	int NX = Nx/2, NY = Ny/2;
	int in_idx = (ix) + (iy+1)*Nx + (iz)*Nx*Ny;
	int IN_IDX = (IX) + (IY)*NX + (IZ)*NX*NY;
	int fy = iy%2; // flag y
	phiFine[in_idx] += phiCoarse[IN_IDX+fy*NX];
}

__global__ void prolongation_vertex_SWB(double *phiFine, const double *phiCoarse, int Nx, int Ny, int Nz){
	int ix = 0;
	int iy = 0;
	int iz = 0;
	int IX = ix/2, IY = iy/2, IZ = iz/2;
	int NX = Nx/2, NY = Ny/2;
	int in_idx = (ix) + (iy)*Nx + (iz)*Nx*Ny;
	int IN_IDX = (IX) + (IY)*NX + (IZ)*NX*NY;
	phiFine[in_idx] += phiCoarse[IN_IDX];
}

__global__ void prolongation_vertex_SWT(double *phiFine, const double *phiCoarse, int Nx, int Ny, int Nz){
	int ix = 0;
	int iy = 0;
	int iz = Nz-1;
	int IX = ix/2, IY = iy/2, IZ = iz/2;
	int NX = Nx/2, NY = Ny/2;
	int in_idx = (ix) + (iy)*Nx + (iz)*Nx*Ny;
	int IN_IDX = (IX) + (IY)*NX + (IZ)*NX*NY;
	phiFine[in_idx] += phiCoarse[IN_IDX];
}

__global__ void prolongation_vertex_SEB(double *phiFine, const double *phiCoarse, int Nx, int Ny, int Nz){
	int ix = Nx-1;
	int iy = 0;
	int iz = 0;
	int IX = ix/2, IY = iy/2, IZ = iz/2;
	int NX = Nx/2, NY = Ny/2;
	int in_idx = (ix) + (iy)*Nx + (iz)*Nx*Ny;
	int IN_IDX = (IX) + (IY)*NX + (IZ)*NX*NY;
	phiFine[in_idx] += phiCoarse[IN_IDX];
}

__global__ void prolongation_vertex_SET(double *phiFine, const double *phiCoarse, int Nx, int Ny, int Nz){
	int ix = Nx-1;
	int iy = 0;
	int iz = Nz-1;
	int IX = ix/2, IY = iy/2, IZ = iz/2;
	int NX = Nx/2, NY = Ny/2;
	int in_idx = (ix) + (iy)*Nx + (iz)*Nx*Ny;
	int IN_IDX = (IX) + (IY)*NX + (IZ)*NX*NY;
	phiFine[in_idx] += phiCoarse[IN_IDX];
}

__global__ void prolongation_vertex_NEB(double *phiFine, const double *phiCoarse, int Nx, int Ny, int Nz){
	int ix = Nx-1;
	int iy = Ny-1;
	int iz = 0;
	int IX = ix/2, IY = iy/2, IZ = iz/2;
	int NX = Nx/2, NY = Ny/2;
	int in_idx = (ix) + (iy)*Nx + (iz)*Nx*Ny;
	int IN_IDX = (IX) + (IY)*NX + (IZ)*NX*NY;
	phiFine[in_idx] += phiCoarse[IN_IDX];
}

__global__ void prolongation_vertex_NET(double *phiFine, const double *phiCoarse, int Nx, int Ny, int Nz){
	int ix = Nx-1;
	int iy = Ny-1;
	int iz = Nz-1;
	int IX = ix/2, IY = iy/2, IZ = iz/2;
	int NX = Nx/2, NY = Ny/2;
	int in_idx = (ix) + (iy)*Nx + (iz)*Nx*Ny;
	int IN_IDX = (IX) + (IY)*NX + (IZ)*NX*NY;
	phiFine[in_idx] += phiCoarse[IN_IDX];
}

__global__ void prolongation_vertex_NWB(double *phiFine, const double *phiCoarse, int Nx, int Ny, int Nz){
	int ix = 0;
	int iy = Ny-1;
	int iz = 0;
	int IX = ix/2, IY = iy/2, IZ = iz/2;
	int NX = Nx/2, NY = Ny/2;
	int in_idx = (ix) + (iy)*Nx + (iz)*Nx*Ny;
	int IN_IDX = (IX) + (IY)*NX + (IZ)*NX*NY;
	phiFine[in_idx] += phiCoarse[IN_IDX];
}

__global__ void prolongation_vertex_NWT(double *phiFine, const double *phiCoarse, int Nx, int Ny, int Nz){
	int ix = 0;
	int iy = Ny-1;
	int iz = Nz-1;
	int IX = ix/2, IY = iy/2, IZ = iz/2;
	int NX = Nx/2, NY = Ny/2;
	int in_idx = (ix) + (iy)*Nx + (iz)*Nx*Ny;
	int IN_IDX = (IX) + (IY)*NX + (IZ)*NX*NY;
	phiFine[in_idx] += phiCoarse[IN_IDX];
}

void prolongation(double *phiFine, const double *phiCoarse, int Nx, int Ny, int Nz,
	dim3 grid, dim3 block){
	prolongation_interior_linear3D<<<gridXY,blockXY>>>(phiFine,phiCoarse,Nx,Ny,Nz)	;
	prolongation_side_bottom<<<gridXY,blockXY>>>(phiFine,phiCoarse,Nx,Ny,Nz);
	prolongation_side_top<<<gridXY,blockXY>>>(phiFine,phiCoarse,Nx,Ny,Nz);
	prolongation_side_south<<<gridXZ,blockXZ>>>(phiFine,phiCoarse,Nx,Ny,Nz);
	prolongation_side_north<<<gridXZ,blockXZ>>>(phiFine,phiCoarse,Nx,Ny,Nz);
	prolongation_side_west<<<gridYZ,blockYZ>>>(phiFine,phiCoarse,Nx,Ny,Nz);
	prolongation_side_east<<<gridYZ,blockYZ>>>(phiFine,phiCoarse,Nx,Ny,Nz);
	prolongation_edge_X_South_Bottom<<<grid.x,block.x>>>(phiFine,phiCoarse,Nx,Ny,Nz);
	prolongation_edge_X_South_Top<<<grid.x,block.x>>>(phiFine,phiCoarse,Nx,Ny,Nz);
	prolongation_edge_X_North_Bottom<<<grid.x,block.x>>>(phiFine,phiCoarse,Nx,Ny,Nz);
	prolongation_edge_X_North_Top<<<grid.x,block.x>>>(phiFine,phiCoarse,Nx,Ny,Nz);

	prolongation_edge_Z_South_West<<<grid.z,block.z>>>(phiFine,phiCoarse,Nx,Ny,Nz);
	prolongation_edge_Z_South_East<<<grid.z,block.z>>>(phiFine,phiCoarse,Nx,Ny,Nz);
	prolongation_edge_Z_North_West<<<grid.z,block.z>>>(phiFine,phiCoarse,Nx,Ny,Nz);
	prolongation_edge_Z_North_East<<<grid.z,block.z>>>(phiFine,phiCoarse,Nx,Ny,Nz);

	prolongation_edge_Y_West_Bottom<<<grid.y,block.y>>>(phiFine,phiCoarse,Nx,Ny,Nz);
	prolongation_edge_Y_West_Top<<<grid.y,block.y>>>(phiFine,phiCoarse,Nx,Ny,Nz);
	prolongation_edge_Y_East_Bottom<<<grid.y,block.y>>>(phiFine,phiCoarse,Nx,Ny,Nz);
	prolongation_edge_Y_East_Top<<<grid.y,block.y>>>(phiFine,phiCoarse,Nx,Ny,Nz);
	prolongation_vertex_SWB<<<1,1>>>(phiFine,phiCoarse,Nx,Ny,Nz);
	prolongation_vertex_SWT<<<1,1>>>(phiFine,phiCoarse,Nx,Ny,Nz);
	prolongation_vertex_SEB<<<1,1>>>(phiFine,phiCoarse,Nx,Ny,Nz);
	prolongation_vertex_SET<<<1,1>>>(phiFine,phiCoarse,Nx,Ny,Nz);
	prolongation_vertex_NEB<<<1,1>>>(phiFine,phiCoarse,Nx,Ny,Nz);
	prolongation_vertex_NET<<<1,1>>>(phiFine,phiCoarse,Nx,Ny,Nz);
	prolongation_vertex_NWB<<<1,1>>>(phiFine,phiCoarse,Nx,Ny,Nz);
	prolongation_vertex_NWT<<<1,1>>>(phiFine,phiCoarse,Nx,Ny,Nz);
	cudaDeviceSynchronize();
}
```

# up_residual_3D.cu

```cu
/**
* @file up_residual_3D.cu
* @brief Updates the residual as rk = b - A_h * xk.
*        Here, A_h is the matrix operator at the cell size level h.
*        The matrix coefficients originate from a Poisson-like equation with variable coefficients.
*        This routine is used in the cell-centered multigrid method.
*
*
* @author Lucas Bessone (contact: lcbessone@gmail.com)
*
* @copyright This file is part of the EU-PAR software.
*            Copyright (C) 2025 Lucas Bessone
*
* @license This program is free software: you can redistribute it and/or modify
*          it under the terms of the GNU General Public License as published by
*          the Free Software Foundation, either version 3 of the License, or
*          (at your option) any later version.
*
*          This program is distributed in the hope that it will be useful,
*          but WITHOUT ANY WARRANTY; without even the implied warranty of
*          MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*          GNU General Public License for more details.
*
*          You should have received a copy of the GNU General Public License
*          along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/
#include <cuda_runtime_api.h>
#define neumann 0
#define periodic 1
#define dirichlet 2
#define gridXY dim3(grid.x,grid.y)
#define blockXY dim3(block.x,block.y)
#define gridXZ dim3(grid.x,grid.z)
#define blockXZ dim3(block.x,block.z)
#define gridYZ dim3(grid.y,grid.z)
#define blockYZ dim3(block.y,block.z)

// #############################################################
// #####   routine: UPDATE RESIDUAL    ########################
// #####   rk_1 = b - A_l*xk_1    ##############################
// #############################################################
__global__ void update_int(double *r_out, double *h_in, const double *rhs, const double *K, double dxdx, int Nx, int Ny, int Nz){
	int ix = threadIdx.x + blockIdx.x*blockDim.x;
	int iy = threadIdx.y + blockIdx.y*blockDim.y;
	if (ix >= Nx-2 || iy >= Ny-2) return;
	int stride = Nx*Ny;
	double result;
	int in_idx, out_idx;
	in_idx = (ix+1) + (iy+1)*Nx;
	double h_top, h_bottom, h_current;
	double K_top, K_bottom, K_current;
	h_current = h_in[in_idx];
	K_current = K[in_idx];
	out_idx = in_idx;
	in_idx += stride;
	h_top = h_in[in_idx];
	K_top = K[in_idx];
	in_idx += stride;

	for(int iz=1; iz<Nz-1; ++iz){
		h_bottom = h_current;
		h_current = h_top;
		h_top = h_in[in_idx];
		K_bottom = K_current;
		K_current = K_top;
		K_top = K[in_idx];
		in_idx += stride;
		out_idx += stride;
		result=0.0;
		result -= 2.0*(h_current - h_in[out_idx+1 ]) /  (1.0/K_current  +  1.0/K[out_idx+1 ]);
		result -= 2.0*(h_current - h_in[out_idx+Nx]) /  (1.0/K_current  +  1.0/K[out_idx+Nx]);
		result -= 2.0*(h_current - h_in[out_idx-1 ]) /  (1.0/K_current  +  1.0/K[out_idx-1 ]);
		result -= 2.0*(h_current - h_in[out_idx-Nx]) /  (1.0/K_current  +  1.0/K[out_idx-Nx ]);
		result -= 2.0*(h_current - h_top           ) /  (1.0/K_current  +  1.0/K_top         );
		result -= 2.0*(h_current - h_bottom        ) /  (1.0/K_current  +  1.0/K_bottom      );
		r_out[out_idx] = rhs[out_idx] - result/dxdx;
	}
}

__global__ void update_side_bottom(double *r_out, double *h_in, const double *rhs, const double *K, double dxdx, int Nx, int Ny, int Nz, int BCtype){
	int ix = threadIdx.x + blockIdx.x*blockDim.x;
	int iy = threadIdx.y + blockIdx.y*blockDim.y;
	if (ix >= Nx-2 || iy >= Ny-2) return;
	int stride = Nx*Ny;
	double result;
	int in_idx;
	int iz=0;
	in_idx = (ix+1) + (iy+1)*Nx + iz*stride;
	double HC=h_in[in_idx], KC=K[in_idx];
	result=0.0;
	result -= 2.0*(HC - h_in[in_idx+1 ])     /  (1.0/KC  +  1.0/K[in_idx+1 ]);
	result -= 2.0*(HC - h_in[in_idx+Nx])     /  (1.0/KC  +  1.0/K[in_idx+Nx]);
	result -= 2.0*(HC - h_in[in_idx-1 ])     /  (1.0/KC  +  1.0/K[in_idx-1 ]);
	result -= 2.0*(HC - h_in[in_idx-Nx])     /  (1.0/KC  +  1.0/K[in_idx-Nx]);
	result -= 2.0*(HC - h_in[in_idx+stride]) /  (1.0/KC  +  1.0/K[in_idx+stride ]);
	if(BCtype==periodic) result -= 2.0*(HC - h_in[in_idx+(Nz-1)*stride]) /  (1.0/KC  +  1.0/K[in_idx+(Nz-1)*stride ]);
	if(BCtype==dirichlet) result -= 2.0*HC*KC;
	r_out[in_idx] = rhs[in_idx] - result/dxdx;
}

__global__ void update_side_top(double *r_out, double *h_in, const double *rhs, const double *K, double dxdx, int Nx, int Ny, int Nz, int BCtype){
	int ix = threadIdx.x + blockIdx.x*blockDim.x;
	int iy = threadIdx.y + blockIdx.y*blockDim.y;
	if (ix >= Nx-2 || iy >= Ny-2) return;
	int stride = Nx*Ny;
	double result;
	int in_idx;
	int iz=Nz-1;
	in_idx = (ix+1) + (iy+1)*Nx + iz*stride;
	double HC=h_in[in_idx], KC=K[in_idx];
	result=0.0;
	result -= 2.0*(HC - h_in[in_idx+1 ])     /  (1.0/KC  +  1.0/K[in_idx+1 ]);
	result -= 2.0*(HC - h_in[in_idx+Nx])     /  (1.0/KC  +  1.0/K[in_idx+Nx]);
	result -= 2.0*(HC - h_in[in_idx-1 ])     /  (1.0/KC  +  1.0/K[in_idx-1 ]);
	result -= 2.0*(HC - h_in[in_idx-Nx])     /  (1.0/KC  +  1.0/K[in_idx-Nx]);
	result -= 2.0*(HC - h_in[in_idx-stride]) /  (1.0/KC  +  1.0/K[in_idx-stride ]);
	if(BCtype==periodic) result -= 2.0*(HC - h_in[in_idx-(Nz-1)*stride]) /  (1.0/KC  +  1.0/K[in_idx-(Nz-1)*stride ]);
	if(BCtype==dirichlet) result -= 2.0*HC*KC;
	r_out[in_idx] = rhs[in_idx] - result/dxdx;
}

__global__ void update_side_south(double *r_out, double *h_in, const double *rhs, const double *K, double dxdx, int Nx, int Ny, int Nz, int BCtype){
	int ix = threadIdx.x + blockIdx.x*blockDim.x;
	int iz = threadIdx.y + blockIdx.y*blockDim.y;
	if (ix >= Nx-2 || iz >= Nz-2) return;
	int stride = Nx*Ny;
	double result;
	int in_idx;
	int iy = 0;
	in_idx = (ix + 1) + iy*Nx + (iz + 1)*stride;
	double HC=h_in[in_idx], KC=K[in_idx];
	result=0.0;
	result -= 2.0*(HC - h_in[in_idx+1 ])     /  (1.0/KC  +  1.0/K[in_idx+1     ]);
	result -= 2.0*(HC - h_in[in_idx+stride]) /  (1.0/KC  +  1.0/K[in_idx+stride]);
	result -= 2.0*(HC - h_in[in_idx-1 ])     /  (1.0/KC  +  1.0/K[in_idx-1     ]);
	result -= 2.0*(HC - h_in[in_idx-stride]) /  (1.0/KC  +  1.0/K[in_idx-stride]);
	result -= 2.0*(HC - h_in[in_idx+Nx])     /  (1.0/KC  +  1.0/K[in_idx+Nx    ]);
	if(BCtype==periodic) result -= 2.0*(HC - h_in[in_idx+(Ny-1)*Nx]) /  (1.0/KC  +  1.0/K[in_idx+(Ny-1)*Nx ]);
	if(BCtype==dirichlet) result -= 2.0*HC*KC;
	r_out[in_idx] = rhs[in_idx] - result/dxdx;
}

__global__ void update_side_north(double *r_out, double *h_in, const double *rhs, const double *K, double dxdx, int Nx, int Ny, int Nz, int BCtype){
	int ix = threadIdx.x + blockIdx.x*blockDim.x;
	int iz = threadIdx.y + blockIdx.y*blockDim.y;
	if (ix >= Nx-2 || iz >= Nz-2) return;
	int stride = Nx*Ny;
	double result;
	int in_idx;
	int iy = Ny-1;
	in_idx = (ix + 1) + iy*Nx + (iz + 1)*stride;
	double HC=h_in[in_idx], KC=K[in_idx];
	result=0.0;
	result -= 2.0*(HC - h_in[in_idx+1 ])     /  (1.0/KC  +  1.0/K[in_idx+1     ]);
	result -= 2.0*(HC - h_in[in_idx+stride]) /  (1.0/KC  +  1.0/K[in_idx+stride]);
	result -= 2.0*(HC - h_in[in_idx-1 ])     /  (1.0/KC  +  1.0/K[in_idx-1     ]);
	result -= 2.0*(HC - h_in[in_idx-stride]) /  (1.0/KC  +  1.0/K[in_idx-stride]);
	result -= 2.0*(HC - h_in[in_idx-Nx])     /  (1.0/KC  +  1.0/K[in_idx-Nx    ]);
	if(BCtype==periodic) result -= 2.0*(HC - h_in[in_idx-(Ny-1)*Nx]) /  (1.0/KC  +  1.0/K[in_idx-(Ny-1)*Nx ]);
	if(BCtype==dirichlet) result -= 2.0*HC*KC;
	r_out[in_idx] = rhs[in_idx] - result/dxdx;
}

__global__ void update_side_west(double *r_out, double *h_in, const double *rhs, const double *K, double dxdx, int Nx, int Ny, int Nz, int BCtype){
	int iy = threadIdx.x + blockIdx.x*blockDim.x;
	int iz  = threadIdx.y + blockIdx.y*blockDim.y;
	if (iy >= Ny-2 || iz >= Nz-2) return;
	int stride = Nx*Ny;
	double result;
	int in_idx;
	int ix = 0;
	in_idx = ix + (iy + 1)*Nx + (iz + 1)*stride;
	double HC=h_in[in_idx], KC=K[in_idx];
	result=0.0;
	result -= 2.0*(HC - h_in[in_idx+Nx ])     /  (1.0/KC  +  1.0/K[in_idx+Nx     ]);
	result -= 2.0*(HC - h_in[in_idx+stride])  /  (1.0/KC  +  1.0/K[in_idx+stride]);
	result -= 2.0*(HC - h_in[in_idx-Nx ])     /  (1.0/KC  +  1.0/K[in_idx-Nx     ]);
	result -= 2.0*(HC - h_in[in_idx-stride])  /  (1.0/KC  +  1.0/K[in_idx-stride]);
	result -= 2.0*(HC - h_in[in_idx+1])       /  (1.0/KC  +  1.0/K[in_idx+1    ]);
	if(BCtype==periodic) result -= 2.0*(HC - h_in[in_idx+(Nx-1)]) /  (1.0/KC  +  1.0/K[in_idx+(Nx-1) ]);
	if(BCtype==dirichlet) result -= 2.0*HC*KC;
	r_out[in_idx] = rhs[in_idx] - result/dxdx;
}

__global__ void update_side_east(double *r_out, double *h_in, const double *rhs, const double *K, double dxdx, int Nx, int Ny, int Nz, int BCtype){
	int iy = threadIdx.x + blockIdx.x*blockDim.x;
	int iz  = threadIdx.y + blockIdx.y*blockDim.y;
	if (iy >= Ny-2 || iz >= Nz-2) return;
	int stride = Nx*Ny;
	double result;
	int in_idx;
	int ix = Nx-1;
	in_idx = ix + (iy + 1)*Nx + (iz + 1)*stride;
	double HC=h_in[in_idx], KC=K[in_idx];
	result=0.0;
	result -= 2.0*(HC - h_in[in_idx+Nx ])     /  (1.0/KC  +  1.0/K[in_idx+Nx     ]);
	result -= 2.0*(HC - h_in[in_idx+stride])  /  (1.0/KC  +  1.0/K[in_idx+stride]);
	result -= 2.0*(HC - h_in[in_idx-Nx ])     /  (1.0/KC  +  1.0/K[in_idx-Nx     ]);
	result -= 2.0*(HC - h_in[in_idx-stride])  /  (1.0/KC  +  1.0/K[in_idx-stride]);
	result -= 2.0*(HC - h_in[in_idx-1])       /  (1.0/KC  +  1.0/K[in_idx-1    ]);
	if(BCtype==periodic) result -= 2.0*(HC - h_in[in_idx-(Nx-1)]) /  (1.0/KC  +  1.0/K[in_idx-(Nx-1) ]);
	if(BCtype==dirichlet) result -= 2.0*HC*KC;
	r_out[in_idx] = rhs[in_idx] - result/dxdx;
}

__global__ void update_edge_X_South_Bottom(double *r_out, double *h_in, const double *rhs, const double *K, double dxdx, int Nx, int Ny, int Nz, int BCsouth, int BCbottom){
	int ix = threadIdx.x + blockIdx.x*blockDim.x;
	if (ix >= Nx-2) return;
	int stride = Nx*Ny;
	int iy = 0;
	int iz = 0;
	int in_idx;
	in_idx = (ix+1) + iy*Nx + iz*stride;
	double result;
	double HC=h_in[in_idx], KC=K[in_idx];
	result=0.0;
	result -= 2.0*(HC - h_in[in_idx+1])       /  (1.0/KC  +  1.0/K[in_idx+1    ]);
	result -= 2.0*(HC - h_in[in_idx-1])       /  (1.0/KC  +  1.0/K[in_idx-1    ]);
	result -= 2.0*(HC - h_in[in_idx+Nx ])     /  (1.0/KC  +  1.0/K[in_idx+Nx     ]);
	result -= 2.0*(HC - h_in[in_idx+stride])  /  (1.0/KC  +  1.0/K[in_idx+stride]);
	if(BCsouth==periodic) result -= 2.0*(HC - h_in[in_idx+(Ny-1)*Nx]) /  (1.0/KC  +  1.0/K[in_idx+(Ny-1)*Nx ]);
	if(BCbottom==periodic) result -= 2.0*(HC - h_in[in_idx+(Nz-1)*stride]) /  (1.0/KC  +  1.0/K[in_idx+(Nz-1)*stride ]);
	if(BCsouth==dirichlet) result -= 2.0*HC*KC;
	if(BCbottom==dirichlet) result -= 2.0*HC*KC;
	r_out[in_idx] = rhs[in_idx] - result/dxdx;
}

__global__ void update_edge_X_South_Top(double *r_out, double *h_in, const double *rhs, const double *K, double dxdx, int Nx, int Ny, int Nz, int BCsouth, int BCtop){
	int ix = threadIdx.x + blockIdx.x*blockDim.x;
	if (ix >= Nx-2) return;
	int stride = Nx*Ny;
	int iy = 0;
	int iz = Nz-1;
	int in_idx;
	in_idx = (ix+1) + iy*Nx + iz*stride;
	double result;
	double HC=h_in[in_idx], KC=K[in_idx];
	result=0.0;
	result -= 2.0*(HC - h_in[in_idx+1])       /  (1.0/KC  +  1.0/K[in_idx+1    ]);
	result -= 2.0*(HC - h_in[in_idx-1])       /  (1.0/KC  +  1.0/K[in_idx-1    ]);
	result -= 2.0*(HC - h_in[in_idx+Nx ])     /  (1.0/KC  +  1.0/K[in_idx+Nx     ]);
	result -= 2.0*(HC - h_in[in_idx-stride])  /  (1.0/KC  +  1.0/K[in_idx-stride]);
	if(BCsouth==periodic) result -= 2.0*(HC - h_in[in_idx+(Ny-1)*Nx]) /  (1.0/KC  +  1.0/K[in_idx+(Ny-1)*Nx ]);
	if(BCtop==periodic) result -= 2.0*(HC - h_in[in_idx-(Nz-1)*stride]) /  (1.0/KC  +  1.0/K[in_idx-(Nz-1)*stride ]);
	if(BCsouth==dirichlet) result -= 2.0*HC*KC;
	if(BCtop==dirichlet) result -= 2.0*HC*KC;
	r_out[in_idx] = rhs[in_idx] - result/dxdx;
}

__global__ void update_edge_X_North_Bottom(double *r_out, double *h_in, const double *rhs, const double *K, double dxdx, int Nx, int Ny, int Nz, int BCnorth, int BCbottom){
	int ix = threadIdx.x + blockIdx.x*blockDim.x;
	if (ix >= Nx-2) return;
	int stride = Nx*Ny;
	int iy = Ny-1;
	int iz = 0;
	int in_idx;
	in_idx = (ix+1) + iy*Nx + iz*stride;
	double result;
	double HC=h_in[in_idx], KC=K[in_idx];
	result=0.0;
	result -= 2.0*(HC - h_in[in_idx+1])       /  (1.0/KC  +  1.0/K[in_idx+1    ]);
	result -= 2.0*(HC - h_in[in_idx-1])       /  (1.0/KC  +  1.0/K[in_idx-1    ]);
	result -= 2.0*(HC - h_in[in_idx-Nx ])     /  (1.0/KC  +  1.0/K[in_idx-Nx     ]);
	result -= 2.0*(HC - h_in[in_idx+stride])  /  (1.0/KC  +  1.0/K[in_idx+stride]);
	if(BCnorth==periodic) result -= 2.0*(HC - h_in[in_idx-(Ny-1)*Nx]) /  (1.0/KC  +  1.0/K[in_idx-(Ny-1)*Nx ]);
	if(BCbottom==periodic) result -= 2.0*(HC - h_in[in_idx+(Nz-1)*stride]) /  (1.0/KC  +  1.0/K[in_idx+(Nz-1)*stride ]);
	if(BCnorth==dirichlet) result -= 2.0*HC*KC;
	if(BCbottom==dirichlet) result -= 2.0*HC*KC;
	r_out[in_idx] = rhs[in_idx] - result/dxdx;
}

__global__ void update_edge_X_North_Top(double *r_out, double *h_in, const double *rhs, const double *K, double dxdx, int Nx, int Ny, int Nz, int BCnorth, int BCtop){
	int ix = threadIdx.x + blockIdx.x*blockDim.x;
	if (ix >= Nx-2) return;
	int stride = Nx*Ny;
	int iy = Ny-1;
	int iz = Nz-1;
	int in_idx;
	in_idx = (ix+1) + iy*Nx + iz*stride;
	double result;
	double HC=h_in[in_idx], KC=K[in_idx];
	result=0.0;
	result -= 2.0*(HC - h_in[in_idx+1])       /  (1.0/KC  +  1.0/K[in_idx+1    ]);
	result -= 2.0*(HC - h_in[in_idx-1])       /  (1.0/KC  +  1.0/K[in_idx-1    ]);
	result -= 2.0*(HC - h_in[in_idx-Nx ])     /  (1.0/KC  +  1.0/K[in_idx-Nx     ]);
	result -= 2.0*(HC - h_in[in_idx-stride])  /  (1.0/KC  +  1.0/K[in_idx-stride]);
	if(BCnorth==periodic) result -= 2.0*(HC - h_in[in_idx-(Ny-1)*Nx]) /  (1.0/KC  +  1.0/K[in_idx-(Ny-1)*Nx ]);
	if(BCtop==periodic) result -= 2.0*(HC - h_in[in_idx-(Nz-1)*stride]) /  (1.0/KC  +  1.0/K[in_idx-(Nz-1)*stride ]);
	if(BCnorth==dirichlet) result -= 2.0*HC*KC;
	if(BCtop==dirichlet) result -= 2.0*HC*KC;
	r_out[in_idx] = rhs[in_idx] - result/dxdx;
}

__global__ void update_edge_Z_South_West(double *r_out, double *h_in, const double *rhs, const double *K, double dxdx, int Nx, int Ny, int Nz, int BCsouth, int BCwest){
	int iz = threadIdx.x + blockIdx.x*blockDim.x;
	if (iz >= Nz-2) return;
	int stride = Nx*Ny;
	int ix = 0;
	int iy = 0;
	int in_idx;
	in_idx = ix + iy*Nx + (iz+1)*stride;
	double result;
	double HC=h_in[in_idx], KC=K[in_idx];
	result=0.0;
	result -= 2.0*(HC - h_in[in_idx+stride])  /  (1.0/KC  +  1.0/K[in_idx+stride    ]);
	result -= 2.0*(HC - h_in[in_idx-stride])  /  (1.0/KC  +  1.0/K[in_idx-stride    ]);
	result -= 2.0*(HC - h_in[in_idx+Nx ])     /  (1.0/KC  +  1.0/K[in_idx+Nx     ]);
	result -= 2.0*(HC - h_in[in_idx+1])       /  (1.0/KC  +  1.0/K[in_idx+1]);
	if(BCsouth==periodic) result -= 2.0*(HC - h_in[in_idx+(Ny-1)*Nx]) /  (1.0/KC  +  1.0/K[in_idx+(Ny-1)*Nx ]);
	if(BCwest==periodic) result -= 2.0*(HC - h_in[in_idx+(Nx-1)]) /  (1.0/KC  +  1.0/K[in_idx+(Nx-1) ]);
	if(BCsouth==dirichlet) result -= 2.0*HC*KC;
	if(BCwest==dirichlet) result -= 2.0*HC*KC;
	r_out[in_idx] = rhs[in_idx] - result/dxdx;
}

__global__ void update_edge_Z_South_East(double *r_out, double *h_in, const double *rhs, const double *K, double dxdx, int Nx, int Ny, int Nz, int BCsouth, int BCeast){
	int iz = threadIdx.x + blockIdx.x*blockDim.x;
	if (iz >= Nz-2) return;
	int stride = Nx*Ny;
	int ix = Nx-1;
	int iy = 0;
	int in_idx;
	in_idx = ix + iy*Nx + (iz+1)*stride;
	double result;
	double HC=h_in[in_idx], KC=K[in_idx];
	result=0.0;
	result -= 2.0*(HC - h_in[in_idx+stride])  /  (1.0/KC  +  1.0/K[in_idx+stride    ]);
	result -= 2.0*(HC - h_in[in_idx-stride])  /  (1.0/KC  +  1.0/K[in_idx-stride    ]);
	result -= 2.0*(HC - h_in[in_idx+Nx ])     /  (1.0/KC  +  1.0/K[in_idx+Nx     ]);
	result -= 2.0*(HC - h_in[in_idx-1])       /  (1.0/KC  +  1.0/K[in_idx-1]);
	if(BCsouth==periodic) result -= 2.0*(HC - h_in[in_idx+(Ny-1)*Nx]) /  (1.0/KC  +  1.0/K[in_idx+(Ny-1)*Nx ]);
	if(BCeast==periodic) result -= 2.0*(HC - h_in[in_idx-(Nx-1)]) /  (1.0/KC  +  1.0/K[in_idx-(Nx-1) ]);
	if(BCsouth==dirichlet) result -= 2.0*HC*KC;
	if(BCeast==dirichlet) result -= 2.0*HC*KC;
	r_out[in_idx] = rhs[in_idx] - result/dxdx;
}

__global__ void update_edge_Z_North_West(double *r_out, double *h_in, const double *rhs, const double *K, double dxdx, int Nx, int Ny, int Nz, int BCnorth, int BCwest){
	int iz = threadIdx.x + blockIdx.x*blockDim.x;
	if (iz >= Nz-2) return;
	int stride = Nx*Ny;
	int ix = 0;
	int iy = Ny-1;
	int in_idx;
	in_idx = ix + iy*Nx + (iz+1)*stride;
	double result;
	double HC=h_in[in_idx], KC=K[in_idx];
	result=0.0;
	result -= 2.0*(HC - h_in[in_idx+stride])  /  (1.0/KC  +  1.0/K[in_idx+stride    ]);
	result -= 2.0*(HC - h_in[in_idx-stride])  /  (1.0/KC  +  1.0/K[in_idx-stride    ]);
	result -= 2.0*(HC - h_in[in_idx-Nx ])     /  (1.0/KC  +  1.0/K[in_idx-Nx     ]);
	result -= 2.0*(HC - h_in[in_idx+1])       /  (1.0/KC  +  1.0/K[in_idx+1]);
	if(BCnorth==periodic) result -= 2.0*(HC - h_in[in_idx-(Ny-1)*Nx]) /  (1.0/KC  +  1.0/K[in_idx-(Ny-1)*Nx ]);
	if(BCwest==periodic) result -= 2.0*(HC - h_in[in_idx+(Nx-1)]) /  (1.0/KC  +  1.0/K[in_idx+(Nx-1) ]);
	if(BCnorth==dirichlet) result -= 2.0*HC*KC;
	if(BCwest==dirichlet) result -= 2.0*HC*KC;
	r_out[in_idx] = rhs[in_idx] - result/dxdx;
}

__global__ void update_edge_Z_North_East(double *r_out, double *h_in, const double *rhs, const double *K, double dxdx, int Nx, int Ny, int Nz, int BCnorth, int BCeast){
	int iz = threadIdx.x + blockIdx.x*blockDim.x;
	if (iz >= Nz-2) return;
	int stride = Nx*Ny;
	int ix = Nx-1;
	int iy = Ny-1;
	int in_idx;
	in_idx = ix + iy*Nx + (iz+1)*stride;
	double result;
	double HC=h_in[in_idx], KC=K[in_idx];
	result=0.0;
	result -= 2.0*(HC - h_in[in_idx+stride])  /  (1.0/KC  +  1.0/K[in_idx+stride    ]);
	result -= 2.0*(HC - h_in[in_idx-stride])  /  (1.0/KC  +  1.0/K[in_idx-stride    ]);
	result -= 2.0*(HC - h_in[in_idx-Nx ])     /  (1.0/KC  +  1.0/K[in_idx-Nx     ]);
	result -= 2.0*(HC - h_in[in_idx-1])       /  (1.0/KC  +  1.0/K[in_idx-1]);
	if(BCnorth==periodic) result -= 2.0*(HC - h_in[in_idx-(Ny-1)*Nx]) /  (1.0/KC  +  1.0/K[in_idx-(Ny-1)*Nx ]);
	if(BCeast==periodic) result -= 2.0*(HC - h_in[in_idx-(Nx-1)]) /  (1.0/KC  +  1.0/K[in_idx-(Nx-1) ]);
	if(BCnorth==dirichlet) result -= 2.0*HC*KC;
	if(BCeast==dirichlet) result -= 2.0*HC*KC;
	r_out[in_idx] = rhs[in_idx] - result/dxdx;
}

__global__ void update_edge_Y_West_Bottom(double *r_out, double *h_in, const double *rhs, const double *K, double dxdx, int Nx, int Ny, int Nz, int BCbottom, int BCwest){
	int iy = threadIdx.x + blockIdx.x*blockDim.x;
	if (iy >= Ny-2) return;
	int stride = Nx*Ny;
	int ix = 0;
	int iz = 0;
	int in_idx;
	in_idx = ix + (iy+1)*Nx + iz*stride;
	double result;
	double HC=h_in[in_idx], KC=K[in_idx];
	result=0.0;
	result -= 2.0*(HC - h_in[in_idx+Nx])  /  (1.0/KC  +  1.0/K[in_idx+Nx    ]);
	result -= 2.0*(HC - h_in[in_idx-Nx])  /  (1.0/KC  +  1.0/K[in_idx-Nx    ]);
	result -= 2.0*(HC - h_in[in_idx+stride ])     /  (1.0/KC  +  1.0/K[in_idx+stride     ]);
	result -= 2.0*(HC - h_in[in_idx+1])       /  (1.0/KC  +  1.0/K[in_idx+1]);
	if(BCbottom==periodic) result -= 2.0*(HC - h_in[in_idx+(Nz-1)*stride]) /  (1.0/KC  +  1.0/K[in_idx+(Nz-1)*stride ]);
	if(BCwest==periodic) result -= 2.0*(HC - h_in[in_idx+(Nx-1)]) /  (1.0/KC  +  1.0/K[in_idx+(Nx-1) ]);
	if(BCbottom==dirichlet) result -= 2.0*HC*KC;
	if(BCwest==dirichlet) result -= 2.0*HC*KC;
	r_out[in_idx] = rhs[in_idx] - result/dxdx;
}

__global__ void update_edge_Y_West_Top(double *r_out, double *h_in, const double *rhs, const double *K, double dxdx, int Nx, int Ny, int Nz, int BCtop, int BCwest){
	int iy = threadIdx.x + blockIdx.x*blockDim.x;
	if (iy >= Ny-2) return;
	int stride = Nx*Ny;
	int ix = 0;
	int iz = Nz-1;
	int in_idx;
	in_idx = ix + (iy+1)*Nx + iz*stride;
	double result;
	double HC=h_in[in_idx], KC=K[in_idx];
	result=0.0;
	result -= 2.0*(HC - h_in[in_idx+Nx])  /  (1.0/KC  +  1.0/K[in_idx+Nx    ]);
	result -= 2.0*(HC - h_in[in_idx-Nx])  /  (1.0/KC  +  1.0/K[in_idx-Nx    ]);
	result -= 2.0*(HC - h_in[in_idx-stride ])     /  (1.0/KC  +  1.0/K[in_idx-stride     ]);
	result -= 2.0*(HC - h_in[in_idx+1])       /  (1.0/KC  +  1.0/K[in_idx+1]);
	if(BCtop==periodic) result -= 2.0*(HC - h_in[in_idx-(Nz-1)*stride]) /  (1.0/KC  +  1.0/K[in_idx-(Nz-1)*stride ]);
	if(BCwest==periodic) result -= 2.0*(HC - h_in[in_idx+(Nx-1)]) /  (1.0/KC  +  1.0/K[in_idx+(Nx-1) ]);
	if(BCtop==dirichlet) result -= 2.0*HC*KC;
	if(BCwest==dirichlet) result -= 2.0*HC*KC;
	r_out[in_idx] = rhs[in_idx] - result/dxdx;
}

__global__ void update_edge_Y_East_Bottom(double *r_out, double *h_in, const double *rhs, const double *K, double dxdx, int Nx, int Ny, int Nz, int BCbottom, int BCeast){
	int iy = threadIdx.x + blockIdx.x*blockDim.x;
	if (iy >= Ny-2) return;
	int stride = Nx*Ny;
	int ix = Nx-1;
	int iz = 0;
	int in_idx;
	in_idx = ix + (iy+1)*Nx + iz*stride;
	double result;
	double HC=h_in[in_idx], KC=K[in_idx];
	result=0.0;
	result -= 2.0*(HC - h_in[in_idx+Nx])  /  (1.0/KC  +  1.0/K[in_idx+Nx    ]);
	result -= 2.0*(HC - h_in[in_idx-Nx])  /  (1.0/KC  +  1.0/K[in_idx-Nx    ]);
	result -= 2.0*(HC - h_in[in_idx+stride ])     /  (1.0/KC  +  1.0/K[in_idx+stride     ]);
	result -= 2.0*(HC - h_in[in_idx-1])       /  (1.0/KC  +  1.0/K[in_idx-1]);
	if(BCbottom==periodic) result -= 2.0*(HC - h_in[in_idx+(Nz-1)*stride]) /  (1.0/KC  +  1.0/K[in_idx+(Nz-1)*stride ]);
	if(BCeast==periodic) result -= 2.0*(HC - h_in[in_idx-(Nx-1)]) /  (1.0/KC  +  1.0/K[in_idx-(Nx-1) ]);
	if(BCbottom==dirichlet) result -= 2.0*HC*KC;
	if(BCeast==dirichlet) result -= 2.0*HC*KC;
	r_out[in_idx] = rhs[in_idx] - result/dxdx;
}

__global__ void update_edge_Y_East_Top(double *r_out, double *h_in, const double *rhs, const double *K, double dxdx, int Nx, int Ny, int Nz, int BCtop, int BCeast){
	int iy = threadIdx.x + blockIdx.x*blockDim.x;
	if (iy >= Ny-2) return;
	int stride = Nx*Ny;
	int ix = Nx-1;
	int iz = Nz-1;
	int in_idx;
	in_idx = ix + (iy+1)*Nx + iz*stride;
	double result;
	double HC=h_in[in_idx], KC=K[in_idx];
	result=0.0;
	result -= 2.0*(HC - h_in[in_idx+Nx])  /  (1.0/KC  +  1.0/K[in_idx+Nx    ]);
	result -= 2.0*(HC - h_in[in_idx-Nx])  /  (1.0/KC  +  1.0/K[in_idx-Nx    ]);
	result -= 2.0*(HC - h_in[in_idx-stride ])     /  (1.0/KC  +  1.0/K[in_idx-stride   ]);
	result -= 2.0*(HC - h_in[in_idx-1])       /  (1.0/KC  +  1.0/K[in_idx-1]);
	if(BCtop==periodic) result -= 2.0*(HC - h_in[in_idx-(Nz-1)*stride]) /  (1.0/KC  +  1.0/K[in_idx-(Nz-1)*stride ]);
	if(BCeast==periodic) result -= 2.0*(HC - h_in[in_idx-(Nx-1)]) /  (1.0/KC  +  1.0/K[in_idx-(Nx-1) ]);
	if(BCtop==dirichlet) result -= 2.0*HC*KC;
	if(BCeast==dirichlet) result -= 2.0*HC*KC;
	r_out[in_idx] = rhs[in_idx] - result/dxdx;
}

__global__ void update_vertex_SWB(double *r_out, double *h_in, const double *rhs, const double *K, double dxdx, int Nx, int Ny, int Nz,int BCsouth, int BCwest, int BCbottom, bool pin1stCell){
	int stride = Nx*Ny;
	int ix = 0;
	int iy = 0;
	int iz = 0;
	int in_idx;
	in_idx = ix + iy*Nx + iz*stride;
	double result;
	double HC=h_in[in_idx], KC=K[in_idx];
	double aC = 0.0;
	double KN = 0.0;
	result=0.0;
	KN = 2.0/  (1.0/KC  +  1.0/K[in_idx+Nx]);
	result -= (HC - h_in[in_idx+Nx])*KN;
	aC += KN;

	KN = 2.0/  (1.0/KC  +  1.0/K[in_idx+stride   ]);
	result -= (HC - h_in[in_idx+stride ])*KN;
	aC += KN;

	KN = 2.0/  (1.0/KC  +  1.0/K[in_idx+1]);
	result -= (HC - h_in[in_idx+1])*KN;
	aC += KN;

	if(BCbottom==periodic) result -= 2.0*(HC - h_in[in_idx+(Nz-1)*stride]) /  (1.0/KC  +  1.0/K[in_idx+(Nz-1)*stride ]);
	if(BCwest==periodic) result -= 2.0*(HC - h_in[in_idx+(Nx-1)]) /  (1.0/KC  +  1.0/K[in_idx+(Nx-1) ]);
	if(BCsouth==periodic) result -= 2.0*(HC - h_in[in_idx+(Ny-1)*Nx]) /  (1.0/KC  +  1.0/K[in_idx+(Ny-1)*Nx ]);

	if(BCbottom==dirichlet) result -= 2.0*HC*KC;
	if(BCwest==dirichlet) result -= 2.0*HC*KC;
	if(BCsouth==dirichlet) result -= 2.0*HC*KC;

	if(pin1stCell) r_out[in_idx] = rhs[in_idx] - (result-HC*aC)/dxdx; //equivalent to aC*=2.0; if pin first cell for solvability
	else r_out[in_idx] = rhs[in_idx] - result/dxdx;
}
__global__ void update_vertex_SWT(double *r_out, double *h_in, const double *rhs, const double *K, double dxdx, int Nx, int Ny, int Nz, int BCsouth, int BCwest, int BCtop){
	int stride = Nx*Ny;
	int ix = 0;
	int iy = 0;
	int iz = Nz-1;
	int in_idx;
	in_idx = ix + iy*Nx + iz*stride;
	double result;
	double HC=h_in[in_idx], KC=K[in_idx];
	result=0.0;
	result -= 2.0*(HC - h_in[in_idx+Nx])      /  (1.0/KC  +  1.0/K[in_idx+Nx    ]);
	result -= 2.0*(HC - h_in[in_idx-stride ]) /  (1.0/KC  +  1.0/K[in_idx-stride   ]);
	result -= 2.0*(HC - h_in[in_idx+1])       /  (1.0/KC  +  1.0/K[in_idx+1]);
	if(BCtop==periodic) result -= 2.0*(HC - h_in[in_idx-(Nz-1)*stride]) /  (1.0/KC  +  1.0/K[in_idx-(Nz-1)*stride ]);
	if(BCwest==periodic) result -= 2.0*(HC - h_in[in_idx+(Nx-1)]) /  (1.0/KC  +  1.0/K[in_idx+(Nx-1) ]);
	if(BCsouth==periodic) result -= 2.0*(HC - h_in[in_idx+(Ny-1)*Nx]) /  (1.0/KC  +  1.0/K[in_idx+(Ny-1)*Nx ]);
	if(BCtop==dirichlet) result -= 2.0*HC*KC;
	if(BCwest==dirichlet) result -= 2.0*HC*KC;
	if(BCsouth==dirichlet) result -= 2.0*HC*KC;
	r_out[in_idx] = rhs[in_idx] - result/dxdx;
}

__global__ void update_vertex_SEB(double *r_out, double *h_in, const double *rhs, const double *K, double dxdx, int Nx, int Ny, int Nz, int BCsouth, int BCeast, int BCbottom){
	int stride = Nx*Ny;
	int ix = Nx-1;
	int iy = 0;
	int iz = 0;
	int in_idx;
	in_idx = ix + iy*Nx + iz*stride;
	double result;
	double HC=h_in[in_idx], KC=K[in_idx];
	result=0.0;
	result -= 2.0*(HC - h_in[in_idx+Nx])      /  (1.0/KC  +  1.0/K[in_idx+Nx    ]);
	result -= 2.0*(HC - h_in[in_idx+stride ]) /  (1.0/KC  +  1.0/K[in_idx+stride   ]);
	result -= 2.0*(HC - h_in[in_idx-1])       /  (1.0/KC  +  1.0/K[in_idx-1]);
	if(BCbottom==periodic) result -= 2.0*(HC - h_in[in_idx+(Nz-1)*stride]) /  (1.0/KC  +  1.0/K[in_idx+(Nz-1)*stride ]);
	if(BCeast==periodic) result -= 2.0*(HC - h_in[in_idx-(Nx-1)]) /  (1.0/KC  +  1.0/K[in_idx-(Nx-1) ]);
	if(BCsouth==periodic) result -= 2.0*(HC - h_in[in_idx+(Ny-1)*Nx]) /  (1.0/KC  +  1.0/K[in_idx+(Ny-1)*Nx ]);
	if(BCbottom==dirichlet) result -= 2.0*HC*KC;
	if(BCeast==dirichlet) result -= 2.0*HC*KC;
	if(BCsouth==dirichlet) result -= 2.0*HC*KC;
	r_out[in_idx] = rhs[in_idx] - result/dxdx;
}

__global__ void update_vertex_SET(double *r_out, double *h_in, const double *rhs, const double *K, double dxdx, int Nx, int Ny, int Nz, int BCsouth, int BCeast, int BCtop){
	int stride = Nx*Ny;
	int ix = Nx-1;
	int iy = 0;
	int iz = Nz-1;
	int in_idx;
	in_idx = ix + iy*Nx + iz*stride;
	double result;
	double HC=h_in[in_idx], KC=K[in_idx];
	result=0.0;
	result -= 2.0*(HC - h_in[in_idx+Nx])      /  (1.0/KC  +  1.0/K[in_idx+Nx    ]);
	result -= 2.0*(HC - h_in[in_idx-stride ]) /  (1.0/KC  +  1.0/K[in_idx-stride   ]);
	result -= 2.0*(HC - h_in[in_idx-1])       /  (1.0/KC  +  1.0/K[in_idx-1]);
	if(BCtop==periodic) result -= 2.0*(HC - h_in[in_idx-(Nz-1)*stride]) /  (1.0/KC  +  1.0/K[in_idx-(Nz-1)*stride ]);
	if(BCeast==periodic) result -= 2.0*(HC - h_in[in_idx-(Nx-1)]) /  (1.0/KC  +  1.0/K[in_idx-(Nx-1) ]);
	if(BCsouth==periodic) result -= 2.0*(HC - h_in[in_idx+(Ny-1)*Nx]) /  (1.0/KC  +  1.0/K[in_idx+(Ny-1)*Nx ]);
	if(BCtop==dirichlet) result -= 2.0*HC*KC;
	if(BCeast==dirichlet) result -= 2.0*HC*KC;
	if(BCsouth==dirichlet) result -= 2.0*HC*KC;
	r_out[in_idx] = rhs[in_idx] - result/dxdx;
}

__global__ void update_vertex_NEB(double *r_out, double *h_in, const double *rhs, const double *K, double dxdx, int Nx, int Ny, int Nz, int BCnorth, int BCeast, int BCbottom){
	int stride = Nx*Ny;
	int ix = Nx-1;
	int iy = Ny-1;
	int iz = 0;
	int in_idx;
	in_idx = ix + iy*Nx + iz*stride;
	double result;
	double HC=h_in[in_idx], KC=K[in_idx];
	result=0.0;
	result -= 2.0*(HC - h_in[in_idx-Nx])      /  (1.0/KC  +  1.0/K[in_idx-Nx    ]);
	result -= 2.0*(HC - h_in[in_idx+stride ]) /  (1.0/KC  +  1.0/K[in_idx+stride   ]);
	result -= 2.0*(HC - h_in[in_idx-1])       /  (1.0/KC  +  1.0/K[in_idx-1]);
	if(BCbottom==periodic) result -= 2.0*(HC - h_in[in_idx+(Nz-1)*stride]) /  (1.0/KC  +  1.0/K[in_idx+(Nz-1)*stride ]);
	if(BCeast==periodic) result -= 2.0*(HC - h_in[in_idx-(Nx-1)]) /  (1.0/KC  +  1.0/K[in_idx-(Nx-1) ]);
	if(BCnorth==periodic) result -= 2.0*(HC - h_in[in_idx-(Ny-1)*Nx]) /  (1.0/KC  +  1.0/K[in_idx-(Ny-1)*Nx ]);
	if(BCbottom==dirichlet) result -= 2.0*HC*KC;
	if(BCeast==dirichlet) result -= 2.0*HC*KC;
	if(BCnorth==dirichlet) result -= 2.0*HC*KC;
	r_out[in_idx] = rhs[in_idx] - result/dxdx;
}

__global__ void update_vertex_NET(double *r_out, double *h_in, const double *rhs, const double *K, double dxdx, int Nx, int Ny, int Nz, int BCnorth, int BCeast, int BCtop){
	int stride = Nx*Ny;
	int ix = Nx-1;
	int iy = Ny-1;
	int iz = Nz-1;
	int in_idx;
	in_idx = ix + iy*Nx + iz*stride;
	double result;
	double HC=h_in[in_idx], KC=K[in_idx];
	result=0.0;
	result -= 2.0*(HC - h_in[in_idx-Nx])      /  (1.0/KC  +  1.0/K[in_idx-Nx    ]);
	result -= 2.0*(HC - h_in[in_idx-stride ]) /  (1.0/KC  +  1.0/K[in_idx-stride   ]);
	result -= 2.0*(HC - h_in[in_idx-1])       /  (1.0/KC  +  1.0/K[in_idx-1]);
	if(BCtop==periodic) result -= 2.0*(HC - h_in[in_idx-(Nz-1)*stride]) /  (1.0/KC  +  1.0/K[in_idx-(Nz-1)*stride ]);
	if(BCeast==periodic) result -= 2.0*(HC - h_in[in_idx-(Nx-1)]) /  (1.0/KC  +  1.0/K[in_idx-(Nx-1) ]);
	if(BCnorth==periodic) result -= 2.0*(HC - h_in[in_idx-(Ny-1)*Nx]) /  (1.0/KC  +  1.0/K[in_idx-(Ny-1)*Nx ]);
	if(BCtop==dirichlet) result -= 2.0*HC*KC;
	if(BCeast==dirichlet) result -= 2.0*HC*KC;
	if(BCnorth==dirichlet) result -= 2.0*HC*KC;
	r_out[in_idx] = rhs[in_idx] - result/dxdx;
}

__global__ void update_vertex_NWB(double *r_out, double *h_in, const double *rhs, const double *K, double dxdx, int Nx, int Ny, int Nz, int BCnorth, int BCwest, int BCbottom){
	int stride = Nx*Ny;
	int ix = 0;
	int iy = Ny-1;
	int iz = 0;
	int in_idx;
	in_idx = ix + iy*Nx + iz*stride;
	double result;
	double HC=h_in[in_idx], KC=K[in_idx];
	result=0.0;
	result -= 2.0*(HC - h_in[in_idx-Nx])      /  (1.0/KC  +  1.0/K[in_idx-Nx    ]);
	result -= 2.0*(HC - h_in[in_idx+stride ]) /  (1.0/KC  +  1.0/K[in_idx+stride   ]);
	result -= 2.0*(HC - h_in[in_idx+1])       /  (1.0/KC  +  1.0/K[in_idx+1]);
	if(BCbottom==periodic) result -= 2.0*(HC - h_in[in_idx+(Nz-1)*stride]) /  (1.0/KC  +  1.0/K[in_idx+(Nz-1)*stride ]);
	if(BCwest==periodic) result -= 2.0*(HC - h_in[in_idx+(Nx-1)]) /  (1.0/KC  +  1.0/K[in_idx+(Nx-1) ]);
	if(BCnorth==periodic) result -= 2.0*(HC - h_in[in_idx-(Ny-1)*Nx]) /  (1.0/KC  +  1.0/K[in_idx-(Ny-1)*Nx ]);
	if(BCbottom==dirichlet) result -= 2.0*HC*KC;
	if(BCwest==dirichlet) result -= 2.0*HC*KC;
	if(BCnorth==dirichlet) result -= 2.0*HC*KC;
	r_out[in_idx] = rhs[in_idx] - result/dxdx;
}

__global__ void update_vertex_NWT(double *r_out, double *h_in, const double *rhs, const double *K, double dxdx, int Nx, int Ny, int Nz, int BCnorth, int BCwest, int BCtop){
	int stride = Nx*Ny;
	int ix = 0;
	int iy = Ny-1;
	int iz = Nz-1;
	int in_idx;
	in_idx = ix + iy*Nx + iz*stride;
	double result;
	double HC=h_in[in_idx], KC=K[in_idx];
	result=0.0;
	result -= 2.0*(HC - h_in[in_idx-Nx])      /  (1.0/KC  +  1.0/K[in_idx-Nx    ]);
	result -= 2.0*(HC - h_in[in_idx-stride ]) /  (1.0/KC  +  1.0/K[in_idx-stride   ]);
	result -= 2.0*(HC - h_in[in_idx+1])       /  (1.0/KC  +  1.0/K[in_idx+1]);
	if(BCtop==periodic) result -= 2.0*(HC - h_in[in_idx-(Nz-1)*stride]) /  (1.0/KC  +  1.0/K[in_idx-(Nz-1)*stride ]);
	if(BCwest==periodic) result -= 2.0*(HC - h_in[in_idx+(Nx-1)]) /  (1.0/KC  +  1.0/K[in_idx+(Nx-1) ]);
	if(BCnorth==periodic) result -= 2.0*(HC - h_in[in_idx-(Ny-1)*Nx]) /  (1.0/KC  +  1.0/K[in_idx-(Ny-1)*Nx ]);
	if(BCtop==dirichlet) result -= 2.0*HC*KC;
	if(BCwest==dirichlet) result -= 2.0*HC*KC;
	if(BCnorth==dirichlet) result -= 2.0*HC*KC;
	r_out[in_idx] = rhs[in_idx] - result/dxdx;
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// %%%%%%%%   AUXILIARY FUNCTION FOR UPDATE RESIDUAL   %%%%%%%%%
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
void update_res(double *rk_1, double *xk, const double *rhs, const double *K,
	double dxdx, int Nx, int Ny, int Nz,
	int BCbottom, int BCtop, int BCsouth, int BCnorth, int BCwest, int BCeast, bool pin1stCell,
	dim3 grid, dim3 block){

update_int        <<<gridXY,blockXY>>>(rk_1,xk,rhs,K,dxdx,Nx,Ny,Nz);

update_side_bottom<<<gridXY,blockXY>>>(rk_1,xk,rhs,K,dxdx,Nx,Ny,Nz,BCbottom);
update_side_top   <<<gridXY,blockXY>>>(rk_1,xk,rhs,K,dxdx,Nx,Ny,Nz,BCtop);

update_side_south<<<gridXZ,blockXZ>>>(rk_1,xk,rhs,K,dxdx,Nx,Ny,Nz,BCsouth);
update_side_north<<<gridXZ,blockXZ>>>(rk_1,xk,rhs,K,dxdx,Nx,Ny,Nz,BCnorth);

update_side_west<<<gridYZ,blockYZ>>>(rk_1,xk,rhs,K,dxdx,Nx,Ny,Nz,BCwest);
update_side_east<<<gridYZ,blockYZ>>>(rk_1,xk,rhs,K,dxdx,Nx,Ny,Nz,BCeast);

update_edge_X_South_Bottom<<<grid.x,block.x>>>(rk_1,xk,rhs,K,dxdx,Nx,Ny,Nz,BCsouth,BCbottom);
update_edge_X_South_Top   <<<grid.x,block.x>>>(rk_1,xk,rhs,K,dxdx,Nx,Ny,Nz,BCsouth,BCtop);
update_edge_X_North_Bottom<<<grid.x,block.x>>>(rk_1,xk,rhs,K,dxdx,Nx,Ny,Nz,BCnorth,BCbottom);
update_edge_X_North_Top   <<<grid.x,block.x>>>(rk_1,xk,rhs,K,dxdx,Nx,Ny,Nz,BCnorth,BCtop);

update_edge_Z_South_West<<<grid.z,block.z>>>(rk_1,xk,rhs,K,dxdx,Nx,Ny,Nz,BCsouth,BCwest);
update_edge_Z_South_East<<<grid.z,block.z>>>(rk_1,xk,rhs,K,dxdx,Nx,Ny,Nz,BCsouth,BCeast);
update_edge_Z_North_West<<<grid.z,block.z>>>(rk_1,xk,rhs,K,dxdx,Nx,Ny,Nz,BCnorth,BCwest);
update_edge_Z_North_East<<<grid.z,block.z>>>(rk_1,xk,rhs,K,dxdx,Nx,Ny,Nz,BCnorth,BCeast);

update_edge_Y_West_Bottom<<<grid.y,block.y>>>(rk_1,xk,rhs,K,dxdx,Nx,Ny,Nz,BCbottom,BCwest);
update_edge_Y_West_Top   <<<grid.y,block.y>>>(rk_1,xk,rhs,K,dxdx,Nx,Ny,Nz,BCtop,BCwest);
update_edge_Y_East_Bottom<<<grid.y,block.y>>>(rk_1,xk,rhs,K,dxdx,Nx,Ny,Nz,BCbottom,BCeast);
update_edge_Y_East_Top   <<<grid.y,block.y>>>(rk_1,xk,rhs,K,dxdx,Nx,Ny,Nz,BCtop,BCeast);

//pin1stCell <- true, if pin 1st cell for solvability (i.e. all homogeneous neumann BC)
update_vertex_SWB<<<1,1>>>(rk_1,xk,rhs,K,dxdx,Nx,Ny,Nz,BCsouth,BCwest,BCbottom,pin1stCell);
update_vertex_SWT<<<1,1>>>(rk_1,xk,rhs,K,dxdx,Nx,Ny,Nz,BCsouth,BCwest,BCtop);
update_vertex_SEB<<<1,1>>>(rk_1,xk,rhs,K,dxdx,Nx,Ny,Nz,BCsouth,BCeast,BCbottom);
update_vertex_SET<<<1,1>>>(rk_1,xk,rhs,K,dxdx,Nx,Ny,Nz,BCsouth,BCeast,BCtop);

update_vertex_NEB<<<1,1>>>(rk_1,xk,rhs,K,dxdx,Nx,Ny,Nz,BCnorth,BCeast,BCbottom);
update_vertex_NET<<<1,1>>>(rk_1,xk,rhs,K,dxdx,Nx,Ny,Nz,BCnorth,BCeast,BCtop);
update_vertex_NWB<<<1,1>>>(rk_1,xk,rhs,K,dxdx,Nx,Ny,Nz,BCnorth,BCwest,BCbottom);
update_vertex_NWT<<<1,1>>>(rk_1,xk,rhs,K,dxdx,Nx,Ny,Nz,BCnorth,BCwest,BCtop);
cudaDeviceSynchronize();
}
```
