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