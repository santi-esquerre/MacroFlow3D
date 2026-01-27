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