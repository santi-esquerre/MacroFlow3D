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
