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
