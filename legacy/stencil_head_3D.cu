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
