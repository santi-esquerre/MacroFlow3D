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
