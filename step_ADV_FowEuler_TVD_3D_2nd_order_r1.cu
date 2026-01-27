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