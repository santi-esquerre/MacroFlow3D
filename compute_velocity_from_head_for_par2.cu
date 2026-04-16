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
