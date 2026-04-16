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
