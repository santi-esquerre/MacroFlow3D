
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