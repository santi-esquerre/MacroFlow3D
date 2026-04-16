#include "header/routines_CCMG.h"
#include "header/MG_struct.h"
#include "cublas_v2.h"
#include "header/linear_operator.h"

#define boundaryCond dirichletBottom,dirichletTop,dirichletSouth,dirichletNorth,dirichletWest,dirichletEast,pin1stCell

void V_cycle(double **e_pre, double **r, double **rr, double **K,
	dim3 *grid, dim3 *block,
	int dirichletBottom, int dirichletTop,
	int dirichletSouth, int dirichletNorth,
	int dirichletWest, int dirichletEast, bool pin1stCell,
	MG_levels MG, int l, cublasHandle_t handle, int ratioX, int ratioY, int ratioZ, double Ly){
	int Nx = pow(2,l)*ratioX;
	int Ny = pow(2,l)*ratioY;
	int Nz = pow(2,l)*ratioZ;
	double h = Ly/(double)Ny;// carasteristic length is Ly
	double dxdx = h*h;
	double h_H = h*2.0;
	cudaMemset(e_pre[l],0,sizeof(double)*Nx*Ny*Nz);
	//el_pre <- smooth(0,rl); //rrl <- rl-A*el_pre;
	smooth_GSRB(e_pre[l],r[l],rr[l],K[l],dxdx,Nx,Ny,Nz,boundaryCond,MG.npre,true,grid[l],block[l]);
	//rrl1 <- restrict(rrl)
	restriction(r[l-1],rr[l],Nx/2,Ny/2,Nz/2,grid[l-1],block[l-1]);
	//solve coarse system
	if (l==2) SolveCoarseSystemGSRB(e_pre[l-1],r[l-1],rr[l-1],K[l-1],h_H*h_H,Nx/2,Ny/2,Nz/2,10000,grid[l-1],block[l-1],handle,boundaryCond);
	//e[l-1] <- MGCYCLE(rl1)
	else V_cycle(e_pre,r,rr,K,grid,block,boundaryCond,MG,l-1,handle,ratioX,ratioY,ratioZ,Ly);
	//el_cgc = prolong(el1); //el_sum = el_pre+el_cgc
	prolongation(e_pre[l],e_pre[l-1],Nx,Ny,Nz,grid[l],block[l]);
	//el <- smooth(el_sum,rl)
	smooth_GSRB(e_pre[l],r[l],rr[l],K[l],dxdx,Nx,Ny,Nz,boundaryCond,MG.npos,false,grid[l],block[l]);
}

int solver_CG(Matrix_t &M, Matrix_t &precond, blas_t &BLAS, double *x, double *P, double *r, double *z, double *y, double tol_abs, double tol_rel, int iter_max, int print_monitor);

void V_cycle2(double **e_pre, double **r, double **rr, double **K,
	dim3 *grid, dim3 *block,
	int dirichletBottom, int dirichletTop,
	int dirichletSouth, int dirichletNorth,
	int dirichletWest, int dirichletEast, bool pin1stCell,
	MG_levels MG, int l, cublasHandle_t handle, int ratioX, int ratioY, int ratioZ, double Ly,
	Matrix_t &M, Matrix_t &precond, blas_t &BLAS, double *aux, double *aux2
	){
	int Nx = pow(2,l)*ratioX;
	int Ny = pow(2,l)*ratioY;
	int Nz = pow(2,l)*ratioZ;
	double h = Ly/(double)Ny;
	double dxdx = h*h;
	// double h_H = h*2.0;
	cudaMemset(e_pre[l],0,sizeof(double)*Nx*Ny*Nz);
	//el_pre <- smooth(0,rl); //rrl <- rl-A*el_pre;
	smooth_GSRB(e_pre[l],r[l],rr[l],K[l],dxdx,Nx,Ny,Nz,boundaryCond,MG.npre,true,grid[l],block[l]);
	//rrl1 <- restrict(rrl)
	restriction(r[l-1],rr[l],Nx/2,Ny/2,Nz/2,grid[l-1],block[l-1]);
	//solve coarse system
	if (l==1)solver_CG(M,precond,BLAS,e_pre[l-1],rr[l-1],aux,r[l-1],aux2,1e-16,0,1000,0);
	//e[l-1] <- MGCYCLE(rl1)
	else V_cycle2(e_pre,r,rr,K,grid,block,boundaryCond,MG,l-1,handle,ratioX,ratioY,ratioZ,Ly,M,precond,BLAS,aux,aux2);
	//el_cgc = prolong(el1); //el_sum = el_pre+el_cgc
	prolongation(e_pre[l],e_pre[l-1],Nx,Ny,Nz,grid[l],block[l]);
	//el <- smooth(el_sum,rl)
	smooth_GSRB(e_pre[l],r[l],rr[l],K[l],dxdx,Nx,Ny,Nz,boundaryCond,MG.npos,false,grid[l],block[l]);
}

void Precond_CCMG_Vcycle(double *e0fine, const double *rfine,
	double **e, double **r, double **rr, double **K,
	dim3 *grid, dim3 *block,
	int dirichletBottom, int dirichletTop,
	int dirichletSouth, int dirichletNorth,
	int dirichletWest, int dirichletEast, bool pin1stCell,
	int Nx, int Ny, int Nz, MG_levels MG, cublasHandle_t handle, int ratioX, int ratioY, int ratioZ, double Ly){
	int l = MG.L-1;
	double h = Ly/(double)Ny;
	double dxdx = h*h;
	smooth_GSRB(e0fine,rfine,rr[l],K[l],dxdx,Nx,Ny,Nz,boundaryCond,MG.npre,true,grid[l],block[l]);
	restriction(r[l-1],rr[l],Nx/2,Ny/2,Nz/2,grid[l-1],block[l-1]);
	V_cycle(e,r,rr,K,grid,block,boundaryCond,MG,l-1,handle,ratioX,ratioY,ratioZ,Ly);
	prolongation(e0fine,e[l-1],Nx,Ny,Nz,grid[l],block[l]);
	smooth_GSRB(e0fine,rfine,rr[l],K[l],dxdx,Nx,Ny,Nz,boundaryCond,MG.npos,false,grid[l],block[l]);
}

void Precond_CCMG_Vcycle2(double *e0fine, const double *rfine,
	double **e, double **r, double **rr, double **K,
	dim3 *grid, dim3 *block,
	int dirichletBottom, int dirichletTop,
	int dirichletSouth, int dirichletNorth,
	int dirichletWest, int dirichletEast, bool pin1stCell,
	int Nx, int Ny, int Nz, MG_levels MG, cublasHandle_t handle, int ratioX, int ratioY, int ratioZ, double Ly,
	Matrix_t &M, Matrix_t &precond, blas_t &BLAS, double *aux, double *aux2
	){
	int l = MG.L-1;
	double h = Ly/(double)Ny;
	double dxdx = h*h;
	smooth_GSRB(e0fine,rfine,rr[l],K[l],dxdx,Nx,Ny,Nz,boundaryCond,MG.npre,true,grid[l],block[l]);
	restriction(r[l-1],rr[l],Nx/2,Ny/2,Nz/2,grid[l-1],block[l-1]);
	V_cycle2(e,r,rr,K,grid,block,boundaryCond,MG,l-1,handle,ratioX,ratioY,ratioZ,Ly,M,precond,BLAS,aux,aux2);
	prolongation(e0fine,e[l-1],Nx,Ny,Nz,grid[l],block[l]);
	smooth_GSRB(e0fine,rfine,rr[l],K[l],dxdx,Nx,Ny,Nz,boundaryCond,MG.npos,false,grid[l],block[l]);
}

// // grid defined from  z-direction
// void V_cycle3(double **e_pre, double **r, double **rr,
// 	dim3 *gridXY, dim3 *blockXY, dim3 *gridXZ, dim3 *blockXZ, dim3 *gridYZ, dim3 *blockYZ,
// 	bool dirichletBottom, bool dirichletTop,
// 	bool dirichletSouth, bool dirichletNorth,
// 	bool dirichletWest, bool dirichletEast, bool pin1stCell,
// 	MG_levels MG, int level, cublasHandle_t handle, int ratioZX, int ratioZY, double Lz,
// 	Matrix_t &M, Matrix_t &precond, blas_t &BLAS, double *aux, double *aux2
// 	){
// 	int Nz = pow(2,level);
// 	int Nx = Nz*ratioZX;
// 	int Ny = Nz*ratioZY;

// 	double h = Lz/(double)Nz;
// 	double dxdx = h*h;
// 	// double h_H = h*2.0;
// 	cudaMemset(e_pre[level],0,sizeof(double)*Nx*Ny*Nz);
// 	//el_pre <- smooth(0,rl); //rrl <- rl-A*el_pre;
// 	smooth_GSRB(e_pre[level],r[level],rr[level],dxdx,Nx,Ny,Nz,boundaryCond,MG.npre,true,GRID3D_L);
// 	//rrl1 <- restrict(rrl)
// 	restriction(r[level-1],rr[level],Nx/2,Ny/2,Nz/2,gridXY[level-1],blockXY[level-1]);
// 	//solve coarse system
// 	if (level==2)solver_CG(M,precond,BLAS,e_pre[level-1],rr[level-1],aux,r[level-1],aux2,1e-16,0,10000,0);
// 	//e[level-1] <- MGCYCLE(rl1)
// 	else V_cycle3(e_pre,r,rr,gridXY,blockXY,gridXZ,blockXZ,gridYZ,blockYZ,boundaryCond,MG,level-1,handle,ratioZX,ratioZY,Lz,M,precond,BLAS,aux,aux2);
// 	//el_cgc = prolong(el1); //el_sum = el_pre+el_cgc
// 	prolongation(e_pre[level],e_pre[level-1],Nx,Ny,Nz,GRID3D_L);
// 	//el <- smooth(el_sum,rl)
// 	smooth_GSRB(e_pre[level],r[level],rr[level],dxdx,Nx,Ny,Nz,boundaryCond,MG.npos,false,GRID3D_L);
// }

// void Precond_CCMG_Vcycle3(double *e0fine, const double *rfine,
// 	double **e, double **r, double **rr,
// 	dim3 *gridXY, dim3 *blockXY, dim3 *gridXZ, dim3 *blockXZ, dim3 *gridYZ, dim3 *blockYZ,
// 	bool dirichletBottom, bool dirichletTop,
// 	bool dirichletSouth, bool dirichletNorth,
// 	bool dirichletWest, bool dirichletEast, bool pin1stCell,
// 	int Nx, int Ny, int Nz, MG_levels MG, cublasHandle_t handle, int ratioZX, int ratioZY, double Lz,
// 	Matrix_t &M, Matrix_t &precond, blas_t &BLAS, double *aux, double *aux2
// 	){
// 	int level = MG.L-1;
// 	double h = Lz/(double)Nz;
// 	double dxdx = h*h;
// 	smooth_GSRB(e0fine,rfine,rr[level],dxdx,Nx,Ny,Nz,boundaryCond,MG.npre,true,GRID3D_L);
// 	restriction(r[level-1],rr[level],Nx/2,Ny/2,Nz/2,gridXY[level-1],blockXY[level-1]);
// 	V_cycle3(e,r,rr,gridXY,blockXY,gridXZ,blockXZ,gridYZ,blockYZ,boundaryCond,MG,level-1,handle,ratioZX,ratioZY,Lz,M,precond,BLAS,aux,aux2);
// 	prolongation(e0fine,e[level-1],Nx,Ny,Nz,GRID3D_L);
// 	smooth_GSRB(e0fine,rfine,rr[level],dxdx,Nx,Ny,Nz,boundaryCond,MG.npos,false,GRID3D_L);
// }

// int solver_BiCGStab(Matrix_t &M, Matrix_t &precond, blas_t &BLAS,
// 	double *x, double *s, double *r, double *r_,
// 	double *Mp, double *Ms,double *AMs, double *AMp, double *RHS,
// 	double tol_abs, double tol_rel, int iter_max, int print_monitor);

// // grid defined from  z-direction
// void V_cycle4(double **e_pre, double **r, double **rr,
// 	dim3 *gridXY, dim3 *blockXY, dim3 *gridXZ, dim3 *blockXZ, dim3 *gridYZ, dim3 *blockYZ,
// 	bool dirichletBottom, bool dirichletTop,
// 	bool dirichletSouth, bool dirichletNorth,
// 	bool dirichletWest, bool dirichletEast, bool pin1stCell,
// 	MG_levels MG, int level, cublasHandle_t handle, int ratioZX, int ratioZY, double Lz,
// 	Matrix_t &M, Matrix_t &precond, blas_t &BLAS, double *aux, double *aux2,
// 	double *Mp, double *Ms, double *AMs, double *AMp
// 	){
// 	int Nz = pow(2,level);
// 	int Nx = Nz*ratioZX;
// 	int Ny = Nz*ratioZY;

// 	double h = Lz/(double)Nz;
// 	double dxdx = h*h;
// 	// double h_H = h*2.0;
// 	cudaMemset(e_pre[level],0,sizeof(double)*Nx*Ny*Nz);
// 	//el_pre <- smooth(0,rl); //rrl <- rl-A*el_pre;
// 	smooth_GSRB(e_pre[level],r[level],rr[level],dxdx,Nx,Ny,Nz,boundaryCond,MG.npre,true,GRID3D_L);
// 	//rrl1 <- restrict(rrl)
// 	restriction(r[level-1],rr[level],Nx/2,Ny/2,Nz/2,gridXY[level-1],blockXY[level-1]);
// 	//solve coarse system
// 	if (level==2) solver_BiCGStab(M,precond,BLAS,e_pre[level-1],rr[level-1],aux,aux2,Mp,Ms,AMs,AMp,r[level-1],1e-16,0,10000,0);
// 	//e[level-1] <- MGCYCLE(rl1)
// 	else V_cycle4(e_pre,r,rr,gridXY,blockXY,gridXZ,blockXZ,gridYZ,blockYZ,boundaryCond,MG,level-1,handle,ratioZX,ratioZY,Lz,M,precond,BLAS,aux,aux2,Mp,Ms,AMs,AMp);
// 	//el_cgc = prolong(el1); //el_sum = el_pre+el_cgc
// 	prolongation(e_pre[level],e_pre[level-1],Nx,Ny,Nz,GRID3D_L);
// 	//el <- smooth(el_sum,rl)
// 	smooth_GSRB(e_pre[level],r[level],rr[level],dxdx,Nx,Ny,Nz,boundaryCond,MG.npos,false,GRID3D_L);
// }

// void Precond_CCMG_Vcycle4(double *e0fine, const double *rfine,
// 	double **e, double **r, double **rr,
// 	dim3 *gridXY, dim3 *blockXY, dim3 *gridXZ, dim3 *blockXZ, dim3 *gridYZ, dim3 *blockYZ,
// 	bool dirichletBottom, bool dirichletTop,
// 	bool dirichletSouth, bool dirichletNorth,
// 	bool dirichletWest, bool dirichletEast, bool pin1stCell,
// 	int Nx, int Ny, int Nz, MG_levels MG, cublasHandle_t handle, int ratioZX, int ratioZY, double Lz,
// 	Matrix_t &M, Matrix_t &precond, blas_t &BLAS, double *aux, double *aux2,double *Mp, double *Ms, double *AMs, double *AMp
// 	){
// 	int level = MG.L-1;
// 	double h = Lz/(double)Nz;
// 	double dxdx = h*h;
// 	smooth_GSRB(e0fine,rfine,rr[level],dxdx,Nx,Ny,Nz,boundaryCond,MG.npre,true,GRID3D_L);
// 	restriction(r[level-1],rr[level],Nx/2,Ny/2,Nz/2,gridXY[level-1],blockXY[level-1]);
// 	V_cycle4(e,r,rr,gridXY,blockXY,gridXZ,blockXZ,gridYZ,blockYZ,boundaryCond,MG,level-1,handle,ratioZX,ratioZY,Lz,M,precond,BLAS,aux,aux2,Mp,Ms,AMs,AMp);
// 	prolongation(e0fine,e[level-1],Nx,Ny,Nz,GRID3D_L);
// 	smooth_GSRB(e0fine,rfine,rr[level],dxdx,Nx,Ny,Nz,boundaryCond,MG.npos,false,GRID3D_L);
// }
#undef boundaryCond
