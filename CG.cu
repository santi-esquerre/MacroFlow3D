#include <iostream>
#include "header/linear_operator.h"

// for residual evolution monitoring
void resMonitor(int levelPrint, int cccc, double *rr_0_h, double &res_relative_old, double *rr_new_h, int i){
	if(levelPrint==2 && cccc==1){
		std::cout<< "<b,b>: "<<*rr_new_h<<std::endl;
		std::cout<<"Iters       ||r||_2     conv.rate  ||r||_2/||b||_2"<<std::endl;
	    std::cout<<"-----    ------------   ---------  ------------"<<std::endl;
	}
	if(levelPrint==2 && cccc==2){
		std::cout.width(5); std::cout << std::right<<i<<"    ";
		std::cout<<std::right<<std::scientific<<pow(*rr_new_h,0.5)<<"    "<<std::right<<std::fixed<<(pow(*rr_new_h,0.5)/pow(*rr_0_h,0.5))/res_relative_old<<"    ";
		res_relative_old = pow(*rr_new_h,0.5)/pow(*rr_0_h,0.5);
		std::cout<<std::right<<std::scientific<<res_relative_old<<std::endl;
	}
	if(levelPrint==1 && cccc==0){
	std::cout.width(5); std::cout << std::right<<i<<"    ";
	std::cout<<std::right<<std::scientific<<pow(*rr_new_h,0.5)<<std::endl;
	}
}

// WARNING!
// Whenever possible the following is used: 
// b (RHS) vector is used as the auxiliary vector P
// z is used auxiliary vector y
// otherwise you must modify to add the vectors as input and
// replace z by y, P by RHS within algorithm
// P is RHS at the beginning
int solver_CG(Matrix_t &M, Matrix_t &precond, blas_t &BLAS, double *x, double *P, double *r, double *z, double *y, double tol_abs, double tol_rel, int iter_max, int print_monitor){
	double *rr_new, *rho, *rho_old, *pAp;
	double *rr_0_h, *rr_new_h;
	cudaMalloc(&rr_new 	, sizeof(double));
	cudaMalloc(&rho    	, sizeof(double));
	cudaMalloc(&rho_old	, sizeof(double));
	cudaMalloc(&pAp		, sizeof(double));
	rr_0_h = new double[1]; rr_new_h = new double[1];
	M.stencil(y,x);	
	BLAS.AXPBY3D(z,y,r,1.0,-1.0);
	BLAS.Ddot(r,r,rr_new);
	BLAS.copyScalar_d2h(rr_0_h, rr_new);
	*rr_new_h = *rr_0_h;
	int i = 0;
	double res_relative_old = 1;// auxiliar for printing
	resMonitor(print_monitor,1,rr_0_h,res_relative_old,rr_new_h,i); // if print_monitor==2 (print table)
	while( ( pow(*rr_new_h,0.5) > tol_abs + pow(*rr_0_h,0.5) * tol_rel ) && i<iter_max){
		precond.stencil(z,r);
		BLAS.copyScalar_d2d(rho_old, rho);
		BLAS.Ddot(r,z,rho);
		if (i==0) BLAS.copyVector_d2d(P,z);
		else BLAS.beta3D(z,P,rho,rho_old);
		M.stencil(y,P);
		BLAS.Ddot(y,P,pAp);
		BLAS.alpha3D(P,x,rho,pAp,true);
		BLAS.alpha3D(y,r,rho,pAp,false);
		BLAS.Ddot(r,r,rr_new);
		BLAS.copyScalar_d2h(rr_new_h,rr_new);
		i+=1;
		// if print_monitor==2 (print table)
		resMonitor(print_monitor,2,rr_0_h,res_relative_old,rr_new_h,i);
	}
	// if print_monitor==1 (print finalAbsRes, iterNumber)
	resMonitor(print_monitor,0,rr_0_h,res_relative_old,rr_new_h,i); 
    cudaFree(rr_new);
    cudaFree(rho);
    cudaFree(rho_old);
	cudaFree(pAp);
	delete [] rr_new_h;
	delete [] rr_0_h;
	return i;
}