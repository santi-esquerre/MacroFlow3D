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
int solver_BiCGStab(Matrix_t &M, Matrix_t &precond, blas_t &BLAS, 
	double *x, double *s, double *r, double *r_, 
	double *Mp, double *Ms,double *AMs, double *AMp, double *RHS,
	double tol_abs, double tol_rel, int iter_max, int print_monitor){

	double *rr_, *AMpr_, *AMss, *AMsAMs, *rr_new;
	double *rr__new;
	cudaMalloc(&rr_new    , sizeof(double));
	cudaMalloc(&rr_    , sizeof(double));
	cudaMalloc(&rr__new    , sizeof(double));
	cudaMalloc(&AMpr_		 , sizeof(double));
	cudaMalloc(&AMss    	 , sizeof(double));
	cudaMalloc(&AMsAMs		 , sizeof(double));	
	double *rr_0_h, *rr_new_h, *ss;
	rr_0_h = new double[1]; rr_new_h = new double[1]; ss = new double[1];	
	M.stencil(r,x); // y <- Ax	
	BLAS.AXPBY3D(RHS,r,r,1.0,-1.0); // r <- b - A*x	
	BLAS.copyVector_d2d(RHS,r); //p <- r
	// BLAS.copyVector_d2d(P,r); //p <- r	
	BLAS.copyVector_d2d(r_,r); //r_ = hat(r) // r_star <- r
	BLAS.Ddot(r_,r,rr_);
	BLAS.copyScalar_d2h(rr_0_h,rr_);
	*rr_new_h = *rr_0_h;	
	int i = 0;
	double res_relative_old = 1;// auxiliar for printing
	resMonitor(print_monitor,1,rr_0_h,res_relative_old,rr_new_h,i); // if print_monitor==2 (print table)
	while( ( pow(*rr_new_h,0.5) > tol_abs + pow(*rr_0_h,0.5) * tol_rel ) && i<iter_max){
		precond.stencil(Mp,RHS); // Mp = M*p
		// precond.stencil(Mp,P); // Mp = M*p	
		M.stencil(AMp,Mp); // AMp = A*Mp	
		// alpha = (r_j, r_star) / (A*M*p, r_star)
		BLAS.Ddot(AMp,r_,AMpr_); //AMpr_ = (A*M*p,r_star)		
		// s_j = r_j - alpha * AMp /////rr_ = (r_j, r_star)
		BLAS.alphaU3D(AMp,r,s,rr_,AMpr_); //Ap = A*M*p
		BLAS.Ddot(s,s,AMsAMs); //AMsAMs = ss = <s,s>;
		BLAS.copyScalar_d2h(ss,AMsAMs);
		if( ( pow(*ss,0.5) < tol_abs + pow(*rr_0_h,0.5) * tol_rel ) || (i>=iter_max)   ){
			BLAS.alpha3D(Mp,x,rr_,AMpr_,true); // x += alpha*M*p_j
			break;
		}		
		precond.stencil(Ms,s); // Ms = M*s_j		
		M.stencil(AMs,Ms); // AMs = A*Ms
		// omega = (AMs, s) / (AMs, AMs)		
		BLAS.Ddot(AMs,s,AMss); //Ass, //(AMs, s)
		BLAS.Ddot(AMs,AMs,AMsAMs); //AsAs, //(AMs, AMs)
		BLAS.omegaX3D(Mp,x,Ms,rr_,AMpr_,AMss,AMsAMs); //x_{j+1} = x_j + alpha*M*p_j + omega*M*s_j		
		BLAS.omegaR3D(s,r,AMs,AMss,AMsAMs); //r_{j+1} = s_j - omega*A*M*s		
		// beta_j = (r_{j+1}, r_star) / (r_j, r_star) * (alpha/omega)	
		BLAS.Ddot(r_,r,rr__new); //rr__new = (r_{j+1}, r_star)
		// p_{j+1} = r_{j+1} + beta*(p_j - omega*A*M*p)
		BLAS.betaU3D(r,RHS,AMp,rr__new,AMpr_,AMss,AMsAMs);
		// BLAS.betaU3D(r,P,AMp,rr__new,AMpr_,AMss,AMsAMs);
		//r_r_star_old = r_r_star_new;
		BLAS.copyScalar_d2d(rr_,rr__new);
		BLAS.Ddot(r,r,rr_new);
		BLAS.copyScalar_d2h(rr_new_h, rr_new);
		i+=1;
		// if print_monitor==2 (print table)
		resMonitor(print_monitor,2,rr_0_h,res_relative_old,rr_new_h,i);
	}
	// if print_monitor==1 (print finalAbsRes, iterNumber)
	resMonitor(print_monitor,0,rr_0_h,res_relative_old,rr_new_h,i); 

	cudaFree(rr_new);
    cudaFree(rr_);
	cudaFree(rr__new);
    cudaFree(AMpr_);
    cudaFree(AMss);    
	cudaFree(AMsAMs);

	delete [] rr_0_h;
	delete [] rr_new_h;
	delete [] ss;
	return i;
}