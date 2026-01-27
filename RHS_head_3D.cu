/**
* @file RHS_head_3D.cu
* @brief Compute RHS for the flow equation (no-flow, periodic, or Dirichlet BCs are admitted).
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

#define neumann 0
#define periodic 1
#define dirichlet 2
#define gridXY dim3(grid.x,grid.y)
#define blockXY dim3(block.x,block.y)
#define gridXZ dim3(grid.x,grid.z)
#define blockXZ dim3(block.x,block.z)
#define gridYZ dim3(grid.y,grid.z)
#define blockYZ dim3(block.y,block.z)


__global__ void RHS_head_int(double *RHS, const double *K, int Nx, int Ny, int Nz){
	const int ix = threadIdx.x + blockIdx.x*blockDim.x;
	const int iy = threadIdx.y + blockIdx.y*blockDim.y;
	if (ix >= Nx-2 || iy >= Ny-2) return;
	int in_idx = (ix + 1) + (iy + 1)*Nx;
	int stride = Nx*Ny;
	for(int i=1; i<Nz-1; ++i){ 
		in_idx += stride;
		RHS[in_idx] = 0.0;
	}
}

__global__ void RHS_head_side_bottom(double *RHS, const double *K, int Nx, int Ny, int Nz, double A, double h, int BCtype, double Hb){
	int ix = threadIdx.x + blockIdx.x*blockDim.x;
	int iy = threadIdx.y + blockIdx.y*blockDim.y;
	if (ix >= Nx-2 || iy >= Ny-2) return;
	int stride = Nx*Ny;
	int in_idx;
	int iz=0;
	in_idx = (ix+1) + (iy+1)*Nx + iz*stride;
	double KC = K[in_idx];
	if(BCtype==dirichlet) RHS[in_idx] -= 2.0*Hb*KC/h*A/h/h/h; //dirichlet contribution
}

__global__ void RHS_head_side_top(double *RHS, const double *K, int Nx, int Ny, int Nz, double A, double h, int BCtype, double Hb){
    int ix = threadIdx.x + blockIdx.x*blockDim.x;
    int iy = threadIdx.y + blockIdx.y*blockDim.y;
    if (ix >= Nx-2 || iy >= Ny-2) return;
    int stride = Nx*Ny;
    int iz=Nz-1;
    int in_idx = (ix+1) + (iy+1)*Nx + iz*stride;
    double KC = K[in_idx];
    if(BCtype==dirichlet) RHS[in_idx] -= 2.0*Hb*KC/h*A/h/h/h; //dirichlet contribution
}

__global__ void RHS_head_side_south(double *RHS, const double *K, int Nx, int Ny, int Nz, double A, double h, int BCtype, double Hb){
	int ix = threadIdx.x + blockIdx.x*blockDim.x;
	int iz = threadIdx.y + blockIdx.y*blockDim.y;
	if (ix >= Nx-2 || iz >= Nz-2) return;
	int stride = Nx*Ny;
	int in_idx;
	int iy = 0;
	in_idx = (ix + 1) + iy*Nx + (iz + 1)*stride;
    double KC = K[in_idx];
    if(BCtype==dirichlet) RHS[in_idx] -= 2.0*Hb*KC/h*A/h/h/h; //dirichlet contribution
}

__global__ void RHS_head_side_north(double *RHS, const double *K, int Nx, int Ny, int Nz, double A, double h, int BCtype, double Hb){
	int ix = threadIdx.x + blockIdx.x*blockDim.x;
	int iz = threadIdx.y + blockIdx.y*blockDim.y;
	if (ix >= Nx-2 || iz >= Nz-2) return;
	int stride = Nx*Ny;
	int in_idx;
	int iy = Ny-1;
	in_idx = (ix + 1) + iy*Nx + (iz + 1)*stride;
    double KC = K[in_idx];
    if(BCtype==dirichlet) RHS[in_idx] -= 2.0*Hb*KC/h*A/h/h/h; //dirichlet contribution
}

__global__ void RHS_head_side_west(double *RHS, const double *K, int Nx, int Ny, int Nz, double A, double h, int BCtype, double Hb){
	int iy = threadIdx.x + blockIdx.x*blockDim.x;
	int iz  = threadIdx.y + blockIdx.y*blockDim.y;
	if (iy >= Ny-2 || iz >= Nz-2) return;
	int stride = Nx*Ny;
	int in_idx;
	int ix = 0;
	in_idx = ix + (iy + 1)*Nx + (iz + 1)*stride; 
    double KC = K[in_idx];
    if(BCtype==dirichlet) RHS[in_idx] -= 2.0*Hb*KC/h*A/h/h/h; //dirichlet contribution
}

__global__ void RHS_head_side_east(double *RHS, const double *K, int Nx, int Ny, int Nz, double A, double h, int BCtype, double Hb){
	int iy = threadIdx.x + blockIdx.x*blockDim.x;
	int iz  = threadIdx.y + blockIdx.y*blockDim.y;
	if (iy >= Ny-2 || iz >= Nz-2) return;
	int stride = Nx*Ny;
	int in_idx;
	int ix = Nx-1;
	in_idx = ix + (iy + 1)*Nx + (iz + 1)*stride;
    double KC = K[in_idx];
    if(BCtype==dirichlet) RHS[in_idx] -= 2.0*Hb*KC/h*A/h/h/h; //dirichlet contribution
}

__global__ void RHS_head_edge_X_South_Bottom(double *RHS, const double *K, int Nx, int Ny, int Nz, double A, double h, int BCsouth, int BCbottom, double Hsouth, double Hbottom){
	int ix = threadIdx.x + blockIdx.x*blockDim.x;
	if (ix >= Nx-2) return;
	int stride = Nx*Ny;
	int iy = 0;
	int iz = 0;
	int in_idx;
	in_idx = (ix+1) + iy*Nx + iz*stride;
    double KC = K[in_idx];
    if(BCsouth==dirichlet) RHS[in_idx] -= 2.0*Hsouth*KC/h*A/h/h/h; //dirichlet contribution
    if(BCbottom==dirichlet) RHS[in_idx] -= 2.0*Hbottom*KC/h*A/h/h/h; //dirichlet contribution
   }


__global__ void RHS_head_edge_X_South_Top(double *RHS, const double *K, int Nx, int Ny, int Nz, double A, double h, int BCsouth, int BCtop, double Hsouth, double Htop){
	int ix = threadIdx.x + blockIdx.x*blockDim.x;
	if (ix >= Nx-2) return;
	int stride = Nx*Ny;
	int iy = 0;
	int iz = Nz-1;
	int in_idx = (ix+1) + iy*Nx + iz*stride;
    double KC = K[in_idx];
    if(BCsouth==dirichlet) RHS[in_idx] -= 2.0*Hsouth*KC/h*A/h/h/h; 
    if(BCtop==dirichlet) RHS[in_idx] -= 2.0*Htop*KC/h*A/h/h/h; 
}

__global__ void RHS_head_edge_X_North_Bottom(double *RHS, const double *K, int Nx, int Ny, int Nz, double A, double h, int BCnorth, int BCbottom, double Hnorth, double Hbottom){
    int ix = threadIdx.x + blockIdx.x*blockDim.x;
    if (ix >= Nx-2) return; 
    int stride = Nx*Ny;
    int iy = Ny-1; 
    int iz = 0; 
    int in_idx = (ix+1) + iy*Nx + iz*stride; 
    double KC = K[in_idx];
    if(BCnorth==dirichlet) RHS[in_idx] -= 2.0*Hnorth*KC/h*A/h/h/h; 
    if(BCbottom==dirichlet) RHS[in_idx] -= 2.0*Hbottom*KC/h*A/h/h/h; 
}

__global__ void RHS_head_edge_X_North_Top(double *RHS, const double *K, int Nx, int Ny, int Nz, double A, double h, int BCnorth, int BCtop, double Hnorth, double Htop){
    int ix = threadIdx.x + blockIdx.x*blockDim.x;
    if (ix >= Nx-2) return; 
    int stride = Nx*Ny;
    int iy = Ny-1; 
    int iz = Nz-1; 
    int in_idx = (ix+1) + iy*Nx + iz*stride; 
    double KC = K[in_idx];
    if(BCnorth==dirichlet) RHS[in_idx] -= 2.0*Hnorth*KC/h*A/h/h/h; 
    if(BCtop==dirichlet) RHS[in_idx] -= 2.0*Htop*KC/h*A/h/h/h; 
}   

__global__ void RHS_head_edge_Z_South_West(double *RHS, const double *K, int Nx, int Ny, int Nz, double A, double h, int BCsouth, int BCwest, double Hsouth, double Hwest){
	int iz = threadIdx.x + blockIdx.x*blockDim.x;
	if (iz >= Nz-2) return;
	int stride = Nx*Ny;
	int ix = 0;
	int iy = 0;
	int in_idx;
	in_idx = ix + iy*Nx + (iz+1)*stride;
    double KC = K[in_idx];
    if(BCsouth==dirichlet) RHS[in_idx] -= 2.0*Hsouth*KC/h*A/h/h/h; 
    if(BCwest==dirichlet) RHS[in_idx] -= 2.0*Hwest*KC/h*A/h/h/h; 
}

__global__ void RHS_head_edge_Z_South_East(double *RHS, const double *K, int Nx, int Ny, int Nz, double A, double h, int BCsouth, int BCeast, double Hsouth, double Heast){
	int iz = threadIdx.x + blockIdx.x*blockDim.x;
	if (iz >= Nz-2) return;
	int stride = Nx*Ny;
	int ix = Nx-1;
	int iy = 0;
	int in_idx;
	in_idx = ix + iy*Nx + (iz+1)*stride;
    double KC = K[in_idx];
    if(BCsouth==dirichlet) RHS[in_idx] -= 2.0*Hsouth*KC/h*A/h/h/h; 
    if(BCeast==dirichlet) RHS[in_idx] -= 2.0*Heast*KC/h*A/h/h/h; 
}

__global__ void RHS_head_edge_Z_North_West(double *RHS, const double *K, int Nx, int Ny, int Nz, double A, double h, int BCnorth, int BCwest, double Hnorth, double Hwest){
	int iz = threadIdx.x + blockIdx.x*blockDim.x;
	if (iz >= Nz-2) return;
	int stride = Nx*Ny;
	int ix = 0;
	int iy = Ny-1;
	int in_idx;
	in_idx = ix + iy*Nx + (iz+1)*stride;
    double KC = K[in_idx];
    if(BCnorth==dirichlet) RHS[in_idx] -= 2.0*Hnorth*KC/h*A/h/h/h; 
    if(BCwest==dirichlet) RHS[in_idx] -= 2.0*Hwest*KC/h*A/h/h/h; 
}

__global__ void RHS_head_edge_Z_North_East(double *RHS, const double *K, int Nx, int Ny, int Nz, double A, double h, int BCnorth, int BCeast, double Hnorth, double Heast){
	int iz = threadIdx.x + blockIdx.x*blockDim.x;
	if (iz >= Nz-2) return;
	int stride = Nx*Ny;
	int ix = Nx-1;
	int iy = Ny-1;
	int in_idx;
	in_idx = ix + iy*Nx + (iz+1)*stride;
    double KC = K[in_idx];
    if(BCnorth==dirichlet) RHS[in_idx] -= 2.0*Hnorth*KC/h*A/h/h/h; 
    if(BCeast==dirichlet) RHS[in_idx] -= 2.0*Heast*KC/h*A/h/h/h; 
}

__global__ void RHS_head_edge_Y_Bottom_West(double *RHS, const double *K, int Nx, int Ny, int Nz, double A, double h, int BCbottom, int BCwest, double Hbottom, double Hwest){
	int iy = threadIdx.x + blockIdx.x*blockDim.x;
	if (iy >= Ny-2) return;
	int stride = Nx*Ny;
	int ix = 0;
	int iz = 0;
	int in_idx;
	in_idx = ix + (iy+1)*Nx + iz*stride;
    double KC = K[in_idx];
    if(BCwest==dirichlet) RHS[in_idx] -= 2.0*Hwest*KC/h*A/h/h/h; 
    if(BCbottom==dirichlet) RHS[in_idx] -= 2.0*Hbottom*KC/h*A/h/h/h; 
}

__global__ void RHS_head_edge_Y_Top_West(double *RHS, const double *K, int Nx, int Ny, int Nz, double A, double h, int BCtop, int BCwest, double Htop, double Hwest){
	int iy = threadIdx.x + blockIdx.x*blockDim.x;
	if (iy >= Ny-2) return;
	int stride = Nx*Ny;
	int ix = 0;
	int iz = Nz-1;
	int in_idx;
	in_idx = ix + (iy+1)*Nx + iz*stride;
    double KC = K[in_idx];
    if(BCwest==dirichlet) RHS[in_idx] -= 2.0*Hwest*KC/h*A/h/h/h; 
    if(BCtop==dirichlet) RHS[in_idx] -= 2.0*Htop*KC/h*A/h/h/h; 
}

__global__ void RHS_head_edge_Y_Bottom_East(double *RHS, const double *K, int Nx, int Ny, int Nz, double A, double h, int BCbottom, int BCeast, double Hbottom, double Heast){
	int iy = threadIdx.x + blockIdx.x*blockDim.x;
	if (iy >= Ny-2) return;
	int stride = Nx*Ny;
	int ix = Nx-1;
	int iz = 0;
	int in_idx;
	in_idx = ix + (iy+1)*Nx + iz*stride;
    double KC = K[in_idx];
    if(BCeast==dirichlet) RHS[in_idx] -= 2.0*Heast*KC/h*A/h/h/h; 
    if(BCbottom==dirichlet) RHS[in_idx] -= 2.0*Hbottom*KC/h*A/h/h/h; 
}

__global__ void RHS_head_edge_Y_Top_East(double *RHS, const double *K, int Nx, int Ny, int Nz, double A, double h, int BCtop, int BCeast, double Htop, double Heast){
	int iy = threadIdx.x + blockIdx.x*blockDim.x;
	if (iy >= Ny-2) return;
	int stride = Nx*Ny;
	int ix = Nx-1;
	int iz = Nz-1;
	int in_idx;
	in_idx = ix + (iy+1)*Nx + iz*stride;
    double KC = K[in_idx];
    if(BCeast==dirichlet) RHS[in_idx] -= 2.0*Heast*KC/h*A/h/h/h; 
    if(BCtop==dirichlet) RHS[in_idx] -= 2.0*Htop*KC/h*A/h/h/h; 
}

__global__ void RHS_head_vertex_SWB(double *RHS, const double *K, int Nx, int Ny, int Nz, double A, double h, 
                                  int BCsouth, int BCwest, int BCbottom, 
                                  double Hsouth, double Hwest, double Hbottom) {
    int ix = 0;
    int iy = 0;
    int iz = 0;
    int stride = Nx*Ny;
    int in_idx = ix + iy*Nx + iz*stride;
    double KC = K[in_idx];
    if(BCsouth==dirichlet) RHS[in_idx] -= 2.0*Hsouth*KC/h*A/h/h/h;
    if(BCwest==dirichlet) RHS[in_idx] -= 2.0*Hwest*KC/h*A/h/h/h;
    if(BCbottom==dirichlet) RHS[in_idx] -= 2.0*Hbottom*KC/h*A/h/h/h;
}

__global__ void RHS_head_vertex_SWT(double *RHS, const double *K, int Nx, int Ny, int Nz, double A, double h, 
                                  int BCsouth, int BCwest, int BCtop, 
                                  double Hsouth, double Hwest, double Htop) {
	int stride = Nx*Ny;
	int ix = 0;
	int iy = 0;
	int iz = Nz-1;
	int in_idx;
	in_idx = ix + iy*Nx + iz*stride;
    double KC = K[in_idx];
    if(BCsouth==dirichlet) RHS[in_idx] -= 2.0*Hsouth*KC/h*A/h/h/h;
    if(BCwest==dirichlet) RHS[in_idx] -= 2.0*Hwest*KC/h*A/h/h/h;
    if(BCtop==dirichlet) RHS[in_idx] -= 2.0*Htop*KC/h*A/h/h/h;
}

__global__ void RHS_head_vertex_SEB(double *RHS, const double *K, int Nx, int Ny, int Nz, double A, double h, 
                                  int BCsouth, int BCeast, int BCbottom, 
                                  double Hsouth, double Heast, double Hbottom) {
	int stride = Nx*Ny;
	int ix = Nx-1;
	int iy = 0;
	int iz = 0;
	int in_idx;
	in_idx = ix + iy*Nx + iz*stride;
    double KC = K[in_idx];
    if(BCsouth==dirichlet) RHS[in_idx] -= 2.0*Hsouth*KC/h*A/h/h/h;
    if(BCeast==dirichlet) RHS[in_idx] -= 2.0*Heast*KC/h*A/h/h/h;
    if(BCbottom==dirichlet) RHS[in_idx] -= 2.0*Hbottom*KC/h*A/h/h/h;
}

__global__ void RHS_head_vertex_SET(double *RHS, const double *K, int Nx, int Ny, int Nz, double A, double h, 
                                  int BCsouth, int BCeast, int BCtop, 
                                  double Hsouth, double Heast, double Htop) {
	int stride = Nx*Ny;
	int ix = Nx-1;
	int iy = 0;
	int iz = Nz-1;
	int in_idx;
	in_idx = ix + iy*Nx + iz*stride;
    double KC = K[in_idx];
    if(BCsouth==dirichlet) RHS[in_idx] -= 2.0*Hsouth*KC/h*A/h/h/h;
    if(BCeast==dirichlet) RHS[in_idx] -= 2.0*Heast*KC/h*A/h/h/h;
    if(BCtop==dirichlet) RHS[in_idx] -= 2.0*Htop*KC/h*A/h/h/h;
}

__global__ void RHS_head_vertex_NWB(double *RHS, const double *K, int Nx, int Ny, int Nz, double A, double h, 
                                  int BCnorth, int BCwest, int BCbottom, 
                                  double Hnorth, double Hwest, double Hbottom) {
	int stride = Nx*Ny;
	int ix = 0;
	int iy = Ny-1;
	int iz = 0;
	int in_idx;
	in_idx = ix + iy*Nx + iz*stride;
    double KC = K[in_idx];
    if(BCnorth==dirichlet) RHS[in_idx] -= 2.0*Hnorth*KC/h*A/h/h/h;
    if(BCwest==dirichlet) RHS[in_idx] -= 2.0*Hwest*KC/h*A/h/h/h;
    if(BCbottom==dirichlet) RHS[in_idx] -= 2.0*Hbottom*KC/h*A/h/h/h;
}

__global__ void RHS_head_vertex_NWT(double *RHS, const double *K, int Nx, int Ny, int Nz, double A, double h, 
                                  int BCnorth, int BCwest, int BCtop, 
                                  double Hnorth, double Hwest, double Htop) {
	int stride = Nx*Ny;
	int ix = 0;
	int iy = Ny-1;
	int iz = Nz-1;
	int in_idx;
	in_idx = ix + iy*Nx + iz*stride;

    double KC = K[in_idx];
    if(BCnorth==dirichlet) RHS[in_idx] -= 2.0*Hnorth*KC/h*A/h/h/h;
    if(BCwest==dirichlet) RHS[in_idx] -= 2.0*Hwest*KC/h*A/h/h/h;
    if(BCtop==dirichlet) RHS[in_idx] -= 2.0*Htop*KC/h*A/h/h/h;
}

__global__ void RHS_head_vertex_NEB(double *RHS, const double *K, int Nx, int Ny, int Nz, double A, double h, 
                                  int BCnorth, int BCeast, int BCbottom, 
                                  double Hnorth, double Heast, double Hbottom) {
	int stride = Nx*Ny;
	int ix = Nx-1;
	int iy = Ny-1;
	int iz = 0;
	int in_idx;
	in_idx = ix + iy*Nx + iz*stride;
    double KC = K[in_idx];
    if(BCnorth==dirichlet) RHS[in_idx] -= 2.0*Hnorth*KC/h*A/h/h/h;
    if(BCeast==dirichlet) RHS[in_idx] -= 2.0*Heast*KC/h*A/h/h/h;
    if(BCbottom==dirichlet) RHS[in_idx] -= 2.0*Hbottom*KC/h*A/h/h/h;
}

__global__ void RHS_head_vertex_NET(double *RHS, const double *K, int Nx, int Ny, int Nz, double A, double h, 
                                  int BCnorth, int BCeast, int BCtop, 
                                  double Hnorth, double Heast, double Htop) {
	int stride = Nx*Ny;
	int ix = Nx-1;
	int iy = Ny-1;
	int iz = Nz-1;
	int in_idx;
	in_idx = ix + iy*Nx + iz*stride;
    double KC = K[in_idx];
    if(BCnorth==dirichlet) RHS[in_idx] -= 2.0*Hnorth*KC/h*A/h/h/h;
    if(BCeast==dirichlet) RHS[in_idx] -= 2.0*Heast*KC/h*A/h/h/h;
    if(BCtop==dirichlet) RHS[in_idx] -= 2.0*Htop*KC/h*A/h/h/h;
}

void RHS_head(double *RHS, const double *K,
	int Nx, int Ny, int Nz, double A, double h,
	int BCbottom, int BCtop, int BCsouth, int BCnorth, int BCwest, int BCeast, 
	double Hbottom, double Htop, double Hsouth, double Hnorth, double Hwest, double Heast, 
	dim3 grid, dim3 block){
	RHS_head_int		<<<gridXY,blockXY>>>(RHS,K,Nx,Ny,Nz);
	RHS_head_side_bottom<<<gridXY,blockXY>>>(RHS,K,Nx,Ny,Nz,A,h,BCbottom,Hbottom);
	RHS_head_side_top	<<<gridXY,blockXY>>>(RHS,K,Nx,Ny,Nz,A,h,BCtop,Htop);
	RHS_head_side_south<<<gridXZ,blockXZ>>>(RHS,K,Nx,Ny,Nz,A,h,BCsouth,Hsouth);
	RHS_head_side_north<<<gridXZ,blockXZ>>>(RHS,K,Nx,Ny,Nz,A,h,BCnorth,Hnorth);
	RHS_head_side_west<<<gridYZ,blockYZ>>>(RHS,K,Nx,Ny,Nz,A,h,BCwest,Hwest);
	RHS_head_side_east<<<gridYZ,blockYZ>>>(RHS,K,Nx,Ny,Nz,A,h,BCeast,Heast);
	RHS_head_edge_X_South_Bottom<<<grid.x,block.x>>>(RHS,K,Nx,Ny,Nz,A,h,BCsouth,BCbottom,Hsouth,Hbottom);
	RHS_head_edge_X_South_Top	<<<grid.x,block.x>>>(RHS,K,Nx,Ny,Nz,A,h,BCsouth,BCtop,Hsouth,Htop);
	RHS_head_edge_X_North_Bottom<<<grid.x,block.x>>>(RHS,K,Nx,Ny,Nz,A,h,BCnorth,BCbottom,Hnorth,Hbottom);
	RHS_head_edge_X_North_Top	<<<grid.x,block.x>>>(RHS,K,Nx,Ny,Nz,A,h,BCnorth,BCtop,Hnorth,Htop);
	RHS_head_edge_Z_South_West<<<grid.z,block.z>>>(RHS,K,Nx,Ny,Nz,A,h,BCsouth,BCwest,Hsouth,Hwest);
	RHS_head_edge_Z_South_East<<<grid.z,block.z>>>(RHS,K,Nx,Ny,Nz,A,h,BCsouth,BCeast,Hsouth,Heast);
	RHS_head_edge_Z_North_West<<<grid.z,block.z>>>(RHS,K,Nx,Ny,Nz,A,h,BCnorth,BCwest,Hnorth,Hwest);
	RHS_head_edge_Z_North_East<<<grid.z,block.z>>>(RHS,K,Nx,Ny,Nz,A,h,BCnorth,BCeast,Hnorth,Heast);
	RHS_head_edge_Y_Bottom_West	<<<grid.y,block.y>>>(RHS,K,Nx,Ny,Nz,A,h,BCbottom,BCwest,Hbottom,Hwest);
	RHS_head_edge_Y_Top_West	<<<grid.y,block.y>>>(RHS,K,Nx,Ny,Nz,A,h,BCtop,BCwest,Htop,Hwest);
	RHS_head_edge_Y_Bottom_East	<<<grid.y,block.y>>>(RHS,K,Nx,Ny,Nz,A,h,BCbottom,BCeast,Hbottom,Heast);
	RHS_head_edge_Y_Top_East	<<<grid.y,block.y>>>(RHS,K,Nx,Ny,Nz,A,h,BCtop,BCeast,Htop,Heast);
	RHS_head_vertex_SWB<<<1,1>>>(RHS,K,Nx,Ny,Nz,A,h,BCsouth,BCwest,BCbottom,Hsouth,Hwest,Hbottom);
	RHS_head_vertex_SWT<<<1,1>>>(RHS,K,Nx,Ny,Nz,A,h,BCsouth,BCwest,BCtop,Hsouth,Hwest,Htop);
	RHS_head_vertex_SEB<<<1,1>>>(RHS,K,Nx,Ny,Nz,A,h,BCsouth,BCeast,BCbottom,Hsouth,Heast,Hbottom);
	RHS_head_vertex_SET<<<1,1>>>(RHS,K,Nx,Ny,Nz,A,h,BCsouth,BCeast,BCtop,Hsouth,Heast,Htop);
	RHS_head_vertex_NWB<<<1,1>>>(RHS,K,Nx,Ny,Nz,A,h,BCnorth,BCwest,BCbottom,Hnorth,Hwest,Hbottom);
	RHS_head_vertex_NWT<<<1,1>>>(RHS,K,Nx,Ny,Nz,A,h,BCnorth,BCwest,BCtop,Hnorth,Hwest,Htop);
	RHS_head_vertex_NEB<<<1,1>>>(RHS,K,Nx,Ny,Nz,A,h,BCnorth,BCeast,BCbottom,Hnorth,Heast,Hbottom);
	RHS_head_vertex_NET<<<1,1>>>(RHS,K,Nx,Ny,Nz,A,h,BCnorth,BCeast,BCtop,Hnorth,Heast,Htop);

	cudaDeviceSynchronize();
}