#define stride (Nx*Ny)
// #define OFFSETS5(FACE, Nx, Ny) OFFSETS_##FACE(Nx, Ny)
#define OFFSETS_BOTTOM int offsets[5] = { -1, 1, -Nx, Nx,  stride }
#define OFFSETS_TOP int    offsets[5] = { -1, 1, -Nx, Nx, -stride }
#define OFFSETS_EAST int offsets[5] = { -1, -Nx,  Nx, -stride, stride }
#define OFFSETS_WEST int offsets[5] = {  1, -Nx,  Nx, -stride, stride }
#define OFFSETS_NORTH int offsets[5] = { -1,  1, -Nx, -stride,  stride }
#define OFFSETS_SOUTH int offsets[5] = { -1,  1,  Nx, -stride,  stride }

// #define OFFSETS4(FACE1,FACE2,Nx,Ny) OFFSETS_##FACE1##_##FACE2(Nx, Ny)se
// edge x
#define OFFSETS_SOUTH_BOTTOM int offsets[4] = { -1,  1,  Nx,  stride }
#define OFFSETS_SOUTH_TOP    int offsets[4] = { -1,  1,  Nx, -stride }
#define OFFSETS_NORTH_BOTTOM int offsets[4] = { -1,  1, -Nx,  stride }
#define OFFSETS_NORTH_TOP    int offsets[4] = { -1,  1, -Nx, -stride }

// edge y
#define OFFSETS_WEST_BOTTOM int offsets[4] = { 1, -Nx, Nx,  stride }
#define OFFSETS_WEST_TOP    int offsets[4] = { 1, -Nx, Nx, -stride }
#define OFFSETS_EAST_BOTTOM int offsets[4] = {-1, -Nx, Nx,  stride }
#define OFFSETS_EAST_TOP    int offsets[4] = {-1, -Nx, Nx, -stride }

// edge z
#define OFFSETS_WEST_SOUTH int offsets[4] = {  1,  Nx, -stride, stride }
#define OFFSETS_WEST_NORTH int offsets[4] = {  1, -Nx, -stride, stride }
#define OFFSETS_EAST_SOUTH int offsets[4] = { -1,  Nx, -stride, stride }
#define OFFSETS_EAST_NORTH int offsets[4] = { -1, -Nx, -stride, stride }

// #define OFFSETS3(FACE1,FACE2,FACE3,Nx,Ny) OFFSETS_##FACE1##_##FACE2##_##FACE3(Nx, Ny)
// vertex
#define OFFSETS_WEST_SOUTH_BOTTOM int offsets[3] = { 1,  Nx,  stride }
#define OFFSETS_WEST_SOUTH_TOP    int offsets[3] = { 1,  Nx, -stride }
#define OFFSETS_WEST_NORTH_BOTTOM int offsets[3] = { 1, -Nx,  stride }
#define OFFSETS_WEST_NORTH_TOP    int offsets[3] = { 1, -Nx, -stride }
#define OFFSETS_EAST_SOUTH_BOTTOM int offsets[3] = {-1,  Nx,  stride }
#define OFFSETS_EAST_SOUTH_TOP    int offsets[3] = {-1,  Nx, -stride }
#define OFFSETS_EAST_NORTH_BOTTOM int offsets[3] = {-1, -Nx,  stride }
#define OFFSETS_EAST_NORTH_TOP    int offsets[3] = {-1, -Nx, -stride }

#define PERIODIC_BOTTOM ((Nz-1)*stride)
#define PERIODIC_TOP (-(Nz-1)*stride)
#define PERIODIC_WEST (Nx-1)
#define PERIODIC_EAST (-(Nx-1))
#define PERIODIC_SOUTH ((Ny-1)*Nx)
#define PERIODIC_NORTH (-(Ny-1)*Nx)

#define COMPUTE_INDEX_BOTTOM \
int ix = threadIdx.x + blockIdx.x*blockDim.x; \
int iy = threadIdx.y + blockIdx.y*blockDim.y; \
if (ix >= Nx-2 || iy >= Ny-2) return; \
int iz=0; \
int in_idx = (ix+1) + (iy+1)*Nx + (iz)*stride;

#define COMPUTE_INDEX_TOP \
int ix = threadIdx.x + blockIdx.x*blockDim.x; \
int iy = threadIdx.y + blockIdx.y*blockDim.y; \
if (ix >= Nx-2 || iy >= Ny-2) return; \
int iz=Nz-1; \
int in_idx = (ix+1) + (iy+1)*Nx + (iz)*stride;

#define COMPUTE_INDEX_SOUTH \
int ix = threadIdx.x + blockIdx.x*blockDim.x; \
int iz = threadIdx.y + blockIdx.y*blockDim.y; \
if (ix >= Nx-2 || iz >= Nz-2) return; \
int iy=0; \
int in_idx = (ix+1) + (iy)*Nx + (iz+1)*stride;

#define COMPUTE_INDEX_NORTH \
int ix = threadIdx.x + blockIdx.x*blockDim.x; \
int iz = threadIdx.y + blockIdx.y*blockDim.y; \
if (ix >= Nx-2 || iz >= Nz-2) return; \
int iy=Ny-1; \
int in_idx = (ix+1) + (iy)*Nx + (iz+1)*stride;

#define COMPUTE_INDEX_WEST \
int iy = threadIdx.x + blockIdx.x*blockDim.x; \
int iz = threadIdx.y + blockIdx.y*blockDim.y; \
if (iy >= Ny-2 || iz >= Nz-2) return; \
int ix=0; \
int in_idx = (ix) + (iy+1)*Nx + (iz+1)*stride;

#define COMPUTE_INDEX_EAST \
int iy = threadIdx.x + blockIdx.x*blockDim.x; \
int iz = threadIdx.y + blockIdx.y*blockDim.y; \
if (iy >= Ny-2 || iz >= Nz-2) return; \
int ix=Nx-1; \
int in_idx = (ix) + (iy+1)*Nx + (iz+1)*stride;

#define COMPUTE_INDEX_SOUTH_BOTTOM \
int ix = threadIdx.x + blockIdx.x*blockDim.x; \
if (ix >= Nx-2) return; \
int iy = 0; \
int iz = 0; \
int in_idx = (ix+1) + iy*Nx + iz*stride;

#define COMPUTE_INDEX_SOUTH_TOP \
int ix = threadIdx.x + blockIdx.x*blockDim.x; \
if (ix >= Nx-2) return; \
int iy = 0; \
int iz = Nz-1; \
int in_idx = (ix+1) + iy*Nx + iz*stride;

#define COMPUTE_INDEX_NORTH_BOTTOM \
int ix = threadIdx.x + blockIdx.x*blockDim.x; \
if (ix >= Nx-2) return; \
int iy = Ny-1; \
int iz = 0; \
int in_idx = (ix+1) + iy*Nx + iz*stride;

#define COMPUTE_INDEX_NORTH_TOP \
int ix = threadIdx.x + blockIdx.x*blockDim.x; \
if (ix >= Nx-2) return; \
int iy = Ny-1; \
int iz = Nz-1; \
int in_idx = (ix+1) + iy*Nx + iz*stride;

#define COMPUTE_INDEX_WEST_BOTTOM \
int iy = threadIdx.x + blockIdx.x*blockDim.x; \
if (iy >= Ny-2) return; \
int ix = 0; \
int iz = 0; \
int in_idx = ix + (iy+1)*Nx + iz*stride;

#define COMPUTE_INDEX_WEST_TOP \
int iy = threadIdx.x + blockIdx.x*blockDim.x; \
if (iy >= Ny-2) return; \
int ix = 0; \
int iz = Nz-1; \
int in_idx = ix + (iy+1)*Nx + iz*stride;

#define COMPUTE_INDEX_EAST_BOTTOM \
int iy = threadIdx.x + blockIdx.x*blockDim.x; \
if (iy >= Ny-2) return; \
int ix = Nx-1; \
int iz = 0; \
int in_idx = ix + (iy+1)*Nx + iz*stride;

#define COMPUTE_INDEX_EAST_TOP \
int iy = threadIdx.x + blockIdx.x*blockDim.x; \
if (iy >= Ny-2) return; \
int ix = Nx-1; \
int iz = Nz-1; \
int in_idx = ix + (iy+1)*Nx + iz*stride;

#define COMPUTE_INDEX_WEST_SOUTH \
int iz = threadIdx.x + blockIdx.x*blockDim.x; \
if (iz >= Nz-2) return; \
int ix = 0; \
int iy = 0; \
int in_idx = ix + iy*Nx + (iz+1)*stride;

#define COMPUTE_INDEX_EAST_SOUTH \
int iz = threadIdx.x + blockIdx.x*blockDim.x; \
if (iz >= Nz-2) return; \
int ix = Nx-1; \
int iy = 0; \
int in_idx = ix + iy*Nx + (iz+1)*stride;

#define COMPUTE_INDEX_WEST_NORTH \
int iz = threadIdx.x + blockIdx.x*blockDim.x; \
if (iz >= Nz-2) return; \
int ix = 0; \
int iy = Ny-1; \
int in_idx = ix + iy*Nx + (iz+1)*stride;

#define COMPUTE_INDEX_EAST_NORTH \
int iz = threadIdx.x + blockIdx.x*blockDim.x; \
if (iz >= Nz-2) return; \
int ix = Nx-1; \
int iy = Ny-1; \
int in_idx = ix + iy*Nx + (iz+1)*stride;

#define COMPUTE_INDEX_WEST_SOUTH_BOTTOM \
int ix = 0; \
int iy = 0; \
int iz = 0; \
int in_idx = ix + iy*Nx + iz*stride;

#define COMPUTE_INDEX_WEST_SOUTH_TOP \
int ix = 0; \
int iy = 0; \
int iz = Nz-1; \
int in_idx = ix + iy*Nx + iz*stride;

#define COMPUTE_INDEX_WEST_NORTH_BOTTOM \
int ix = 0; \
int iy = Ny-1; \
int iz = 0; \
int in_idx = ix + iy*Nx + iz*stride;

#define COMPUTE_INDEX_WEST_NORTH_TOP \
int ix = 0; \
int iy = Ny-1; \
int iz = Nz-1; \
int in_idx = ix + iy*Nx + iz*stride;

#define COMPUTE_INDEX_EAST_SOUTH_BOTTOM \
int ix = Nx-1; \
int iy = 0; \
int iz = 0; \
int in_idx = ix + iy*Nx + iz*stride;

#define COMPUTE_INDEX_EAST_SOUTH_TOP \
int ix = Nx-1; \
int iy = 0; \
int iz = Nz-1; \
int in_idx = ix + iy*Nx + iz*stride;

#define COMPUTE_INDEX_EAST_NORTH_BOTTOM \
int ix = Nx-1; \
int iy = Ny-1; \
int iz = 0; \
int in_idx = ix + iy*Nx + iz*stride;

#define COMPUTE_INDEX_EAST_NORTH_TOP \
int ix = Nx-1; \
int iy = Ny-1; \
int iz = Nz-1; \
int in_idx = ix + iy*Nx + iz*stride;

#define GRIDBLOCK_BOTTOM dim3(grid.x,grid.y),dim3(block.x,block.y)
#define GRIDBLOCK_TOP GRIDBLOCK_BOTTOM
#define GRIDBLOCK_SOUTH dim3(grid.x,grid.z),dim3(block.x,block.z)
#define GRIDBLOCK_NORTH GRIDBLOCK_SOUTH
#define GRIDBLOCK_WEST dim3(grid.y,grid.z),dim3(block.y,block.z)
#define GRIDBLOCK_EAST GRIDBLOCK_WEST

#define GRIDBLOCK_SOUTH_BOTTOM grid.x, block.x
#define GRIDBLOCK_NORTH_BOTTOM GRIDBLOCK_SOUTH_BOTTOM
#define GRIDBLOCK_SOUTH_TOP GRIDBLOCK_SOUTH_BOTTOM
#define GRIDBLOCK_NORTH_TOP GRIDBLOCK_SOUTH_BOTTOM
#define GRIDBLOCK_WEST_SOUTH grid.z, block.z
#define GRIDBLOCK_WEST_NORTH GRIDBLOCK_WEST_SOUTH
#define GRIDBLOCK_EAST_SOUTH GRIDBLOCK_WEST_SOUTH
#define GRIDBLOCK_EAST_NORTH GRIDBLOCK_WEST_SOUTH
#define GRIDBLOCK_WEST_BOTTOM grid.y, block.y
#define GRIDBLOCK_WEST_TOP GRIDBLOCK_WEST_SOUTH
#define GRIDBLOCK_EAST_BOTTOM GRIDBLOCK_WEST_SOUTH
#define GRIDBLOCK_EAST_TOP GRIDBLOCK_WEST_SOUTH
