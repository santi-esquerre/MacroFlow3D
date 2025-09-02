#define COMPUTE_INDEX_NORMAL_VELOCITY_BOTTOM \
int idx_U = (ix+1+1) + (iy+1)*(Nx+1) + (iz)*(Nx+1)*Ny; \
int idx_V = (ix+1)   + (iy+1+1)*(Nx) + (iz)*Nx*(Ny+1); \
int idx_W = (ix+1)   + (iy+1)*(Nx)   + (iz+1)*stride;
#define COMPUTE_INDEX_NORMAL_VELOCITY_TOP COMPUTE_INDEX_NORMAL_VELOCITY_BOTTOM

#define COMPUTE_INDEX_NORMAL_VELOCITY_SOUTH \
int idx_U = (ix+1+1) + (iy)*(Nx+1) + (iz+1)*(Nx+1)*Ny; \
int idx_V = (ix+1)   + (iy+1)*(Nx) + (iz+1)*Nx*(Ny+1); \
int idx_W = (ix+1)   + (iy)*(Nx)   + (iz+1+1)*stride;
#define COMPUTE_INDEX_NORMAL_VELOCITY_NORTH COMPUTE_INDEX_NORMAL_VELOCITY_SOUTH

#define COMPUTE_INDEX_NORMAL_VELOCITY_WEST \
int idx_U = (ix+1) + (iy+1)*(Nx+1) + (iz+1)*(Nx+1)*Ny; \
int idx_V = (ix)   + (iy+1+1)*(Nx) + (iz+1)*Nx*(Ny+1); \
int idx_W = (ix)   + (iy+1)*(Nx)   + (iz+1+1)*stride;
#define COMPUTE_INDEX_NORMAL_VELOCITY_EAST COMPUTE_INDEX_NORMAL_VELOCITY_WEST

#define COMPUTE_INDEX_NORMAL_VELOCITY_SOUTH_BOTTOM \
int idx_U = (ix+1+1) + (iy)*(Nx+1) + (iz)*(Nx+1)*Ny; \
int idx_V = (ix+1)   + (iy+1)*(Nx) + (iz)*Nx*(Ny+1); \
int idx_W = (ix+1)   + (iy)*(Nx)   + (iz+1)*stride;
#define COMPUTE_INDEX_NORMAL_VELOCITY_SOUTH_TOP COMPUTE_INDEX_NORMAL_VELOCITY_SOUTH_BOTTOM
#define COMPUTE_INDEX_NORMAL_VELOCITY_NORTH_BOTTOM COMPUTE_INDEX_NORMAL_VELOCITY_SOUTH_BOTTOM
#define COMPUTE_INDEX_NORMAL_VELOCITY_NORTH_TOP COMPUTE_INDEX_NORMAL_VELOCITY_SOUTH_BOTTOM

#define COMPUTE_INDEX_NORMAL_VELOCITY_WEST_BOTTOM \
int idx_U = (ix+1) + (iy+1)*(Nx+1) + (iz)*(Nx+1)*Ny; \
int idx_V = (ix)   + (iy+1+1)*(Nx) + (iz)*Nx*(Ny+1); \
int idx_W = (ix)   + (iy+1)*(Nx)   + (iz+1)*stride;
#define COMPUTE_INDEX_NORMAL_VELOCITY_WEST_TOP COMPUTE_INDEX_NORMAL_VELOCITY_WEST_BOTTOM
#define COMPUTE_INDEX_NORMAL_VELOCITY_EAST_BOTTOM COMPUTE_INDEX_NORMAL_VELOCITY_WEST_BOTTOM
#define COMPUTE_INDEX_NORMAL_VELOCITY_EAST_TOP COMPUTE_INDEX_NORMAL_VELOCITY_WEST_BOTTOM

#define COMPUTE_INDEX_NORMAL_VELOCITY_WEST_SOUTH \
int idx_U = (ix+1) + (iy)*(Nx+1) + (iz+1)*(Nx+1)*Ny; \
int idx_V = (ix)   + (iy+1)*(Nx) + (iz+1)*Nx*(Ny+1); \
int idx_W = (ix)   + (iy)*(Nx)   + (iz+1+1)*stride;
#define COMPUTE_INDEX_NORMAL_VELOCITY_EAST_SOUTH COMPUTE_INDEX_NORMAL_VELOCITY_WEST_SOUTH
#define COMPUTE_INDEX_NORMAL_VELOCITY_WEST_NORTH COMPUTE_INDEX_NORMAL_VELOCITY_WEST_SOUTH
#define COMPUTE_INDEX_NORMAL_VELOCITY_EAST_NORTH COMPUTE_INDEX_NORMAL_VELOCITY_WEST_SOUTH

#define COMPUTE_INDEX_NORMAL_VELOCITY_WEST_SOUTH_BOTTOM \
int idx_U = (ix+1) + (iy)*(Nx+1) + (iz)*(Nx+1)*Ny; \
int idx_V = (ix)   + (iy+1)*(Nx) + (iz)*Nx*(Ny+1); \
int idx_W = (ix)   + (iy)*(Nx)   + (iz+1)*stride;
#define COMPUTE_INDEX_NORMAL_VELOCITY_WEST_SOUTH_TOP COMPUTE_INDEX_NORMAL_VELOCITY_WEST_SOUTH_BOTTOM
#define COMPUTE_INDEX_NORMAL_VELOCITY_WEST_NORTH_BOTTOM COMPUTE_INDEX_NORMAL_VELOCITY_WEST_SOUTH_BOTTOM
#define COMPUTE_INDEX_NORMAL_VELOCITY_WEST_NORTH_TOP COMPUTE_INDEX_NORMAL_VELOCITY_WEST_SOUTH_BOTTOM
#define COMPUTE_INDEX_NORMAL_VELOCITY_EAST_SOUTH_BOTTOM COMPUTE_INDEX_NORMAL_VELOCITY_WEST_SOUTH_BOTTOM
#define COMPUTE_INDEX_NORMAL_VELOCITY_EAST_SOUTH_TOP COMPUTE_INDEX_NORMAL_VELOCITY_WEST_SOUTH_BOTTOM
#define COMPUTE_INDEX_NORMAL_VELOCITY_EAST_NORTH_BOTTOM COMPUTE_INDEX_NORMAL_VELOCITY_WEST_SOUTH_BOTTOM
#define COMPUTE_INDEX_NORMAL_VELOCITY_EAST_NORTH_TOP COMPUTE_INDEX_NORMAL_VELOCITY_WEST_SOUTH_BOTTOM

// The order must match with the OFFSETS array as they are iterated in the same loop (see macros_indez_kernel.h)
// #define OFFSETS_BOTTOM int offsets[5] = { -1, 1, -Nx, Nx,  stride }
// #define OFFSETS_TOP int    offsets[5] = { -1, 1, -Nx, Nx, -stride }
// #define OFFSETS_EAST int offsets[5] = { -1, -Nx,  Nx, -stride, stride }
// #define OFFSETS_WEST int offsets[5] = {  1, -Nx,  Nx, -stride, stride }
// #define OFFSETS_NORTH int offsets[5] = { -1,  1, -Nx, -stride,  stride }
// #define OFFSETS_SOUTH int offsets[5] = { -1,  1,  Nx, -stride,  stride }
#define LOAD_FACE_FLUX_BOTTOM double m_f[6] = {-Up[idx_U-1]*A, Up[idx_U]*A, -Vp[idx_V-Nx]*A, Vp[idx_V]*A, Wp[idx_W]*A, -Wp[idx_W-stride]*A}
#define LOAD_FACE_FLUX_TOP double m_f[6] = {-Up[idx_U-1]*A, Up[idx_U]*A, -Vp[idx_V-Nx]*A, Vp[idx_V]*A, -Wp[idx_W-stride]*A, Wp[idx_W]*A}
#define LOAD_FACE_FLUX_EAST double m_f[6] = {-Up[idx_U-1]*A, -Vp[idx_V-Nx]*A, Vp[idx_V]*A, -Wp[idx_W-stride]*A, Wp[idx_W]*A, Up[idx_U]*A}
#define LOAD_FACE_FLUX_WEST double m_f[6] = { Up[idx_U]*A, -Vp[idx_V-Nx]*A, Vp[idx_V]*A, -Wp[idx_W-stride]*A, Wp[idx_W]*A, -Up[idx_U-1]*A}
#define LOAD_FACE_FLUX_NORTH double m_f[6] = {-Up[idx_U-1]*A, Up[idx_U]*A, -Vp[idx_V-Nx]*A, -Wp[idx_W-stride]*A, Wp[idx_W]*A, Vp[idx_V]*A}
#define LOAD_FACE_FLUX_SOUTH double m_f[6] = {-Up[idx_U-1]*A, Up[idx_U]*A, Vp[idx_V]*A, -Wp[idx_W-stride]*A, Wp[idx_W]*A, -Vp[idx_V-Nx]*A}

// edge x
#define LOAD_FACE_FLUX_SOUTH_BOTTOM double m_f[6] = {-Up[idx_U-1]*A, Up[idx_U]*A, Vp[idx_V]*A, Wp[idx_W]*A, -Vp[idx_V-Nx]*A, -Wp[idx_W-stride]*A}
#define LOAD_FACE_FLUX_SOUTH_TOP    double m_f[6] = {-Up[idx_U-1]*A, Up[idx_U]*A, Vp[idx_V]*A, -Wp[idx_W-stride]*A, -Vp[idx_V-Nx]*A, Wp[idx_W]*A}
#define LOAD_FACE_FLUX_NORTH_BOTTOM double m_f[6] = {-Up[idx_U-1]*A, Up[idx_U]*A, -Vp[idx_V-Nx]*A, Wp[idx_W]*A, Vp[idx_V]*A, -Wp[idx_W-stride]*A}
#define LOAD_FACE_FLUX_NORTH_TOP    double m_f[6] = {-Up[idx_U-1]*A, Up[idx_U]*A, -Vp[idx_V-Nx]*A, -Wp[idx_W-stride]*A, Vp[idx_V]*A, Wp[idx_W]*A}
// edge y
#define LOAD_FACE_FLUX_WEST_BOTTOM double m_f[6] = { Up[idx_U]*A, -Vp[idx_V-Nx]*A, Vp[idx_V]*A, Wp[idx_W]*A, -Up[idx_U-1]*A, -Wp[idx_W-stride]*A}
#define LOAD_FACE_FLUX_WEST_TOP    double m_f[6] = { Up[idx_U]*A, -Vp[idx_V-Nx]*A, Vp[idx_V]*A, -Wp[idx_W-stride]*A, -Up[idx_U-1]*A, Wp[idx_W]*A}
#define LOAD_FACE_FLUX_EAST_BOTTOM double m_f[6] = {-Up[idx_U-1]*A, -Vp[idx_V-Nx]*A, Vp[idx_V]*A, Wp[idx_W]*A, Up[idx_U]*A, -Wp[idx_W-stride]*A}
#define LOAD_FACE_FLUX_EAST_TOP    double m_f[6] = {-Up[idx_U-1]*A, -Vp[idx_V-Nx]*A, Vp[idx_V]*A, -Wp[idx_W-stride]*A, Up[idx_U]*A, Wp[idx_W]*A}

// edge z
#define LOAD_FACE_FLUX_WEST_SOUTH double m_f[6] = {Up[idx_U]*A, Vp[idx_V]*A, -Wp[idx_W-stride]*A, Wp[idx_W]*A, -Up[idx_U-1]*A, -Vp[idx_V-Nx]*A}
#define LOAD_FACE_FLUX_WEST_NORTH double m_f[6] = {Up[idx_U]*A, -Vp[idx_V-Nx]*A, -Wp[idx_W-stride]*A, Wp[idx_W]*A, -Up[idx_U-1]*A, Vp[idx_V]*A}
#define LOAD_FACE_FLUX_EAST_SOUTH double m_f[6] = {-Up[idx_U-1]*A, Vp[idx_V]*A, -Wp[idx_W-stride]*A, Wp[idx_W]*A, Up[idx_U]*A, -Vp[idx_V-Nx]*A}
#define LOAD_FACE_FLUX_EAST_NORTH double m_f[6] = {-Up[idx_U-1]*A, -Vp[idx_V-Nx]*A, -Wp[idx_W-stride]*A, Wp[idx_W]*A, Up[idx_U]*A, Vp[idx_V]*A}

#define LOAD_FACE_FLUX_WEST_SOUTH_BOTTOM double m_f[6] = { Up[idx_U]*A, Vp[idx_V]*A, Wp[idx_W]*A, -Up[idx_U-1]*A, -Vp[idx_V-Nx]*A, -Wp[idx_W-stride]*A}
#define LOAD_FACE_FLUX_WEST_SOUTH_TOP    double m_f[6] = { Up[idx_U]*A, Vp[idx_V]*A, -Wp[idx_W-stride]*A, -Up[idx_U-1]*A, -Vp[idx_V-Nx]*A, Wp[idx_W]*A}
#define LOAD_FACE_FLUX_WEST_NORTH_BOTTOM double m_f[6] = { Up[idx_U]*A, -Vp[idx_V-Nx]*A, Wp[idx_W]*A, -Up[idx_U-1]*A, Vp[idx_V]*A, -Wp[idx_W-stride]*A}
#define LOAD_FACE_FLUX_WEST_NORTH_TOP    double m_f[6] = { Up[idx_U]*A, -Vp[idx_V-Nx]*A, -Wp[idx_W-stride]*A, -Up[idx_U-1]*A, Vp[idx_V]*A, Wp[idx_W]*A}
#define LOAD_FACE_FLUX_EAST_SOUTH_BOTTOM double m_f[6] = {-Up[idx_U-1]*A, Vp[idx_V]*A, Wp[idx_W]*A, Up[idx_U]*A, -Vp[idx_V-Nx]*A, -Wp[idx_W-stride]*A}
#define LOAD_FACE_FLUX_EAST_SOUTH_TOP    double m_f[6] = {-Up[idx_U-1]*A, Vp[idx_V]*A, -Wp[idx_W-stride]*A, Up[idx_U]*A, -Vp[idx_V-Nx]*A, Wp[idx_W]*A}
#define LOAD_FACE_FLUX_EAST_NORTH_BOTTOM double m_f[6] = {-Up[idx_U-1]*A, -Vp[idx_V-Nx]*A, Wp[idx_W]*A, Up[idx_U]*A, Vp[idx_V]*A, -Wp[idx_W-stride]*A}
#define LOAD_FACE_FLUX_EAST_NORTH_TOP    double m_f[6] = {-Up[idx_U-1]*A, -Vp[idx_V-Nx]*A, -Wp[idx_W-stride]*A, Up[idx_U]*A, Vp[idx_V]*A, Wp[idx_W]*A}
