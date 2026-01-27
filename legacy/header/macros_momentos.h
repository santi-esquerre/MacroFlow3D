#define COMPUTE_MOMENTO_BOTTOM \
momento1x[in_idx] = (float)((ix+1.5)*h)  *  (float)phiC  *(float)(h*h*h); \
momento1y[in_idx] = (float)((iy+1.5)*h ) *  (float)phiC  *(float)(h*h*h); \
momento1z[in_idx] = (float)((iz+0.5)*h)  *  (float)phiC  *(float)(h*h*h); \
momento2x[in_idx] = (float)powf((ix+1.5f)*(float)h,2.0f)*(float)phiC*(float)(h*h*h); \
momento2y[in_idx] = (float)powf((iy+1.5f)*(float)h,2.0f)*(float)phiC*(float)(h*h*h); \
momento2z[in_idx] = (float)powf((iz+0.5f)*(float)h,2.0f) *(float)phiC*(float)(h*h*h); \
C_float[in_idx] = (float)phiC;

#define COMPUTE_MOMENTO_TOP \
momento1x[in_idx] = (float)((ix+1.5)*h)  *  (float)phiC  *(float)(h*h*h); \
momento1y[in_idx] = (float)((iy+1.5)*h ) *  (float)phiC  *(float)(h*h*h); \
momento1z[in_idx] = (float)((iz+0.5)*h)  *  (float)phiC  *(float)(h*h*h); \
momento2x[in_idx] = (float)powf((ix+1.5f)*(float)h,2.0f)*(float)phiC*(float)(h*h*h); \
momento2y[in_idx] = (float)powf((iy+1.5f)*(float)h,2.0f)*(float)phiC*(float)(h*h*h); \
momento2z[in_idx] = (float)powf((iz+0.5f)*(float)h,2.0f) *(float)phiC*(float)(h*h*h); \
C_float[in_idx] = (float)phiC;

#define COMPUTE_MOMENTO_SOUTH \
momento1x[in_idx] = (float)((ix+1.5)*h)  *  (float)phiC  *(float)(h*h*h); \
momento1y[in_idx] = (float)((iy+0.5)*h ) *  (float)phiC  *(float)(h*h*h); \
momento1z[in_idx] = (float)((iz+1.5)*h)  *  (float)phiC  *(float)(h*h*h); \
momento2x[in_idx] = (float)powf((ix+1.5f)*(float)h,2.0f)*(float)phiC*(float)(h*h*h); \
momento2y[in_idx] = (float)powf((iy+0.5f)*(float)h,2.0f)*(float)phiC*(float)(h*h*h); \
momento2z[in_idx] = (float)powf((iz+1.5f)*(float)h,2.0f) *(float)phiC*(float)(h*h*h); \
C_float[in_idx] = (float)phiC;

#define COMPUTE_MOMENTO_NORTH \
momento1x[in_idx] = (float)((ix+1.5)*h)  *  (float)phiC  *(float)(h*h*h); \
momento1y[in_idx] = (float)((iy+0.5)*h ) *  (float)phiC  *(float)(h*h*h); \
momento1z[in_idx] = (float)((iz+1.5)*h)  *  (float)phiC  *(float)(h*h*h); \
momento2x[in_idx] = (float)powf((ix+1.5f)*(float)h,2.0f)*(float)phiC*(float)(h*h*h); \
momento2y[in_idx] = (float)powf((iy+0.5f)*(float)h,2.0f)*(float)phiC*(float)(h*h*h); \
momento2z[in_idx] = (float)powf((iz+1.5f)*(float)h,2.0f) *(float)phiC*(float)(h*h*h); \
C_float[in_idx] = (float)phiC;

#define COMPUTE_MOMENTO_WEST \
momento1x[in_idx] = (float)((ix+0.5)*h)  *  (float)phiC  *(float)(h*h*h); \
momento1y[in_idx] = (float)((iy+1.5)*h ) *  (float)phiC  *(float)(h*h*h); \
momento1z[in_idx] = (float)((iz+1.5)*h)  *  (float)phiC  *(float)(h*h*h); \
momento2x[in_idx] = (float)powf((ix+0.5f)*(float)h,2.0f)*(float)phiC*(float)(h*h*h); \
momento2y[in_idx] = (float)powf((iy+1.5f)*(float)h,2.0f)*(float)phiC*(float)(h*h*h); \
momento2z[in_idx] = (float)powf((iz+1.5f)*(float)h,2.0f) *(float)phiC*(float)(h*h*h); \
C_float[in_idx] = (float)phiC;

#define COMPUTE_MOMENTO_EAST \
momento1x[in_idx] = (float)((ix+0.5)*h)  *  (float)phiC  *(float)(h*h*h); \
momento1y[in_idx] = (float)((iy+1.5)*h ) *  (float)phiC  *(float)(h*h*h); \
momento1z[in_idx] = (float)((iz+1.5)*h)  *  (float)phiC  *(float)(h*h*h); \
momento2x[in_idx] = (float)powf((ix+0.5f)*(float)h,2.0f)*(float)phiC*(float)(h*h*h); \
momento2y[in_idx] = (float)powf((iy+1.5f)*(float)h,2.0f)*(float)phiC*(float)(h*h*h); \
momento2z[in_idx] = (float)powf((iz+1.5f)*(float)h,2.0f) *(float)phiC*(float)(h*h*h); \
C_float[in_idx] = (float)phiC;

#define COMPUTE_MOMENTO_SOUTH_BOTTOM \
momento1x[in_idx] = (float)((ix+1.5)*h)  *  (float)phiC  *(float)(h*h*h); \
momento1y[in_idx] = (float)((iy+0.5)*h ) *  (float)phiC  *(float)(h*h*h); \
momento1z[in_idx] = (float)((iz+0.5)*h)  *  (float)phiC  *(float)(h*h*h); \
momento2x[in_idx] = (float)powf((ix+1.5f)*(float)h,2.0f)*(float)phiC*(float)(h*h*h); \
momento2y[in_idx] = (float)powf((iy+0.5f)*(float)h,2.0f)*(float)phiC*(float)(h*h*h); \
momento2z[in_idx] = (float)powf((iz+0.5f)*(float)h,2.0f) *(float)phiC*(float)(h*h*h); \
C_float[in_idx] = (float)phiC;

#define COMPUTE_MOMENTO_SOUTH_TOP \
momento1x[in_idx] = (float)((ix+1.5)*h)  *  (float)phiC  *(float)(h*h*h); \
momento1y[in_idx] = (float)((iy+0.5)*h ) *  (float)phiC  *(float)(h*h*h); \
momento1z[in_idx] = (float)((iz+0.5)*h)  *  (float)phiC  *(float)(h*h*h); \
momento2x[in_idx] = (float)powf((ix+1.5f)*(float)h,2.0f)*(float)phiC*(float)(h*h*h); \
momento2y[in_idx] = (float)powf((iy+0.5f)*(float)h,2.0f)*(float)phiC*(float)(h*h*h); \
momento2z[in_idx] = (float)powf((iz+0.5f)*(float)h,2.0f) *(float)phiC*(float)(h*h*h); \
C_float[in_idx] = (float)phiC;

#define COMPUTE_MOMENTO_NORTH_BOTTOM \
momento1x[in_idx] = (float)((ix+1.5)*h)  *  (float)phiC  *(float)(h*h*h); \
momento1y[in_idx] = (float)((iy+0.5)*h ) *  (float)phiC  *(float)(h*h*h); \
momento1z[in_idx] = (float)((iz+0.5)*h)  *  (float)phiC  *(float)(h*h*h); \
momento2x[in_idx] = (float)powf((ix+1.5f)*(float)h,2.0f)*(float)phiC*(float)(h*h*h); \
momento2y[in_idx] = (float)powf((iy+0.5f)*(float)h,2.0f)*(float)phiC*(float)(h*h*h); \
momento2z[in_idx] = (float)powf((iz+0.5f)*(float)h,2.0f) *(float)phiC*(float)(h*h*h); \
C_float[in_idx] = (float)phiC;

#define COMPUTE_MOMENTO_NORTH_TOP \
momento1x[in_idx] = (float)((ix+1.5)*h)  *  (float)phiC  *(float)(h*h*h); \
momento1y[in_idx] = (float)((iy+0.5)*h ) *  (float)phiC  *(float)(h*h*h); \
momento1z[in_idx] = (float)((iz+0.5)*h)  *  (float)phiC  *(float)(h*h*h); \
momento2x[in_idx] = (float)powf((ix+1.5f)*(float)h,2.0f)*(float)phiC*(float)(h*h*h); \
momento2y[in_idx] = (float)powf((iy+0.5f)*(float)h,2.0f)*(float)phiC*(float)(h*h*h); \
momento2z[in_idx] = (float)powf((iz+0.5f)*(float)h,2.0f) *(float)phiC*(float)(h*h*h); \
C_float[in_idx] = (float)phiC;

#define COMPUTE_MOMENTO_WEST_BOTTOM \
momento1x[in_idx] = (float)((ix+0.5)*h)  *  (float)phiC  *(float)(h*h*h); \
momento1y[in_idx] = (float)((iy+1.5)*h ) *  (float)phiC  *(float)(h*h*h); \
momento1z[in_idx] = (float)((iz+0.5)*h)  *  (float)phiC  *(float)(h*h*h); \
momento2x[in_idx] = (float)powf((ix+0.5f)*(float)h,2.0f)*(float)phiC*(float)(h*h*h); \
momento2y[in_idx] = (float)powf((iy+1.5f)*(float)h,2.0f)*(float)phiC*(float)(h*h*h); \
momento2z[in_idx] = (float)powf((iz+0.5f)*(float)h,2.0f) *(float)phiC*(float)(h*h*h); \
C_float[in_idx] = (float)phiC;

#define COMPUTE_MOMENTO_WEST_TOP \
momento1x[in_idx] = (float)((ix+0.5)*h)  *  (float)phiC  *(float)(h*h*h); \
momento1y[in_idx] = (float)((iy+1.5)*h ) *  (float)phiC  *(float)(h*h*h); \
momento1z[in_idx] = (float)((iz+0.5)*h)  *  (float)phiC  *(float)(h*h*h); \
momento2x[in_idx] = (float)powf((ix+0.5f)*(float)h,2.0f)*(float)phiC*(float)(h*h*h); \
momento2y[in_idx] = (float)powf((iy+1.5f)*(float)h,2.0f)*(float)phiC*(float)(h*h*h); \
momento2z[in_idx] = (float)powf((iz+0.5f)*(float)h,2.0f) *(float)phiC*(float)(h*h*h); \
C_float[in_idx] = (float)phiC;

#define COMPUTE_MOMENTO_EAST_BOTTOM \
momento1x[in_idx] = (float)((ix+0.5)*h)  *  (float)phiC  *(float)(h*h*h); \
momento1y[in_idx] = (float)((iy+1.5)*h ) *  (float)phiC  *(float)(h*h*h); \
momento1z[in_idx] = (float)((iz+0.5)*h)  *  (float)phiC  *(float)(h*h*h); \
momento2x[in_idx] = (float)powf((ix+0.5f)*(float)h,2.0f)*(float)phiC*(float)(h*h*h); \
momento2y[in_idx] = (float)powf((iy+1.5f)*(float)h,2.0f)*(float)phiC*(float)(h*h*h); \
momento2z[in_idx] = (float)powf((iz+0.5f)*(float)h,2.0f) *(float)phiC*(float)(h*h*h); \
C_float[in_idx] = (float)phiC;

#define COMPUTE_MOMENTO_EAST_TOP \
momento1x[in_idx] = (float)((ix+0.5)*h)  *  (float)phiC  *(float)(h*h*h); \
momento1y[in_idx] = (float)((iy+1.5)*h ) *  (float)phiC  *(float)(h*h*h); \
momento1z[in_idx] = (float)((iz+0.5)*h)  *  (float)phiC  *(float)(h*h*h); \
momento2x[in_idx] = (float)powf((ix+0.5f)*(float)h,2.0f)*(float)phiC*(float)(h*h*h); \
momento2y[in_idx] = (float)powf((iy+1.5f)*(float)h,2.0f)*(float)phiC*(float)(h*h*h); \
momento2z[in_idx] = (float)powf((iz+0.5f)*(float)h,2.0f) *(float)phiC*(float)(h*h*h); \
C_float[in_idx] = (float)phiC;

#define COMPUTE_MOMENTO_WEST_SOUTH \
momento1x[in_idx] = (float)((ix+0.5)*h)  *  (float)phiC  *(float)(h*h*h); \
momento1y[in_idx] = (float)((iy+0.5)*h ) *  (float)phiC  *(float)(h*h*h); \
momento1z[in_idx] = (float)((iz+1.5)*h)  *  (float)phiC  *(float)(h*h*h); \
momento2x[in_idx] = (float)powf((ix+0.5f)*(float)h,2.0f)*(float)phiC*(float)(h*h*h); \
momento2y[in_idx] = (float)powf((iy+0.5f)*(float)h,2.0f)*(float)phiC*(float)(h*h*h); \
momento2z[in_idx] = (float)powf((iz+1.5f)*(float)h,2.0f) *(float)phiC*(float)(h*h*h); \
C_float[in_idx] = (float)phiC;

#define COMPUTE_MOMENTO_EAST_SOUTH \
momento1x[in_idx] = (float)((ix+0.5)*h)  *  (float)phiC  *(float)(h*h*h); \
momento1y[in_idx] = (float)((iy+0.5)*h ) *  (float)phiC  *(float)(h*h*h); \
momento1z[in_idx] = (float)((iz+1.5)*h)  *  (float)phiC  *(float)(h*h*h); \
momento2x[in_idx] = (float)powf((ix+0.5f)*(float)h,2.0f)*(float)phiC*(float)(h*h*h); \
momento2y[in_idx] = (float)powf((iy+0.5f)*(float)h,2.0f)*(float)phiC*(float)(h*h*h); \
momento2z[in_idx] = (float)powf((iz+1.5f)*(float)h,2.0f) *(float)phiC*(float)(h*h*h); \
C_float[in_idx] = (float)phiC;

#define COMPUTE_MOMENTO_WEST_NORTH \
momento1x[in_idx] = (float)((ix+0.5)*h)  *  (float)phiC  *(float)(h*h*h); \
momento1y[in_idx] = (float)((iy+0.5)*h ) *  (float)phiC  *(float)(h*h*h); \
momento1z[in_idx] = (float)((iz+1.5)*h)  *  (float)phiC  *(float)(h*h*h); \
momento2x[in_idx] = (float)powf((ix+0.5f)*(float)h,2.0f)*(float)phiC*(float)(h*h*h); \
momento2y[in_idx] = (float)powf((iy+0.5f)*(float)h,2.0f)*(float)phiC*(float)(h*h*h); \
momento2z[in_idx] = (float)powf((iz+1.5f)*(float)h,2.0f) *(float)phiC*(float)(h*h*h); \
C_float[in_idx] = (float)phiC;

#define COMPUTE_MOMENTO_EAST_NORTH \
momento1x[in_idx] = (float)((ix+0.5)*h)  *  (float)phiC  *(float)(h*h*h); \
momento1y[in_idx] = (float)((iy+0.5)*h ) *  (float)phiC  *(float)(h*h*h); \
momento1z[in_idx] = (float)((iz+1.5)*h)  *  (float)phiC  *(float)(h*h*h); \
momento2x[in_idx] = (float)powf((ix+0.5f)*(float)h,2.0f)*(float)phiC*(float)(h*h*h); \
momento2y[in_idx] = (float)powf((iy+0.5f)*(float)h,2.0f)*(float)phiC*(float)(h*h*h); \
momento2z[in_idx] = (float)powf((iz+1.5f)*(float)h,2.0f) *(float)phiC*(float)(h*h*h); \
C_float[in_idx] = (float)phiC;

#define COMPUTE_MOMENTO_WEST_SOUTH_BOTTOM \
momento1x[in_idx] = (float)((ix+0.5)*h)  *  (float)phiC  *(float)(h*h*h); \
momento1y[in_idx] = (float)((iy+0.5)*h ) *  (float)phiC  *(float)(h*h*h); \
momento1z[in_idx] = (float)((iz+0.5)*h)  *  (float)phiC  *(float)(h*h*h); \
momento2x[in_idx] = (float)powf((ix+0.5f)*(float)h,2.0f)*(float)phiC*(float)(h*h*h); \
momento2y[in_idx] = (float)powf((iy+0.5f)*(float)h,2.0f)*(float)phiC*(float)(h*h*h); \
momento2z[in_idx] = (float)powf((iz+0.5f)*(float)h,2.0f) *(float)phiC*(float)(h*h*h); \
C_float[in_idx] = (float)phiC;

#define COMPUTE_MOMENTO_WEST_SOUTH_TOP COMPUTE_MOMENTO_WEST_SOUTH_BOTTOM
#define COMPUTE_MOMENTO_WEST_NORTH_BOTTOM COMPUTE_MOMENTO_WEST_SOUTH_BOTTOM
#define COMPUTE_MOMENTO_WEST_NORTH_TOP COMPUTE_MOMENTO_WEST_SOUTH_BOTTOM
#define COMPUTE_MOMENTO_EAST_SOUTH_BOTTOM COMPUTE_MOMENTO_WEST_SOUTH_BOTTOM
#define COMPUTE_MOMENTO_EAST_SOUTH_TOP COMPUTE_MOMENTO_WEST_SOUTH_BOTTOM
#define COMPUTE_MOMENTO_EAST_NORTH_BOTTOM COMPUTE_MOMENTO_WEST_SOUTH_BOTTOM
#define COMPUTE_MOMENTO_EAST_NORTH_TOP COMPUTE_MOMENTO_WEST_SOUTH_BOTTOM