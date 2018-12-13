#ifndef CPAB_OPS
#define CPAB_OPS

int stride(int ndim, const int* nc);
int param_pr_cell(int ndim);
int mymin(int a, double b);
int findcellidx_1D(const float* p, const int ncx);
int findcellidx_2D(const float* p, const int ncx, const int ncy);
int findcellidx_3D(const float* p, const int nx, const int ny, const int nz)
int findcellidx(int ndim, const float* p, const int* nc);
void A_times_b(int ndim, float x[], const float* A, const float* b);
void A_times_b_linear(int ndim, float x[], const float* A, float* b);
void cpab_forward(  float* newpoints, const float* points, const float* trels,
                    const int* nstepsolver, const float* nc,
                    const int ndim, const int nP, const int batch_size);
void cpab_backward( float* grad, const float* points, const float* As,
                    const float* Bs, const int* nstepsolver, const float* nc,
                    const int n_theta, const int d,
                    const int ndim, const int nP, const int nC);       
#endif