#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "../core/cpab_ops.cuh"

#define DIV_UP(a, b) (((a) + (b)-1) / (b))

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

typedef Eigen::GpuDevice GPUDevice;

void cpab_cuda_forward(const GPUDevice& device, const float* points, const float* trels,
                        const int* nstepsolver, const int* nc, const int broadcast,
                        const int ndim, const int nP, const int batch_size, float* output){
    // Kernel configuration
    dim3 bc((int)ceil(nP/256.0), batch_size);
    dim3 tpb(256, 1);
    
    // Launch kernel
    // We do it in this way, since dynamically allocating memory in CUDA sucks!
    if(ndim == 1){
         cpab_cuda_kernel_forward_1D<<<bc, tpb>>>(nP, batch_size, output,
                                                  points, trels, nstepsolver,
                                                  nc, broadcast);
	}
	if(ndim == 2){
         cpab_cuda_kernel_forward_2D<<<bc, tpb>>>(nP, batch_size, output,
                                                  points, trels, nstepsolver,
                                                  nc, broadcast);
	}
	if(ndim == 3){
        cpab_cuda_kernel_forward_3D<<<bc, tpb>>>(nP, batch_size, output,
                                                 points, trels, nstepsolver,
                                                 nc, broadcast);
    }            
    return;                        
}

void cpab_cuda_backward(const GPUDevice& device, const float* points, const float* As,
                        const float* Bs, const int* nstepsolver, const int* nc,
                        const int broadcast, const int ndim, const int nP,
                        const int n_theta, const int d, const int nC, float* output){
    // Kernel configuration
    dim3 tpb = dim3(std::min((int)nP, 128), std::min((int)n_theta, 4), std::min((int)d, 1));
    dim3 bc = dim3(DIV_UP(nP, tpb.x), DIV_UP(n_theta, tpb.y), DIV_UP(d, tpb.z));
    dim3 vtc = dim3(nP, n_theta, d);
    
    // Launch kernel
    // We do it in this way, since dynamically allocating memory in CUDA sucks!
	if(ndim == 1){
         cpab_cuda_kernel_backward_1D<<<bc, tpb>>>(vtc, n_theta, d, nP, nC,
                                                   output, points, As, Bs,
                                                   nstepsolver, nc, broadcast);
	}
	if(ndim == 2){
         cpab_cuda_kernel_backward_2D<<<bc, tpb>>>(vtc, n_theta, d, nP, nC,
                                                   output, points, As, Bs,
                                                   nstepsolver, nc, broadcast);
	}
 	if(ndim == 3){
         cpab_cuda_kernel_backward_3D<<<bc, tpb>>>(vtc, n_theta, d, nP, nC,
                                                   output, points, As, Bs,
                                                   nstepsolver, nc, broadcast);
    }
    gpuErrchk( cudaPeekAtLastError() );
    return;
}

#endif
