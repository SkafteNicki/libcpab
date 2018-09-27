#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel declaration
__global__ void cpab_cuda_kernel_forward(int ndim, int nP, int batch_size,
                                         float *output,
                                         float *points,
                                         float *trels,
                                         int *nstepsolver,
                                         int *nc_in){
    int point_index = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_index = blockIdx.y * blockDim.y + threadIdx.y;
    if(point_index < nP && batch_index < batch_size) {
    
    }
    return;
}

__global__ void cpab_cuda_kernel_backward(){

    return;
}

// Kernel launcher declaration
at::Tensor cpab_cuda_forward(at::Tensor points_in, 
                             at::Tensor trels_in,  
                             at::Tensor nstepsolver_in, 
                             at::Tensor nc_in){
    // Problem size
    const auto ndim = points_in.size(0);
    const auto nP = points_in.size(1);
    const auto batch_size = trels_in.size(0);

    // Allocate output
    auto output = at::CUDA(at::kFloat).zeros({batch_size, ndim, nP}); // [batch_size, ndim, nP]                     
    
    // Kernel configuration
    dim3 bc(ceil(nP/256), batch_size);
    dim3 tpb(256, 1);
    
    // Launch kernel
    cpab_cuda_kernel_forward<<<bc, tpb>>>(ndim, nP, batch_size,
                                          output.data<float>(),
                                          points_in.data<float>(),
                                          trels_in.data<float>(),
                                          nstepsolver_in.data<int>(),
                                          nc_in.data<int>());
    return output;
                             
}
at::Tensor cpab_cuda_backward(at::Tensor points_in, 
                              at::Tensor As_in, 
                              at::Tensor Bs_in, 
                              at::Tensor nstepsolver_in,
                              at::Tensor nc){
                              
                              
}


