#ifndef _CUDA_KERNEL_HELPER_H_
#define _CUDA_KERNEL_HELPER_H_
#if GOOGLE_CUDA
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/platform/types.h"

#define CUDA_AXIS_KERNEL_LOOP(i, n, axis)                                 \
  for(int i = blockIdx.axis * blockDim.axis + threadIdx.axis; i < n.axis; \
         i += blockDim.axis * gridDim.axis)

#define DIV_UP(a, b) (((a) + (b)-1) / (b))
             
using namespace tensorflow;

typedef Eigen::GpuDevice GPUDevice;

struct Cuda2DLaunchConfig {
    dim3 virtual_thread_count;
    dim3 thread_per_block;
    dim3 block_count;
};

struct Cuda3DLaunchConfig {
    dim3 virtual_thread_count;
    dim3 thread_per_block;
    dim3 block_count;
};

inline Cuda2DLaunchConfig GetCuda2DLaunchConfigOWN(int xdim, int ydim){
    Cuda2DLaunchConfig config;
    
    config.virtual_thread_count = dim3(xdim, ydim);
    config.thread_per_block = dim3(256, 1);
    config.block_count = dim3((int)ceil(xdim/256.0), ydim);
    
    return config;
}

inline Cuda3DLaunchConfig GetCuda3DLaunchConfigOWN(int xdim, int ydim, int zdim){
    Cuda3DLaunchConfig config;
        
    int threadx = std::min(xdim, 4);
    int thready = std::min(ydim, 128);
    int threadz = std::min(zdim, 1);
    int blocksx = DIV_UP(xdim, threadx);
    int blocksy = DIV_UP(ydim, thready);
    int blocksz = DIV_UP(zdim, threadz);
    
    config.virtual_thread_count = dim3(xdim, ydim, zdim);
    config.thread_per_block = dim3(threadx, thready, threadz);
    config.block_count = dim3(blocksx, blocksy, blocksz);
    
    return config;
}

#endif  // GOOGLE_CUDA

#endif  // TENSORFLOW_CORE_UTIL_CUDA_KERNEL_HELPER_H_

