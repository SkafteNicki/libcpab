#ifndef _CUDA_KERNEL_HELPER_H_
#define _CUDA_KERNEL_HELPER_H_

#if GOOGLE_CUDA

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/platform/types.h"

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

#define CUDA_AXIS_KERNEL_LOOP(i, n, axis)                                 \
  for(int i = blockIdx.axis * blockDim.axis + threadIdx.axis; i < n.axis; \
	     i += blockDim.axis * gridDim.axis)

#define DIV_UP(a, b) (((a) + (b)-1) / (b))
			 
using namespace tensorflow;

typedef Eigen::GpuDevice GPUDevice;

struct CudaLaunchConfig {
  // Logical number of thread that works on the elements. If each logical
  // thread works on exactly a single element, this is the same as the working
  // element count.
  int virtual_thread_count = -1;
  // Number of threads per block.
  int thread_per_block = -1;
  // Number of blocks for Cuda kernel launch.
  int block_count = -1;
};

// Calculate the Cuda launch config we should use for a kernel launch.
// This is assuming the kernel is quite simple and will largely be
// memory-limited.
inline CudaLaunchConfig GetCudaLaunchConfig(int work_element_count,
                                            const GPUDevice& d) {
  const int virtual_thread_count = work_element_count;
  const int physical_thread_count = std::min(
      d.getNumCudaMultiProcessors() * d.maxCudaThreadsPerMultiProcessor(),
      virtual_thread_count);
  const int thread_per_block = std::min(512, d.maxCudaThreadsPerBlock());
  const int block_count = std::min(
      (physical_thread_count + thread_per_block - 1) / thread_per_block,
      d.getNumCudaMultiProcessors());

  CudaLaunchConfig config;
  config.virtual_thread_count = virtual_thread_count;
  config.thread_per_block = thread_per_block;
  config.block_count = block_count;
  return config;
}

struct Cuda2DLaunchConfig {
  dim3 virtual_thread_count;
  dim3 thread_per_block;
  dim3 block_count;
};

inline Cuda2DLaunchConfig GetCuda2DLaunchConfig(int xdim, int ydim,
                                                const GPUDevice& d) {
  Cuda2DLaunchConfig config;

  config.virtual_thread_count = dim3(xdim, ydim, 1);

  const int kThreadsPerBlock = 256;
  int block_cols = std::min(xdim, kThreadsPerBlock);
  // ok to round down here and just do more loops in the kernel
  int block_rows = std::max(kThreadsPerBlock / block_cols, 1);

  const int physical_thread_count =
      d.getNumCudaMultiProcessors() * d.maxCudaThreadsPerMultiProcessor();

  const int max_blocks = std::max(physical_thread_count / kThreadsPerBlock, 1);

  config.thread_per_block = dim3(block_cols, block_rows, 1);

  int grid_x = std::min((xdim + block_cols - 1) / block_cols, max_blocks);

  config.block_count = dim3(
      grid_x, std::min(max_blocks / grid_x, std::max(ydim / block_rows, 1)), 1);

  return config;
}

struct Cuda3DLaunchConfig {
  dim3 virtual_thread_count;
  dim3 thread_per_block;
  dim3 block_count;
};

template <typename DeviceFunc> inline Cuda3DLaunchConfig GetCuda3DLaunchConfig(
		int xdim, int ydim, int zdim, const GPUDevice& d, DeviceFunc func,
		size_t dynamic_shared_memory_size, int block_size_limit) {
  
	//Cuda3DLaunchConfig config;
	Cuda3DLaunchConfig config;

	if (xdim <= 0 || ydim <= 0 || zdim <= 0) {
		return config; //config;
	}

	int dev;
	cudaGetDevice(&dev);
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, dev);
	int xthreadlimit = deviceProp.maxThreadsDim[0];
	int ythreadlimit = deviceProp.maxThreadsDim[1];
	int zthreadlimit = deviceProp.maxThreadsDim[2];
	int xgridlimit = deviceProp.maxGridSize[0];
	int ygridlimit = deviceProp.maxGridSize[1];
	int zgridlimit = deviceProp.maxGridSize[2];
	std::cout << xthreadlimit << " " << ythreadlimit << " " << zthreadlimit << std::endl;
	std::cout << xgridlimit << " " << ygridlimit << " " << zgridlimit << std::endl;

	int block_count = 0;
	int thread_per_block = 0;
	cudaError_t err = cudaOccupancyMaxPotentialBlockSize(
		&block_count, &thread_per_block, func, dynamic_shared_memory_size,
		block_size_limit);
	std::cout << (int)(err == cudaSuccess) << std::endl;
	std::cout << block_count << std::endl;
	std::cout << thread_per_block << std::endl;
	
	#define MIN3(a, b, c) std::min((a), std::min((b), (c)))
	int threadsx = MIN3(xdim, thread_per_block, xthreadlimit);
	int threadsy = MIN3(ydim, std::max(thread_per_block / threadsx, 1), ythreadlimit);
	int threadsz = MIN3(zdim, std::max(thread_per_block / (threadsx * threadsy), 1), zthreadlimit);

	int blocksx = MIN3(block_count, DIV_UP(xdim, threadsx), xgridlimit);
	int blocksy = MIN3(DIV_UP(block_count, blocksx), DIV_UP(ydim, threadsy), ygridlimit);
	int blocksz = MIN3(DIV_UP(block_count, (blocksx * blocksy)), DIV_UP(zdim, threadsz), zgridlimit);
	#undef MIN3

	config.virtual_thread_count = dim3(xdim, ydim, zdim);
	config.thread_per_block = dim3(threadsx, threadsy, threadsz);
	config.block_count = dim3(blocksx, blocksy, blocksz);

	//return config;
	return config;
}


inline Cuda3DLaunchConfig GetCuda3DLaunchConfigOWN(int xdim, int ydim, int zdim){
	//Cuda3DLaunchConfig config;
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

