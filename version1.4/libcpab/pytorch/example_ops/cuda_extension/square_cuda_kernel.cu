#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Kernel declaration
__global__ void square_kernel_forward(float* __restrict__ output, 
                                      const float* __restrict__ input, 
                                      size_t N){
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < N){

        output[i] = input[i] * input[i];
    }
    return;
}

__global__ void square_kernel_backward(float* __restrict__ output, 
                                       const float* __restrict__ input, 
                                       size_t N){
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < N){
        output[i] = 2.0 * input[i];
    }
    return;
}

// Kernel launcher declaration
at::Tensor square_cuda_forward(at::Tensor input){
    const auto N = input.numel();
    auto output = at::zeros_like(input);    
    const int blockSize = 256;
    const int numBlocks = ceil((N + blockSize - 1) / blockSize);
    square_kernel_forward<<<numBlocks, blockSize>>>(output.data<float>(),
                                                    input.data<float>(),
                                                    N);
    return output;
}

at::Tensor square_cuda_backward(at::Tensor input){
    const auto N = input.numel();
    auto output = at::zeros_like(input);
    const int blockSize = 256;
    const int numBlocks = ceil((N + blockSize - 1) / blockSize);
	square_kernel_backward<<<numBlocks, blockSize>>>(output.data<float>(),
                                                     input.data<float>(),
                                                     N);
    return output;
}
