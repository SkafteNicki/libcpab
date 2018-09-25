#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Kernel declaration
namespace {

template <typename scalar_t>
__global__ void square_kernel_forward(scalar_t* __restrict__ output, 
                                      const scalar_t* __restrict__ input, 
                                      size_t N){
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < N){
        output[i] = input[i] * input[i];
    }
    return;
}

template <typename scalar_t>
__global__ void square_kernel_backward(scalar_t* __restrict__ output, 
                                       const scalar_t* __restrict__ input, 
                                       size_t N){
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < N){
        output[i] = 2.0 * input[i];
    }
    return;
}

} // end namespace

// Kernel launcher declaration
std::vector<at::Tensor> square_cuda_forward(at::Tensor input){
    const auto N = input.numel();
    auto output = at::zeros_like(input);    
    const int blockSize = 512;
    const int numBlocks = (N + blockSize - 1) / blockSize;
    
    AT_DISPATCH_FLOATING_TYPES(input.type(), "square_cuda_forward", ([&] {
        square_kernel_forward<scalar_t><<<numBlocks, blockSize>>>(
            output.data<scalar_t>(),
            input.data<scalar_t>(),
            N);
    }));
    
    return {output};
}

std::vector<at::Tensor> square_cuda_backward(at::Tensor input){
    const auto N = input.numel();
    auto output = at::zeros_like(input);
    const int blockSize = 512;
    const int numBlocks = (N + blockSize - 1) / blockSize;
    AT_DISPATCH_FLOATING_TYPES(input.type(), "square_cuda_forward", ([&] {
        square_kernel_backward<scalar_t><<<numBlocks, blockSize>>>(
            output.data<scalar_t>(),
            input.data<scalar_t>(),
            N);
    }));
    return {output};
}
