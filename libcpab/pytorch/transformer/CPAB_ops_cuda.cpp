#include <torch/torch.h>

// Cuda forward declaration
at::Tensor cpab_cuda_forward(at::Tensor points_in, at::Tensor trels_in,  
                             at::Tensor nstepsolver_in, at::Tensor nc_in);
at::Tensor cpab_cuda_backward(at::Tensor points_in, at::Tensor As_in, 
                              at::Tensor Bs_in, at::Tensor nstepsolver_in,
                              at::Tensor nc);
                              
// Shortcuts for checking
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Function declaration
at::Tensor cpab_forward(at::Tensor points_in, //[ndim, n_points]
                        at::Tensor trels_in,  //[batch_size, nC, ndim, ndim+1]
                        at::Tensor nstepsolver_in, // scalar
                        at::Tensor nc_in){ // ndim length tensor
    // Do input checking
    CHECK_INPUT(points_in);
    CHECK_INPUT(trels_in);
    CHECK_INPUT(nstepsolver_in);
    CHECK_INPUT(nc_in);
    
    // Call kernel launcher
    output = cpab_cuda_forward(input);
    return output;
}

at::Tensor cpab_backward(at::Tensor points_in, // [ndim, nP]
                         at::Tensor As_in, // [n_theta, nC, ndim, ndim+1]
                         at::Tensor Bs_in, // [d, nC, ndim, ndim+1]
                         at::Tensor nstepsolver_in, // scalar
                         at::Tensor nc){ // ndim length tensor
    // Do input checking
    CHECK_INPUT(points_in);
    CHECK_INPUT(As_in);
    CHECK_INPUT(Bs_in);
    CHECK_INPUT(nstepsolver_in);
    CHECK_INPUT(nc);
    
    // Call kernel launcher
    output = cpab_cuda_backward(input);
    return output;
}

// Binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &cpab_forward, "Cpab transformer forward");
    m.def("backward", &cpab_backward, "Cpab transformer backward");
}