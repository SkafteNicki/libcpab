#include <torch/torch.h>
#include <vector>

// Cuda forward declaration
at::Tensor square_cuda_forward(at::Tensor input);
at::Tensor square_cuda_backward(at::Tensor input);

// Shortcuts for checking
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Function declaration
at::Tensor square_forward(at::Tensor input){
    CHECK_INPUT(input);
    return square_cuda_forward(input);
}

at::Tensor square_backward(at::Tensor input){
    CHECK_INPUT(input);
    return square_cuda_backward(input);
}

// Binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &square_forward, "Square forward (CUDA)");
    m.def("backward", &square_backward, "Square backward (CUDA)");
}
