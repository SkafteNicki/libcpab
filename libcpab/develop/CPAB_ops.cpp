#include <torch/torch.h>
#include <vector>

at::Tensor CPAB_forward_cpu() {


};

at::Tensor CPAB_backwards_cpu() {


};

at::Tensor CPAB_forward_gpu() {


};

at::Tensor CPAB_backwards_gpu() {


};


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_cpu", &CPAB_forward_cpu, "CPAB forward cpu");
    m.def("backward_cpu", &CPAB_backwards_cpu, "CPAB backward cpu");
    m.def("forward_gpu", &CPAB_backwards_gpu, "CPAB forward gpu");
    m.def("backward_gpu", &CPAB_backwards_gpu, "CPAB backward gpu");
}