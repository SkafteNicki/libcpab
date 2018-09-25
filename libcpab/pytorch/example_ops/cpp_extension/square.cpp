#include <torch/torch.h>

// Function declaration
at::Tensor square_forward(at::Tensor input){
    const auto N = input.numel();
    auto input_d = input.data<float>();
    auto output = at::zeros_like(input);
    auto output_d = output.data<float>();
    for(int i = 0; i < N; i++){
        output_d[i] = input_d[i] * input_d[i];
    }
    return output;
}

at::Tensor square_backward(at::Tensor input){
    const auto N = input.numel();
    auto input_d = input.data<float>();
    auto output = at::zeros_like(input);
    auto output_d = output.data<float>();
    for(int i = 0; i < N; i++){
        output_d[i] = 2.0*input_d[i];
    }
    return output;
}

// Binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &square_forward, "Square forward");
    m.def("backward", &square_backward, "Square backward");
}