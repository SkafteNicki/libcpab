#define EIGEN_USE_THREADS
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "iostream"
#include "../core/cpab_ops.h"

using namespace tensorflow;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

REGISTER_OP("CalcTrans")
    .Input("points: float")         // dim x nP or n_theta x dim x nP
    .Input("trels: float")          // n_theta x nC x 1 x 2
    .Input("ntimestep: int32")
    .Input("ncx: int32")
    .Output("newpoints: float")     // n_theta x 1 x nP
    .Doc(R"doc(CPAB transformation implementation)doc");
    
REGISTER_OP("CalcGrad")
    .Input("points: float")        // 2 x nP
    .Input("as: float")            // n_theta x nC x 1 x 2
    .Input("bs: float")            // d x nC x 1 x 2
    .Input("ntimestep: int32")    
    .Input("ncx: int32")
    .Output("grad: float")         // d x n_theta x 1 x nP
    .Doc(R"doc(Gradient of CPAB transformation implementation)doc");
    
class CalcTransCPU : public OpKernel {
    public:
        explicit CalcTransCPU(OpKernelConstruction* context) : OpKernel(context) {}
        void Compute(OpKernelContext* context) override {
            // Grap input
            const Tensor& points_in = context->input(0);
            const Tensor& trels_in = context->input(1);
            const Tensor& nstepsolver_in = context->input(2);
            const Tensor& nc_in = context->input(3);
            
            // Determine if grid is matrix or tensor
            const int broadcast = (int)(points_in.dims() == 3 & points_in.dim_size(0) == trels_in.dim_size(0));
            
            // Problem size
            const int ndim = (broadcast) ? points_in.dim_size(1) : points_in.dim_size(0);
            const int nP = (broadcast) ? points_in.dim_size(2) : points_in.dim_size(1);
            const int batch_size = trels_in.dim_size(0);

            // Create and allocate output tensor
            Tensor* newpoints_out = NULL;
            std::initializer_list< int64 > s0 = {batch_size, ndim, nP};
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape(s0), &newpoints_out));
            float* newpoints = (newpoints_out->flat<float>()).data();

            // Setup data view
            const auto points = (points_in.flat<float>()).data();
            const auto trels = (trels_in.flat<float>()).data();
            const auto nstepsolver = (nstepsolver_in.flat<int>()).data();
            const auto nc = (nc_in.flat<int>()).data();
            
            // Call function
            cpab_forward_op(newpoints, points, trels, nstepsolver, nc,
                            ndim, nP, batch_size, broadcast);
            return;
        } // end compute method
};
        
// Forward decleration of kernel launcher 
void cpab_cuda_forward(const GPUDevice& device, const float* points, const float* trels,
                        const int* nstepsolver, const int* nc, const int broadcast,
                        const int ndim, const int nP, const int batch_size, float* output);

class CalcTransGPU : public OpKernel {
    public:
        explicit CalcTransGPU(OpKernelConstruction* context) : OpKernel(context) {}
        void Compute(OpKernelContext* context) override {
            // Grap input
            const Tensor& points_in = context->input(0);
            const Tensor& trels_in = context->input(1);
            const Tensor& nstepsolver_in = context->input(2);
            const Tensor& nc_in = context->input(3);
            
            // Determine if grid is matrix or tensor
            const int broadcast = (int)(points_in.dims() == 3 & points_in.dim_size(0) == trels_in.dim_size(0));
            
            // Problem size
            const int ndim = (broadcast) ? points_in.dim_size(1) : points_in.dim_size(0);
            const int nP = (broadcast) ? points_in.dim_size(2) : points_in.dim_size(1);
            const int batch_size = trels_in.dim_size(0);
            
            // Create and allocate output tensor
            Tensor* newpoints_out = NULL;
            std::initializer_list< int64 > s = {batch_size, ndim, nP};
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape(s), &newpoints_out));            
            float* newpoints = (newpoints_out->flat<float>()).data();
            
            // Grap GPU device
            const GPUDevice& eigen_device = context->eigen_device<GPUDevice>();
            
            // Setup data view
            const auto points = (points_in.flat<float>()).data();
            const auto trels = (trels_in.flat<float>()).data();
            const auto nstepsolver = (nstepsolver_in.flat<int>()).data();
            const auto nc = (nc_in.flat<int>()).data();
            
            // Call solver
            cpab_cuda_forward(eigen_device, points, trels, nstepsolver, nc, 
                            broadcast, ndim, nP, batch_size, newpoints);
            return;
        } // end compute method
}; // end CalcTransGPU

class CalcGradCPU : public OpKernel {
    public:
        explicit CalcGradCPU(OpKernelConstruction* context) : OpKernel(context) {}
        void Compute(OpKernelContext* context) override {
            // Grap input
            const Tensor& points_in = context->input(0);
            const Tensor& As_in = context->input(1);
            const Tensor& Bs_in = context->input(2);
            const Tensor& nstepsolver_in = context->input(3);
            const Tensor& nc_in = context->input(4);
            
            // Determine if grid is matrix or tensor
            const int broadcast = (int)(points_in.dims() == 3 & points_in.dim_size(0) == As_in.dim_size(0));
            
            // Problem size
            const int ndim = (broadcast) ? points_in.dim_size(1) : points_in.dim_size(0);
            const int nP = (broadcast) ? points_in.dim_size(2) : points_in.dim_size(1);
            const int n_theta = As_in.dim_size(0);
            const int d = Bs_in.dim_size(0);
            const int nC = Bs_in.dim_size(1);
            
            // Allocate output
            Tensor* grad_out = NULL;
            std::initializer_list< int64 > s = {d, n_theta, ndim, nP};
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape(s), &grad_out));            
            float* grad = (grad_out->flat<float>()).data();
                   
            // Setup data view
            const auto points = (points_in.flat<float>()).data();
            const auto As = (As_in.flat<float>()).data();            
            const auto Bs = (Bs_in.flat<float>()).data();     
            const auto nstepsolver = (nstepsolver_in.flat<int>()).data();
            const auto nc = (nc_in.flat<int>()).data();
            
            // Call solver
            cpab_backward_op(grad, points, As, Bs, nstepsolver, nc,
                            n_theta, d, ndim, nP, nC, broadcast);
            return;
        } // end compute method
}; // end CalcGradCPU

void cpab_cuda_backward(const GPUDevice& device, const float* points, const float* As,
                        const float* Bs, const int* nstepsolver, const int* nc,
                        const int broadcast, const int ndim, const int nP,
                        const int n_theta, const int d, const int nC, float* output);

class CalcGradGPU : public OpKernel {
    public:
        explicit CalcGradGPU(OpKernelConstruction* context) : OpKernel(context) {}
        void Compute(OpKernelContext* context) override {
            // Grap input
            const Tensor& points_in = context->input(0);
            const Tensor& As_in = context->input(1);
            const Tensor& Bs_in = context->input(2);
            const Tensor& nstepsolver_in = context->input(3);
            const Tensor& nc_in = context->input(4);
            
            // Determine if grid is matrix or tensor
            const int broadcast = (int)(points_in.dims() == 3 & points_in.dim_size(0) == As_in.dim_size(0));
            
            // Create and allocate output tensor
            const int ndim = (broadcast) ? points_in.dim_size(1) : points_in.dim_size(0);
            const int nP = (broadcast) ? points_in.dim_size(2) : points_in.dim_size(1);
            const int n_theta = As_in.dim_size(0);
            const int d = Bs_in.dim_size(0);
            const int nC = Bs_in.dim_size(1);
            
            // Allocate output
            Tensor* grad_out = NULL;
            std::initializer_list< int64 > s = {d, n_theta, ndim, nP};
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape(s), &grad_out));            
            float* grad = (grad_out->flat<float>()).data();
            
            // Grap GPU device
            const GPUDevice& eigen_device = context->eigen_device<GPUDevice>();
            
            // Setup data view
            const float* points = (points_in.flat<float>()).data();
            const float* As = (As_in.flat<float>()).data();            
            const float* Bs = (Bs_in.flat<float>()).data();            
            const int* nstepsolver = (nstepsolver_in.flat<int>()).data();            
            const int* nc = (nc_in.flat<int>()).data();            
            
            // Call solver
            cpab_cuda_backward(eigen_device, points, As, Bs, nstepsolver, 
                                nc, broadcast, ndim, nP, n_theta, d, nC, grad);
            return;
        } // end compute method
}; // end CalcGradGPU

// Register kernels to OP's
REGISTER_KERNEL_BUILDER(Name("CalcTrans").Device(DEVICE_CPU), CalcTransCPU);
REGISTER_KERNEL_BUILDER(Name("CalcTrans").Device(DEVICE_GPU), CalcTransGPU);
REGISTER_KERNEL_BUILDER(Name("CalcGrad").Device(DEVICE_CPU), CalcGradCPU);
REGISTER_KERNEL_BUILDER(Name("CalcGrad").Device(DEVICE_GPU), CalcGradGPU);