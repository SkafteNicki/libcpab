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

using namespace tensorflow;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

REGISTER_OP("CalcTrans1")
    .Input("points: float")         // 1 x nP
    .Input("trels: float")          // n_theta x nC x 1 x 2
    .Input("ntimestep: int32")
    .Input("ncx: int32")
    .Output("newpoints: float")     // n_theta x 1 x nP
    .Doc(R"doc(CPAB transformation implementation)doc");
    
REGISTER_OP("CalcGrad1")
    .Input("points: float")        // 2 x nP
    .Input("as: float")            // n_theta x nC x 1 x 2
    .Input("bs: float")            // d x nC x 1 x 2
    .Input("ntimestep: int32")    
    .Input("ncx: int32")
    .Input("inc_x: float")
    .Output("grad: float")         // d x n_theta x 1 x nP
    .Doc(R"doc(Gradient of CPAB transformation implementation)doc");
    
    
    
    
class CalcTransCPU : public OpKernel {
    public:
        explicit CalcTransCPU(OpKernelConstruction* context) : OpKernel(context) {}
        void Compute(OpKernelContext* context) override {
            // Grap input
            const Tensor& points_in = context->input(0);
            const Tensor& Trels_in = context->input(1);
            const Tensor& nStepSolver_in = context->input(2);
            const Tensor& ncx_in = context->input(3);
                
            // Problem size
            const int nP = points_in.dim_size(1);
            const int batch_size = Trels_in.dim_size(0);

            // Create and allocate output tensor
            const int NDIMS = 3;        
            Tensor* newpoints_out = NULL;
            std::initializer_list< int64 > s0 = {batch_size, 1, nP};
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape(s0), &newpoints_out));            
            typename TTypes<float, NDIMS>::Tensor newpoints = newpoints_out->tensor<float, NDIMS>();
                   
            // Setup data view
            typename TTypes<float>::ConstMatrix points = points_in.matrix<float>();
            const float* Trels = Trels_in.flat<float>().data();
            const int nStepSolver = nStepSolver_in.flat<int>()(0);
            const int ncx = ncx_in.flat<int>()(0);

            // Loop over all transformations and all points    
            for(int t = 0; t < batch_size; t++){
                // Define start index for the matrices belonging to this batch
                // batch * 2 param pr cell * cell in x
                int start_idx = t * 2 * ncx; 
                for(int i = 0; i < nP; i++){
                    // Current point
                    float point[1];
                    point[0] = points(0,i);
                        
                    // Iterate in nStepSolver
                    int cellidx;
                    for(int n = 0; n < nStepSolver; n++){
                        // Find cell idx
                        cellidx = findcellidx(point, ncx);
    
                        // Extract the mapping in the cell
                        const float* Trels_idx = Trels + 2*cellidx + start_idx;                
                        
                        // Calculate trajectory of point
                        float point_updated[1];                
                        A_times_b(point_updated, Trels_idx, point);
                        
                        point[0] = point_updated[0];
                    }
                    // Copy to output
                    newpoints(t,0,i) = point[0];
                }    
            } 
        } // end compute method
    private:
        int mymin(int a, double b) {
            return !(b<a)?a:round(b);
        }
    
        int findcellidx(const float* p, const int ncx) {           
            // Floor value to find cell
            int idx = std::floor(p[0] * ncx);
            idx = std::min(0, std::max(idx, ncx));
            return idx;
        }
        
        void A_times_b(float x[], const float* A, float* b) {
            x[0] = A[0]*b[0] + A[1];
            return;
        }
};
        
// Forward decleration of kernel launcher 
void calcTrans_kernel_launcher(const GPUDevice& device, const int nP, const int batch_size,
                                float* newpoints, const float* points, 
                                const float* Trels, const int* nStepSolver,
                                const int* ncx);

class CalcTransGPU : public OpKernel {
    public:
        explicit CalcTransGPU(OpKernelConstruction* context) : OpKernel(context) {}
        
        void Compute(OpKernelContext* context) override {
            // Grap input
            const Tensor& points_in = context->input(0);
            const Tensor& Trels_in = context->input(1);
            const Tensor& nStepSolver_in = context->input(2);
            const Tensor& ncx_in = context->input(3);
            
            // Problem size
            const int nP = points_in.dim_size(1);
            const int batch_size = Trels_in.dim_size(0);
            
            // Create and allocate output tensor
            Tensor* newpoints_out = NULL;
            std::initializer_list< int64 > s = {batch_size, 2, nP};
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape(s), &newpoints_out));            
            float* newpoints = newpoints_out->flat<float>().data();
                        
            // Setup data view
            const float* points = points_in.flat<float>().data();
            const float* Trels = Trels_in.flat<float>().data();
            const int* nStepSolver = nStepSolver_in.flat<int>().data();
            const int* ncx = ncx_in.flat<int>().data();
            
            // Grap GPU device
            const GPUDevice& eigen_device = context->eigen_device<GPUDevice>();

            // Launch kernel
            calcTrans_kernel_launcher(eigen_device, nP, batch_size,
                                      newpoints, points, Trels,
                                      nStepSolver, ncx);
            
            return;
        }
};

class CalcGradCPU : public OpKernel {
    public:
        explicit CalcGradCPU(OpKernelConstruction* context) : OpKernel(context) {}
        
        void Compute(OpKernelContext* context) override {
            // Grap input
            const Tensor& points_in = context->input(0);
            const Tensor& As_in = context->input(1);
            const Tensor& Bs_in = context->input(2);
            const Tensor& nStepSolver_in = context->input(3);
            const Tensor& ncx_in = context->input(4);
                
            // Create and allocate output tensor
            const int n_theta = As_in.dim_size(0);
            const int d = Bs_in.dim_size(0);
            const int nP = points_in.dim_size(1);
            const int nC = Bs_in.dim_size(1);
            
            Tensor* grad_out = NULL;
            std::initializer_list< int64 > s = {d, n_theta, 1, nP};
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape(s), &grad_out));            
            float* grad = (grad_out->flat<float>()).data();
                   
            // Setup data view
            const float* points = (points_in.flat<float>()).data();
            const float* As = (As_in.flat<float>()).data();            
            const float* Bs = (Bs_in.flat<float>()).data();     
            const int nStepSolver = nStepSolver_in.flat<int>()(0);
            const int ncx = ncx_in.flat<int>()(0);
            
            // Allocate memory for computations
            float p[1], v[1], pMid[1], vMid[1], q[1], qMid[1];
            float B_times_T[1], A_times_dTdAlpha[1], u[1], uMid[1];
            float Alocal[2], Blocal[2];
            int cellidx;
            
            // Loop over all transformers
            for(int batch_index = 0; batch_index < n_theta; batch_index++) {
                // For all points
                for(int point_index = 0; point_index < nP; point_index++) {
                    // For all parameters in the transformers
                    for(int dim_index = 0; dim_index < d; dim_index++) {
                        int index = nP * batch_index + point_index;
                        int boxsize = nP * n_theta;
                        
                        // Define start index for the matrices belonging to this batch
                        // batch * 2 param pr cell * cell in x
                        int start_idx = batch_index * 2 * ncx; 
                        
                        // Initilize gradient to zero
                        grad[dim_index*boxsize + index] = 0;
                        
                        // Get point
                        p[0] = points[point_index];
                        
                        // Step size for solver
                        double h = (1.0 / nStepSolver);
                        
                        // Iterate a number of times
                        for(int t=0; t<nStepSolver; t++) {
                            // Get current cell
                            cellidx = findcellidx(p, ncx);
                        
                            // Get index of A
                            int As_idx = 2*cellidx;
                        
                            // Extract local A
                            for(int i = 0; i < 2; i++){
                                Alocal[i] = (As + As_idx + start_idx)[i];
                            }
                        
                            // Compute velocity at current location
                            A_times_b(v, Alocal, p);
                        
                            // Compute midpoint
                            pMid[0] = p[0] + h*v[0]/2.0;
                        
                            // Compute velocity at midpoint
                            A_times_b(vMid, Alocal, pMid);
                        
                            // Get index of B
                            int Bs_idx = 2 * dim_index * nC + As_idx;
                        
                            // Get local B
                            for(int i = 0; i < 2; i++){
                                Blocal[i] = (Bs + Bs_idx)[i];
                            }
                        
                            // Copy q
                            q[0] = grad[dim_index*boxsize + index];
                
                            // Step 1: Compute u using the old location
                            // Find current RHS (term 1 + term 2)
                            A_times_b(B_times_T, Blocal, p); // Term 1
                            A_times_b_linear(A_times_dTdAlpha, Alocal, q); // Term 2
                
                            // Sum both terms
                            u[0] = B_times_T[0] + A_times_dTdAlpha[0];
                
                            // Step 2: Compute mid "point"
                            qMid[0] = q[0] + h * u[0]/2.0;
                
                            // Step 3: Compute uMid
                            A_times_b(B_times_T, Blocal, pMid); // Term 1
                            A_times_b_linear(A_times_dTdAlpha, Alocal, qMid); // Term 2
                
                            // Sum both terms
                            uMid[0] = B_times_T[0] + A_times_dTdAlpha[0];

                            // Update q
                            q[0] += uMid[0] * h;
                    
                            // Update gradient
                            grad[dim_index * boxsize + index] = q[0];
                        
                            // Update p
                            p[0] += vMid[0]*h;
                        }
                    }
                }    
            }
        } // end compute method
    private:
        int mymin(int a, double b) {
            return !(b<a)?a:round(b);
        }
    
        int findcellidx(const float* p, const int ncx) {
            // Floor value to find cell
            int idx = std::floor(p[0] * ncx);
            idx = std::min(0, std::max(idx, ncx));
            return idx;
        }
        
        void A_times_b(float x[], const float* A, float* b) {
            x[0] = A[0]*b[0] + A[1];
            return;
        }
        
        void A_times_b_linear(float x[], const float* A, float* b) {
            x[0] = A[0]*b[0];
            return;
        }
}; // end CalcGradCPU

void calcGrad_kernel_launcher(const GPUDevice& device, 
                              const int n_theta, const int d, const int nP, const int nC,
                              float* grad, const float* points, const float* As, const float* Bs,
                              const int* nStepSolver, const int* ncx);

class CalcGradGPU : public OpKernel {
    public:
        explicit CalcGradGPU(OpKernelConstruction* context) : OpKernel(context) {}

        void Compute(OpKernelContext* context) override {
            // Grap input
            const Tensor& points_in = context->input(0);
            const Tensor& As_in = context->input(1);
            const Tensor& Bs_in = context->input(2);
            const Tensor& nStepSolver_in = context->input(3);
            const Tensor& ncx_in = context->input(4);

            // Create and allocate output tensor
            const int n_theta = As_in.dim_size(0);
            const int d = Bs_in.dim_size(0);
            const int nP = points_in.dim_size(1);
            const int nC = Bs_in.dim_size(1);
            
            Tensor* grad_out = NULL;
            std::initializer_list< int64 > s = {d, n_theta, 2, nP};
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape(s), &grad_out));            
            float* grad = (grad_out->flat<float>()).data();
            
            // Setup data view
            const float* points = (points_in.flat<float>()).data();
            const float* As = (As_in.flat<float>()).data();            
            const float* Bs = (Bs_in.flat<float>()).data();            
            const int* nStepSolver = (nStepSolver_in.flat<int>()).data();            
            const int* ncx = (ncx_in.flat<int>()).data();            
                       
            // Get GPU information
            const GPUDevice& eigen_device = context->eigen_device<GPUDevice>();
            
            // Launch kernel
            calcGrad_kernel_launcher(eigen_device, n_theta, d, nP, nC,
                                             grad, points, As, Bs,
                                             nStepSolver, ncx);
            return;
        } // end compute method
}; // end CalcGradGPU

// Register kernels to OP's
REGISTER_KERNEL_BUILDER(Name("CalcTrans1").Device(DEVICE_CPU), CalcTransCPU);
REGISTER_KERNEL_BUILDER(Name("CalcTrans1").Device(DEVICE_GPU), CalcTransGPU);
REGISTER_KERNEL_BUILDER(Name("CalcGrad1").Device(DEVICE_CPU), CalcGradCPU);
REGISTER_KERNEL_BUILDER(Name("CalcGrad1").Device(DEVICE_GPU), CalcGradGPU);


