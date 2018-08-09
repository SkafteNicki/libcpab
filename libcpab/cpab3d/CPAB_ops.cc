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

REGISTER_OP("CalcTrans3")
    .Input("points: float")         // 3 x nP
    .Input("trels: float")          // n_theta x nC x 3 x 4
    .Input("ntimestep: int32")
    .Input("nx: int32")
    .Input("ny: int32")
    .Input("nz: int32")
    .Output("newpoints: float")     // n_theta x 3 x nP
    .Doc(R"doc(CPAB transformation implementation)doc");
    
REGISTER_OP("CalcGrad3")
    .Input("points: float")        // 3 x nP
    .Input("as: float")            // n_theta x nC x 3 x 4
    .Input("bs: float")            // d x nC x 3 x 4
    .Input("ntimestep: int32")    
    .Input("nx: int32")
    .Input("ny: int32")
    .Input("nz: int32")
    .Output("grad: float")         // d x n_theta x 3 x nP
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
            const Tensor& ncy_in = context->input(4);
        	   const Tensor& ncz_in = context->input(5);
                
            // Problem size
            const int nP = points_in.dim_size(1);
            const int batch_size = Trels_in.dim_size(0);

            // Create and allocate output tensor
            const int NDIMS = 3;        
            Tensor* newpoints_out = NULL;
            std::initializer_list< int64 > s0 = {batch_size, 3, nP};
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape(s0), &newpoints_out));            
            typename TTypes<float, NDIMS>::Tensor newpoints = newpoints_out->tensor<float, NDIMS>();
                              
            // Setup data view
            typename TTypes<float>::ConstMatrix points = points_in.matrix<float>();
            const float* Trels = Trels_in.flat<float>().data();
            const int nStepSolver = nStepSolver_in.flat<int>()(0);
            const int ncx = ncx_in.flat<int>()(0);
            const int ncy = ncy_in.flat<int>()(0);
        	   const int ncz = ncz_in.flat<int>()(0);
        
            // Loop over all transformations and all points    
            for(int t = 0; t < batch_size; t++){
                // Define start index for the matrices belonging to this batch
                // batch * 12 params pr cell * 6 3D-triangles pr cell * cell in x * cell in y * cell_z
                int start_idx = t * 12 * 6 * nx * ny * nz; 
                for(int i = 0; i < nP; i++){
                    // Current point
                    float point[3];
                    point[0] = points(0,i);
                    point[1] = points(1,i);
            		  point[2] = points(2,i);
                        
                    // Iterate in nStepSolver
                    int cellidx;
                    for(int n = 0; n < nStepSolver; n++){
                        // Find cell idx
                        cellidx = findcellidx(point, nx, ny, nz);
    
                        // Extract the mapping in the cell
                        const float* Trels_idx = Trels + 12*cellidx + start_idx;                
                        
                        // Calculate trajectory of point
                        float point_updated[3];                
                        A_times_b(point_updated, Trels_idx, point);
                        
                        point[0] = point_updated[0];
                        point[1] = point_updated[1];
                        point[2] = point_updated[2];
                    }
                        
                    // Copy to output
                    newpoints(t,0,i) = point[0];
                    newpoints(t,1,i) = point[1];
            		  newpoints(t,2,i) = point[2];
                }    
            } 
        } // end compute method
    private:
        int mymin(int a, double b) {
            return !(b<a)?a:floor(b);
        }
    
        int findcellidx(const float* p, const int nx, const int ny, const int nz) {
            // Copy point
            double point[3];
            point[0] = p[0];
            point[1] = p[1];
        	   point[2] = p[2];
            
            // Find row, col, layer placement
            int p0 = mymin( nx - 1 , std::max(0.0, point[0]*nx) );
            int p1 = mymin( ny - 1 , std::max(0.0, point[1]*ny) );
        	   int p2 = mymin( nz - 1 , std::max(0.0, point[2]*nz) );
            
            int cell_idx = 6 * ( p0 + p1*nx + p2*nx*ny );

        	   const double x = point[0]*nx - p0;
        	   const double y = point[1]*ny - p1;
        	   const double z = point[2]*nz - p2;

        	   if( x<=y && 1-x>y && x<z && 1-x>=z ){ cell_idx += 0; }
        	   if( x>y && x<=1-y && y<z && 1-y>=z ){ cell_idx += 1; }
            if( x>=z && x<1-z && y>=z && y<1-z ){ cell_idx += 2; }
            if( x<=z && x>1-z && y<=z && y>1-z ){ cell_idx += 3; }
            if( x<y && x>=1-y && y>z && 1-y<=z ){ cell_idx += 4; }
            if( x>=y && 1-x<y && x>z && 1-x<=z ){ cell_idx += 5; }

            return cell_idx;
        }
        
        void A_times_b(float x[], const float* A, float* b) {
            x[0] = A[0]*b[0] + A[1]*b[1] + A[2]*b[2] + A[3];
            x[1] = A[4]*b[0] + A[5]*b[1] + A[6]*b[2] + A[7];
            x[2] = A[8]*b[0] + A[9]*b[1] + A[10]*b[2] + A[11];
            return;
        }
};
        
// Forward decleration of kernel launcher 
void calcTrans_kernel_launcher(const GPUDevice& device, const int nP, const int batch_size,
                                float* newpoints, /*int* index,*/ const float* points, 
                                const float* Trels, const int* nStepSolver,
                                const int* nx, const int* ny, const int* nz); 

class CalcTransGPU : public OpKernel {
    public:
        explicit CalcTransGPU(OpKernelConstruction* context) : OpKernel(context) {}
        
        void Compute(OpKernelContext* context) override {
            // Grap input
            const Tensor& points_in = context->input(0);
            const Tensor& Trels_in = context->input(1);
            const Tensor& nStepSolver_in = context->input(2);
            const Tensor& nx_in = context->input(3);
            const Tensor& ny_in = context->input(4);
	    const Tensor& nz_in = context->input(5);
            
            // Problem size
            const int nP = points_in.dim_size(1);
            const int batch_size = Trels_in.dim_size(0);
            
            // Create and allocate output tensor
            Tensor* newpoints_out = NULL;
            std::initializer_list< int64 > s = {batch_size, 3, nP};
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape(s), &newpoints_out));            
            float* newpoints = newpoints_out->flat<float>().data();
            
/*            Tensor* index_out = NULL;
            std::initializer_list< int64 > s1 = {batch_size, nP, 50}; //TODO: This should be general
            OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape(s1), &index_out));
            int* index = index_out->flat<int>().data();
*/            
            // Setup data view
            const float* points = points_in.flat<float>().data();
            const float* Trels = Trels_in.flat<float>().data();
            const int* nStepSolver = nStepSolver_in.flat<int>().data();
            const int* nx = nx_in.flat<int>().data();
            const int* ny = ny_in.flat<int>().data();
	    const int* nz = nz_in.flat<int>().data();
            
            // Grap GPU device
            const GPUDevice& eigen_device = context->eigen_device<GPUDevice>();

            // Launch kernel
            calcTrans_kernel_launcher(eigen_device, nP, batch_size,
                                      newpoints, /*index,*/ points, Trels,
                                      nStepSolver, nx, ny, nz);
            
            return;
        }
};

void calcT_batch_grad_kernel_launcher(    const GPUDevice& device, 
                                        const int n_theta, const int d, const int nP, const int nC,
                                        float* grad, const float* points, const float* As, const float* Bs,
                                        const int* nStepSolver, const int* nx, const int* ny, 
                                        const int* nz);

class CalcGradGPU : public OpKernel {
    public:
        explicit CalcGradGPU(OpKernelConstruction* context) : OpKernel(context) {}

        void Compute(OpKernelContext* context) override {
            // Grap input
            const Tensor& points_in = context->input(0);
            const Tensor& As_in = context->input(1);
            const Tensor& Bs_in = context->input(2);
            const Tensor& nStepSolver_in = context->input(3);
            const Tensor& nx_in = context->input(5);
            const Tensor& ny_in = context->input(6);
            const Tensor& nz_in = context->input(7);

            // Create and allocate output tensor
            const int n_theta = As_in.dim_size(0);
            const int d = Bs_in.dim_size(0);
            const int nP = points_in.dim_size(1);
                        const int nC = Bs_in.dim_size(1);
            
            Tensor* grad_out = NULL;
            std::initializer_list< int64 > s = {d, n_theta, 3, nP};
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape(s), &grad_out));            
            float* grad = (grad_out->flat<float>()).data();

            // Setup data view
            const float* points = (points_in.flat<float>()).data();
            const float* As = (As_in.flat<float>()).data();
            const float* Bs = (Bs_in.flat<float>()).data();
            const int* nStepSolver = (nStepSolver_in.flat<int>()).data();
            const int* nx = (nx_in.flat<int>()).data();
            const int* ny = (ny_in.flat<int>()).data();
	    const int* nz = (nz_in.flat<int>()).data();
            
            // Get GPU information
            const GPUDevice& eigen_device = context->eigen_device<GPUDevice>();

            // Launch kernel
            calcT_batch_grad_kernel_launcher(eigen_device, n_theta, d, nP, nC,
                                             grad, points, As, Bs,
                                             nStepSolver, nx, ny, nz);
            return;
        } // end compute method
}; // end CalcTBatchGradGPU

// Register kernels to OP's
REGISTER_KERNEL_BUILDER(Name("CalcTrans3").Device(DEVICE_CPU), CalcTransCPU);
REGISTER_KERNEL_BUILDER(Name("CalcTrans3").Device(DEVICE_GPU), CalcTransGPU);
// TODO: Implement CalcGrad for CPU: REGISTER_KERNEL_BUILDER(Name("CalcGrad3").Device(DEVICE_CPU), CalcGradCPU);
REGISTER_KERNEL_BUILDER(Name("CalcGrad3").Device(DEVICE_GPU), CalcGradGPU);


