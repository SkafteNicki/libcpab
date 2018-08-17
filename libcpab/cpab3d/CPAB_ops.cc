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
    .Doc(R"doc(CPAB transformation implementation)doc")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c){
        c->set_output(0, c->Tensor(c->Dim(c->input(1), 0), 3, c->Dim(c->input(0), 1)));
        return Status::OK();
    });
    
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
                int start_idx = t * 12 * 6 * ncx * ncy * ncz; 
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
                        cellidx = findcellidx(point, ncx, ncy, ncz);
    
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
                                float* newpoints, const float* points, 
                                const float* Trels, const int* nStepSolver,
                                const int* ncx, const int* ncy, const int* ncz); 

class CalcTransGPU : public OpKernel {
    public:
        explicit CalcTransGPU(OpKernelConstruction* context) : OpKernel(context) {}
        
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
            Tensor* newpoints_out = NULL;
            std::initializer_list< int64 > s = {batch_size, 3, nP};
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape(s), &newpoints_out));            
            float* newpoints = newpoints_out->flat<float>().data();
               
            // Setup data view
            const float* points = points_in.flat<float>().data();
            const float* Trels = Trels_in.flat<float>().data();
            const int* nStepSolver = nStepSolver_in.flat<int>().data();
            const int* ncx = ncx_in.flat<int>().data();
            const int* ncy = ncy_in.flat<int>().data();
        	   const int* ncz = ncz_in.flat<int>().data();
            
            // Grap GPU device
            const GPUDevice& eigen_device = context->eigen_device<GPUDevice>();

            // Launch kernel
            calcTrans_kernel_launcher(eigen_device, nP, batch_size,
                                      newpoints, points, Trels,nStepSolver, ncx, ncy, ncz);
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
            const Tensor& ncy_in = context->input(5);
            const Tensor& ncz_in = context->input(6);
                
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
            const int nStepSolver = nStepSolver_in.flat<int>()(0);
            const int ncx = ncx_in.flat<int>()(0);
            const int ncy = ncy_in.flat<int>()(0);
            const int ncz = ncz_in.flat<int>()(0);
            
            // Allocate memory for computations
            float p[3], v[3], pMid[3], vMid[3], q[3], qMid[3];
            float B_times_T[3], A_times_dTdAlpha[3], u[3], uMid[3];
            float Alocal[12], Blocal[12];
            int cellidx;
            
            // Loop over all transformers
            for(int batch_index = 0; batch_index < n_theta; batch_index++) {
                // For all points
                for(int point_index = 0; point_index < nP; point_index++) {
                    // For all parameters in the transformers
                    for(int dim_index = 0; dim_index < d; dim_index++) {
                        int index = 3 * nP * batch_index + point_index;
                        int boxsize = 3 * nP * n_theta;
                        
                        // Define start index for the matrices belonging to this batch
                        // batch * 12 params pr cell * 6 3D-triangles pr cell * cell in x * cell in y * cell_z
                        int start_idx = batch_index * 12 * 6 * ncx * ncy * ncz; 
                        
                        // Initilize gradient to zero
                        grad[dim_index*boxsize + index] = 0;
                        grad[dim_index*boxsize + index + nP] = 0;
                        grad[dim_index*boxsize + index + 2*nP] = 0;
                        
                        // Get point
                        p[0] = points[point_index];
                        p[1] = points[point_index + nP];
                        p[2] = points[point_index + 2*nP];
                        
                        // Step size for solver
                        double h = (1.0 / nStepSolver);
                        
                        // Iterate a number of times
                        for(int t=0; t<nStepSolver; t++) {
                            // Get current cell
                            cellidx = findcellidx(p, ncx, ncy, ncz);
                        
                            // Get index of A
                            int As_idx = 12*cellidx;
                        
                            // Extract local A
                            for(int i = 0; i < 12; i++){
                                Alocal[i] = (As + As_idx + start_idx)[i];
                            }
                        
                            // Compute velocity at current location
                            A_times_b(v, Alocal, p);
                        
                            // Compute midpoint
                            pMid[0] = p[0] + h*v[0]/2.0;
                            pMid[1] = p[1] + h*v[1]/2.0;
                            pMid[2] = p[2] + h*v[2]/2.0;
                        
                            // Compute velocity at midpoint
                            A_times_b(vMid, Alocal, pMid);
                        
                            // Get index of B
                            int Bs_idx = 12 * dim_index * nC + As_idx;
                        
                            // Get local B
                            for(int i = 0; i < 12; i++){
                                Blocal[i] = (Bs + Bs_idx)[i];
                            }
                        
                            // Copy q
                            q[0] = grad[dim_index*boxsize + index];
                            q[1] = grad[dim_index*boxsize + index + nP];
                            q[2] = grad[dim_index*boxsize + index + 2*nP];
                
                            // Step 1: Compute u using the old location
                            // Find current RHS (term 1 + term 2)
                            A_times_b(B_times_T, Blocal, p); // Term 1
                            A_times_b_linear(A_times_dTdAlpha, Alocal, q); // Term 2
                
                            // Sum both terms
                            u[0] = B_times_T[0] + A_times_dTdAlpha[0];
                            u[1] = B_times_T[1] + A_times_dTdAlpha[1];
                            u[2] = B_times_T[2] + A_times_dTdAlpha[2];
                
                            // Step 2: Compute mid "point"
                            qMid[0] = q[0] + h * u[0]/2.0;
                            qMid[1] = q[1] + h * u[1]/2.0;
                            qMid[2] = q[2] + h * u[2]/2.0;
                
                            // Step 3: Compute uMid
                            A_times_b(B_times_T, Blocal, pMid); // Term 1
                            A_times_b_linear(A_times_dTdAlpha, Alocal, qMid); // Term 2
                
                            // Sum both terms
                            uMid[0] = B_times_T[0] + A_times_dTdAlpha[0];
                            uMid[1] = B_times_T[1] + A_times_dTdAlpha[1];
                            uMid[2] = B_times_T[2] + A_times_dTdAlpha[2];

                            // Update q
                            q[0] += uMid[0] * h;
                            q[1] += uMid[1] * h;
                            q[2] += uMid[2] * h;
                    
                            // Update gradient
                            grad[dim_index * boxsize + index] = q[0];
                            grad[dim_index * boxsize + index + nP] = q[1];
                            grad[dim_index * boxsize + index + 2*nP] = q[2];
                        
                            // Update p
                            p[0] += vMid[0]*h;
                            p[1] += vMid[1]*h;
                            p[2] += vMid[2]*h;
                        }
                    }
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
        
        void A_times_b_linear(float x[], const float* A, float* b) {
            x[0] = A[0]*b[0] + A[1]*b[1] + A[2]*b[2];
            x[1] = A[4]*b[0] + A[5]*b[1] + A[6]*b[2];
            x[2] = A[8]*b[0] + A[9]*b[1] + A[10]*b[2];
            return;
        }  
}; // end CalcGradCPU

void calcGrad_kernel_launcher(  const GPUDevice& device, 
                                const int n_theta, const int d, const int nP, const int nC,
                                float* grad, const float* points, const float* As, const float* Bs,
                                const int* nStepSolver, const int* ncx, const int* ncy, 
                                const int* ncz);

class CalcGradGPU : public OpKernel {
    public:
        explicit CalcGradGPU(OpKernelConstruction* context) : OpKernel(context) {}

        void Compute(OpKernelContext* context) override {
            // Grap input
            const Tensor& points_in = context->input(0);
            const Tensor& As_in = context->input(1);
            const Tensor& Bs_in = context->input(2);
            const Tensor& nStepSolver_in = context->input(3);
            const Tensor& ncx_in = context->input(5);
            const Tensor& ncy_in = context->input(6);
            const Tensor& ncz_in = context->input(7);

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
            const int* ncx = (ncx_in.flat<int>()).data();
            const int* ncy = (ncy_in.flat<int>()).data();
            const int* ncz = (ncz_in.flat<int>()).data();
            
            // Get GPU information
            const GPUDevice& eigen_device = context->eigen_device<GPUDevice>();

            // Launch kernel
            calcGrad_kernel_launcher(eigen_device, n_theta, d, nP, nC,
                                     grad, points, As, Bs,
                                     nStepSolver, ncx, ncy, ncz);
            return;
        } // end compute method
}; // end CalcTBatchGradGPU

// Register kernels to OP's
REGISTER_KERNEL_BUILDER(Name("CalcTrans3").Device(DEVICE_CPU), CalcTransCPU);
REGISTER_KERNEL_BUILDER(Name("CalcTrans3").Device(DEVICE_GPU), CalcTransGPU);
REGISTER_KERNEL_BUILDER(Name("CalcGrad3").Device(DEVICE_CPU), CalcGradCPU);
REGISTER_KERNEL_BUILDER(Name("CalcGrad3").Device(DEVICE_GPU), CalcGradGPU);