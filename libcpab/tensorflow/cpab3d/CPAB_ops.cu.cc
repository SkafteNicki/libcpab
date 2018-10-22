#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "cuda_kernel_helper.h"
//#include "tensorflow/core/util/cuda_kernel_helper.h"

__device__ int mymin(int a, double b) {
    return !(b<a)?a:floor(b);
}

__device__ int findcellidx(const float* p, const int ncx, const int ncy, const int ncz) {
    // Move with respect to the lower bound
    double point[3];
    point[0] = p[0];
    point[1] = p[1];
    point[2] = p[2];
    
    // Find initial row, col placement
    int p0 = mymin( ncx-1 , max(0.0, point[0]*ncx ) );
    int p1 = mymin( ncy-1 , max(0.0, point[1]*ncy ) );
    int p2 = mymin( ncz-1 , max(0.0, point[2]*ncz ) );
            
    int cell_idx = 6 * (p0 + p1*ncx + p2*ncx*ncy);

    const double x = point[0]*ncx - p0;
    const double y = point[1]*ncy - p1;
    const double z = point[2]*ncz - p2;

    if( x<=y && 1-x>y && x<z && 1-x>=z ){ cell_idx += 0; }
    if( x>y && x<=1-y && y<z && 1-y>=z ){ cell_idx += 1; }
    if( x>=z && x<1-z && y>=z && y<1-z ){ cell_idx += 2; }
    if( x<=z && x>1-z && y<=z && y>1-z ){ cell_idx += 3; }
    if( x<y && x>=1-y && y>z && 1-y<=z ){ cell_idx += 4; }
    if( x>=y && 1-x<y && x>z && 1-x<=z ){ cell_idx += 5; }
        
    return cell_idx;
}


__device__ void A_times_b(float x[], const float* A, float* b) {
    x[0] = A[0]*b[0] + A[1]*b[1] + A[2]*b[2] + A[3];
    x[1] = A[4]*b[0] + A[5]*b[1] + A[6]*b[2] + A[7];
    x[2] = A[8]*b[0] + A[9]*b[1] + A[10]*b[2] + A[11];
    return;
}

__device__ void A_times_b_linear(float x[], const float* A, float* b) {
    x[0] = A[0]*b[0] + A[1]*b[1] + A[2]*b[2];
    x[1] = A[4]*b[0] + A[5]*b[1] + A[6]*b[2];
    x[2] = A[8]*b[0] + A[9]*b[1] + A[10]*b[2];
    return;
}

__global__ void calcTrans_kernel(const int nP, const int batch_size,
                                 float* newpoints, const float* points, 
                                 const float* Trels, const int* nStepSolver,
                                 const int* ncx, const int* ncy, const int* ncz) {
    
    int point_index = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_index = blockIdx.y * blockDim.y + threadIdx.y;
    if(point_index < nP && batch_index < batch_size) {
        // Get point
        float point[3];
        point[0] = points[point_index];
        point[1] = points[point_index + nP];
        point[2] = points[point_index + 2*nP];
    
        // Define start index for the matrices belonging to this batch
        // batch * 12 params pr cell * 6 triangles pr cell * cell in x * cell in y * cell in z
        int start_idx = batch_index * 12 * 6 * ncx[0] * ncy[0] * ncz[0]; 
    
        // Iterate in nStepSolver
        int cellidx;
        for(int n = 0; n < nStepSolver[0]; n++){
            // Find cell idx
            cellidx = findcellidx(point, ncx[0], ncy[0], ncx[0]);
            
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
        newpoints[3 * nP * batch_index + point_index] = point[0];
        newpoints[3 * nP * batch_index + point_index + nP] = point[1];
        newpoints[3 * nP * batch_index + point_index + 2 * nP] = point[2];    
    }
    return;                            
}

void calcTrans_kernel_launcher(const GPUDevice& d, const int nP, const int batch_size,
                               float* newpoints, const float* points, 
                               const float* Trels, const int* nStepSolver, 
                               const int* ncx, const int* ncy, const int* ncz) {
    
    // Get GPU 2D cuda configuration
    dim3 bc((int)ceil(nP/256.0), batch_size);
    dim3 tpb(256, 1);
    
    // Launch kernel with configuration    
    calcTrans_kernel<<<bc, tpb, 0, d.stream()>>>(nP, batch_size, newpoints,
                                                 points, Trels, nStepSolver,
                                                 ncx, ncy, ncz);
    
    return;            
}


__global__ void  calcGrad_kernel(dim3 nthreads, const int n_theta, const int d, const int nP, const int nC,
                                float* grad, const float* points, const float* As, const float* Bs,
                                const int* nStepSolver, const int* ncx, const int* ncy, const int* ncz) {
        
        // Allocate memory for computations
        float p[3], v[3], pMid[3], vMid[3], q[3], qMid[3];
        float B_times_T[3], A_times_dTdAlpha[3], u[3], uMid[3];
        float Alocal[12], Blocal[12];
        int cellidx;
        
        // Thread index
        int point_index = threadIdx.x + blockIdx.x * blockDim.x;
        int batch_index = threadIdx.y + blockIdx.y * blockDim.y;
        int dim_index = threadIdx.z + blockIdx.z * blockIdx.z;
        
        // Make sure we are within bounds
        if(point_index < nP && batch_index < n_theta && dim_index < d){
            int index = 3 * nP * batch_index + point_index;
            int boxsize = 3 * nP * n_theta;
        
            // Define start index for the matrices belonging to this batch
            // batch * 12 params pr cell * 6 triangles pr cell * cell in x * cell in y * cell in z
            int start_idx = batch_index * 12 * 6 * ncx[0] * ncy[0] * ncz[0]; 
            
            // Initilize gradient to zero
            grad[dim_index*boxsize + index] = 0;
            grad[dim_index*boxsize + index + nP] = 0;
            grad[dim_index*boxsize + index + 2 * nP] = 0;

            // Get point
            p[0] = points[point_index];
            p[1] = points[point_index + nP];
            p[2] = points[point_index + 2 * nP];
            
            // Step size for solver
            double h = (1.0 / nStepSolver[0]);
        
            // Iterate a number of times
            for(int t=0; t<nStepSolver[0]; t++) {
                // Get current cell
                cellidx = findcellidx(p, ncx[0], ncy[0], ncz[0]);
                
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
                q[2] = grad[dim_index*boxsize + index + 2 * nP];
        
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
        
                // Ubcpdate gradient
                grad[dim_index * boxsize + index] = q[0];
                grad[dim_index * boxsize + index + nP] = q[1];
                grad[dim_index * boxsize + index + 2 * nP] = q[2];
                
                // Update p
                p[0] += vMid[0]*h;
                p[1] += vMid[1]*h;
                p[2] += vMid[2]*h;
            }
        }
        return;
}


void calcGrad_kernel_launcher(const GPUDevice& device, 
                              const int n_theta, const int d, const int nP, const int nC,
                              float* grad, const float* points, const float* As, const float* Bs,
                              const int* nStepSolver, const int* ncx, const int* ncy, const int* ncz){    
    // Get GPU 3D configuration
    Cuda3DLaunchConfig config = GetCuda3DLaunchConfigOWN(n_theta, nP, d);
    dim3 vtc = config.virtual_thread_count;
    dim3 tpb = config.thread_per_block;
    dim3 bc = config.block_count;
    
    // Launch kernel
    calcGrad_kernel<<<bc, tpb, 0, device.stream()>>>(vtc, n_theta, d, nP, 
                                                    nC, grad, points, As, Bs, 
                                                    nStepSolver, ncx, ncy, ncz);
    return;
}

#endif