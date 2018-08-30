#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "cuda_kernel_helper.h"

__device__ int mymin(int a, double b) {
    return !(b<a)?a:round(b);
}

__device__ int findcellidx(const float* p, const int ncx) {           
    // Floor value to find cell
    int idx = floor(p[0] * ncx);
    idx = min(0, max(idx, ncx));
    return idx;                            
}

__device__ void A_times_b(float x[], const float* A, float* b) {
    x[0] = A[0]*b[0] + A[1];
    return;
}

__device__ void A_times_b_linear(float x[], const float* A, float* b) {
    x[0] = A[0]*b[0];
    return;
}

__global__ void calcTrans_kernel(const int nP, const int batch_size,
                                 float* newpoints, const float* points,
                                 const float* Trels, const int* nStepSolver,
                                 const int* ncx) {
    
    int point_index = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_index = blockIdx.y * blockDim.y + threadIdx.y;
    if(point_index < nP && batch_index < batch_size) {
        // Get point
        float point[1];
        point[0] = points[point_index];
    
        // Define start index for the matrices belonging to this batch
        // batch * 2 params pr cell * cell in x
        int start_idx = batch_index * 2 * ncx[0]; 
    
        // Iterate in nStepSolver
        int cellidx;
        for(int n = 0; n < nStepSolver[0]; n++){
            // Find cell idx
            cellidx = findcellidx(point, ncx[0]);
            
            // Extract the mapping in the cell
            const float* Trels_idx = Trels + 2*cellidx + start_idx;                
                     
            // Calculate trajectory of point
            float point_updated[1];                
            A_times_b(point_updated, Trels_idx, point);
            point[0] = point_updated[0];
        }
    
        // Copy to output
        newpoints[nP * batch_index + point_index] = point[0];
    }
    return;                            
}

void calcTrans_kernel_launcher(const GPUDevice& d, const int nP, const int batch_size,
                               float* newpoints, const float* points, 
                               const float* Trels, const int* nStepSolver, 
                               const int* ncx) {
    
    // Get GPU 2D cuda configuration
    dim3 bc((int)ceil(nP/256.0), batch_size);
    dim3 tpb(256, 1);
    
    // Launch kernel with configuration    
    calcTrans_kernel<<<bc, tpb, 0, d.stream()>>>(nP, batch_size,
                                                 newpoints, 
                                                 points, Trels, nStepSolver, ncx);
    
    return;            
}


__global__ void  calcGrad_kernel(dim3 nthreads, const int n_theta, const int d, const int nP, const int nC,
                                        float* grad, const float* points, const float* As, const float* Bs,
                                        const int* nStepSolver, const int* ncx) {
        
        // Allocate memory for computations
        float p[1], v[1], pMid[1], vMid[1], q[1], qMid[1];
        float B_times_T[1], A_times_dTdAlpha[1], u[1], uMid[1];
        float Alocal[2], Blocal[2];
        int cellidx;
        
        CUDA_AXIS_KERNEL_LOOP(batch_index, nthreads, x) {
            CUDA_AXIS_KERNEL_LOOP(point_index, nthreads, y) {
                CUDA_AXIS_KERNEL_LOOP(dim_index, nthreads, z) {
                    int index = nP * batch_index + point_index;
                    int boxsize = nP * n_theta;
                
                    // Define start index for the matrices belonging to this batch
                    // batch * 2 params pr cell * cell in x
                    int start_idx = batch_index * 2 * ncx[0]; 
                    
                    // Initilize gradient to zero
                    grad[dim_index*boxsize + index] = 0;

                    // Get point
                    p[0] = points[point_index];
                    
                    // Step size for solver
                    double h = (1.0 / nStepSolver[0]);
                
                    // Iterate a number of times
                    for(int t=0; t<nStepSolver[0]; t++) {
                        // Get current cell
                        cellidx = findcellidx(p, ncx[0]);
                        
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
        return;
}


void calcGrad_kernel_launcher(const GPUDevice& device, 
                              const int n_theta, const int d, const int nP, const int nC,
                              float* grad, const float* points, const float* As, const float* Bs,
                              const int* nStepSolver, const int* ncx){

    // Get GPU 3D configuration
    Cuda3DLaunchConfig config = GetCuda3DLaunchConfigOWN(n_theta, nP, d);
    dim3 vtc = config.virtual_thread_count;
    dim3 tpb = config.thread_per_block;
    dim3 bc = config.block_count;
    
    // Launch kernel
    calcGrad_kernel<<<bc, tpb, 0, device.stream()>>>(vtc, n_theta, d, nP, 
                                                    nC, grad, points, As, Bs, 
                                                    nStepSolver, ncx);
    return;
}

#endif