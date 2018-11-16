#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "cuda_kernel_helper.h"

__device__ int mymin(int a, double b) {
    return !(b<a)?a:round(b);
}

__device__ double cuda_fmod(double numer, double denom){
    double tquou = floor(numer / denom);
    return numer - tquou * denom;
}

__device__ int findcellidx(const float* p, const int ncx, const int ncy) {
    // Copy point
    double point[2];
    point[0] = p[0];
    point[1] = p[1];
    
    // Cell size
    const float inc_x = 1.0 / ncx;
    const float inc_y = 1.0 / ncy;
    
    // Find initial row, col placement
    double p0 = min((ncx * inc_x - 0.000000001), max(0.0, point[0]));
    double p1 = min((ncy * inc_y - 0.000000001), max(0.0, point[1]));

    double xmod = cuda_fmod((double)p0, (double)inc_x);
    double ymod = cuda_fmod((double)p1, (double)inc_y);

    double x = xmod / inc_x;
    double y = ymod / inc_y;
            
    int cell_idx =     mymin(ncx-1, (p0 - xmod) / inc_x) + 
                    mymin(ncy-1, (p1 - ymod) / inc_y) * ncx;        
    cell_idx *= 4;
            
    // Out of bound (left)
    if(point[0]<=0){
        if(point[1] <= 0 && point[1]/inc_y<point[0]/inc_x){
            // Nothing to do here
        } else if(point[1] >= ncy * inc_y && point[1]/inc_y-ncy > -point[0]/inc_x) {
            cell_idx += 2;
        } else {
            cell_idx += 3;
        }
        return cell_idx;
    }
            
    // Out of bound (right)
    if(point[0] >= ncx*inc_x){
        if(point[1]<=0 && -point[1]/inc_y > point[0]/inc_x - ncx){
            // Nothing to do here
        } else if(point[1] >= ncy*inc_y && point[1]/inc_y - ncy > point[0]/inc_x-ncx){
            cell_idx += 2;
        } else {
            cell_idx += 1;
        }
        return cell_idx;
    }
            
    // Out of bound (up)
    if(point[1] <= 0){
        return cell_idx;
    }
            
    // Out of bound (bottom)
    if(point[1] >= ncy*inc_y){
        cell_idx += 2;
        return cell_idx;
    }
            
    // OK, we are inbound
    if(x<y){
        if(1-x<y){
            cell_idx += 2;
        } else {
            cell_idx += 3;
        }
    } else if(1-x<y) {
        cell_idx += 1;
    }
                                
    return cell_idx;
}

__device__ void A_times_b(float x[], const float* A, float* b) {
    x[0] = A[0]*b[0] + A[1]*b[1] + A[2];
    x[1] = A[3]*b[0] + A[4]*b[1] + A[5];
    return;
}

__device__ void A_times_b_linear(float x[], const float* A, float* b) {
    x[0] = A[0]*b[0] + A[1]*b[1];
    x[1] = A[3]*b[0] + A[4]*b[1];
    return;
}

__global__ void calcTrans_kernel(const int nP, const int batch_size,
                                 float* newpoints, const float* points,
                                 const float* Trels, const int* nStepSolver,
                                 const int* ncx, const int* ncy) {
    
    int point_index = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_index = blockIdx.y * blockDim.y + threadIdx.y;
    if(point_index < nP && batch_index < batch_size) {
        // Get point
        float point[2];
        point[0] = points[point_index];
        point[1] = points[point_index + nP];
    
        // Define start index for the matrices belonging to this batch
        // batch * num_elem * 4 triangles pr cell * cell in x * cell in y
        int start_idx = batch_index * 6 * 4 * ncx[0] * ncy[0]; 
    
        // Iterate in nStepSolver
        int cellidx;
        for(int n = 0; n < nStepSolver[0]; n++){
            // Find cell idx
            cellidx = findcellidx(point, ncx[0], ncy[0]);
            
            // Extract the mapping in the cell
            const float* Trels_idx = Trels + 6*cellidx + start_idx;                
                     
            // Calculate trajectory of point
            float point_updated[2];                
            A_times_b(point_updated, Trels_idx, point);

            point[0] = point_updated[0];
            point[1] = point_updated[1];
        }
    
        // Copy to output
        newpoints[2 * nP * batch_index + point_index] = point[0];
        newpoints[2 * nP * batch_index + point_index + nP] = point[1];    
    }
    return;                            
}

void calcTrans_kernel_launcher(const GPUDevice& d, const int nP, const int batch_size,
                               float* newpoints, const float* points, 
                               const float* Trels, const int* nStepSolver, 
                               const int* ncx, const int* ncy) {
    
    // Get GPU 2D cuda configuration
    dim3 bc((int)ceil(nP/256.0), batch_size);
    dim3 tpb(256, 1);
    
    // Launch kernel with configuration    
    calcTrans_kernel<<<bc, tpb, 0, d.stream()>>>(nP, batch_size,
                                                 newpoints, 
                                                 points, Trels, nStepSolver,
                                                 ncx, ncy);
    
    return;            
}


__global__ void  calcGrad_kernel(dim3 nthreads, const int n_theta, const int d, const int nP, const int nC,
                                    float* grad, const float* points, const float* As, const float* Bs,
                                    const int* nStepSolver, const int* ncx, const int* ncy) {
        
        // Allocate memory for computations
        float p[2], v[2], pMid[2], vMid[2], q[2], qMid[2];
        float B_times_T[2], A_times_dTdAlpha[2], u[2], uMid[2];
        float Alocal[6], Blocal[6];
        int cellidx;
        
        // Thread index
        int point_index = threadIdx.x + blockIdx.x * blockDim.x;
        int batch_index = threadIdx.y + blockIdx.y * blockDim.y;
        int dim_index = threadIdx.z + blockIdx.z * blockIdx.z;
        
        // Make sure we are within bounds
        if(point_index < nP && batch_index < n_theta && dim_index < d){
            int index = 2 * nP * batch_index + point_index;
            int boxsize = 2 * nP * n_theta;
        
            // Define start index for the matrices belonging to this batch
            // batch * num_elem * 4 triangles pr cell * cell in x * cell in y
            int start_idx = batch_index * 6 * 4 * ncx[0] * ncy[0]; 
            
            // Initilize gradient to zero
            grad[dim_index*boxsize + index] = 0;
            grad[dim_index*boxsize + index + nP] = 0;

            // Get point
            p[0] = points[point_index];
            p[1] = points[point_index + nP];
            
            // Step size for solver
            double h = (1.0 / nStepSolver[0]);
        
            // Iterate a number of times
            for(int t=0; t<nStepSolver[0]; t++) {
                // Get current cell
                cellidx = findcellidx(p, ncx[0], ncy[0]);
                
                // Get index of A
                int As_idx = 6*cellidx;
                
                // Extract local A
                for(int i = 0; i < 6; i++){
                    Alocal[i] = (As + As_idx + start_idx)[i];
                }
                
                // Compute velocity at current location
                A_times_b(v, Alocal, p);
                
                // Compute midpoint
                pMid[0] = p[0] + h*v[0]/2.0;
                pMid[1] = p[1] + h*v[1]/2.0;
                
                // Compute velocity at midpoint
                A_times_b(vMid, Alocal, pMid);
                
                // Get index of B
                int Bs_idx = 6 * dim_index * nC + As_idx;
                
                // Get local B
                for(int i = 0; i < 6; i++){
                    Blocal[i] = (Bs + Bs_idx)[i];
                }
                
                // Copy q
                q[0] = grad[dim_index*boxsize + index];
                q[1] = grad[dim_index*boxsize + index + nP];
        
                // Step 1: Compute u using the old location
                // Find current RHS (term 1 + term 2)
                A_times_b(B_times_T, Blocal, p); // Term 1
                A_times_b_linear(A_times_dTdAlpha, Alocal, q); // Term 2
        
                // Sum both terms
                u[0] = B_times_T[0] + A_times_dTdAlpha[0];
                u[1] = B_times_T[1] + A_times_dTdAlpha[1];
        
                // Step 2: Compute mid "point"
                qMid[0] = q[0] + h * u[0]/2.0;
                qMid[1] = q[1] + h * u[1]/2.0;
        
                // Step 3: Compute uMid
                A_times_b(B_times_T, Blocal, pMid); // Term 1
                A_times_b_linear(A_times_dTdAlpha, Alocal, qMid); // Term 2
        
                // Sum both terms
                uMid[0] = B_times_T[0] + A_times_dTdAlpha[0];
                uMid[1] = B_times_T[1] + A_times_dTdAlpha[1];

                // Update q
                q[0] += uMid[0] * h;
                q[1] += uMid[1] * h;
        
                // Update gradient
                grad[dim_index * boxsize + index] = q[0];
                grad[dim_index * boxsize + index + nP] = q[1];
                
                // Update p
                p[0] += vMid[0]*h;
                p[1] += vMid[1]*h;
            }
        }
        return;
}


void calcGrad_kernel_launcher(const GPUDevice& device, 
                              const int n_theta, const int d, const int nP, const int nC,
                              float* grad, const float* points, const float* As, const float* Bs,
                              const int* nStepSolver, const int* ncx, const int* ncy){

    // Get GPU 3D configuration

    Cuda3DLaunchConfig config = GetCuda3DLaunchConfigOWN(n_theta, nP, d);
    dim3 vtc = config.virtual_thread_count;
    dim3 tpb = config.thread_per_block;
    dim3 bc = config.block_count;
    
    // Launch kernel
    calcGrad_kernel<<<bc, tpb, 0, device.stream()>>>(vtc, n_theta, d, nP, 
                                                    nC, grad, points, As, Bs, 
                                                    nStepSolver, ncx, ncy);
    return;
}

#endif