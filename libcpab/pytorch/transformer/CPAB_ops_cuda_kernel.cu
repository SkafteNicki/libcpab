#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

#define DIV_UP(a, b) (((a) + (b)-1) / (b))

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


// Support functions
__device__ int mymin(int a, double b) {
    return !(b<a)?a:round(b);
}

__device__ double cuda_fmod(double numer, double denom){
    double tquou = floor(numer / denom);
    return numer - tquou * denom;
}

__device__ int findcellidx_1D(const float* p, const int ncx) {           
    // Floor value to find cell
    int idx = floor(p[0] * ncx);
    idx = max(0, min(idx, ncx-1));
    return idx;                            
}

__device__ int findcellidx_2D(const float* p, const int ncx, const int ncy) {
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
            
    int cell_idx =  mymin(ncx-1, (p0 - xmod) / inc_x) + 
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

__device__ int findcellidx_3D(const float* p, const int ncx, const int ncy, const int ncz) {
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


__device__ void A_times_b_1D(float x[], const float* A, float* b) {
    x[0] = A[0]*b[0] + A[1];
    return;
}

__device__ void A_times_b_2D(float x[], const float* A, float* b) {
    x[0] = A[0]*b[0] + A[1]*b[1] + A[2];
    x[1] = A[3]*b[0] + A[4]*b[1] + A[5];
    return;
}

__device__ void A_times_b_3D(float x[], const float* A, float* b) {
    x[0] = A[0]*b[0] + A[1]*b[1] + A[2]*b[2] + A[3];
    x[1] = A[4]*b[0] + A[5]*b[1] + A[6]*b[2] + A[7];
    x[2] = A[8]*b[0] + A[9]*b[1] + A[10]*b[2] + A[11];
    return;
}

__device__ void A_times_b_linear_1D(float x[], const float* A, float* b) {
    x[0] = A[0]*b[0];
    return;
}

__device__ void A_times_b_linear_2D(float x[], const float* A, float* b) {
    x[0] = A[0]*b[0] + A[1]*b[1];
    x[1] = A[3]*b[0] + A[4]*b[1];
    return;
}

__device__ void A_times_b_linear_3D(float x[], const float* A, float* b) {
    x[0] = A[0]*b[0] + A[1]*b[1] + A[2]*b[2];
    x[1] = A[4]*b[0] + A[5]*b[1] + A[6]*b[2];
    x[2] = A[8]*b[0] + A[9]*b[1] + A[10]*b[2];
    return;
}


// Kernel declaration
__global__ void cpab_cuda_kernel_forward_1D(const int nP, const int batch_size,
                                            float* newpoints, const float* points,
                                            const float* Trels, const int* nStepSolver,
                                            const int* nc) {
    
    int point_index = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_index = blockIdx.y * blockDim.y + threadIdx.y;
    if(point_index < nP && batch_index < batch_size) {
        // Get point
        float point[1];
        point[0] = points[point_index];
    
        // Define start index for the matrices belonging to this batch
        // batch * 2 params pr cell * cell in x
        int start_idx = batch_index * 2 * nc[0]; 
    
        // Iterate in nStepSolver
        int cellidx;
        for(int n = 0; n < nStepSolver[0]; n++){
            // Find cell idx
            cellidx = findcellidx_1D(point, nc[0]);
            
            // Extract the mapping in the cell
            const float* Trels_idx = Trels + 2*cellidx + start_idx;                
                     
            // Calculate trajectory of point
            float point_updated[1];                
            A_times_b_1D(point_updated, Trels_idx, point);
            point[0] = point_updated[0];
        }
    
        // Copy to output
        newpoints[nP * batch_index + point_index] = point[0];
    }
    return;                            
}

__global__ void cpab_cuda_kernel_forward_2D(const int nP, const int batch_size,
                                            float* newpoints, const float* points,
                                            const float* Trels, const int* nStepSolver,
                                            const int* nc) {

    int point_index = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_index = blockIdx.y * blockDim.y + threadIdx.y;
    if(point_index < nP && batch_index < batch_size) {
        // Get point
        float point[2];
        point[0] = points[point_index];
        point[1] = points[point_index + nP];
    
        // Define start index for the matrices belonging to this batch
        // batch * num_elem * 4 triangles pr cell * cell in x * cell in y
        int start_idx = batch_index * 6 * 4 * nc[0] * nc[1]; 
    
        // Iterate in nStepSolver
        int cellidx;
        for(int n = 0; n < nStepSolver[0]; n++){
            // Find cell idx
            cellidx = findcellidx_2D(point, nc[0], nc[1]);
            
            // Extract the mapping in the cell
            const float* Trels_idx = Trels + 6*cellidx + start_idx;                
                     
            // Calculate trajectory of point
            float point_updated[2];                
            A_times_b_2D(point_updated, Trels_idx, point);

            point[0] = point_updated[0];
            point[1] = point_updated[1];
        }
    
        // Copy to output
        newpoints[2 * nP * batch_index + point_index] = point[0];
        newpoints[2 * nP * batch_index + point_index + nP] = point[1];    
    }
    return;                            
}

__global__ void cpab_cuda_kernel_forward_3D(const int nP, const int batch_size,
                                            float* newpoints, const float* points, 
                                            const float* Trels, const int* nStepSolver,
                                            const int* nc) {
    
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
        int start_idx = batch_index * 12 * 6 * nc[0] * nc[1] * nc[2]; 
    
        // Iterate in nStepSolver
        int cellidx;
        for(int n = 0; n < nStepSolver[0]; n++){
            // Find cell idx
            cellidx = findcellidx_3D(point, nc[0], nc[1], nc[2]);
            
            // Extract the mapping in the cell
            const float* Trels_idx = Trels + 12*cellidx + start_idx;                
                     
            // Calculate trajectory of point
            float point_updated[3];                
            A_times_b_3D(point_updated, Trels_idx, point);

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

__global__ void cpab_cuda_kernel_backward_1D(dim3 nthreads, const int n_theta, const int d, const int nP, const int nC,
                                             float* grad, const float* points, const float* As, const float* Bs,
                                             const int* nStepSolver, const int* nc) {
        
        // Allocate memory for computations
        float p[1], v[1], pMid[1], vMid[1], q[1], qMid[1];
        float B_times_T[1], A_times_dTdAlpha[1], u[1], uMid[1];
        float Alocal[2], Blocal[2];
        int cellidx;
        
        // Thread index
        int point_index = threadIdx.x + blockIdx.x * blockDim.x;
        int batch_index = threadIdx.y + blockIdx.y * blockDim.y;
        int dim_index = threadIdx.z + blockIdx.z * blockIdx.z;
        
        // Make sure we are within bounds
        if(point_index < nP && batch_index < n_theta && dim_index < d){
            int index = nP * batch_index + point_index;
            int boxsize = nP * n_theta;
        
            // Define start index for the matrices belonging to this batch
            // batch * 2 params pr cell * cell in x
            int start_idx = batch_index * 2 * nc[0]; 
            
            // Initilize gradient to zero
            grad[dim_index*boxsize + index] = 0;

            // Get point
            p[0] = points[point_index];
            
            // Step size for solver
            double h = (1.0 / nStepSolver[0]);
        
            // Iterate a number of times
            for(int t=0; t<nStepSolver[0]; t++) {
                // Get current cell
                cellidx = findcellidx_1D(p, nc[0]);
                
                // Get index of A
                int As_idx = 2*cellidx;
                
                // Extract local A
                for(int i = 0; i < 2; i++){
                    Alocal[i] = (As + As_idx + start_idx)[i];
                }
                
                // Compute velocity at current location
                A_times_b_1D(v, Alocal, p);
                
                // Compute midpoint
                pMid[0] = p[0] + h*v[0]/2.0;
                
                // Compute velocity at midpoint
                A_times_b_1D(vMid, Alocal, pMid);
                
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
                A_times_b_1D(B_times_T, Blocal, p); // Term 1
                A_times_b_linear_1D(A_times_dTdAlpha, Alocal, q); // Term 2
        
                // Sum both terms
                u[0] = B_times_T[0] + A_times_dTdAlpha[0];
        
                // Step 2: Compute mid "point"
                qMid[0] = q[0] + h * u[0]/2.0;
        
                // Step 3: Compute uMid
                A_times_b_1D(B_times_T, Blocal, pMid); // Term 1
                A_times_b_linear_1D(A_times_dTdAlpha, Alocal, qMid); // Term 2
        
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
        return;
}

__global__ void   cpab_cuda_kernel_backward_2D(dim3 nthreads, const int n_theta, const int d, const int nP, const int nC,
                                               float* grad, const float* points, const float* As, const float* Bs,
                                               const int* nStepSolver, const int* nc) {
        
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
            int start_idx = batch_index * 6 * 4 * nc[0] * nc[1]; 
            
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
                cellidx = findcellidx_2D(p, nc[0], nc[1]);
                
                // Get index of A
                int As_idx = 6*cellidx;
                
                // Extract local A
                for(int i = 0; i < 6; i++){
                    Alocal[i] = (As + As_idx + start_idx)[i];
                }
                
                // Compute velocity at current location
                A_times_b_2D(v, Alocal, p);
                
                // Compute midpoint
                pMid[0] = p[0] + h*v[0]/2.0;
                pMid[1] = p[1] + h*v[1]/2.0;
                
                // Compute velocity at midpoint
                A_times_b_2D(vMid, Alocal, pMid);
                
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
                A_times_b_2D(B_times_T, Blocal, p); // Term 1
                A_times_b_linear_2D(A_times_dTdAlpha, Alocal, q); // Term 2
        
                // Sum both terms
                u[0] = B_times_T[0] + A_times_dTdAlpha[0];
                u[1] = B_times_T[1] + A_times_dTdAlpha[1];
        
                // Step 2: Compute mid "point"
                qMid[0] = q[0] + h * u[0]/2.0;
                qMid[1] = q[1] + h * u[1]/2.0;
        
                // Step 3: Compute uMid
                A_times_b_2D(B_times_T, Blocal, pMid); // Term 1
                A_times_b_linear_2D(A_times_dTdAlpha, Alocal, qMid); // Term 2
        
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

__global__ void   cpab_cuda_kernel_backward_3D(dim3 nthreads, const int n_theta, const int d, const int nP, const int nC,
                                               float* grad, const float* points, const float* As, const float* Bs,
                                               const int* nStepSolver, const int* nc) {
        
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
            int start_idx = batch_index * 12 * 6 * nc[0] * nc[1] * nc[2]; 
            
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
                cellidx = findcellidx_3D(p, nc[0], nc[1], nc[2]);
                
                // Get index of A
                int As_idx = 12*cellidx;
                
                // Extract local A
                for(int i = 0; i < 12; i++){
                    Alocal[i] = (As + As_idx + start_idx)[i];
                }
                
                // Compute velocity at current location
                A_times_b_3D(v, Alocal, p);
                
                // Compute midpoint
                pMid[0] = p[0] + h*v[0]/2.0;
                pMid[1] = p[1] + h*v[1]/2.0;
                pMid[2] = p[2] + h*v[2]/2.0;
                
                // Compute velocity at midpoint
                A_times_b_3D(vMid, Alocal, pMid);
                
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
                A_times_b_3D(B_times_T, Blocal, p); // Term 1
                A_times_b_linear_3D(A_times_dTdAlpha, Alocal, q); // Term 2
        
                // Sum both terms
                u[0] = B_times_T[0] + A_times_dTdAlpha[0];
                u[1] = B_times_T[1] + A_times_dTdAlpha[1];
                u[2] = B_times_T[2] + A_times_dTdAlpha[2];
        
                // Step 2: Compute mid "point"
                qMid[0] = q[0] + h * u[0]/2.0;
                qMid[1] = q[1] + h * u[1]/2.0;
                qMid[2] = q[2] + h * u[2]/2.0;
        
                // Step 3: Compute uMid
                A_times_b_3D(B_times_T, Blocal, pMid); // Term 1
                A_times_b_linear_3D(A_times_dTdAlpha, Alocal, qMid); // Term 2
        
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

// Kernel launcher declaration
at::Tensor cpab_cuda_forward(at::Tensor points_in, 
                             at::Tensor trels_in,  
                             at::Tensor nstepsolver_in, 
                             at::Tensor nc_in, 
							 at::Tensor output){
    // Problem size
    const auto ndim = points_in.size(0);
    const auto nP = points_in.size(1);
    const auto batch_size = trels_in.size(0);        
    
    // Kernel configuration
    dim3 bc((int)ceil(nP/256.0), batch_size);
    dim3 tpb(256, 1);
    
    // Launch kernel
    // We do it in this way, since dynamically allocating memory in CUDA sucks!
    if(ndim == 1){
         cpab_cuda_kernel_forward_1D<<<bc, tpb>>>(nP, batch_size,
                                                  output.data<float>(),
                                                  points_in.data<float>(),
                                                  trels_in.data<float>(),
                                                  nstepsolver_in.data<int>(),
                                                  nc_in.data<int>());
	}
	if(ndim == 2){
         cpab_cuda_kernel_forward_2D<<<bc, tpb>>>(nP, batch_size,
                                                  output.data<float>(),
                                                  points_in.data<float>(),
                                                  trels_in.data<float>(),
                                                  nstepsolver_in.data<int>(),
                                                  nc_in.data<int>());
	}
	if(ndim == 3){
        	cpab_cuda_kernel_forward_3D<<<bc, tpb>>>(nP, batch_size,
                                                	output.data<float>(),
                                                  points_in.data<float>(),
                                                  trels_in.data<float>(),
                                                  nstepsolver_in.data<int>(),
                                                  nc_in.data<int>());
    }
    gpuErrchk( cudaPeekAtLastError() );                                             
    return output;           
}

at::Tensor cpab_cuda_backward(at::Tensor points_in, 
                              at::Tensor As_in, 
                              at::Tensor Bs_in, 
                              at::Tensor nstepsolver_in,
                              at::Tensor nc_in,
                              at::Tensor output){
                              
    // Problem size
    const auto n_theta = As_in.size(0);
    const auto d = Bs_in.size(0);
    const auto ndim = points_in.size(0);
    const auto nP = points_in.size(1);
    const auto nC = Bs_in.size(1);
    
    // Kernel configuration
    dim3 tpb = dim3(std::min((int)nP, 128), std::min((int)n_theta, 4), std::min((int)d, 1));
    dim3 bc = dim3(DIV_UP(nP, tpb.x), DIV_UP(n_theta, tpb.y), DIV_UP(d, tpb.z));
    dim3 vtc = dim3(nP, n_theta, d);
    
    // Launch kernel
    // We do it in this way, since dynamically allocating memory in CUDA sucks!
	if(ndim == 1){
         cpab_cuda_kernel_backward_1D<<<bc, tpb>>>(vtc, n_theta, d, nP, nC,
                                                   output.data<float>(), 
                                                   points_in.data<float>(), 
                                                   As_in.data<float>(), 
                                                   Bs_in.data<float>(),
                                                   nstepsolver_in.data<int>(), 
                                                   nc_in.data<int>());
	}
	if(ndim == 2){
         cpab_cuda_kernel_backward_2D<<<bc, tpb>>>(vtc, n_theta, d, nP, nC,
                                                   output.data<float>(), 
                                                   points_in.data<float>(), 
                                                   As_in.data<float>(), 
                                                   Bs_in.data<float>(),
                                                   nstepsolver_in.data<int>(), 
                                                   nc_in.data<int>());
	}
 	if(ndim == 3){
         cpab_cuda_kernel_backward_3D<<<bc, tpb>>>(vtc, n_theta, d, nP, nC,
                                                   output.data<float>(), 
                                                   points_in.data<float>(), 
                                                   As_in.data<float>(), 
                                                   Bs_in.data<float>(),
                                                   nstepsolver_in.data<int>(), 
                                                   nc_in.data<int>());
    }
    gpuErrchk( cudaPeekAtLastError() );                                               
    return output;
}