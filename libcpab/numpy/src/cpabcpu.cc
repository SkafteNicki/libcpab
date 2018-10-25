#include <iostream>
#include <omp.h>

// Support functions
int stride(int ndim, const int* nc){
    int s = 0;
    if (ndim == 1){ s = 2; } // two parameters per cell
    if (ndim == 2){ s = 6 * 4; } // 6 parameters per triangle, 4 triangles per cell
    if (ndim == 3){ s = 12 * 6; } // 12 parameters per pyramid, 6 pyramids per cell
    for (int j = 0; j < ndim; j++) {
        s *= nc[j];
    }
    return s;
}

int param_pr_cell(int ndim){
    if(ndim == 1){ return 2; } // two parameters per cell
    if(ndim == 2){ return 6; } // 6 parameters per triangle
    if(ndim == 3){ return 12; } // 12 parameters per pyramid
    return -1; // should never happen, just there to silence the compiler
}

template <class FLOAT>
int mymin(int a, FLOAT b) {
    return !(b<a)?a:round(b);
}

template <class FLOAT>
int findcellidx1(const FLOAT* p, const int ncx) {           
    // Floor value to find cell
    int idx = std::floor(p[0] * ncx);
    idx = std::min(0, std::max(idx, ncx));
    return idx;
}

template <class FLOAT>
int findcellidx2(const FLOAT* p, const int ncx, const int ncy) {
    // Copy point                        
    double point[2];
    point[0] = p[0];
    point[1] = p[1];
    
    // Cell size
    const float inc_x = 1.0 / ncx;
    const float inc_y = 1.0 / ncy;
    
    // Find initial row, col placement
    double p0 = std::min((ncx * inc_x - 0.000000001), std::max(0.0, point[0]));
    double p1 = std::min((ncy * inc_y - 0.000000001), std::max(0.0, point[1]));
    double xmod = fmod(p0, inc_x);
    double ymod = fmod(p1, inc_y);
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

template <class FLOAT>
int findcellidx3(const FLOAT* p, const int nx, const int ny, const int nz) {
    // Copy point
    FLOAT point[3];
    point[0] = p[0];
    point[1] = p[1];
    point[2] = p[2];
    
    // Find row, col, layer placement
    int p0 = mymin( nx - 1 , std::max(0.0, point[0]*nx) );
    int p1 = mymin( ny - 1 , std::max(0.0, point[1]*ny) );
    int p2 = mymin( nz - 1 , std::max(0.0, point[2]*nz) );
    
    int cell_idx = 6 * ( p0 + p1*nx + p2*nx*ny );

    const FLOAT x = point[0]*nx - p0;
    const FLOAT y = point[1]*ny - p1;
    const FLOAT z = point[2]*nz - p2;

    if( x<=y && 1-x>y && x<z && 1-x>=z ){ cell_idx += 0; }
    if( x>y && x<=1-y && y<z && 1-y>=z ){ cell_idx += 1; }
    if( x>=z && x<1-z && y>=z && y<1-z ){ cell_idx += 2; }
    if( x<=z && x>1-z && y<=z && y>1-z ){ cell_idx += 3; }
    if( x<y && x>=1-y && y>z && 1-y<=z ){ cell_idx += 4; }
    if( x>=y && 1-x<y && x>z && 1-x<=z ){ cell_idx += 5; }

    return cell_idx;
}

template <class FLOAT>
int findcellidx(int ndim, const FLOAT* p, const int* nc){
    if(ndim == 1){ return findcellidx1(p, nc[0]);} 
    if(ndim == 2){ return findcellidx2(p, nc[0], nc[1]);}
    if(ndim == 3){ return findcellidx3(p, nc[0], nc[1], nc[2]); }
    return -1; // should never happen; just there to silence compiler
}

template <class FLOAT>
void A_times_b(int ndim, FLOAT x[], const FLOAT *A, const FLOAT *b) {
    if (ndim == 1) {
      x[0] = A[0]*b[0] + A[1];
    }
    if (ndim == 2) {
      x[0] = A[0]*b[0] + A[1]*b[1] + A[2];
      x[1] = A[3]*b[0] + A[4]*b[1] + A[5];    
    }
    if (ndim == 3) {
      x[0] = A[0]*b[0] + A[1]*b[1] + A[2]*b[2] + A[3];
      x[1] = A[4]*b[0] + A[5]*b[1] + A[6]*b[2] + A[7];
      x[2] = A[8]*b[0] + A[9]*b[1] + A[10]*b[2] + A[11];
    }
}

template <class FLOAT>
void A_times_b_linear(int ndim, FLOAT x[], const FLOAT *A, FLOAT *b) {
    if (ndim == 1) {
      x[0] = A[0]*b[0];
    }
    if (ndim == 2) {
      x[0] = A[0]*b[0] + A[1]*b[1];
      x[1] = A[3]*b[0] + A[4]*b[1];    
    }
    if (ndim == 3) {
      x[0] = A[0]*b[0] + A[1]*b[1] + A[2]*b[2];
      x[1] = A[4]*b[0] + A[5]*b[1] + A[6]*b[2];
      x[2] = A[8]*b[0] + A[9]*b[1] + A[10]*b[2];
    }
}

template <class FLOAT>
void cpab_forward_cpu(const FLOAT *points, // [ndim, n_points]
                      const FLOAT *trels,  // [batch_size, nC, ndim, ndim+1]
                      int ndim, int nP, int batch_size, int nstepsolver, 
                      const int *nc, // [ndim]
                      FLOAT *newpoints // [ndim, n_points]
                     )
{
    // Main loop
    #pragma omp parallel for
    for (int t = 0; t < batch_size; t++) { // for all batches
        // Start index for batch
        const int start_idx = t * stride(ndim, nc);
        
        #pragma omp parallel for
        for (int i = 0; i < nP; i++) { // for all points
            // Current point
            FLOAT point[ndim];
            for (int j = 0; j < ndim; j++) {
                point[j] = points[i + j*nP];
            }
            // Iterate in nStepSolver
            for (int n = 0; n < nstepsolver; n++) {
                // Find cell index
                int idx = findcellidx(ndim, point, nc);
                
                // Get mapping
                const FLOAT* tidx = trels + param_pr_cell(ndim)*idx + start_idx;  
                
                // Update points
                FLOAT newpoint[ndim];
                A_times_b(ndim, newpoint, tidx, point);
                for (int j = 0; j < ndim; j++){
                    point[j] = newpoint[j];
                }
            }
            // Update output
            for (int j = 0; j < ndim; j++) {
                newpoints[t * ndim * nP + i + j * nP] = point[j]; 
            }            
        }
    }
}


template <class FLOAT>
void cpab_backward_cpu(const FLOAT *points, // [ndim, nP]
                       const FLOAT *As, // [batch_size, nC, ndim, ndim+1]
                       const FLOAT *Bs, // [d, nC, ndim, ndim+1]
                       int ndim, int nP, int batch_size, int nstepsolver, int d, int nC, // scalar
                       const int *nc, // [ndim]
                       FLOAT *grad) { // [d, batch_size, ndim, nP]
    
    const FLOAT h = (1.0 / nstepsolver);
    const int params_size = param_pr_cell(ndim);

    // Loop over all transformations
    #pragma omp parallel for
    for (int batch_index = 0; batch_index < batch_size; batch_index++) {
        // Define start index for the matrices belonging to this batch
        const int start_idx = batch_index * stride(ndim, nc);
                
        const int boxsize = 2 * nP * batch_size;
 
        // For all points
        #pragma omp parallel for
        for (int point_index = 0; point_index < nP; point_index++) {
            // Make data structures for calculations
            FLOAT p[ndim], v[ndim], pMid[ndim], vMid[ndim], q[ndim], qMid[ndim];
            FLOAT B_times_T[ndim], A_times_dTdAlpha[ndim], u[ndim], uMid[ndim];
            //FLOAT Alocal[ndim*(ndim+1)], Blocal[ndim*(ndim+1)];

            // For all parameters in the transformations
            for (int dim_index = 0; dim_index < d; dim_index++) {
                const int index = 2 * nP * batch_index + point_index;
                
                // Get point
                for (int j = 0; j < ndim; j++) {
                    p[j] = points[point_index + j*index];
                }
                
                // Integrate velocity field
                for (int t = 0; t < nstepsolver; t++) {
                    // Get current cell
                    const int cellidx = findcellidx(ndim, p, nc);
                    
                    // Get index of A
                    const int As_idx = params_size*cellidx;
                    
                    // Extract local A
                    const FLOAT *Alocal = As + As_idx + start_idx;
                    //for (int i = 0; i < params_size; i++) {
                    //    Alocal[i] = (As + As_idx + start_idx)[i];
                    //}
                    
                    // Compute velocity at current location
                    A_times_b(ndim, v, Alocal, p);
                    
                    // Compute midpoint
                    for (int j = 0; j < ndim; j++) {
                        pMid[j] = p[j] + h*v[j]/2.0;
                    }
                    
                    // Compute velocity at midpoint
                    A_times_b(ndim, vMid, Alocal, pMid);
                    
                    // Get index of B
                    const int Bs_idx = params_size * dim_index * nC + As_idx;
                    
                    // Get local B
                    const FLOAT *Blocal = Bs + Bs_idx;
                    //for (int i = 0; i < params_size; i++) {
                    //    Blocal[i] = (Bs + Bs_idx)[i];
                    //}
                    
                    // Copy q
                    for (int j = 0; j < ndim; j++) {
                        q[j] = grad[dim_index*boxsize + index + j*nP];
                    }
                    
                    // Step 1: Compute u using the old location
                    // Find current RHS (term 1 + trem 2)
                    A_times_b(ndim, B_times_T, Blocal, p); // term 1
                    A_times_b_linear(ndim, A_times_dTdAlpha, Alocal, q); // term 2
                    
                    // Sum both terms
                    for (int j = 0; j < ndim; j++) {
                        u[j] = B_times_T[j] + A_times_dTdAlpha[j];
                    }
                    
                    // Step 2: Compute mid point
                    for (int j = 0; j < ndim; j++) {
                        qMid[j] = q[j] + h * u[j]/2.0;
                    }
                    
                    // Step 3: Compute uMid
                    A_times_b(ndim, B_times_T, Blocal, pMid);
                    A_times_b_linear(ndim, A_times_dTdAlpha, Alocal, qMid);
                    
                    // Sum both terms
                    for (int j = 0; j < ndim; j++) {
                        uMid[j] = B_times_T[j] + A_times_dTdAlpha[j];
                    }
                    
                    // Update q
                    for (int j = 0; j < ndim; j++) {
                        q[j] += uMid[j] * h;
                    }
                    
                    // Update gradient
                    for (int j = 0; j < ndim; j++) {
                        grad[dim_index * boxsize + index + j*nP] = q[j];
                    }
                    
                    // Update p
                    for (int j = 0; j < ndim; j++) {
                        p[j] += vMid[j]*h;
                    }
                    
                }
            }
        }
    }
}

