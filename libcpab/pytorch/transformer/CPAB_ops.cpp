#include <torch/torch.h>

// Support functions
int stride(int ndim, const int* nc){
    int s;
    switch(ndim) {
        case 1: s=2; // two parameters per cell
        case 2: s=6 * 4; // 4 triangles per cell, 6 parameters per triangle
        case 3: s=12 * 6; // 6 pyramids per cell, 12 parameters per paramid
    }
    for(int j = 0; j < ndim; j++) {
        s *= nc[j];
    }
    return s;
}

int param_pr_cell(int ndim){
    switch(ndim) {
        case 1: return 2; // two parameters per cell
        case 2: return 6; // 6 parameters per triangle
        case 3: return 12;  // 12 parameters per paramid
    }
}

int mymin(int a, double b) {
    return !(b<a)?a:round(b);
}
    
int findcellidx_1D(const float* p, const int ncx) {           
    // Floor value to find cell
    int idx = std::floor(p[0] * ncx);
    idx = std::min(0, std::max(idx, ncx));
    return idx;
}

int findcellidx_2D(const float* p, const int ncx, const int ncy) {
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

int findcellidx_3D(const float* p, const int nx, const int ny, const int nz) {
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

int findcellidx(int ndim, const float* p, const int* nc){
    switch(ndim) {
        case 1: return findcellidx_1D(p, nc[0]);
        case 2: return findcellidx_2D(p, nc[0], nc[1]);
        case 3: return findcellidx_3D(p, nc[0], nc[1], nc[2]);
    }
}
    
void A_times_b(int ndim, float x[], const float* A, float* b){
    switch(ndim) {
        case 1: 
            x[0] = A[0]*b[0] + A[1];
        case 2: 
            x[0] = A[0]*b[0] + A[1]*b[1] + A[2];
            x[1] = A[3]*b[0] + A[4]*b[1] + A[5];
        case 3:
            x[0] = A[0]*b[0] + A[1]*b[1] + A[2]*b[2] + A[3];
            x[1] = A[4]*b[0] + A[5]*b[1] + A[6]*b[2] + A[7];
            x[2] = A[8]*b[0] + A[9]*b[1] + A[10]*b[2] + A[11];
    }
    return;
}

// Function declaration
at::Tensor cpab_forward(at::Tensor points_in, //[ndim, n_points]
                        at::Tensor trels_in,  //[batch_size, nC, ndim, ndim+1]
                        at::Tensor nstepsolver_in, // scalar
                        at::Tensor nc_in){ // ndim length tensor
    
    // Problem size
    const auto ndim = points_in.size(0);
    const auto nP = points_in.size(1);
    const auto batch_size = trels_in.size(0);

    // Allocate output
    auto output = at::CPU(at::kFloat).zeros({batch_size, ndim, nP}); // [batch_size, ndim, nP]
    
    // Convert to pointers
    auto points = points_in.data<float>();
    auto trels = trels_in.data<float>();
    auto nstepsolver = nstepsolver_in.data<int>();
    auto nc = nc_in.data<int>();
    
    // Make pointers
    float point[ndim], newpoint[ndim];
    int t, i, j, n, idx, start_idx;
    for(t = 0; t < batch_size; t++) { // for all batches
        start_idx = t * stride(ndim, nc);
        for(i = 0; i < nP; i++) { // for all points
            // Current point            
            for(j = 0; j < ndim; j++){
                point[j] = points[i + j*nP];
            }
            // Iterate in nStepSolver
            for(n = 0; n < nstepsolver[0]; n++){
                // Find cell index
                idx = findcellidx(ndim, point, nc);
                
                // Get mapping
                const float* tidx = trels + param_pr_cell(ndim)*idx + start_idx;  
                               
                // Update points
                A_times_b(ndim, newpoint, tidx, point);
                for(j = 0; j < ndim; j++){
                    point[j] = newpoint[j];
                }
            }
            // Update output
            for(j = 0; j < ndim; j++){
                output[t * ndim * nP + i * ndim + j * nP] = point[j]; 
            }
        }
    }
    return output;
}

at::Tensor cpab_backward(at::Tensor points_in, // [ndim, nP]
                         at::Tensor As_in, // [n_theta, nC, ndim, ndim+1]
                         at::Tensor Bs_in, // [d, nC, ndim, ndim+1]
                         at::Tensor nstepsolver_in, // scalar
                         at::Tensor nc){ // ndim length tensor
    // Problem size
    const auto n_theta = As_in.size(0);
    const auto d = Bs_in.size(0);
    const auto ndim = points_in.size(0);
    const auto nP = points_in.size(1);
    const auto nC = Bs_in.size(1);
    
    // Allocate output
    auto output = at::CPU(at::kFloat).zeros({d, n_theta, ndim, nP});
    
    return output;
}
            
// Binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &cpab_forward, "Cpab transformer forward");
    m.def("backward", &cpab_backward, "Cpab transformer backward");
}