#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 09:59:09 2017

@author: nsde
"""

#%%
import numpy as np
from .utility import make_hashable

#%%
def get_constrain_matrix_3D(nc, domain_min, domain_max,
                            valid_outside, zero_boundary, volume_perservation):
    tess = Tesselation3D(nc, domain_min, domain_max, zero_boundary, volume_perservation)
    return tess.get_constrain_matrix()

#%%
class Tesselation:
    """ Base tesselation class. This function is not meant to be called,
        but descripes the base structure that needs to be implemented in
        1D, 2D, and 3D. Additionally, some functionallity is shared across
        the different dimensions.
        
    Args:
        nc: list with number of cells
        domain_min: value of the lower bound(s) of the domain
        domain_max: value of the upper bound(s) of the domain
        zero_boundary: bool, if true the velocity is zero on the boundary
        volume_perservation: bool, if true volume is perserved
        
    Methods that should not be implemented in subclasses:
        @get_constrain_matrix:
        @get_cell_centers:
        @create_continuity_constrains:
        @create_zero_trace_constrains:
            
    Methods that should be implemented in subclasses:
        @find_verts:
        @find_verts_outside:
        @create_zero_boundary_constrains:
        
    """
    def __init__(self, nc, domain_min, domain_max,
                 zero_boundary = True, volume_perservation=False):
        """ Initilization of the class that create the constrain matrix L
        Arguments:
            nc: list, number of cells in each dimension
            domain_min: list, lower domain bound in each dimension
            domain_max: list, upper domain bound in each dimension
            zero_boundary: bool, determines is the velocity at the boundary is zero
            volume_perservation: bool, determine if the transformation is
                volume perservating
        """
        
        # Save parameters
        self.nc = nc
        self.domain_min = domain_min
        self.domain_max = domain_max
        self.zero_boundary = zero_boundary
        self.volume_perservation = volume_perservation
    
        # Get vertices
        self.find_verts()
        
        # Find shared vertices
        self.find_shared_verts()
        
        # find auxility vertices, if transformation is valid outside
        if not zero_boundary: self.find_verts_outside()
        
        # Get continuity constrains
        self.L = self.create_continuity_constrains()
        
        # If zero boundary, add constrains
        if zero_boundary:
            temp = self.create_zero_boundary_constrains()
            self.L = np.concatenate((self.L, temp), axis=0)
            
        # If volume perservation, add constrains
        if volume_perservation:
            temp = self.create_zero_trace_constrains()
            self.L = np.concatenate((self.L, temp), axis=0)
    
    def get_constrain_matrix(self):
        """ Function for getting the constructed constrain matrix L """
        return self.L
    
    def get_cell_centers(self):
        """ Get the centers of all the cells """
        return np.mean(self.verts[:,:,:self.ndim], axis=1)
    
    def find_verts(self):
        """ Function that should find the different vertices of all cells in
            the tesselation """
        raise NotImplementedError
        
    def find_shared_verts(self):
        """ Find pairs of cells that share ndim-vertices. It is these pairs,
            where we need to add continuity constrains at """
        # Iterate over all pairs of cell to find cells with intersecting cells
        shared_v, shared_v_idx = [ ], [ ]
        for i in range(self.nC):
            for j in range(self.nC):
                if i != j:
                    vi = make_hashable(self.verts[i])
                    vj = make_hashable(self.verts[j])
                    shared_verts = set(vi).intersection(vj)
                    if len(shared_verts) >= self.ndim and (j,i) not in shared_v_idx:
                        shared_v.append(list(shared_verts)[:self.ndim])
                        shared_v_idx.append((i,j))
        
        # Save result
        self.shared_v = np.asarray(shared_v)
        self.shared_v_idx = shared_v_idx
        
    def find_verts_outside(self):
        """ If the transformation should be valid outside, this function should
            add additional auxilliry points to the tesselation that secures
            continuity outside the domain """
        raise NotImplementedError
        
    def create_continuity_constrains(self):
        """ This function goes through all pairs (i,j) of cells that share a
            boundary. In N dimension we need to add N*N constrains (one for each
            dimension times one of each vertex in the boundary) """
        Ltemp = np.zeros(shape=(0,self.n_params*self.nC))
        for idx, (i,j) in enumerate(self.shared_v_idx):
            for vidx in range(self.ndim):
                for k in range(self.ndim):
                    index1 = self.n_params*i + k*(self.ndim+1)
                    index2 = self.n_params*j + k*(self.ndim+1)
                    row = np.zeros(shape=(1,self.n_params*self.nC))
                    row[0,index1:index1+(self.ndim+1)] = self.shared_v[idx][vidx]
                    row[0,index2:index2+(self.ndim+1)] = -self.shared_v[idx][vidx]
                    Ltemp = np.vstack((Ltemp, row))
        return Ltemp
        
    def create_zero_boundary_constrains(self):
        """ Function that creates a constrain matrix L, containing constrains that
            secure 0 velocity at the boundary """
        raise NotImplementedError
        
    def create_zero_trace_constrains(self):
        """ The volume perservation constrains, that corresponds to the trace
            of each matrix being 0. These can be written general for all dims."""
        Ltemp = np.zeros((self.nC, self.n_params*self.nC))
        row = np.concatenate((np.eye(self.ndim), np.zeros((self.ndim, 1))), axis=1).flatten()
        for c in range(self.nC):
            Ltemp[c,self.n_params*c:self.n_params*(c+1)] = row
        return Ltemp
    
class Tesselation3D(Tesselation):
    def __init__(self, nc, domain_min, domain_max,
                 zero_boundary = True, volume_perservation=False):
        # 1D parameters
        self.n_params = 12
        self.nC = 6*np.prod(nc) # 6 triangle per cell
        self.ndim = 3
        
        # Initialize super class
        super(Tesselation3D, self).__init__(nc, domain_min, domain_max,
             zero_boundary, volume_perservation)
    
    def find_verts(self):
        Vx = np.linspace(self.domain_min[0], self.domain_max[0], self.nc[0]+1)
        Vy = np.linspace(self.domain_min[1], self.domain_max[1], self.nc[1]+1)
        Vz = np.linspace(self.domain_min[2], self.domain_max[2], self.nc[2]+1)
        
        # Find cell index and verts for each cell
        cells, verts = [ ], [ ]
        for i in range(self.nc[2]):
            for j in range(self.nc[1]):        
                for k in range(self.nc[0]):
                    cnt = tuple([(Vx[k]+Vx[k+1])/2.0, (Vy[j]+Vy[j+1])/2.0, (Vz[i]+Vz[i+1])/2.0, 1])
                    lnl = tuple([Vx[k], Vy[j], Vz[i], 1])
                    lnu = tuple([Vx[k], Vy[j], Vz[i+1], 1])
                    lfl = tuple([Vx[k], Vy[j+1], Vz[i], 1])
                    lfu = tuple([Vx[k], Vy[j+1], Vz[i+1], 1])
                    rnl = tuple([Vx[k+1], Vy[j], Vz[i], 1])
                    rnu = tuple([Vx[k+1], Vy[j], Vz[i+1], 1])
                    rfl = tuple([Vx[k+1], Vy[j+1], Vz[i], 1])
                    rfu = tuple([Vx[k+1], Vy[j+1], Vz[i+1], 1])
                    
                    verts.append((cnt, lnl, lnu, lfl, lfu))
                    verts.append((cnt, lnl, lnu, rnl, rnu))
                    verts.append((cnt, lnl, lfl, rnl, rfl))
                    verts.append((cnt, lnu, lfu, rnu, rfu))                    
                    verts.append((cnt, lfl, lfu, rfl, rfu))
                    verts.append((cnt, rnl, rnu, rfl, rfu))

                    cells.append((k,j,i,0))
                    cells.append((k,j,i,1))
                    cells.append((k,j,i,2))
                    cells.append((k,j,i,3))
                    cells.append((k,j,i,4))
                    cells.append((k,j,i,5))
        
        # Convert to array
        self.verts = np.asarray(verts)
        self.cells = cells

    def find_verts_outside(self):
        shared_verts, shared_verts_idx = [ ], [ ]
        # Iterate over all pairs of cells
        for i in range(self.nC):
            for j in range(self.nC):
                if i != j:
                    # Add constrains for each side
                    for d in range(self.ndim):
                        # Get cell vertices
                        vi = self.verts[i]    
                        vj = self.verts[j]
                        
                        # Conditions for adding a constrain
                        upper_cond = sum(vi[:,d]==self.domain_min[d]) == 4 and \
                                     sum(vj[:,d]==self.domain_min[d]) == 4
                        lower_cond = sum(vi[:,d]==self.domain_max[d]) == 4 and \
                                     sum(vj[:,d]==self.domain_max[d]) == 4
                        dist_cond = np.sqrt(np.sum(np.power(vi[0] - vj[0], 2.0))) <= 1/self.nc[d]+1e-5
                        idx_cond = (j,i) not in shared_verts_idx
                        
                        if (upper_cond or lower_cond) and dist_cond and idx_cond:
                            # Find the shared points
                            vi = make_hashable(vi)
                            vj = make_hashable(vj)
                            sv = set(vi).intersection(vj)
                            # Add auxilliry point halfway between the centers
                            center = [(v1 + v2) / 2.0 for v1,v2 in zip(vi[0], vj[0])]
                            shared_verts.append(list(sv.union([tuple(center)])))
                            shared_verts_idx.append((i,j))
        
        # Add to already found pairs
        if shared_verts:
            self.shared_v = np.concatenate((self.shared_v, np.asarray(shared_verts)))
            self.shared_v_idx += shared_verts_idx
        
    def create_zero_boundary_constrains(self):
        nx, ny, nz = self.nc
        Ltemp = np.zeros(shape=(2*((nx+1)*(ny+1) + (nx+1)*(nz+1) + (ny+1)*(nz+1)), self.n_params*self.nC))
        # xy-plane
        sr = 0
        z = [0,0,0]
        for i in [0,nz-1]:
            for j in range(ny+1):
                for k in range(nx+1):
                    c_idx = 6 * ( nx*ny*i + nx*min(j,ny-1) + min(k,nx-1) )
                    c_idx += 2 if i==0 else 3

                    vrt = [ k/nx, j/ny, i/(nz-1) ]

                    Ltemp[sr,c_idx*12:(c_idx+1)*12] = np.matrix(z+z+vrt+[0,0,1])
                    sr += 1
        # xz-plane
        for j in [0,ny-1]:
            for i in range(nz+1):
                for k in range(nx+1):
                    c_idx = 6 * ( nx*ny*min(i,nz-1) + nx*j + min(k,nx-1) )
                    c_idx += 1 if j==0 else 4

                    vrt = [ k/nx, j/(ny-1), i/nz ]

                    Ltemp[sr,c_idx*12:(c_idx+1)*12] = np.matrix(z+vrt+z+[0,1,0])
                    sr += 1
        # yz-plane
        for k in [0,nx-1]:
            for i in range(nz+1):
                for j in range(ny+1):
                    c_idx = 6 * ( nx*ny*min(i,nz-1) + nx*min(j,ny-1) + k )
                    c_idx += 0 if k==0 else 5
            
                    vrt = [ k/(nx-1), j/ny, i/nz ]

                    Ltemp[sr,c_idx*12:(c_idx+1)*12] = np.matrix(vrt+z+z+[1,0,0])
                    sr += 1
                    
        return Ltemp

#%%
def get_constrain_matrix_3D_old(nc, domain_min, domain_max,
                            valid_outside, zero_boundary, volume_perservation):
    nx, ny, nz = nc
    
    N = 6 * 12*nx*ny*nz
    if zero_boundary:
        S = 9*( nx*ny*(nz-1) + nx*(ny-1)*nz + (nx-1)*ny*nz + 12*nx*ny*nz  ) + 2*( (nx+1)*(ny+1) + (nx+1)*(nz+1) + (ny+1)*(nz+1) )
    else:
        S = 9*( nx*ny*(nz-1) + nx*(ny-1)*nz + (nx-1)*ny*nz + 12*nx*ny*nz )
    L = np.zeros((S,N))
    z = [0,0,0]
    c = 0

    # Parametrizing continuity condition surfaces in yz-plane
    for i in range(nz):
        for j in range(ny):
            for k in range(nx-1):
                idx = nx*ny*i + nx*j + k

                c_idx_left = 6*idx + 5
                c_idx_right = 6*(idx+1)

                rnl = [ (k+1)/nx, j/ny, i/nz ]
                rfl = [ (k+1)/nx, (j+1)/ny, i/nz ]
                rfu = [ (k+1)/nx, (j+1)/ny, (i+1)/nz ]

                L,c = set_continuity_constraint(rnl, rfl, rfu, c_idx_left, c_idx_right, L, c)

    # Parametrizing continuity conditions surfaces in xz-plane
    for i in range(nz):
        for j in range(ny-1):
            for k in range(nx):
                idx = nx*ny*i + nx*j + k

                c_idx_near = idx*6 + 4
                c_idx_far = (idx+nx)*6 + 1

                lfl = [ k/nx, (j+1)/ny, i/nz ]
                rfl = [ (k+1)/nx, (j+1)/ny, i/nz ]
                rfu = [ (k+1)/nx, (j+1)/ny, (i+1)/nz ]

                L,c = set_continuity_constraint(lfl, rfl, rfu, c_idx_near, c_idx_far, L, c)

    # Parametrizing continuity conditions surfaces in xy-plane
    for i in range(nz-1):
        for j in range(ny):
            for k in range(nx):
                idx = nx*ny*i + nx*j + k

                c_idx_lower = idx*6 + 3
                c_idx_upper = (idx + nx*ny)*6 + 2

                lnu = [ k/nx, j/ny, (i+1)/nz ]
                rnu = [ (k+1)/nx, j/ny, (i+1)/nz ]
                rfu = [ (k+1)/nx, (j+1)/ny, (i+1)/nz ]

                L,c = set_continuity_constraint(lnu, rnu, rfu, c_idx_lower, c_idx_upper, L, c)

    # Parametrizing continuity conditions within boxes
    for i in range(nz):
        for j in range(ny):
            for k in range(nx):
                c_idx = 6 * ( nx*ny*i + nx*j + k )

                # Corner and center voxels of box
                cnt = [(k+0.5)/nx, (j+0.5)/ny, (i+0.5)/nz ]
                lnl = [ k/nx, j/ny, i/nz ]
                lnu = [ k/nx, j/ny, (i+1)/nz ]
                lfl = [ k/nx, (j+1)/ny, i/nz]
                lfu = [ k/nx, (j+1)/ny, (i+1)/nz]
                rnl = [ (k+1)/nx, j/ny, i/nz]
                rnu = [ (k+1)/nx, j/ny, (i+1)/nz]
                rfl = [ (k+1)/nx, (j+1)/ny, i/nz]
                rfu = [ (k+1)/nx, (j+1)/ny, (i+1)/nz]

                L,c = set_continuity_constraint(cnt, lnl, lnu, c_idx, c_idx+1, L, c)
                L,c = set_continuity_constraint(cnt, lnl, lfl, c_idx, c_idx+2, L, c)
                L,c = set_continuity_constraint(cnt, lnu, lfu, c_idx, c_idx+3, L, c)
                L,c = set_continuity_constraint(cnt, lfl, lfu, c_idx, c_idx+4, L, c)

                L,c = set_continuity_constraint(cnt, lnl, rnl, c_idx+1, c_idx+2, L, c)
                L,c = set_continuity_constraint(cnt, lnu, rnu, c_idx+1, c_idx+3, L, c)
                L,c = set_continuity_constraint(cnt, rnl, rnu, c_idx+1, c_idx+5, L, c)

                L,c = set_continuity_constraint(cnt, lfl, rfl, c_idx+2, c_idx+4, L, c)
                L,c = set_continuity_constraint(cnt, rnl, rfl, c_idx+2, c_idx+5, L, c)

                L,c = set_continuity_constraint(cnt, lfu, rfu, c_idx+3, c_idx+4, L, c)
                L,c = set_continuity_constraint(cnt, rnu, rfu, c_idx+3, c_idx+5, L, c)

                L,c = set_continuity_constraint(cnt, rfl, rfu, c_idx+4, c_idx+5, L, c)

    # Setting up image boundary conditions
    if zero_boundary:
        ## Boundary points
        # xy-plane
        sr = 0
        for i in [0,nz-1]:
            for j in range(ny+1):
                for k in range(nx+1):
                    c_idx = 6 * ( nx*ny*i + nx*min(j,ny-1) + min(k,nx-1) )
                    c_idx += 2 if i==0 else 3

                    vrt = [ k/nx, j/ny, i/(nz-1) ]

                    m = np.matrix(z+z+vrt+[0,0,1])

                    L[c*9+sr:c*9+sr+1,c_idx*12:(c_idx+1)*12] = m
                    sr += 1
        # xz-plane
        for j in [0,ny-1]:
            for i in range(nz+1):
                for k in range(nx+1):
                    c_idx = 6 * ( nx*ny*min(i,nz-1) + nx*j + min(k,nx-1) )
                    c_idx += 1 if j==0 else 4

                    vrt = [ k/nx, j/(ny-1), i/nz ]

                    m = np.matrix(z+vrt+z+[0,1,0])

                    L[c*9+sr:c*9+sr+1,c_idx*12:(c_idx+1)*12] = m
                    sr += 1
        # yz-plane
        for k in [0,nx-1]:
            for i in range(nz+1):
                for j in range(ny+1):
                    c_idx = 6 * ( nx*ny*min(i,nz-1) + nx*min(j,ny-1) + k )
                    c_idx += 0 if k==0 else 5
                    c_idx

                    vrt = [ k/(nx-1), j/ny, i/nz ]

                    m = np.matrix(vrt+z+z+[1,0,0])

                    L[c*9+sr:c*9+sr+1,c_idx*12:(c_idx+1)*12] = m
                    sr += 1

    return L

#%%
def set_continuity_constraint(p1,p2,p3,c_idx_1,c_idx_2,L,c):
    z = [0,0,0]

    m = np.concatenate( (np.matrix([ p1 + z + z, z + p1 + z, z + z + p1]), np.identity(3)), axis=1)
    L[c*9:c*9+3,c_idx_1*12:(c_idx_1+1)*12] = m
    L[c*9:c*9+3,c_idx_2*12:(c_idx_2+1)*12] = -m

    m = np.concatenate( (np.matrix([ p2 + z + z, z + p2 + z, z + z + p2 ]), np.identity(3)), axis=1)
    L[c*9+3:c*9+6,c_idx_1*12:(c_idx_1+1)*12] = m
    L[c*9+3:c*9+6,c_idx_2*12:(c_idx_2+1)*12] = -m

    m = np.concatenate( (np.matrix([ p3 + z + z, z + p3 + z, z + z + p3]), np.identity(3)), axis=1)
    L[c*9+6:(c+1)*9,c_idx_1*12:(c_idx_1+1)*12] = m
    L[c*9+6:(c+1)*9,c_idx_2*12:(c_idx_2+1)*12] = -m

    c += 1
    return L, c

#%%
if __name__ == '__main__':
    pass
