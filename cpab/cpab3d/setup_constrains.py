#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 09:59:09 2017

@author: nsde
"""

#%%
import numpy as np

#%%
def get_constrain_matrix_3D(nc, domain_min, domain_max,
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
