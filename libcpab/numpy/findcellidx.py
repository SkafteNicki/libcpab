# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 14:31:34 2018

@author: nsde
"""

#%%
import numpy as np

#%%
def mymin(a, b):
    return np.where(a < b, a, np.round(b))

#%%
def findcellidx(ndim, p, nc):
    if ndim==1:   return findcellidx1D(p, *nc)
    elif ndim==2: return findcellidx2D(p, *nc)
    elif ndim==3: return findcellidx3D(p, *nc)
    
#%%
def findcellidx1D(p, nx):
    p = p.copy()
    idx = np.floor(p[0] * nx)
    idx = np.maximum(0, np.minimum(idx, nx-1))
    idx = idx.flatten().astype(np.int32)
    return idx

#%%
def findcellidx2D(p, nx, ny):
    p = p.copy()
    # Conditions for points outside
    cond = np.logical_or(np.logical_or(np.logical_or(
            p[0,:] < 0.0, p[0,:] > 1.0), p[1,:] < 0.0), p[1,:] > 1.0)
    
    # Push the points inside boundary
    inc_x = 1.0 / nx
    inc_y = 1.0 / ny
    half = 0.5
    points_outside = p[:, cond]
    points_outside -= half
    abs_x = np.abs(points_outside[0])
    abs_y = np.abs(points_outside[1])
    push_x = (half*inc_x)*(abs_x < abs_y)
    push_y = (half*inc_y)*(abs_x > abs_y)
    cond_x = abs_x > half
    cond_y = abs_y > half
    points_outside[0,cond_x] = np.copysign(half - push_x[cond_x], points_outside[0, cond_x])
    points_outside[1,cond_y] = np.copysign(half - push_y[cond_y], points_outside[1, cond_y])
    points_outside += half
    p[:, cond] = points_outside
    
    # Find initial cell index
    p0 = np.minimum(1.0 - 1e-5, p[0,:])
    p1 = np.minimum(1.0 - 1e-5, p[1,:])
    p0nx = p0 * nx
    p1ny = p1 * ny
    ip0nx = np.floor(p0nx)
    ip1ny = np.floor(p1ny)
    idx = 4 * (ip0nx + ip1ny * nx)
    
    # Subtriangle
    x = p0nx - ip0nx
    y = p1ny - ip1ny
    idx[np.logical_and(x < y , 1-x <= y)] += 2
    idx[np.logical_and(x < y , 1-x > y)] += 3
    idx[np.logical_and(x >= y, 1-x < y)] += 1
    idx = idx.flatten().astype(np.int32)
    return idx
    
#%%
def findcellidx3D(p, nx, ny, nz):
    p = p.copy()
    # Conditions for points outside
    cond =  np.logical_or(np.logical_or(
            np.logical_or(p[0,:] < 0.0, p[0,:] > 1.0),
            np.logical_or(p[1,:] < 0.0, p[1,:] > 1.0)),
            np.logical_or(p[2,:] < 0.0, p[2,:] > 1.0))
        
    # Push the points inside boundary
    inc_x, inc_y, inc_z = 1.0 / nx, 1.0 / ny, 1.0 / nz
    half = 0.5
    points_outside = p[:, cond]
    points_outside -= half
    abs_x = np.abs(points_outside[0])
    abs_y = np.abs(points_outside[1])
    abs_z = np.abs(points_outside[2])
    push_x = (half * inc_x)*(np.logical_and(abs_x < abs_y, abs_x < abs_z))
    push_y = (half * inc_y)*(np.logical_and(abs_y < abs_x, abs_x < abs_z))
    push_z = (half * inc_z)*(np.logical_and(abs_z < abs_x, abs_x < abs_y))
    cond_x = abs_x > half
    cond_y = abs_y > half
    cond_z = abs_z > half
    points_outside[0, cond_x] = np.copysign(half - push_x[cond_x], points_outside[0, cond_x])
    points_outside[1, cond_y] = np.copysign(half - push_y[cond_y], points_outside[1, cond_y])
    points_outside[2, cond_z] = np.copysign(half - push_z[cond_z], points_outside[2, cond_z])
    points_outside += half
    p[:, cond] = points_outside

    # Find row, col, depth placement and cell placement
    inc_x, inc_y, inc_z = 1.0/nx, 1.0/ny, 1.0/nz
    p0 = np.minimum(nx * inc_x - 1e-8, np.maximum(0.0, p[0]))
    p1 = np.minimum(ny * inc_y - 1e-8, np.maximum(0.0, p[1]))
    p2 = np.minimum(nz * inc_z - 1e-8, np.maximum(0.0, p[2]))

    xmod = np.mod(p0, inc_x)
    ymod = np.mod(p1, inc_y)
    zmod = np.mod(p2, inc_z)
    
    i = mymin(nx - 1, ((p0 - xmod) / inc_x))
    hest = mymin(ny - 1, ((p1 - ymod) / inc_y))
    k = mymin(nz - 1, ((p2 - zmod) / inc_z))
    idx = 5 * (i + hest * nx + k * nx * ny)

    x = xmod / inc_x
    y = ymod / inc_y
    z = zmod / inc_z
    
    # Find subcell location
    cond = np.logical_or(np.logical_or(np.logical_or(
            ((k%2==0) & (i%2==0) & (hest%2==1)),
            ((k%2==0) & (i%2==1) & (hest%2==0))),
            ((k%2==1) & (i%2==0) & (hest%2==0))),
            ((k%2==1) & (i%2==1) & (hest%2==1)))

    tmp = x.copy()
    x[cond] = y[cond]
    y[cond] = 1-tmp[cond]
    
    cond1 = -x-y+z >= 0
    cond2 = x+y+z-2 >= 0
    cond3 = -x+y-z >= 0
    cond4 = x-y-z >= 0
    idx[cond1] += 1
    idx[cond2 & ~cond1] += 2
    idx[cond3 & ~cond1 & ~cond2] += 3
    idx[cond4 & ~cond1 & ~cond2 & ~cond3] += 4
    idx = idx.flatten().astype(np.int32)
    return idx