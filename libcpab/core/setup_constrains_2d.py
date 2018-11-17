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
def get_constrain_matrix_2D(nc, domain_min, domain_max,
                            valid_outside, zero_boundary, volume_perservation):
    ncx = nc[0]
    ncy = nc[1]
    
    # Call tessalation and get vertices of cells
    cells_multiidx, cells_verts  = tessalation(ncx, ncy, domain_min, domain_max)
    nC = len(cells_verts)
        
    # Find shared vertices (edges) where a continuity constrain needs to hold
    shared_v, shared_v_idx = find_shared_verts(nC, cells_verts)
            
    # If the transformation should be valid outside of the image domain, 
    # calculate the auxiliary points and add them to the edges where a 
    # continuity constrain should be
    if valid_outside:
        shared_v_outside, shared_v_idx_outside = find_shared_verts_outside(nC, ncx, ncy, 
                                                                           cells_verts, cells_multiidx)
        if shared_v_outside.size != 0:
            shared_v = np.concatenate((shared_v, shared_v_outside))
            shared_v_idx = np.concatenate((shared_v_idx, shared_v_idx_outside))
            
    # Create L
    L = create_continuity_constrains(nC, shared_v_idx, shared_v)
            
    # Update L with extra constrains if needed
    if zero_boundary:
        Ltemp = create_zero_boundary_constrains(nC, domain_min, domain_max, cells_verts)
        L = np.vstack((L, Ltemp))

    if volume_perservation:
        Ltemp = create_zero_trace_constrains(nC)
        L = np.vstack((L, Ltemp))
            
    return L
            
#%%        
def tessalation(ncx, ncy, domain_min, domain_max):
    """ Finds the coordinates of all cell vertices """
    xmin, ymin = domain_min
    xmax, ymax = domain_max
    Vx = np.linspace(xmin, xmax, ncx+1)
    Vy = np.linspace(ymin, ymax, ncy+1)
    cells_x = [ ]
    cells_x_verts = [ ]
    for i in range(ncy):
        for j in range(ncx):
            ul = tuple([Vx[j],Vy[i],1])
            ur = tuple([Vx[j+1],Vy[i],1])
            ll = tuple([Vx[j],Vy[i+1],1])
            lr = tuple([Vx[j+1],Vy[i+1],1])
            
            center = [(Vx[j]+Vx[j+1])/2,(Vy[i]+Vy[i+1])/2,1]
            center = tuple(center)                 
            
            cells_x_verts.append((center,ul,ur))  # order matters!
            cells_x_verts.append((center,ur,lr))  # order matters!
            cells_x_verts.append((center,lr,ll))  # order matters!
            cells_x_verts.append((center,ll,ul))  # order matters!                

            cells_x.append((j,i,0))
            cells_x.append((j,i,1))
            cells_x.append((j,i,2))
            cells_x.append((j,i,3))
    
    return  cells_x, np.asarray(cells_x_verts)

#%%
def find_shared_verts(nC, cells_verts):
    """ Find all pair of cells that share a vertices that encode continuity
        constrains inside the domain
    """
    shared_v = [ ]
    shared_v_idx = [ ]
    for i in range(nC):
        for j in range(nC):
            vi = make_hashable(cells_verts[i])
            vj = make_hashable(cells_verts[j])
            shared_verts = set(vi).intersection(vj)
            if len(shared_verts) == 2 and (j,i) not in shared_v_idx:
                shared_v.append(list(shared_verts))
                shared_v_idx.append((i,j))
            
    return np.array(shared_v), shared_v_idx

#%%
def find_shared_verts_outside(nC, ncx, ncy, cells_verts, cells_multiidx):
    """ Find all pair of cells that share a vertices that encode continuity
        constrains outside the domain
    """
    shared_v = [ ]
    shared_v_idx = [ ]

    left =   np.zeros((nC, nC), np.bool)    
    right =  np.zeros((nC, nC), np.bool) 
    top =    np.zeros((nC, nC), np.bool) 
    bottom = np.zeros((nC, nC), np.bool) 

    for i in range(nC):
        for j in range(nC):
            
            vi = make_hashable(cells_verts[i])
            vj = make_hashable(cells_verts[j])
            shared_verts = set(vi).intersection(vj)
            
            mi = cells_multiidx[i]
            mj = cells_multiidx[j]
    
            # leftmost col, left triangle, adjacent rows
            if  mi[0]==mj[0]==0 and \
                mi[2]==mj[2]==3 and \
                np.abs(mi[1]-mj[1])==1: 
                    
                left[i,j]=True
            
            # rightmost col, right triangle, adjacent rows                 
            if  mi[0]==mj[0]==ncx-1 and \
                mi[2]==mj[2]==1 and \
                np.abs(mi[1]-mj[1])==1: 

                right[i,j]=True
            
            # uppermost row, upper triangle , adjacent cols                    
            if  mi[1]==mj[1]==0 and \
                mi[2]==mj[2]==0 and \
                np.abs(mi[0]-mj[0])==1:
                    
                top[i,j]=True
            
            # lowermost row, # lower triangle, # adjacent cols            
            if  mi[1]==mj[1]==ncy-1 and \
                mi[2]==mj[2]==2 and \
                np.abs(mi[0]-mj[0])==1:
                    
                bottom[i,j]=True
                            
            if  len(shared_verts) == 1 and \
                any([left[i,j],right[i,j],top[i,j],bottom[i,j]]) and \
                (j,i) not in shared_v_idx:
                    
                v_aux = list(shared_verts)[0] # v_aux is a tuple
                v_aux = list(v_aux) # Now v_aux is a list (i.e. mutable)
                if left[i,j] or right[i,j]:
                    v_aux[0]-=10 # Create a new vertex  with the same y
                elif top[i,j] or bottom[i,j]:
                    v_aux[1]-=10 # Create a new vertex  with the same x
                else:
                    raise ValueError("WTF?")                        
                shared_verts = [tuple(shared_verts)[0], tuple(v_aux)]
                shared_v.append(shared_verts)
                shared_v_idx.append((i,j))
    
    return np.array(shared_v), shared_v_idx
    
#%%     
def create_continuity_constrains(nC, shared_v_idx, shared_v):
    """ Based on the vertices found that are shared by cells, construct
        continuity constrains 
    """
    Ltemp = np.zeros(shape=(0,6*nC))
    count = 0
    for i,j in shared_v_idx:

        # Row 1 [x_a^T 0_{1x3} -x_a^T 0_{1x3}]
        row1 = np.zeros(shape=(6*nC))
        row1[(6*i):(6*(i+1))] = np.append(np.array(shared_v[count][0]), 
                                          np.zeros((1,3)))
        row1[(6*j):(6*(j+1))] = np.append(-np.array(shared_v[count][0]), 
                                          np.zeros((1,3)))
        
        # Row 2 [0_{1x3} x_a^T 0_{1x3} -x_a^T]
        row2 = np.zeros(shape=(6*nC))
        row2[(6*i):(6*(i+1))] = np.append(np.zeros((1,3)), 
                                          np.array(shared_v[count][0]))
        row2[(6*j):(6*(j+1))] = np.append(np.zeros((1,3)), 
                                          -np.array(shared_v[count][0]))
        
        # Row 3 [x_b^T 0_{1x3} -x_b^T 0_{1x3}]
        row3 = np.zeros(shape=(6*nC))
        row3[(6*i):(6*(i+1))] = np.append(np.array(shared_v[count][1]), 
                                          np.zeros((1,3)))
        row3[(6*j):(6*(j+1))] = np.append(-np.array(shared_v[count][1]), 
                                          np.zeros((1,3)))
        
        # Row 4 [0_{1x3} x_b^T 0_{1x3} -x_b^T]
        row4 = np.zeros(shape=(6*nC))
        row4[(6*i):(6*(i+1))] = np.append(np.zeros((1,3)), 
                                          np.array(shared_v[count][1]))
        row4[(6*j):(6*(j+1))] = np.append(np.zeros((1,3)), 
                                          -np.array(shared_v[count][1]))
                    
        Ltemp = np.vstack((Ltemp, row1, row2, row3, row4))
        
        count += 1
    
    return Ltemp

#%%    
def create_zero_boundary_constrains(nC, domain_min, domain_max, cells_verts):
    """ Construct zero boundary i.e. fixed boundary constrains. Note that 
        points on the upper and lower bound can still move to the left and 
        right and points on the left and right bound can still move up 
        and down. Thus, they are only partial zero. 
    """
    xmin, ymin = domain_min
    xmax, ymax = domain_max
    Ltemp = np.zeros(shape=(0,6*nC))
    for c in range(nC):
        for v in cells_verts[c]:
            if(v[0] == xmin or v[0] == xmax): 
                row = np.zeros(shape=(6*nC))
                row[(6*c):(6*(c+1))] = np.append(np.zeros((1,3)),v)
                Ltemp = np.vstack((Ltemp, row))
            if(v[1] == ymin or v[1] == ymax): 
                row = np.zeros(shape=(6*nC))
                row[(6*c):(6*(c+1))] = np.append(v,np.zeros((1,3)))
                Ltemp = np.vstack((Ltemp, row))
    return Ltemp

#%%
def create_zero_trace_constrains(nC):
    """ Construct zero trace (volume perservation) constrains """
    Ltemp = np.zeros(shape=(nC, 6*nC))
    for c in range(nC):
        Ltemp[c,(6*c):(6*(c+1))] = np.array([1,0,0,0,1,0])
    return Ltemp

#%%
if __name__ == '__main__':
    pass

