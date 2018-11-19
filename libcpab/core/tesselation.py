# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 14:23:25 2018

@author: nsde
"""

#%%
import numpy as np

#%%
class tesselation:
    """ Base tesselation class. This function is not meant to be called,
        but descripes the base structure that needs to be implemented in
        1D, 2D, and 3D.
        
    Args:
        nc: list with number of cells
        domain_min: value of the lower bound(s) of the domain
        domain_max: value of the upper bound(s) of the domain
        zero_boundary: bool, if true the velocity is zero on the boundary
        volume_perservation: bool, if true volume is perserved
        
    Methods that should not be implemented in subclasses:
        @get_constrain_matrix:
        @get_cell_centers:
            
    Methods that should be implemented in subclasses:
        @find_verts:
        @find_verts_outside:
        @create_continuity_constrains:
        @create_zero_boundary_constrains:
        @create_zero_trace_constrains:
    """
    def __init__(self, nc, domain_min=0, domain_max=1,
                 zero_boundary = True, volume_perservation=False):
        # Save parameters
        self.nc = nc
        self.nC = np.prod(nc)
        self.domain_min = domain_min
        self.domain_max = domain_max
        self.zero_boundary = zero_boundary
        self.volume_perservation = volume_perservation
    
        # Get vertices
        self.vertices = self.find_verts()
        self.inner_vertices = self.vertices
        
        # find auxility vertices, if transformation is valid outside
        if not zero_boundary: 
            temp = self.find_verts_outside()
            self.vertices = np.concatenate((self.vertices, temp), axis=0)
        
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
        return self.L
    
    def get_cell_centers(self):
        return np.mean(self.inner_vertices)
    
    def find_verts(self):
        raise NotImplemented
        
    def find_verts_outside(self):
        raise NotImplemented
        
    def create_continuity_constrains(self):
        raise NotImplemented
        
    def create_zero_boundary_constrains(self):
        raise NotImplemented
        
    def create_zero_trace_constrains(self):
        raise NotImplemented
        
#%%
class tesselation1D(tesselation):
    def __init__(self, nc, domain_min=0, domain_max=1,
                 zero_boundary = True, volume_perservation=False):
        super(tesselation1D, self).__init__(nc, domain_min, domain_max,
             zero_boundary, volume_perservation)
        self.n_params = 2
    
    def find_verts(self):
        pass
    
    def find_verts_outside(self):
        return np.empty((0, self.nC))
    
    def create_continuity_constrains(self):
        pass
        
    def create_zero_boundary_constrains(self):
        Ltemp = np.zeros((2,2*self.nC))
        Ltemp[0,:2] = [self.domain_min, 1]
        Ltemp[1,-2:] = [self.domain_max, 1]
        return Ltemp
        
    def create_zero_trace_constrains(self):
        Ltemp = np.zeros(shape=(self.nC, 2*self.nC))
        for c in range(self.nC):
            Ltemp[c,2*c] = 1
        return Ltemp
    
