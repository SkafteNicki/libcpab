# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 14:23:25 2018

@author: nsde
"""

#%%
import numpy as np

#%%
class tesselation1D:
    def __init__(self, nc, domain_min=0, domain_max=1,
                 zero_boundary = True, volume_perservation=False):
        # Save parameters
        self.nc = nc
        self.domain_min = domain_min
        self.domain_max = domain_max
        self.zero_boundary = zero_boundary
        self.volume_perservation = volume_perservation
    
        # Get vertices
        self.vertices = self.find_verts()
        
        if not zero_boundary: # find auxility vertices
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
    
    def find_verts(self):
        pass
    
    def find_verts_outside(self):
        pass
    
    def create_continuity_constrains(self):
        pass
        
    def create_zero_boundary_constrains(self):
        pass
        
    def create_zero_trace_constrains(self):
        pass
    
#%%
class tesselation2D:
    def __init__(self, nc, domain_min=0, domain_max=1,
                 zero_boundary = True, volume_perservation=False):
        # Save parameters
        self.nc = nc
        self.domain_min = domain_min
        self.domain_max = domain_max
        self.zero_boundary = zero_boundary
        self.volume_perservation = volume_perservation
    
    def get_constrain_matrix(self):
        pass
    
    def find_verts(self):
        pass
    
    def find_verts_outside(self):
        pass
    
    def create_continuity_constrains(self):
        pass
        
    def create_zero_boundary_constrains(self):
        pass
        
    def create_zero_trace_constrains(self):
        pass
    
#%%
class tesselation3D:
    def __init__(self, nc, domain_min=0, domain_max=1,
                 zero_boundary = True, volume_perservation=False):
        # Save parameters
        self.nc = nc
        self.domain_min = domain_min
        self.domain_max = domain_max
        self.zero_boundary = zero_boundary
        self.volume_perservation = volume_perservation
    
    def get_constrain_matrix(self):
        pass
    
    def find_verts(self):
        pass
    
    def find_verts_outside(self):
        pass
    
    def create_continuity_constrains(self):
        pass
        
    def create_zero_boundary_constrains(self):
        pass
        
    def create_zero_trace_constrains(self):
        pass
    