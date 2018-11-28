# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 16:13:00 2018

@author: nsde
"""

#%%
from tqdm import tqdm

#%%
class alignment:
    def __init__(self, T):
        ''' T is an instance of the core cpab class '''
        self.T = T
        
        if self.T.backend_name == 'numpy':
            from .numpy import functions as backend
        elif self.T.backend_name == 'tensorflow':
            from .tensorflow import functions as backend
        elif self.T.backend_name == 'pytorch':
            from .pytorch import functions as backend
        self.backend = backend
    
    #%%
    def alignment_by_sampling(self, x1, x2, maxiter=100):
        ''' MCMC sampling '''
        self.T._check_type(x1)
        self.T._check_type(x2)
        current_sample = self.T.identity(1)
        current_error = self.backend.norm(x1 - x2)
        accept_ratio = 0
        for i in tqdm(range(maxiter), desc='mcmc sampling'):
            # Sample and transform 
            theta = self.T.sample_transformation(1, mean=current_sample)
            x1_trans = self.T.transform_data(x1, theta)
            
            # Calculate new error
            new_error = self.backend.norm(x1_trans - x2)
            
            if new_error < current_error:
                current_sample = theta
                current_error = new_error
                accept_ratio += 1
        print('Acceptence ratio: ', accept_ratio / maxiter * 100, '%')
        return current_sample    
    
    #%%
    def alignment_by_gradient(self, x1, x2, maxiter=100):
        ''' Gradient based minimization '''
        assert self.T.backend_name != 'numpy', \
            ''' Cannot do gradient decent when using the numpy backend '''
        self.T._check_type(x1)
        self.T._check_type(x2)
        
#        theta = self.backend.Variable(self.T.identity())
#        for i in tqdm(range(maxiter), desc='graident decent'):
#            x1_trans = self.T.transform_data(x1, theta)
#            loss = self.backend.norm(x1_trans - x2)
#            self.backend.optimize(loss)
#        return theta