# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 16:13:00 2018

@author: nsde
"""

#%%
from tqdm import tqdm

#%%
class DataAligner:
    def __init__(self, cpab_class):
        ''' T is an instance of the core cpab class '''
        self.T = cpab_class
        
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
    def alignment_by_gradient(self, x1, x2, maxiter=100, lr=1e-4):
        ''' Gradient based minimization '''
        assert self.T.backend_name != 'numpy', \
            ''' Cannot do gradient decent when using the numpy backend '''
        self.T._check_type(x1)
        self.T._check_type(x2)
        
        # TODO: write this as general when tensorflow backend is done
        import torch
        theta = torch.autograd.Variable(self.T.identity(1, epsilon=1e-2))
        theta.requires_grad = True
        optimizer = torch.optim.Adam([theta], lr=lr)
        
        progress_bar = tqdm(desc='Gradient decent optimizer', 
                            total=maxiter, unit='iterations')
        for i in range(maxiter):
            optimizer.zero_grad()
            x1_trans = self.T.transform_data(x1, theta, outsize=x1.shape[2:])
            loss = self.backend.norm(x1_trans - x2)
            loss.backward()
            optimizer.step()
            progress_bar.update()
            progress_bar.set_postfix({'loss': loss.item()})

        return theta, x1_trans.detach()