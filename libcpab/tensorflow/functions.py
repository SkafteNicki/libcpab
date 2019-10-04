# -*- coding: utf-8 -*- 
"""
Created on Fri Nov 16 16:01:52 2018

@author: nsde
"""

#%%
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python.framework.ops import Tensor
from tensorflow.python.ops.resource_variable_ops import ResourceVariable
from .interpolation import interpolate
from .transformer import CPAB_transformer as transformer
from .findcellidx import findcellidx

#%%
def assert_version():
    numbers = tf.__version__.split('.')
    version = float(numbers[0] + '.' + numbers[1])
    assert version >= 2.0, \
        ''' You are using a older installation of pytorch, please install 2.0.0
            or newer '''

#%%
def to(x, dtype=tf.float32, device=None):
    with tf.device(device):
        x = tf.identity(tf.cast(x, dtype=dtype))
        return x

#%%
def tonumpy(x):
    return x.cpu().numpy()

#%%
def check_device(x, device_name):
    if x.device=='': # if x is placeholder, accept
        return True
    else: # else check if we match
        return ('GPU' in x.device) == (device_name=="gpu")

#%%
def backend_type():
    return (Tensor, ResourceVariable)

#%%
def pdist(mat):
    norm = tf.reduce_sum(mat * mat, 1)
    norm = tf.reshape(norm, (-1, 1))
    D = norm - 2*tf.matmul(mat, tf.transpose(mat)) + tf.transpose(norm)
    return D

#%%
def norm(x):
    return tf.norm(x)

#%%
def matmul(x,y):
    return tf.matmul(x,y)

#%%
def transpose(x):
    return tf.transpose(x)

#%%
def exp(x):
    return tf.exp(x)

#%%
def zeros(*s):
    return tf.zeros(*s)
    
#%%
def ones(*s):
    return tf.ones(*s)

#%%
def arange(x):
    return tf.range(x)
    
#%%
def repeat(x, reps):
    return tf.tile([x], [reps])

#%%
def batch_repeat(x, reps):
    return x.repeat(reps, *(x.dim()*[1]))

#%%
def maximum(x):
    return tf.reduce_max(x)

#%%
def sample_transformation(d, n_sample=1, mean=None, cov=None, device='cpu'):
    with tf.device(device):
        mean = tf.zeros((d,), dtype=tf.float32) if mean is None else mean
        cov = tf.eye(d, dtype=tf.float32) if cov is None else cov
        distribution = tfp.distributions.MultivariateNormalFullCovariance(mean, cov)
        return distribution.sample(n_sample)
    
#%%
def identity(d, n_sample=1, epsilon=0, device='cpu'):
    assert epsilon>=0, "epsilon need to be larger than 0"
    with tf.device(device):
        return tf.zeros((n_sample, d), dtype=tf.float32) + epsilon

#%%
def uniform_meshgrid(ndim, domain_min, domain_max, n_points, device='cpu'):
    with tf.device(device):
        lin = [tf.linspace(tf.cast(domain_min[i], tf.float32), 
               tf.cast(domain_max[i], tf.float32), n_points[i]) for i in range(ndim)]
        mesh = tf.meshgrid(*lin[::-1])
        grid = tf.concat([tf.reshape(array, (1, -1)) for array in mesh[::-1]], axis=0)
        return grid

#%%
def calc_vectorfield(grid, theta, params):
    # Calculate velocity fields
    B = to(params.basis, dtype=theta.dtype, device=theta.device)
    Avees = tf.tensordot(B, tf.reshape(theta, (-1,)), 1)
    As = tf.reshape(Avees, (params.nC, *params.Ashape))
    
    # Find cell index
    idx = findcellidx(params.ndim, grid, params.nc)
    
    # Do indexing
    Aidx = tf.gather(As, idx)
    
    # Convert to homogeneous coordinates
    with tf.device(grid.device):
        onerow = tf.ones((1, grid.shape[1]))
    grid = tf.concat((grid, onerow), axis=0)
    grid = tf.transpose(grid[None], perm=[2,1,0])
    
    # Do matrix multiplication
    v = tf.matmul(Aidx, grid)
    return tf.transpose(tf.squeeze(v))
