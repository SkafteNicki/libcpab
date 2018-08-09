# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 09:21:24 2018

@author: nsde
"""
#%%
import tensorflow as tf

'''
# Optimize theta by aligning time series

# Placeholder for training set / Convert training set to tensor
X = tf.convert_to_tensor(X_train, dtype=tf.float32, name='X')

# Placeholders for indices
n_idx = tf.placeholder(tf.int32, shape=(), name='n')
m_idx = tf.placeholder(tf.int32, shape=(), name='m')

#tf.while_loop

# Fetch pair of time series from data set
x_ts = tf.gather(X, n_idx)
y_ts = tf.gather(X, m_idx)

# Two time series from data set
x_np = np.linspace(0,1,ts_length)
y_np = np.linspace(0,1,ts_length)

# Variables for two time series to be aligned
x = tf.Variable(x_np, trainable=False, dtype=tf.float32, name='x')
y = tf.Variable(y_np, trainable=False, dtype=tf.float32, name='y')

# Vector of parameters to be optimized
# Initialize as zero (identity) transformation
# Add small noise, optimization fails if theta is zero vector
theta = tf.Variable(tf.zeros([d, 1]) + 1e-10, trainable=True, name='theta')

# Convert B to tensorflow variable
B = tf.convert_to_tensor(B, dtype=tf.float32, name='B')

# Compute A
A = tf.matmul(B, theta)

# Transform time series
x_trans = transformation_v2(A, x, N_step, N_p)

# Interpolate values from 'x' time series
x_ts_interp = tf_linear_interpolation(x, x_trans, x_ts, ts_length)

# Update cost function value
loss = tf.reduce_mean(tf.pow(x_ts_interp-y_ts, 2), name='loss')

# Optimization using gradients
learning_rate = 0.01
opt = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

'''
def tf_cpab_transformer():
    pass

def tf_findcellidx():
    
    pass

#%%
def transformation_v2(A, U, N_step, N_p, t=1):
    delta_t = float(t) / N_step
    
    phi = U
    
    for j in range(N_step):
        
        # Find cell index
        idx = tf.floor(N_p * phi)
        idx = tf.clip_by_value(idx, clip_value_min=0, clip_value_max=N_p-1)
        idx = tf.cast(idx, tf.int32)
        
        # Fetch values from A (vector field)
        a = tf.reshape(tf.gather(A, 2*idx), [-1])
        b = tf.reshape(tf.gather(A, 2*idx+1), [-1])
        
        # Perform psi computation
        phi = tf.where(tf.equal(a, 0), psi_a_eq_zero(phi, a, b, delta_t), psi_a_noteq_zero(phi, a, b, delta_t))
        
    return phi

#%%
def psi_a_eq_zero(x, a, b, t):
    tb = tf.multiply(t,b)
    psi = tf.add(x, tb)
    return psi

#%%
def psi_a_noteq_zero(x, a, b, t):
    c1 = tf.exp(tf.multiply(t, a))
    c2 = tf.truediv(tf.multiply(b, tf.subtract(c1, 1)), a)
    psi = tf.add(tf.multiply(c1, x), c2)
    return psi