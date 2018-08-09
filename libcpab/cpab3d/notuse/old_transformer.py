#%% Packages
import tensorflow as tf
from tensorflow.python.framework import function
from scipy.linalg import expm as scipy_expm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
#from Transform import TransformNS
import pickle
import os.path
import time
from ..helper.utility import load_basis, get_dir
from sys import platform as _platform

#%% Load dynamic module
def load_dynamic_modules():
    dir_path = get_dir(__file__)
    transformer_module = tf.load_op_library(dir_path + '/./CPAB_ops.so')
    transformer_op = transformer_module.calc_trans
    grad_op = transformer_module.calc_grad
    
    return transformer_op, grad_op

if _platform == "linux" or _platform == "linux2" or _platform == "darwin":    
    transformer_op, grad_op = load_dynamic_modules()


#%%
def _tf_log2(x):
    ''' Computes log-2 of a tensor '''
    with tf.variable_scope('log2'):
        numerator = tf.log(x)
        denominator = tf.log(tf.constant(2, dtype=numerator.dtype))
        return numerator / denominator

#%%
def _tf_pade13(A):
    ''' Pade approximation of order 13 '''
    with tf.variable_scope('pade_13_approximation'):
        b = tf.constant([64764752532480000.,
                         32382376266240000.,
                         7771770303897600.,
                         1187353796428800.,
                         129060195264000.,
                         10559470521600.,
                         670442572800.,
                         33522128640.,
                         1323241920.,
                         40840800.,
                         960960.,
                         16380.,
                         182.,
                         1.], dtype=A.dtype)
        ident = tf.eye(tf.shape(A)[1], dtype=A.dtype)
        A2 = tf.matmul(A,A)
        A4 = tf.matmul(A2,A2)
        A6 = tf.matmul(A4,A2)
        U = tf.matmul(A, tf.matmul(A6, b[13]*A6 + b[11]*A4 + b[9]*A2) + b[7]*A6 + b[5]*A4 + b[3]*A2 + b[1]*ident)
        V = tf.matmul(A6, b[12]*A6 + b[10]*A4 + b[8]*A2) + b[6]*A6 + b[4]*A4 + b[2]*A2 + b[0]*ident
        return U, V

#%%
def _tf_squaring(elems):
    ''' Function for squring a matrix a number of times. Works with tf.map_fn.
        Note that elems = (A, n_squarins), where A is a matrix and n_squarins
        is an integer'''
    with tf.variable_scope('matrix_squaring'):
        A = elems[0]
        n_squarings = elems[1]
        i = tf.constant(0.0, tf.float64)
        def body(i, a): return i+1, tf.matmul(a,a)
        def cond(i, a): return tf.less(i, n_squarings)
        res = tf.while_loop(cond = cond, body = body, loop_vars = [i, A])
        return res[1]

#%%
def tf_expm(A):
    ''' Computes the matrix exponential of a  batch of square matrices using
        13-Order Pade approximation.
    Parameters:
        A: 3D-Tensor [N_batch, n, n] with square matrices
    Output
        expA: 3D-Tensor [N_batch, n, n] with matrix exponentials of A
    '''
    with tf.variable_scope('expm'):
        # Cast A to float32 datatype
        A = tf.cast(A, tf.float64)

        A_fro = tf.norm(A, axis=(1,2)) # Computes the frobenius norm of the matrix

        # To make the computations more stable, downscale the matrix at fist, do
        # the calculations and then upscale the matrix exponential
        maxnorm = tf.constant(5.371920351148152, dtype=tf.float64)
        zero = tf.constant(0.0, dtype=tf.float64)
        two = tf.constant(2.0, dtype=tf.float64)
        n_squarings = tf.maximum(zero, tf.ceil(_tf_log2(A_fro / maxnorm)))
        Ascaled = A / tf.expand_dims(tf.expand_dims(tf.pow(two, n_squarings), 1), 2)
        U,V = _tf_pade13(Ascaled)

        P = U + V  # p_m(A) : numerator
        Q = -U + V # q_m(A) : denominator
        R = tf.matrix_solve(Q,P)

        # squaring step to undo scaling (calls comp as long i < n_squarings)
        elems = (R, n_squarings)
        expA = tf.map_fn(_tf_squaring, elems, dtype=(tf.float64))
        return expA

#%%
def _calc_trans(points, theta):
    with tf.variable_scope('calc_trans'):
        # Make sure that both inputs are in float32 format
        points = tf.cast(points, tf.float32) # format [3, nb_points]
        theta  = tf.cast(theta, tf.float32) # format [1, dim]

        tesselation = pickle.load( open( "tesselation.pkl", "rb" ) )

        nx = tf.cast( tesselation["nx"], dtype=tf.int32 )
        ny = tf.cast( tesselation["ny"], dtype=tf.int32 )
        nz = tf.cast( tesselation["nz"], dtype=tf.int32 )
        Z  = tf.convert_to_tensor( tesselation["Basis"], dtype=tf.float32 )

        N = 6 * nx * ny * nz

        theta = tf.transpose(theta)
        Ab = tf.reshape( tf.matmul(Z,theta), [N,4,3] )
        A = Ab[:,0:3,0:3]
        b = tf.reshape(Ab[:,3,:], [N,3,1])

        C = tf.concat( [tf.concat([A,b], axis=2), tf.zeros([N,1,4])], axis=1 )

        nStepSolver = tf.cast(50, dtype = tf.int32) # Change in CPAB_ops.cc if this is changed
        dT = 1.0 / tf.cast(nStepSolver , tf.float32)

        MatExp = tf_expm( dT*C )
        Mx = tf.cast( tf.reshape(MatExp[:,0:3,:], [1,N,3,4]), dtype=tf.float32 )

        with tf.variable_scope('calc_trans_op'):
            newpoints = transformer_op(points, Mx, nStepSolver, nx, ny, nz)
        return newpoints

#%%
def _calc_grad_numeric(op, grad): #grad: n_theta x 2 x nP
    points = op.inputs[0] # 3 x n
    theta = op.inputs[1] # n_theta x d

    # Finite difference permutation size
    h = tf.cast(0.0001, tf.float32)

    # Base function evaluation
    f0 = _calc_trans(points, theta) # n_theta x 2 x nP

    gradient = [ ]
    for i in range(theta.get_shape()[1].value):
        # Add small permutation to i element in theta
        temp = tf.concat([theta[:,:i], tf.expand_dims(theta[:,i]+h,1), theta[:,(i+1):]], 1)

        # Calculate new function value
        f1 = _calc_trans(points, temp) # n_theta x 3 x nP

        # Finite difference
        diff = (f1 - f0) / h # n_theta x 3 x nP

        if i != 0:
            # Gradient
            gradient = tf.concat([gradient, tf.expand_dims(tf.reduce_sum(grad * diff, axis=[1,2]), 1)], 1)
        else:
            gradient = tf.expand_dims(tf.reduce_sum(grad * diff, axis=[1,2]), 1)

    return [None, gradient]

#%%
@function.Defun(tf.float32, tf.float32, func_name = 'tf_CPAB_transformer', python_grad_func = _calc_grad_numeric)
def tf_cpab_transformer(points, theta):
	return _calc_trans(points, theta)
