# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 17:06:05 2019

@author: nsde
"""

#%%
import tensorflow as tf

#%%
def expm(A):
    """ """
    n_A = A.shape[0]
    A_fro = tf.sqrt(tf.reduce_sum(tf.pow(tf.abs(A), 2.0), axis=[1,2], keepdim=True))
    
    # Scaling step
    maxnorm = tf.cast([5.371920351148152], dtype=A.dtype, device=A.device)
    zero = tf.cast([0.0], dtype=A.dtype, device=A.device)
    n_squarings = tf.maximum(zero, tf.ceil(log2(A_fro / maxnorm)))
    Ascaled = A / 2.0**n_squarings
    n_squarings = tf.cast(tf.reshape(n_squarings, (-1, )), tf.int64)
    
    # Pade 13 approximation
    U, V = pade13(Ascaled)
    P = U + V
    Q = -U + V
    R, _ = tf.matrix_solve(Q, P)
    
    # Unsquaring step
    n = tf.reduce_max(n_squarings)
    res = [R]
    for i in range(n):
        res.append(tf.matmul(res[-1], res[-1]))
    R = tf.stack(res)
    expmA = R[n_squarings, tf.range(n_A)]
    return expmA

#%%
def log2(x):
    return tf.log(x) / tf.log(tf.cast([2.0], dtype=x.dtype, device=x.device))

#%%
def pade13(A):
    with tf.device(A):
        b = tf.cast([64764752532480000., 32382376266240000., 7771770303897600.,
                     1187353796428800., 129060195264000., 10559470521600.,
                     670442572800., 33522128640., 1323241920., 40840800.,
                     960960., 16380., 182., 1.], dtype=A.dtype)
        ident = tf.eye(A.shape[1], dtype=A.dtype)
    A2 = tf.matmul(A,A)
    A4 = tf.matmul(A2,A2)
    A6 = tf.matmul(A4,A2)
    U = tf.matmul(A, tf.matmul(A6, b[13]*A6 + b[11]*A4 + b[9]*A2) + b[7]*A6 + b[5]*A4 + b[3]*A2 + b[1]*ident)
    V = tf.matmul(A6, b[12]*A6 + b[10]*A4 + b[8]*A2) + b[6]*A6 + b[4]*A4 + b[2]*A2 + b[0]*ident
    return U, V
