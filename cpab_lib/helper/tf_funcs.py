#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 09:46:23 2017

@author: nsde
"""

#%%
import tensorflow as tf

#%%
def tf_repeat_matrix(x_in, n_repeats):
    """ Concatenate n_repeats copyies of x_in by a new 0 axis """
    with tf.name_scope('repeat_matrix'):
        x_out = tf.reshape(x_in, (-1,))
        x_out = tf.tile(x_out, [n_repeats])
        x_out = tf.reshape(x_out, (n_repeats, tf.shape(x_in)[0], tf.shape(x_in)[1]))
        return x_out

#%%
def tf_repeat_matrix(x_in, n_repeats):
    """ Concatenate n_repeats copyies of x_in by a new 0 axis """
    with tf.name_scope('repeat_matrix'):
        x_out = tf.reshape(x_in, (-1,))
        x_out = tf.tile(x_out, [n_repeats])
        x_out = tf.reshape(x_out, (n_repeats, tf.shape(x_in)[0], tf.shape(x_in)[1]))
        return x_out

#%%
def tf_repeat(x, n_repeats):
    """
    Tensorflow implementation of np.repeat(x, n_repeats)
    """
    with tf.name_scope('repeat'):
        ones = tf.ones(shape=(n_repeats, ))
        rep = tf.transpose(tf.expand_dims(ones, 1), [1, 0])
        rep = tf.cast(rep, x.dtype)
        x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
        return tf.reshape(x, [-1])

#%%
def tf_log2(x):
    ''' Computes log-2 of a tensor '''
    with tf.name_scope('log2'):
        numerator = tf.log(x)
        denominator = tf.log(tf.constant(2, dtype=numerator.dtype))
        return numerator / denominator
    
#%%
def _tf_norm_l1_batch(A, batch_size):
    ''' Computes the l1-norm (frobenius norm) for a batch of matrices '''
    with tf.name_scope('l1_norm'):
        A_flat = tf.reshape(A, [batch_size, -1])
        return tf.sqrt(tf.reduce_sum(tf.pow(tf.abs(A_flat), 2.0), 1))

#%%
def _tf_pade7(A):
    ''' Pade approximation of order 9 '''
    with tf.name_scope('pade_9_approximation'):
        b = (17643225600., 8821612800., 2075673600., 302702400., 30270240.,
             2162160., 110880., 3960., 90., 1.)
        ident = tf.eye(tf.shape(A)[1], tf.shape(A)[2], [tf.shape(A)[0]], dtype=A.dtype)
        A2 = tf.matmul(A,A)
        A4 = tf.matmul(A2,A2)
        A6 = tf.matmul(A4,A2)
        A8 = tf.matmul(A6,A2)
        U = tf.matmul(A, b[9]*A8 + b[7]*A6 + b[5]*A4 + b[3]*A2 + b[1]*ident)
        V = b[8]*A8 + b[6]*A6 + b[4]*A4 + b[2]*A2 + b[0]*ident
        return U, V

#%%
def _tf_squaring(elems):
    ''' Function for squring a matrix a number of times. Works with tf.map_fn.
        Note that elems = (A, n_squarins), where A is a matrix and n_squarins
        is an integer
    '''
    with tf.name_scope('matrix_squaring'):
        A = elems[0]
        n_squarings = elems[1]
        i = tf.constant(0, tf.float32)
        def body(i, a): return i+1, tf.matmul(a,a)
        def cond(i, a): return tf.less(i, n_squarings)
        res = tf.while_loop(cond = cond, body = body, loop_vars = [i, A])
        return res[1]

#%%
def tf_expm3x3(A):
    ''' Specilized function for computing the matrix exponential of 3x3 square
        matrix. This function does not rely on tf.matrix_solve, which at the
        moment does not run on the gpu. Very similar to tf_expm.
        Parameters
            A: 3D tensor with shape [num_matrices, 3, 3]
        Output:
            expA: 3D tensor with shape [num_matrices, 3, 3] containing matrix exponentials
    '''
    with tf.name_scope('expm3x3'):
        A = tf.cast(A, tf.float32)
        batch_size = tf.shape(A)[0]

        A_L1 = _tf_norm_l1_batch(A, batch_size) # Computes the frobenius norm of the matrix
        maxnorm = tf.constant(5.371920351148152, dtype=A_L1.dtype)
        n_squarings = tf.maximum(0.0, tf.ceil(tf_log2(A_L1 / maxnorm)))
        A = A / tf.expand_dims(tf.expand_dims(tf.pow(2.0, n_squarings), 1), 2)
        
        U, V = _tf_pade7(A)
        
        P = U + V  # p_m(A) : numerator
        Q = -U + V # q_m(A) : denominator
    
        # Special case 3x3 matrix solver
        R = tf.matmul(tf_inv3x3_batch(Q), P)
        
        # Squaring step to undo scaling
        expA = tf.map_fn(fn = _tf_squaring, elems = (R, n_squarings), dtype = tf.float32)
        
        return expA        

#%%    
def tf_det3x3_batch(A):
    """ Tensorflow implementation of determinant for a batch of 3x3 matrices.
    Arguments
        A: 3D-`Tensor` [N,3,3]. Input square matrices.
    Output:
        D: `Vector` [D,]. Determintant of each input matrix.
    """
    with tf.name_scope('det3x3'):
        D = A[:, 0, 0]*A[:, 1, 1]*A[:, 2, 2] \
          + A[:, 0, 1]*A[:, 1, 2]*A[:, 2, 0] \
          + A[:, 0, 2]*A[:, 1, 0]*A[:, 2, 1] \
          - A[:, 0, 2]*A[:, 1, 1]*A[:, 2, 0] \
          - A[:, 0, 1]*A[:, 1, 0]*A[:, 2, 2] \
          - A[:, 0, 0]*A[:, 1, 2]*A[:, 2, 1]
        return D
    
#%%
def tf_inv3x3_batch(X):
    """ Tensorflow implementation of matrix inverse for 3x3 matrices.
    Arguments
        X: 3D-`Tensor` [N,3,3]. Input square matrices.
    Output:
        Y: 3D-`Tensor` [N,3,3]. Matrix inverse of each input matrix.
    """
    with tf.name_scope('inv3x3'):
        detX = tf_det3x3_batch(X) # Nx1
        
        Y00 =  (X[:, 1, 1]*X[:, 2, 2] - X[:, 1, 2]*X[:, 2, 1]) # Nx1
        Y10 = -(X[:, 1, 0]*X[:, 2, 2] - X[:, 1, 2]*X[:, 2, 0]) # Nx1
        Y20 =  (X[:, 1, 0]*X[:, 2, 1] - X[:, 1, 1]*X[:, 2, 0]) # Nx1
        Y01 = -(X[:, 0, 1]*X[:, 2, 2] - X[:, 0, 2]*X[:, 2, 1]) # Nx1
        Y11 =  (X[:, 0, 0]*X[:, 2, 2] - X[:, 0, 2]*X[:, 2, 0]) # Nx1
        Y21 = -(X[:, 0, 0]*X[:, 2, 1] - X[:, 0, 1]*X[:, 2, 0]) # Nx1
        Y02 =  (X[:, 0, 1]*X[:, 1, 2] - X[:, 0, 2]*X[:, 1, 1]) # Nx1
        Y12 = -(X[:, 0, 0]*X[:, 1, 2] - X[:, 0, 2]*X[:, 1, 0]) # Nx1
        Y22 =  (X[:, 0, 0]*X[:, 1, 1] - X[:, 0, 1]*X[:, 1, 0]) # Nx1
        
        Y0 = tf.stack([Y00, Y01, Y02],1) # Nx3
        Y1 = tf.stack([Y10, Y11, Y12],1) # Nx3
        Y2 = tf.stack([Y20, Y21, Y22],1) # Nx3
    
        Y = tf.stack([Y0, Y1, Y2], 1) # Nx3x3
        Y = Y / tf.expand_dims(tf.expand_dims(detX,1),2)
        
        return Y

#%%
def _complex_case(a,b,c,d,e,f,x,y):
    ''' Complex solution for the expm function for 3x3 matrices '''
    nom = 1/((-a**2-2*a*e-e**2-x**2)*x)
    sinx = tf.sin(0.5*x)
    cosx = tf.cos(0.5*x)
    expea = tf.exp(0.5*(a+e))
    
    Ea = -(4*((a-e)*sinx+cosx*x))*(a*e-b*d)*expea * nom
    Eb = -8*b*(a*e-b*d)*sinx*expea * nom
    Ec = -4*(((-c*e**2+(a*c+b*f)*e+b*(a*f-2*c*d))*sinx-cosx*x*(b*f-c*e))*expea+(b*f-c*e)*x)*nom
    Ed = -8*d*(a*e-b*d)*sinx*expea * nom
    Ee = 4*((a-e)*sinx-cosx*x)*(a*e-b*d)*expea * nom
    Ef = 4*(((a**2*f+(-c*d-e*f)*a+d*(2*b*f-c*e))*sinx-x*cosx*(a*f-c*d))*expea+x*(a*f-c*d))*nom
    
    E = tf.stack([tf.stack([Ea, Eb, Ec], axis=1),
                  tf.stack([Ed, Ee, Ef], axis=1)], axis=1)
    return E
    
#%%
def _real_case(a,b,c,d,e,f,x,y):
    ''' Real solution for the expm function for 3x3 matrices '''
    eap = a+e+x
    eam = a+e-x
    nom = 1/(x*eam*eap)
    expeap = tf.exp(0.5*eap)
    expeam = tf.exp(0.5*eam)
      
    Ea = -2*(a*e-b*d)*((a-e-x)*expeam-(a-e+x)*expeap)*nom
    Eb = -4*b*(expeam-expeap)*(a*e-b*d)*nom
    Ec = (((4*c*d-2*f*eap)*b-2*c*e*(a-e-x))*expeam +
         ((-4*c*d+2*f*eam)*b+2*c*e*(a-e+x))*expeap+4*(b*f-c*e)*x)*nom
    Ed = -4*d*(expeam-expeap)*(a*e-b*d)*nom
    Ee = 2*(a*e-b*d)*((a-e+x)*expeam-(a-e-x)*expeap)*nom
    Ef = ((2*a**2*f+(-2*c*d-2*f*(e-x))*a+4*d*(b*f-(1/2)*c*(e+x)))*expeam + 
         (-2*a**2*f+(2*c*d+2*f*(e+x))*a-4*(b*f-(1/2)*c*(e-x))*d)*expeap - 
         (4*(a*f-c*d))*x ) * nom

    E = tf.stack([tf.stack([Ea, Eb, Ec], axis=1),
                  tf.stack([Ed, Ee, Ef], axis=1)], axis=1)
    return E

#%%
def _limit_case(a,b,c,d,e,f,x,y):
    """ Limit solution for the expm function for 3x3 matrices """
    ea2 = (a + e)**2
    expea = tf.exp(0.5*(a + e))
    Ea = 2*(a - e + 2)*(a * e - b * d) * expea / ea2
    Eb = 4 * b * (a*e - b*d) * expea /ea2
    Ec = ((-2*c*e**2+(2*b*f+2*c*(a+2))*e+2*b*(-2*c*d+f*(a-2)))*expea+4*b*f-4*c*e)/ea2
    Ed = 4*d*(a*e - b*d) * expea / ea2
    Ee = -(2*(a-e-2))*(a*e-b*d) * expea /ea2
    Ef = ((-2*a**2*f+(2*c*d+2*f*(e+2))*a-4*d*(b*f-0.5*c*(e-2)))*expea-4*a*f+4*c*d)/ea2    

    E = tf.stack([tf.stack([Ea, Eb, Ec], axis=1),
                  tf.stack([Ed, Ee, Ef], axis=1)], axis=1)
    return E    

#%%
def tf_expm3x3_analytic(A):
    """ Tensorflow implementation for finding the matrix exponential of a batch
        of 3x3 matrices that have special form (last row is zero).
    
    Arguments:
        A: 3D-`Tensor` [N,2,3]. Batch of input matrices. It is assumed
            that the third row of each matrix is zero, and it should therefore
            not supplied. 
        
    Output:
        expA: 3D-`Tensor` [N,2,3]. Matrix exponential for each matrix
            in input tensor A. The last row is not returned but is [0,0,1] for 
            each matrix.
    """
    with tf.name_scope('expm3x3_analytic'):
        # Initilial computations
        a,b,c = A[:,0,0], A[:,0,1], A[:,0,2]
        d,e,f = A[:,1,0], A[:,1,1], A[:,1,2]
        y = a**2 - 2*a*e + 4*b*d + e**2
        x = tf.sqrt(tf.abs(y))
        
        # Calculate all three cases and then choose bases on y
        real_res = _real_case(a,b,c,d,e,f,x,y)
        complex_res = _complex_case(a,b,c,d,e,f,x,y)
        limit_res = _limit_case(a,b,c,d,e,f,x,y)
        expA = tf.where(y > 0, real_res, tf.where(y < 0, complex_res, limit_res))
        return expA