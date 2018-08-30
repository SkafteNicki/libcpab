# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 12:35:48 2018

@author: nsde
"""
#%%
import tensorflow as tf

#%%
def _real_case2x2(a,b):
    """ Real solution for expm function for 2x2 special form matrices"""
    Ea = tf.expand_dims(tf.expand_dims(tf.exp(a), axis=1), axis=2)
    Eb = tf.expand_dims(tf.expand_dims(b*(tf.exp(a)-1)/a, axis=1), axis=2)
    return tf.concat([Ea, Eb], axis=2)
    
#%%
def _limit_case2x2(a,b):
    """ Limit solution for expm function for 2x2 special form matrices"""
    Ea = tf.expand_dims(tf.expand_dims(tf.ones_like(a), axis=1), axis=2)
    Eb = tf.expand_dims(tf.expand_dims(b, axis=1), axis=2)
    return tf.concat([Ea, Eb], axis=2)

#%%
def tf_expm2x2(A):
    """ Tensorflow implementation for finding the matrix exponential of a batch
        of 2x2 matrices that have special form (last row is zero).
    
    Arguments:
        A: 3D-`Tensor` [N,2,2]. Batch of input matrices. It is assumed
            that the second row of each matrix is zero.
        
    Output:
        expA: 3D-`Tensor` [N,2,2]. Matrix exponential for each matrix in input tensor A.
    """
    with tf.name_scope('expm2x2'):
        n_batch = tf.shape(A)[0]
        a, b = A[:,0,0], A[:,0,1]
        
        real_res = _real_case2x2(a,b)
        limit_res = _limit_case2x2(a,b)
        E = tf.where(tf.equal(a,0), limit_res, real_res)
        
        zero = tf.zeros((n_batch,1,1))
        ones = tf.ones((n_batch,1,1))
        expA = tf.concat([E, tf.concat([zero, ones], axis=2)], axis=1)        
        return expA
    
#%%
def _complex_case3x3(a,b,c,d,e,f,x,y):
    ''' Complex solution for the expm function for special form 3x3 matrices '''
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
def _real_case3x3(a,b,c,d,e,f,x,y):
    ''' Real solution for the expm function for special form 3x3 matrices '''
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
def _limit_case3x3(a,b,c,d,e,f,x,y):
    """ Limit solution for the expm function for special form 3x3 matrices """
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
def _zero_case3x3(a,b,c,d,e,f,x,y):
    ones = tf.ones_like(a)
    zeros = tf.zeros_like(a)*a
    Ea, Ee = ones, ones
    Eb, Ec, Ed, Ef = zeros, zeros, zeros, zeros
    E = tf.stack([tf.stack([Ea, Eb, Ec], axis=1),
                  tf.stack([Ed, Ee, Ef], axis=1)], axis=1)
    return E

#%%
def tf_expm3x3(A):
    """ Tensorflow implementation for finding the matrix exponential of a batch
        of 3x3 matrices that have special form (last row is zero).
    
    Arguments:
        A: 3D-`Tensor` [N,3,3]. Batch of input matrices. It is assumed
            that the second row of each matrix is zero.
        
    Output:
        expA: 3D-`Tensor` [N,3,3]. Matrix exponential for each matrix in input tensor A.
    """
    with tf.name_scope('expm3x3'):
        n_batch = tf.shape(A)[0]
        
        # Initilial computations
        a,b,c = A[:,0,0], A[:,0,1], A[:,0,2]
        d,e,f = A[:,1,0], A[:,1,1], A[:,1,2]
        y = a**2 - 2*a*e + 4*b*d + e**2
        x = tf.sqrt(tf.abs(y))
        
        # Calculate all three cases and then choose bases on y
        real_res = _real_case3x3(a,b,c,d,e,f,x,y)
        complex_res = _complex_case3x3(a,b,c,d,e,f,x,y)
        limit_res = _limit_case3x3(a,b,c,d,e,f,x,y)
        zero_res = _zero_case3x3(a,b,c,d,e,f,x,y)
        E = tf.where(y > 0, real_res, tf.where(y < 0, complex_res, 
                     tf.where(tf.logical_and(tf.equal(a,0), tf.equal(e,0)),
                     zero_res, limit_res)))
        
        zero = tf.zeros(shape=(n_batch, 1, 2), dtype=E.dtype)
        ones = tf.ones(shape=(n_batch, 1, 1), dtype=E.dtype)

        expA = tf.concat([E, tf.concat([zero, ones], axis=2)], axis=1)
        
        return expA

#%%
def _tf_log2(x):
    """ Computes log-2 of a tensor """
    with tf.variable_scope('log2'):
        numerator = tf.log(x)
        denominator = tf.log(tf.constant(2, dtype=numerator.dtype))
        return numerator / denominator

#%%
def _tf_pade13(A):
    """ Pade approximation of order 13 """
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
    """ Function for squring a matrix a number of times. Works with tf.map_fn.
        Note that elems = (A, n_squarins), where A is a matrix and n_squarins
        is an integer"""
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
        13-Order Pade approximation. Works with generic type matrices.
    
    Arguments:
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
if __name__ == '__main__':
    pass