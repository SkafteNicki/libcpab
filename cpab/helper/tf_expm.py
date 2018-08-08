# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 12:35:48 2018

@author: nsde
"""
#%%
import tensorflow as tf

#%%
def _real_case2x2(a,b):
    """ """
    Ea = tf.expand_dims(tf.expand_dims(tf.exp(a), axis=1), axis=2)
    Eb = tf.expand_dims(tf.expand_dims(b*(tf.exp(a)-1)/a, axis=1), axis=2)
    return tf.concat([Ea, Eb], axis=2)
    
#%%
def _limit_case2x2(a,b):
    """ """
    Ea = tf.expand_dims(tf.expand_dims(tf.ones_like(a), axis=1), axis=2)
    Eb = tf.expand_dims(tf.expand_dims(b, axis=1), axis=2)
    return tf.concat([Ea, Eb], axis=2)

#%%
def tf_expm2x2(A):
    """ """
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
def _real_case3x3(a,b,c,d,e,f,x,y):
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
def _limit_case3x3(a,b,c,d,e,f,x,y):
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
def tf_expm3x3(A):
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
    with tf.name_scope('expm3x3'):
        # Initilial computations
        a,b,c = A[:,0,0], A[:,0,1], A[:,0,2]
        d,e,f = A[:,1,0], A[:,1,1], A[:,1,2]
        y = a**2 - 2*a*e + 4*b*d + e**2
        x = tf.sqrt(tf.abs(y))
        
        # Calculate all three cases and then choose bases on y
        real_res = _real_case3x3(a,b,c,d,e,f,x,y)
        complex_res = _complex_case3x3(a,b,c,d,e,f,x,y)
        limit_res = _limit_case3x3(a,b,c,d,e,f,x,y)
        expA = tf.where(y > 0, real_res, tf.where(y < 0, complex_res, limit_res))
        return expA
    
