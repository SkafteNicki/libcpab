    #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 09:30:10 2017

@author: nsde
"""

#%% Packages
from sys import platform as _platform
import tensorflow as tf
from tensorflow.python.framework import function
from ddtn.helper.utility import load_basis, get_dir
from ddtn.helper.tf_funcs import tf_repeat_matrix, tf_expm3x3_analytic, tf_findcellidx

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
def _calc_trans(points, theta):
    """ Tensorflow wrapper function for calculating the CPAB transformations.
        The function extracts information for the current tesselation basis, and
        then call the dynamic library functions compiled from the cpp code which
        do the actual computations
        
    Arguments:
        points: `Matrix` [2, nb_points]. Grid of 2D points to transform
        theta: `Matrix` [n_theta, dim]. Batch of parametrization vectors. Each
            row specifies a specific transformation
        
    Output:
        newpoints: 3D-`Tensor` [n_theta, 2, nb_points]. Tensor of transformed points.
            The slice newpoints[i] corresponds to the input points transformed
            using the parametrization vector theta[i].
        o
    """
    with tf.name_scope('calc_trans'):
        # Make sure that both inputs are in float32 format
        points = tf.cast(points, tf.float32) # format [2, nb_points]
        theta = tf.cast(theta, tf.float32) # format [n_theta, dim]
        n_theta = tf.shape(theta)[0]
        
        # Load file with basis
        file = load_basis()
        
        # Tessalation information
        nC = tf.cast(file['nC'], tf.int32)
        ncx = tf.cast(file['ncx'], tf.int32)
        ncy = tf.cast(file['ncy'], tf.int32)
        inc_x = tf.cast(file['inc_x'], tf.float32)
        inc_y = tf.cast(file['inc_y'], tf.float32)
        
        # Steps sizes
        # NOTE: If this number is changed, then the allocation of the cell index
        # need to be changed in the CPAB_ops.cc file as well
        nStepSolver = tf.cast(50, dtype = tf.int32) 
        dT = 1.0 / tf.cast(nStepSolver , tf.float32)
        
        # Get cpab basis
        B = tf.cast(file['B'], tf.float32)

        # Repeat basis for batch multiplication
        B = tf_repeat_matrix(B, n_theta)

        # Calculate the row-flatted affine transformations Avees 
        Avees = tf.matmul(B, tf.expand_dims(theta, 2))
		
        # Reshape into (number of cells, 2, 3) tensor
        As = tf.reshape(Avees, shape = (n_theta * nC, 2, 3)) # format [n_theta * nC, 2, 3]
        
        # Multiply by the step size and do matrix exponential on each matrix
        Trels = tf_expm3x3_analytic(dT*As)
        Trels = tf.reshape(Trels, shape=(n_theta, nC, 2, 3))

        # Call the dynamic library
        with tf.name_scope('calc_trans_op'):
	        newpoints = transformer_op(points, Trels, nStepSolver, ncx, ncy, inc_x, inc_y)
        return newpoints

#%%
def _calc_grad(op, grad): #grad: n_theta x 2 x nP
    """ Tensorflow wrapper function for calculating the gradient of the CPAB 
        transformations. The function extracts information for the current 
        tesselation basis, and then call the dynamic library functions compiled 
        from the cpp code which do the actual computations
        
    Arguments:
        op: tensorflow operation class. The class holds information about the
            input and output of the original operation we are trying to 
            differentiate
        grad: 4D-`Tensor` [dim, n_theta, 2, nb_points]. Incoming gradient that
            is propegated onwards by this layer. It can be viewed as the gradient
            vector in each point, for all thetas and for all parameters of each
            theta.
        
    Output:
        gradient: list of 2 elements. Each element corresponds to the gradient
        w.r.t the input to the original function _calc_trans(points, theta). 
        Since we are only interested in the gradient w.r.t. theta, the first
        element is None. The second is a `Matrix` [dim, n_theta] i.e. the gradient
        of each element in all theta vectors.
        
    """
    with tf.name_scope('calc_grad'):
        # Grap input
        points = op.inputs[0] # 2 x nP
        theta = op.inputs[1] # n_theta x d
        n_theta = tf.shape(theta)[0]
    
        # Load file with basis
        file = load_basis()
        
        # Tessalation information
        nC = tf.cast(file['nC'], tf.int32)
        ncx = tf.cast(file['ncx'], tf.int32)
        ncy = tf.cast(file['ncy'], tf.int32)
        inc_x = tf.cast(file['inc_x'], tf.float32)
        inc_y = tf.cast(file['inc_y'], tf.float32)
        
        # Steps sizes
        nStepSolver = tf.cast(50, dtype = tf.int32)
    
        # Get cpab basis
        B = tf.cast(file['B'], tf.float32)
        Bs = tf.reshape(tf.transpose(B), (-1, nC, 2, 3))
        B = tf_repeat_matrix(B, n_theta)
        
        # Calculate the row-flatted affine transformations Avees 
        Avees = tf.matmul(B, tf.expand_dims(theta, 2))
        
        # Reshape into (ntheta, number of cells, 2, 3) tensor
        As = tf.reshape(Avees, shape = (n_theta, nC, 2, 3)) # n_theta x nC x 2 x 3
        
        # Call cuda code
        with tf.name_scope('calcT_batch_grad_operator'):
            gradient = grad_op(points, As, Bs, nStepSolver,
                               ncx, ncy, inc_x, inc_y) # gradient: d x n_theta x 2 x n
        
        # Reduce into: d x 1 vector
        gradient = tf.reduce_sum(grad * gradient, axis = [2,3])
        gradient = tf.transpose(gradient)
                                  
        return [None, gradient]
    
#%%
def _calc_grad_numeric(op, grad): #grad: n_theta x 2 x nP
    """ Similar to the _calc_grad(...) function above. Only difference is that 
        this function does a finite difference of the gradient by calling the
        _calc_trans(...) again and again for small permutations of the input
        theta vector, and then compare to the actual value of theta.
        Arguments and output is the same _calc_grad(...).
    """
    points = op.inputs[0] # 2 x n
    theta = op.inputs[1] # n_theta x d
    
    # Finite difference permutation size
    h = tf.cast(0.01, tf.float32)
    
    # Base function evaluation
    f0 = _calc_trans(points, theta) # n_theta x 2 x nP
    
    gradient = [ ]
    for i in range(theta.get_shape()[1].value):
        # Add small permutation to i element in theta
        temp = tf.concat([theta[:,:i], tf.expand_dims(theta[:,i]+h,1), theta[:,(i+1):]], 1)
        
        # Calculate new function value
        f1 = _calc_trans(points, temp) # n_theta x 2 x nP
        
        # Finite difference
        diff = (f1 - f0) / h # n_theta x 2 x nP
        
        if i != 0:
            # Gradient
            gradient = tf.concat([gradient, tf.expand_dims(tf.reduce_sum(grad * diff, axis=[1,2]), 1)], 1)
        else:
            gradient = tf.expand_dims(tf.reduce_sum(grad * diff, axis=[1,2]), 1)

    return [None, gradient]        

#%%
@function.Defun(tf.float32, tf.float32, func_name='tf_CPAB_transformer', python_grad_func=_calc_grad)
def tf_cuda_CPAB_transformer(points, theta):
    transformed_points = _calc_trans(points, theta)
    return transformed_points

#%%
@function.Defun(tf.float32, tf.float32, func_name = 'tf_CPAB_transformer_numeric_grad', python_grad_func = _calc_grad_numeric)
def tf_cuda_CPAB_transformer_numeric_grad(points, theta):
    """ Similar to tf_CPAB_transformer(...) where the analytic gradient is have
        been replaced with a numeric finite difference gradient
    """
    transformed_points = _calc_trans(points, theta)
    return transformed_points

#%%
def tf_pure_CPAB_transformer(points, theta):
    """ CPAB transformer in pure tensorflow. 
        Transform the input points by repeatly appling the matrix-exponentials 
        parametrized by theta. This function should automatic be able to calculate 
        the gradient of the output w.r.t. theta.
    
    Arguments:
        points: `Matrix` [2, n_points]. 2D input points to transform
        
        theta: `Matrix` [n_theta, dim]. Parametrization to use. 
            
    Output:
        trans_points: 3D-`Tensor` [n_theta, 2, n_points]. The transformed points
            for each parametrization in theta.
    """
    with tf.name_scope('CPAB_transformer'):
        # Make sure that both inputs are in float32 format
        points = tf.cast(points, tf.float32) # format [2, nb_points]
        theta = tf.cast(theta, tf.float32) # format [n_theta, dim]
        n_theta = tf.shape(theta)[0]
        n_points = tf.shape(points)[1]
        
        # Repeat point matrix, one for each theta
        newpoints = tf_repeat_matrix(points, n_theta) # [n_theta, 2, nb_points]
        
        # Reshape into a [nb_points*n_theta, 2] matrix
        newpoints = tf.reshape(tf.transpose(newpoints, perm=[0,2,1]), (-1, 2))
        
        # Add a row of ones, creating a [nb_points*n_theta, 3] matrix
        newpoints = tf.concat([newpoints, tf.ones((n_theta*n_points, 1))], axis=1)
        
        # Expand dims for matrix multiplication later -> [nb_points*n_theta, 3, 1] tensor
        newpoints = tf.expand_dims(newpoints, 2)
        
        # Load file with basis
        file = load_basis()
        
        # Tessalation information
        nC = tf.cast(file['nC'], tf.int32)
        ncx = tf.cast(file['ncx'], tf.int32)
        ncy = tf.cast(file['ncy'], tf.int32)
        inc_x = tf.cast(file['inc_x'], tf.float32)
        inc_y = tf.cast(file['inc_y'], tf.float32)
        
        # Steps sizes
        nStepSolver = 50 # Change this for more precision
        dT = 1.0 / tf.cast(nStepSolver, tf.float32)
        
        # Get cpab basis
        B = tf.cast(file['B'], tf.float32)

        # Repeat basis for batch multiplication
        B = tf_repeat_matrix(B, n_theta)
        
        # Calculate the row-flatted affine transformations Avees 
        Avees = tf.matmul(B, tf.expand_dims(theta, 2))
		
        # Reshape into (number of cells, 2, 3) tensor
        As = tf.reshape(Avees, shape = (n_theta * nC, 2, 3)) # format [n_theta * nC, 2, 3]
        
        # Multiply by the step size and do matrix exponential on each matrix
        Trels = tf_expm3x3_analytic(dT*As)
        Trels = tf.concat([Trels, tf.cast(tf.reshape(tf.tile([0,0,1], 
                [n_theta*nC]), (n_theta*nC, 1, 3)), tf.float32)], axis=1)
        
        # Batch index to add to correct for the batch effect
        batch_idx = (4*ncx*ncy) * tf.reshape(tf.transpose(tf.ones((n_points, n_theta), 
                    dtype=tf.int32)*tf.cast(tf.range(n_theta), tf.int32)),(-1,))
        
        # Body function for while loop (executes the computation)
        def body(i, points):
            # Find cell index of each point
            idx = tf_findcellidx(points, ncx, ncy, inc_x, inc_y)
            
            # Correct for batch
            corrected_idx = tf.cast(idx, tf.int32) + batch_idx
            
            # Gether relevant matrices
            Tidx = tf.gather(Trels, corrected_idx)
            
            # Transform points
            newpoints = tf.matmul(Tidx, points)
            
            # Shape information is lost, but tf.while_loop requires shape 
            # invariance so we need to manually set it (easy in this case)
            newpoints.set_shape((None, 3, 1)) 
            return i+1, newpoints
        
        # Condition function for while loop (indicates when to stop)
        def cond(i, points):
            # Return iteration bound
            return tf.less(i, nStepSolver)
        
        # Run loop
        trans_points = tf.while_loop(cond, body, [tf.constant(0), newpoints],
                                     parallel_iterations=10, back_prop=True)[1]
        # Reshape to batch format
        trans_points = tf.reshape(tf.transpose(trans_points[:,:2], perm=[1,0,2]), 
                                 (n_theta, 2, n_points))
        return trans_points

#%%
if __name__ == '__main__':
    from ddtn.transformers.setup_CPAB_transformer import setup_CPAB_transformer
    import numpy as np
    import matplotlib.pyplot as plt
    
    
    # Create basis
    s = setup_CPAB_transformer(2, 2, 
                               valid_outside=True, 
                               zero_boundary=False, 
                               override=True)
    
    # Sample parametrization and grid
    theta = 0.5*s.sample_theta_without_prior(1)
    points = s.sample_grid(20)
    
    # Convert to tf tensors
    theta_tf = tf.cast(theta, tf.float32)    
    points_tf = tf.cast(points, tf.float32)
    
    # Create computaitons
    newpoints_ana_tf = tf_cuda_CPAB_transformer(points_tf, theta_tf)
    newpoints_num_tf = tf_cuda_CPAB_transformer_numeric_grad(points_tf, theta_tf)
    newpoints_pur_tf = tf_pure_CPAB_transformer(points_tf, theta_tf)
    grad_ana_tf = tf.gradients(newpoints_ana_tf, [theta_tf])[0]
    grad_num_tf = tf.gradients(newpoints_num_tf, [theta_tf])[0]
    grad_pur_tf = tf.gradients(newpoints_pur_tf, [theta_tf])[0]
    
    sess = tf.Session()
    p1, p2, p3, g1, g2, g3 = sess.run([newpoints_ana_tf, 
                                       newpoints_num_tf, 
                                       newpoints_pur_tf,
                                       grad_ana_tf, 
                                       grad_num_tf,
                                       grad_pur_tf])
    
    # Print gradient res
    print('Analytic gradient:')
    print(g1.round(3))
    print('Numeric gradient:')
    print(g2.round(3))
    print('Pure gradient:')
    print(g3.round(3))
    print('Difference ana-num:', (np.linalg.norm(g1 - g2) / np.linalg.norm(g1)).round(3))
    print('Difference ana-pur:', (np.linalg.norm(g1 - g3) / np.linalg.norm(g1)).round(3))
    
    # Show deformation and velocity field
    fig = plt.figure()
    plt.plot(points[0], points[1], 'b.', label='original grid')
    plt.plot(p1[0,0], p1[0,1], 'r.', label='deformed grid')
    plt.plot(p3[0,0], p3[0,1], 'g.', label='deformed grid')
    plt.legend(fontsize=15)
    s.visualize_vectorfield_arrow(theta.flatten())
    plt.show()
    