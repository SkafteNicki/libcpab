# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 13:50:53 2018

@author: nsde
"""

# TODO: find and fix the sparse tensor -> tensor consume lots of memory error
# TODO: check the gradient at the identity 
#       only seems to be a problem for the pure transformer case
# TODO: fix the problem in the 1D pure transformer shape mismatch between Trels 
#       and points - shape is maybe changed in in the loop?
# TODO: get demo3 to work
# TODO: get demo4 to work
# TODO: test interpolation for 1D
# TODO: test interpolation for 3D

#%%
if __name__ == '__main__':
    import tensorflow as tf
    from libcpab import cpab
    from libcpab.helper.utility import load_basis
    from libcpab.helper.tf_funcs import tf_shape_i, tf_repeat_matrix
    from libcpab.helper.tf_expm import tf_expm2x2
    from libcpab.helper.tf_findcellidx import tf_findcellidx_1D
    
    T1 = cpab([4,])
    theta = T1.sample_transformation(2)
    points = T1.uniform_meshgrid([50,])
    
    # Tessalation information
    tess = load_basis()
    ndim = tess['ndim']
    nC = tf.cast(tess['nC'], tf.int32)
    ncx = tf.cast(tess['nc'][0], tf.int32)
    nStepSolver = tf.cast(tess['nstepsolver'], tf.int32)
    
    # Make sure that both inputs are in float32 format
    points = tf.cast(points, tf.float32) # format [1, nb_points]
    theta = tf.cast(theta, tf.float32) # format [n_theta, dim]
    n_theta = tf_shape_i(theta, 0)#tf.shape(theta)[0]
    n_points = tf_shape_i(points, 1)#points.get_shape().as_list()[1]
    
    # Repeat point matrix, one for each theta
    newpoints = tf_repeat_matrix(points, n_theta) # [n_theta, 1, nb_points]
    
    # Reshape into a [nb_points*n_theta, 1] matrix
    newpoints = tf.reshape(tf.transpose(newpoints, perm=[0,2,1]), (-1, ndim))
    
    # Add a row of ones, creating a [nb_points*n_theta, 2] matrix
    newpoints = tf.concat([newpoints, tf.ones((n_theta*n_points, 1))], axis=1)
    
    # Expand dims for matrix multiplication later -> [nb_points*n_theta, 2, 1] tensor
    newpoints = tf.expand_dims(newpoints, axis=2)
    
    # Steps sizes
    dT = 1.0 / tf.cast(nStepSolver, tf.float32)
    
    # Get cpab basis
    B = tf.cast(tess['basis'], tf.float32)

    # Calculate the row-flatted affine transformations Avees 
    Avees = tf.transpose(tf.matmul(B, tf.transpose(theta)))
		
    # Reshape into (n_theta*number_of_cells, 1, 2) tensor
    As = tf.reshape(Avees, shape = (n_theta * nC, *tess['Ashape'])) # format [n_theta * nC, 1, 2]
    
    # Multiply by the step size and do matrix exponential on each matrix
    Trels = tf_expm2x2(dT*As)
    
    # Batch index to add to correct for the batch effect
    batch_idx = nC * tf.reshape(tf.transpose(tf.ones((n_points, n_theta), 
                dtype=tf.int32)*tf.cast(tf.range(n_theta), tf.int32)),(-1,))
    
    # Body function for while loop (executes the computation)
    def body(i, points):
        # Find cell index of each point
        idx = tf_findcellidx_1D(points, ncx)
        
        # Correct for batch
        corrected_idx = tf.cast(idx, tf.int32) + batch_idx
        
        # Gether relevant matrices
        Tidx = tf.gather(Trels, corrected_idx)
        
        print(Trels)
        print(idx)
        print(batch_idx)
        print(corrected_idx)
        print(Tidx)
        
        # Transform points
        newpoints = tf.matmul(Tidx, points)
        
        # Shape information is lost, but tf.while_loop requires shape 
        # invariance so we need to manually set it (easy in this case)
        newpoints.set_shape((None, ndim+1, 1)) 
        return i+1, newpoints
    
    # Condition function for while loop (indicates when to stop)
    def cond(i, points):
        # Return iteration bound
        return tf.less(i, nStepSolver)
    
    # Run loop
    trans_points = tf.while_loop(cond, body, [tf.constant(0), newpoints],
                                 parallel_iterations=10, back_prop=True)[1]
    # Reshape to batch format
    trans_points = tf.transpose(tf.reshape(tf.squeeze(trans_points[:,:ndim]), 
                        (n_theta, n_points, ndim)), perm=[0,2,1])
    return trans_points


    
    
    
    
    
    