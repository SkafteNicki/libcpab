#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 12:53:10 2018

@author: nsde
"""

if self._is_tf_tensor(n_points):
            lin_p = [tf.linspace(tf.cast(self._domain_min[i], tf.float32), 
                                 tf.cast(self._domain_max[i], tf.float32), n_points[i])
                    for i in range(self._ndim)]
            mesh_p = tf.meshgrid(*lin_p)                        
            grid = tf.concat([tf.reshape(array, (1, -1)) for array in mesh_p], axis=0)
            return grid
        else:    