# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 15:40:59 2018

@author: nsde
"""

if __name__ == "__main__":
    import tensorflow as tf
    import numpy as np
    import matplotlib.pyplot as plt
    from libcpab import Cpab

    from tensorflow import keras
    
    class mylayer(keras.layers.Layer):
        def __init__(self):
            self.cpab = Cpab([5,], backend='tensorflow', device='gpu')
            super().__init__()
            
        def call(self, x):
            return self.cpab.transform_data(x,
                                            self.cpab.sample_transformation(100),
                                            outsize=(50,))
            
    model = keras.Sequential()
    model.add(keras.layers.InputLayer(input_shape=(50,10), batch_size=100))
    model.add(mylayer())
    model.add(keras.layers.Conv1D(32, 7, activation='relu'))
    model.add(keras.layers.Conv1D(64, 5, activation='relu'))
    model.add(keras.layers.Conv1D(128, 3, activation='relu'))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(10, activation='softmax'))
    
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    
    X = np.random.rand(1000,50,10)
    y = np.random.random_integers(0,9,1000)
    y = keras.utils.to_categorical(y, 10)
    
    model.fit(X,y,batch_size=100,epochs=10)
    model.save_weights('myweights.h5')
    model.load_weights('myweights.h5')
    
    
#    T = Cpab([3, 3], backend='tensorflow', device='gpu', zero_boundary=True,
#             volume_perservation=False, override=False)
#
#    theta = T.sample_transformation()
#    #theta = T.identity()
#
#    grid = T.uniform_meshgrid([350, 350])
#
#    grid_t = T.transform_grid(grid, theta)
#    
#    im = plt.imread('version1.4/data/cat.jpg')
#    im = np.expand_dims(im, 0) / 255
#    im = tf.cast(im, tf.float32)
#    
#    im_t = T.interpolate(im, grid_t, outsize=(350, 350))
#    
#    im = im.numpy()
#    im_t = im_t.numpy()
#    
#    plt.figure()
#    plt.imshow(im[0])
#    plt.axis('off')    
#    plt.tight_layout()
#    plt.imsave('data/cat.jpg', im[0])
#    
#    plt.figure()
#    plt.imshow(im_t[0])
#    plt.axis('off')    
#    plt.tight_layout()
#    plt.imsave('data/deform_cat.jpg', im_t[0])
#
#    T.visualize_vectorfield(theta)
#    plt.axis([0,1,0,1])
#    plt.tight_layout()
#    plt.savefig('data/velocity_field.jpg')
#    plt.show()
