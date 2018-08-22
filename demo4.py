# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 12:50:17 2018

@author: nsde
"""

#%%
from tensorflow.python.keras._impl.keras.layers.core import Layer
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense, InputLayer

from cpablib import cpab

#%%
class TransformerLayer(Layer):
    
    def __init__(self, transformer_class, localization_net, **kwargs):
        self.transformer_class = transformer_class
        self.locnet = localization_net
        super(TransformerLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.locnet.build(input_shape)
        self._trainable_weights = self.locnet.trainable_weights
        super(TransformerLayer, self).build(input_shape)
    
    def compute_output_shape(self, input_shape):
        return (None, input_shape[-1])

    def get_config(self):
        config = super(TransformerLayer, self).get_config()
        config['localization_net'] = self.locnet
        config['transformer_class'] = self.transformer_class
        return config
    
    def call(self, X, mask=None):
        theta = self.locnet.call(X)
        output = self.transformer_class.transform_data(X, theta)
        return output


#%%
if __name__ == '__main__':
    # Load data
    # ...
    
    # Make cpab transformer
    T = cpab(tess_size=[5,], return_tf_tensors=True)
    
    # Construct localization network
    d = T.get_theta_dim()
    locnet = Sequential()
    locnet.add(Dense(128, activation='tanh', input_shape=input_shape))
    locnet.add(Dense(64, activation='tanh'))
    locnet.add(Dense(d, activation='tanh'))
    
    # Construct full model
    model = Sequential()
    model.add(InputLayer(input_shape=input_shape))
    model.add(TransformerLayer(T, locnet))
    model.add(Dense(128, activation='tanh'))
    model.add(Dense(64, activation='tanh'))
    model.add(Dense(1, activation='tanh'))
 
    # Comile model
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    
    # Get summary of model
    model.summary()
    
    # Fit model
    model.fit(X_train, y_train,
              batch_size=args['batch_size'],
              epochs=args['num_epochs'],
              validation_data=(X_test, y_test),
              callbacks=[logger])    
    