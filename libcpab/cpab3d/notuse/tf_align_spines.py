import tensorflow as tf
import numpy as np
import time
import imageio
import scipy.io as sio
import scipy
import matplotlib
import pickle
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from matplotlib.pyplot import savefig
from tf_CPAB_transformer import tf_CPAB_transformer

#%% Optimizing alignment
sess = tf.Session()
	
#%% Reading tesselation parameters	
tesselation = pickle.load( open( "tesselation.pkl", "rb" ) )

nx = tf.cast( tesselation["nx"], dtype=tf.int32 )
ny = tf.cast( tesselation["ny"], dtype=tf.int32 )
nz = tf.cast( tesselation["nz"], dtype=tf.int32 )
m  = tesselation["len_theta"]

s = 1 # Spine to be transformed
t = 3 # Target spine

#%% reading spine and target points
string_s = 'spine'+str(s)
string_t = 'spine'+str(t)

S = pickle.load( open( "spine_points.pkl", "rb" ) )

S_p = np.transpose(S[string_s])
T_p = np.transpose(S[string_t])

spine = tf.convert_to_tensor( S_p, dtype=tf.float32 )
target = tf.convert_to_tensor( T_p, dtype=tf.float32 )

#%% Inializing optimization parameters
theta = tf.Variable( tf.zeros([1,m]), name="theta", dtype=tf.float32)
newspine = tf_CPAB_transformer(spine, theta)
	    
loss = tf.reduce_sum( tf.square(newspine-target) )
optimizer = tf.train.GradientDescentOptimizer(0.1)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

def optimize():
	with tf.Session() as session:
		session.run(init)
		for step in range(250):
		    session.run(train)
		return(session.run(theta), sess.run(loss))

#%% Optimizing
t_opt, l = optimize()

print(l)

theta_opt = tf.reshape( tf.convert_to_tensor(t_opt, dtype=tf.float32), [1,m])
spine_algn = tf_CPAB_transformer(spine, theta_opt)

s_algn = np.reshape( spine_algn.eval(session=sess), [3,100] )

fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
plt.xlim([0,1])
plt.ylim([0,1])
#ax.plot(S_p[0], S_p[1], S_p[2])
#ax.plot(T_p[0], T_p[1], T_p[2])
#ax.plot(s_algn[0], s_algn[1], s_algn[2])
plt.plot(S_p[1], S_p[2])
#plt.plot(T_p[1], T_p[2])
plt.plot(s_algn[1], s_algn[2])

plt.show()

