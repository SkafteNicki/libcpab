import tensorflow as tf
import numpy as np
import pickle

from scipy.io import loadmat, savemat
from tf_CPAB_transformer import tf_CPAB_transformer
from trilinear_interp import trilinear_sampler

sess = tf.Session()
tesselation = pickle.load( open( "tesselation.pkl", "rb" ) )

nx = tf.cast( tesselation["nx"], dtype=tf.int32 )
ny = tf.cast( tesselation["ny"], dtype=tf.int32 )
nz = tf.cast( tesselation["nz"], dtype=tf.int32 )
m  = tesselation["len_theta"]

w = 512
d = 512
h = 560
x = np.matlib.repmat(np.linspace(0+1/(2*w),1-1/(2*w),w),1,d*h)
y = np.matlib.repmat(np.repeat(np.linspace(0+1/(2*d),1-1/(2*d),d),w),1,h)
z = np.reshape( np.repeat(np.linspace(0+1/(2*h),1-1/(2*h),h),w*d), [1,w*d*h] )

V = tf.convert_to_tensor( np.concatenate( (x,y,z), axis=0 ), dtype=tf.float32 )

#theta = tf.random_uniform([1,m], minval=-1, maxval=1, dtype=tf.float32)
t_1_2 = np.load('alignments/transform_spine1_3x3x3.npy')
theta = tf.convert_to_tensor( np.reshape(-t_1_2[0],[1,m]), dtype=tf.float32)

nV = tf_CPAB_transformer(V, theta)

trn = np.squeeze(nV.eval(session=sess))

case = loadmat('../../../resized/case1.mat')

Im = case['Im']

nIm = np.reshape( trilinear_sampler(Im, trn), [512,512,560] )

savemat('transformed/case1.mat',{'Im': nIm},do_compression=True)


