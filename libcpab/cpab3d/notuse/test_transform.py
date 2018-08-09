import tensorflow as tf
import numpy as np
import pickle

from scipy.io import loadmat, savemat
from tf_CPAB_transformer import tf_CPAB_transformer
from trilinear_interp import trilinear_sampler

tesselation = pickle.load( open( "tesselation.pkl", "rb" ) )

m  = tesselation["len_theta"]

All_algn = np.zeros((21,m))
Loss = np.zeros(21)
it = 0;
for i in range(6):
	thetas = np.load( 'transform_spine' + str(i+1) + '_3x3x3.npy' )
	loss =  np.load( 'transform_loss_spine' + str(i+1) + '.npy' )
	for j in range(i,6):
		All_algn[it] = thetas[j]
		Loss[it] = loss[j]
		it += 1

print(np.shape(Loss))

All_algn = All_algn[Loss<1]
n,_ = np.shape(All_algn)
for i in range(n):
	All_algn = np.concatenate( (All_algn, np.reshape(-All_algn[i],[1,m])), 0)

#All_algn = All_algn - np.mean(All_algn, 0) # zero-mean;
#print( np.mean(All_algn,0) )

#%% Finding covariance
Cov = np.cov(All_algn.T)

l,v = np.linalg.eigh(Cov)

theta = np.reshape( 3*np.sqrt(l[-1])*v[-1], [1,m] )

with tf.Session() as sess:

	w = 512
	d = 512
	h = 560
	x = np.matlib.repmat(np.linspace(0+1/(2*w),1-1/(2*w),w),1,d*h)
	y = np.matlib.repmat(np.repeat(np.linspace(0+1/(2*d),1-1/(2*d),d),w),1,h)
	z = np.reshape( np.repeat(np.linspace(0+1/(2*h),1-1/(2*h),h),w*d), [1,w*d*h] )

	V = tf.convert_to_tensor( np.concatenate( (x,y,z), axis=0 ), dtype=tf.float32 )

	t_sample = np.random.multivariate_normal( np.zeros([m]), Cov )
	theta = tf.convert_to_tensor( theta, dtype=tf.float32 )

	nV = tf_CPAB_transformer(V, theta)
	trn = np.squeeze(nV.eval(session=sess))

	nIm = trilinear_sampler('case1', trn)

	savemat('transformed/case1.mat',{'Im': nIm},do_compression=True)
