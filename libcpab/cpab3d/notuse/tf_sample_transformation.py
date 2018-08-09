import tensorflow as tf
import numpy as np
import pickle

from trilinear_interp import trilinear_sampler
from scipy.io import loadmat, savemat
from tf_CPAB_transformer import tf_CPAB_transformer

def tf_sample_transformation(spine, Cov, j):
	#with tf.Graph().as_default():
	with tf.Session() as sess:
		tesselation = pickle.load( open( "tesselation.pkl", "rb" ) )

		m  = tesselation["len_theta"]

		w = 512
		d = 512
		h = 560
		x = np.matlib.repmat(np.linspace(0+1/(2*w),1-1/(2*w),w),1,d*h)
		y = np.matlib.repmat(np.repeat(np.linspace(0+1/(2*d),1-1/(2*d),d),w),1,h)
		z = np.reshape( np.repeat(np.linspace(0+1/(2*h),1-1/(2*h),h),w*d), [1,w*d*h] )

		V = tf.convert_to_tensor( np.concatenate( (x,y,z), axis=0 ), dtype=tf.float32 )

		t_sample = np.random.multivariate_normal( np.zeros([m]), Cov )

		theta = tf.convert_to_tensor( np.reshape(-t_sample,[1,m]), dtype=tf.float32 )

		nV = tf_CPAB_transformer(V, theta)
		del V
		nV = np.squeeze(nV.eval(session=sess))

		sess.close()
	tf.reset_default_graph()
	nIm = trilinear_sampler(spine, trn)
	
	string = 'transformed/' + spine + '_' + str(j) + '.mat'
	savemat( string, {'Im': nIm}, do_compression=True )
	del nIm

	label = 'labled/' + spine + '_label'

	nIm = np.round( trilinear_sampler(label, trn) )
	string = 'transformed/labled/' + spine + '_label_' + str(j) + '.mat'
	savemat( string, {'T': nIm}, do_compression=True )
	sess.close()
