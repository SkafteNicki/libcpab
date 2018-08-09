import numpy as np
import pickle
import matplotlib.pyplot as plt

from tf_sample_transformation import tf_sample_transformation

tesselation = pickle.load( open( "tesselation.pkl", "rb" ) )

m  = tesselation["len_theta"]

#%% Loading all optimized transformations
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

spine = 'case7'

tf_sample_transformation(spine, Cov, np.random.randint(0,100))
	

