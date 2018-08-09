import tensorflow as tf
import numpy as np
from scipy.misc import imresize

from scipy.io import loadmat, savemat

def trilinear_sampler(spine, trn):
	string = '../../../translated/' + spine + '.mat'
	case = loadmat(string)
	
	if len(spine) > 8:
		Im = case['T']
	else:
		Im = case['Im']
	
	w,d,h = np.shape(Im)

	trn = np.minimum( np.maximum(trn,0.00001), 0.99999) 
	c = np.zeros(np.shape(trn)[1])

	x = (w-1)*trn[0]
	y = (d-1)*trn[1]
	z = (h-1)*trn[2]

	x0 = np.floor( x ).astype(int)
	x1 = x0+1
	y0 = np.floor( y ).astype(int)
	y1 = y0+1
	z0 = np.floor( z ).astype(int)
	z1 = z0+1

	xd = (x-x0)/(x1-x0)
	yd = (y-y0)/(y1-y0)
	zd = (z-z0)/(z1-z0)

	c000 = Im[x0,y0,z0]
	c001 = Im[x0,y0,z1]
	c010 = Im[x0,y1,z0]
	c011 = Im[x0,y1,z1]
	c100 = Im[x1,y0,z0]
	c101 = Im[x1,y0,z1]
	c110 = Im[x1,y1,z0]
	c111 = Im[x1,y1,z1]

	c00 = c000*(1-xd) + c100*xd
	c01 = c001*(1-xd) + c101*xd
	c10 = c010*(1-xd) + c110*xd
	c11 = c011*(1-xd) + c111*xd

	c0 = c00*(1-yd) + c10*yd
	c1 = c01*(1-yd) + c11*yd

	c = c0*(1-zd) + c1*zd

	return c
	
