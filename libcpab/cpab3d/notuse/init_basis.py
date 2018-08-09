import tensorflow as tf
import numpy as np
import pickle
import os.path
from Transform import TransformNS

nx = 3
ny = 3
nz = 3

def init_basis(nx, ny, nz):
    with tf.variable_scope('calc_trans'):
        file_name = "TrNS/"+str(nx)+"x"+str(ny)+"x"+str(nz)+"_TrNS_B"
        if not os.path.isfile(file_name+".npy"):
            print("Computing null space for given tesselation")
            Z,_ = TransformNS(nx,ny,nz,True)
            np.save(file_name, Z)
        else:
            Z = np.load(file_name+".npy")
        
        _,m = np.shape(Z)
        
        N = 6*nx*ny*nz
        
        tesselation = {"nx": nx, "ny": ny, "nz": nz, "N": N, "Basis": Z, "len_theta": m}
        pickle.dump( tesselation, open( "tesselation.pkl", "wb" )  )
        return m

init_basis(nx,ny,nz)
