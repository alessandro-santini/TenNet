from ..tensors import MatrixProductOperators as MPO
import tensorflow as tf
import numpy as np

def IsingChain(L, J=1., h_z=1., h_x=0.):    
    if isinstance(J,float):   J = J*np.ones(L)
    if isinstance(h_x,float): h_x = h_x*np.ones(L)
    if isinstance(h_z,float): h_z = h_z*np.ones(L)
    
    tensors = []
    
    sigma_z = np.array([[1.,0.],[0.,-1.]])
    sigma_x = np.array([[0.,1.],[1.,0]])
    
    WL = np.zeros((1,3,2,2))
    WR = np.zeros((3,1,2,2))
    Wbulk = np.zeros((3,3,2,2))
    
    WL[0,0,:,:] = -h_x[0]*sigma_x-h_z[0]*sigma_z; WL[0,1,:,:] = -J[0]*sigma_x; WL[0,2,: ,:] = np.eye(2)
    WR[0,0,:,:] = np.eye(2); WR[1,0,:,:] = sigma_x; WR[2,0,:,:] = -h_x[-1]*sigma_x-h_z[-1]*sigma_z;
    tensors.append(WL)
    for i in range(1,L-1):
        Wbulk[0,0,:,:] = np.eye(2); Wbulk[1,0,:,:] = sigma_x; Wbulk[2,0,:,:] = -h_x[i]*sigma_x-h_z[i]*sigma_z
        Wbulk[2,1,:,:] = -J[i]*sigma_x; Wbulk[2,2,:,:] = np.eye(2)
        tensors.append(Wbulk)
    tensors.append(WR)
    tensors = list(map(tf.convert_to_tensor,tensors))
    return MPO.MPO(L, d=2, tensors = tensors)
    