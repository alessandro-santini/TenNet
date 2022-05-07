from ..tensors.MatrixProductState import MPS
import numpy as np

def all_up(L):
    tensors = [np.array([1.,0.]).reshape(1,2,1)]*L
    psi =  MPS(L,tensors = tensors)
    psi.center = 0
    return psi

def all_down(L):
    tensors = [np.array([0.,1.]).reshape(1,2,1)]*L
    psi =  MPS(L,tensors = tensors)
    psi.center = 0
    return psi

def Neel(L):
    tensors = [np.array([1.,0.]).reshape(1,2,1),np.array([0.,1.]).reshape(1,2,1)]*(L//2)
    psi =  MPS(L,tensors = tensors)
    psi.center = 0
    return psi    
    
    