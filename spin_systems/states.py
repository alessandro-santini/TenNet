from ..tensors.MatrixProductState import MPS
import numpy as np

def all_up(L):
    tensors = [np.array([1.,0.]).reshape(1,2,1)]*L
    return MPS(L,tensors = tensors)

def all_down(L):
    tensors = [np.array([0.,1.]).reshape(1,2,1)]*L
    return MPS(L,tensors = tensors)

def Neel(L):
    tensors = [np.array([1.,0.]).reshape(1,2,1),np.array([0.,1.]).reshape(1,2,1)]*(L//2)
    return MPS(L,tensors=tensors)
    
    
    