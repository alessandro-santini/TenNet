from ..tensors.MatrixProductState import MPS
import tensorflow as tf
import numpy as np

def all_up(L):
    tensors = [np.array([1.,0.]).reshape(1,2,1)]*L
    tensors = list(map(tf.convert_to_tensor,tensors))
    return MPS(L,tensors = tensors)

def all_down(L):
    tensors = [np.array([0.,1.]).reshape(1,2,1)]*L
    tensors = list(map(tf.convert_to_tensor,tensors))
    return MPS(L,tensors = tensors)

    
    
    