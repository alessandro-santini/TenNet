import tensorflow as tf
import numpy as np
import opt_einsum as oe
import copy
from ..tools import contract

class MPO:
    def __init__(self, L, d=2, tensors = None, dtype=tf.float64):
        self.L = L
        self.d = d
        self.dtype = tf.float64
        if tensors != None:
            self.tensors = copy.deepcopy(tensors)
        else:
            self.tensors = [0]*L
            
    def contractMPOtoMPS(self, psi):
       if(psi.L != self.L): raise Exception('psi MPO length are different')
       if(psi.d != self.d): raise Exception('psi MPO local dimensions are different')
       
       R_env = tf.ones((1,1,1), dtype = psi.dtype)
       if psi.dtype != self.dtype:
           for i in range(self.L-1,0,-1):
               R_env = contract.contract_right(psi.tensors[i], tf.cast(self.tensors[i],psi.dtype), tf.math.conj(psi.tensors[i]), R_env)
           return contract.contract_right(psi.tensors[0], tf.cast(self.tensors[0],psi.dtype), tf.math.conj(psi.tensors[0])).numpy().item()
       else:
           for i in range(self.L-1,0,-1):
               R_env = contract.contract_right(psi.tensors[i], self.tensors[i], psi.tensors[i], R_env)
           return contract.contract_right(psi.tensors[0], self.tensors[0], psi.tensors[0], R_env).numpy().item()