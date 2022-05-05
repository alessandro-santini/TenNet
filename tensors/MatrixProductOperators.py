import numpy as np
import copy
from ..tools import contract
import IOhdf5

class MPO:
    def __init__(self, L, d=2, tensors = None):
        self.L = L
        self.d = d
        if tensors != None:
            self.tensors = copy.deepcopy(tensors)
        else:
            self.tensors = [0]*L
            
    def contractMPOtoMPS(self, psi):
       if(psi.L != self.L): raise Exception('psi MPO length are different')
       if(psi.d != self.d): raise Exception('psi MPO local dimensions are different')
       
       R_env = np.ones((1,1,1))
       for i in range(self.L-1,0,-1):
           R_env = contract.contract_right(psi.tensors[i], self.tensors[i], psi.tensors[i], R_env)
       return contract.contract_right(psi.tensors[0], self.tensors[0], psi.tensors[0], R_env).item()
   
    def save(self, file_pointer, subgroup):
        IOhdf5.save_hdf5(self, file_pointer, subgroup)
    def load(self, file_pointer, subgroup):
        IOhdf5.load_hdf5(self, file_pointer, subgroup)