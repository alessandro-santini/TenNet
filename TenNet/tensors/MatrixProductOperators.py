import numpy as np
import copy
from ..tools import contract
from ..tools.svd_truncate import svd_truncate
from . import IOhdf5
import opt_einsum as oe

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
   
    def compressMPO(self,options={'trunc_cut':1e-15}):
        L = self.L
        old_tensors = copy.deepcopy(self.tensors)

        for j in range(L-1):
            W = self.tensors[j].transpose(0,2,3,1)
            shpW = W.shape
            q, r = np.linalg.qr(W.reshape(shpW[0]*shpW[1]*shpW[2],shpW[3]))
            q.reshape(shpW[0],shpW[1],shpW[2],q.shape[1]).transpose(0, 3, 1, 2)            
            self.tensors[j] = q.reshape(shpW[0],shpW[1],shpW[2],q.shape[1]).transpose(0, 3, 1, 2)
            self.tensors[j+1] = oe.contract('ij,jklm->iklm',r,self.tensors[j+1])

        for j in range(L-1,0,-1):
            W = self.tensors[j].transpose(0,2,3,1)
            shpW = W.shape
            Snorm = np.linalg.norm( np.linalg.svd(W.reshape(shpW[0],shpW[1]*shpW[2]*shpW[3]),compute_uv=False) )
            (U,S,V),_ = svd_truncate(W.reshape(shpW[0],shpW[1]*shpW[2]*shpW[3]),options)
            self.tensors[j] = V.reshape(S.size,shpW[1],shpW[2],shpW[3]).transpose(0,3,1,2)
            self.tensors[j-1] = oe.contract('ijlm,jk,k->iklm',self.tensors[j-1],U,S)*Snorm
        
        ######################
        # Estimate the error #
        ######################
        Rtemp_1 = np.ones((1,1))
        Rtemp_2 = np.ones((1,1))
        for i in range(L):
            Rtemp_1 = oe.contract('ijkl,mnlk,im->jn',old_tensors[i],old_tensors[i],Rtemp_1)
            Rtemp_2 = oe.contract('ijkl,mnlk,im->jn',old_tensors[i],self.tensors[i],Rtemp_2)
        print('Err compression MPO',np.abs(Rtemp_1.item()-Rtemp_2.item())/np.abs(Rtemp_1.item()))
                        
    def save(self, file_pointer, subgroup):
        IOhdf5.save_hdf5(self, file_pointer, subgroup)
    def load(self, file_pointer, subgroup):
        IOhdf5.load_hdf5(self, file_pointer, subgroup)