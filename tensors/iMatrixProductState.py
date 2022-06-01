import numpy as np
import opt_einsum as oe
from . import IOhdf5

class iMPS:
    def __init__(self,Sv,B1,B2,d=2):
        self.B1 = B1.copy()
        self.B2 = B2.copy()
        self.Sv = Sv.copy()
        self.d  = d
        
    def compute_norm(self):
        self.norm = np.real_if_close(oe.contract('i,ijk,klm,ijn,nlm',self.Sv**2,self.B1,self.B2,self.B1.conj(),self.B2.conj()))
        return self.norm
        
    def compute_local_observable(self,op):
        return np.real_if_close(oe.contract('i,ijk,ilk,jl',self.Sv**2,self.B1,self.B1.conj(),op).item())
    def compute_two_body_observable(self,op2):
        return np.real_if_close(oe.contract('a,abc,cde,afg,ghe,bdfh',self.Sv**2,self.B1,self.B2,self.B1.conj(),self.B2.conj(),op2).item())
    
    def compute_corr(self,r,opi,opj=None):
        corr = np.zeros(r,complex)
        if opj is None:
            opj = opi
        corr[0] = self.compute_local_observable(opi@opj)
        L = oe.contract('a,abc,ade,bd->ce',self.Sv**2,self.B1,self.B1.conj(),opi)
        for j in range(1,r):
            if j % 2 == 0:
                corr[j] = oe.contract('ab,acd,bed,ce',L,self.B1,self.B1.conj(),opj).item()
                L = oe.contract('ab,acd,bcf->df',L,self.B1,self.B1.conj())
            else:
                corr[j] = oe.contract('ab,acd,bed,ce',L,self.B2,self.B2.conj(),opj).item()
                L = oe.contract('ab,acd,bcf->df',L,self.B2,self.B2.conj())
        return np.real_if_close(corr)
    def compute_connected_corr(self,r,opi,opj=None):
        if opj is None:
            opj = opi
        return self.compute_corr(r,opi,opj)-self.compute_local_observable(opi)*self.compute_local_observable(opj)
    
    def compute_entanglement_entropy(self):
        return -np.sum(self.Sv**2*np.log(self.Sv**2))
    
    def set_tensors(self):
        self.tensors=[self.Sv,self.B1,self.B2]
    
    def save(self, file_pointer, subgroup):
        self.set_tensors()
        IOhdf5.save_hdf5(self, file_pointer, subgroup)
    def load(self, file_pointer, subgroup):
        self.set_tensors()
        IOhdf5.load_hdf5(self, file_pointer, subgroup)