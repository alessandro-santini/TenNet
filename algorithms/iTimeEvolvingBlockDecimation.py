import numpy as np
import opt_einsum as oe
from ..tools.svd_truncate import svd_truncate
from scipy.sparse.linalg import expm

class iTEBD:
    def __init__(self, Sv, B1, B2, H, dt, d=2, options={}):
        self.B1, self.B2= B1.copy(), B2.copy()
        self.Sv = Sv.copy()
        self.d = d
        self.H = H.reshape(d,d,d,d)
        self.U = expm(-1j*H*dt).reshape(d,d,d,d).transpose(2,3,0,1)
        self.options = options
        if 'trunc_cut' not in self.options.keys(): self.options.update({'trunc_cut':1e-10})
        if 'chi_max'   not in self.options.keys(): self.options.update({'chi_max':128})
        if 'svd_min'   not in self.options.keys(): self.options.update({'svd_min':1e-10})
        self.truncation_err = 0.
        self.initial_energy = self.compute_energy()
        
    def time_step(self):
        for _ in range(2):
            shp1,shp2 = self.B1.shape,self.B2.shape
            theta = oe.contract('a,abc,cde,bdfg->afge',self.Sv,self.B1,self.B2,self.U)
            theta = theta.reshape(shp1[0]*shp1[1],shp2[1]*shp2[2])
            (U,S,V), err = svd_truncate(theta,self.options)
            U, V = U.reshape(shp1[0],shp1[1],S.size), V.reshape(S.size,shp2[1],shp2[2])
            self.truncation_err = max(err,self.truncation_err)
            self.B2 = V
            self.B1 = oe.contract('i,ijk,k->ijk',1./self.Sv,U,S)
            self.Sv, self.B1, self.B2 = S, self.B2, self.B1
            
    def err_normalization(self):
        self.norm = oe.contract('i,ijk,klm,ijn,nlm',self.Sv**2,self.B1,self.B2,self.B1.conj(),self.B2.conj())
        return np.abs(self.norm-1)
    
    def compute_local_observable(self,op):
        return oe.contract('i,ijk,ilk,jl',self.Sv**2,self.B1,self.B1.conj(),op).item().real
    
    def compute_energy(self):
        return oe.contract('a,abc,cde,afg,ghe,bdfh',self.Sv**2,self.B1,self.B2,self.B1.conj(),self.B2.conj(),self.H).item().real
    
    def err_energy(self):
        return np.abs(self.compute_energy()-self.initial_energy)
    
    def compute_corr(self,r,opi,opj=None):
        corr = np.zeros(r)
        if opj == None:
            opj = opi
        corr[0] = self.compute_local_observable(opi@opj)
        L = oe.contract('a,abc,ade,bd->ce',self.Sv**2,self.B1,self.B1.conj(),opi)
        for j in range(1,r):
            if j % 2 == 0:
                corr[j] = oe.contract('ab,acd,bed,ce',L,self.B1,self.B1.conj(),opj).item().real
                L = oe.contract('ab,acd,bcf->df',L,self.B1,self.B1.conj())
            else:
                corr[j] = oe.contract('ab,acd,bed,ce',L,self.B2,self.B2.conj(),opj).item().real
                L = oe.contract('ab,acd,bcf->df',L,self.B2,self.B2.conj())
        return corr
            
            
        