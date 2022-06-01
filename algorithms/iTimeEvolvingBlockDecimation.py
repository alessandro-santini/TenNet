import numpy as np
import opt_einsum as oe
from ..tools.svd_truncate import svd_truncate
from scipy.sparse.linalg import expm

import copy

class iTEBD:
    def __init__(self, psi, H, dt, d=2, options={}):
        """
        Parameters
        ----------
        psi : iMPS
            state as an infinite MPS.
        H : np.ndarray
            two-body interaction.
        dt : float
            time_step.
        d : int, optional
            local space dimension. The default is 2.
        options : dict, optional
            svd_truncate options. The default is {}.
        """
        assert psi.d == d
        self.d = d
        self.psi = copy.deepcopy( psi )
        self.Hmat = H.copy()
        self.H = H.reshape(d,d,d,d)
        self.options = options
        if 'order'     not in self.options.keys(): self.options.update({'order':1})
        if 'trunc_cut' not in self.options.keys(): self.options.update({'trunc_cut':1e-10})
        if 'chi_max'   not in self.options.keys(): self.options.update({'chi_max':128})
        if 'svd_min'   not in self.options.keys(): self.options.update({'svd_min':1e-10})
        self.set_U(dt,order=options['order'])        
        self.truncation_err = 0.
        self.initial_energy = self.compute_energy()
        
    def set_U(self,dt,order=1):
        d = self.d
        H = self.Hmat
        self.U = []
        if order == 1:
            self.U.append(expm(-1j*H*dt).reshape(d,d,d,d).transpose(2,3,0,1))
            self.U.append(expm(-1j*H*dt).reshape(d,d,d,d).transpose(2,3,0,1))
        if order == 2:
            self.U.append(expm(-1j*H*dt/2).reshape(d,d,d,d).transpose(2,3,0,1))
            self.U.append(expm(-1j*H*dt).reshape(d,d,d,d).transpose(2,3,0,1))
            self.U.append(expm(-1j*H*dt/2).reshape(d,d,d,d).transpose(2,3,0,1))
        if order == 4:
            tau1 = 1./(4.-4.**(1./3.))*dt; tau2=tau1;
            tau3 = dt-2*tau1-2*tau2
            for tau in [tau1/2,tau1,(tau1+tau2)/2,tau2,(tau2+tau3)/2,tau3,tau3/2 ]:
                self.U.append(expm(-1j*H*tau/2).reshape(d,d,d,d).transpose(2,3,0,1))
        
    def time_step(self):
        for K in self.U:
            shp1,shp2 = self.psi.B1.shape,self.psi.B2.shape
            theta = oe.contract('a,abc,cde,bdfg->afge',self.psi.Sv,self.psi.B1,self.psi.B2, K)
            theta = theta.reshape(shp1[0]*shp1[1],shp2[1]*shp2[2])
            (U,S,V), err = svd_truncate(theta,self.options)
            U, V = U.reshape(shp1[0],shp1[1],S.size), V.reshape(S.size,shp2[1],shp2[2])
            self.truncation_err = max(err,self.truncation_err)
            self.psi.B2 = V
            self.psi.B1 = oe.contract('i,ijk,k->ijk',1./self.psi.Sv,U,S)
            self.psi.Sv, self.psi.B1, self.psi.B2 = S, self.psi.B2, self.psi.B1
            
    def err_normalization(self):
        return np.abs(self.psi.compute_norm()-1)
    
    def compute_energy(self):
        return self.psi.compute_two_body_observable(self.H)
    
    def err_energy(self):
        return np.abs(self.compute_energy()-self.initial_energy)