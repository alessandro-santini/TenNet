import numpy as np
import opt_einsum as oe
from ..tools import contract, lanczos
from ..tools.svd_truncate import svd_truncate
import copy
import warnings

class TDVP:
    def __init__(self, psi, H, options={}):
        assert psi.L == H.L
        self.H = H
        self.L = self.H.L
        self.psi = copy.deepcopy(psi)
        self.options = options
        self.sites = options.get('sites', 1)
        if 'trunc_cut' not in self.options.keys(): self.options.update({'trunc_cut':1e-10})
        if 'chi_max'   not in self.options.keys(): self.options.update({'chi_max':64})
        if 'krydim'    not in self.options.keys(): self.options.update({'krydim':10})
        self.initialize()
        
    def initialize(self):
        if self.psi.center != 0: 
            warnings.warn('Initialization: psi was not right normalized')
            self.psi.right_normalize()
        self.L_env = [0]*(self.L+1)
        self.R_env = [0]*(self.L+1)
        
        self.L_env[-1] = np.ones((1,1,1))
        self.R_env[-1] = np.ones((1,1,1))
        
        for j in range(self.L-1,0,-1):
            self.R_env[j] = contract.contract_right(self.psi.tensors[j], self.H.tensors[j], self.psi.tensors[j].conj(), self.R_env[j+1])
        if self.sites == 2:
            self.H12 = [oe.contract('ijkl,jabc->iakblc',self.H.tensors[i],self.H.tensors[i+1]) for i in range(self.L-1)]
            self.truncation_err = np.zeros(self.L-1)
        self.initial_energy = self.energy()
    def energy(self):
        return self.H.contractMPOtoMPS(self.psi).real
    def energy_err(self):
        return self.energy()-self.initial_energy
    ###########################
    # Single site tdvp sweeps #
    ###########################
    def right_sweep_single_site(self,delta):
        for i in range(self.L):
            assert self.psi.center == i
            shp = self.psi.tensors[i].shape
            psi = local_exponentiation_single_site('Heff_single', self.psi.tensors[i], self.L_env[i-1], self.H.tensors[i], self.R_env[i+1], delta, self.options['krydim'])
            U,S,V = np.linalg.svd(psi.reshape(shp[0]*shp[1],shp[2]),full_matrices=False); S /= np.linalg.norm(S)
            self.psi.tensors[i] = U.reshape(shp[0],shp[1],S.size)
            self.L_env[i] = contract.contract_left(self.psi.tensors[i], self.H.tensors[i], self.psi.tensors[i], self.L_env[i-1])
            if i != self.L-1:
                self.psi.singular_values[i] = S
                self.psi.center += 1
                C = oe.contract('i,ij->ij',S,V)
                psi = local_exponentiation_single_site('Hfree', C, self.L_env[i],' ', self.R_env[i+1], -delta, self.options['krydim'])
                self.psi.tensors[i+1] = oe.contract('ij,jkl->ikl',psi.reshape(C.shape),self.psi.tensors[i+1])
    def left_sweep_single_site(self,delta):
        for i in range(self.L-1,-1,-1):
            assert self.psi.center == i
            shp = self.psi.tensors[i].shape
            psi = local_exponentiation_single_site('Heff_single', self.psi.tensors[i], self.L_env[i-1], self.H.tensors[i], self.R_env[i+1], delta, self.options['krydim'])
            U,S,V = np.linalg.svd(psi.reshape(shp[0],shp[1]*shp[2]),full_matrices=False); S /= np.linalg.norm(S)
            self.psi.tensors[i] = V.reshape(S.size,shp[1],shp[2])
            self.R_env[i] = contract.contract_right(self.psi.tensors[i], self.H.tensors[i], self.psi.tensors[i], self.R_env[i+1])
            if i != 0:
                self.psi.center -= 1
                self.psi.singular_values[i-1] = S
                C = oe.contract('ij,j->ij',U,S)
                psi = local_exponentiation_single_site('Hfree', C, self.L_env[i-1],' ', self.R_env[i], -delta, self.options['krydim'])
                self.psi.tensors[i-1] = oe.contract('ijk,kl->ijl',self.psi.tensors[i-1],psi.reshape(C.shape))
    
    ###########################
    #  Two sites tdvp sweeps  #
    ###########################
    def right_sweep_two_sites(self,delta):
        for i in range(self.L-1):
            assert self.psi.center == i
            shp1 = self.psi.tensors[i].shape; shp2 = self.psi.tensors[i+1].shape
            M12 = oe.contract('ijk,klm->ijlm', self.psi.tensors[i], self.psi.tensors[i+1])
            psi = local_exponentiation_two_sites('Heff_two', M12, self.L_env[i-1], self.H12[i],self.R_env[i+2], delta, self.options['krydim'])
            (U,S,V), err = svd_truncate(psi.reshape(shp1[0]*shp1[1],shp2[1]*shp2[2]),self.options)
            self.psi.singular_values[i] = S; self.psi.center += 1
            self.truncation_err[i] = max(err, self.truncation_err[i])
            self.psi.tensors[i] = U.reshape(shp1[0],shp1[1],S.size)
            self.psi.tensors[i+1] = oe.contract('i,ij->ij',S,V).reshape(S.size,shp2[1],shp2[2])
            if i != self.L-2:
                self.L_env[i] = contract.contract_left(self.psi.tensors[i], self.H.tensors[i], self.psi.tensors[i], self.L_env[i-1])
                shp2 = self.psi.tensors[i+1].shape
                self.psi.tensors[i+1] = local_exponentiation_two_sites('Heff_single', self.psi.tensors[i+1], self.L_env[i], self.H.tensors[i+1], self.R_env[i+2], -delta, self.options['krydim']).reshape(shp2)
    def left_sweep_two_sites(self,delta):
            for i in range(self.L-1,0,-1):
                assert self.psi.center == i
                shp1 = self.psi.tensors[i-1].shape; shp2 = self.psi.tensors[i].shape
                M12  = oe.contract('ijk,klm->ijlm', self.psi.tensors[i-1], self.psi.tensors[i]) 
                psi  = local_exponentiation_two_sites('Heff_two', M12, self.L_env[i-2], self.H12[i-1], self.R_env[i+1], delta, self.options['krydim'])
                (U,S,V), err = svd_truncate(psi.reshape(shp1[0]*shp1[1],shp2[1]*shp2[2]),self.options)
                self.psi.singular_values[i-1] = S; self.psi.center -= 1
                self.psi.tensors[i] = V.reshape(S.size,shp2[1],shp2[2])
                self.psi.tensors[i-1] = oe.contract('ij,j->ij',U,S).reshape(shp1[0],shp1[1],S.size)
                if i != 1:
                    self.R_env[i] = contract.contract_right(self.psi.tensors[i],self.H.tensors[i],self.psi.tensors[i],self.R_env[i+1])
                    shp1 = self.psi.tensors[i-1].shape
                    self.psi.tensors[i-1] = local_exponentiation_two_sites('Heff_single', self.psi.tensors[i-1], self.L_env[i-2], self.H.tensors[i-1], self.R_env[i], -delta, self.options['krydim']).reshape(shp1)
    def time_step(self,delta):
        if self.sites == 1:
            self.right_sweep_single_site(delta)
            self.left_sweep_single_site(delta)
        if self.sites == 2:
            self.right_sweep_two_sites(delta)
            self.left_sweep_two_sites(delta)
        self.psi.update_bonds_infos()
            
            
def apply_Heff_single_site(L,H,R,M):
    return oe.contract('adf,decg,beh,acb->fgh',L,H,R,M)
def apply_Hfree_single_site(L,R,C):
    return oe.contract('acd,bce,ab->de',L,R,C)
def local_exponentiation_single_site(method, M, L, H, R, delta, numiter):
    if   method == 'Heff_single':
        Afunc = lambda x: apply_Heff_single_site(L, H, R, x.reshape(M.shape)).ravel()
    elif method == 'Hfree':
        Afunc = lambda x: apply_Hfree_single_site(L, R, x.reshape(M.shape)).ravel()
    v = lanczos.expm_krylov_lanczos(Afunc, M.ravel(), -1j*delta/2, numiter=numiter)
    return v/np.linalg.norm(v)

def apply_Heff_two_sites(L,H12,R,M12):
    return oe.contract('ijk,jalmbc,nad,ilmn->kbcd',L,H12,R,M12)
def local_exponentiation_two_sites(method, M, L, H, R, delta, numiter):
    if   method == 'Heff_two':
        Afunc = lambda x: apply_Heff_two_sites(L, H, R, x.reshape(M.shape)).ravel()
    elif method == 'Heff_single':
        Afunc = lambda x: apply_Heff_single_site(L, H, R, x.reshape(M.shape)).ravel()
    v = lanczos.expm_krylov_lanczos(Afunc, M.ravel(), -1j*delta/2, numiter=numiter)
    return v/np.linalg.norm(v)