import numpy as np
import opt_einsum as oe
from ..tensors.MatrixProductState import MPS
from ..tensors.MatrixProductOperators import MPO
from ..tools import contract, lanczos
from ..tools.svd_truncate import svd_truncate

class DMRG:
    def __init__(self, H, options = {'chi_max':64,'trunc_cut':1e-10}, psi = None): 
        self.H = H
        self.L = H.L
        self.options = options
        self.psi = psi
        self.sites = options.get('sites', 1)
        if 'trunc_cut' not in self.options.keys(): self.options.update({'trunc_cut':1e-10})
        if 'chi_init' not in self.options.keys(): self.options.update({'chi_init':64})
        self.initialize()
        
    def initialize(self):
        if self.psi is None:
            if self.sites == 1: self.psi = MPS(self.L,self.H.d,chi=self.options['chi_max']) 
            if self.sites == 2: self.psi = MPS(self.L,self.H.d,chi=self.options['chi_init']) 
        self.psi.left_normalize()
        self.psi.right_normalize()
        
        self.L_env = [0]*(self.L+1)
        self.R_env = [0]*(self.L+1)
        
        self.L_env[-1] = np.ones((1,1,1))
        self.R_env[-1] = np.ones((1,1,1))
        
        for j in range(self.L-1,0,-1):
            self.R_env[j] = contract.contract_right(self.psi.tensors[j], self.H.tensors[j], self.psi.tensors[j], self.R_env[j+1])
        if self.sites == 2:
            self.H12 = [oe.contract('ijkl,jabc->iakblc',self.H.tensors[i],self.H.tensors[i+1]) for i in range(self.L-1)]
        
        Hsquared = [0]*self.L
        for i in range(self.L):
            shpH = self.H.tensors[i].shape
            Hsquared[i] = oe.contract('ijkl,mnlo->imjnko',self.H.tensors[i],self.H.tensors[i]).reshape(shpH[0]**2,shpH[1]**2,shpH[2],shpH[3])
        self.Hsquared = MPO(self.L,tensors=Hsquared)
    def check_convergence(self):
        try:
            self.energy_err = self.Hsquared.contractMPOtoMPS(self.psi).real-self.energy**2
        except:
            self.energy = self.H.contractMPOtoMPS(self.psi)
            self.energy_err = self.Hsquared.contractMPOtoMPS(self.psi).real-self.energy**2
        return self.energy_err
    
    ###########################
    # Single site dmrg sweeps #
    ###########################
    def right_sweep_single_site(self):
        for i in range(self.L-1):
            assert self.psi.center == i
            shp = self.psi.tensors[i].shape
            psi, e = local_minimization_single_site(self.psi.tensors[i], self.L_env[i-1], self.H.tensors[i], self.R_env[i+1])
            self.psi.tensors[i] = psi.reshape(shp); self.energy = e
            self.psi.move_center_one_step(i,direction='right')
            self.L_env[i] = contract.contract_left(self.psi.tensors[i], self.H.tensors[i], self.psi.tensors[i], self.L_env[i-1])
    def left_sweep_single_site(self):
        for i in range(self.L-1,0,-1):
            assert self.psi.center == i
            shp = self.psi.tensors[i].shape
            psi, e = local_minimization_single_site(self.psi.tensors[i], self.L_env[i-1], self.H.tensors[i], self.R_env[i+1])
            self.psi.tensors[i] = psi.reshape(shp); self.energy = e
            self.psi.move_center_one_step(i, direction='left')
            self.R_env[i] = contract.contract_right(self.psi.tensors[i], self.H.tensors[i], self.psi.tensors[i], self.R_env[i+1])
    ###########################
    #  Two sites dmrg sweeps  #
    ###########################    
    def right_sweep_two_sites(self):
        for i in range(self.L-1):
            assert self.psi.center == i
            M12  = oe.contract('ijk,klm->ijlm', self.psi.tensors[i], self.psi.tensors[i+1])
            shp1 = self.psi.tensors[i].shape; shp2 = self.psi.tensors[i+1].shape
            psi, e = local_minimization_two_sites(M12, self.L_env[i-1], self.H12[i], self.R_env[i+2])
            (U,S,V), err = svd_truncate(psi.reshape(shp1[0]*shp1[1],shp2[1]*shp2[2]),self.options)
            self.truncation_err[i] = max(err,self.truncation_err[i])
            self.energy = e
            self.psi.singular_values[i] = S
            
            self.psi.tensors[i]   = U.reshape(shp1[0],shp1[1],S.size)
            self.psi.tensors[i+1] = oe.contract('i,ij->ij', S, V).reshape(S.size,shp2[1],shp2[2])
            self.psi.center += 1
            
            self.L_env[i] = contract.contract_left(self.psi.tensors[i], self.H.tensors[i], self.psi.tensors[i], self.L_env[i-1])
    def left_sweep_two_sites(self):
        for i in range(self.L-1,0,-1):
            assert self.psi.center == i
            M12  = oe.contract('ijk,klm->ijlm', self.psi.tensors[i-1], self.psi.tensors[i])
            shp1 = self.psi.tensors[i-1].shape; shp2 = self.psi.tensors[i].shape
            
            psi, e = local_minimization_two_sites(M12, self.L_env[i-2], self.H12[i-1], self.R_env[i+1])
            (U,S,V), err = svd_truncate(psi.reshape(shp1[0]*shp1[1],shp2[1]*shp2[2]),self.options)
            self.truncation_err[i-1] = max(err,self.truncation_err[i-1])
            self.energy = e
            self.psi.singular_values[i-1] = S
            
            self.psi.tensors[i] = V.reshape(S.size,shp2[1],shp2[2])
            self.psi.tensors[i-1] = oe.contract('ij,j->ij',U,S).reshape(shp1[0],shp1[1],S.size)
            self.psi.center -= 1
            
            self.R_env[i] = contract.contract_right(self.psi.tensors[i], self.H.tensors[i], self.psi.tensors[i], self.R_env[i+1])
    
    def dmrg_step(self):
        if self.sites == 1:
            self.right_sweep_single_site()
            self.left_sweep_single_site()
        if self.sites == 2:
            self.truncation_err = np.zeros(self.L-1)
            self.right_sweep_two_sites()
            self.left_sweep_two_sites()
        self.psi.update_bonds_infos()
        
def apply_Heff_single_site(L,H,R,M):
    return oe.contract('adf,decg,beh,acb->fgh',L,H,R,M)
def local_minimization_single_site(M, L, H, R, numiter=10):
    Afunc = lambda x: apply_Heff_single_site(L, H, R, x.reshape(M.shape)).ravel()
    return lanczos.optimize_lanczos(Afunc, M.ravel(), numiter)

def apply_Heff_two_sites(L,H12,R,M12):
    return oe.contract('ijk,jalmbc,nad,ilmn->kbcd',L,H12,R,M12)
def local_minimization_two_sites(M12, L, H12, R, numiter=10):
    Afunc = lambda x: apply_Heff_two_sites(L, H12, R, x.reshape(M12.shape)).ravel()
    return lanczos.optimize_lanczos(Afunc, M12.ravel(), numiter)
