import tensorflow as tf
import opt_einsum as oe
from ..tensors.MatrixProductState import MPS
from ..tools import contract, lanczos

class DMRG:
    def __init__(self, H, options = {}, psi = None):
        self.H = H
        self.L = H.L
        self.options = options
        self.chi_MAX = options.get('chi_max', 64)
        self.psi = psi
        self.sites = options.get('sites', 1)
        self.initialize()
        
    def initialize(self):
        if self.psi is None:
            self.psi = MPS(self.L,self.H.d,chi=self.chi_MAX) 
        self.psi.left_normalize()
        self.psi.right_normalize()
        
        self.L_env = [0]*(self.L+1)
        self.R_env = [0]*(self.L+1)
        
        self.L_env[-1] = tf.ones((1,1,1), dtype=tf.float64)
        self.R_env[-1] = tf.ones((1,1,1), dtype=tf.float64)
        
        for j in range(self.L-1,0,-1):
            self.R_env[j] = contract.contract_right(self.psi.tensors[j], self.H.tensors[j], self.psi.tensors[j], self.R_env[j+1])
        
    def right_sweep_single_site(self):
        for i in range(self.L-1):
            assert self.psi.center == i
            shp = self.psi.tensors[i].shape
            psi, e = local_minimization_single_site(self.psi.tensors[i], self.L_env[i-1], self.H.tensors[i], self.R_env[i+1])
            self.psi.tensors[i] = tf.reshape(psi, shp); self.energy = e
            self.psi.move_center_one_step(i,direction='right')
            self.L_env[i] = contract.contract_left(self.psi.tensors[i], self.H.tensors[i], self.psi.tensors[i], self.L_env[i-1])
    def left_sweep_single_site(self):
        for i in range(self.L-1,0,-1):
            assert self.psi.center == i
            shp = self.psi.tensors[i].shape
            psi, e = local_minimization_single_site(self.psi.tensors[i], self.L_env[i-1], self.H.tensors[i], self.R_env[i+1])
            self.psi.tensors[i] = tf.reshape(psi, shp); self.energy = e
            self.psi.move_center_one_step(i, direction='left')
            self.R_env[i] = contract.contract_right(self.psi.tensors[i], self.H.tensors[i], self.psi.tensors[i], self.R_env[i+1])
    def dmrg_step(self):
        if self.sites == 1:
            self.right_sweep_single_site()
            self.left_sweep_single_site()
        
def apply_Heff_single_site(L,H,R,M):
    return oe.contract('adf,decg,beh,acb->fgh',L,H,R,M)
def local_minimization_single_site(M, L, H, R, Lsteps=10):
    Afunc = lambda x: tf.reshape(apply_Heff_single_site(L, H, R, tf.reshape(x,M.shape)),[-1]).numpy()
    return lanczos.optimize_lanczos(Afunc, tf.reshape(M,[-1]), Lsteps)