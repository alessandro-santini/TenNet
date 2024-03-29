import numpy as np
import opt_einsum as oe
import copy
from . import IOhdf5

class DM_MPS:
    def __init__(self,L,d=4,chi=64,tensors=None,basis=None):
        self.L = L
        self.d = d
        self.center = -1
        self.initialize(tensors)
        self.singular_values = [0]*(self.L-1)
        self.set_basis(basis)
        
    def initialize(self,tensors):
        if tensors is None:
            chi_space = [1]+[self.chi]*(self.L-1)+[1]
            self.tensors = [np.random.rand(chi_space[i],self.d,chi_space[i+1]) for i in range(self.L)]
        else:
            self.tensors = copy.deepcopy(tensors)
        
    def set_basis(self, basis):
        if basis is None:
            self.basis = [0]*4
            self.basis[0] = np.eye(2)/np.sqrt(2)
            self.basis[1] = np.array([[0,1],[1,0]])/np.sqrt(2)
            self.basis[2] = np.array([[0,-1j],[1j,0]])/np.sqrt(2)
            self.basis[3] = np.array([[1,0],[0,-1]])/np.sqrt(2)
        else:
            self.basis = copy.deepcopy(basis)
    
    ###########################################################
    
    ###########################################################
    
    ###########################################################
    
    def move_center_one_step(self,i,direction='right',options={'decomp_way':'svd'}):
            assert (direction =='right') | (direction =='left')
            assert self.center == i
            
            shpi = self.tensors[i].shape
            if direction == 'right':
                self.center = i+1
                U, S, V = np.linalg.svd( self.tensors[i].reshape(shpi[0]*shpi[1], shpi[2]), full_matrices=False )
                S /= np.linalg.norm(S);
                self.tensors[i] = U.reshape(shpi[0], shpi[1], S.shape[0])
                self.tensors[i+1] = oe.contract('i,ij,jkl->ikl', S, V, self.tensors[i+1])
                self.singular_values[i] = S
            else:
                self.center = i-1
                U, S, V = np.linalg.svd( self.tensors[i].reshape(shpi[0],shpi[1]*shpi[2]),full_matrices=False ) 
                S /= np.linalg.norm(S);
                self.tensors[i] = V.reshape(S.shape[0],shpi[1],shpi[2])
                self.tensors[i-1] = oe.contract('ij,j,kli->klj',U,S,self.tensors[i-1])
                self.singular_values[i-1] = S
        
    def set_center(self,i):
        while self.center > i:
            ic = self.center
            self.move_center_one_step(ic, 'left')    
        while self.center < i:
            ic = self.center
            self.move_center_one_step(ic, 'right')
    
    def left_normalize(self):
        self.center = 0
        for i in range(self.L-1):
            self.move_center_one_step(i,direction='right')
        shp = self.tensors[-1].shape
        U, S, V = np.linalg.svd(self.tensors[-1].reshape(shp[0]*shp[1],shp[2]),full_matrices=False)
        S /= np.linalg.norm(S)
        self.tensors[-1] = oe.contract('ij,j,jk->ik',U,S,V).reshape(shp)
    def right_normalize(self):
        self.center=self.L-1
        for i in range(self.L-1,0,-1):
            self.move_center_one_step(i,direction='left')
        shp = self.tensors[0].shape
        U, S, V = np.linalg.svd(self.tensors[0].reshape(shp[0]*shp[1],shp[2]),full_matrices=False)
        S /= np.linalg.norm(S); 
        self.tensors[0] = oe.contract('ij,j,jk->ik',U,S,V).reshape(shp)
        
    def compute_normalization(self):
        L_env = np.ones((1,1))
        for i in range(self.L):
            L_env = oe.contract('ij,ikl,jkm->lm', L_env, self.tensors[i], self.tensors[i].conj())
        return L_env.item()
    
    def compute_entanglement_entropy(self):
        Sent = np.zeros(self.L-1, dtype=np.float64)
        for i,x in enumerate(self.singular_values):
            x = x[x>1e-15]
            Sent[i] = -np.sum(x**2*np.log(x**2))
        return Sent
    
    def update_bonds_infos(self):
        self.bonds_infos = [([x.shape[0],x.shape[1]],max([x.shape[0],x.shape[1]])) for x in self.tensors]
        
    ###########################################################
    
    ###########################################################
    
    ###########################################################
    
    def save(self, file_pointer, subgroup):
        IOhdf5.save_hdf5(self, file_pointer, subgroup)
    def load(self, file_pointer, subgroup):
        IOhdf5.load_hdf5(self, file_pointer, subgroup)