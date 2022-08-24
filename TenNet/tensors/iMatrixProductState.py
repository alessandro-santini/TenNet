import numpy as np
import opt_einsum as oe
from . import IOhdf5
from scipy.sparse.linalg import eigs

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
    
    def compute_transfer_matrix(self):
        shp1 = self.B1.shape;
        T   = oe.contract('abe,ehf,cbd,dhg->acfg',self.B1,self.B2,self.B1.conj(),self.B2.conj())
        self.Tnot_reshaped = T.copy()
        self.T = T.reshape(shp1[0]**2,shp1[0]**2)    
        
    def compute_corr_length(self):
        self.compute_transfer_matrix()
        eig =np.sort(  np.abs(eigs(self.T, k=6, ncv=300, which='LM',return_eigenvectors=False)) )[::-1]
        self.Teig = eig
        return (-2./np.log(eig[1]),-2./np.log(eig[2]))
        
    def compute_long_distance_observable_degenerate(self,op):
        self.compute_transfer_matrix()
        shp1 = self.B1.shape
        eig, w = eigs(self.T, k=3, ncv=100, which='LR')
        self.Teigs, self.Tw = eig, w
        
        c = np.eye(shp1[0]).ravel(); c/=np.linalg.norm(c)
        w = np.vdot(c,w[:,1])*w[:,0]-np.vdot(c,w[:,0])*w[:,1]
        w = (w/np.linalg.norm(w)).reshape(shp1[0],shp1[0])
        w=c.reshape(shp1[0],shp1[0])
        return np.abs(np.real_if_close(oe.contract('i,jk,ijl,ikm,lno,mnp,op',self.Sv**2,op,self.B1,self.B1.conj(),self.B2,self.B2.conj(),w).item()))
        
    def compute_long_distance_observable(self,op):
        self.compute_transfer_matrix()
        eig, w = eigs(self.T, which='LR')
        self.Teigs,self.Tw = eig, w
        shp1 = self.B1.shape
        r_eigenvector = w[:,0].reshape(shp1[0],shp1[0])
        return np.real_if_close( oe.contract('i,ijk,ilm,kab,mac,jl,bc',self.Sv**2,self.B1,self.B1.conj(),self.B2,self.B2.conj(),op,r_eigenvector).item() )
    
    def save(self, file_pointer, subgroup):
        self.set_tensors()
        IOhdf5.save_hdf5(self, file_pointer, subgroup)
    def load(self, file_pointer, subgroup):
        self.set_tensors()
        IOhdf5.load_hdf5(self, file_pointer, subgroup)