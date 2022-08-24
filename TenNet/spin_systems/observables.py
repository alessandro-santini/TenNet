from ..tools.contract import compute_local_operator
import numpy as np
import copy
import opt_einsum as oe

def local_mz(psi):
    sigma_z = np.array([[1.,0.],[0.,-1.]], complex)
    return compute_local_operator(sigma_z, psi).real
def local_mx(psi):
    sigma_x = np.array([[0.,1.],[1.,0.]], complex)
    return compute_local_operator(sigma_x, psi).real
def local_my(psi):
    sigma_y = np.array([[0.,-1j],[1j,0.]], complex)
    return compute_local_operator(sigma_y, psi).real

def compute_corr_function(psi, i, j, opi, opj):
    psi_temp = copy.deepcopy(psi)
    if i > j: i, j = j, i;  opi, opj = opj, opi
    psi_temp.set_center(i)
    corr_ = oe.contract('ijk,jl,ilm->mk', psi_temp.tensors[i], opi, psi_temp.tensors[i].conj())
    for x in range(i+1,j):
        corr_ = oe.contract('ij,ikl,jkm->lm',corr_, psi_temp.tensors[x],  psi_temp.tensors[x].conj() )
    return oe.contract('ij,ikl,km,jml',corr_, psi_temp.tensors[j], opj, psi_temp.tensors[j].conj() ).item()

def compute_corr_matrix(psi, opi, opj):
    psi_temp = copy.deepcopy(psi)
    L = psi_temp.L
    corr_matrix = np.zeros((L, L),complex)
     
    for i in range(L-1):
        psi_temp.set_center(i)
        corr_i = oe.contract('ijk,jl,ilm->km', psi_temp.tensors[i],opi,psi_temp.tensors[i].conj() )
        for j in range(i+1, L):
            corr_matrix[i,j] = oe.contract('im,ikj,kl,mlj',corr_i, psi_temp.tensors[j], opj, psi_temp.tensors[j].conj() )
            corr_i = oe.contract('im,ikj,mkl->jl',corr_i, psi_temp.tensors[j], psi_temp.tensors[j].conj() )
    
    if np.allclose(opi, opj):
       corr_matrix += np.diag( compute_local_operator(opi@opi, psi_temp) )
    return (corr_matrix+np.triu(corr_matrix,k=1).T).real