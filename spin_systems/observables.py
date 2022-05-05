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

def compute_corr_function(psi,i,j,opi,opj):
    psi_temp = copy.deepcopy(psi)
    if i > j: i, j = j, i;  opi, opj = opj, opi
    
    while psi_temp.center > i:
        ic = psi_temp.center
        psi_temp.move_center_one_step(ic, 'left')    
    while psi_temp.center < i:
        ic = psi_temp.center
        psi_temp.move_center_one_step(ic, 'right')

    corr_ = oe.contract('ijk,jl,ilm->mk', psi_temp.tensors[i], opi, psi_temp.tensors[i].conj())
    for x in range(i+1,j):
        corr_ = oe.contract('ij,ikl,jkm->lm',corr_, psi_temp.tensors[x],  psi_temp.tensors[x].conj() )
    return oe.contract('ij,ikl,km,jml',corr_, psi_temp.tensors[j], opj, psi_temp.tensors[j].conj() )