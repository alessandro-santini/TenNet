import opt_einsum as oe
import copy
import numpy as np

def contract_right(A,W,B,R):
    return oe.contract('acb,decg,fgh,beh->adf',A,W,B.conj(),R)

def contract_left(A,W,B,L):
    return oe.contract('acb,decg,fgh,adf->beh',A,W,B.conj(),L)

def compute_local_operator(op, psi):
    psi_temp = copy.deepcopy(psi)
    if psi_temp.center != 0: psi_temp.right_normalize()
    op_avg = np.zeros(psi.L,complex)
    for i in range(psi_temp.L-1):
        op_avg[i] = oe.contract('ijk,jl,ilk', psi_temp.tensors[i], op, psi_temp.tensors[i].conj() )
        psi_temp.move_center_one_step(i,'right')
    op_avg[-1] = oe.contract('ijk,jl,ilk', psi_temp.tensors[-1], op, psi_temp.tensors[-1].conj() )
    return op_avg