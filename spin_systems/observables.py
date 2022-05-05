from ..tools.contract import compute_local_operator
import numpy as np

def local_mz(psi):
    sigma_z = np.array([[1.,0.],[0.,-1.]],complex)
    return compute_local_operator(sigma_z, psi).real
def local_mx(psi):
    sigma_x = np.array([[0.,1.],[1.,0.]],complex)
    return compute_local_operator(sigma_x, psi).real
def local_my(psi):
    sigma_y = np.array([[0.,-1j],[1j,0.]],complex)
    return compute_local_operator(sigma_y, psi).real

