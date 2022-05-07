import numpy as np
from ..tensors.MatrixProductState import MPS

def Ising_ThermalState(L, K):
    """
    Parameters
    ----------
    L : int 
        size of the chain.
    K : float
        \beta*J.

    Returns
    -------
    rho_termal : MPDO
        classicla Ising thermal distribution.
    """
    tensors = [0]*L
    
    sigma_x = np.array([[0,1],[1,0]])
    with np.errstate(over='ignore'):
        c_K =  np.cosh(K/2)/np.sqrt(2*np.cosh(K))
        s_K = -np.sinh(K/2)/np.sqrt(2*np.cosh(K))
    if np.isnan(c_K): c_K = 1./2
    if np.isnan(s_K): s_K = -1./2*np.sign(K)
    boundL = (np.array([np.eye(2),sigma_x])/np.sqrt(2)).reshape(1,2,2,2)
    bulk   = np.array([[c_K*np.eye(2), c_K * sigma_x],[s_K*sigma_x,s_K*np.eye(2)]]).reshape(2,2,2,2)
    boundR = np.array( [c_K*np.eye(2),  s_K*sigma_x]).reshape(2,2,2,1)
    
    tensors[0] = boundL
    tensors[-1] = boundR
    for i in range(1,L-1):
        tensors[i] = bulk
    for i in range(L):
        shp = tensors[i].shape
        tensors[i] = tensors[i].reshape(shp[0],shp[1]*shp[2],shp[3])
    rho_t = MPS(L,d=4,tensors=tensors)
    rho_t.right_normalize()
    return rho_t