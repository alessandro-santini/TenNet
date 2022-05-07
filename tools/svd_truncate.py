import numpy as np
from scipy.linalg import svd as scipy_svd

def svd_truncate(A, options={}):
    """
    Parameters
    ----------
    A : Tensor
        two legged M x N tensor.
    options : dict, optional
       chi_min: keeps at least chi_min Schmidt Values
       chi_max: keeps at most  chi_max Schmidt Values
       svd_min: discard S[i] < svd_min
       trunc_cut: discard all the S as long as \sum_j S[j]**2 < trunc_cut.
    Returns
    -------
    U : Tensor
        shape, M x new_chi.
    S : Tensor
        shape, new_chi.
    V : Tensor
        shape, new_chi x N.
    """
    try:
        U, S, V = np.linalg.svd(A,full_matrices=False)
    except:
        U, S, V = scipy_svd(A,full_matrices=False,lapack_driver='gesvd')
    S = S/np.linalg.norm(S)
    S, err  = truncate(S, options)
    chi_new = S.size
    U = U[:,:chi_new]; V = V[:chi_new,:]
    return (U, S, V), err    
   
def truncate(S, options={}):
    """
    Parameters
    ----------
    S : Tensor
        singular values, should be normalized np.sum(S*S) = 1.
    options : dict, optional
       chi_min: keeps at least chi_min Schmidt Values
       chi_max: keeps at most  chi_max Schmidt Values
       svd_min: discard S[i] < svd_min
       trunc_cut: discard all the S as long as \sum_j S[j]**2 < trunc_cut.
    Returns
    -------
    Snew : Tensor
        truncated and normalized to 1 singular values.
    truncation_error : float
        error after the truncation \sum_k Snew[k]**2.
    """
    chi_min = options.get('chi_min', 1)
    chi_max = options.get('chi_max', 64)
    svd_min = options.get('svd_min', 1e-14)
    trunc_cut = options.get('trunc_cut', 1e-14)
    
    indices = np.ones(len(S),dtype=bool)
    # Keep at most chi_max
    if chi_max is not None:
        indices1 = np.ones(len(S),dtype=bool)
        indices1[chi_max:] = False
        indices = np.logical_and(indices,indices1)
    # Discard small singular values
    indices = np.logical_and(indices, S >= svd_min)
    # Truncation Error
    indices = np.logical_and(indices, np.flipud(np.cumsum(np.flipud(S)**2) >= trunc_cut))
    # Keeps at least chi_min
    indices[:chi_min] = True

    Snew = S[indices]
    truncation_error = 1.-np.sum(Snew**2)
    Snew /= np.linalg.norm(Snew)
    return Snew, truncation_error.item()