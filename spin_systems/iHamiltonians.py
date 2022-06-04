import numpy as np

def IsingChainWithAncilla(J=1., h_z=0.25, h_x=0.):
    sz = np.array([[1,0],[0,-1]])
    sx = np.array([[0,1],[1,0]])
    Id2 = np.eye(2)
    Id4 = np.eye(4)
    
    sz_sys = np.kron(sz,Id2)
    sx_sys = np.kron(sx,Id2)
    
    sz_anc = np.kron(Id2,sz)
    sx_anc = np.kron(Id2,sz)

    H  = -J*(np.kron(sx_sys,sx_sys))-h_z*(np.kron(sz_sys,Id4)+np.kron(Id4,sz_sys))-h_x*(np.kron(sx_sys,Id4)+np.kron(Id4,sx_sys))
    H  -= -J*(np.kron(sx_anc,sx_anc))-h_z*(np.kron(sz_anc,Id4)+np.kron(Id4,sz_anc))-h_x*(np.kron(sx_anc,Id4)+np.kron(Id4,sx_anc))
    
    return H