from ..tensors.MatrixProductOperators import MPO
import numpy as np
import opt_einsum as oe

def IsingChain(L, J=1., h_z=1., h_x=0.):    
    if isinstance(J,  (int,float)):   J = J*np.ones(L)
    if isinstance(h_x,(int,float)): h_x = h_x*np.ones(L)
    if isinstance(h_z,(int,float)): h_z = h_z*np.ones(L)
    
    tensors = []
    
    sigma_z = np.array([[1.,0.],[0.,-1.]])
    sigma_x = np.array([[0.,1.],[1.,0]])
    
    WL = np.zeros((1,3,2,2))
    WR = np.zeros((3,1,2,2))
    Wbulk = np.zeros((3,3,2,2))
    
    WL[0,0,:,:] = -h_x[0]*sigma_x-h_z[0]*sigma_z; WL[0,1,:,:] = -J[0]*sigma_x; WL[0,2,: ,:] = np.eye(2)
    WR[0,0,:,:] = np.eye(2); WR[1,0,:,:] = sigma_x; WR[2,0,:,:] = -h_x[-1]*sigma_x-h_z[-1]*sigma_z;
    tensors.append(WL)
    for i in range(1,L-1):
        Wbulk[0,0,:,:] = np.eye(2); Wbulk[1,0,:,:] = sigma_x; Wbulk[2,0,:,:] = -h_x[i]*sigma_x-h_z[i]*sigma_z
        Wbulk[2,1,:,:] = -J[i]*sigma_x; Wbulk[2,2,:,:] = np.eye(2)
        tensors.append(Wbulk)
    tensors.append(WR)
    return MPO(L, d=2, tensors = tensors)
    
def IsingChainWithAncilla(L, J=1., h_z=1., h_x=0.):
    tensors = [0]*L
    
    sigma_z = np.array([[1.,0.],[0.,-1.]])
    sigma_x = np.array([[0.,1.],[1.,0]])

    Zsys = np.kron(sigma_z,np.eye(2)); Xsys = np.kron(sigma_x,np.eye(2))
    Zanc = np.kron(np.eye(2),sigma_z); Xanc = np.kron(np.eye(2),sigma_x)
    
    Wbulk = np.zeros((4,4,4,4))
    
    Wbulk[0,0,:,:] = np.eye(4)
    Wbulk[1,0,:,:] = Xsys
    Wbulk[2,0,:,:] = Xanc
    Wbulk[3,0,:,:] = -h_x*(Xsys+Xanc)-h_z*(Zsys+Zanc)
    Wbulk[3,1,:,:] = -J*Xsys
    Wbulk[3,2,:,:] = -J*Xanc
    Wbulk[3,3,:,:] = np.eye(4)
    
    tensors[0]  = (Wbulk[-1,:,:,:].copy()).reshape(1,4,4,4)
    tensors[-1] =  Wbulk[:,0,:,:].copy().reshape(4,1,4,4)
    for i in range(1,L-1):
        tensors[i] = Wbulk
    return MPO(L, d=4, tensors=tensors)

def Hierarchical_Dyson_model(N,sigma,h,J=1):
    L = 2**N
    
    # Build coupling matrix
    real_states = [np.vectorize(np.binary_repr)(np.arange(2**N),N)][0]
    t = np.array([2.**( - (1 + sigma)*k ) for k in np.arange(0, N) ])
    A = np.zeros((L,L))
    for i, state_a in enumerate(real_states):
        for j, state_b in enumerate(real_states):
            if i != j :
                k = N
                while( state_a[:k] != state_b[:k] or k < 0 ):
                    k = k-1
                else:
                    A[i,j] = t[(N-k-1)]
    A = -J*A # Ferromagnetic coupling matrix
    
    sigma_z = np.array([[1.,0.],[0.,-1.]])
    sigma_x = np.array([[0.,1.],[1.,0]])
    
    tensors = [0]*L
    
    tensors[0] = np.zeros([1,L+1,2,2])
    tensors[0][0,0,:,:]  = -h*sigma_z
    tensors[0][0,-1,:,:] = np.eye(2)
    tensors[0][0,1:L,:,:] = oe.contract('i,jk->ijk',A[0,1:],sigma_x)
    
    tensors[-1] = np.zeros([3,1,2,2])
    tensors[-1][0,0,:,:] = np.eye(2)
    tensors[-1][1,0,:,:] = sigma_x
    tensors[-1][2,0,:,:] = -h*sigma_z
    
    for k in range(1,L-1):
        tensors[k] = np.zeros([L+2-k, L+1-k, 2, 2])
        tensors[k][0,0,:,:] = np.eye(2); tensors[k][-1,-1,:,:] = np.eye(2)
        tensors[k][1,0,:,:] = sigma_x; tensors[k][-1,0,:,:] = -h*sigma_z
        for i in range(1,L-k):
            tensors[k][i+1,i] = np.eye(2)
            if k+i<L:
                tensors[k][-1,i,:,:] = A[k,k+i]*sigma_x
    return MPO(L,tensors=tensors)


########################
###### 2D Systems ######
########################
def IsingMPO2D(J, h_z, h_x, config):
    N = config.size
    L = config.shape[0]
    
    if isinstance(h_x,(int,float)): h_x = h_x*np.ones(N)
    if isinstance(h_z,(int,float)): h_z = h_z*np.ones(N)
    
    def compute_dim_MPO(L):
        dim = []
        nearest = []
        for x in range(L*L):
            i,j = np.where(config == x)
            i = i[0]; j = j[0]
            nn = []
            if i !=0: nn.append(config[i-1,j])
            if j !=0: nn.append(config[i,j-1])
            if i !=L-1: nn.append(config[i+1,j])
            if j !=L-1: nn.append(config[i,j+1])
            nn = np.array(nn)
            nearest.append( np.array(nn[nn>config[i,j]]-config[i,j],int))
            new_dim = int(np.max(nn-config[i,j]) + 2)
            if x > 0:
                if new_dim < dim[x-1][1]:
                   dim.append((dim[x-1][1],dim[x-1][1]-1,2,2))
                else:
                    dim.append((dim[x-1][1],new_dim,2,2))
            else:
                dim.append((1,new_dim,2,2))
        dim[-1] = (dim[-2][1],1,2,2)
        return dim, nearest
    
    sigma_z = np.array([[1.,0.],[0.,-1.]])
    sigma_x = np.array([[0.,1.],[1.,0]])
    
    dimW, nn = compute_dim_MPO(L)
    
    tensors = [0]*N
    for i,dim in enumerate(dimW):
        tensors[i] = np.zeros(dim)
    
    tensors[0][0,0,:,:] = -h_z[0]*sigma_z -h_x[0]*sigma_x
    for k in nn[0]:
        tensors[0][0,k,:,:] = -J*sigma_x
    tensors[0][0,-1,:,:] = np.eye(2)
    
    tensors[-1][0,0,:,:] = np.eye(2)
    tensors[-1][1,0,:,:] = sigma_x
    tensors[-1][2,0,:,:] = -h_z[-1]*sigma_z - h_x[-1]*sigma_x
    
    for i in range(1,N-1):
        tensors[i][0,0,:,:]   = np.eye(2)
        tensors[i][-1,-1,:,:] = np.eye(2)
        tensors[i][-1,0,:,:]  = -h_z[i]*sigma_z - h_x[i]*sigma_x
        tensors[i][1,0,:,:]   = sigma_x
        for k in nn[i]:
            tensors[i][-1,k,:,:] = -J*sigma_x
        for k in range(2,tensors[i].shape[0]-1):
            tensors[i][k,k-1,:,:] = np.eye(2)
    H =  MPO(N,tensors=tensors)
    H.config = config.copy()
    return H