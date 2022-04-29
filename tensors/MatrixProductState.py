import tensorflow as tf
import copy

class MPS:
        def __init__(self, L, d=2, chi=64, tensors=None):
            self.L = L
            self.chi = chi
            self.d = d
            self.center = -1
            self.initialize(tensors)
            
        def initialize(self,tensors):
            if tensors is None:
                chi_space = [1]+[self.chi]*(self.L-1)+[1]
                self.tensors = [tf.random.uniform((chi_space[i],2,chi_space[i+1]),dtype=tf.float64) for i in range(self.L)]
            else:
                self.tensors = copy.deepcopy(tensors)
                
        def move_center_one_step(self,i,direction='right',options={'decomp_way':'svd'}):
            assert (direction =='right') | (direction =='left')
            assert self.center == i
            
            shpi = self.tensors[i].shape
            if direction == 'right':
                self.center = i+1
                S, U, V = tf.linalg.svd(tf.reshape(self.tensors[i], (shpi[0]*shpi[1], shpi[2])  )  )
                S /= tf.linalg.norm(S)
                self.tensors[i] = tf.reshape(U, (shpi[0], shpi[1], S.shape[0]))
                self.tensors[i+1] = tf.einsum('i,ij,jkl->ikl',S,tf.linalg.adjoint(V), self.tensors[i+1])
                
            else:
                self.center = i-1
                S, U, V = tf.linalg.svd( tf.reshape(self.tensors[i],(shpi[0],shpi[1]*shpi[2]) ) )
                S /= tf.linalg.norm(S)
                self.tensors[i] = tf.reshape(tf.linalg.adjoint(V),(S.shape[0],shpi[1],shpi[2]))
                self.tensors[i-1] = tf.einsum('ij,j,jkl->ikl',U,S,self.tensors[i-1])
                
        def left_normalize(self):
            if self.center == -1:
                shp0 = self.tensors[0].shape
                S, U, V = tf.linalg.svd( tf.reshape(self.tensors[0],(shp0[0]*shp0[1],shp0[2])))
                S /= tf.linalg.norm(S)
                self.tensors[0] = tf.reshape(tf.matmul(U, tf.matmul(tf.linalg.diag(S),V, adjoint_b=True)),shp0)
                self.center = 0
            for i in range(self.L-1):
                self.move_center_one_step(i,direction='right')
                
        def compute_normalization(self):
            L_env = tf.ones((1,1),dtype=tf.complex128)
            for i in range(self.L):
                1 ==0
                
            
psi = MPS(10,2,64)
psi.left_normalize()
