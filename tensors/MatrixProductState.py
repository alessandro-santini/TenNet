import tensorflow as tf
import numpy as np
import opt_einsum as oe
import copy

class MPS:
        def __init__(self, L, d=2, chi=64, tensors=None, dtype = tf.float64):
            self.L = L
            self.chi = chi
            self.d = d
            self.center = -1
            self.dtype = dtype
            self.initialize(tensors)
            self.singular_values = [0]*(self.L-1)
            
        def initialize(self,tensors):
            if tensors is None:
                chi_space = [1]+[self.chi]*(self.L-1)+[1]
                if self.dtype == tf.float64:
                    self.tensors = [tf.random.uniform((chi_space[i],2,chi_space[i+1]),dtype=tf.float64) for i in range(self.L)]
                elif self.dtype == tf.complex128:
                    self.tensors = [tf.complex(tf.random.uniform((chi_space[i],2,chi_space[i+1]),dtype=tf.float64),
                                               tf.random.uniform((chi_space[i],2,chi_space[i+1]),dtype=tf.float64))
                                               for i in range(self.L)]
            else:
                self.tensors = copy.deepcopy(tensors)
                
        def move_center_one_step(self,i,direction='right',options={'decomp_way':'svd'}):
            assert (direction =='right') | (direction =='left')
            assert self.center == i
            
            shpi = self.tensors[i].shape
            if direction == 'right':
                self.center = i+1
                S, U, V = tf.linalg.svd(tf.reshape(self.tensors[i], (shpi[0]*shpi[1], shpi[2])  )  )
                S /= tf.linalg.norm(S); S = tf.cast(S,self.dtype)
                self.tensors[i] = tf.reshape(U, (shpi[0], shpi[1], S.shape[0]))
                self.tensors[i+1] = oe.contract('i,ij,jkl->ikl',S,tf.linalg.adjoint(V), self.tensors[i+1])
                self.singular_values[i] = S
            else:
                self.center = i-1
                S, U, V = tf.linalg.svd( tf.reshape(self.tensors[i],(shpi[0],shpi[1]*shpi[2]) ) )
                S /= tf.linalg.norm(S); S = tf.cast(S,self.dtype)
                self.tensors[i] = tf.reshape(tf.linalg.adjoint(V),(S.shape[0],shpi[1],shpi[2]))
                self.tensors[i-1] = oe.contract('ij,j,kli->klj',U,S,self.tensors[i-1])
                self.singular_values[i-1] = S
                
        def left_normalize(self):
            self.center = 0
            for i in range(self.L-1):
                self.move_center_one_step(i,direction='right')
            shp = self.tensors[-1].shape
            S, U, V = tf.linalg.svd( tf.reshape(self.tensors[-1],(shp[0]*shp[1],shp[2])))
            S /= tf.linalg.norm(S); S = tf.cast(S,self.dtype)
            self.tensors[-1] = tf.reshape(tf.matmul(U, tf.matmul(tf.linalg.diag(S),V, adjoint_b=True)),shp)
        def right_normalize(self):
            self.center=self.L-1
            for i in range(self.L-1,0,-1):
                self.move_center_one_step(i,direction='left')
            shp = self.tensors[0].shape
            S, U, V = tf.linalg.svd( tf.reshape(self.tensors[0],(shp[0]*shp[1],shp[2])))
            S /= tf.linalg.norm(S); S = tf.cast(S,self.dtype)
            self.tensors[0] = tf.reshape(tf.matmul(U, tf.matmul(tf.linalg.diag(S),V, adjoint_b=True)),shp)
            
        def compute_normalization(self):
            L_env = tf.ones((1,1), dtype=self.dtype)
            for i in range(self.L):
                L_env = oe.contract('ij,ikl,jkm->lm', L_env,self.tensors[i],tf.math.conj(self.tensors[i]))
            return L_env.numpy().item()
        
        def compute_entanglement_entropy(self):
            Sent = np.zeros(self.L-1,dtype=np.float64)
            for i,x in enumerate(self.singular_values):
                Sent[i] = -tf.math.reduce_sum(x**2*tf.math.log(x**2)).numpy()
            return Sent
        
psi = MPS(100,2,64)
psi.right_normalize()


