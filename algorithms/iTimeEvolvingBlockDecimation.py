import numpy as np
import opt_einsum as oe
from ..tensors.MatrixProductState import MPS
from ..tensors.MatrixProductOperators import MPO
from ..tools import contract, lanczos
from ..tools.svd_truncate import svd_truncate

import copy

class iTEBD:
    def __init__(self, A, Sa, B, Sb, U, options={}):
        """
        --Sa--A---Sb---B--Sa--
              |_______|
              | U(dt) |
              |͞ ͞ ͞ ͞ ͞ ͞ ͞ |
              |       |
        
        --Sb--B---Sa---A--Sb--
              |_______|
              | U(dt) |
              |͞ ͞ ͞ ͞ ͞ ͞ ͞ |
              |       |

        Parameters
        ----------
        A : np.ndarray
            first tensor.
        Sa : np.ndarray
            singular values.
        B : np.ndarray
            second tensor.
        Sb : np.ndarray
            singular values.
        U : np.ndarray
            two-body potential.
        options : dict, optional
            svd_truncate options. The default is {}.
            
        """
        self.A, self.B= A.copy(), B.copy()
        self.Sa, self.Sb = Sa.copy(), Sb.copy()
        self.U = U
        self.options = options
        if 'trunc_cut' not in self.options.keys(): self.options.update({'trunc_cut':1e-10})
        if 'chi_max'   not in self.options.keys(): self.options.update({'chi_max':64})
        if 'krydim'    not in self.options.keys(): self.options.update({'krydim':10})
        self.truncation_err = 0.
        
    def time_step(self, dt):
        for _ in range(2):
            shpA, shpB = self.A.shape, self.B.shape
            theta = oe.contract('a,acd,d,dfg,g,cflm->almg', self.Sa,self.A,self.Sb,self.B,self.Sa,self.U)
            (U,S,V),err = svd_truncate(theta.reshape(shpA[0]*shpA[1],shpB[1]*shpB[2]), self.options)
            self.truncation_err = max(err, self.truncation_err)
            self.Sb = S
            U = U.reshape(shpA[0],shpA[1],S.size)
            self.B = V.reshape(S.size,shpB[1],shpB[2])
            self.A = oe.contract('j,jkl->jkl', 1./self.Sa, U)
            self.A,self.Sa,self.B,self.Sb = self.B,self.Sb,self.A,self.Sa