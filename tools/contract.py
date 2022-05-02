import opt_einsum as oe
import tensorflow as tf

def contract_right(A,W,B,R):
    return oe.contract('acb,decg,fgh,beh->adf',A,W,tf.math.conj(B),R)

def contract_left(A,W,B,L):
    return oe.contract('acb,decg,fgh,adf->beh',A,W,tf.math.conj(B),L)