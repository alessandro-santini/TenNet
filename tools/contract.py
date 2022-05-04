import opt_einsum as oe

def contract_right(A,W,B,R):
    return oe.contract('acb,decg,fgh,beh->adf',A,W,B.conj(),R)

def contract_left(A,W,B,L):
    return oe.contract('acb,decg,fgh,adf->beh',A,W,B.conj(),L)