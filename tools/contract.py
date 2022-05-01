import opt_einsum as oe

def contract_right(A,W,B,R):
    return oe.contract('acb,decg,fgh->adf',A,W,B,R)

def contract_left(A,W,B,L):
    return oe.contract('acb,decg,fgh->beh',A,W,B,L)