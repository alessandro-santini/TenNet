import numpy as np


def rho_beta(beta):
    rho_thermal = MPS_(64,2)
    
    sigma_x = np.array([[0,1],[1,0]])
    
    J = -1
    beta = 2
    beta *= J
    c_beta =  np.cosh(beta/2)/np.sqrt(2*np.cosh(beta))
    s_beta = -np.sinh(beta/2)/np.sqrt(2*np.cosh(beta))
    
    boundL = (np.array([np.eye(2),sigma_x])/np.sqrt(2)).reshape(1,2,2,2)
    bulk   = np.array([[c_beta*np.eye(2),c_beta * sigma_x],[s_beta*sigma_x,s_beta*np.eye(2)]]).reshape(2,2,2,2)
    boundR = np.array([c_beta*np.eye(2),s_beta*sigma_x]).reshape(2,2,2,1)
    
    rho_thermal.M[0] = boundL
    rho_thermal.M[-1] = boundR
    for i in range(1,rho_thermal.L-1):
        rho_thermal.M[i] = bulk
    rho_thermal.right_normalize()