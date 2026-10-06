import numpy as np
import scipy
# import jax
# jax.config.update("jax_enable_x64", True)
# from jax import numpy as jnp
from functools import partial
import sparse

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# @partial(jax.jit, static_argnums=(1,))
def shuffle_forward(x, n):
    reshaped_x = x.reshape(( *tuple( n*[2] ), *tuple( n*[2] )))
    order = np.arange(2*n).reshape(2, n)
    order = order.reshape((1, -1), order= 'F')[0]
    reshaped_x = reshaped_x.transpose(order)

    return reshaped_x

# import torch

# def shuffle_forward_torch(x, n):
#     reshaped_x = x.reshape((*((2,) * n), *((2,) * n)))  
#     order = torch.arange(2 * n).reshape(2, n)  
#     order = order.reshape((1, -1), order='F').squeeze(0)  
#     reshaped_x = reshaped_x.permute(*order)  
#     return reshaped_x

# @partial(jax.jit, static_argnums=(1,))
def shuffle_adjoint(x, n):
    reshaped_x = x.reshape(( *tuple( n*[2] ), *tuple( n*[2] )))
    order = np.concatenate((np.arange(0, 2*n, 2), np.arange(1, 2*n+1, 2)))
    reshaped_x = reshaped_x.transpose(order)
    reshaped_x = reshaped_x.reshape(2**n, 2**n)

    return reshaped_x

# def shuffle_adjoint_torch(x, n):
#     reshaped_x = x.reshape((*((2,) * n), *((2,) * n)))  
#     order = torch.cat((torch.arange(0, 2 * n, 2), torch.arange(1, 2 * n + 1, 2)))  
#     reshaped_x = reshaped_x.permute(*order)  
#     reshaped_x = reshaped_x.reshape(2**n, 2**n)  

#     return reshaped_x

# @partial(jax.jit, static_argnums=(1,))
def qst1(x, n):
    x = shuffle_forward(x, n)
    I = np.eye(2)
    X = np.array( [ [0,1],[1,0]] )
    Y = np.array( [[ 0,-1j], [1j, 0] ] )
    Z = np.array( [[1,0],[0,-1]] )
    paulis = np.vstack([I.ravel(), X.ravel(), Y.ravel(), Z.ravel()]).T

    for i in range(n):
        x = x.reshape((4, -1))
        x = paulis.conj().T@x 
        x = x.T
    return x.reshape(-1, )


# def qst1_torch(x, n):

#     x = torch.tensor(x, dtype=torch.complex64, device=device)
#     # Convert the input `x` to a PyTorch tensor
#     x = shuffle_forward_torch(x, n)

#     # Define the Pauli matrices as PyTorch tensors
#     I = torch.eye(2, dtype=torch.complex64)
#     X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64)
#     Y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64)
#     Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64)
    
#     paulis = torch.stack([I.ravel(), X.ravel(), Y.ravel(), Z.ravel()], dim=1)

#     for i in range(n):
#         x = x.view(4, -1)
#         x = torch.matmul(paulis.conj().T, x)  # Conjugate transpose of `paulis` and matmul with `x`
#         x = x.T

#     return x.view(-1).cpu().numpy()



# @partial(jax.jit, static_argnums=(1,))
def qst2(x, n):

    I = np.eye(2)
    X = np.array( [ [0,1],[1,0]] )
    Y = np.array( [[ 0,-1j], [1j, 0] ] )
    Z = np.array( [[1,0],[0,-1]] )
    paulis = np.vstack([I.ravel(), X.ravel(), Y.ravel(), Z.ravel()]).T

    for i in range(n):
        x = x.reshape(4, -1)
        x = paulis@x 
        x = x.T

    x = shuffle_adjoint(x, n)
    x = 0.5*(x+x.conj().T)   

    return x


# def qst2_torch(x, n):
#     # Convert the input `x` to a PyTorch tensor
#     x = torch.tensor(x, dtype=torch.complex64, device=device)

#     # Define the Pauli matrices as PyTorch tensors
#     I = torch.eye(2, dtype=torch.complex64)
#     X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64)
#     Y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64)
#     Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64)
    
#     paulis = torch.stack([I.ravel(), X.ravel(), Y.ravel(), Z.ravel()], dim=1)

#     for i in range(n):
#         x = x.view(4, -1)
#         x = torch.matmul(paulis, x)  # Matrix multiplication with `paulis`
#         x = x.T

#     # Assuming `shuffle_adjoint` is a function you have, which applies some operation to `x`
#     x = shuffle_adjoint_torch(x, n)

#     # Apply conjugate transpose operation
#     x = 0.5 * (x + x.conj().T)

#     return x.cpu().p()



def prim(u, A):
    return A@u

def fun(u, nQubits, y, primitive1 = None, primitive2 = None, rank = None, A = None, lowmem = None, mList=None):
    m = 4**nQubits - 1

    if u.shape[0] == 2**nQubits and u.shape[1] == 2**nQubits:
        f = primitive1(u, nQubits)
        fplus = 0.5*(f[0]+f[1:])
        fminus = 0.5*(f[0]-f[1:])
        
 
        val = -1*np.sum(y[0:4**nQubits-1]*np.log(fplus))-1*np.sum(y[4**nQubits-1:]*np.log(fminus)) 

    else:
        tau = 2*np.sum(y)
        idx = int(u.size/2)
        u = u[0:idx].reshape((2**nQubits, rank), order= 'C') + 1j*u[idx:].reshape((2**nQubits, rank), order = 'C')
    
        f = primitive1(u@u.conj().T, nQubits)
        fplus = 0.5*(f[0]+f[1:])
        fminus = 0.5*(f[0]-f[1:])
 
        val = -1*np.sum(y[0:4**nQubits-1]*np.log(fplus))-1*np.sum(y[4**nQubits-1:]*np.log(fminus)) + 0.5*tau*np.linalg.norm(u,  'fro')**2

    return val
        

def grad(u, nQubits, y, primitive1 = None, primitive2 = None, rank = None, A = None, lowmem = None, mList=None):
    m = 4**nQubits - 1
    
    if u.shape[0] == 2**nQubits and u.shape[1] == 2**nQubits:
        f = primitive1(u, nQubits)
        fplus = 0.5*(f[0]+f[1:])
        fminus = 0.5*(f[0]-f[1:])
        
        fplus = np.concatenate((np.array([0]), y[0:4**(nQubits)-1]/fplus), axis = 0)
        fminus = np.concatenate((np.array([0]), y[4**(nQubits)-1:]/fminus), axis = 0)
        
        der = -1* 0.5*(primitive2(fplus, nQubits) + (np.sum(fplus))*np.eye(2**nQubits) -1*primitive2(fminus,nQubits) + (np.sum(fminus))*np.eye(2**nQubits) )
        
    else:
        tau = 2*np.sum(y)

        idx = int(u.size/2)
        u = u[0:idx].reshape((2**nQubits, rank), order= 'C') + 1j*u[idx:].reshape((2**nQubits, rank), order = 'C')
   
        f = primitive1(u@u.conj().T, nQubits)
        fplus = 0.5*(f[0]+f[1:])
        fminus = 0.5*(f[0]-f[1:])

        fplus = np.concatenate((np.array([0]), y[0:4**nQubits-1]/fplus), axis = 0)
        fminus = np.concatenate((np.array([0]), y[4**nQubits-1:]/fminus), axis = 0)

        der = -2* 0.5*(primitive2(fplus, nQubits) + (np.sum(fplus))*np.eye(2**nQubits)- primitive2(fminus,nQubits) + (np.sum(fminus))*np.eye(2**nQubits) )@u+ tau*u
        der = np.vstack((np.real(der).reshape((rank*2**nQubits,1), order='C') , np.imag(der).reshape((rank*2**nQubits,1), order='C')))
   
    return der


def fidelity(sigma, rho):

    S = scipy.linalg.sqrtm(sigma)

    return np.real((np.trace(scipy.linalg.sqrtm(S @ rho @ S))) ** 2)

    # return None



def pX(U):
    """ apply Pauli X operator, that is, 
    U = [ [U1], [U2]]  -->  [ [U2], [U1] ]
    since sigma_X = [ [0,1],[1,0]]
    """
    return np.array( [U[1], U[0]] )

def pY(U):
    """ apply Pauli Y operator, that is, 
    U = [ [U1], [U2]]  -->  [ -1j*[U2], 1j*[U1] ]
    since sigma_Y = [ [0,-1j],[1j,0]]
    """
    return np.array( [-1j*U[1], 1j*U[0]] )

def pZ(U):
    """ apply Pauli Z operator, that is, 
    U = [ [U1], [U2]]  -->  [ [U1], -[U2] ]
    since sigma_Z = [ [1,0],[0,-1]]
    """
    return np.array( [U[0], -U[1]] )
def pI(U):
    """ identity operator """
    return U


# PauliFcn = {0:pI, 1:pZ, 2:pX, 3:pY}

PauliFcn = {0:pI, 1:pX, 2:pY, 3:pZ}    


def int2lst(j, nQubits ):
    """ input: 0 <= j < 4^nQubits,
    where j represents one of the measurements.
    We'll take the 4-nary representation of j
    and return it as a list of length nQubits
    Ex: j = 19 and nQubits = 4 returns [0, 0, 1, 0, 3]
    """
    pad = nQubits-np.floor(np.log(np.maximum(j,.3))/np.log(4))-1
    lst = np.base_repr( j, base=4, padding=int(pad) )
    return [int(i) for i in lst ]


def lowmemA(u,nQubits, mList, rank, y=None):
    """
    y = lowmemA( U, mList)
        returns y = A( UU^* )
        i.e. y[i] = tr( A_i UU^* )
        where mList is a list of Pauli operators to specify i

        In detail, ... 
    """
    print(u.shape)
    if u.ndim == 1: u = u.reshape( (-1,1) )  
    m = len(mList)
    f = np.zeros( (m,1),dtype = 'complex_' )
    # nQubits = int( np.log2( u.shape[0] ) )
    d = nQubits # alias
    idx = int(u.size/2)
    u = u[0:idx].reshape((2**nQubits, rank), order= 'C') + 1j*u[idx:].reshape((2**nQubits, rank), order = 'C')
    # r = u.shape[1]
    r = rank

    # We'll be doing y_i = trace( A_i uu^* ) = trace( u^* A_i u)
    #   so then V u <-- A_i u, so  = trace( u^* v) = vdot(u,v)

    axs = np.arange(d)
    newAxs = np.roll( axs, 1 )

    if r==1:
        U = u.reshape( tuple( d*[2] ) ) # not sure if U and u are separate objectors or not...
        # todo: check np.may_share_memory(U,u) to see
    
    else:
        U = u.reshape( ( *tuple( d*[2] ), r ) )
        axs = np.hstack( [axs,d] ) # when u is a matrix with r > 1 columns
        newAxs = np.hstack( [newAxs,d] )    
    gradVal = np.zeros((U.shape), dtype=np.complex64)
    fVal = 0

    for i, meas in enumerate([0]):
        V0 = U.copy()
        for p in int2lst(meas, nQubits ):
            V0 = PauliFcn[p]( V0 )
            V0 = np.moveaxis( V0 , axs, newAxs )
        f0 = np.vdot( u, V0)

    for i, meas in enumerate(mList):
        V = U.copy()
        for p in int2lst(meas, nQubits ):
            V = PauliFcn[p]( V )
            V = np.moveaxis( V , axs, newAxs )
        f[i] = np.vdot( u, V) # V_i is there

        if y is not None:  
            gradVal = gradVal + y[i]*(U + V)/(f0+f[i]) + y[i+m]*(U - V)/(f0-f[i])

    if y is not None:
        gradVal = -2*gradVal  + 2*np.sum(y)*U  
        gradVal = np.vstack((np.real(gradVal).reshape((rank*2**nQubits,1), order='C') , np.imag(gradVal).reshape((rank*2**nQubits,1), order='C')))

        f = f.ravel()
        fplus = 0.5*(f0+f)
        fminus = 0.5*(f0-f)
    
        fVal = -1*np.sum(y[0:m]*np.log(fplus))-1*np.sum(y[m:]*np.log(fminus)) + 0.5*2*np.sum(y)*np.linalg.norm(u,  'fro')**2
    
        return fVal, gradVal
    
    return np.vstack((f0, f))

def true_prob(u,mList):
    """
    y = lowmemA( U, mList)
        returns y = A( UU^* )
        i.e. y[i] = tr( A_i UU^* )
        where mList is a list of Pauli operators to specify i

        In detail, ... 
    """
    if u.ndim == 1: u = u.reshape( (-1,1) )  
    m = len(mList)
    f = np.zeros( (m,1),dtype = 'complex_' )
    nQubits = int( np.log2( u.shape[0] ) )
    d = nQubits # alias
    r = u.shape[1]

    # We'll be doing y_i = trace( A_i uu^* ) = trace( u^* A_i u)
    #   so then V u <-- A_i u, so  = trace( u^* v) = vdot(u,v)

    axs = np.arange(d)
    newAxs = np.roll( axs, 1 )

    if r==1:
        U = u.reshape( tuple( d*[2] ) ) # not sure if U and u are separate objectors or not...
        # todo: check np.may_share_memory(U,u) to see
    
    else:
        U = u.reshape( ( *tuple( d*[2] ), r ) )
        axs = np.hstack( [axs,d] ) # when u is a matrix with r > 1 columns
        newAxs = np.hstack( [newAxs,d] )    

    for i, meas in enumerate([0]):
        V0 = U.copy()
        for p in int2lst(meas, nQubits ):
            V0 = PauliFcn[p]( V0 )
            V0 = np.moveaxis( V0 , axs, newAxs )
        f0 = np.vdot( u, V0)

    for i, meas in enumerate(mList):
        V = U.copy()
        for p in int2lst(meas, nQubits ):
            V = PauliFcn[p]( V )
            V = np.moveaxis( V , axs, newAxs )
        f[i] = np.vdot( u, V)
    
    return np.vstack((f0, f))
