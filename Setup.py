import numpy as np
from collections import Counter
import random
#import sparse
from qutip import *
from joblib import Parallel, delayed
from utils import *
import scipy.sparse as sp

def createSetupQST(nQubits, nShots, r=None, Wstate = True, Depolarize = 0, rng = np.random.default_rng(), lowmem = False ):
    # random.seed(42)
    #prepare true state
    if r is None:
        r = np.random.randint(1, 2**nQubits+1)
    if Wstate:
        u = np.zeros((2**nQubits, 1))
        for i in range(1, nQubits+1):
            u[2**(i-1)] = 1/np.sqrt(nQubits)
    else:
        u = np.random.rand(2**nQubits, r)
    
    if Depolarize > 0:
        rhoTrue = (1-Depolarize)*u@u.T + Depolarize/(2**nQubits)*np.eye(2**nQubits)
    
    else:
        pass
        rhoTrue = u@np.conj(u.T)

    # for extra safety:
    rhoTrue = rhoTrue/np.trace(rhoTrue)
    u = u/np.linalg.norm(u) 
    
    # true probabilities for Pauli matrices
    if not lowmem:
        m = 4**nQubits
        mList = None
        pi = qst1(rhoTrue, nQubits)
        # print('check1')
    else:
        m = 10
        mList = random.sample(range(1, 4**nQubits), m)
        # mList = np.concatenate([[0], mList])
        pi = true_prob(u, mList)
        pi = pi.ravel()

    fi = np.hstack([0.5*(pi[0]+pi[1:]), 0.5*(pi[0]-pi[1:])])
    # true probabilities for Pauli POVMs
    pi = 0.5*(pi[0]+pi[1:])   
    if not lowmem:
        sampled_pauli = random.choices(np.arange(1, m), k=nShots)
    else:
        sampled_pauli = random.choices(mList, k=nShots)
    # print('check2')
    count_pauli = Counter(sampled_pauli)
    count_pauli = dict(sorted(count_pauli.items()))
    yPlus = np.random.binomial( list(count_pauli.values()), np.real_if_close(pi))
    yMinus = list(count_pauli.values()) - yPlus
    # print('check3')
    #excluding identity Pauli corresponding to index 0 
    if nQubits <=7:
        A = create_A_qutip(nQubits, 1, 1)
    else: A = None    
    
    return rhoTrue, u, yPlus, yMinus, qutip_to_sparse(A, 1, nQubits), list(count_pauli.values()), mList, fi


# defining the pauli operators in qutip
I = qeye(2)
X = sigmax()
Y = sigmay()
Z = sigmaz()

ind_to_pauli = {'0': I, '1': X, '2': Y, '3':Z}

def base_4(int, nQubits):
    base_4_digits = np.base_repr(int, base=4)
    padded_base_4_digits = base_4_digits.zfill(nQubits)

    return padded_base_4_digits
def Kronecker_Pauli_qutip(int, nQubits):
    int_lst = base_4(int, nQubits)
    A = ind_to_pauli[int_lst[0]]
    for digit in int_lst[1:]:
        A = tensor(A, ind_to_pauli[digit])
    A.dims = [[2**nQubits], [2**nQubits]]
    return A 


def create_A_qutip(nQubits, povm, exclude_identity):
    out = []
    d = 2**nQubits
    for i in range(exclude_identity, 4**nQubits):
        out.append((povm*qeye(2**nQubits) + Kronecker_Pauli_qutip(i, nQubits=nQubits))/ (1 + povm*1))
    return out


def qobj_to_csr(qobj):
    # Direct numpy → scipy sparse
    return sp.csr_matrix(qobj.full())


def qobj_to_coo(qobj):
    # return sparse.csr_array(qobj.full())
    return sparse.COO.from_numpy(qobj.full())

def qutip_to_sparse(qobj, exclude_identity, nQubits):
    if qobj==None:
        return None
    # Use parallel processing to convert the list
    sparse_array = Parallel(n_jobs=-1)(delayed(qobj_to_coo)(obj) for obj in qobj)

    # If you want a NumPy array containing the sparse matrices
    # sparse_array = scipy.sparse.vstack(sparse_array, format = 'csr')
    sparse_array = sparse.stack(sparse_array, axis = 0)
    sparse_array = sparse.reshape(sparse_array, (4**nQubits-exclude_identity, 2**nQubits, 2**nQubits))


    return sparse_array


def qutip_to_sparse2(qobj_list, exclude_identity, nQubits):

    if qobj_list is None:
        return None

    # Parallel conversion to SciPy sparse
    sparse_list = Parallel(n_jobs=-1)(
        delayed(qobj_to_csr)(obj) for obj in qobj_list
    )

    return sparse_list

