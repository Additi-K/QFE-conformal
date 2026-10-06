import sys
import os

from utils import qst1
# from algos import *
# from Setup import *
from models import *
from UniformSamplingOnSimplex import *
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import scipy
import random
import copy
import torch.optim as optim
from time import perf_counter
from joblib import Parallel, delayed
import cvxpy as cp
import time
import pandas as pd
import warnings
from numpy.linalg import qr


# Check if MPS is available
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
device = "cpu" 


M = torch.zeros((4, 2, 2), dtype=torch.complex64)#, device=device)
M[0] = torch.tensor([[1.0, 0.0],
                    [0.0, 1.0]], dtype=torch.complex64)#, device=device)
M[1] = torch.tensor([[0.0, 1.0],
                    [1.0, 0.0]], dtype=torch.complex64)#, device=device)
M[2] = torch.tensor([[0.0, -1j],
                    [1j, 0.0]], dtype=torch.complex64)#, device=device)
M[3] = torch.tensor([[1.0, 0.0],
                    [0.0, -1.0]], dtype=torch.complex64)#, device=device)


def get_WState(nQubits):

    sigma = np.zeros((2**nQubits, 1))
    for i in range(1, nQubits + 1):
        sigma[2**(i-1)] = 1/np.sqrt(nQubits)
    sigma /= np.linalg.norm(sigma)    

    return sigma

def HS_state(N, do_complex = True, rng = np.random.default_rng() ):
    A = rng.standard_normal(size=(N, N))
    if do_complex:
        A = A + 1j*rng.standard_normal(size=(N, N))
    A = A.conj().T @ A
    return A/np.trace(A)


def fidelity(rho, sigma):
    """
    Fidelity between two quantum states.

    Works for:
        - State vectors (pure states)
        - Density matrices (mixed states)
    
    Args:
        rho, sigma: np.ndarray or torch.Tensor
            - If vector: shape (d,)
            - If density matrix: shape (d,d)
    
    Returns:
        float: fidelity between 0 and 1
    """
    # Convert tensors to numpy
    if isinstance(rho, torch.Tensor):
        rho = rho.detach().cpu().numpy()
    if isinstance(sigma, torch.Tensor):
        sigma = sigma.detach().cpu().numpy()
    
    # If both are vectors (pure states)
    if rho.ndim == 1 and sigma.ndim == 1:
        return np.abs(np.vdot(rho, sigma))**2

    # If one is vector, one is density matrix
    elif rho.ndim == 1 and sigma.ndim == 2:
        return np.real((rho.conj().T @ sigma @ rho))
    
    elif rho.ndim == 2 and sigma.ndim == 1:
        return np.real((sigma.conj().T @ rho @ sigma))
        # np.real( rho @ sigma @ (sigma.conj().T))
    
    # Both are density matrices (rank-1 though)
    elif rho.ndim == 2 and sigma.ndim == 2:
        return np.vdot(sigma, rho)
    
    else:
        raise ValueError("Input shapes not recognized")



def qmt_torch(X, operators, allow_negative=True):
    """
    Simplifying the computational complexity of mixed state measurements using the product structure of POVM, see 
    the matlab version in paper ``Superfast maximum likelihood reconstruction for quantum tomography``,
    this is the [torch] version we implemented.

    Args:
        X (tensor): Density matrix.
        operators (list, [tensor]): The set of single-qubit measurement, such as [M1, M2, ....], the size of M is (k, 2, 2).
        allow_negative (bool): Flag for whether to set the calculated value negative to zero.
    """
    if not isinstance(operators, list):
        operators = [operators]
    # X = X.to(dtype=torch.complex64)
    
    # if not isinstance(operators, torch.Tensor):
    #     operators = torch.tensor(operators)
    # operators = operators.to(device=X.device, dtype=torch.complex64)
    
    N = len(operators)  # qubits number
    Ks = torch.zeros(N, dtype=torch.int)
    Ds = torch.zeros(N, dtype=torch.int)
    for i in range(N):
        dims = operators[i].shape
        Ks[i] = dims[0]
        Ds[i] = dims[1]
        operators[i] = operators[i].reshape((Ks[i], Ds[i]**2))

    X = shuffle_forward_torch(X, Ds)
    X = X.reshape(-1, Ds[-1]**2)

    for i in range(N - 1, -1, -1):
        P = operators[i]
        X = torch.matmul(P, X.T)

        if i > 0:
            X = X.reshape(-1, Ds[i]**2)

    P_all = torch.real(X.reshape(-1))
    if not allow_negative:
        P_all = torch.maximum(P_all, torch.tensor(0))
        P_all /= torch.sum(P_all)

    return P_all

def shuffle_forward_torch(X, dims):
    """
    To transpose the density matrix, see
    the matlab version in paper ``Superfast maximum likelihood reconstruction for quantum tomography``,
    this is the [torch] version we implemented.
    """
    N = len(dims)
    X = X.T
    X = X.reshape(tuple(torch.cat([dims, dims], 0)))
    ordering = torch.reshape(torch.arange(2*N).reshape(2, -1).T, (1, -1))[0]
    X = X.permute(tuple(ordering))
    return X



def measure(nQubits, nShots, sigma, distr = "uniform", f_tar_fixed = 0.85, rank=1, dep = False, HS = True, random_shots = False, seed=None):
    """
    Generate a perturbed quantum state rho with target fidelity drawn from a Beta distribution.
    Returns:
        sigma       : original state
        densityMat  : perturbed state (density matrix)
        probs       : measurement probabilities for Pauli POVMs (y+/y-)
    """
    d = 2**nQubits
    max_try = 10000
    if distr == "beta":
        alpha_beta = 2   # Beta distribution alpha > 1 downweights near 0
        beta_beta  = 2  # Beta distribution beta < 1 upweights near 1
        f_tar = np.random.beta(alpha_beta, beta_beta, size=1).item()
    elif distr == "uniform":
        f_tar = np.random.uniform(0.01, 1.0, size=1).item()
    elif distr == "fixed":
        f_tar = f_tar_fixed
    
    if sigma.ndim == 1:
        sigma = sigma @ sigma.T.conj()

    for i in range(max_try):
        if not dep:
            if HS:
                if seed is not None:
                    rng = np.random.default_rng(seed)
                    tempState = rng.standard_normal((d, rank))+ 1j * rng.standard_normal((d, rank))
                else:
                    tempState = np.random.randn(d, rank) + 1j * np.random.randn(d, rank)
                tempState = tempState @ tempState.conj().T
                tempState /= np.trace(tempState)
            if not HS:
                tempState = haar_mixed_state(N= d)    
            
            c = np.vdot(tempState.flatten(), sigma.flatten())
            alpha = (f_tar - c)/(1 - c)

            if 0 < alpha < 1:
                rho = alpha * sigma + (1 - alpha) * tempState
                densityMat = rho# / np.trace(rho)
                # print(np.trace(densityMat), fidelity(densityMat, sigma), f_tar)
                
                break

            # fallback if max_try reached
            if i == max_try - 1:
                warnings.warn("Max try exceeded; using random convex combination fallback")
                eps = np.random.uniform(0.05, 0.95)
                densityMat = (1 - eps) * sigma + eps * tempState
                densityMat /= np.trace(densityMat)
        else:
            tempState = np.eye(d)
            alpha = (f_tar - 1)/(1/d - 1) 
            # if 0 < alpha < 1:
            rho = (1-alpha) * sigma + alpha * tempState/ d
            densityMat = rho

            break
            # if i == max_try - 1:
            #     warnings.warn("Max try exceeded; using random convex combination fallback")
            #     eps = np.random.uniform(0.05, 0.95)
            #     densityMat = (1 - eps) * sigma + eps * tempState/d
            #     densityMat /= np.trace(densityMat)

    # print(f_tar, fidelity(densityMat, sigma))    

    # --- Pauli measurement probabilities ---
    m = 4**nQubits
    fi = qst1(densityMat, nQubits)
    tot_shots = (m-1)*nShots

    # Return measurement probabilities
    if nShots is None:
        probs = np.vstack((0.5*(fi[0]+fi[1:]), 0.5*(fi[0]-fi[1:])))
        return sigma, densityMat, probs

    fi_prob = 0.5*(fi[0] + fi[1:])

    if random_shots == False:
        weights = [nShots]*(m-1)
    else:
        weights = np.random.multinomial(tot_shots, np.ones(m-1) / (m-1))

    yPlus = np.random.binomial(weights, np.real(fi_prob))
    yMinus = nShots - yPlus

    # probs = np.vstack((yPlus/nShots, yMinus/nShots))
    probs = np.vstack((yPlus/weights, yMinus/weights))


    return sigma, densityMat, probs

def counts(x, nQubits, nShots):
    m = 4**nQubits
    fi = qst1(x, nQubits)

    # Return measurement probabilities
    if nShots is None:
        probs = np.vstack((0.5*(fi[0]+fi[1:]), 0.5*(fi[0]-fi[1:])))
        return probs

    fi_prob = 0.5*(fi[0] + fi[1:])

    weights = [nShots]*(m-1)

    yPlus = np.random.binomial(weights, np.real(fi_prob))
    yMinus = nShots - yPlus

    # probs = np.vstack((yPlus/nShots, yMinus/nShots))
    probs = np.vstack((yPlus/weights, yMinus/weights))


    return probs.ravel()

def createDataset(nSamples, nShots = 100, sigma = None, distr = "uniform", f_tar_fixed = 0.85, rank = 1, dep = False, HS = True, random_shots = False, seed = None):
    nQubits = int(np.log2(sigma.shape[0]))
    x = np.zeros(( nSamples, 2*(4**nQubits - 1)), dtype=np.float32)     # f_k
    y = np.zeros((nSamples, 1), dtype=np.float32)    # fid
    rhos = []

    for i in range(nSamples):
        sigma, rho, counts = measure(nQubits, nShots=nShots, sigma=sigma, distr=distr, f_tar_fixed = f_tar_fixed, rank=rank, dep=dep, HS=HS , random_shots=random_shots, seed=seed)
        x[i] = np.real(counts.ravel())
        y[i] = np.real_if_close(fidelity(rho, sigma))
        # rhos.append(rho)

    return (x, y, rhos)


class Conformal():
    def __init__(self, nQubits, X, Y, X_test, Y_test, sigma, alpha, seed, optimizer):
        self.nQubits = nQubits
        self.x = X
        self.y = Y
        self.n = Y.shape[0]
        # idx = np.random.permutation(self.n)
        rng = np.random.default_rng(seed)
        idx = rng.permutation(self.n)
        n_half = int(np.floor(self.n/2))
        self.idx_train, self.idx_cal = idx[:n_half], idx[n_half:2*n_half]
        self.n2 = self.idx_cal.shape[0]
        self.test_x = X_test
        self.test_y = Y_test
        self.sigma = sigma
        self.alpha = alpha
        self.gamma = 1e-3
        
     
        # if optimizer == "lbfgs":
        #     self.optimizer = lbfgs_nn(nQubits=self.nQubits, P_idxs=np.arange(1, 4**self.nQubits), M=M, rank=1, device=device)
        if optimizer == "cvx":
            self.optimizer = cvx(nQubits=self.nQubits)
        elif optimizer == "least-squares":
            self.optimizer = PLS2(nQubits=self.nQubits)


        # self.pred = self.optimizer.batch_estimate(self.x, self.sigma)
        self.pred = np.zeros_like(self.y)
        self.pred[self.idx_train], _ = self.optimizer.batch_estimate(self.x[self.idx_train], self.sigma)
        self.pred[self.idx_cal], _   = self.optimizer.batch_estimate(self.x[self.idx_cal], self.sigma)
        self.test_pred, _ = self.optimizer.batch_estimate(self.test_x, self.sigma) 
        

    def conformal_analysis(self):
        t_local_str = time.time()
        y_lower_local, y_upper_local =  self.local_conformal()
        t_local = time.time()-t_local_str

        coverage_local, length_local, miscoverage_left_local, miscoverage_right_local  = compute_coverage(self.test_y,
                                                            y_lower_local,
                                                            y_upper_local
                                                            ) 
        

        
        t_adap_str = time.time()
        y_lower_adap, y_upper_adap = self.adaptive_conformal()
        t_adap = time.time() - t_adap_str
        coverage_adap, length_adap, miscoverage_left_adap, miscoverage_right_adap = compute_coverage(self.test_y,
                                                            y_lower_adap,
                                                            y_upper_adap) 
        
        return coverage_local, coverage_adap, length_local, length_adap, miscoverage_left_local, miscoverage_left_adap, miscoverage_right_local, miscoverage_right_adap, t_local, t_adap



    def local_conformal(self):
        # use all data as calibration
        self.score = np.abs(np.asarray(self.pred[self.idx_cal])-np.asarray(self.y[self.idx_cal]).ravel())
        self.quantile = np.quantile(self.score, np.ceil((self.n2+1)*(1-self.alpha))/self.n2, method='higher')

        return self.test_pred - self.quantile, self.test_pred + self.quantile


    def adaptive_conformal(self):
        trainer = optimizer_NN(
            in_size=2*(4**self.nQubits-1),
            out_size=1,
            hidden_size=256,
            ch_mult=[1]
        )

    
        x_train = torch.tensor(
            self.x[self.idx_train],
            dtype=torch.float32
        )

        y_train = torch.tensor(
            np.abs(
                self.y[self.idx_train].ravel()
                - self.pred[self.idx_train].ravel()
            ),
            dtype=torch.float32
        )
        # y_train = torch.log(y_train)

        trainer.train(x_train, y_train)
        self.sigma_model = trainer

        sigma_cal = self.sigma_model.predict(self.x[self.idx_cal]).detach().cpu().numpy().ravel()
        # sigma_cal = np.exp(sigma_cal)
        self.score_adap = np.abs(self.pred[ self.idx_cal] - self.y[self.idx_cal].ravel()) /( sigma_cal + self.gamma)
        self.quantile_adap = np.quantile(self.score_adap, np.ceil((self.n2+1)*(1-self.alpha))/self.n2, method='higher')
        sigma = self.sigma_model.predict(self.test_x).detach().cpu().numpy().ravel()

        return self.test_pred - sigma*self.quantile_adap, self.test_pred + sigma*self.quantile_adap

        

def compute_coverage(y_test,y_lower,y_upper):
    in_the_range = np.sum((y_test >= y_lower) & (y_test <= y_upper))
    coverage = in_the_range / len(y_test)
    sys.stdout.flush()

    avg_length = np.mean(y_upper - y_lower)
    sys.stdout.flush()

    miscoverage_left = np.mean(y_test < y_lower)
    miscoverage_right = np.mean(y_test > y_upper)


    return coverage, avg_length

def compute_coverage_per_sample(y_test,y_lower,y_upper):
    coverage = (y_test >= y_lower) & (y_test <= y_upper)

    sys.stdout.flush()

    length = np.abs(y_upper - y_lower)
    sys.stdout.flush()

    miscoverage_left = np.mean(y_test < y_lower)
    miscoverage_right = np.mean(y_test > y_upper)


    return coverage, length



#t-bootstrap or studentized bootstrap
def t_bootstrap(n_boot, boot_hist, confidence = 0.95):

    
    t_value = scipy.stats.t.ppf((1 + confidence) / 2.0, df=n_boot - 1)

    se = 0.0
    for acc in boot_hist:
        se += (acc - np.mean(boot_hist)) ** 2
    se = np.sqrt((1.0 / (n_boot- 1)) * se)

    ci_length = t_value * se

    ci_lower = np.mean(boot_hist) - ci_length
    ci_upper = np.mean(boot_hist) + ci_length

    return (ci_lower, ci_upper)



class Bootstrap():
    def __init__(self, nQubits, nShots, n_bootstrap, data, sigma, optimizer, alpha):
        self.nQubits = nQubits
        self.data = data
        self.sigma = sigma
        self.alpha = alpha
        self.nShots = nShots
        self.n_bootstrap = n_bootstrap
        self.optimizer = optimizer
        
        _, self.state_est = self.optimizer.batch_estimate(self.data, self.sigma)
        self.fidelity = fidelity(self.state_est[0], self.sigma)

    def bootstrap(self, resample_method = None):
    
        if resample_method == "parametric":
            self.parametric_resample()
        elif resample_method == "non-parametric":
            self.nonparametric_resample()
        
        self.bootstrap_analysis("percentile")

    def nonparametric_resample(self):
        # data based
        fp = np.random.binomial( self.nShots, self.data[0:4**self.nQubits-1], size=( self.n_bootstrap, 4**self.nQubits-1))/self.nShots
        fm = 1-fp
        data_boot = np.hstack((fp, fm))

        self.bootstrap_log, _ = self.optimizer.batch_estimate(data_boot, self.sigma)


    def parametric_resample(self):
        
        born_probab = np.real(qst1(self.state_est[0], self.nQubits))
        self.probab = np.concatenate((0.5*(born_probab[0]+born_probab[1:]) , 0.5*(born_probab[0]-born_probab[1:])))
        # parametric sampling
        fp = np.random.binomial( self.nShots, self.probab[0:4**self.nQubits-1], size=( self.n_bootstrap , 4**self.nQubits-1))/self.nShots
        fm = 1-fp
        self.data_boot = np.hstack((fp, fm))

        self.bootstrap_log, _ = self.optimizer.batch_estimate(self.data_boot, self.sigma)


    def bootstrap_analysis(self, method):

        if method == "percentile":
            self.percentile_bootstrap()
        elif method == "reverse-percentile":
            self.reverse_percentile_bootstrap()
        else:
            raise NotImplementedError(f"Bootstrap method '{method}' is not implemented.")
        
    def percentile_bootstrap(self):

        self.q_lower = np.percentile(self.bootstrap_log, 100 * self.alpha/2)
        self.q_upper = np.percentile(self.bootstrap_log, 100 * (1 - self.alpha/2))
        self.conf_int = np.array([self.q_lower, self.q_upper])

        # print(self.conf_int)

    def reverse_percentile_bootstrap(self):

        self.q_lower = np.percentile(self.bootstrap_log, 100 * (1 - self.alpha/2))
        self.q_upper = np.percentile(self.bootstrap_log, 100 * self.alpha/2)
        self.conf_int = np.array([2*self.fidelity-self.q_lower, 2*self.fidelity-self.q_upper])
        # print(self.conf_int)

    from numpy.linalg import qr


def Haar(N=2, do_complex=True, rng = np.random.default_rng() ):
    A = rng.standard_normal(size=(N, N))
    if do_complex:
        A = A + 1j*rng.standard_normal(size=(N, N))
    Q, R = scipy.linalg.qr(A)
    Q = Q * np.sign(np.diag(R))
    return Q

def haar_mixed_state(N=2,do_complex=True, rng = np.random.default_rng() ):
    Q = Haar(N, do_complex, rng)
    w = uniform_sampling_on_simplex(N, n_pts=1).ravel()
    return Q @ np.diag(w) @ Q.conj().T

def pure_state(N=2,do_complex=True, rng = np.random.default_rng() ):
    Q = Haar(N, do_complex, rng)
    return np.outer(Q[:,0], Q[:, 0].conj())



