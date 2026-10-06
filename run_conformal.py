import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import random
from uq_utils import *
import pickle
import seaborn as sns
from plot_util import *

# Check if MPS is available
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print('using device:', device) 

import sys


#https://github.com/yromano/cqr
from cqr import helper
from nonconformist.nc import RegressorNc
# from nonconformist.cp import IcpRegressor
from nonconformist.nc import AbsErrorErrFunc
# from nonconformist.nc import SignErrorErrFunc
from nonconformist.nc import RegressorNormalizer
from nonconformist.nc import QuantileRegErrFunc
from nonconformist.nc import QuantileRegAsymmetricErrFunc



def run_experiment(method, nQubits, x_train, y_train, x_test, y_test, idx_train, idx_cal, alpha, random_state = None):
    # initialize data to save
    # in_shape = 2*(4**nQubits-1)
    in_shape = x_train.shape[1]
    size_hidden = 256
    NN_map = {
        2: {'hidden': size_hidden , 'layers':[1]},
        3: {'hidden': size_hidden , 'layers':[1]},
        4: {'hidden': size_hidden, 'layers':[1]},
        5: {'hidden': size_hidden, 'layers':[1]},
        6: {'hidden': size_hidden, 'layers':[1]}
    }
    size_hidden = 256
    CQR_map = {
        2: {'hidden': size_hidden , 'layers':[1]},
        3: {'hidden': size_hidden , 'layers':[1]},
        4: {'hidden': size_hidden, 'layers':[1]},
        5: {'hidden': size_hidden, 'layers':[1]},
        6: {'hidden': size_hidden, 'layers':[1]}
    }

    # pytorch's optimizer object
    nn_learn_func = torch.optim.Adam

    # number of epochs
    epochs = 2000
    epochs_cqr = 2000

    # learning rate
    lr = 1e-4

    # mini-batch size
    batch_size = 128

    # dropout regularization rate
    dropout = 0.1

    # weight decay regularization
    wd = 1e-6

    # Ask for a reduced coverage when tuning the network parameters by 
    # cross-validataion to avoid too concervative initial estimation of the 
    # prediction interval. This estimation will be conformalized by CQR.
    # desired miscoverage error
    
    quantiles_net = [alpha/2, 1- alpha/2]

    cv_test_ratio = 0.05

    cv_random_state = random_state

    strt = time.time()
    if method == "Neural Net":
        model = helper.MSENet_RegressorAdapter(model=None,
                                       fit_params=None,
                                       in_shape = in_shape,
                                       hidden_size = NN_map[nQubits]["hidden"],
                                       ch_mult = NN_map[nQubits]["layers"],
                                       learn_func = nn_learn_func,
                                       epochs = epochs,
                                       batch_size=batch_size,
                                       dropout=dropout,
                                       lr=lr,
                                       wd=wd,
                                       test_ratio=cv_test_ratio,
                                       random_state=cv_random_state)
                
        nc = RegressorNc(model)

        y_lower, y_upper = helper.run_icp(nc, x_train, y_train, x_test, idx_train, idx_cal, alpha)

        method_name = "Split Conformal Neural Network"

        # compute and print average coverage and average length
        coverage, length = helper.compute_coverage(y_test,
                                                    y_lower,
                                                    y_upper,
                                                    alpha,
                                                    method_name)
        
    if method == "Neural Net Local":
        # mean_estimator 
        model = helper.MSENet_RegressorAdapter(model=None,
                                       fit_params=None,
                                       in_shape = in_shape,
                                       hidden_size = NN_map[nQubits]["hidden"],
                                       ch_mult = NN_map[nQubits]["layers"],
                                       learn_func = nn_learn_func,
                                       epochs = epochs,
                                       batch_size=batch_size,
                                       dropout=dropout,
                                       lr=lr,
                                       wd=wd,
                                       test_ratio=cv_test_ratio,
                                       random_state=cv_random_state)
# absolute deviation
        mad_estimator  = helper.MSENet_RegressorAdapter(model=None,
                                            fit_params=None,
                                            in_shape = in_shape,
                                            hidden_size = NN_map[nQubits]["hidden"],
                                            ch_mult = NN_map[nQubits]["layers"],
                                            learn_func = nn_learn_func,
                                            epochs = epochs,
                                            batch_size=batch_size,
                                            dropout=dropout,
                                            lr=lr,
                                            wd=wd,
                                            test_ratio=cv_test_ratio,
                                            random_state=cv_random_state)

        normalizer = RegressorNormalizer(model, 
                                        # mean_estimator,
                                        mad_estimator,
                                        AbsErrorErrFunc())

        # nc = RegressorNc(mean_estimator, AbsErrorErrFunc(), normalizer, beta = 1e-3)
        nc = RegressorNc(model, AbsErrorErrFunc(), normalizer, beta = 1e-3)

        y_lower, y_upper = helper.run_icp(nc, x_train, y_train, x_test, idx_train, idx_cal, alpha)

        method_name = "Locally Adaptive Conformal Neural Network"

        # compute and print average coverage and average length
        coverage, length = helper.compute_coverage(y_test,
                                                                            y_lower,
                                                                            y_upper,
                                                                            alpha,
                                                                            method_name)
    if method == "Quantile Neural Net (Non-Conformal)":
        model = helper.AllQNet_RegressorAdapter(model=None,
                                       fit_params=None,
                                       in_shape = in_shape,
                                       hidden_size = CQR_map[nQubits]["hidden"],
                                       ch_mult = CQR_map[nQubits]["layers"],
                                       quantiles=quantiles_net,
                                       learn_func = nn_learn_func,
                                       epochs = epochs_cqr,
                                       batch_size=batch_size,
                                       dropout=dropout,
                                       lr=lr,
                                       wd=wd,
                                       test_ratio=cv_test_ratio,
                                       random_state=cv_random_state)
                
        model.fit(x_train, y_train)
        tmp = model.predict(x_test)
        y_lower = tmp[:,0]
        y_upper = tmp[:,1]
    

        method_name = "Quantile Neural Network"

        # compute and print average coverage and average length
        coverage, length = helper.compute_coverage(y_test,
                                                                    y_lower,
                                                                    y_upper,
                                                                    alpha,
                                                                    method_name)
        
    if method == "Symmetric CQR Neural Net":
        # quantile_estimator_symm 
        model = helper.AllQNet_RegressorAdapter(model=None,
                                                    fit_params=None,
                                                    in_shape=in_shape,
                                                    hidden_size = CQR_map[nQubits]["hidden"],
                                                    ch_mult = CQR_map[nQubits]["layers"],
                                                    quantiles=quantiles_net,
                                                    learn_func=nn_learn_func,
                                                    epochs=epochs_cqr,
                                                    batch_size=batch_size,
                                                    dropout=dropout,
                                                    lr=lr,
                                                    wd=wd,
                                                    test_ratio=cv_test_ratio,
                                                    random_state=cv_random_state,
                                                    use_rearrangement=False)

        # define a CQR object, computes the absolute residual error of points 
        # located outside the estimated quantile neural network band 
        # nc = RegressorNc(quantile_estimator_symm, QuantileRegErrFunc())      # can be replaced by QuantileRegAsymmetricErrFunc()
        nc = RegressorNc(model, QuantileRegErrFunc())
        # run CQR procedure
        y_lower, y_upper = helper.run_icp(nc, x_train, y_train, x_test, idx_train, idx_cal, alpha)

        # compute and print average coverage and average length
        coverage, length = helper.compute_coverage(y_test,
                                                                y_lower,
                                                                y_upper,
                                                                alpha,
                                                                "CQR Symmetric Neural Net")
        
    if method == "Asymmetric CQR Neural Net":
        # quantile_estimator_asymm
        model = helper.AllQNet_RegressorAdapter(model=None,
                                                    fit_params=None,
                                                    in_shape=in_shape,
                                                    hidden_size = CQR_map[nQubits]["hidden"],
                                                    ch_mult = CQR_map[nQubits]["layers"],
                                                    quantiles=quantiles_net,
                                                    learn_func=nn_learn_func,
                                                    epochs=epochs_cqr,
                                                    batch_size=batch_size,
                                                    dropout=dropout,
                                                    lr=lr,
                                                    wd=wd,
                                                    test_ratio=cv_test_ratio,
                                                    random_state=cv_random_state,
                                                    use_rearrangement=False)

        # define the CQR object, computing the absolute residual error of points 
        # located outside the estimated quantile neural network band 
        # nc = RegressorNc(quantile_estimator_asymm, QuantileRegAsymmetricErrFunc())
        nc = RegressorNc(model, QuantileRegAsymmetricErrFunc())

        # run CQR procedure
        y_lower, y_upper = helper.run_icp(nc, x_train, y_train, x_test, idx_train, idx_cal, alpha)

        # compute and print average coverage and average length
        coverage, length = helper.compute_coverage(y_test,
                                                                        y_lower,
                                                                        y_upper,
                                                                        alpha,
                                                                        "CQR Asymmetric Neural Net") 
        


          
    return coverage, length, y_lower, y_upper, time.time() - strt, model
   
    


if __name__ == "__main__":
    
    test_methods = ['Asymmetric CQR Neural Net']#,'Neural Net Local','Quantile Neural Net (Non-Conformal)', 'Asymmetric CQR Neural Net']
             
    
    n_qubits_list = [2]
    n_shots_list = [100]
    n_reps = 5
    alpha = 0.1
    save_path = "./results"
    os.makedirs(save_path, exist_ok=True)
    
    records = []
    save = True

    for nQubits in n_qubits_list:
        sigma = np.random.randn(2**nQubits, 1) + 1j*np.random.randn(2**nQubits, 1)
        sigma_vec = sigma/np.linalg.norm(sigma)
        sigma = sigma @ sigma.T.conj()
        sigma /= np.trace(sigma)

        train_states = [HS_state(N= 2**nQubits) for _ in range(20000)]
        
        for nShots in n_shots_list:
            X = [counts(rho, nQubits, nShots) for rho in train_states]
            Y = [np.real(fidelity(rho, sigma)) for rho in train_states]

            X = np.asarray(X, dtype=np.float32)
            Y = np.asarray(Y, dtype=np.float32)
            for i in range(n_reps):

                # divide the dataset into test and train based on the test_ratio parameter
                x_train, x_test, y_train, y_test = train_test_split(X,
                                                                    Y.ravel(),
                                                                    test_size=0.2,
                                                                    random_state=i)

                # compute input dimensions
                n_train = x_train.shape[0]
                n_test = x_test.shape[0]
                in_shape = x_train.shape[1]

                # divide the data into proper training set and calibration set
                idx = np.random.permutation(n_train)
                n_half = int(np.floor(n_train/2))
                idx_train, idx_cal = idx[:n_half], idx[n_half:2*n_half]

                for method in test_methods:
                    print(f"\n# qubits:{nQubits}, iteration:{i}, method:{method}")
                    coverage, length, y_lower, y_upper, t, _ = run_experiment(method, nQubits, x_train, y_train, x_test, y_test, idx_train, idx_cal, alpha)
                    records.append({
                        "nQubits": nQubits,
                        "nShots": nShots,
                        "iter": i,
                        "solver" : "NN",
                        "method": method,
                        'coverage': coverage,
                        'avg_length': length,
                        'miscoverage_left': np.sum(y_test < y_lower)/n_test,
                        'miscoverage_right': np.sum(y_test > y_upper)/n_test,
                        'time': t })

    if save:
        df_results = pd.DataFrame(records)
        df_results.to_csv(os.path.join(save_path, "test.csv"), index=False)
        plot_conformal_coverage_from_csv(os.path.join(save_path, "test.csv"), save_dir=save_path)
        plot_conformal_length_from_csv(os.path.join(save_path, "test.csv"), save_dir=save_path)

    print("Code Executed Successfully!") 
    










    
    


