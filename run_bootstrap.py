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

# Check if MPS is available
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
# device = "cpu"
print('using device:', device) 

import sys
import os

from cqr import helper



def run_experiment(method, nQubits, nShots):
    # initialize data to save

    alpha = 0.1
    
    # sigma = get_randomState(nQubits, rank=1)
    sigma = pure_state(N=2**nQubits)

    x, y, _ = createDataset(nSamples=1, nShots= nShots, sigma= sigma)
    y = y.ravel()

    # # divide the dataset into test and train based on the test_ratio parameter
    # x_train, x_test, y_train, y_test = train_test_split(X,
    #                                                     Y.ravel(),
    #                                                     test_size=test_ratio,
    #                                                     random_state=random_state_train_test)


    # reshape the data
    x_test = np.asarray(x)
    y_test = np.asarray(y)

    resampling, bootstrapping = method

    bootstrap = Bootstrap(nQubits=nQubits, data = x_test, sigma=sigma, alpha = alpha,  optimizer= CVX(nQubits), nShots=nShots, n_bootstrap=1000)
    bootstrap.bootstrap(resample_method=resampling)
    bootstrap.bootstrap_analysis(method = bootstrapping)
    y_lower, y_upper = float(bootstrap.conf_int[0]), float(bootstrap.conf_int[1])


    # compute and print average coverage and average length
    coverage, length = helper.compute_coverage(y_test,
                                                y_lower,
                                                y_upper,
                                                alpha) 
   
    return coverage, length



if __name__ == "__main__":
    # use the same bootstrap samples for percentile and reverse-percentile methods for a particular 
    # resampling technique

    resampling_methods = ['parametric']
    bootstrap_methods = ['percentile']
    n_qubits_list = [2]#, 4, 5, 6]
    n_shots_list = [50, 100, 200, 500, 1000, 100000]
    n_reps = 2
    
    records= []
    save = True
    save_path = "./results"
    os.makedirs(save_path, exist_ok=True)

    for resampling in resampling_methods:
        for bootstrap in bootstrap_methods:
            for n_shots in n_shots_list:
                for n_qubits in n_qubits_list:
                    results_dict = {}
                    for i in range(n_reps):
                        print(f'method:{resampling}, {bootstrap}, # shots:{n_shots}, # qubits:{n_qubits}, iteration:{i}')
                        # pass the current results dict
                        coverage, length = run_experiment((resampling, bootstrap), n_qubits, n_shots)
                        records.append({
                        "nQubits": n_qubits,
                        "nShots": n_shots,
                        "iter": i,
                        "method": resampling,
                        'percentile': bootstrap,
                        'coverage': coverage,
                        'avg_length': length})
                    
                    

    df_results = pd.DataFrame(records)
    if save:
        df_results.to_csv(os.path.join(save_path, "bootstrap_results.csv"), index=False)
    print("Code Executed Successfully!") 

    # plot results
    if save:
        df_results = pd.read_csv(os.path.join(save_path, "bootstrap_results.csv"))
        plt.rcParams.update({'font.size': 14})
        df_results['method'] = "Bootstrap"
        df_results['coverage'] = df_results['coverage']/100
        summary = df_results.copy()
        print(summary)
        summary = summary.groupby(["nShots"]).agg(
            mean_cov=("coverage", "mean"),
            mean_len=("avg_length", "mean"),
            std_cov=("coverage", "std"),
            n_runs=("coverage", "count")
        ).reset_index()


        # Compute 95% Confidence Intervals
        summary["ci_cov"] = 1.96 * (summary["std_cov"] / np.sqrt(summary["n_runs"]))

        # 2. Plot Coverage Comparison
        sns.lineplot(
            data=df_results, x="nShots", y="coverage", hue="method", markersize = 8 , err_kws={"capsize": 4},
            marker="o", linestyle = "-.",err_style="bars", errorbar=("ci", 95)
        )
        plt.rcParams.update({'font.size': 14})
        plt.axhline(0.9, linestyle="dotted", color='black', label="Target Coverage")
        plt.xlabel(r"$N$")
        plt.ylabel("Empirical Coverage")
        plt.xscale('log')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(save_path, "bootstrap_cov.pdf"), bbox_inches='tight', dpi=300)
        plt.show()

        # 3. Plot Interval Length Comparison
        sns.lineplot(
            data=df_results, x="nShots", y="avg_length", hue="method", markersize = 8 , err_kws={"capsize": 4},
            marker="o", linestyle = "-.", err_style="bars", errorbar=("ci", 95)
        )
        plt.rcParams.update({'font.size': 14})
        plt.xlabel(r"$N$")
        plt.ylabel("Interval Length")
        plt.xscale('log')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(save_path, "bootstrap_len.pdf"), bbox_inches='tight', dpi=300)
        plt.show()