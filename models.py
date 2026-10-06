import sys
import os
from Setup import *
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
import torch.optim as optim
from time import perf_counter
import cvxpy as cp
from joblib import Parallel, delayed
import torch.multiprocessing as mp
import sparse
from scipy.sparse import csr_matrix, vstack, eye
from scipy import sparse
from joblib import Parallel, delayed

# Check if MPS is available
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
device = torch.device("cpu")


class loadDataset(Dataset):
    
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):

        return (self.y).shape[0]
    

    def __getitem__(self, idx):

        return self.x[idx], self.y[idx]

def MLE_loss(f_out, f_star):
    """Negative log-likelihood function"""
    if isinstance(f_star, np.ndarray):
        f_star = torch.tensor(f_star, dtype=torch.float32, device=device)
    else:
        f_star = f_star.to(dtype=torch.float32, device=device)

    f_star_flat = f_star.flatten()
    f_out_flat = f_out.flatten()
    out_idx = f_out_flat > 1e-16
    loss = -torch.dot(f_star_flat[out_idx], torch.log(f_out_flat[out_idx]))

    return loss


class NN_model(nn.Module):

    def __init__(self, in_size, out_size, ch_mult, base_ch):
        super().__init__()
        self.input_size = in_size
        self.output_size = out_size
        self.ch_mult = ch_mult
        self.n_layers = len(self.ch_mult)
        self.base_ch = base_ch
        # self.sigmoid = sigmoid
        self.input_layer = nn.Linear(self.input_size, int(self.ch_mult[0]*self.base_ch))
        self.hidden = nn.ModuleList()
        self.dropout = 0.1

        # for i in range(self.n_layers-1):
        #     self.hidden.append(nn.Linear(int(self.ch_mult[i]*self.base_ch),int( self.ch_mult[i+1]*self.base_ch)))

        for i in range(1, self.n_layers):
            self.hidden.append(nn.Linear(int(self.ch_mult[i-1]*self.base_ch),int( self.ch_mult[i]*self.base_ch)))

        self.output_layer = nn.Linear(int(self.ch_mult[-1]*self.base_ch), self.output_size)

        self.act1 = nn.ReLU()
        self.dropout_layer = nn.Dropout(self.dropout)

        self.init_weights()


    def init_weights(self):
        """ Initialize the network parameters
        """
        # Initialize input layer
        nn.init.orthogonal_(self.input_layer.weight)
        nn.init.constant_(self.input_layer.bias, 0)

        # Initialize each hidden layer
        for layer in self.hidden:
            nn.init.orthogonal_(layer.weight)
            nn.init.constant_(layer.bias, 0)

        # Initialize output layer
        nn.init.orthogonal_(self.output_layer.weight)
        nn.init.constant_(self.output_layer.bias, 0)    


    def forward(self, x):

        z = self.input_layer(x)
        z = self.act1(z)
        z = self.dropout_layer(z)

        for layer in self.hidden:
            z = layer(z)
            z = self.act1(z)
            z = self.dropout_layer(z)

        z = self.output_layer(z)
        
        z = z.squeeze(-1)    

        return z   


    def loss(self, x, y):

        prediction = self.forward(x)
        # loss = torch.mean((prediction - y) ** 2)
        mse = torch.nn.MSELoss()
        loss = mse(prediction.flatten(), y.flatten())

        return loss 

# functions for training and evaluating the model
class optimizer_NN():

    def __init__(
        self,
        in_size,
        out_size,
        hidden_size,
        ch_mult,
        optimizer=torch.optim.Adam,
        epochs=2000,
        batch_size=128,
        lr=1e-4,
        wd=1e-6,
        test_ratio=0.1,
        random_state=0
    ):

        # FIXED: remove trailing commas
        self.in_size = in_size
        self.out_size = out_size
        self.hidden_size = hidden_size
        self.ch_mult = ch_mult
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.wd = wd
        self.test_ratio = test_ratio
        self.random_state = random_state
        self.device = "mps"

        self.model = NN_model(
            in_size=in_size,
            out_size=out_size,
            ch_mult=ch_mult,
            base_ch=hidden_size
        ).to(self.device)

        self.optimizer = optimizer(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.wd
        )

        self.loss_func = nn.MSELoss()

    # --------------------------------------------------

    def train(
    self,
    x,
    y,
    print_freq=100,
    name="model_f",
    save=False
):
        # Split data
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=self.test_ratio, random_state=self.random_state)
        
        # Ensure tensors
        if not torch.is_tensor(x_train):
            x_train = torch.tensor(x_train, dtype=torch.float32)
        if not torch.is_tensor(y_train):
            y_train = torch.tensor(y_train, dtype=torch.float32)
        if not torch.is_tensor(x_test):
            x_test = torch.tensor(x_test, dtype=torch.float32)
        if not torch.is_tensor(y_test):
            y_test = torch.tensor(y_test, dtype=torch.float32)

        train_dataset = TensorDataset(x_train, y_train)
        test_dataset = TensorDataset(x_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        train_loss_hist = []
        test_loss_hist = []

        for epoch in range(self.epochs):
            self.model.train()
            epoch_train_losses = []

            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                loss = self.model.loss(x_batch, y_batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # predictions = self.model.forward(x_batch)

                epoch_train_losses.append(loss.item())

            train_loss = np.mean(epoch_train_losses)
            train_loss_hist.append(train_loss)

            # Evaluate on test set
            self.model.eval()
            test_losses = []
            with torch.no_grad():
                for x_batch, y_batch in test_loader:
                    x_batch = x_batch.to(self.device)
                    y_batch = y_batch.to(self.device)

                    loss = self.model.loss(x_batch, y_batch)
                    test_losses.append(loss.item())
            test_loss = np.mean(test_losses)
            test_loss_hist.append(test_loss)

            # Print every print_freq epochs
            # if (epoch + 1) % print_freq == 0:
            #     print(f"Epoch {epoch+1}/{self.epochs} | Train Loss: {train_loss:.4e} | Test Loss: {test_loss:.4e}")

        # Save checkpoint
        if save:
            checkpoint = {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epoch": self.epochs,
                "train_loss": train_loss_hist,
                "test_loss": test_loss_hist
            }
            save_path = f"{name}_ckpt.pt"
            torch.save(checkpoint, save_path)
            print(f"Checkpoint saved to {save_path}")

        # Plot training & test loss
        # plt.figure()
        # plt.plot(train_loss_hist, label="Train Loss")
        # plt.plot(test_loss_hist, label="Test Loss")
        # plt.yscale("log")
        # plt.xlabel("Epoch")
        # plt.ylabel("MSE Loss")
        # plt.legend()
        # plt.show(block=False)  # <- don't block execution
        # plt.pause(0.1)          # optional: let the figure render
        # plt.close()             # automatically close


    def evaluate(self, data, model_path=None):

        if model_path is not None:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model"])

        self.model.eval()

        errors = []
        with torch.no_grad():
            pbar = tqdm(data)
            for x, y in pbar:
                x = x.to(self.device)
                y = y.to(self.device)

                pred = self.model(x)
                err = torch.mean((pred - y) ** 2)
                errors.append(err.item())

                pbar.set_description(
                    f"Validation MSE: {np.mean(errors):.4e}"
                )

    def predict(self, x, model_path=None):

        if model_path is not None:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model"])

        self.model.eval()
        with torch.no_grad():
            if not torch.is_tensor(x):
                x = torch.tensor(x, dtype=torch.float32)

            x = x.to(self.device)
            pred = self.model(x)

        return pred.detach()     


class lbfgs_nn(nn.Module):

    def __init__(self,
                 nQubits,
                 P_idxs,
                 M, rank, device):
        super().__init__()
                    
        self.N = nQubits
        self.P_idxs = P_idxs
        self.M = M
        self.device = device 
        self.rank = rank    

        d = 2**nQubits
        params = torch.randn((2, d, self.rank), requires_grad=True).to(torch.float32)
        self.params = nn.Parameter(params)


    def forward(self):
        rho = self.Rho()
        f_out = self.Measure_rho(rho).to(self.device)
        return f_out

    def Rho(self):
        U = torch.complex(self.params[0,:,:], self.params[1,:,:])
        rho = torch.matmul(U, U.conj().T)
        rho = rho / torch.trace(rho)
        return rho

    def Measure_rho(self, rho):
        """Born's Rule"""
        from uq_utils import qmt_torch

        f_all = qmt_torch(rho, [self.M] * self.N)
        tr = f_all[0:1]
        f = f_all[self.P_idxs]

        return torch.vstack((0.5*(tr+f).real, 0.5*(tr-f).real))#.to(torch.float32)

    # def batch_estimate(self, x, sigma):
    #     if isinstance(x, np.ndarray):
    #         x = torch.tensor(x, dtype=torch.float32, device=device)
    #     else:
    #         x = x.to(dtype=torch.float32, device=device)
        
    #     self.pred = []
    #     n = x.shape[0]
    #     for i in range(n):
    #         # self.reset_parameters()
    #         # print(i)
    #         fid = mle_estimate(self, x[i], sigma)[0]
    #         self.pred.append(fid)
        
    #     self.pred = np.array(self.pred)     

    #     return self.pred  


    def batch_estimate(self, x, sigma, n_jobs=8):
        
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)

        mp.set_start_method("spawn", force=True)

        args = [
            (
                i,                 # seed
                x[:, i],              # data sample
                sigma,
                self.N,             # nQubits
                self.P_idxs,
                self.M,
                self.rank,
                self.device,
            )
            for i in range(x.shape[1])
        ]

        with mp.Pool(processes=n_jobs) as pool:
            results = pool.map(lbfgs_worker, args)
        
        # results is a list of tuples (fidelity, state)
        preds, states = zip(*results)  # <-- unpack correctly
        return np.asarray(preds), list(states)
        
        # return np.asarray(preds), states 

    def reset_parameters(self):
        with torch.no_grad():
            self.params.copy_(torch.randn_like(self.params))

def lbfgs_worker(args):
    (
        seed,
        x_i,
        sigma,
        nQubits,
        P_idxs,
        M,
        rank,
        device,
    ) = args
    # ---- safety ----
    torch.set_num_threads(1)
    # torch.set_num_interop_threads(1)

    torch.manual_seed(seed)
    np.random.seed(seed)

    # ---- build fresh model ----
    model = lbfgs_nn(
        nQubits=nQubits,
        P_idxs=P_idxs,
        M=M,
        rank=rank,
        device=device,
    ).to(device)

    # ---- run MLE ----
    fid, state = mle_estimate(model, x_i, sigma)

    return fid, state 

class lbfgs_trainer():
    def __init__(self, generator, f_star, rho_star, learning_rate=0.01):
        self.generator = generator
        self.f_star = f_star
        self.criterion = MLE_loss
        self.rho_star = rho_star
        self.optim = optim.LBFGS(self.generator.parameters(), lr=learning_rate, max_iter=1000, 
                               tolerance_grad=1e-06, tolerance_change=1e-06, 
                               history_size=10, line_search_fn=None)
      
        self.overhead_t = 0
        self.epoch = 0
        self.time_all = 0 

    def track_parameters(self, loss, result_save):
        from uq_utils import fidelity
        """Callback to store parameter updates (excluding computation time)."""
        
        start_overhead = perf_counter()  # Start timing overhead
        self.generator.eval()

        with torch.no_grad():
            rho = self.generator.Rho()
            penalty =  2 * torch.sum(self.f_star) * torch.norm(self.generator.params, p=2) ** 2
            
            Fq = fidelity(rho, self.rho_star)

            result_save['epoch'].append(self.epoch)
            result_save['Fq'].append(Fq)
            result_save['loss'].append(loss.item() - penalty)
            self.epoch += 1

        self.overhead_t = perf_counter() - start_overhead 

    def train(self, result_save):
        """Net training."""
        pbar = tqdm(range(1), mininterval=0.01, disable=True)
        epoch = 0
        mse = torch.nn.MSELoss()

        for _ in pbar:
            epoch += 1
            

            self.generator.train()

            def closure():
                self.generator.train()
                time_b = perf_counter()
                self.optim.zero_grad()
                f_out = self.generator()#.detach().cpu()
                loss = self.criterion(f_out, self.f_star)
                loss += 2 * torch.sum(self.f_star) * torch.norm(self.generator.params, p=2) ** 2
                
                assert not torch.isnan(loss), "Loss is NaN" 
                loss.backward()
                
                self.track_parameters(loss, result_save)
                # Update tracking (exclude overhead from time_all)
                raw_t = perf_counter()
                self.time_all += raw_t - time_b - self.overhead_t
                result_save['time'].append(self.time_all)

                return loss.detach()

            self.optim.step(closure)

        pbar.close()

    def get_state_estimate(self):
        with torch.no_grad():
            U = torch.complex(self.generator.params[0,:,:], self.generator.params[1,:,:]).detach().cpu()
            U = U/torch.norm(U)

        return U

def mle_estimate(generator, data, sigma = None, device = None):    
    
    if isinstance(data, np.ndarray):
        data = torch.tensor(data, dtype=torch.float32, device=device)
    else:
        data = data.to(dtype=torch.float32, device=device)

    if isinstance(sigma, np.ndarray):
        sigma = torch.tensor(sigma, dtype=torch.float32, device=device)
    else:
        sigma = sigma.to(dtype=torch.float32, device=device)

    net = lbfgs_trainer(generator=generator, f_star=data, rho_star=sigma)
    result_save = {
            'time': [],
            'epoch': [],
            'Fq': [], 
            'loss': []}
    net.train(result_save)
    fidelity = result_save['Fq'][-1]
    # state = net.get_state_estimate()
    rho = generator.Rho()

    return fidelity, rho.detach().cpu().numpy()


class CVX():
    def __init__(self, nQubits):
        self.nQubits = nQubits
        self.d = 2**nQubits
        d_2 = 4**nQubits -1 
        self.b = cp.Parameter(2*d_2, nonneg=True)
        self.rho = cp.Variable((self.d, self.d), hermitian = True)
        self.M_op = cp.Constant(self.operator())
        
        rho_vec = cp.vec(self.rho, order="C")
        # probabilities = cp.real(self.M_op @ cp.vec(self.rho))
        # self.error = cp.sum_squares(self.M_op @ rho_vec - self.b)

        self.error = -cp.sum(cp.multiply(self.b, cp.log(cp.real(self.M_op @ rho_vec) + 1e-9)))
        self.constraints = [self.rho >> 0, cp.real(cp.trace(self.rho)) == 1]

        self.obj = cp.Minimize(self.error)
        self.prob = cp.Problem(self.obj, self.constraints)
        tol = 1e-9
        self.SCSopts  = {'max_iters':10000, 'eps':tol}
        
    
    # def batch_estimate(self, X, sigma):
    #     from uq_utils import fidelity
    #     self.rho_values = []
    #     self.fid_values = []
    #     for i in range(X.shape[0]):
            
    #         self.b.value = X[i]
    #         self.prob.solve(solver=cp.SCS, warm_start=True, verbose = False)
    #         self.rho_values.append(self.rho.value)
    #         self.fid_values.append(np.real(fidelity(self.rho.value, sigma)))
       
    #     self.fid_values = np.array(self.fid_values)    

    #     return self.fid_values, self.rho_values  
    
    def operator(self):
        W_qobjs = create_A_qutip(self.nQubits, 0, 1)
        M_p = []
        M_m = []
        for W_q in W_qobjs:
            # Wi is a complex matrix (especially for Y-basis)
            Wi = W_q.full() 
            # Flattening (Column-major to match cp.vec)
            M_p.append(0.5 * (np.eye(self.d) + Wi).flatten(order='F'))
            M_m.append(0.5 * (np.eye(self.d) - Wi).flatten(order='F'))
        M_p.extend(M_m)
        return np.array(M_p, dtype=np.complex128)
        
    
    def _solve_single(self, x_data, sigma):
        """Worker function for parallel solving"""
        from uq_utils import fidelity
        
        # Set the parameter for this specific instance
        self.b.value = np.real(x_data).astype(np.float64)
        
        # try:
        #     # Clarabel is superior for PSD + Log-cone problems
        #     self.prob.solve(solver=self.solver, verbose=False)
        #     res_rho = self.rho.value
        # except:
        #     # Fallback to maximally mixed state on failure
        #     res_rho = np.eye(self.d) / self.d
            
        # if res_rho is None:
        #     res_rho = np.eye(self.d) / self.d

        self.prob.solve(solver =cp.SCS, warm_start=True , verbose=False, **self.SCSopts)
        res_rho = self.rho.value
        fid = np.real(fidelity(res_rho, sigma))
        return res_rho, fid

    def batch_estimate(self, X, sigma, n_jobs=-1):
        """
        Runs the batch in parallel. 
        Note: joblib handles the serialization of the CVX object.
        """
        # Parallel solve bypasses the sequential canonicalization bottleneck
        results = Parallel(n_jobs=n_jobs)(
            delayed(self._solve_single)(X[i], sigma) for i in range(X.shape[0])
        )
        
        rho_values, fid_values = zip(*results)
        return np.array(fid_values), list(rho_values)



class PLS():
    def __init__(self, nQubits):
        self.nQubits = nQubits
        W = create_A_qutip(self.nQubits, 0, 1)
        self.W = qutip_to_sparse(W, 1, self.nQubits)
        self.W_dense = np.stack([Wi.todense() for Wi in self.W], axis=0)  # shape: (n_ops, d, d)
        self.m = len(W)
        self.d = 2**nQubits
        self.eye_d = np.eye(self.d, dtype=np.complex64)

    def least_squares_batch(self, X):
        """
        Vectorized computation of least squares rho for a batch
        X: shape (batch_size, 2*m)
        Returns: rho_ls_batch of shape (batch_size, d, d)
        """
        batch_size = X.shape[0]
        rho_ls_batch = np.zeros((batch_size, self.d, self.d), dtype=np.complex64)

        # Vectorized accumulation over measurement operators
        for i in range(self.m):
            # X[:, i] - X[:, i+self.m] -> shape (batch_size,)
            coeffs = (X[:, i] - X[:, i+self.m])[:, None, None]  # shape (batch_size,1,1)
            rho_ls_batch += coeffs * self.W_dense[i][None, :, :]  # broadcast multiplication

        rho_ls_batch += self.eye_d[None, :, :]
        rho_ls_batch /= self.d

        return rho_ls_batch

    def project_single(self, rho_ls):
        """
        Project a single rho_ls onto valid density matrix
        """
        rho_H = 0.5 * (rho_ls + rho_ls.conj().T)

        # Eigendecomposition
        eigvals, eigvecs = np.linalg.eigh(rho_H)
        idx = np.argsort(eigvals)[::-1]
        eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]

        # Find threshold x0 for projection
        S = 0
        d = len(eigvals)
        for k in range(1, d+1):
            S += eigvals[k-1]
            x0 = (S - 1)/k
            if k == d or x0 >= eigvals[k]:
                break

        eigvals_proj = np.maximum(eigvals - x0, 0)
        rho_proj = eigvecs @ np.diag(eigvals_proj) @ eigvecs.conj().T
        return rho_proj

    def batch_estimate(self, X, sigma, n_jobs=-1):
        """
        Compute PLS estimates and fidelities for a batch of inputs
        X: shape (batch_size, 2*m)
        sigma: true state for fidelity computation
        n_jobs: number of parallel jobs for projection
        """
        from uq_utils import fidelity

        # Step 1: Vectorized least squares for batch
        rho_ls_batch = self.least_squares_batch(X)

        # Step 2: Parallel projection of each rho_ls
        rho_pls_list = Parallel(n_jobs=n_jobs)(
            delayed(self.project_single)(rho_ls_batch[i]) for i in range(X.shape[0])
        )
        rho_pls_batch = np.array(rho_pls_list)

        # Step 3: Compute fidelities
        fid_values = np.array([fidelity(sigma, rho_pls_batch[i]) for i in range(X.shape[0])])

        self.rho_values = rho_pls_batch
        self.fid_values = fid_values

        return fid_values, rho_pls_batch



    
if __name__ == "__main__":
    
    solver = CVX(nQubits=2)