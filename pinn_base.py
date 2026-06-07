

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time
import os
import csv
from scipy.interpolate import interp1d


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class NoisyDataGenerator:
   

    SUPPORTED = ['burgers', 'heat', 'wave', 'allen_cahn']

    def __init__(self, pde='burgers', nu=0.01/np.pi, alpha=0.01,
                 c=1.0, device=device):
        assert pde in self.SUPPORTED, f"PDE must be one of {self.SUPPORTED}"
        self.pde        = pde
        self.nu         = nu
        self.alpha      = alpha
        self.c          = c
        self.device     = device

    def _to_t(self, a):
        return torch.tensor(a, dtype=torch.float32).to(self.device)

    def _ic(self, x):
        """Initial condition u(x, 0) per PDE."""
        if self.pde == 'burgers':
            return -np.sin(np.pi * x)
        elif self.pde == 'heat':
            return np.sin(np.pi * x)
        elif self.pde == 'wave':
            return np.sin(np.pi * x)
        elif self.pde == 'allen_cahn':
            return x**2 * np.cos(np.pi * x)

    def _ic_dt(self, x):
      
        return np.zeros_like(x)

    def generate(self, N_ic=1000, N_bc=1000, N_f=8000, noise_eps=0.0):
       
        x_ic = np.random.uniform(-1, 1, (N_ic, 1))
        t_ic = np.zeros((N_ic, 1))
        u_ic = self._ic(x_ic)
        if noise_eps > 0:
            u_ic += noise_eps * np.random.randn(*u_ic.shape)

        t_bc       = np.random.uniform(0, 1, (N_bc, 1))
        half       = N_bc // 2
        x_bc_left  = -np.ones((half, 1))
        x_bc_right =  np.ones((half, 1))
        t_bc_left  = t_bc[:half]
        t_bc_right = t_bc[half:]
        u_bc_left  = np.zeros((half, 1))
        u_bc_right = np.zeros((half, 1))
        if noise_eps > 0:
            u_bc_left  += noise_eps * np.random.randn(*u_bc_left.shape)
            u_bc_right += noise_eps * np.random.randn(*u_bc_right.shape)

        x_f = np.random.uniform(-1, 1, (N_f, 1))
        t_f = np.random.uniform( 0, 1, (N_f, 1))

        data = {
            'x_ic': self._to_t(x_ic), 't_ic': self._to_t(t_ic),
            'u_ic': self._to_t(u_ic),
            'x_bc_left':  self._to_t(x_bc_left),
            't_bc_left':  self._to_t(t_bc_left),
            'u_bc_left':  self._to_t(u_bc_left),
            'x_bc_right': self._to_t(x_bc_right),
            't_bc_right': self._to_t(t_bc_right),
            'u_bc_right': self._to_t(u_bc_right),
            'x_f': self._to_t(x_f),
            't_f': self._to_t(t_f),
        }

        if self.pde == 'wave':
            u_ic_dt = self._ic_dt(x_ic)
            if noise_eps > 0:
                u_ic_dt += noise_eps * np.random.randn(*u_ic_dt.shape)
            data['u_ic_dt'] = self._to_t(u_ic_dt)

        return data

    def true_ic(self, x):
        """Return noiseless IC values for comparison plots."""
        return self._ic(x)


class CurriculumSampler:
    """
    4-stage residual-based curriculum sampler.

    Stage 1 (0–25%)   : easiest 25% of points by residual magnitude
    Stage 2 (25–50%)  : easiest 50%
    Stage 3 (50–75%)  : easiest 75%
    Stage 4 (75–100%) : full domain

    Points are re-sampled every `resample_every` epochs based on
    current model residuals → truly adaptive.
    """

    STAGE_THRESHOLDS = [0.25, 0.50, 0.75, 1.00]

    def __init__(self, N_pool=20000, resample_every=500, device=device):
        self.N_pool         = N_pool
        self.resample_every = resample_every
        self.device         = device

        # Pre-generate large candidate pool
        x_pool = np.random.uniform(-1, 1, (N_pool, 1))
        t_pool = np.random.uniform( 0, 1, (N_pool, 1))
        self.x_pool = torch.tensor(x_pool, dtype=torch.float32).to(device)
        self.t_pool = torch.tensor(t_pool, dtype=torch.float32).to(device)

    def get_stage(self, epoch, total_epochs):
        """Return current stage index (0–3) based on training progress."""
        progress = epoch / total_epochs
        for i, threshold in enumerate(self.STAGE_THRESHOLDS):
            if progress <= threshold:
                return i
        return 3

    def sample(self, model, pde_residual_fn, epoch, total_epochs, N_f=8000):
        """
        Sample N_f collocation points according to current curriculum stage.

        Parameters
        ----------
        model           : current PINN model
        pde_residual_fn : function(model, x, t) → residual tensor
        epoch           : current epoch
        total_epochs    : total training epochs
        N_f             : number of points to return
        """
        stage = self.get_stage(epoch, total_epochs)
        threshold = self.STAGE_THRESHOLDS[stage]

        if epoch % self.resample_every == 0:
            model.eval()
            with torch.enable_grad():
                x_p = self.x_pool.clone().requires_grad_(True)
                t_p = self.t_pool.clone().requires_grad_(True)
                res = pde_residual_fn(model, x_p, t_p)
                self._scores = torch.abs(res).detach().squeeze().cpu().numpy()
            model.train()

        if not hasattr(self, '_scores'):
            
            idx = np.random.choice(self.N_pool, N_f, replace=False)
        else:
           
            sorted_idx = np.argsort(self._scores)           
            cutoff     = int(threshold * self.N_pool)
            cutoff     = max(cutoff, N_f)                   
            eligible   = sorted_idx[:cutoff]
            idx        = np.random.choice(eligible, N_f, replace=False)

        return self.x_pool[idx], self.t_pool[idx]


class _PINNNet(nn.Module):
    """Shared MLP backbone used by both PINNSolver and ACPINNSolver."""

    def __init__(self, layers):
        super().__init__()
        self.activation = nn.Tanh()
        self.linears    = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.linears.append(nn.Linear(layers[i], layers[i+1]))
        self._init_weights()

    def _init_weights(self):
        for layer in self.linears:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x, t):
        a = torch.cat([x, t], dim=1)
        for linear in self.linears[:-1]:
            a = self.activation(linear(a))
        return self.linears[-1](a)


def burgers_residual(model, x, t, nu):
    x.requires_grad_(True); t.requires_grad_(True)
    u    = model(x, t)
    u_t  = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u),
                                create_graph=True, retain_graph=True)[0]
    u_x  = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                create_graph=True, retain_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x),
                                create_graph=True, retain_graph=True)[0]
    return u_t + u * u_x - nu * u_xx


def heat_residual(model, x, t, alpha):
    x.requires_grad_(True); t.requires_grad_(True)
    u    = model(x, t)
    u_t  = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u),
                                create_graph=True, retain_graph=True)[0]
    u_x  = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                create_graph=True, retain_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x),
                                create_graph=True, retain_graph=True)[0]
    return u_t - alpha * u_xx


def wave_residual(model, x, t, c):
    x.requires_grad_(True); t.requires_grad_(True)
    u    = model(x, t)
    u_t  = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u),
                                create_graph=True, retain_graph=True)[0]
    u_tt = torch.autograd.grad(u_t, t, grad_outputs=torch.ones_like(u_t),
                                create_graph=True, retain_graph=True)[0]
    u_x  = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                create_graph=True, retain_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x),
                                create_graph=True, retain_graph=True)[0]
    return u_tt - c**2 * u_xx


def allen_cahn_residual(model, x, t, epsilon):
    x.requires_grad_(True); t.requires_grad_(True)
    u    = model(x, t)
    u_t  = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u),
                                create_graph=True, retain_graph=True)[0]
    u_x  = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                create_graph=True, retain_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x),
                                create_graph=True, retain_graph=True)[0]
    return u_t - epsilon**2 * u_xx - u + u**3


PDE_RESIDUALS = {
    'burgers':    burgers_residual,
    'heat':       heat_residual,
    'wave':       wave_residual,
    'allen_cahn': allen_cahn_residual,
}

PDE_PARAMS = {
    'burgers':    {'nu': 0.01/np.pi},
    'heat':       {'alpha': 0.01},
    'wave':       {'c': 1.0},
    'allen_cahn': {'epsilon': 0.01},
}


class PINNSolver(nn.Module):
    """
    Vanilla PINN solver. Supports all 4 PDEs.
    Loss weights are fixed throughout training.
    """

    def __init__(self, pde='burgers', layers=None, pde_params=None,
                 lambda_ic=1.0, lambda_bc=1.0, lambda_pde=5.0, device=device):
        super().__init__()
        self.pde        = pde
        self.device     = device
        self.lambda_ic  = lambda_ic
        self.lambda_bc  = lambda_bc
        self.lambda_pde = lambda_pde
        self.pde_params = pde_params or PDE_PARAMS[pde]

        if layers is None:
            layers = [2, 64, 64, 64, 64, 1]
        self.network  = _PINNNet(layers).to(device)
        self.residual_fn = PDE_RESIDUALS[pde]

    def forward(self, x, t):
        return self.network(x, t)

    def _compute_loss(self, data, x_f=None, t_f=None):
        mse = nn.MSELoss()

        # IC loss
        u_pred_ic = self.network(data['x_ic'], data['t_ic'])
        loss_ic   = mse(u_pred_ic, data['u_ic'])

        # BC loss
        u_pred_left  = self.network(data['x_bc_left'],  data['t_bc_left'])
        u_pred_right = self.network(data['x_bc_right'], data['t_bc_right'])
        loss_bc = mse(u_pred_left,  data['u_bc_left']) + \
                  mse(u_pred_right, data['u_bc_right'])

        # Wave: velocity IC
        if self.pde == 'wave' and 'u_ic_dt' in data:
            x_ic_ = data['x_ic'].clone().requires_grad_(True)
            t_ic_ = data['t_ic'].clone().requires_grad_(True)
            u_    = self.network(x_ic_, t_ic_)
            u_t_  = torch.autograd.grad(u_, t_ic_,
                        grad_outputs=torch.ones_like(u_),
                        create_graph=True)[0]
            loss_ic = loss_ic + mse(u_t_, data['u_ic_dt'])

        # PDE residual loss
        xf = data['x_f'] if x_f is None else x_f
        tf = data['t_f'] if t_f is None else t_f
        xf = xf.clone().requires_grad_(True)
        tf = tf.clone().requires_grad_(True)
        f_pred   = self.residual_fn(self.network, xf, tf, **self.pde_params)
        loss_pde = mse(f_pred, torch.zeros_like(f_pred))

        total = self.lambda_ic * loss_ic + \
                self.lambda_bc * loss_bc + \
                self.lambda_pde * loss_pde
        return total, loss_ic, loss_bc, loss_pde

    def fit(self, data, epochs=10000, lr=1e-3, print_every=500, label=''):
        optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=3000, gamma=0.5)
        history = {'total': [], 'ic': [], 'bc': [], 'pde': []}
        start   = time.time()

        for epoch in range(epochs):
            self.network.train()
            optimizer.zero_grad()
            total, l_ic, l_bc, l_pde = self._compute_loss(data)
            total.backward()
            optimizer.step()
            scheduler.step()

            history['total'].append(total.item())
            history['ic'].append(l_ic.item())
            history['bc'].append(l_bc.item())
            history['pde'].append(l_pde.item())

            if epoch % print_every == 0:
                prefix = f"{label} | " if label else ""
                print(f"[{prefix}Epoch {epoch:5d}/{epochs}] "
                      f"Total: {total.item():.6f} | "
                      f"IC: {l_ic.item():.6f} | BC: {l_bc.item():.6f} | "
                      f"PDE: {l_pde.item():.6f}")

        self.runtime = time.time() - start
        print(f"\nTraining complete in {self.runtime:.2f}s")
        return history

    def predict(self, x, t):
        self.network.eval()
        with torch.no_grad():
            x_t = torch.tensor(x, dtype=torch.float32).to(self.device)
            t_t = torch.tensor(t, dtype=torch.float32).to(self.device)
            return self.network(x_t, t_t).cpu().numpy()

    def predict_grid(self, nx=200, nt=100):
        x_v = np.linspace(-1, 1, nx)
        t_v = np.linspace( 0, 1, nt)
        X, T = np.meshgrid(x_v, t_v)
        x_t  = torch.tensor(X.flatten()[:, None], dtype=torch.float32).to(self.device)
        t_t  = torch.tensor(T.flatten()[:, None], dtype=torch.float32).to(self.device)
        self.network.eval()
        with torch.no_grad():
            U = self.network(x_t, t_t).cpu().numpy().reshape(T.shape)
        return X, T, U

    def plot_solution(self, title=None):
        X, T, U = self.predict_grid()
        plt.figure(figsize=(8, 5))
        c = plt.contourf(X, T, U, levels=100, cmap='jet')
        plt.colorbar(c, label='u(x,t)')
        plt.xlabel('x'); plt.ylabel('t')
        plt.title(title or f'PINN Solution — {self.pde}')
        plt.tight_layout(); plt.show()

    def plot_loss_history(self, history):
        plt.figure(figsize=(10, 5))
        for key, vals in history.items():
            plt.plot(vals, label=key.upper() + ' Loss')
        plt.yscale('log'); plt.xlabel('Epoch'); plt.ylabel('Loss')
        plt.title('Training Loss Curves')
        plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

    def plot_initial_condition_comparison(self, generator):
        x_plot = np.linspace(-1, 1, 200)[:, None]
        t_plot = np.zeros_like(x_plot)
        u_pred = self.predict(x_plot, t_plot)
        u_true = generator.true_ic(x_plot)

        plt.figure(figsize=(8, 4))
        plt.plot(x_plot, u_true, lw=2, label='True IC')
        plt.plot(x_plot, u_pred, '--', lw=2, label='PINN Prediction')
        plt.xlabel('x'); plt.ylabel('u(x,0)')
        plt.title('Initial Condition Comparison')
        plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()


class ACPINNSolver(PINNSolver):


    def __init__(self, pde='burgers', layers=None, pde_params=None,
                 lambda_ic=1.0, lambda_bc=1.0, lambda_pde=5.0,
                 weight_strategy='gradient',
                 N_pool=20000, resample_every=500,
                 device=device):
        super().__init__(pde=pde, layers=layers, pde_params=pde_params,
                         lambda_ic=lambda_ic, lambda_bc=lambda_bc,
                         lambda_pde=lambda_pde, device=device)
        assert weight_strategy in ['gradient', 'ratio', 'both']
        self.weight_strategy = weight_strategy
        self.sampler = CurriculumSampler(
            N_pool=N_pool, resample_every=resample_every, device=device)

    def _update_weights_ratio(self, l_ic, l_bc, l_pde):
     
        with torch.no_grad():
            total = l_ic.item() + l_bc.item() + l_pde.item() + 1e-10
            self.lambda_ic  = float(l_ic.item()  / total * 3.0)
            self.lambda_bc  = float(l_bc.item()  / total * 3.0)
            self.lambda_pde = float(l_pde.item() / total * 3.0)
            # Clamp to reasonable range
            self.lambda_ic  = np.clip(self.lambda_ic,  0.1, 10.0)
            self.lambda_bc  = np.clip(self.lambda_bc,  0.1, 10.0)
            self.lambda_pde = np.clip(self.lambda_pde, 0.1, 10.0)

    def _update_weights_gradient_full(self, l_ic, l_bc, l_pde, optimizer):
        
        last_layer_params = list(self.network.linears[-1].parameters())

        def safe_grad_norm(scalar_loss):
            try:
                grads = torch.autograd.grad(
                    scalar_loss, last_layer_params,
                    retain_graph=True, allow_unused=True
                )
                total_norm = sum(
                    g.norm(2).item()**2 for g in grads if g is not None
                )
                return total_norm**0.5 + 1e-10
            except Exception:
                return 1.0

        n_ic    = safe_grad_norm(l_ic)
        n_bc    = safe_grad_norm(l_bc)
        n_pde   = safe_grad_norm(l_pde)
        total_n = n_ic + n_bc + n_pde

        self.lambda_ic  = float(np.clip(total_n / n_ic  * 1.0, 0.1, 10.0))
        self.lambda_bc  = float(np.clip(total_n / n_bc  * 1.0, 0.1, 10.0))
        self.lambda_pde = float(np.clip(total_n / n_pde * 1.0, 0.1, 10.0))

    def fit(self, data, epochs=10000, lr=1e-3, print_every=500,
            weight_update_every=200, label=''):
        optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=3000, gamma=0.5)

        history = {
            'total': [], 'ic': [], 'bc': [], 'pde': [],
            'lambda_ic': [], 'lambda_bc': [], 'lambda_pde': [],
            'stage': []
        }
        start = time.time()

        for epoch in range(epochs):
            self.network.train()
            optimizer.zero_grad()

           
            stage = self.sampler.get_stage(epoch, epochs)

            def _res_fn(model, x, t):
                return self.residual_fn(model, x, t, **self.pde_params)

            x_f, t_f = self.sampler.sample(
                self.network, _res_fn, epoch, epochs,
                N_f=data['x_f'].shape[0]
            )

          
            total, l_ic, l_bc, l_pde = self._compute_loss(data, x_f, t_f)

            if epoch % weight_update_every == 0 and epoch > 0:
                use_gradient = (
                    self.weight_strategy == 'gradient' or
                    (self.weight_strategy == 'both' and stage < 2)
                )
                use_ratio = (
                    self.weight_strategy == 'ratio' or
                    (self.weight_strategy == 'both' and stage >= 2)
                )
                if use_gradient:
                    self._update_weights_gradient_full(l_ic, l_bc, l_pde, optimizer)
                  
                    optimizer.zero_grad()
                    x_f_new = x_f.detach().clone()
                    t_f_new = t_f.detach().clone()
                    total, l_ic, l_bc, l_pde = self._compute_loss(data, x_f_new, t_f_new)
                elif use_ratio:
                    self._update_weights_ratio(l_ic, l_bc, l_pde)
                    total = self.lambda_ic  * l_ic + \
                            self.lambda_bc  * l_bc + \
                            self.lambda_pde * l_pde

            total.backward()
            optimizer.step()
            scheduler.step()

            history['total'].append(total.item())
            history['ic'].append(l_ic.item())
            history['bc'].append(l_bc.item())
            history['pde'].append(l_pde.item())
            history['lambda_ic'].append(self.lambda_ic)
            history['lambda_bc'].append(self.lambda_bc)
            history['lambda_pde'].append(self.lambda_pde)
            history['stage'].append(stage)

            if epoch % print_every == 0:
                prefix = f"{label} | " if label else ""
                print(f"[{prefix}Epoch {epoch:5d}/{epochs}] Stage {stage+1}/4 | "
                      f"Total: {total.item():.6f} | "
                      f"IC: {l_ic.item():.6f} | BC: {l_bc.item():.6f} | "
                      f"PDE: {l_pde.item():.6f} | "
                      f"λ=({self.lambda_ic:.2f},{self.lambda_bc:.2f},{self.lambda_pde:.2f})")

        self.runtime = time.time() - start
        print(f"\nAC-PINN training complete in {self.runtime:.2f}s")
        return history

    def plot_weight_history(self, history):
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        axes[0].plot(history['lambda_ic'],  label='λ_ic')
        axes[0].plot(history['lambda_bc'],  label='λ_bc')
        axes[0].plot(history['lambda_pde'], label='λ_pde')
        axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Weight value')
        axes[0].set_title('Adaptive Loss Weights Over Training')
        axes[0].legend(); axes[0].grid(True)

        stages = np.array(history['stage'])
        axes[1].plot(stages + 1)
        axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Curriculum Stage')
        axes[1].set_title('Curriculum Stage Progression')
        axes[1].set_yticks([1, 2, 3, 4])
        axes[1].grid(True)

        plt.tight_layout(); plt.show()



class _BaseFDM:

    def __init__(self, nx, nt):
        self.nx = nx; self.nt = nt
        self.x  = np.linspace(-1, 1, nx)
        self.dx = self.x[1] - self.x[0]
        self.dt = 1.0 / nt
        self.u  = None
        self.runtime = None

    def _thomas(self, lower, main, upper, rhs):
        n = len(rhs); c = np.zeros(n); d = rhs.copy()
        c[0] = upper[0] / main[0]; d[0] /= main[0]
        for i in range(1, n):
            denom = main[i] - lower[i-1] * c[i-1]
            c[i]  = upper[i] / denom if i < n-1 else 0.0
            d[i]  = (d[i] - lower[i-1] * d[i-1]) / denom
        x = np.zeros(n); x[-1] = d[-1]
        for i in range(n-2, -1, -1):
            x[i] = d[i] - c[i] * x[i+1]
        return x

    def get_solution_at_time(self, t):
        idx = int(np.clip(t * self.nt, 0, self.nt-1))
        return self.x, self.u[idx]

    def plot_solution(self, title='FDM Solution'):
        T_v = np.linspace(0, 1, self.nt)
        X, T = np.meshgrid(self.x, T_v)
        plt.figure(figsize=(8, 5))
        c = plt.contourf(X, T, self.u, levels=100, cmap='jet')
        plt.colorbar(c, label='u(x,t)')
        plt.xlabel('x'); plt.ylabel('t')
        plt.title(title); plt.tight_layout(); plt.show()

    def plot_time_slices(self, t_values=[0.0, 0.25, 0.5, 0.75, 1.0]):
        plt.figure(figsize=(8, 4))
        for t in t_values:
            x, u = self.get_solution_at_time(t)
            plt.plot(x, u, label=f't={t}')
        plt.xlabel('x'); plt.ylabel('u'); plt.legend()
        plt.grid(True); plt.tight_layout(); plt.show()


class BurgersFDM(_BaseFDM):

    def __init__(self, nx=256, nt=2000, nu=0.01/np.pi):
        super().__init__(nx, nt)
        self.nu = nu

    def solve(self):
        start = time.time()
        dx, dt, nu = self.dx, self.dt, self.nu
        r  = nu * dt / dx**2
        N  = self.nx - 2
        md = np.ones(N) * (1.0 + r)
        up = np.ones(N-1) * (-r/2)
        lo = np.ones(N-1) * (-r/2)
        u  = np.zeros((self.nt, self.nx))
        u[0] = -np.sin(np.pi * self.x)

        for n in range(self.nt - 1):
            u_n = u[n]
            pos = u_n[1:-1] > 0
            adv = np.where(pos,
                           u_n[1:-1]*(u_n[1:-1]-u_n[:-2])/dx,
                           u_n[1:-1]*(u_n[2:]-u_n[1:-1])/dx)
            diff_exp = (r/2)*(u_n[:-2] - 2*u_n[1:-1] + u_n[2:])
            rhs = u_n[1:-1] - dt*adv + diff_exp
            u[n+1, 1:-1] = self._thomas(lo, md, up, rhs)
            u[n+1, 0] = u[n+1, -1] = 0.0

        self.u = u; self.runtime = time.time()-start
        print(f'BurgersFDM solved in {self.runtime:.4f}s')
        return u


class HeatFDM(_BaseFDM):

    def __init__(self, nx=256, nt=1000, alpha=0.01):
        super().__init__(nx, nt)
        self.alpha = alpha

    def solve(self):
        start = time.time()
        dx, dt, alpha = self.dx, self.dt, self.alpha
        r  = alpha * dt / dx**2
        N  = self.nx - 2
        md = np.ones(N) * (1.0 + r)
        up = np.ones(N-1) * (-r/2)
        lo = np.ones(N-1) * (-r/2)
        u  = np.zeros((self.nt, self.nx))
        u[0] = np.sin(np.pi * self.x)

        for n in range(self.nt - 1):
            u_n  = u[n]
            diff = (r/2)*(u_n[:-2] - 2*u_n[1:-1] + u_n[2:])
            rhs  = u_n[1:-1] + diff
            u[n+1, 1:-1] = self._thomas(lo, md, up, rhs)
            u[n+1, 0] = u[n+1, -1] = 0.0

        self.u = u; self.runtime = time.time()-start
        print(f'HeatFDM solved in {self.runtime:.4f}s')
        return u


class WaveFDM(_BaseFDM):

    def __init__(self, nx=256, nt=2000, c=1.0):
        super().__init__(nx, nt)
        self.c = c

    def solve(self):
        start  = time.time()
        dx, dt, c = self.dx, self.dt, self.c
        r = (c * dt / dx)**2
        assert r <= 1.0, f"CFL violated: r={r:.3f}. Increase nt."
        u = np.zeros((self.nt, self.nx))
        u[0] = np.sin(np.pi * self.x)
        # First step using zero velocity IC
        u[1, 1:-1] = u[0, 1:-1] + (r/2)*(u[0, :-2]-2*u[0, 1:-1]+u[0, 2:])

        for n in range(1, self.nt - 1):
            u[n+1, 1:-1] = (2*u[n, 1:-1] - u[n-1, 1:-1] +
                            r*(u[n, :-2] - 2*u[n, 1:-1] + u[n, 2:]))
            u[n+1, 0] = u[n+1, -1] = 0.0

        self.u = u; self.runtime = time.time()-start
        print(f'WaveFDM solved in {self.runtime:.4f}s')
        return u


class AllenCahnFDM(_BaseFDM):

    def __init__(self, nx=256, nt=5000, epsilon=0.01):
        super().__init__(nx, nt)
        self.epsilon = epsilon

    def solve(self):
        start = time.time()
        dx, dt, eps = self.dx, self.dt, self.epsilon
        r  = eps**2 * dt / dx**2
        N  = self.nx - 2
        md = np.ones(N) * (1.0 + r)
        up = np.ones(N-1) * (-r/2)
        lo = np.ones(N-1) * (-r/2)
        u  = np.zeros((self.nt, self.nx))
        u[0] = (self.x)**2 * np.cos(np.pi * self.x)

        for n in range(self.nt - 1):
            u_n     = u[n]
            react   = dt * (u_n[1:-1] - u_n[1:-1]**3)
            diff_ex = (r/2)*(u_n[:-2] - 2*u_n[1:-1] + u_n[2:])
            rhs     = u_n[1:-1] + react + diff_ex
            u[n+1, 1:-1] = self._thomas(lo, md, up, rhs)
            u[n+1, 0] = u[n+1, -1] = 0.0

        self.u = u; self.runtime = time.time()-start
        print(f'AllenCahnFDM solved in {self.runtime:.4f}s')
        return u


FDM_SOLVERS = {
    'burgers':    BurgersFDM,
    'heat':       HeatFDM,
    'wave':       WaveFDM,
    'allen_cahn': AllenCahnFDM,
}



class Benchmark:

    def __init__(self, fdm_solver, nx=200, nt=100):
        self.fdm   = fdm_solver
        self.nx    = nx
        self.nt    = nt
        self.models = {}   # name → model
        self.U_preds = {}  # name → U array
        self.X = self.T = self.U_fdm = None

    def add(self, name, model):
        self.models[name] = model
        return self

    def run(self):
        x_v = np.linspace(-1, 1, self.nx)
        t_v = np.linspace( 0, 1, self.nt)
        X, T = np.meshgrid(x_v, t_v)
        self.X, self.T = X, T

        # FDM reference — interpolate if resolution differs
        step  = max(1, self.fdm.nt // self.nt)
        U_fdm = self.fdm.u[::step, :][:self.nt, :]
        if U_fdm.shape[1] != self.nx:
            f     = interp1d(self.fdm.x, U_fdm, axis=1, kind='linear')
            U_fdm = f(x_v)
        self.U_fdm = U_fdm

        # Evaluate each model
        x_t = torch.tensor(X.flatten()[:, None], dtype=torch.float32).to(device)
        t_t = torch.tensor(T.flatten()[:, None], dtype=torch.float32).to(device)
        for name, model in self.models.items():
            model.network.eval()
            with torch.no_grad():
                U = model.network(x_t, t_t).cpu().numpy().reshape(T.shape)
            self.U_preds[name] = U
        return self

    def compare_metrics(self):
        print('=' * 65)
        print(f"  {'Model':<25} {'Rel L2':>10} {'Max Err':>10} {'MAE':>10} {'RMSE':>10}")
        print('=' * 65)
        results = {}
        for name, U_pred in self.U_preds.items():
            diff  = U_pred - self.U_fdm
            l2    = np.linalg.norm(diff) / (np.linalg.norm(self.U_fdm) + 1e-10)
            max_e = np.max(np.abs(diff))
            mae   = np.mean(np.abs(diff))
            rmse  = np.sqrt(np.mean(diff**2))
            print(f"  {name:<25} {l2:>10.6f} {max_e:>10.6f} {mae:>10.6f} {rmse:>10.6f}")
            results[name] = {'l2': l2, 'max_error': max_e, 'mae': mae, 'rmse': rmse}
        print('=' * 65)
        return results

    def plot_comparison(self, save_path=None):
        n_models = len(self.U_preds)
        n_cols   = n_models + 2  # FDM + models + error cols
        vmin = min(self.U_fdm.min(), min(u.min() for u in self.U_preds.values()))
        vmax = max(self.U_fdm.max(), max(u.max() for u in self.U_preds.values()))
        kw   = dict(levels=100, cmap='jet', vmin=vmin, vmax=vmax)

        fig, axes = plt.subplots(2, n_models+1, figsize=(5*(n_models+1), 8))

        # FDM
        axes[0, 0].contourf(self.X, self.T, self.U_fdm, **kw)
        axes[0, 0].set_title('FDM (Ground Truth)')
        axes[1, 0].axis('off')

        for col, (name, U_pred) in enumerate(self.U_preds.items(), 1):
            c = axes[0, col].contourf(self.X, self.T, U_pred, **kw)
            axes[0, col].set_title(name)
            err = axes[1, col].contourf(self.X, self.T,
                                         np.abs(U_pred - self.U_fdm),
                                         levels=100, cmap='hot_r')
            axes[1, col].set_title(f'|{name} - FDM|')
            fig.colorbar(err, ax=axes[1, col])

        for ax in axes.flat:
            ax.set_xlabel('x'); ax.set_ylabel('t')

        plt.suptitle('Model Comparison vs FDM Ground Truth', fontsize=13)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

    def plot_time_slices(self, t_values=[0.25, 0.5, 0.75, 1.0], save_path=None):
        fig, axes = plt.subplots(1, len(t_values), figsize=(4*len(t_values), 4))
        if len(t_values) == 1: axes = [axes]
        x = self.X[0, :]

        for ax, t in zip(axes, t_values):
            idx = int(np.clip(t * self.nt, 0, self.nt-1))
            ax.plot(x, self.U_fdm[idx], lw=2.5, label='FDM', zorder=5)
            for name, U_pred in self.U_preds.items():
                ax.plot(x, U_pred[idx], '--', lw=1.8, label=name)
            ax.set_title(f't = {t}'); ax.set_xlabel('x'); ax.set_ylabel('u')
            ax.legend(fontsize=8); ax.grid(True)

        plt.suptitle('Time Slice Comparison', fontsize=13)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

    def plot_error_over_time(self, save_path=None):
        t_vals = self.T[:, 0]
        plt.figure(figsize=(9, 4))
        for name, U_pred in self.U_preds.items():
            l2_t = np.linalg.norm(U_pred - self.U_fdm, axis=1) / \
                   (np.linalg.norm(self.U_fdm, axis=1) + 1e-10)
            plt.plot(t_vals, l2_t, lw=2, label=name)
        plt.xlabel('t'); plt.ylabel('Relative L2 error')
        plt.title('L2 Error Over Time vs FDM')
        plt.legend(); plt.grid(True); plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()


def save_metrics(metrics, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, metrics)
    print(f'Saved: {path}')

def load_metrics(path):
    return np.load(path, allow_pickle=True).item()

def save_history(history, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, history)
    print(f'Saved: {path}')

def load_history(path):
    return np.load(path, allow_pickle=True).item()


def save_training_plots(history, save_path, label=''):
    """Plot loss/weight/curriculum curves for a training run, save PNG + CSV."""
    is_ac = 'lambda_ic' in history

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # Top left: total loss
    axes[0, 0].plot(history['total'])
    axes[0, 0].set_yscale('log')
    axes[0, 0].set_xlabel('Epoch'); axes[0, 0].set_ylabel('Total Loss')
    axes[0, 0].set_title('Total Loss'); axes[0, 0].grid(True)

    # Top right: IC + BC + PDE loss components
    axes[0, 1].plot(history['ic'],  label='IC')
    axes[0, 1].plot(history['bc'],  label='BC')
    axes[0, 1].plot(history['pde'], label='PDE')
    axes[0, 1].set_yscale('log')
    axes[0, 1].set_xlabel('Epoch'); axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Loss Components')
    axes[0, 1].legend(); axes[0, 1].grid(True)

    # Bottom left: adaptive loss weights (AC-PINN only)
    if is_ac:
        axes[1, 0].plot(history['lambda_ic'],  label='λ_ic')
        axes[1, 0].plot(history['lambda_bc'],  label='λ_bc')
        axes[1, 0].plot(history['lambda_pde'], label='λ_pde')
        axes[1, 0].set_xlabel('Epoch'); axes[1, 0].set_ylabel('Weight value')
        axes[1, 0].set_title('Adaptive Loss Weights')
        axes[1, 0].legend(); axes[1, 0].grid(True)
    else:
        axes[1, 0].axis('off')
        axes[1, 0].text(0.5, 0.5, 'N/A - Vanilla PINN', ha='center', va='center', fontsize=12)

    # Bottom right: curriculum stage progression (AC-PINN only)
    if is_ac:
        stages = np.array(history['stage'])
        axes[1, 1].plot(stages + 1)
        axes[1, 1].set_xlabel('Epoch'); axes[1, 1].set_ylabel('Curriculum Stage')
        axes[1, 1].set_title('Curriculum Stage Progression')
        axes[1, 1].set_yticks([1, 2, 3, 4])
        axes[1, 1].grid(True)
    else:
        axes[1, 1].axis('off')
        axes[1, 1].text(0.5, 0.5, 'N/A - Vanilla PINN', ha='center', va='center', fontsize=12)

    fig.suptitle(label, fontsize=14)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

    # CSV with raw numbers
    csv_path = os.path.splitext(save_path)[0] + '.csv'
    fieldnames = ['epoch', 'total', 'ic', 'bc', 'pde']
    if is_ac:
        fieldnames += ['lambda_ic', 'lambda_bc', 'lambda_pde', 'stage']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        for i in range(len(history['total'])):
            row = [i, history['total'][i], history['ic'][i],
                   history['bc'][i], history['pde'][i]]
            if is_ac:
                row += [history['lambda_ic'][i], history['lambda_bc'][i],
                        history['lambda_pde'][i], history['stage'][i]]
            writer.writerow(row)

    print(f'Saved: {save_path}')
    print(f'Saved: {csv_path}')
