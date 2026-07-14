

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time
import os
import csv
import random
from scipy.interpolate import interp1d


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed):
    """Set all relevant RNG seeds (random, numpy, torch, CUDA) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# If PINN_SEED is set in the environment, seed everything at import time.
# This lets an external driver (e.g. multi_seed_run.py) reproduce a given
# seed for an entire notebook run without having to modify the notebook.
_env_seed = os.environ.get('PINN_SEED')
if _env_seed is not None:
    set_seed(int(_env_seed))


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
    n-stage residual-based curriculum sampler (4 stages by default).

    With n_stages=4 (default):
    Stage 1 (0–25%)   : easiest 25% of points by residual magnitude
    Stage 2 (25–50%)  : easiest 50%
    Stage 3 (50–75%)  : easiest 75%
    Stage 4 (75–100%) : full domain

    In general, the n_stages difficulty bands and fixed-epoch fractions
    are split equally: [1/n_stages, 2/n_stages, ..., 1.0].

    Points are re-sampled every `resample_every` epochs based on
    current model residuals → truly adaptive.

    Stage ADVANCEMENT (i.e. when to move from one difficulty band to the
    next) can work in two modes:

    - dynamic_curriculum=True (default): a stage advances once the mean
      PDE residual over the last `N_window` collocation points drops
      below `stage_thresholds[stage] * initial_residual`, where
      `initial_residual` is the mean residual measured at the start of
      training. Each stage must also last at least `min_epochs_per_stage`
      epochs before it's allowed to advance. If the residual never drops
      enough, the sampler falls back to the fixed epoch fractions of
      `total_epochs` so training can't stall in an early stage forever.
      `stage_thresholds` must have `n_stages - 1` entries (one per
      possible transition); if not given, a default decreasing sequence
      from 0.5 to 0.05 is generated automatically.
    - dynamic_curriculum=False: stages advance purely at the fixed epoch
      fractions, matching the original behavior.
    """

    def __init__(self, N_pool=20000, resample_every=500, device=device,
                 dynamic_curriculum=True, stage_thresholds=None,
                 min_epochs_per_stage=500, N_window=200, n_stages=4):
        assert n_stages >= 2, "n_stages must be at least 2"
        self.N_pool         = N_pool
        self.resample_every = resample_every
        self.device         = device

        self.dynamic_curriculum = dynamic_curriculum
        self.n_stages           = n_stages
        # Difficulty cutoff (fraction of pool) and fixed-mode / fallback
        # epoch fractions, split equally across n_stages: [1/n, ..., 1.0]
        self.STAGE_THRESHOLDS  = [ (i + 1) / n_stages for i in range(n_stages) ]
        self.FIXED_EPOCH_FRACS = list(self.STAGE_THRESHOLDS)

        if stage_thresholds is None:
            stage_thresholds = list(np.geomspace(0.5, 0.05, n_stages - 1))
        else:
            assert len(stage_thresholds) == n_stages - 1, (
                f"stage_thresholds must have n_stages-1={n_stages-1} "
                f"entries, got {len(stage_thresholds)}")
        self.stage_thresholds     = stage_thresholds
        self.min_epochs_per_stage = min_epochs_per_stage
        self.N_window             = N_window

        # Pre-generate large candidate pool
        x_pool = np.random.uniform(-1, 1, (N_pool, 1))
        t_pool = np.random.uniform( 0, 1, (N_pool, 1))
        self.x_pool = torch.tensor(x_pool, dtype=torch.float32).to(device)
        self.t_pool = torch.tensor(t_pool, dtype=torch.float32).to(device)

        # Dynamic stage-tracking state
        self._stage             = 0
        self._stage_start_epoch = 0
        self._initial_residual  = None
        self.stage_transitions  = []  # [{from_stage, to_stage, epoch, residual, fallback}, ...]

        # Resample-gap tracking (so plateau-triggered resampling never
        # fires more often than `resample_every`, and epoch 0 always
        # triggers an initial resample)
        self._last_resample_epoch = -resample_every

    def get_stage(self, epoch, total_epochs):
        """Return current stage index (0 to n_stages-1) based on training progress."""
        if not self.dynamic_curriculum:
            progress = epoch / total_epochs
            for i, frac in enumerate(self.FIXED_EPOCH_FRACS):
                if progress <= frac:
                    return i
            return self.n_stages - 1
        return self._stage

    def _maybe_advance_stage(self, epoch, total_epochs, mean_residual=None):
        """
        Residual-driven stage advancement with fixed-epoch fallback.

        `mean_residual=None` means "no fresh residual score this call" --
        used for the every-epoch fallback-only check (called
        unconditionally from ACPINNSolver.fit(), regardless of whether a
        resample happened this epoch) so the fixed-epoch fallback can
        never be delayed past a resample event. The residual-based check
        still only runs when a real `mean_residual` is supplied, i.e.
        from sample() right after residual scores are recomputed.
        """
        if self._stage >= self.n_stages - 1:
            return

        if mean_residual is not None and self._initial_residual is None:
            self._initial_residual = mean_residual + 1e-12

        epochs_in_stage = epoch - self._stage_start_epoch
        if epochs_in_stage < self.min_epochs_per_stage:
            return

        progress     = epoch / total_epochs
        fallback_due = progress >= self.FIXED_EPOCH_FRACS[self._stage]

        residual_met = False
        if mean_residual is not None and self._initial_residual is not None:
            threshold    = self.stage_thresholds[self._stage] * self._initial_residual
            residual_met = mean_residual < threshold

        if residual_met or fallback_due:
            self.stage_transitions.append({
                'from_stage': self._stage,
                'to_stage':   self._stage + 1,
                'epoch':      epoch,
                'residual':   float(mean_residual) if mean_residual is not None else None,
                'fallback':   bool(fallback_due and not residual_met),
            })
            self._stage += 1
            self._stage_start_epoch = epoch

    def should_resample(self, epoch, adaptive_resample=True, plateau_detected=False):
        """
        Decide whether residual scores should be recomputed this epoch.

        `resample_every` always acts as a minimum gap between resamples,
        regardless of mode:
        - adaptive_resample=True : resample when `plateau_detected` is True
          (as determined by the caller from its own loss history), but
          never more often than every `resample_every` epochs.
        - adaptive_resample=False: resample every `resample_every` epochs,
          matching the original fixed-interval behavior.
        """
        gap_ok = (epoch - self._last_resample_epoch) >= self.resample_every
        if not gap_ok:
            return False

        if adaptive_resample:
            do_resample = plateau_detected
        else:
            do_resample = (epoch % self.resample_every == 0)

        if do_resample:
            self._last_resample_epoch = epoch
        return do_resample

    def sample(self, model, pde_residual_fn, epoch, total_epochs, N_f=8000,
               force_resample=False):
        """
        Sample N_f collocation points according to current curriculum stage.

        Parameters
        ----------
        model           : current PINN model
        pde_residual_fn : function(model, x, t) → residual tensor
        epoch           : current epoch
        total_epochs    : total training epochs
        N_f             : number of points to return
        force_resample  : recompute residual scores this epoch regardless
                           of any internal schedule (caller-driven, e.g.
                           fixed-interval or loss-plateau detection)
        """
        if force_resample or not hasattr(self, '_scores'):
            model.eval()
            with torch.enable_grad():
                x_p = self.x_pool.clone().requires_grad_(True)
                t_p = self.t_pool.clone().requires_grad_(True)
                res = pde_residual_fn(model, x_p, t_p)
                abs_res = torch.abs(res).detach().squeeze()
                self._scores = abs_res.cpu().numpy()
            model.train()

            if self.dynamic_curriculum:
                mean_residual = float(abs_res[:self.N_window].mean().item())
                self._maybe_advance_stage(epoch, total_epochs, mean_residual)

        stage     = self.get_stage(epoch, total_epochs)
        threshold = self.STAGE_THRESHOLDS[stage]

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

    def fit(self, data, epochs=10000, lr=1e-3, print_every=500, label='', seed=None):
        if seed is not None:
            set_seed(seed)

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
        plt.title(title or f'PINN Solution - {self.pde}')
        plt.tight_layout(); plt.show()

    def plot_loss_history(self, history):
        plt.figure(figsize=(10, 5))
        for key in ['total', 'ic', 'bc', 'pde']:
            if key in history:
                plt.plot(history[key], label=key.upper() + ' Loss')
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
                 dynamic_curriculum=True,
                 stage_thresholds=None,
                 min_epochs_per_stage=500,
                 switch_eval_every=1000,
                 min_epochs_before_switch=2000,
                 n_stages=4,
                 device=device):
        super().__init__(pde=pde, layers=layers, pde_params=pde_params,
                         lambda_ic=lambda_ic, lambda_bc=lambda_bc,
                         lambda_pde=lambda_pde, device=device)
        assert weight_strategy in ['gradient', 'ratio', 'both', 'auto']
        self.weight_strategy = weight_strategy
        self.dynamic_curriculum = dynamic_curriculum
        if stage_thresholds is None and n_stages == 4:
            stage_thresholds = [0.5, 0.2, 0.05]  # preserve original default
        self.sampler = CurriculumSampler(
            N_pool=N_pool, resample_every=resample_every, device=device,
            dynamic_curriculum=dynamic_curriculum,
            stage_thresholds=stage_thresholds,
            min_epochs_per_stage=min_epochs_per_stage,
            n_stages=n_stages)

        # 'auto' mode: dynamically switch between 'ratio' and 'gradient'
        self.switch_eval_every        = switch_eval_every
        self.min_epochs_before_switch = min_epochs_before_switch
        self._current_auto_strategy   = 'ratio'
        self._strategy_history        = {'ratio': [], 'gradient': []}

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

    def _switch_auto_strategy(self, epoch, from_strategy, to_strategy, reason, history):
        self._current_auto_strategy = to_strategy
        history['weight_strategy_log'].append({
            'epoch':          epoch,
            'from_strategy':  from_strategy,
            'to_strategy':    to_strategy,
            'reason':         reason,
        })

    def _evaluate_auto_strategy(self, epoch, history):
        """
        Every `switch_eval_every` epochs (once `min_epochs_before_switch`
        has elapsed), compare the current strategy's loss reduction over
        the last `switch_eval_every` epochs against the other strategy's
        historical average reduction, and switch if the other strategy
        has performed better on average. If the other strategy has no
        recorded performance yet, switch to it once to gather a baseline.
        """
        window = history['total'][-self.switch_eval_every:]
        if len(window) < self.switch_eval_every:
            return

        current = self._current_auto_strategy
        other   = 'gradient' if current == 'ratio' else 'ratio'

        reduction = (window[0] - window[-1]) / self.switch_eval_every
        self._strategy_history[current].append(reduction)

        if not self._strategy_history[other]:
            reason = f"exploring '{other}' (no historical data yet)"
            self._switch_auto_strategy(epoch, current, other, reason, history)
            return

        other_avg = sum(self._strategy_history[other]) / len(self._strategy_history[other])
        if other_avg > reduction:
            reason = (f"'{other}' historical avg reduction {other_avg:.6g} "
                      f"> '{current}' last-window reduction {reduction:.6g}")
            self._switch_auto_strategy(epoch, current, other, reason, history)

    @staticmethod
    def _loss_plateau_detected(total_history, plateau_window, plateau_tol):
        """
        True once the moving average of total loss over the last
        `plateau_window` epochs improves by less than `plateau_tol`
        relative to the moving average over the `plateau_window` epochs
        before that.
        """
        if len(total_history) < 2 * plateau_window:
            return False
        recent   = total_history[-plateau_window:]
        previous = total_history[-2 * plateau_window:-plateau_window]
        prev_avg = sum(previous) / plateau_window
        curr_avg = sum(recent) / plateau_window
        improvement = prev_avg - curr_avg
        return improvement < plateau_tol

    def fit(self, data, epochs=10000, lr=1e-3, print_every=500,
            weight_update_every=200, label='',
            adaptive_resample=True, plateau_window=200, plateau_tol=1e-4,
            seed=None):
        if seed is not None:
            set_seed(seed)

        optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=3000, gamma=0.5)

        history = {
            'total': [], 'ic': [], 'bc': [], 'pde': [],
            'lambda_ic': [], 'lambda_bc': [], 'lambda_pde': [],
            'stage': [], 'stage_transitions': [], 'weight_strategy_log': []
        }
        start = time.time()

        for epoch in range(epochs):
            self.network.train()
            optimizer.zero_grad()

            if self.sampler.dynamic_curriculum:
                # Fallback-only check, every epoch, independent of whether
                # a resample happens this epoch -- so the fixed-epoch
                # fallback can never be delayed past a resample event.
                # (Residual-based advancement still only happens inside
                # sample() when a fresh residual score is computed.)
                self.sampler._maybe_advance_stage(epoch, epochs)

            stage = self.sampler.get_stage(epoch, epochs)

            if (self.weight_strategy == 'auto' and epoch > 0
                    and epoch >= self.min_epochs_before_switch
                    and epoch % self.switch_eval_every == 0):
                self._evaluate_auto_strategy(epoch, history)

            def _res_fn(model, x, t):
                return self.residual_fn(model, x, t, **self.pde_params)

            plateau_detected = adaptive_resample and self._loss_plateau_detected(
                history['total'], plateau_window, plateau_tol)
            do_resample = self.sampler.should_resample(
                epoch, adaptive_resample=adaptive_resample,
                plateau_detected=plateau_detected)

            x_f, t_f = self.sampler.sample(
                self.network, _res_fn, epoch, epochs,
                N_f=data['x_f'].shape[0], force_resample=do_resample
            )

          
            total, l_ic, l_bc, l_pde = self._compute_loss(data, x_f, t_f)

            if epoch % weight_update_every == 0 and epoch > 0:
                active_strategy = (
                    self._current_auto_strategy
                    if self.weight_strategy == 'auto' else self.weight_strategy
                )
                use_gradient = (
                    active_strategy == 'gradient' or
                    (active_strategy == 'both' and stage < 2)
                )
                use_ratio = (
                    active_strategy == 'ratio' or
                    (active_strategy == 'both' and stage >= 2)
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
                print(f"[{prefix}Epoch {epoch:5d}/{epochs}] "
                      f"Stage {stage+1}/{self.sampler.n_stages} | "
                      f"Total: {total.item():.6f} | "
                      f"IC: {l_ic.item():.6f} | BC: {l_bc.item():.6f} | "
                      f"PDE: {l_pde.item():.6f} | "
                      f"lam=({self.lambda_ic:.2f},{self.lambda_bc:.2f},{self.lambda_pde:.2f})")

        history['stage_transitions'] = self.sampler.stage_transitions

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
        axes[1].set_yticks(range(1, int(stages.max()) + 2))
        axes[1].grid(True)

        plt.tight_layout(); plt.show()

    def plot_curriculum_stages(self, history, data, save_path):
        """
        Scatter-plot the training collocation points over the (x, t)
        domain, colored by which curriculum stage first made each point
        eligible for sampling (stage 0 = easiest, n_stages-1 = hardest).

        Since the sampler only keeps its most recent residual snapshot
        (not a per-epoch history of which point entered at which epoch),
        "first introduced in" is reconstructed by re-scoring
        `data['x_f']`/`data['t_f']` with the final trained network and
        mapping each point's difficulty rank through the same
        `STAGE_THRESHOLDS` cutoffs used during training. The epoch/residual
        of each actual stage transition (from `history['stage_transitions']`)
        is annotated alongside the plot for context.
        """
        x_f = data['x_f'].detach().cpu().numpy().flatten()
        t_f = data['t_f'].detach().cpu().numpy().flatten()

        self.network.eval()
        x_t = data['x_f'].clone().requires_grad_(True)
        t_t = data['t_f'].clone().requires_grad_(True)
        with torch.enable_grad():
            res = self.residual_fn(self.network, x_t, t_t, **self.pde_params)
            scores = torch.abs(res).detach().cpu().numpy().flatten()
        self.network.train()

        n_points   = len(scores)
        n_stages   = self.sampler.n_stages
        thresholds = np.array(self.sampler.STAGE_THRESHOLDS)

        sorted_idx = np.argsort(scores)              # ascending difficulty
        rank       = np.empty(n_points, dtype=int)
        rank[sorted_idx] = np.arange(n_points)
        fracs = (rank + 1) / n_points

        point_stage = np.searchsorted(thresholds, fracs, side='left')
        point_stage = np.clip(point_stage, 0, n_stages - 1)

        fig, ax = plt.subplots(figsize=(9, 6))
        cmap = plt.get_cmap('viridis', n_stages)
        sc = ax.scatter(x_f, t_f, c=point_stage, cmap=cmap,
                         vmin=-0.5, vmax=n_stages - 0.5, s=10, alpha=0.7)
        cbar = fig.colorbar(sc, ax=ax, ticks=range(n_stages))
        cbar.set_label(f'Curriculum stage first introduced '
                        f'(0=easy .. {n_stages-1}=hard)')

        transitions = history.get('stage_transitions', [])
        if transitions:
            lines = []
            for t in transitions:
                residual_str = (f"{t['residual']:.4g}" if t.get('residual') is not None
                                 else 'N/A')
                fallback_str = ', fallback' if t.get('fallback') else ''
                lines.append(
                    f"Stage {t['from_stage']}→{t['to_stage']} @ epoch {t['epoch']} "
                    f"(residual={residual_str}{fallback_str})"
                )
            fig.text(0.99, 0.02, '\n'.join(lines), fontsize=7,
                     ha='right', va='bottom', transform=fig.transFigure)

        ax.set_xlabel('x'); ax.set_ylabel('t')
        ax.set_title('Curriculum Stage Introduction Map')
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f'Saved: {save_path}')


class SAWeightPINN(PINNSolver):
    """
    Self-Adaptive PINN (SA-PINN), Wang et al. 2022.

    The IC/BC/PDE loss-term weights are trainable parameters (softmax
    normalized so they always sum to 3, matching the default
    lambda_ic + lambda_bc + lambda_pde = 3 scale of the vanilla PINN).
    The network minimizes the weighted loss while the weights themselves
    are updated by GRADIENT ASCENT on that same loss -- an adversarial
    min-max game where terms the network is fitting poorly get pushed to
    higher weight automatically.
    """

    def __init__(self, pde='burgers', layers=None, pde_params=None,
                 lambda_ic=1.0, lambda_bc=1.0, lambda_pde=5.0,
                 weight_lr=1e-3, device=device):
        super().__init__(pde=pde, layers=layers, pde_params=pde_params,
                          lambda_ic=lambda_ic, lambda_bc=lambda_bc,
                          lambda_pde=lambda_pde, device=device)
        self.weight_lr = weight_lr
        init_logits = torch.log(torch.tensor(
            [lambda_ic, lambda_bc, lambda_pde], dtype=torch.float32))
        self.raw_weights = nn.Parameter(init_logits.to(device))

    def _softmax_weights(self):
        w = torch.softmax(self.raw_weights, dim=0) * 3.0
        return w[0], w[1], w[2]

    def _compute_loss(self, data, x_f=None, t_f=None):
        mse = nn.MSELoss()

        u_pred_ic = self.network(data['x_ic'], data['t_ic'])
        loss_ic   = mse(u_pred_ic, data['u_ic'])

        u_pred_left  = self.network(data['x_bc_left'],  data['t_bc_left'])
        u_pred_right = self.network(data['x_bc_right'], data['t_bc_right'])
        loss_bc = mse(u_pred_left,  data['u_bc_left']) + \
                  mse(u_pred_right, data['u_bc_right'])

        if self.pde == 'wave' and 'u_ic_dt' in data:
            x_ic_ = data['x_ic'].clone().requires_grad_(True)
            t_ic_ = data['t_ic'].clone().requires_grad_(True)
            u_    = self.network(x_ic_, t_ic_)
            u_t_  = torch.autograd.grad(u_, t_ic_,
                        grad_outputs=torch.ones_like(u_),
                        create_graph=True)[0]
            loss_ic = loss_ic + mse(u_t_, data['u_ic_dt'])

        xf = data['x_f'] if x_f is None else x_f
        tf = data['t_f'] if t_f is None else t_f
        xf = xf.clone().requires_grad_(True)
        tf = tf.clone().requires_grad_(True)
        f_pred   = self.residual_fn(self.network, xf, tf, **self.pde_params)
        loss_pde = mse(f_pred, torch.zeros_like(f_pred))

        w_ic, w_bc, w_pde = self._softmax_weights()
        self.lambda_ic  = float(w_ic.item())
        self.lambda_bc  = float(w_bc.item())
        self.lambda_pde = float(w_pde.item())

        total = w_ic * loss_ic + w_bc * loss_bc + w_pde * loss_pde
        return total, loss_ic, loss_bc, loss_pde

    def fit(self, data, epochs=10000, lr=1e-3, print_every=500, label=''):
        optimizer_net = torch.optim.Adam(self.network.parameters(), lr=lr)
        optimizer_w   = torch.optim.Adam([self.raw_weights], lr=self.weight_lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer_net, step_size=3000, gamma=0.5)
        history = {
            'total': [], 'ic': [], 'bc': [], 'pde': [],
            'lambda_ic': [], 'lambda_bc': [], 'lambda_pde': []
        }
        start = time.time()

        for epoch in range(epochs):
            self.network.train()
            optimizer_net.zero_grad()
            optimizer_w.zero_grad()

            total, l_ic, l_bc, l_pde = self._compute_loss(data)
            total.backward()

            # Network descends the weighted loss...
            optimizer_net.step()
            # ...while the weight logits ascend it (adversarial min-max):
            # flip the sign of the gradient the shared backward() already
            # computed for raw_weights, then step with a normal (descent)
            # optimizer.
            self.raw_weights.grad.neg_()
            optimizer_w.step()

            scheduler.step()

            history['total'].append(total.item())
            history['ic'].append(l_ic.item())
            history['bc'].append(l_bc.item())
            history['pde'].append(l_pde.item())
            history['lambda_ic'].append(self.lambda_ic)
            history['lambda_bc'].append(self.lambda_bc)
            history['lambda_pde'].append(self.lambda_pde)

            if epoch % print_every == 0:
                prefix = f"{label} | " if label else ""
                print(f"[{prefix}Epoch {epoch:5d}/{epochs}] "
                      f"Total: {total.item():.6f} | "
                      f"IC: {l_ic.item():.6f} | BC: {l_bc.item():.6f} | "
                      f"PDE: {l_pde.item():.6f} | "
                      f"w=({self.lambda_ic:.2f},{self.lambda_bc:.2f},{self.lambda_pde:.2f})")

        self.runtime = time.time() - start
        print(f"\nSA-PINN training complete in {self.runtime:.2f}s")
        return history


class RARPINNSolver(PINNSolver):
    """
    RAR-PINN (Residual-based Adaptive Refinement), Lu et al. 2021 (DeepXDE).

    Loss weights are fixed, exactly like the vanilla PINN. Every
    `rar_update_every` epochs, the PDE residual is evaluated on a fresh
    dense candidate grid (`rar_pool_size` random points) and the
    `rar_k` highest-residual points are ADDED to the collocation set --
    points are only appended, never replaced or removed, up to a cap of
    `rar_max_points`.
    """

    def __init__(self, pde='burgers', layers=None, pde_params=None,
                 lambda_ic=1.0, lambda_bc=1.0, lambda_pde=5.0,
                 rar_update_every=1000, rar_k=50, rar_pool_size=10000,
                 rar_max_points=20000, device=device):
        super().__init__(pde=pde, layers=layers, pde_params=pde_params,
                          lambda_ic=lambda_ic, lambda_bc=lambda_bc,
                          lambda_pde=lambda_pde, device=device)
        self.rar_update_every = rar_update_every
        self.rar_k            = rar_k
        self.rar_pool_size    = rar_pool_size
        self.rar_max_points   = rar_max_points

    def _rar_add_points(self, x_f, t_f):
        """Evaluate the residual on a fresh dense random grid and append
        the rar_k highest-residual points to the (x_f, t_f) collocation
        set, capped at rar_max_points."""
        self.network.eval()
        x_pool = torch.tensor(
            np.random.uniform(-1, 1, (self.rar_pool_size, 1)),
            dtype=torch.float32).to(self.device).requires_grad_(True)
        t_pool = torch.tensor(
            np.random.uniform(0, 1, (self.rar_pool_size, 1)),
            dtype=torch.float32).to(self.device).requires_grad_(True)
        with torch.enable_grad():
            res    = self.residual_fn(self.network, x_pool, t_pool, **self.pde_params)
            scores = torch.abs(res).detach().squeeze()
        self.network.train()

        top_idx = torch.argsort(scores, descending=True)[:self.rar_k]
        x_new = x_pool[top_idx].detach()
        t_new = t_pool[top_idx].detach()

        x_f = torch.cat([x_f, x_new], dim=0)
        t_f = torch.cat([t_f, t_new], dim=0)
        if x_f.shape[0] > self.rar_max_points:
            x_f = x_f[-self.rar_max_points:]
            t_f = t_f[-self.rar_max_points:]
        return x_f, t_f

    def fit(self, data, epochs=10000, lr=1e-3, print_every=500, label=''):
        optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=3000, gamma=0.5)
        history = {
            'total': [], 'ic': [], 'bc': [], 'pde': [], 'n_collocation': []
        }
        start = time.time()

        x_f = data['x_f'].clone()
        t_f = data['t_f'].clone()

        for epoch in range(epochs):
            self.network.train()

            if epoch > 0 and epoch % self.rar_update_every == 0:
                x_f, t_f = self._rar_add_points(x_f, t_f)

            optimizer.zero_grad()
            total, l_ic, l_bc, l_pde = self._compute_loss(data, x_f, t_f)
            total.backward()
            optimizer.step()
            scheduler.step()

            history['total'].append(total.item())
            history['ic'].append(l_ic.item())
            history['bc'].append(l_bc.item())
            history['pde'].append(l_pde.item())
            history['n_collocation'].append(x_f.shape[0])

            if epoch % print_every == 0:
                prefix = f"{label} | " if label else ""
                print(f"[{prefix}Epoch {epoch:5d}/{epochs}] "
                      f"Total: {total.item():.6f} | "
                      f"IC: {l_ic.item():.6f} | BC: {l_bc.item():.6f} | "
                      f"PDE: {l_pde.item():.6f} | "
                      f"N_f: {x_f.shape[0]}")

        self.runtime = time.time() - start
        print(f"\nRAR-PINN training complete in {self.runtime:.2f}s")
        return history


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

        # FDM reference - interpolate if resolution differs
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
    # NOTE: 'lambda_ic' alone is NOT a reliable AC-PINN indicator -- SAWeightPINN
    # histories also carry lambda_ic/bc/pde (its learned softmax weights) but
    # have no 'stage' key, since it has no curriculum. Check each panel's own
    # required key independently instead of one combined is_ac flag.
    has_weights = 'lambda_ic' in history
    has_stage   = 'stage' in history

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

    # Bottom left: adaptive/learned loss weights (AC-PINN and SA-PINN)
    if has_weights:
        axes[1, 0].plot(history['lambda_ic'],  label='lambda_ic')
        axes[1, 0].plot(history['lambda_bc'],  label='lambda_bc')
        axes[1, 0].plot(history['lambda_pde'], label='lambda_pde')
        axes[1, 0].set_xlabel('Epoch'); axes[1, 0].set_ylabel('Weight value')
        axes[1, 0].set_title('Adaptive Loss Weights')
        axes[1, 0].legend(); axes[1, 0].grid(True)
    else:
        axes[1, 0].axis('off')
        axes[1, 0].text(0.5, 0.5, 'N/A - fixed loss weights', ha='center', va='center', fontsize=12)

    # Bottom right: curriculum stage progression (AC-PINN only)
    if has_stage:
        stages = np.array(history['stage'])
        axes[1, 1].plot(stages + 1)
        axes[1, 1].set_xlabel('Epoch'); axes[1, 1].set_ylabel('Curriculum Stage')
        axes[1, 1].set_title('Curriculum Stage Progression')
        axes[1, 1].set_yticks(range(1, int(stages.max()) + 2))
        axes[1, 1].grid(True)
    else:
        axes[1, 1].axis('off')
        axes[1, 1].text(0.5, 0.5, 'N/A - no curriculum stages', ha='center', va='center', fontsize=12)

    fig.suptitle(label, fontsize=14)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

    # CSV with raw numbers
    csv_path = os.path.splitext(save_path)[0] + '.csv'
    fieldnames = ['epoch', 'total', 'ic', 'bc', 'pde']
    if has_weights:
        fieldnames += ['lambda_ic', 'lambda_bc', 'lambda_pde']
    if has_stage:
        fieldnames += ['stage']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        for i in range(len(history['total'])):
            row = [i, history['total'][i], history['ic'][i],
                   history['bc'][i], history['pde'][i]]
            if has_weights:
                row += [history['lambda_ic'][i], history['lambda_bc'][i],
                        history['lambda_pde'][i]]
            if has_stage:
                row += [history['stage'][i]]
            writer.writerow(row)

    print(f'Saved: {save_path}')
    print(f'Saved: {csv_path}')
