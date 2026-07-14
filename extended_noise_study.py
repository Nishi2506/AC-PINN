

import os
import numpy as np
import matplotlib.pyplot as plt

from pinn_base import (
    NoisyDataGenerator, PINNSolver, ACPINNSolver,
    FDM_SOLVERS, PDE_PARAMS, Benchmark, save_metrics
)

EPSILONS = [0.01, 0.05, 0.1, 0.15, 0.2, 0.3]
PDES     = ['burgers', 'heat', 'wave', 'allen_cahn']
LAYERS   = [2, 64, 64, 64, 64, 64, 1]
EPOCHS   = 5000

FDM_CONFIG = {
    'burgers':    dict(nx=256, nt=2000),
    'heat':       dict(nx=256, nt=1000),
    'wave':       dict(nx=256, nt=2000),
    'allen_cahn': dict(nx=256, nt=5000),
}

FIGURES_DIR = 'figures/comparison/'


def run_pde_noise_study(pde):
    """Train Vanilla PINN and AC-PINN (both) at each noise level for one
    PDE, benchmark both against the FDM ground truth, and return a dict
    keyed by epsilon -> {'Vanilla': {...}, 'AC-PINN (both)': {...}}."""
    pde_params = PDE_PARAMS[pde]
    results_dir = f'results/{pde}/'
    os.makedirs(results_dir, exist_ok=True)

    GEN_PARAMS = {'nu', 'alpha', 'c'}
    gen_params = {k: v for k, v in pde_params.items() if k in GEN_PARAMS}
    gen = NoisyDataGenerator(pde=pde, **gen_params)

    fdm = FDM_SOLVERS[pde](**FDM_CONFIG[pde], **pde_params)
    fdm.solve()

    results = {}
    for eps in EPSILONS:
        print(f'\n=== {pde} | epsilon={eps} ===')
        data = gen.generate(N_ic=50, N_bc=50, N_f=3000, noise_eps=eps)

        vanilla = PINNSolver(pde=pde, layers=LAYERS, pde_params=pde_params)
        vanilla.fit(data, epochs=EPOCHS, print_every=2500,
                    label=f'ExtNoise | {pde} | Vanilla eps={eps}')

        acpinn = ACPINNSolver(pde=pde, layers=LAYERS, pde_params=pde_params,
                               weight_strategy='both')
        acpinn.fit(data, epochs=EPOCHS, print_every=2500,
                   label=f'ExtNoise | {pde} | AC-PINN eps={eps}')

        bench = Benchmark(fdm)
        bench.add('Vanilla', vanilla)
        bench.add('AC-PINN (both)', acpinn)
        bench.run()
        results[eps] = bench.compare_metrics()

    save_metrics(results, results_dir + 'extended_noise_metrics.npy')
    return results


def plot_extended_noise_study(all_results, save_path):
    """2x2 grid, one subplot per PDE, epsilon (x) vs Rel L2 error (y)
    for Vanilla PINN and AC-PINN (both)."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for ax, pde in zip(axes, PDES):
        results  = all_results[pde]
        epsilons = sorted(results.keys())
        vanilla_l2 = [results[eps]['Vanilla']['l2'] for eps in epsilons]
        ac_l2      = [results[eps]['AC-PINN (both)']['l2'] for eps in epsilons]

        ax.plot(epsilons, vanilla_l2, 'o-', label='Vanilla PINN')
        ax.plot(epsilons, ac_l2,      's-', label='AC-PINN (both)')
        ax.set_xlabel('Noise level (epsilon)')
        ax.set_ylabel('Rel L2 Error')
        ax.set_title(pde.replace('_', ' ').title())
        ax.legend(); ax.grid(True)

    fig.suptitle('Extended Noise Robustness Study', fontsize=14)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'Saved: {save_path}')


def main():
    all_results = {}
    for pde in PDES:
        all_results[pde] = run_pde_noise_study(pde)

    plot_extended_noise_study(all_results, FIGURES_DIR + 'extended_noise_study.png')
    print('\nExtended noise robustness study complete.')


if __name__ == '__main__':
    main()
