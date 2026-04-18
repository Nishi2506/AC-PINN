# AC-PINN: Adaptive Curriculum Physics-Informed Neural Networks for Stable PDE Solving under Sparse/Noisy Data

**Authors:** Suyash Vasal Jain, Nishita Raghvendra

---

## Overview

This repository implements **AC-PINN**, a novel extension of Physics-Informed Neural Networks (PINNs) that incorporates:

1. **4-stage curriculum learning** - residual-based collocation point sampling that starts from easy regions and progressively introduces harder regions (shock fronts, stiff interfaces)
2. **Adaptive loss weighting** - two strategies (gradient magnitude based, loss ratio based) that dynamically rebalance IC/BC/PDE loss terms during training
3. **Robustness experiments** - systematic evaluation under clean/noisy × dense/sparse data conditions across 4 PDEs

## PDEs Covered

| PDE | Equation | IC | Architecture |
|---|---|---|---|
| Burgers | $u_t + uu_x = \nu u_{xx}$ | $-\sin(\pi x)$ | `[2,64,64,64,64,64,1]` |
| Heat | $u_t = \alpha u_{xx}$ | $\sin(\pi x)$ | `[2,32,32,32,1]` |
| Wave | $u_{tt} = c^2 u_{xx}$ | $\sin(\pi x)$ | `[2,64,64,64,64,64,1]` |
| Allen-Cahn | $u_t = \varepsilon^2 u_{xx} + u - u^3$ | $x^2\cos(\pi x)$ | `[2,128,128,128,128,128,1]` |

## Repository Structure

```
ac-pinn-project/
├── pinn_base.py                   ← all model classes
├── run_all.py                     ← run full suite at once
├── requirements.txt
├── .gitignore
├── notebooks/
│   ├── 00_setup_and_verify.ipynb  ← environment check
│   ├── 01_burgers.ipynb
│   ├── 02_heat.ipynb
│   ├── 03_wave.ipynb
│   ├── 04_allen_cahn.ipynb
│   ├── 05_ablation.ipynb          ← isolates component contributions
│   └── 06_final_comparison.ipynb  ← paper figures
├── results/                       ← saved metrics (gitignored)
└── figures/                       ← saved plots (gitignored)
```

## Classes in `pinn_base.py`

| Class | Description |
|---|---|
| `NoisyDataGenerator` | Generates IC/BC/collocation data with controllable noise (ε) and sparsity |
| `CurriculumSampler` | 4-stage residual-based collocation point sampler |
| `PINNSolver` | Vanilla PINN - fixed loss weights, random collocation |
| `ACPINNSolver` | AC-PINN - curriculum + adaptive weights (gradient / ratio / both) |
| `BurgersFDM` | Crank-Nicolson FDM for Burgers equation |
| `HeatFDM` | Crank-Nicolson FDM for Heat equation |
| `WaveFDM` | Leapfrog FDM for Wave equation |
| `AllenCahnFDM` | IMEX FDM for Allen-Cahn equation |
| `Benchmark` | Compares multiple models against FDM ground truth |

## Setup

### Local

```bash
git clone https://github.com/Nishi2506/AC-PINN.git
cd ac-pinn-project

python -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

### Google Colab

```python
!git clone https://github.com/Nishi2506/AC-PINN.git
%cd ac-pinn-project
!pip install -r requirements.txt -q

from google.colab import drive
drive.mount('/content/drive')
```

## Running

### Run all notebooks sequentially

```bash
python run_all.py
```

### Run individual notebooks

```bash
jupyter notebook notebooks/01_burgers.ipynb
```

### Quick usage example

```python
from pinn_base import NoisyDataGenerator, PINNSolver, ACPINNSolver, BurgersFDM, Benchmark
import numpy as np

# Data
gen  = NoisyDataGenerator(pde='burgers', nu=0.01/np.pi)
data = gen.generate(N_ic=1000, N_bc=1000, N_f=8000, noise_eps=0.1)

# FDM ground truth
fdm = BurgersFDM(nx=256, nt=2000)
fdm.solve()

# Vanilla PINN
vanilla = PINNSolver(pde='burgers', layers=[2,64,64,64,64,64,1])
vanilla.fit(data, epochs=10000)

# AC-PINN
ac = ACPINNSolver(pde='burgers', layers=[2,64,64,64,64,64,1], weight_strategy='both')
ac.fit(data, epochs=10000)

# Benchmark
bench = Benchmark(fdm)
bench.add('Vanilla', vanilla).add('AC-PINN', ac)
bench.run()
bench.compare_metrics()
bench.plot_comparison()
```

## Experiments

Each PDE notebook runs 4 experiments:

| Experiment | Data Condition |
|---|---|
| 1 | Vanilla PINN - clean dense |
| 2 | Vanilla PINN - noisy sparse (ε=0.1, N_ic=20) |
| 3 | AC-PINN - clean dense |
| 4 | AC-PINN - noisy sparse (ε=0.1, N_ic=20) |

Plus a **noise level study** across ε ∈ {0.05, 0.1, 0.2}.

The ablation notebook isolates:
- Curriculum only
- Adaptive weights (ratio) only
- Adaptive weights (gradient) only
- Full AC-PINN (ratio)
- Full AC-PINN (gradient)
- Full AC-PINN (both strategies) ← best

## Training Config

| Parameter | Value |
|---|---|
| Epochs | 10,000 |
| Optimizer | Adam |
| Learning rate | 1e-3 (step decay ×0.5 every 3000 epochs) |
| Curriculum stages | 4 |
| Pool size | 20,000 (30,000 for Allen-Cahn) |
| Resample every | 500 epochs |
| Weight update every | 200 epochs |
| Noise levels tested | ε = 0.05, 0.1, 0.2 |

## License

MIT
