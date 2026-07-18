

import subprocess
import sys
import os
import shutil
import time
import numpy as np

N_SEEDS = 3
SEEDS   = [42, 123, 456]

PDE_NOTEBOOKS = {
    'burgers':    'notebooks/01_burgers.ipynb',
    'heat':       'notebooks/02_heat.ipynb',
    'wave':       'notebooks/03_wave.ipynb',
    'allen_cahn': 'notebooks/04_allen_cahn.ipynb',
}

ROOT        = os.path.dirname(os.path.abspath(__file__))
TMP_NB_DIR  = os.path.join(ROOT, '.multi_seed_tmp')
METRIC_KEYS = ['l2', 'max_error', 'mae', 'rmse']


def run_notebook_with_seed(nb_path, seed, timeout=7200):
    """
    Execute a notebook with PINN_SEED set in the environment (picked up by
    pinn_base.py at import time, seeding random/numpy/torch/CUDA before any
    model is constructed). The executed copy is written to a scratch
    directory so the original notebook file is never modified.
    """
    os.makedirs(TMP_NB_DIR, exist_ok=True)
    nb_name  = os.path.basename(nb_path)
    out_name = f'{os.path.splitext(nb_name)[0]}_seed{seed}.ipynb'

    env = os.environ.copy()
    env['PINN_SEED'] = str(seed)

    print(f'\n{"="*60}')
    print(f'  Running: {nb_path}  (seed={seed})')
    print(f'{"="*60}')
    start = time.time()

    result = subprocess.run([
        sys.executable, '-m', 'nbconvert',
        '--to', 'notebook',
        '--execute',
        f'--ExecutePreprocessor.timeout={timeout}',
        nb_path,
        '--output-dir', TMP_NB_DIR,
        '--output', out_name,
    ], capture_output=True, text=True)

    elapsed = time.time() - start

    if result.returncode == 0:
        print(f'  [DONE] Done in {elapsed:.1f}s')
        return True
    else:
        print(f'  [FAIL] FAILED after {elapsed:.1f}s')
        print(f'  stderr: {result.stderr[-800:]}')
        return False


def collect_metrics(pde, seed):
    """
    Copy the benchmark_metrics.npy the run just produced into a
    seed-tagged file, so it survives the next seed overwriting it.
    """
    src = os.path.join(ROOT, 'results', pde, 'benchmark_metrics.npy')
    dst = os.path.join(ROOT, 'results', pde, f'benchmark_metrics_seed{seed}.npy')
    if not os.path.exists(src):
        raise FileNotFoundError(
            f'{src} not found - notebook for "{pde}" did not produce '
            f'benchmark_metrics.npy (check nbconvert output above).')
    shutil.copy2(src, dst)
    return dst


def mean_std_across_seeds(paths):
    """
    Load N benchmark_metrics.npy dicts (model_name -> {l2, max_error, mae,
    rmse}) and compute mean +/- std per model per metric across seeds.
    """
    runs = [np.load(p, allow_pickle=True).item() for p in paths]
    models = runs[0].keys()

    summary = {}
    for model in models:
        summary[model] = {}
        for metric in METRIC_KEYS:
            values = np.array([run[model][metric] for run in runs], dtype=float)
            summary[model][metric] = {
                'mean':   float(values.mean()),
                'std':    float(values.std()),
                'values': values.tolist(),
            }
    return summary


def print_summary_table(pde, summary):
    print('=' * 100)
    print(f'  {pde.upper()}  -  mean +/- std over {N_SEEDS} seeds {SEEDS}')
    print('=' * 100)
    header = f"  {'Model':<25}" + ''.join(f"{m:>18}" for m in METRIC_KEYS)
    print(header)
    print('-' * 100)
    for model, metrics in summary.items():
        row = f"  {model:<25}"
        for m in METRIC_KEYS:
            mean = metrics[m]['mean']
            std  = metrics[m]['std']
            row += f"{mean:>9.5f}+/-{std:<7.5f}"
        print(row)
    print('=' * 100)


def main():
    print('\nAC-PINN Multi-Seed Run')
    print(f'Seeds: {SEEDS}  (N_SEEDS={N_SEEDS})')
    print('=' * 60)

    all_summaries = {}
    failed = []

    for pde, nb_path in PDE_NOTEBOOKS.items():
        print(f'\n{"#"*70}\n# PDE: {pde}\n{"#"*70}')

        if not os.path.exists(nb_path):
            print(f'  WARNING: {nb_path} not found, skipping.')
            continue

        seed_metric_paths = []
        for seed in SEEDS:
            ok = run_notebook_with_seed(nb_path, seed)
            if not ok:
                failed.append((pde, seed))
                continue
            seed_metric_paths.append(collect_metrics(pde, seed))

        if len(seed_metric_paths) < 2:
            print(f'  Not enough successful seed runs for {pde}, skipping summary.')
            continue

        summary  = mean_std_across_seeds(seed_metric_paths)
        out_path = os.path.join(ROOT, 'results', pde, 'mean_std_metrics.npy')
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.save(out_path, summary)
        print(f'\nSaved: {out_path}')

        all_summaries[pde] = summary
        print_summary_table(pde, summary)

    print(f'\n\n{"#"*70}\n# FINAL SUMMARY - ALL PDEs\n{"#"*70}')
    for pde, summary in all_summaries.items():
        print_summary_table(pde, summary)

    if failed:
        print(f'\nFailed runs: {failed}')
    print('\nMulti-seed run complete.')

    if failed:
        sys.exit(1)


if __name__ == '__main__':
    main()
