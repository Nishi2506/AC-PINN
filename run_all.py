"""
run_all.py
==========
Executes the full AC-PINN notebook suite in order.
Run from the project root:
    python run_all.py

Each notebook is executed in place and its outputs are saved.
Results are stored in results/ and figures/ directories.
"""

import subprocess
import sys
import os
import time

NOTEBOOKS = [
    'notebooks/00_setup_and_verify.ipynb',
    'notebooks/01_burgers.ipynb',
    'notebooks/02_heat.ipynb',
    'notebooks/03_wave.ipynb',
    'notebooks/04_allen_cahn.ipynb',
    'notebooks/05_ablation.ipynb',
    'notebooks/06_final_comparison.ipynb',
]

def run_notebook(path):
    print(f'\n{"="*60}')
    print(f'  Running: {path}')
    print(f'{"="*60}')
    start = time.time()
    result = subprocess.run([
        sys.executable, '-m', 'nbconvert',
        '--to', 'notebook',
        '--execute',
        '--inplace',
        '--ExecutePreprocessor.timeout=7200',  # 2 hour timeout per notebook
        path
    ], capture_output=True, text=True)

    elapsed = time.time() - start

    if result.returncode == 0:
        print(f'  ✓ Done in {elapsed:.1f}s')
    else:
        print(f'  ✗ FAILED after {elapsed:.1f}s')
        print(f'  stderr: {result.stderr[-500:]}')
        return False
    return True

def main():
    print('\nAC-PINN Full Suite Runner')
    print('Authors: Suyash Vasal Jain, Nishita Raghvendra, Priyal Agrawal')
    print('='*60)

    # Ensure results/figures dirs exist
    for pde in ['burgers', 'heat', 'wave', 'allen_cahn']:
        os.makedirs(f'results/{pde}', exist_ok=True)
        os.makedirs(f'figures/{pde}', exist_ok=True)
    os.makedirs('figures/comparison', exist_ok=True)

    total_start = time.time()
    failed = []

    for nb in NOTEBOOKS:
        if not os.path.exists(nb):
            print(f'  WARNING: {nb} not found, skipping.')
            continue
        success = run_notebook(nb)
        if not success:
            failed.append(nb)
            print(f'  Stopping suite due to failure in {nb}')
            break

    total_time = time.time() - total_start
    print(f'\n{"="*60}')
    print(f'  Suite complete in {total_time/60:.1f} minutes')
    if failed:
        print(f'  Failed notebooks: {failed}')
    else:
        print(f'  All notebooks ran successfully.')
    print(f'{"="*60}\n')

if __name__ == '__main__':
    main()
