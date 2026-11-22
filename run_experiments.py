"""
Run all fair mean estimation experiments and save results.

Usage
-----
    python run_experiments.py                 # all experiments
    python run_experiments.py --exp 1_1 1_2   # specific experiments
    python run_experiments.py --fast           # fewer MC draws (quick check)

Outputs
-------
    results/   CSV files with raw data
    figures/   PNG plots (one subdirectory per experiment)
"""

import sys
import argparse
import time
from pathlib import Path

# -- Make sure the codes directory is importable --
sys.path.insert(0, str(Path(__file__).parent))

import matplotlib
matplotlib.use('Agg')    # non-interactive backend for file output

from fair_estimation.experiments import (
    exp_1_1, exp_1_2, exp_2_1_1, exp_2_1_2, exp_2_2
)
from fair_estimation.plots import (
    plot_estimator_means, plot_mse_panel, plot_mse_combined, plot_payments,
    plot_group_payments, plot_privacy_loss, plot_distribution_comparison,
    plot_mse_payment_tradeoff, plot_active_participants,
)

# ---------------------------------------------------------------------------
# Output directories
# ---------------------------------------------------------------------------
BASE    = Path(__file__).parent
RESULTS = BASE / 'results'
FIGS    = BASE / 'figures'
for d in [RESULTS, FIGS]:
    d.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Individual experiment runners
# ---------------------------------------------------------------------------

def run_1_1(n_mc):
    print("\n" + "=" * 60)
    print("Experiment 1 — Vary Population Proportions (Homogeneous)")
    print("=" * 60)
    t0  = time.time()
    df  = exp_1_1(n_mc=n_mc)
    df.to_csv(RESULTS / 'exp_1_1.csv', index=False)

    sub = FIGS / 'exp_1_1'
    sub.mkdir(exist_ok=True)
    plot_estimator_means(df, 'p_A', sub / 'means.png',
                         title='Exp 1 — Estimator Mean vs. pₐ')
    plot_mse_panel      (df, 'p_A', sub / 'mse.png',
                         title='Exp 1 — MSE breakdown vs. pₐ')
    plot_payments       (df, 'p_A', sub / 'payments.png',
                         title='Exp 1 — Total Payment vs. pₐ')
    plot_privacy_loss   (df, 'p_A', sub / 'privacy.png',
                         title='Exp 1 — Avg privacy loss vs. pₐ')
    plot_active_participants(df, 'p_A', sub / 'active.png',
                             title='Exp 1 — Active participants vs. pₐ')
    print(f"  → 5 figures  ({time.time()-t0:.1f}s)")
    return df


def run_1_2(n_mc):
    print("\n" + "=" * 60)
    print("Experiment 2 — Vary Fairness Strength λ (Homogeneous)")
    print("=" * 60)
    t0  = time.time()
    df  = exp_1_2(n_mc=n_mc)
    df.to_csv(RESULTS / 'exp_1_2.csv', index=False)

    sub = FIGS / 'exp_1_2'
    sub.mkdir(exist_ok=True)
    plot_estimator_means    (df, 'lambda', sub / 'means.png',
                             title='Exp 2 — Estimator Mean vs. λ')
    plot_mse_combined       (df, 'lambda', sub / 'mse.png',
                             title='Exp 2 — MSE Breakdown vs. λ')
    plot_payments           (df, 'lambda', sub / 'payments.png',
                             title='Exp 2 — Total Payment vs. λ')
    plot_group_payments     (df, 'lambda', sub / 'group_payments.png',
                             title='Exp 2 — Group Average Payments vs. λ')
    plot_active_participants(df, 'lambda', sub / 'active.png',
                             title='Exp 2 — Acquired participants vs. λ')
    print(f"  → figures generated  ({time.time()-t0:.1f}s)")
    return df


def run_2_1_1(n_mc):
    print("\n" + "=" * 60)
    print("Experiment 3.1 — Vary Cost Disparity (Heterogeneous, Equal Proportions)")
    print("=" * 60)
    t0  = time.time()
    df  = exp_2_1_1(n_mc=n_mc)
    df.to_csv(RESULTS / 'exp_2_1_1.csv', index=False)

    sub = FIGS / 'exp_2_1_1'
    sub.mkdir(exist_ok=True)
    plot_estimator_means    (df, 'cost_ratio', sub / 'means.png',
                             title='Exp 3.1 — Estimator Mean vs. Cost Ratio')
    plot_mse_panel          (df, 'cost_ratio', sub / 'mse.png',
                             title='Exp 3.1 — MSE vs. Cost Ratio')
    plot_payments           (df, 'cost_ratio', sub / 'payments.png',
                             title='Exp 3.1 — Total Payment vs. Cost Ratio')
    plot_privacy_loss       (df, 'cost_ratio', sub / 'privacy.png',
                             title='Exp 3.1 — Avg privacy loss vs. Cost Ratio')
    plot_active_participants(df, 'cost_ratio', sub / 'active.png',
                             title='Exp 3.1 — Acquired participants vs. Cost Ratio')
    print(f"  → 5 figures  ({time.time()-t0:.1f}s)")
    return df


def run_2_1_2(n_mc):
    print("\n" + "=" * 60)
    print("Experiment 3.2 — Vary Proportions (Heterogeneous Costs)")
    print("=" * 60)
    t0  = time.time()
    df  = exp_2_1_2(n_mc=n_mc)
    df.to_csv(RESULTS / 'exp_2_1_2.csv', index=False)

    sub = FIGS / 'exp_2_1_2'
    sub.mkdir(exist_ok=True)
    plot_estimator_means    (df, 'p_A', sub / 'means.png',
                             title='Exp 3.2 — Estimator Mean vs. pₐ  (cᵦ=3·cₐ)')
    plot_mse_panel          (df, 'p_A', sub / 'mse.png',
                             title='Exp 3.2 — MSE vs. pₐ')
    plot_payments           (df, 'p_A', sub / 'payments.png',
                             title='Exp 3.2 — Total Payment vs. pₐ')
    plot_privacy_loss       (df, 'p_A', sub / 'privacy.png',
                             title='Exp 3.2 — Privacy loss vs. pₐ')
    plot_active_participants(df, 'p_A', sub / 'active.png',
                             title='Exp 3.2 — Acquired participants vs. pₐ')
    print(f"  → 5 figures  ({time.time()-t0:.1f}s)")
    return df


def run_2_2(n_mc):
    print("\n" + "=" * 60)
    print("Experiment 4 — Distribution Comparison")
    print("=" * 60)
    t0  = time.time()
    df  = exp_2_2(n_mc=max(n_mc, 10))   # more draws for distribution exp
    df.to_csv(RESULTS / 'exp_2_2.csv', index=False)

    sub = FIGS / 'exp_2_2'
    sub.mkdir(exist_ok=True)
    plot_distribution_comparison(df, sub / 'comparison.png',
                                 title='Experiment 4 — Distribution Comparison')
    print(f"  → 1 figure  ({time.time()-t0:.1f}s)")
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

RUNNERS = {
    '1_1':  run_1_1,
    '1_2':  run_1_2,
    '2_1_1': run_2_1_1,
    '2_1_2': run_2_1_2,
    '2_2':  run_2_2,
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp',  nargs='+', default=list(RUNNERS.keys()),
                        help='Which experiments to run (default: all)')
    parser.add_argument('--fast', action='store_true',
                        help='Use n_mc=3 for a quick sanity check')
    parser.add_argument('--mc',   type=int, default=5,
                        help='Monte Carlo draws per configuration (default 5)')
    args = parser.parse_args()

    n_mc = 3 if args.fast else args.mc

    t_total = time.time()
    results = {}
    for key in args.exp:
        if key not in RUNNERS:
            print(f"[warn] Unknown experiment '{key}'. Skipping.")
            continue
        results[key] = RUNNERS[key](n_mc)

    print("\n" + "=" * 60)
    print(f"All done in {time.time() - t_total:.1f}s")
    print(f"  Results : {RESULTS}")
    print(f"  Figures : {FIGS}")
