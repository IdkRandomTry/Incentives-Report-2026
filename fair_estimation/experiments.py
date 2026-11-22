"""
Experiment runners for all five experiments.

Each runner returns a pandas DataFrame. One row = one (parameter config,
estimator type) combination, with all metrics averaged over n_mc Monte
Carlo draws of the cost vector.

Experiment map
--------------
  exp_1_1   Vary population proportions pₐ   (homogeneous costs)
  exp_1_2   Vary fairness strength λ          (homogeneous costs)
  exp_2_1_1 Vary cost disparity cᵦ/cₐ        (heterogeneous, equal props)
  exp_2_1_2 Vary population proportions pₐ   (heterogeneous costs)
  exp_2_2   Compare cost distributions        (heterogeneous costs)
"""

import numpy as np
import pandas as pd
from typing import Callable, Sequence

from .core import GroupParams, solve, compute_metrics
from .costs import sample_costs, group_costs

# All three estimator types used throughout
ESTIMATOR_TYPES = ['fair', 'no_fairness', 'groupless']


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _run_all(psi: np.ndarray, costs: np.ndarray,
             params: GroupParams, seed: int = 0) -> dict:
    """
    Solve for all three estimator types and return a dict of metric dicts.
    """
    out = {}
    for et in ESTIMATOR_TYPES:
        sol = solve(psi, params, estimator_type=et,
                    n_restarts=5, seed=seed)
        out[et] = compute_metrics(sol, psi, costs, params, estimator_type=et)
    return out


def _mc_average(cost_fn: Callable[[int], tuple],
                params: GroupParams,
                n_mc: int = 5,
                base_seed: int = 0) -> dict:
    """
    Average metrics over n_mc independent cost draws.

    cost_fn(seed) → (costs, virtual_costs)
    Returns {estimator_type: {metric: mean_value}}.
    """
    accum = {et: [] for et in ESTIMATOR_TYPES}

    for s in range(n_mc):
        costs, psi = cost_fn(base_seed + s)
        per_draw   = _run_all(psi, costs, params, seed=s)
        for et, metrics in per_draw.items():
            accum[et].append(metrics)

    averaged = {}
    for et, runs in accum.items():
        df = pd.DataFrame(runs)
        averaged[et] = {col: float(df[col].mean()) for col in df.columns}
    return averaged


def _to_rows(avg: dict, fixed_cols: dict) -> list:
    """Convert per-estimator averaged metrics into a list of row dicts."""
    rows = []
    for et, metrics in avg.items():
        rows.append({**fixed_cols, 'estimator': et, **metrics})
    return rows


# ---------------------------------------------------------------------------
# Experiment 1.1 — Vary population proportions (homogeneous costs)
# ---------------------------------------------------------------------------

def exp_1_1(
    mu_A: float = 0.0,
    mu_B: float = 1.0,
    sigma2: float = 1.0,
    n_total: int = 50,
    lam: float = 10.0,
    mean_cost: float = 0.5,
    p_A_values: Sequence[float] = None,
    n_mc: int = 5,
) -> pd.DataFrame:
    """
    Fix homogeneous costs and λ.  Sweep pₐ ∈ p_A_values.

    Shows: how the estimator mean and MSEs shift as the minority group
    shrinks, and how the fairness term counteracts that drift.
    """
    if p_A_values is None:
        p_A_values = np.linspace(0.1, 0.9, 9)

    rows = []
    for p_A in p_A_values:
        n_A = max(1, round(p_A * n_total))
        n_B = n_total - n_A
        params = GroupParams(mu_A, mu_B, sigma2, n_A, n_B, lam)

        def cost_fn(seed, _nA=n_A, _nB=n_B):
            return sample_costs('uniform', _nA + _nB, mean_cost, seed=seed)

        avg = _mc_average(cost_fn, params, n_mc=n_mc)
        rows.extend(_to_rows(avg, {
            'experiment': '1.1', 'p_A': float(p_A),
            'n_A': n_A, 'n_B': n_B, 'lam': lam,
        }))
        print(f"  [1.1] p_A={p_A:.2f}  n_A={n_A}  n_B={n_B}  done.")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Experiment 1.2 — Vary fairness strength λ (homogeneous costs)
# ---------------------------------------------------------------------------

def exp_1_2(
    mu_A: float = 0.0,
    mu_B: float = 1.0,
    sigma2: float = 1.0,
    n_A: int = 15,
    n_B: int = 35,
    mean_cost: float = 0.5,
    lam_values: Sequence[float] = None,
    n_mc: int = 5,
) -> pd.DataFrame:
    """
    Fix population (pₐ ≈ 0.3) and costs.  Sweep λ.

    Shows: estimator mean moving from θ (λ=0, min-MSE) toward the group
    midpoint (λ→∞, max-fairness).  Also tracks whether total payments
    rise or fall as λ increases.
    """
    if lam_values is None:
        lam_values = [0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]

    rows = []
    for lam in lam_values:
        params = GroupParams(mu_A, mu_B, sigma2, n_A, n_B, float(lam))

        def cost_fn(seed, _lam=lam):
            return sample_costs('uniform', n_A + n_B, mean_cost, seed=seed)

        avg = _mc_average(cost_fn, params, n_mc=n_mc)
        rows.extend(_to_rows(avg, {
            'experiment': '1.2', 'lambda': float(lam),
            'n_A': n_A, 'n_B': n_B,
        }))
        print(f"  [1.2] lambda={lam}  done.")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Experiment 2.1.1 — Vary cost disparity (heterogeneous, equal proportions)
# ---------------------------------------------------------------------------

def exp_2_1_1(
    mu_A: float = 0.0,
    mu_B: float = 1.0,
    sigma2: float = 1.0,
    n_A: int = 25,
    n_B: int = 25,
    lam: float = 10.0,
    mean_cost_A: float = 0.5,
    mean_cost_B_values: Sequence[float] = None,
    n_mc: int = 5,
) -> pd.DataFrame:
    """
    Fix equal proportions and λ.  Sweep Group B cost mean.

    Group A cost fixed at mean_cost_A; Group B cost increases.
    Groupless estimator is flagged as 'not applicable' in heterogeneous
    setting but still included for reference.
    """
    if mean_cost_B_values is None:
        mean_cost_B_values = [0.5, 2.5, 5.0, 10.0]  # Ratios: 1.0, 5.0, 10.0, 20.0

    rows = []
    for mc_B in mean_cost_B_values:
        params = GroupParams(mu_A, mu_B, sigma2, n_A, n_B, lam)

        def cost_fn(seed, _mcB=mc_B):
            return group_costs('uniform', n_A, mean_cost_A, n_B, _mcB,
                               seed=seed)

        avg = _mc_average(cost_fn, params, n_mc=n_mc)
        rows.extend(_to_rows(avg, {
            'experiment': '2.1.1',
            'mean_cost_A': mean_cost_A,
            'mean_cost_B': float(mc_B),
            'cost_ratio':  mc_B / mean_cost_A,
            'lam': lam,
        }))
        print(f"  [2.1.1] mean_cost_B={mc_B:.2f}  ratio={mc_B/mean_cost_A:.2f}  done.")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Experiment 2.1.2 — Vary proportions (heterogeneous costs)
# ---------------------------------------------------------------------------

def exp_2_1_2(
    mu_A: float = 0.0,
    mu_B: float = 1.0,
    sigma2: float = 1.0,
    n_total: int = 50,
    lam: float = 10.0,
    mean_cost_A: float = 0.5,
    mean_cost_B: float = 1.5,
    p_A_values: Sequence[float] = None,
    n_mc: int = 5,
) -> pd.DataFrame:
    """
    Fix cost disparity (B = 3× A) and λ.  Sweep pₐ.

    Studies interaction between cost asymmetry and group size.
    """
    if p_A_values is None:
        p_A_values = np.linspace(0.1, 0.9, 9)

    rows = []
    for p_A in p_A_values:
        n_A = max(1, round(p_A * n_total))
        n_B = n_total - n_A
        params = GroupParams(mu_A, mu_B, sigma2, n_A, n_B, lam)

        def cost_fn(seed, _nA=n_A, _nB=n_B):
            return group_costs('uniform', _nA, mean_cost_A, _nB, mean_cost_B,
                               seed=seed)

        avg = _mc_average(cost_fn, params, n_mc=n_mc)
        rows.extend(_to_rows(avg, {
            'experiment': '2.1.2', 'p_A': float(p_A),
            'n_A': n_A, 'n_B': n_B,
            'mean_cost_A': mean_cost_A,
            'mean_cost_B': mean_cost_B,
        }))
        print(f"  [2.1.2] p_A={p_A:.2f}  n_A={n_A}  n_B={n_B}  done.")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Experiment 2.2 — Distribution comparison
# ---------------------------------------------------------------------------

def exp_2_2(
    mu_A: float = 0.0,
    mu_B: float = 1.0,
    sigma2: float = 1.0,
    n_A: int = 25,
    n_B: int = 25,
    lam: float = 10.0,
    mean_cost_A: float = 0.5,
    mean_cost_B: float = 1.0,
    dist_types: Sequence[str] = None,
    n_mc: int = 15,
) -> pd.DataFrame:
    """
    Fix group moments.  Compare Uniform, LogNormal, TruncNormal distributions.

    All distributions are matched to the same (mean, variance) so that
    any differences arise from distributional shape, not first/second moments.
    Variance is matched to U(0, 2·mean) for each group: var = mean²/3.
    """
    if dist_types is None:
        dist_types = ['uniform', 'lognormal', 'truncnorm']

    # Match variance to the uniform distribution for fair comparison
    var_A = mean_cost_A**2 / 3.0
    var_B = mean_cost_B**2  / 3.0

    rows = []
    for dist in dist_types:
        params = GroupParams(mu_A, mu_B, sigma2, n_A, n_B, lam)

        def cost_fn(seed, _d=dist):
            return group_costs(_d, n_A, mean_cost_A, n_B, mean_cost_B,
                               var_A=var_A, var_B=var_B, seed=seed)

        avg = _mc_average(cost_fn, params, n_mc=n_mc)
        rows.extend(_to_rows(avg, {
            'experiment':   '2.2',
            'distribution': dist,
            'mean_cost_A':  mean_cost_A,
            'mean_cost_B':  mean_cost_B,
        }))
        print(f"  [2.2] dist={dist}  done.")

    return pd.DataFrame(rows)
