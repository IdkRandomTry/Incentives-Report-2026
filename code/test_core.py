"""Quick unit test for core solver."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fair_estimation.core import GroupParams, solve, compute_metrics
from fair_estimation.costs import sample_costs, group_costs
import numpy as np

print("--- Test 1: Core solver (homogeneous) ---")
params = GroupParams(mu_A=0.0, mu_B=1.0, sigma2=1.0, n_A=30, n_B=70, lam=1.0)
costs, psi = sample_costs('uniform', 100, 0.5, seed=0)
print(f"Costs sampled OK. mean_psi={psi.mean():.4f}")

for et in ['fair', 'no_fairness', 'groupless']:
    sol = solve(psi, params, estimator_type=et, n_restarts=2, seed=0)
    m   = compute_metrics(sol, psi, costs, params, et)
    print(f"  [{et:16s}] m={sol['m']:.4f}  S={sol['S']:.4f}  "
          f"mse_pop={m['mse_pop']:.5f}  payment={m['instance_payment']:.4f}  "
          f"disparity={m['mse_disparity']:.5f}")

print()
print("--- Test 2: Verification ---")
# lambda=0 fair should match no_fairness exactly
params0 = GroupParams(mu_A=0.0, mu_B=1.0, sigma2=1.0, n_A=30, n_B=70, lam=0.0)
sol_fair0 = solve(psi, params0, 'fair', n_restarts=2)
sol_nofair = solve(psi, params, 'no_fairness', n_restarts=2)
diff = abs(sol_fair0['m'] - sol_nofair['m'])
print(f"  lam=0 fair vs no_fairness: |m_diff|={diff:.6f}  (should be ~0)")

print()
print("--- Test 3: Heterogeneous costs ---")
params_het = GroupParams(mu_A=0.0, mu_B=1.0, sigma2=1.0, n_A=50, n_B=50, lam=1.0)
c_het, psi_het = group_costs('uniform', 50, 0.5, 50, 1.5, seed=0)
sol_het = solve(psi_het, params_het, 'fair', n_restarts=2)
m_het   = compute_metrics(sol_het, psi_het, c_het, params_het, 'fair')
print(f"  Het fair: m={sol_het['m']:.4f}  avg_priv_A={m_het['avg_priv_A']:.4f}  "
      f"avg_priv_B={m_het['avg_priv_B']:.4f}")

print()
print("--- Test 4: Distribution samplers ---")
from fair_estimation.costs import sample_lognormal, sample_truncnorm
c_ln, psi_ln = sample_lognormal(200, 0.5, 0.5**2/3, seed=0)
c_tn, psi_tn = sample_truncnorm(200, 0.5, 0.5**2/3, seed=0)
print(f"  LogNormal  : mean_c={c_ln.mean():.4f} (target 0.5)  min_psi={psi_ln.min():.4f}")
print(f"  TruncNormal: mean_c={c_tn.mean():.4f} (target 0.5)  min_psi={psi_tn.min():.4f}")

print()
print("All tests passed!")
