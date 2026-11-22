"""
Fair Mean Estimation — Core Model, Solver, and Metrics.

Mechanism
---------
  θ̂ = Σᵢ wᵢXᵢ + Z,   Z ~ Lap(1/S),   S = Σᵢ yᵢ,   yᵢ = S·wᵢ

Platform objective (fairness-augmented, in w and S)
----------------------------------------------------
  J(w,S) = (n+1) [σ²·Σwᵢ² + 2/S² + (m(w)−θ)²]
           + λ·Δ²·[2m(w)−μₐ−μᵦ]²
           + S · Σᵢ ψᵢwᵢ

where
  m(w)  = Σᵢ wᵢ·μg(i)   estimator mean
  θ     = pₐ·μₐ + pᵦ·μᵦ  true population mean
  Δ     = μᵦ − μₐ        group mean gap
  ψᵢ    = virtual cost    = cᵢ + Fᵢ(cᵢ)/fᵢ(cᵢ)

Three estimator modes
---------------------
  'fair'        : full objective with λ from GroupParams
  'no_fairness' : same but λ_eff = 0  (group-aware, optimal MSE-payment)
  'groupless'   : μg(i) = θ for all i, λ_eff = 0
                  (not applicable when costs are group-dependent)
"""

import numpy as np
from scipy.optimize import minimize
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

@dataclass
class GroupParams:
    """All parameters for a two-group fair estimation problem."""
    mu_A:   float          # Group A population mean
    mu_B:   float          # Group B population mean
    sigma2: float          # Common within-group variance σ²
    n_A:    int            # Participants in Group A
    n_B:    int            # Participants in Group B
    lam:    float = 0.0    # Fairness strength λ ≥ 0

    # -- Derived --

    @property
    def n(self):       return self.n_A + self.n_B
    @property
    def p_A(self):     return self.n_A / self.n
    @property
    def p_B(self):     return self.n_B / self.n
    @property
    def theta(self):   return self.p_A * self.mu_A + self.p_B * self.mu_B
    @property
    def Delta(self):   return self.mu_B - self.mu_A
    @property
    def midpoint(self): return (self.mu_A + self.mu_B) / 2.0
    @property
    def mu_g(self):
        """Group mean for each participant, shape (n,)."""
        return np.array([self.mu_A] * self.n_A + [self.mu_B] * self.n_B)


# ---------------------------------------------------------------------------
# Estimator-type configuration
# ---------------------------------------------------------------------------

_CONFIGS = {
    # (use actual group means?,  use lambda from params?)
    'fair':        (True,  True),
    'no_fairness': (True,  False),
    'groupless':   (False, False),
}

ESTIMATOR_LABELS = {
    'fair':        'Fair (λ>0)',
    'no_fairness': 'Group-aware, no fairness (λ=0)',
    'groupless':   'Groupless',
}


def _effective(estimator_type, params):
    """Return (mu_g_eff, lam_eff) for the given estimator type."""
    use_groups, use_lam = _CONFIGS[estimator_type]
    mu_g_eff = params.mu_g if use_groups else np.full(params.n, params.theta)
    lam_eff  = params.lam  if use_lam    else 0.0
    return mu_g_eff, lam_eff


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

def objective(w, S, psi, mu_g_eff, params, lam_eff):
    """
    Evaluate J(w, S).

    Parameters
    ----------
    w        : (n,) weights on participants (must sum to 1)
    S        : float, total privacy budget  (must be > 0)
    psi      : (n,) virtual costs
    mu_g_eff : (n,) effective group means (params.mu_g or θ for groupless)
    params   : GroupParams
    lam_eff  : effective λ
    """
    n1 = params.n + 1
    m  = w @ mu_g_eff

    variance = params.sigma2 * (w @ w) + 2.0 / S**2
    bias_sq  = (m - params.theta) ** 2
    fairness = lam_eff * params.Delta**2 * (2*m - params.mu_A - params.mu_B)**2
    payment  = S * (psi @ w)

    return n1 * (variance + bias_sq) + fairness + payment


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

def solve(psi, params, estimator_type='fair', n_restarts=3, seed=0):
    """
    Minimise J(w, S) over the probability simplex × ℝ₊ using SLSQP.

    Parameters
    ----------
    psi            : (n,) virtual costs
    params         : GroupParams
    estimator_type : 'fair' | 'no_fairness' | 'groupless'
    n_restarts     : number of random restarts (3 deterministic + rest random)
    seed           : RNG seed for random restarts

    Returns
    -------
    dict  {'w': (n,), 'S': float, 'm': float, 'obj': float}
    """
    mu_g_eff, lam_eff = _effective(estimator_type, params)
    n   = params.n
    rng = np.random.default_rng(seed)

    # --- Objective wrapper (x = [w₁…wₙ, S]) ---
    def obj_fn(x):
        w, S = x[:n], x[n]
        return objective(w, S, psi, mu_g_eff, params, lam_eff)

    # Simplex constraint
    constraint = {'type': 'eq', 'fun': lambda x: x[:n].sum() - 1.0}
    bounds     = [(0.0, 1.0)] * n + [(1e-4, 1e3)]

    # Starting S heuristic from envelope condition at uniform weights
    psi_avg = max(float(np.mean(psi)), 1e-8)
    S0      = max((4.0 * (n + 1) / psi_avg) ** (1.0 / 3.0), 0.05)

    starts = [
        np.append(np.ones(n) / n, S0),
        np.append(np.ones(n) / n, S0 * 0.4),
        np.append(np.ones(n) / n, S0 * 2.5),
    ]
    for _ in range(max(0, n_restarts - 3)):
        starts.append(np.append(rng.dirichlet(np.ones(n)),
                                S0 * rng.uniform(0.2, 4.0)))

    best_x, best_v = starts[0], np.inf

    for x0 in starts:
        try:
            res = minimize(obj_fn, x0,
                           method='SLSQP',
                           bounds=bounds,
                           constraints=constraint,
                           options={'ftol': 1e-12, 'maxiter': 3000})
            if res.fun < best_v:
                best_v, best_x = res.fun, res.x.copy()
        except Exception:
            pass

    # Numerical clean-up
    w_opt = np.clip(best_x[:n], 0.0, None)
    w_opt /= max(w_opt.sum(), 1e-12)
    S_opt  = float(np.clip(best_x[n], 1e-4, 1e3))
    m_opt  = float(w_opt @ mu_g_eff)

    return {'w': w_opt, 'S': S_opt, 'm': m_opt, 'obj': best_v}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(sol, psi, costs, params, estimator_type='fair'):
    """
    Compute all experiment metrics from a solved allocation.

    Parameters
    ----------
    sol            : dict from solve()
    psi            : (n,) virtual costs
    costs          : (n,) actual privacy costs cᵢ
    params         : GroupParams
    estimator_type : str

    Returns
    -------
    dict of floats — all metrics tracked in the experiments.
    """
    mu_g_eff, _ = _effective(estimator_type, params)

    w, S = sol['w'], sol['S']
    n_A, n_B = params.n_A, params.n_B

    y = S * w                           # actual privacy losses yᵢ
    m = float(w @ mu_g_eff)             # estimator mean (using effective μg)

    # Variance component (noise + sampling variance)
    var_term = params.sigma2 * float(w @ w) + 2.0 / S**2

    # MSE breakdown
    mse_pop = var_term + (m - params.theta) ** 2
    mse_A   = var_term + (m - params.mu_A)  ** 2
    mse_B   = var_term + (m - params.mu_B)  ** 2

    # Group privacy losses
    y_A, y_B = y[:n_A], y[n_A:]

    # Payments
    #   instance_payment  = Σᵢ cᵢ·yᵢ   (direct cost paid to participant)
    #   virtual_payment   = Σᵢ ψᵢ·yᵢ   (proxy for E[payment] via virtual cost)
    costs_A, costs_B = costs[:n_A], costs[n_A:]
    pay_A = float(costs_A @ y_A) if n_A > 0 else 0.0
    pay_B = float(costs_B @ y_B) if n_B > 0 else 0.0
    inst_pay = pay_A + pay_B
    virt_pay = float(psi   @ y)

    # Acquired participants (data acquired and weighted, wᵢ > 1e-5)
    w_A, w_B = w[:n_A], w[n_A:]
    w_thr = 1e-5
    n_acq_A = int((w_A > w_thr).sum())
    n_acq_B = int((w_B > w_thr).sum())
    n_acq_tot = n_acq_A + n_acq_B

    # Fairness disparity (always computed, regardless of estimator type)
    # Measures how much unfairness this allocation incurs
    disp_val = params.Delta**2 * (2*m - params.mu_A - params.mu_B)**2

    return {
        # Reference means
        'estimator_mean': m,
        'true_mean':      params.theta,
        'midpoint':       params.midpoint,

        # MSE
        'mse_pop':      mse_pop,
        'mse_A':        mse_A,
        'mse_B':        mse_B,
        'mse_disparity': abs(mse_A - mse_B),   # |MSEₐ − MSE_B|

        # Fairness penalty value (Δ²·(2m−μₐ−μᵦ)², without λ)
        # For 'fair' estimator: params.lam × this = penalty in objective
        'disparity_sq':      disp_val,
        'fairness_penalty':  params.lam * disp_val,

        # Total privacy budget
        'S': S,

        # Group privacy losses
        'total_priv_A': float(y_A.sum()),
        'total_priv_B': float(y_B.sum()),
        'avg_priv_A':   float(y_A.mean()) if n_A > 0 else 0.0,
        'avg_priv_B':   float(y_B.mean()) if n_B > 0 else 0.0,
        'total_priv':   float(y.sum()),

        # Acquired participants (from whom data is acquired)
        'n_acquired':     n_acq_tot,
        'n_acquired_A':   n_acq_A,
        'n_acquired_B':   n_acq_B,
        'pct_acquired':   (n_acq_tot / params.n) * 100.0 if params.n > 0 else 0.0,
        'pct_acquired_A': (n_acq_A / n_A) * 100.0 if n_A > 0 else 0.0,
        'pct_acquired_B': (n_acq_B / n_B) * 100.0 if n_B > 0 else 0.0,

        # Backward compatibility
        'n_active_A': n_acq_A,
        'n_active_B': n_acq_B,

        # Payments
        'instance_payment': inst_pay,
        'virtual_payment':  virt_pay,
        'total_pay_A':      pay_A,
        'total_pay_B':      pay_B,
        'avg_pay_A':        pay_A / n_A if n_A > 0 else 0.0,
        'avg_pay_B':        pay_B / n_B if n_B > 0 else 0.0,
    }
