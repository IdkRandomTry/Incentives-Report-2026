"""
Privacy cost distributions and virtual cost (ψ) computation.

For a participant with private cost c drawn from distribution F with density f,
Myerson's virtual cost is:

    ψ(c) = c + F(c) / f(c)

Implemented distributions
--------------------------
  Uniform  U(0, 2μ)         ψ(c) = 2c               (closed-form)
  LogNormal LN(μ_L, σ_L²)   ψ(c) = c·(1 + σ_L·Φ(z)/φ(z))   (analytical)
  Truncated Normal TN≥0      ψ(c) = c + F_TN(c)/f_TN(c)      (numerical)

All samplers return (costs, virtual_costs) as numpy arrays of shape (n,).
Costs are guaranteed positive; virtual costs may exceed costs.
"""

import numpy as np
from scipy.stats import norm, truncnorm
from scipy.optimize import fsolve
from typing import Literal

DistType = Literal['uniform', 'lognormal', 'truncnorm']


# ---------------------------------------------------------------------------
# Uniform U(0, 2μ)
# ---------------------------------------------------------------------------

def sample_uniform(n: int, mean_c: float, seed: int = 0):
    """
    Sample n costs from Uniform(0, 2·mean_c).

    Mean = mean_c, Var = mean_c²/3.
    Virtual cost: ψ(c) = c + F/f = c + (c−0)·(2·mean_c) / (2·mean_c) = c + c = 2c.
    """
    rng   = np.random.default_rng(seed)
    costs = rng.uniform(0.0, 2.0 * mean_c, n)
    # Ensure no zeros (would give ψ = 0, which is degenerate)
    costs = np.maximum(costs, 1e-10)
    virtual = 2.0 * costs
    return costs, virtual


# ---------------------------------------------------------------------------
# Log-Normal LN(μ_L, σ_L²)
# ---------------------------------------------------------------------------

def _lognorm_params(mean_c: float, var_c: float):
    """
    Solve for (μ_L, σ_L) of LN(μ_L, σ_L²) matching given mean and variance.

    mean_c = exp(μ_L + σ_L²/2)
    var_c  = (exp(σ_L²) − 1) · mean_c²
    """
    sigma_L2 = np.log(1.0 + var_c / mean_c**2)
    mu_L     = np.log(mean_c) - 0.5 * sigma_L2
    return mu_L, np.sqrt(sigma_L2)


def sample_lognormal(n: int, mean_c: float, var_c: float, seed: int = 0):
    """
    Sample n costs from LogNormal matched to (mean_c, var_c).

    For LN(μ_L, σ_L²) with z = (log c − μ_L)/σ_L:
        F(c)/f(c) = c · σ_L · Φ(z) / φ(z)
        ψ(c) = c + c · σ_L · Φ(z)/φ(z)
    """
    mu_L, sigma_L = _lognorm_params(mean_c, var_c)
    rng   = np.random.default_rng(seed)
    costs = rng.lognormal(mu_L, sigma_L, n)

    z       = (np.log(costs) - mu_L) / sigma_L
    Phi_z   = norm.cdf(z)
    phi_z   = np.maximum(norm.pdf(z), 1e-15)
    virtual = costs * (1.0 + sigma_L * Phi_z / phi_z)
    return costs, virtual


# ---------------------------------------------------------------------------
# Truncated Normal TN(μ, σ², 0, ∞)
# ---------------------------------------------------------------------------

def _truncnorm_params(mean_c: float, var_c: float):
    """
    Solve for (μ, σ) of TN(μ, σ², 0, ∞) matching given mean and variance.
    Uses scipy.optimize.fsolve; falls back to (mean_c, sqrt(var_c)) on failure.
    """
    def equations(params):
        mu, log_sigma = params
        sigma = np.exp(log_sigma)
        a = -mu / max(sigma, 1e-10)
        rv = truncnorm(a=a, b=np.inf, loc=mu, scale=sigma)
        return [rv.mean() - mean_c,
                rv.var()  - var_c]

    sigma_init = np.sqrt(var_c)
    try:
        sol = fsolve(equations, [mean_c, np.log(sigma_init)], full_output=True)
        mu_sol, log_sigma_sol = sol[0]
        sigma_sol = np.exp(log_sigma_sol)
        # Sanity check: recovered moments should be close
        a = -mu_sol / max(sigma_sol, 1e-10)
        rv = truncnorm(a=a, b=np.inf, loc=mu_sol, scale=sigma_sol)
        if abs(rv.mean() - mean_c) < 0.1 * mean_c and sigma_sol > 0:
            return float(mu_sol), float(sigma_sol)
    except Exception:
        pass
    return mean_c, sigma_init   # fallback


def sample_truncnorm(n: int, mean_c: float, var_c: float, seed: int = 0):
    """
    Sample n costs from Truncated Normal on [0, ∞) matched to (mean_c, var_c).

    Virtual cost computed numerically using scipy.stats.truncnorm:
        ψ(c) = c + F_TN(c)/f_TN(c)
    """
    mu, sigma = _truncnorm_params(mean_c, var_c)
    a_std = -mu / max(sigma, 1e-10)      # lower truncation in standard form
    rv    = truncnorm(a=a_std, b=np.inf, loc=mu, scale=sigma)

    rng   = np.random.default_rng(seed)
    costs = rv.rvs(n, random_state=rng)
    costs = np.maximum(costs, 1e-10)

    F_vals = rv.cdf(costs)
    f_vals = np.maximum(rv.pdf(costs), 1e-15)
    virtual = costs + F_vals / f_vals
    return costs, virtual


# ---------------------------------------------------------------------------
# Dispatch helpers
# ---------------------------------------------------------------------------

def sample_costs(dist_type: DistType, n: int, mean_c: float,
                 var_c: float = None, seed: int = 0):
    """
    Sample costs and virtual costs from the named distribution.

    For 'uniform', var_c is ignored (variance is mean_c²/3 from U(0, 2·mean_c)).
    For 'lognormal' and 'truncnorm', var_c defaults to mean_c²/3 (same as uniform).

    Returns
    -------
    costs   : (n,) actual costs cᵢ > 0
    virtual : (n,) virtual costs ψᵢ
    """
    if dist_type == 'uniform':
        return sample_uniform(n, mean_c, seed)

    if var_c is None:
        var_c = mean_c**2 / 3.0     # match uniform variance by default

    if dist_type == 'lognormal':
        return sample_lognormal(n, mean_c, var_c, seed)
    if dist_type == 'truncnorm':
        return sample_truncnorm(n, mean_c, var_c, seed)

    raise ValueError(f"Unknown dist_type: {dist_type!r}. "
                     "Choose 'uniform', 'lognormal', or 'truncnorm'.")


def group_costs(dist_type: DistType,
                n_A: int, mean_A: float,
                n_B: int, mean_B: float,
                var_A: float = None, var_B: float = None,
                seed: int = 0):
    """
    Sample costs for two groups independently (different mean / variance).

    Returns
    -------
    costs   : (n_A + n_B,) concatenated costs    [group A first]
    virtual : (n_A + n_B,) concatenated ψ values
    """
    c_A, psi_A = sample_costs(dist_type, n_A, mean_A, var_A, seed=seed)
    c_B, psi_B = sample_costs(dist_type, n_B, mean_B, var_B, seed=seed + 10_000)
    return (np.concatenate([c_A, c_B]),
            np.concatenate([psi_A, psi_B]))
