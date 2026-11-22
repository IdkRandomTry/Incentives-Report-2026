"""
Reusable plotting utilities for fair mean estimation experiments.

All public functions accept a DataFrame (from experiments.py) and an x_col
string, produce a matplotlib Figure, and optionally save it to out_path.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------

ESTIMATOR_TYPES = ['fair', 'no_fairness', 'groupless']

STYLES = {
    'fair':        dict(color='#1565C0', marker='o',  ls='-',  lw=2.0,
                        label='Fair  (λ>0)'),
    'no_fairness': dict(color='#E65100', marker='s',  ls='--', lw=1.8,
                        label='Group-aware  (λ=0)'),
    'groupless':   dict(color='#757575', marker='^',  ls=':',  lw=1.5,
                        label='Groupless'),
}

METRIC_LABELS = {
    'estimator_mean':   'Estimator mean  m(w*)',
    'mse_pop':          'Population MSE',
    'mse_A':            'MSE — Group A',
    'mse_B':            'MSE — Group B',
    'mse_disparity':    'Fairness gap  |MSEₐ − MSEᵦ|',
    'instance_payment': 'Total payment  Σ cᵢyᵢ',
    'virtual_payment':  'Virtual payment  Σ ψᵢyᵢ',
    'avg_priv_A':       'Avg privacy loss — Group A',
    'avg_priv_B':       'Avg privacy loss — Group B',
    'total_priv_A':     'Total privacy loss — Group A',
    'total_priv_B':     'Total privacy loss — Group B',
    'fairness_penalty': 'Fairness penalty  λ·Δ²·(2m−μₐ−μᵦ)²',
    'n_active_A':       'Active participants — Group A',
    'n_active_B':       'Active participants — Group B',
    'n_acquired':       'Total acquired participants',
    'n_acquired_A':     'Acquired — Group A',
    'n_acquired_B':     'Acquired — Group B',
    'pct_acquired':     'Total acquired (%)',
}

X_LABELS = {
    'p_A':          'Proportion of Group A  (pₐ)',
    'lambda':       'Fairness strength  λ',
    'cost_ratio':   'Cost ratio  cᵦ / cₐ',
    'distribution': 'Cost distribution',
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fig(nrows=1, ncols=1, figsize=None):
    if figsize is None:
        figsize = (5.5 * ncols, 4.0 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize,
                             facecolor='#FAFAFA')
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes[np.newaxis, :]
    elif ncols == 1:
        axes = axes[:, np.newaxis]
    return fig, axes


def _style_ax(ax, xlabel='', ylabel='', title=''):
    ax.set_xlabel(xlabel or '', fontsize=10)
    ax.set_ylabel(ylabel or '', fontsize=10)
    ax.set_title(title or '', fontsize=11, pad=6)
    ax.grid(True, alpha=0.25, lw=0.8)
    ax.spines[['top', 'right']].set_visible(False)


def _add_refs(ax, df):
    """Add θ and midpoint as horizontal reference lines if constant."""
    if 'true_mean' in df.columns:
        vals = df['true_mean'].dropna().unique()
        if len(vals) == 1:
            ax.axhline(vals[0], color='black', ls='--', lw=1.0, alpha=0.5,
                       label=f'True mean θ={vals[0]:.2f}')
    if 'midpoint' in df.columns:
        vals = df['midpoint'].dropna().unique()
        if len(vals) == 1:
            ax.axhline(vals[0], color='#2E7D32', ls=':', lw=1.0, alpha=0.6,
                       label=f'Midpoint={vals[0]:.2f}')


def _plot_lines(ax, df, x_col, y_col, estimator_types=None):
    """Draw one line per estimator type."""
    if estimator_types is None:
        estimator_types = ESTIMATOR_TYPES
    for et in estimator_types:
        sub = df[df['estimator'] == et].sort_values(x_col)
        if sub.empty:
            continue
        ax.plot(sub[x_col], sub[y_col],
                **{k: v for k, v in STYLES[et].items()},
                markersize=6, zorder=3)


def _save(fig, out_path):
    if out_path is not None:
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)


def _xlbl(x_col):
    return X_LABELS.get(x_col, x_col.replace('_', ' ').title())


# ---------------------------------------------------------------------------
# Public plots
# ---------------------------------------------------------------------------

def plot_estimator_means(df: pd.DataFrame, x_col: str,
                          out_path=None, title=None):
    """
    Estimator mean m(w*) vs. x_col for all estimator types,
    with reference lines for θ and the group midpoint.
    """
    fig, axes = _fig(1, 1)
    ax = axes[0, 0]

    _plot_lines(ax, df, x_col, 'estimator_mean')
    _add_refs(ax, df)

    _style_ax(ax, _xlbl(x_col), METRIC_LABELS['estimator_mean'],
              title or f'Estimator Mean vs. {_xlbl(x_col)}')
    ax.legend(fontsize=8, loc='best')
    fig.tight_layout()
    _save(fig, out_path)
    return fig, ax


def plot_group_payments(df: pd.DataFrame, x_col: str = 'lambda',
                        out_path=None, title=None):
    """
    Plot average per-capita payment for both groups (avg_pay_A, avg_pay_B)
    on the same graph vs. x_col.
    """
    fair = df[df['estimator'] == 'fair'].sort_values(x_col)

    fig, axes = _fig(1, 1, figsize=(5.5, 4.0))
    ax = axes[0, 0]

    if 'avg_pay_A' in fair.columns:
        ax.plot(fair[x_col], fair['avg_pay_A'], label=r'Group A (Minority, $\bar{t}_A$)',
                color='#D32F2F', marker='o', ls='-', lw=2.0, markersize=5.5)
    if 'avg_pay_B' in fair.columns:
        ax.plot(fair[x_col], fair['avg_pay_B'], label=r'Group B (Majority, $\bar{t}_B$)',
                color='#1976D2', marker='s', ls='-', lw=2.0, markersize=5.5)

    _style_ax(ax, _xlbl(x_col), r'Average Payment per Capita ($\bar{t}_g$)',
              title or f'Group Average Payments vs. {_xlbl(x_col)}')
    ax.legend(fontsize=8, loc='best')
    fig.tight_layout()
    _save(fig, out_path)
    return fig, ax


def plot_mse_panel(df: pd.DataFrame, x_col: str,
                    out_path=None, title=None):
    """
    2 × 2 panel: population MSE, Group A MSE, Group B MSE, fairness gap.
    """
    metrics = ['mse_pop', 'mse_A', 'mse_B', 'mse_disparity']
    fig, axes = _fig(2, 2, figsize=(12, 8))

    for ax, metric in zip(axes.flatten(), metrics):
        _plot_lines(ax, df, x_col, metric)
        _style_ax(ax, _xlbl(x_col), METRIC_LABELS[metric], METRIC_LABELS[metric])
        ax.legend(fontsize=7)

    fig.suptitle(title or f'MSE breakdown vs. {_xlbl(x_col)}', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, out_path)
    return fig, axes


def plot_mse_combined(df: pd.DataFrame, x_col: str = 'lambda',
                      out_path=None, title=None):
    """
    Plot the 4 key MSE metrics on the same single graph with distinct lines:
    1. Group A MSE (mse_A)
    2. Group B MSE (mse_B)
    3. Population MSE (mse_pop)
    4. Fairness Disparity |MSE_A - MSE_B| (mse_disparity)

    Filters for the 'fair' estimator to show dynamic evolution across x_col.
    """
    fair = df[df['estimator'] == 'fair'].sort_values(x_col)

    fig, axes = _fig(1, 1, figsize=(5.5, 4.0))
    ax = axes[0, 0]

    curves = [
        ('mse_A',         'MSE — Group A',               '#D32F2F', 'o', '-'),
        ('mse_B',         'MSE — Group B',               '#1976D2', 's', '-'),
        ('mse_disparity', 'Fairness gap |MSEₐ − MSEᵦ|',   '#E65100', '^', '--'),
        ('mse_pop',       'Population MSE',             '#2E7D32', 'd', '-.'),
    ]

    for col, label, color, marker, ls in curves:
        if col in fair.columns:
            ax.plot(fair[x_col], fair[col], label=label, color=color,
                    marker=marker, ls=ls, lw=2.0, markersize=5.5)

    _style_ax(ax, _xlbl(x_col), 'Mean Squared Error (MSE)',
              title or f'MSE Breakdown vs. {_xlbl(x_col)}')
    ax.legend(fontsize=8, loc='best')
    fig.tight_layout()
    _save(fig, out_path)
    return fig, ax


def plot_group_payments(df: pd.DataFrame, x_col: str = 'lambda',
                        out_path=None, title=None):
    """
    Plot average per-capita payment for both groups (avg_pay_A, avg_pay_B)
    on the same graph vs. x_col.
    """
    fair = df[df['estimator'] == 'fair'].sort_values(x_col)

    fig, axes = _fig(1, 1, figsize=(5.5, 4.0))
    ax = axes[0, 0]

    if 'avg_pay_A' in fair.columns:
        ax.plot(fair[x_col], fair['avg_pay_A'], label=r'Group A (Minority, $\bar{t}_A$)',
                color='#D32F2F', marker='o', ls='-', lw=2.0, markersize=5.5)
    if 'avg_pay_B' in fair.columns:
        ax.plot(fair[x_col], fair['avg_pay_B'], label=r'Group B (Majority, $\bar{t}_B$)',
                color='#1976D2', marker='s', ls='-', lw=2.0, markersize=5.5)

    _style_ax(ax, _xlbl(x_col), r'Average Payment per Capita ($\bar{t}_g$)',
              title or f'Group Average Payments vs. {_xlbl(x_col)}')
    ax.legend(fontsize=8, loc='best')
    fig.tight_layout()
    _save(fig, out_path)
    return fig, ax


def plot_payments(df: pd.DataFrame, x_col: str,
                   out_path=None, title=None, include_virtual=False):
    """
    Plot total instance payment Σcᵢyᵢ vs. x_col.

    By default, produces a single-panel plot with figsize=(5.5, 4.0), perfectly
    matching plot_estimator_means in dimensions, aspect ratio, and font scale.
    If include_virtual=True, produces a 2-panel figure with virtual payments.
    """
    if include_virtual:
        fig, axes = _fig(1, 2, figsize=(12, 4))
        for ax, metric in zip(axes[0], ['instance_payment', 'virtual_payment']):
            _plot_lines(ax, df, x_col, metric)
            _style_ax(ax, _xlbl(x_col), METRIC_LABELS[metric], METRIC_LABELS[metric])
            ax.legend(fontsize=7)
        fig.suptitle(title or f'Payments vs. {_xlbl(x_col)}', fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        _save(fig, out_path)
        return fig, axes

    fig, axes = _fig(1, 1)
    ax = axes[0, 0]
    _plot_lines(ax, df, x_col, 'instance_payment')
    _style_ax(ax, _xlbl(x_col), METRIC_LABELS['instance_payment'],
              title or f'Total Payment vs. {_xlbl(x_col)}')
    ax.legend(fontsize=8, loc='best')
    fig.tight_layout()
    _save(fig, out_path)
    return fig, ax


def plot_group_payments(df: pd.DataFrame, x_col: str = 'lambda',
                        out_path=None, title=None):
    """
    Plot average per-capita payment for both groups (avg_pay_A, avg_pay_B)
    on the same graph vs. x_col.
    """
    fair = df[df['estimator'] == 'fair'].sort_values(x_col)

    fig, axes = _fig(1, 1, figsize=(5.5, 4.0))
    ax = axes[0, 0]

    if 'avg_pay_A' in fair.columns:
        ax.plot(fair[x_col], fair['avg_pay_A'], label=r'Group A (Minority, $\bar{t}_A$)',
                color='#D32F2F', marker='o', ls='-', lw=2.0, markersize=5.5)
    if 'avg_pay_B' in fair.columns:
        ax.plot(fair[x_col], fair['avg_pay_B'], label=r'Group B (Majority, $\bar{t}_B$)',
                color='#1976D2', marker='s', ls='-', lw=2.0, markersize=5.5)

    _style_ax(ax, _xlbl(x_col), r'Average Payment per Capita ($\bar{t}_g$)',
              title or f'Group Average Payments vs. {_xlbl(x_col)}')
    ax.legend(fontsize=8, loc='best')
    fig.tight_layout()
    _save(fig, out_path)
    return fig, ax


def plot_group_payments(df: pd.DataFrame, x_col: str = 'lambda',
                        out_path=None, title=None):
    """
    Plot average per-capita payment for both groups (avg_pay_A, avg_pay_B)
    on the same graph vs. x_col.
    """
    fair = df[df['estimator'] == 'fair'].sort_values(x_col)

    fig, axes = _fig(1, 1, figsize=(5.5, 4.0))
    ax = axes[0, 0]

    if 'avg_pay_A' in fair.columns:
        ax.plot(fair[x_col], fair['avg_pay_A'], label=r'Group A (Minority, $\bar{t}_A$)',
                color='#D32F2F', marker='o', ls='-', lw=2.0, markersize=5.5)
    if 'avg_pay_B' in fair.columns:
        ax.plot(fair[x_col], fair['avg_pay_B'], label=r'Group B (Majority, $\bar{t}_B$)',
                color='#1976D2', marker='s', ls='-', lw=2.0, markersize=5.5)

    _style_ax(ax, _xlbl(x_col), r'Average Payment per Capita ($\bar{t}_g$)',
              title or f'Exp 2 — Group Average Payments vs. {_xlbl(x_col)}')
    ax.legend(fontsize=8, loc='best')
    fig.tight_layout()
    _save(fig, out_path)
    return fig, ax


def plot_privacy_loss(df: pd.DataFrame, x_col: str,
                       out_path=None, title=None):
    """
    Average privacy loss per group (ȳₐ, ȳᵦ) vs. x_col.
    """
    fig, axes = _fig(1, 2, figsize=(12, 4))

    for ax, (metric, grp) in zip(axes[0],
                                   [('avg_priv_A', 'A'), ('avg_priv_B', 'B')]):
        _plot_lines(ax, df, x_col, metric)
        _style_ax(ax, _xlbl(x_col), f'Avg privacy loss — Group {grp}',
                  f'Avg privacy loss — Group {grp}')
        ax.legend(fontsize=7)

    fig.suptitle(title or f'Privacy Loss vs. {_xlbl(x_col)}', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save(fig, out_path)
    return fig, axes


def plot_mse_payment_tradeoff(df: pd.DataFrame, x_col: str = 'lambda',
                               out_path=None, title=None):
    """
    Scatter: population MSE vs. instance payment for the fair estimator
    as λ varies, showing the MSE–payment–fairness tradeoff.
    """
    sub  = df[df['estimator'] == 'fair'].sort_values(x_col)
    fig, axes = _fig(1, 1)
    ax = axes[0, 0]

    sc = ax.scatter(sub['instance_payment'], sub['mse_pop'],
                    c=sub[x_col], cmap='viridis', s=80, zorder=3)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(f'λ', fontsize=9)

    ax.set_xlabel(METRIC_LABELS['instance_payment'], fontsize=10)
    ax.set_ylabel(METRIC_LABELS['mse_pop'], fontsize=10)
    _style_ax(ax, title=title or 'MSE vs. Payment tradeoff (fair estimator)')
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, out_path)
    return fig, ax


def plot_group_payments(df: pd.DataFrame, x_col: str = 'lambda',
                        out_path=None, title=None):
    """
    Plot average per-capita payment for both groups (avg_pay_A, avg_pay_B)
    on the same graph vs. x_col.
    """
    fair = df[df['estimator'] == 'fair'].sort_values(x_col)

    fig, axes = _fig(1, 1, figsize=(5.5, 4.0))
    ax = axes[0, 0]

    if 'avg_pay_A' in fair.columns:
        ax.plot(fair[x_col], fair['avg_pay_A'], label=r'Group A (Minority, $\bar{t}_A$)',
                color='#D32F2F', marker='o', ls='-', lw=2.0, markersize=5.5)
    if 'avg_pay_B' in fair.columns:
        ax.plot(fair[x_col], fair['avg_pay_B'], label=r'Group B (Majority, $\bar{t}_B$)',
                color='#1976D2', marker='s', ls='-', lw=2.0, markersize=5.5)

    _style_ax(ax, _xlbl(x_col), r'Average Payment per Capita ($\bar{t}_g$)',
              title or f'Group Average Payments vs. {_xlbl(x_col)}')
    ax.legend(fontsize=8, loc='best')
    fig.tight_layout()
    _save(fig, out_path)
    return fig, ax


def plot_distribution_comparison(df: pd.DataFrame, out_path=None, title=None):
    """
    Grouped bar chart: compare distributions across key metrics.
    Used in Experiment 2.2.
    """
    dist_types = df['distribution'].unique().tolist()
    metrics    = ['mse_pop', 'mse_disparity', 'instance_payment',
                  'n_acquired', 'avg_priv_A', 'avg_priv_B']

    n_dists  = len(dist_types)
    n_et     = len(ESTIMATOR_TYPES)
    fig, axes = _fig(1, len(metrics), figsize=(4.2 * len(metrics), 5))

    bar_w   = 0.22
    offsets = np.linspace(-(n_et - 1) / 2 * bar_w,
                           (n_et - 1) / 2 * bar_w, n_et)
    x = np.arange(n_dists)

    for ax, metric in zip(axes[0], metrics):
        for i, et in enumerate(ESTIMATOR_TYPES):
            sub  = df[df['estimator'] == et]
            vals = [float(sub[sub['distribution'] == d][metric].mean())
                    if not sub[sub['distribution'] == d].empty else 0.0
                    for d in dist_types]
            s = STYLES[et]
            ax.bar(x + offsets[i], vals, bar_w,
                   color=s['color'], alpha=0.85, label=s['label'],
                   edgecolor='white', lw=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels(
            [d.replace('truncnorm', 'trunc.\nnormal') for d in dist_types],
            fontsize=8)
        ax.set_ylabel(METRIC_LABELS.get(metric, metric), fontsize=9)
        ax.set_title(METRIC_LABELS.get(metric, metric), fontsize=9, pad=4)
        ax.legend(fontsize=6, loc='upper right')
        ax.grid(True, alpha=0.2, axis='y')
        ax.spines[['top', 'right']].set_visible(False)

    fig.suptitle(title or 'Experiment 4 — Distribution Comparison', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save(fig, out_path)
    return fig, axes


def plot_active_participants(df: pd.DataFrame, x_col: str,
                              out_path=None, title=None):
    """Number of acquired participants (wᵢ > 1e-5) overall and per group."""
    fig, axes = _fig(1, 3, figsize=(16, 4))

    panels = [
        ('n_acquired',   'Total Acquired'),
        ('n_acquired_A', 'Group A Acquired'),
        ('n_acquired_B', 'Group B Acquired'),
    ]

    for ax, (metric, label) in zip(axes[0], panels):
        _plot_lines(ax, df, x_col, metric)
        _style_ax(ax, _xlbl(x_col), f'{label}', f'{label}')
        ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.legend(fontsize=7)

    fig.suptitle(title or f'Data-acquired participants vs. {_xlbl(x_col)}', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save(fig, out_path)
    return fig, axes
