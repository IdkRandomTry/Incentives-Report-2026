from pathlib import Path
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent.parent / 'results'

# Table 1: exp_1_1
df1 = pd.read_csv(RESULTS_DIR / 'exp_1_1.csv')
print('=== TABLE 1: Exp 1.1 ===')
for p in [0.1, 0.5, 0.9]:
    sub = df1[df1['p_A'] == p]
    for idx, r in sub.iterrows():
        est_name = 'Fair' if r['estimator'] == 'fair' else ('Group-Aware ($\\lambda=0$)' if r['estimator'] == 'no_fairness' else 'Groupless')
        p_str = f"{p:.1f}" if r['estimator'] == 'fair' else "     "
        pool_str = f"({int(r['n_A'])}, {int(r['n_B'])})" if r['estimator'] == 'fair' else "          "
        acq_str = f"{r['n_acquired']:.1f} / {r['n_acquired_A']:.1f} / {r['n_acquired_B']:.1f}"
        print(f"{p_str} & {pool_str} & {est_name:25s} & {r['estimator_mean']:.4f} & {r['true_mean']:.2f} & {r['mse_disparity']:.4f} & {r['mse_pop']:.4f} & {r['instance_payment']:.4f} & {acq_str} & {r['avg_priv_A']:.4f} & {r['avg_priv_B']:.4f} \\\\")
    if p != 0.9:
        print(r'\midrule')

# Table 2: exp_1_2
df2 = pd.read_csv(RESULTS_DIR / 'exp_1_2.csv')
print('\n=== TABLE 2: Exp 1.2 ===')
df2_fair = df2[df2['estimator'] == 'fair']
for idx, r in df2_fair.iterrows():
    acq_str = f"{r['n_acquired']:.1f} / {r['n_acquired_A']:.1f} / {r['n_acquired_B']:.1f}"
    lam_val = r['lambda'] if 'lambda' in r else r['lam']
    print(f"{lam_val:5.1f} & {r['estimator_mean']:.4f} & {r['mse_disparity']:.4f} & {r['mse_pop']:.4f} & {r['mse_A']:.4f} & {r['mse_B']:.4f} & {r['instance_payment']:.4f} & {acq_str} & {r['avg_priv_A']:.4f} & {r['avg_priv_B']:.4f} \\\\")

# Table 3: exp_2_1_1
df3 = pd.read_csv(RESULTS_DIR / 'exp_2_1_1.csv')
print('\n=== TABLE 3: Exp 2.1.1 ===')
for cr in [1.0, 2.0, 4.0, 6.0]:
    sub = df3[df3['cost_ratio'] == cr]
    for idx, r in sub.iterrows():
        est_name = 'Fair' if r['estimator'] == 'fair' else ('Group-Aware ($\\lambda=0$)' if r['estimator'] == 'no_fairness' else 'Groupless')
        cr_str = f"{cr:.1f}" if r['estimator'] == 'fair' else "   "
        acq_str = f"{r['n_acquired']:.1f} / {r['n_acquired_A']:.1f} / {r['n_acquired_B']:.1f}"
        print(f"{cr_str} & {est_name:25s} & {r['estimator_mean']:.4f} & {r['mse_disparity']:.4f} & {r['mse_pop']:.4f} & {r['instance_payment']:.4f} & {acq_str} & {r['avg_priv_A']:.4f} & {r['avg_priv_B']:.4f} \\\\")
    if cr != 6.0:
        print(r'\midrule')

# Table 4: exp_2_1_2
df4 = pd.read_csv(RESULTS_DIR / 'exp_2_1_2.csv')
print('\n=== TABLE 4: Exp 2.1.2 ===')
for p in [0.1, 0.5, 0.9]:
    sub = df4[df4['p_A'] == p]
    for idx, r in sub.iterrows():
        est_name = 'Fair' if r['estimator'] == 'fair' else ('Group-Aware ($\\lambda=0$)' if r['estimator'] == 'no_fairness' else 'Groupless')
        p_str = f"{p:.1f}" if r['estimator'] == 'fair' else "   "
        pool_str = f"({int(r['n_A'])}, {int(r['n_B'])})" if r['estimator'] == 'fair' else "          "
        acq_str = f"{r['n_acquired']:.1f} / {r['n_acquired_A']:.1f} / {r['n_acquired_B']:.1f}"
        print(f"{p_str} & {pool_str} & {est_name:25s} & {r['estimator_mean']:.4f} & {r['mse_disparity']:.4f} & {r['mse_pop']:.4f} & {r['instance_payment']:.4f} & {acq_str} & {r['avg_priv_A']:.4f} & {r['avg_priv_B']:.4f} \\\\")
    if p != 0.9:
        print(r'\midrule')

# Table 5: exp_2_2
df5 = pd.read_csv(RESULTS_DIR / 'exp_2_2.csv')
print('\n=== TABLE 5: Exp 2.2 ===')
for d in ['uniform', 'lognormal', 'truncnorm']:
    d_display = 'Uniform' if d == 'uniform' else ('Log-Normal' if d == 'lognormal' else 'Truncated Normal')
    sub = df5[df5['distribution'] == d]
    for idx, r in sub.iterrows():
        est_name = 'Fair' if r['estimator'] == 'fair' else ('Group-Aware ($\\lambda=0$)' if r['estimator'] == 'no_fairness' else 'Groupless')
        d_str = d_display if r['estimator'] == 'fair' else '                '
        acq_str = f"{r['n_acquired']:.1f} / {r['n_acquired_A']:.1f} / {r['n_acquired_B']:.1f}"
        print(f"{d_str:16s} & {est_name:25s} & {r['estimator_mean']:.4f} & {r['mse_pop']:.4f} & {r['mse_disparity']:.4f} & {r['instance_payment']:.4f} & {acq_str} & {r['avg_priv_A']:.4f} & {r['avg_priv_B']:.4f} \\\\")
    if d != 'truncnorm':
        print(r'\midrule')
