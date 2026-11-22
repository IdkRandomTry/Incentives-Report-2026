from pathlib import Path
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent.parent / 'results'

def format_all_tables():
    # Table 1: Exp 1.1
    df1 = pd.read_csv(RESULTS_DIR / 'exp_1_1.csv')
    print('=== TABLE 1 (Exp 1.1) ===')
    print(r'''\begin{table}[htbp]
\centering
\caption{Experiment 1.1: Performance, sample sizes, and participant acquisition across demographic proportions $p_A$ (total population $n = 50$, homogeneous costs, $\lambda = 10.0$, 10 Monte Carlo draws).}
\label{tab:exp1_1}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lrrrrrrrrr}
\toprule
Estimator & $\widehat{m}$ & Disparity & $\MSE_{\mathrm{pop}}$ & Payment & Acquired ($N / A / B$) & $\bar{\epsilon}_A$ & $\bar{\epsilon}_B$ & $\bar{t}_A$ & $\bar{t}_B$ \\
\midrule''')
    for p in [0.1, 0.5, 0.9]:
        regime = "Minority Group A" if p == 0.1 else ("Equal Proportions" if p == 0.5 else "Majority Group A")
        sub = df1[df1['p_A'] == p]
        r0 = sub.iloc[0]
        print(f"\\multicolumn{{10}}{{l}}{{\\textbf{{{regime}}} \\quad ($p_A = {p:.1f}$, \\ Pool: $n_A = {int(r0['n_A'])}, n_B = {int(r0['n_B'])}$, \\ Target $\\theta = {r0['true_mean']:.2f}$)}} \\\\")
        print(r'\midrule')
        for idx, r in sub.iterrows():
            est_name = 'Fair' if r['estimator'] == 'fair' else ('Group-Aware' if r['estimator'] == 'no_fairness' else 'Groupless')
            acq_str = f"{r['n_acquired']:.1f} / {r['n_acquired_A']:.1f} / {r['n_acquired_B']:.1f}"
            print(f"{est_name:12s} & {r['estimator_mean']:.4f} & {r['mse_disparity']:.4f} & {r['mse_pop']:.4f} & {r['instance_payment']:.4f} & {acq_str} & {r['avg_priv_A']:.4f} & {r['avg_priv_B']:.4f} & {r['avg_pay_A']:.4f} & {r['avg_pay_B']:.4f} \\\\")
        if p != 0.9:
            print(r'\midrule')
    print(r'''\bottomrule
\end{tabular}%
}
\end{table}''')

    # Table 2: Exp 1.2
    df2 = pd.read_csv(RESULTS_DIR / 'exp_1_2.csv')
    print('\n=== TABLE 2 (Exp 1.2) ===')
    print(r'''\begin{table}[htbp]
\centering
\caption{Experiment 1.2: Dynamics under increasing fairness strength $\lambda$ ($p_A = 0.3$, $n_A = 15, n_B = 35$, total $n = 50$, 10 Monte Carlo draws).}
\label{tab:exp1_2}
\resizebox{\textwidth}{!}{%
\begin{tabular}{rrrrrrrrrrrr}
\toprule
$\lambda$ & $\widehat{m}$ & Disparity & $\MSE_{\mathrm{pop}}$ & $\MSE_A$ & $\MSE_B$ & Payment & Acquired ($N / A / B$) & $\bar{\epsilon}_A$ & $\bar{\epsilon}_B$ & $\bar{t}_A$ & $\bar{t}_B$ \\
\midrule''')
    df2_fair = df2[df2['estimator'] == 'fair']
    for idx, r in df2_fair.iterrows():
        acq_str = f"{r['n_acquired']:.1f} / {r['n_acquired_A']:.1f} / {r['n_acquired_B']:.1f}"
        lam_val = r['lambda'] if 'lambda' in r else r['lam']
        print(f"{lam_val:5.1f} & {r['estimator_mean']:.4f} & {r['mse_disparity']:.4f} & {r['mse_pop']:.4f} & {r['mse_A']:.4f} & {r['mse_B']:.4f} & {r['instance_payment']:.4f} & {acq_str} & {r['avg_priv_A']:.4f} & {r['avg_priv_B']:.4f} & {r['avg_pay_A']:.4f} & {r['avg_pay_B']:.4f} \\\\")
    print(r'''\bottomrule
\end{tabular}%
}
\end{table}''')

    # Table 3: Exp 2.1.1
    df3 = pd.read_csv(RESULTS_DIR / 'exp_2_1_1.csv')
    print('\n=== TABLE 3 (Exp 2.1.1) ===')
    print(r'''\begin{table}[htbp]
\centering
\caption{Experiment 2.1.1: Cost ratio sweep with equal demographic sizes ($n_A = 25, n_B = 25$, total $n = 50$, $\lambda = 10.0$, 10 Monte Carlo draws). \emph{Note: Groupless estimator omitted due to theoretical inapplicability.}}
\label{tab:exp2_1_1}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lrrrrrrrrr}
\toprule
Estimator & $\widehat{m}$ & Disparity & $\MSE_{\mathrm{pop}}$ & Payment & Acquired ($N / A / B$) & $\bar{\epsilon}_A$ & $\bar{\epsilon}_B$ & $\bar{t}_A$ & $\bar{t}_B$ \\
\midrule''')
    for cr in [1.0, 5.0, 10.0, 20.0]:
        sub = df3[(df3['cost_ratio'] == cr) & (df3['estimator'] != 'groupless')]
        print(f"\\multicolumn{{10}}{{l}}{{\\textbf{{Cost Ratio $c_B/c_A = {cr:.1f}$}} \\quad ($\\bar{{c}}_A = 0.5, \\bar{{c}}_B = {0.5*cr:.2f}$, \\ Pool: $25$ A, $25$ B)}} \\\\")
        print(r'\midrule')
        for idx, r in sub.iterrows():
            est_name = 'Fair' if r['estimator'] == 'fair' else 'Group-Aware'
            acq_str = f"{r['n_acquired']:.1f} / {r['n_acquired_A']:.1f} / {r['n_acquired_B']:.1f}"
            print(f"{est_name:12s} & {r['estimator_mean']:.4f} & {r['mse_disparity']:.4f} & {r['mse_pop']:.4f} & {r['instance_payment']:.4f} & {acq_str} & {r['avg_priv_A']:.4f} & {r['avg_priv_B']:.4f} & {r['avg_pay_A']:.4f} & {r['avg_pay_B']:.4f} \\\\")
        if cr != 20.0:
            print(r'\midrule')
    print(r'''\bottomrule
\end{tabular}%
}
\end{table}''')

    # Table 4: Exp 2.1.2
    df4 = pd.read_csv(RESULTS_DIR / 'exp_2_1_2.csv')
    print('\n=== TABLE 4 (Exp 2.1.2) ===')
    print(r'''\begin{table}[htbp]
\centering
\caption{Experiment 2.1.2: Demographic sweep under heterogeneous costs ($\bar{c}_A = 0.5, \bar{c}_B = 1.5$, total $n = 50$, $\lambda = 10.0$, 10 Monte Carlo draws).}
\label{tab:exp2_1_2}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lrrrrrrrrr}
\toprule
Estimator & $\widehat{m}$ & Disparity & $\MSE_{\mathrm{pop}}$ & Payment & Acquired ($N / A / B$) & $\bar{\epsilon}_A$ & $\bar{\epsilon}_B$ & $\bar{t}_A$ & $\bar{t}_B$ \\
\midrule''')
    for p in [0.1, 0.5, 0.9]:
        regime = "Minority Cheap Group A (Majority Expensive B)" if p == 0.1 else ("Equal Representation" if p == 0.5 else "Majority Cheap Group A (Minority Expensive B)")
        sub = df4[(df4['p_A'] == p) & (df4['estimator'] != 'groupless')]
        r0 = sub.iloc[0]
        print(f"\\multicolumn{{10}}{{l}}{{\\textbf{{{regime}}} \\quad ($p_A = {p:.1f}$, \\ Pool: $n_A = {int(r0['n_A'])}, n_B = {int(r0['n_B'])}$, \\ Target $\\theta = {r0['true_mean']:.2f}$)}} \\\\")
        print(r'\midrule')
        for idx, r in sub.iterrows():
            est_name = 'Fair' if r['estimator'] == 'fair' else 'Group-Aware'
            acq_str = f"{r['n_acquired']:.1f} / {r['n_acquired_A']:.1f} / {r['n_acquired_B']:.1f}"
            print(f"{est_name:12s} & {r['estimator_mean']:.4f} & {r['mse_disparity']:.4f} & {r['mse_pop']:.4f} & {r['instance_payment']:.4f} & {acq_str} & {r['avg_priv_A']:.4f} & {r['avg_priv_B']:.4f} & {r['avg_pay_A']:.4f} & {r['avg_pay_B']:.4f} \\\\")
        if p != 0.9:
            print(r'\midrule')
    print(r'''\bottomrule
\end{tabular}%
}
\end{table}''')

    # Table 5: Exp 2.2
    df5 = pd.read_csv(RESULTS_DIR / 'exp_2_2.csv')
    print('\n=== TABLE 5 (Exp 2.2) ===')
    print(r'''\begin{table}[htbp]
\centering
\caption{Experiment 2.2: Distribution comparison with matched first and second moments ($n_A = 25, n_B = 25$, total $n = 50$, $\lambda = 10.0$, 10 Monte Carlo draws).}
\label{tab:exp2_2}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lrrrrrrrrr}
\toprule
Estimator & $\widehat{m}$ & $\MSE_{\mathrm{pop}}$ & Disparity & Payment & Acquired ($N / A / B$) & $\bar{\epsilon}_A$ & $\bar{\epsilon}_B$ & $\bar{t}_A$ & $\bar{t}_B$ \\
\midrule''')
    for d in ['uniform', 'lognormal', 'truncnorm']:
        d_display = 'Uniform Distribution' if d == 'uniform' else ('Log-Normal Distribution' if d == 'lognormal' else 'Truncated Normal Distribution')
        sub = df5[(df5['distribution'] == d) & (df5['estimator'] != 'groupless')]
        print(f"\\multicolumn{{10}}{{l}}{{\\textbf{{{d_display}}} \\quad ($\\bar{{c}}_A = 0.5, \\bar{{c}}_B = 1.0, \\Var(c) = \\bar{{c}}^2/3$)}} \\\\")
        print(r'\midrule')
        for idx, r in sub.iterrows():
            est_name = 'Fair' if r['estimator'] == 'fair' else 'Group-Aware'
            acq_str = f"{r['n_acquired']:.1f} / {r['n_acquired_A']:.1f} / {r['n_acquired_B']:.1f}"
            print(f"{est_name:12s} & {r['estimator_mean']:.4f} & {r['mse_pop']:.4f} & {r['mse_disparity']:.4f} & {r['instance_payment']:.4f} & {acq_str} & {r['avg_priv_A']:.4f} & {r['avg_priv_B']:.4f} & {r['avg_pay_A']:.4f} & {r['avg_pay_B']:.4f} \\\\")
        if d != 'truncnorm':
            print(r'\midrule')
    print(r'''\bottomrule
\end{tabular}%
}
\end{table}''')

if __name__ == '__main__':
    format_all_tables()
