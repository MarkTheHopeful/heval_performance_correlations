import json
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def load_metrics(path):
    records = []
    with open(path, 'r') as f:
        for line in f:
            records.append(json.loads(line))
    return pd.DataFrame(records)


def plot_pass_at_k(df, output_dir):
    plt.figure(figsize=(10, 5))
    plt.bar(df['task_id'], df['pass@k'])
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('pass@k')
    plt.title('Pass@k by Task')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pass_at_k_by_task.png'))
    plt.close()


def plot_scatter(df, x_col, y_col, output_dir):
    plt.figure(figsize=(6, 6))
    plt.scatter(df[x_col], df[y_col])
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f'{x_col} vs. {y_col}')
    plt.tight_layout()
    fname = f'scatter_{x_col}_vs_{y_col}.png'.replace(' ', '_').replace(':','')
    plt.savefig(os.path.join(output_dir, fname))
    plt.close()


def plot_correlation_heatmap(df, output_dir):
    numeric = df.select_dtypes(include=[np.number])
    corr = numeric.corr()
    plt.figure(figsize=(8, 8))
    plt.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(label='pearson')
    ticks = range(len(corr))
    plt.xticks(ticks, corr.columns, rotation=90)
    plt.yticks(ticks, corr.columns)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'))
    plt.close()


def plot_histograms(df, cols, output_dir):
    for col in cols:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            plt.figure(figsize=(6, 4))
            plt.hist(df[col].dropna(), bins=10)
            plt.title(f'Distribution of {col}')
            plt.xlabel(col)
            plt.ylabel('count')
            plt.tight_layout()
            fname = f'hist_{col}.png'.replace(' ', '_').replace(':','')
            plt.savefig(os.path.join(output_dir, fname))
            plt.close()


def explode_runs(df):
    # identify which columns are per‐run lists
    list_cols = [
        'passes',
        'ComM: Gestalt text similarity',
        'ComM: CFG code similarity',
        'SolM: Total text length',
        'SolM: Lines count',
        'SolM: CFG Triviality',
        'Interest'
    ]

    # start by exploding the pass flags
    exploded = df[['task_id', 'passes']].explode('passes').rename(columns={'passes': 'pass'})

    # for each metric list, explode and join back by task_id + run‐index
    for col in list_cols:
        if col == 'passes':
            continue
        # explode each into its own DF
        tmp = df[['task_id', col]].explode(col)
        col_new_name = col.replace(':', "").replace(" ", "_")
        tmp = tmp.rename(columns={col: col.replace(':', '').replace(' ', '_')})
        exploded = exploded.reset_index(drop=True)
        tmp = tmp.reset_index(drop=True)
        exploded = pd.concat([exploded, tmp[tmp.columns.difference(exploded.columns)]], axis=1)
        exploded[col_new_name] = pd.to_numeric(exploded[col_new_name], errors='coerce')

    return exploded


def plot_histograms_by_pass(df_runs, metric_cols, output_dir):
    for col in metric_cols:
        if col not in df_runs.columns or not pd.api.types.is_numeric_dtype(df_runs[col]):
            continue

        plt.figure(figsize=(6, 4))

        vals_pass = df_runs.loc[df_runs['pass'] == True, col].dropna()
        vals_fail = df_runs.loc[df_runs['pass'] == False, col].dropna()
        all_vals = pd.concat([vals_pass, vals_fail])
        vmin, vmax = all_vals.min(), all_vals.max()
        edges = np.linspace(vmin, vmax, 21)

        plt.hist(vals_pass, bins=edges, alpha=0.6, label='passed', edgecolor='white')
        plt.hist(vals_fail, bins=edges, alpha=0.6, label='failed', edgecolor='white', color='red')

        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.legend()
        plt.tight_layout()

        fname = f'hist_{col}.png'.replace(' ', '_').replace(':', '')
        plt.savefig(os.path.join(output_dir, fname))
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize HumanEval metrics from JSONL file with configurable plots")
    parser.add_argument('--metrics',      required=True, help='Path to metrics JSONL file')
    parser.add_argument('--output-dir',   default='visualizations', help='Directory to save output plots')
    parser.add_argument('--no-bar',       action='store_true', help='Skip bar plot of pass@k')
    parser.add_argument('--scatter',      nargs=2, metavar=('X_COL', 'Y_COL'), help='Plot scatter of two columns')
    parser.add_argument('--heatmap',      action='store_true', help='Include correlation heatmap')
    parser.add_argument('--hist',         nargs='+', metavar='COL', help='List of columns for histogram plots')
    parser.add_argument('--pass-by-sol', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    df = load_metrics(args.metrics)

    # bar chart
    if not args.no_bar:
        if 'pass@k' in df.columns:
            plot_pass_at_k(df, args.output_dir)
        else:
            print("Warning: 'pass@k' column not found, skipping bar plot.")

    if args.pass_by_sol:
        e_df = explode_runs(df)
        plot_histograms_by_pass(e_df, ['ComM_Gestalt_text_similarity',
        'ComM_CFG_code_similarity',
        'SolM_Total_text_length',
        'SolM_Lines_count',
        'SolM_CFG_Triviality',
        'Interest'
    ], args.output_dir)

    # scatter plot
    if args.scatter:
        x_col, y_col = args.scatter
        if x_col in df.columns and y_col in df.columns:
            plot_scatter(df, x_col, y_col, args.output_dir)
        else:
            print(f"Warning: columns {x_col}, {y_col} not both present; skipping scatter.")

    # heatmap
    if args.heatmap:
        plot_correlation_heatmap(df, args.output_dir)

    # histograms
    if args.hist:
        plot_histograms(df, args.hist, args.output_dir)
    else:
        default_hist = ['pass@k', 'TasM: Total text length', 'Mean SolM: Total text length']
        plot_histograms(df, default_hist, args.output_dir)

    print(f"Visualizations saved to {args.output_dir}")

if __name__ == "__main__":
    main()