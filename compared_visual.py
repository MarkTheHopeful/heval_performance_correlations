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


def plot_grouped_bar(merged, label1, label2, field, output_dir):
    tasks = merged['task_id']
    x = np.arange(len(tasks))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(tasks)*0.3), 5))
    ax.bar(x - width/2, merged[f'{field}_{label1}'], width, label=label1)
    ax.bar(x + width/2, merged[f'{field}_{label2}'], width, label=label2)
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=45, ha='right')
    ax.set_ylabel(field)
    ax.set_title(f'Comparative {field} by Task')
    ax.legend()
    plt.tight_layout()
    fname = f'comparative_{field}.png'
    plt.savefig(os.path.join(output_dir, fname))
    plt.close()


def plot_scatter(merged, label1, label2, field, output_dir):
    x = merged[f'{field}_{label1}']
    y = merged[f'{field}_{label2}']

    plt.figure(figsize=(6, 6))
    plt.scatter(x, y)
    plt.xlabel(f'{field}_{label1}')
    plt.ylabel(f'{field}_{label2}')
    plt.title(f'{field}: {label1} vs {label2}')
    plt.tight_layout()
    fname = f'scatter_{field}_{label1}_vs_{label2}.png'
    plt.savefig(os.path.join(output_dir, fname))
    plt.close()


def plot_diff_histogram(merged, label1, label2, field, output_dir):
    diff = merged[f'{field}_{label1}'] - merged[f'{field}_{label2}']
    plt.figure(figsize=(6, 4))
    plt.hist(diff.dropna(), bins=10)
    plt.title(f'Difference in {field} ({label1} - {label2})')
    plt.xlabel(f'{field} difference')
    plt.ylabel('Count')
    plt.tight_layout()
    fname = f'hist_diff_{field}_{label1}_minus_{label2}.png'
    plt.savefig(os.path.join(output_dir, fname))
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Compare two HumanEval metrics files and visualize specified metrics"
    )
    parser.add_argument('--metrics1', required=True, help='First metrics JSONL file')
    parser.add_argument('--metrics2', required=True, help='Second metrics JSONL file')
    parser.add_argument('--label1',   required=True, help='Label for first run (used in legends)')
    parser.add_argument('--label2',   required=True, help='Label for second run')
    parser.add_argument('--fields',   required=True, nargs='+',
                        help='List of metric fields (exact column names) to compare')
    parser.add_argument('--output-dir', default='compared_visuals', help='Directory to save plots')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df1 = load_metrics(args.metrics1)
    df2 = load_metrics(args.metrics2)

    for field in args.fields:
        if field not in df1.columns or field not in df2.columns:
            print(f"Warning: Field '{field}' not found in both metrics files, skipping.")
            continue

        m1 = df1[['task_id', field]].copy()
        m1.rename(columns={field: f'{field}_{args.label1}'}, inplace=True)
        m2 = df2[['task_id', field]].copy()
        m2.rename(columns={field: f'{field}_{args.label2}'}, inplace=True)

        merged = pd.merge(m1, m2, on='task_id', how='inner')

        plot_grouped_bar(merged, args.label1, args.label2, field, args.output_dir)
        plot_scatter(merged, args.label1, args.label2, field, args.output_dir)
        plot_diff_histogram(merged, args.label1, args.label2, field, args.output_dir)
        print(f"Generated visuals for field '{field}'")

    print(f"Comparative visualizations saved to {args.output_dir}")

if __name__ == "__main__":
    main()