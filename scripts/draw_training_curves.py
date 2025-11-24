#!/usr/bin/env python3
import argparse
import os
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_loss_curves(csv_paths: list[list[str]], output_dir: str, subplot_titles: Optional[list[str]] = None):
    """
    Plots training and validation loss curves for multiple CSV files.
    If multiple CSVs are provided, the mean and standard deviation are plotted.

    Parameters:
        csv_paths: List of grouped CSV file paths (each group is a list of paths).
        output_dir: Directory to write loss curves figure to.
        subplot_titles: List of subplot titles.
    """    
    FONTSIZE = 15

    num_subplots = len(csv_paths)

    fig, axes = plt.subplots(1, num_subplots, figsize=(6*num_subplots, 4), sharey=False)
    if num_subplots == 1: # axes is not iterable if num_subplots==1
        axes = [axes]

    for i, (group, ax) in enumerate(zip(csv_paths, axes)):
        dfs = [pd.read_csv(path) for path in group]
        epochs = dfs[0]['epoch']

        train_losses = np.array([df['train_loss'] for df in dfs])
        eval_losses = np.array([df['eval_loss'] for df in dfs])

        train_loss_mean = train_losses.mean(axis=0)
        train_loss_std = train_losses.std(axis=0)
        eval_loss_mean = eval_losses.mean(axis=0)
        eval_loss_std = eval_losses.std(axis=0)

        ax.plot(epochs, train_loss_mean, color="blue", label="Training Loss (Mean)")
        ax.fill_between(epochs, train_loss_mean - train_loss_std, train_loss_mean + train_loss_std, color="blue", alpha=0.2, label="Training Loss (Std Dev)")

        ax.plot(epochs, eval_loss_mean, color="orange", label="Validation Loss (Mean)")
        ax.fill_between(epochs, eval_loss_mean - eval_loss_std, eval_loss_mean + eval_loss_std, color="orange", alpha=0.2, label="Validation Loss (Std Dev)")

        if subplot_titles is not None:
            ax.set_title(subplot_titles[i], fontsize=FONTSIZE)
        ax.set_xlabel("Epoch #", fontsize=FONTSIZE)
        ax.set_ylabel("Loss", fontsize=FONTSIZE)
        _, ymax = ax.get_ylim()
        ax.set_yticks([0.0, round(ymax/2, 1), round(ymax, 1)])

    # Add a single legend for the entire figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=2, fontsize=FONTSIZE)

    fig.suptitle("Training and Validation Loss over Epochs", y=1.05, fontsize=FONTSIZE)
    plt.rcParams["xtick.labelsize"] = FONTSIZE
    plt.rcParams["ytick.labelsize"] = FONTSIZE

    savepath = os.path.join(output_dir, f"loss_curves.png")
    fig.savefig(savepath, bbox_inches='tight')

def draw_training_curves(csv_paths: list[str], output_dir: str, subplot_titles: list[str]):

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # For k-fold handling of 2 experiments
    if (len(csv_paths) % 2 == 0) and (len(csv_paths) > 2):
        k = len(csv_paths) // 2
        print(f"Detected {k} folds to pair up")
        plot_loss_curves([[csv_paths[i] for i in range(k)], [csv_paths[i] for i in range(k,2*k)]], output_dir, subplot_titles)
    else:
        print(f"Treating each input csv individually")
        plot_loss_curves([[csv_path] for csv_path in csv_paths], output_dir, subplot_titles)

    # There's no support to do this for each CSV path yet
    df = pd.read_csv(csv_paths[0])

    # Metrics to plot individually
    metrics = [
        'learning_rate',
        'grad_norm',
        'eval_accuracy_branch1',
        'eval_precision_branch1',
        'eval_recall_branch1',
        'eval_f1_branch1',
        'eval_accuracy_branch2',
        'eval_precision_branch2',
        'eval_recall_branch2',
        'eval_f1_branch2',
        'eval_lambda',
        'eval_tn_branch2',
        'eval_fp_branch2',
        'eval_fn_branch2',
        'eval_tp_branch2',
        'eval_loss_branch1',
        'eval_loss_branch2',
    ]

    for metric in metrics:
        if metric in df.columns:
            plt.figure()
            plt.plot(df['epoch'], df[metric])
            plt.xlabel('Epoch')
            plt.ylabel(metric.replace('_', ' ').title())
            plt.title(f'{metric.replace("_", " ").title()} Over Epochs')
            plt.savefig(os.path.join(output_dir, f'{metric}_curve.png'))
            plt.close()
        else:
            print(f"WARNING: Did not find {metric}")

    print(f"Plots saved to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Train a domain adaptation model, given a config file. Evaluation will be attempted after training as well.")

    parser.add_argument("training_curves_csv_file", type=str, nargs='+', help="Path to CSV file containing data needed to draw training curves. This would be named `training_curves.csv` after using `run_training.py`. Specify more than one to plot a side-by-side comparison of equivalent metrics among the CSV's.")
    parser.add_argument("output_dir", type=str, help="Path to directory to put the training curve plots into. This directory doesn't need to already exist.")
    parser.add_argument("--subplot_titles", type=str, nargs='+', required=False, help="Titles for each subplot.")
    args = parser.parse_args()

    draw_training_curves(args.training_curves_csv_file, args.output_dir, args.subplot_titles)

if __name__ == "__main__":
    main()
