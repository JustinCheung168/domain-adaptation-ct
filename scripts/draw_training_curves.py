#!/usr/bin/env python3
import argparse
import os
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt

def plot_loss_curves(csv_paths: list[str], output_dir: str, subplot_titles: Optional[list[str]] = None):
    """
    Plots training and validation loss curves for multiple CSV files.
    
    Parameters:
        csv_paths: List of CSV file paths.
        output_dir: Directory to write loss curves figure to.
        subplot_titles: List of subplot titles.
    """    
    FONTSIZE = 15

    num_subplots = len(csv_paths)

    fig, axes = plt.subplots(1, num_subplots, figsize=(6*num_subplots, 4), sharey=False)
    if num_subplots == 1: # axes is not iterable if num_subplots==1
        axes = [axes]

    for i, (csv_path, ax) in enumerate(zip(csv_paths, axes)):
        df = pd.read_csv(csv_path)
        print(df)
        epochs = df['epoch']
        train_loss = df['train_loss']
        eval_loss = df['eval_loss']
        ax.plot(epochs, train_loss, color="blue", label="Training Loss")
        ax.plot(epochs, eval_loss, color="orange", label="Validation Loss")
        if subplot_titles is not None:
            ax.set_title(subplot_titles[i], fontsize=FONTSIZE)
        ax.set_xlabel("Epoch #", fontsize=FONTSIZE)
        ax.set_ylabel("Loss", fontsize=FONTSIZE)
        _, ymax = ax.get_ylim()
        ax.set_yticks([0.0, round(ymax/2, 1), round(ymax, 1)])
        ax.legend()

    fig.suptitle("Training and Validation Loss over Epochs", y=1.02, fontsize=FONTSIZE)
    plt.rcParams["xtick.labelsize"] = FONTSIZE
    plt.rcParams["ytick.labelsize"] = FONTSIZE

    savepath = os.path.join(output_dir, f"loss_curves.png")
    fig.savefig(savepath, bbox_inches='tight')

def draw_training_curves(csv_paths: list[str], output_dir: str, subplot_titles: list[str]):

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    plot_loss_curves(csv_paths, output_dir, subplot_titles)

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
