#!/usr/bin/env python3
import argparse
import os
import json

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

RESULTS_ROOT = "/repo/results/metrics/final_results"
FIGS_ROOT = "/repo/results/figs"

METRIC_NAME_READABLE2COL = {
    "Test Accuracy (Labels)": {"eval_accuracy_branch1", "eval_accuracy"},
    "Test Macro Precision (Labels)": {"eval_precision_branch1", "eval_precision"},
    "Test Macro Recall (Labels)": {"eval_recall_branch1", "eval_recall"},
    "Test Macro F1 (Labels)": {"eval_f1_branch1", "eval_f1"},
}

def reverse_dict_of_set(d: dict[str, set[str]]) -> dict[str, str]:
    r = {}
    for k, vset in d.items():
        for v in vset:
            r[v] = k
    return r
METRIC_NAME_COL2READABLE = reverse_dict_of_set(METRIC_NAME_READABLE2COL)

MODEL_NAME_READABLE2STD = {
    "Train on Original": "baseline_train_on_original",
    "Train on Uniform Noise": "baseline_train_on_Uniform_Noise",
    r"Train on Rotated 90$\degree$": "baseline_train_on_Rotate_90deg",
    "Train on Ring Artifact": "baseline_train_on_ring_data",
    r"Train on all but Rotated 90$\degree$": "augmentation_train_exclude_Rotate_90deg",
    "Train on all but Ring Artifact": "augmentation_train_exclude_ring_data",
    r"DANN ($\mathcal{S}=\text{Original}, \mathcal{T}=\text{Rotated 90}\degree$)": "dann_target_domain_Rotate_90deg",
    r"DANN ($\mathcal{S}=\text{Original}, \mathcal{T}=\text{Ring Artifact}$)": "dann_target_domain_ring_data",
}

TESTSET_NAME_PUBLISHED2INDEX = {
    "Original": "original",
    "Uniform Noise": "Uniform_Noise",
    r"Rotated 90$\degree$": "Rotate_90deg",
    "Ring Artifact": "ring_data",
}

def mean_std_dfs_to_heatmap(mean_df, std_df, ttl, vmin=0, vmax=1):
    FONTSIZE_L=23
    FONTSIZE_S=18
    FIGSIZE=(14,9)
    
    plusminus_df = mean_df.map(lambda m: f"{m:.3f}") + "\n±" + std_df.map(lambda s: f"{s:.3f}")
    
    # Plot heatmap
    plt.figure(figsize=FIGSIZE)
    ax = sns.heatmap(
        mean_df,
        cmap="Blues",
        annot=plusminus_df,
        fmt="", # Keep format from above
        linewidths=0.5,
        vmin=vmin,
        vmax=vmax,
        annot_kws={"size": FONTSIZE_S}
    )
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=FONTSIZE_S)

    plt.title(ttl, pad=20, fontsize=FONTSIZE_L)
    plt.xlabel("Model", fontsize=FONTSIZE_L)
    plt.ylabel("Test Data Distortion", fontsize=FONTSIZE_L)
    plt.xticks(rotation=30, ha='right', fontsize=FONTSIZE_L)
    plt.yticks(rotation=0, fontsize=FONTSIZE_L)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS_ROOT, ttl + ".png"))
    plt.show()

def create_kfold_test_matrices():
    # Collect all relevant metrics from files
    metric_tables = {}
    for metric_name_readable in METRIC_NAME_READABLE2COL.keys():
        metric_tables[metric_name_readable] = {}
        for stat in ["mean", "std"]:
            metric_tables[metric_name_readable][stat] = {}
            for model_name_readable in MODEL_NAME_READABLE2STD.keys():
                metric_tables[metric_name_readable][stat][model_name_readable] = {}
                model_name_std = MODEL_NAME_READABLE2STD[model_name_readable]
                for testset_name_published in TESTSET_NAME_PUBLISHED2INDEX.keys():
                    testset_name_index = TESTSET_NAME_PUBLISHED2INDEX[testset_name_published]
                    testset_folder = os.path.join(RESULTS_ROOT,model_name_std, testset_name_index)
                    fold_test_result_fps = [os.path.join(testset_folder, x, "test_metrics.csv") for x in os.listdir(testset_folder) if os.path.isdir(os.path.join(testset_folder, x))]
                    
                    assert len(fold_test_result_fps) == 5, "Expecting 5 folds"
                
                    fold_metrics = []
                    for fold_num in range(len(fold_test_result_fps)):
                        fold_test_result_fp = fold_test_result_fps[fold_num]
                        df = pd.read_csv(fold_test_result_fp)
                        df = df.rename(columns=METRIC_NAME_COL2READABLE)
                        fold_metrics.append(df[metric_name_readable].item())
                    
                    if stat == "mean":
                        metric_tables[metric_name_readable][stat][model_name_readable][testset_name_published] = np.mean(fold_metrics)
                    elif stat == "std":
                        metric_tables[metric_name_readable][stat][model_name_readable][testset_name_published] = np.std(fold_metrics)
                    else:
                        assert False

        mean_df = pd.DataFrame(metric_tables[metric_name_readable]["mean"])
        std_df = pd.DataFrame(metric_tables[metric_name_readable]["std"])
        mean_std_dfs_to_heatmap(mean_df, std_df, metric_name_readable)

def main():
    parser = argparse.ArgumentParser(description="Draw heatmaps.")
    args = parser.parse_args()

    create_kfold_test_matrices()

if __name__ == "__main__":
    main()