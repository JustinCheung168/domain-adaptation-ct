#!/bin/bash

# Go to this script's location.
SCRIPT_DIR="$(dirname "$0")"
cd "${SCRIPT_DIR}"

# Assume the repository root is one level up from here
REPO_PATH="$(realpath ..)"

# Base directory for the configuration files
BASE_DIR="${REPO_PATH}/experiment_configs"

# Do the hyperparameter tuning runs
${REPO_PATH}/scripts/run_training.py ${REPO_PATH}/experiment_configs/dann_d21_target_domain_rings/explore_lambda_scheduling/dann_fold0_train_d21_parabolic_increasing.yaml;
${REPO_PATH}/scripts/run_training.py ${REPO_PATH}/experiment_configs/dann_d21_target_domain_rings/explore_lambda_scheduling/dann_fold0_train_d21_linear_increasing.yaml;
${REPO_PATH}/scripts/run_training.py ${REPO_PATH}/experiment_configs/dann_d21_target_domain_rings/explore_lambda_scheduling/dann_fold0_train_d21_logistic_increasing.yaml;
${REPO_PATH}/scripts/run_training.py ${REPO_PATH}/experiment_configs/dann_d21_target_domain_rings/explore_lambda_scheduling/dann_fold0_train_d21_parabolic_decreasing.yaml;
${REPO_PATH}/scripts/run_training.py ${REPO_PATH}/experiment_configs/dann_d21_target_domain_rings/explore_lambda_scheduling/dann_fold0_train_d21_linear_decreasing.yaml;
${REPO_PATH}/scripts/run_training.py ${REPO_PATH}/experiment_configs/dann_d21_target_domain_rings/explore_lambda_scheduling/dann_fold0_train_d21_constant.yaml;

# Manually inspect the runs, and check which one did the best. Then, modify the 5-fold training files to use that lambda scheduling strategy for the final evaluation.
