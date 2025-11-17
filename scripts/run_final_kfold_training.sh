#!/bin/bash

# Go to this script's location.
SCRIPT_DIR="$(dirname "$0")"
cd "${SCRIPT_DIR}"

# Assume the repository root is one level up from here
REPO_PATH="$(realpath ..)"

# Base directory for the configuration files
BASE_DIR="${REPO_PATH}/experiment_configs"

# Do the D21 runs
${REPO_PATH}/scripts/run_training.py experiment_configs/dann_d21_target_domain_rings/dann_fold0_train_d21.yaml;
${REPO_PATH}/scripts/run_training.py experiment_configs/dann_d21_target_domain_rings/dann_fold1_train_d21.yaml;
${REPO_PATH}/scripts/run_training.py experiment_configs/dann_d21_target_domain_rings/dann_fold2_train_d21.yaml;
${REPO_PATH}/scripts/run_training.py experiment_configs/dann_d21_target_domain_rings/dann_fold3_train_d21.yaml;
${REPO_PATH}/scripts/run_training.py experiment_configs/dann_d21_target_domain_rings/dann_fold4_train_d21.yaml;

# Do the D20 runs
${REPO_PATH}/scripts/run_training.py experiment_configs/dann_d20_target_domain_rotated/dann_fold0_train_d20.yaml;
${REPO_PATH}/scripts/run_training.py experiment_configs/dann_d20_target_domain_rotated/dann_fold1_train_d20.yaml;
${REPO_PATH}/scripts/run_training.py experiment_configs/dann_d20_target_domain_rotated/dann_fold2_train_d20.yaml;
${REPO_PATH}/scripts/run_training.py experiment_configs/dann_d20_target_domain_rotated/dann_fold3_train_d20.yaml;
${REPO_PATH}/scripts/run_training.py experiment_configs/dann_d20_target_domain_rotated/dann_fold4_train_d20.yaml;
