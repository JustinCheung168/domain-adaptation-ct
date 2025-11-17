#!/bin/bash

# Go to this script's location.
SCRIPT_DIR="$(dirname "$0")"
cd "${SCRIPT_DIR}"

# Assume the repository root is one level up from here
REPO_PATH="$(realpath ..)"

# Base directory for the configuration files
BASE_DIR="${REPO_PATH}/experiment_configs"

${REPO_PATH}/scripts/run_evaluation.py experiment_configs/dann_d21_target_domain_rings/inference/dann_allfolds_test_d21_original.yaml;
${REPO_PATH}/scripts/run_evaluation.py experiment_configs/dann_d21_target_domain_rings/inference/dann_allfolds_test_d21_Uniform_Noise.yaml;
${REPO_PATH}/scripts/run_evaluation.py experiment_configs/dann_d21_target_domain_rings/inference/dann_allfolds_test_d21_Rotate_90deg.yaml;
${REPO_PATH}/scripts/run_evaluation.py experiment_configs/dann_d21_target_domain_rings/inference/dann_allfolds_test_d21_ring_data.yaml;
${REPO_PATH}/scripts/run_evaluation.py experiment_configs/dann_d20_target_domain_rotated/inference/dann_allfolds_test_d20_original.yaml;
${REPO_PATH}/scripts/run_evaluation.py experiment_configs/dann_d20_target_domain_rotated/inference/dann_allfolds_test_d20_Uniform_Noise.yaml;
${REPO_PATH}/scripts/run_evaluation.py experiment_configs/dann_d20_target_domain_rotated/inference/dann_allfolds_test_d20_Rotate_90deg.yaml;
${REPO_PATH}/scripts/run_evaluation.py experiment_configs/dann_d20_target_domain_rotated/inference/dann_allfolds_test_d20_ring_data.yaml;
