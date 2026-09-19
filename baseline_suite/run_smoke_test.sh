#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
DATA_FILE="${DATA_FILE:-../ConformalBiomarkerTrajectories/data/data.csv}"
SPLITS_FILE="${SPLITS_FILE:-./results/multivariate_cp_dkgp_4_cal020/subject_splits.csv}"
BIOMARKERS_FILE="${BIOMARKERS_FILE:-./baseline_suite/biomarkers_4.example.json}"
RESULTS_DIR="${RESULTS_DIR:-./results/multivariate_baselines_smoke_test}"
GPU_ID="${GPU_ID:-0}"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/baseline_suite:${PYTHONPATH:-}"

python baseline_suite/run_multivariate_baselines.py \
  --file "$DATA_FILE" \
  --subject-splits "$SPLITS_FILE" \
  --biomarkers-json "$BIOMARKERS_FILE" \
  --models mlp,drmc,dqr,gam \
  --fold-start 0 \
  --n-folds 1 \
  --alpha 0.10 \
  --seed 42 \
  --gpuid "$GPU_ID" \
  --mlp-epochs 5 \
  --mc-samples 10 \
  --output-dir "$RESULTS_DIR" \
  --continue-on-error
