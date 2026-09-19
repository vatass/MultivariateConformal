#!/usr/bin/env bash
set -euo pipefail

# Run this from the project root containing functions.py and models.py.
python cp_dkgp_risk_pipeline/run_pipeline.py \
  --file ../ConformalBiomarkerTrajectories/data/data.csv \
  --labels-file ./mci_conversion_labels.csv \
  --label-id-column anon_id \
  --label-column label \
  --biomarkers-json ./cp_dkgp_risk_pipeline/biomarkers_4.json \
  --output-dir ./results/cp_dkgp_high_risk_pipeline \
  --outer-test-fraction 0.20 \
  --crossfit-folds 5 \
  --calibration-fraction 0.20 \
  --alpha 0.10 \
  --uncertainty epistemic \
  --iterations 100 \
  --feature-time-mode observed \
  --time-scale-factor 12 \
  --gpuid 0 \
  --seed 42
