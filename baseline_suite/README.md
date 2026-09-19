# Multivariate trajectory baseline suite

This suite trains four independent predictors, one for each configured biomarker,
and applies the same four interval procedures to every predictor:

1. the model's native/raw interval;
2. single-biomarker trajectory conformal prediction;
3. Bonferroni conformal prediction at `alpha / 4`;
4. joint multivariate conformal prediction using the maximum over time and all
   four biomarkers.

Implemented predictors:

- `mlp`: deterministic MLP with a fit-set robust residual scale;
- `drmc`: Monte Carlo dropout regression;
- `bootstrap`: subject-level bootstrap MLP ensemble;
- `dqr`: deep quantile regression with normalized CQR;
- `exact_gp`: exact RBF Gaussian process;
- `dmegp`: deep mean network plus a variational residual GP;
- `lmm`: random-intercept/random-time-slope mixed model;
- `gam`: cubic spline in time plus selected linear covariates.

## Use the 20% DKGP split

Every baseline must use the same fitting, calibration, and test subjects as the
primary 20% DKGP experiment. The expected split file is:

```text
./results/multivariate_cp_dkgp_4_cal020/subject_splits.csv
```

The same calibration subjects are then used for all four biomarkers and every
baseline. Do not let an individual baseline resample its own calibration set.

## Required project files

Run from the project root, where `functions.py` and the dataset are accessible.
The suite uses the project's existing:

```python
from functions import process_temporal_singletask_data
```

Install any missing dependencies with:

```bash
pip install -r baseline_suite/requirements_baselines.txt
```

## One-fold smoke test

Run this first. It uses only five neural-network epochs and ten MC draws so it
checks data loading, split alignment, output shapes, and conformalization rather
than final performance.

```bash
bash baseline_suite/run_smoke_test.sh
```

The defaults match the current project layout. Paths can be overridden:

```bash
DATA_FILE=../ConformalBiomarkerTrajectories/data/data.csv \
SPLITS_FILE=./results/multivariate_cp_dkgp_4_cal020/subject_splits.csv \
GPU_ID=0 \
bash baseline_suite/run_smoke_test.sh
```

## Run every baseline

```bash
bash baseline_suite/run_all_baselines.sh
```

Default paths and settings:

```text
DATA_FILE       ../ConformalBiomarkerTrajectories/data/data.csv
SPLITS_FILE     ./results/multivariate_cp_dkgp_4_cal020/subject_splits.csv
BIOMARKERS_FILE ./baseline_suite/biomarkers_4.example.json
RESULTS_DIR     ./results/multivariate_baselines_cal020
GPU_ID          0
ALPHA           0.10
SEED            42
```

Override any value as an environment variable. For example:

```bash
GPU_ID=1 \
RESULTS_DIR=./results/multivariate_baselines_cal020_gpu1 \
bash baseline_suite/run_all_baselines.sh
```

Run only selected models with:

```bash
RUN_MODELS=mlp,drmc,dqr,gam bash baseline_suite/run_all_baselines.sh
```

The launcher runs models sequentially, writes a separate log for each one, and
continues to later models if one baseline fails.

## Direct Python command

```bash
python baseline_suite/run_multivariate_baselines.py \
  --file ../ConformalBiomarkerTrajectories/data/data.csv \
  --subject-splits ./results/multivariate_cp_dkgp_4_cal020/subject_splits.csv \
  --biomarkers-json ./baseline_suite/biomarkers_4.example.json \
  --models mlp,drmc,bootstrap,dqr,exact_gp,dmegp,lmm,gam \
  --alpha 0.10 \
  --seed 42 \
  --gpuid 0 \
  --uncertainty epistemic \
  --output-dir ./results/multivariate_baselines_cal020 \
  --continue-on-error
```

Running models separately is safer for long jobs. For example:

```bash
python baseline_suite/run_multivariate_baselines.py \
  --file ../ConformalBiomarkerTrajectories/data/data.csv \
  --subject-splits ./results/multivariate_cp_dkgp_4_cal020/subject_splits.csv \
  --biomarkers-json ./baseline_suite/biomarkers_4.example.json \
  --models drmc \
  --mc-samples 100 \
  --mlp-epochs 200 \
  --alpha 0.10 \
  --gpuid 0 \
  --output-dir ./results/multivariate_baselines_cal020
```

## Conformal scores

For uncertainty-based or scale-based predictors:

```text
R_i,k = max_t |Y_i,t,k - mean_i,t,k| / scale_i,t,k
```

For DQR:

```text
R_i,k = max_t max(L_i,t,k - Y_i,t,k,
                  Y_i,t,k - U_i,t,k,
                  0) / s_k
```

The joint score is:

```text
R_i,joint = max_k R_i,k
```

At `alpha=0.10`, Bonferroni calibration uses `alpha/4 = 0.025` for each
biomarker.

## Model details

### MLP

A deterministic `128 -> 64 -> 1` network. Its conformal normalizer is a robust
fit-set residual scale, so calibration and test outcomes are never used to
estimate the scale.

### DRMC

The same network with dropout `0.20` retained at inference. The normalizer is
the empirical standard deviation over 100 MC predictions.

### Bootstrap

Ten MLPs trained on subject-level bootstrap samples. All visits from a sampled
subject are retained together. The normalizer is the ensemble standard
deviation.

### DQR

A three-output quantile network. The default native interval is the 5th to 95th
quantile. Joint CQR scores are normalized using a robust fit-set residual scale
to prevent outcome units from dominating the cross-biomarker maximum.

### Exact GP

An RBF GP on the original input representation. With roughly 7,000-9,000
longitudinal observations per fold, this may be computationally demanding even
with iterative GPyTorch inference. Run one fold first and do not silently
subsample if it fails.

### DMEGP

A standalone deep-mean plus variational-residual-GP implementation. Before a
final manuscript comparison, align it with the exact DMEGP implementation and
hyperparameters used in the earlier study.

### LMM

Random intercept and random time slope. Predictions for unseen test subjects use
the fixed-effect mean because their random effects are unobserved. To avoid a
singular high-dimensional model, fit-set feature screening retains at most 20
non-time predictors.

### GAM

A cubic spline for time plus selected linear covariates, estimated with ridge
regularization. Fit-set feature screening retains at most 30 non-time
predictors.

## Outputs

Each model has its own directory containing:

```text
calibration_predictions_long.csv
test_predictions_long.csv
calibration_scores_by_subject_biomarker.csv
calibration_scores_common_subjects_wide.csv
conformal_quantiles.csv
test_intervals_long.csv
test_trajectory_coverage_by_biomarker.csv
test_joint_coverage_by_subject.csv
fold_metrics.csv
overall_metrics.csv
training_runtime_and_counts.csv
```

Aggregate completed models with:

```bash
python baseline_suite/aggregate_baseline_results.py \
  --results-dir ./results/multivariate_baselines_cal020
```

This creates combined overall, fold-level, runtime, and joint-summary CSV files.

## Recommended order

1. Run `run_smoke_test.sh`.
2. Run `mlp`, `drmc`, `dqr`, and `gam` over all folds.
3. Run `bootstrap` and `dmegp`.
4. Run `lmm` and inspect convergence warnings.
5. Run `exact_gp` last because it is the most computationally demanding.
