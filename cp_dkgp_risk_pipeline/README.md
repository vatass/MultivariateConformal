# CP-DKGP high-risk classification pipeline

This package implements the complete subject-level design discussed for the
high-risk versus low-risk experiment using **only CP-DKGP** as the longitudinal
prediction method.

## Design

The labeled classification cohort is split once into:

- 80% development subjects
- 20% untouched final-test subjects

The final-test subjects are excluded from every DKGP fitting set, every
conformal calibration set, all logistic-regression tuning, and logistic-
regression fitting.

Within the development cohort, five-fold cross-fitting is used. For each fold:

1. The fold is held out for feature generation.
2. A common 20% conformal-calibration subset is selected from the remaining
   eligible development subjects.
3. The four DKGPs are fitted without the held-out, calibration, or final-test
   subjects. Unlabeled or non-classification subjects may still contribute to
   DKGP fitting.
4. A subject-level joint score is computed by maximizing the standardized
   residual over time and the four biomarkers.
5. Joint conformal intervals are generated for the held-out development fold.
6. Point RoC and RoCB features are calculated.

After all five folds, every development subject has one out-of-fold feature
vector. A final CP-DKGP model is then fitted using the development population,
a final development-only calibration subset, and the untouched final-test
subjects are predicted once.

Three logistic regressions are compared:

- `point_roc`: four point-slope features
- `rocb`: eight lower/upper RoC-bound features
- `point_roc_plus_rocb`: all twelve features

Interval width is not included as a separate feature.

## Label file

Provide one row per subject eligible for the downstream classification task,
for example baseline-MCI subjects only:

```csv
anon_id,label
SUBJECT_001,0
SUBJECT_002,1
```

Use `0` for low risk/stable MCI and `1` for high risk/MCI-to-AD conversion. The
trajectory dataset may contain additional subjects. Those additional subjects
can be used for DKGP fitting but are never used in the logistic regression.

## Main command

Run from the project root containing `functions.py` and `models.py`:

```bash
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
```

Set `--time-scale-factor 12` only when the input time unit is months and annual
RoC values are desired.

## Time-grid choice

### Observed random times

```bash
--feature-time-mode observed
```

This uses the same randomly timed predictions for which the joint trajectory
coverage is evaluated. Subjects need at least two unique observed times. The
resulting slope can depend on visit count and follow-up length, so the visit
schedule should not contain post-outcome information.

### Fixed prediction grid

```bash
--feature-time-mode fixed_grid \
--feature-grid "0,12,24,36"
```

This repeats the subject's earliest input vector and replaces the last input
column with the requested times. Use this only when all non-time inputs are
baseline/static covariates. The script checks within-subject invariance.

A fixed grid prevents the classifier from exploiting differences in visit
schedule. However, the current conformal score is calibrated over observed
random times, so coverage on arbitrary unobserved grid points requires an
additional methodological justification. The script stores both observed-time
intervals and fixed-grid feature intervals so this distinction remains clear.

## RoC bound calculation

For times `t_j`, the OLS slope is a linear functional with weights

```text
w_j = (t_j - mean(t)) / sum_l (t_l - mean(t))^2
```

The point RoC uses the predicted means. The exact minimum and maximum slopes over
the rectangular conformal trajectory band are obtained by selecting the lower
or upper endpoint according to the sign of each weight. This is more accurate
than fitting one line through all lower endpoints and another through all upper
endpoints.

## Important outputs

### Splits

```text
splits/outer_subject_split.csv
splits/development_crossfit_subject_roles.csv
splits/final_model_subject_roles.csv
```

These files document every subject's role and should be archived with the
analysis.

### Development out-of-fold data

```text
development_oof_observed_intervals_long.csv
development_oof_roc_rocb_features.csv
development_crossfit_qhats.csv
```

### Untouched final-test data

```text
final_test_observed_intervals_long.csv
final_test_roc_rocb_features.csv
final_model/feature_intervals_long.csv
```

The interval files contain one row per subject, biomarker, and time, including:

```text
id, biomarker, time, y, mean, std, qhat, lower, upper
```

These are the files to preserve for later meta-analysis.

### Logistic-regression results

```text
logistic_regression/final_test_metrics.csv
logistic_regression/final_test_predictions.csv
logistic_regression/final_test_bootstrap_intervals.csv
logistic_regression/logistic_coefficients.csv
logistic_regression/logistic_cv_results.csv
```

## Resume after interruption

The pipeline saves each cross-fit fold separately. Add:

```bash
--resume
```

to reuse folds and the final CP-DKGP stage whose required outputs already
exist.

## Interpretation of validity

The joint conformal bands target simultaneous coverage of all four observed
biomarker trajectories under the exchangeability assumptions of split
conformal prediction. The RoCB inherits coverage as a linear functional when it
is calculated from the same covered trajectory points. The downstream logistic
regression does not receive a conformal classification guarantee; it is an
ordinary predictive model evaluated on the untouched final-test cohort.
