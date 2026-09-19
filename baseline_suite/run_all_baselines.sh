#!/usr/bin/env bash
set -u -o pipefail

# Run from the project root, i.e. the directory containing functions.py.
PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
DATA_FILE="${DATA_FILE:-../ConformalBiomarkerTrajectories/data/data.csv}"
SPLITS_FILE="${SPLITS_FILE:-./results/multivariate_cp_dkgp_4_cal020/subject_splits.csv}"
BIOMARKERS_FILE="${BIOMARKERS_FILE:-./baseline_suite/biomarkers_4.example.json}"
RESULTS_DIR="${RESULTS_DIR:-./results/multivariate_baselines_cal020}"
GPU_ID="${GPU_ID:-0}"
ALPHA="${ALPHA:-0.10}"
SEED="${SEED:-42}"

cd "$PROJECT_ROOT" || exit 1
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/baseline_suite:${PYTHONPATH:-}"

for required in "$DATA_FILE" "$SPLITS_FILE" "$BIOMARKERS_FILE"; do
  if [[ ! -f "$required" ]]; then
    echo "Missing required file: $required" >&2
    exit 2
  fi
done

mkdir -p "$RESULTS_DIR/logs"

BASE_ARGS=(
  --file "$DATA_FILE"
  --subject-splits "$SPLITS_FILE"
  --biomarkers-json "$BIOMARKERS_FILE"
  --alpha "$ALPHA"
  --seed "$SEED"
  --gpuid "$GPU_ID"
  --output-dir "$RESULTS_DIR"
)

# Override with, for example: RUN_MODELS="mlp,drmc,dqr"
RUN_MODELS="${RUN_MODELS:-mlp,drmc,bootstrap,dqr,exact_gp,dmegp,lmm,gam}"
IFS=',' read -r -a MODELS <<< "$RUN_MODELS"

failed_models=()
for model in "${MODELS[@]}"; do
  model="$(echo "$model" | xargs)"
  [[ -z "$model" ]] && continue
  echo
  echo "================================================================================"
  echo "Running baseline: $model"
  echo "================================================================================"

  EXTRA_ARGS=()
  case "$model" in
    mlp)
      EXTRA_ARGS+=(--mlp-epochs 200 --mlp-learning-rate 0.01)
      ;;
    drmc)
      EXTRA_ARGS+=(--mlp-epochs 200 --mlp-learning-rate 0.01 --dropout 0.20 --mc-samples 100)
      ;;
    bootstrap)
      EXTRA_ARGS+=(--mlp-epochs 200 --mlp-learning-rate 0.01 --bootstrap-models 10)
      ;;
    dqr)
      EXTRA_ARGS+=(--mlp-epochs 200 --mlp-learning-rate 0.01 --dropout 0.20 --dqr-lower 0.05 --dqr-upper 0.95)
      ;;
    exact_gp)
      EXTRA_ARGS+=(--exact-gp-iterations 100 --exact-gp-learning-rate 0.10 --max-cholesky-size 1000 --uncertainty epistemic)
      ;;
    dmegp)
      EXTRA_ARGS+=(--dmegp-latent-dim 64 --dmegp-inducing-points 256 --dmegp-epochs 50 --dmegp-learning-rate 0.001 --dmegp-weight-decay 0.001 --uncertainty epistemic)
      ;;
    lmm)
      EXTRA_ARGS+=(--lmm-max-fixed-features 20 --lmm-maxiter 500)
      ;;
    gam)
      EXTRA_ARGS+=(--gam-knots 8 --gam-degree 3 --gam-ridge-alpha 1.0 --gam-max-linear-features 30)
      ;;
    *)
      echo "Unknown model: $model" >&2
      failed_models+=("$model")
      continue
      ;;
  esac

  log_file="$RESULTS_DIR/logs/${model}.log"
  if python baseline_suite/run_multivariate_baselines.py \
      "${BASE_ARGS[@]}" \
      --models "$model" \
      "${EXTRA_ARGS[@]}" 2>&1 | tee "$log_file"; then
    echo "Completed: $model"
  else
    echo "FAILED: $model. See $log_file" >&2
    failed_models+=("$model")
  fi
done

if python baseline_suite/aggregate_baseline_results.py \
    --results-dir "$RESULTS_DIR" 2>&1 | tee "$RESULTS_DIR/logs/aggregate.log"; then
  echo "Aggregated completed models."
else
  echo "Aggregation failed or no model completed." >&2
fi

if (( ${#failed_models[@]} > 0 )); then
  printf 'Models with failures:' >&2
  printf ' %s' "${failed_models[@]}" >&2
  printf '\n' >&2
  exit 1
fi

echo "All requested baselines completed. Results: $RESULTS_DIR"
