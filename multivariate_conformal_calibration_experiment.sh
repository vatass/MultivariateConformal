for spec in "0.05 cal005" "0.10 cal010" "0.20 cal020"
do
  set -- $spec
  fraction=$1
  tag=$2

  python multivariate_cp_dkgp.py \
    --file ../ConformalBiomarkerTrajectories/data/data.csv \
    --folds-dir ../ConformalBiomarkerTrajectories/data/folds \
    --biomarkers-json ./biomarkers_4_example.json \
    --calibration-fraction "$fraction" \
    --alpha 0.10 \
    --uncertainty epistemic \
    --iterations 100 \
    --seed 42 \
    --gpuid 0 \
    --output-dir "./results/multivariate_cp_dkgp_4_${tag}"
done