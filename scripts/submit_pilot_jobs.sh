#!/usr/bin/env bash
set -euo pipefail

BOOTSTRAPS="${BOOTSTRAPS:-10}"
JOB_FILE="${JOB_FILE:-job.yml}"

jobs=(
  "Camptothecin|pan-cancer|both"
  "Camptothecin|pan-cancer|mutations"
  "Camptothecin|pan-cancer|expression"
  "Camptothecin|Breast Carcinoma|both"
  "Camptothecin|Non-Small Cell Lung Carcinoma|both"
  "Camptothecin|Glioblastoma|both"
  "Cisplatin|pan-cancer|both"
  "Cisplatin|Breast Carcinoma|mutations"
  "Paclitaxel|pan-cancer|both"
  "Paclitaxel|Non-Small Cell Lung Carcinoma|expression"
)

for job_spec in "${jobs[@]}"; do
  IFS="|" read -r drug_name cancer_type feature_config <<< "$job_spec"

  echo "Submitting: drug=${drug_name}, cancer_type=${cancer_type}, feature_config=${feature_config}, bootstraps=${BOOTSTRAPS}"
  az ml job create \
    -f "$JOB_FILE" \
    --set \
      inputs.drug_name="$drug_name" \
      inputs.cancer_type="$cancer_type" \
      inputs.feature_config="$feature_config" \
      inputs.n_bootstraps="$BOOTSTRAPS"
done
