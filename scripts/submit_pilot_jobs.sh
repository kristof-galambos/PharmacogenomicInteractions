#!/usr/bin/env bash
set -euo pipefail

BOOTSTRAPS="${BOOTSTRAPS:-10}"
JOB_FILE="${JOB_FILE:-job.yml}"

jobs=(
  "Camptothecin|pan-cancer|both"
  "Camptothecin|pan-cancer|mutations"
  "Camptothecin|pan-cancer|expression"
  "Camptothecin|breast|both"
  "Camptothecin|lung_NSCLC|both"
  "Camptothecin|large_intestine|both"
  "Cisplatin|pan-cancer|both"
  "Cisplatin|breast|mutations"
  "Paclitaxel|pan-cancer|both"
  "Paclitaxel|lung_NSCLC|expression"
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
