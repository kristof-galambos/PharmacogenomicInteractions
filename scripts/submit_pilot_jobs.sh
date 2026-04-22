#!/usr/bin/env bash
set -euo pipefail

BOOTSTRAPS="${BOOTSTRAPS:-10}"
JOB_FILE="${JOB_FILE:-job.yml}"

jobs=(
  "Camptothecin|pan-cancer|mutation_expression"
  "Camptothecin|pan-cancer|mutations"
  "Camptothecin|pan-cancer|expression"
  "Camptothecin|pan-cancer|copy_number"
  "Camptothecin|pan-cancer|all"
  "Camptothecin|Breast Carcinoma|mutation_expression"
  "Camptothecin|Non-Small Cell Lung Carcinoma|mutation_expression"
  "Camptothecin|Glioblastoma|mutation_expression"
  "Cisplatin|pan-cancer|mutation_expression"
  "Cisplatin|Breast Carcinoma|mutations"
  "Paclitaxel|pan-cancer|mutation_expression"
  "Paclitaxel|Non-Small Cell Lung Carcinoma|expression"
)

for job_spec in "${jobs[@]}"; do
  IFS="|" read -r drug_name cancer_type feature_config <<< "$job_spec"

  echo "Submitting: drug=${drug_name}, cancer_type=${cancer_type}, feature_config=${feature_config}, bootstraps=${BOOTSTRAPS}"
  az ml job create \
    -f "$JOB_FILE" \
    --set \
      "inputs.drug_name=${drug_name}" \
      "inputs.cancer_type=${cancer_type}" \
      "inputs.feature_config=${feature_config}" \
      "inputs.n_bootstraps=${BOOTSTRAPS}"
done
