#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
import sys
import traceback

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DOWNLOADS_PATH = Path('/Users/kristof/Downloads')
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from data_reader import read_data
from main import run_modelling

PILOT_JOBS = [
    ("Camptothecin", "pan-cancer", "both"),
    ("Camptothecin", "pan-cancer", "mutations"),
    ("Camptothecin", "pan-cancer", "expression"),
    ("Camptothecin", "pan-cancer", "copy_number"),
    ("Camptothecin", "pan-cancer", "all"),
    ("Camptothecin", "Breast Carcinoma", "both"),
    ("Camptothecin", "Non-Small Cell Lung Carcinoma", "both"),
    ("Camptothecin", "Glioblastoma", "both"),
    ("Cisplatin", "pan-cancer", "both"),
    ("Cisplatin", "Breast Carcinoma", "mutations"),
    ("Paclitaxel", "pan-cancer", "both"),
    ("Paclitaxel", "Non-Small Cell Lung Carcinoma", "expression"),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the same pilot matrix as scripts/submit_pilot_jobs.sh locally."
    )
    parser.add_argument(
        "--mutation-csv",
        default=str(DOWNLOADS_PATH / "mutations_summary_20260316.csv"),
        # default=str(REPO_ROOT / "data" / "mutations_summary_20260316.csv"),
    )
    parser.add_argument(
        "--expression-csv",
        default=str(DOWNLOADS_PATH / "rnaseq_merged_rsem_tpm_20260323.csv"),
        # default=str(REPO_ROOT / "data" / "rnaseq_merged_rsem_tpm_20260323.csv"),
    )
    parser.add_argument(
        "--copy-number-csv",
        default=str(DOWNLOADS_PATH / "WES_pureCN_CNV_genes_total_copy_number_20250207.csv"),
    )
    parser.add_argument(
        "--ic50-csv",
        default=str(DOWNLOADS_PATH / "GDSC2_fitted_dose_response_27Oct23.xlsx"),
        # default=str(REPO_ROOT / "data" / "GDSC2_fitted_dose_response_27Oct23.xlsx"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "local_outputs"),
    )
    parser.add_argument("--n-bootstraps", type=int, default=int(os.environ.get("BOOTSTRAPS", 5)))
    parser.add_argument("--random-seed", type=int, default=42)
    return parser.parse_args()


def safe_name(value):
    return "".join(char if char.isalnum() else "_" for char in value).strip("_").lower()


def numeric_or_none(value):
    if value is None or pd.isna(value):
        return None
    return float(value)


def get_job_outdir(args, drug_name, cancer_type, feature_config):
    return (
        Path(args.output_dir)
        / safe_name(drug_name)
        / safe_name(cancer_type)
        / safe_name(feature_config)
    )


def write_job_error(job_outdir, drug_name, cancer_type, feature_config, error):
    job_outdir.mkdir(parents=True, exist_ok=True)
    error_info = {
        "drug_name": drug_name,
        "cancer_type": cancer_type,
        "feature_config": feature_config,
        "status": "failed",
        "error_type": type(error).__name__,
        "error_message": str(error),
        "traceback": traceback.format_exc(),
    }
    with open(job_outdir / "error.json", "w") as f:
        json.dump(error_info, f, indent=2)
    return {
        "drug_name": drug_name,
        "cancer_type": cancer_type,
        "feature_config": feature_config,
        "status": "failed",
        "error_file": str(job_outdir / "error.json"),
    }


def run_pilot_job(args, drug_name, cancer_type, feature_config):
    job_outdir = get_job_outdir(args, drug_name, cancer_type, feature_config)
    job_outdir.mkdir(parents=True, exist_ok=True)

    data_cancer_type = None if cancer_type == "pan-cancer" else cancer_type
    data = read_data(
        args.ic50_csv,
        args.mutation_csv,
        args.expression_csv,
        args.copy_number_csv,
        drug_name=drug_name,
        cancer_type=data_cancer_type,
    )
    if data.empty:
        raise ValueError(
            f"No rows found for drug_name={drug_name!r}, cancer_type={cancer_type!r}."
        )

    metrics_rows = []
    prediction_frames = []
    for bootstrap_id in range(args.n_bootstraps):
        bootstrap_seed = args.random_seed + bootstrap_id
        print(
            "Running: "
            f"drug={drug_name}, cancer_type={cancer_type}, "
            f"feature_config={feature_config}, bootstrap={bootstrap_id}, "
            f"seed={bootstrap_seed}"
        )
        y_test, y_test_pred, test_r, _final_model, model_info = run_modelling(
            data,
            feature_config=feature_config,
            random_state=bootstrap_seed,
        )

        metrics_rows.append({
            "drug_name": drug_name,
            "cancer_type": cancer_type,
            "feature_config": feature_config,
            "bootstrap_id": bootstrap_id,
            "random_seed": bootstrap_seed,
            "n_samples": int(len(data)),
            "n_test_samples": int(len(y_test)),
            "test_pearson_r": numeric_or_none(test_r),
            **model_info,
        })
        prediction_frames.append(pd.DataFrame({
            "drug_name": drug_name,
            "cancer_type": cancer_type,
            "feature_config": feature_config,
            "bootstrap_id": bootstrap_id,
            "sample_id": y_test.index,
            "observed_ic50": y_test.values,
            "predicted_ic50": y_test_pred,
        }))

    metrics_by_bootstrap = pd.DataFrame(metrics_rows)
    metrics = {
        "drug_name": drug_name,
        "cancer_type": cancer_type,
        "feature_config": feature_config,
        "status": "succeeded",
        "n_bootstraps": args.n_bootstraps,
        "n_samples": int(len(data)),
        "mean_test_pearson_r": numeric_or_none(metrics_by_bootstrap["test_pearson_r"].mean()),
    }

    metrics_by_bootstrap.to_csv(job_outdir / "metrics_by_bootstrap.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(
        job_outdir / "predictions.csv",
        index=False,
    )
    with open(job_outdir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Wrote local pilot outputs to {job_outdir}")
    return metrics


def main():
    args = parse_args()
    if args.n_bootstraps <= 0:
        raise ValueError("--n-bootstraps must be greater than 0.")

    all_metrics = []
    for drug_name, cancer_type, feature_config in PILOT_JOBS:
        try:
            all_metrics.append(run_pilot_job(args, drug_name, cancer_type, feature_config))
        except Exception as error:
            job_outdir = get_job_outdir(args, drug_name, cancer_type, feature_config)
            error_info = write_job_error(
                job_outdir,
                drug_name,
                cancer_type,
                feature_config,
                error,
            )
            print(
                "Failed: "
                f"drug={drug_name}, cancer_type={cancer_type}, "
                f"feature_config={feature_config}. "
                f"Wrote error details to {job_outdir / 'error.json'}"
            )
            all_metrics.append(error_info)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_metrics).to_csv(output_dir / "summary.csv", index=False)
    with open(output_dir / "summary.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Wrote local pilot summary to {output_dir}")


if __name__ == "__main__":
    main()
