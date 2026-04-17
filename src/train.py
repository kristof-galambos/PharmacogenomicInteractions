import argparse
import json
from pathlib import Path

import pandas as pd
from joblib import dump

from data_reader import read_data
from main import run_modelling


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mutation-csv", type=str, required=True)
    parser.add_argument("--expression-csv", type=str, required=True)
    parser.add_argument("--ic50-csv", type=str, required=True)
    parser.add_argument("--drug-name", type=str, required=True)
    parser.add_argument("--cancer-type", type=str, default="pan-cancer")
    parser.add_argument(
        "--feature-config",
        type=str,
        choices=["mutations", "expression", "both"],
        default="both",
    )
    parser.add_argument("--n-bootstraps", type=int, default=1)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.n_bootstraps <= 0:
        raise ValueError("--n-bootstraps must be greater than 0.")

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    cancer_type = None if args.cancer_type == "pan-cancer" else args.cancer_type
    data = read_data(
        args.ic50_csv,
        args.mutation_csv,
        args.expression_csv,
        drug_name=args.drug_name,
        cancer_type=cancer_type,
    )
    if data.empty:
        raise ValueError(
            f"No rows found for drug_name={args.drug_name!r}, "
            f"cancer_type={args.cancer_type!r}."
        )

    metrics_rows = []
    prediction_frames = []
    models_dir = outdir / "models"
    models_dir.mkdir(exist_ok=True)

    for bootstrap_id in range(args.n_bootstraps):
        bootstrap_seed = args.random_seed + bootstrap_id
        y_test, y_test_pred, test_r, final_model, model_info = run_modelling(
            data,
            feature_config=args.feature_config,
            random_state=bootstrap_seed,
        )

        metrics_row = {
            "drug_name": args.drug_name,
            "cancer_type": args.cancer_type,
            "feature_config": args.feature_config,
            "bootstrap_id": bootstrap_id,
            "random_seed": bootstrap_seed,
            "n_samples": int(len(data)),
            "n_test_samples": int(len(y_test)),
            "test_pearson_r": None if pd.isna(test_r) else float(test_r),
            **model_info,
        }
        metrics_rows.append(metrics_row)

        prediction_frames.append(pd.DataFrame({
            "drug_name": args.drug_name,
            "cancer_type": args.cancer_type,
            "feature_config": args.feature_config,
            "bootstrap_id": bootstrap_id,
            "sample_id": y_test.index,
            "observed_ic50": y_test.values,
            "predicted_ic50": y_test_pred,
        }))

        dump(final_model, models_dir / f"model_bootstrap_{bootstrap_id}.joblib")

    metrics = {
        "drug_name": args.drug_name,
        "cancer_type": args.cancer_type,
        "feature_config": args.feature_config,
        "n_bootstraps": args.n_bootstraps,
        "n_samples": int(len(data)),
        "mean_test_pearson_r": float(pd.DataFrame(metrics_rows)["test_pearson_r"].mean()),
    }

    pd.concat(prediction_frames, ignore_index=True).to_csv(
        outdir / "predictions.csv",
        index=False,
    )
    pd.DataFrame(metrics_rows).to_csv(outdir / "metrics_by_bootstrap.csv", index=False)

    with open(outdir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print('note that the predictions and the metrics were written to')
    print(str(outdir))


if __name__ == "__main__":
    main()
