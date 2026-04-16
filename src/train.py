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
    parser.add_argument("--output-dir", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    data = read_data(args.ic50_csv, args.mutation_csv, args.expression_csv, drug_name='Camptothecin')

    y_test, y_test_pred, test_r, final_model = run_modelling(data)

    metrics = {
        "drug_name": args.drug_name,
        "n_samples": int(len(y_test)),
        "test_pearson_r": None if pd.isna(test_r) else float(test_r),
    }

    pd.DataFrame({
        "sample_id": y_test.index,
        "observed_ic50": y_test.values,
        "predicted_ic50": y_test_pred,
    }).to_csv(outdir / "predictions.csv", index=False)

    with open(outdir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    dump(final_model, outdir / "model.joblib")

    print(json.dumps(metrics, indent=2))
    print('note that the predictions and the metrics were written to')
    print(str(outdir))


if __name__ == "__main__":
    main()
