import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from joblib import dump


from data_reader import read_data
from data_split import split_data
from model import fit_en, predict_en
from utils import get_mutation_columns, pearson_correlation, get_gene_expression_columns
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler


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

    #################### this is from the local code ###################

    data = read_data(args.ic50_csv, args.mutation_csv, args.expression_csv, drug_name='Camptothecin')
    print('hi')
    print(data.shape)

    y = data['LN_IC50']
    mutation_columns = get_mutation_columns(data.columns)
    gene_expression_columns = get_gene_expression_columns(data.columns)
    X = data[mutation_columns + gene_expression_columns]
    print(X.shape)
    print(y.shape)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    print(X_train.shape, X_val.shape, X_test.shape)
    print(y_train.shape)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    alpha_grid = [0.01, 0.05, 0.1, 0.5, 1.0]
    l1_ratio_grid = [round(x, 1) for x in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]]

    best_model = None
    best_alpha = None
    best_l1_ratio = None
    best_val_r = float('-inf')

    for alpha in alpha_grid:
        for l1_ratio in l1_ratio_grid:
            en_model = fit_en(X_train_scaled, y_train, alpha=alpha, l1_ratio=l1_ratio)
            y_val_pred = predict_en(en_model, X_val_scaled)
            val_r = pearson_correlation(y_val, y_val_pred)

            if val_r > best_val_r:
                best_model = en_model
                best_alpha = alpha
                best_l1_ratio = l1_ratio
                best_val_r = val_r

    print('best_alpha =', best_alpha)
    print('best_l1_ratio =', best_l1_ratio)
    print('validation_pearson =', best_val_r)

    y_val_pred = predict_en(best_model, X_val_scaled)
    r2 = r2_score(y_val, y_val_pred)
    print('validation_r2 =', r2)

    X_train_val = pd.concat([X_train, X_val], ignore_index=True)
    y_train_val = pd.concat([y_train, y_val], ignore_index=True)

    final_scaler = StandardScaler()
    X_train_val_scaled = final_scaler.fit_transform(X_train_val)
    X_test_scaled = final_scaler.transform(X_test)

    final_model = fit_en(X_train_val_scaled, y_train_val, alpha=best_alpha, l1_ratio=best_l1_ratio)
    y_test_pred = predict_en(final_model, X_test_scaled)
    test_r = pearson_correlation(y_test, y_test_pred)
    print('test_pearson =', test_r)

    #################### end of local code ######################

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