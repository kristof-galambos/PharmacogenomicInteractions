import numpy as np


def get_mutation_columns(cols):
    return [x for x in cols if x.startswith('MUT_')]


def get_gene_expression_columns(cols):
    return [x for x in cols if x.startswith('GEX_')]


def pearson_correlation(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    y_true_centered = y_true - y_true.mean()
    y_pred_centered = y_pred - y_pred.mean()

    denominator = np.sqrt(np.sum(y_true_centered ** 2) * np.sum(y_pred_centered ** 2))
    if denominator == 0:
        return 0.0
    return float(np.sum(y_true_centered * y_pred_centered) / denominator)