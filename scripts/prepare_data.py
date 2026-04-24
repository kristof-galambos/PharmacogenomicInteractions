import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.data_reader import prepare_expression, reduce_expression_features, \
    prepare_mutations, sanity_check, prepare_copy_number, reduce_copy_number_features, select_top_variance_columns


FEATURE_TS = pd.Timestamp("2026-04-24T00:00:00Z")


def get_prepared_gdsc_data(gdsc_path):
    print('reading gdsc (62 sec)')
    gdsc_data = pd.read_excel(gdsc_path)
    # top_drugs = gdsc_data.groupby('DRUG_NAME').size().sort_values(ascending=False)
    # print(top_drugs)

    print('GDSC DATA:')
    print(gdsc_data.shape)
    print(gdsc_data.columns)

    gdsc_subset = gdsc_data[["DRUG_NAME", "CANCER_TYPE", "SANGER_MODEL_ID", "LN_IC50"]].copy()
    gdsc_subset["SANGER_MODEL_ID"] = gdsc_subset["SANGER_MODEL_ID"].astype(str)

    # dataset = gdsc_subset.merge(
    #     mutations,
    #     on="SANGER_MODEL_ID",
    #     how="inner",
    # )
    # dataset = dataset.merge(
    #     expression,
    #     on="SANGER_MODEL_ID",
    #     how="inner",
    # )

    dataset = gdsc_subset.dropna(subset=["LN_IC50"]).drop_duplicates(subset=["SANGER_MODEL_ID"])
    dataset = dataset.reset_index(drop=True)

    return dataset


def get_prepared_mutations_data(mutations_path):
    print('reading mutations (all: 33 sec, summary: few sec)')
    mutations_data = pd.read_csv(mutations_path)

    print('MUTATIONS DATA:')
    print(mutations_data.shape)
    print(mutations_data.columns)

    mutations = prepare_mutations(mutations_data)

    sanity_check(mutations)

    mutations = mutations.rename(columns={'model_id': 'SANGER_MODEL_ID'})

    return mutations


def get_prepared_expression_data(expression_path):
    print('reading gene expression (10 sec)')
    expression_data = pd.read_csv(expression_path, low_memory=False)

    print('GENE EXPRESSION DATA:')
    print(expression_data.shape)
    print(expression_data.columns)

    expression = prepare_expression(expression_data)

    expression = expression.rename(columns={'model_id': 'SANGER_MODEL_ID'})

    return expression


def get_prepared_cnv_data(cnv_path):
    cnv_data = pd.read_csv(cnv_path, low_memory=False)

    print('COPY NUMBER DATA:')
    print(cnv_data.shape)
    print(cnv_data.columns)

    cnv = prepare_copy_number(cnv_data)
    cnv = cnv.rename(columns={'model_id': 'SANGER_MODEL_ID'})

    return cnv


def write_parquet_data(df, target_path):
    print('writing. df.shape =', df.shape)
    # Convert DataFrame to Apache Arrow Table
    table = pa.Table.from_pandas(df)

    pq.write_table(table, target_path)
    print('wrote to', target_path)


def add_feature_timestamp(df, id_column="SANGER_MODEL_ID"):
    if id_column not in df.columns:
        raise ValueError(f"Expected {id_column!r} column before adding feature_ts.")

    out = df.copy()
    insert_at = out.columns.get_loc(id_column) + 1
    out.insert(insert_at, "feature_ts", FEATURE_TS)
    return out


if __name__ == '__main__':
    gdsc = get_prepared_gdsc_data(gdsc_path='/Users/kristof/Downloads/GDSC2_fitted_dose_response_27Oct23.xlsx')
    write_parquet_data(gdsc, '../data/gdsc.parquet')

    mut = get_prepared_mutations_data(mutations_path='/Users/kristof/Downloads/mutations_summary_20260316.csv')
    mut = add_feature_timestamp(mut)
    write_parquet_data(mut, '../data/mutations.parquet')

    EXPRESSION_TOP_N = None  # or 500
    expr = get_prepared_expression_data(expression_path='/Users/kristof/Downloads/rnaseq_merged_rsem_tpm_20260323.csv')
    expr = add_feature_timestamp(expr)
    if EXPRESSION_TOP_N:
        expr = reduce_expression_features(expr, top_variance_top_n=EXPRESSION_TOP_N)
        write_parquet_data(expr, f'../data/gene_expressions_{EXPRESSION_TOP_N}.parquet')
    else:
        write_parquet_data(expr, '../data/gene_expressions.parquet')
        order = select_top_variance_columns(expr, top_n=None)
        write_parquet_data(order, '../data/gene_expressions_variance_order.parquet')

    CNV_TOP_N = None  # or 500
    cnv = get_prepared_cnv_data(cnv_path='/Users/kristof/Downloads/WES_pureCN_CNV_genes_total_copy_number_20250207.csv')
    cnv = add_feature_timestamp(cnv)
    if CNV_TOP_N:
        cnv = reduce_copy_number_features(cnv, top_variance_top_n=CNV_TOP_N)
        write_parquet_data(cnv, f'../data/copy_number_variations_{CNV_TOP_N}.parquet')
    else:
        write_parquet_data(cnv, '../data/copy_number_variations.parquet')
        order = select_top_variance_columns(cnv, top_n=None)
        write_parquet_data(order, '../data/copy_number_variations_variance_order.parquet')
