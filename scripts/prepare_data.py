import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.data_reader import prepare_expression, reduce_expression_features, \
    prepare_mutations, sanity_check, prepare_copy_number, reduce_copy_number_features


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


def get_prepared_expression_data(expression_path, top_n=1000):
    print('reading gene expression (10 sec)')
    expression_data = pd.read_csv(expression_path, low_memory=False)

    print('GENE EXPRESSION DATA:')
    print(expression_data.shape)
    print(expression_data.columns)

    expression = prepare_expression(expression_data)

    expression = expression.rename(columns={'model_id': 'SANGER_MODEL_ID'})

    expression = reduce_expression_features(expression, top_variance_top_n=top_n)

    return expression


def get_prepared_cnv_data(cnv_path, top_n=500):
    cnv_data = pd.read_csv(cnv_path, low_memory=False)

    print('COPY NUMBER DATA:')
    print(cnv_data.shape)
    print(cnv_data.columns)

    cnv = prepare_copy_number(cnv_data)
    cnv = cnv.rename(columns={'model_id': 'SANGER_MODEL_ID'})
    cnv = reduce_copy_number_features(cnv, top_variance_top_n=top_n)

    return cnv


def write_parquet_data(df, target_path):
    print('writing. df.shape =', df.shape)
    # Convert DataFrame to Apache Arrow Table
    table = pa.Table.from_pandas(df)

    pq.write_table(table, target_path)
    print('wrote to', target_path)


if __name__ == '__main__':
    gdsc = get_prepared_gdsc_data(gdsc_path='/Users/kristof/Downloads/GDSC2_fitted_dose_response_27Oct23.xlsx')
    write_parquet_data(gdsc, '../data/gdsc.parquet')

    mut = get_prepared_mutations_data(mutations_path='/Users/kristof/Downloads/mutations_summary_20260316.csv')
    write_parquet_data(mut, '../data/mutations.parquet')

    TOP_N = 500
    expr = get_prepared_expression_data(expression_path='/Users/kristof/Downloads/rnaseq_merged_rsem_tpm_20260323.csv',
                                        top_n=TOP_N)
    write_parquet_data(expr, f'../data/gene_expressions_{TOP_N}.parquet')

    cnv = get_prepared_cnv_data(
        cnv_path='/Users/kristof/Downloads/WES_pureCN_CNV_genes_total_copy_number_20250207.csv',
        top_n=TOP_N,
    )
    write_parquet_data(cnv, f'../data/copy_number_variations_{TOP_N}.parquet')
