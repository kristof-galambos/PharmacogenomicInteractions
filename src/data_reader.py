import numpy as np
import pandas as pd
import sys

from src.utils import get_gene_expression_columns


def read_data(gdsc_path='/Users/kristof/Downloads/GDSC2_fitted_dose_response_27Oct23.xlsx',
              mutations_path='/Users/kristof/Downloads/mutations_summary_20260316.csv',
              expression_path='/Users/kristof/Downloads/rnaseq_merged_rsem_tpm_20260323.csv',
              drug_name='Ulixertinib', cancer_type=None):
    print('reading data')
    print('reading gdsc (62 sec)')
    gdsc_data = pd.read_excel(gdsc_path)
    # top_drugs = gdsc_data.groupby('DRUG_NAME').size().sort_values(ascending=False)
    # print(top_drugs)
    print('reading mutations (all: 33 sec, summary: few sec)')
    mutations_data = pd.read_csv(mutations_path)
    print('reading gene expression (10 sec)')
    expression_data = pd.read_csv(expression_path)

    print('GDSC DATA:')
    print(gdsc_data.shape)
    print(gdsc_data.columns)
    print('MUTATIONS DATA:')
    print(mutations_data.shape)
    print(mutations_data.columns)
    # print(cnv_data.columns)
    print('GENE EXPRESSION DATA:')
    print(expression_data.shape)
    print(expression_data.columns)

    print('preparing data')
    mutations = prepare_mutations(mutations_data)
    expression = prepare_expression(expression_data)
    print(mutations.shape)
    print(expression.shape)
    sanity_check(mutations)

    gdsc_subset = gdsc_data[["DRUG_NAME", "CANCER_TYPE", "SANGER_MODEL_ID", "LN_IC50"]].copy()
    gdsc_subset["SANGER_MODEL_ID"] = gdsc_subset["SANGER_MODEL_ID"].astype(str)
    mutations = mutations.rename(columns={'model_id': 'SANGER_MODEL_ID'})
    expression = expression.rename(columns={'model_id': 'SANGER_MODEL_ID'})

    dataset = gdsc_subset.merge(
        mutations,
        on="SANGER_MODEL_ID",
        how="inner",
    )
    expression = reduce_expression_features(expression)
    dataset = dataset.merge(
        expression,
        on="SANGER_MODEL_ID",
        how="inner",
    )

    dataset = dataset[dataset["DRUG_NAME"] == drug_name].copy()
    if cancer_type is not None:
        dataset = dataset[dataset["CANCER_TYPE"] == cancer_type].copy()

    dataset = dataset.dropna(subset=["LN_IC50"]).drop_duplicates(subset=["SANGER_MODEL_ID"])
    dataset = dataset.reset_index(drop=True)

    return dataset


def sanity_check(df):
    # check that no numeric values of the mutation matrix are above 1
    numeric_df = df.select_dtypes(include=["uint8"])
    print('are there any values above 1:')
    print((numeric_df > 1).any().any())
    # check how many and what percentage of the entries are 1
    num_entries = numeric_df.shape[0] * numeric_df.shape[1]
    print('fraction of non-zero entries:')
    print(np.sum(numeric_df) / num_entries)


def prepare_mutations(df):
    # -----------------------------
    # 1. Keep only rows with key IDs
    # -----------------------------
    mut = df.copy()
    mut = mut.dropna(subset=["model_id", "gene_symbol"])

    # Optional: standardize IDs as strings
    mut["model_id"] = mut["model_id"].astype(str)
    mut["gene_symbol"] = mut["gene_symbol"].astype(str)

    # -----------------------------
    # 2. Optional filtering
    # -----------------------------
    # A. Keep only coding mutations.
    # The source file stores booleans as strings like "t"/"f", not Python True/False.
    if "coding" in mut.columns:
        coding_values = (
            mut["coding"]
            .astype(str)
            .str.strip()
            .str.lower()
        )
        mut_coding = mut[coding_values.isin({"true", "t", "1", "yes"})].copy()
    else:
        mut_coding = mut.copy()

    # B. Keep only likely functional effects
    functional_effects = {
        "missense",
        "missense_variant",
        "frameshift",
        "frameshift_variant",
        "nonsense",
        "stop_gained",
        "stop_lost",
        "start_lost",
        "ess_splice",
        "splice_donor_variant",
        "splice_acceptor_variant",
        "splice_region",
        "inframe",
        "inframe_insertion",
        "inframe_deletion",
        "protein_altering_variant",
    }

    if "effect" in mut_coding.columns:
        effect_values = (
            mut_coding["effect"]
            .astype(str)
            .str.strip()
            .str.lower()
        )
        mut_coding = mut_coding[
            effect_values.isin(functional_effects)
        ].copy()

    # If the optional filters remove everything, fall back to the ID-cleaned data
    # so the pivot still produces a usable mutation matrix.
    if mut_coding.empty:
        mut_coding = mut.copy()

    # C. Optional: keep only cancer driver mutations
    # Uncomment this if you want a paper-like filtered version
    # if "cancer_driver" in mut_coding.columns:
    #     mut_coding = mut_coding[mut_coding["cancer_driver"] == True].copy()

    # -----------------------------
    # 3. Collapse to one mutation flag per (cell line, gene)
    # -----------------------------
    # If a cell line has multiple mutations in the same gene, keep a single 1
    mut_binary_long = (
        mut_coding[["model_id", "gene_symbol"]]
        .drop_duplicates()
        .assign(mutated=1)
    )

    # -----------------------------
    # 4. Pivot to wide matrix
    # -----------------------------
    mutation_matrix = mut_binary_long.pivot(
        index="model_id",
        columns="gene_symbol",
        values="mutated"
    ).fillna(0).astype("uint8")

    # Make column names easier to identify after merging with other omics
    mutation_matrix.columns = [f"MUT_{g}" for g in mutation_matrix.columns]

    # Bring model_id back as a normal column if you want
    mutation_matrix = mutation_matrix.reset_index()

    print(mutation_matrix.shape)
    print(mutation_matrix.head())

    return mutation_matrix


def prepare_expression(df):
    expr = df.copy()

    if expr.shape[1] < 4 or expr.shape[0] < 4:
        raise ValueError("Expression dataframe does not match an expected format.")

    metadata_cols = list(expr.columns[:3])
    model_cols = list(expr.columns[3:])

    expr = expr.iloc[3:].copy()
    expr = expr.rename(columns={
        metadata_cols[0]: "gene_symbol",
        metadata_cols[1]: "ensembl_gene_id",
        metadata_cols[2]: "gene_id",
    })
    expr = expr.dropna(subset=["gene_symbol"])
    expr["gene_symbol"] = expr["gene_symbol"].astype(str)

    expr_numeric = expr[model_cols].apply(pd.to_numeric, errors="coerce")
    expr_numeric.index = expr["gene_symbol"]

    expression_matrix = expr_numeric.groupby(level=0).mean().transpose().fillna(0.0)
    expression_matrix.index.name = "model_id"

    expression_matrix.columns = [f"GEX_{gene}" for gene in expression_matrix.columns]
    expression_matrix = expression_matrix.reset_index()

    print(expression_matrix.shape)
    print(expression_matrix.head())

    return expression_matrix


def prepare_copy_number(df):
    cnv = df.copy()

    if cnv.shape[1] < 2 or cnv.shape[0] < 4:
        raise ValueError("Copy number dataframe does not match an expected format.")

    gene_col = cnv.columns[0]
    sample_cols = list(cnv.columns[1:])

    row_labels = cnv[gene_col].astype(str).str.strip()
    row_labels_lower = row_labels.str.lower()

    model_id_rows = row_labels_lower == "model_id"
    if not model_id_rows.any():
        raise ValueError("Copy number dataframe is missing the model_id metadata row.")

    model_ids = (
        cnv.loc[model_id_rows, sample_cols]
        .iloc[0]
        .astype(str)
        .str.strip()
    )
    valid_model_ids = model_ids.notna() & (model_ids != "") & (model_ids.str.lower() != "nan")
    if not valid_model_ids.any():
        raise ValueError("Copy number dataframe does not contain any usable model IDs.")

    metadata_rows = {"model_id", "source", "symbol"}
    gene_rows = ~row_labels_lower.isin(metadata_rows)
    gene_symbols = row_labels[gene_rows]
    valid_gene_symbols = (
        gene_symbols.notna()
        & (gene_symbols != "")
        & (gene_symbols.str.lower() != "nan")
    )
    if not valid_gene_symbols.any():
        raise ValueError("Copy number dataframe does not contain any usable gene symbols.")

    cnv_gene_rows = cnv.loc[gene_rows, sample_cols].loc[valid_gene_symbols].copy()
    gene_symbols = gene_symbols.loc[valid_gene_symbols].astype(str)

    cnv_numeric = cnv_gene_rows.apply(pd.to_numeric, errors="coerce")
    cnv_numeric.index = gene_symbols

    # Some source files contain duplicate gene symbols and repeated model IDs
    # from multiple sources. Average them to keep one feature per gene and one
    # row per cell line.
    copy_number_by_gene = cnv_numeric.groupby(level=0).mean()
    copy_number_by_gene = copy_number_by_gene.loc[:, valid_model_ids.to_numpy()]
    copy_number_by_gene.columns = model_ids.loc[valid_model_ids].to_numpy()

    copy_number_matrix = copy_number_by_gene.transpose().groupby(level=0).mean()
    copy_number_matrix = copy_number_matrix.fillna(2.0)
    copy_number_matrix.index.name = "model_id"

    copy_number_matrix.columns = [f"CNV_{gene}" for gene in copy_number_matrix.columns]
    copy_number_matrix = copy_number_matrix.reset_index()

    print(copy_number_matrix.shape)
    print(copy_number_matrix.head())

    return copy_number_matrix


def reduce_expression_features(df, top_variance_top_n=500):
    gene_cols = get_gene_expression_columns(df)
    if gene_cols:
        top_variance_gene_cols = select_top_variance_columns(
            df[gene_cols],
            top_n=top_variance_top_n,
        )
        keep_cols = ['SANGER_MODEL_ID'] + top_variance_gene_cols
        kept = df[keep_cols]
        print(
            f'Pre-join top-variance selection on gene expression data completed: '
            f'kept {len(top_variance_gene_cols)} genes (top_n={top_variance_top_n}), '
            f'gene expression data shape now={kept.shape}'
        )
        return kept
    else:
        print('Pre-join top-variance enabled, but no gene columns were detected in gene expression data.')
        return df


def select_top_variance_columns(
    df: pd.DataFrame,
    top_n: int = 1000,
):
    if top_n <= 0:
        raise ValueError("top_n must be > 0.")

    numeric_df = df.select_dtypes(include=["number"])
    if numeric_df.empty:
        raise ValueError("No numeric columns found for variance selection.")

    variances = numeric_df.var(axis=0, skipna=True)
    variances = variances.fillna(0.0)
    selected = variances.sort_values(ascending=False).head(top_n).index.tolist()

    if len(selected) < top_n:
        print(
            f"Requested top_n={top_n} columns, but only {len(selected)} numeric columns are available."
        )

    return selected
