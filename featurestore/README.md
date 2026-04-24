## Azure ML Feature Store Scaffold

This directory holds the Azure ML managed feature store assets for the three
molecular modalities in this repo.

Layout:

- `entities/cell_line.yaml`: shared entity keyed by `SANGER_MODEL_ID`
- `featuresets/expression/`
- `featuresets/copy_number/`
- `featuresets/mutations/`

Each feature set has:

- `featureset_asset.yaml`: the Azure ML feature set asset
- `spec/FeaturesetSpec.yaml`: the source/specification used by the asset

Important constraints:

1. Azure ML feature set specs require an explicit `features:` list.
2. Azure ML feature set specs also require a timestamp column in the source
   data.

Your current prepared Parquet files appear to be cell-line keyed, but they
don't currently declare a feature timestamp in this repo's prep code. Before
registering the feature sets, add a static timestamp column such as
`feature_ts` to each modality parquet.

Then generate the final spec files with:

```bash
python3 scripts/generate_featurestore_specs.py \
  --storage-account <storage-account-name> \
  --container pgx-features
```

By default the script reads:

- `data/gene_expressions.parquet`
- `data/copy_number_variations.parquet`
- `data/mutations.parquet`

and writes the final `FeaturesetSpec.yaml` files in this directory.

After that, create the entity and feature sets:

```bash
az ml feature-store-entity create \
  --file featurestore/entities/cell_line.yaml \
  --resource-group <resource-group> \
  --feature-store-name <feature-store-name>

az ml feature-set create \
  --file featurestore/featuresets/expression/featureset_asset.yaml \
  --resource-group <resource-group> \
  --feature-store-name <feature-store-name>

az ml feature-set create \
  --file featurestore/featuresets/copy_number/featureset_asset.yaml \
  --resource-group <resource-group> \
  --feature-store-name <feature-store-name>

az ml feature-set create \
  --file featurestore/featuresets/mutations/featureset_asset.yaml \
  --resource-group <resource-group> \
  --feature-store-name <feature-store-name>
```

That registers the entity and feature sets in the Azure ML feature store.
It does not create the Azure ML feature store resource itself, and it assumes:

- the feature store already exists
- the Parquet files are already uploaded to ADLS Gen2
- the identities involved have the required storage permissions
