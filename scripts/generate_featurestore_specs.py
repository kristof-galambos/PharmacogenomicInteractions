from __future__ import annotations

import argparse
from pathlib import Path


def azure_type_from_dtype(dtype_name: str) -> str:
    name = dtype_name.lower()
    if name in {"bool", "boolean"}:
        return "boolean"
    if name in {"int8", "int16", "int32", "uint8", "uint16", "uint32"}:
        return "integer"
    if name in {"int64", "uint64"}:
        return "long"
    if name in {"float16", "float32"}:
        return "float"
    if name in {"float64"}:
        return "double"
    if "datetime" in name:
        return "datetime"
    return "string"


def render_spec(source_path: str, feature_names_and_types: list[tuple[str, str]]) -> str:
    feature_lines = []
    for name, feature_type in feature_names_and_types:
        feature_lines.append(f"  - name: {name}")
        feature_lines.append(f"    type: {feature_type}")

    return "\n".join(
        [
            "$schema: http://azureml/sdk-2-0/FeatureSetSpec.json",
            "",
            "source:",
            "  type: parquet",
            f"  path: {source_path}",
            "  timestamp_column:",
            "    name: feature_ts",
            "features:",
            *feature_lines,
            "index_columns:",
            "  - name: SANGER_MODEL_ID",
            "    type: string",
            "",
        ]
    )


def load_feature_columns(parquet_path: Path) -> list[tuple[str, str]]:
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    required_columns = {"SANGER_MODEL_ID", "feature_ts"}
    missing = required_columns - set(df.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(
            f"{parquet_path} is missing required column(s): {missing_str}. "
            "Add a static feature_ts column before generating the Azure feature set specs."
        )

    feature_columns: list[tuple[str, str]] = []
    for column_name, dtype in df.dtypes.items():
        if column_name in required_columns:
            continue
        feature_columns.append((column_name, azure_type_from_dtype(str(dtype))))

    if not feature_columns:
        raise ValueError(f"{parquet_path} does not contain any feature columns.")

    return feature_columns


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--container", default="pgx-features")
    parser.add_argument("--storage-account", required=True)
    parser.add_argument("--expression-parquet", default="data/gene_expressions.parquet")
    parser.add_argument("--copy-number-parquet", default="data/copy_number_variations.parquet")
    parser.add_argument("--mutations-parquet", default="data/mutations.parquet")
    parser.add_argument("--expression-destination", default="molecular/expression/v1/gene_expressions.parquet")
    parser.add_argument("--copy-number-destination", default="molecular/copy_number/v1/copy_number_variations.parquet")
    parser.add_argument("--mutations-destination", default="molecular/mutations/v1/mutations.parquet")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    specs_root = repo_root / "featurestore" / "featuresets"

    configs = [
        (
            Path(args.expression_parquet),
            specs_root / "expression" / "spec" / "FeaturesetSpec.yaml",
            args.expression_destination,
        ),
        (
            Path(args.copy_number_parquet),
            specs_root / "copy_number" / "spec" / "FeaturesetSpec.yaml",
            args.copy_number_destination,
        ),
        (
            Path(args.mutations_parquet),
            specs_root / "mutations" / "spec" / "FeaturesetSpec.yaml",
            args.mutations_destination,
        ),
    ]

    for local_parquet, output_spec, destination in configs:
        feature_columns = load_feature_columns(repo_root / local_parquet)
        source_path = (
            f"abfss://{args.container}@{args.storage_account}.dfs.core.windows.net/{destination}"
        )
        write_text(output_spec, render_spec(source_path, feature_columns))
        print(f"wrote {output_spec}")


if __name__ == "__main__":
    main()
