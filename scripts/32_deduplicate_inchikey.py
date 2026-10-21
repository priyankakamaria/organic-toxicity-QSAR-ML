
import pandas as pd
from pathlib import Path

input_file = Path(
    "output/prospective_pyrethroid/candidates_not_in_training.csv"
)

output_dir = Path("output/prospective_pyrethroid")

output_file = output_dir / "unique_inchikey_candidates.csv"
duplicate_file = output_dir / "duplicate_inchikey_records.csv"

df = pd.read_csv(input_file)

if "InChIKey" not in df.columns:
    raise ValueError("InChIKey column is missing.")

# Normalize whitespace and identify missing keys
df["InChIKey"] = df["InChIKey"].astype("string").str.strip()

missing_keys = df["InChIKey"].isna() | (df["InChIKey"] == "")

missing_df = df[missing_keys].copy()
with_keys = df[~missing_keys].copy()

# Keep one record for each exact full InChIKey
duplicates = with_keys[
    with_keys.duplicated(subset=["InChIKey"], keep="first")
].copy()

unique = with_keys.drop_duplicates(
    subset=["InChIKey"],
    keep="first"
).copy()

unique.to_csv(output_file, index=False)
duplicates.to_csv(duplicate_file, index=False)

print("=" * 60)
print("INCHIKEY DEDUPLICATION")
print("=" * 60)
print("Input records:", len(df))
print("Records with missing InChIKey:", len(missing_df))
print("Duplicate full InChIKeys:", len(duplicates))
print("Unique full InChIKeys:", len(unique))
print("Saved unique candidates:", output_file.resolve())
print("Saved duplicate records:", duplicate_file.resolve())
print("=" * 60)
