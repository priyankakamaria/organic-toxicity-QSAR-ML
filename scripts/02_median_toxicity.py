import pandas as pd

print("=" * 60)
print("CALCULATING MEDIAN EC50 FOR DUPLICATE COMPOUND-SPECIES PAIRS")
print("=" * 60)

# Load dataset
df = pd.read_csv("data/organic_toxicity.csv", encoding="latin1")

print(f"\nOriginal rows : {len(df):,}")

# Columns used for grouping
group_cols = [
    "SMILES_Canonical_RDKit",
    "organism"
]

# Columns whose first value should be retained
first_cols = [
    "CAS",
    "chemical_name",
    "Conc_sign",
    "Duration_Value",
    "effect",
    "endpoint",
    "species_group",
    "SMILES"
]

# Aggregate
median_df = (
    df.groupby(group_cols, as_index=False)
      .agg({
          "mgperL": "median",
          **{col: "first" for col in first_cols}
      })
)

# Reorder columns
median_df = median_df[
    [
        "CAS",
        "chemical_name",
        "organism",
        "species_group",
        "Duration_Value",
        "effect",
        "endpoint",
        "mgperL",
        "SMILES",
        "SMILES_Canonical_RDKit",
        "Conc_sign"
    ]
]

print(f"Rows after median calculation : {len(median_df):,}")
print(f"Duplicate rows removed        : {len(df)-len(median_df):,}")

# Save
output_file = "output/organic_toxicity_median.csv"
median_df.to_csv(output_file, index=False)

print(f"\nDataset saved as:\n{output_file}")

print("\nFirst five rows:")
print(median_df.head())

print("\nCompleted Successfully.")