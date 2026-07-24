import os
import pandas as pd
import numpy as np

print("=" * 70)
print("REMOVING HIGHLY CORRELATED FEATURES")
print("=" * 70)

# ----------------------------------------------------------
# Input / Output
# ----------------------------------------------------------

input_file = "output/combined_no_low_variance_features.csv"
output_file = "output/combined_no_correlated_features.csv"

# Correlation threshold
correlation_threshold = 0.95

# ----------------------------------------------------------
# Load dataset
# ----------------------------------------------------------

print("\nLoading dataset...")

df = pd.read_csv(input_file)

print("\nDataset shape:")
print(df.shape)

# ----------------------------------------------------------
# Separate metadata and descriptors
# ----------------------------------------------------------

metadata_df = df.select_dtypes(include=["object", "string"]).copy()
descriptor_df = df.select_dtypes(include=["number"]).copy()

print("\nMetadata columns:")
print(metadata_df.shape[1])

print("Descriptor columns:")
print(descriptor_df.shape[1])

# ----------------------------------------------------------
# Calculate correlation matrix
# ----------------------------------------------------------

print("\nCalculating correlation matrix...")

corr_matrix = descriptor_df.corr().abs()

print("Done.")

# ----------------------------------------------------------
# Find highly correlated features
# ----------------------------------------------------------

print("\nFinding highly correlated features...")

upper = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
)

to_drop = [
    column for column in upper.columns
    if any(upper[column] > correlation_threshold)
]

print("\nHighly correlated features found:")
print(len(to_drop))

# ----------------------------------------------------------
# Remove correlated features
# ----------------------------------------------------------

descriptor_clean = descriptor_df.drop(columns=to_drop)

print("\nRemaining descriptors:")
print(descriptor_clean.shape[1])

# ----------------------------------------------------------
# Combine metadata + descriptors
# ----------------------------------------------------------

final_df = pd.concat(
    [metadata_df, descriptor_clean],
    axis=1
)

print("\nFinal dataset shape:")
print(final_df.shape)

# ----------------------------------------------------------
# Save
# ----------------------------------------------------------

os.makedirs("output", exist_ok=True)

final_df.to_csv(output_file, index=False)

print("\nSaved file:")
print(output_file)

print("\nCompleted Successfully.")
print("=" * 70)