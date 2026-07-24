import os
import pandas as pd
from sklearn.feature_selection import VarianceThreshold

print("=" * 70)
print("REMOVING LOW VARIANCE FEATURES")
print("=" * 70)

# ----------------------------------------------------------
# Input / Output
# ----------------------------------------------------------

input_file = "output/combined_no_constant_features.csv"
output_file = "output/combined_no_low_variance_features.csv"

# Recommended threshold
threshold = 0.01

# ----------------------------------------------------------
# Load dataset
# ----------------------------------------------------------

print("\nLoading dataset...")

df = pd.read_csv(input_file)

print("\nDataset shape:")
print(df.shape)

# ----------------------------------------------------------
# Automatically separate metadata and descriptors
# ----------------------------------------------------------

metadata_df = df.select_dtypes(include=["object", "string"]).copy()
descriptor_df = df.select_dtypes(include=["number"]).copy()

print("\nMetadata columns:")
print(list(metadata_df.columns))

print("\nNumber of metadata columns:")
print(metadata_df.shape[1])

print("\nNumber of descriptor columns:")
print(descriptor_df.shape[1])

# ----------------------------------------------------------
# Remove low variance descriptors
# ----------------------------------------------------------

print("\nRemoving low variance features...")

selector = VarianceThreshold(threshold=threshold)

descriptor_selected = selector.fit_transform(descriptor_df)

selected_columns = descriptor_df.columns[selector.get_support()]
removed_columns = descriptor_df.columns[~selector.get_support()]

print("\nLow variance features removed:")
print(len(removed_columns))

print("\nDescriptors remaining:")
print(len(selected_columns))

# ----------------------------------------------------------
# Create final dataset
# ----------------------------------------------------------

selected_df = pd.DataFrame(
    descriptor_selected,
    columns=selected_columns,
    index=df.index
)

final_df = pd.concat([metadata_df, selected_df], axis=1)

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