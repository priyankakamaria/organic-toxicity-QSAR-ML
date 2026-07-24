import os
import pandas as pd

print("=" * 70)
print("REMOVING CONSTANT FEATURES")
print("=" * 70)

# ----------------------------------------------------------
# Input and Output
# ----------------------------------------------------------

input_file = "output/combined_rdkit_morgan_maccs_dataset.csv"
output_file = "output/combined_no_constant_features.csv"

# ----------------------------------------------------------
# Load dataset
# ----------------------------------------------------------

print("\nLoading dataset...")

df = pd.read_csv(input_file)

print("\nOriginal dataset shape:")
print(df.shape)

# ----------------------------------------------------------
# Metadata columns
# ----------------------------------------------------------

meta_cols = ["CAS", "chemical_name", "SMILES"]

print("\nMetadata columns:")
print(meta_cols)

descriptor_cols = [col for col in df.columns if col not in meta_cols]

print("\nNumber of descriptors:")
print(len(descriptor_cols))

# ----------------------------------------------------------
# Separate metadata and descriptors
# ----------------------------------------------------------

meta_df = df[meta_cols]
X = df[descriptor_cols]

# ----------------------------------------------------------
# Find constant features
# ----------------------------------------------------------

print("\nChecking constant features...")

constant_features = [
    col for col in X.columns
    if X[col].nunique(dropna=False) == 1
]

print("\nConstant features found:")
print(len(constant_features))

if len(constant_features) > 0:
    print("\nFirst few constant features:")
    print(constant_features[:20])

# ----------------------------------------------------------
# Remove constant features
# ----------------------------------------------------------

X_clean = X.drop(columns=constant_features)

# ----------------------------------------------------------
# Combine metadata and descriptors
# ----------------------------------------------------------

final_df = pd.concat([meta_df, X_clean], axis=1)

print("\nFinal dataset shape:")
print(final_df.shape)

print("\nDescriptors remaining:")
print(X_clean.shape[1])

print("\nDescriptors removed:")
print(len(constant_features))

# ----------------------------------------------------------
# Save dataset
# ----------------------------------------------------------

os.makedirs("output", exist_ok=True)

final_df.to_csv(output_file, index=False)

print("\nSaved file:")
print(output_file)

print("\nCompleted Successfully.")
print("=" * 70)