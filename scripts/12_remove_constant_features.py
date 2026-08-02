import pandas as pd
import os

print("=" * 70)
print("CONSTANT FEATURE REMOVAL")
print("TRAINING DATA ONLY")
print("=" * 70)

# ---------------------------------------------------------
# Input
# ---------------------------------------------------------

train_file = "output/train_dataset.csv"
test_file = "output/test_dataset.csv"

# ---------------------------------------------------------
# Outputs
# ---------------------------------------------------------

train_output = "output/train_no_constant_features.csv"
test_output = "output/test_no_constant_features.csv"

feature_file = "output/constant_filter_features.csv"

# ---------------------------------------------------------
# Load
# ---------------------------------------------------------

train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)

print("\nTraining shape:", train_df.shape)
print("Testing shape :", test_df.shape)

# ---------------------------------------------------------
# Metadata
# ---------------------------------------------------------

metadata_cols = [
    "CAS",
    "chemical_name",
    "SMILES",
    "organism",
    "SMILES_Canonical_RDKit",
    "Standardized_SMILES",
    "Molecular_Weight",
    "EC50_mol_L",
    "mgperL",
    "pEC50"
]

metadata_cols = [
    c for c in metadata_cols
    if c in train_df.columns
]

# ---------------------------------------------------------
# Feature columns
# ---------------------------------------------------------

feature_cols = [
    c for c in train_df.columns
    if c not in metadata_cols
]

print("\nInitial features:", len(feature_cols))

# ---------------------------------------------------------
# Find constant features using TRAINING DATA ONLY
# ---------------------------------------------------------

constant_features = []

for col in feature_cols:

    if train_df[col].nunique(dropna=False) <= 1:
        constant_features.append(col)

print("\nConstant features found:")
print(len(constant_features))

# ---------------------------------------------------------
# Retain features
# ---------------------------------------------------------

selected_features = [
    c for c in feature_cols
    if c not in constant_features
]

# ---------------------------------------------------------
# Apply SAME feature list to train and test
# ---------------------------------------------------------

train_final = train_df[
    metadata_cols + selected_features
]

test_final = test_df[
    metadata_cols + selected_features
]

# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

train_final.to_csv(
    train_output,
    index=False
)

test_final.to_csv(
    test_output,
    index=False
)

pd.DataFrame({
    "Feature": selected_features
}).to_csv(
    feature_file,
    index=False
)

print("\nFinal training shape:")
print(train_final.shape)

print("\nFinal testing shape:")
print(test_final.shape)

print("\nRemaining features:")
print(len(selected_features))

print("\nSaved:")
print(train_output)
print(test_output)
print(feature_file)

print("=" * 70)
print("Completed Successfully")
print("=" * 70)