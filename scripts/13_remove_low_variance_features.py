import pandas as pd
import os

from sklearn.feature_selection import VarianceThreshold

print("=" * 70)
print("LOW-VARIANCE FEATURE REMOVAL")
print("TRAINING DATA ONLY")
print("=" * 70)

# ---------------------------------------------------------
# Input
# ---------------------------------------------------------

train_file = "output/train_no_constant_features.csv"
test_file = "output/test_no_constant_features.csv"

# ---------------------------------------------------------
# Output
# ---------------------------------------------------------

train_output = "output/train_no_low_variance_features.csv"
test_output = "output/test_no_low_variance_features.csv"

feature_file = "output/low_variance_features.csv"

# ---------------------------------------------------------
# Load
# ---------------------------------------------------------

train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)

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

X_train = train_df[feature_cols]

print("\nTraining shape:")
print(X_train.shape)

# ---------------------------------------------------------
# Variance filter
# ---------------------------------------------------------

threshold = 0.01

selector = VarianceThreshold(
    threshold=threshold
)

selector.fit(X_train)

support = selector.get_support()

selected_features = [
    feature_cols[i]
    for i, keep in enumerate(support)
    if keep
]

removed_features = [
    feature_cols[i]
    for i, keep in enumerate(support)
    if not keep
]

print("\nVariance threshold:", threshold)

print("Initial features:", len(feature_cols))
print("Removed:", len(removed_features))
print("Remaining:", len(selected_features))

# ---------------------------------------------------------
# Apply same features to train and test
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

print("\nTraining final shape:")
print(train_final.shape)

print("Testing final shape:")
print(test_final.shape)

print("\nSaved:")
print(train_output)
print(test_output)
print(feature_file)

print("=" * 70)
print("Completed Successfully")
print("=" * 70)