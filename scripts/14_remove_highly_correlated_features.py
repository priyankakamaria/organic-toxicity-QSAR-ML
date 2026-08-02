import pandas as pd
import numpy as np

print("=" * 70)
print("HIGH CORRELATION FEATURE REMOVAL")
print("TRAINING DATA ONLY")
print("=" * 70)

# ---------------------------------------------------------
# Input
# ---------------------------------------------------------

train_file = "output/train_no_low_variance_features.csv"
test_file = "output/test_no_low_variance_features.csv"

# ---------------------------------------------------------
# Output
# ---------------------------------------------------------

train_output = "output/train_no_correlation_features.csv"
test_output = "output/test_no_correlation_features.csv"

feature_file = "output/correlation_filter_features.csv"

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

feature_cols = [
    c for c in train_df.columns
    if c not in metadata_cols
]

X_train = train_df[feature_cols]

print("\nInitial features:")
print(X_train.shape[1])

# ---------------------------------------------------------
# Correlation threshold
# ---------------------------------------------------------

threshold = 0.90

# ---------------------------------------------------------
# Correlation matrix
# ---------------------------------------------------------

corr_matrix = X_train.corr().abs()

upper = corr_matrix.where(
    np.triu(
        np.ones(corr_matrix.shape),
        k=1
    ).astype(bool)
)

to_remove = [
    column
    for column in upper.columns
    if any(upper[column] > threshold)
]

selected_features = [
    c for c in feature_cols
    if c not in to_remove
]

print("\nCorrelation threshold:", threshold)

print("Initial features:", len(feature_cols))
print("Removed:", len(to_remove))
print("Remaining:", len(selected_features))

# ---------------------------------------------------------
# Apply same feature list
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