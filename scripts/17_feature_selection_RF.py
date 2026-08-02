import pandas as pd
from sklearn.ensemble import RandomForestRegressor

print("="*70)
print("RANDOM FOREST FEATURE SELECTION")
print("TRAINING DATA ONLY")
print("="*70)

# ---------------------------------------------------------
# Input
# ---------------------------------------------------------

input_file = "output/train_no_correlation_features.csv"

# ---------------------------------------------------------
# Outputs
# ---------------------------------------------------------

output_dataset = "output/selected_features_train.csv"
output_names = "output/selected_feature_names.csv"

# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------

print("\nLoading training dataset...")

df = pd.read_csv(input_file)

print("\nDataset shape:")
print(df.shape)

# ---------------------------------------------------------
# Target column
# ---------------------------------------------------------

target = "pEC50"

# ---------------------------------------------------------
# Metadata columns
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
    "mgperL"
]

metadata_cols = [col for col in metadata_cols if col in df.columns]

print("\nMetadata columns removed:")
print(metadata_cols)

# ---------------------------------------------------------
# Prepare X and y
# ---------------------------------------------------------

X = df.drop(columns=metadata_cols + [target])
y = df[target]

# Keep only numeric columns

X = X.select_dtypes(include=["number"])

print("\nNumber of descriptors after preprocessing:")
print(X.shape[1])

# ---------------------------------------------------------
# Train Random Forest
# ---------------------------------------------------------

print("\nTraining Random Forest...")

rf = RandomForestRegressor(
    n_estimators=500,
    random_state=42,
    n_jobs=-1
)

rf.fit(X, y)

# ---------------------------------------------------------
# Feature importance
# ---------------------------------------------------------

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

# ---------------------------------------------------------
# Select Top 100 Features
# ---------------------------------------------------------

top_features = importance.head(100)["Feature"].tolist()

print("\nTop features selected:", len(top_features))

# Save selected feature names

pd.DataFrame({
    "Selected_Features": top_features
}).to_csv(
    output_names,
    index=False
)

# Save feature importance for publication

importance.to_csv(
    "output/rf_feature_importance_all.csv",
    index=False
)

# ---------------------------------------------------------
# Create training dataset with selected features
# ---------------------------------------------------------

selected_df = df[
    metadata_cols +
    [target] +
    top_features
]

selected_df.to_csv(
    output_dataset,
    index=False
)

# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

print("\nInitial descriptors :", X.shape[1])
print("Selected descriptors:", len(top_features))

print("\nFinal training dataset shape:")
print(selected_df.shape)

print("\nSaved files:")
print(output_dataset)
print(output_names)
print("output/rf_feature_importance_all.csv")

print("="*70)
print("Completed Successfully")
print("="*70)