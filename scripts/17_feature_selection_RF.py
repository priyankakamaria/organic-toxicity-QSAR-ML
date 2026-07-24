import pandas as pd
import numpy as np
import os

from sklearn.ensemble import RandomForestRegressor


print("="*70)
print("RANDOM FOREST FEATURE SELECTION")
print("="*70)


# Input
input_file = "output/train_dataset.csv"

# Outputs
output_dataset = "output/selected_features_train.csv"
output_names = "output/selected_feature_names.csv"


# Load dataset

print("\nLoading training dataset...")

df = pd.read_csv(input_file)

print("\nDataset shape:")
print(df.shape)


# Target column

target = "pEC50"


# Metadata columns

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


# Remove unavailable columns

metadata_cols = [
    col for col in metadata_cols
    if col in df.columns
]


print("\nMetadata columns removed:")
print(metadata_cols)


# Prepare X and y

X = df.drop(
    columns=metadata_cols + [target]
)

y = df[target]


print("\nNumber of descriptors:")
print(X.shape[1])


# Ensure numeric

X = X.select_dtypes(include=["number"])


print("\nNumeric descriptors:")
print(X.shape[1])


# Random Forest model

print("\nTraining Random Forest...")

rf = RandomForestRegressor(
    n_estimators=500,
    random_state=42,
    n_jobs=-1
)


rf.fit(X, y)


# Feature importance

importance = pd.DataFrame(
    {
        "Feature": X.columns,
        "Importance": rf.feature_importances_
    }
)


importance = importance.sort_values(
    by="Importance",
    ascending=False
)


# Select top 100

top_features = importance.head(100)["Feature"].tolist()


print("\nTop features selected:")
print(len(top_features))


# Save feature names

pd.DataFrame(
    {
        "Selected_Features": top_features
    }
).to_csv(
    output_names,
    index=False
)


# Create selected dataset

selected_df = df[
    metadata_cols +
    [target] +
    top_features
]


selected_df.to_csv(
    output_dataset,
    index=False
)


print("\nFinal selected dataset:")
print(selected_df.shape)


print("\nSaved files:")
print(output_dataset)
print(output_names)


print("\nCompleted Successfully")
print("="*70)