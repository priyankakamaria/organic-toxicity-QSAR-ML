import pandas as pd

# Load dataset
df = pd.read_csv("data/organic_toxicity.csv", encoding="latin1")

print("=" * 60)
print("ORGANIC TOXICITY DATASET SUMMARY")
print("=" * 60)

# Dataset shape
print(f"\nRows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")

# Column names
print("\nColumn Names:")
print(df.columns.tolist())

# Data types
print("\nData Types:")
print(df.dtypes)

# Missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Duplicate rows
print(f"\nDuplicate Rows : {df.duplicated().sum()}")

# Duplicate SMILES
print(f"Duplicate SMILES : {df['SMILES'].duplicated().sum()}")

# Target statistics
print("\nLogToxicity Statistics")
print(df["LogToxicity"].describe())

print("\nFirst 5 Rows")
print(df.head())

print("\nDataset checking completed successfully.")