import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
import os

print("="*70)
print("CALCULATING pEC50")
print("="*70)

# Input file
input_file = "output/combined_no_correlated_features.csv"

# Output file
output_file = "output/dataset_with_pEC50.csv"


# Load dataset
print("\nLoading dataset...")

df = pd.read_csv(input_file)

print("\nOriginal dataset shape:")
print(df.shape)


# Check columns
if "SMILES" not in df.columns:
    raise Exception("SMILES column not found")

if "mgperL" not in df.columns:
    raise Exception("mgperL toxicity column not found")


# Calculate molecular weight

print("\nCalculating molecular weights...")


def calculate_mw(smiles):

    try:
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            return np.nan

        return Descriptors.MolWt(mol)

    except:
        return np.nan


df["Molecular_Weight"] = df["SMILES"].apply(calculate_mw)


# Convert mg/L to mol/L

print("Converting mg/L to mol/L...")


df["EC50_mol_L"] = (
    df["mgperL"] /
    (1000 * df["Molecular_Weight"])
)


# Calculate pEC50

print("Calculating pEC50...")


df["pEC50"] = -np.log10(df["EC50_mol_L"])


# Remove invalid values

before = len(df)


df = df.replace([np.inf, -np.inf], np.nan)

df = df.dropna(
    subset=[
        "Molecular_Weight",
        "EC50_mol_L",
        "pEC50"
    ]
)


after = len(df)


print("\nQuality check:")
print("-----------------------------")
print("Records before cleaning :", before)
print("Records after cleaning  :", after)
print("Removed records        :", before-after)


print("\npEC50 statistics:")
print(df["pEC50"].describe())


# Save

os.makedirs("output", exist_ok=True)

df.to_csv(
    output_file,
    index=False
)


print("\nSaved file:")
print(output_file)


print("\nCompleted Successfully")
print("="*70)