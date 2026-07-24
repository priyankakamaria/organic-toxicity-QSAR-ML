# ============================================================
# STEP 06: REMOVE INVALID SMILES
# ============================================================

import pandas as pd
from rdkit import Chem
import os


print("="*60)
print("REMOVING INVALID SMILES")
print("="*60)


# ------------------------------------------------------------
# Input and output files
# ------------------------------------------------------------

input_file = "output/organic_toxicity_median.csv"

output_file = "output/clean_smiles_dataset.csv"


# ------------------------------------------------------------
# Load dataset
# ------------------------------------------------------------

df = pd.read_csv(input_file)

print("\nOriginal compounds :", len(df))


# ------------------------------------------------------------
# Check SMILES validity
# ------------------------------------------------------------

def validate_smiles(smiles):

    try:
        mol = Chem.MolFromSmiles(str(smiles))

        if mol is not None:
            return True
        else:
            return False

    except:
        return False



# Apply validation

df["SMILES_valid"] = df["SMILES"].apply(validate_smiles)



# ------------------------------------------------------------
# Separate valid and invalid compounds
# ------------------------------------------------------------

invalid_df = df[df["SMILES_valid"] == False].copy()

clean_df = df[df["SMILES_valid"] == True].copy()



# ------------------------------------------------------------
# Remove helper column
# ------------------------------------------------------------

clean_df = clean_df.drop(columns=["SMILES_valid"])

invalid_df = invalid_df.drop(columns=["SMILES_valid"])



# ------------------------------------------------------------
# Create output folder
# ------------------------------------------------------------

os.makedirs("output", exist_ok=True)



# ------------------------------------------------------------
# Save files
# ------------------------------------------------------------

clean_df.to_csv(
    output_file,
    index=False
)


invalid_df.to_csv(
    "output/invalid_smiles_removed.csv",
    index=False
)



# ------------------------------------------------------------
# Report
# ------------------------------------------------------------

print("\nInvalid SMILES removed :", len(invalid_df))

print("Valid compounds        :", len(clean_df))


print("\nClean dataset saved:")
print(output_file)


print("\nInvalid compounds saved:")
print("output/invalid_smiles_removed.csv")


print("\nCompleted Successfully")