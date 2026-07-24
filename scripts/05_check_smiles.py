from rdkit import Chem
import pandas as pd

print("=" * 60)
print("CHECKING SMILES VALIDITY")
print("=" * 60)

# Load dataset
df = pd.read_csv("output/organic_toxicity_median.csv")

valid = 0
invalid = []

for i, smi in enumerate(df["SMILES_Canonical_RDKit"]):
    mol = Chem.MolFromSmiles(str(smi))

    if mol is None:
        invalid.append(i)
    else:
        valid += 1

print(f"\nTotal compounds : {len(df):,}")
print(f"Valid SMILES    : {valid:,}")
print(f"Invalid SMILES  : {len(invalid):,}")

if len(invalid) > 0:
    invalid_df = df.iloc[invalid]
    invalid_df.to_csv("output/invalid_smiles.csv", index=False)
    print("\nInvalid SMILES saved to:")
    print("output/invalid_smiles.csv")
else:
    print("\nAll SMILES are valid.")

print("\nCompleted Successfully.")