
import pandas as pd
from pathlib import Path
from rdkit import Chem

file = Path(
    "output/prospective_pyrethroid/candidates_not_in_training.csv"
)

df = pd.read_csv(file)

print("=" * 60)
print("CANDIDATE QUALITY CHECK")
print("=" * 60)
print("Rows:", len(df))
print("Columns:", df.columns.tolist())

smiles_col = next(
    (
        c for c in
        ["ConnectivitySMILES", "SMILES", "CanonicalSMILES"]
        if c in df.columns
    ),
    None
)

if smiles_col is None:
    raise ValueError("No SMILES column found.")

valid_count = 0
invalid_count = 0
missing_count = 0
motif_count = 0

motif = Chem.MolFromSmarts(
    "[C](=O)[O][C;X4]([C]#N)[c]1cccc([O][c]2ccccc2)c1"
)

for smiles in df[smiles_col]:
    if pd.isna(smiles) or not str(smiles).strip():
        missing_count += 1
        continue

    mol = Chem.MolFromSmiles(str(smiles))

    if mol is None:
        invalid_count += 1
        continue

    valid_count += 1

    if motif is not None and mol.HasSubstructMatch(motif):
        motif_count += 1

print("Valid SMILES:", valid_count)
print("Invalid SMILES:", invalid_count)
print("Missing SMILES:", missing_count)
print("Structures matching motif:", motif_count)

for col in ["CID", "InChIKey", "IUPACName"]:
    if col in df.columns:
        print(f"\nFirst 5 values in {col}:")
        print(df[col].head().to_string(index=False))

print("=" * 60)
