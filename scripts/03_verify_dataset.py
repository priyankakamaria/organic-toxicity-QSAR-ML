import pandas as pd

# Load median dataset
df = pd.read_csv("output/organic_toxicity_median.csv")

print("=" * 60)
print("DATASET VERIFICATION")
print("=" * 60)

print(f"\nUnique Canonical SMILES : {df['SMILES_Canonical_RDKit'].nunique()}")
print(f"Unique CAS : {df['CAS'].nunique()}")

# SMILES with multiple CAS
smiles_cas = df.groupby("SMILES_Canonical_RDKit")["CAS"].nunique()
multiple_cas = smiles_cas[smiles_cas > 1]

print(f"\nSMILES having multiple CAS numbers : {len(multiple_cas)}")

# Save detailed report
problem_smiles = multiple_cas.index.tolist()

details = df[df["SMILES_Canonical_RDKit"].isin(problem_smiles)]
details = details.sort_values(["SMILES_Canonical_RDKit", "CAS"])

details.to_csv("output/multiple_CAS_same_SMILES.csv", index=False)

print("\nDetails saved to:")
print("output/multiple_CAS_same_SMILES.csv")

# Duplicate check
duplicates = df.duplicated(
    subset=["SMILES_Canonical_RDKit", "organism"],
    keep=False
)

print(f"\nDuplicate compound-organism pairs : {duplicates.sum()}")

print("\nVerification completed.")