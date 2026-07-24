import pandas as pd
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize


print("="*60)
print("SMILES STANDARDIZATION")
print("="*60)


# Input file
input_file = "output/clean_smiles_dataset.csv"

# Output files
output_file = "output/standardized_smiles_dataset.csv"
failed_file = "output/failed_standardization.csv"


# Load dataset
df = pd.read_csv(input_file)

print("\nOriginal compounds :", len(df))


# Identify SMILES column
smiles_col = None

for col in df.columns:
    if "smiles" in col.lower():
        smiles_col = col
        break

if smiles_col is None:
    raise Exception("SMILES column not found")

print("SMILES column :", smiles_col)


# Standardizers
uncharger = rdMolStandardize.Uncharger()
normalizer = rdMolStandardize.Normalizer()
fragment_chooser = rdMolStandardize.LargestFragmentChooser()


standardized_smiles = []
failed_records = []


for index, row in df.iterrows():

    smi = str(row[smiles_col])

    try:

        # Convert SMILES to molecule
        mol = Chem.MolFromSmiles(smi)

        if mol is None:
            failed_records.append(row)
            continue


        # Normalize molecule
        mol = normalizer.normalize(mol)


        # Remove salts and keep largest fragment
        mol = fragment_chooser.choose(mol)


        # Neutralize charges
        mol = uncharger.uncharge(mol)


        # Generate canonical SMILES
        new_smiles = Chem.MolToSmiles(
            mol,
            canonical=True
        )


        standardized_smiles.append(new_smiles)


    except Exception:
        failed_records.append(row)



# Add standardized SMILES

df_standard = df.loc[
    df.index[:len(standardized_smiles)]
].copy()

df_standard["Standardized_SMILES"] = standardized_smiles


# Save standardized dataset

df_standard.to_csv(
    output_file,
    index=False
)


# Save failed molecules

if len(failed_records) > 0:
    
    failed_df = pd.DataFrame(failed_records)

    failed_df.to_csv(
        failed_file,
        index=False
    )


print("\nStandardization completed")
print("--------------------------------")
print("Standardized compounds :", len(df_standard))
print("Failed compounds       :", len(failed_records))


print("\nSaved files:")
print(output_file)

if len(failed_records)>0:
    print(failed_file)

print("\nCompleted Successfully")