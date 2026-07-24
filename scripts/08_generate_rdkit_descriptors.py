import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors


print("=" * 60)
print("RDKit MOLECULAR DESCRIPTOR GENERATION")
print("=" * 60)


# Input and output files
input_file = "output/standardized_smiles_dataset.csv"
output_file = "output/rdkit_descriptors_dataset.csv"
failed_file = "output/failed_descriptor_calculation.csv"


# Load dataset
df = pd.read_csv(input_file)

print("\nOriginal compounds :", len(df))


# Identify standardized SMILES column
smiles_col = "Standardized_SMILES"

if smiles_col not in df.columns:
    raise Exception("Standardized_SMILES column not found")


print("SMILES column :", smiles_col)


# Get all RDKit descriptors
descriptor_names = [
    desc[0] for desc in Descriptors.descList
]

print("Number of RDKit descriptors :", len(descriptor_names))


# Descriptor calculator
calculator = MoleculeDescriptors.MolecularDescriptorCalculator(
    descriptor_names
)


descriptor_data = []
failed_data = []


print("\nCalculating descriptors...")


for idx, row in df.iterrows():

    smiles = row[smiles_col]

    try:

        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            failed_data.append(row)
            continue


        descriptors = calculator.CalcDescriptors(mol)

        descriptor_data.append(descriptors)


    except Exception:
        failed_data.append(row)



# Convert descriptors to dataframe

descriptor_df = pd.DataFrame(
    descriptor_data,
    columns=descriptor_names
)


# Keep only successful molecules
successful_df = df.iloc[
    :len(descriptor_df)
].reset_index(drop=True)


# Combine original data + descriptors

final_df = pd.concat(
    [
        successful_df,
        descriptor_df
    ],
    axis=1
)


# Save final descriptor dataset

final_df.to_csv(
    output_file,
    index=False
)


# Save failed molecules

if len(failed_data) > 0:

    failed_df = pd.DataFrame(failed_data)

    failed_df.to_csv(
        failed_file,
        index=False
    )


print("\nDescriptor calculation completed")
print("-" * 40)

print("Successful molecules :", len(descriptor_df))
print("Failed molecules     :", len(failed_data))

print("\nFinal dataset shape:")
print(final_df.shape)

print("\nSaved file:")
print(output_file)

if len(failed_data) > 0:
    print("Failed file:")
    print(failed_file)


print("\nCompleted Successfully")