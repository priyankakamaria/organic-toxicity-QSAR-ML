import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator


print("="*60)
print("MORGAN FINGERPRINT GENERATION")
print("="*60)


# Input file
input_file = "output/rdkit_descriptors_cleaned_optionA.csv"

# Output file
output_file = "output/morgan_fingerprints_dataset.csv"


# Load dataset

df = pd.read_csv(input_file)

print("\nTotal compounds :", len(df))


# SMILES column

smiles_col = "Standardized_SMILES"


# Fingerprint settings

n_bits = 2048


# Create Morgan fingerprint generators

morgan_ecfp4 = rdFingerprintGenerator.GetMorganGenerator(
    radius=2,
    fpSize=n_bits
)


morgan_ecfp6 = rdFingerprintGenerator.GetMorganGenerator(
    radius=3,
    fpSize=n_bits
)



ecfp4_data = []
ecfp6_data = []

failed = []


print("\nGenerating fingerprints...")


for index, row in df.iterrows():

    smiles = row[smiles_col]

    try:

        mol = Chem.MolFromSmiles(smiles)


        if mol is None:

            failed.append(index)
            continue



        # ECFP4 fingerprint

        fp4 = morgan_ecfp4.GetFingerprint(mol)


        # ECFP6 fingerprint

        fp6 = morgan_ecfp6.GetFingerprint(mol)



        # Convert fingerprint to array

        arr4 = np.zeros(
            (n_bits,),
            dtype=int
        )


        arr6 = np.zeros(
            (n_bits,),
            dtype=int
        )



        for bit in fp4.GetOnBits():
            arr4[bit] = 1


        for bit in fp6.GetOnBits():
            arr6[bit] = 1



        ecfp4_data.append(arr4)

        ecfp6_data.append(arr6)



    except Exception as e:

        failed.append(index)



# Convert to dataframe

ecfp4_df = pd.DataFrame(
    ecfp4_data,
    columns=[
        f"ECFP4_{i}"
        for i in range(n_bits)
    ]
)



ecfp6_df = pd.DataFrame(
    ecfp6_data,
    columns=[
        f"ECFP6_{i}"
        for i in range(n_bits)
    ]
)



# Combine original data + fingerprints

final_df = pd.concat(
    [
        df.reset_index(drop=True),
        ecfp4_df,
        ecfp6_df
    ],
    axis=1
)



# Save

final_df.to_csv(
    output_file,
    index=False
)



print("\nFingerprint generation completed")
print("-"*40)

print("Successful molecules :", len(final_df))

print("Failed molecules     :", len(failed))


print("\nFinal dataset shape:")
print(final_df.shape)


print("\nSaved file:")
print(output_file)


print("\nCompleted Successfully")