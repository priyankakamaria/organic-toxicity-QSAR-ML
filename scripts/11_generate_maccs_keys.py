import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import MACCSkeys


print("="*60)
print("MACCS KEY GENERATION")
print("="*60)


# Input file

input_file = "output/morgan_fingerprints_dataset.csv"


# Output file

output_file = "output/combined_rdkit_morgan_maccs_dataset.csv"



# Load dataset

df = pd.read_csv(input_file)


print("\nTotal compounds :", len(df))


smiles_col = "Standardized_SMILES"



maccs_data = []

failed = []



print("\nGenerating MACCS keys...")



for index, row in df.iterrows():

    smiles = row[smiles_col]


    try:

        mol = Chem.MolFromSmiles(smiles)


        if mol is None:

            failed.append(index)
            continue



        fp = MACCSkeys.GenMACCSKeys(mol)


        arr = np.zeros(
            (167,),
            dtype=int
        )


        for bit in fp.GetOnBits():

            arr[bit] = 1



        maccs_data.append(arr)



    except Exception:

        failed.append(index)




# Convert to dataframe

maccs_df = pd.DataFrame(

    maccs_data,

    columns=[
        f"MACCS_{i}"
        for i in range(167)
    ]

)



# Combine

final_df = pd.concat(

    [
        df.reset_index(drop=True),
        maccs_df

    ],

    axis=1

)



# Save

final_df.to_csv(

    output_file,

    index=False

)



print("\nMACCS generation completed")
print("-"*40)


print("Successful molecules :", len(final_df))

print("Failed molecules     :", len(failed))


print("\nFinal dataset shape:")

print(final_df.shape)



print("\nSaved file:")

print(output_file)


print("\nCompleted Successfully")