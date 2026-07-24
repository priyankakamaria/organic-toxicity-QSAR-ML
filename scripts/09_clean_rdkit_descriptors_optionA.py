import pandas as pd
import numpy as np


print("="*60)
print("RDKit DESCRIPTOR CLEANING - OPTION A")
print("="*60)


input_file = "output/rdkit_descriptors_dataset.csv"

output_file = "output/rdkit_descriptors_cleaned_optionA.csv"


# Load data

df = pd.read_csv(input_file)


print("\nOriginal dataset:")
print(df.shape)



# Descriptor columns

descriptor_start = list(df.columns).index("Standardized_SMILES") + 1

descriptor_cols = list(df.columns[descriptor_start:])


print("\nOriginal descriptors:", len(descriptor_cols))



# Descriptors with missing/inf values

remove_descriptors = [

    "MaxPartialCharge",
    "MinPartialCharge",
    "MaxAbsPartialCharge",
    "MinAbsPartialCharge",

    "BCUT2D_MWHI",
    "BCUT2D_MWLOW",
    "BCUT2D_CHGHI",
    "BCUT2D_CHGLO",
    "BCUT2D_LOGPHI",
    "BCUT2D_LOGPLOW",
    "BCUT2D_MRHI",
    "BCUT2D_MRLOW"

]


# Remove if present

remove_descriptors = [
    x for x in remove_descriptors
    if x in df.columns
]


df = df.drop(
    columns=remove_descriptors
)



print("\nRemoved problematic descriptors:")
for x in remove_descriptors:
    print(x)



# Remove constant descriptors

descriptor_cols = [
    x for x in descriptor_cols
    if x in df.columns
]


constant_cols = []


for col in descriptor_cols:

    if df[col].nunique() <= 1:
        constant_cols.append(col)



df = df.drop(
    columns=constant_cols
)


print("\nConstant descriptors removed:")
for x in constant_cols:
    print(x)



# Replace infinite values if any

df = df.replace(
    [np.inf, -np.inf],
    np.nan
)



# Fill remaining missing descriptor values with median

descriptor_cols = [
    x for x in descriptor_cols
    if x in df.columns
]


df[descriptor_cols] = df[descriptor_cols].fillna(
    df[descriptor_cols].median()
)



# Save

df.to_csv(
    output_file,
    index=False
)


print("\nFinal dataset:")
print(df.shape)


print("\nSaved:")
print(output_file)


print("\nCompleted Successfully")