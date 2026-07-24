import pandas as pd
import numpy as np


print("="*60)
print("RDKit DESCRIPTOR QUALITY CHECK")
print("="*60)


# Input file
input_file = "output/rdkit_descriptors_dataset.csv"

# Output file
output_file = "output/rdkit_descriptors_cleaned.csv"


# Load dataset
df = pd.read_csv(input_file)


print("\nDataset shape:")
print(df.shape)


# Identify descriptor columns
# RDKit descriptors start after Standardized_SMILES column

descriptor_start = list(df.columns).index("Standardized_SMILES") + 1

descriptor_cols = df.columns[descriptor_start:]


print("\nNumber of descriptor columns:")
print(len(descriptor_cols))


# -------------------------------
# QC 1: Missing values
# -------------------------------

print("\n1. Missing Value Check")
print("-"*40)

missing = df[descriptor_cols].isnull().sum()

missing_values = missing[missing > 0]


if len(missing_values) == 0:
    print("No missing values found")
else:
    print(missing_values)



# -------------------------------
# QC 2: Infinite values
# -------------------------------

print("\n2. Infinite Value Check")
print("-"*40)


inf_check = np.isinf(
    df[descriptor_cols]
)

inf_count = inf_check.sum().sum()


print("Infinite values:", inf_count)



# Replace infinite values

df[descriptor_cols] = df[descriptor_cols].replace(
    [np.inf, -np.inf],
    np.nan
)


# Remove rows with missing descriptor values

before_rows = len(df)

df = df.dropna(
    subset=descriptor_cols
)

after_rows = len(df)


print("Rows removed due to missing/inf:",
      before_rows - after_rows)



# -------------------------------
# QC 3: Constant descriptors
# -------------------------------

print("\n3. Constant Descriptor Check")
print("-"*40)


constant_cols = []


for col in descriptor_cols:

    if df[col].nunique() <= 1:
        constant_cols.append(col)


print("Constant descriptors:",
      len(constant_cols))


if len(constant_cols) > 0:

    print("\nRemoved descriptors:")
    for col in constant_cols:
        print(col)


    df = df.drop(
        columns=constant_cols
    )


# Update descriptor list

descriptor_cols = [
    col for col in df.columns
    if col in descriptor_cols
    and col not in constant_cols
]



# -------------------------------
# QC 4: Duplicate descriptors
# -------------------------------

print("\n4. Duplicate Descriptor Check")
print("-"*40)


duplicate_cols = []

for i in range(len(descriptor_cols)):

    for j in range(i):

        if df[descriptor_cols[i]].equals(
            df[descriptor_cols[j]]
        ):

            duplicate_cols.append(
                descriptor_cols[i]
            )

            break



print("Duplicate descriptors:",
      len(duplicate_cols))


if len(duplicate_cols) > 0:

    print("\nRemoved duplicate descriptors:")

    for col in duplicate_cols:
        print(col)


    df = df.drop(
        columns=duplicate_cols
    )



# -------------------------------
# Save cleaned dataset
# -------------------------------

df.to_csv(
    output_file,
    index=False
)


print("\nFinal cleaned dataset shape:")
print(df.shape)


print("\nSaved file:")
print(output_file)


print("\nQC Completed Successfully")