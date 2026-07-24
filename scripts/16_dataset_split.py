import pandas as pd
import os
from sklearn.model_selection import train_test_split

print("="*70)
print("DATASET SPLIT (TRAIN-TEST)")
print("="*70)


# Input file
input_file = "output/dataset_with_pEC50.csv"


# Output files
train_file = "output/train_dataset.csv"
test_file = "output/test_dataset.csv"


# Load dataset
print("\nLoading dataset...")

df = pd.read_csv(input_file)

print("\nOriginal dataset shape:")
print(df.shape)


# Check target
if "pEC50" not in df.columns:
    raise Exception("pEC50 column not found")


# Split dataset

print("\nPerforming 80:20 split...")


train_df, test_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42
)


# Save

os.makedirs("output", exist_ok=True)


train_df.to_csv(
    train_file,
    index=False
)

test_df.to_csv(
    test_file,
    index=False
)


# Summary

print("\nSplit completed")
print("-----------------------------")

print("Training dataset:")
print(train_df.shape)

print("\nTesting dataset:")
print(test_df.shape)


print("\nSaved files:")
print(train_file)
print(test_file)


print("\nCompleted Successfully")
print("="*70)