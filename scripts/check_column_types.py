import pandas as pd

df = pd.read_csv("output/combined_no_constant_features.csv")

print("=" * 60)
print("NON-NUMERIC COLUMNS")
print("=" * 60)

object_cols = df.select_dtypes(include=["object"]).columns

for col in object_cols:
    print(col)

print("\nTotal non-numeric columns:", len(object_cols))