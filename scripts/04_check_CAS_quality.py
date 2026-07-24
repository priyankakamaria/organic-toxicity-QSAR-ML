import pandas as pd
import re

print("=" * 60)
print("CAS NUMBER QUALITY CHECK")
print("=" * 60)

# Load dataset
df = pd.read_csv("output/organic_toxicity_median.csv")

print(f"\nTotal records : {len(df):,}")

# Standard CAS format
cas_pattern = r'^\d{2,7}-\d{2}-\d$'

invalid = df[
    ~df["CAS"].astype(str).str.match(cas_pattern, na=False)
]

print(f"\nInvalid CAS numbers : {len(invalid)}")

if len(invalid) > 0:
    print("\nInvalid CAS entries:")
    print(invalid[["CAS", "chemical_name"]])

    invalid.to_csv(
        "output/invalid_CAS_numbers.csv",
        index=False
    )

    print("\nSaved:")
    print("output/invalid_CAS_numbers.csv")

else:
    print("\nAll CAS numbers are valid.")

# Missing CAS
print(f"\nMissing CAS : {df['CAS'].isna().sum()}")

print("\nCompleted Successfully.")