
from pathlib import Path
import pandas as pd

BASE = Path("data/ecotox_ascii_09_15_2026")
VALIDATION = BASE / "validation"


def read_table(path):
    df = pd.read_csv(
        path,
        sep="|",
        dtype=str,
        encoding="utf-8-sig",
        low_memory=False
    )
    df.columns = df.columns.str.strip()
    return df


print("=" * 70)
print("ECOTOX CODE AND FIELD INSPECTION")
print("=" * 70)

# Inspect official endpoint and effect descriptions
for filename in ["endpoint_codes.txt", "effect_codes.txt"]:
    path = VALIDATION / filename
    print(f"\n--- {filename} ---")

    if path.exists():
        df = read_table(path)
        print(df.to_string(index=False))
    else:
        print("File not found:", path)

# Inspect result field values without loading the full results table
result_file = BASE / "results.txt"
result_fields = [
    "endpoint",
    "effect",
    "measurement",
    "obs_duration_unit",
    "conc1_type",
    "conc1_unit",
]

print("\n--- Common result field values ---")

counts = {field: {} for field in result_fields}

for chunk in pd.read_csv(
    result_file,
    sep="|",
    dtype=str,
    usecols=result_fields,
    chunksize=200_000,
    encoding="utf-8-sig",
    low_memory=False
):
    chunk.columns = chunk.columns.str.strip()

    for field in result_fields:
        values = chunk[field].fillna("").str.strip()
        for value, count in values.value_counts().items():
            counts[field][value] = counts[field].get(value, 0) + int(count)

for field, value_counts in counts.items():
    print(f"\n{field}:")
    for value, count in sorted(
        value_counts.items(),
        key=lambda item: item[1],
        reverse=True
    )[:30]:
        print(f"  {value!r}: {count}")

# Inspect test duration and organism fields
test_file = BASE / "tests.txt"
test_fields = [
    "test_id",
    "test_cas",
    "species_number",
    "organism_habitat",
    "exposure_duration_mean",
    "exposure_duration_unit",
]

print("\n--- Common test field values ---")

test_counts = {
    field: {}
    for field in [
        "organism_habitat",
        "exposure_duration_unit"
    ]
}

for chunk in pd.read_csv(
    test_file,
    sep="|",
    dtype=str,
    usecols=test_fields,
    chunksize=200_000,
    encoding="utf-8-sig",
    low_memory=False
):
    chunk.columns = chunk.columns.str.strip()

    for field in test_counts:
        values = chunk[field].fillna("").str.strip()
        for value, count in values.value_counts().items():
            test_counts[field][value] = (
                test_counts[field].get(value, 0) + int(count)
            )

for field, value_counts in test_counts.items():
    print(f"\n{field}:")
    for value, count in sorted(
        value_counts.items(),
        key=lambda item: item[1],
        reverse=True
    )[:30]:
        print(f"  {value!r}: {count}")

print("\nInspection complete.")
