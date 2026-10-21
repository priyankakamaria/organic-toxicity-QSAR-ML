
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data" / "ecotox_ascii_09_15_2026"

files = [
    DATA / "validation" / "chemicals.txt",
    DATA / "chemical_carriers.txt",
    DATA / "tests.txt",
]

for path in files:
    print("\n" + "=" * 80)
    print("FILE:", path.name)

    # Read only a small sample for initial inspection.
    df = pd.read_csv(
        path,
        sep="|",
        dtype=str,
        nrows=5,
        encoding="utf-8",
        on_bad_lines="skip"
    )

    df.columns = df.columns.str.strip()

    print("Columns:")
    print(df.columns.tolist())

    print("\nSample:")
    print(df.head(3).to_string(index=False))

print("\nDiagnostic complete.")
