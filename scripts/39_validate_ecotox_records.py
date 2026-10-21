
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "prospective_pyrethroid"

files = {
    "EC50": OUT / "ecotox_candidate_EC50_records.csv",
    "LC50": OUT / "ecotox_candidate_LC50_records.csv",
}

for endpoint, path in files.items():
    print("\n" + "=" * 70)
    print(f"{endpoint} RECORD INSPECTION")
    print("=" * 70)

    if not path.exists():
        print(f"Missing file: {path}")
        continue

    df = pd.read_csv(path, dtype=str, low_memory=False)
    print("Rows:", len(df))
    print("\nColumns:")
    for col in df.columns:
        print(" -", col)

    # Inspect columns relevant to endpoint and exposure.
    terms = (
        "species", "common_name", "latin_name",
        "fish_flag", "duration", "endpoint", "effect",
        "measurement", "conc1", "unit", "candidate",
        "cas", "test_id", "reference"
    )

    relevant = [c for c in df.columns if any(t in c.lower() for t in terms)]

    if relevant:
        print("\nSample values:")
        print(df[relevant].head(5).to_string(index=False))

    # Print available duration and endpoint values.
    for col in df.columns:
        if any(t in col.lower() for t in [
            "obs_duration_unit",
            "exposure_duration_unit",
            "endpoint",
            "conc1_unit",
        ]):
            print(f"\nMost frequent values in {col}:")
            print(df[col].value_counts(dropna=False).head(15).to_string())

print("\nInspection complete. No records were modified.")
