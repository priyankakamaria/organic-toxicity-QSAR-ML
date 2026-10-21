
import pandas as pd
from pathlib import Path

input_file = Path(
    "output/prospective_pyrethroid/unique_inchikey_candidates.csv"
)

output_dir = Path("output/prospective_pyrethroid")
output_file = output_dir / "toxicity_screening_list.csv"

df = pd.read_csv(input_file)

required = ["CID", "InChIKey", "IUPACName", "SMILES"]
missing = [col for col in required if col not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

screen = df[required].copy()

# Add fields to record the evidence gathered during screening
screen["PubChem_fish_toxicity_checked"] = "Not checked"
screen["EPA_ECOTOX_checked"] = "Not checked"
screen["Literature_checked"] = "Not checked"
screen["Experimental_fish_EC50_record_found"] = "Not assessed"
screen["Screening_notes"] = ""
screen["Screening_date"] = ""

screen.to_csv(output_file, index=False)

print("=" * 60)
print("TOXICITY SCREENING LIST CREATED")
print("=" * 60)
print("Compounds:", len(screen))
print("Output:", output_file.resolve())
print("Next: screen each compound for experimental fish-toxicity data.")
print("=" * 60)
