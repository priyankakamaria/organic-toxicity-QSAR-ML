
import pandas as pd
from pathlib import Path
from rdkit import Chem

# Input files
candidate_file = Path(
    "output/prospective_pyrethroid/pubchem_candidates.csv"
)
training_file = Path(
    "output/clean_smiles_dataset.csv"
)

# Output files
output_dir = Path("output/prospective_pyrethroid")
output_dir.mkdir(parents=True, exist_ok=True)

clean_file = output_dir / "candidates_not_in_training.csv"
invalid_file = output_dir / "invalid_candidate_smiles.csv"


def canonicalize(smiles):
    if pd.isna(smiles):
        return None

    mol = Chem.MolFromSmiles(str(smiles))

    if mol is None:
        return None

    return Chem.MolToSmiles(
        mol,
        canonical=True,
        isomericSmiles=True
    )


def find_smiles_column(df):
    possible = [
        "ConnectivitySMILES",
        "SMILES",
        "CanonicalSMILES",
        "smiles",
        "canonical_smiles"
    ]

    for column in possible:
        if column in df.columns:
            return column

    raise ValueError(
        f"No SMILES column found. Available columns: {df.columns.tolist()}"
    )


def main():
    candidates = pd.read_csv(candidate_file)
    training = pd.read_csv(training_file)

    candidate_smiles_col = find_smiles_column(candidates)
    training_smiles_col = find_smiles_column(training)

    candidates["canonical_smiles"] = (
        candidates[candidate_smiles_col].apply(canonicalize)
    )

    training["canonical_smiles"] = (
        training[training_smiles_col].apply(canonicalize)
    )

    # Keep invalid structures separately for inspection
    invalid = candidates[
        candidates["canonical_smiles"].isna()
    ].copy()

    invalid.to_csv(invalid_file, index=False)

    valid = candidates[
        candidates["canonical_smiles"].notna()
    ].copy()

    # Remove duplicate candidate structures
    valid = valid.drop_duplicates(
        subset=["canonical_smiles"]
    )

    # Compare canonical structures against training compounds
    training_smiles = set(
        training["canonical_smiles"].dropna()
    )

    valid["in_training_set"] = (
        valid["canonical_smiles"].isin(training_smiles)
    )

    overlap_count = int(valid["in_training_set"].sum())

    new_candidates = valid[
        ~valid["in_training_set"]
    ].copy()

    new_candidates.to_csv(clean_file, index=False)

    print("=" * 60)
    print("TRAINING-OVERLAP CHECK")
    print("=" * 60)
    print("Original PubChem candidates:", len(candidates))
    print("Valid candidate structures:", len(valid) + len(invalid))
    print("Invalid candidate structures:", len(invalid))
    print("Unique valid structures:", len(valid))
    print("Exact canonical-SMILES overlap:", overlap_count)
    print("Candidates not in training set:", len(new_candidates))
    print("Saved new candidates to:", clean_file.resolve())
    print("Saved invalid structures to:", invalid_file.resolve())


if __name__ == "__main__":
    main()
