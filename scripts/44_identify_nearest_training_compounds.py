
import pandas as pd
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize

CANDIDATE_FILE = (
    "output/prospective_pyrethroid/"
    "prospective_catboost_predictions.csv"
)
TRAIN_FILE = "output/selected_features_train.csv"
OUTPUT_FILE = (
    "output/prospective_pyrethroid/"
    "nearest_training_compound_audit.csv"
)

candidates = pd.read_csv(CANDIDATE_FILE)
train = pd.read_csv(TRAIN_FILE)

candidate_smiles_col = "original_smiles_used"
train_smiles_col = next(
    (c for c in [
        "Standardized_SMILES",
        "SMILES_Canonical_RDKit",
        "SMILES",
        "canonical_smiles",
    ] if c in train.columns),
    None
)

if candidate_smiles_col not in candidates.columns:
    raise ValueError("Candidate SMILES column not found.")

if train_smiles_col is None:
    raise ValueError("Training SMILES column not found.")

normalizer = rdMolStandardize.Normalizer()
chooser = rdMolStandardize.LargestFragmentChooser()
uncharger = rdMolStandardize.Uncharger()

generator = rdFingerprintGenerator.GetMorganGenerator(
    radius=3, fpSize=2048
)

def get_fp(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None

    mol = normalizer.normalize(mol)
    mol = chooser.choose(mol)
    mol = uncharger.uncharge(mol)
    return generator.GetFingerprint(mol)

# Build fingerprint library while retaining source row and identifiers.
train_fps = []
train_rows = []

for idx, row in train.iterrows():
    fp = get_fp(row[train_smiles_col])
    if fp is not None:
        train_fps.append(fp)
        train_rows.append((idx, row))

results = []

for _, candidate in candidates.iterrows():
    cfp = get_fp(candidate[candidate_smiles_col])

    if cfp is None:
        results.append({
            "CID": candidate.get("CID"),
            "error": "Candidate SMILES could not be parsed"
        })
        continue

    sims = np.asarray(
        DataStructs.BulkTanimotoSimilarity(cfp, train_fps)
    )
    order = np.argsort(-sims)[:3]

    for position, i in enumerate(order, start=1):
        train_idx, train_row = train_rows[int(i)]

        results.append({
            "CID": candidate.get("CID"),
            "candidate_pEC50": candidate.get("predicted_pEC50"),
            "candidate_fragment_count":
                candidate.get("original_fragment_count"),
            "nearest_rank": position,
            "training_row_index": train_idx,
            "training_chemical_name":
                train_row.get("chemical_name", ""),
            "training_CAS": train_row.get("CAS", ""),
            "training_organism": train_row.get("organism", ""),
            "training_species_group":
                train_row.get("species_group", ""),
            "training_endpoint": train_row.get("endpoint", ""),
            "training_effect": train_row.get("effect", ""),
            "training_duration": train_row.get("Duration_Value", ""),
            "training_SMILES": train_row.get(train_smiles_col, ""),
            "ECFP6_Tanimoto": float(sims[int(i)]),
        })

pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)
print("Saved:", OUTPUT_FILE)
print("Inspect the nearest three training records for each candidate.")
