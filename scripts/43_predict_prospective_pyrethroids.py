
import os
import warnings
import numpy as np
import pandas as pd
import joblib

from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdFingerprintGenerator, MACCSkeys
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.ML.Descriptors import MoleculeDescriptors

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================
CANDIDATE_FILE = (
    "output/prospective_pyrethroid/"
    "diverse_20_candidates_for_model.csv"
)
FEATURE_FILE = "output/selected_feature_names.csv"
MODEL_FILE = "output/catboost/best_catboost_model.pkl"
TRAIN_FEATURE_FILE = "output/selected_features_train.csv"

OUTPUT_DIR = "output/prospective_pyrethroid"
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR, "prospective_catboost_predictions.csv"
)
FAILED_FILE = os.path.join(
    OUTPUT_DIR, "prospective_prediction_failures.csv"
)

N_BITS = 2048

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("PROSPECTIVE PYRETHROID TOXICITY PREDICTION")
print("Using saved CatBoost model; no retraining")
print("=" * 70)

# ============================================================
# LOAD MODEL AND INPUTS
# ============================================================
candidates = pd.read_csv(CANDIDATE_FILE)
feature_table = pd.read_csv(FEATURE_FILE)
train_df = pd.read_csv(TRAIN_FEATURE_FILE)

selected_features = (
    feature_table["Selected_Features"]
    .dropna()
    .astype(str)
    .tolist()
)

# Preserve the feature order used by the original model.
if len(selected_features) != 100:
    raise ValueError(
        f"Expected 100 selected features; found {len(selected_features)}"
    )

if len(set(selected_features)) != len(selected_features):
    raise ValueError("Duplicate feature names in selected feature list.")

model = joblib.load(MODEL_FILE)

model_features = list(getattr(model, "feature_names_", []))
if model_features and model_features != selected_features:
    raise ValueError(
        "Saved model feature order differs from selected_feature_names.csv."
    )

if "SMILES" in candidates.columns:
    smiles_col = "SMILES"
elif "canonical_smiles" in candidates.columns:
    smiles_col = "canonical_smiles"
else:
    raise ValueError("Candidate file has no SMILES column.")

print("Candidates:", len(candidates))
print("Selected features:", len(selected_features))
print("Model:", type(model).__name__)

# ============================================================
# ORIGINAL TRAINING FEATURE GENERATION
# ============================================================
descriptor_names = [x[0] for x in Descriptors.descList]
descriptor_calculator = MoleculeDescriptors.MolecularDescriptorCalculator(
    descriptor_names
)

fp4_generator = rdFingerprintGenerator.GetMorganGenerator(
    radius=2, fpSize=N_BITS
)
fp6_generator = rdFingerprintGenerator.GetMorganGenerator(
    radius=3, fpSize=N_BITS
)

normalizer = rdMolStandardize.Normalizer()
fragment_chooser = rdMolStandardize.LargestFragmentChooser()
uncharger = rdMolStandardize.Uncharger()


def standardize_molecule(smiles):
    """Apply the same broad standardization steps as the training pipeline."""
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError("SMILES could not be parsed.")

    original_fragments = len(Chem.GetMolFrags(mol))

    mol = normalizer.normalize(mol)
    mol = fragment_chooser.choose(mol)
    mol = uncharger.uncharge(mol)

    standardized_smiles = Chem.MolToSmiles(
        mol, canonical=True
    )

    return mol, standardized_smiles, original_fragments


def calculate_feature_row(mol):
    """Calculate descriptors and fingerprints with training-compatible names."""
    values = {}

    descriptors = descriptor_calculator.CalcDescriptors(mol)
    values.update(dict(zip(descriptor_names, descriptors)))

    fp4 = fp4_generator.GetFingerprint(mol)
    fp6 = fp6_generator.GetFingerprint(mol)

    for bit in range(N_BITS):
        values[f"ECFP4_{bit}"] = int(fp4.GetBit(bit))
        values[f"ECFP6_{bit}"] = int(fp6.GetBit(bit))

    maccs = MACCSkeys.GenMACCSKeys(mol)
    for bit in range(167):
        values[f"MACCS_{bit}"] = int(maccs.GetBit(bit))

    return values, fp6


# ============================================================
# TRAINING-SET FINGERPRINTS FOR SIMILARITY
# ============================================================
train_smiles_col = next(
    (c for c in [
        "Standardized_SMILES",
        "SMILES_Canonical_RDKit",
        "SMILES",
        "canonical_smiles",
    ] if c in train_df.columns),
    None
)

training_fingerprints = []
training_smiles_used = 0

if train_smiles_col is not None:
    for smi in train_df[train_smiles_col].dropna():
        try:
            mol, _, _ = standardize_molecule(smi)
            training_fingerprints.append(
                fp6_generator.GetFingerprint(mol)
            )
            training_smiles_used += 1
        except Exception:
            continue

print("Training structures available for similarity:",
      training_smiles_used)

# ============================================================
# PROCESS CANDIDATES
# ============================================================
prediction_rows = []
failure_rows = []

for row_index, row in candidates.iterrows():
    candidate_id = row.get("CID", row_index + 1)
    original_smiles = row.get(smiles_col, "")

    try:
        mol, standardized_smiles, fragment_count = standardize_molecule(
            original_smiles
        )

        feature_values, candidate_fp = calculate_feature_row(mol)

        missing = [
            feature for feature in selected_features
            if feature not in feature_values
        ]
        if missing:
            raise ValueError(
                "Missing required features: " + ", ".join(missing[:10])
            )

        X_candidate = pd.DataFrame(
            [[feature_values[f] for f in selected_features]],
            columns=selected_features
        )

        X_candidate = X_candidate.apply(
            pd.to_numeric, errors="coerce"
        )

        if not np.isfinite(X_candidate.to_numpy(dtype=float)).all():
            raise ValueError(
                "Selected features contain NaN or infinite values."
            )

        predicted_pEC50 = float(model.predict(X_candidate)[0])

        max_tanimoto = np.nan
        if training_fingerprints:
            similarities = DataStructs.BulkTanimotoSimilarity(
                candidate_fp, training_fingerprints
            )
            max_tanimoto = float(max(similarities))

        result = row.to_dict()
        result.update({
            "original_smiles_used": original_smiles,
            "standardized_smiles_used": standardized_smiles,
            "original_fragment_count": fragment_count,
            "largest_fragment_used_for_prediction": (
                fragment_count > 1
            ),
            "predicted_pEC50": predicted_pEC50,
            "nearest_training_Tanimoto_ECFP6": max_tanimoto,
            "prediction_status": "Prediction generated",
            "structure_review_required": (
                fragment_count > 1
                or pd.isna(max_tanimoto)
                or max_tanimoto < 0.30
            ),
        })
        prediction_rows.append(result)

    except Exception as exc:
        failure_rows.append({
            "candidate_row": row_index,
            "CID": candidate_id,
            "input_smiles": original_smiles,
            "error": str(exc),
        })

# ============================================================
# SAVE RESULTS
# ============================================================
if prediction_rows:
    results = pd.DataFrame(prediction_rows)
    results.to_csv(OUTPUT_FILE, index=False)

    print("\nPredictions generated:", len(results))
    print("Output:", OUTPUT_FILE)

if failure_rows:
    failures = pd.DataFrame(failure_rows)
    failures.to_csv(FAILED_FILE, index=False)

    print("Candidates requiring troubleshooting:", len(failures))
    print("Failure report:", FAILED_FILE)

print("\nImportant:")
print("- Predicted pEC50 values are computational estimates.")
print("- Multi-fragment inputs were reduced to their largest fragment.")
print("- Those structures are flagged for chemical identity review.")
print("- Similarity is a screening indicator, not formal AD validation.")
print("- No model retraining was performed.")
print("=" * 70)
