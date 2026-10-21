
from pathlib import Path
import pickle
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

BASE = Path(".")
OUT = BASE / "output" / "prospective_pyrethroid"

candidate_path = OUT / "diverse_20_candidates_for_model.csv"
training_path = BASE / "output" / "clean_smiles_dataset.csv"
x_train_path = BASE / "output" / "X_train.csv"
model_path = BASE / "output" / "catboost" / "best_catboost_model.pkl"

candidates = pd.read_csv(candidate_path)
training = pd.read_csv(training_path)
X_train = pd.read_csv(x_train_path)

print("=" * 70)
print("1. INPUT FILES")
print("=" * 70)
print("Candidates:", candidates.shape)
print("Training records:", training.shape)
print("Training feature matrix:", X_train.shape)
print("\nTraining dataset columns:")
print(training.columns.tolist())
print("\nX_train columns (first 25):")
print(X_train.columns.tolist()[:25])

print("\n" + "=" * 70)
print("2. CANDIDATE STRUCTURE AUDIT")
print("=" * 70)

audit = []

for _, row in candidates.iterrows():
    smi = str(row.get("canonical_smiles", ""))
    mol = Chem.MolFromSmiles(smi)

    if mol is None:
        audit.append({
            "shortlist_rank": row.get("shortlist_rank"),
            "CID": row.get("CID"),
            "valid_smiles": False,
            "fragments": None,
            "MW_RDKit": None,
            "heavy_atoms": None,
            "formal_charge": None,
            "formula": None,
        })
        continue

    fragments = Chem.GetMolFrags(mol)
    audit.append({
        "shortlist_rank": row.get("shortlist_rank"),
        "CID": row.get("CID"),
        "valid_smiles": True,
        "fragments": len(fragments),
        "MW_RDKit": round(Descriptors.MolWt(mol), 2),
        "heavy_atoms": mol.GetNumHeavyAtoms(),
        "formal_charge": Chem.GetFormalCharge(mol),
        "formula": rdMolDescriptors.CalcMolFormula(mol),
    })

audit_df = pd.DataFrame(audit)
audit_path = OUT / "shortlist_structure_audit.csv"
audit_df.to_csv(audit_path, index=False)
print(audit_df.to_string(index=False))
print("\nSaved:", audit_path)

print("\n" + "=" * 70)
print("3. TRAINING FEATURE MATRIX")
print("=" * 70)
print("Number of model input columns:", X_train.shape[1])
print("First 20 feature names:")
print(X_train.columns.tolist()[:20])
print("Data types:")
print(X_train.dtypes.value_counts().to_dict())

print("\n" + "=" * 70)
print("4. MODEL INTERFACE")
print("=" * 70)

with open(model_path, "rb") as f:
    model = pickle.load(f)

print("Model class:", type(model).__name__)
print("Model feature count:", getattr(model, "n_features_in_", "not exposed"))

if hasattr(model, "feature_names_in_"):
    model_features = list(model.feature_names_in_)
    print("Model exposes feature names:", len(model_features))
    print("First 25 model feature names:", model_features[:25])
else:
    print("Model does not expose feature_names_in_.")

print("\nAudit complete. No predictions were generated.")
