
from pathlib import Path
import pandas as pd
import numpy as np

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.ML.Cluster import Butina

BASE = Path(".")
OUT = BASE / "output" / "prospective_pyrethroid"
OUT.mkdir(parents=True, exist_ok=True)

candidate_file = OUT / "preliminary_candidates_no_likely_fish_record.csv"
train_file = BASE / "output" / "X_train.csv"
clean_file = BASE / "output" / "clean_smiles_dataset.csv"

# Load preliminary candidates
df = pd.read_csv(candidate_file)
df = df.drop_duplicates(subset="InChIKey").copy()
df["canonical_smiles"] = df["canonical_smiles"].fillna("").astype(str)

# Keep valid molecular structures
mols = []
valid_rows = []

for idx, smi in zip(df.index, df["canonical_smiles"]):
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        mols.append(mol)
        valid_rows.append(idx)

df = df.loc[valid_rows].reset_index(drop=True)
mols = mols[:len(df)]

if len(df) != len(mols):
    raise RuntimeError("Candidate molecule alignment failed.")

print("Valid candidates:", len(df))

# Morgan radius 3 fingerprints (ECFP6-style, 2048 bits)
fps = [
    AllChem.GetMorganFingerprintAsBitVect(
        mol, radius=3, nBits=2048
    )
    for mol in mols
]

# Cluster by structural similarity; distance = 1 - Tanimoto
n = len(fps)
distances = []
for i in range(1, n):
    sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
    distances.extend([1.0 - s for s in sims])

clusters = Butina.ClusterData(
    distances, n, 0.35, isDistData=True, reordering=True
)

# Select one representative from each cluster, prioritizing
# candidates with the strongest preliminary eligibility evidence.
cluster_representatives = []

for cluster_id, members in enumerate(clusters):
    members = list(members)

    # Butina's first member is the cluster centre/representative.
    # Use it as the reproducible starting point.
    rep = members[0]
    cluster_representatives.append((cluster_id, rep, len(members)))

cluster_representatives.sort(key=lambda x: (x[2], -x[1]), reverse=True)

# Prefer representatives from distinct clusters, then fill from
# remaining clusters if fewer than 20 clusters are available.
chosen = []
for cluster_id, rep, cluster_size in cluster_representatives:
    chosen.append((rep, cluster_id, cluster_size))
    if len(chosen) >= 20:
        break

# If there are fewer than 20 clusters, add unselected compounds
# by choosing structures maximally dissimilar to the selected set.
if len(chosen) < 20:
    chosen_indices = {x[0] for x in chosen}
    remaining = [i for i in range(n) if i not in chosen_indices]

    while remaining and len(chosen) < 20:
        if not chosen:
            best = remaining[0]
        else:
            scores = []
            for i in remaining:
                max_sim = max(
                    DataStructs.TanimotoSimilarity(fps[i], fps[j])
                    for j, _, _ in chosen
                )
                scores.append((max_sim, i))
            best = min(scores)[1]

        cluster_id = next(
            (cid for cid, members in enumerate(clusters) if best in members),
            -1
        )
        cluster_size = len(clusters[cluster_id]) if cluster_id >= 0 else 1
        chosen.append((best, cluster_id, cluster_size))
        remaining.remove(best)

selected_indices = [x[0] for x in chosen]
shortlist = df.iloc[selected_indices].copy().reset_index(drop=True)
shortlist.insert(0, "shortlist_rank", range(1, len(shortlist) + 1))
shortlist["cluster_id"] = [x[1] for x in chosen]
shortlist["cluster_size"] = [x[2] for x in chosen]

# Check exact overlap against the original training dataset's SMILES,
# where available. This is a separate check from fingerprint similarity.
training_smiles = set()
if clean_file.exists():
    train_df = pd.read_csv(clean_file)
    for col in ["canonical_smiles", "SMILES", "smiles"]:
        if col in train_df.columns:
            for smi in train_df[col].dropna().astype(str):
                mol = Chem.MolFromSmiles(smi)
                if mol is not None:
                    training_smiles.add(Chem.MolToSmiles(mol))

shortlist["exact_training_smiles_overlap"] = (
    shortlist["canonical_smiles"].map(
        lambda s: Chem.MolToSmiles(Chem.MolFromSmiles(s))
        in training_smiles
    ) if training_smiles else False
)

# Save shortlist
out_file = OUT / "diverse_20_candidates_for_model.csv"
shortlist.to_csv(out_file, index=False)

print("\nSelected:", len(shortlist))
print("Exact SMILES overlaps detected:", int(
    shortlist["exact_training_smiles_overlap"].sum()
))
print("Saved:", out_file)
print("\nShortlist:")
print(shortlist[
    ["shortlist_rank", "CID", "IUPACName", "MolecularWeight",
     "cluster_id", "cluster_size", "exact_training_smiles_overlap"]
].to_string(index=False))
