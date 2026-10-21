
from pathlib import Path
import pandas as pd
import re

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "prospective_pyrethroid"

CANDIDATE_FILE = OUT / "unique_inchikey_candidates.csv"
EC50_FILE = OUT / "ecotox_candidate_EC50_records.csv"
LC50_FILE = OUT / "ecotox_candidate_LC50_records.csv"

# Fish names/genera used for a conservative first-pass screen.
# This is a screening rule, not definitive taxonomic verification.
FISH_PATTERN = re.compile(
    r"\b(trout|zebrafish|medaka|minnow|carp|guppy|"
    r"bluegill|perch|bass|tilapia|catfish|salmon|"
    r"fathead|killifish|mullet|herring|cod|flounder|"
    r"sole|pike|stickleback|sunfish|goldfish|"
    r"rainbowfish|mosquitofish)\b|"
    r"\b(Oncorhynchus|Danio|Pimephales|Lepomis|"
    r"Cyprinus|Carassius|Oryzias|Poecilia|Gambusia|"
    r"Salmo|Salvelinus|Ictalurus|Micropterus|Perca|"
    r"Oreochromis|Clarias|Lates|Fundulus|Morone|"
    r"Esox|Rutilus|Gasterosteus|Takifugu|"
    r"Dicentrarchus|Sparus|Pangasianodon)\b",
    flags=re.IGNORECASE,
)

# Explicitly exclude common misleading invertebrate names.
INVERTEBRATE_PATTERN = re.compile(
    r"\b(louse|lice|amphipod|water flea|daphnia|"
    r"copepod|shrimp|crab|lobster|worm|mussel|"
    r"oyster|snail|clam|insect|mosquito|fly|thrips)\b",
    flags=re.IGNORECASE,
)


def read_csv(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    df = pd.read_csv(path, dtype=str, low_memory=False)
    df.columns = df.columns.str.strip()
    return df


def find_col(df, options):
    lookup = {c.lower(): c for c in df.columns}
    for option in options:
        if option.lower() in lookup:
            return lookup[option.lower()]
    return None


def duration_is_96h(row):
    """Check observation duration first, then test exposure duration."""
    for value_col, unit_col in [
        ("obs_duration_mean", "obs_duration_unit"),
        ("exposure_duration_mean", "exposure_duration_unit"),
    ]:
        if value_col not in row.index or unit_col not in row.index:
            continue

        value = str(row.get(value_col, "")).strip()
        unit = str(row.get(unit_col, "")).strip().lower()

        try:
            number = float(value)
        except (ValueError, TypeError):
            continue

        if unit in {"h", "hr", "hrs", "hour", "hours"} and number == 96:
            return True

        if unit in {"d", "day", "days"} and number == 4:
            return True

    return False


def fish_status(row):
    common = str(row.get("common_name", "") or "")
    latin = str(row.get("latin_name", "") or "")
    phylum = str(row.get("phylum_division", "") or "")
    group = str(row.get("ecotox_group", "") or "")

    common_latin = f"{common} {latin}".strip()
    all_text = f"{common_latin} {phylum} {group}".lower()

    if INVERTEBRATE_PATTERN.search(common_latin):
        return "exclude_nonfish_likely"

    if FISH_PATTERN.search(common_latin):
        return "fish_candidate_review"

    # Taxonomy/group fields can support review, but are not enough
    # on their own to establish that a species is a fish.
    if re.search(r"\bfish\b|pisces|actinopteryg", all_text):
        return "possible_fish_manual_review"

    return "not_identified_as_fish"


print("=" * 70)
print("CANDIDATE-LEVEL ECOTOX SCREENING")
print("=" * 70)

candidates = read_csv(CANDIDATE_FILE)
candidates = candidates.reset_index(drop=True)

# Preserve the row IDs used in script 38.
# Its candidate_row_id is the zero-based row index in candidates_with_cas.csv.
with_cas_path = OUT / "candidates_with_cas.csv"
with_cas = read_csv(with_cas_path)

if "candidate_row_id" not in candidates.columns:
    candidates["candidate_row_id"] = ""

if "candidate_row_id" not in with_cas.columns:
    with_cas["candidate_row_id"] = with_cas.index.astype(str)

# Map screening row IDs to candidate chemical identity.
identity_cols = [
    c for c in [
        "candidate_row_id", "CID", "SMILES", "canonical_smiles",
        "InChIKey", "IUPACName", "MolecularFormula",
        "MolecularWeight", "CAS_RNs"
    ]
    if c in with_cas.columns
]

candidate_identity = with_cas[identity_cols].copy()
candidate_identity["candidate_row_id"] = (
    candidate_identity["candidate_row_id"].astype(str)
)
candidate_identity = candidate_identity.drop_duplicates(
    subset=["candidate_row_id"]
)

ec50 = read_csv(EC50_FILE)
lc50 = read_csv(LC50_FILE)

for df in (ec50, lc50):
    if "candidate_row_id" in df.columns:
        df["candidate_row_id"] = df["candidate_row_id"].astype(str)

required = {"candidate_row_id", "common_name", "latin_name"}
for label, df in [("EC50", ec50), ("LC50", lc50)]:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{label} file missing columns {sorted(missing)}. "
            f"Found: {df.columns.tolist()}"
        )

# Apply species screen and preserve the decisions for auditing.
for label, df in [("EC50", ec50), ("LC50", lc50)]:
    df["fish_screen_status"] = df.apply(fish_status, axis=1)
    df["is_96h"] = df.apply(duration_is_96h, axis=1)

    df.to_csv(
        OUT / f"ecotox_{label.lower()}_fish_screen_review.csv",
        index=False,
    )

    print(f"\n{label} species-screen counts:")
    print(df["fish_screen_status"].value_counts(dropna=False).to_string())

# Records flagged as likely fish require manual confirmation.
def collect_candidate_flags(df, prefix):
    likely = df[df["fish_screen_status"] == "fish_candidate_review"]
    possible = df[
        df["fish_screen_status"] == "possible_fish_manual_review"
    ]

    all_rows = []
    for _, row in likely.iterrows():
        all_rows.append({
            "candidate_row_id": row["candidate_row_id"],
            f"{prefix}_likely_fish_record": True,
            f"{prefix}_possible_fish_record": True,
            f"{prefix}_96h_likely_fish_record": bool(row["is_96h"]),
        })

    for _, row in possible.iterrows():
        all_rows.append({
            "candidate_row_id": row["candidate_row_id"],
            f"{prefix}_likely_fish_record": False,
            f"{prefix}_possible_fish_record": True,
            f"{prefix}_96h_likely_fish_record": False,
        })

    if not all_rows:
        return pd.DataFrame(columns=[
            "candidate_row_id",
            f"{prefix}_likely_fish_record",
            f"{prefix}_possible_fish_record",
            f"{prefix}_96h_likely_fish_record",
        ])

    temp = pd.DataFrame(all_rows)
    return temp.groupby("candidate_row_id", as_index=False).agg({
        f"{prefix}_likely_fish_record": "max",
        f"{prefix}_possible_fish_record": "max",
        f"{prefix}_96h_likely_fish_record": "max",
    })


ec50_flags = collect_candidate_flags(ec50, "EC50")
lc50_flags = collect_candidate_flags(lc50, "LC50")

screen = candidate_identity.copy()

for flags in [ec50_flags, lc50_flags]:
    screen = screen.merge(flags, on="candidate_row_id", how="left")

flag_cols = [
    "EC50_likely_fish_record",
    "EC50_possible_fish_record",
    "EC50_96h_likely_fish_record",
    "LC50_likely_fish_record",
    "LC50_possible_fish_record",
    "LC50_96h_likely_fish_record",
]

for col in flag_cols:
    if col not in screen.columns:
        screen[col] = False
    screen[col] = screen[col].fillna(False).astype(bool)

screen["likely_fish_EC50_or_LC50_found"] = (
    screen["EC50_likely_fish_record"]
    | screen["LC50_likely_fish_record"]
)

screen["possible_fish_record_needs_review"] = (
    screen["EC50_possible_fish_record"]
    | screen["LC50_possible_fish_record"]
)

screen["screening_status"] = "NO_LIKELY_FISH_MATCH_FOUND_REQUIRES_REVIEW"

screen.loc[
    screen["possible_fish_record_needs_review"],
    "screening_status"
] = "POSSIBLE_FISH_RECORD_MANUAL_REVIEW"

screen.loc[
    screen["likely_fish_EC50_or_LC50_found"],
    "screening_status"
] = "LIKELY_FISH_RECORD_FOUND_EXCLUDE_PENDING_VERIFICATION"

# A candidate with a likely fish record is not eligible for the
# no-experimental-data shortlist until the record is checked.
screen["preliminary_eligible"] = (
    ~screen["likely_fish_EC50_or_LC50_found"]
    & ~screen["possible_fish_record_needs_review"]
)

screen.to_csv(OUT / "candidate_level_ecotox_exclusion_table.csv", index=False)

shortlist = screen[screen["preliminary_eligible"]].copy()
shortlist.to_csv(
    OUT / "preliminary_candidates_no_likely_fish_record.csv",
    index=False,
)

print("\n" + "=" * 70)
print("CANDIDATE-LEVEL SUMMARY")
print("=" * 70)
print("Total candidates in CAS screening list:", len(screen))
print(
    "Candidates with likely fish EC50/LC50 records:",
    int(screen["likely_fish_EC50_or_LC50_found"].sum()),
)
print(
    "Candidates requiring species/manual review:",
    int(screen["possible_fish_record_needs_review"].sum()),
)
print(
    "Preliminary candidates with no likely fish record:",
    len(shortlist),
)

print("\nOutput files:")
print(OUT / "candidate_level_ecotox_exclusion_table.csv")
print(OUT / "preliminary_candidates_no_likely_fish_record.csv")
print(OUT / "ecotox_ec50_fish_screen_review.csv")
print(OUT / "ecotox_lc50_fish_screen_review.csv")

print(
    "\nIMPORTANT: This is a preliminary CAS-matched screen. "
    "It does not prove that experimental data are absent. "
    "Verify candidate identities, species, endpoint, and records "
    "before finalizing a 20-compound prospective set."
)
