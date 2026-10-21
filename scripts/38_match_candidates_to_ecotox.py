

from pathlib import Path
import re
import sys
import pandas as pd

# ============================================================
# 1. PATHS AND SETTINGS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "ecotox_ascii_09_15_2026"
VALIDATION_DIR = DATA_DIR / "validation"

CANDIDATE_FILE = (
    ROOT / "output" / "prospective_pyrethroid"
    / "candidates_with_cas.csv"
)

OUTPUT_DIR = ROOT / "output" / "prospective_pyrethroid"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHEMICALS_FILE = VALIDATION_DIR / "chemicals.txt"
CARRIERS_FILE = DATA_DIR / "chemical_carriers.txt"
TESTS_FILE = DATA_DIR / "tests.txt"
RESULTS_FILE = DATA_DIR / "results.txt"
SPECIES_FILE = VALIDATION_DIR / "species.txt"

CHUNK_SIZE = 100_000

# ============================================================
# 2. GENERAL HELPERS
# ============================================================

def clean_column_name(value):
    """Remove BOM and surrounding spaces from a column name."""
    return str(value).replace("\ufeff", "").strip()


def normalize_cas(value):
    """Normalize a CAS number for matching, e.g. 123-45-6 -> 123456."""
    if pd.isna(value):
        return ""

    value = str(value).strip()

    if not value or value.lower() in {"nan", "none", "null", "na"}:
        return ""

    digits = re.sub(r"\D", "", value)

    # Avoid treating arbitrary short numeric values as CAS numbers.
    if len(digits) < 5:
        return ""

    return digits


def normalize_columns(df):
    df.columns = [clean_column_name(c) for c in df.columns]
    return df


def read_header(path):
    """Read only a pipe-delimited file's header."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    header = pd.read_csv(
        path,
        sep="|",
        nrows=0,
        dtype=str,
        encoding="utf-8-sig",
        skipinitialspace=True,
    )

    return [clean_column_name(c) for c in header.columns]


def find_column(columns, alternatives):
    """Find a column by exact case-insensitive name."""
    lookup = {str(c).strip().lower(): c for c in columns}

    for name in alternatives:
        if name.lower() in lookup:
            return lookup[name.lower()]

    return None


def read_pipe_table(path, usecols=None):
    df = pd.read_csv(
        path,
        sep="|",
        dtype=str,
        encoding="utf-8-sig",
        skipinitialspace=True,
        usecols=usecols,
        on_bad_lines="skip",
        low_memory=False,
    )
    return normalize_columns(df)


def split_cas_values(value):
    """Return normalized CAS keys from a possibly multi-value field."""
    if pd.isna(value):
        return []

    text = str(value).strip()

    if not text or text.lower() in {"nan", "none", "null", "na"}:
        return []

    # Candidate CAS fields may contain semicolon- or comma-separated IDs.
    parts = re.split(r"[;,]", text)

    normalized = []
    for part in parts:
        cas_key = normalize_cas(part)
        if cas_key:
            normalized.append(cas_key)

    return sorted(set(normalized))


def save_csv(df, filename):
    path = OUTPUT_DIR / filename
    df.to_csv(path, index=False)
    print(f"Saved: {path} ({len(df):,} rows)")


# ============================================================
# 3. CHECK INPUT FILES
# ============================================================

for path in [
    CANDIDATE_FILE,
    CHEMICALS_FILE,
    CARRIERS_FILE,
    TESTS_FILE,
    RESULTS_FILE,
    SPECIES_FILE,
]:
    if not path.exists():
        sys.exit(f"ERROR: Required file not found: {path}")

print("=" * 70)
print("ECOTOX PROSPECTIVE CANDIDATE MATCHING")
print("=" * 70)

# ============================================================
# 4. LOAD CANDIDATES AND IDENTIFY CAS COLUMN
# ============================================================

candidates = pd.read_csv(
    CANDIDATE_FILE,
    dtype=str,
    encoding="utf-8-sig",
    low_memory=False,
)

candidates.columns = [clean_column_name(c) for c in candidates.columns]

print(f"\nCandidates loaded: {len(candidates):,}")
print("Candidate columns:", candidates.columns.tolist())

cas_col = find_column(
    candidates.columns,
    [
        "cas_number",
        "cas_numbers",
        "cas",
        "cas_rn",
        "cas_rns",
        "cas_registry_number",
        "registry_number",
        "pubchem_cas",
    ],
)

if cas_col is None:
    # Fall back to a column whose name contains "cas".
    possible = [c for c in candidates.columns if "cas" in c.lower()]
    if len(possible) == 1:
        cas_col = possible[0]
    else:
        sys.exit(
            "ERROR: Could not identify the candidate CAS column.\n"
            f"Columns found: {candidates.columns.tolist()}\n"
            f"Possible CAS columns: {possible}\n"
            "Inspect candidates_with_cas.csv and set cas_col manually."
        )

print(f"Candidate CAS column: {cas_col}")

# Preserve original candidate row number for traceability.
candidates = candidates.reset_index(drop=True)
candidates["candidate_row_id"] = candidates.index.astype(str)
candidates["cas_match_key"] = candidates[cas_col].apply(split_cas_values)

candidate_cas_rows = []

for _, row in candidates.iterrows():
    for cas_key in row["cas_match_key"]:
        candidate_cas_rows.append(
            {
                "candidate_row_id": row["candidate_row_id"],
                "candidate_cas_raw": row[cas_col],
                "cas_key": cas_key,
            }
        )

candidate_cas_map = pd.DataFrame(
    candidate_cas_rows,
    columns=["candidate_row_id", "candidate_cas_raw", "cas_key"],
).drop_duplicates()

print(
    "Unique normalized candidate CAS identifiers:",
    candidate_cas_map["cas_key"].nunique()
    if not candidate_cas_map.empty else 0,
)

# ============================================================
# 5. MATCH AGAINST ECOTOX CHEMICAL VALIDATION TABLE
# ============================================================

chemical_columns = read_header(CHEMICALS_FILE)
chemical_cas_col = find_column(
    chemical_columns,
    ["cas_number", "cas", "cas_rn"],
)

if chemical_cas_col is None:
    sys.exit(
        "ERROR: CAS column not found in validation/chemicals.txt.\n"
        f"Columns found: {chemical_columns}"
    )

chemicals = read_pipe_table(
    CHEMICALS_FILE,
    usecols=[
        c for c in [
            chemical_cas_col,
            find_column(chemical_columns, ["chemical_name"]),
            find_column(chemical_columns, ["ecotox_group"]),
            find_column(chemical_columns, ["dtxsid"]),
        ]
        if c is not None
    ],
)

chemicals["cas_key"] = chemicals[chemical_cas_col].apply(normalize_cas)

chemical_matches = candidate_cas_map.merge(
    chemicals,
    on="cas_key",
    how="inner",
)

save_csv(chemical_matches, "ecotox_cas_chemical_matches.csv")

print(
    "ECOTOX chemical-table CAS matches:",
    chemical_matches["candidate_row_id"].nunique()
    if not chemical_matches.empty else 0,
)

# ============================================================
# 6. MATCH DIRECT TEST_CAS IDENTIFIERS
# ============================================================

test_columns = read_header(TESTS_FILE)

test_id_col = find_column(test_columns, ["test_id"])
test_cas_col = find_column(test_columns, ["test_cas"])
species_number_col = find_column(test_columns, ["species_number"])

if test_id_col is None or test_cas_col is None:
    sys.exit(
        "ERROR: Required test_id or test_cas column missing in tests.txt.\n"
        f"Columns found: {test_columns}"
    )

test_metadata_cols = [
    c for c in [
        test_id_col,
        test_cas_col,
        species_number_col,
        find_column(test_columns, ["exposure_duration_mean"]),
        find_column(test_columns, ["exposure_duration_unit"]),
        find_column(test_columns, ["organism_habitat"]),
        find_column(test_columns, ["reference_number"]),
    ]
    if c is not None
]

direct_test_matches = []

for chunk in pd.read_csv(
    TESTS_FILE,
    sep="|",
    dtype=str,
    usecols=lambda c: clean_column_name(c) in test_metadata_cols,
    chunksize=CHUNK_SIZE,
    encoding="utf-8-sig",
    skipinitialspace=True,
    on_bad_lines="skip",
    low_memory=False,
):
    chunk = normalize_columns(chunk)

    chunk["cas_key"] = chunk[test_cas_col].apply(normalize_cas)

    matched = chunk.merge(
        candidate_cas_map,
        on="cas_key",
        how="inner",
    )

    if not matched.empty:
        direct_test_matches.append(matched)

if direct_test_matches:
    direct_tests = pd.concat(direct_test_matches, ignore_index=True)
    direct_tests = direct_tests.drop_duplicates(
        subset=["candidate_row_id", test_id_col, "cas_key"]
    )
else:
    direct_tests = pd.DataFrame()

save_csv(direct_tests, "ecotox_candidate_direct_test_matches.csv")

print(
    "Direct test_cas matches:",
    direct_tests["candidate_row_id"].nunique()
    if not direct_tests.empty else 0,
)

# ============================================================
# 7. MATCH CHEMICAL CARRIERS
# ============================================================

carrier_columns = read_header(CARRIERS_FILE)

carrier_id_col = find_column(carrier_columns, ["carrier_id"])
carrier_test_id_col = find_column(carrier_columns, ["test_id"])
carrier_cas_col = find_column(carrier_columns, ["cas_number"])
carrier_name_col = find_column(carrier_columns, ["chem_name"])
carrier_purpose_col = find_column(carrier_columns, ["purpose"])

if carrier_test_id_col is None or carrier_cas_col is None:
    sys.exit(
        "ERROR: Required test_id or cas_number column missing "
        "in chemical_carriers.txt.\n"
        f"Columns found: {carrier_columns}"
    )

carrier_usecols = [
    c for c in [
        carrier_id_col,
        carrier_test_id_col,
        carrier_cas_col,
        carrier_name_col,
        carrier_purpose_col,
    ]
    if c is not None
]

carrier_matches_list = []

print("\nScanning chemical_carriers.txt in chunks...")

for chunk in pd.read_csv(
    CARRIERS_FILE,
    sep="|",
    dtype=str,
    usecols=lambda c: clean_column_name(c) in carrier_usecols,
    chunksize=CHUNK_SIZE,
    encoding="utf-8-sig",
    skipinitialspace=True,
    on_bad_lines="skip",
    low_memory=False,
):
    chunk = normalize_columns(chunk)

    chunk["cas_key"] = chunk[carrier_cas_col].apply(normalize_cas)

    matched = chunk.merge(
        candidate_cas_map,
        on="cas_key",
        how="inner",
    )

    if not matched.empty:
        carrier_matches_list.append(matched)

if carrier_matches_list:
    carrier_matches = pd.concat(carrier_matches_list, ignore_index=True)
    carrier_matches = carrier_matches.drop_duplicates(
        subset=["candidate_row_id", carrier_test_id_col, "cas_key"]
    )
else:
    carrier_matches = pd.DataFrame(
        columns=carrier_usecols
        + ["cas_key", "candidate_row_id", "candidate_cas_raw"]
    )

save_csv(carrier_matches, "ecotox_candidate_carrier_matches.csv")

print(
    "Carrier matches:",
    carrier_matches["candidate_row_id"].nunique()
    if not carrier_matches.empty else 0,
)

# ============================================================
# 8. COMBINE TEST IDs FROM DIRECT AND CARRIER MATCHES
# ============================================================

test_id_values = []

if not direct_tests.empty:
    test_id_values.extend(
        direct_tests[test_id_col].dropna().astype(str).tolist()
    )

if not carrier_matches.empty:
    test_id_values.extend(
        carrier_matches[carrier_test_id_col].dropna().astype(str).tolist()
    )

matched_test_ids = sorted(
    set(x.strip() for x in test_id_values if str(x).strip())
)

print("Unique matched ECOTOX test IDs:", len(matched_test_ids))

if not matched_test_ids:
    print(
        "\nNo ECOTOX tests were matched by normalized CAS identifiers.\n"
        "This does not prove that experimental data are absent. "
        "Name-based checks may still be needed."
    )

# Build candidate-to-test relationships from both match routes.
relationships = []

if not direct_tests.empty:
    for _, row in direct_tests.iterrows():
        relationships.append(
            {
                "candidate_row_id": row["candidate_row_id"],
                "test_id": str(row[test_id_col]),
                "match_route": "tests.test_cas",
                "matched_cas_key": row["cas_key"],
            }
        )

if not carrier_matches.empty:
    for _, row in carrier_matches.iterrows():
        relationships.append(
            {
                "candidate_row_id": row["candidate_row_id"],
                "test_id": str(row[carrier_test_id_col]),
                "match_route": "chemical_carriers.cas_number",
                "matched_cas_key": row["cas_key"],
            }
        )

relationships_df = pd.DataFrame(
    relationships,
    columns=[
        "candidate_row_id",
        "test_id",
        "match_route",
        "matched_cas_key",
    ],
).drop_duplicates()

save_csv(relationships_df, "ecotox_candidate_test_relationships.csv")

# ============================================================
# 9. LOAD SPECIES TABLE AND FLAG POSSIBLE FISH
# ============================================================

species_columns = read_header(SPECIES_FILE)

species_number_key = find_column(species_columns, ["species_number"])

if species_number_key is None:
    sys.exit(
        "ERROR: species_number column not found in species.txt.\n"
        f"Columns found: {species_columns}"
    )

species_keep = [
    c for c in [
        species_number_key,
        find_column(species_columns, ["common_name"]),
        find_column(species_columns, ["latin_name"]),
        find_column(species_columns, ["kingdom"]),
        find_column(species_columns, ["phylum_division"]),
        find_column(species_columns, ["ecotox_group"]),
        find_column(species_columns, ["ncbi_taxid"]),
    ]
    if c is not None
]

species = read_pipe_table(SPECIES_FILE, usecols=species_keep)

# Heuristic flag only: taxonomy/group labels are not a definitive
# taxonomic verification of every species.
species_text_cols = [
    c for c in species.columns
    if c != species_number_key
]

species_text = (
    species[species_text_cols]
    .fillna("")
    .astype(str)
    .agg(" ".join, axis=1)
    .str.lower()
)

fish_pattern = (
    r"\bfish\b|teleost|actinopteryg|"
    r"chondrichth|osteichth|"
    r"\bpisces\b|salmon|trout|carp|"
    r"minnow|guppy|medaka|zebrafish|"
    r"fathead|bluegill|perch|bass|"
    r"tilapia|catfish|danio rerio"
)

species["fish_flag_heuristic"] = species_text.str.contains(
    fish_pattern,
    regex=True,
    na=False,
)

save_csv(species, "ecotox_fish_species_lookup.csv")

print(
    "Species flagged as possible fish:",
    int(species["fish_flag_heuristic"].sum()),
)

# ============================================================
# 10. READ TEST METADATA FOR MATCHED TEST IDs
# ============================================================

if matched_test_ids:
    matched_test_set = set(matched_test_ids)
    test_metadata_parts = []

    for chunk in pd.read_csv(
        TESTS_FILE,
        sep="|",
        dtype=str,
        usecols=lambda c: clean_column_name(c) in test_metadata_cols,
        chunksize=CHUNK_SIZE,
        encoding="utf-8-sig",
        skipinitialspace=True,
        on_bad_lines="skip",
        low_memory=False,
    ):
        chunk = normalize_columns(chunk)

        selected = chunk[
            chunk[test_id_col].astype(str).isin(matched_test_set)
        ]

        if not selected.empty:
            test_metadata_parts.append(selected)

    if test_metadata_parts:
        matched_test_metadata = pd.concat(
            test_metadata_parts, ignore_index=True
        ).drop_duplicates(subset=[test_id_col])
    else:
        matched_test_metadata = pd.DataFrame(columns=test_metadata_cols)
else:
    matched_test_metadata = pd.DataFrame(columns=test_metadata_cols)

if not matched_test_metadata.empty and species_number_col:
    matched_test_metadata = matched_test_metadata.merge(
        species,
        left_on=species_number_col,
        right_on=species_number_key,
        how="left",
        suffixes=("", "_species"),
    )

save_csv(
    matched_test_metadata,
    "ecotox_candidate_test_matches.csv",
)

# ============================================================
# 11. EXTRACT RESULTS FOR MATCHED TESTS
# ============================================================

result_columns = read_header(RESULTS_FILE)

result_test_id_col = find_column(result_columns, ["test_id"])
result_id_col = find_column(result_columns, ["result_id"])
endpoint_col = find_column(result_columns, ["endpoint"])
effect_col = find_column(result_columns, ["effect"])
measurement_col = find_column(result_columns, ["measurement"])

if result_test_id_col is None or endpoint_col is None:
    sys.exit(
        "ERROR: test_id or endpoint column missing in results.txt.\n"
        f"Columns found: {result_columns}"
    )

result_usecols = [
    c for c in [
        result_id_col,
        result_test_id_col,
        endpoint_col,
        effect_col,
        measurement_col,
        find_column(result_columns, ["obs_duration_mean"]),
        find_column(result_columns, ["obs_duration_unit"]),
        find_column(result_columns, ["conc1_type"]),
        find_column(result_columns, ["conc1_mean_op"]),
        find_column(result_columns, ["conc1_mean"]),
        find_column(result_columns, ["conc1_unit"]),
    ]
    if c is not None
]

result_parts = []

if matched_test_ids:
    matched_test_set = set(matched_test_ids)

    print("\nScanning results.txt in chunks...")

    for chunk in pd.read_csv(
        RESULTS_FILE,
        sep="|",
        dtype=str,
        usecols=lambda c: clean_column_name(c) in result_usecols,
        chunksize=CHUNK_SIZE,
        encoding="utf-8-sig",
        skipinitialspace=True,
        on_bad_lines="skip",
        low_memory=False,
    ):
        chunk = normalize_columns(chunk)

        selected = chunk[
            chunk[result_test_id_col].astype(str).isin(matched_test_set)
        ]

        if not selected.empty:
            result_parts.append(selected)

if result_parts:
    toxicity_results = pd.concat(result_parts, ignore_index=True)
else:
    toxicity_results = pd.DataFrame(columns=result_usecols)

# Attach the candidate relationship and test metadata.
if not toxicity_results.empty and not relationships_df.empty:
    toxicity_results = toxicity_results.merge(
        relationships_df,
        left_on=result_test_id_col,
        right_on="test_id",
        how="left",
    )

if (
    not toxicity_results.empty
    and not matched_test_metadata.empty
    and test_id_col in matched_test_metadata.columns
):
    toxicity_results = toxicity_results.merge(
        matched_test_metadata,
        left_on=result_test_id_col,
        right_on=test_id_col,
        how="left",
        suffixes=("", "_test"),
    )

save_csv(
    toxicity_results,
    "ecotox_candidate_toxicity_results.csv",
)

# ============================================================
# 12. SEPARATE EC50 AND LC50 RECORDS
# ============================================================

if not toxicity_results.empty:
    endpoint_values = (
        toxicity_results[endpoint_col]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    ec50_records = toxicity_results[
        endpoint_values.str.contains(r"\bEC50\b", regex=True, na=False)
    ].copy()

    lc50_records = toxicity_results[
        endpoint_values.str.contains(r"\bLC50\b", regex=True, na=False)
    ].copy()
else:
    ec50_records = toxicity_results.copy()
    lc50_records = toxicity_results.copy()

save_csv(ec50_records, "ecotox_candidate_EC50_records.csv")
save_csv(lc50_records, "ecotox_candidate_LC50_records.csv")

# ============================================================
# 13. SUMMARY
# ============================================================

summary_rows = [
    ("candidate_count", len(candidates)),
    (
        "candidates_with_CAS",
        int(candidates["cas_match_key"].apply(bool).sum()),
    ),
    (
        "unique_normalized_candidate_CAS",
        candidate_cas_map["cas_key"].nunique()
        if not candidate_cas_map.empty else 0,
    ),
    (
        "candidates_matched_chemical_table",
        chemical_matches["candidate_row_id"].nunique()
        if not chemical_matches.empty else 0,
    ),
    (
        "candidates_matched_direct_test_CAS",
        direct_tests["candidate_row_id"].nunique()
        if not direct_tests.empty else 0,
    ),
    (
        "candidates_matched_carrier_CAS",
        carrier_matches["candidate_row_id"].nunique()
        if not carrier_matches.empty else 0,
    ),
    ("unique_matched_test_ids", len(matched_test_ids)),
    ("matched_result_rows", len(toxicity_results)),
    ("EC50_result_rows", len(ec50_records)),
    ("LC50_result_rows", len(lc50_records)),
]

summary = pd.DataFrame(summary_rows, columns=["metric", "value"])
save_csv(summary, "ecotox_matching_summary.csv")

print("\n" + "=" * 70)
print("MATCHING SUMMARY")
print("=" * 70)
print(summary.to_string(index=False))

print("\nOutput directory:")
print(OUTPUT_DIR)

print(
    "\nInterpretation note: these are identifier-based database matches. "
    "Review chemical identity, species, endpoint, exposure duration, "
    "concentration units, and study quality before using any record "
    "as an experimental reference."
)
