
import time
import requests
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]

INPUT = BASE / "output/prospective_pyrethroid/unique_inchikey_candidates.csv"
OUTPUT = BASE / "output/prospective_pyrethroid/candidates_with_cas.csv"
ERRORS = BASE / "output/prospective_pyrethroid/pubchem_cas_errors.csv"

df = pd.read_csv(INPUT, dtype=str).fillna("")

if "CID" not in df.columns:
    raise ValueError("Input file does not contain a CID column.")

df["CID"] = df["CID"].str.strip()
df["CAS_RNs"] = ""
df["CAS_lookup_status"] = "Not queried"

session = requests.Session()
session.headers.update({
    "User-Agent": "ProspectiveQSARResearch/1.0 (academic research)"
})

errors = []

def get_cas_numbers(cid):
    url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/"
        f"compound/cid/{cid}/xrefs/RN/JSON"
    )

    response = session.get(url, timeout=30)

    if response.status_code == 404:
        return [], "No RN cross-reference returned"

    response.raise_for_status()
    data = response.json()

    identifiers = (
        data.get("InformationList", {})
        .get("Information", [{}])[0]
        .get("RN", [])
    )

    # Keep CAS-shaped identifiers only; this is a format filter,
    # not independent confirmation that each identifier is a CAS RN.
    cas_like = sorted({
        str(x).strip()
        for x in identifiers
        if str(x).strip()
        and len(str(x).strip().split("-")) == 3
        and all(part.isdigit()
                for part in str(x).strip().split("-"))
    })

    return cas_like, "Retrieved" if cas_like else "No CAS-shaped RN returned"

for i, row in df.iterrows():
    cid = row["CID"]

    if not cid.isdigit():
        df.at[i, "CAS_lookup_status"] = "Invalid CID"
        errors.append({"CID": cid, "error": "Invalid CID"})
        continue

    try:
        cas_numbers, status = get_cas_numbers(cid)
        df.at[i, "CAS_RNs"] = ";".join(cas_numbers)
        df.at[i, "CAS_lookup_status"] = status

    except Exception as exc:
        df.at[i, "CAS_lookup_status"] = "Lookup error"
        errors.append({"CID": cid, "error": str(exc)})

    if (i + 1) % 25 == 0:
        print(f"Processed {i + 1}/{len(df)} compounds")
        df.to_csv(OUTPUT, index=False)

    # Be considerate of PubChem's request limits.
    time.sleep(0.25)

df.to_csv(OUTPUT, index=False)
pd.DataFrame(errors, columns=["CID", "error"]).to_csv(ERRORS, index=False)

print("\nCAS lookup completed.")
print("Total candidates:", len(df))
print("With CAS-shaped identifiers:", (df["CAS_RNs"] != "").sum())
print("Without CAS-shaped identifiers:", (df["CAS_RNs"] == "").sum())
print("Output:", OUTPUT)
print("Errors:", ERRORS)
