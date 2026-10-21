
import time
import requests
import pandas as pd
from pathlib import Path
from rdkit import Chem

# PubChem API
BASE_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

# Output folder
OUTPUT_DIR = Path("output/prospective_pyrethroid")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Initial substructure pattern for a pyrethroid-related motif.
# This is a preliminary filter, not a complete definition
# of all pyrethroids.
SMARTS = "[C](=O)[O][C;X4]([C]#N)[c]1cccc([O][c]2ccccc2)c1"


def main():

    print("=" * 60)
    print("PYRETHROID-RELATED CANDIDATE EXTRACTION")
    print("=" * 60)

    query = Chem.MolFromSmarts(SMARTS)

    if query is None:
        raise ValueError("The SMARTS pattern is invalid.")

    # Submit substructure search to PubChem
    search_url = (
        f"{BASE_URL}/compound/fastsubstructure/smarts/cids/JSON"
    )

    print("Connecting to PubChem...")
    response = requests.post(
        search_url,
        data={"smarts": SMARTS, "MaxRecords": 5000},
        timeout=120
    )

    print("PubChem response status:", response.status_code)

    if not response.ok:
        print("PubChem response:")
        print(response.text[:2000])
        response.raise_for_status()

    data = response.json()
    cids = data.get("IdentifierList", {}).get("CID", [])

    print("Compound IDs found:", len(cids))

    if not cids:
        print("No compounds returned for this pattern.")
        print("The query may need to be revised.")
        return

    # Retrieve compound properties in batches
    records = []

    for start in range(0, len(cids), 100):
        batch = cids[start:start + 100]
        cid_string = ",".join(map(str, batch))

        properties_url = (
            f"{BASE_URL}/compound/cid/{cid_string}/property/"
            "IUPACName,CanonicalSMILES,IsomericSMILES,"
            "MolecularFormula,MolecularWeight,InChIKey/JSON"
        )

        result = requests.get(properties_url, timeout=60)

        if result.ok:
            properties = result.json().get(
                "PropertyTable", {}
            ).get("Properties", [])
            records.extend(properties)
        else:
            print(
                "Property retrieval failed for batch:",
                start,
                result.status_code
            )

        time.sleep(0.3)

    if not records:
        print("No compound properties were retrieved.")
        return

    df = pd.DataFrame(records)

    # Find the SMILES field returned by PubChem
    smiles_column = next(
        (
            col for col in
            ["ConnectivitySMILES", "SMILES", "CanonicalSMILES"]
            if col in df.columns
        ),
        None
    )

    if smiles_column is None:
        print("Available columns:", df.columns.tolist())
        raise ValueError("Could not identify the SMILES column.")

    # Independently verify the motif using RDKit
    def verify_motif(smiles):
        mol = Chem.MolFromSmiles(str(smiles))
        return (
            mol is not None
            and mol.HasSubstructMatch(query)
        )

    df["motif_verified"] = df[smiles_column].apply(verify_motif)
    df = df[df["motif_verified"]].copy()
    df = df.drop_duplicates(subset=["CID"])

    output_file = OUTPUT_DIR / "pubchem_candidates.csv"
    df.to_csv(output_file, index=False)

    print("-" * 60)
    print("RDKit-verified candidates:", len(df))
    print("Output saved to:", output_file.resolve())
    print("=" * 60)


if __name__ == "__main__":
    main()
