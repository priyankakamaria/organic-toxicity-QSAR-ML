

from pathlib import Path

DATA_DIR = Path("data/ecotox_ascii_09_15_2026")

targets = [
    "chemicals.txt",
    "chemical_carriers.txt",
    "tests.txt",
    "results.txt",
    "species.txt",
    "endpoint_codes.txt",
    "effect_codes.txt",
]

print("=" * 75)
print("ECOTOX FILE DISCOVERY")
print("=" * 75)

if not DATA_DIR.exists():
    raise FileNotFoundError(f"Folder not found: {DATA_DIR.resolve()}")

all_files = [
    p for p in DATA_DIR.rglob("*")
    if p.is_file()
]

for target in targets:
    matches = [
        p for p in all_files
        if p.name.lower() == target.lower()
    ]

    print(f"\nTARGET: {target}")

    if not matches:
        print("NOT FOUND")
        continue

    for path in matches:
        print("PATH:", path)
        print("SIZE:", path.stat().st_size, "bytes")

        try:
            with path.open(
                "r",
                encoding="utf-8-sig",
                errors="replace"
            ) as f:
                header = f.readline().rstrip("\r\n")
                print("HEADER:")
                print(header[:5000])
        except Exception as exc:
            print("READ ERROR:", exc)

print("\nDiscovery completed.")
