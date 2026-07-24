import os
import pandas as pd
from datetime import datetime

# ==============================
# PATH SETTINGS
# ==============================

PROJECT_DIR = os.getcwd()

OUTPUT_DIR = os.path.join(PROJECT_DIR, "output")
RESULT_DIR = os.path.join(PROJECT_DIR, "results")

SUMMARY_FILE = "manuscript_summary.txt"


# ==============================
# HELPER FUNCTION
# ==============================

def check_file(path):
    return os.path.exists(path)


def read_csv_info(file_path):
    try:
        df = pd.read_csv(file_path)

        info = []
        info.append(f"\nFile: {os.path.basename(file_path)}")
        info.append(f"Rows: {df.shape[0]}")
        info.append(f"Columns: {df.shape[1]}")
        info.append(f"Column names:")
        info.append(", ".join(df.columns.tolist()))

        return "\n".join(info)

    except Exception as e:
        return f"Could not read {file_path}: {e}"


# ==============================
# START SUMMARY
# ==============================

summary = []

summary.append("="*70)
summary.append("QSAR MANUSCRIPT SUMMARY")
summary.append("="*70)

summary.append(
    f"\nGenerated on: {datetime.now()}"
)


# ==============================
# SEARCH IMPORTANT FILES
# ==============================

important_files = [

    "model_results.csv",
    "feature_importance.csv",
    "shap_features.csv",
    "shap_summary.csv",
    "y_randomization_results.csv",
    "applicability_domain.csv",
    "final_dataset.csv",
    "organic_toxicity_median.csv"

]


summary.append("\n\nAVAILABLE FILES")
summary.append("-"*50)


for root, dirs, files in os.walk(PROJECT_DIR):

    for file in files:

        if file.lower() in [x.lower() for x in important_files]:

            full_path = os.path.join(root,file)

            summary.append(
                f"\nFOUND: {full_path}"
            )

            summary.append(
                read_csv_info(full_path)
            )


# ==============================
# SCAN ALL CSV FILES
# ==============================

summary.append("\n\nALL CSV FILES")
summary.append("-"*50)


for root, dirs, files in os.walk(PROJECT_DIR):

    for file in files:

        if file.endswith(".csv"):

            summary.append(
                os.path.join(root,file)
            )


# ==============================
# SAVE SUMMARY
# ==============================

with open(
    SUMMARY_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write("\n".join(summary))


print("\nCompleted Successfully")
print("Created:")
print(SUMMARY_FILE)