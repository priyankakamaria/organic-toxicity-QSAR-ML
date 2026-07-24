# ==============================================================
# GENERATE PUBLICATION TABLES
# Creates CSV + Excel tables for manuscript
# ==============================================================

import os
import pandas as pd
from openpyxl import Workbook

print("=" * 70)
print("GENERATING PUBLICATION TABLES")
print("=" * 70)

# --------------------------------------------------------------
# Create output folder
# --------------------------------------------------------------

output_dir = "output/publication_tables"
os.makedirs(output_dir, exist_ok=True)

# ==============================================================
# TABLE 1 : DATASET SUMMARY
# ==============================================================

train = pd.read_csv("output/train_dataset.csv")
test = pd.read_csv("output/test_dataset.csv")

table1 = pd.DataFrame({

    "Parameter":[
        "Total compounds",
        "Training compounds",
        "Testing compounds",
        "Response variable",
        "Endpoint"
    ],

    "Value":[
        len(train)+len(test),
        len(train),
        len(test),
        "pEC50",
        "EC50"
    ]

})

table1.to_csv(
    os.path.join(output_dir,"Table1_Dataset_Summary.csv"),
    index=False
)

# ==============================================================
# TABLE 2 : FEATURE REDUCTION
# ==============================================================

table2 = pd.DataFrame({

    "Step":[
        "Initial descriptors",
        "After constant removal",
        "After low variance",
        "After correlation removal",
        "Random Forest selected"
    ],

    "Features":[
        4470,
        4419,
        1275,
        1038,
        100
    ]

})

table2.to_csv(
    os.path.join(output_dir,"Table2_Descriptor_Reduction.csv"),
    index=False
)

# ==============================================================
# TABLE 3 : MODEL PERFORMANCE
# ==============================================================

performance = pd.read_csv("output/model_performance_results.csv")

# Add CatBoost if available
catboost_file = "output/catboost/catboost_metrics.csv"

if os.path.exists(catboost_file):

    cat = pd.read_csv(catboost_file)

    performance = pd.concat(
        [performance, cat],
        ignore_index=True
    )

performance.to_csv(
    os.path.join(output_dir,"Table3_Model_Performance.csv"),
    index=False
)

# ==============================================================
# TABLE 4 : TOP 20 SHAP FEATURES
# ==============================================================

shap = pd.read_csv("output/SHAP/shap_feature_importance.csv")

table4 = shap.head(20)

table4.to_csv(
    os.path.join(output_dir,"Table4_Top20_SHAP_Features.csv"),
    index=False
)

# ==============================================================
# TABLE 5 : APPLICABILITY DOMAIN
# ==============================================================

summary_file = "output/applicability_domain/applicability_domain_summary.txt"

rows = []

with open(summary_file,"r") as f:

    for line in f:

        if ":" in line:

            a,b = line.strip().split(":",1)

            rows.append([a.strip(),b.strip()])

table5 = pd.DataFrame(
    rows,
    columns=["Parameter","Value"]
)

table5.to_csv(
    os.path.join(output_dir,"Table5_Applicability_Domain.csv"),
    index=False
)

# ==============================================================
# TABLE 6 : Y RANDOMIZATION
# ==============================================================

yrand = pd.read_csv("output/y_randomization_results.csv")

table6 = yrand.describe().T

table6.to_csv(
    os.path.join(output_dir,"Table6_Y_Randomization.csv")
)

# ==============================================================
# CREATE EXCEL WORKBOOK
# ==============================================================

wb = Workbook()

wb.remove(wb.active)

tables = [

    ("Dataset Summary",table1),
    ("Feature Reduction",table2),
    ("Model Performance",performance),
    ("Top20 SHAP",table4),
    ("Applicability Domain",table5),
    ("Y Randomization",table6)

]

for sheet_name,df in tables:

    ws = wb.create_sheet(title=sheet_name)

    ws.append(list(df.columns))

    for row in df.itertuples(index=False):

        ws.append(list(row))

wb.save(
    os.path.join(output_dir,"Publication_Tables.xlsx")
)

print("\nGenerated Files")
print("-"*40)

print("Table1_Dataset_Summary.csv")
print("Table2_Descriptor_Reduction.csv")
print("Table3_Model_Performance.csv")
print("Table4_Top20_SHAP_Features.csv")
print("Table5_Applicability_Domain.csv")
print("Table6_Y_Randomization.csv")
print("Publication_Tables.xlsx")

print("\nSaved to:")
print(output_dir)

print("=" * 70)
print("Completed Successfully")
print("=" * 70)