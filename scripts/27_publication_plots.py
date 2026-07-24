# ==============================================================
# PUBLICATION QUALITY PLOTS
# Best Model: XGBoost
# ==============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score

# -------------------------------------------------------------
# Create output folder
# -------------------------------------------------------------

output_dir = "output/publication_plots"
os.makedirs(output_dir, exist_ok=True)

print("="*70)
print("PUBLICATION QUALITY PLOTS")
print("="*70)

# -------------------------------------------------------------
# Load prediction file
# -------------------------------------------------------------

df = pd.read_csv("output/model_predictions.csv")

y_true = df["Experimental_pEC50"]
y_pred = df["XGBoost_Prediction"]

# -------------------------------------------------------------
# Residuals
# -------------------------------------------------------------

residuals = y_true - y_pred

r2 = r2_score(y_true, y_pred)

# =============================================================
# Figure 1
# Experimental vs Predicted
# =============================================================

plt.figure(figsize=(7,7))

plt.scatter(
    y_true,
    y_pred,
    alpha=0.7
)

min_val = min(y_true.min(), y_pred.min())
max_val = max(y_true.max(), y_pred.max())

plt.plot(
    [min_val, max_val],
    [min_val, max_val],
    'r--',
    linewidth=2
)

plt.xlabel("Experimental pEC50", fontsize=12)
plt.ylabel("Predicted pEC50", fontsize=12)

plt.title(f"Experimental vs Predicted\nXGBoost (R² = {r2:.3f})")

plt.tight_layout()

plt.savefig(
    f"{output_dir}/Figure1_Experimental_vs_Predicted.png",
    dpi=600
)

plt.close()

# =============================================================
# Figure 2
# Residual Plot
# =============================================================

plt.figure(figsize=(7,6))

plt.scatter(
    y_pred,
    residuals,
    alpha=0.7
)

plt.axhline(
    y=0,
    color="red",
    linestyle="--"
)

plt.xlabel("Predicted pEC50")
plt.ylabel("Residuals")

plt.title("Residual Plot")

plt.tight_layout()

plt.savefig(
    f"{output_dir}/Figure2_Residual_Plot.png",
    dpi=600
)

plt.close()

# =============================================================
# Figure 3
# Residual Distribution
# =============================================================

plt.figure(figsize=(7,6))

plt.hist(
    residuals,
    bins=30
)

plt.xlabel("Residual")
plt.ylabel("Frequency")

plt.title("Residual Distribution")

plt.tight_layout()

plt.savefig(
    f"{output_dir}/Figure3_Residual_Distribution.png",
    dpi=600
)

plt.close()

# =============================================================
# Figure 4
# Model Comparison
# =============================================================

performance = pd.DataFrame({

    "Model":[
        "RF",
        "GB",
        "XGBoost",
        "CatBoost",
        "SVR"
    ],

    "R2":[
        0.6333,
        0.6410,
        0.6480,
        0.6448,
        -0.0032
    ]

})

plt.figure(figsize=(8,6))

plt.bar(
    performance["Model"],
    performance["R2"]
)

plt.ylabel("Test R²")

plt.title("Model Comparison")

plt.tight_layout()

plt.savefig(
    f"{output_dir}/Figure4_Model_Comparison.png",
    dpi=600
)

plt.close()

print("\nFigures saved to:")
print(output_dir)

print("\nGenerated:")
print("1. Experimental vs Predicted")
print("2. Residual Plot")
print("3. Residual Distribution")
print("4. Model Comparison")

print("="*70)
print("Completed Successfully")
print("="*70)