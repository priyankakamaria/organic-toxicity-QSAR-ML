import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("="*70)
print("APPLICABILITY DOMAIN (WILLIAMS PLOT)")
print("="*70)

os.makedirs("output/applicability_domain", exist_ok=True)

print("\nLoading datasets...")

X_train = pd.read_csv("output/X_train.csv")
X_test = pd.read_csv("output/X_test.csv")
pred = pd.read_csv("output/model_predictions.csv")

# Experimental and predicted values
y_true = pred["Experimental_pEC50"]
y_pred = pred["XGBoost_Prediction"]

print(f"Training compounds : {len(X_train)}")
print(f"Testing compounds  : {len(X_test)}")

# ============================================================
# Calculate leverage
# ============================================================

print("\nCalculating leverage values...")

Xtrain = X_train.values
Xtest = X_test.values

XtX = Xtrain.T @ Xtrain
XtX_inv = np.linalg.pinv(XtX)

leverages = np.sum((Xtest @ XtX_inv) * Xtest, axis=1)

# ============================================================
# Standardized residuals
# ============================================================

print("Calculating standardized residuals...")

residuals = y_true - y_pred

std_residuals = residuals / np.std(residuals)

# ============================================================
# Warning leverage
# ============================================================

n = X_train.shape[0]
p = X_train.shape[1]

h_star = (3 * (p + 1)) / n

print(f"\nWarning leverage (h*) = {h_star:.4f}")

# ============================================================
# Classify compounds
# ============================================================

inside = (
    (leverages < h_star) &
    (np.abs(std_residuals) <= 3)
)

outside = ~inside

inside_count = np.sum(inside)
outside_count = np.sum(outside)

print("\nApplicability Domain Summary")
print("-----------------------------")
print(f"Inside AD : {inside_count}")
print(f"Outside AD: {outside_count}")

# ============================================================
# Save results
# ============================================================

results = pd.DataFrame({
    "Experimental_pEC50": y_true,
    "Predicted_pEC50": y_pred,
    "Residual": residuals,
    "Standardized_Residual": std_residuals,
    "Leverage": leverages,
    "Inside_AD": inside
})

results.to_csv(
    "output/applicability_domain/applicability_domain_results.csv",
    index=False
)

# ============================================================
# Save summary
# ============================================================

with open(
    "output/applicability_domain/applicability_domain_summary.txt",
    "w"
) as f:

    f.write("Applicability Domain Summary\n")
    f.write("============================\n\n")
    f.write(f"Training compounds : {n}\n")
    f.write(f"Testing compounds  : {len(X_test)}\n\n")
    f.write(f"Descriptors : {p}\n")
    f.write(f"Warning leverage (h*) : {h_star:.4f}\n\n")
    f.write(f"Inside AD : {inside_count}\n")
    f.write(f"Outside AD: {outside_count}\n")

# ============================================================
# Williams Plot
# ============================================================

plt.figure(figsize=(8,6))

plt.scatter(
    leverages,
    std_residuals,
    c="blue",
    edgecolors="black",
    alpha=0.7
)

plt.axhline(3, color="red", linestyle="--")
plt.axhline(-3, color="red", linestyle="--")
plt.axvline(h_star, color="green", linestyle="--")

plt.xlabel("Leverage")
plt.ylabel("Standardized Residual")

plt.title("Williams Plot")

plt.tight_layout()

plt.savefig(
    "output/applicability_domain/Williams_plot.png",
    dpi=300
)

plt.close()

print("\nFiles saved")
print("-----------------------------")
print("output/applicability_domain/applicability_domain_results.csv")
print("output/applicability_domain/applicability_domain_summary.txt")
print("output/applicability_domain/Williams_plot.png")

print("\nCompleted Successfully")
print("="*70)