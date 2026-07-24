import os
import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt

print("=" * 70)
print("SHAP ANALYSIS")
print("=" * 70)

# ----------------------------------------------------
# Output folder
# ----------------------------------------------------

os.makedirs("output/SHAP", exist_ok=True)

# ----------------------------------------------------
# Load model
# ----------------------------------------------------

print("\nLoading model...")

model = joblib.load("output/best_xgboost_model.pkl")

# ----------------------------------------------------
# Load data
# ----------------------------------------------------

print("Loading training data...")

X_train = pd.read_csv("output/X_train.csv")

print("Training shape:", X_train.shape)

# ----------------------------------------------------
# Calculate SHAP values
# ----------------------------------------------------

print("\nCalculating SHAP values...")

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X_train)

print("Done.")

# ----------------------------------------------------
# Mean absolute SHAP values
# ----------------------------------------------------

importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Mean_SHAP": abs(shap_values).mean(axis=0)
})

importance = importance.sort_values(
    by="Mean_SHAP",
    ascending=False
)

importance.to_csv(
    "output/SHAP/shap_feature_importance.csv",
    index=False
)

print("\nTop 20 SHAP Features")

print(importance.head(20))

# ----------------------------------------------------
# SHAP Summary Plot
# ----------------------------------------------------

plt.figure(figsize=(10,8))

shap.summary_plot(
    shap_values,
    X_train,
    show=False
)

plt.tight_layout()

plt.savefig(
    "output/SHAP/SHAP_summary.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ----------------------------------------------------
# SHAP Bar Plot
# ----------------------------------------------------

plt.figure(figsize=(8,7))

shap.summary_plot(
    shap_values,
    X_train,
    plot_type="bar",
    show=False
)

plt.tight_layout()

plt.savefig(
    "output/SHAP/SHAP_bar.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nFiles saved")

print("output/SHAP/shap_feature_importance.csv")
print("output/SHAP/SHAP_summary.png")
print("output/SHAP/SHAP_bar.png")

print("=" * 70)
print("Completed Successfully")
print("=" * 70)