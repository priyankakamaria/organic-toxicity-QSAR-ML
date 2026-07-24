import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBRegressor

print("=" * 70)
print("SAVE OPTIMIZED XGBOOST MODEL")
print("FEATURE IMPORTANCE ANALYSIS")
print("=" * 70)

# ---------------------------------------------------
# Create output folders
# ---------------------------------------------------

os.makedirs("output/feature_importance", exist_ok=True)

# ---------------------------------------------------
# Load datasets
# ---------------------------------------------------

print("\nLoading datasets...")

X_train = pd.read_csv("output/X_train.csv")
y_train = pd.read_csv("output/y_train.csv").values.ravel()

print("Training data:", X_train.shape)

# ---------------------------------------------------
# Train optimized model
# ---------------------------------------------------

print("\nTraining optimized XGBoost model...")

model = XGBRegressor(
    n_estimators=800,
    max_depth=7,
    learning_rate=0.01,
    subsample=0.7,
    colsample_bytree=0.7,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# ---------------------------------------------------
# Save model
# ---------------------------------------------------

joblib.dump(model, "output/best_xgboost_model.pkl")

print("\nModel saved successfully.")

# ---------------------------------------------------
# Feature importance
# ---------------------------------------------------

importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

importance.to_csv(
    "output/feature_importance/feature_importance.csv",
    index=False
)

print("\nTop 20 features")

print(importance.head(20))

# ---------------------------------------------------
# Plot
# ---------------------------------------------------

top20 = importance.head(20)

plt.figure(figsize=(9,7))

plt.barh(
    top20["Feature"][::-1],
    top20["Importance"][::-1]
)

plt.xlabel("Importance")
plt.title("Top 20 XGBoost Feature Importances")

plt.tight_layout()

plt.savefig(
    "output/feature_importance/top20_feature_importance.png",
    dpi=300
)

plt.close()

print("\nFiles saved")

print("output/best_xgboost_model.pkl")
print("output/feature_importance/feature_importance.csv")
print("output/feature_importance/top20_feature_importance.png")

print("=" * 70)
print("Completed Successfully")
print("=" * 70)