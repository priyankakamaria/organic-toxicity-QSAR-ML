import os
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from xgboost import XGBRegressor

print("=" * 70)
print("FEATURE SUBSET OPTIMIZATION")
print("=" * 70)

os.makedirs("output/feature_subset_optimization", exist_ok=True)

# -------------------------------------------------------
# Load datasets
# -------------------------------------------------------

print("\nLoading datasets...")

X_train = pd.read_csv("output/X_train.csv")
y_train = pd.read_csv("output/y_train.csv").values.ravel()

X_test = pd.read_csv("output/X_test.csv")
y_test = pd.read_csv("output/y_test.csv").values.ravel()

print("Training:", X_train.shape)
print("Testing :", X_test.shape)

# -------------------------------------------------------
# Rank features using Random Forest
# -------------------------------------------------------

print("\nRanking features...")

rf = RandomForestRegressor(
    n_estimators=500,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": rf.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
).reset_index(drop=True)

importance.to_csv(
    "output/feature_subset_optimization/full_feature_ranking.csv",
    index=False
)

# -------------------------------------------------------
# Feature subset sizes
# -------------------------------------------------------

feature_sizes = [25, 50, 75, 100, 150, 200, 300]

results = []

cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# -------------------------------------------------------
# Loop
# -------------------------------------------------------

for n in feature_sizes:

    print("\n" + "=" * 50)
    print(f"Top {n} Features")
    print("=" * 50)

    selected = importance["Feature"][:n].tolist()

    Xtr = X_train[selected]
    Xte = X_test[selected]

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

    cv_r2 = cross_val_score(
        model,
        Xtr,
        y_train,
        cv=cv,
        scoring="r2",
        n_jobs=-1
    ).mean()

    model.fit(Xtr, y_train)

    pred = model.predict(Xte)

    r2 = r2_score(y_test, pred)

    rmse = np.sqrt(mean_squared_error(y_test, pred))

    mae = mean_absolute_error(y_test, pred)

    print(f"CV R2   : {cv_r2:.4f}")
    print(f"Test R2 : {r2:.4f}")

    results.append([
        n,
        cv_r2,
        r2,
        rmse,
        mae
    ])

# -------------------------------------------------------
# Save
# -------------------------------------------------------

results = pd.DataFrame(
    results,
    columns=[
        "Features",
        "CV_R2",
        "Test_R2",
        "RMSE",
        "MAE"
    ]
)

results.to_csv(
    "output/feature_subset_optimization/feature_subset_results.csv",
    index=False
)

best = results.loc[
    results["Test_R2"].idxmax()
]

with open(
    "output/feature_subset_optimization/best_feature_subset.txt",
    "w"
) as f:

    f.write(str(best))

print("\n")
print("=" * 70)
print(results)

print("\nBest feature subset")

print(best)

print("\nSaved")

print("output/feature_subset_optimization/feature_subset_results.csv")

print("output/feature_subset_optimization/best_feature_subset.txt")

print("=" * 70)
print("Completed Successfully")
print("=" * 70)