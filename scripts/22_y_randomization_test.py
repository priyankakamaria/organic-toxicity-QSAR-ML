import pandas as pd
import numpy as np

from xgboost import XGBRegressor

from sklearn.model_selection import KFold, cross_val_score
from sklearn.utils import shuffle


print("="*70)
print("Y-RANDOMIZATION TEST")
print("QSAR MODEL VALIDATION")
print("="*70)


# Load datasets

print("\nLoading datasets...")

X_train = pd.read_csv(
    "output/X_train.csv"
)

y_train = pd.read_csv(
    "output/y_train.csv"
).values.ravel()


print("Training dataset:")
print(X_train.shape)


# Best XGBoost parameters obtained earlier

model = XGBRegressor(
    n_estimators=800,
    learning_rate=0.01,
    max_depth=7,
    subsample=0.7,
    colsample_bytree=0.7,
    random_state=42,
    n_jobs=-1
)


cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# Original model CV

print("\nOriginal model performance")

original_scores = cross_val_score(
    model,
    X_train,
    y_train,
    cv=cv,
    scoring="r2",
    n_jobs=-1
)


print(
    "Original CV R2:",
    original_scores.mean()
)



# Y randomization

n_iterations = 20

random_results=[]


print("\nStarting Y-randomization...")
print("Iterations:", n_iterations)


for i in range(n_iterations):

    print(
        "Iteration",
        i+1
    )


    y_random = shuffle(
        y_train,
        random_state=i
    )


    random_scores = cross_val_score(
        model,
        X_train,
        y_random,
        cv=cv,
        scoring="r2",
        n_jobs=-1
    )


    random_results.append(
        random_scores.mean()
    )


# Save results

results = pd.DataFrame(
    {
        "Iteration":
        range(1,n_iterations+1),

        "Randomized_R2":
        random_results
    }
)


results.to_csv(
    "output/y_randomization_results.csv",
    index=False
)


print("\n")
print("="*70)
print("Y-RANDOMIZATION SUMMARY")
print("="*70)


print(
    "Original CV R2:",
    original_scores.mean()
)


print(
    "Randomized mean R2:",
    np.mean(random_results)
)


print(
    "Randomized max R2:",
    np.max(random_results)
)


print("\nSaved file:")
print(
    "output/y_randomization_results.csv"
)


print("="*70)
print("Completed Successfully")