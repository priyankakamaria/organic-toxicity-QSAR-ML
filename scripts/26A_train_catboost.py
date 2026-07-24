# ==============================================================
# CATBOOST QSAR MODEL TRAINING
# 5-FOLD CROSS VALIDATION + HYPERPARAMETER OPTIMIZATION
# ==============================================================

import os
import pandas as pd
import numpy as np

from catboost import CatBoostRegressor

from sklearn.model_selection import (
    KFold,
    cross_val_score,
    RandomizedSearchCV
)

from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)

import joblib


print("="*70)
print("CATBOOST QSAR MODEL TRAINING")
print("5-FOLD CROSS VALIDATION + OPTIMIZATION")
print("="*70)


# --------------------------------------------------------------
# PATHS
# --------------------------------------------------------------

X_train_path = "output/X_train.csv"
y_train_path = "output/y_train.csv"

X_test_path = "output/X_test.csv"
y_test_path = "output/y_test.csv"


output_dir = "output/catboost"

os.makedirs(output_dir, exist_ok=True)


# --------------------------------------------------------------
# LOAD DATA
# --------------------------------------------------------------

print("\nLoading datasets...")

X_train = pd.read_csv(X_train_path)
y_train = pd.read_csv(y_train_path)

X_test = pd.read_csv(X_test_path)
y_test = pd.read_csv(y_test_path)


# Convert y to 1D array

y_train = y_train.values.ravel()
y_test = y_test.values.ravel()


print("\nTraining:")
print(X_train.shape)

print("Testing:")
print(X_test.shape)


# --------------------------------------------------------------
# BASE CATBOOST MODEL
# --------------------------------------------------------------

model = CatBoostRegressor(
    loss_function="RMSE",
    random_seed=42,
    verbose=0
)


# --------------------------------------------------------------
# 5 FOLD CROSS VALIDATION
# --------------------------------------------------------------

print("\n")
print("="*60)
print("INITIAL 5-FOLD CROSS VALIDATION")
print("="*60)


kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


cv_scores = cross_val_score(
    model,
    X_train,
    y_train,
    cv=kf,
    scoring="r2",
    n_jobs=-1
)


print("\nCV R2 Scores:")
print(cv_scores)

print("\nMean CV R2:")
print(cv_scores.mean())


# --------------------------------------------------------------
# HYPERPARAMETER OPTIMIZATION
# --------------------------------------------------------------

print("\n")
print("="*60)
print("CATBOOST HYPERPARAMETER OPTIMIZATION")
print("="*60)


param_grid = {

    "iterations":[300,500,800,1000],

    "depth":[4,5,6,7,8],

    "learning_rate":[0.01,0.03,0.05,0.1],

    "l2_leaf_reg":[1,3,5,7,9],

    "bagging_temperature":[0,1,3,5],

    "random_strength":[1,2,5]

}


search = RandomizedSearchCV(

    estimator=model,

    param_distributions=param_grid,

    n_iter=20,

    scoring="r2",

    cv=5,

    random_state=42,

    verbose=2,

    n_jobs=-1

)


print("\nStarting optimization...")


search.fit(
    X_train,
    y_train
)


best_model = search.best_estimator_


print("\nBest Parameters:")
print(search.best_params_)


print("\nBest CV R2:")
print(search.best_score_)



# --------------------------------------------------------------
# TEST SET EVALUATION
# --------------------------------------------------------------

print("\n")
print("="*60)
print("FINAL CATBOOST TEST EVALUATION")
print("="*60)


best_model.fit(
    X_train,
    y_train
)


pred = best_model.predict(
    X_test
)


r2 = r2_score(
    y_test,
    pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        pred
    )
)

mae = mean_absolute_error(
    y_test,
    pred
)


print("\nTest R2:")
print(r2)

print("\nTest RMSE:")
print(rmse)

print("\nTest MAE:")
print(mae)



# --------------------------------------------------------------
# SAVE MODEL
# --------------------------------------------------------------

joblib.dump(
    best_model,
    f"{output_dir}/best_catboost_model.pkl"
)


# --------------------------------------------------------------
# SAVE PREDICTIONS
# --------------------------------------------------------------

prediction_df = pd.DataFrame({

    "Experimental_pEC50": y_test,

    "CatBoost_Prediction": pred

})


prediction_df.to_csv(
    f"{output_dir}/catboost_predictions.csv",
    index=False
)



# --------------------------------------------------------------
# SAVE METRICS
# --------------------------------------------------------------

metrics = pd.DataFrame({

    "Model":["CatBoost"],

    "CV_R2":[search.best_score_],

    "Test_R2":[r2],

    "Test_RMSE":[rmse],

    "Test_MAE":[mae]

})


metrics.to_csv(
    f"{output_dir}/catboost_metrics.csv",
    index=False
)



# --------------------------------------------------------------
# FEATURE IMPORTANCE
# --------------------------------------------------------------

importance = pd.DataFrame({

    "Feature":X_train.columns,

    "Importance":best_model.feature_importances_

})


importance = importance.sort_values(
    by="Importance",
    ascending=False
)


importance.to_csv(
    f"{output_dir}/catboost_feature_importance.csv",
    index=False
)



print("\n")
print("="*70)
print("CATBOOST TRAINING COMPLETED")
print("="*70)


print("\nSaved files:")
print(output_dir)

print("\nModel:")
print("best_catboost_model.pkl")

print("\nMetrics:")
print("catboost_metrics.csv")

print("\nFeature importance:")
print("catboost_feature_importance.csv")

print("="*70)