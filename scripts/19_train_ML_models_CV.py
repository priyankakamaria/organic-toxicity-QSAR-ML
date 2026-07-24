import pandas as pd
import numpy as np
import os

from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)
from sklearn.svm import SVR
from xgboost import XGBRegressor

from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)


print("="*70)
print("QSAR MACHINE LEARNING MODEL TRAINING")
print("5-FOLD CROSS VALIDATION")
print("="*70)


# Input files

X_train_file = "output/X_train.csv"
y_train_file = "output/y_train.csv"

X_test_file = "output/X_test.csv"
y_test_file = "output/y_test.csv"


# Output files

performance_file = "output/model_performance_results.csv"
prediction_file = "output/model_predictions.csv"


# Load datasets

print("\nLoading datasets...")

X_train = pd.read_csv(X_train_file)
y_train = pd.read_csv(y_train_file).values.ravel()

X_test = pd.read_csv(X_test_file)
y_test = pd.read_csv(y_test_file).values.ravel()


print("\nTraining data:")
print(X_train.shape)

print("\nTesting data:")
print(X_test.shape)


# Define models

models = {

    "Random Forest":
        RandomForestRegressor(
            n_estimators=500,
            random_state=42,
            n_jobs=-1
        ),


    "XGBoost":
        XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        ),


    "SVR":
        SVR(
            kernel="rbf",
            C=10,
            gamma="scale"
        ),


    "Gradient Boosting":
        GradientBoostingRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=5,
            random_state=42
        )
}


# 5-fold CV

cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


results = []

predictions = pd.DataFrame({
    "Experimental_pEC50": y_test
})


# Train models

for name, model in models.items():

    print("\n" + "="*50)
    print("MODEL:", name)
    print("="*50)


    # Cross validation

    r2_scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="r2"
    )


    rmse_scores = np.sqrt(
        -cross_val_score(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring="neg_mean_squared_error"
        )
    )


    mae_scores = -cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="neg_mean_absolute_error"
    )


    cv_r2 = r2_scores.mean()
    cv_rmse = rmse_scores.mean()
    cv_mae = mae_scores.mean()


    print("CV R2:", cv_r2)
    print("CV RMSE:", cv_rmse)
    print("CV MAE:", cv_mae)


    # Train final model

    print("Training final model...")

    model.fit(
        X_train,
        y_train
    )


    # External test prediction

    y_pred = model.predict(X_test)


    test_r2 = r2_score(
        y_test,
        y_pred
    )

    test_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            y_pred
        )
    )

    test_mae = mean_absolute_error(
        y_test,
        y_pred
    )


    print("Test R2:", test_r2)
    print("Test RMSE:", test_rmse)
    print("Test MAE:", test_mae)


    results.append(
        [
            name,
            cv_r2,
            cv_rmse,
            cv_mae,
            test_r2,
            test_rmse,
            test_mae
        ]
    )


    predictions[name+"_Prediction"] = y_pred



# Save performance

results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "CV_R2",
        "CV_RMSE",
        "CV_MAE",
        "Test_R2",
        "Test_RMSE",
        "Test_MAE"
    ]
)


results_df.to_csv(
    performance_file,
    index=False
)


predictions.to_csv(
    prediction_file,
    index=False
)


print("\n" + "="*70)
print("MODEL TRAINING COMPLETED")
print("="*70)


print("\nPerformance file:")
print(performance_file)

print("\nPrediction file:")
print(prediction_file)

print("="*70)