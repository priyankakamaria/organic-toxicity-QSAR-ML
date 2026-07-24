import pandas as pd
import numpy as np

from sklearn.model_selection import RandomizedSearchCV, KFold

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

from xgboost import XGBRegressor

from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)

import os


print("="*70)
print("HYPERPARAMETER OPTIMIZATION")
print("5-FOLD CROSS VALIDATION")
print("="*70)


# Files

X_train_file = "output/X_train.csv"
y_train_file = "output/y_train.csv"

X_test_file = "output/X_test.csv"
y_test_file = "output/y_test.csv"


# Load data

print("\nLoading datasets...")

X_train = pd.read_csv(X_train_file)
y_train = pd.read_csv(y_train_file).values.ravel()

X_test = pd.read_csv(X_test_file)
y_test = pd.read_csv(y_test_file).values.ravel()


print("Training:", X_train.shape)
print("Testing :", X_test.shape)


# CV

cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# Models and parameters

models = {


"Random Forest":

(
RandomForestRegressor(
    random_state=42,
    n_jobs=-1
),

{
"n_estimators":[200,500,800],
"max_depth":[5,10,20,None],
"min_samples_split":[2,5,10],
"min_samples_leaf":[1,2,4]
}

),



"XGBoost":

(
XGBRegressor(
    random_state=42,
    n_jobs=-1
),

{
"n_estimators":[200,500,800],
"learning_rate":[0.01,0.05,0.1],
"max_depth":[3,5,7],
"subsample":[0.7,0.8,1],
"colsample_bytree":[0.7,0.8,1]
}

),



"Gradient Boosting":

(
GradientBoostingRegressor(
    random_state=42
),

{
"n_estimators":[200,500,800],
"learning_rate":[0.01,0.05,0.1],
"max_depth":[3,5,7],
"min_samples_split":[2,5,10]
}

)

}



results=[]


best_models={}


for name,(model,param_grid) in models.items():

    print("\n"+"="*60)
    print("Optimizing:",name)
    print("="*60)


    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=20,
        scoring="r2",
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )


    search.fit(
        X_train,
        y_train
    )


    best_model = search.best_estimator_

    best_models[name]=best_model


    print("\nBest parameters:")
    print(search.best_params_)


    print(
        "Best CV R2:",
        search.best_score_
    )


    # Test prediction

    best_model.fit(
        X_train,
        y_train
    )


    prediction = best_model.predict(
        X_test
    )


    test_r2 = r2_score(
        y_test,
        prediction
    )


    test_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            prediction
        )
    )


    test_mae = mean_absolute_error(
        y_test,
        prediction
    )


    print("Test R2:",test_r2)
    print("Test RMSE:",test_rmse)
    print("Test MAE:",test_mae)



    results.append(
        [
            name,
            search.best_score_,
            test_r2,
            test_rmse,
            test_mae,
            str(search.best_params_)
        ]
    )



# Save results

results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Best_CV_R2",
        "Test_R2",
        "Test_RMSE",
        "Test_MAE",
        "Best_Parameters"
    ]
)


results_df.to_csv(
    "output/optimized_model_results.csv",
    index=False
)


print("\nCompleted Successfully")

print("\nSaved:")
print("output/optimized_model_results.csv")

print("="*70)