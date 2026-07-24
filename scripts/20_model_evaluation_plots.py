import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


print("="*70)
print("QSAR MODEL EVALUATION PLOTS")
print("="*70)


# Input file

input_file = "output/model_predictions.csv"


# Output folder

output_folder = "output/model_plots"

os.makedirs(output_folder, exist_ok=True)


# Load prediction data

print("\nLoading prediction file...")

df = pd.read_csv(input_file)

print(df.head())


# Experimental values

y_true = df["Experimental_pEC50"]


# Models available

models = [
    "Random Forest_Prediction",
    "XGBoost_Prediction",
    "Gradient Boosting_Prediction"
]


results = []


for model_name in models:

    print("\nProcessing:", model_name)

    y_pred = df[model_name]


    # Metrics

    r2 = r2_score(y_true, y_pred)

    rmse = np.sqrt(
        mean_squared_error(y_true, y_pred)
    )

    mae = mean_absolute_error(
        y_true,
        y_pred
    )


    results.append(
        [
            model_name.replace("_Prediction",""),
            r2,
            rmse,
            mae
        ]
    )


    # =====================================
    # Experimental vs Predicted plot
    # =====================================

    plt.figure(figsize=(6,6))


    plt.scatter(
        y_true,
        y_pred,
        alpha=0.6
    )


    minimum = min(
        y_true.min(),
        y_pred.min()
    )

    maximum = max(
        y_true.max(),
        y_pred.max()
    )


    plt.plot(
        [minimum, maximum],
        [minimum, maximum]
    )


    plt.xlabel("Experimental pEC50")
    plt.ylabel("Predicted pEC50")

    plt.title(
        model_name.replace("_Prediction","")
    )


    plt.text(
        minimum,
        maximum-1,
        f"R² = {r2:.3f}\nRMSE = {rmse:.3f}\nMAE = {mae:.3f}"
    )


    plt.tight_layout()


    plt.savefig(
        f"{output_folder}/{model_name}_predicted_vs_experimental.png",
        dpi=300
    )


    plt.close()



    # =====================================
    # Residual plot
    # =====================================

    residuals = y_true - y_pred


    plt.figure(figsize=(7,5))


    plt.scatter(
        y_pred,
        residuals,
        alpha=0.6
    )


    plt.axhline(
        0
    )


    plt.xlabel("Predicted pEC50")
    plt.ylabel("Residual")


    plt.title(
        model_name.replace("_Prediction","")
        + " Residual Plot"
    )


    plt.tight_layout()


    plt.savefig(
        f"{output_folder}/{model_name}_residual_plot.png",
        dpi=300
    )


    plt.close()



# Save metrics

metrics_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "R2",
        "RMSE",
        "MAE"
    ]
)


metrics_df.to_csv(
    "output/initial_model_metrics.csv",
    index=False
)


print("\nCompleted Successfully")

print("\nPlots saved in:")
print(output_folder)

print("\nMetrics saved:")
print("output/initial_model_metrics.csv")

print("="*70)