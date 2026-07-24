import pandas as pd
import os


print("="*70)
print("PREPARING ML DATASETS")
print("="*70)


# Input files

train_file = "output/selected_features_train.csv"
test_file = "output/test_dataset.csv"
feature_file = "output/selected_feature_names.csv"


# Output files

X_train_file = "output/X_train.csv"
y_train_file = "output/y_train.csv"

X_test_file = "output/X_test.csv"
y_test_file = "output/y_test.csv"


# Load data

print("\nLoading datasets...")

train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)

features_df = pd.read_csv(feature_file)


# Feature list

selected_features = features_df["Selected_Features"].tolist()


print("\nNumber of selected features:")
print(len(selected_features))


# Check missing features

missing_train = set(selected_features) - set(train_df.columns)
missing_test = set(selected_features) - set(test_df.columns)


if missing_train:
    print("Missing features in training:")
    print(missing_train)
    raise Exception("Training features missing")


if missing_test:
    print("Missing features in testing:")
    print(missing_test)
    raise Exception("Testing features missing")


# Prepare X and y

X_train = train_df[selected_features]

y_train = train_df[["pEC50"]]


X_test = test_df[selected_features]

y_test = test_df[["pEC50"]]


# Save

X_train.to_csv(
    X_train_file,
    index=False
)

y_train.to_csv(
    y_train_file,
    index=False
)


X_test.to_csv(
    X_test_file,
    index=False
)

y_test.to_csv(
    y_test_file,
    index=False
)


print("\nDataset preparation completed")
print("-----------------------------")


print("\nX_train:")
print(X_train.shape)

print("\ny_train:")
print(y_train.shape)


print("\nX_test:")
print(X_test.shape)

print("\ny_test:")
print(y_test.shape)


print("\nSaved files:")
print(X_train_file)
print(y_train_file)
print(X_test_file)
print(y_test_file)


print("\nCompleted Successfully")
print("="*70)