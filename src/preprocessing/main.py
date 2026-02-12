import os
from pathlib import Path
import numpy as np  # اضافه شد
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer  # اضافه شد برای مدیریت NaN ها
from sklearn.pipeline import Pipeline  # اضافه شد برای ساخت پایپ‌لاین تمیزتر
import pickle


def load_and_preprocess_data(raw_filepath, processed_dir):
    try:
        df = pd.read_csv(raw_filepath, sep=";")
    except Exception:
        df = pd.read_csv(raw_filepath)

    df.replace("unknown", np.nan, inplace=True)

    if "duration" in df.columns:
        df = df.drop(columns=["duration"])

    df = df.dropna(subset=["y"])

    X = df.drop(columns=["y"])
    y = df["y"]

    le = LabelEncoder()
    y = le.fit_transform(y)

    numeric_features = ["age", "balance", "campaign", "pdays", "previous", "day"]
    categorical_features = [
        "job",
        "marital",
        "education",
        "default",
        "housing",
        "loan",
        "contact",
        "month",
        "poutcome",
    ]

    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        verbose_feature_names_out=False,
    )

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.1765, random_state=42, stratify=y_temp
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()

    train_df = pd.DataFrame(X_train_processed, columns=feature_names)
    train_df["target"] = y_train

    val_df = pd.DataFrame(X_val_processed, columns=feature_names)
    val_df["target"] = y_val

    test_df = pd.DataFrame(X_test_processed, columns=feature_names)
    test_df["target"] = y_test

    os.makedirs(processed_dir, exist_ok=True)

    train_df.to_csv(os.path.join(processed_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(processed_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(processed_dir, "test.csv"), index=False)

    print(f"Data saved to {processed_dir}")
    print(f"Train shape: {train_df.shape}")
    print(f"Val shape: {val_df.shape}")
    print(f"Test shape: {test_df.shape}")

    # Save preprocessor to src/models/ directory (standardized location)
    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(exist_ok=True)
    preprocessor_path = models_dir / "preprocessor.pkl"
    with open(preprocessor_path, "wb") as f:
        pickle.dump(preprocessor, f)
    print(f"Preprocessor saved to {preprocessor_path}")

    return train_df, val_df, test_df


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    bank_full_path = "data/raw/bank/bank-full.csv"
    dataset_path = project_root / bank_full_path
    processed_path = project_root / "data/processed"

    try:
        train_df, val_df, test_df = load_and_preprocess_data(dataset_path, processed_path)
    except FileNotFoundError:
        print(f"Error: Raw data file not found at {dataset_path}")
