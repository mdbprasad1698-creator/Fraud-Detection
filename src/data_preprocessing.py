"""
Data Preprocessing Module for Financial Fraud Detection
========================================================
Handles loading, cleaning, and feature engineering of transaction data.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")


# ─── Column constants ────────────────────────────────────────────────────────
CATEGORICAL_COLS = [
    "payment_method", "card_type", "device_type",
    "operating_system", "browser", "merchant_category",
    "transaction_type", "day_of_week",
]

NUMERIC_COLS = [
    "transaction_amount", "user_account_age_days",
    "fee_amount", "hour", "time_diff",
]

DROP_COLS = [
    "transaction_id", "user_id", "organization",
    "transaction_timestamp", "currency", "country", "city",
]

TARGET = "is_fraud"


# ─── Loading ──────────────────────────────────────────────────────────────────
def load_data(filepath: str) -> pd.DataFrame:
    """Load CSV dataset and parse timestamps."""
    print(f"[INFO] Loading data from: {filepath}")
    df = pd.read_csv(filepath, parse_dates=["transaction_timestamp"])
    print(f"[INFO] Loaded {len(df):,} rows × {df.shape[1]} columns")
    return df


# ─── Inspection ──────────────────────────────────────────────────────────────
def inspect_data(df: pd.DataFrame) -> None:
    """Print a concise data quality report."""
    print("\n=== Dataset Overview ===")
    print(f"Shape        : {df.shape}")
    print(f"Fraud rate   : {df[TARGET].mean():.4%}  ({df[TARGET].sum():,} fraud / {len(df):,} total)")
    print(f"\nMissing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    print(f"\nNumeric summary:\n{df[NUMERIC_COLS].describe().round(2)}")
    print(f"\nCategorical value counts (top 5 per column):")
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            print(f"  {col}: {df[col].value_counts().head(5).to_dict()}")


# ─── Feature Engineering ─────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features that improve fraud signal."""
    df = df.copy()

    # Ratio features
    df["fee_ratio"]          = df["fee_amount"] / (df["transaction_amount"] + 1e-6)
    df["amount_per_acc_age"] = df["transaction_amount"] / (df["user_account_age_days"] + 1)

    # Time-based flags
    df["is_weekend"] = df["day_of_week"].isin(["Saturday", "Sunday"]).astype(int)
    df["is_high_risk_hour"] = df["hour"].between(0, 5).astype(int)  # midnight–5 am

    # Log-transform skewed columns
    df["log_amount"]    = np.log1p(df["transaction_amount"])
    df["log_time_diff"] = np.log1p(df["time_diff"])

    # Security flags
    df["no_otp_high_amount"] = (
        (df["otp_used"] == 0) & (df["transaction_amount"] > df["transaction_amount"].quantile(0.75))
    ).astype(int)
    df["unverified_new_user"] = (
        (df["kyc_verified"] == 0) & (df["user_account_age_days"] < 90)
    ).astype(int)

    print(f"[INFO] Feature engineering complete. New shape: {df.shape}")
    return df


# ─── Encoding ────────────────────────────────────────────────────────────────
def encode_features(
    df: pd.DataFrame,
    encoders: dict | None = None,
    fit: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    Label-encode categorical columns.

    Parameters
    ----------
    df       : Input DataFrame (will be copied).
    encoders : Pre-fitted encoders dict (pass during inference).
    fit      : If True, fit new encoders; otherwise reuse supplied encoders.

    Returns
    -------
    (encoded_df, encoders_dict)
    """
    df = df.copy()
    if encoders is None:
        encoders = {}

    cat_cols = [c for c in CATEGORICAL_COLS if c in df.columns]
    for col in cat_cols:
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            le = encoders[col]
            # Handle unseen labels gracefully
            known = set(le.classes_)
            df[col] = df[col].astype(str).apply(
                lambda x: x if x in known else le.classes_[0]
            )
            df[col] = le.transform(df[col])

    return df, encoders


# ─── Full Preprocessing Pipeline ─────────────────────────────────────────────
def preprocess(
    filepath: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """
    End-to-end preprocessing pipeline.

    Returns a dict with:
        X_train, X_test, y_train, y_test, feature_names,
        scaler, encoders, df_raw
    """
    df = load_data(filepath)
    inspect_data(df)

    df = engineer_features(df)
    df, encoders = encode_features(df, fit=True)

    # Drop non-feature columns
    drop = [c for c in DROP_COLS if c in df.columns]
    df.drop(columns=drop, inplace=True)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=feature_names
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=feature_names
    )

    print(f"\n[INFO] Train: {X_train.shape}  |  Test: {X_test.shape}")
    print(f"[INFO] Train fraud rate: {y_train.mean():.4%}  |  Test fraud rate: {y_test.mean():.4%}")

    return dict(
        X_train=X_train,
        X_test=X_test,
        X_train_scaled=X_train_scaled,
        X_test_scaled=X_test_scaled,
        y_train=y_train,
        y_test=y_test,
        feature_names=feature_names,
        scaler=scaler,
        encoders=encoders,
    )


if __name__ == "__main__":
    result = preprocess("../data/improved_fraud_dataset.csv")
    print("\n[DONE] Preprocessing complete.")
