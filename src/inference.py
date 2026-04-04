"""
Inference Module
================
Load a trained model and run predictions on new transaction data.
"""

import os
import joblib
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

MODEL_DIR = "../models"


class FraudDetector:
    """
    End-to-end fraud detection inference wrapper.

    Usage
    -----
    detector = FraudDetector.load("lightgbm")
    result   = detector.predict(transactions_df)
    """

    def __init__(self, model, scaler, encoders: dict, feature_names: list):
        self.model = model
        self.scaler = scaler
        self.encoders = encoders
        self.feature_names = feature_names

    # ── Persistence ──────────────────────────────────────────────────────────
    def save(self, name: str = "fraud_detector", directory: str = MODEL_DIR) -> None:
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"{name}.pkl")
        joblib.dump(self, path)
        print(f"[SAVED] FraudDetector → {path}")

    @classmethod
    def load(cls, name: str = "fraud_detector", directory: str = MODEL_DIR) -> "FraudDetector":
        path = os.path.join(directory, f"{name}.pkl")
        print(f"[INFO] Loading FraudDetector from {path}")
        return joblib.load(path)

    # ── Preprocessing (mirrors data_preprocessing.py) ─────────────────────────
    def _engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["fee_ratio"]          = df["fee_amount"] / (df["transaction_amount"] + 1e-6)
        df["amount_per_acc_age"] = df["transaction_amount"] / (df["user_account_age_days"] + 1)
        df["is_weekend"]         = df["day_of_week"].isin(["Saturday", "Sunday"]).astype(int)
        df["is_high_risk_hour"]  = df["hour"].between(0, 5).astype(int)
        df["log_amount"]         = np.log1p(df["transaction_amount"])
        df["log_time_diff"]      = np.log1p(df["time_diff"])
        df["no_otp_high_amount"] = (
            (df["otp_used"] == 0) & (df["transaction_amount"] > 2000)
        ).astype(int)
        df["unverified_new_user"] = (
            (df["kyc_verified"] == 0) & (df["user_account_age_days"] < 90)
        ).astype(int)
        return df

    def _encode(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col, le in self.encoders.items():
            if col in df.columns:
                known = set(le.classes_)
                df[col] = df[col].astype(str).apply(
                    lambda x: x if x in known else le.classes_[0]
                )
                df[col] = le.transform(df[col])
        return df

    # ── Public Prediction API ─────────────────────────────────────────────────
    def predict(
        self,
        df: pd.DataFrame,
        threshold: float = 0.5,
        return_proba: bool = True,
    ) -> pd.DataFrame:
        """
        Run fraud detection on raw transaction DataFrame.

        Parameters
        ----------
        df            : Raw transactions (same schema as training data).
        threshold     : Decision threshold for binary label.
        return_proba  : Include fraud probability column in output.

        Returns
        -------
        DataFrame with columns: transaction_id, fraud_score, is_fraud_predicted
        """
        df_proc = self._engineer(df)
        df_proc = self._encode(df_proc)

        DROP = [
            "transaction_id", "user_id", "organization",
            "transaction_timestamp", "currency", "country", "city", "is_fraud",
        ]
        drop_cols = [c for c in DROP if c in df_proc.columns]
        X = df_proc.drop(columns=drop_cols)[self.feature_names]

        proba = self.model.predict_proba(X)[:, 1]
        pred  = (proba >= threshold).astype(int)

        output = pd.DataFrame({
            "transaction_id": df.get("transaction_id", pd.RangeIndex(len(df))),
            "fraud_score":    np.round(proba, 4),
            "is_fraud_predicted": pred,
        })

        if not return_proba:
            output.drop(columns=["fraud_score"], inplace=True)

        return output

    def predict_single(self, transaction: dict, threshold: float = 0.5) -> dict:
        """Predict a single transaction dict."""
        row = pd.DataFrame([transaction])
        result = self.predict(row, threshold=threshold)
        return {
            "transaction_id": result["transaction_id"].iloc[0],
            "fraud_score":    float(result["fraud_score"].iloc[0]),
            "is_fraud":       bool(result["is_fraud_predicted"].iloc[0]),
        }


# ─── CLI demo ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    # Demo: load detector and score 10 random rows
    try:
        detector = FraudDetector.load("fraud_detector")
    except FileNotFoundError:
        print("[ERROR] No saved detector found. Run main.py first.")
        sys.exit(1)

    sample_df = pd.read_csv("../data/improved_fraud_dataset.csv").sample(10, random_state=1)
    results = detector.predict(sample_df)
    print(results.to_string(index=False))
