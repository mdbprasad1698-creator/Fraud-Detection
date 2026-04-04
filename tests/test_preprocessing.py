"""
Unit Tests – Fraud Detection Project
======================================
Run with: pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import pytest

from data_preprocessing import (
    engineer_features,
    encode_features,
    CATEGORICAL_COLS,
    NUMERIC_COLS,
    TARGET,
)


# ─── Fixtures ────────────────────────────────────────────────────────────────
@pytest.fixture
def sample_df():
    """Minimal synthetic DataFrame matching the dataset schema."""
    n = 200
    rng = np.random.default_rng(42)

    return pd.DataFrame({
        "organization":            ["OrgA"] * n,
        "transaction_id":          [f"TXN{i:05d}" for i in range(n)],
        "user_id":                 [f"U{i:04d}" for i in range(n)],
        "user_account_age_days":   rng.integers(10, 2000, n),
        "kyc_verified":            rng.integers(0, 2, n),
        "payment_method":          rng.choice(["bkash", "nagad", "rocket", "card"], n),
        "card_type":               rng.choice(["debit", "credit"], n),
        "transaction_amount":      rng.uniform(10, 30000, n),
        "currency":                ["BDT"] * n,
        "fee_amount":              rng.uniform(0, 600, n),
        "transaction_timestamp":   pd.date_range("2024-01-01", periods=n, freq="h"),
        "day_of_week":             rng.choice(
            ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"], n
        ),
        "city":                    ["Dhaka"] * n,
        "country":                 ["Bangladesh"] * n,
        "device_type":             rng.choice(["mobile", "desktop", "tablet"], n),
        "operating_system":        rng.choice(["Android", "iOS", "Windows"], n),
        "browser":                 rng.choice(["Chrome", "Safari", "Edge"], n),
        "merchant_category":       rng.choice(["grocery", "fashion", "electronics", "travel"], n),
        "transaction_type":        rng.choice(["purchase", "withdrawal", "transfer"], n),
        "otp_used":                rng.integers(0, 2, n),
        "is_fraud":                rng.integers(0, 2, n),
        "hour":                    rng.integers(0, 24, n),
        "is_night":                rng.integers(0, 2, n),
        "time_diff":               rng.uniform(0, 1e7, n),
    })


# ─── Feature Engineering Tests ───────────────────────────────────────────────
class TestEngineerFeatures:

    def test_new_columns_created(self, sample_df):
        out = engineer_features(sample_df)
        expected = [
            "fee_ratio", "amount_per_acc_age", "is_weekend",
            "is_high_risk_hour", "log_amount", "log_time_diff",
            "no_otp_high_amount", "unverified_new_user",
        ]
        for col in expected:
            assert col in out.columns, f"Missing column: {col}"

    def test_no_original_columns_dropped(self, sample_df):
        original_cols = set(sample_df.columns)
        out = engineer_features(sample_df)
        for col in original_cols:
            assert col in out.columns

    def test_fee_ratio_non_negative(self, sample_df):
        out = engineer_features(sample_df)
        assert (out["fee_ratio"] >= 0).all()

    def test_log_amount_non_negative(self, sample_df):
        out = engineer_features(sample_df)
        assert (out["log_amount"] >= 0).all()

    def test_binary_flags_are_binary(self, sample_df):
        out = engineer_features(sample_df)
        for col in ["is_weekend", "is_high_risk_hour", "no_otp_high_amount", "unverified_new_user"]:
            assert set(out[col].unique()).issubset({0, 1}), f"{col} has non-binary values"

    def test_is_weekend_correct(self, sample_df):
        out = engineer_features(sample_df)
        weekend_mask = out["day_of_week"].isin(["Saturday", "Sunday"])
        assert (out.loc[weekend_mask, "is_weekend"] == 1).all()
        assert (out.loc[~weekend_mask, "is_weekend"] == 0).all()

    def test_no_mutation_of_input(self, sample_df):
        original_cols = list(sample_df.columns)
        engineer_features(sample_df)
        assert list(sample_df.columns) == original_cols


# ─── Encoding Tests ───────────────────────────────────────────────────────────
class TestEncodeFeatures:

    def test_categorical_cols_are_numeric(self, sample_df):
        out_df, _ = encode_features(sample_df, fit=True)
        cat_cols = [c for c in CATEGORICAL_COLS if c in out_df.columns]
        for col in cat_cols:
            assert pd.api.types.is_numeric_dtype(out_df[col]), f"{col} not numeric after encoding"

    def test_encoders_dict_populated(self, sample_df):
        _, encoders = encode_features(sample_df, fit=True)
        cat_cols = [c for c in CATEGORICAL_COLS if c in sample_df.columns]
        for col in cat_cols:
            assert col in encoders, f"Encoder missing for {col}"

    def test_inference_mode_handles_unseen_labels(self, sample_df):
        _, encoders = encode_features(sample_df, fit=True)

        # Create a row with an unseen payment method
        new_row = sample_df.iloc[:5].copy()
        new_row["payment_method"] = "UNKNOWN_METHOD"

        # Should not raise
        out_df, _ = encode_features(new_row, encoders=encoders, fit=False)
        assert pd.api.types.is_numeric_dtype(out_df["payment_method"])

    def test_no_mutation_of_input(self, sample_df):
        original_payment = sample_df["payment_method"].copy()
        encode_features(sample_df, fit=True)
        assert (sample_df["payment_method"] == original_payment).all()


# ─── Data Integrity Tests ─────────────────────────────────────────────────────
class TestDataIntegrity:

    def test_fraud_label_is_binary(self, sample_df):
        assert set(sample_df[TARGET].unique()).issubset({0, 1})

    def test_transaction_amount_non_negative(self, sample_df):
        assert (sample_df["transaction_amount"] >= 0).all()

    def test_hour_in_valid_range(self, sample_df):
        assert sample_df["hour"].between(0, 23).all()

    def test_no_all_null_columns(self, sample_df):
        for col in sample_df.columns:
            assert not sample_df[col].isnull().all(), f"Column '{col}' is all nulls"


# ─── Run standalone ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
