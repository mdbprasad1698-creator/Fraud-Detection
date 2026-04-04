"""
Main Pipeline Orchestrator
===========================
Run the full fraud detection pipeline end-to-end.

Usage
-----
    python main.py                        # full pipeline
    python main.py --skip-eda            # skip EDA plots
    python main.py --skip-shap           # skip SHAP (faster)
    python main.py --data /path/to.csv   # custom data path
"""

import argparse
import os
import sys
import time

import joblib
import pandas as pd

# Local imports (run from project root)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_preprocessing import preprocess
from eda import run_full_eda
from model_training import run_training_pipeline
from explainability import run_explainability
from inference import FraudDetector


DATA_PATH   = "data/improved_fraud_dataset.csv"
MODEL_DIR   = "models"
REPORTS_DIR = "reports/figures"


def banner(text: str) -> None:
    width = 60
    print(f"\n{'='*width}")
    print(f"  {text}")
    print(f"{'='*width}\n")


def main():
    parser = argparse.ArgumentParser(description="Fraud Detection Pipeline")
    parser.add_argument("--data",       default=DATA_PATH, help="Path to CSV dataset")
    parser.add_argument("--skip-eda",   action="store_true", help="Skip EDA step")
    parser.add_argument("--skip-shap",  action="store_true", help="Skip SHAP step")
    parser.add_argument("--model",      default="LightGBM",
                        choices=["LogisticRegression", "XGBoost", "LightGBM"],
                        help="Best model for SHAP analysis")
    args = parser.parse_args()

    total_start = time.time()

    # ── 1. EDA ──────────────────────────────────────────────────────────────
    if not args.skip_eda:
        banner("Step 1 – Exploratory Data Analysis")
        run_full_eda(args.data, output_dir=REPORTS_DIR)
    else:
        print("[SKIP] EDA skipped.\n")

    # ── 2. Preprocessing ─────────────────────────────────────────────────────
    banner("Step 2 – Data Preprocessing & Feature Engineering")
    data = preprocess(args.data)

    # ── 3. Model Training ────────────────────────────────────────────────────
    banner("Step 3 – Model Training & Evaluation")
    results = run_training_pipeline(data)

    # ── 4. Save Best Detector ────────────────────────────────────────────────
    banner("Step 4 – Persisting Best Model Detector")
    best_name = args.model
    best_model = results[best_name]["model"]
    detector = FraudDetector(
        model=best_model,
        scaler=data["scaler"],
        encoders=data["encoders"],
        feature_names=data["feature_names"],
    )
    detector.save("fraud_detector", directory=MODEL_DIR)

    # ── 5. SHAP Explainability ───────────────────────────────────────────────
    if not args.skip_shap:
        banner("Step 5 – SHAP Model Explainability")
        run_explainability(
            model=best_model,
            X_test=data["X_test"],
            model_name=best_name.lower(),
            output_dir=REPORTS_DIR,
        )
    else:
        print("[SKIP] SHAP skipped.\n")

    # ── 6. Summary ───────────────────────────────────────────────────────────
    banner("Pipeline Complete – Results Summary")
    metrics_rows = []
    for name, res in results.items():
        m = res["metrics"]
        metrics_rows.append({
            "Model": name,
            "CV AUC-ROC": f"{m['cv_auc_mean']:.4f} ± {m['cv_auc_std']:.4f}",
            "Test AUC-ROC": f"{m['test_auc_roc']:.4f}",
            "Avg Precision": f"{m['test_avg_precision']:.4f}",
            "F1-Score": f"{m['test_f1']:.4f}",
        })

    summary = pd.DataFrame(metrics_rows)
    print(summary.to_string(index=False))

    elapsed = time.time() - total_start
    print(f"\nTotal runtime: {elapsed/60:.1f} minutes")
    print(f"Figures saved to : {REPORTS_DIR}/")
    print(f"Models saved to  : {MODEL_DIR}/")


if __name__ == "__main__":
    main()
