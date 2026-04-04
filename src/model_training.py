"""
Model Training Module
=====================
Trains XGBoost, LightGBM, and a Logistic Regression baseline
for fraud detection with class-imbalance handling and evaluation.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, roc_auc_score, average_precision_score,
    confusion_matrix, roc_curve, precision_recall_curve,
    f1_score, ConfusionMatrixDisplay,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")


MODEL_DIR = "../models"
FIGURE_DIR = "../reports/figures"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)


# ─── Model Definitions ───────────────────────────────────────────────────────
def get_models(scale_pos_weight: float = 1.0) -> dict:
    """Return dict of unfitted models."""
    return {
        "LogisticRegression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            use_label_encoder=False,
            eval_metric="aucpr",
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        ),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        ),
    }


# ─── Training ────────────────────────────────────────────────────────────────
def train_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """
    Train all models, evaluate, and return results dict.

    Returns
    -------
    {model_name: {"model": fitted_model, "metrics": {...}, "y_pred_proba": array}}
    """
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"[INFO] scale_pos_weight = {scale_pos_weight:.2f}")

    models = get_models(scale_pos_weight)
    results = {}

    for name, model in models.items():
        print(f"\n{'='*55}")
        print(f"  Training: {name}")
        print(f"{'='*55}")

        # Cross-validation (AUC-ROC)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(
            model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1
        )
        print(f"  CV AUC-ROC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        # Final fit
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "cv_auc_mean": cv_scores.mean(),
            "cv_auc_std": cv_scores.std(),
            "test_auc_roc": roc_auc_score(y_test, y_proba),
            "test_avg_precision": average_precision_score(y_test, y_proba),
            "test_f1": f1_score(y_test, y_pred),
        }

        print(f"  Test AUC-ROC  : {metrics['test_auc_roc']:.4f}")
        print(f"  Test Avg Prec : {metrics['test_avg_precision']:.4f}")
        print(f"  Test F1-Score : {metrics['test_f1']:.4f}")
        print(f"\n{classification_report(y_test, y_pred, target_names=['Legit','Fraud'])}")

        results[name] = {
            "model": model,
            "metrics": metrics,
            "y_pred": y_pred,
            "y_pred_proba": y_proba,
        }

        # Save model
        model_path = os.path.join(MODEL_DIR, f"{name.lower()}_model.pkl")
        joblib.dump(model, model_path)
        print(f"  [SAVED] {model_path}")

    return results


# ─── Evaluation Plots ─────────────────────────────────────────────────────────
def plot_roc_curves(results: dict, y_test: pd.Series, output_dir: str = FIGURE_DIR) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#3498DB", "#E74C3C", "#2ECC71"]

    for (name, res), color in zip(results.items(), colors):
        fpr, tpr, _ = roc_curve(y_test, res["y_pred_proba"])
        auc = res["metrics"]["test_auc_roc"]
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.4f})", color=color, linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random (AUC=0.5)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves – Model Comparison", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, "08_roc_curves", output_dir)


def plot_pr_curves(results: dict, y_test: pd.Series, output_dir: str = FIGURE_DIR) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#3498DB", "#E74C3C", "#2ECC71"]
    baseline = y_test.mean()

    for (name, res), color in zip(results.items(), colors):
        prec, rec, _ = precision_recall_curve(y_test, res["y_pred_proba"])
        ap = res["metrics"]["test_avg_precision"]
        ax.plot(rec, prec, label=f"{name} (AP={ap:.4f})", color=color, linewidth=2)

    ax.axhline(baseline, color="black", linestyle="--", linewidth=1,
               label=f"Random baseline ({baseline:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves – Model Comparison", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, "09_pr_curves", output_dir)


def plot_confusion_matrices(results: dict, y_test: pd.Series, output_dir: str = FIGURE_DIR) -> None:
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]
    fig.suptitle("Confusion Matrices", fontsize=14, fontweight="bold")

    for ax, (name, res) in zip(axes, results.items()):
        cm = confusion_matrix(y_test, res["y_pred"])
        disp = ConfusionMatrixDisplay(cm, display_labels=["Legit", "Fraud"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(name)

    plt.tight_layout()
    _save(fig, "10_confusion_matrices", output_dir)


def plot_metrics_comparison(results: dict, output_dir: str = FIGURE_DIR) -> None:
    metric_keys = ["test_auc_roc", "test_avg_precision", "test_f1"]
    metric_labels = ["AUC-ROC", "Avg Precision", "F1-Score"]
    names = list(results.keys())

    x = np.arange(len(metric_keys))
    width = 0.25
    colors = ["#3498DB", "#E74C3C", "#2ECC71"]

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (name, color) in enumerate(zip(names, colors)):
        vals = [results[name]["metrics"][k] for k in metric_keys]
        bars = ax.bar(x + i * width, vals, width, label=name, color=color, alpha=0.85, edgecolor="white")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", fontsize=9)

    ax.set_xticks(x + width)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.1)
    ax.set_title("Model Performance Comparison", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    _save(fig, "11_metrics_comparison", output_dir)


def _save(fig, name: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{name}.png")
    fig.savefig(path, bbox_inches="tight", dpi=120)
    print(f"[SAVED] {path}")
    plt.close(fig)


# ─── Entry Point ─────────────────────────────────────────────────────────────
def run_training_pipeline(data: dict) -> dict:
    """
    Accept the output of data_preprocessing.preprocess() and run full training.
    Returns results dict.
    """
    results = train_models(
        data["X_train"], data["y_train"],
        data["X_test"],  data["y_test"],
    )

    plot_roc_curves(results, data["y_test"])
    plot_pr_curves(results, data["y_test"])
    plot_confusion_matrices(results, data["y_test"])
    plot_metrics_comparison(results)

    return results


if __name__ == "__main__":
    from data_preprocessing import preprocess
    data = preprocess("../data/improved_fraud_dataset.csv")
    results = run_training_pipeline(data)
    print("\n[DONE] Training pipeline complete.")
