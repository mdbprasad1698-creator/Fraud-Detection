"""
Model Explainability Module (SHAP)
===================================
Generates SHAP-based feature importance and explanation plots
for the best-performing model (LightGBM by default).
"""

import os
import shap
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

FIGURE_DIR = "../reports/figures"
MODEL_DIR  = "../models"


def load_best_model(model_name: str = "lightgbm"):
    path = os.path.join(MODEL_DIR, f"{model_name}_model.pkl")
    print(f"[INFO] Loading model from: {path}")
    return joblib.load(path)


def compute_shap_values(
    model,
    X_sample: pd.DataFrame,
    model_name: str = "lightgbm",
) -> shap.Explanation:
    """Compute SHAP values. Returns a shap.Explanation object."""
    print(f"[INFO] Computing SHAP values for {len(X_sample):,} samples ...")

    if model_name.lower() in ("lightgbm", "xgboost"):
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.LinearExplainer(model, X_sample)

    shap_values = explainer(X_sample)

    # For binary classifiers, keep class-1 (fraud) SHAP values
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif shap_values.values.ndim == 3:
        shap_values = shap.Explanation(
            values=shap_values.values[:, :, 1],
            base_values=shap_values.base_values[:, 1],
            data=shap_values.data,
            feature_names=shap_values.feature_names,
        )

    print("[INFO] SHAP computation complete.")
    return shap_values


def plot_shap_summary(
    shap_values: shap.Explanation,
    output_dir: str = FIGURE_DIR,
) -> None:
    """Beeswarm summary plot."""
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.plots.beeswarm(shap_values, max_display=20, show=False)
    plt.title("SHAP Summary – Feature Impact on Fraud Prediction", fontsize=13, fontweight="bold")
    path = os.path.join(output_dir, "12_shap_summary.png")
    plt.savefig(path, bbox_inches="tight", dpi=120)
    print(f"[SAVED] {path}")
    plt.close()


def plot_shap_bar(
    shap_values: shap.Explanation,
    output_dir: str = FIGURE_DIR,
) -> None:
    """Mean absolute SHAP bar chart."""
    os.makedirs(output_dir, exist_ok=True)
    mean_abs = pd.Series(
        np.abs(shap_values.values).mean(axis=0),
        index=shap_values.feature_names,
    ).sort_values(ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(mean_abs.index[::-1], mean_abs.values[::-1], color="#3498DB", edgecolor="white")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Top 15 Features by Mean Absolute SHAP Value", fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "13_shap_bar_importance.png")
    plt.savefig(path, bbox_inches="tight", dpi=120)
    print(f"[SAVED] {path}")
    plt.close(fig)


def plot_shap_dependence(
    shap_values: shap.Explanation,
    feature: str = "log_amount",
    output_dir: str = FIGURE_DIR,
) -> None:
    """Dependence plot for a single feature."""
    os.makedirs(output_dir, exist_ok=True)

    if feature not in shap_values.feature_names:
        print(f"[WARN] Feature '{feature}' not found. Skipping dependence plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    shap.plots.scatter(shap_values[:, feature], show=False)
    plt.title(f"SHAP Dependence – {feature}", fontsize=13, fontweight="bold")
    path = os.path.join(output_dir, f"14_shap_dependence_{feature}.png")
    plt.savefig(path, bbox_inches="tight", dpi=120)
    print(f"[SAVED] {path}")
    plt.close()


def run_explainability(
    model,
    X_test: pd.DataFrame,
    model_name: str = "lightgbm",
    n_samples: int = 5000,
    output_dir: str = FIGURE_DIR,
) -> shap.Explanation:
    """
    Full explainability pipeline.

    Parameters
    ----------
    model      : Fitted model object (or None to load from disk).
    X_test     : Test features DataFrame.
    model_name : Name for loading from disk if model is None.
    n_samples  : Number of random samples for SHAP (full set is slow).
    output_dir : Where to save plots.

    Returns
    -------
    shap_values : shap.Explanation object.
    """
    if model is None:
        model = load_best_model(model_name)

    # Sample for speed
    sample = X_test.sample(min(n_samples, len(X_test)), random_state=42)

    shap_values = compute_shap_values(model, sample, model_name)
    plot_shap_summary(shap_values, output_dir)
    plot_shap_bar(shap_values, output_dir)
    plot_shap_dependence(shap_values, "log_amount", output_dir)
    plot_shap_dependence(shap_values, "transaction_amount", output_dir)

    return shap_values


if __name__ == "__main__":
    from data_preprocessing import preprocess
    data = preprocess("../data/improved_fraud_dataset.csv")

    # Load best model
    model = load_best_model("lightgbm")
    run_explainability(model, data["X_test"], model_name="lightgbm")
    print("\n[DONE] Explainability pipeline complete.")
