"""
Exploratory Data Analysis (EDA) Module
=======================================
Generates all visualizations for the fraud detection project.
Run standalone or import individual functions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os

# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 120,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
})
PALETTE = {"fraud": "#E74C3C", "legit": "#2ECC71"}
FRAUD_COLOR, LEGIT_COLOR = "#E74C3C", "#2ECC71"


def save_fig(fig, name: str, output_dir: str = "../reports/figures") -> None:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{name}.png")
    fig.savefig(path, bbox_inches="tight")
    print(f"[SAVED] {path}")
    plt.close(fig)


# ── 1. Class Distribution ─────────────────────────────────────────────────────
def plot_class_distribution(df: pd.DataFrame, output_dir: str = "../reports/figures") -> None:
    counts = df["is_fraud"].value_counts()
    labels = ["Legitimate", "Fraud"]
    colors = [LEGIT_COLOR, FRAUD_COLOR]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Transaction Class Distribution", fontsize=14, fontweight="bold")

    # Bar chart
    axes[0].bar(labels, counts.values, color=colors, edgecolor="white", linewidth=1.5)
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 5000, f"{v:,}\n({v/len(df):.1%})", ha="center", fontsize=10)
    axes[0].set_ylabel("Count")
    axes[0].set_title("Count of Transactions")

    # Pie chart
    axes[1].pie(counts.values, labels=labels, colors=colors,
                autopct="%1.2f%%", startangle=140,
                wedgeprops={"edgecolor": "white", "linewidth": 2})
    axes[1].set_title("Proportion")

    save_fig(fig, "01_class_distribution", output_dir)


# ── 2. Transaction Amount Distribution ───────────────────────────────────────
def plot_amount_distribution(df: pd.DataFrame, output_dir: str = "../reports/figures") -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Transaction Amount by Fraud Label", fontsize=14, fontweight="bold")

    for label, color, name in [(0, LEGIT_COLOR, "Legitimate"), (1, FRAUD_COLOR, "Fraud")]:
        subset = df[df["is_fraud"] == label]["transaction_amount"]
        axes[0].hist(subset, bins=60, alpha=0.6, color=color, label=name, density=True)

    axes[0].set_xlabel("Transaction Amount")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Distribution (raw)")
    axes[0].legend()

    # Log scale
    for label, color, name in [(0, LEGIT_COLOR, "Legitimate"), (1, FRAUD_COLOR, "Fraud")]:
        subset = np.log1p(df[df["is_fraud"] == label]["transaction_amount"])
        axes[1].hist(subset, bins=60, alpha=0.6, color=color, label=name, density=True)

    axes[1].set_xlabel("log(1 + Transaction Amount)")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Distribution (log scale)")
    axes[1].legend()

    save_fig(fig, "02_amount_distribution", output_dir)


# ── 3. Fraud Rate by Categorical Feature ─────────────────────────────────────
def plot_fraud_rate_by_category(df: pd.DataFrame, output_dir: str = "../reports/figures") -> None:
    cat_cols = [
        "payment_method", "card_type", "device_type",
        "merchant_category", "transaction_type",
        "day_of_week", "operating_system",
    ]
    cat_cols = [c for c in cat_cols if c in df.columns]

    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    axes = axes.flatten()
    fig.suptitle("Fraud Rate by Categorical Feature", fontsize=15, fontweight="bold", y=1.01)

    for i, col in enumerate(cat_cols):
        rates = df.groupby(col)["is_fraud"].mean().sort_values(ascending=False)
        colors = [FRAUD_COLOR if r > df["is_fraud"].mean() else "#3498DB" for r in rates]
        axes[i].bar(rates.index, rates.values * 100, color=colors, edgecolor="white")
        axes[i].axhline(df["is_fraud"].mean() * 100, color="black", linestyle="--", linewidth=1, label="Overall avg")
        axes[i].set_title(col.replace("_", " ").title())
        axes[i].set_ylabel("Fraud Rate (%)")
        axes[i].tick_params(axis="x", rotation=25)
        axes[i].legend(fontsize=8)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    save_fig(fig, "03_fraud_rate_by_category", output_dir)


# ── 4. Temporal Analysis ─────────────────────────────────────────────────────
def plot_temporal_patterns(df: pd.DataFrame, output_dir: str = "../reports/figures") -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Temporal Fraud Patterns", fontsize=14, fontweight="bold")

    # By hour
    hourly = df.groupby("hour")["is_fraud"].mean() * 100
    axes[0].plot(hourly.index, hourly.values, marker="o", color=FRAUD_COLOR, linewidth=2)
    axes[0].fill_between(hourly.index, hourly.values, alpha=0.2, color=FRAUD_COLOR)
    axes[0].set_xlabel("Hour of Day")
    axes[0].set_ylabel("Fraud Rate (%)")
    axes[0].set_title("Fraud Rate by Hour of Day")
    axes[0].set_xticks(range(0, 24))

    # By day of week
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    daily = df.groupby("day_of_week")["is_fraud"].mean().reindex(day_order) * 100
    colors = [FRAUD_COLOR if d in ["Saturday", "Sunday"] else "#3498DB" for d in day_order]
    axes[1].bar(day_order, daily.values, color=colors, edgecolor="white")
    axes[1].set_xlabel("Day of Week")
    axes[1].set_ylabel("Fraud Rate (%)")
    axes[1].set_title("Fraud Rate by Day of Week")
    axes[1].tick_params(axis="x", rotation=30)

    save_fig(fig, "04_temporal_patterns", output_dir)


# ── 5. Correlation Heatmap ────────────────────────────────────────────────────
def plot_correlation_heatmap(df: pd.DataFrame, output_dir: str = "../reports/figures") -> None:
    num_cols = [
        "transaction_amount", "user_account_age_days",
        "fee_amount", "hour", "time_diff",
        "kyc_verified", "otp_used", "is_night", "is_fraud",
    ]
    num_cols = [c for c in num_cols if c in df.columns]
    corr = df[num_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="RdYlGn", center=0, vmin=-1, vmax=1,
        linewidths=0.5, ax=ax,
    )
    ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold", pad=15)
    save_fig(fig, "05_correlation_heatmap", output_dir)


# ── 6. KYC & OTP Analysis ────────────────────────────────────────────────────
def plot_security_signals(df: pd.DataFrame, output_dir: str = "../reports/figures") -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Security Signal Analysis", fontsize=14, fontweight="bold")

    for i, (col, title) in enumerate([("kyc_verified", "KYC Status"), ("otp_used", "OTP Usage")]):
        rates = df.groupby(col)["is_fraud"].mean() * 100
        labels = ["No", "Yes"]
        colors = [FRAUD_COLOR, LEGIT_COLOR]
        axes[i].bar(labels, rates.values, color=colors, edgecolor="white", linewidth=1.5)
        for j, v in enumerate(rates.values):
            axes[i].text(j, v + 0.2, f"{v:.2f}%", ha="center", fontweight="bold")
        axes[i].set_title(f"Fraud Rate by {title}")
        axes[i].set_ylabel("Fraud Rate (%)")

    save_fig(fig, "06_security_signals", output_dir)


# ── 7. Account Age vs Fraud ───────────────────────────────────────────────────
def plot_account_age_analysis(df: pd.DataFrame, output_dir: str = "../reports/figures") -> None:
    df = df.copy()
    df["age_bucket"] = pd.cut(
        df["user_account_age_days"],
        bins=[0, 90, 180, 365, 730, 2000],
        labels=["<3m", "3-6m", "6-12m", "1-2y", ">2y"],
    )

    rates = df.groupby("age_bucket", observed=True)["is_fraud"].mean() * 100
    counts = df.groupby("age_bucket", observed=True).size()

    fig, ax1 = plt.subplots(figsize=(10, 5))
    bars = ax1.bar(rates.index, rates.values, color="#3498DB", alpha=0.8, edgecolor="white")
    ax1.set_ylabel("Fraud Rate (%)", color="#3498DB")
    ax1.set_xlabel("Account Age Bucket")
    ax1.set_title("Fraud Rate by Account Age", fontsize=13, fontweight="bold")

    ax2 = ax1.twinx()
    ax2.plot(rates.index, counts.values, color=FRAUD_COLOR, marker="o", linewidth=2, label="# Transactions")
    ax2.set_ylabel("Transaction Count", color=FRAUD_COLOR)

    save_fig(fig, "07_account_age_analysis", output_dir)


# ── Run All ───────────────────────────────────────────────────────────────────
def run_full_eda(filepath: str, output_dir: str = "../reports/figures") -> None:
    print("[EDA] Loading data...")
    df = pd.read_csv(filepath)
    print(f"[EDA] {len(df):,} transactions loaded.\n")

    plot_class_distribution(df, output_dir)
    plot_amount_distribution(df, output_dir)
    plot_fraud_rate_by_category(df, output_dir)
    plot_temporal_patterns(df, output_dir)
    plot_correlation_heatmap(df, output_dir)
    plot_security_signals(df, output_dir)
    plot_account_age_analysis(df, output_dir)

    print("\n[EDA] All plots saved.")


if __name__ == "__main__":
    run_full_eda("../data/improved_fraud_dataset.csv")
