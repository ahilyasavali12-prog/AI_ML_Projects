"""
evaluation/plots.py
All 6 research graphs for the paper.
Call after experiments complete.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional
from sklearn.metrics import roc_curve

# ── Style ──────────────────────────────────────────────────────────────────────
COLORS    = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0"]
METHODS   = ["FedAvg", "FedProx", "SCAFFOLD", "PerFedHypID", "Adaptive FL"]
ALPHAS    = ["IID", "α=1.0", "α=0.5", "α=0.1"]
PLOT_DIR  = Path("./results/plots")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3
})


# ── 1. Convergence Curve ───────────────────────────────────────────────────────
def plot_convergence_curves(results_per_condition: Dict, save: bool = True):
    """
    Line plot of AUC-ROC over communication rounds.
    One line per method × condition combination.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    conditions = [("IID", "solid"), ("Non-IID α=0.1", "dashed")]

    for ax, (cond, ls) in zip(axes, conditions):
        for i, method in enumerate(METHODS):
            key = f"{method}_{cond}"
            if key not in results_per_condition:
                continue
            rounds_data = results_per_condition[key]["rounds"]
            rounds = [r["round"] for r in rounds_data]
            aucs   = [r["metrics"]["auc_roc"] for r in rounds_data]
            ax.plot(rounds, aucs, color=COLORS[i], linestyle=ls,
                    label=method, linewidth=2)

        ax.set_title(f"Convergence — {cond}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Communication Rounds")
        ax.set_ylabel("AUC-ROC")
        ax.set_ylim(0.4, 1.0)
        ax.legend(fontsize=9)

    plt.suptitle("AUC-ROC Convergence: IID vs. Non-IID", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(fig, "1_convergence_curves.png", save)


# ── 2. Bar Chart — F1 Score Comparison ────────────────────────────────────────
def plot_f1_comparison(f1_scores: Dict, save: bool = True):
    """
    Grouped bar chart: F1 score per method, grouped by data condition.
    f1_scores = {method: {condition: f1_value}}
    """
    n_methods    = len(METHODS)
    n_conditions = len(ALPHAS)
    x = np.arange(n_conditions)
    width = 0.15

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, method in enumerate(METHODS):
        vals = [f1_scores.get(method, {}).get(cond, 0) for cond in ALPHAS]
        bars = ax.bar(x + i * width, vals, width, label=method,
                      color=COLORS[i], alpha=0.85, edgecolor="white")

    ax.set_xlabel("Data Condition", fontsize=12)
    ax.set_ylabel("Macro F1 Score", fontsize=12)
    ax.set_title("F1 Score: All Methods × All Conditions", fontsize=13, fontweight="bold")
    ax.set_xticks(x + width * (n_methods - 1) / 2)
    ax.set_xticklabels(ALPHAS)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9, loc="lower left")
    plt.tight_layout()
    _save(fig, "2_f1_comparison.png", save)


# ── 3. Heatmap — AUC-ROC vs Heterogeneity ─────────────────────────────────────
def plot_heatmap(auc_matrix: np.ndarray, save: bool = True):
    """
    Heatmap rows=methods, cols=conditions, cells=AUC-ROC.
    auc_matrix shape: (n_methods, n_conditions)
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(
        auc_matrix,
        annot=True, fmt=".3f", cmap="RdYlGn",
        xticklabels=ALPHAS, yticklabels=METHODS,
        vmin=0.5, vmax=1.0,
        linewidths=0.5, linecolor="white",
        ax=ax, cbar_kws={"label": "AUC-ROC"}
    )
    ax.set_title("AUC-ROC Heatmap: Methods × Heterogeneity Level",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Data Condition (Dirichlet α)")
    ax.set_ylabel("Aggregation Method")
    plt.tight_layout()
    _save(fig, "3_heatmap_auc.png", save)


# ── 4. ROC Curves ─────────────────────────────────────────────────────────────
def plot_roc_curves(
    errors_dict: Dict,     # {method: reconstruction_errors_array}
    y_true: np.ndarray,
    condition: str = "Non-IID α=0.1",
    save: bool = True
):
    """
    ROC curve per method for a given data condition.
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    for i, method in enumerate(METHODS):
        if method not in errors_dict:
            continue
        fpr, tpr, _ = roc_curve(y_true, errors_dict[method])
        auc = np.trapz(tpr, fpr)
        ax.plot(fpr, tpr, color=COLORS[i], linewidth=2,
                label=f"{method} (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random (AUC=0.5)")
    ax.set_xlabel("False Alarm Rate", fontsize=12)
    ax.set_ylabel("Detection Rate", fontsize=12)
    ax.set_title(f"ROC Curves — {condition}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    plt.tight_layout()
    _save(fig, "4_roc_curves.png", save)


# ── 5. Box Plot — Stability Across Seeds ──────────────────────────────────────
def plot_stability_boxplot(f1_by_seed: Dict, save: bool = True):
    """
    Box plot showing F1 score distribution across random seeds.
    f1_by_seed = {method: [f1_seed1, f1_seed2, f1_seed3]}
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    data    = [f1_by_seed.get(m, [0]) for m in METHODS]

    bp = ax.boxplot(data, patch_artist=True, notch=False,
                    medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp["boxes"], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xticklabels(METHODS, fontsize=10)
    ax.set_ylabel("Macro F1 Score", fontsize=12)
    ax.set_title("Model Stability Across Random Seeds (Non-IID α=0.1)",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    _save(fig, "5_stability_boxplot.png", save)


# ── 6. Bar Chart — Convergence Rounds ─────────────────────────────────────────
def plot_convergence_rounds(conv_rounds: Dict, save: bool = True):
    """
    Bar chart of rounds needed to converge per method × condition.
    conv_rounds = {method: {condition: n_rounds}}
    """
    n_methods    = len(METHODS)
    n_conditions = len(ALPHAS)
    x = np.arange(n_conditions)
    width = 0.15

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, method in enumerate(METHODS):
        vals = [conv_rounds.get(method, {}).get(cond, 100) for cond in ALPHAS]
        ax.bar(x + i * width, vals, width, label=method,
               color=COLORS[i], alpha=0.85, edgecolor="white")

    ax.set_xlabel("Data Condition", fontsize=12)
    ax.set_ylabel("Rounds to Convergence", fontsize=12)
    ax.set_title("Convergence Speed: Fewer Rounds = Better",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x + width * (n_methods - 1) / 2)
    ax.set_xticklabels(ALPHAS)
    ax.legend(fontsize=9)
    plt.tight_layout()
    _save(fig, "6_convergence_rounds.png", save)


# ── Utility ────────────────────────────────────────────────────────────────────
def _save(fig, filename: str, save: bool):
    if save:
        path = PLOT_DIR / filename
        fig.savefig(path, bbox_inches="tight")
        print(f"[Plot] Saved: {path}")
    plt.show()
    plt.close(fig)


def generate_all_plots(results: Dict):
    """
    Generate all 6 plots from a completed experiment results dict.
    Call this after running all conditions.
    """
    print("Generating all research plots...")
    # These will be populated from your actual results
    # Placeholder calls — replace with real data after running experiments
    print("Plots will be generated once experiment results are available.")
    print("See each plot function above for the expected input format.")
