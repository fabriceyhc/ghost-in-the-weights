"""
Cross-experiment visualization utilities.

Provides plotting functions for all five experiments with consistent styling.
"""

from typing import Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np


def _setup_style():
    """Apply consistent matplotlib style."""
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.figsize": (10, 6),
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "figure.dpi": 150,
    })


# ---------------------------------------------------------------------------
# Experiment 1: Temporal Persistence
# ---------------------------------------------------------------------------

def plot_decay_curves(
    results: Dict,
    probe_points: List[int],
    title: str = "Representation Decay Over Context Distance",
    save_path: Optional[str] = None,
):
    """
    Plot exponential decay curves for each condition.

    Args:
        results: Dict mapping condition name to {mean_similarities, std_similarities, fit_params}
        probe_points: Token distances
        title: Plot title
        save_path: If provided, save figure
    """
    import matplotlib.pyplot as plt
    _setup_style()

    fig, ax = plt.subplots()
    colors = {"self": "#e41a1c", "matched_ai": "#377eb8",
              "named_entity": "#4daf4a", "generic_fact": "#984ea3"}

    for condition, data in results.items():
        if condition.startswith("_"):
            continue
        color = colors.get(condition, "#999999")
        means = data["mean_similarities"]
        stds = data.get("std_similarities", [0] * len(means))

        ax.errorbar(probe_points[:len(means)], means, yerr=stds,
                     marker="o", label=f"{condition} (τ={data.get('mean_tau', 0):.0f})",
                     color=color, capsize=3)

        # Plot fitted curve if available
        fit = data.get("fit_params")
        if fit and fit.get("success"):
            from ..src.utils import exponential_decay
            x_fit = np.linspace(probe_points[0], probe_points[-1], 100)
            y_fit = exponential_decay(x_fit, fit["A"], fit["tau"], fit["C"])
            ax.plot(x_fit, y_fit, "--", color=color, alpha=0.5)

    ax.set_xlabel("Context Distance (tokens)")
    ax.set_ylabel("Cosine Similarity to Original Representation")
    ax.set_title(title)
    ax.legend()
    ax.set_xscale("log")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved: {save_path}")
    return fig


# ---------------------------------------------------------------------------
# Experiment 2: Grammatical vs Semantic
# ---------------------------------------------------------------------------

def plot_factorial_results(
    results: Dict,
    title: str = "Ghost Score Decomposition: Person x Referent",
    save_path: Optional[str] = None,
):
    """
    Plot 2x2 factorial results as grouped bar chart.

    Args:
        results: Dict with keys like '1p_self_ai', '2p_self_ai', etc.
    """
    import matplotlib.pyplot as plt
    _setup_style()

    fig, ax = plt.subplots()

    conditions = ["self_ai", "other_ai", "human"]
    x = np.arange(len(conditions))
    width = 0.35

    first_person = []
    second_person = []
    for cond in conditions:
        fp_key = f"1p_{cond}"
        sp_key = f"2p_{cond}"
        first_person.append(results.get(fp_key, {}).get("mean_slope", 0))
        second_person.append(results.get(sp_key, {}).get("mean_slope", 0))

    ax.bar(x - width / 2, first_person, width, label="1st person (I am...)",
           color="#e41a1c", alpha=0.8)
    ax.bar(x + width / 2, second_person, width, label="2nd person (You are...)",
           color="#377eb8", alpha=0.8)

    ax.set_xlabel("Referent")
    ax.set_ylabel("Restoration Slope")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(["Self (AI)", "Other (AI)", "Human"])
    ax.legend()
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Experiment 3: Self/Other Geometry
# ---------------------------------------------------------------------------

def plot_principal_angles(
    aligned_angles: List[float],
    divergent_angles: List[float],
    title: str = "Self/Other Subspace Angles",
    save_path: Optional[str] = None,
):
    """Plot principal angles for aligned vs divergent belief scenarios."""
    import matplotlib.pyplot as plt
    _setup_style()

    fig, ax = plt.subplots()

    x = np.arange(len(aligned_angles))
    ax.plot(x, sorted(aligned_angles, reverse=True), "o-",
            label="Aligned beliefs", color="#377eb8")
    ax.plot(x[:len(divergent_angles)], sorted(divergent_angles, reverse=True), "s-",
            label="Divergent beliefs", color="#e41a1c")

    ax.set_xlabel("Principal Angle Index")
    ax.set_ylabel("Angle (degrees)")
    ax.set_title(title)
    ax.legend()
    ax.set_ylim(0, 90)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Experiment 4: Behavioral Self-Recognition
# ---------------------------------------------------------------------------

def plot_attribution_accuracy(
    results: Dict,
    title: str = "Self-Attribution Accuracy by Difficulty",
    save_path: Optional[str] = None,
):
    """Plot accuracy vs chance for each difficulty level."""
    import matplotlib.pyplot as plt
    _setup_style()

    fig, ax = plt.subplots()

    difficulties = list(results.keys())
    accuracies = [results[d]["accuracy"] for d in difficulties]
    p_values = [results[d].get("p_value", 1.0) for d in difficulties]

    colors = ["#2ca02c" if p < 0.05 else "#d62728" for p in p_values]
    bars = ax.bar(difficulties, accuracies, color=colors, alpha=0.8)

    ax.axhline(y=1 / 3, color="gray", linestyle="--", label="Chance (1/3)")
    ax.set_xlabel("Difficulty Level")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.legend()

    # Add significance stars
    for i, (bar, p) in enumerate(zip(bars, p_values)):
        if p < 0.001:
            star = "***"
        elif p < 0.01:
            star = "**"
        elif p < 0.05:
            star = "*"
        else:
            star = "ns"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                star, ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Experiment 5: Metacognitive Uncertainty
# ---------------------------------------------------------------------------

def plot_attention_divergence(
    baseline_entropy: List[float],
    steered_entropy: List[float],
    layer_labels: Optional[List[str]] = None,
    title: str = "Attention Entropy: Baseline vs Uncertainty-Steered",
    save_path: Optional[str] = None,
):
    """Plot attention entropy comparison across layers."""
    import matplotlib.pyplot as plt
    _setup_style()

    fig, ax = plt.subplots()

    x = np.arange(len(baseline_entropy))
    if layer_labels is None:
        layer_labels = [str(i) for i in x]

    ax.plot(x, baseline_entropy, "o-", label="Baseline", color="#377eb8")
    ax.plot(x, steered_entropy, "s-", label="Uncertainty-steered", color="#e41a1c")

    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean Attention Entropy")
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Cross-experiment summary
# ---------------------------------------------------------------------------

def plot_model_comparison(
    model_results: Dict[str, Dict],
    metric: str,
    title: str = "Cross-Model Comparison",
    save_path: Optional[str] = None,
):
    """
    Plot a metric across all models as a grouped bar chart.

    Args:
        model_results: Dict mapping model_name to experiment results
        metric: Key to extract from each model's results
    """
    import matplotlib.pyplot as plt
    _setup_style()

    fig, ax = plt.subplots(figsize=(14, 6))

    models = list(model_results.keys())
    values = [model_results[m].get(metric, 0) for m in models]

    # Color by family
    colors = []
    for m in models:
        if "qwen" in m.lower():
            colors.append("#e41a1c")
        elif "gemma" in m.lower():
            colors.append("#377eb8")
        elif "llama" in m.lower():
            colors.append("#4daf4a")
        else:
            colors.append("#999999")

    ax.bar(range(len(models)), values, color=colors, alpha=0.8)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([m.split("/")[-1] for m in models], rotation=45, ha="right")
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    return fig
