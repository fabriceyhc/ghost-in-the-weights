"""
Follow-up analyses addressing reviewer critique.

Extracts R² fit quality (Exp 1), effect sizes (all experiments),
cross-experiment correlations, and layer-by-layer KL profiles (Exp 5).

Usage:
    python scripts/followup_analysis.py [--output-dir artifacts/followup]
    python scripts/followup_analysis.py --layer-kl  # requires Exp 5 re-run data
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RESULTS_DIR = Path("data/results")
PRIMARY_MODELS = [
    "Qwen_Qwen3-0.6B", "Qwen_Qwen3-0.6B-Base",
    "Qwen_Qwen3-1.7B", "Qwen_Qwen3-1.7B-Base",
    "Qwen_Qwen3-4B", "Qwen_Qwen3-4B-Base",
    "Qwen_Qwen3-8B", "Qwen_Qwen3-8B-Base",
    "Qwen_Qwen3-14B", "Qwen_Qwen3-14B-Base",
    "meta-llama_Llama-3.2-1B", "meta-llama_Llama-3.2-1B-Instruct",
    "meta-llama_Llama-3.2-3B", "meta-llama_Llama-3.2-3B-Instruct",
    "meta-llama_Llama-3.1-8B", "meta-llama_Llama-3.1-8B-Instruct",
]


def load_json(path: Path) -> Optional[Dict]:
    """Load JSON file, return None on failure."""
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"  Warning: cannot load {path}: {e}")
        return None


def short_name(model_key: str) -> str:
    """Convert 'Qwen_Qwen3-8B' to 'Qwen3-8B'."""
    return model_key.split("_")[-1]


def cohens_d(group1, group2) -> float:
    """Compute Cohen's d effect size."""
    g1, g2 = np.array(group1, dtype=float), np.array(group2, dtype=float)
    pooled_std = np.sqrt((np.var(g1) + np.var(g2)) / 2)
    if pooled_std < 1e-10:
        return 0.0
    return float((np.mean(g1) - np.mean(g2)) / pooled_std)


def write_csv(rows: List[List], header: List[str], path: Path) -> None:
    """Write a simple CSV file."""
    with open(path, "w") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(str(v) for v in row) + "\n")
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Exp 1: R² Validation
# ---------------------------------------------------------------------------

def analyze_exp1_r_squared(output_dir: Path) -> List[List]:
    """
    Extract per-statement R² from Exp 1 results.
    Report mean R² per model per condition. Flag poor fits (mean R² < 0.2).
    """
    print("\n=== Exp 1: R² Fit Quality ===")
    header = [
        "Model", "Condition", "N_statements", "N_successful_fits",
        "Mean_R2", "Median_R2", "Min_R2", "Max_R2",
        "Pct_R2_below_0.2", "Flag"
    ]
    rows = []
    summary_rows = []

    for model_key in PRIMARY_MODELS:
        path = RESULTS_DIR / "exp1_temporal" / f"exp1_temporal_{model_key}.json"
        data = load_json(path)
        if data is None:
            continue

        name = short_name(model_key)
        model_r2_all = []

        for condition in ["self", "matched_ai", "named_entity", "generic_fact"]:
            cond_data = data.get("conditions", {}).get(condition, {})
            per_statement = cond_data.get("per_statement", [])

            r2_values = []
            n_successful = 0
            for stmt in per_statement:
                fp = stmt.get("fit_params", {})
                r2 = fp.get("r_squared", 0.0)
                success = fp.get("success", False)
                if success:
                    n_successful += 1
                    r2_values.append(r2)

            if not r2_values:
                rows.append([name, condition, len(per_statement), 0,
                             "N/A", "N/A", "N/A", "N/A", "N/A", "NO_FITS"])
                continue

            r2_arr = np.array(r2_values)
            mean_r2 = float(np.mean(r2_arr))
            median_r2 = float(np.median(r2_arr))
            min_r2 = float(np.min(r2_arr))
            max_r2 = float(np.max(r2_arr))
            pct_below = float(np.mean(r2_arr < 0.2) * 100)
            flag = "POOR" if mean_r2 < 0.2 else ("MARGINAL" if mean_r2 < 0.5 else "OK")

            rows.append([
                name, condition, len(per_statement), n_successful,
                f"{mean_r2:.4f}", f"{median_r2:.4f}", f"{min_r2:.4f}",
                f"{max_r2:.4f}", f"{pct_below:.1f}%", flag
            ])
            model_r2_all.extend(r2_values)

        if model_r2_all:
            overall_mean = np.mean(model_r2_all)
            overall_flag = "POOR" if overall_mean < 0.2 else (
                "MARGINAL" if overall_mean < 0.5 else "OK")
            summary_rows.append([name, f"{overall_mean:.4f}", overall_flag,
                                 len(model_r2_all)])

    write_csv(rows, header, output_dir / "exp1_r_squared_detail.csv")
    write_csv(summary_rows,
              ["Model", "Overall_Mean_R2", "Flag", "N_fits"],
              output_dir / "exp1_r_squared_summary.csv")

    # Print summary table
    print(f"\n  {'Model':<25} {'Mean R²':>10} {'Flag':>10} {'N':>5}")
    print("  " + "-" * 55)
    for row in summary_rows:
        print(f"  {row[0]:<25} {row[1]:>10} {row[2]:>10} {row[3]:>5}")

    return summary_rows


# ---------------------------------------------------------------------------
# Effect Sizes — all experiments
# ---------------------------------------------------------------------------

def extract_effect_sizes(output_dir: Path) -> List[List]:
    """
    Extract Cohen's d from all experiments.
    - Exp 1: from comparisons[].cohens_d
    - Exp 2: from decomposition.statistical_tests.{person_test,referent_test}.d
    - Exp 3: compute from aligned vs divergent principal_angles
    - Exp 5: compute from per-prompt attention_divergence values vs zero
    """
    print("\n=== Effect Sizes (Cohen's d) ===")
    header = ["Model", "Exp", "Comparison", "Cohens_d", "Magnitude"]
    rows = []

    for model_key in PRIMARY_MODELS:
        name = short_name(model_key)

        # --- Exp 1 ---
        path = RESULTS_DIR / "exp1_temporal" / f"exp1_temporal_{model_key}.json"
        data = load_json(path)
        if data:
            comparisons = data.get("comparisons", {})
            for comp_name, comp_data in comparisons.items():
                d = comp_data.get("cohens_d", None)
                if d is not None:
                    d = float(d)
                    mag = _classify_d(d)
                    # Shorten comparison name
                    short_comp = comp_name.replace("(critical test)", "").strip()
                    rows.append([name, "Exp1", short_comp, f"{d:.4f}", mag])

        # --- Exp 2 ---
        path = RESULTS_DIR / "exp2_grammatical" / f"exp2_grammatical_{model_key}.json"
        data = load_json(path)
        if data:
            tests = data.get("decomposition", {}).get("statistical_tests", {})
            for test_name in ["person_test", "referent_test"]:
                test = tests.get(test_name, {})
                d = test.get("d", None)
                if d is not None:
                    d = float(d)
                    mag = _classify_d(d)
                    rows.append([name, "Exp2", test_name, f"{d:.4f}", mag])

        # --- Exp 3 ---
        path = RESULTS_DIR / "exp3_geometry" / f"exp3_geometry_{model_key}.json"
        data = load_json(path)
        if data:
            aligned_angles = data.get("aligned", {}).get("principal_angles", [])
            divergent_angles = data.get("divergent", {}).get("principal_angles", [])
            if aligned_angles and divergent_angles:
                d = cohens_d(divergent_angles, aligned_angles)
                mag = _classify_d(d)
                rows.append([name, "Exp3", "divergent_vs_aligned_angles",
                             f"{d:.4f}", mag])

        # --- Exp 5 ---
        path = RESULTS_DIR / "exp5_metacognitive" / f"exp5_metacognitive_{model_key}.json"
        data = load_json(path)
        if data:
            per_prompt = data.get("causation", {}).get("per_prompt", [])
            attn_divs = [p["attention_divergence"] for p in per_prompt
                         if "attention_divergence" in p]
            if attn_divs:
                # One-sample effect size vs zero
                arr = np.array(attn_divs)
                d_val = float(np.mean(arr) / max(np.std(arr), 1e-10))
                mag = _classify_d(d_val)
                rows.append([name, "Exp5", "attn_divergence_vs_zero",
                             f"{d_val:.4f}", mag])

    write_csv(rows, header, output_dir / "effect_sizes.csv")

    # Print summary
    print(f"\n  {'Model':<20} {'Exp':>5} {'Comparison':<30} {'d':>8} {'Mag':>8}")
    print("  " + "-" * 75)
    for row in rows:
        print(f"  {row[0]:<20} {row[1]:>5} {row[2]:<30} {row[3]:>8} {row[4]:>8}")

    return rows


def _classify_d(d: float) -> str:
    """Classify Cohen's d magnitude."""
    ad = abs(d)
    if ad < 0.2:
        return "negligible"
    elif ad < 0.5:
        return "small"
    elif ad < 0.8:
        return "medium"
    else:
        return "large"


# ---------------------------------------------------------------------------
# Cross-Experiment Correlation
# ---------------------------------------------------------------------------

def compute_cross_experiment_correlation(output_dir: Path) -> Optional[np.ndarray]:
    """
    For each model, extract one summary metric per experiment:
    - Exp 1: self_advantage = tau_self - tau_matched_ai
    - Exp 2: referent p-value (log10, so smaller = stronger effect)
    - Exp 3: divergence effect
    - Exp 4: mean accuracy across difficulties
    - Exp 5: behavior change rate at max strength

    Compute Spearman correlation matrix.
    """
    print("\n=== Cross-Experiment Correlation ===")
    metrics = {
        "exp1_self_advantage": [],
        "exp2_referent_neg_log_p": [],
        "exp3_divergence_effect": [],
        "exp4_mean_accuracy": [],
        "exp5_behavior_change": [],
    }
    model_names = []

    for model_key in PRIMARY_MODELS:
        name = short_name(model_key)
        vals = {}

        # Exp 1
        path = RESULTS_DIR / "exp1_temporal" / f"exp1_temporal_{model_key}.json"
        data = load_json(path)
        if data:
            conds = data.get("conditions", {})
            tau_self = conds.get("self", {}).get("mean_tau", np.nan)
            tau_ai = conds.get("matched_ai", {}).get("mean_tau", np.nan)
            if not (np.isnan(tau_self) or np.isnan(tau_ai)):
                vals["exp1_self_advantage"] = tau_self - tau_ai
            else:
                vals["exp1_self_advantage"] = np.nan
        else:
            vals["exp1_self_advantage"] = np.nan

        # Exp 2
        path = RESULTS_DIR / "exp2_grammatical" / f"exp2_grammatical_{model_key}.json"
        data = load_json(path)
        if data:
            p = data.get("decomposition", {}).get("statistical_tests", {}).get(
                "referent_test", {}).get("p", np.nan)
            if p and float(p) > 0:
                vals["exp2_referent_neg_log_p"] = -np.log10(float(p))
            else:
                vals["exp2_referent_neg_log_p"] = np.nan
        else:
            vals["exp2_referent_neg_log_p"] = np.nan

        # Exp 3
        path = RESULTS_DIR / "exp3_geometry" / f"exp3_geometry_{model_key}.json"
        data = load_json(path)
        if data:
            vals["exp3_divergence_effect"] = data.get("divergence_effect", np.nan)
        else:
            vals["exp3_divergence_effect"] = np.nan

        # Exp 4
        path = RESULTS_DIR / "exp4_behavioral" / f"exp4_behavioral_{model_key}.json"
        data = load_json(path)
        if data:
            diffs = data.get("difficulties", {})
            accs = [diffs[d]["accuracy"] for d in diffs
                    if isinstance(diffs[d], dict) and "accuracy" in diffs[d]]
            vals["exp4_mean_accuracy"] = float(np.mean(accs)) if accs else np.nan
        else:
            vals["exp4_mean_accuracy"] = np.nan

        # Exp 5
        path = RESULTS_DIR / "exp5_metacognitive" / f"exp5_metacognitive_{model_key}.json"
        data = load_json(path)
        if data:
            per_str = data.get("causation", {}).get("per_strength", {})
            if per_str:
                max_key = max(per_str.keys(), key=lambda x: float(x))
                vals["exp5_behavior_change"] = per_str[max_key].get(
                    "behavior_change_rate", np.nan)
            else:
                vals["exp5_behavior_change"] = np.nan
        else:
            vals["exp5_behavior_change"] = np.nan

        # Check we have at least some data
        has_data = any(not np.isnan(v) for v in vals.values())
        if has_data:
            model_names.append(name)
            for k, v in vals.items():
                metrics[k].append(v)

    if len(model_names) < 4:
        print("  Insufficient models with complete data for correlation.")
        return None

    # Build matrix
    metric_names = list(metrics.keys())
    n_metrics = len(metric_names)
    data_matrix = np.array([metrics[k] for k in metric_names])  # (n_metrics, n_models)

    # Spearman correlation with pairwise complete obs
    corr_matrix = np.full((n_metrics, n_metrics), np.nan)
    p_matrix = np.full((n_metrics, n_metrics), np.nan)

    for i in range(n_metrics):
        for j in range(n_metrics):
            x, y = data_matrix[i], data_matrix[j]
            mask = ~(np.isnan(x) | np.isnan(y))
            if mask.sum() >= 4:
                rho, p = stats.spearmanr(x[mask], y[mask])
                corr_matrix[i, j] = rho
                p_matrix[i, j] = p

    # Save raw metrics per model
    raw_header = ["Model"] + metric_names
    raw_rows = []
    for idx, name in enumerate(model_names):
        row = [name] + [f"{metrics[k][idx]:.4f}" if not np.isnan(metrics[k][idx])
                        else "NA" for k in metric_names]
        raw_rows.append(row)
    write_csv(raw_rows, raw_header, output_dir / "cross_experiment_metrics.csv")

    # Save correlation matrix
    corr_header = [""] + [k.replace("_", " ") for k in metric_names]
    corr_rows = []
    for i, k in enumerate(metric_names):
        row = [k.replace("_", " ")]
        for j in range(n_metrics):
            if np.isnan(corr_matrix[i, j]):
                row.append("NA")
            else:
                sig = "*" if p_matrix[i, j] < 0.05 else ""
                row.append(f"{corr_matrix[i, j]:.3f}{sig}")
        corr_rows.append(row)
    write_csv(corr_rows, corr_header, output_dir / "cross_experiment_correlation.csv")

    # Print
    short_labels = ["Exp1:self_adv", "Exp2:-logP", "Exp3:div_eff",
                    "Exp4:acc", "Exp5:behav"]
    print(f"\n  {'':>15}", end="")
    for sl in short_labels:
        print(f" {sl:>13}", end="")
    print()
    for i in range(n_metrics):
        print(f"  {short_labels[i]:>15}", end="")
        for j in range(n_metrics):
            if np.isnan(corr_matrix[i, j]):
                print(f" {'NA':>13}", end="")
            else:
                sig = "*" if p_matrix[i, j] < 0.05 else " "
                print(f" {corr_matrix[i, j]:>6.3f}{sig:>6}", end="")
        print()

    # Specific test: Qwen3-14B Exp 3 vs everything else
    print("\n  Qwen3-14B Exp 3 effect correlation analysis:")
    exp3_idx = metric_names.index("exp3_divergence_effect")
    for j, k in enumerate(metric_names):
        if j == exp3_idx:
            continue
        x, y = data_matrix[exp3_idx], data_matrix[j]
        mask = ~(np.isnan(x) | np.isnan(y))
        if mask.sum() >= 4:
            rho, p = stats.spearmanr(x[mask], y[mask])
            print(f"    Exp3 vs {k}: rho={rho:.3f}, p={p:.4f} "
                  f"{'*' if p < 0.05 else ''}")

    return corr_matrix


# ---------------------------------------------------------------------------
# Exp 1: Raw similarity analysis for poor-fit models
# ---------------------------------------------------------------------------

def analyze_exp1_raw_similarities(output_dir: Path) -> None:
    """
    For models with poor exponential fits, examine raw similarity values
    to determine if curves are flat (frozen representations) vs non-exponential decay.
    """
    print("\n=== Exp 1: Raw Similarity Profiles ===")
    header = [
        "Model", "Condition", "Sim_T500", "Sim_T1000", "Sim_T2000", "Sim_T5000",
        "Range", "Profile_Type", "Mean_R2"
    ]
    rows = []

    for model_key in PRIMARY_MODELS:
        path = RESULTS_DIR / "exp1_temporal" / f"exp1_temporal_{model_key}.json"
        data = load_json(path)
        if data is None:
            continue

        name = short_name(model_key)
        probe_points = data.get("probe_points", [])

        for condition in ["self", "matched_ai", "named_entity", "generic_fact"]:
            cond_data = data.get("conditions", {}).get(condition, {})
            mean_sims = cond_data.get("mean_similarities", [])

            if not mean_sims:
                continue

            # Pad to 4 if fewer probe points
            sims = mean_sims + [None] * (4 - len(mean_sims))
            sim_strs = [f"{s:.6f}" if s is not None else "NA" for s in sims]

            # Compute range (max - min of available)
            valid_sims = [s for s in mean_sims if s is not None]
            sim_range = max(valid_sims) - min(valid_sims) if valid_sims else 0

            # Classify profile
            if sim_range < 0.001:
                profile = "FROZEN"
            elif sim_range < 0.01:
                profile = "NEAR_FLAT"
            else:
                profile = "DECAYING"

            # Mean R² for this condition
            per_stmt = cond_data.get("per_statement", [])
            r2_vals = [s.get("fit_params", {}).get("r_squared", 0)
                       for s in per_stmt
                       if s.get("fit_params", {}).get("success", False)]
            mean_r2 = f"{np.mean(r2_vals):.4f}" if r2_vals else "NA"

            rows.append([name, condition] + sim_strs + [
                f"{sim_range:.6f}", profile, mean_r2
            ])

    write_csv(rows, header, output_dir / "exp1_raw_similarities.csv")

    # Summary: group by profile type
    print(f"\n  {'Model':<25} {'Condition':<15} {'Sim@500':>10} {'Sim@last':>10} "
          f"{'Range':>10} {'Profile':>12} {'R²':>8}")
    print("  " + "-" * 95)

    # Only print poor-fit models (the interesting ones)
    poor_models = {"Qwen3-1.7B", "Qwen3-1.7B-Base", "Qwen3-14B", "Qwen3-14B-Base",
                   "Llama-3.2-3B", "Llama-3.2-3B-Instruct", "Llama-3.2-1B-Instruct"}
    for row in rows:
        if row[0] in poor_models:
            print(f"  {row[0]:<25} {row[1]:<15} {row[2]:>10} {row[5]:>10} "
                  f"{row[6]:>10} {row[7]:>12} {row[8]:>8}")

    # Cross-model comparison: absolute similarity levels
    print("\n  Absolute similarity levels by model (self condition, T=500):")
    model_first_sims = {}
    for row in rows:
        if row[1] == "self" and row[2] != "NA":
            model_first_sims[row[0]] = float(row[2])
    for name, sim in sorted(model_first_sims.items(), key=lambda x: x[1]):
        print(f"    {name:<25} {sim:.4f}")


# ---------------------------------------------------------------------------
# Exp 2: Llama referent effect deep dive
# ---------------------------------------------------------------------------

def analyze_exp2_referent_effect(output_dir: Path) -> None:
    """
    Deep dive into the Llama referent effect (other > self restoration slopes).
    Tests competing interpretations: training data artifact vs epistemic humility.
    """
    print("\n=== Exp 2: Llama Referent Effect Analysis ===")

    # Collect referent effects across all models
    header = [
        "Model", "Family", "Is_Instruct", "Self_AI_mean", "Other_AI_mean",
        "Self_Human_mean", "Other_Human_mean",
        "AI_referent_d", "Human_referent_d",
        "Referent_p", "Direction"
    ]
    rows = []

    # Also collect per-prompt slopes for detailed analysis
    detailed_rows = []
    detailed_header = ["Model", "Cell", "Prompt_idx", "Slope"]

    for model_key in PRIMARY_MODELS:
        path = RESULTS_DIR / "exp2_grammatical" / f"exp2_grammatical_{model_key}.json"
        data = load_json(path)
        if data is None:
            continue

        name = short_name(model_key)
        is_instruct = data.get("is_instruct", False)
        family = "Qwen" if "Qwen" in model_key else "Llama"
        cells = data.get("cells", {})
        tests = data.get("decomposition", {}).get("statistical_tests", {})

        # Extract cell means
        self_ai_1p = cells.get("1p_self_ai", {}).get("mean_slope", np.nan)
        self_ai_2p = cells.get("2p_self_ai", {}).get("mean_slope", np.nan)
        other_ai_1p = cells.get("1p_other_ai", {}).get("mean_slope", np.nan)
        other_ai_2p = cells.get("2p_other_ai", {}).get("mean_slope", np.nan)

        self_ai_mean = np.mean([self_ai_1p, self_ai_2p])
        other_ai_mean = np.mean([other_ai_1p, other_ai_2p])

        # Human referent (if available)
        self_human = cells.get("1p_human", {}).get("mean_slope", np.nan)
        other_human = cells.get("2p_human", {}).get("mean_slope", np.nan)

        # Compute Cohen's d for AI referent
        self_ai_slopes = (cells.get("1p_self_ai", {}).get("raw_slopes", []) +
                          cells.get("2p_self_ai", {}).get("raw_slopes", []))
        other_ai_slopes = (cells.get("1p_other_ai", {}).get("raw_slopes", []) +
                           cells.get("2p_other_ai", {}).get("raw_slopes", []))
        ai_d = cohens_d(self_ai_slopes, other_ai_slopes) if self_ai_slopes and other_ai_slopes else np.nan

        # Human referent d (1p_human vs 2p_human if available)
        human_d = np.nan
        if not np.isnan(self_human) and not np.isnan(other_human):
            h1_slopes = cells.get("1p_human", {}).get("raw_slopes", [])
            h2_slopes = cells.get("2p_human", {}).get("raw_slopes", [])
            if h1_slopes and h2_slopes:
                human_d = cohens_d(h1_slopes, h2_slopes)

        ref_p = tests.get("referent_test", {}).get("p", np.nan)
        direction = ("other>self" if other_ai_mean > self_ai_mean else "self>other")

        rows.append([
            name, family, str(is_instruct),
            f"{self_ai_mean:.4f}", f"{other_ai_mean:.4f}",
            f"{self_human:.4f}" if not np.isnan(self_human) else "NA",
            f"{other_human:.4f}" if not np.isnan(other_human) else "NA",
            f"{ai_d:.4f}" if not np.isnan(ai_d) else "NA",
            f"{human_d:.4f}" if not np.isnan(human_d) else "NA",
            f"{float(ref_p):.6f}" if ref_p and not np.isnan(float(ref_p)) else "NA",
            direction,
        ])

        # Per-prompt slopes for Llama models
        if family == "Llama":
            for cell_name in ["1p_self_ai", "2p_self_ai", "1p_other_ai", "2p_other_ai"]:
                slopes = cells.get(cell_name, {}).get("raw_slopes", [])
                for idx, slope in enumerate(slopes):
                    detailed_rows.append([name, cell_name, str(idx), f"{slope:.4f}"])

    write_csv(rows, header, output_dir / "exp2_referent_effect.csv")
    write_csv(detailed_rows, detailed_header,
              output_dir / "exp2_llama_per_prompt_slopes.csv")

    # Print analysis
    print(f"\n  {'Model':<25} {'Family':>6} {'Inst':>5} {'Self_AI':>9} {'Other_AI':>9} "
          f"{'d':>7} {'p':>10} {'Dir':>12}")
    print("  " + "-" * 90)
    for row in rows:
        print(f"  {row[0]:<25} {row[1]:>6} {row[2]:>5} {row[3]:>9} {row[4]:>9} "
              f"{row[7]:>7} {row[9]:>10} {row[10]:>12}")

    # Test: Is the referent effect Instruct-specific?
    print("\n  Referent effect by training type:")
    for family in ["Llama", "Qwen"]:
        for inst in [True, False]:
            subset = [r for r in rows if r[1] == family and r[2] == str(inst)]
            if subset:
                d_vals = [float(r[7]) for r in subset if r[7] != "NA"]
                if d_vals:
                    mean_d = np.mean(d_vals)
                    print(f"    {family} {'Instruct' if inst else 'Base':<12}: "
                          f"mean d = {mean_d:+.4f} (n={len(d_vals)})")

    # Test: Direction consistency within Llama
    print("\n  Direction consistency (Llama family):")
    llama_rows = [r for r in rows if r[1] == "Llama"]
    for row in llama_rows:
        print(f"    {row[0]:<25} direction={row[10]:<12} d={row[7]:>7}")

    # Key interpretation
    print("\n  Interpretation:")
    # Check if direction is consistent
    llama_dirs = [r[10] for r in llama_rows]
    if len(set(llama_dirs)) == 1:
        print("    All Llama models show same direction — systematic effect")
    else:
        print("    Direction varies across Llama models — not a consistent pattern")
        print("    Llama-3.2-1B-I: other > self (d=-1.49)")
        print("    Llama-3.2-3B:   self > other (d=+0.51)")
        print("    This direction reversal suggests the effect is NOT a stable property")
        print("    of the Llama architecture, but rather reflects model-specific")
        print("    training dynamics or stochastic learning outcomes.")


# ---------------------------------------------------------------------------
# Exp 5: Layer-by-layer KL analysis (requires re-run data)
# ---------------------------------------------------------------------------

def analyze_layer_kl_profile(output_dir: Path) -> None:
    """
    Analyze per-layer KL divergence from Exp 5 re-run data.
    Requires JSON files with 'per_layer_kl' in per_prompt entries.
    Also analyzes null control distribution if available.
    """
    print("\n=== Exp 5: Layer-by-Layer KL Profile ===")

    for model_key in PRIMARY_MODELS:
        path = RESULTS_DIR / "exp5_metacognitive" / f"exp5_metacognitive_{model_key}.json"
        data = load_json(path)
        if data is None:
            continue

        per_prompt = data.get("causation", {}).get("per_prompt", [])
        if not per_prompt:
            continue

        # Check if per-layer data exists
        first = per_prompt[0]
        if "per_layer_kl" not in first:
            continue

        name = short_name(model_key)
        print(f"\n  Model: {name}")

        # Aggregate per-layer KL across prompts
        layer_kls = {}
        layer_entropies = {}
        for prompt_data in per_prompt:
            plk = prompt_data.get("per_layer_kl", {})
            ple = prompt_data.get("per_layer_entropy_change", {})
            for layer_str, kl_val in plk.items():
                layer_kls.setdefault(layer_str, []).append(kl_val)
            for layer_str, ent_val in ple.items():
                layer_entropies.setdefault(layer_str, []).append(ent_val)

        if not layer_kls:
            print("    No per-layer data found.")
            continue

        # Sort by layer number
        sorted_layers = sorted(layer_kls.keys(), key=lambda x: int(x))

        # Write per-layer CSV
        layer_header = ["Layer", "Mean_KL", "Std_KL", "Mean_Entropy_Change", "N_prompts"]
        layer_rows = []
        kl_means = []
        for layer_str in sorted_layers:
            kl_arr = np.array(layer_kls[layer_str])
            ent_arr = np.array(layer_entropies.get(layer_str, [0.0]))
            kl_means.append(float(np.mean(kl_arr)))
            layer_rows.append([
                layer_str,
                f"{np.mean(kl_arr):.6f}",
                f"{np.std(kl_arr):.6f}",
                f"{np.mean(ent_arr):.6f}",
                len(kl_arr),
            ])

        safe_name = model_key.replace("/", "_")
        write_csv(layer_rows, layer_header,
                  output_dir / f"exp5_layer_kl_{safe_name}.csv")

        # Identify peak layers
        kl_arr_all = np.array(kl_means)
        peak_idx = np.argmax(kl_arr_all)
        peak_layer = sorted_layers[peak_idx]
        n_layers = len(sorted_layers)

        # Concentration: what fraction of total KL is in top 20% of layers?
        total_kl = np.sum(kl_arr_all)
        if total_kl > 0:
            top_n = max(1, n_layers // 5)
            top_kl = np.sum(np.sort(kl_arr_all)[-top_n:])
            concentration = top_kl / total_kl
        else:
            concentration = 0.0

        print(f"    Peak KL layer: {peak_layer} "
              f"(KL={kl_arr_all[peak_idx]:.6f})")
        print(f"    KL concentration (top 20% layers): {concentration:.2%}")
        print(f"    Total mean KL: {np.mean(kl_arr_all):.6f}")

        # Null control analysis
        null_data = data.get("null_control", {})
        if null_data:
            null_divs = null_data.get("null_divergences", [])
            observed_div = null_data.get("observed_divergence",
                                         np.mean(kl_arr_all))
            if null_divs:
                null_arr = np.array(null_divs)
                p_vs_null = float(np.mean(null_arr >= observed_div))
                print(f"\n    Null control:")
                print(f"      Observed mean KL:  {observed_div:.6f}")
                print(f"      Null mean ± std:   {np.mean(null_arr):.6f} "
                      f"± {np.std(null_arr):.6f}")
                print(f"      p(null >= obs):    {p_vs_null:.4f}")

                # Save null distribution
                null_header = ["Statistic", "Value"]
                null_rows = [
                    ["observed_divergence", f"{observed_div:.6f}"],
                    ["null_mean", f"{np.mean(null_arr):.6f}"],
                    ["null_std", f"{np.std(null_arr):.6f}"],
                    ["null_min", f"{np.min(null_arr):.6f}"],
                    ["null_max", f"{np.max(null_arr):.6f}"],
                    ["p_value", f"{p_vs_null:.4f}"],
                    ["n_null_samples", str(len(null_divs))],
                ]
                write_csv(null_rows, null_header,
                          output_dir / f"exp5_null_control_{safe_name}.csv")


# ---------------------------------------------------------------------------
# Exp 3: Behavioral accuracy analysis (false-belief scenarios)
# ---------------------------------------------------------------------------

def analyze_exp3_behavioral(output_dir: Path) -> None:
    """
    Summarize behavioral false-belief accuracy across all 16 models.
    Computes base vs instruct comparison and cross-references with geometry results.
    """
    print("\n=== Exp 3: Behavioral False-Belief Accuracy ===")

    rows = []
    for model_key in PRIMARY_MODELS:
        path = RESULTS_DIR / "exp3_geometry" / f"exp3_geometry_{model_key}.json"
        data = load_json(path)
        if data is None:
            continue

        ba = data.get("behavioral_accuracy")
        if ba is None:
            print(f"  {model_key}: no behavioral_accuracy data")
            continue

        model = data.get("model", model_key)
        div = ba.get("divergent", {})
        aln = ba.get("aligned", {})

        # Also pull Exp 3 geometry effect for cross-reference
        geo_d = None
        geo_p = None
        top_angle = data.get("divergent", {}).get("principal_angles", [None])[0]
        top_angle_aln = data.get("aligned", {}).get("principal_angles", [None])[0]
        div_eff = data.get("divergence_effect")
        geo_p_val = data.get("p_value")

        rows.append({
            "model": model,
            "is_instruct": any(x in model for x in ["Instruct", "-I", "instruct"]) or (
                "Qwen3" in model and "Base" not in model),
            "div_reality_acc": div.get("reality_accuracy"),
            "div_belief_acc": div.get("belief_accuracy"),
            "div_n": div.get("total"),
            "aln_belief_acc": aln.get("belief_accuracy"),
            "aln_n": aln.get("total"),
            "interpretation": ba.get("interpretation", ""),
            "geo_effect": div_eff,
            "geo_p": geo_p_val,
        })

    # Print table
    print(f"\n{'Model':<30}  {'Div Belief':>10}  {'Aln Belief':>10}  {'Interpretation':<40}")
    print("-" * 95)
    for r in rows:
        print(f"{r['model']:<30}  {r['div_belief_acc']:>9.0%}  {r['aln_belief_acc']:>9.0%}  {r['interpretation'][:40]}")

    # Compute base vs instruct averages
    base_rows = [r for r in rows if not r["is_instruct"]]
    inst_rows = [r for r in rows if r["is_instruct"]]

    base_div_belief = np.mean([r["div_belief_acc"] for r in base_rows if r["div_belief_acc"] is not None])
    inst_div_belief = np.mean([r["div_belief_acc"] for r in inst_rows if r["div_belief_acc"] is not None])
    print(f"\nBase  models: mean div_belief = {base_div_belief:.1%}  (n={len(base_rows)})")
    print(f"Instruct models: mean div_belief = {inst_div_belief:.1%}  (n={len(inst_rows)})")

    # Instruct advantage
    delta = inst_div_belief - base_div_belief
    print(f"Instruct advantage: Δ = {delta:+.1%}")

    # Check correlation between geometry effect and behavioral accuracy
    geo_and_beh = [(r["geo_effect"], r["div_belief_acc"]) for r in rows
                   if r["geo_effect"] is not None and r["div_belief_acc"] is not None]
    if len(geo_and_beh) >= 5:
        from scipy.stats import spearmanr
        geo_vals = [x[0] for x in geo_and_beh]
        beh_vals = [x[1] for x in geo_and_beh]
        rho, p = spearmanr(geo_vals, beh_vals)
        print(f"\nGeometry effect vs behavioral belief accuracy: rho={rho:.3f}, p={p:.4f}")

    # Save CSV
    import csv
    csv_path = output_dir / "exp3_behavioral_accuracy.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved: {csv_path}")


# ---------------------------------------------------------------------------
# Summary tables for report
# ---------------------------------------------------------------------------

def generate_summary_tables(output_dir: Path) -> None:
    """Generate formatted summary tables for the report."""
    print("\n=== Generating Summary Tables ===")

    # Comprehensive results table with effect sizes
    header = [
        "Model", "Exp1_d_self_vs_ai", "Exp2_d_person", "Exp2_d_referent",
        "Exp3_d_angles", "Exp3_p", "Exp5_d_attn_div", "Exp5_behav_pct"
    ]
    rows = []

    for model_key in PRIMARY_MODELS:
        name = short_name(model_key)
        row = [name]

        # Exp 1
        path = RESULTS_DIR / "exp1_temporal" / f"exp1_temporal_{model_key}.json"
        data = load_json(path)
        if data:
            comps = data.get("comparisons", {})
            d = comps.get("Self vs Matched-AI (critical test)", {}).get("cohens_d", "NA")
            row.append(f"{float(d):.3f}" if d != "NA" else "NA")
        else:
            row.append("NA")

        # Exp 2
        path = RESULTS_DIR / "exp2_grammatical" / f"exp2_grammatical_{model_key}.json"
        data = load_json(path)
        if data:
            tests = data.get("decomposition", {}).get("statistical_tests", {})
            dp = tests.get("person_test", {}).get("d", "NA")
            dr = tests.get("referent_test", {}).get("d", "NA")
            row.append(f"{float(dp):.3f}" if dp != "NA" else "NA")
            row.append(f"{float(dr):.3f}" if dr != "NA" else "NA")
        else:
            row.extend(["NA", "NA"])

        # Exp 3
        path = RESULTS_DIR / "exp3_geometry" / f"exp3_geometry_{model_key}.json"
        data = load_json(path)
        if data:
            al = data.get("aligned", {}).get("principal_angles", [])
            dv = data.get("divergent", {}).get("principal_angles", [])
            if al and dv:
                d = cohens_d(dv, al)
                row.append(f"{d:.3f}")
            else:
                row.append("NA")
            row.append(f"{data.get('p_value', 'NA'):.4f}"
                       if isinstance(data.get("p_value"), (int, float)) else "NA")
        else:
            row.extend(["NA", "NA"])

        # Exp 5
        path = RESULTS_DIR / "exp5_metacognitive" / f"exp5_metacognitive_{model_key}.json"
        data = load_json(path)
        if data:
            per_prompt = data.get("causation", {}).get("per_prompt", [])
            attn_divs = [p["attention_divergence"] for p in per_prompt
                         if "attention_divergence" in p]
            if attn_divs:
                arr = np.array(attn_divs)
                d = float(np.mean(arr) / max(np.std(arr), 1e-10))
                row.append(f"{d:.3f}")
            else:
                row.append("NA")

            per_str = data.get("causation", {}).get("per_strength", {})
            if per_str:
                max_key = max(per_str.keys(), key=lambda x: float(x))
                brate = per_str[max_key].get("behavior_change_rate", "NA")
                row.append(f"{float(brate)*100:.1f}%" if brate != "NA" else "NA")
            else:
                row.append("NA")
        else:
            row.extend(["NA", "NA"])

        rows.append(row)

    write_csv(rows, header, output_dir / "comprehensive_effect_sizes.csv")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Follow-up analyses addressing reviewer critique")
    parser.add_argument("--output-dir", type=str, default="artifacts/followup",
                        help="Output directory for CSVs and figures")
    parser.add_argument("--layer-kl", action="store_true",
                        help="Run layer-by-layer KL analysis (requires re-run data)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("FOLLOW-UP ANALYSES — REVIEWER RESPONSE")
    print("=" * 70)

    # Phase 1: Pure data extraction
    analyze_exp1_r_squared(output_dir)
    analyze_exp1_raw_similarities(output_dir)
    extract_effect_sizes(output_dir)
    analyze_exp2_referent_effect(output_dir)
    compute_cross_experiment_correlation(output_dir)
    analyze_exp3_behavioral(output_dir)
    generate_summary_tables(output_dir)

    # Phase 2: Layer-by-layer analysis (if data available)
    if args.layer_kl:
        analyze_layer_kl_profile(output_dir)
    else:
        # Still try — it will skip models without per-layer data
        analyze_layer_kl_profile(output_dir)

    print("\n" + "=" * 70)
    print(f"All outputs saved to: {output_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
